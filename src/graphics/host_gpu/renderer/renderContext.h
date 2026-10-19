#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/indirectWriteTables.h"
#include "graphics/host_gpu/renderer/occlusionQueries.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "kernel/eventQueue.h"

#include <array>
#include <atomic>
#include <memory>
#include <shared_mutex>
#include <span>
#include <deque>
#include <utility>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

class GuestGpu;

// The renderer's lock (recursive, as Common::Mutex). The GPU thread that executes the draws and dispatches takes it
// around each of them (thousands a frame), other threads (presentation, a capture) a few times a frame: an interlocked
// acquisition and release per draw stalled that thread on its store buffer. So its acquisition is a store and a load
// (chenxiao07's asymmetric Dekker lock): another thread takes the mutex, raises `requested`, makes the owner's stores
// visible and its later loads see the request (FlushProcessWriteBuffers) and waits until the owner is outside (`busy`
// clear); the owner that finds a request takes the mutex. KYTY_RENDER_MUTEX=0: nobody becomes the owner, every thread
// takes the mutex.
class RenderMutex {
public:
	RenderMutex()  = default;
	~RenderMutex() = default;
	KYTY_CLASS_NO_COPY(RenderMutex);

	// The calling thread (the one that executes the GPU operations) takes the owner's side from now on, until it
	// resigns (outside any lock).
	void BecomeOwner();
	void ResignOwner();

	void Lock() {
		if (!t_owner) return RequesterLock();
		if (m_owner_depth++ != 0) return;
		m_busy.store(1, std::memory_order_relaxed);
		// (The store before the load in program order: the requester's flush is what orders them on the CPU.)
		std::atomic_signal_fence(std::memory_order_seq_cst);
		if (!m_asymmetric || m_requested.load(std::memory_order_acquire) != 0) OwnerLockSlow();
	}
	void Unlock() {
		if (!t_owner) return RequesterUnlock();
		if (--m_owner_depth != 0) return;
		if (m_owner_locked) return OwnerUnlockSlow();
		m_busy.store(0, std::memory_order_release);
	}

private:
	void OwnerLockSlow();
	void OwnerUnlockSlow();
	void RequesterLock();
	void RequesterUnlock();

	Common::Mutex m_mutex;
	// The owner's (written at each of its acquisitions): it is inside, and its depth.
	alignas(64) std::atomic<uint32_t> m_busy {0};
	uint32_t                          m_owner_depth  = 0;
	bool                              m_owner_locked = false; // (its outermost acquisition took the mutex)
	// The requesters' (written under the mutex): one holds the mutex and waits for the owner to leave; an owner exists.
	alignas(64) std::atomic<uint32_t> m_requested {0};
	bool                              m_owner_exists = false;
	bool                              m_asymmetric   = false; // (the host can flush other processors' stores)
	inline static thread_local bool     t_owner           = false;
	inline static thread_local uint32_t t_requester_depth = 0;
};

class RenderLockGuard {
public:
	// NOLINTNEXTLINE(google-runtime-references)
	explicit RenderLockGuard(RenderMutex& mutex): m_mutex(mutex) { m_mutex.Lock(); }
	~RenderLockGuard() { m_mutex.Unlock(); }
	KYTY_CLASS_NO_COPY(RenderLockGuard);

private:
	RenderMutex& m_mutex;
};

class RenderContext {
public:
	explicit RenderContext(GraphicContext& graphics);
	~RenderContext();
	KYTY_CLASS_NO_COPY(RenderContext);

	[[nodiscard]] GraphicContext&           GetGraphics() const noexcept { return m_graphics; }
	void                                    InitializeGpu(VideoOut::VideoOutDriver* video_out);
	void                                    ShutdownGpu();
	[[nodiscard]] GuestGpu&                 GetGpu() const;
	[[nodiscard]] VideoOut::VideoOutDriver& GetVideoOut() const;

	RenderMutex&        GetMutex() { return m_mutex; }
	CommandScheduler&   GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&      GetPipelineCache() { return m_pipeline_cache; }
	DescriptorHeap&     GetDescriptorHeap() { return m_descriptor_heap; }
	SamplerCache&       GetSamplerCache() { return m_sampler_cache; }
	BufferCache&        GetBufferCache() { return m_buffer_cache; }
	TextureCache&       GetTextureCache() { return m_texture_cache; }
	IndirectWriteTables& GetIndirectWriteTables() { return m_indirect_write_tables; }
	OcclusionQueries&   GetOcclusionQueries() { return m_occlusion_queries; }
	RenderExecutor&     GetRenderExecutor() { return m_render_executor; }

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	[[nodiscard]] bool TryWriteFaultWindow(uint64_t fault_vaddr);
	// Before the emulated GPU writes guest memory (labels, WRITE_DATA): resolves a tracked page
	// as its write fault would, without raising the exception.
	void PrepareGpuWrite(const void* destination, uint64_t size) noexcept;
	// Execution thread, before each operation (see PrepareBda()).
	void NoteOperation(uint64_t submit_id, bool after_wait) noexcept {
		if (submit_id != m_bda_submit || after_wait) {
			m_bda_submit = submit_id;
			m_bda_generation++;
		}
	}
	// A small command-processor write that GPU-owned pages would turn into a readback of all
	// queued GPU work (see BufferCache::WriteAroundGpuWrites()); false when the caller must
	// PrepareGpuWrite() and store the bytes itself.
	[[nodiscard]] bool WriteAroundGpuWrites(void* destination, const void* data, uint64_t size);
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	// Before a host write of all of the range: GPU-written bytes it replaces are dropped.
	[[nodiscard]] bool InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool InvalidateImages(uint64_t vaddr, uint64_t size);
	// The pages of a range the trackers protect (PageManager::WatchedSpan).
	[[nodiscard]] bool WatchedSpan(uint64_t vaddr, uint64_t size, uint64_t* first, uint64_t* end) const noexcept {
		return m_page_manager.WatchedSpan(vaddr, size, first, end);
	}
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	// Advances whenever a range is mapped or unmapped.
	[[nodiscard]] uint64_t MappedRangesVersion() const noexcept;
	// Execution thread: the guest frames (suspend points) executed so far.
	[[nodiscard]] uint64_t FrameNumber() const noexcept { return m_frame_number; }
	// Whether part of the range was unmapped after MappedRangesVersion() was `version` (true
	// when the log of unmaps no longer reaches back that far).
	[[nodiscard]] bool UnmappedSince(uint64_t version, uint64_t vaddr, uint64_t size) const;
	void               MapMemory(uint64_t vaddr, uint64_t size);
	// releasing: the range is unmapped or decommitted right after (the trackers' unwatching leaves its host protection to
	// the unmap, KYTY_UNMAP_PROTECT_SKIP).
	void               UnmapMemory(uint64_t vaddr, uint64_t size, bool releasing = false);
	// A releasing UnmapMemory whose unmap failed: the range's pages get their trackers' protection back.
	void               ReapplyProtection(uint64_t vaddr, uint64_t size) { m_page_manager.ReapplyProtection(vaddr, size); }
	void               CacheDmaBases(const ShaderStageRuntime& runtime);
	// Uploads the CPU writes of the guest memory that the device-address reads of the stages can
	// reach (see ShaderInfo::dma_windows); false when a stage's reach is not known, and nothing
	// was uploaded.
	[[nodiscard]] bool SynchronizeDmaFootprint(std::span<const ShaderStageRuntime* const> stages);
	// may_write: a shader of the operation may store through device addresses. full_sync: upload
	// the CPU writes of all cached memory first (SynchronizeDmaFootprint() did not cover the
	// operation).
	void               PrepareBda(bool may_write, bool full_sync = true);
	void               RunGarbageCollector();

	void AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void TriggerInterrupt(int event_id, uint32_t context_id);

private:
	struct InterruptEqRegistration {
		LibKernel::EventQueue::KernelEqueue eq       = LibKernel::EventQueue::KERNEL_EQUEUE_INVALID;
		int                                 event_id = 0;
	};

	GraphicContext&           m_graphics;
	RenderMutex               m_mutex;
	RenderExecutor            m_render_executor;
	CommandScheduler          m_command_scheduler;
	DescriptorHeap            m_descriptor_heap;
	PipelineCache             m_pipeline_cache;
	SamplerCache              m_sampler_cache;
	PageManager               m_page_manager;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	IndirectWriteTables       m_indirect_write_tables {*this};
	OcclusionQueries          m_occlusion_queries {*this};
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	uint64_t                  m_mapped_ranges_version = 0;
	// Ranges mapped since the last BDA synchronization (under m_mapped_ranges_mutex): that
	// synchronization forces their regions (BufferCache::ForceSynchronization()).
	std::vector<std::pair<uint64_t, uint64_t>> m_bda_mapped_since_sync;
	// The latest unmaps (under m_mapped_ranges_mutex): the version each moved to, and the range.
	struct UnmapEntry {
		uint64_t version = 0, vaddr = 0, size = 0;
	};
	std::deque<UnmapEntry> m_unmap_log;
	uint64_t               m_unmap_log_floor = 0; // the version of the newest unmap dropped
	uint64_t               m_frame_number    = 0;
	// SynchronizeDmaFootprint(): the spans of an operation.
	std::vector<std::pair<uint64_t, uint64_t>> m_dma_spans;
	// Epochs (CPU dirty, buffer registrations, mapped ranges) read before the last full BDA
	// synchronization.
	std::array<uint64_t, 3> m_bda_sync_epochs {UINT64_MAX, UINT64_MAX, UINT64_MAX};
	// Advances with each guest submission an operation belongs to, after each WAIT_REG_MEM the
	// command processor resolved and whenever the execution thread itself writes guest memory
	// the CPU tracks (see PrepareBda()).
	uint64_t m_bda_generation        = 1;
	uint64_t m_bda_synced_generation = 0;
	uint64_t m_bda_submit            = UINT64_MAX;
	std::unique_ptr<GuestGpu> m_gpu;
	VideoOut::VideoOutDriver* m_video_out = nullptr;
	bool                      m_fault_process_pending = false;
	// Device-address accesses ran since the use bitmap was last read; reads one in UseScanPeriod
	// collections (the buffer cache's age collection waits much longer).
	bool                      m_use_scan_pending      = false;
	uint32_t                  m_use_scan_countdown    = 0;
	bool                      m_bda_logged = false;

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
