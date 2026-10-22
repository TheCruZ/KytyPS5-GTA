#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCachePages.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <atomic>
#include <deque>
#include <functional>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static_assert(CACHING_PAGEBITS == BufferCachePages::kPageBits &&
	              CACHING_NUMPAGES == BufferCachePages::kNumPages &&
	              LOWER_ADDRESS_SIZE == BufferCachePages::kLowerAddressSize &&
	              LibKernel::Memory::kExtendedMemoryBase == BufferCachePages::kExtendedMemoryBase,
	              "bufferCachePages.h is out of date");
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);
	// The fault buffer holds three bitmaps of CACHING_NUMPAGES bits: pages that device-address
	// accesses found without a cached buffer, pages that V#-table stores wrote, and pages that
	// device-address accesses found cached (their buffers are in use).
	static constexpr uint64_t WRITE_BITMAP_OFFSET = CACHING_NUMPAGES / 8;
	static constexpr uint64_t USE_BITMAP_OFFSET   = 2 * WRITE_BITMAP_OFFSET;
	static constexpr uint64_t FAULT_BUFFER_SIZE   = 3 * WRITE_BITMAP_OFFSET;

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	// A guest write faulted on a page the cache protects.
	void                   InvalidateFaultedPage(uint64_t vaddr);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	// needs_device_address: the caller reads the data through a buffer device address, which
	// the stream buffer used for small CPU-written reads does not have.
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer      = false,
	                                                        BufferId id                   = {},
	                                                        bool     needs_device_address = false);
	// A written binding of which the shader's stores reach only [written_vaddr, written_vaddr +
	// written_size): the whole range is synchronized, only that part becomes GPU-written.
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferWritten(uint64_t vaddr, uint64_t size,
	                                                               uint64_t written_vaddr,
	                                                               uint64_t written_size,
	                                                               BufferId id = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer*       GetGdsBuffer() noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer*       GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	// GPU write bitmap (KYTY_WRITE_BITMAP=1, rework.md Phase 2 first step): one bit per 4 KiB
	// guest page below 1 TiB, set by shaders through bindings whose writes the host cannot bound.
	// Its device address, or 0 when it is off or unavailable.
	[[nodiscard]] vk::DeviceAddress WriteBitmapAddress();
	// KYTY_SRT_VERIFY: the report buffer verifying shaders write when a value the host used
	// differs from the GPU's (count, hash low, hash high), host-coherent; 0 when unavailable.
	[[nodiscard]] vk::DeviceAddress SrtVerifyReportAddress();
	// The shader the report names when its count changed since the last call, or 0; `detail`
	// gets the first differing pair's index, the host's value and the GPU's.
	[[nodiscard]] uint64_t TakeSrtVerifyReport(std::array<uint32_t, 3>* detail = nullptr);
	// ObtainBuffer for such a written binding: its pages become bitmap-tracked, so a guest read
	// of a page the GPU did not write needs no download (ReadMemoryStep).
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferTracked(uint64_t vaddr, uint64_t size,
	                                                               BufferId id);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	// Fills memory the emulator itself rewrites, such as consumed color metadata, without
	// invalidating images over it. Memory the GPU owns is filled on the GPU, so the fill never
	// drains it; other memory gets a backing write.
	void FillInternalMemory(uint64_t vaddr, uint64_t size, uint32_t value);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool              IsRegionRegistered(uint64_t vaddr, uint64_t size);
	// Color metadata the GPU found free of the clear keys in `keys` (bits of code >> 5) stays so
	// until a GPU write or download.
	void               MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys);
	[[nodiscard]] bool IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const;
	[[nodiscard]] bool              HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// The same, under the dirty-range lock: for a GPU-read delegate (Memory::SetGpuReadDelegate).
	[[nodiscard]] bool              HasGpuDirtyBytesShared(uint64_t vaddr, uint64_t size);
	// Whether an asynchronous download of bytes in the range is still in flight: those bytes are
	// no longer GPU-dirty, but guest memory holds them only once the download is published.
	[[nodiscard]] bool              HasPendingDownload(uint64_t vaddr, uint64_t size);
	// GPU thread: waits until no download of bytes in the range is in flight (published).
	void                            WaitDownloadsLanded(uint64_t vaddr, uint64_t size);
	// GPU thread, before it reads guest bytes on the host (indirect arguments, COND_EXEC): shared
	// buffers are not page-protected, so a read there does not fault into a sync. Waits for the
	// GPU writes to them recorded so far.
	void WaitSharedWrites(uint64_t vaddr, uint64_t size);
	// GPU thread (KYTY_SRT_DEFER): records and submits the download of the GPU-written bytes of
	// [vaddr, vaddr + size) without waiting. Returns the tick to wait for (its completion and
	// priority operations publish the bytes), or 0 when the bytes can be read now.
	[[nodiscard]] uint64_t StartSrtReadback(uint64_t vaddr, uint64_t size);
	// GPU thread (KYTY_GPU_SIZED_DYNAMIC): see Memory::PeekGpuDword.
	[[nodiscard]] uint32_t PeekDword(uint64_t vaddr);
	// Diagnostics: the write-tracking state of the page holding vaddr.
	[[nodiscard]] std::string DescribeWriteState(uint64_t vaddr);
	// Any thread: none of the bytes is GPU-written, or on its way back from the GPU, so guest
	// memory holds their current value even when their page is protected.
	[[nodiscard]] bool              IsCleanForConcurrentRead(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] bool              IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool              IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void                            ProcessFaultBuffer();
	// Synchronizes the buffers over the ranges the tracker queued after the CPU wrote to them
	// (QueueBdaSync); the ranges are cut to the mapped ones.
	void                            SynchronizeBda(const RangeSet& mapped);
	void                            QueueBdaSync(uint64_t vaddr, uint64_t size) {
		m_memory_tracker.QueueBdaSync(vaddr, size);
	}
	// GPU thread: writes bytes on a page protected because the GPU wrote to it, without
	// downloading the page: the host copy through the backing store, the GPU copy in the command
	// stream, after the GPU's earlier writes. Bytes the GPU had written are then current on both
	// sides and no longer need a download. False when there is nothing to save (the page is not
	// protected as GPU-written) or it would be wrong (a download of these bytes is in flight, or
	// no cached buffer covers them); the caller then writes normally.
	[[nodiscard]] bool              WriteClean(uint64_t vaddr, const void* data, uint64_t size);
	[[nodiscard]] ShaderFaultReport CollectFaults() { return m_fault_manager.CollectFaults(); }
	[[nodiscard]] uint64_t          UnattributedFaults() const noexcept {
		return m_fault_manager.UnattributedFaults();
	}
	void                            SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	// Same, but visits only the tracker regions that may hold CPU-dirty pages.
	void                            SynchronizeCpuDirtyBuffersInRange(uint64_t vaddr, uint64_t size);
	void                            RunGarbageCollector();
	// Diagnostics, GPU thread: makes guest memory current for [vaddr, vaddr + size) by
	// downloading what the GPU wrote there (drains the GPU).
	void                            DownloadRangeForDiagnostics(uint64_t vaddr, uint64_t size);
	// Diagnostics (GPU thread): the GPU copy of [vaddr, vaddr + size) as the cached buffer holding
	// it has it now, waiting for the GPU; false when no single buffer holds the range.
	bool                            PeekGpuCopy(uint64_t vaddr, uint64_t size, std::vector<uint32_t>& out);
	// Diagnostics (GPU thread): records copies of the GPU's bytes of each guest range, and of
	// [raw_offset, raw_offset + raw_size) of the host buffer raw, behind all work recorded so far,
	// and calls done with them once that work has run (no wait). A range no single cached buffer
	// holds comes back empty.
	using SnapshotDone = std::function<void(std::vector<std::vector<uint32_t>>&, std::vector<uint32_t>&)>;
	void SnapshotGpu(const std::vector<std::pair<uint64_t, uint64_t>>& ranges, vk::Buffer raw,
	                 uint64_t raw_offset, uint64_t raw_size, SnapshotDone done);

	// Diagnostics: the guest shader whose bindings are being prepared on this thread, if any.
	inline static thread_local uint64_t s_diag_shader_hash = 0;

	// A ReadMemoryStep result with this bit set names a copy-queue readback (see bufferCache.cpp)
	// for the caller to complete, instead of a tick to wait for.
	static constexpr uint64_t CopyReadbackMarker = uint64_t {1} << 63u;
	// Any thread: waits for a copy-queue readback and publishes its bytes to guest memory (once;
	// a marker whose readback was already completed returns at once).
	void CompleteCopyReadback(uint64_t marker);
	// GPU thread, at the end of a submission: records the downloads of the windows the game
	// threads keep reading back that the submission wrote to (KYTY_EAGER_READBACK).
	void EagerDownloads();
	// GPU thread, before guest memory is unmapped: drops the shared buffers over it (their
	// imports point into that memory).
	void UnmapShared(uint64_t vaddr, uint64_t size);

private:
	// Shared buffers (KYTY_SHARED_BUFFERS=1): a small buffer the game threads keep reading back is
	// replaced by one over the guest's own memory (VK_EXT_external_memory_host), so GPU writes land
	// in guest memory and the reads no longer wait for downloads. GPU thread only.
	[[nodiscard]] bool IsShared(uint64_t vaddr, uint64_t size) const {
		return !m_shared_ranges.Empty() && m_shared_ranges.Intersects(vaddr, size);
	}
	// A game-thread readback of this buffer: votes for sharing it, and shares it at enough votes.
	// True when the buffer is shared now and the read may proceed.
	bool VoteShared(BufferId id);
	RangeSet                                   m_shared_ranges;
	std::unordered_map<uint64_t, uint32_t>     m_shared_votes;
	std::unordered_map<uint64_t, uint32_t>     m_shared_attempts;
	// The tick of the latest GPU write recorded into each shared buffer, until it completes.
	std::unordered_map<uint64_t, uint64_t>     m_shared_write_ticks;
	friend struct BufferCacheTestAccess;

	void ReleaseBuffer(BufferId id);
	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	// Records bytes a binding makes GPU-written (after SynchronizeBuffer marked their pages).
	void MarkGpuWritten(uint64_t vaddr, uint64_t size, bool tracked = false);
	// ReadMemoryStep: drops the pages of the readback window the bitmap shows the GPU did not
	// write. True with wait_tick set when an asynchronous caller must first wait for that tick.
	bool RefineWithWriteBitmap(const Buffer& buffer, uint64_t vaddr, uint64_t size, bool async,
	                           uint64_t& wait_tick);
	// When a submission has completed: tracked pages its shaders did not write are released
	// (no longer GPU-written, no longer protected). GPU thread.
	void SettleWriteBitmap(uint64_t completed_tick);
	// Whole pages whose GPU-written bytes were synced (downloaded or overwritten) leave
	// m_bitmap_written. Caller holds m_dirty_ranges_mutex.
	void ClearWriteBits(uint64_t vaddr, uint64_t size);
	// Once per frame (GPU thread): the next epoch's bitmap moves its bits into m_bitmap_written
	// and is cleared, then takes the frame's writes.
	void AdvanceWriteBitmapEra();
	// The written bits of [word_begin, word_end) ORed over all epochs (or of only_epoch), read
	// with streaming loads.
	void LoadWrittenWords(uint64_t word_begin, uint64_t word_end, std::vector<uint32_t>& out,
	                      int only_epoch = -1) const;
	uint64_t m_bitmap_settle_frame = UINT64_MAX;
	// Eras: one per frame. At each era's end the GPU copies the era's bitmap words to host-cached
	// staging and zeroes them, in command order, so each copy holds exactly the pages that era
	// wrote (the host never reads the bitmap itself: device memory, ~1 us a word).
	// m_bitmap_written: pages known written and not synced since.
	static constexpr uint32_t                          WriteBitmapEpochs = 1;
	uint64_t                                           m_bitmap_era      = 0;
	std::array<uint64_t, 4>                            m_era_start_tick {};
	// region -> last era it was marked in; erased once that era's copy is processed
	std::unordered_map<uint64_t, uint64_t>             m_region_era;
	RangeSet                                           m_bitmap_written;
	struct BitmapCopy {
		uint64_t                                   era  = 0;
		uint64_t                                   tick = 0;
		std::unique_ptr<Buffer>                    staging;
		std::vector<std::pair<uint64_t, uint64_t>> spans;   // bitmap word ranges
		std::vector<uint64_t>                      offsets; // their first word in staging
	};
	std::deque<BitmapCopy>                 m_bitmap_copies;
	std::vector<std::unique_ptr<Buffer>>   m_bitmap_staging_pool;
	// Write prediction (KYTY_WRITE_PREDICT, on with the bitmap): unbounded written bindings
	// protect only the pages the GPU wrote in recent eras; a page written outside them is
	// protected when its era's copy is processed.
	RangeSet                               m_predicted;
	uint64_t                               m_predicted_generation = 0;
	std::unordered_map<uint64_t, uint64_t> m_predict_seen; // page -> last era written
	std::unordered_map<uint64_t, uint64_t> m_predict_seen_blocks; // 32-page block -> last era
	struct BindingPrediction {
		uint64_t era        = 0; // era + 1 it was last marked in
		uint64_t generation = 0; // of m_predicted then
		uint32_t eras_seen  = 0; // the first eras mark it whole, to learn its writes
	};
	std::unordered_map<uint64_t, BindingPrediction> m_predict_marked;
	// Tracked bindings obtained so far, and per page the count when WaitSharedWrites last
	// protected it: a page is protected again only after another tracked binding.
	uint64_t                                        m_tracked_serial = 0;
	std::unordered_map<uint64_t, uint64_t>          m_host_read_serial;
	void               ProcessBitmapCopies();
	void               ProcessBitmapCopy(BitmapCopy& copy);
	[[nodiscard]] bool PredictWrites() const;
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void                   Unregister(BufferId id);
	template <bool insert>
	void                     ChangeRegister(BufferId id);
	void                     DeleteBuffer(BufferId id);
	[[nodiscard]] bool       SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                           bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// One readback attempt on the GPU thread: 0 when done, else the tick to wait for.
	[[nodiscard]] uint64_t ReadMemoryStep(uint64_t vaddr, uint64_t size, bool is_write,
	                                      bool from_gpu_thread, bool async);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	template <bool async>
	[[nodiscard]] bool DownloadBufferWindow(Buffer& buffer, uint64_t vaddr, uint64_t size);
	template <bool async>
	void DownloadBufferCopies(Buffer& buffer, std::vector<vk::BufferCopy> copies,
	                          uint64_t total_size);

	// Copy-queue readbacks (KYTY_COPY_QUEUE_READBACK, default on). GPU thread: when every GPU
	// write to the window has executed, copies its GPU-written bytes on the second queue and
	// returns the readback's marker (0: not taken, the caller downloads as before).
	[[nodiscard]] uint64_t TryCopyQueueReadback(Buffer& buffer, uint64_t window_begin,
	                                            uint64_t window_end);
	// GPU thread: the marker of an unfinished copy-queue readback overlapping the range, or 0.
	[[nodiscard]] uint64_t FindCopyReadback(uint64_t vaddr, uint64_t size) const;
	// Any thread: waits until no copy-queue readback is reading a cache buffer (before one is
	// destroyed). Does not publish.
	void                   WaitCopyReadbacks();
	// GPU thread: the tick the bytes MarkGpuWritten records are written in, by 1 MiB block.
	void                   NoteGpuWriteTick(uint64_t vaddr, uint64_t size);
	[[nodiscard]] uint64_t LatestGpuWriteTick(uint64_t begin, uint64_t end) const;
	struct CopyReadbackSlot {
		std::mutex                  mutex;
		std::atomic<uint32_t>       busy {0};
		uint32_t                    generation = 0;
		vk::CommandPool             pool       = nullptr;
		vk::CommandBuffer           command    = nullptr;
		vk::Fence                   fence      = nullptr;
		std::unique_ptr<Buffer>     staging;
		uint64_t                    buffer_address = 0;
		uint64_t                    window_begin   = 0;
		uint64_t                    window_end     = 0;
		uint64_t                    total_size     = 0;
		std::vector<vk::BufferCopy> copies;
		bool                        snapshot       = false; // KYTY_READ_SNAPSHOT
	};
	// KYTY_READ_SNAPSHOT: the GPU's current bytes of a window whose writers are still running,
	// copied without waiting for them; and the windows whose snapshot is in guest memory, waiting
	// for the GPU thread to let reads through.
	[[nodiscard]] uint64_t TrySnapshotReadback(Buffer& buffer, uint64_t window_begin,
	                                           uint64_t window_end);
	std::mutex                                        m_snapshot_mutex;
	std::vector<std::pair<uint64_t, uint64_t>>        m_snapshots_ready;
	static constexpr uint32_t CopyReadbackSlots = 8;
	std::array<CopyReadbackSlot, CopyReadbackSlots>   m_copy_slots;
	std::atomic<uint32_t>                             m_copy_in_flight {0};
	std::unordered_map<uint64_t, uint64_t>            m_write_ticks;
	// Writes too large to note block by block: begin, end, tick.
	std::vector<std::array<uint64_t, 3>>              m_large_writes;
	void                                              NoteHotReadback(uint64_t vaddr);
	void                                              NoteEagerWrite(uint64_t vaddr, uint64_t size);
	std::unordered_map<uint64_t, uint64_t>            m_hot_windows;
	std::unordered_set<uint64_t>                      m_eager_pending;
	uint64_t                                          m_eager_count  = 0;
	uint64_t                                          m_copy_taken   = 0;
	uint64_t                                          m_copy_busy    = 0;
	uint64_t                                          m_copy_running = 0;
	void               ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size);
	// Reads GPU-written bytes whose last writes were already submitted: a copy on the readback
	// queue waits for those submissions only, not for the work queued after them.
	[[nodiscard]] bool DownloadOnReadbackQueue(Buffer& buffer, std::span<const vk::BufferCopy> copies,
	                                           uint64_t total_size);
	void NoteWrite(uint64_t vaddr, uint64_t size) {
		// Draws bind the same written ranges again and again.
		if (m_pending_writes.empty() || m_pending_writes.back() != std::pair {vaddr, size}) {
			m_pending_writes.emplace_back(vaddr, size);
			for (const auto& hot: m_readback_hot) {
				if (vaddr < hot.end && hot.begin < vaddr + size) {
					m_flush_requested = true;
					break;
				}
			}
		}
	}
	// A readback of [begin, end) waited for writes still being recorded (`drained`) or for
	// submitted ones; `device_address` when they were device-address writes.
	void NoteReadbackWait(uint64_t begin, uint64_t end, bool device_address, bool drained);

	GraphicContext&                                    m_graphics;
	CommandScheduler&                                  m_scheduler;
	FaultManager                                       m_fault_manager;
	Buffer                                             m_gds_buffer;
	Buffer                                             m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                         m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                          m_buffers;
	PageTable                                          m_page_table;
	RangeSet                                           m_gpu_modified_ranges;
	// Bytes whose download is recorded but not yet in guest memory.
	RangeSet                                           m_downloading_ranges;
	// Every in-flight download's ranges, one entry each: m_downloading_ranges is their union. A
	// publication removes only its own entries (ISSUES #26: the first of two downloads of one
	// window cleared the range while the second was in flight; a game thread then took the
	// page as current, wrote it, and the second publication put older GPU bytes over that).
	std::vector<std::pair<uint64_t, uint64_t>>         m_downloading_list;
	// Write bitmap (WriteBitmapAddress): the buffer, the whole pages whose GPU writes all came
	// through tracked bindings, and per 4 MiB region the last tick a tracked binding wrote it.
	// Guarded by m_dirty_ranges_mutex, as m_gpu_modified_ranges.
	std::unique_ptr<Buffer>                            m_write_bitmap;
	std::unique_ptr<Buffer>                            m_srt_verify_report;
	bool                                               m_srt_verify_report_failed = false;
	uint32_t                                           m_srt_verify_reports_seen  = 0;
	bool                                               m_write_bitmap_failed = false;
	RangeSet                                           m_bitmap_tracked;
	std::unordered_map<uint64_t, uint64_t>             m_bitmap_ticks;
	// Tracked written bindings whose ticks may still run, in tick order (KYTY_BITMAP_RANGE_TICKS):
	// a readback waits for the writers of its window, not for any writer of its 4 MiB region.
	struct PendingBitmapWrite {
		uint64_t begin;
		uint64_t end;
		uint64_t tick;
	};
	std::deque<PendingBitmapWrite>                     m_pending_bitmap_writes;
	bool m_narrow_window = false; // RefineWithWriteBitmap: download only the faulting pages
	bool m_srt_readback  = false; // StartSrtReadback is asking (KYTY_SRT_NARROW_READBACK)
	uint64_t                                           m_bitmap_dropped_pages = 0;
	uint64_t                                           m_bitmap_written_pages = 0;
	// The tick of the latest asynchronous readback (KYTY_ASYNC_READBACK); GPU thread only.
	uint64_t                                           m_last_async_download_tick = 0;
	std::atomic<uint64_t>                              m_download_publish_tick {0}; // latest download's publication tick
	// PeekDword: a small host-visible buffer of dword copies, one slot per guest address.
	struct PeekSlot {
		uint32_t index = 0;
		uint64_t tick  = 0;
		uint32_t value = 0;
	};
	std::unique_ptr<Buffer>                            m_peek_buffer;
	std::unordered_map<uint64_t, PeekSlot>             m_peek_slots;
	// Guards changes to both range sets (GPU thread and download completions) against
	// IsCleanForConcurrentRead; the GPU thread reads them without it.
	mutable std::shared_mutex                          m_dirty_ranges_mutex;
	MemoryTracker                                      m_memory_tracker;
	StreamBuffer                                       m_staging_buffer;
	// Guest reads UploadCopies leaves to the recording thread: staging destination, guest
	// address, bytes.
	struct UploadRead {
		uint8_t* destination;
		uint64_t address;
		uint64_t size;
	};
	std::vector<UploadRead> m_upload_reads;
	StreamBuffer                                       m_stream_buffer;
	StreamBuffer                                       m_download_buffer;
	StreamBuffer                                       m_device_buffer;
	TextureCache&                                      m_texture_cache;
	uint64_t                                           m_total_used_memory = 0;
	// One per TileManager::ColorClearSlots key slot.
	std::array<RangeSet, 8>                           m_checked_color_metadata;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	// As configured; the ones above come down while memory is short (as in the texture cache).
	uint64_t m_base_trigger_gc_memory     = 0;
	uint64_t m_base_critical_gc_memory    = 0;
	uint64_t m_applied_trigger_gc_memory  = UINT64_MAX; // as in the texture cache
	uint64_t m_applied_critical_gc_memory = UINT64_MAX;
	uint64_t m_gc_tick            = 0;
	// The LRU clock: presented frames, advanced by the collector's own ticks as well so a
	// stretch without presents still ages its entries.
	[[nodiscard]] uint64_t LruClock() const noexcept;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
