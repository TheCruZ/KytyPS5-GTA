#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/shader/shaderBindings.h"
#include "libs/errno.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <thread>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#undef MemoryBarrier
#endif

namespace Libs::Graphics {

namespace {

// chenxiao07's KYTY_UNMAP_PROTECT_SKIP (default on; 0 restores the protection of every released watcher).
bool UnmapProtectSkipEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_UNMAP_PROTECT_SKIP");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// chenxiao07's write windows (KYTY_WRITE_FAULT_WINDOW, default on; 0: the faulting page, or the pages up to the next
// 64 KiB when the page below is already written). A guest write fault in memory only the CPU owns releases its aligned
// 16 KiB at once, and the next 64 KiB when it follows the window this thread's last such fault ended in within 2 ms (a
// stream of writes): Demon's Souls took ~800 write faults a frame, each an exception and a protection change.
bool WriteFaultWindowEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_WRITE_FAULT_WINDOW");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

uint64_t WindowClockNow() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// KYTY_RENDER_MUTEX=0: the renderer's lock is the plain mutex for every thread, the draw thread included.
bool RenderMutexEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_RENDER_MUTEX");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// Makes the stores of every thread of the process visible to all processors (false: the host cannot).
bool FlushProcessWriteBuffers() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::FlushProcessWriteBuffers();
	return true;
#else
	return false;
#endif
}

} // namespace

void RenderMutex::BecomeOwner() {
	if (!RenderMutexEnabled()) {
		return;
	}
	EXIT_IF(t_owner);
	m_mutex.Lock(); // (no requester inside: a later one finds the owner)
	m_owner_exists = true;
	m_asymmetric   = FlushProcessWriteBuffers();
	m_owner_depth  = 0;
	m_owner_locked = false;
	t_owner        = true;
	m_mutex.Unlock();
}

void RenderMutex::ResignOwner() {
	if (!t_owner) {
		return;
	}
	EXIT_IF(m_owner_depth != 0);
	m_mutex.Lock();
	t_owner        = false;
	m_owner_exists = false;
	m_mutex.Unlock();
}

void RenderMutex::OwnerLockSlow() {
	// A requester holds the mutex or is about to: out, and in after it (or the host has no flush: the mutex always).
	m_busy.store(0, std::memory_order_release);
	m_mutex.Lock();
	m_owner_locked = true;
}

void RenderMutex::OwnerUnlockSlow() {
	m_owner_locked = false;
	m_mutex.Unlock();
}

void RenderMutex::RequesterLock() {
	if (t_requester_depth++ != 0) return;
	m_mutex.Lock();
	if (!m_owner_exists || !m_asymmetric) return;
	m_requested.store(1, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_seq_cst);
	// The owner's `busy` store is visible now if it made it, and its next acquisition sees the request.
	(void)FlushProcessWriteBuffers();
	while (m_busy.load(std::memory_order_acquire) != 0) std::this_thread::yield();
}

void RenderMutex::RequesterUnlock() {
	if (--t_requester_depth != 0) return;
	m_requested.store(0, std::memory_order_release);
	m_mutex.Unlock();
}

// Any guest thread, at a write fault: true when the fault's write window was released (no image, no GPU-written page
// and all mapped in it).
bool RenderContext::TryWriteFaultWindow(uint64_t fault_vaddr) {
	constexpr uint64_t WindowSize = 4 * TRACKER_PAGE_SIZE;
	constexpr uint64_t StreamSize = 16 * TRACKER_PAGE_SIZE;
	constexpr uint64_t StreamNs   = 2'000'000;
	thread_local uint64_t last_end = 0, last_time = 0;
	const auto            window   = Common::AlignDown(fault_vaddr, WindowSize);
	const auto            now      = WindowClockNow();
	const auto            release  = [&](uint64_t size) {
		if (!GuestRange {window, size}.Valid() || !IsMapped(window, size) ||
		    m_buffer_cache.IsRegionGpuModified(window, size) || m_texture_cache.HasImagesInRegion(window, size)) {
			return false;
		}
		m_buffer_cache.InvalidateMemory(window, size);
		last_end  = window + size;
		last_time = now;
		return true;
	};
	return (window == last_end && now - last_time < StreamNs && release(StreamSize)) || release(WindowSize);
}

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this),
      m_command_scheduler(*this, graphics,
                          (Config::GpuPipelineStages() & Config::GPU_PIPELINE_RECORDING) != 0),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		// Cache tracking may still protect a page whose range left the GPU-mapped set; that
		// fault is ours to resolve, not a guest access violation.
		if (!GuestRange {fault_vaddr, fault_size}.Valid() || !m_page_manager.IsWatched(fault_vaddr)) {
			return false;
		}
	}
	// No watcher holds the page, so no cache expects this fault: the protection outlived its watchers (a release the
	// unmap left to the host, KYTY_UNMAP_PROTECT_SKIP, whose unmap did not complete) and would fault forever. It gets
	// the guest's own protection back.
	if (access == PageFaultAccess::Write && UnmapProtectSkipEnabled() &&
	    m_page_manager.RestoreIfUnwatched(fault_vaddr)) {
		return true;
	}
	if (access == PageFaultAccess::Write && GuestGpu::IsGpuThread()) {
		// A write of the command stream: device-address readers after it must see it.
		m_bda_generation++;
	}
	if (access == PageFaultAccess::Write && WriteFaultWindowEnabled() && TryWriteFaultWindow(fault_vaddr)) {
		return true;
	}
	if (access == PageFaultAccess::Write) {
		m_buffer_cache.InvalidateFaultedPage(fault_vaddr);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

void RenderContext::PrepareGpuWrite(const void* destination, uint64_t size) noexcept {
	const auto vaddr     = reinterpret_cast<uint64_t>(destination);
	const auto page_size = m_page_manager.GetPageSize();
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	// A watched page is protected: the write would fault into HandleFault().
	for (auto page = vaddr & ~(page_size - 1); page < vaddr + size; page += page_size) {
		if (m_page_manager.IsWatched(page)) {
			(void)HandleFault(PageFaultAccess::Write, std::max(page, vaddr));
		}
	}
}

bool RenderContext::WriteAroundGpuWrites(void* destination, const void* data, uint64_t size) {
	const auto vaddr = reinterpret_cast<uint64_t>(destination);
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	if (m_texture_cache.IsRegionGpuModified(vaddr, size)) {
		// A write over memory a GPU-written image holds: the faulting path reads the GPU's
		// buffer writes of the page back (draining the GPU) and leaves the image CPU-dirty. The
		// write-around keeps the page GPU-owned instead, which needs no readback, and the image
		// becomes CPU-dirty all the same.
		if (!m_buffer_cache.WriteAroundGpuWrites(vaddr, data, size)) {
			return false;
		}
		m_texture_cache.InvalidateMemory(vaddr, size);
		return true;
	}
	return m_buffer_cache.WriteAroundGpuWrites(vaddr, data, size);
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemoryForWrite(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::InvalidateImages(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

uint64_t RenderContext::MappedRangesVersion() const noexcept {
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges_version;
}

bool RenderContext::UnmappedSince(uint64_t version, uint64_t vaddr, uint64_t size) const {
	std::shared_lock lock(m_mapped_ranges_mutex);
	if (version == m_mapped_ranges_version) {
		return false;
	}
	if (version < m_unmap_log_floor) {
		return true; // unmaps since then left the log
	}
	for (auto it = m_unmap_log.rbegin(); it != m_unmap_log.rend() && it->version > version; ++it) {
		if (vaddr < it->vaddr + it->size && it->vaddr < vaddr + size) {
			return true;
		}
	}
	return false;
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_mapped_ranges_version++;
	m_bda_mapped_since_sync.emplace_back(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size, bool releasing) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size, releasing] {
		// Released memory loses its host protection with the mapping: the watchers removed below need not restore it
		// page run by page run first.
		struct UnmappingRange {
			explicit UnmappingRange(uint64_t begin, uint64_t end) { PageManager::SetUnmappingRange(begin, end); }
			~UnmappingRange() { PageManager::SetUnmappingRange(0, 0); }
		};
		std::optional<UnmappingRange> unmapping;
		if (releasing && UnmapProtectSkipEnabled()) {
			unmapping.emplace(vaddr, vaddr + size);
		}
		// Finishing the GPU protects GPU writes to the range and completions that may write guest
		// memory there (downloads, write scans). Queued GPU work only reads cached copies and
		// host-only completions (cache bookkeeping, flips, interrupts) never touch the range, so
		// without GPU-owned pages, images or pending guest completions writing the range the
		// unmap needs no drain.
		if (m_command_scheduler.Active()) {
			m_command_scheduler.PopPendingOperations();
		}
		const bool drain = m_command_scheduler.Active() &&
		                   (m_buffer_cache.IsRegionGpuModified(vaddr, size) ||
		                    m_texture_cache.HasImagesInRegion(vaddr, size) ||
		                    m_command_scheduler.HasPendingGuestOperations(vaddr, size));
		if (drain) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_occlusion_queries.OnUnmap(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
		m_mapped_ranges_version++;
		if (m_unmap_log.size() >= 4096) {
			m_unmap_log_floor = m_unmap_log.front().version;
			m_unmap_log.pop_front();
		}
		m_unmap_log.push_back({m_mapped_ranges_version, vaddr, size});
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	// The guest unmaps memory once the GPU work that used it executed: it waits on labels the
	// execution thread writes after the operations before them. The operations queued since do not
	// use the range, so the unmap runs before them instead of making the guest wait for them;
	// streaming games unmap many times per frame, holding their own locks.
	m_gpu->SendUrgentCommandSync(unmap);
}

void RenderContext::CacheDmaBases(const ShaderStageRuntime& runtime) {
	const auto&      program   = *runtime.program;
	const auto&      user_data = runtime.resources->user_data;
	std::shared_lock lock(m_mapped_ranges_mutex);
	for (const auto reg: program.info.dma_base_registers) {
		const auto index = static_cast<uint64_t>(reg) - program.user_data_base;
		if (reg < program.user_data_base || index + 1u >= user_data.size()) {
			continue;
		}
		const auto base = user_data[index] | (static_cast<uint64_t>(user_data[index + 1u]) << 32u);
		// The registers of an access that never runs may hold stale data. Mapped memory never
		// reaches the top of the address space, so a window that would overflow is skipped.
		if (base > UINT64_MAX - BufferCache::CACHING_PAGESIZE) {
			continue;
		}
		// DMA reaches only memory with a cached buffer; any other access records a fault and
		// reads zero, and the buffer arrives after the shader has run. That loses data the guest
		// writes for a single dispatch, such as the glyph bitmaps GTA V copies into its font atlas.
		m_mapped_ranges.ForEachInRange(base, BufferCache::CACHING_PAGESIZE,
		                               [this](uint64_t start, uint64_t end) {
			                               (void)m_buffer_cache.FindBuffer(start, end - start);
		                               });
	}
}

// chenxiao07's BDA read ranges (KYTY_BDA_READ_RANGES, default on; 0: a program whose DMA footprint the tracking pass did
// not bound synchronizes all cached memory): a program whose footprint it did not bound may still have a read plan.
static bool BdaReadRangesEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BDA_READ_RANGES");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool RenderContext::SynchronizeDmaFootprint(std::span<const ShaderStageRuntime* const> stages) {
	// Larger V# ranges are no buffers a draw reads (see FindBuffers()).
	constexpr uint64_t MaxBufferBytes = uint64_t {256} * 1024 * 1024;
	// An index offset or thread ID added to a V# record stays within a wave of records past it.
	constexpr uint64_t WaveRecords = 64;
	m_dma_spans.clear();
	// (Reused per thread: every operation with an unbounded footprint evaluates its plan.)
	thread_local std::vector<GuestRange> plan_ranges;
	std::shared_lock lock(m_mapped_ranges_mutex);
	const auto add = [&](uint64_t address, uint64_t size) {
		if (size != 0 && address <= UINT64_MAX - size) {
			m_mapped_ranges.ForEachInRange(address, size, [&](uint64_t start, uint64_t end) {
				m_dma_spans.emplace_back(start, end);
			});
		}
	};
	for (const auto* stage: stages) {
		const auto& program = *stage->program;
		if (!program.info.uses_dma) {
			continue;
		}
		if (!program.info.dma_bounded) {
			// (chenxiao07's read plan, KYTY_BDA_READ_RANGES: what the shader's address loads can reach, from its user
			// data and flattened SRT; a plan that does not evaluate leaves the reach unknown.)
			if (BdaReadRangesEnabled() &&
			    ShaderRecompiler::IR::EvaluateBdaReadPlan(program.bda_read_plan, *stage->resources, plan_ranges)) {
				for (const auto& range: plan_ranges) {
					add(range.address, range.size);
				}
				continue;
			}
			return false;
		}
		const auto& user_data = stage->resources->user_data;
		const auto  base_of   = [&](uint32_t reg, uint64_t& base) {
			const auto index = static_cast<uint64_t>(reg) - program.user_data_base;
			if (reg < program.user_data_base || index + 1u >= user_data.size()) {
				return false;
			}
			base = user_data[index] | (static_cast<uint64_t>(user_data[index + 1u]) << 32u);
			return true;
		};
		for (const auto& window: program.info.dma_windows) {
			uint64_t base = 0;
			if (!base_of(window.base_register, base)) {
				return false;
			}
			// Scalar address reads clear the low two bits of the base.
			const auto before = static_cast<uint64_t>(-window.first) + 4u;
			const auto start  = base > before ? base - before : 0;
			add(start, std::min(base, UINT64_MAX - window.last) + window.last - start);
		}
		for (const auto& table: program.info.dma_tables) {
			uint64_t base = 0;
			if (!base_of(table.base_register, base)) {
				return false;
			}
			const auto entries = (base & ~uint64_t {3}) + static_cast<uint64_t>(table.immediate);
			for (uint32_t offset = table.offset_bits;; offset = (offset - 1u) & table.offset_bits) {
				// The V# the shader would read, from guest memory that the GPU did not write.
				const auto address = entries + offset;
				if (!m_mapped_ranges.Contains(address, 12) ||
				    m_buffer_cache.IsRegionGpuModified(address, 12)) {
					return false;
				}
				ShaderBufferResource descriptor;
				std::memcpy(descriptor.fields, reinterpret_cast<const void*>(address), 12);
				const auto stride = uint64_t {descriptor.Stride()};
				if (descriptor.Base48() != 0 && descriptor.NumRecords() != 0) {
					add(descriptor.Base48(),
					    std::min(descriptor.GetSize() + std::max<uint64_t>(stride, 4u) * WaveRecords,
					             MaxBufferBytes));
				}
				if (offset == 0) {
					break;
				}
			}
		}
	}
	lock.unlock();
	m_buffer_cache.SynchronizeBuffersInSpans(m_dma_spans);
	return true;
}

void RenderContext::PrepareBda(bool may_write, bool full_sync) {
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	m_buffer_cache.PrepareFaultBuffer();
	if (may_write) {
		m_buffer_cache.NoteDeviceAddressUse();
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	// Nothing to upload when no page became CPU dirty, no buffer was registered and no range was
	// mapped since the last synchronization started.
	const std::array<uint64_t, 3> epochs {RegionManager::CpuDirtyEpoch(),
	                                      m_buffer_cache.RegisterEpoch(), m_mapped_ranges_version};
	// The guest's CPU writes that the work of a submission may read through device addresses
	// precede the submission, or a WAIT_REG_MEM that the work waited for; the GPU sees later ones
	// only after such a wait. So once a synchronization ran for a submission, pages the CPU
	// dirties afterwards wait for the next submission or wait (they stay dirty meanwhile),
	// unless the execution thread itself wrote them for the command stream (labels, CP writes)
	// or the buffers or mappings changed.
	const bool same_layout = epochs[1] == m_bda_sync_epochs[1] && epochs[2] == m_bda_sync_epochs[2];
	if (full_sync && epochs != m_bda_sync_epochs &&
	    (!same_layout || m_bda_synced_generation != m_bda_generation)) {
		m_bda_synced_generation = m_bda_generation;
		// A new buffer or mapped range may cover CPU-dirty pages that earlier passes skipped: the
		// buffer cache forces the regions of new buffers, and of the ranges mapped since. Only this
		// thread takes the ranges out; MapMemory() adds them under the exclusive lock.
		for (const auto& [vaddr, size]: m_bda_mapped_since_sync) {
			m_buffer_cache.ForceSynchronization(vaddr, size);
		}
		m_bda_mapped_since_sync.clear();
		m_buffer_cache.SynchronizeBuffersInRanges(m_mapped_ranges, false);
		m_bda_sync_epochs = epochs;
	}
	m_fault_process_pending = true;
	m_use_scan_pending      = true;
}

void RenderContext::RunGarbageCollector() {
	m_frame_number++;
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	constexpr uint32_t UseScanPeriod = 256;
	if (m_use_scan_countdown != 0) {
		m_use_scan_countdown--;
	} else if (m_use_scan_pending) {
		m_use_scan_pending   = false;
		m_use_scan_countdown = UseScanPeriod;
		m_buffer_cache.ProcessUseBuffer();
	}
	if (m_indirect_write_tables.TakePendingWrites()) {
		m_buffer_cache.ProcessWriteBuffer([this](std::span<const uint64_t> pages) {
			m_indirect_write_tables.NoteWrites(pages);
		});
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
