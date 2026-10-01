#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/profiler.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/hackNotice.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <mutex>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace Libs::Graphics {

// vkQueueSubmit, and the queue lock that present also holds, runs on a queue thread instead
// of the GPU thread. Frames reach the presenter only after their flip tick completed, and the
// presenter's own scheduler stays synchronous. KYTY_ASYNC_SUBMIT=0 submits on the GPU thread.
// KYTY_RECORD_THREAD=0: no recording thread (CommandScheduler::EnableRecordingThread), which
// submits too. Done while the members are constructed: the BufferCache begins the first command
// buffer (it clears the BDA tables), and the recording thread has to be enabled before that.
static CommandScheduler& EnableSchedulerThreads(CommandScheduler& scheduler) {
	const char* record_thread = std::getenv("KYTY_RECORD_THREAD");
	const char* async_submit  = std::getenv("KYTY_ASYNC_SUBMIT");
	if (record_thread == nullptr || std::strcmp(record_thread, "0") != 0) {
		scheduler.EnableRecordingThread();
	} else if (async_submit == nullptr || std::strcmp(async_submit, "0") != 0) {
		scheduler.EnableAsyncSubmit();
	}
	return scheduler;
}

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, EnableSchedulerThreads(m_command_scheduler).GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_bindless_table(graphics, m_command_scheduler, m_page_manager) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_texture_cache.on_bindless_unregister = [this](ImageId id) {
		m_bindless_table.OnImageUnregistered(id);
	};
	m_texture_cache.on_image_release = [this](const Image& image) {
		if (!m_bindless_table.Enabled()) {
			return false;
		}
		std::vector<vk::ImageView> views;
		views.reserve(image.views.size());
		for (const auto& cached: image.views) {
			views.push_back(cached.view);
		}
		return m_bindless_table.ReleaseViews(views);
	};
	LOGF("Recording thread: %s, async queue submit: %s\n",
	     m_command_scheduler.Threaded() ? "on" : "off",
	     m_command_scheduler.AsyncSubmit() ? "on" : "off");
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void WarmUpLocalMemory(GraphicContext& graphics, CommandScheduler& scheduler);

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	WarmUpLocalMemory(m_graphics, m_command_scheduler);
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
			(KYTY_SUBMIT_REASON(), m_command_scheduler.Finish());
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
	if (access == PageFaultAccess::Write) {
		// Diagnostics: KYTY_TRACE_WRITERS=<hex address>:<hex size>: guest CPU write faults in that
		// range ([cpu-write], first 200), with whether the page holds GPU-written bytes.
		static const std::pair<uint64_t, uint64_t> trace = [] {
			std::pair<uint64_t, uint64_t> range {0, 0};
			if (const char* value = std::getenv("KYTY_TRACE_WRITERS"); value != nullptr) {
				char* end    = nullptr;
				range.first  = std::strtoull(value, &end, 16);
				range.second = (end != nullptr && *end == ':') ? std::strtoull(end + 1, nullptr, 16) : 4u;
			}
			return range;
		}();
		static std::atomic<uint32_t> traced {0};
		if (trace.first != 0 && fault_vaddr >= trace.first && fault_vaddr < trace.first + trace.second &&
		    traced.fetch_add(1) < 200u) {
			std::fprintf(stderr, "[cpu-write] 0x%012llx frame=%llu gpu_dirty_page=%d\n",
			             static_cast<unsigned long long>(fault_vaddr),
			             static_cast<unsigned long long>(GetGraphics().presented_frames.load()),
			             m_buffer_cache.HasGpuDirtyBytes(fault_vaddr & ~uint64_t {0xfff}, 0x1000) ? 1 : 0);
		}
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_bindless_table.OnCpuWrite(fault_vaddr, fault_size);
	} else {
		KYTY_PROFILER_BLOCK("ReadMemory: read fault");
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::CanServeCleanRead(uint64_t fault_vaddr, uint64_t vaddr,
                                      uint64_t size) const noexcept {
	return IsMapped(vaddr, size) && m_page_manager.IsReadWatched(fault_vaddr) &&
	       m_buffer_cache.IsCleanForConcurrentRead(vaddr, size);
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

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_bindless_table.OnCpuWrite(vaddr, size);
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

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_buffer_cache.QueueBdaSync(vaddr, size);
	m_mapped_ranges_version++;
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	// The kernel unmaps every free range it is about to (re)map, most of which were never GPU
	// mapped. Buffers, images and GPU-dirty pages are only tracked inside mapped ranges (faults
	// and invalidations outside them are ignored), so such an unmap has nothing to invalidate:
	// skip the GPU-thread round trip and the full drain it forces. KYTY_UNMAP_SKIP_UNMAPPED=0
	// sends every unmap to the GPU thread again.
	static const bool skip_unmapped = [] {
		const auto* value = std::getenv("KYTY_UNMAP_SKIP_UNMAPPED");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	if (skip_unmapped && GuestRange {vaddr, size}.Valid()) {
		std::shared_lock lock(m_mapped_ranges_mutex);
		if (!m_mapped_ranges.Intersects(vaddr, size)) {
			return;
		}
	}
	const auto unmap = [this, vaddr, size] {
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
			(KYTY_SUBMIT_REASON(), m_command_scheduler.Finish());
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.UnmapShared(vaddr, size);
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_bindless_table.UnwatchRange(vaddr, size);
		m_occlusion_queries.OnUnmap(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
		m_mapped_ranges_version++;
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
	// KYTY_FPS_LOG: time the GPU-thread round trips of unmaps every 5 s.
	static const bool stats = std::getenv("KYTY_FPS_LOG") != nullptr;
	const auto        begin = std::chrono::steady_clock::now();
	m_gpu->SendUrgentCommandSync(unmap);
	if (stats) {
		static std::mutex mutex;
		static double     total_ms = 0.0, max_ms = 0.0;
		static uint64_t   count    = 0;
		static auto       last     = std::chrono::steady_clock::now();
		const auto        now      = std::chrono::steady_clock::now();
		const double      ms = std::chrono::duration<double, std::milli>(now - begin).count();
		std::lock_guard   lock(mutex);
		total_ms += ms;
		max_ms = std::max(max_ms, ms);
		count++;
		if (now - last > std::chrono::seconds(5)) {
			std::fprintf(stderr, "[unmap-sync] %llu GPU-thread unmaps in %.1f s: %.1f ms total, max %.1f ms\n",
			             static_cast<unsigned long long>(count),
			             std::chrono::duration<double>(now - last).count(), total_ms, max_ms);
			total_ms = max_ms = 0.0;
			count             = 0;
			last              = now;
		}
	}
}

void RenderContext::UnmapMemoryRanges(std::span<const std::pair<uint64_t, uint64_t>> ranges) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion\n");
	}
	std::vector<std::pair<uint64_t, uint64_t>> mapped;
	{
		std::shared_lock lock(m_mapped_ranges_mutex);
		for (const auto& [vaddr, size]: ranges) {
			if (GuestRange {vaddr, size}.Valid() && m_mapped_ranges.Intersects(vaddr, size)) {
				mapped.emplace_back(vaddr, size);
			}
		}
	}
	if (mapped.empty()) {
		return;
	}
	const auto unmap = [this, &mapped] {
		// As UnmapMemory: drain only for GPU-owned pages, images or pending guest completions.
		bool drain = false;
		if (m_command_scheduler.Active()) {
			m_command_scheduler.PopPendingOperations();
			for (const auto& [vaddr, size]: mapped) {
				drain |= m_buffer_cache.IsRegionGpuModified(vaddr, size) ||
				         m_texture_cache.HasImagesInRegion(vaddr, size) ||
				         m_command_scheduler.HasPendingGuestOperations(vaddr, size);
			}
		}
		if (drain) {
			const auto tick = m_command_scheduler.CurrentTick();
			(KYTY_SUBMIT_REASON(), m_command_scheduler.Finish());
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		for (const auto& [vaddr, size]: mapped) {
			m_buffer_cache.UnmapShared(vaddr, size);
			m_buffer_cache.InvalidateMemory(vaddr, size);
			m_texture_cache.UnmapMemory(vaddr, size);
			m_bindless_table.UnwatchRange(vaddr, size);
		}
		std::lock_guard lock(m_mapped_ranges_mutex);
		for (const auto& [vaddr, size]: mapped) {
			m_mapped_ranges.Subtract(vaddr, size);
		}
	};
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
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

void RenderContext::PrepareBda() {
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	m_buffer_cache.PrepareFaultBuffer();
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

// Diagnostics: KYTY_DUMP_GUEST=<frame>:<address>+<size>[,<address>+<size>...] (frame decimal,
// the rest hex) writes guest memory, with the GPU's writes downloaded first, to
// dump_<frame>_<address>.bin once that many frames have been presented. A frame written +<n>
// counts from the first use of the KYTY_WATCH_SHADER shader.
static void DumpGuestMemory(GraphicContext& graphics, BufferCache& cache) {
	struct Request {
		uint64_t                                   frame    = 0;
		bool                                       relative = false;
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
	};
	static const Request request = [] {
		Request     result;
		const char* value = std::getenv("KYTY_DUMP_GUEST");
		if (value == nullptr) {
			return result;
		}
		char* end       = nullptr;
		result.relative = *value == '+';
		result.frame    = std::strtoull(result.relative ? value + 1 : value, &end, 10);
		value        = end != nullptr && *end == ':' ? end + 1 : nullptr;
		while (value != nullptr && *value != '\0') {
			const auto address = std::strtoull(value, &end, 16);
			uint64_t   size    = 4;
			if (end != nullptr && *end == '+') {
				size = std::strtoull(end + 1, &end, 16);
			}
			result.ranges.emplace_back(address, size);
			value = end != nullptr && *end == ',' ? end + 1 : nullptr;
		}
		return result;
	}();
	static bool done = false;
	if (done || request.ranges.empty()) {
		return;
	}
	const auto first = WatchedShaderFirstFrame();
	if (request.relative && first == UINT64_MAX) {
		return;
	}
	const auto frame = request.relative ? first + request.frame : request.frame;
	if (graphics.presented_frames.load(std::memory_order_relaxed) < frame) {
		return;
	}
	done = true;
	for (const auto& [address, size]: request.ranges) {
		cache.DownloadRangeForDiagnostics(address, size);
		std::vector<uint8_t> bytes(size);
		const bool read = LibKernel::Memory::TryReadBacking(address, bytes.data(), size);
		const auto name = "dump_" + std::to_string(frame) + "_" +
		                  std::to_string(address) + ".bin";
		if (read) {
			if (auto* file = std::fopen(name.c_str(), "wb"); file != nullptr) {
				std::fwrite(bytes.data(), 1, bytes.size(), file);
				std::fclose(file);
			}
		}
		LOGF("DumpGuest: 0x%016" PRIx64 "+0x%" PRIx64 " -> %s (%s)\n", address, size, name.c_str(),
		     read ? "written" : "not readable");
	}
}

void RenderContext::RunGarbageCollector() {
	DumpGuestMemory(GetGraphics(), m_buffer_cache);
	LogWatchedRereads(m_buffer_cache, GetGraphics().presented_frames.load(std::memory_order_relaxed));
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
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
