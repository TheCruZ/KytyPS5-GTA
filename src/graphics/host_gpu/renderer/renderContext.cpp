#include "common/syncAcct.h"
#include <cinttypes>
#include <cstdio>
#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/ablation.h"
#include "graphics/host_gpu/gpuProfiler.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/cache/cacheStats.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/shader/shaderBindings.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

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
	GpuProfiler::TimedWait timed(access == PageFaultAccess::Write ? "ev: guest write fault"
	                                                              : "ev: guest read fault");
	KYTY_SYNC_SITE(s_wfault, "fault: GPU-tracked page write", Tax);
	KYTY_SYNC_SITE(s_rfault, "fault: GPU-tracked page read (readback)", Tax);
	Common::SyncAcct::Scope sync_scope(access == PageFaultAccess::Write ? s_wfault : s_rfault, fault_vaddr);
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		// Cache tracking may still protect a page whose range left the GPU-mapped set; that
		// fault is ours to resolve, not a guest access violation.
		if (!GuestRange {fault_vaddr, fault_size}.Valid() || !m_page_manager.IsWatched(fault_vaddr)) {
			std::printf("HandleFault: untracked fault 0x%016" PRIx64
			            " outside the GPU-mapped ranges (access=%u backing=%d)\n",
			            fault_vaddr, static_cast<uint32_t>(access),
			            LibKernel::Memory::IsBackingReadable(fault_vaddr, 1) ? 1 : 0);
			std::fflush(stdout);
			return false;
		}
		static std::atomic_uint32_t logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			std::printf("HandleFault: tracked page 0x%016" PRIx64
			            " is outside the GPU-mapped ranges (access=%u)\n",
			            fault_vaddr, static_cast<uint32_t>(access));
			std::fflush(stdout);
		}
	}
	if (access == PageFaultAccess::Write && GuestGpu::IsGpuThread()) {
		// A write of the command stream: device-address readers after it must see it.
		m_bda_generation++;
	}
	CacheStats::Add(access == PageFaultAccess::Write ? CacheStats::WriteFaults
	                                                 : CacheStats::ReadFaults);
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
	m_mapped_ranges_version++;
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	GpuProfiler::TimedWait timed("ev: unmap");
	KYTY_SYNC_SITE(s_unmap, "game: GPU unmap (urgent, waits exec)", Work);
	Common::SyncAcct::Scope sync_scope(s_unmap, vaddr, size);
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
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
		if (drain && GpuProfiler::PhaseEnabled()) {
			// Debug: why unmaps drain the GPU.
			GpuProfiler::AddCount(m_buffer_cache.IsRegionGpuModified(vaddr, size) ? "unmap drain: gpu pages" : "unmap drain: no gpu pages");
			GpuProfiler::AddCount(m_texture_cache.HasImagesInRegion(vaddr, size) ? "unmap drain: images" : "unmap drain: no images");
			GpuProfiler::AddCount(m_command_scheduler.HasPendingGuestOperations(vaddr, size) ? "unmap drain: guest ops" : "unmap drain: no guest ops");
		}
		if (drain) {
			const auto tick = m_command_scheduler.CurrentTick();
			GpuProfiler::WaitTagScope tag("unmap Finish");
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
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
	static const bool urgent = [] {
		const char* env = std::getenv("KYTY_URGENT_UNMAP");
		return env == nullptr || env[0] != '0';
	}();
	if (urgent) {
		m_gpu->SendUrgentCommandSync(unmap);
	} else {
		m_gpu->SendCommandSync(unmap);
	}
}

void RenderContext::CacheDmaBases(const ShaderStageRuntime& runtime) {
	const auto&      program   = *runtime.program;
	const auto&      user_data = runtime.resources->user_data;
	if (GpuProfiler::PhaseEnabled()) {
		// Debug (rain-perf): which programs use device addresses.
		char tag[64];
		std::snprintf(tag, sizeof(tag), "dma prog %016llx st%u", static_cast<unsigned long long>(program.shader_hash),
		              static_cast<unsigned>(program.stage));
		GpuProfiler::AddWaitString(tag, 0);
	}
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
		{
			// Debug (rain-perf): the DMA bases of the first draws.
			static std::unordered_map<uint64_t, int> printed;
			if (GpuProfiler::PhaseEnabled() && printed[program.shader_hash]++ < 3) {
				std::printf("DMA base: prog %016llx reg %u base 0x%llx (user data %zu dwords) "
				            "bounded %d windows %zu tables %zu\n",
				            static_cast<unsigned long long>(program.shader_hash), reg,
				            static_cast<unsigned long long>(base), user_data.size(),
				            program.info.dma_bounded ? 1 : 0, program.info.dma_windows.size(),
				            program.info.dma_tables.size());
				for (const auto& w: program.info.dma_windows) {
					std::printf("  window reg %u [%lld, %llu)\n", w.base_register,
					            static_cast<long long>(w.first), static_cast<unsigned long long>(w.last));
				}
				for (const auto& t: program.info.dma_tables) {
					std::printf("  table reg %u imm %lld bits 0x%x\n", t.base_register,
					            static_cast<long long>(t.immediate), t.offset_bits);
				}
			}
		}
		m_mapped_ranges.ForEachInRange(base, BufferCache::CACHING_PAGESIZE,
		                               [this](uint64_t start, uint64_t end) {
			                               (void)m_buffer_cache.FindBuffer(start, end - start);
		                               });
	}
}

bool RenderContext::SynchronizeDmaFootprint(std::span<const ShaderStageRuntime* const> stages) {
	// Larger V# ranges are no buffers a draw reads (see FindBuffers()).
	constexpr uint64_t MaxBufferBytes = uint64_t {256} * 1024 * 1024;
	// An index offset or thread ID added to a V# record stays within a wave of records past it.
	constexpr uint64_t WaveRecords = 64;
	m_dma_spans.clear();
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
	GpuProfiler::AddCount("dma: footprint spans", m_dma_spans.size());
	const auto dbg_start = std::chrono::steady_clock::now();
	m_buffer_cache.SynchronizeBuffersInSpans(m_dma_spans);
	if (GpuProfiler::PhaseEnabled()) {
		// Debug (rain-perf): cost and reach of the footprint synchronization per program.
		uint64_t bytes = 0;
		for (const auto& [first, last]: m_dma_spans) bytes += last - first;
		char tag[96];
		std::snprintf(tag, sizeof(tag), "dma fp %016llx", static_cast<unsigned long long>(
		                                                     stages.empty() ? 0 : stages[0]->program->shader_hash));
		GpuProfiler::AddWaitString(tag, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                                         std::chrono::steady_clock::now() - dbg_start)
		                                                         .count()));
		std::snprintf(tag, sizeof(tag), "dma fp bytes %016llx", static_cast<unsigned long long>(
		                                                           stages.empty() ? 0 : stages[0]->program->shader_hash));
		GpuProfiler::AddWaitString(tag, bytes); // "ms/frame" = MB/frame
	}
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
	const auto       begin = std::chrono::steady_clock::now();
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
	// Debug (rain-perf ablation): no BDA synchronization (device-address reads see stale data).
	if (full_sync && epochs != m_bda_sync_epochs && !KYTY_ABL_AT(NO_BDA_SYNC) &&
	    (!same_layout || m_bda_synced_generation != m_bda_generation)) {
		m_bda_synced_generation = m_bda_generation;
		// A new buffer or mapped range may cover CPU-dirty pages that earlier passes skipped.
		const bool all = epochs[1] != m_bda_sync_epochs[1] || epochs[2] != m_bda_sync_epochs[2];
		// Debug: cost of full and incremental passes.
		GpuProfiler::TimedWait timed_sync(all ? (epochs[2] != m_bda_sync_epochs[2] ? "bda: full sync (mapping)" : "bda: full sync (new buffer)")
		                                      : "bda: incremental sync");
		m_buffer_cache.SynchronizeBuffersInRanges(m_mapped_ranges, all);
		m_bda_sync_epochs = epochs;
	}
	m_fault_process_pending = true;
	m_use_scan_pending      = true;
	CacheStats::Add(CacheStats::BdaSyncs);
	CacheStats::Add(CacheStats::BdaSyncUs,
	                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                          std::chrono::steady_clock::now() - begin)
	                                          .count()));
}

static void PrintCacheStats() {
	static const bool enabled = std::getenv("KYTY_CACHE_STATS") != nullptr;
	if (!enabled) {
		return;
	}
	static auto last = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) {
		return;
	}
	const auto seconds = std::chrono::duration<double>(now - last).count();
	last               = now;
	// Deltas from a snapshot: the hitch report of the phase profiler reads the same counters.
	static uint64_t previous[CacheStats::Count] {};
	uint64_t        values[CacheStats::Count] {};
	for (uint32_t i = 0; i < CacheStats::Count; i++) {
		const auto value = CacheStats::g_counters[i].load(std::memory_order_relaxed);
		values[i]        = value - previous[i];
		previous[i]      = value;
	}
	std::printf("CacheStats %.1fs: submits=%" PRIu64 " wfaults=%" PRIu64 " rfaults=%" PRIu64
	            " readbacks=%" PRIu64 " readback_wait_ms=%" PRIu64 " bda_syncs=%" PRIu64
	            " bda_sync_ms=%" PRIu64 " bda_buffers=%" PRIu64 " tex_uploads=%" PRIu64
	            " tex_upload_mb=%" PRIu64 " tex_gc_frees=%" PRIu64 " tex_gc_dropped_gpu=%" PRIu64
	            " buf_gc_frees=%" PRIu64 " indirect_targets=%" PRIu64 " indirect_dispatches=%" PRIu64
	            " indirect_prepare_ms=%" PRIu64 " readbacks_in_indirect=%" PRIu64
	            " tex_uploads_from_buffer=%" PRIu64 " tex_uploads_in_indirect=%" PRIu64 "\n",
	            seconds, values[CacheStats::Submissions], values[CacheStats::WriteFaults],
	            values[CacheStats::ReadFaults], values[CacheStats::ReadbackDownloads],
	            values[CacheStats::ReadbackWaitUs] / 1000, values[CacheStats::BdaSyncs],
	            values[CacheStats::BdaSyncUs] / 1000, values[CacheStats::BdaSyncBuffers],
	            values[CacheStats::TextureUploads], values[CacheStats::TextureUploadBytes] >> 20u,
	            values[CacheStats::TextureGcFrees], values[CacheStats::TextureGcGpuDataDropped],
	            values[CacheStats::BufferGcFrees], values[CacheStats::IndirectTargets],
	            values[CacheStats::IndirectDispatches], values[CacheStats::IndirectPrepareUs] / 1000,
	            values[CacheStats::ReadbackIndirect], values[CacheStats::TextureUploadsFromBuffer],
	            values[CacheStats::TextureUploadsInIndirect]);
	std::printf("CacheStats marked: fill=%" PRIu64 " copy=%" PRIu64 " storage=%" PRIu64
	            " storage_dynamic=%" PRIu64 " indirect=%" PRIu64 " | readbacks storage=%" PRIu64
	            " dynamic=%" PRIu64 " dma=%" PRIu64 " | storage_over_gpu_image w=%" PRIu64
	            " rmw=%" PRIu64 "\n",
	            values[CacheStats::ImagesMarkedFill], values[CacheStats::ImagesMarkedCopy],
	            values[CacheStats::ImagesMarkedStorage],
	            values[CacheStats::ImagesMarkedStorageDynamic],
	            values[CacheStats::ImagesMarkedIndirect], values[CacheStats::ReadbackStorage],
	            values[CacheStats::ReadbackDynamic], values[CacheStats::ReadbackDma],
	            values[CacheStats::StorageWriteOverGpuImage],
	            values[CacheStats::StorageRmwOverGpuImage]);
	std::fflush(stdout);
}

void RenderContext::RunGarbageCollector() {
	CacheStats::Add(CacheStats::Submissions);
	PrintCacheStats();
	if (GpuProfiler::PhaseEnabled()) {
		static auto last = std::chrono::steady_clock::now();
		if (std::chrono::steady_clock::now() - last > std::chrono::seconds(10)) {
			last = std::chrono::steady_clock::now();
			m_graphics.LogAllocationCounts();
		}
	}
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
