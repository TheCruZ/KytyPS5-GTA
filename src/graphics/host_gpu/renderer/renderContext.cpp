#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/shader/shaderBindings.h"
#include "libs/errno.h"

#include <algorithm>
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
	if (access == PageFaultAccess::Write && GuestGpu::IsGpuThread()) {
		// A write of the command stream: device-address readers after it must see it.
		m_bda_generation++;
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
		// A new buffer or mapped range may cover CPU-dirty pages that earlier passes skipped.
		const bool all = epochs[1] != m_bda_sync_epochs[1] || epochs[2] != m_bda_sync_epochs[2];
		m_buffer_cache.SynchronizeBuffersInRanges(m_mapped_ranges, all);
		m_bda_sync_epochs = epochs;
	}
	m_fault_process_pending = true;
	m_use_scan_pending      = true;
}

void RenderContext::RunGarbageCollector() {
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
