#include "common/hackNotice.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "gpu_tiler_shaders/fault_buffer_process_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr size_t MaxPageFaults       = 1024;
constexpr size_t PageFaultAreaSize   = MaxPageFaults * sizeof(uint64_t);
constexpr size_t PageFaultBitsetSize = BufferCache::CACHING_NUMPAGES / 8;
// Pages V#-table stores may write between two reads; more stay in the bitmap for the next read.
constexpr size_t MaxPageWrites       = 16384;
// Pages device-address accesses found cached between two reads; more stay for the next read.
constexpr size_t MaxPageUses         = 65536;

size_t DownloadAreaSize(const GraphicContext& graphics) {
	const auto& limits = graphics.physical_device_properties.limits;
	const auto  alignment =
	    std::max(limits.nonCoherentAtomSize, limits.minStorageBufferOffsetAlignment);
	return (PageFaultAreaSize + sizeof(ShaderTrapRecord) + alignment - 1) & ~(alignment - 1);
}

// A bitmap's readback area: the count, then the pages (`max_pages` at most).
size_t BitmapAreaSize(const GraphicContext& graphics, size_t max_pages) {
	const auto& limits = graphics.physical_device_properties.limits;
	const auto  alignment =
	    std::max(limits.nonCoherentAtomSize, limits.minStorageBufferOffsetAlignment);
	return ((max_pages + 1) * sizeof(uint64_t) + alignment - 1) & ~(alignment - 1);
}

size_t WriteAreaSize(const GraphicContext& graphics) {
	return BitmapAreaSize(graphics, MaxPageWrites);
}

} // namespace

FaultManager::FaultManager(GraphicContext& graphics, CommandScheduler& scheduler,
                           BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_buffer_cache(buffer_cache),
      m_download_area_size(DownloadAreaSize(graphics)),
      m_fault_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                     BufferCache::FAULT_BUFFER_SIZE),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
                        MaxPendingFaults * m_download_area_size),
      m_write_download(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
                       MaxPendingFaults * WriteAreaSize(graphics)),
      m_write_area_size(WriteAreaSize(graphics)),
      m_use_download(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
                     MaxPendingFaults * BitmapAreaSize(graphics, MaxPageUses)),
      m_use_area_size(BitmapAreaSize(graphics, MaxPageUses)) {
	SetVulkanObjectNameF(m_graphics.device, m_fault_buffer.Handle(), "Fault Buffer");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                                 &m_fault_process_desc_layout),
	                     "create fault-buffer descriptor layout");

	const auto module = CompileSPV(FAULT_BUFFER_PROCESS_SPV, m_graphics.device);

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &m_fault_process_desc_layout;
	RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                            &m_fault_process_pipeline_layout),
	                     "create fault-buffer pipeline layout");

	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_fault_process_pipeline_layout;
	const auto result    = m_graphics.device.createComputePipelines(
	    nullptr, 1, &pipeline_info, nullptr, &m_fault_process_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create fault-buffer pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_fault_process_pipeline, "Fault Buffer Parser");
}

FaultManager::~FaultManager() {
	m_graphics.device.destroyPipeline(m_fault_process_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_fault_process_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_fault_process_desc_layout, nullptr);
}

Buffer* FaultManager::GetFaultBuffer() noexcept {
	// Construction happens before the scheduler is active. Initialize on the GPU
	// thread at first binding, before any shader can read or claim the record.
	if (!m_initialized) {
		m_fault_buffer.Fill(0, m_fault_buffer.Size(), 0);
		m_initialized = true;
	}
	return &m_fault_buffer;
}

void FaultManager::ProcessFaultBuffer() {
	ProcessFaultBufferImpl(nullptr);
}

ShaderFaultReport FaultManager::CollectFaults() {
	ShaderFaultReport report;
	ProcessFaultBufferImpl(&report);
	// The report is owned by this stack frame; its deferred callback must finish
	// before returning. Newly populated pages are uploaded in the next recording.
	(KYTY_SUBMIT_REASON(), m_scheduler.Finish());
	return report;
}

void FaultManager::ProcessFaultBufferImpl(ShaderFaultReport* report) {
	KYTY_PROFILER_FUNCTION();
	(void)GetFaultBuffer();
	if (const auto wait_tick = m_fault_areas[m_current_area]; wait_tick != 0) {
		m_scheduler.Wait(wait_tick);
		m_scheduler.PopPendingOperations();
	}

	const auto offset = m_current_area * m_download_area_size;
	auto*      mapped = m_download_buffer.Mapped().data() + offset;
	std::memset(mapped, 0, m_download_area_size);
	m_download_buffer.Flush(offset, m_download_area_size);

	constexpr auto bitmap_size = BufferCache::CACHING_NUMPAGES / 8;
	vk::BufferMemoryBarrier2 pre_barrier {};
	pre_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	pre_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	pre_barrier.dstStageMask =
	    vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
	pre_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead |
	                            vk::AccessFlagBits2::eShaderWrite |
	                            vk::AccessFlagBits2::eTransferRead;
	pre_barrier.buffer        = m_fault_buffer.Handle();
	pre_barrier.offset        = 0;
	pre_barrier.size          = m_fault_buffer.Size();
	auto post_barrier         = pre_barrier;
	post_barrier.srcStageMask =
	    vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
	post_barrier.srcAccessMask =
	    vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferRead;
	post_barrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	post_barrier.dstAccessMask =
	    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;

	const vk::DescriptorBufferInfo infos[] {
	    {m_fault_buffer.Handle(), 0, PageFaultBitsetSize},
	    {m_download_buffer.Handle(), offset, PageFaultAreaSize},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	m_scheduler.EndRendering();
	auto               command = m_scheduler.Current().Recorder();
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &pre_barrier;
	command.pipelineBarrier2(dependency);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline_layout,
	                             0, writes);
	const auto num_threads    = BufferCache::CACHING_NUMPAGES / 32;
	const auto num_workgroups = (num_threads + 63) / 64;
	command.dispatch(static_cast<uint32_t>(num_workgroups), 1, 1);
	const vk::BufferCopy trap_copy {PageFaultBitsetSize, offset + PageFaultAreaSize,
	                                sizeof(ShaderTrapRecord)};
	command.copyBuffer(m_fault_buffer.Handle(), m_download_buffer.Handle(), trap_copy);
	dependency.pBufferMemoryBarriers = &post_barrier;
	command.pipelineBarrier2(dependency);
	vk::BufferMemoryBarrier2 download_barrier {};
	download_barrier.srcStageMask =
	    vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
	download_barrier.srcAccessMask =
	    vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
	download_barrier.dstStageMask    = vk::PipelineStageFlagBits2::eHost;
	download_barrier.dstAccessMask   = vk::AccessFlagBits2::eHostRead;
	download_barrier.buffer          = m_download_buffer.Handle();
	download_barrier.offset          = offset;
	download_barrier.size            = m_download_area_size;
	dependency.pBufferMemoryBarriers = &download_barrier;
	command.pipelineBarrier2(dependency);

	// Preserve the record in readback, then release it for the next attempt. Fill
	// supplies the transfer-read -> transfer-write -> shader-access dependencies.
	m_fault_buffer.Fill(PageFaultBitsetSize, sizeof(ShaderTrapRecord), 0);
	const auto area = m_current_area;
	// The fault scan only caches pages still mapped; it never writes guest memory, so an unmap
	// does not wait for it.
	m_scheduler.DeferHostOperation([this, mapped, offset, area, report] {
		m_download_buffer.Invalidate(offset, m_download_area_size);
		ShaderTrapRecord trap;
		std::memcpy(&trap, mapped + PageFaultAreaSize, sizeof(trap));
		if (trap.claimed != 0 && report == nullptr) {
			const auto  hash    = (uint64_t {trap.shader_hash_high} << 32) | trap.shader_hash_low;
			const auto* pending = std::bit_cast<const uint64_t*>(mapped);
			EXIT("GPU shader trap: hash=0x%016" PRIx64
			     " pc=0x%08x code=0x%02x pending_pages=%" PRIu64 " first_page=%016" PRIx64 "\n",
			     hash, trap.pc, trap.code, pending[0], pending[1]);
		}
		RangeSet    fault_ranges;
		const auto* faults = std::bit_cast<const uint64_t*>(mapped);
		if (report != nullptr) {
			report->page_count = faults[0];
			report->trap       = trap;
		} else if (faults[0] != 0) {
			m_unattributed_faults.fetch_add(1, std::memory_order_release);
		}
		// The parser's counter can exceed its bounded output array. Never consume
		// unrecorded addresses as pages (or read beyond the mapped readback area).
		const auto count = std::min<uint64_t>(faults[0], MaxPageFaults - 1);
		if (faults[0] > count) {
			LOGF("GPU page-fault report truncated: %" PRIu64 " entries, %" PRIu64 " recorded\n",
			     faults[0], count);
		}
		for (uint32_t index = 1; index <= count; ++index) {
			const auto address = BufferCache::GuestAddress(faults[index]);
			fault_ranges.Add(address, BufferCache::CACHING_PAGESIZE);
			LOGF("Accessed non-GPU cached memory at 0x%016" PRIx64 "\n", address);
		}
		fault_ranges.ForEach([this](uint64_t start, uint64_t end) {
			EXIT_IF(end - start > std::numeric_limits<uint32_t>::max());
			// The guest may have unmapped the range since the GPU accessed it: an unmap does not
			// wait for this completion.
			if (m_scheduler.Context().IsMapped(start, end - start)) {
				m_buffer_cache.CacheFaultedMemory(start, end - start);
			} else {
				Common::HackNotice::Report(
				    "device-address fault on memory the guest has unmapped ignored",
				    "FaultManager::ProcessFaultBuffer", 0, 0, "the access is not cached");
			}
		});
		if (report != nullptr) {
			fault_ranges.ForEach([this](uint64_t start, uint64_t end) {
				m_buffer_cache.SynchronizeBuffersInRange(start, end - start);
			});
		}
		m_fault_areas[area] = 0;
	});

	m_fault_areas[m_current_area++] = m_scheduler.CurrentTick();
	m_current_area %= MaxPendingFaults;
}

void FaultManager::ProcessBitmap(uint64_t bitmap_offset, Buffer& download, size_t area_size,
                                 std::array<uint64_t, MaxPendingFaults>& areas, uint32_t& current,
                                 PagesHandler&& handler, bool host_only) {
	KYTY_PROFILER_FUNCTION();
	(void)GetFaultBuffer();
	if (const auto wait_tick = areas[current]; wait_tick != 0) {
		m_scheduler.Wait(wait_tick);
		m_scheduler.PopPendingOperations();
	}

	const auto offset = current * area_size;
	auto*      mapped = download.Mapped().data() + offset;
	// The parser appends after the count; the host reads only the entries it counted.
	std::memset(mapped, 0, sizeof(uint64_t));
	download.Flush(offset, sizeof(uint64_t));

	vk::BufferMemoryBarrier2 pre_barrier {};
	pre_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	pre_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	pre_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	pre_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
	pre_barrier.buffer        = m_fault_buffer.Handle();
	pre_barrier.offset        = bitmap_offset;
	pre_barrier.size          = PageFaultBitsetSize;
	auto post_barrier          = pre_barrier;
	post_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	post_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	post_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	post_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderWrite;

	const vk::DescriptorBufferInfo infos[] {
	    {m_fault_buffer.Handle(), bitmap_offset, PageFaultBitsetSize},
	    {download.Handle(), offset, area_size},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	m_scheduler.EndRendering();
	auto               command = m_scheduler.Current().Recorder();
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &pre_barrier;
	command.pipelineBarrier2(dependency);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline_layout,
	                             0, writes);
	const auto num_threads    = BufferCache::CACHING_NUMPAGES / 32;
	const auto num_workgroups = (num_threads + 63) / 64;
	command.dispatch(static_cast<uint32_t>(num_workgroups), 1, 1);
	dependency.pBufferMemoryBarriers = &post_barrier;
	command.pipelineBarrier2(dependency);
	vk::BufferMemoryBarrier2 download_barrier {};
	download_barrier.srcStageMask    = vk::PipelineStageFlagBits2::eComputeShader;
	download_barrier.srcAccessMask   = vk::AccessFlagBits2::eShaderWrite;
	download_barrier.dstStageMask    = vk::PipelineStageFlagBits2::eHost;
	download_barrier.dstAccessMask   = vk::AccessFlagBits2::eHostRead;
	download_barrier.buffer          = download.Handle();
	download_barrier.offset          = offset;
	download_barrier.size            = area_size;
	dependency.pBufferMemoryBarriers = &download_barrier;
	command.pipelineBarrier2(dependency);

	const auto area = current;
	// The write scan marks the pages the GPU wrote as GPU-owned: an unmap waits for it, or a remap
	// of the range would inherit the marks. The use scan only changes host state.
	auto completion = [this, mapped, offset, area, &download, area_size, &areas,
	                   handler = std::move(handler)]() mutable {
		download.Invalidate(offset, area_size);
		const auto* entries = std::bit_cast<const uint64_t*>(mapped);
		// The parser counts every set bit but stores only the entries the area holds; it leaves
		// the others set for the next read.
		const auto capacity = area_size / sizeof(uint64_t) - 1;
		const auto count    = std::min<uint64_t>(static_cast<uint32_t>(entries[0]), capacity);
		std::vector<uint64_t> pages;
		pages.reserve(count);
		for (uint64_t index = 1; index <= count; ++index) {
			pages.push_back(BufferCache::GuestAddress(entries[index]));
		}
		if (!pages.empty()) {
			handler(pages);
		}
		areas[area] = 0;
	};
	if (host_only) {
		m_scheduler.DeferHostOperation(std::move(completion));
	} else {
		m_scheduler.DeferOperation(std::move(completion));
	}

	areas[current++] = m_scheduler.CurrentTick();
	current %= MaxPendingFaults;
}

void FaultManager::ProcessWriteBuffer(PagesHandler&& handler) {
	ProcessBitmap(BufferCache::WRITE_BITMAP_OFFSET, m_write_download, m_write_area_size,
	              m_write_areas, m_current_write_area, std::move(handler), false);
}

void FaultManager::ProcessUseBuffer(PagesHandler&& handler) {
	ProcessBitmap(BufferCache::USE_BITMAP_OFFSET, m_use_download, m_use_area_size, m_use_areas,
	              m_current_use_area, std::move(handler), true);
}

} // namespace Libs::Graphics
