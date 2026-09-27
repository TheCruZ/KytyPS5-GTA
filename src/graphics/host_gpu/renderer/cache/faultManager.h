#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_

#include "common/abi.h"
#include "graphics/host_gpu/renderer/cache/bufferCachePages.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <span>

namespace Libs::Graphics {

class BufferCache;

struct ShaderFaultReport {
	uint64_t         page_count = 0;
	ShaderTrapRecord trap;
};

// The fault buffer holds two page bitmaps (one bit per BufferCache caching page): shaders record
// device-address accesses that found no cached buffer in the first, and pages that V#-table
// stores wrote in the second (after the shader trap record). Both are read back asynchronously.
class FaultManager {
	static constexpr size_t MaxPendingFaults = 8;

public:
	using PagesHandler = Common::UniqueFunction<void, std::span<const uint64_t>>;

	FaultManager(GraphicContext& graphics, CommandScheduler& scheduler, BufferCache& buffer_cache);
	~FaultManager();
	KYTY_CLASS_NO_COPY(FaultManager);

	using PagesHandler = Common::UniqueFunction<void, std::span<const uint64_t>>;

	[[nodiscard]] Buffer*           GetFaultBuffer() noexcept;
	void                            ProcessFaultBuffer();
	[[nodiscard]] ShaderFaultReport CollectFaults();
	// Clears the write bitmap and, once the GPU work recorded so far completes, passes the guest
	// addresses of the written caching pages to `handler`.
	void                            ProcessWriteBuffer(PagesHandler&& handler);
	// Page faults reported while no dispatch transaction was collecting (for example by a
	// dispatch that ran without recovery). A change re-arms dispatch recovery.
	[[nodiscard]] uint64_t UnattributedFaults() const noexcept {
		return m_unattributed_faults.load(std::memory_order_acquire);
	}

private:
	void                                   ProcessFaultBufferImpl(ShaderFaultReport* report);
	GraphicContext&                        m_graphics;
	CommandScheduler&                      m_scheduler;
	BufferCache&                           m_buffer_cache;
	size_t                                 m_download_area_size;
	Buffer                                 m_fault_buffer;
	Buffer                                 m_download_buffer;
	std::array<uint64_t, MaxPendingFaults> m_fault_areas {};
	uint32_t                               m_current_area                  = 0;
	bool                                   m_initialized                   = false;
	vk::DescriptorSetLayout                m_fault_process_desc_layout     = nullptr;
	vk::Pipeline                           m_fault_process_pipeline        = nullptr;
	vk::PipelineLayout                     m_fault_process_pipeline_layout = nullptr;
	std::atomic<uint64_t>                  m_unattributed_faults {0};
	// Readback of the write bitmap, in its own areas.
	Buffer                                 m_write_download;
	size_t                                 m_write_area_size;
	std::array<uint64_t, MaxPendingFaults> m_write_areas {};
	uint32_t                               m_current_write_area = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_
