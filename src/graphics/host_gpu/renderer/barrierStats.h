#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERSTATS_H_

#include <source_location>

namespace Libs::Graphics {

// KYTY_BARRIER_STATS=1: counts every vkCmdPipelineBarrier / vkCmdPipelineBarrier2 by call site and
// stage masks, and prints the busiest sites every 5 s ([barriers]). CommandRecorder names its
// caller through SetBarrierSite; direct vk::CommandBuffer calls are named by their host stack.
void InstallBarrierStats();
void SetBarrierSite(const std::source_location* site) noexcept;

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERSTATS_H_ */
