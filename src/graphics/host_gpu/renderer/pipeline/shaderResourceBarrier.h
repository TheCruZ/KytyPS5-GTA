#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_

#include "graphics/host_gpu/graphicContext.h"

#include <source_location>
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

namespace Libs::Graphics {

struct ShaderStageRuntime;
class CommandRecorder;

vk::ShaderStageFlagBits NativeShaderStage(ShaderType stage);
vk::PipelineStageFlags  ShaderPipelineStages(vk::ShaderStageFlags stages);
vk::MemoryBarrier       MakeShaderAccessDependency();
vk::MemoryBarrier       MakeShaderWriteHazardDependency();
vk::MemoryBarrier       MakeShaderWriteDependency();
vk::BufferMemoryBarrier MakeGdsDependency(vk::Buffer buffer);
bool HasShaderBufferWrites(const ShaderStageRuntime& runtime);
void ShaderAccessBarrier(const CommandRecorder& vk_buffer, vk::PipelineStageFlags source_stages,
                         std::source_location site = std::source_location::current());
void ShaderWriteHazardBarrier(const CommandRecorder& vk_buffer,
                              vk::PipelineStageFlags destination_stages,
                              std::source_location   site = std::source_location::current());
void ShaderWriteBarrier(const CommandRecorder& vk_buffer, vk::PipelineStageFlags source_stages);

// KYTY_DISPATCH_SYNC=0 (experiment, default 1): no barrier around every compute dispatch; the
// guest's own sync points order dispatches instead (CS/PS partial flushes already emit a global
// barrier, and ACQUIRE_MEM then emits one too).
inline bool DispatchSyncEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DISPATCH_SYNC");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_
