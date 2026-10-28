#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// Host device capabilities that change the emitted SPIR-V. Built once per device.
struct Profile {
	bool maintenance8         = false; // Dynamic texel offsets on image samples.
	bool native_float_min_max = false; // NMin/NMax give the guest's NaN and signed-zero results.
	// OpReadClockKHR Device tick in picoseconds; 0 without shaderDeviceClock.
	uint32_t device_clock_period_ps = 0;
};

inline constexpr Profile DefaultProfile {};

// S_MEMREALTIME counts at 100 MHz. The whole divisor that brings a device clock with this tick
// nearest to that rate: 10 for a 1 ns clock (NVIDIA), 1 for a 100 MHz one (AMD), at most 1024.
constexpr uint32_t DeviceClockDivisor(uint32_t device_clock_period_ps) {
	if (device_clock_period_ps == 0u) {
		return 1u;
	}
	const uint32_t divisor = (10000u + device_clock_period_ps / 2u) / device_clock_period_ps;
	return divisor < 1u ? 1u : (divisor > 1024u ? 1024u : divisor);
}

// Resets and writes backend IDs and descriptor indices; do not emit the same program concurrently.
std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t       push_data_start_dword = 0,
                                  const Profile& profile               = DefaultProfile,
                                  bool           emit_debug_info       = false,
                                  bool           robust_buffer_access2 = false);
} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
