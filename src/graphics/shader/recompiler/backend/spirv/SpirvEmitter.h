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
};

inline constexpr Profile DefaultProfile {};

// Resets and writes backend IDs and descriptor indices; do not emit the same program concurrently.
std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t       push_data_start_dword = 0,
                                  const Profile& profile               = DefaultProfile,
                                  bool emit_debug_info = false,
                                  bool robust_buffer_access2 = false);

// Optional image features of the device, set once by the device layer.
struct HostImageFeatures {
	// shaderResourceMinLod: the MinLod image operand, used for IMAGE_SAMPLE*_CL.
	bool min_lod = false;
};

void              SetHostImageFeatures(const HostImageFeatures& features);
HostImageFeatures GetHostImageFeatures();

// S_MEMREALTIME counts at 100 MHz. The emitter reads the device clock (OpReadClockKHR, device
// scope) and divides it by this number to reach that rate. The device layer sets it from the
// device's clock rate: 10 for a 1 GHz clock (NVIDIA, the default), 1 for a 100 MHz one (AMD).
void     SetDeviceClockDivisor(uint32_t divisor);
uint32_t GetDeviceClockDivisor();

// A coherent (GLC) buffer load is followed by a device-scope acquire barrier, so that a load
// polling for another workgroup's write sees it. Without one, AMD's drivers keep serving the
// cached value (BufferWorkgroupPublication, ISSUES.md #16); NVIDIA reloads anyway. The device
// layer enables it on every vendor but NVIDIA; KYTY_COHERENT_LOAD_ACQUIRE=0/1 overrides that.
void SetCoherentLoadAcquire(bool enabled);
bool GetCoherentLoadAcquire();

// The device runs 32-bit float shaders in the SPIR-V DenormFlushToZero mode
// (shaderDenormFlushToZeroFloat32, with independent denormal behaviour per width). Shaders whose
// MODE flushes FP32 denormals (every one this title runs) then declare the mode and drop the
// software flushes around MAD/MAC, conversions and transcendentals: the host flushes the inputs
// and outputs of every FP32 operation itself, as the guest ALU does. KYTY_DENORM_FTZ_MODE=0 keeps
// the software flushes.
void SetDenormFlushToZero32(bool supported);
bool GetDenormFlushToZero32();

// Why a mesh program cannot run in passes (ShaderMeshInputInfo::passes), or nullptr.
[[nodiscard]] const char* MeshPassesUnsupported(const IR::Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
