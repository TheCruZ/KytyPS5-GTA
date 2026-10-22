#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_

#include <array>
#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Descriptor set 1 of a pipeline that samples bindless images (the host side is BindlessTable).
inline constexpr uint32_t BindlessDescriptorSet  = 1;
inline constexpr uint32_t BindlessImages2D       = 0;
inline constexpr uint32_t BindlessImages2DArray  = 1;
inline constexpr uint32_t BindlessImagesCube     = 2;
inline constexpr uint32_t BindlessImages3D       = 3;
inline constexpr uint32_t BindlessTranslation    = 4;
inline constexpr uint32_t BindlessFeedback       = 5;
// Samplers mirrored from guest sampler heaps (16-byte S# records); slot 0 is the default sampler.
inline constexpr uint32_t BindlessSamplers       = 6;
// A translation entry for a key whose texture is not resident yet; the shader samples slot 0.
inline constexpr uint32_t BindlessPending        = 0xffffffffu;
// In a 3D binding's region: the key's T# is a 2D or 2D-array texture, held in the 2D-array
// binding at the slot in the low bits. The hardware addresses by the T# type, not the
// instruction's dimension, so a 3D sample of it reads layer round(z) (Marvel's Wolverine
// particles). Pending (all bits set) is told apart by the bit above.
inline constexpr uint32_t BindlessSlotArrayFlag  = 0x40000000u;
inline constexpr uint32_t BindlessSlotFlagMask   = 0xc0000000u;
// Diagnostics (KYTY_KEY_CAPTURE=<hash>): that shader marks every image key it samples in feedback
// word BindlessCaptureImages + (key & 0xffff) and every sampler key in BindlessCaptureSamplers +
// (key & 0xffff); the host prints them ([key-capture]). Above any heap region.
inline constexpr uint32_t BindlessCaptureImages   = 0xe0000u;
// Diagnostics (KYTY_PROBE_RECT with KYTY_PROBE_PC): the probed pixel shader's MRT0 export per
// pixel of a screen rectangle, 4 words each (x | y << 16, then three value bits); 16384 pixels.
inline constexpr uint32_t BindlessCaptureProbe    = 0xd0000u;
inline constexpr uint32_t BindlessCaptureProbeWords = 0x10000u;
inline constexpr uint32_t BindlessCaptureSamplers = 0xf0000u;
// The S# a sampler without a usable heap entry gets: wrap on every axis, trilinear, full LOD range.
inline constexpr std::array<uint32_t, 4> BindlessDefaultSampler {
    0x00000000u,                             // wrap on every axis
    0xfffu << 12u,                           // max_lod 255.9
    (1u << 20u) | (1u << 22u) | (2u << 26u), // bilinear mag/min, linear mip
    0x00000000u};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_
