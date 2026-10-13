#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_RAYTRACINGPATCH_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_RAYTRACINGPATCH_H_

#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <cstdint>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::Decoder {

// Native ray tracing of guest BVH traversal loops.
//
// GTA V traverses its BVHs in software: a wave-wide loop pops a node from a stack kept in VGPR
// lanes, tests it with IMAGE_BVH_INTERSECT_RAY and pushes the children the lanes hit. Box nodes
// are 9 of 10 node visits. The patch keeps the guest code that sets up an instance (object
// space ray, BLAS descriptors) and the code of a leaf (triangle tests, alpha test, results), and
// replaces the stack walk with ray queries over host acceleration structures whose primitives
// are the guest leaves (see NativeRayTracing): each lane's query yields the leaves its ray
// reaches, the wave runs the guest leaf code for one (instance, leaf) at a time with the lanes
// that reached it, and a lane whose leaf code accepted a hit stops its query.
//
// The original loop stays in the shader: RT_INIT falls back to it when the host has no
// acceleration structure for the TLAS the shader traces.
//
// Pseudo instructions (Instruction::offset = site):
//   RT_INIT    image_nsa_addr[0..7] = origin xyz, direction xyz, t max and cull mask VGPRs,
//              src0 = SGPR pair of the guest TLAS header address. SCC = native traversal.
//   RT_SELECT  dst = SGPR of the instance node, dst2 = SGPR pair of the lane mask; advances the
//              queries, picks the candidate of the first lane that has one and sets EXEC (and
//              the mask) to the lanes with the same candidate. SCC = some lane has a candidate.
//   RT_LEAF    dst = SGPR of the leaf node, dst2 = SGPR pair that receives EXEC, src0 = SGPR pair
//              of the lanes entering the leaf.
//   RT_RESOLVE src0 = SGPR pair of the lanes still searching after the leaf: the other lanes that
//              entered it accepted a hit and stop.
//   RT_MODE    SCC = the site is traversing natively.
//   RT_EXIT    EXEC = the lanes that started the traversal; ends the native traversal.
struct RayTracingSite {
	// The instruction that falls through to the loop header; the patch enters there.
	uint32_t entry_pc = 0;
	// The loop header (the fallback target).
	uint32_t loop_pc = 0;
	uint32_t origin[3] {};
	uint32_t direction[3] {};
	uint32_t t_max     = 0;
	uint32_t cull_mask = 0;
	// SGPR pair of the TLAS header address at entry_pc.
	uint32_t header_sgpr = 0;
	// SGPR that turns the alpha test off (every triangle opaque) when not 0 (NoAlphaOff: none).
	uint32_t alpha_off_sgpr = 0;
	// The VGPRs of an accepted hit: distance, primitive word, instance record address (pair).
	uint32_t result_t        = 0;
	uint32_t result_word     = 0;
	uint32_t result_instance = 0;
	// The instance setup and the register it reads the instance node from.
	uint32_t setup_pc      = 0;
	uint32_t instance_sgpr = 0;
	// Where the instance setup leaves when the instance mask culls every lane.
	uint32_t cull_exit_pc = 0;
	// The BLAS loop header, replaced by the entry to the leaf code.
	uint32_t blas_head_pc = 0;
	// The leaf code, the registers it reads the leaf from, the lane mask and the saved EXEC.
	uint32_t leaf_pc         = 0;
	uint32_t leaf_sgpr       = 0;
	uint32_t mask_sgpr       = 0;
	uint32_t saved_exec_sgpr = 0;
	// The end of the leaf code (the BLAS loop latch), replaced by the end of the iteration.
	uint32_t tail_pc = 0;
	// Instructions run after each leaf (copies; 0 = none), e.g. restoring a spilled SGPR.
	uint32_t after_leaf_pc = 0;
	// Where the loop exits to.
	uint32_t exit_pc = 0;
};

constexpr uint32_t NoAlphaOff = 0xffffffffu;

struct RayTracingPatch {
	uint64_t                         shader_hash = 0;
	std::span<const RayTracingSite> sites;
	// The guest TLAS header: the 64-bit value at user data pointer `user_sgpr` + `header_offset`.
	uint32_t                         user_sgpr     = 0;
	uint32_t                         header_offset = 0;
	uint32_t                         root          = 0;
};

// The patch of a shader, or null.
[[nodiscard]] const RayTracingPatch* FindRayTracingPatch(uint64_t shader_hash) noexcept;

// Applies `patch` to a decoded program. Returns false, leaving the program unchanged, when the
// program does not match what the patch expects.
bool ApplyRayTracingPatch(const RayTracingPatch& patch, Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::Decoder

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_RAYTRACINGPATCH_H_
