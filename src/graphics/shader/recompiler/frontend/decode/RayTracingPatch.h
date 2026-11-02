#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_RAYTRACINGPATCH_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_RAYTRACINGPATCH_H_

#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Decoder {

// Native ray tracing of guest BVH traversal loops.
//
// Games using sce::Psr traverse its BVHs in software, with a traversal loop of the library
// compiled into their shaders (the variants GTA V uses are recognized; Astro Bot's are not yet).
// In the packet variant a wave-wide loop pops a node from a stack kept in
// VGPR lanes, tests it with IMAGE_BVH_INTERSECT_RAY and pushes the children the lanes hit. Box nodes
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
//   RT_ALPHA   before RT_INIT of a site with a RayTracingAlphaTest: image_nsa_addr[0] = the level
//              of detail VGPR, [1] = material V# offset, [2] = buffer table V# offset,
//              [3] = threshold, [4] = level of detail offset, [5] = reference VGPR (NoRegister:
//              threshold and factor); src0 = spread register, src1 = factor SGPR (none for a
//              reference VGPR), src2 / src3 = SGPR pairs of the material and buffer table user
//              pointers; dmask = RayTracingAlphaLayout.
//
// A closest hit site (RayTracingKind::ClosestOpaque) is a per lane traversal whose leaf code
// accepts the nearest triangle of either facing: the hardware finds the hit and RT_COMMIT writes
// every register the guest leaf code writes for it. When the leaf code alpha tests the triangles
// whose primitive word is not marked opaque (the site has a RayTracingAlphaTest with the Lane
// layout), the query takes every triangle as a candidate (the guest ignores the instance's
// opacity flags) and confirms the opaque ones and the alpha tested ones the alpha test accepts.
enum class RayTracingKind : uint32_t {
	Packet,        // Wave-wide traversal, the guest leaf code tests the alpha tested triangles.
	ClosestOpaque, // Closest hit, both facings.
};

// A register a site does not have.
constexpr uint32_t NoRegister = 0xffffffffu;

// The alpha test of a Packet site's leaf code, when it is GTA V's (see DeriveAlphaTest in
// RayTracingPatch.cpp): the native traversal runs it per lane on each alpha tested candidate the
// hardware reports and confirms the candidates it accepts, so the guest leaf code never runs.
// Per triangle (node N, primitive word W) of instance record R:
//   B  = ((R[0x34] & 0x3fffff) << 32 | R[0x30]) << 3      the BLAS
//   m  = B[B[0x28] + ((N >> 3) - B[0x7c]) * 4] + R[0x78]    the material index
//   M  = 12 bytes of the material table at m * 12
//   indices: V# (M1 >> 8) * 16 of the buffer table, formatted elements (W & 0x3fffffff) * 3 + k
//   uv:      V# (M0 >> 8) * 16 of the buffer table, two floats at (M1 & 0xfc) + (M0 & 0xfc) * index
//   level:   min(M2 & 15, log2(2 * (spread * t + lod)) + lod_offset)
//   accepted when the W channel of T# (M2 >> 4) of the leaf code's bindless table at uv exceeds
//   threshold * (M0 bit 0 ? factor : 1), or the lane's reference.
// Every register holds its value from the site entry through the traversal.
enum class RayTracingAlphaLayout : uint32_t {
	// A packet site's scalar leaf code: the material record is 12 bytes at m * 12, the vertices
	// are read through their V#, the reference is threshold * (M0 bit 0 ? factor : 1).
	Scalar = 0,
	// A closest hit site's per lane leaf code: the material record is dwords m * 3 + k of a
	// structured buffer, the vertices are read at the base address of their V# (from the buffer
	// table as a structured buffer), the reference is a VGPR.
	Lane = 1,
};

struct RayTracingAlphaTest {
	bool                  valid  = false;
	RayTracingAlphaLayout layout = RayTracingAlphaLayout::Scalar;
	// The lane's level of detail term (VGPR), the cone spread (an SGPR, or a VGPR in the Lane
	// layout) and the threshold factor (SGPR, Scalar layout) or reference (VGPR, Lane layout).
	uint32_t lod_vgpr       = 0;
	uint32_t spread_reg     = 0;
	uint32_t factor_sgpr    = 0;
	uint32_t reference_vgpr = NoRegister;
	// The V#s of the material table and of the vertex and index buffer table: 16 bytes at user
	// data pointer (SGPR pair) `*_user` + `*_offset`.
	uint32_t material_user   = 0;
	uint32_t material_offset = 0;
	uint32_t buffer_user     = 0;
	uint32_t buffer_offset   = 0;
	// Float bits.
	uint32_t threshold  = 0;
	uint32_t lod_offset = 0;
};

struct RayTracingSite {
	RayTracingKind kind = RayTracingKind::Packet;
	// The instruction that falls through to the loop header; the patch enters there.
	uint32_t entry_pc = 0;
	// The loop header (the fallback target).
	uint32_t loop_pc = 0;
	uint32_t origin[3] {};
	uint32_t direction[3] {};
	uint32_t t_max     = 0;
	// The VGPR of the ray's instance mask (NoRegister: the constant mask below).
	uint32_t cull_mask = 0;
	uint32_t cull_mask_constant = 0xff;
	// SGPR pair of the TLAS header address at entry_pc.
	uint32_t header_sgpr = 0;
	// SGPR that turns the alpha test off (every triangle opaque) when not 0 (NoAlphaOff: none).
	uint32_t alpha_off_sgpr = 0;
	// The VGPRs of an accepted hit: distance, primitive word, instance record address (pair).
	uint32_t result_t        = 0;
	uint32_t result_word     = 0;
	uint32_t result_instance = 0;
	// Closest hit sites: the VGPRs of its barycentrics (as the guest leaf code divides them) and
	// of its guest triangle node.
	uint32_t result_u    = NoRegister;
	uint32_t result_v    = NoRegister;
	uint32_t result_node = NoRegister;
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
	// The alpha test the native traversal runs instead of the leaf code.
	RayTracingAlphaTest alpha;
};

constexpr uint32_t NoAlphaOff = 0xffffffffu;

struct RayTracingPatch {
	std::vector<RayTracingSite> sites;
	// The guest TLAS header: the 64-bit value at user data pointer `user_sgpr` + `header_offset`.
	uint32_t                    user_sgpr     = 0;
	uint32_t                    header_offset = 0;
	uint32_t                    root          = 0;
};

// Finds the traversal loops of a decoded program by their structure (see RayTracingPatch.cpp);
// false when the program has none, or one that does not match in every detail (the program
// keeps its own traversal).
bool DeriveRayTracingPatch(const Program& program, RayTracingPatch& patch);

// Applies `patch` to a decoded program (and records its TLAS source in
// Program::ray_tracing_source). Returns false, leaving the program unchanged, when the program
// does not match what the patch expects.
bool ApplyRayTracingPatch(const RayTracingPatch& patch, Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::Decoder

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_RAYTRACINGPATCH_H_
