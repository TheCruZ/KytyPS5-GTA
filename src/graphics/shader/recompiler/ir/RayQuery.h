#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_RAYQUERY_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_RAYQUERY_H_

#include <cstdint>

// The flags of the RayQuery* IR operations (native ray tracing, see RayTracingPatch.h): each
// patched traversal (site) owns one host ray query per lane and a short list of the candidate
// leaves the query found; the wave processes one (instance, leaf) key at a time, the smallest
// any lane has, with every lane that has it.
namespace Libs::Graphics::ShaderRecompiler::IR {

constexpr uint32_t MaxRayQuerySites = 4;
// Candidates a lane collects before the wave processes them.
constexpr uint32_t RayQueryListSize = 32;

// RayQueryFlag: wave-uniform or per-lane state.
enum class RayQueryFlagKind : uint32_t {
	InTraversal = 0, // The lane started the native traversal.
	Reuse       = 1, // The selected key's instance is the one whose setup ran last.
	NeedRestore = 2, // An instance setup ran in this traversal (its spills must be undone).
	InstanceMatch = 3, // The lane's next key has the selected key's instance.
	Matched       = 4, // The lane has the selected key.
};

// RayQueryResult: the lane's opaque hit (found by the hardware), for lanes whose guest code did
// not accept another one.
enum class RayQueryResultKind : uint32_t {
	Has        = 0, // 1 when the lane has one.
	T          = 1, // The hit distance (float bits).
	Word       = 2, // The primitive word of its guest list entry.
	InstanceLo = 3, // The address of its guest instance record.
	InstanceHi = 4,
	U          = 5, // Barycentrics as the guest triangle test reports them (closest hit sites).
	V          = 6,
	Node       = 7, // The guest triangle node of the hit.
};

// RayQueryInit variant bits.
constexpr uint32_t RayQueryInitBothFacings = 1u;
// Opacity from the geometry only (the primitive word's opaque bit; instance flags do not
// count): every triangle is a candidate, the opaque ones are confirmed and the alpha tested
// ones the per-lane alpha test accepts.
constexpr uint32_t RayQueryInitGeometryOpacity = 2u;
// No guest leaf code (Decoder::RayTracingKind::PacketLane): without the per-lane alpha test the
// site keeps the guest traversal. The two last arguments flip the facing the lanes cull, and allow
// the native traversal.
constexpr uint32_t RayQueryInitNoLeaf = 4u;
// Every triangle is a candidate of the guest leaf code, which tests its facing and opacity
// (Decoder::RayTracingKind::ClosestLane).
constexpr uint32_t RayQueryInitAllCandidates = 8u;

// RayQueryNext, RayQueryMatch and RayQuerySelectedLeaf variant: the wave takes the smallest
// instance any lane has, and each lane with a key of it its own smallest leaf of it.
constexpr uint32_t RayQueryKeyLaneLeaf = 1u;

// RayQueryAlpha variant: how the alpha test reads its data (Decoder::RayTracingAlphaLayout).
constexpr uint32_t RayQueryAlphaScalar = 0u;
constexpr uint32_t RayQueryAlphaLane   = 1u;
// With the Lane layout: the reference is threshold * (M0 bit 0 ? factor : 1).
constexpr uint32_t RayQueryAlphaFactor = 2u;

// RayQueryMark: state changes; the argument selects lanes.
enum class RayQueryMarkKind : uint32_t {
	Enter       = 0, // The selected lanes in the argument entered the leaf code.
	Resolve     = 1, // After the leaf: selected lanes that entered and are not in the argument
	                 // accepted a hit and stop; the other selected lanes drop the key.
	CullResolve = 2, // The instance setup culled every selected lane: they drop the key, and
	                 // no instance setup is complete.
	SetupBegin  = 3, // The instance setup of the selected key starts, for the lanes in the
	                 // argument (whose object space ray it computes).
	Exit        = 4, // Ends the native traversal of the site.
};

[[nodiscard]] constexpr uint32_t RayQueryFlags(uint32_t site, uint32_t variant) noexcept {
	return site | (variant << 8u);
}
[[nodiscard]] constexpr uint32_t RayQuerySite(uint32_t flags) noexcept {
	return flags & 0xffu;
}
[[nodiscard]] constexpr uint32_t RayQueryVariant(uint32_t flags) noexcept {
	return flags >> 8u;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_RAYQUERY_H_
