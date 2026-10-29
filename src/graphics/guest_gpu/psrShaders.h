#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_PSRSHADERS_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_PSRSHADERS_H_

#include <cstdint>

namespace Libs::Graphics::Psr {

// sce::Psr is Sony's ray tracing BVH library, linked statically into the games that use it (GTA
// V, Astro Bot, Marvel's Wolverine). It builds and refits BVHs on the GPU with its own compute
// shaders, dispatched like any other compute work (there is nothing to intercept on the CPU), and
// registers each shader by name with the driver (owner "sce::Psr", see
// sceAgcDriverRegisterResource). The emulator recognizes them by that name, not by the hash of
// their code, which differs from game to game, and reads their user data with the layout of the
// library's generation.
//
// Only the shaders the emulator replaces or watches are listed.
enum class Shader : uint8_t {
	None,

	// Top level build (kBase format), in dispatch order. Some of them (scans, sorts) serve other
	// builds too.
	MortonComputeBoundingBoxForInstances,
	MortonInitializeIndirectArgsTop,
	SinglePassScanInitializeDescriptors,
	SinglePassScanUint32,
	MortonComputeValidInstanceIndices,
	MortonSetupIndirectArgsTop,
	MortonComputeCodeForInstances,
	BitonicSortPassKeyValueIndirect,
	MortonBuildQbvhHierarchyForInstances,
	MortonBuildInstanceNode,
	MortonRefitQbvhHierarchyForInstances,
	MortonRotateTreeFirstRunTop,
	MortonRotateTreeTop,
	CollapseToQbvhSimpleComputeCompactedNodesIndirect,
	SinglePassScanSetupIndirectArgsVec2u,
	SinglePassScanInitializeDescriptorsVec2uIndirect,
	SinglePassScanVec2uIndirect,
	CollapseToQbvhSimpleTop,

	// Refit of a bottom level BVH built for triangle fan refits, in dispatch order (the second one
	// once per geometry).
	QbvhFanRefitCopyDispatchArgs,
	QbvhFanRefitAsync,
	QbvhFanRefitRecoverAabbs,
	QbvhRefitBottomLevel,

	Count
};

// The generations of the library whose user data layouts the emulator knows.
enum class Generation : uint8_t {
	Unknown,
	// 460 shaders: GTA V (build J03180688) and Astro Bot (J03093718).
	Base460,
};

// The generation of the library the game registered (Unknown until it registered its shaders,
// and for generations whose layouts are not known).
[[nodiscard]] Generation LibraryGeneration();

// The Psr shader whose code starts at `code_address`, or None (not a Psr shader, a shader the
// emulator does not list, or a library of an unknown generation).
[[nodiscard]] Shader IdentifyShader(uint64_t code_address);

// Whether `shader` belongs to the top level (kBase) build.
[[nodiscard]] bool IsTopLevelBuildShader(Shader shader);

} // namespace Libs::Graphics::Psr

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_PSRSHADERS_H_
