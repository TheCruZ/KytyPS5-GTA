#include "graphics/guest_gpu/psrShaders.h"

#include "common/logging/log.h"
#include "graphics/shader/shader.h"

#include <atomic>
#include <cstring>
#include <string_view>

namespace Libs::Graphics::Psr {

namespace {

struct NamedShader {
	std::string_view name;
	Shader           shader;
};

// The registered names of the listed shaders. The library keeps byte-identical copies of a few
// shaders under two names (psr_morton_initialize_indirect_args_top and its _trinity copy, the
// _uint32 and _float descriptor initializations of the scan): both names do the same thing.
constexpr NamedShader Names[] = {
    {"psr_morton_compute_bounding_box_for_instances", Shader::MortonComputeBoundingBoxForInstances},
    {"psr_morton_initialize_indirect_args_top", Shader::MortonInitializeIndirectArgsTop},
    {"psr_morton_initialize_indirect_args_top_trinity", Shader::MortonInitializeIndirectArgsTop},
    {"psr_single_pass_scan_initialize_descriptors_uint32", Shader::SinglePassScanInitializeDescriptors},
    {"psr_single_pass_scan_initialize_descriptors_float", Shader::SinglePassScanInitializeDescriptors},
    {"psr_single_pass_scan_uint32", Shader::SinglePassScanUint32},
    {"psr_morton_compute_valid_instance_indices", Shader::MortonComputeValidInstanceIndices},
    {"psr_morton_setup_indirect_args_top", Shader::MortonSetupIndirectArgsTop},
    {"psr_morton_compute_code_for_instances", Shader::MortonComputeCodeForInstances},
    {"psr_bitonic_sort_pass_key_value_indirect", Shader::BitonicSortPassKeyValueIndirect},
    {"psr_morton_build_qbvh_hierarchy_for_instances", Shader::MortonBuildQbvhHierarchyForInstances},
    {"psr_morton_build_instance_node", Shader::MortonBuildInstanceNode},
    {"psr_morton_refit_qbvh_hierarchy_for_instances", Shader::MortonRefitQbvhHierarchyForInstances},
    {"psr_morton_rotate_tree_first_run_top", Shader::MortonRotateTreeFirstRunTop},
    {"psr_morton_rotate_tree_top", Shader::MortonRotateTreeTop},
    {"psr_collapse_to_qbvh_simple_compute_compacted_nodes_indirect",
     Shader::CollapseToQbvhSimpleComputeCompactedNodesIndirect},
    {"psr_single_pass_scan_setup_indirect_args_vec2u", Shader::SinglePassScanSetupIndirectArgsVec2u},
    {"psr_single_pass_scan_initialize_descriptors_vec2u_indirect",
     Shader::SinglePassScanInitializeDescriptorsVec2uIndirect},
    {"psr_single_pass_scan_vec2u_indirect", Shader::SinglePassScanVec2uIndirect},
    {"psr_collapse_to_qbvh_simple_top", Shader::CollapseToQbvhSimpleTop},
    {"psr_qbvh_fan_refit_copy_dispatch_args", Shader::QbvhFanRefitCopyDispatchArgs},
    {"psr_qbvh_fan_refit_async", Shader::QbvhFanRefitAsync},
    {"psr_qbvh_fan_refit_recover_aabbs", Shader::QbvhFanRefitRecoverAabbs},
    {"psr_qbvh_refit_bottom_level", Shader::QbvhRefitBottomLevel},
    {"psr_morton_compute_code_for_mesh_uint8x3", Shader::MortonComputeCodeForMesh},
    {"psr_morton_compute_code_for_mesh_uint16x3", Shader::MortonComputeCodeForMesh},
    {"psr_morton_compute_code_for_mesh_uint32x3", Shader::MortonComputeCodeForMesh},
    {"psr_morton_compute_code_for_mesh_low_precision_uint8x3", Shader::MortonComputeCodeForMesh},
    {"psr_morton_compute_code_for_mesh_low_precision_uint16x3", Shader::MortonComputeCodeForMesh},
    {"psr_morton_compute_code_for_mesh_low_precision_uint32x3", Shader::MortonComputeCodeForMesh},
    {"psr_collapse_to_qbvh_simple_bottom", Shader::CollapseToQbvhSimpleBottom},
    {"psr_collapse_to_qbvh_simple_bottom_with_refit", Shader::CollapseToQbvhSimpleBottom},
};

std::atomic<Generation> g_generation {Generation::Unknown};
std::atomic<bool>       g_generation_known {false};

// The generation of a library that registered its shaders. Generation Base460 has shaders the
// next one removed (the vertex bounds of the Morton build, the rebraiding instance copies) and
// none of the ones it added (nested BVH counts, flattening).
Generation DetectGeneration() {
	const bool base460 = ShaderNameRegistered("psr_morton_compute_bounding_box_for_vertices") &&
	                     ShaderNameRegistered("psr_rebraiding_copy_srt_instance") &&
	                     !ShaderNameRegistered("psr_count_nested_bvh") &&
	                     !ShaderNameRegistered("psr_create_flattened_header");
	return base460 ? Generation::Base460 : Generation::Unknown;
}

} // namespace

Generation LibraryGeneration() {
	return g_generation.load(std::memory_order_acquire);
}

Shader IdentifyShader(uint64_t code_address) {
	const char* name = ShaderRegisteredName(code_address);
	if (name == nullptr || std::strncmp(name, "psr_", 4) != 0) {
		return Shader::None;
	}
	// The library registers all its shaders at once, before it dispatches any: the first one
	// seen decides the generation.
	if (!g_generation_known.load(std::memory_order_acquire)) {
		const auto generation = DetectGeneration();
		g_generation.store(generation, std::memory_order_release);
		g_generation_known.store(true, std::memory_order_release);
		LOGF("Psr: library generation %s\n",
		     generation == Generation::Base460 ? "Base460" : "unknown (its shaders are emulated)");
	}
	if (g_generation.load(std::memory_order_acquire) != Generation::Base460) {
		return Shader::None;
	}
	for (const auto& entry: Names) {
		if (entry.name == name) {
			return entry.shader;
		}
	}
	return Shader::None;
}

bool IsTopLevelBuildShader(Shader shader) {
	return shader >= Shader::MortonComputeBoundingBoxForInstances &&
	       shader <= Shader::CollapseToQbvhSimpleTop;
}

} // namespace Libs::Graphics::Psr
