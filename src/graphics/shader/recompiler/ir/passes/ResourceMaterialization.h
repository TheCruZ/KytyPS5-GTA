#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		bool                   scalar_empty                    = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		uint32_t                      table                      = ImageResource::NoImageTable;
		uint32_t                      table_capacity             = 0;
		uint32_t                      table_mapping_offset       = 0;
		uint32_t                      table_entry_mask           = 0;
		bool                          cube                       = false;
		bool                          fmask                      = false;
		bool                          operator==(const Image&) const = default;
	};

	struct Sampler {
		uint32_t indirect_root = SamplerResource::NoIndirectSampler;
		uint32_t indirect_mapping_offset = 0;
		uint32_t indirect_search_iterations = 0;
		bool operator==(const Sampler&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;
	std::vector<Sampler> samplers;

	bool operator==(const ResourceSpecialization&) const = default;
};

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
ResourcePlan ExtractResourcePlan(const Program& program);

// Makes indirect descriptor layouts stable: mappings reserve room for a power of two of keys
// (compute shaders at least `keys`, other stages at least 8) and indirect image roots bind a power
// of two of images (compute shaders at least `images`), so that the guest data they select from
// changes the shader permutation less often. Until it is called, layouts keep exact counts.
void SetComputeIndirectFloors(size_t keys, size_t images);

// The shape of the image descriptors the last materialization of one plan specialized, and the image
// specialization it derived (chenxiao07's KYTY_SPECIALIZATION_GUARD, default on; 0 turns it off). A
// materialization whose descriptors have the same shape (null or not, format, type, mip range and swizzle
// bits: the only bits the derivation reads) takes the stored image specialization instead of deriving it
// again. Addresses, sizes, SRT values and sampler payloads always come from the current materialization,
// and so does the buffer specialization. Only plans of direct images are captured. Owned by whoever
// serializes the materializations of the plan (the program cache's lock); one guard per plan.
struct ResourceSpecializationGuard {
	struct ImageProbe {
		uint32_t format, interpretation;
		bool     null;
	};

	const ResourcePlan*                    owner = nullptr;
	std::vector<ImageProbe>                images;
	std::vector<ResourceSpecialization::Image> specialization;
	bool                                   compact_images = false;
	// Counts the shapes captured: unchanged by a materialization that reused the stored one.
	uint64_t                               publication = 0;
};

// Refreshes cached resources and specialization in place. A failed refresh must not be used. With a guard
// of the plan, the image specialization is reused when the descriptors' shape is unchanged.
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                          ResourceSpecializationGuard* shape_guard = nullptr);

void SetSpecializationGuardEnabledForTesting(bool enabled);

// The specialization of resources whose descriptors the caller evaluated (table mode, chenxiao07's KytyPS5 fork):
// snapshot.buffers, images and samplers hold one descriptor per resource of the plan, as resource materialization
// derives it from them. False for a plan with indirect descriptors, or descriptors no specialization takes.
bool SpecializeDirectResources(const ResourcePlan& program, ResourceSnapshot& snapshot,
                               ResourceSpecialization& specialization);

// The same refresh through SrtWalker alone, without the compiled plan; tests compare the two.
bool MaterializeResourcesReference(const ResourcePlan& program, const SrtRuntime& runtime,
                                   ResourceSnapshot& snapshot,
                                   ResourceSpecialization& specialization);

// Whether the host can bind a bindless table T# in the descriptor array of a specialized root.
bool ImageTableSlotCompatible(const ImageResource& root, const DescriptorValue& descriptor);

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
