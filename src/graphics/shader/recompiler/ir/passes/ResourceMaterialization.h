#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		// Written at addresses the host cannot bound: the shader records the pages it writes in
		// the GPU write bitmap (BufferCache::WriteBitmap, rework.md Phase 2 first step).
		bool                   write_tracked                   = false;
		bool                   scalar_empty                    = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		// KYTY_GPU_SIZED_DYNAMIC: see BufferResource::gpu_records_flat.
		uint32_t               gpu_records_flat                = BufferResource::NoGpuRecords;
		uint32_t               gpu_records_add_flat            = BufferResource::NoGpuRecords;
		uint32_t               gpu_records_add                 = 0;
		bool                   gpu_records_ssa                 = false;
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
		bool                          bindless                   = false;
		bool                          operator==(const Image&) const = default;
	};

	struct Sampler {
		uint32_t indirect_root = SamplerResource::NoIndirectSampler;
		uint32_t indirect_mapping_offset = 0;
		uint32_t indirect_search_iterations = 0;
		bool     bindless                = false;
		uint32_t bindless_mapping_offset = 0;
		bool operator==(const Sampler&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;
	std::vector<Sampler> samplers;

	bool operator==(const ResourceSpecialization&) const = default;
};

// KYTY_WRITE_BITMAP=1 (ResourceSpecialization::Buffer::write_tracked).
[[nodiscard]] bool WriteBitmapEnabled();

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
ResourcePlan ExtractResourcePlan(const Program& program);

// Makes indirect descriptor layouts stable: mappings reserve room for a power of two of keys
// (compute shaders at least `keys`, other stages at least 8) and indirect image roots bind a power
// of two of images (compute shaders at least `images`), so that the guest data they select from
// changes the shader permutation less often. Until it is called, layouts keep exact counts.
void SetComputeIndirectFloors(size_t keys, size_t images);

// Refreshes cached resources and specialization in place. A failed refresh must not be used.
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization);

// Diagnostics: the source line of the indirect-image check that failed the last
// MaterializeResources call on this thread, or 0.
int LastIndirectImageFailureLine();

// Whether the host can bind a bindless table T# in the descriptor array of a specialized root.
bool ImageTableSlotCompatible(const ImageResource& root, const DescriptorValue& descriptor);

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

// Kind of an image heap entry as root sees it, or 0 (and a zeroed value) when root reads it as
// null. Entries of one kind specialize identically.
uint32_t ImageHeapKind(const ImageResource& root, DescriptorValue& value);
// The sampled resource that root resolves entries of kind as; kind 0 gives the identity of
// root's view of a heap.
ImageResource ImageHeapResource(const ImageResource& root, uint32_t kind);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
