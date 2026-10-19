#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/recompiler/ir/BdaReadPlan.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.h"
#include "graphics/shader/shader.h"

#include <array>
#include <bit>
#include <deque>
#include <list>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace Xbyak {
class CodeGenerator;
}

namespace Libs::Graphics::ShaderRecompiler::IR {

class SrtWalker;

// Finite descriptor domains also bound the arithmetic used to prove their indices.
constexpr uint32_t MaxIndirectDescriptorProbes = 65536u;

enum class ResourceKind {
	None,
	ScalarBuffer,
	ScalarAddress,
	Buffer,
	IndirectBuffer,
	Flat,
	FlatLocal,
	Global,
	Scratch,
	Lds,
	Gds,
	Image,
	Sampler
};

[[nodiscard]] constexpr bool IsAddressResourceKind(ResourceKind kind) {
	return kind == ResourceKind::ScalarAddress || kind == ResourceKind::Flat ||
	       kind == ResourceKind::FlatLocal || kind == ResourceKind::Global ||
	       kind == ResourceKind::Scratch;
}

struct MemoryInfo {
	ResourceKind            kind                     = ResourceKind::None;
	uint32_t                resource                 = 0;
	uint32_t                sampler                  = 0;
	uint32_t                offset                   = 0;
	uint32_t                secondary_offset         = 0;
	uint32_t                dmask                    = 0;
	uint32_t                data_dwords              = 1;
	uint32_t                data_bits                = 32;
	uint32_t                component_index          = 0;
	uint32_t                component_count          = 1;
	uint32_t                data_format              = 0;
	uint32_t                number_format            = 0;
	uint32_t                image_sample_flags       = 0;
	Decoder::ImageDimension image_dimension          = Decoder::ImageDimension::Unknown;
	uint32_t                image_address_components = 0;
	bool                    address_is_full                                       = false;
	bool                    data_signed                                           = false;
	bool                    typed                                                 = false;
	bool                    formatted                                             = false;
	bool                    image_has_mip                                         = false;
	bool                    image_r128                                            = false;
	bool                    idxen                                                 = false;
	bool                    offen                                                 = false;
	bool                    coherent                                              = false;
	bool                    planning_only                                         = false;

	[[nodiscard]] bool SupportsIndirectBufferLoad(ValueOpcode opcode) const {
		if (typed) return false;
		if (data_bits == 8u) return !formatted && opcode == ValueOpcode::LoadBufferU8;
		if (data_bits == 16u) return !formatted && opcode == ValueOpcode::LoadBufferU16;
		return data_bits == 32u &&
		       (formatted ? opcode == ValueOpcode::LoadBufferU32
		                  : opcode == ValueOpcode::ReadConstBuffer ||
		                        opcode == ValueOpcode::LoadBufferU32 ||
		                        opcode == ValueOpcode::LoadBufferU32x2 ||
		                        opcode == ValueOpcode::LoadBufferU32x3 ||
		                        opcode == ValueOpcode::LoadBufferU32x4);
	}
	// Raw DWORD stores through a V# selected at runtime from a descriptor table.
	[[nodiscard]] bool SupportsIndirectBufferStore(ValueOpcode opcode) const {
		return !formatted && !typed && data_bits == 32u &&
		       (opcode == ValueOpcode::StoreBufferU32 || opcode == ValueOpcode::StoreBufferU32x2 ||
		        opcode == ValueOpcode::StoreBufferU32x3 || opcode == ValueOpcode::StoreBufferU32x4);
	}
	[[nodiscard]] bool SupportsIndirectBufferAccess(ValueOpcode opcode) const {
		return SupportsIndirectBufferLoad(opcode) || SupportsIndirectBufferStore(opcode);
	}

	bool operator==(const MemoryInfo& other) const = default;
};

enum class ExportTargetKind { Unknown, Null, Position, Primitive, Parameter, Mrt, MrtZ };

struct ExportInfo {
	ExportTargetKind kind   = ExportTargetKind::Unknown;
	uint32_t         target = 0;
	uint32_t         index  = 0;
	uint32_t         en     = 0;
	bool             done   = false;
	bool             compr  = false;
	bool             vm     = false;

	bool operator==(const ExportInfo& other) const = default;
};

struct BufferResource {
	static constexpr uint32_t NoImageAlias = UINT32_MAX;
	static constexpr uint32_t NoIndirectBuffer = UINT32_MAX;

	uint32_t               source             = 0;
	uint32_t               descriptor_index   = UINT32_MAX;
	// The buffer's slot among the uniform buffers too (DescriptorBindingKind::UniformBuffers),
	// which its constant reads use, or UINT32_MAX.
	uint32_t               uniform_index      = UINT32_MAX;
	uint32_t               first_use_pc       = 0;
	uint32_t               max_byte_extent    = 0;
	uint32_t               packed_stride      = 0;
	Prospero::BufferFormat descriptor_format  = Prospero::BufferFormat::kInvalid;
	uint32_t               descriptor_swizzle = DstSel(4, 5, 6, 7);
	uint32_t               image_alias        = NoImageAlias;
	bool                   read               = false;
	bool                   written            = false;
	bool                   atomic             = false;
	bool                   formatted          = false;
	bool                   scalar             = false;
	bool                   live               = false;
	// Only read by ReadConstBuffer at immediate offsets inside UniformBufferBytes, so it can be
	// bound as a uniform buffer too (see ShaderInfoCollection).
	bool                   constant_only      = false;
	uint32_t               indirect_root              = NoIndirectBuffer;
	uint32_t               indirect_mapping_offset    = 0;
	uint32_t               indirect_search_iterations = 0;
	std::vector<uint32_t>  indirect_resources;
	// This buffer is a table of V#s (16-byte entries starting at indirect_table_offset) that
	// selects the targets of IndirectBuffer stores; the host caches every entry it names.
	bool                   indirect_write_table  = false;
	uint32_t               indirect_table_offset = 0;

	bool operator==(const BufferResource& other) const = default;
};

enum class ImageMipMode { None, Dynamic };

constexpr uint32_t ShaderImageIdentitySwizzle = 0x00000facu;

struct ImageResource {
	static constexpr uint32_t NoIndirectImage = UINT32_MAX;
	static constexpr uint32_t NoImageTable    = UINT32_MAX;

	uint32_t                      source            = 0;
	uint32_t                      descriptor_index  = UINT32_MAX;
	uint32_t                      first_use_pc      = 0;
	ImageResourceClass            resource_class    = ImageResourceClass::None;
	Prospero::TextureNumericClass numeric_class     = Prospero::TextureNumericClass::Unsupported;
	Decoder::ImageDimension       dimension         = Decoder::ImageDimension::Unknown;
	ImageMipMode                  mip_mode          = ImageMipMode::None;
	uint32_t                      mip_count         = 1;
	Prospero::BufferFormat        conversion_format = Prospero::BufferFormat::kInvalid;
	uint32_t                      shader_swizzle    = ShaderImageIdentitySwizzle;
	bool                          read              = false;
	bool                          written           = false;
	bool                          atomic            = false;
	bool                          atomic64          = false;
	bool                          unfiltered_access_only = true;
	bool                          depth_compare     = false;
	bool                          cube              = false;
	bool                          r128              = false;
	uint32_t                      indirect_root     = NoIndirectImage;
	uint32_t                      indirect_mapping_offset   = 0;
	uint32_t                      indirect_search_iterations = 0;
	std::vector<uint32_t>         indirect_resources;
	// A root sampling a GPU-indexed T# table (bindless): the key selects table entry
	// key & table_entry_mask, flattened_srt[table_mapping_offset] holds the entry count and
	// then one slot per entry, and the slot indexes table_capacity descriptors of this image's
	// binding that the host fills from the table (slot 0 is a null image).
	uint32_t                      table                = NoImageTable;
	uint32_t                      table_capacity       = 0;
	uint32_t                      table_mapping_offset = 0;
	uint32_t                      table_entry_mask     = 0;
	// First element of the table's descriptors in this image's binding, after the binding's
	// resources; roots of one table and binding share them.
	uint32_t                      table_descriptor_base = UINT32_MAX;

	bool operator==(const ImageResource& other) const = default;
};

[[nodiscard]] constexpr Prospero::BufferFormat
RawSignedStorageFormat(const ImageResource& image, Prospero::BufferFormat format) {
	if (image.resource_class != ImageResourceClass::Storage || !image.written)
		return Prospero::BufferFormat::kInvalid;
	if (format == Prospero::BufferFormat::k32SInt && (image.atomic ? !image.atomic64 : !image.read))
		return Prospero::BufferFormat::k32UInt;
	if (format == Prospero::BufferFormat::k16SInt && !image.read && !image.atomic)
		return Prospero::BufferFormat::k16UInt;
	return Prospero::BufferFormat::kInvalid;
}

struct SamplerResource {
	static constexpr uint32_t NoIndirectSampler = UINT32_MAX;

	uint32_t source                = 0;
	uint32_t first_use_pc          = 0;
	// Native filtering/border variants share the original sampler's runtime descriptor.
	uint32_t snapshot_index        = 0;
	uint32_t indirect_root = NoIndirectSampler;
	uint32_t indirect_mapping_offset = 0;
	uint32_t indirect_search_iterations = 0;
	std::vector<uint32_t> indirect_resources;
	bool     force_point_filtering = false;
	bool     depth_compare         = false;
	bool     integer_border        = false;
	bool     gather_lod            = false;

	bool operator==(const SamplerResource& other) const = default;
};

struct SampledResourcePair {
	uint32_t image        = 0;
	uint32_t sampler      = 0;
	uint32_t first_use_pc = 0;

	bool operator==(const SampledResourcePair& other) const = default;
};

enum class TessellationAttribute {
	LocalOutput,
	ControlInput,
	ControlOutput,
	EvaluationInput,
	PatchOutput,
	Factor,
	PatchInput
};

enum class StageInputKind {
	VertexIndex,
	InvocationId,
	PrimitiveId,
	TessCoord,
	InstanceIndex,
	FragCoord,
	FrontFacing,
	PackedAncillary,
	Layer,
	SampleId,
	BaryCoordSmooth,
	BaryCoordSmoothSample,
	BaryCoordSmoothCentroid,
	BaryCoordNoPerspective,
	WorkgroupId,
	NumWorkgroups,
	LocalInvocationId,
	LocalInvocationIndex,
	GlobalInvocationId,
	Parameter,
};

enum class StageOutputKind {
	Position,
	Parameter,
	Mrt,
	ObjectPrimitiveId,
	Depth,
	SampleMask,
	PointSize,
	ClipDistance,
	CullDistance,
	Layer,
	ViewportIndex
};

struct PositionExportComponent {
	uint32_t clip_distance = UINT32_MAX;
	uint32_t cull_distance = UINT32_MAX;
	bool     point_size     = false;
	bool     layer          = false;
	bool     viewport       = false;
};

inline PositionExportComponent DecodePositionExportComponent(uint32_t control,
	                                                           uint32_t pos_index,
	                                                           uint32_t component) {
	PositionExportComponent result;
	if (pos_index == 0 || component >= 4) {
		return result;
	}

	uint32_t slot   = pos_index - 1;
	uint32_t vector = 3;
	for (uint32_t i = 0; i < 3; i++) {
		if ((control & (1u << (21u + i))) != 0) {
			if (slot == 0) {
				vector = i;
				break;
			}
			slot--;
		}
	}
	if (vector == 3) {
		return result;
	}

	if (vector == 0) {
		result.point_size = component == 0 && (control & (1u << 16u)) != 0;
		result.layer      = component == 2 && (control & (1u << 18u)) != 0;
		result.viewport   = component == 2 && (control & (1u << 19u)) != 0;
		return result;
	}

	const auto scalar = (vector - 1) * 4 + component;
	const auto lower  = (1u << scalar) - 1u;
	const auto clip   = control & 0xffu;
	const auto cull   = (control >> 8u) & 0xffu;
	if ((clip & (1u << scalar)) != 0) {
		result.clip_distance = std::popcount(clip & lower);
	}
	if ((cull & (1u << scalar)) != 0) {
		result.cull_distance = std::popcount(cull & lower);
	}
	return result;
}

struct StageInput {
	StageInputKind kind            = StageInputKind::VertexIndex;
	uint32_t       location        = 0;
	uint32_t       component_count = 1;
	std::string    debug_name;
	bool           per_vertex = false;

	bool operator==(const StageInput& other) const = default;
};

struct StageOutput {
	StageOutputKind kind     = StageOutputKind::Parameter;
	uint32_t        index    = 0;
	uint32_t        location = 0;
	std::string     debug_name;

	bool operator==(const StageOutput& other) const = default;
};

inline constexpr uint32_t FirstImageBinding           = 1u;
inline constexpr uint32_t FirstComparisonImageBinding = 22u;
inline constexpr uint32_t FirstStorageImageBinding    = 29u;
inline constexpr uint32_t ImageBindingCount           = 48u;

enum class DescriptorBindingKind : uint32_t {
	Buffers  = 0u,
	Samplers = FirstImageBinding + ImageBindingCount,
	Gds,
	BdaPagetable,
	FaultBuffer,
	FlattenedSrt,
	ShaderData,
	SharedMemory,
	UniformBuffers,
	Count,
};

// The uniform buffer range every host binds (maxUniformBufferRange is at least 64 KiB on the
// GPUs that run the emulator); constant reads stay below it with the base adjustment.
inline constexpr uint32_t UniformBufferBytes = 65536u;
inline constexpr uint32_t UniformBufferBaseSlack = 256u;

static_assert(static_cast<uint32_t>(DescriptorBindingKind::Samplers) == 49u);
static_assert(static_cast<uint32_t>(DescriptorBindingKind::Count) == 57u);

struct PushData {
	static constexpr uint32_t DwordCount = 32;
	static constexpr uint32_t MeshDrawDwordCount = 6;
	static constexpr uint32_t NoStart    = UINT32_MAX;
	std::array<uint32_t, DwordCount> dwords {};

	[[nodiscard]] static constexpr uint32_t GraphicsDrawDwords(bool mesh, bool object_id) {
		return mesh ? MeshDrawDwordCount : static_cast<uint32_t>(object_id);
	}

	[[nodiscard]] static constexpr bool CanFit(uint32_t start, uint32_t size) {
		return size != 0 && start <= DwordCount && size <= DwordCount - start;
	}
	[[nodiscard]] static constexpr uint32_t StartFor(uint32_t cursor, uint32_t size) {
		return CanFit(cursor, size) ? cursor : NoStart;
	}
};

static_assert(sizeof(PushData) == 128);
constexpr uint32_t NativePushConstantSize = sizeof(PushData);

[[nodiscard]] constexpr uint32_t NativeBinding(ShaderType stage, DescriptorBindingKind kind) {
	const uint32_t group = stage == ShaderType::Pixel                    ? 1u
	                       : stage == ShaderType::TessellationControl    ? 2u
	                       : stage == ShaderType::TessellationEvaluation ? 3u
	                                                                     : 0u;
	return static_cast<uint32_t>(kind) +
	       group * static_cast<uint32_t>(DescriptorBindingKind::Count);
}

[[nodiscard]] constexpr ImageResourceClass ImageBindingResourceClass(DescriptorBindingKind kind) {
	const auto value = static_cast<uint32_t>(kind);
	if (value >= FirstImageBinding && value < FirstStorageImageBinding) {
		return ImageResourceClass::Sampled;
	}
	if (value >= FirstStorageImageBinding &&
	    value < static_cast<uint32_t>(DescriptorBindingKind::Samplers)) {
		return ImageResourceClass::Storage;
	}
	return ImageResourceClass::None;
}

[[nodiscard]] constexpr uint32_t ImageBindingIndex(DescriptorBindingKind kind) {
	return static_cast<uint32_t>(kind) - FirstImageBinding;
}

[[nodiscard]] constexpr std::optional<DescriptorBindingKind>
DescriptorBindingForImage(const ImageResource& image) {
	constexpr uint32_t SampledFloatBinding = 1u;
	constexpr uint32_t SampledUintBinding  = 8u;
	constexpr uint32_t SampledSintBinding  = 15u;
	constexpr uint32_t StorageFloatBinding = FirstStorageImageBinding;
	constexpr uint32_t StorageUintBinding  = StorageFloatBinding + 5u;
	constexpr uint32_t AtomicUintBinding   = StorageUintBinding + 5u;

	uint32_t base    = 0;
	bool     sampled = false;
	if (image.resource_class == ImageResourceClass::Sampled) {
		if (image.atomic) {
			return std::nullopt;
		}
		sampled = true;
		switch (image.numeric_class) {
			case Prospero::TextureNumericClass::Float:
				base = image.depth_compare ? FirstComparisonImageBinding : SampledFloatBinding;
				break;
			case Prospero::TextureNumericClass::Uint: base = SampledUintBinding; break;
			case Prospero::TextureNumericClass::Sint: base = SampledSintBinding; break;
			case Prospero::TextureNumericClass::Unsupported: return std::nullopt;
			default: return std::nullopt;
		}
		if (image.depth_compare && image.numeric_class != Prospero::TextureNumericClass::Float) {
			return std::nullopt;
		}
	} else if (image.resource_class == ImageResourceClass::Storage) {
		if (image.atomic) {
			if (image.numeric_class != Prospero::TextureNumericClass::Uint) {
				return std::nullopt;
			}
			base = AtomicUintBinding + (image.atomic64 ? 5u : 0u);
		} else {
			switch (image.numeric_class) {
				case Prospero::TextureNumericClass::Float: base = StorageFloatBinding; break;
				case Prospero::TextureNumericClass::Uint: base = StorageUintBinding; break;
				case Prospero::TextureNumericClass::Sint:
				case Prospero::TextureNumericClass::Unsupported: return std::nullopt;
				default: return std::nullopt;
			}
		}
	} else {
		return std::nullopt;
	}

	uint32_t dimension = 0;
	switch (image.dimension) {
		case Decoder::ImageDimension::Dim1D: break;
		case Decoder::ImageDimension::Dim1DArray: dimension = 1u; break;
		case Decoder::ImageDimension::Dim2D: dimension = 2u; break;
		case Decoder::ImageDimension::Dim2DArray: dimension = 3u; break;
		case Decoder::ImageDimension::Dim2DMsaa:
			if (!sampled) {
				return std::nullopt;
			}
			dimension = 4u;
			break;
		case Decoder::ImageDimension::Dim2DMsaaArray:
			if (!sampled) {
				return std::nullopt;
			}
			dimension = 5u;
			break;
		case Decoder::ImageDimension::Dim3D: dimension = sampled ? 6u : 4u; break;
		case Decoder::ImageDimension::Unknown: return std::nullopt;
		default: return std::nullopt;
	}
	return static_cast<DescriptorBindingKind>(base + dimension);
}

// The first table descriptor of each image (UINT32_MAX without a bindless table): the
// descriptors of a table follow the resources of its roots' binding, whose count grows by the
// table capacity; roots of one table and binding share them.
[[nodiscard]] inline std::vector<uint32_t> AllocateImageTableDescriptors(
    const std::vector<ImageResource>& images,
    std::array<uint32_t, static_cast<size_t>(DescriptorBindingKind::Count)>& counts) {
	std::vector<uint32_t> bases(images.size(), UINT32_MAX);
	for (size_t i = 0; i < images.size(); i++) {
		const auto& image = images[i];
		if (image.table_capacity == 0u) {
			continue;
		}
		const auto kind = *DescriptorBindingForImage(image);
		for (size_t j = 0; j < i && bases[i] == UINT32_MAX; j++) {
			if (bases[j] != UINT32_MAX && images[j].table == image.table &&
			    DescriptorBindingForImage(images[j]) == kind) {
				if (images[j].table_capacity != image.table_capacity) {
					EXIT("shader binding layout failed: image table %u has two capacities",
					     image.table);
				}
				bases[i] = bases[j];
			}
		}
		if (bases[i] == UINT32_MAX) {
			auto& count = counts[static_cast<size_t>(kind)];
			bases[i]    = count;
			count += image.table_capacity;
		}
	}
	return bases;
}

struct Bindings {
	std::array<uint32_t, static_cast<size_t>(DescriptorBindingKind::Count)> descriptor_counts {};
	uint64_t descriptor_mask = 0;
	uint32_t push_data_start_dword = PushData::NoStart;
	uint32_t dispatch_thread_dword = PushData::NoStart;
	uint32_t memory_offset_dword = 0;
	uint32_t table_block_dwords  = 0; // table mode: the block's device address (TablePlan)

	[[nodiscard]] uint32_t TableBlockDword() const {
		return memory_offset_dword +
		       (descriptor_counts[static_cast<size_t>(DescriptorBindingKind::Buffers)] + 3u) / 4u;
	}
	[[nodiscard]] uint32_t ShaderDataDwords() const {
		return TableBlockDword() + table_block_dwords;
	}
	[[nodiscard]] bool UsesPushData() const {
		return push_data_start_dword != PushData::NoStart;
	}
	void AdvancePushData(uint32_t& cursor) const {
		if (UsesPushData()) {
			cursor = push_data_start_dword + ShaderDataDwords();
		}
	}

	bool operator==(const Bindings& other) const = default;
};

struct ShaderInfo {
	static constexpr uint32_t MaxBufferRoots  = 64;
	static constexpr uint32_t MaxImages       = 64;
	static constexpr uint32_t MaxSamplers     = 32;
	static constexpr uint32_t MaxSampledPairs = 64;

	std::vector<BufferResource>      buffers;
	std::vector<ImageResource>       images;
	std::vector<SamplerResource>     samplers;
	std::vector<SampledResourcePair> sampled_pairs;
	std::vector<StageInput>          inputs;
	std::vector<StageOutput>         outputs;
	std::vector<uint32_t>            user_data_registers;
	// User-data registers holding the low dword of a 64-bit base address that DMA (FLAT/global)
	// accesses offset from; the next register holds the high dword.
	std::vector<uint32_t>            dma_base_registers;
	// The guest memory DMA reads, when every DMA access is a read whose address is a user-data
	// base pair plus an offset with known possible bits (see DmaFootprint).
	struct DmaWindow {
		// Bytes [base + first, base + last) from the base in registers base_register and +1.
		uint32_t base_register = 0;
		int64_t  first         = 0;
		uint64_t last          = 0;

		bool operator==(const DmaWindow& other) const = default;
	};
	// V#s read from base + immediate + offset, for every offset whose set bits lie within
	// offset_bits; the shader reads buffer memory through them.
	struct DmaDescriptorTable {
		uint32_t base_register = 0;
		int64_t  immediate     = 0;
		uint32_t offset_bits   = 0;

		bool operator==(const DmaDescriptorTable& other) const = default;
	};
	std::vector<DmaWindow>           dma_windows;
	std::vector<DmaDescriptorTable>  dma_tables;
	std::array<uint8_t, 32>          vertex_fetch_components {};
	int32_t                          vertex_offset_sgpr = -1;
	int32_t                          instance_offset_sgpr = -1;
	bool                            has_bitwise_xor               = false;
	bool                            uses_dma                      = false;
	// dma_windows and dma_tables hold everything DMA can read (only with uses_dma).
	bool                            dma_bounded                   = false;
	bool                            uses_swizzle                  = false;
	bool                            uses_lds                      = false;
	bool                            uses_gds                      = false;
	bool                            uses_flattened_srt            = false;
	bool                            uses_dispatch_threads         = false;
	bool                            bvh                           = false;
	bool                            subgroup_ballot               = false;
	bool                            subgroup_vote                 = false;
	bool                            subgroup_barrier              = false;
	bool                            subgroup_shuffle              = false;
	bool                            subgroup_arithmetic           = false;
	bool                            subgroup_local_invocation_id  = false;
	bool                            compute_derivatives           = false;
	bool                            image_gather_extended         = false;
	bool                            function_lds                  = false;
	bool                            function_scratch              = false;
	bool                            pixel_valid_mask              = false;
	bool                            buffer_int64_atomics          = false;
	bool                            buffer_u8                     = false;
	bool                            buffer_u16                    = false;
	bool                            shared_int64_atomics          = false;
	bool                            coherent_buffers              = false;
	bool                            float64                       = false;
	bool                            float16                       = false;

	bool operator==(const ShaderInfo& other) const = default;
};

struct DescriptorSource {
	struct IndirectDescriptor {
		struct SelectorRead {
			struct CountedRows {
				uint32_t source = UINT32_MAX;
				uint32_t stride = 0; // Byte spacing between selector rows, not their length.

				bool operator==(const CountedRows& other) const = default;
			};
			uint32_t source = UINT32_MAX;
			uint32_t stride = 0;
			uint32_t offset = 0;
			uint32_t maximum = UINT32_MAX;
			uint32_t bit_offset = 0;
			uint32_t bit_mask = UINT32_MAX;
			bool global = false;
			// Four DWORD bitmap words per record; set-bit ranks subtract from the last key.
			bool bitset = false;
			std::optional<CountedRows> rows;
			std::vector<std::pair<uint32_t, uint32_t>> byte_ranges;

			bool operator==(const SelectorRead& other) const = default;
		};
		struct PointerTable {
			uint32_t offset = 0;
			uint32_t stride = 0;
			uint32_t count = 1;
			bool indexed = false; // Parent pointers use raw MUBUF record addressing.

			bool operator==(const PointerTable& other) const = default;
		};
		std::optional<SelectorRead> selector;
		std::optional<uint32_t> table_maximum;
		std::optional<PointerTable> pointer_table;
		uint32_t table_source    = 0;
		uint32_t table_offset    = 0;
		uint32_t table_immediate = 0;
		uint32_t table_stride    = 0;
		uint32_t table_record_bytes = 0;
		bool     table_scalar = false;
		bool     indexed = false;
		bool     signed_count = false;
		uint32_t workgroup_axis  = UINT32_MAX;
		// The table entry is at ((key * table_stride) & table_mask) + table_offset.
		uint32_t table_mask      = UINT32_MAX;
		// A bindless T# table: every table entry is a candidate.
		bool     table_array     = false;
		Value    key_count;
		// First selector index, or first descriptor key for a direct or bitmap range.
		Value                 selector_first;
		Value    selector_mask;
		std::vector<uint32_t> sources;

		bool operator==(const IndirectDescriptor& other) const = default;
	};

	std::array<Value, 8>         dwords {};
	uint32_t                     dword_count = 0;
	std::optional<IndirectDescriptor> indirect_descriptor;

	bool operator==(const DescriptorSource& other) const = default;
};

struct SrtRead {
	Value    value;
	uint32_t flat_offset = 0;

	bool operator==(const SrtRead& other) const = default;
};

struct ResourceBlock {
	// Conditional successors are ordered true, false; an empty condition follows every edge.
	Value                 condition;
	std::vector<uint32_t> successors;
	std::vector<uint32_t> sources;
	std::vector<uint32_t> srt_reads;
};

// Table mode (Program::table_mode, chenxiao07's KytyPS5 fork): the flattened SRT's slots, the V#s of the buffers the
// shader reads by device address, and the T#s and S#s of its images and samplers. Slot s holds the dword at
// ((high:low) & 0xffff'ffff'ffff) + (offset & ~3) + immediate, dword aligned (a bounded slot, read through a V#: high and
// `records` are its words 1 and 2, and a read past its range, NUM_RECORDS bytes or NUM_RECORDS times the stride, fails
// the evaluation, as the normal path's); a descriptor is its words. An operand is an immediate, a user data register or
// a lower slot. For each dispatch or draw the renderer evaluates the slots `cpu` marks (those the shader reads, `gpu`,
// the image and sampler words, which choose its descriptors, and what their addresses are read through) as the guest
// memory holds them then, and gives the shader a block: dword s is slot s; from BufferDword(i), buffer i's device
// address (two dwords), size in dwords (0: none), buffer word and stride, resolved from its V#. The shader reads the
// block through its device address (Bindings::TableBlockDword).
struct TablePlan {
	struct Operand {
		// ShaderBaseLow/High: the low or high dword of the stage's code address plus `value` (a V# of constants the
		// code embeds, addressed from s_getpc_b64; the renderer has the address from the registers).
		enum class Kind : uint8_t { Immediate, UserData, Slot, ShaderBaseLow, ShaderBaseHigh };
		Kind     kind  = Kind::Immediate;
		uint32_t value = 0; // the immediate, the scalar register, the slot or the offset from the code address
		bool     operator==(const Operand&) const = default;
	};
	struct Slot {
		Operand low, high, offset;
		int32_t immediate = 0; // dword aligned
		Operand records;       // a bounded slot's V# word 2
		bool    bounded = false;
		bool    operator==(const Slot&) const = default;
	};
	struct Descriptor {
		std::array<Operand, 8> words {};
		uint32_t               count = 0;
		bool                   operator==(const Descriptor&) const = default;
	};
	std::vector<Slot>                   slots;
	std::vector<std::array<Operand, 4>> buffers;
	std::vector<Descriptor>             images, samplers;
	std::vector<uint8_t>                gpu, cpu; // per slot
	// Flat or global memory reads: the renderer readies their page table ranges at every use, as the normal path's
	// PrepareBda does.
	bool global_memory = false;
	// An indirect image whose candidates and key mapping the normal path's materialization makes (an
	// IndirectDescriptor with a key search: a compute shader's light loop picking one T# of a table by a record's key):
	// the renderer enumerates them at every use as materialization does (EnumerateIndirectImage), binds them at the
	// root's `elements` (its indirect_resources, root first) and writes the key mapping ([count, (key, ordinal)...],
	// `dwords` of it) into the block at `block`, which the shader's search reads there (the root's
	// indirect_mapping_offset in table mode). `flat_mapping` and `search_iterations` are the mapping's place and size in
	// the normal path's flattened SRT, which the enumeration must give again (the program's layout).
	struct IndirectImage {
		uint32_t              root = 0;
		uint32_t              flat_mapping = 0, search_iterations = 0;
		uint32_t              block = 0, dwords = 0;
		std::vector<uint32_t> elements;
		bool                  operator==(const IndirectImage&) const = default;
	};
	std::vector<IndirectImage> indirect;

	// A buffer's dwords in the block: its device address (2), size in dwords, buffer word and stride, all from the V#
	// the renderer resolved.
	static constexpr uint32_t BufferDwords = 5;
	[[nodiscard]] uint32_t BufferDword(size_t buffer) const {
		return static_cast<uint32_t>(slots.size() + BufferDwords * buffer);
	}
	[[nodiscard]] uint32_t BlockDwords() const {
		uint32_t dwords = BufferDword(buffers.size());
		for (const auto& image: indirect) dwords += image.dwords;
		return dwords;
	}
	bool                   operator==(const TablePlan&) const = default;
};

// Stable shader metadata consumed by the renderer after native IR has been discarded.
struct CompiledShaderInfo {
	ShaderType                    stage               = ShaderType::Unknown;
	uint64_t                      shader_hash         = 0;
	uint32_t                      wave_size           = 64;
	uint32_t                      user_data_base      = 0;
	uint32_t                      user_data_count     = 64;
	uint32_t                      scratch_dwords      = 0;
	uint32_t                      param_export_mask   = 0;
	bool                          has_address_writes  = false;
	bool                          table_mode          = false; // see Program::table_mode
	TablePlan                     table_plan;
	ShaderInfo                    info;
	Bindings                      bindings;
	// For each flat SRT slot, the written info.buffers that a store or atomic of the shader
	// may update before a read of that slot in program order.
	std::vector<std::vector<uint32_t>> srt_reads_after_buffer_writes;
	// The same for each scalar load the shader issues itself, by memory-info index.
	std::vector<std::vector<uint32_t>> raw_reads_after_buffer_writes;
	// The guest memory the shader reads through device addresses, from the dispatch's user data and flattened SRT
	// (chenxiao07's BdaReadPlan, KYTY_BDA_READ_RANGES): what its BDA synchronization has to make current.
	BdaReadPlan bda_read_plan;
};

// Whether a write of the shader to a buffer (an info.buffers index) can be visible to one of
// its specialization reads. A read that a flattened scalar load made before any write to that
// buffer in program order is not: the shader consumes the value from the flattened SRT, and
// each wave performs that load before its own writes, so the dispatch-time snapshot is a value
// the load observes on hardware.
bool SpecializationReadFollowsBufferWrite(const CompiledShaderInfo& program,
                                          const ResourceSnapshot& resources, size_t read,
                                          uint32_t buffer);

struct UniformFillPlan {
	UniformFill          fill;
	std::array<Value, 4> values;
	Value                pattern_size;
	Value                element_count;
};

struct CompiledSrtPlan;
struct LinearSrtPlan;

// Resource analysis retained by the shader cache. It owns immutable descriptor/SRT,
// condition and fill values without translated blocks, plus reusable evaluation scratch.
struct ResourcePlan {
	struct SelectorKeys {
		uint32_t              source = UINT32_MAX;
		std::vector<uint32_t> values;
		bool                  resolved = false;
	};

	struct EvaluationContext {
		struct Entry {
			uint64_t value      = 0;
			uint64_t generation = 0;
		};

		std::vector<Entry> values;
		uint64_t           generation = 0;
		const SrtWalker*    session = nullptr;
		Value              active_mask;
		uint8_t            read_policy = 0;
	};

	ResourcePlan();
	~ResourcePlan();

	ResourcePlan(const ResourcePlan&)            = delete;
	ResourcePlan& operator=(const ResourcePlan&) = delete;
	ResourcePlan(ResourcePlan&&) noexcept;
	ResourcePlan& operator=(ResourcePlan&& other) noexcept;

	ShaderType                    stage           = ShaderType::Unknown;
	uint64_t                      shader_hash     = 0;
	uint32_t                      user_data_base  = 0;
	uint32_t                      user_data_count = 64;
	std::list<Inst>                     value_storage;
	std::vector<MemoryInfo>             memory_info;
	std::vector<DescriptorSource>       descriptor_sources;
	std::vector<uint32_t>               selector_groups;
	std::vector<ResourceBlock>          control_flow;
	std::unordered_map<const Inst*, std::vector<uint32_t>> control_phi_predecessors;
	std::vector<SrtRead>                srt_reads;
	bool                                requires_specialization_memory = false;
	std::vector<std::pair<uint64_t, uint64_t>> source_reads;
	bool                                capture_specialization_reads = false;
	bool                                srt_plan_complete          = false;
	bool                                resource_tracking_complete = false;
	// Flattened evaluation program, built on the first refresh (null: SrtWalker evaluates).
	// The plan's values must not change after that refresh. Kept next to the flags that
	// every refresh reads.
	mutable bool                                   compiled_srt_built = false;
	mutable std::shared_ptr<const CompiledSrtPlan> compiled_srt;
	// Its native form (LinearSrt.h), built on first use per read policy (without, with a clean evaluator).
	struct LinearSrtSlot {
		bool                                 built = false;
		std::shared_ptr<const LinearSrtPlan> plan;
	};
	mutable std::array<LinearSrtSlot, 2> linear_srt;
	ShaderInfo                          info;
	UniformFillPlan                     uniform_fill;
	std::unique_ptr<Xbyak::CodeGenerator> srt_code;
	// GPU-thread scratch for nested clean/EXEC memos, activity and material keys.
	mutable std::deque<EvaluationContext> evaluation_contexts;
	mutable uint32_t                       evaluation_value_count = 0;
	mutable std::vector<uint8_t>            active_sources;
	mutable std::vector<uint8_t>            visited_blocks;
	mutable std::vector<uint8_t>            reachable_blocks;
	mutable std::vector<uint32_t>           pending_blocks;
	mutable std::vector<uint32_t>           material_keys;
	mutable std::vector<SelectorKeys>       selector_keys;
};

struct Program: ResourcePlan {
	Program() = default;
	~Program();

	Program(const Program&)            = delete;
	Program& operator=(const Program&) = delete;
	Program(Program&&) noexcept         = default;
	Program& operator=(Program&& other) noexcept;
	CompiledShaderInfo TakeCompiledInfo() &&;

	std::vector<std::unique_ptr<Block>> block_storage;
	BlockList                           blocks;
	uint32_t                      wave_size      = 64;
	uint32_t                      scratch_dwords = 0;
	bool                          dispatcher_fallback = false;
	CFG::FailureKind              cfg_failure_kind    = CFG::FailureKind::None;
	std::string                   fallback_reason;
	struct ScalarWrite { uint32_t pc; ScalarReg reg; Value value; };
	std::vector<ScalarWrite>      scalar_writes;
	// Typed memory and export instructions reference shader-local metadata by dense index.
	// Decoder-only details (such as NSA register numbers) have already become IR operands.
	std::vector<ExportInfo>       export_info;
	bool                          has_address_writes = false;
	bool                          shader_info_complete = false;
	Bindings                      bindings;
	// Table mode (EnterTableMode): the SRT reads and the buffers' V#s come from a block the renderer fills for each use,
	// read through its device address; no flattened SRT or buffer descriptors.
	bool                          table_mode = false;
	TablePlan                     table_plan;

};

uint32_t StorageBufferElementBits(const Program& program, const MemoryInfo& memory);

std::string ProgramToString(const Program& program);
bool        HasShaderMemoryWrites(const Program& program);
// raw (optional) receives the same per memory-info index of the shader's own scalar loads.
std::vector<std::vector<uint32_t>> SrtReadsAfterBufferWrites(
    const Program& program, std::vector<std::vector<uint32_t>>* raw = nullptr);

void  ValidateProgram(const Program& program, bool require_ssa);
bool  EquivalentValue(const ResourcePlan& program, Value left, Value right);
bool  EquivalentSelectorDomain(const ResourcePlan& program,
                               const DescriptorSource::IndirectDescriptor& left,
                               const DescriptorSource::IndirectDescriptor& right);
Value ResolveInvariantPhi(const ResourcePlan& program, Value value);
Value ResolveActiveU32(Value value, Value active);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_ */
