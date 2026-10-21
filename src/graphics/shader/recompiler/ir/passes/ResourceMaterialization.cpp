#include "common/hackNotice.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "common/assert.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <fmt/format.h>
#include <functional>
#include <numeric>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint64_t AddressMask            = 0x0000ffffffffffffull;
// Keys an indirect descriptor mapping reserves room for at least (see MaterializeIndirectDescriptor()).
constexpr size_t MinIndirectMappingCapacity = 8u;
// See SetComputeIndirectFloors().
std::atomic<size_t> g_compute_indirect_keys {MinIndirectMappingCapacity};
std::atomic<size_t> g_compute_indirect_images {1u};
// Set by SetComputeIndirectFloors(): without it mappings and images keep their exact counts.
std::atomic<bool> g_stable_indirect_layouts {false};

bool SpecializationFail(std::string_view message) {
	std::fprintf(stderr, "shader resource specialization failed: %.*s\n",
	             static_cast<int>(message.size()), message.data());
	return false;
}

Decoder::ImageDimension DescriptorDimension(const DescriptorValue&  descriptor,
                                            Decoder::ImageDimension requested) {
	const bool is_array = requested == Decoder::ImageDimension::Dim1DArray ||
	                      requested == Decoder::ImageDimension::Dim2DArray ||
	                      requested == Decoder::ImageDimension::Dim2DMsaaArray;
	switch (static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu)) {
		case Prospero::ImageType::kColor1D: return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor1DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim1DArray;
			}
			return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor3D: return Decoder::ImageDimension::Dim3D;
		case Prospero::ImageType::kCube: return Decoder::ImageDimension::Dim2DArray;
		case Prospero::ImageType::kColor2DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DArray;
			}
			return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaaArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DMsaaArray;
			}
			return Decoder::ImageDimension::Dim2DMsaa;
		case Prospero::ImageType::kColor2D: return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaa: return Decoder::ImageDimension::Dim2DMsaa;
		default: return Decoder::ImageDimension::Unknown;
	}
}

bool NullImageDescriptor(const DescriptorValue& descriptor) {
	return descriptor.dwords[0] == 0 && (descriptor.dwords[1] & 0xffu) == 0;
}

bool NullBufferDescriptor(Value value) {
	const auto* handle = value.Resolve().TryInstruction();
	if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetBufferResource ||
	    handle->NumArgs() != 4u) return false;
	for (size_t index = 0; index < 4u; ++index)
		if (handle->Arg(index).Resolve() != Value(0u)) return false;
	return true;
}

bool ValidImageDescriptor(const DescriptorValue& descriptor, bool r128 = false) {
	const auto& words = descriptor.dwords;
	// Reject texture descriptors with nonzero reserved bits.
	if ((words[1] & 0x20000000u) != 0u || (words[2] & 0x70003000u) != 0u ||
	    (!r128 && ((words[4] & 0xe000e000u) != 0u || (words[5] & 0xf9000000u) != 0u ||
	               (words[6] & 0x00007b00u) != 0u))) {
		return false;
	}
	const auto type   = static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu);
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	if (type < Prospero::ImageType::kColor1D || format == Prospero::BufferFormat::kInvalid ||
	    format > Prospero::BufferFormat::kBc7Srgb) {
		return false;
	}
	if (r128 && type != Prospero::ImageType::kColor1D && type != Prospero::ImageType::kColor2D &&
	    type != Prospero::ImageType::kColor2DMsaa) {
		return false;
	}
	if (!IsValidImageSwizzle(words[3] & 0xfffu)) return false;
	const bool array = type == Prospero::ImageType::kColor1DArray ||
	                   type == Prospero::ImageType::kColor2DArray ||
	                   type == Prospero::ImageType::kColor2DMsaaArray ||
	                   type == Prospero::ImageType::kCube;
	if (array && ((words[4] >> 16u) & 0x1fffu) > (words[4] & 0x1fffu)) {
		return false;
	}
	const auto base_level = (words[3] >> 12u) & 0xfu;
	const auto last_level = (words[3] >> 16u) & 0xfu;
	if (type == Prospero::ImageType::kColor2DMsaa ||
	    type == Prospero::ImageType::kColor2DMsaaArray) {
		const auto max_mip    = (descriptor.dwords[5] >> 4u) & 0xfu;
		return base_level == 0 && last_level >= 1 && last_level <= 3 &&
		       (r128 || max_mip == last_level);
	}
	return base_level <= last_level;
}

uint32_t DescriptorImageSwizzle(const DescriptorValue& descriptor) {
	return descriptor.dwords[3] & 0xfffu;
}

Prospero::BufferFormat ImageConversionFormat(Prospero::BufferFormat format) {
	return Prospero::RemapTextureFormat(format) != format ? format
	                                                      : Prospero::BufferFormat::kInvalid;
}

bool ValidatePackedUnormSampling(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                                 uint32_t image_index) {
	ShaderTextureResource texture;
	std::copy_n(snapshot.images[image_index].dwords.begin(), std::size(texture.fields),
	            texture.fields);
	if (texture.Type() != Prospero::ImageType::kColor2D ||
	    texture.BaseLevel() != texture.LastLevel() || texture.MinLod() != 0u ||
	    texture.CornerSample()) {
		return SpecializationFail("packed UNorm sampling requires a centered, single-mip 2D image");
	}
	for (const auto& pair: program.info.sampled_pairs) {
		if (pair.image != image_index) continue;
		if (pair.sampler >= snapshot.samplers.size() ||
		    program.descriptor_sources[program.info.samplers[pair.sampler].source].indirect_descriptor) {
			return SpecializationFail("packed UNorm sampling requires a direct sampler");
		}
		ShaderSamplerResource sampler;
		std::copy_n(snapshot.samplers[pair.sampler].dwords.begin(), std::size(sampler.fields),
		            sampler.fields);
		const auto linear = static_cast<uint32_t>(Prospero::SamplerFilter::kBilinear);
		const auto clamp = static_cast<uint32_t>(Prospero::SamplerClampMode::kClampLastTexel);
		if (sampler.XyMagFilter() != linear || sampler.XyMinFilter() != linear ||
		    sampler.ClampX() != clamp || sampler.ClampY() != clamp ||
		    sampler.ForceUnormCoords() || sampler.MaxAnisoRatio() != 0u ||
		    sampler.FilterMode() != 0u || sampler.MinLod() != 0u || sampler.ForceSrgb() ||
		    sampler.TruncCoord() || (sampler.fields[0] & (1u << 19u)) != 0u) { // MC_COORD_TRUNC
			return SpecializationFail(fmt::format(
			    "packed UNorm image {} requires normalized bilinear clamp sampling (sampler {})",
			    image_index, pair.sampler));
		}
	}
	return true;
}

enum class SamplerClass : uint8_t { Float, Integer };

SamplerClass ClassifySampler(const ImageResource& image) {
	return image.numeric_class == Prospero::TextureNumericClass::Uint ||
	               image.numeric_class == Prospero::TextureNumericClass::Sint
	           ? SamplerClass::Integer : SamplerClass::Float;
}

bool DescriptorIsCube(const DescriptorValue& descriptor) {
	return static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu) ==
	       Prospero::ImageType::kCube;
}

uint32_t ImageMipCount(const ImageResource& image, const DescriptorValue& descriptor) {
	if (image.mip_mode != ImageMipMode::Dynamic || NullImageDescriptor(descriptor)) {
		return 1;
	}
	const auto base = (descriptor.dwords[3] >> 12u) & 0xfu;
	const auto last = (descriptor.dwords[3] >> 16u) & 0xfu;
	return last - base + 1u;
}

bool DecodeBufferDescriptor(const DescriptorValue& descriptor, ShaderBufferResource& result) {
	if (descriptor.dword_count != std::size(result.fields)) {
		return false;
	}
	std::copy_n(descriptor.dwords.begin(), std::size(result.fields), result.fields);
	return true;
}

const DescriptorSource* Source(const ResourcePlan& program, uint32_t source) {
	if (source >= program.descriptor_sources.size()) {
		return nullptr;
	}
	return &program.descriptor_sources[source];
}

void MarkCleanReads(const std::unordered_set<const Inst*>& planned_reads,
                    const DescriptorSource* source, Value extra = {}) {
	if (source == nullptr && extra.IsEmpty()) {
		return;
	}
	std::vector<Value> pending;
	if (source != nullptr) {
		pending.assign(source->dwords.begin(), source->dwords.begin() + source->dword_count);
	}
	if (!extra.IsEmpty()) pending.push_back(extra);
	std::vector<const Inst*> visited;
	while (!pending.empty()) {
		auto* inst = pending.back().Resolve().TryInstruction();
		pending.pop_back();
		if (inst == nullptr || std::ranges::find(visited, inst) != visited.end()) {
			continue;
		}
		visited.push_back(inst);
		if (planned_reads.contains(inst)) {
			auto flags = inst->Flags<SrtReadFlags>();
			flags.clean = 1u;
			inst->SetFlags(flags);
		}
		for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
			pending.push_back(inst->Arg(arg));
		}
	}
}

bool ReadTable(uint64_t base, uint64_t size, uint64_t offset,
               const SrtRuntime& runtime, std::span<uint32_t> words) {
	const auto count = std::min<uint64_t>(words.size(), offset < size ? (size - offset) / 4u : 0u);
	std::ranges::fill(words.subspan(count), 0u);
	if (count == 0u) {
		return true;
	}
	if (base > AddressMask || offset > AddressMask - base) {
		return false;
	}
	const auto address = base + offset;
	const auto prefix = words.first(count);
	return prefix.size_bytes() - 1u <= AddressMask - address &&
	       runtime.read_specialization_memory != nullptr &&
	       runtime.read_specialization_memory(runtime.userdata, address, prefix);
}

thread_local int g_indirect_failure_line = 0;
} // namespace
// Which part of MaterializeResources is reading (diagnostics: KYTY_LOG_GPU_OWNED_SRT).
thread_local const char* g_materialize_phase = "";
// The image descriptors as evaluated, before invalid ones are zeroed (KYTY_DEBUG_RESOURCES).
thread_local std::vector<DescriptorValue> g_debug_raw_images;
const char* MaterializePhase() { return g_materialize_phase; }
namespace {

bool FailIndirect(int line) {
	if (g_indirect_failure_line == 0) {
		g_indirect_failure_line = line;
	}
	return false;
}

// The shader looks the key up in the bindless translation table: nothing to enumerate. Record
// the heap for the host and reserve the two words it patches (region base, entry count); zeros
// resolve every key to the placeholder in slot 0.
bool MaterializeBindlessImage(const DescriptorSource::IndirectDescriptor& indirect,
                               const DescriptorValue& table_value, uint32_t image_index,
                               ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	ShaderBufferResource heap;
	if (table_value.dword_count != 4u || !DecodeBufferDescriptor(table_value, heap)) {
		return FailIndirect(__LINE__);
	}
	const auto mapping_offset = static_cast<uint32_t>(snapshot.flattened_srt.size());
	snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
	snapshot.images[image_index] = {.dword_count = 8u};
	snapshot.bindless_heaps.push_back({.base           = heap.Base48(),
	                                   .size           = heap.GetSize(),
	                                   .table_offset   = indirect.table_offset,
	                                   .image          = image_index,
	                                   .mapping_offset = mapping_offset});
	auto& root                      = specialization.images[image_index];
	root.indirect_root              = image_index;
	root.indirect_mapping_offset    = mapping_offset;
	root.indirect_search_iterations = 0;
	root.bindless                   = true;
	return true;
}

bool ReadScalarTable(uint64_t base, uint64_t size, uint64_t offset,
                     const SrtRuntime& runtime, std::span<uint32_t> words) {
	return ReadTable(base & AddressMask & ~uint64_t {3}, size, offset & ~uint64_t {3}, runtime, words);
}

bool NormalizeIndirectBuffer(DescriptorValue& value, bool indexed, const BufferResource& resource) {
	ShaderBufferResource descriptor;
	if (!DecodeBufferDescriptor(value, descriptor)) return false;
	const bool formatted_read = resource.formatted && resource.read && !resource.written;
	if (descriptor.Type() != 0u || descriptor.Format() == Prospero::BufferFormat::kInvalid ||
	    (!formatted_read && (descriptor.Base48() == 0u || descriptor.GetSize() == 0u))) {
		value.dwords.fill(0u);
		return true;
	}
	if (descriptor.SwizzleEnabled() || descriptor.AddTid() ||
	    descriptor.OutOfBounds() != (indexed ? 0u : 3u)) {
		return SpecializationFail(fmt::format(
		    "indirect buffers require exact linear index or byte bounds: "
		    "descriptor={:08x},{:08x},{:08x},{:08x} swizzle={} add_tid={} oob={} "
		    "indexed={} formatted={} read={} written={} max_extent={}",
		    value.dwords[0], value.dwords[1], value.dwords[2], value.dwords[3],
		    descriptor.SwizzleEnabled(), descriptor.AddTid(), descriptor.OutOfBounds(), indexed,
		    resource.formatted, resource.read, resource.written, resource.max_byte_extent));
	}
	if (resource.formatted) {
		const auto format = Format::GetFormatInfo(descriptor.Format());
		if (!indexed || format.type == Format::ComponentType::Unknown)
			return SpecializationFail("indirect formatted buffers require a supported indexed format");
		if (!formatted_read) {
			const bool scalar = resource.max_byte_extent == 4u &&
			    (descriptor.Format() == Prospero::BufferFormat::k32UInt ||
			     descriptor.Format() == Prospero::BufferFormat::k8SInt);
			const bool rgba16 = resource.max_byte_extent == 16u &&
			    descriptor.Format() == Prospero::BufferFormat::k16_16_16_16Float;
			if ((!scalar && !rgba16) || descriptor.Stride() != format.byte_size ||
			    descriptor.DstSelXYZW() != (rgba16 ? DstSel(4, 5, 6, 7) : DstSel(4, 0, 0, 0)))
				return SpecializationFail("indirect formatted stores require tight identity SInt8, UInt32 or RGBA16Float records");
		}
		if (formatted_read &&
		    (descriptor.Base48() == 0u || descriptor.Stride() == 0u || descriptor.NumRecords() == 0u)) {
			// Every index is out of bounds. Preserve the format and swizzle constants.
			value.dwords[1] &= ~(0x3fffu << 16u);
			value.dwords[2] = 0u;
			return true;
		}
		const auto alignment = std::min(4u, format.byte_size);
		if (descriptor.Stride() < format.byte_size ||
		    (descriptor.Stride() & (alignment - 1u)) != 0u ||
		    (descriptor.Base48() & (alignment - 1u)) != 0u)
			return SpecializationFail("indirect formatted buffers require aligned full-format records");
	} else if ((!indexed && descriptor.Stride() != 1u) ||
	           (indexed && resource.max_byte_extent != 1u && descriptor.Stride() != 4u)) {
		return SpecializationFail("indirect buffer stores require exact linear DWORD or byte bounds");
	}
	return true;
}

bool IsBoundedDescriptorTable(const ResourcePlan& program,
                              const DescriptorSource::IndirectDescriptor& indirect) {
	const auto* table = Source(program, indirect.table_source);
	return indirect.sources.empty() && (!indirect.selector || indirect.table_scalar) &&
	       (indirect.key_count.IsEmpty() || indirect.table_maximum) && indirect.workgroup_axis == UINT32_MAX &&
	       table != nullptr && table->dword_count == 4u;
}

bool CanReadIndexedSelector(const ShaderBufferResource& material, uint32_t offset) {
	if (material.Type() != 0u || material.SwizzleEnabled() || material.AddTid() ||
	    material.OutOfBounds() != 0u) return false;
	if (material.Format() == Prospero::BufferFormat::kInvalid || material.NumRecords() == 0u ||
	    offset >= material.Stride()) return true;
	return material.Base48() != 0u &&
	       ((material.Base48() | material.Stride() | offset) & 3u) == 0u &&
	       material.NumRecords() <= MaxIndirectDescriptorProbes &&
	       uint64_t {material.NumRecords() - 1u} * material.Stride() + offset + 4u <=
	           uint64_t {UINT32_MAX} + 1u;
}

template <typename Walker>
bool EvaluateIndirectCount(const DescriptorSource::IndirectDescriptor& indirect,
                           const SrtRuntime& runtime, Walker& clean, uint32_t& count) {
	if (indirect.workgroup_axis != UINT32_MAX) {
		if (runtime.workgroup_counts.size() != 3u || indirect.workgroup_axis >= 3u) return false;
		count = std::ranges::find(runtime.workgroup_counts, 0u) != runtime.workgroup_counts.end()
		    ? 0u : runtime.workgroup_counts[indirect.workgroup_axis];
		return true;
	}
	if (!clean.Evaluate(indirect.key_count, count)) return false;
	if (indirect.signed_count && std::bit_cast<int32_t>(count) < 0) count = 0;
	return true;
}

template <typename Walker>
bool ResolveSelectorKeys(const DescriptorSource::IndirectDescriptor& indirect,
                         const DescriptorValue& material_value, const SrtRuntime& runtime,
                         Walker& clean, std::vector<uint32_t>& keys) {
	const auto* selector = &*indirect.selector;
	keys.clear();
	const auto read_selector = [&](uint64_t base, uint64_t size, uint64_t offset,
	                               std::span<uint32_t> words) {
		return selector->global ? ReadTable(base, size, offset, runtime, words)
		                        : ReadScalarTable(base, size, offset, runtime, words);
	};
	const auto read_keys = [&](uint64_t base, uint64_t size, uint64_t first,
	                           uint64_t step, uint64_t count) {
		keys.resize(count);
		if (count == 0u) return true;
		if (step == 4u) {
			return read_selector(base, size, first, keys);
		}
		if (step < 4u) return false;
		for (uint64_t copied = 0; copied < count;) {
			const auto offset = first + copied * step;
			// Scalar reads align each strided offset independently.
			const auto bias = selector->global ? 0u : offset & 3u;
			const auto length = std::min(count - copied,
			    1u + (uint64_t {MaxIndirectDescriptorProbes - 1u} * 4u - bias) / step);
			keys.resize(count + (bias + (length - 1u) * step) / 4u + 1u);
			const auto batch = std::span(keys).subspan(count);
			if (!read_selector(base, size, offset, batch)) {
				keys.resize(count);
				return false;
			}
			for (uint64_t index = 0; index < length; ++index)
				keys[copied + index] = batch[(bias + index * step) / 4u];
			copied += length;
		}
		keys.resize(count);
		return true;
	};
	if (selector->rows) {
		const auto& rows = *selector->rows;
		DescriptorValue count_value;
		ShaderBufferResource counts, material;
		if (!clean.EvaluateDescriptor(rows.source, count_value) ||
		    !DecodeBufferDescriptor(count_value, counts) ||
		    !DecodeBufferDescriptor(material_value, material) || selector->stride != 4u ||
		    selector->offset != 0u || rows.stride == 0u || (rows.stride & 3u) != 0u ||
		    counts.GetSize() / 4u > MaxIndirectDescriptorProbes) return false;
		std::vector<uint32_t> lengths(counts.GetSize() / 4u);
		if (!ReadScalarTable(counts.Base48(), counts.GetSize(), 0u, runtime, lengths)) return false;
		uint32_t total = 0;
		for (size_t row = 0; row < lengths.size(); ++row) {
			const auto length = lengths[row];
			if (length > MaxIndirectDescriptorProbes - total ||
			    (length != 0u && uint64_t {row} * rows.stride + uint64_t {length} * 4u >
			                         UINT32_MAX + 1ull)) return false;
			total += length;
		}
		keys.reserve(total);
		std::vector<uint32_t> batch;
		for (size_t row = 0; row < lengths.size();) {
			if (lengths[row] == 0u) { ++row; continue; }
			const auto first = uint64_t {row} * rows.stride / 4u;
			auto end = first + lengths[row];
			size_t next = row + 1u;
			for (; next < lengths.size(); ++next) {
				if (lengths[next] == 0u) continue;
				const auto candidate_end =
				    std::max(end, uint64_t {next} * rows.stride / 4u + lengths[next]);
				if (candidate_end - first > MaxIndirectDescriptorProbes) break;
				end = candidate_end;
			}
			// Coalesce small gaps, but interpret only selected prefixes. The strict
			// reader captures the entire fetched span for writable-alias rejection.
			batch.resize(end - first);
			if (!ReadScalarTable(material.Base48(), material.GetSize(), first * 4u, runtime, batch))
				return false;
			for (; row < next; ++row) {
				if (lengths[row] == 0u) continue;
				const auto begin = batch.begin() + (uint64_t {row} * rows.stride / 4u - first);
				keys.insert(keys.end(), begin, begin + lengths[row]);
			}
		}
	} else if (selector->bitset) {
		ShaderBufferResource material;
		std::array<uint32_t, 4> masks {};
		uint32_t first = 0, count = 0;
		if (!DecodeBufferDescriptor(material_value, material) ||
		    !clean.Evaluate(indirect.selector_first, first) ||
		    !EvaluateIndirectCount(indirect, runtime, clean, count) || count != masks.size() * 32u ||
		    selector->stride != sizeof(masks) || selector->offset != 0u ||
		    material.Type() != 0u || material.Stride() != sizeof(masks) ||
		    material.SwizzleEnabled() || material.AddTid() || material.OutOfBounds() != 0u ||
		    (material.Base48() & 3u) != 0u) return false;
		if (material.Format() != Prospero::BufferFormat::kInvalid) {
			const auto words = material.GetSize() / sizeof(uint32_t);
			// Bitmap storage can grow with resolution while its candidate domain stays
			// at 128 keys. Bound scratch memory, preserving each coherent read's capture.
			for (uint64_t first_word = 0; first_word < words; first_word += keys.size()) {
				keys.resize(std::min<uint64_t>(words - first_word, MaxIndirectDescriptorProbes));
				if (!ReadScalarTable(material.Base48(), material.GetSize(), first_word * 4u,
				                     runtime, keys)) return false;
				for (size_t word = 0; word < keys.size(); ++word) masks[word % masks.size()] |= keys[word];
				if (std::ranges::all_of(masks, [](uint32_t mask) { return mask == UINT32_MAX; })) break;
			}
			keys.clear();
			for (uint32_t word = 0; word < masks.size(); ++word) {
				auto mask = masks[word];
				while (mask != 0u) {
					const auto rank = word * 32u + std::countr_zero(mask);
					keys.push_back(first + count - 1u - rank);
					mask &= mask - 1u;
				}
			}
		}
	} else if (!selector->byte_ranges.empty()) {
		if (material_value.dword_count != 2u) return false;
		const auto base = (uint64_t {material_value.dwords[1]} << 32u) | material_value.dwords[0];
		std::array<bool, 256> selected {};
		for (const auto [offset, size]: selector->byte_ranges) {
			if (size > MaxIndirectDescriptorProbes || ((offset | size) & 3u) != 0u) return false;
			keys.resize(size / 4u);
			if (!ReadScalarTable(base, UINT64_MAX, offset, runtime, keys)) return false;
			for (const auto word: keys)
				for (uint32_t shift = 0; shift < 32u; shift += 8u)
					selected[(word >> shift) & 255u] = true;
		}
		keys.clear();
		for (uint32_t key = 0; key < selected.size(); ++key)
			if (selected[key]) keys.push_back(key);
	} else if (!indirect.selector_first.IsEmpty()) {
		ShaderBufferResource material;
		uint64_t             base = 0, size = UINT64_MAX;
		uint32_t             first = 0, count = 0;
		if (material_value.dword_count == 2u) {
			if (selector->stride == 0u) return false;
			base = (uint64_t {material_value.dwords[1]} << 32u) | material_value.dwords[0];
		} else {
			if (!DecodeBufferDescriptor(material_value, material)) return false;
			base = material.Base48();
			size = material.GetSize();
		}
		if (!clean.Evaluate(indirect.selector_first, first) ||
		    !EvaluateIndirectCount(indirect, runtime, clean, count) || count > MaxIndirectDescriptorProbes)
			return false;
		// SMEM uses its instruction's byte stride; V# stride only supplies its bounds.
		const auto step = selector->stride != 0u ? selector->stride : material.Stride();
		if (selector->stride == 0u &&
		    (material.Type() != 0u || material.SwizzleEnabled() || material.AddTid() ||
		     material.OutOfBounds() != 0u || uint64_t {selector->offset} + 4u > step ||
		     uint64_t {first} + count > material.NumRecords())) return false;
		const auto last = count != 0u ? uint64_t {first} + count - 1u : 0u;
		if (last > UINT32_MAX) return false;
		const auto end = count != 0u ? last * step + selector->offset + 4u : 0u;
		if (end > uint64_t {UINT32_MAX} + 1u ||
		    (indirect.workgroup_axis == UINT32_MAX && end > size))
			return false;
		if (!read_keys(base, size, uint64_t {first} * step + selector->offset, step, count))
			return false;
	} else if (!indirect.selector_mask.IsEmpty()) {
		uint32_t mask = 0;
		uint32_t count = 0;
		if (material_value.dword_count != 2u ||
		    !clean.Evaluate(indirect.selector_mask, mask) ||
		    !EvaluateIndirectCount(indirect, runtime, clean, count) || count == 0u || count > 32u) {
			return false;
		}
		if (count < 32u) mask &= (1u << count) - 1u;
		const auto material_base =
		    (static_cast<uint64_t>(material_value.dwords[1]) << 32u) | material_value.dwords[0];
		keys.reserve(std::popcount(mask));
		while (mask != 0u) {
			const auto index = std::countr_zero(mask);
			const auto offset = static_cast<uint64_t>(selector->offset) +
			                    static_cast<uint64_t>(index) * selector->stride;
			if (offset > UINT32_MAX) return false;
			uint32_t key = 0;
			if (!read_selector(material_base, UINT64_MAX, static_cast<uint32_t>(offset),
			                   {&key, 1})) return false;
			keys.push_back(key);
			mask &= mask - 1u;
		}
	} else if (selector->stride == 0u) {
		ShaderBufferResource material;
		if (!DecodeBufferDescriptor(material_value, material) ||
		    !CanReadIndexedSelector(material, selector->offset)) return false;
		if (material.Format() != Prospero::BufferFormat::kInvalid && material.NumRecords() != 0u &&
		    selector->offset < material.Stride() &&
		    !read_keys(material.Base48(), material.GetSize(), selector->offset,
		               material.Stride(), material.NumRecords())) return false;
		keys.push_back(0u); // Every out-of-bounds indexed load returns zero.
	} else {
		ShaderBufferResource material;
		if (!DecodeBufferDescriptor(material_value, material) || selector->stride < 4u ||
		    !std::has_single_bit(selector->stride) || (selector->offset & 3u) != 0u)
			return false;
		// Empty selector bounds denote the complete U32 congruence class. SMEM
		// adds its aligned immediate after the dynamic offset, without U32 wrap.
		const auto domain_count = (uint64_t {1} << 32u) / selector->stride;
		const auto available = material.GetSize() > selector->offset
		    ? material.GetSize() - selector->offset : 0u;
		const auto count = std::min(domain_count,
		    available < 4u ? 0u : (available - 4u) / selector->stride + 1u);
		if (count > MaxIndirectDescriptorProbes ||
		    !read_keys(material.Base48(), material.GetSize(), selector->offset, selector->stride, count)) return false;
		if (count < domain_count) keys.push_back(0u); // An out-of-bounds scalar read returns zero.
	}
	if (selector->byte_ranges.empty()) {
		if (selector->bit_offset != 0u || selector->bit_mask != UINT32_MAX || selector->maximum != UINT32_MAX)
			for (auto& key: keys)
				key = std::min((key >> selector->bit_offset) & selector->bit_mask, selector->maximum);
		std::ranges::sort(keys);
		keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
	}
	return true;
}

template <typename Specialization, typename Normalize, typename Walker>
bool MaterializeIndirectDescriptor(const ResourcePlan&                         program,
                                   uint32_t source_index,
                                   uint32_t resource_index, uint32_t dword_count, uint32_t read_dwords,
                                   const SrtRuntime& runtime, Walker& clean,
                                   ResourceSnapshot&             snapshot,
                                   std::vector<DescriptorValue>& descriptors,
                                   std::vector<Specialization>&  specializations,
                                   Normalize&& normalize) {
	const auto& indirect = *program.descriptor_sources[source_index].indirect_descriptor;
	const auto  descriptor_bytes = read_dwords * sizeof(uint32_t);
	const auto& sources = indirect.sources;
	const auto* selector = indirect.selector ? &*indirect.selector : nullptr;
	const bool byte_keys = IsBoundedDescriptorTable(program, indirect);
	auto* selector_keys = selector != nullptr
	    ? &program.selector_keys[program.selector_groups[source_index]] : nullptr;
	auto& keys = selector_keys != nullptr ? selector_keys->values : program.material_keys;
	auto& byte_offsets = program.material_keys;
	const auto children_begin = descriptors.size();
	const auto root_resource = specializations[resource_index];
	const auto intern_candidate = [&](const DescriptorValue& candidate) {
		if (candidate == descriptors[resource_index]) return 0u;
		const auto found = std::find(descriptors.begin() + children_begin, descriptors.end(), candidate);
		const auto ordinal = static_cast<uint32_t>(found - descriptors.begin() - children_begin + 1u);
		if (found == descriptors.end()) {
			descriptors.push_back(candidate);
			auto child = root_resource;
			child.indirect_root = resource_index;
			specializations.push_back(child);
		}
		return ordinal;
	};
	uint64_t table_base = 0;
	uint64_t table_size = UINT64_MAX; // Scalar addresses have no buffer descriptor bounds.
	DescriptorValue table_value;
	ShaderBufferResource table;
	const auto failed = [&](std::string_view reason, std::string_view details = {}) {
		const auto* pointer = indirect.pointer_table ? &*indirect.pointer_table : nullptr;
		return SpecializationFail(fmt::format(
		    "indirect {} resource={} source={} table_source={} words={:08x},{:08x},{:08x},{:08x} "
		    "base={:#x} size={} type={} runtime_stride={} records={} swizzle={} addtid={} oob={} "
		    "maximum={} offset={} immediate={} record_bytes={} child_count={} child_stride={} "
		    "child_offset={} indexed_pointer={} candidates={} {}",
		    reason, resource_index, source_index, indirect.table_source,
		    table_value.dwords[0], table_value.dwords[1], table_value.dwords[2], table_value.dwords[3],
		    table_base, table_size, table.Type(), table.Stride(), table.NumRecords(),
		    table.SwizzleEnabled(), table.AddTid(), table.OutOfBounds(),
		    indirect.table_maximum ? fmt::format("{}", *indirect.table_maximum) : "none",
		    indirect.table_offset, indirect.table_immediate, indirect.table_record_bytes,
		    pointer != nullptr ? pointer->count : 1u, pointer != nullptr ? pointer->stride : 0u,
		    pointer != nullptr ? pointer->offset : 0u, pointer != nullptr && pointer->indexed,
		    descriptors.size(), details));
	};
	const auto read_failed = [&](std::string_view reason, uint32_t row, uint32_t child,
	                             uint64_t base, uint64_t offset, size_t bytes) {
		return failed(reason, fmt::format("row={} child={} read_base={:#x} read_offset={:#x} "
		                                 "read_address={:#x} requested_bytes={}", row, child, base, offset,
		    (base & AddressMask & ~uint64_t {3}) + (offset & ~uint64_t {3}), bytes));
	};
	const auto normalization_failed = [&](uint64_t offset, uint32_t key, uint64_t step,
	                                      std::string_view key_kind) {
		return failed("candidate", fmt::format("candidate_offset={:#x} {}={} stride={} step={}",
		    offset & ~uint64_t {3}, key_kind, key, indirect.table_stride, step));
	};
	if (sources.empty()) {
		if (selector == nullptr) keys.clear();
		DescriptorValue material_value;
		if ((selector != nullptr && (byte_keys || !selector_keys->resolved) &&
		     !clean.EvaluateDescriptor(selector->source, material_value)) ||
		    !clean.EvaluateDescriptor(indirect.table_source, table_value)) {
			return FailIndirect(__LINE__);
		}
		if (table_value.dword_count == 2u) {
			table_base = (static_cast<uint64_t>(table_value.dwords[1]) << 32u) | table_value.dwords[0];
		} else if (DecodeBufferDescriptor(table_value, table)) {
			table_base = table.Base48();
			table_size = table.GetSize();
		} else {
			return FailIndirect(__LINE__);
		}
		if (byte_keys && table_size > UINT32_MAX) return failed("table geometry");
		bool use_selector = selector != nullptr;
		if (byte_keys && selector != nullptr) {
			ShaderBufferResource material;
			if (!DecodeBufferDescriptor(material_value, material)) return false;
			use_selector = CanReadIndexedSelector(material, selector->offset);
			if (use_selector && material.Format() != Prospero::BufferFormat::kInvalid &&
			    material.NumRecords() != 0u && selector->offset < material.Stride()) {
				const auto begin = material.Base48() + selector->offset;
				const auto end = begin + uint64_t {material.NumRecords() - 1u} * material.Stride() + 4u;
				for (uint32_t i = 0; i < program.info.buffers.size(); ++i) {
					const auto& buffer = program.info.buffers[i];
					if (!buffer.written) continue;
					ShaderBufferResource written;
					if (program.descriptor_sources[buffer.source].indirect_descriptor ||
					    !DecodeBufferDescriptor(snapshot.buffers[i], written) ||
					    (written.Base48() != 0u && written.GetSize() != 0u &&
					     begin < written.Base48() + written.GetSize() &&
					     written.Base48() < end)) {
						use_selector = false;
						break;
					}
				}
			}
		}
		if (byte_keys && !use_selector) {
			if (!indirect.table_scalar && table.Format() == Prospero::BufferFormat::kInvalid) {
				descriptors[resource_index] = {.dword_count = dword_count};
				return true;
			}
			if (indirect.table_record_bytes != 0u &&
			     (table.Type() != 0u || table.SwizzleEnabled() || table.AddTid() ||
			      table.OutOfBounds() != 0u || (table_base & 3u) != 0u ||
			      (table.Stride() & 3u) != 0u || table.Stride() < indirect.table_record_bytes))
				return failed("table geometry");
			const auto* pointer = indirect.pointer_table ? &*indirect.pointer_table : nullptr;
			const bool indexed_pointer = pointer != nullptr && pointer->indexed;
			const auto child_count = pointer != nullptr ? pointer->count : 1u;
			uint64_t domain = indirect.table_maximum ? uint64_t {*indirect.table_maximum} + 1u
			                                       : uint64_t {1} << 32u;
			if (!indirect.key_count.IsEmpty()) {
				uint32_t key_count = 0u;
				if (!EvaluateIndirectCount(indirect, runtime, clean, key_count)) return false;
				domain = std::min(domain, uint64_t {key_count});
			}
			if (indexed_pointer &&
			    (!indirect.table_maximum || table.Type() != 0u || table.SwizzleEnabled() || table.AddTid() ||
			     table.OutOfBounds() != 0u || (table_base & 3u) != 0u || (table.Stride() & 3u) != 0u ||
			     uint64_t {indirect.table_immediate} + 8u > table.Stride() ||
			     domain > table.NumRecords()))
				return failed("indexed pointer geometry", fmt::format("rows={}", domain));
			const bool affine = indirect.table_stride != 0u || indexed_pointer;
			const bool image_heap = dword_count == 8u && pointer == nullptr;
			const uint64_t step = indexed_pointer ? table.Stride()
			    : indirect.table_maximum ? indirect.table_stride
			    : affine ? std::max(uint64_t {1} << std::countr_zero(indirect.table_stride), uint64_t {4})
			             : 4u;
			const uint64_t first = ((indirect.table_maximum ? indirect.table_offset
			    : indirect.table_offset % step) & ~uint64_t {3}) +
			                       (indirect.table_immediate & ~3u);
			if (!indirect.table_maximum) domain /= step;
			const auto count = table_size >= first + 4u
			    ? std::min((table_size - first - 4u) / step + 1u, domain)
			    : 0u;
			if (child_count == 0u || (!image_heap && count * child_count > MaxIndirectDescriptorProbes))
				return failed("candidate count", fmt::format("rows={} limit={}", count, MaxIndirectDescriptorProbes));
			// Affine SMEM offsets wrap in U32 before adding the aligned immediate.
			// Image heaps retain one bulk snapshot and their absolute byte-offset mapping.
			auto& heap_words = program.material_keys;
			if (image_heap) {
				heap_words.resize(table_size / 4u);
				if (!ReadScalarTable(table_base, table_size, 0u, runtime, heap_words))
					return read_failed("image heap read", 0u, 0u, table_base, 0u, heap_words.size() * sizeof(uint32_t));
			}
			const auto mapping_offset = snapshot.flattened_srt.size();
			descriptors[resource_index] = {.dword_count = dword_count};
			snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
			for (uint32_t word = 0; word < count; ++word) {
				const auto table_offset = affine ? first + word * step : uint64_t {word} * 4u;
				uint64_t pointer_base = 0u;
				if (pointer != nullptr) {
					std::array<uint32_t, 2> address {};
					if (!ReadScalarTable(table_base, table_size, table_offset, runtime, address))
						return read_failed("parent pointer read", word, 0u, table_base, table_offset, sizeof(address));
					pointer_base = (uint64_t {address[1]} << 32u) | address[0];
				}
				for (uint32_t child = 0; child < child_count; ++child) {
					DescriptorValue candidate {.dword_count = dword_count};
					if (pointer != nullptr) {
						if (!ReadScalarTable(pointer_base, UINT64_MAX,
						        uint64_t {child} * pointer->stride + pointer->offset, runtime,
						        std::span(candidate.dwords).first(read_dwords)))
							return read_failed("child descriptor read", word, child, pointer_base,
							    uint64_t {child} * pointer->stride + pointer->offset, descriptor_bytes);
					} else if (!image_heap) {
						if (!ReadScalarTable(table_base, table_size, table_offset, runtime,
						                     std::span(candidate.dwords).first(read_dwords))) return false;
					} else {
						const auto table_word = table_offset / 4u;
						std::copy_n(heap_words.begin() + table_word,
						            std::min<size_t>(read_dwords, heap_words.size() - table_word),
						            candidate.dwords.begin());
					}
					if (!normalize(candidate))
						return normalization_failed(table_offset, word, step,
						                            indirect.table_maximum ? "key" : "probe");
					const auto ordinal = intern_candidate(candidate);
					if (ordinal == 0u) continue;
					const auto mapping_word = image_heap ? table_offset / 4u : word * child_count + child;
					snapshot.flattened_srt.resize(mapping_offset + 2u + mapping_word, 0u);
					snapshot.flattened_srt[mapping_offset + 1u + mapping_word] = ordinal;
				}
			}

			if (descriptors.size() == children_begin) {
				snapshot.flattened_srt.resize(mapping_offset);
			} else {
				snapshot.flattened_srt[mapping_offset] =
				    static_cast<uint32_t>(snapshot.flattened_srt.size() - mapping_offset - 1u);
				auto& root = specializations[resource_index];
				root.indirect_root = resource_index;
				root.indirect_mapping_offset = static_cast<uint32_t>(mapping_offset);
				root.indirect_search_iterations = 0u;
			}
			return true;
		}
		if (selector == nullptr) {
			uint32_t key_count = 0;
			uint32_t first_key = 0;
			if (!EvaluateIndirectCount(indirect, runtime, clean, key_count)) return false;
			if (!indirect.selector_first.IsEmpty() && !clean.Evaluate(indirect.selector_first, first_key))
				return false;
			if (indirect.table_stride == 0u || key_count > MaxIndirectDescriptorProbes ||
			    (indirect.selector_first.IsEmpty() && key_count != 0u &&
			         uint64_t {indirect.table_offset} + indirect.table_immediate +
			         uint64_t {key_count - 1u} * indirect.table_stride + descriptor_bytes >
			             UINT32_MAX + 1ull)) {
				return FailIndirect(__LINE__);
			}
			keys.resize(key_count);
			std::iota(keys.begin(), keys.end(), first_key);
			if (!keys.empty() && keys.back() < keys.front())
				std::rotate(keys.begin(), keys.begin() + (UINT32_MAX - first_key + 1ull), keys.end());
		} else {
			if (!indirect.selector_mask.IsEmpty() && table_value.dword_count != 2u) return false;
			if (!selector_keys->resolved) {
				if (!ResolveSelectorKeys(indirect, material_value, runtime, clean, keys)) return false;
				selector_keys->resolved = true;
			}
		}
		if (byte_keys) {
			// Keep cached selector values independent of each destination table's
			// U32 arithmetic, and use the same byte keys as the full-heap GPU path.
			byte_offsets.clear();
			byte_offsets.reserve(keys.size());
			for (const auto key: keys) {
				const auto offset = uint64_t {(key * indirect.table_stride + indirect.table_offset) & ~3u} +
				                    (indirect.table_immediate & ~3u);
				byte_offsets.push_back(offset <= UINT32_MAX ? static_cast<uint32_t>(offset) : UINT32_MAX);
			}
			std::ranges::sort(byte_offsets);
			byte_offsets.erase(std::unique(byte_offsets.begin(), byte_offsets.end()), byte_offsets.end());
		}
	}

	const auto mapping_offset = snapshot.flattened_srt.size();
	const auto selected_keys = byte_keys ? std::span<const uint32_t>(byte_offsets)
	                                    : std::span<const uint32_t>(keys);
	const auto key_count = sources.empty() ? selected_keys.size() : sources.size();
	// The mapping's layout is part of the shader permutation (its offset in the flattened SRT
	// and the number of search steps), while the number of keys follows guest data from dispatch
	// to dispatch (GTA V's ray tracing shader saw dozens of permutations, each a multi-second
	// pipeline compile). Reserve room for a power-of-two number of keys instead: the shader
	// searches the runtime count, and later mappings keep their offsets until a count crosses a
	// power of two.
	// Compute shaders can keep one layout for more keys (see SetComputeIndirectFloors()).
	const bool stable_layout = g_stable_indirect_layouts.load(std::memory_order_relaxed);
	const auto key_capacity  = !stable_layout ? key_count : std::max<size_t>(
	    program.stage == ShaderType::Compute ? g_compute_indirect_keys.load(std::memory_order_relaxed)
	                                         : MinIndirectMappingCapacity,
	    std::bit_ceil(key_count));
	snapshot.flattened_srt.resize(mapping_offset + 1u + key_capacity * 2u);
	snapshot.flattened_srt[mapping_offset] = static_cast<uint32_t>(key_count);
	for (uint32_t entry = 0; entry < key_count; ++entry) {
		const auto key = sources.empty() ? selected_keys[entry] : entry;
		uint64_t table_offset = 0;
		DescriptorValue candidate;
		candidate.dword_count = dword_count;
		if (sources.empty()) {
			table_offset = byte_keys ? key
			    : uint64_t {(key * indirect.table_stride + indirect.table_offset) & ~3u} +
			          (indirect.table_immediate & ~3u);
			if (!(byte_keys && key == UINT32_MAX) &&
			    !ReadScalarTable(table_base, table_size, table_offset, runtime,
			                     std::span(candidate.dwords).first(read_dwords))) {
				return read_failed("table descriptor read", key, 0u, table_base, table_offset, descriptor_bytes);
			}
		} else if (!clean.EvaluateDescriptor(sources[entry], candidate)) {
			return FailIndirect(__LINE__);
		}
		if (!normalize(candidate))
			return normalization_failed(table_offset, key, indirect.table_stride, "key");
		uint32_t ordinal = 0;
		if (entry == 0 && !byte_keys) {
			descriptors[resource_index] = candidate;
		} else {
			ordinal = intern_candidate(candidate);
		}
		snapshot.flattened_srt[mapping_offset + 1u + entry * 2u] = key;
		snapshot.flattened_srt[mapping_offset + 2u + entry * 2u] = ordinal;
	}
	if (key_count > 4096u) {
		static std::atomic<uint32_t> printed {0};
		if (printed.fetch_add(1, std::memory_order_relaxed) < 16) {
			uint32_t nulls = 0;
			for (auto d = children_begin; d < descriptors.size(); ++d)
				nulls += std::ranges::all_of(descriptors[d].dwords, [](uint32_t w) { return w == 0u; });
			std::fprintf(stderr, "[material-descs] %016llx keys=%zu distinct=%zu null=%u root_null=%d\n",
			             static_cast<unsigned long long>(program.shader_hash), size_t(key_count),
			             descriptors.size() - children_begin, nulls,
			             std::ranges::all_of(descriptors[resource_index].dwords,
			                                 [](uint32_t w) { return w == 0u; }) ? 1 : 0);
		}
	}
	if (descriptors.size() == children_begin) {
		snapshot.flattened_srt.resize(mapping_offset);
	} else {
		if (std::is_same_v<Specialization, ResourceSpecialization::Image> && stable_layout) {
			// The same for the distinct T#s: pad them with copies of the last one, which no key
			// selects, so that with the root they are a power of two (at least 8 for compute).
			const auto children = descriptors.size() - children_begin;
			const auto total    = std::max<size_t>(
                program.stage == ShaderType::Compute
                    ? g_compute_indirect_images.load(std::memory_order_relaxed)
                    : 1u,
                std::bit_ceil(children + 1u));
			for (auto padded = children;
			     padded + 1u < total; ++padded) {
				descriptors.push_back(descriptors.back());
				specializations.push_back(specializations.back());
			}
		}
		auto& root                      = specializations[resource_index];
		root.indirect_root              = resource_index;
		root.indirect_mapping_offset = static_cast<uint32_t>(mapping_offset);
		root.indirect_search_iterations = std::bit_width(key_capacity);
	}
	return true;
}

// Exemplar that specializes a table root: a T# with the root's dimension and a native format,
// preferring the float textures that sampled material tables mostly hold.
bool TableExemplar(const ImageResource& root, const DescriptorValue& descriptor, bool float_only) {
	if (NullImageDescriptor(descriptor) || DescriptorIsCube(descriptor) != root.cube) {
		return false;
	}
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	const auto numeric_class = Prospero::SampledTextureNumericClass(format);
	return DescriptorDimension(descriptor, root.dimension) == root.dimension &&
	       numeric_class != Prospero::TextureNumericClass::Unsupported &&
	       (!float_only || numeric_class == Prospero::TextureNumericClass::Float) &&
	       ImageConversionFormat(format) == Prospero::BufferFormat::kInvalid &&
	       !Prospero::IsFmaskTextureFormat(format);
}

uint32_t ImageTableCapacity(size_t slots) {
	return std::bit_ceil(std::max<uint32_t>(static_cast<uint32_t>(slots), 256u));
}

// A bindless table: the key the shader computes from GPU data selects entry key & entry_mask,
// so every entry the table holds is a candidate. Distinct valid T#s become descriptor slots and
// the per-entry slot mapping is appended to the flattened SRT.
bool MaterializeImageTable(const ResourcePlan& program,
                           const DescriptorSource::IndirectDescriptor& indirect,
                           const DescriptorValue& table_value, uint32_t image_index,
                           const SrtRuntime& runtime, ResourceSnapshot& snapshot,
                           ResourceSpecialization& specialization,
                           std::vector<ImageTableSnapshot>& previous,
                           std::vector<uint32_t>& mapping_offsets) {
	ShaderBufferResource table;
	if (table_value.dword_count != 4u || !DecodeBufferDescriptor(table_value, table) ||
	    (indirect.table_offset & 31u) != 0u) {
		return false;
	}
	const auto base       = table.Base48();
	const auto size       = table.GetSize();
	const auto entry_mask = indirect.key_mask;
	const auto entries    = std::min<uint64_t>(
	    {size > indirect.table_offset ? (size - indirect.table_offset) / 32u : 0u,
	     uint64_t {entry_mask} + 1u, MaxIndirectDescriptorProbes});
	const auto& root = program.info.images[image_index];
	const auto same_table = [&](const ImageTableSnapshot& candidate) {
		return candidate.base == base && candidate.size == size &&
		       candidate.offset == indirect.table_offset && candidate.entry_mask == entry_mask;
	};
	auto found = std::ranges::find_if(snapshot.image_tables, same_table);
	if (found == snapshot.image_tables.end()) {
		ImageTableSnapshot next {.base = base, .size = size, .offset = indirect.table_offset,
		                         .entry_mask = entry_mask};
		// Tables span megabytes and rarely change: read into a reused buffer and keep the
		// previous snapshot's words when they are equal.
		static thread_local std::vector<uint32_t> words;
		words.resize(static_cast<size_t>(entries) * 8u);
		if (!words.empty() && !ReadScalarTable(base, size, indirect.table_offset, runtime, words)) {
			return false;
		}
		const auto reusable = std::ranges::find_if(previous, [&](const ImageTableSnapshot& old) {
			return same_table(old) && old.raw == words && !old.slots.empty();
		});
		// The tables this thread built last, by any program: resolutions ahead of execution
		// materialize into a ring of snapshots, so `previous` rarely holds the table.
		static thread_local std::array<ImageTableSnapshot, 4> built;
		static thread_local uint32_t                          built_next = 0;
		const auto cached = reusable != previous.end()
		                        ? built.end()
		                        : std::ranges::find_if(built, [&](const ImageTableSnapshot& old) {
			                          return same_table(old) && !old.slots.empty() &&
			                                 old.raw == words;
		                          });
		if (reusable != previous.end()) {
			next.raw     = std::move(reusable->raw);
			next.slots   = std::move(reusable->slots);
			next.mapping = std::move(reusable->mapping);
		} else if (cached != built.end()) {
			next.raw.swap(words);
			next.slots   = cached->slots;
			next.mapping = cached->mapping;
		} else {
			next.raw.swap(words);
			next.slots.push_back(DescriptorValue {.dword_count = 8u});
			next.mapping.assign(static_cast<size_t>(entries) + 1u, 0u);
			next.mapping[0] = static_cast<uint32_t>(entries);
			std::unordered_map<std::string_view, uint32_t> slot_of;
			for (uint32_t entry = 0; entry < entries; entry++) {
				DescriptorValue candidate {.dword_count = 8u};
				std::copy_n(next.raw.begin() + entry * 8u, 8u, candidate.dwords.begin());
				if (NullImageDescriptor(candidate)) {
					continue;
				}
				if (!ValidImageDescriptor(candidate, root.r128)) {
					Common::HackNotice::Report("invalid T# in a bindless table bound as a null image",
					                           "MaterializeImageTable", program.shader_hash, 0,
					                           "words that only look like a T# are not bound");
					continue;
				}
				const std::string_view key(
				    reinterpret_cast<const char*>(next.raw.data() + entry * 8u),
				    8u * sizeof(uint32_t));
				const auto [slot, inserted] =
				    slot_of.try_emplace(key, static_cast<uint32_t>(next.slots.size()));
				if (inserted) {
					next.slots.push_back(candidate);
				}
				next.mapping[entry + 1u] = slot->second;
			}
			built[built_next++ % built.size()] = next;
		}
		snapshot.image_tables.push_back(std::move(next));
		mapping_offsets.push_back(static_cast<uint32_t>(snapshot.flattened_srt.size()));
		const auto& mapping = snapshot.image_tables.back().mapping;
		snapshot.flattened_srt.insert(snapshot.flattened_srt.end(), mapping.begin(), mapping.end());
		found = snapshot.image_tables.end() - 1;
	}
	const auto table_index     = static_cast<uint32_t>(found - snapshot.image_tables.begin());
	auto&      image           = specialization.images[image_index];
	image.table                = table_index;
	image.table_capacity       = ImageTableCapacity(found->slots.size());
	image.table_mapping_offset = mapping_offsets[table_index];
	image.table_entry_mask     = entry_mask;
	snapshot.images[image_index] = {.dword_count = 8u};
	for (const bool float_only: {true, false}) {
		const auto exemplar = std::ranges::find_if(found->slots, [&](const DescriptorValue& slot) {
			return TableExemplar(root, slot, float_only);
		});
		if (exemplar != found->slots.end()) {
			snapshot.images[image_index] = *exemplar;
			break;
		}
	}
	return true;
}

} // namespace

bool ImageTableSlotCompatible(const ImageResource& root, const DescriptorValue& descriptor) {
	if (NullImageDescriptor(descriptor) || !ValidImageDescriptor(descriptor, root.r128) ||
	    DescriptorIsCube(descriptor) != root.cube) {
		return false;
	}
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	const auto dimension = DescriptorDimension(descriptor, root.dimension);
	const auto is_2d     = [](Decoder::ImageDimension value) {
		return value == Decoder::ImageDimension::Dim2D ||
		       value == Decoder::ImageDimension::Dim2DArray;
	};
	return (dimension == root.dimension || (is_2d(dimension) && is_2d(root.dimension))) &&
	       Prospero::SampledTextureNumericClass(format) == root.numeric_class &&
	       ImageConversionFormat(format) == root.conversion_format &&
	       !Prospero::IsFmaskTextureFormat(format);
}

struct SamplerPlan {
	struct Binding {
		uint32_t     source;
		SamplerClass type;
	};
	std::vector<std::array<uint32_t, 2>> mapping;
	std::vector<Binding>                bindings;
};

template <typename T, typename Keep>
void CompactImages(std::vector<T>& images, Keep&& keep) {
	size_t count = 0;
	for (size_t index = 0; index < images.size(); ++index) {
		if (!keep(index)) continue;
		if (count != index) images[count] = std::move(images[index]);
		++count;
	}
	images.resize(count);
}

struct ImageRemap {
	explicit ImageRemap(const ResourceSpecialization& specialization)
	    : indices(specialization.images.size()),
	      source_count(static_cast<uint32_t>(specialization.images.size())) {
		for (uint32_t index = 0; index < source_count; index++) {
			indices[index] = specialization.images[index].fmask ? UINT32_MAX : count++;
		}
	}

	uint32_t operator[](uint32_t index) const {
		EXIT_IF(index >= source_count);
		return indices[index];
	}

	// what names the use, for the fatal message.
	uint32_t operator()(uint32_t index, const char* what) const {
		if (index >= source_count) {
			EXIT("image remap: %s refers to image %u, but the shader has %u\n", what, index,
			     source_count);
		}
		return indices[index];
	}

	template <typename T>
	void Apply(std::vector<T>& images) const {
		EXIT_IF(images.size() != source_count);
		if (count == source_count) {
			return;
		}
		CompactImages(images, [&](size_t index) { return indices[index] != UINT32_MAX; });
	}

private:
	std::vector<uint32_t> indices;
	uint32_t                                    source_count;
	uint32_t                                    count = 0;
};

static bool BuildResourceSpecialization(const ResourcePlan& program, ResourceSnapshot& snapshot,
                                        ResourceSpecialization& specialization,
                                        bool                    float_image_atomics) {
	for (uint32_t i = 0; i < specialization.images.size(); i++) {
		const auto& descriptor = snapshot.images[i];
		auto&       image      = specialization.images[i];
		const auto  base_index = i < program.info.images.size() ? i : image.indirect_root;
		if (base_index >= program.info.images.size()) {
			return SpecializationFail(fmt::format("image resource {} has an invalid root", i));
		}
		const auto& base = program.info.images[base_index];
		if (base.resource_class == ImageResourceClass::None ||
		    (base.atomic && base.resource_class != ImageResourceClass::Storage)) {
			return SpecializationFail(fmt::format("image resource {} has an invalid class", i));
		}
		image.mip_count = ImageMipCount(base, descriptor);
		if (NullImageDescriptor(descriptor)) {
			image.numeric_class = base.atomic ? Prospero::TextureNumericClass::Uint
			                                  : Prospero::TextureNumericClass::Float;
			image.dimension     = Decoder::ImageDimension::Dim2D;
			image.cube          = false;
			if (image.bindless) {
				// No one descriptor describes a bindless image: keep the dimension the
				// instruction samples, which picks the set-1 array.
				image.dimension = base.dimension == Decoder::ImageDimension::Unknown
				                      ? Decoder::ImageDimension::Dim2D
				                      : base.dimension;
				image.cube      = base.cube;
			}
			continue;
		}
		const auto descriptor_dimension = DescriptorDimension(descriptor, base.dimension);
		if (descriptor_dimension == Decoder::ImageDimension::Unknown) {
			return SpecializationFail(fmt::format(
			    "image descriptor {} has unsupported type {}: {:08x},{:08x},{:08x},{:08x},"
			    "{:08x},{:08x},{:08x},{:08x}",
			    i, (descriptor.dwords[3] >> 28u) & 0xfu, descriptor.dwords[0], descriptor.dwords[1],
			    descriptor.dwords[2], descriptor.dwords[3], descriptor.dwords[4],
			    descriptor.dwords[5], descriptor.dwords[6], descriptor.dwords[7]));
		}
		image.dimension = descriptor_dimension;
		image.cube      = DescriptorIsCube(descriptor);
		const auto format =
		    static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
		// Image atomics operate on the raw texel bits, and only the integer atomics decode, so an
		// atomic on an R32 sint or float surface is a uint atomic on the same bits.
		// --no-float-image-atomics still turns the float case off.
		const bool float_atomic =
		    float_image_atomics && base.atomic && format == Prospero::BufferFormat::k32Float;
		if (base.atomic &&
		    (base.atomic64 ? format != Prospero::BufferFormat::k32_32UInt
		                   : format != Prospero::BufferFormat::k32UInt &&
		                         format != Prospero::BufferFormat::k32SInt && !float_atomic)) {
			return SpecializationFail(
			    fmt::format("atomic image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
		const bool storage      = base.resource_class == ImageResourceClass::Storage;
		image.fmask             = Prospero::IsFmaskTextureFormat(format);
		if (image.fmask) {
			if (storage || base.depth_compare ||
			    image.indirect_root != ImageResource::NoIndirectImage ||
			    std::ranges::any_of(program.info.sampled_pairs,
			                        [&](const auto& pair) { return pair.image == i; })) {
				return SpecializationFail("FMASK requires a direct image load");
			}
		}
		image.conversion_format = ImageConversionFormat(format);
		if (storage || image.conversion_format != Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = DescriptorImageSwizzle(descriptor);
		}
		const auto raw_sint_format = RawSignedStorageFormat(base, format);
		const bool raw_sint_storage = raw_sint_format != Prospero::BufferFormat::kInvalid;
		if (raw_sint_format == Prospero::BufferFormat::k16UInt) image.conversion_format = format;
		image.numeric_class = Prospero::SampledTextureNumericClass(
		    Prospero::RemapTextureFormat(format));
		if (!storage && format == Prospero::BufferFormat::k11_11_10UNorm) {
			if (image.indirect_root != ImageResource::NoIndirectImage) {
				return SpecializationFail("packed UNorm sampling requires a direct image");
			}
			if (std::ranges::any_of(program.info.sampled_pairs,
			                       [&](const auto& pair) { return pair.image == i; }) &&
			    !ValidatePackedUnormSampling(program, snapshot, i)) return false;
		}
		if (storage) {
			if ((!raw_sint_storage && !base.atomic &&
			     image.numeric_class == Prospero::TextureNumericClass::Sint) ||
			    image.numeric_class == Prospero::TextureNumericClass::Unsupported) {
				return SpecializationFail(
				    fmt::format("storage image descriptor {} uses unsupported format {}", i,
				                static_cast<uint32_t>(format)));
			}
			// Integer image atomics operate on the raw 32-bit texel, independently of
			// the descriptor numeric format. R32 float/sint loads and stores preserve
			// the same register bits through this uint view.
			if (raw_sint_storage || base.atomic) {
				image.numeric_class = Prospero::TextureNumericClass::Uint;
			}
		} else if (image.numeric_class == Prospero::TextureNumericClass::Unsupported ||
		           (base.depth_compare &&
		            image.numeric_class != Prospero::TextureNumericClass::Float)) {
			auto message = fmt::format(
			    "sampled image descriptor {} uses unsupported format {}: root={} source={} "
			    "first_use_pc={:#x} r128={} words={:08x},{:08x},{:08x},{:08x},"
			    "{:08x},{:08x},{:08x},{:08x}",
			    i, static_cast<uint32_t>(format), base_index, base.source, base.first_use_pc,
			    base.r128, descriptor.dwords[0], descriptor.dwords[1], descriptor.dwords[2],
			    descriptor.dwords[3], descriptor.dwords[4], descriptor.dwords[5],
			    descriptor.dwords[6], descriptor.dwords[7]);
			const auto* source = Source(program, base.source);
			if (i >= program.info.images.size() && source != nullptr &&
			    source->indirect_descriptor &&
			    IsBoundedDescriptorTable(program, *source->indirect_descriptor) &&
			    specialization.images[base_index].indirect_search_iterations == 0u &&
			    !source->indirect_descriptor->pointer_table) {
				const auto& indirect = *source->indirect_descriptor;
				const size_t mapping = specialization.images[base_index].indirect_mapping_offset;
				const auto& flat = snapshot.flattened_srt;
				if (mapping < flat.size() && flat[mapping] <= flat.size() - mapping - 1u) {
					uint32_t ordinal = 0;
					for (size_t child = program.info.images.size(); child <= i; ++child)
						ordinal += specialization.images[child].indirect_root == base_index;
					uint32_t matches = 0;
					uint64_t first_offset = 0;
					for (uint32_t probe = 0; probe < flat[mapping]; ++probe) {
						if (flat[mapping + 1u + probe] != ordinal) continue;
						if (matches++ == 0u) first_offset = uint64_t {probe} * 4u;
					}
					message += fmt::format(" table_source={} ordinal={} dense_matches={} "
					                       "first_candidate_offset={:#x}",
					                       indirect.table_source, ordinal, matches, first_offset);
				}
			}
			return SpecializationFail(message);
		}
	}
	for (uint32_t root_index = 0; root_index < specialization.images.size(); root_index++) {
		auto& root = specialization.images[root_index];
		if (root.indirect_root != root_index || root.bindless) {
			continue;
		}
		const auto key_count = root.indirect_mapping_offset < snapshot.flattened_srt.size()
		                           ? snapshot.flattened_srt[root.indirect_mapping_offset]
		                           : 0u;
		if (key_count == 0u ||
		    static_cast<size_t>(root.indirect_mapping_offset) + 1u +
		            static_cast<size_t>(key_count) * (root.indirect_search_iterations == 0u ? 1u : 2u) >
		        snapshot.flattened_srt.size()) {
			return SpecializationFail("indirect image specialization has an invalid key mapping");
		}
		uint32_t exemplar       = ImageResource::NoIndirectImage;
		uint32_t resource_count = 0;
		for (uint32_t resource = 0; resource < specialization.images.size(); resource++) {
			if (specialization.images[resource].indirect_root != root_index) {
				continue;
			}
			resource_count++;
			if (exemplar == ImageResource::NoIndirectImage &&
			    !NullImageDescriptor(snapshot.images[resource])) {
				exemplar = resource;
			}
		}
		if (resource_count < 2u || exemplar == ImageResource::NoIndirectImage) {
			return SpecializationFail("indirect image specialization has no typed candidate");
		}
		const auto& image_class = specialization.images[exemplar];
		const auto& base = program.info.images[root_index];
		const bool storage_swizzle = base.unfiltered_access_only &&
		    base.resource_class == ImageResourceClass::Storage;
		const auto is_2d = [](Decoder::ImageDimension dimension) {
			return dimension == Decoder::ImageDimension::Dim2D ||
			       dimension == Decoder::ImageDimension::Dim2DArray;
		};
		const auto is_1d_or_2d = [](Decoder::ImageDimension dimension) {
			return dimension == Decoder::ImageDimension::Dim1D ||
			       dimension == Decoder::ImageDimension::Dim2D;
		};
		for (uint32_t candidate = 0; candidate < specialization.images.size(); candidate++) {
			auto& image = specialization.images[candidate];
			if (image.indirect_root != root_index) {
				continue;
			}
			if (NullImageDescriptor(snapshot.images[candidate])) {
				image.numeric_class     = image_class.numeric_class;
				image.dimension         = image_class.dimension;
				image.mip_count         = image_class.mip_count;
				image.conversion_format = image_class.conversion_format;
				image.shader_swizzle    = image_class.shader_swizzle;
				image.cube              = image_class.cube;
			}
			const bool same_coordinates = image.dimension == image_class.dimension &&
			                              image.cube == image_class.cube;
			const bool unfiltered_coordinates = base.unfiltered_access_only &&
			    base.dimension == Decoder::ImageDimension::Dim2D &&
			    !image.cube && !image_class.cube && is_1d_or_2d(image.dimension) &&
			    is_1d_or_2d(image_class.dimension);
			if ((!same_coordinates && !(is_2d(image.dimension) && is_2d(image_class.dimension)) &&
			     !unfiltered_coordinates) ||
			    image.conversion_format != image_class.conversion_format ||
			    (!storage_swizzle && image.shader_swizzle != image_class.shader_swizzle)) {
				return SpecializationFail(
				    fmt::format("indirect image table at pc 0x{:08x} has incompatible candidates",
				                program.info.images[root_index].first_use_pc));
			}
		}
	}
	CompactImages(snapshot.images, [&](size_t index) { return !specialization.images[index].fmask; });
	return true;
}

bool BuildSamplerPlan(const ShaderInfo& base, SamplerPlan& plan) {
	plan.mapping.resize(base.samplers.size());
	plan.bindings.resize(base.samplers.size());
	std::vector<uint8_t> usage(base.samplers.size());
	for (const auto& pair: base.sampled_pairs) {
		if (pair.image >= base.images.size() || pair.sampler >= base.samplers.size()) {
			return false;
		}
		usage[pair.sampler] |= 1u << static_cast<uint32_t>(ClassifySampler(base.images[pair.image]));
		for (const auto candidate: base.images[pair.image].indirect_resources) {
			if (candidate >= base.images.size()) return false;
			usage[pair.sampler] |= 1u << static_cast<uint32_t>(ClassifySampler(base.images[candidate]));
		}
		for (const auto candidate: base.samplers[pair.sampler].indirect_resources) {
			if (candidate >= base.samplers.size()) return false;
			usage[candidate] |= usage[pair.sampler];
		}
	}
	for (uint32_t index = 0; index < base.samplers.size(); index++) {
		auto& mapping = plan.mapping[index];
		mapping.fill(UINT32_MAX);
		const auto classes = usage[index] == 0u ? 1u : usage[index];
		bool       first   = true;
		for (uint32_t type = 0; type < mapping.size(); type++) {
			if ((classes & (1u << type)) == 0u) continue;
			// A bindless sampler is one slot of the host's table for all of its uses: its other
			// classes share the first one's binding rather than adding copies.
			if (!first && base.samplers[index].bindless) {
				mapping[type] = index;
				continue;
			}
			const auto target = first ? index : static_cast<uint32_t>(plan.bindings.size());
			if (!first) plan.bindings.emplace_back();
			mapping[type]         = target;
			plan.bindings[target] = {index, static_cast<SamplerClass>(type)};
			first                = false;
		}
	}
	return true;
}

template <typename Predicate>
static std::vector<ResourceBlock> ResourceControlFlow(const Program& program, const Predicate& predicate) {
	// The renderer checks captured scalar reads against final buffer and image write ranges.
	if (program.has_address_writes) {
		return {};
	}
	std::unordered_map<const Block*, uint32_t> indices;
	for (uint32_t i = 0; i < program.blocks.size(); i++) {
		if (!indices.emplace(program.blocks[i], i).second) {
			return {};
		}
	}
	std::vector<ResourceBlock> blocks(program.blocks.size());
	for (uint32_t i = 0; i < blocks.size(); i++) {
		auto&                 block      = blocks[i];
		const auto&             info       = *program.blocks[i];
		const auto&           terminator = info.terminator;
		const std::array        targets {terminator.true_block, terminator.false_block};
		std::span<Block* const> successors;
		switch (terminator.kind) {
			case CFG::TerminatorKind::Branch: successors = std::span(targets).first(1); break;
			case CFG::TerminatorKind::ConditionalBranch:
				successors      = targets;
				block.condition = info.condition;
				break;
			case CFG::TerminatorKind::IndirectBranch: successors = info.ImmSuccessors(); break;
			case CFG::TerminatorKind::Return: break;
			default: return {};
		}
		for (const auto successor: successors) {
			const auto found = indices.find(successor);
			if (found == indices.end()) {
				return {};
			}
			block.successors.push_back(found->second);
		}
		for (const auto& inst: *program.blocks[i]) {
			const auto op     = inst.GetOpcode();
			if (op == ValueOpcode::ReadConst) {
				block.srt_reads.push_back(inst.Arg(1).Resolve().U32());
				continue;
			}
			const auto buffer = BufferAccessOf(op);
			const auto image  = ImageOpcodeInfoOf(op);
			if (buffer == BufferAccess::None && image.access == ImageAccess::None) {
				continue;
			}
			const auto& memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			if (memory.planning_only || memory.kind == ResourceKind::IndirectBuffer) {
				continue;
			}
			if (buffer != BufferAccess::None) {
				block.sources.push_back(program.info.buffers.at(memory.resource).source);
			} else {
				block.sources.push_back(program.info.images.at(memory.resource).source);
				if (image.needs_sampler) {
					block.sources.push_back(program.info.samplers.at(memory.sampler).source);
				}
			}
		}
		std::ranges::sort(block.sources);
		block.sources.erase(std::unique(block.sources.begin(), block.sources.end()),
		                    block.sources.end());
	}
	for (auto& block: blocks) block.condition = predicate(block.condition);
	if (std::ranges::none_of(
	        blocks, [](const ResourceBlock& block) { return !block.condition.IsEmpty(); })) {
		return {};
	}
	return blocks;
}

// ResolveComputeBufferFill proves complete workgroups and matching requested extents, so every
// dispatch-bound predicate accepted here is true for every invocation covered by the fill.
static bool IsFullDispatchPredicate(Value value, uint32_t depth = 0) {
	value = value.Resolve();
	if (value == Value(true)) return true;
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 32) return false;
	if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
		return IsFullDispatchPredicate(inst->Arg(0), depth + 1) &&
		       IsFullDispatchPredicate(inst->Arg(1), depth + 1);
	}
	if (inst->GetOpcode() != ValueOpcode::ULessThan32) return false;
	const auto* global = inst->Arg(0).Resolve().TryInstruction();
	const auto* extent = inst->Arg(1).Resolve().TryInstruction();
	if (global == nullptr || global->GetOpcode() != ValueOpcode::GetBuiltin ||
	    global->Arg(0).Resolve() != Value(static_cast<uint32_t>(StageInputKind::GlobalInvocationId)) ||
	    extent == nullptr || extent->GetOpcode() != ValueOpcode::GetDispatchThreadExtent)
		return false;
	const auto axis = global->Arg(1).Resolve();
	return axis.IsImmediate() && axis.U32() < 3 && extent->Arg(0).Resolve() == axis;
}

static std::optional<std::array<uint64_t, 3>> FillIndex(Value value, uint32_t axis, Value guard) {
	return AffineIndex<3>(value, [&](Value at) { return ResolveActiveU32(at, guard); },
	    [&](Value at) -> std::optional<size_t> {
		    const auto* inst = at.TryInstruction();
		    if (inst == nullptr || inst->GetOpcode() != ValueOpcode::GetBuiltin ||
		        inst->Arg(1) != Value(axis)) return {};
		    if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId))) return 1u;
		    if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId))) return 2u;
		    return {};
	    });
}

static UniformFillPlan AnalyzeUniformFill(const Program& program) {
	if (program.stage != ShaderType::Compute || program.blocks.empty() || program.info.uses_dma ||
	    !program.info.samplers.empty()) {
		return {};
	}
	std::unordered_set<const Block*> visited;
	const auto*                      block = program.blocks.front();
	const Inst*                      store = nullptr;
	for (;;) {
		if (block == nullptr || !visited.insert(block).second) return {};
		for (const auto& inst: *block) {
			if (AddressOpcodeInfoOf(inst.GetOpcode()).access != AddressAccess::None) return {};
			if (!inst.MayHaveSideEffects()) continue;
			if (store != nullptr || (BufferAccessOf(inst.GetOpcode()) != BufferAccess::Write &&
			                         inst.GetOpcode() != ValueOpcode::ImageWrite))
				return {};
			store = &inst;
		}
		const auto& term = block->terminator;
		if (term.kind == CFG::TerminatorKind::Return) break;
		if (term.kind != CFG::TerminatorKind::Branch) return {};
		block = term.true_block;
	}
	if (store == nullptr || visited.size() != program.blocks.size()) return {};
	for (const auto& buffer: program.info.buffers) {
		if (buffer.read && (!buffer.scalar || buffer.written)) return {};
	}
	const auto& memory = program.memory_info.at(store->Flags<MemoryFlags>().index);
	UniformFillPlan result;
	result.fill.resource = memory.resource;
	Value data;
	Value guard(true);
	if (store->GetOpcode() == ValueOpcode::ImageWrite) {
		if (program.info.images.size() != 1 || memory.dmask != 1 || memory.data_bits != 32 ||
		    memory.image_has_mip || memory.image_sample_flags != 0 || memory.image_r128 ||
		    memory.image_dimension != Decoder::ImageDimension::Dim2DArray ||
		    !IsFullDispatchPredicate(store->Arg(3))) return {};
		const auto& image = program.info.images[memory.resource];
		if (image.read || image.atomic || image.mip_mode != ImageMipMode::None) return {};
		const auto* address = store->Arg(1).ResolveInstruction();
		if (address == nullptr || address->GetOpcode() != ValueOpcode::MakeImageAddress) return {};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const auto index = FillIndex(address->Arg(axis), axis, guard);
			if (!index || (*index)[0] != 0) return {};
			if (axis < 2) {
				if ((*index)[1] != 1 || (*index)[2] == 0) return {};
			} else if ((*index)[1] != 0 || (*index)[2] != 1) {
				return {};
			}
			result.fill.group_stride[axis] = static_cast<uint32_t>((*index)[2]);
		}
		const auto* values = store->Arg(2).ResolveInstruction();
		if (values == nullptr || values->GetOpcode() != ValueOpcode::CompositeConstructU32x4)
			return {};
		result.fill.kind  = UniformFillKind::Image;
		result.fill.words = 1;
		data = values->Arg(0);
	} else {
		if (!program.info.images.empty()) return {};
		const auto           op = store->GetOpcode();
		constexpr std::array stores {ValueOpcode::StoreBufferU32, ValueOpcode::StoreBufferU32x2,
		                             ValueOpcode::StoreBufferU32x3, ValueOpcode::StoreBufferU32x4};
		const auto           store_op = std::ranges::find(stores, op);
		if (store_op == stores.end() || store->Arg(2).Resolve() != Value(0u) ||
		    store->Arg(3).Resolve() != Value(0u))
			return {};
		guard = store->Arg(5).Resolve();
		if (!IsFullDispatchPredicate(guard)) return {};
		if (!memory.formatted || memory.typed || !memory.idxen || memory.offen || memory.offset != 0 ||
		    memory.data_bits != 32 ||
		    memory.data_dwords != static_cast<uint32_t>(store_op - stores.begin() + 1))
			return {};
		const auto address = FillIndex(store->Arg(1), 0, guard);
		if (!address || (*address)[0] != 0 || (*address)[1] != 1 || (*address)[2] == 0) return {};
		result.fill.kind = UniformFillKind::Buffer;
		result.fill.group_stride[0] = static_cast<uint32_t>((*address)[2]);
		result.fill.words = memory.data_dwords;
		data = store->Arg(4);
	}
	data = ResolveActiveU32(data, guard);
	const auto*          vector = data.TryInstruction();
	constexpr std::array composites {ValueOpcode::CompositeConstructU32x2,
	                                 ValueOpcode::CompositeConstructU32x3,
	                                 ValueOpcode::CompositeConstructU32x4};
	if (result.fill.words > 1 &&
	    (vector == nullptr || vector->GetOpcode() != composites[result.fill.words - 2]))
		return {};
	for (uint32_t i = 0; i < result.fill.words; ++i) {
		const auto word = ResolveActiveU32(result.fill.words == 1 ? data : vector->Arg(i), guard);
		if (word.GetType() != Type::U32 ||
		    !ValidateRuntimeValue(program, word, RuntimeValueType::Integer))
			return {};
		result.values[i] = word;
	}
	return result;
}

namespace {

// Integer operations whose result interval follows from their operands' intervals (see
// WriteInterval).
bool IsIntervalOperation(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::IAdd32:
		case ValueOpcode::IMul32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::SelectU32: return true;
		default: return false;
	}
}

bool IsBoundedInvocationId(const ResourcePlan& program, const Inst& inst) {
	if (inst.GetOpcode() != ValueOpcode::GetBuiltin || program.stage != ShaderType::Compute ||
	    !inst.Arg(0).IsImmediate() || !inst.Arg(1).IsImmediate() || inst.Arg(1).U32() >= 3u) {
		return false;
	}
	switch (static_cast<StageInputKind>(inst.Arg(0).U32())) {
		case StageInputKind::WorkgroupId:
		case StageInputKind::LocalInvocationId:
		case StageInputKind::LocalInvocationIndex:
		case StageInputKind::GlobalInvocationId: return true;
		default: return false;
	}
}

// A store address operand the host can bound per dispatch: values it evaluates, compute invocation
// IDs, and interval operations on them. The select on the store's own predicate only matters for
// the stored lanes.
bool BoundableWriteOperand(const Program& program, Value value, Value predicate, uint32_t depth) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		return value.GetType() == Type::U32;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 32u) {
		return false;
	}
	if (IsBoundedInvocationId(program, *inst)) {
		return true;
	}
	if (IsIntervalOperation(inst->GetOpcode())) {
		if (inst->GetOpcode() == ValueOpcode::SelectU32 && inst->Arg(0).Resolve() == predicate) {
			return BoundableWriteOperand(program, inst->Arg(1), predicate, depth + 1u);
		}
		const auto first = inst->GetOpcode() == ValueOpcode::SelectU32 ? 1u : 0u;
		for (size_t i = first; i < inst->NumArgs(); ++i) {
			if (!BoundableWriteOperand(program, inst->Arg(i), predicate, depth + 1u)) {
				return false;
			}
		}
		return true;
	}
	return value.GetType() == Type::U32 &&
	       ValidateRuntimeValue(program, value, RuntimeValueType::Integer);
}

} // namespace

namespace {
bool GpuSizedWrittenEnabled(uint64_t shader_hash);
bool GpuSizedDynamicEnabled();

// KYTY_GPU_SIZED_ATOMIC_SHADERS=<hash>[,<hash>...] (hex): atomically updated buffers of the
// listed shaders also bind a conservative record count when the GPU wrote theirs (with
// KYTY_GPU_SIZED_WRITTEN on and the shader in its list): only for shaders whose atomics index
// within the real count (the shader's own data), never ones that rely on out-of-range drops.
bool GpuSizedAtomicEnabled(uint64_t shader_hash) {
	// KYTY_GPU_SIZED_DYNAMIC: the shader bounds its atomics by the exact count too, so an atomic
	// past it is dropped as on the console (CS 48ba2226632e8209 waited for its atomic buffer's
	// count every frame on the title screen, ~20 % of the GPU thread). KYTY_GPU_SIZED_ATOMIC_SHADERS
	// still limits it to the listed shaders.
	static const bool listed = std::getenv("KYTY_GPU_SIZED_ATOMIC_SHADERS") != nullptr;
	if (GpuSizedDynamicEnabled() && !listed) {
		return GpuSizedWrittenEnabled(shader_hash);
	}
	static const std::vector<uint64_t> shaders = [] {
		std::vector<uint64_t> result;
		const char*           value = std::getenv("KYTY_GPU_SIZED_ATOMIC_SHADERS");
		if (value == nullptr) return result;
		const std::string list = value;
		size_t            at   = 0;
		while (at < list.size()) {
			const auto end = std::min(list.find_first_of(",; ", at), list.size());
			if (end > at) result.push_back(std::strtoull(list.substr(at, end - at).c_str(), nullptr, 16));
			at = end + 1;
		}
		return result;
	}();
	return !shaders.empty() && GpuSizedWrittenEnabled(shader_hash) &&
	       std::find(shaders.begin(), shaders.end(), shader_hash) != shaders.end();
}

// KYTY_SRT_EXTENTS_NO_WAIT_SHADERS=<hash>[,<hash>...] (hex): KYTY_SRT_EXTENTS_NO_WAIT for the
// listed shaders only. Their write extents are evaluated without waiting for the GPU (the whole
// buffer when they cannot be), so the slots the extents read can be left to the GPU.
bool ExtentsNoWaitShader(uint64_t shader_hash) {
	static const std::vector<uint64_t> shaders = [] {
		std::vector<uint64_t> result;
		const char*           value = std::getenv("KYTY_SRT_EXTENTS_NO_WAIT_SHADERS");
		if (value == nullptr) return result;
		const std::string list = value;
		size_t            at   = 0;
		while (at < list.size()) {
			const auto end  = std::min(list.find_first_of(",; ", at), list.size());
			if (end > at) result.push_back(std::strtoull(list.substr(at, end - at).c_str(), nullptr, 16));
			at = end + 1;
		}
		return result;
	}();
	return !shaders.empty() && std::find(shaders.begin(), shaders.end(), shader_hash) != shaders.end();
}
} // namespace

// Flat slots the host never needs, so their bytes can be left to the GPU when it wrote them
// (SrtGpuFill): the slot's value is a raw scalar read whose address comes from user data and
// immediates alone, and no value the host evaluates refers to the slot (descriptor dwords and
// keys, branch conditions, uniform fills, buffer write bounds, other reads' addresses). The host
// then only uploads the flat buffer; the shader reads the slot.
static std::vector<uint8_t> GpuFillSlots(const ResourcePlan& plan) {
	const auto           count = plan.srt_reads.size();
	std::vector<uint8_t> host(count, 0u);
	std::vector<Value>   pending;
	std::unordered_set<const Inst*> visited;
	// Why a slot stays on the host (KYTY_SRT_FILL_REASON="hash,..." prints it): D descriptor,
	// C condition, U uniform fill, W write extent, A another read's address, S clean slot,
	// K not a raw scalar read, M address depends on memory.
	uint8_t    tag  = 'D';
	// Sources only read-only, non-indirect buffers use: their record count (dword 2) is not a host
	// need (GpuSizedBuffers binds a conservative count when the GPU wrote it).
	std::vector<uint8_t> sized_sources(plan.descriptor_sources.size(), 0u);
	// For KYTY_SRT_FILL_REASON: why a source's record count stays on the host (w written,
	// a atomic, i image, s sampler).
	std::vector<std::string> source_why(plan.descriptor_sources.size());
	{
		std::vector<uint8_t> other(plan.descriptor_sources.size(), 0u);
		for (const auto& buffer: plan.info.buffers) {
			if (buffer.source >= sized_sources.size()) continue;
			if ((!buffer.written || GpuSizedWrittenEnabled(plan.shader_hash)) &&
			    (!buffer.atomic || GpuSizedAtomicEnabled(plan.shader_hash))) {
				sized_sources[buffer.source] = 1u;
			} else {
				other[buffer.source] = 1u;
				source_why[buffer.source] += buffer.atomic ? 'a' : 'w';
			}
		}
		for (const auto& image: plan.info.images) {
			if (image.source < other.size()) {
				other[image.source] = 1u;
				source_why[image.source] += 'i';
			}
		}
		for (const auto& sampler: plan.info.samplers) {
			if (sampler.source < other.size()) {
				other[sampler.source] = 1u;
				source_why[sampler.source] += 's';
			}
		}
		for (size_t i = 0; i < sized_sources.size(); ++i) {
			if (other[i] != 0u || plan.descriptor_sources[i].dword_count != 4u ||
			    plan.descriptor_sources[i].indirect_descriptor.has_value()) {
				sized_sources[i] = 0u;
			}
		}
	}
	std::vector<int64_t> host_desc(count, -1);
	int64_t              current_desc = -1;
	// The plan holds the raw reads themselves (no flat aliases): which slot each one fills.
	std::unordered_map<const Inst*, uint32_t> slot_of;
	for (uint32_t slot = 0; slot < count; slot++) {
		if (const auto* read = plan.srt_reads[slot].value.ResolveInstruction(); read != nullptr) {
			slot_of.emplace(read, slot);
		}
	}
	const auto mark = [&](Value root) {
		pending.push_back(root);
		while (!pending.empty()) {
			const auto  value = pending.back().Resolve();
			pending.pop_back();
			const auto* inst  = value.TryInstruction();
			if (inst == nullptr || !visited.insert(inst).second) {
				continue;
			}
			if (const auto found = slot_of.find(inst); found != slot_of.end()) {
				const auto slot = found->second;
				if (host[slot] == 0u) {
					host[slot]      = tag;
					host_desc[slot] = tag == 'D' ? current_desc : -1;
				}
			}
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				pending.push_back(inst->Arg(arg));
			}
		}
	};
	for (size_t source_index = 0; source_index < plan.descriptor_sources.size(); ++source_index) {
		const auto& source = plan.descriptor_sources[source_index];
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			if (dword == 2u && sized_sources[source_index] != 0u) {
				continue;
			}
			current_desc = static_cast<int64_t>(source_index * 16u + dword);
			mark(source.dwords[dword]);
		}
		current_desc = static_cast<int64_t>(source_index * 16u + 15u);
		if (source.indirect_descriptor.has_value()) {
			mark(source.indirect_descriptor->key_count);
			mark(source.indirect_descriptor->selector_first);
			mark(source.indirect_descriptor->selector_mask);
		}
	}
	// Branch conditions only choose which descriptor sources are active. With SrtConditionsNoWait
	// a condition the host cannot read without waiting for the GPU is "not known" (both successors
	// walked), so the slots it reads need not be on the host: the GPU fills them for the shader.
	tag = 'C';
	if (!SrtConditionsNoWait()) {
		for (const auto& block: plan.control_flow) {
			mark(block.condition);
		}
	}
	tag = 'U';
	for (uint32_t i = 0; i < plan.uniform_fill.fill.words; ++i) {
		mark(plan.uniform_fill.values[i]);
	}
	tag = 'W';
	if (std::getenv("KYTY_SRT_EXTENTS_NO_WAIT") == nullptr && !ExtentsNoWaitShader(plan.shader_hash)) {
		for (const auto& write: plan.buffer_writes) {
			// A GPU-sized buffer's extents are evaluated without waiting (t_gpu_sized_snapshot);
			// they need no host slots.
			if (write.buffer < plan.info.buffers.size() &&
			    plan.info.buffers[write.buffer].source < sized_sources.size() &&
			    sized_sources[plan.info.buffers[write.buffer].source] != 0u) {
				continue;
			}
			mark(write.index);
			mark(write.offset);
			mark(write.soffset);
			mark(write.predicate);
		}
	}
	// Another read's address operands (the read itself is not a host use of its own slot).
	tag = 'A';
	for (const auto& read: plan.srt_reads) {
		if (const auto* inst = read.value.Resolve().TryInstruction(); inst != nullptr) {
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				mark(inst->Arg(arg));
			}
		}
	}
	// The read's own address must not depend on memory: one host read per slot, the slot's own.
	const auto memory_free = [&](const Inst& read) {
		std::vector<Value>              stack;
		std::unordered_set<const Inst*> seen;
		for (size_t arg = 0; arg < read.NumArgs(); arg++) {
			stack.push_back(read.Arg(arg));
		}
		while (!stack.empty()) {
			const auto* inst = stack.back().Resolve().TryInstruction();
			stack.pop_back();
			if (inst == nullptr || !seen.insert(inst).second) {
				continue;
			}
			switch (inst->GetOpcode()) {
				case ValueOpcode::ReadConst:
				case ValueOpcode::LoadAddressU32:
				case ValueOpcode::ReadConstBuffer:
				case ValueOpcode::LoadBufferU32:
				case ValueOpcode::Phi: return false;
				default: break;
			}
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				stack.push_back(inst->Arg(arg));
			}
		}
		return true;
	};
	std::vector<uint8_t> fill(count, 0u);
	for (size_t slot = 0; slot < count; slot++) {
		const auto* inst = plan.srt_reads[slot].value.Resolve().TryInstruction();
		if (inst == nullptr) {
			host[slot] = host[slot] != 0u ? host[slot] : 'K';
			continue;
		}
		if (host[slot] != 0u) {
			continue;
		}
		if (inst->Flags<SrtReadFlags>().clean != 0u) {
			host[slot] = 'S';
			continue;
		}
		const auto op    = inst->GetOpcode();
		const auto index = inst->Flags<SrtReadFlags>().index;
		if ((op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) ||
		    index >= plan.memory_info.size()) {
			host[slot] = 'K';
			continue;
		}
		const auto kind = plan.memory_info[index].kind;
		if (!((op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
		      (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer))) {
			host[slot] = 'K';
			continue;
		}
		// 1: the slot's read is its only read. 2: its address comes from other reads; those are done
		// on the host and only the slot's own read is left to the GPU (SrtWalker::m_fill_inst).
		fill[slot] = memory_free(*inst) ? 1u : 2u;
	}
	if (const char* list = std::getenv("KYTY_SRT_FILL_REASON");
	    list != nullptr &&
	    std::string_view(list).find(fmt::format("{:016x}", plan.shader_hash)) != std::string_view::npos) {
		std::string text;
		for (size_t slot = 0; slot < count; slot++) {
			const auto* inst = plan.srt_reads[slot].value.Resolve().TryInstruction();
			text += fmt::format(" {}:{}({})", slot, fill[slot] != 0u ? 'G' : static_cast<char>(host[slot]),
			                    inst != nullptr ? ValueOpcodeName(inst->GetOpcode()) : "-");
			if (fill[slot] == 0u && host[slot] == 'D' && host_desc[slot] >= 0) {
				const auto  src    = static_cast<size_t>(host_desc[slot] / 16);
				const auto  dw     = host_desc[slot] % 16;
				const auto& source = plan.descriptor_sources[src];
				text += fmt::format("[src{} {} n={} ind={} why={}]", src,
				                    dw == 15 ? std::string("key") : fmt::format("d{}", dw),
				                    source.dword_count, source.indirect_descriptor.has_value() ? 1 : 0,
				                    source_why[src]);
			}
		}
		std::fprintf(stderr, "[srt-fill] %016llx slots=%zu%s\n",
		             static_cast<unsigned long long>(plan.shader_hash), count, text.c_str());
	}
	return fill;
}

ResourcePlan ExtractResourcePlan(const Program& program) {
	ResourcePlan plan;
	plan.stage                      = program.stage;
	plan.shader_hash                = program.shader_hash;
	plan.user_data_base             = program.user_data_base;
	plan.user_data_count            = program.user_data_count;
	plan.info                       = program.info;
	plan.memory_info                = program.memory_info;
	plan.source_reads               = program.source_reads;
	plan.srt_plan_complete          = program.srt_plan_complete;
	plan.resource_tracking_complete = program.resource_tracking_complete;
	plan.bindless_images            = program.bindless_images;

	std::unordered_map<const Inst*, Inst*> cloned;
	std::function<Value(Value)>            Clone = [&](Value value) -> Value {
		value              = value.Resolve();
		const auto* source = value.TryInstruction();
		if (source == nullptr) {
			return value;
		}
		if (source->GetOpcode() == ValueOpcode::ReadConst) {
			const auto slot = source->Arg(1).Resolve();
			EXIT_IF(!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			        slot.U32() >= program.srt_reads.size());
			return Clone(program.srt_reads[slot.U32()].value);
		}
		if (source->GetOpcode() == ValueOpcode::ReadFirstLane &&
		    ValidateRuntimeValue(program, source->Arg(0))) {
			// Uniform values need no new EXEC context; preserve their shared evaluation memo.
			return Clone(source->Arg(0));
		}
		if (source->GetOpcode() == ValueOpcode::Phi) {
			const auto invariant = ResolveInvariantPhi(program, value);
			if (!invariant.IsEmpty() && invariant != value) {
				return Clone(invariant);
			}
		}
		if (const auto found = cloned.find(source); found != cloned.end()) {
			return Value(found->second);
		}
		if (source->GetOpcode() == ValueOpcode::LoadBufferU32) {
			plan.requires_specialization_memory = true;
			plan.capture_specialization_reads   = true;
		}
		auto& target =
		    plan.value_storage.emplace_back(source->GetOpcode(), source->Flags<uint64_t>());
		if (source->GetOpcode() == ValueOpcode::LoadAddressU32 ||
		    source->GetOpcode() == ValueOpcode::ReadConstBuffer ||
		    source->GetOpcode() == ValueOpcode::LoadBufferU32) {
			target.SetFlags(SrtReadFlags {.index = source->Flags<MemoryFlags>().index});
		}
		cloned.emplace(source, &target);
		if (source->GetOpcode() == ValueOpcode::Phi) {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.AddPhiOperand(nullptr, Clone(source->Arg(index)));
			}
		} else {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.SetArg(index, Clone(source->Arg(index)));
			}
		}
		return Value(&target);
	};

	bool capture_indirect_reads = false;
	plan.descriptor_sources.reserve(program.descriptor_sources.size());
	for (const auto& source: program.descriptor_sources) {
		auto& target          = plan.descriptor_sources.emplace_back();
		target.dword_count    = source.dword_count;
		target.indirect_descriptor = source.indirect_descriptor;
		target.bindless_sampler = source.bindless_sampler;
		if (target.indirect_descriptor.has_value()) {
			plan.requires_specialization_memory = true;
			capture_indirect_reads = true;
			target.indirect_descriptor->key_count = Clone(target.indirect_descriptor->key_count);
			target.indirect_descriptor->selector_first =
			    Clone(target.indirect_descriptor->selector_first);
			target.indirect_descriptor->selector_mask =
			    Clone(target.indirect_descriptor->selector_mask);
		}
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			target.dwords[dword] = Clone(source.dwords[dword]);
		}
	}
	// Destination tables share keys when their complete selector domains agree.
	// Resolve those domains lazily at dispatch time, without repeated guest reads.
	plan.selector_groups.resize(plan.descriptor_sources.size(), UINT32_MAX);
	for (uint32_t source = 0; source < plan.descriptor_sources.size(); ++source) {
		const auto& descriptor = program.descriptor_sources[source].indirect_descriptor;
		if (!descriptor || !descriptor->selector) continue;
		const auto& a = *descriptor;
		uint32_t group = 0;
		for (; group < plan.selector_keys.size(); ++group) {
			const auto& b = *program.descriptor_sources[plan.selector_keys[group].source].indirect_descriptor;
			if (EquivalentSelectorDomain(program, a, b)) break;
		}
		if (group == plan.selector_keys.size()) plan.selector_keys.push_back({.source = source});
		plan.selector_groups[source] = group;
	}
	plan.srt_reads.reserve(program.srt_reads.size());
	for (const auto& read: program.srt_reads) {
		plan.srt_reads.push_back({Clone(read.value), read.flat_offset});
	}
	// Retain integer control expressions with opaque varying leaves. Their identities
	// matter when an EXEC factor also guards a selected value; never clone float work.
	bool capture_control_reads = false;
	bool uniform_control = false;
	const auto control_value_count = plan.value_storage.size();
	struct ControlValue {
		Value value;
		bool complete = false;
	};
	std::unordered_map<const Inst*, ControlValue> predicates;
	std::function<Value(Value)> ClonePredicate = [&](Value value) -> Value {
		value = value.Resolve();
		if (value.IsEmpty()) return {};
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			uniform_control |= value.GetType() == Type::U1;
			return ValidateRuntimeValue(program, value, RuntimeValueType::Integer) ? value : Value {};
		}
		const auto [found, inserted] = predicates.try_emplace(inst);
		auto& entry = found->second;
		auto& result = entry.value;
		const auto opaque = [&]() -> Value {
			if (value.GetType() != Type::U1 && value.GetType() != Type::U32) return {};
			return result = Value(&plan.value_storage.emplace_back(
			    value.GetType() == Type::U1 ? ValueOpcode::UndefU1 : ValueOpcode::UndefU32));
		};
		// A recursive projection remains opaque, including unstructured loop Phis.
		if (!inserted) return entry.complete || !result.IsEmpty() ? result : opaque();
		const auto finish = [&](Value projected) {
			entry.complete = true;
			return result = projected;
		};
		if (ValidateRuntimeValue(program, value, RuntimeValueType::Integer)) {
			uniform_control = true;
			capture_control_reads |=
			    !ValidateRuntimeValue(program, value, RuntimeValueType::ImmutableInteger);
			return finish(Clone(value));
		}
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ConditionRef) return finish(ClonePredicate(inst->Arg(0)));
		if (op == ValueOpcode::Phi) {
			if (inst->Parent() == nullptr || inst->Parent()->terminator.loop_header ||
			    inst->NumPhiBlocks() != inst->NumArgs() || inst->NumArgs() == 0u ||
			    (value.GetType() != Type::U1 && value.GetType() != Type::U32)) return finish(opaque());
			std::vector<uint32_t> predecessors;
			for (size_t index = 0; index < inst->NumArgs(); ++index) {
				const auto predecessor = std::ranges::find(program.blocks, inst->PhiBlock(index));
				if (predecessor == program.blocks.end()) return finish(opaque());
				predecessors.push_back(static_cast<uint32_t>(predecessor - program.blocks.begin()));
			}
			auto& node = plan.value_storage.emplace_back(op, inst->Flags<uint64_t>());
			plan.control_phi_predecessors.emplace(&node, std::move(predecessors));
			for (size_t index = 0; index < inst->NumArgs(); ++index)
				node.AddPhiOperand(nullptr, ClonePredicate(inst->Arg(index)));
			return finish(Value(&node));
		}
		if (op != ValueOpcode::LogicalAnd && op != ValueOpcode::LogicalOr &&
		    op != ValueOpcode::LogicalNot && op != ValueOpcode::IEqual32 &&
		    op != ValueOpcode::INotEqual32 && op != ValueOpcode::ULessThan32 &&
		    op != ValueOpcode::ULessThanEqual32 &&
		    op != ValueOpcode::SelectU1 && op != ValueOpcode::SelectU32) return finish(opaque());
		auto& node = plan.value_storage.emplace_back(op);
		for (size_t index = 0; index < inst->NumArgs(); ++index)
			node.SetArg(index, ClonePredicate(inst->Arg(index)));
		return finish(Value(&node));
	};
	plan.control_flow = ResourceControlFlow(program, ClonePredicate);
	if (!uniform_control) {
		plan.control_flow.clear();
		plan.control_phi_predecessors.clear();
		auto first = std::next(plan.value_storage.begin(), control_value_count);
		for (auto it = first; it != plan.value_storage.end(); ++it) it->Invalidate();
		plan.value_storage.erase(first, plan.value_storage.end());
	}
	plan.uniform_fill = program.uniform_fill.fill.kind != UniformFillKind::None
	    ? program.uniform_fill : AnalyzeUniformFill(program);
	for (auto& value: plan.uniform_fill.values) value = Clone(value);
	plan.uniform_fill.pattern_size = Clone(plan.uniform_fill.pattern_size);
	plan.uniform_fill.element_count = Clone(plan.uniform_fill.element_count);
	// A bindless sampler's handle dwords are read with the strict reader: its reads are marked
	// clean below, which needs the captured-reads mode.
	bool bindless_sampler_reads = false;
	for (const auto& sampler: plan.info.samplers) {
		const auto* source = Source(plan, sampler.source);
		if (source != nullptr && source->bindless_sampler.has_value()) {
			bindless_sampler_reads = true;
		}
	}
	if (bindless_sampler_reads) plan.requires_specialization_memory = true;
	plan.capture_specialization_reads |= capture_indirect_reads || capture_control_reads ||
	                                     !plan.source_reads.empty() || bindless_sampler_reads;
	if (plan.capture_specialization_reads) {
		// Every former flat alias now targets its retained raw read. Mark those reads
		// on the normalized graph, so discarded Phi/EXEC dependencies stay discarded.
		std::unordered_set<const Inst*> planned_reads;
		for (const auto& read: plan.srt_reads) {
			auto* inst = read.value.ResolveInstruction();
			EXIT_IF(inst == nullptr || (inst->GetOpcode() != ValueOpcode::LoadAddressU32 &&
			                           inst->GetOpcode() != ValueOpcode::ReadConstBuffer));
			planned_reads.insert(inst);
		}
		for (const auto& source: plan.descriptor_sources) {
			if (!source.indirect_descriptor.has_value()) continue;
			const auto& indirect = *source.indirect_descriptor;
			if (indirect.selector) {
				MarkCleanReads(planned_reads, Source(plan, indirect.selector->source));
				if (indirect.selector->rows)
					MarkCleanReads(planned_reads, Source(plan, indirect.selector->rows->source));
			}
			MarkCleanReads(planned_reads, nullptr, indirect.selector_mask);
			MarkCleanReads(planned_reads, nullptr, indirect.selector_first);
			MarkCleanReads(planned_reads, nullptr, indirect.key_count);
			if (indirect.sources.empty()) {
				MarkCleanReads(planned_reads, Source(plan, indirect.table_source));
			} else {
				for (const auto candidate: indirect.sources) {
					MarkCleanReads(planned_reads, Source(plan, candidate));
				}
			}
		}
		for (const auto& sampler: plan.info.samplers) {
			const auto* source = Source(plan, sampler.source);
			if (source != nullptr && source->bindless_sampler.has_value()) {
				MarkCleanReads(planned_reads, source);
			}
		}
		// Clean writable addresses prove that shader writes cannot overlap captured reads.
		for (const auto& buffer: plan.info.buffers) {
			if (!buffer.written) continue;
			const auto* source = Source(plan, buffer.source);
			// A GPU-sized written buffer (KYTY_GPU_SIZED_WRITTEN_SHADERS) binds a conservative record
			// count, so its dword 2 is no host need here either: the GPU fills the slots under it.
			if (source != nullptr && source->dword_count == 4u &&
			    !source->indirect_descriptor.has_value() && GpuSizedWrittenEnabled(plan.shader_hash) &&
			    (!buffer.atomic || GpuSizedAtomicEnabled(plan.shader_hash))) {
				for (const uint32_t dword: {0u, 1u, 3u}) {
					MarkCleanReads(planned_reads, nullptr, source->dwords[dword]);
				}
				continue;
			}
			MarkCleanReads(planned_reads, source);
		}
		for (const auto& image: plan.info.images) {
			if (image.written) MarkCleanReads(planned_reads, Source(plan, image.source));
		}
	}
	if (capture_indirect_reads || !plan.source_reads.empty())
		plan.resource_tracking_complete &= !program.has_address_writes;
	CompileSrtWalker(plan);
	// After the decisions above: the clones below must not count as the shader's own reads.
	// Store addresses bound the bytes a dispatch writes: a title's heap-wide V# otherwise marks
	// the whole heap GPU-written, and every CPU access to it drains the GPU.
	const auto buffer_count = program.info.buffers.size();
	plan.buffer_writes_bounded.assign(buffer_count, 1u);
	std::vector<uint32_t> listed_writes(buffer_count, 0u);
	for (uint32_t i = 0; i < buffer_count; ++i) {
		if (program.info.buffers[i].image_alias != BufferResource::NoImageAlias) {
			plan.buffer_writes_bounded[i] = 0u;
		}
	}
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto access = BufferAccessOf(inst.GetOpcode());
			if (access != BufferAccess::Write && access != BufferAccess::Atomic) {
				continue;
			}
			const auto flags = inst.Flags<MemoryFlags>();
			if (flags.index >= program.memory_info.size()) {
				plan.buffer_writes_bounded.assign(buffer_count, 0u);
				continue;
			}
			const auto& memory = program.memory_info[flags.index];
			if (memory.kind != ResourceKind::Buffer || memory.resource >= buffer_count) {
				continue;
			}
			const auto buffer = memory.resource;
			if (plan.buffer_writes_bounded[buffer] == 0u) {
				continue;
			}
			const auto predicate = inst.Arg(inst.NumArgs() - 1u).Resolve();
			if (!BoundableWriteOperand(program, inst.Arg(1), predicate, 0u) ||
			    !BoundableWriteOperand(program, inst.Arg(2), predicate, 0u) ||
			    !BoundableWriteOperand(program, inst.Arg(3), predicate, 0u)) {
				plan.buffer_writes_bounded[buffer] = 0u;
				continue;
			}
			listed_writes[buffer]++;
			plan.buffer_writes.push_back({.buffer    = buffer,
			                              .immediate = memory.offset,
			                              .index     = Clone(inst.Arg(1)),
			                              .offset    = Clone(inst.Arg(2)),
			                              .soffset   = Clone(inst.Arg(3)),
			                              .predicate = Clone(predicate)});
		}
	}
	for (uint32_t i = 0; i < buffer_count; ++i) {
		if (listed_writes[i] == 0u) {
			plan.buffer_writes_bounded[i] = 0u;
		}
	}
	plan.gpu_fill_slots = GpuFillSlots(plan);
	return plan;
}

namespace {

struct WriteInterval {
	uint64_t lo = 0;
	uint64_t hi = 0;
};

// The unsigned 32-bit range a store address operand takes over the dispatch: invocation IDs range
// over the workgroup count and size, everything else the host evaluates exactly. Fails on an
// unknown dispatch size or a range that could wrap.
template <typename Walker>
bool EvaluateWriteInterval(const ResourcePlan& program, const SrtRuntime& runtime, Walker& walker,
                           Value value, Value predicate, WriteInterval& out, uint32_t& budget) {
	constexpr uint64_t Max32 = UINT32_MAX;
	if (budget == 0u) {
		return false;
	}
	--budget;
	value = value.Resolve();
	if (value.IsImmediate()) {
		if (value.GetType() != Type::U32) {
			return false;
		}
		out = {value.U32(), value.U32()};
		return true;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (IsBoundedInvocationId(program, *inst)) {
		const auto  axis  = inst->Arg(1).U32();
		if (runtime.workgroup_counts.size() < 3u) {
			return false;
		}
		const auto& count = runtime.workgroup_counts;
		const auto& size  = runtime.workgroup_size;
		uint64_t    total = 0;
		switch (static_cast<StageInputKind>(inst->Arg(0).U32())) {
			case StageInputKind::WorkgroupId: total = count[axis]; break;
			case StageInputKind::LocalInvocationId: total = size[axis]; break;
			case StageInputKind::LocalInvocationIndex:
				total = uint64_t {size[0]} * size[1] * size[2];
				break;
			case StageInputKind::GlobalInvocationId:
				total = uint64_t {count[axis]} * size[axis];
				break;
			default: return false;
		}
		if (total == 0u || total - 1u > Max32) {
			return false;
		}
		out = {0u, total - 1u};
		return true;
	}
	const auto op = inst->GetOpcode();
	if (!IsIntervalOperation(op)) {
		uint32_t exact = 0;
		if (!walker.Evaluate(value, exact)) {
			return false;
		}
		out = {exact, exact};
		return true;
	}
	if (op == ValueOpcode::SelectU32 && inst->Arg(0).Resolve() == predicate) {
		return EvaluateWriteInterval(program, runtime, walker, inst->Arg(1), predicate, out, budget);
	}
	std::array<WriteInterval, 3> args {};
	const size_t                 first = op == ValueOpcode::SelectU32 ? 1u : 0u;
	if (inst->NumArgs() > args.size()) {
		return false;
	}
	for (size_t i = first; i < inst->NumArgs(); ++i) {
		if (!EvaluateWriteInterval(program, runtime, walker, inst->Arg(i), predicate, args[i],
		                           budget)) {
			return false;
		}
	}
	const auto& a = args[0];
	const auto& b = args[1];
	switch (op) {
		case ValueOpcode::IAdd32: out = {a.lo + b.lo, a.hi + b.hi}; break;
		case ValueOpcode::IMul32: out = {a.lo * b.lo, a.hi * b.hi}; break;
		case ValueOpcode::ShiftLeftLogical32:
			if (b.hi > 31u) {
				return false;
			}
			out = {a.lo << b.lo, a.hi << b.hi};
			break;
		case ValueOpcode::ShiftRightLogical32:
			if (b.hi > 31u) {
				return false;
			}
			out = {a.lo >> b.hi, a.hi >> b.lo};
			break;
		case ValueOpcode::BitwiseAnd32: out = {0u, std::min(a.hi, b.hi)}; break;
		case ValueOpcode::BitwiseOr32: {
			const auto high = std::max(a.hi, b.hi);
			out = {std::max(a.lo, b.lo), (uint64_t {1} << std::bit_width(high)) - 1u};
			break;
		}
		case ValueOpcode::UMin32: out = {std::min(a.lo, b.lo), std::min(a.hi, b.hi)}; break;
		case ValueOpcode::UMax32: out = {std::max(a.lo, b.lo), std::max(a.hi, b.hi)}; break;
		case ValueOpcode::BitFieldUExtract: {
			// (base >> offset) masked to count bits: at most both.
			if (b.hi > 31u) {
				return false;
			}
			const auto& count = args[2];
			const auto  mask  = count.hi >= 32u ? Max32 : (uint64_t {1} << count.hi) - 1u;
			out               = {0u, std::min(mask, a.hi >> b.lo)};
			break;
		}
		case ValueOpcode::SelectU32:
			out = {std::min(args[1].lo, args[2].lo), std::max(args[1].hi, args[2].hi)};
			break;
		default: return false;
	}
	return out.hi <= Max32;
}

// Set while a snapshot has a GPU-sized buffer (GpuSizedRecords): its write extents are evaluated
// without waiting, and one that needs GPU-written bytes covers the (bounded) binding.
thread_local bool t_gpu_sized_snapshot = false;
thread_local uint64_t t_extents_shader_hash = 0; // for OptionalScope below

// See ResourceSnapshot::buffer_write_extents.
template <typename Walker>
void ComputeBufferWriteExtents(const ResourcePlan& program, const SrtRuntime& runtime,
                               Walker& walker, ResourceSnapshot& snapshot) {
	snapshot.buffer_write_extents.assign(program.info.buffers.size(), {});
	t_extents_shader_hash = program.shader_hash;
	if (program.buffer_writes.empty()) {
		return;
	}
	std::vector<uint8_t> failed(program.info.buffers.size(), 0u);
	// Extents only narrow the bytes marked GPU-written; one the host cannot evaluate without
	// waiting for the GPU falls back to the whole buffer (SrtConditionsNoWait).
	// Measured 2026-10-09: the whole-buffer fallback marks more memory GPU-written, and later SRT
	// reads of it wait instead (CS 4e7f2c6bb9b158a1: 1.0 -> 4.4 s of waits per 20 s). Opt-in only.
	static const bool extents_no_wait = std::getenv("KYTY_SRT_EXTENTS_NO_WAIT") != nullptr;
	struct OptionalScope {
		bool active = t_gpu_sized_snapshot || extents_no_wait ||
		              ExtentsNoWaitShader(t_extents_shader_hash);
		OptionalScope() {
			if (active) SrtSetEvaluatingCondition(true);
		}
		~OptionalScope() {
			if (active) SrtSetEvaluatingCondition(false);
		}
	} scope;
	for (const auto& write: program.buffer_writes) {
		if (write.buffer >= failed.size() || failed[write.buffer] != 0u) {
			continue;
		}
		auto&                extent = snapshot.buffer_write_extents[write.buffer];
		ShaderBufferResource descriptor;
		WriteInterval        index;
		WriteInterval        offset;
		WriteInterval        soffset;
		uint32_t             budget    = 1024u;
		const auto           predicate = write.predicate.Resolve();
		if (program.buffer_writes_bounded[write.buffer] == 0u ||
		    !DecodeBufferDescriptor(snapshot.buffers[write.buffer], descriptor) ||
		    descriptor.SwizzleEnabled() ||
		    !EvaluateWriteInterval(program, runtime, walker, write.index, predicate, index, budget) ||
		    !EvaluateWriteInterval(program, runtime, walker, write.offset, predicate, offset, budget) ||
		    !EvaluateWriteInterval(program, runtime, walker, write.soffset, predicate, soffset,
		                           budget)) {
			failed[write.buffer] = 1u;
			extent               = {};
			continue;
		}
		if (descriptor.AddTid()) {
			index.hi += 63u;
		}
		// Buffer addressing: base + soffset + offset + index * stride, at most 16 bytes a
		// store (dwordx4, format xyzw), and a store outside the descriptor's range is dropped.
		const uint64_t stride = descriptor.Stride();
		const uint64_t size   = descriptor.GetSize();
		const uint64_t begin =
		    std::min(index.lo * stride + offset.lo + soffset.lo + write.immediate, size);
		const uint64_t end =
		    std::min(index.hi * stride + offset.hi + soffset.hi + write.immediate + 16u, size);
		if (!extent.valid) {
			extent = {.begin = begin, .end = end, .valid = true};
		} else {
			extent.begin = std::min(extent.begin, begin);
			extent.end   = std::max(extent.end, end);
		}
	}
}

} // namespace

int LastIndirectImageFailureLine() {
	return g_indirect_failure_line;
}

namespace {

// KYTY_GPU_SIZED_BUFFERS=0 waits for GPU-written record counts of read-only buffers, as before.
bool GpuSizedBuffersEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_BUFFERS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// The record count bound for a read-only buffer whose count the GPU wrote: KYTY_GPU_SIZED_BUFFER_KB
// (default 256) bytes of records at the V#'s stride (dword 1, bits 16-29), or that many bytes for
// a raw buffer. The binding is clamped to mapped guest memory like any other.
uint32_t GpuSizedRecords(uint32_t dword1, bool written = false) {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_BUFFER_KB");
		const auto  kb    = value != nullptr ? std::strtoull(value, nullptr, 10) : 256ull;
		return (kb == 0 ? 256ull : kb) * 1024ull;
	}();
	static const uint64_t written_bytes = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_WRITTEN_KB");
		const auto  kb    = value != nullptr ? std::strtoull(value, nullptr, 10) : 4096ull;
		return (kb == 0 ? 4096ull : kb) * 1024ull;
	}();
	const auto     limit  = written ? written_bytes : bytes;
	const uint32_t stride = (dword1 >> 16u) & 0x3fffu;
	return static_cast<uint32_t>(stride != 0u ? limit / stride : limit);
}

// KYTY_GPU_SIZED_WRITTEN=0 waits for GPU-written record counts of written buffers, as before.
// On: a written, non-atomic buffer whose count the GPU wrote is bound with a conservative count
// (KYTY_GPU_SIZED_WRITTEN_KB, default 4096 KB of records) instead of waiting for the GPU. Its
// shader tests its records against the count it reads itself (CS fb5ee753d43f3f0b, a sort of
// GPU-counted keys: ~20 ms of GPU-thread wait a frame at the title menu, 2026-10-09).
// Only for the shaders listed (KYTY_GPU_SIZED_WRITTEN_SHADERS, hex hashes, default the sort
// above): applied to every written buffer, CS 380b6ffba703cef1's binding lost the device.
// KYTY_GPU_SIZED_DYNAMIC (on by default, =0 turns it off): a buffer bound with a conservative record count (its count, V#
// dword 2, was written by the GPU) keeps its exact bounds on the GPU: the shader reads the count
// from the flattened SRT (the GPU fills that slot before the dispatch) and bounds its accesses by
// it (BufferResource::gpu_records_flat). Written buffers then size without waiting too: the
// conservative count alone let stores past the real count land (KYTY_GPU_SIZED_WRITTEN lost the
// device), with the exact bound they are dropped as on the console. Default shaders: the two whose
// counts made the GPU thread wait for the GPU every frame in gameplay (fb5ee753d43f3f0b, an async
// sort, and 380b6ffba703cef1); every shader unless KYTY_GPU_SIZED_WRITTEN_SHADERS lists some.
bool GpuSizedDynamicEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_DYNAMIC");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool GpuSizedWrittenEnabled(uint64_t shader_hash) {
	// Opt-in (KYTY_GPU_SIZED_WRITTEN=1): on by default, the title menu lost the device at ~57 s in
	// three runs of three (a DISPATCH_INDIRECT hung, 2026-10-09); off, it did not.
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_WRITTEN");
		return (value != nullptr && value[0] == '1') || GpuSizedDynamicEnabled();
	}();
	// Dynamic bounds make the conservative count safe, so every shader by default: the GPU-thread
	// wait otherwise moves to the next shader reading the same GPU-written counts (CS
	// 380b6ffba703cef1 once fb5ee753d43f3f0b no longer waited, 2026-10-17).
	static const bool all = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_WRITTEN_SHADERS");
		return value != nullptr ? std::string_view(value) == "all" : GpuSizedDynamicEnabled();
	}();
	if (all) {
		return enabled;
	}
	static const std::vector<uint64_t> shaders = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_WRITTEN_SHADERS");
		std::string list  = value != nullptr ? value
		                    : GpuSizedDynamicEnabled() ? "fb5ee753d43f3f0b,2ea3957af2daf690"
		                                               : "fb5ee753d43f3f0b";
		std::vector<uint64_t> result;
		size_t                at = 0;
		while (at < list.size()) {
			const auto end = list.find_first_of(",; ", at);
			const auto item =
			    list.substr(at, end == std::string::npos ? std::string::npos : end - at);
			if (!item.empty()) {
				result.push_back(std::strtoull(item.c_str(), nullptr, 16));
			}
			if (end == std::string::npos) break;
			at = end + 1;
		}
		return result;
	}();
	return enabled && std::find(shaders.begin(), shaders.end(), shader_hash) != shaders.end();
}

// Whether `value` reads a flat slot the GPU fills in this refresh (SrtGpuFill): the host's copy
// of such a slot is a placeholder (0), not the value.
bool DependsOnGpuFill(const ResourcePlan& program, Value value,
                      const std::vector<SrtGpuFill>& fills) {
	if (fills.empty()) {
		return false;
	}
	std::vector<Value>              stack {value};
	std::unordered_set<const Inst*> seen;
	while (!stack.empty() && seen.size() < 256u) {
		const auto* inst = stack.back().Resolve().TryInstruction();
		stack.pop_back();
		if (inst == nullptr || !seen.insert(inst).second) {
			continue;
		}
		for (const auto& read: program.srt_reads) {
			if (read.value.ResolveInstruction() != inst) continue;
			for (const auto& fill: fills) {
				if (fill.flat_offset == read.flat_offset) {
					return true;
				}
			}
		}
		for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
			stack.push_back(inst->Arg(arg));
		}
	}
	// Too large to follow: not known.
	return !stack.empty();
}

// KYTY_GPU_SIZED_DYNAMIC: the bound record count of a buffer the shader bounds by its exact count
// (gpu_records_flat). A fixed bound is either too small (CS fb5ee753d43f3f0b's count passed 4 MB of
// records: its keys past the bound were dropped and a later draw, 618eca57f179de7d, cost 25-80 ms
// of GPU time a frame) or binds, and marks GPU-written, far more memory than the shader touches (64
// MB: other shaders then waited for those bytes, and the game crashed once). The bound is twice
// the largest count seen so far, from copies of the count that never make the host wait
// (SrtRuntime::peek_gpu_dword), at least KYTY_GPU_SIZED_DYNAMIC_MIN_KB (default 256) and at most
// KYTY_GPU_SIZED_DYNAMIC_MAX_KB (default 262144) of records; the fixed bound until a copy lands.
uint32_t LearnedRecords(uint64_t shader_hash, uint32_t buffer, uint32_t flat_offset,
                        uint32_t add_flat, uint32_t add, uint32_t dword1, uint32_t fallback,
                        const SrtRuntime& runtime, const ResourceSnapshot& snapshot) {
	static const uint64_t min_bytes = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_DYNAMIC_MIN_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 256ull) * 1024ull;
	}();
	static const uint64_t max_bytes = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_DYNAMIC_MAX_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 262144ull) * 1024ull;
	}();
	if (runtime.peek_gpu_dword == nullptr) {
		return fallback;
	}
	// A slot the GPU fills: its last seen value; one the host read: its value.
	bool       any       = false;
	const auto slot_value = [&](uint32_t offset) -> uint64_t {
		for (const auto& fill: snapshot.gpu_fills) {
			if (fill.flat_offset == offset) {
				any = true;
				return runtime.peek_gpu_dword(fill.address);
			}
		}
		return offset < snapshot.flattened_srt.size() ? snapshot.flattened_srt[offset] : 0u;
	};
	uint64_t total = slot_value(flat_offset) + add;
	if (add_flat != BufferResource::NoGpuRecords) {
		total += slot_value(add_flat);
	}
	if (!any) {
		return fallback;
	}
	const auto seen = static_cast<uint32_t>(std::min<uint64_t>(total, UINT32_MAX));
	static std::mutex                             mutex;
	static std::unordered_map<uint64_t, uint32_t> largest;
	uint32_t                                      most = 0;
	{
		std::lock_guard lock(mutex);
		auto&           entry = largest[shader_hash ^ (uint64_t {buffer} << 48u)];
		const bool      grew  = seen > entry;
		entry                 = std::max(entry, seen);
		most                  = entry;
		if (grew) {
			if (static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr; log) {
				std::fprintf(stderr, "[gpu-sized-dyn] %016llx buffer %u: largest count %u\n",
				             static_cast<unsigned long long>(shader_hash), buffer, most);
			}
		}
	}
	if (most == 0) {
		return fallback;
	}
	const uint64_t stride = (dword1 >> 16u) & 0x3fffu;
	const uint64_t unit   = stride == 0u ? 1u : stride;
	const uint64_t bytes  = std::clamp<uint64_t>(uint64_t {most} * unit * 2u, min_bytes, max_bytes);
	return static_cast<uint32_t>(std::min<uint64_t>(bytes / unit, UINT32_MAX));
}

void NoteGpuSizedBuffer(uint64_t shader_hash) {
	static std::atomic<uint64_t> count {0};
	const auto                   n = count.fetch_add(1, std::memory_order_relaxed) + 1;
	if ((n & (n - 1)) == 0) {
		Common::HackNotice::Report("buffer record count written by the GPU replaced by a fixed bound",
		                           "gpu-sized buffers", shader_hash, 0,
		                           "KYTY_GPU_SIZED_BUFFER_KB records, without waiting for the GPU");
		std::fprintf(stderr, "[gpu-sized-buffer] %llu bindings sized without waiting (last: %016llx)\n",
		             static_cast<unsigned long long>(n), static_cast<unsigned long long>(shader_hash));
	}
}

} // namespace

void SetComputeIndirectFloors(size_t keys, size_t images) {
	g_compute_indirect_keys.store(std::max(std::bit_ceil(keys), MinIndirectMappingCapacity),
	                              std::memory_order_relaxed);
	g_compute_indirect_images.store(std::max<size_t>(std::bit_ceil(images), 1u),
	                                std::memory_order_relaxed);
	g_stable_indirect_layouts.store(true, std::memory_order_relaxed);
}

template <typename Walker>
static bool MaterializeWith(const ResourcePlan& program, const SrtRuntime& runtime,
                            ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	for (auto& keys: program.selector_keys) keys.resolved = false;
	g_indirect_failure_line = 0;
	t_gpu_sized_snapshot    = false;
	snapshot.bindless_heaps.clear();
	snapshot.bindless_sampler_heaps.clear();
	if (!program.resource_tracking_complete ||
	    ((program.requires_specialization_memory || !program.source_reads.empty()) &&
	     runtime.read_specialization_memory == nullptr)) {
		return false;
	}
	const auto uses_workgroup_count = [](const DescriptorSource& source) {
		return source.indirect_descriptor && source.indirect_descriptor->workgroup_axis != UINT32_MAX;
	};
	const bool capture_reads = program.capture_specialization_reads || !program.source_reads.empty() ||
	    (runtime.indirect_args_address != 0u &&
	     std::ranges::any_of(program.descriptor_sources, uses_workgroup_count));
	auto& reads = snapshot.specialization_reads;
	reads = program.source_reads;
	snapshot.specialization_read_slots.assign(reads.size(), NoSrtSlot);
	SrtReadCapture capture {runtime, reads, &snapshot.specialization_read_slots};
	auto observed = capture_reads ? capture.ObservedRuntime() : runtime;
	// Reads through the fast page path are captured there.
	if (capture_reads) {
		observed.capture_ranges = &reads;
		observed.capture_slots  = &snapshot.specialization_read_slots;
	}
	Walker clean(program, CleanRuntime(observed));
	Walker walker(program, observed,
	              capture_reads || program.requires_specialization_memory ? &clean : nullptr);
	g_materialize_phase = "flat";
	if (!walker.RefreshFlatBuffer(snapshot.flattened_srt)) {
		return SpecializationFail(fmt::format(
		    "SRT refresh complete={} reads={} user_data={} shader={:016x} failure={} at 0x{:x}/0x{:x}",
		    program.srt_plan_complete, program.srt_reads.size(), runtime.user_data.size(),
		    program.shader_hash, walker.ReadFailure() != nullptr ? walker.ReadFailure() : "-",
		    walker.ReadFailureAddress(), clean.ReadFailureAddress()));
	}
	const auto active = std::span<const uint8_t>(program.active_sources);
	std::array<uint32_t, 3> workgroup_counts {};
	if (runtime.indirect_args_address != 0u) {
		for (uint32_t source = 0; source < program.descriptor_sources.size(); ++source) {
			if ((!active.empty() && !active[source]) ||
			    !uses_workgroup_count(program.descriptor_sources[source])) continue;
			if ((runtime.indirect_args_address & 3u) != 0u ||
			    !ReadTable(runtime.indirect_args_address, sizeof(workgroup_counts), 0u,
			               observed, workgroup_counts)) return false;
			observed.workgroup_counts = workgroup_counts;
			break;
		}
	}
	snapshot.uniform_fill = {};
	const auto& fill = program.uniform_fill;
	auto words = fill.fill.words;
	std::array<uint32_t, 4> stored {};
	bool uniform_fill = words != 0;
	if (uniform_fill && !fill.pattern_size.IsEmpty())
		uniform_fill = clean.Evaluate(fill.pattern_size, words) && words >= 1 && words <= 4;
	uint32_t elements = 0;
	if (uniform_fill && !fill.element_count.IsEmpty())
		uniform_fill = clean.Evaluate(fill.element_count, elements) && elements != 0;
	for (uint32_t i = 0; i < words && uniform_fill; ++i) {
		uniform_fill = clean.Evaluate(fill.values[i], stored[i]) && stored[i] == stored[0];
	}
	if (uniform_fill) {
		snapshot.uniform_fill = fill.fill;
		snapshot.uniform_fill.value = stored[0];
		snapshot.uniform_fill.bounded_elements = elements;
	}
	const auto evaluate = [&](uint32_t source, DescriptorValue& value, bool written = false) {
		if (source >= program.descriptor_sources.size()) {
			return false;
		}
		if (active.empty() || active[source]) {
			return (capture_reads && written ? clean : walker).EvaluateDescriptor(source, value) ||
			    SpecializationFail(fmt::format("descriptor evaluation source={}", source));
		}
		value = {};
		value.dword_count = program.descriptor_sources[source].dword_count;
		return true;
	};
	const auto resource_failed = [&](std::string_view kind, uint32_t resource, uint32_t source,
	                                 size_t candidates) {
		return SpecializationFail(fmt::format("indirect {} resource={} source={} candidates={}",
		    kind, resource, source, candidates));
	};
	g_materialize_phase = "buffers";
	snapshot.buffers.resize(program.info.buffers.size());
	specialization.buffers.assign(program.info.buffers.size(), {});
	for (uint32_t i = 0; i < snapshot.buffers.size(); ++i) {
		// KYTY_GPU_SIZED_DYNAMIC: the flat SRT dword holding this buffer's exact record count.
		uint32_t gpu_records_flat = BufferResource::NoGpuRecords;
		uint32_t gpu_records_add_flat = BufferResource::NoGpuRecords;
		uint32_t gpu_records_add      = 0;
		bool     gpu_records_ssa      = false;
		const auto base_index =
		    i < program.info.buffers.size() ? i : specialization.buffers[i].indirect_root;
		if (base_index >= program.info.buffers.size()) return false;
		const auto& base = program.info.buffers[base_index];
		if (i < program.info.buffers.size()) {
			const auto* source = Source(program, base.source);
			if (source == nullptr) return false;
			if (static const char* diag = std::getenv("KYTY_SRT_FILL_REASON");
			    diag != nullptr &&
			    std::string_view(diag).find(fmt::format("{:016x}", program.shader_hash)) !=
			        std::string_view::npos) {
				static std::atomic<uint32_t> printed {0};
				if (printed.fetch_add(1, std::memory_order_relaxed) < 64) {
					std::fprintf(stderr,
					             "[gpu-sized-buffer] %016llx buffer %u source %u written=%d atomic=%d "
					             "scalar=%d indirect=%d active=%d dwords=%u\n",
					             static_cast<unsigned long long>(program.shader_hash), i, base.source,
					             base.written ? 1 : 0, base.atomic ? 1 : 0, base.scalar ? 1 : 0,
					             source->indirect_descriptor.has_value() ? 1 : 0,
					             active.empty() || active[base.source] ? 1 : 0, source->dword_count);
				}
			}
			if (source->indirect_descriptor.has_value()) {
				snapshot.buffers[i] = {.dword_count = 4u};
				if (active.empty() || active[base.source]) {
					if (!MaterializeIndirectDescriptor(
					        program, base.source, i, 4u, 4u, observed, clean, snapshot,
					        snapshot.buffers, specialization.buffers,
					        [&](DescriptorValue& value) {
						        return NormalizeIndirectBuffer(
						            value, source->indirect_descriptor->indexed, base);
					        }))
						return resource_failed("buffer", i, base.source, snapshot.buffers.size());
				}
			} else if ((!base.written || GpuSizedWrittenEnabled(program.shader_hash)) &&
			           (!base.atomic || GpuSizedAtomicEnabled(program.shader_hash)) &&
			           GpuSizedBuffersEnabled() && (active.empty() || active[base.source])) {
				// A read-only buffer whose record count (V# dword 2) the GPU wrote: binding it
				// needs the base, stride and format, not the exact count. Read the descriptor
				// refusing GPU-written bytes; when that fails, read dwords 0, 1 and 3 normally and
				// give dword 2 a conservative count (GpuSizedRecords), so the GPU thread does not
				// wait for the GPU. The shader keeps its own bounds tests (the count in its SGPRs
				// comes from the flat buffer, which the GPU fills).
				auto&       walker_ref = capture_reads && base.written ? clean : walker;
				const auto& src        = program.descriptor_sources[base.source];
				// A refused read can still let an evaluation succeed, with a value that is not
				// known (a condition's): dwords 0, 1 and 3 are read normally, and dword 2 counts as
				// refused when any read under it was (SrtConditionRefusals). Evaluating the whole
				// descriptor refusing reads bound such a value as the count (the title menu lost
				// the device in three runs of three, 2026-10-09).
				bool ok = false;
				if (src.dword_count != 4u) {
					ok = evaluate(base.source, snapshot.buffers[i], base.written);
				} else {
					auto& value       = snapshot.buffers[i];
					value             = {};
					value.dword_count = 4u;
					ok                = true;
					bool sized        = false;
					for (uint32_t d = 0; d < 4u && ok; ++d) {
						if (d == 2u) {
							const auto refusals = SrtConditionRefusals();
							SrtSetEvaluatingCondition(true);
							sized = !walker_ref.Evaluate(src.dwords[2], value.dwords[2]);
							SrtSetEvaluatingCondition(false);
							sized = sized || SrtConditionRefusals() != refusals ||
							        DependsOnGpuFill(program, src.dwords[2], snapshot.gpu_fills);
							continue;
						}
						ok = walker_ref.Evaluate(src.dwords[d], value.dwords[d]);
					}
					if (ok && sized && GpuSizedDynamicEnabled()) {
						// The count must be one flat slot read as it is, so the shader can read
						// it there; bounds of swizzled or lane-indexed buffers are not a plain
						// length. Otherwise a written buffer waits for its exact count.
						// Also flat + flat and flat + immediate (CS 380b6ffba703cef1, f8eeaa7f21b1450a).
						const auto flat_of = [&](Value v) -> uint32_t {
							const auto* inst = v.Resolve().TryInstruction();
							if (inst == nullptr) {
								return BufferResource::NoGpuRecords;
							}
							for (const auto& read: program.srt_reads) {
								if (read.value.ResolveInstruction() == inst) return read.flat_offset;
							}
							return BufferResource::NoGpuRecords;
						};
						const auto imm_of = [](Value v, uint32_t& out) {
							v = v.Resolve();
							if (!v.IsImmediate() || v.GetType() != Type::U32) return false;
							out = v.U32();
							return true;
						};
						const auto* read = src.dwords[2].Resolve().TryInstruction();
						if (read != nullptr && ((value.dwords[1] >> 31u) & 1u) == 0u &&
						    ((value.dwords[3] >> 23u) & 1u) == 0u) {
							if (const auto direct = flat_of(src.dwords[2]);
							    direct != BufferResource::NoGpuRecords) {
								gpu_records_flat = direct;
							} else if (read->GetOpcode() == ValueOpcode::IAdd32) {
								const auto a = flat_of(read->Arg(0));
								const auto b = flat_of(read->Arg(1));
								uint32_t   imm = 0;
								if (a != BufferResource::NoGpuRecords && b != BufferResource::NoGpuRecords) {
									gpu_records_flat     = a;
									gpu_records_add_flat = b;
								} else if (a != BufferResource::NoGpuRecords && imm_of(read->Arg(1), imm)) {
									gpu_records_flat = a;
									gpu_records_add  = imm;
								} else if (b != BufferResource::NoGpuRecords && imm_of(read->Arg(0), imm)) {
									gpu_records_flat = b;
									gpu_records_add  = imm;
								}
							}
						}
						if (static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr; log) {
							static std::mutex                   mutex;
							static std::unordered_set<uint64_t> seen;
							std::lock_guard                     lock(mutex);
							if (seen.insert(program.shader_hash ^ (uint64_t {i} << 56u)).second) {
								// The count's expression, two levels deep, when it is not a flat slot.
								std::string shape;
								const std::function<void(Value, int)> describe = [&](Value v, int depth) {
									v = v.Resolve();
									if (v.IsImmediate()) {
										shape += fmt::format("#{:x}", v.GetType() == Type::U32 ? v.U32() : 0u);
										return;
									}
									const auto* inst = v.TryInstruction();
									if (inst == nullptr) {
										shape += "?";
										return;
									}
									shape += ValueOpcodeName(inst->GetOpcode());
									if (depth == 0 || inst->NumArgs() == 0) return;
									shape += "(";
									for (size_t a = 0; a < inst->NumArgs(); a++) {
										if (a != 0) shape += ",";
										describe(inst->Arg(a), depth - 1);
									}
									shape += ")";
								};
								if (gpu_records_flat == BufferResource::NoGpuRecords) {
									describe(src.dwords[2], 3);
								}
								std::fprintf(stderr,
								             "[gpu-sized-dyn] %016llx buffer %u written=%d flat=%d %s\n",
								             static_cast<unsigned long long>(program.shader_hash),
								             i, base.written ? 1 : 0,
								             gpu_records_flat == BufferResource::NoGpuRecords
								                 ? -1
								                 : static_cast<int>(gpu_records_flat),
								             shape.c_str());
							}
						}
						// No flat slot: the shader bounds its accesses by the count its own
						// descriptor computes (KYTY_GPU_SIZED_SSA=0 waits for the count instead).
						static const bool ssa = [] {
							const char* env = std::getenv("KYTY_GPU_SIZED_SSA");
							return env == nullptr || env[0] != '0';
						}();
						if (gpu_records_flat == BufferResource::NoGpuRecords && ssa &&
						    ((value.dwords[1] >> 31u) & 1u) == 0u &&
						    ((value.dwords[3] >> 23u) & 1u) == 0u) {
							gpu_records_ssa = true;
						}
						if (gpu_records_flat == BufferResource::NoGpuRecords && !gpu_records_ssa &&
						    base.written) {
							ok    = walker_ref.Evaluate(src.dwords[2], value.dwords[2]);
							sized = false;
						}
					}
					if (ok && sized) {
						value.dwords[2] = GpuSizedRecords(value.dwords[1], base.written);
						if (gpu_records_flat != BufferResource::NoGpuRecords) {
							value.dwords[2] = LearnedRecords(program.shader_hash, i, gpu_records_flat,
							                                 gpu_records_add_flat, gpu_records_add,
							                                 value.dwords[1], value.dwords[2],
							                                 runtime, snapshot);
						}
						t_gpu_sized_snapshot = true;
						// With the count read on the GPU (flat slot or the descriptor's own
						// expression) the shader bounds its accesses exactly: only a count it
						// cannot read there stays a fixed bound.
						if (gpu_records_flat == BufferResource::NoGpuRecords && !gpu_records_ssa) {
							NoteGpuSizedBuffer(program.shader_hash);
						}
					}
				}
				if (!ok) {
					return false;
				}
			} else if (!evaluate(base.source, snapshot.buffers[i], base.written)) {
				return false;
			}
		}
		auto&                descriptor_value = snapshot.buffers[i];
		ShaderBufferResource descriptor;
		if (!DecodeBufferDescriptor(descriptor_value, descriptor)) {
			return SpecializationFail(fmt::format("buffer descriptor {} has invalid width", i));
		}
		if (descriptor.Type() != 0) {
			descriptor_value.dwords.fill(0);
			descriptor = {};
		}
		auto       packed_stride = descriptor.PackedStride();
		const auto stride        = packed_stride & 0x3fffu;
		const bool swizzle       = stride != 0u && ((packed_stride >> 14u) & 1u) != 0u;
		if (stride == 0u) {
			packed_stride &= ~((1u << 14u) | (3u << 16u));
		} else if (!swizzle) {
			packed_stride &= ~(3u << 16u);
		}
		auto& buffer         = specialization.buffers[i];
		buffer.packed_stride = packed_stride;
		buffer.descriptor_format =
		    base.formatted ? descriptor.Format() : Prospero::BufferFormat::kInvalid;
		buffer.descriptor_swizzle = base.formatted ? descriptor.DstSelXYZW() : DstSel(4, 5, 6, 7);
		buffer.zero_stride_oob    = descriptor.OutOfBounds() == 0u && stride == 0u;
		buffer.gpu_records_flat   = gpu_records_flat;
		buffer.gpu_records_add_flat = gpu_records_add_flat;
		buffer.gpu_records_add      = gpu_records_add;
		buffer.gpu_records_ssa      = gpu_records_ssa;
		buffer.write_tracked      = WriteBitmapEnabled() && i < program.info.buffers.size() &&
		                            base.written && base.image_alias == BufferResource::NoImageAlias &&
		                            i < program.buffer_writes_bounded.size() &&
		                            program.buffer_writes_bounded[i] == 0u;
		buffer.scalar_empty       = base.scalar && descriptor.GetSize() == 0u;
	}
	{
		// Write extents only size the GPU-written range; their reads do not specialize the
		// shader, so they stay out of the specialization-read proof.
		const auto captured = reads.size();
		g_materialize_phase = "extents";
		ComputeBufferWriteExtents(program, observed, walker, snapshot);
		if (capture_reads) {
			reads.resize(captured);
		}
	}
	g_materialize_phase = "images";
	g_debug_raw_images.assign(program.info.images.size(), DescriptorValue {});
	snapshot.images.resize(program.info.images.size());
	specialization.images.resize(program.info.images.size());
	auto previous_tables = std::move(snapshot.image_tables);
	snapshot.image_tables.clear();
	std::vector<uint32_t> table_mappings;
	for (uint32_t i = 0; i < program.info.images.size(); ++i) {
		const auto& image = program.info.images[i];
		specialization.images[i] = {
		    .numeric_class = image.numeric_class,
		    .dimension = image.dimension,
		    .mip_count = image.mip_count,
		    .conversion_format = image.conversion_format,
		    .shader_swizzle = image.shader_swizzle,
		    .indirect_root = image.indirect_root,
		    .indirect_mapping_offset = image.indirect_mapping_offset,
		    .indirect_search_iterations = image.indirect_search_iterations,
		    .table = image.table,
		    .table_capacity = image.table_capacity,
		    .table_mapping_offset = image.table_mapping_offset,
		    .table_entry_mask = image.table_entry_mask,
		    .cube = image.cube,
		    .bindless = image.bindless,
		};
		const auto* source = Source(program, image.source);
		if (source == nullptr) {
			return FailIndirect(__LINE__);
		}
		if (source->indirect_descriptor.has_value()) {
			snapshot.images[i] = {.dword_count = 8u};
			if (!active.empty() && !active[image.source]) {
				continue;
			}
			const auto& indirect = *source->indirect_descriptor;
			if (indirect.bindless) {
				DescriptorValue table_value;
				if (!clean.EvaluateDescriptor(indirect.table_source, table_value) ||
				    !MaterializeBindlessImage(indirect, table_value, i, snapshot, specialization)) {
					return FailIndirect(__LINE__);
				}
				continue;
			}
			if (indirect.table_array) {
				DescriptorValue table;
				if (!clean.EvaluateDescriptor(indirect.table_source, table) ||
				    !MaterializeImageTable(program, indirect, table, i, observed, snapshot,
				                           specialization, previous_tables, table_mappings)) {
					return FailIndirect(__LINE__);
				}
				continue;
			}
			const auto srt_size        = snapshot.flattened_srt.size();
			const auto image_count     = snapshot.images.size();
			const auto spec_count      = specialization.images.size();
			const auto failure_line    = g_indirect_failure_line;
			const bool bounded_table = IsBoundedDescriptorTable(program, indirect);
			if (!MaterializeIndirectDescriptor(
			        program, image.source, i, 8u, image.r128 ? 4u : 8u, observed, clean, snapshot, snapshot.images,
			        specialization.images, [&](DescriptorValue& value) {
				        // A broad heap also contains resources for other typed image operations.
				        if (NullImageDescriptor(value) || !ValidImageDescriptor(value, image.r128) ||
				            (bounded_table && DescriptorDimension(value, image.dimension) != image.dimension))
					        value.dwords.fill(0u);
				        return true;
			        })) {
				// A store-only image picked per material from a table too large to enumerate
				// (Wolverine's terrain and scatter write a per-material image from a 20 MB
				// material table): bind a null image, dropping those stores, rather than
				// skipping the draw itself.
				static const bool null_writes = [] {
					const char* env = std::getenv("KYTY_NULL_UNENUMERABLE_WRITES");
					return env == nullptr || env[0] != '0';
				}();
				if (!null_writes || image.read || image.atomic || !image.written) {
					return resource_failed("image", i, image.source, snapshot.images.size());
				}
				snapshot.flattened_srt.resize(srt_size);
				snapshot.images.resize(image_count);
				specialization.images.resize(spec_count);
				snapshot.images[i] = {.dword_count = 8u};
				g_indirect_failure_line = failure_line;
				Common::HackNotice::Report(
				    "store-only image from a table too large to enumerate bound as a null image",
				    "MaterializeWith", program.shader_hash, i, "its stores are dropped");
			}
		} else {
			if (!evaluate(image.source, snapshot.images[i], image.written)) {
				return FailIndirect(__LINE__);
			}
			if (g_debug_raw_images.size() < snapshot.images.size()) {
				g_debug_raw_images.resize(snapshot.images.size());
			}
			g_debug_raw_images[i] = snapshot.images[i];
			if (!ValidImageDescriptor(snapshot.images[i], image.r128)) {
				snapshot.images[i].dwords.fill(0);
			}
		}
	}
	g_materialize_phase = "samplers";
	snapshot.samplers.resize(program.info.samplers.size());
	specialization.samplers.assign(program.info.samplers.size(), {});
	for (uint32_t i = 0; i < snapshot.samplers.size(); ++i) {
		const auto base_index =
		    i < program.info.samplers.size() ? i : specialization.samplers[i].indirect_root;
		if (base_index >= program.info.samplers.size()) return false;
		if (i < program.info.samplers.size()) {
			auto source = program.info.samplers[i].source;
			const auto* descriptor = Source(program, source);
			if (descriptor == nullptr) return false;
			if (descriptor->bindless_sampler.has_value()) {
				// The shader indexes the host's mirror of the heap with two patched words (region
				// base, entry count); zeros select the default sampler in slot 0. Set 0 still binds
				// the default sampler for this resource.
				const auto mapping_offset = static_cast<uint32_t>(snapshot.flattened_srt.size());
				snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
				snapshot.samplers[i].dword_count = 4u;
				std::ranges::copy(BindlessDefaultSampler, snapshot.samplers[i].dwords.begin());
				specialization.samplers[i] = {.bindless = true, .bindless_mapping_offset = mapping_offset};
				DescriptorValue      table;
				ShaderBufferResource heap;
				if ((active.empty() || active[source]) &&
				    clean.EvaluateDescriptor(source, table) &&
				    DecodeBufferDescriptor(table, heap) && heap.Base48() != 0) {
					snapshot.bindless_sampler_heaps.push_back(
					    {.base           = heap.Base48(),
					     .size           = heap.GetSize(),
					     .table_offset   = descriptor->bindless_sampler->table_offset,
					     .sampler        = i,
					     .mapping_offset = mapping_offset});
				}
				continue;
			}
			const auto* indirect = descriptor->indirect_descriptor ? &*descriptor->indirect_descriptor : nullptr;
			if (indirect != nullptr && indirect->sources.empty()) {
				snapshot.samplers[i] = {.dword_count = 4u};
				if ((active.empty() || active[source]) && !MaterializeIndirectDescriptor(
				        program, source, i, 4u, 4u, observed, clean, snapshot, snapshot.samplers,
				        specialization.samplers,
				        [](DescriptorValue&) { return true; }))
					return resource_failed("sampler", i, source, snapshot.samplers.size());
			} else {
				if (indirect != nullptr) source = indirect->sources[0];
				if (!evaluate(source, snapshot.samplers[i])) return false;
			}
		}
		if (program.info.samplers[base_index].gather_lod) {
			const auto control = snapshot.samplers[i].dwords[2];
			const auto filter = (control >> 26u) & 3u;
			const auto perf_mip = (snapshot.samplers[i].dwords[1] >> 24u) & 15u;
			// MipNone always selects the base level. Point/linear selection requires zero
			// primary/secondary bias; linear mip snapping is not implemented.
			if (filter > 2u || (filter != 0u && (control & 0xfffffu) != 0u) ||
			    (filter == 2u && perf_mip != 0u)) {
				return SpecializationFail(
				    "explicit-LOD gather requires None, unbiased Point, or unbiased Linear without mip snapping");
			}
		}
	}
	snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	// KYTY_DEBUG_RESOURCES=<hash,...>: print that shader's image and sampler descriptors
	// (first 6 refreshes).
	if (static const std::string debug = [] {
		    const char* env = std::getenv("KYTY_DEBUG_RESOURCES");
		    return env != nullptr ? std::string(env) : std::string();
	    }();
	    !debug.empty() && debug.find(fmt::format("{:016x}", program.shader_hash)) != std::string::npos) {
		static std::mutex                             printed_mutex;
		static std::unordered_map<uint64_t, uint32_t> printed_per_shader;
		bool                                          print = false;
		{
			std::lock_guard lock(printed_mutex);
			print = printed_per_shader[program.shader_hash]++ < 3;
		}
		if (print) {
			std::string text = fmt::format("[debug-res] {:016x} images={} samplers={} buffers={}\n",
			                               program.shader_hash, snapshot.images.size(),
			                               snapshot.samplers.size(), snapshot.buffers.size());
			for (size_t i = 0; i < snapshot.images.size(); i++) {
				const auto& d = snapshot.images[i].dwords;
				const auto  w = ((d[2] >> 14u) & 0x3fffu) + 1u;
				const auto  h = ((d[2] >> 28u) | ((d[3] & 0x3ffu) << 4u)) + 1u;
				text += fmt::format(
				    "  img{} bindless={} indirect_root={} base=0x{:x} fmt={} {}x{} dw={:08x} {:08x} "
				    "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x}\n",
				    i, specialization.images[i].bindless ? 1 : 0,
				    specialization.images[i].indirect_root,
				    (static_cast<uint64_t>(d[1] & 0xffu) << 40u) | (static_cast<uint64_t>(d[0]) << 8u),
				    (d[1] >> 20u) & 0x1ffu, w, h, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
				if (i < g_debug_raw_images.size()) {
					const auto& r = g_debug_raw_images[i].dwords;
					text += fmt::format("    evaluated: {:08x} {:08x} {:08x} {:08x} {:08x} "
					                    "{:08x} {:08x} {:08x} source={} active={}\n",
					                    r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
					                    program.info.images[i].source,
					                    program.active_sources.empty() ||
					                            program.active_sources[program.info.images[i].source]
					                        ? 1 : 0);
				}
			}
			g_debug_raw_images.clear();
			for (size_t i = 0; i < snapshot.buffers.size(); i++) {
				const auto& d = snapshot.buffers[i].dwords;
				text += fmt::format("  buf{} {:08x} {:08x} {:08x} {:08x}\n", i, d[0], d[1], d[2], d[3]);
			}
			std::fputs(text.c_str(), stderr);
		}
	}
	if (!BuildResourceSpecialization(program, snapshot, specialization,
	                                 runtime.float_image_atomics)) {
		return FailIndirect(__LINE__);
	}
	return true;
}

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	// KYTY_COMPILED_SRT=0: the IR walker for every plan. The compiled plan does not model the GPU
	// fills of compute stages (SrtGpuFill) or the machine-code walker of plans without a reader.
	static const bool compiled_enabled = [] {
		const char* value = std::getenv("KYTY_COMPILED_SRT");
		return value == nullptr || value[0] != '0';
	}();
	if (compiled_enabled && runtime.gpu_fills == nullptr && runtime.gpu_written == nullptr &&
	    !(program.srt_code && runtime.read_memory == nullptr) &&
	    CompileSrtPlan(program) != nullptr) {
		return MaterializeWith<CompiledSrtWalker>(program, runtime, snapshot, specialization);
	}
	return MaterializeWith<SrtWalker>(program, runtime, snapshot, specialization);
}

bool MaterializeResourcesReference(const ResourcePlan& program, const SrtRuntime& runtime,
                                   ResourceSnapshot& snapshot,
                                   ResourceSpecialization& specialization) {
	return MaterializeWith<SrtWalker>(program, runtime, snapshot, specialization);
}

void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization) {
	EXIT_IF(!program.resource_tracking_complete || program.shader_info_complete);
	EXIT_IF(program.info.buffers.size() > specialization.buffers.size() ||
	        program.info.images.size() > specialization.images.size() ||
	        program.info.samplers.size() > specialization.samplers.size());

	auto& buffers = program.info.buffers;
	const auto original_buffer_count = buffers.size();
	buffers.reserve(specialization.buffers.size());
	for (uint32_t index = 0; index < specialization.buffers.size(); ++index) {
		const auto& source = specialization.buffers[index];
		if (index >= buffers.size()) {
			EXIT_IF(source.indirect_root >= original_buffer_count);
			buffers.push_back(buffers[source.indirect_root]);
		}
		auto& buffer                      = buffers[index];
		buffer.packed_stride              = source.packed_stride;
		buffer.descriptor_format          = source.descriptor_format;
		buffer.descriptor_swizzle         = source.descriptor_swizzle;
		buffer.write_tracked              = source.write_tracked;
		buffer.indirect_root              = source.indirect_root;
		buffer.indirect_mapping_offset    = source.indirect_mapping_offset;
		buffer.indirect_search_iterations = source.indirect_search_iterations;
		buffer.gpu_records_flat           = source.gpu_records_flat;
		buffer.gpu_records_add_flat       = source.gpu_records_add_flat;
		buffer.gpu_records_add            = source.gpu_records_add;
		buffer.gpu_records_ssa            = source.gpu_records_ssa;
		buffer.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < buffers.size(); ++index) {
		const auto root = buffers[index].indirect_root;
		if (root != BufferResource::NoIndirectBuffer) {
			EXIT_IF(root >= buffers.size());
			buffers[root].indirect_resources.push_back(index);
		}
	}
	auto& images = program.info.images;
	const auto original_image_count = images.size();
	images.reserve(specialization.images.size());
	for (uint32_t index = 0; index < specialization.images.size(); index++) {
		const auto& source = specialization.images[index];
		if (index >= images.size()) {
			EXIT_IF(source.indirect_root >= original_image_count);
			images.push_back(images[source.indirect_root]);
		}
		auto& image                      = images[index];
		image.numeric_class              = source.numeric_class;
		image.dimension                  = source.dimension;
		image.mip_count                  = source.mip_count;
		image.conversion_format          = source.conversion_format;
		image.shader_swizzle             = source.shader_swizzle;
		image.indirect_root              = source.indirect_root;
		image.indirect_mapping_offset    = source.indirect_mapping_offset;
		image.indirect_search_iterations = source.indirect_search_iterations;
		image.bindless                   = source.bindless;
		image.table                      = source.table;
		image.table_capacity             = source.table_capacity;
		image.table_mapping_offset       = source.table_mapping_offset;
		image.table_entry_mask           = source.table_entry_mask;
		image.cube                       = source.cube;
		image.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < images.size(); index++) {
		const auto root = images[index].indirect_root;
		if (root != ImageResource::NoIndirectImage) {
			EXIT_IF(root >= images.size());
			images[root].indirect_resources.push_back(index);
		}
	}

	auto& samplers      = program.info.samplers;
	auto& sampled_pairs = program.info.sampled_pairs;
	const auto base_sampler_count = samplers.size();
	samplers.reserve(specialization.samplers.size());
	for (uint32_t index = 0; index < specialization.samplers.size(); ++index) {
		const auto& source = specialization.samplers[index];
		if (index >= samplers.size()) {
			EXIT_IF(source.indirect_root >= base_sampler_count);
			samplers.push_back(samplers[source.indirect_root]);
			samplers.back().indirect_resources.clear();
		}
		samplers[index].indirect_root = source.indirect_root;
		samplers[index].indirect_mapping_offset = source.indirect_mapping_offset;
		samplers[index].indirect_search_iterations = source.indirect_search_iterations;
	}
	for (uint32_t index = 0; index < samplers.size(); ++index) {
		const auto root = samplers[index].indirect_root;
		if (root != SamplerResource::NoIndirectSampler) {
			EXIT_IF(root >= samplers.size());
			samplers[root].indirect_resources.push_back(index);
		}
	}
	SamplerPlan sampler_plan;
	EXIT_IF(!BuildSamplerPlan(program.info, sampler_plan));
	const auto original_sampler_count = samplers.size();
	for (uint32_t index = 0; index < original_sampler_count && index < specialization.samplers.size();
	     index++) {
		samplers[index].bindless                = specialization.samplers[index].bindless;
		samplers[index].bindless_mapping_offset = specialization.samplers[index].bindless_mapping_offset;
	}
	samplers.reserve(sampler_plan.bindings.size());
	for (uint32_t index = 0; index < sampler_plan.bindings.size(); index++) {
		const auto& binding = sampler_plan.bindings[index];
		if (index >= original_sampler_count) {
			samplers.push_back(samplers[binding.source]);
		}
		samplers[index].snapshot_index = binding.source;
		samplers[index].force_point_filtering = binding.type == SamplerClass::Integer;
		samplers[index].integer_border        = binding.type != SamplerClass::Float;
	}
	for (uint32_t index = 0; index < sampler_plan.bindings.size(); index++) {
		const auto& binding = sampler_plan.bindings[index];
		for (auto& candidate: samplers[index].indirect_resources) {
			candidate = sampler_plan.mapping[candidate][static_cast<uint32_t>(binding.type)];
			EXIT_IF(candidate == UINT32_MAX);
		}
	}
	for (auto& pair: sampled_pairs) {
		const auto type = static_cast<uint32_t>(ClassifySampler(images[pair.image]));
		pair.sampler = sampler_plan.mapping[pair.sampler][type];
		EXIT_IF(pair.sampler == UINT32_MAX);
		samplers[pair.sampler].depth_compare |= images[pair.image].depth_compare;
		for (const auto candidate: samplers[pair.sampler].indirect_resources)
			samplers[candidate].depth_compare |= images[pair.image].depth_compare;
	}

	auto& memory_info = program.memory_info;
	// Resolve empty scalar-table words before folding the indirect descriptors that use them.
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::ReadConstBuffer) continue;
			const auto& memory = memory_info[inst.Flags<MemoryFlags>().index];
			if (memory.kind == ResourceKind::ScalarBuffer &&
			    specialization.buffers[memory.resource].scalar_empty &&
			    buffers[memory.resource].indirect_root == BufferResource::NoIndirectBuffer)
				inst.ReplaceUsesWith(Value(0u));
		}
	}
	const ImageRemap image_remap(specialization);
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
			if (inst.GetOpcode() == ValueOpcode::GetImageResource) {
				inst.SetFlags(image_remap[inst.Flags<uint32_t>()]);
				continue;
			}
			if (BufferAccessOf(inst.GetOpcode()) == BufferAccess::Read) {
				const auto& memory = memory_info[inst.Flags<MemoryFlags>().index];
				const bool native_oob = memory.kind == ResourceKind::Buffer &&
				    specialization.buffers[memory.resource].zero_stride_oob &&
				    buffers[memory.resource].indirect_root == BufferResource::NoIndirectBuffer;
				const bool null_indirect = memory.kind == ResourceKind::IndirectBuffer &&
				    NullBufferDescriptor(inst.Arg(0));
				if (native_oob || null_indirect) {
					// Bounds mode 0 checks offset >= stride, so zero stride
					// makes every vector read out of bounds. An all-zero V# is unbound.
					const auto count = BufferComponentCount(inst.GetOpcode());
					std::array<Value, 4> values {Value(0u), Value(0u), Value(0u), Value(0u)};
					if (native_oob && memory.formatted && !memory.typed) {
						const auto& buffer = buffers[memory.resource];
						const auto format = Format::GetFormatInfo(buffer.descriptor_format);
						const bool d16 = memory.data_bits == 16u;
						const auto components = d16 ? memory.component_count : count;
						std::array<uint32_t, 4> constants {};
						for (uint32_t component = 0; component < components; component++) {
							if (format.type == Format::ComponentType::Unknown ||
							    GetDstSel(buffer.descriptor_swizzle, component) != 1u) continue;
							auto one = Format::FormattedConstantBits(
							    format, Format::FormattedSourceKind::One);
							if (d16 && one == 0x3f800000u) one = 0x3c00u;
							constants[d16 ? component / 2u : component] |=
							    one << (d16 ? (component % 2u) * 16u : 0u);
						}
						for (uint32_t component = 0; component < count; component++) {
							if (constants[component] == 0u) continue;
							values[component] = Value(&*block->PrependNewInst(
							    it, ValueOpcode::SelectU32,
							    {inst.Arg(inst.NumArgs() - 1), Value(constants[component]), Value(0u)}));
						}
					}
					Value result = values[0];
					switch (inst.GetType()) {
						case Type::U8: result = Value(uint8_t {0}); break;
						case Type::U16: result = Value(uint16_t {0}); break;
						case Type::U32x2:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x2,
							                                      {values[0], values[1]}));
							break;
						case Type::U32x3:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x3,
							                                      {values[0], values[1], values[2]}));
							break;
						case Type::U32x4:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x4,
							                                      {values[0], values[1], values[2], values[3]}));
							break;
						default: break;
					}
					inst.ReplaceUsesWith(result);
				}
				continue;
			}
			const auto image_opcode = ImageOpcodeInfoOf(inst.GetOpcode());
			if (image_opcode.access == ImageAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			EXIT_IF(index >= memory_info.size());
			auto& memory = memory_info[index];
			EXIT_IF(memory.resource >= images.size());
			const auto& image = images[memory.resource];
			if (specialization.images[memory.resource].fmask) {
				EXIT_IF(inst.GetOpcode() != ValueOpcode::ImageRead || memory.data_bits != 32u);
				// Vulkan MSAA stores each sample directly; FMASK's four-bit fragment indices
				// therefore map each coverage sample to the same host sample.
				constexpr uint32_t indices[] = {0x76543210u, 0xfedcba98u};
				std::array<Value, 2> fragments;
				for (uint32_t component = 0; component < fragments.size(); component++) {
					const auto selected = block->PrependNewInst(
					    it, ValueOpcode::SelectU32, {inst.Arg(2), Value(indices[component]), Value(0u)});
					fragments[component] = Value(&*selected);
				}
				const auto result = block->PrependNewInst(
				    it, ValueOpcode::CompositeConstructU32x4,
				    {fragments[0], fragments[1], Value(0u), Value(0u)});
				inst.ReplaceUsesWith(Value(&*result));
				continue;
			}
			if (image_opcode.needs_sampler &&
			    memory.sampler < original_sampler_count) {
				const auto type = static_cast<uint32_t>(ClassifySampler(image));
				memory.sampler = sampler_plan.mapping[memory.sampler][type];
				EXIT_IF(memory.sampler == UINT32_MAX);
			}
			// Enumerated candidates only sample; a bindless image is loaded per instruction and
			// serves every image operation.
			EXIT_IF(image.indirect_root == memory.resource && !image.bindless &&
			        inst.GetOpcode() != ValueOpcode::ImageSampleRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageGatherRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageQueryDimensions &&
			        inst.GetOpcode() != ValueOpcode::ImageRead &&
			        inst.GetOpcode() != ValueOpcode::ImageWrite);
			EXIT_IF(image.table_capacity != 0u && inst.GetOpcode() != ValueOpcode::ImageSampleRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageQueryDimensions);
		}
	}
	// Tracking gives an image index only to the accesses that survived its dead-code pass; an
	// access it removed keeps its T# register from translation, which need not be an image index.
	std::vector<bool> live_image_access(memory_info.size());
	for (auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (ImageOpcodeInfoOf(inst.GetOpcode()).access != ImageAccess::None) {
				live_image_access[inst.Flags<MemoryFlags>().index] = true;
			}
		}
	}
	for (uint32_t index = 0; index < memory_info.size(); index++) {
		auto& memory = memory_info[index];
		if (live_image_access[index] && memory.kind == ResourceKind::Image && !memory.planning_only) {
			memory.resource = image_remap(memory.resource, "image access");
		}
	}
	for (auto& buffer: buffers) {
		if (buffer.image_alias != BufferResource::NoImageAlias) {
			buffer.image_alias = image_remap(buffer.image_alias, "buffer image alias");
		}
	}
	for (auto& pair: sampled_pairs) {
		pair.image = image_remap(pair.image, "sampled pair");
	}
	for (auto& image: images) {
		if (image.indirect_root != ImageResource::NoIndirectImage) {
			image.indirect_root = image_remap(image.indirect_root, "indirect root");
		}
		for (auto& resource: image.indirect_resources) {
			resource = image_remap(resource, "indirect resource");
		}
	}
	image_remap.Apply(images);
}

// KYTY_WRITE_BITMAP=1: buffers written at addresses the host cannot bound record the pages they
// write in a GPU bitmap, so a guest read of a page the shader did not write needs no download
// (rework.md, Phase 2 first step; compute shader eaa507075716aa3b adds into a 70 MiB range and
// caused half of all readbacks in a new game).
bool WriteBitmapEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_WRITE_BITMAP");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
