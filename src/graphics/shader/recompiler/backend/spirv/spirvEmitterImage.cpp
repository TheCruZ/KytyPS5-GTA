#include "common/hackNotice.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/recompiler/frontend/decode/ImageOps.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t DmaskComponentIndex(uint32_t dmask, uint32_t component) {
	uint32_t index = 0;
	for (uint32_t i = 0; i < component; i++) {
		index += (dmask >> i) & 1u;
	}
	return index;
}

uint32_t DmaskComponent(uint32_t dmask, uint32_t index) {
	for (uint32_t component = 0; component < 4u; component++) {
		if (((dmask >> component) & 1u) != 0u && index-- == 0u) return component;
	}
	return 0u;
}

uint32_t ImageGatherComponent(uint32_t dmask) {
	switch (dmask) {
		case 0x2u: return 1;
		case 0x4u: return 2;
		case 0x8u: return 3;
		default: return 0;
	}
}

bool HasFlag(const IR::MemoryInfo& mem, uint32_t flag) {
	return (mem.image_sample_flags & flag) != 0u;
}

uint32_t AddressU32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                    uint32_t component) {
	const auto layout = Decoder::ImageAddressComponentLayout(mem.image_sample_flags,
	                                                         mem.image_dimension, component);
	const auto packed = layout.bit_offset / 32u;
	if (packed >= address.NumArgs()) return ConstantU32(ctx.state, 0);
	auto value = ctx.Def(address.Arg(packed));
	if (layout.bit_width == 16u) {
		if ((layout.bit_offset & 31u) != 0u) {
			value = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), value,
			               ConstantU32(ctx.state, 16));
		}
		value = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), value,
		               ConstantU32(ctx.state, 0xffffu));
	}
	return value;
}

uint32_t AddressF32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                    uint32_t component) {
	const auto value  = AddressU32(ctx, mem, address, component);
	const auto layout = Decoder::ImageAddressComponentLayout(mem.image_sample_flags,
	                                                         mem.image_dimension, component);
	return layout.bit_width == 16u ? EmitF16BitsToF32(ctx.state, value)
	                               : Unary(ctx.state, spv::OpBitcast, TypeF32(ctx.state), value);
}

ImageSampleLayout Layout(const IR::MemoryInfo& mem) {
	ImageSampleLayout layout;
	uint32_t          cursor = 0;
	const auto&       info   = ImageDimensionInfoFor(mem.image_dimension);
	if (HasFlag(mem, Decoder::ImageSampleFlagOffset)) layout.offset = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagBias)) layout.bias = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagCompare)) layout.dref = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagDerivative)) {
		layout.grad_x = cursor;
		cursor += info.spatial_components;
		layout.grad_y = cursor;
		cursor += info.spatial_components;
	}
	layout.coord = cursor;
	cursor += info.coordinate_components;
	if (HasFlag(mem, Decoder::ImageSampleFlagLod)) layout.lod = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagLodClamp)) layout.clamp = cursor++;
	return layout;
}

uint32_t ZeroF32(EmitterState& state) {
	return ConstantF32(state, 0);
}

uint32_t GatherMip(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                   const IR::Inst& address, const ImageSampleLayout& layout, uint32_t mip_count) {
	auto& state = ctx.state;
	const auto Extract = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first),
		                           ConstantU32(state, count));
	};
	const auto LodFixed = [&](uint32_t word, uint32_t first) {
		return Binary(state, spv::OpFMul, TypeF32(state),
		              Unary(state, spv::OpConvertUToF, TypeF32(state), Extract(word, first, 12)),
		              ConstantF32Value(state, 1.0f / 256.0f));
	};
	const auto sampler_control = ctx.Arg(inst, 6);
	const auto mip_filter = Extract(sampler_control, 26, 2);
	const auto preclamp = Binary(state, spv::OpINotEqual, TypeBool(state),
	                            Extract(sampler_control, 28, 1), ConstantU32(state, 0));
	const auto zero = ZeroF32(state);
	// Point selects the nearest mip; linear gathers return the lower, primary mip only.
	const auto rounding = Select(state, TypeF32(state),
	    Binary(state, spv::OpIEqual, TypeBool(state), mip_filter, ConstantU32(state, 1)),
	    ConstantF32Value(state, 0.5f), zero);
	const auto base = Unary(state, spv::OpConvertUToF, TypeF32(state),
	                        Extract(ctx.Arg(inst, 4), 12, 4));
	const auto minimum = Binary(state, spv::OpFSub, TypeF32(state),
	                            LodFixed(ctx.Arg(inst, 3), 8), base);
	const auto last = ConstantF32Value(state, static_cast<float>(mip_count - 1u));
	auto lod = Binary(state, spv::OpFAdd, TypeF32(state),
	                  AddressF32(ctx, mem, address, layout.lod),
	                  Select(state, TypeF32(state), preclamp, rounding, zero));
	lod = EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(
	    state, lod, LodFixed(ctx.Arg(inst, 5), 0), LodFixed(ctx.Arg(inst, 5), 12));
	// T# minimum LOD is in physical mip space and applies after the S# view-relative clamp.
	lod = EmitGlsl<GLSLstd450FMax, IR::Type::F32>(state, lod, minimum);
	lod = EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(state, lod, zero, last);
	lod = Binary(state, spv::OpFAdd, TypeF32(state), lod,
	             Select(state, TypeF32(state), preclamp, zero, rounding));
	// The clamped value is nonnegative, so conversion performs the integer mip rounding.
	const auto mip = EmitUMin32(state, Unary(state, spv::OpConvertFToU, TypeU32(state), lod),
	                           ConstantU32(state, mip_count - 1u));
	const auto mip_none = Binary(state, spv::OpIEqual, TypeBool(state),
	                            mip_filter, ConstantU32(state, 0));
	return Select(state, TypeU32(state), mip_none, ConstantU32(state, 0), mip);
}

uint32_t CubeAxis(EmitterState& state, uint32_t value) {
	return Binary(state, spv::OpFSub, TypeF32(state), value, ConstantF32(state, 0x3f800000u));
}

uint32_t CubeLayer(EmitterState& state, uint32_t value) {
	const auto guest = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertFToU, TypeU32(state), guest, value);
	const auto padding = Binary(
	    state, spv::OpShiftLeftLogical, TypeU32(state),
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), guest, ConstantU32(state, 3)),
	    ConstantU32(state, 1));
	const auto host   = Binary(state, spv::OpISub, TypeU32(state), guest, padding);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), result, host);
	return result;
}

uint32_t CoordF32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                  uint32_t first, uint32_t components, bool cube = false) {
	auto x = AddressF32(ctx, mem, address, first);
	if (components == 1u) return x;
	auto y = mem.image_address_components > first + 1u ? AddressF32(ctx, mem, address, first + 1u)
	                                                   : ZeroF32(ctx.state);
	if (cube) {
		x = CubeAxis(ctx.state, x);
		y = CubeAxis(ctx.state, y);
	}
	const auto result = ctx.state.builder.AllocateId();
	if (components == 3u) {
		auto z = mem.image_address_components > first + 2u
		             ? AddressF32(ctx, mem, address, first + 2u)
		             : ZeroF32(ctx.state);
		if (cube) z = CubeLayer(ctx.state, z);
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(ctx.state, 3),
		                              result, x, y, z);
	} else {
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(ctx.state, 2),
		                              result, x, y);
	}
	return result;
}

uint32_t CoordU32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                  ImageDimension dimension) {
	const auto components = ImageDimensionInfoFor(dimension).coordinate_components;
	const auto x          = AddressU32(ctx, mem, address, 0);
	if (components == 1u) return x;
	const auto y      = mem.image_address_components > 1u ? AddressU32(ctx, mem, address, 1)
	                                                      : ConstantU32(ctx.state, 0);
	const auto result = ctx.state.builder.AllocateId();
	if (components == 3u) {
		const auto z = mem.image_address_components > 2u ? AddressU32(ctx, mem, address, 2)
		                                                 : ConstantU32(ctx.state, 0);
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 3),
		                              result, x, y, z);
	} else {
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 2),
		                              result, x, y);
	}
	return result;
}

uint32_t LodU32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                ImageDimension dimension) {
	const auto component = ImageDimensionInfoFor(dimension).coordinate_components;
	return mem.image_has_mip && mem.image_address_components > component
	           ? AddressU32(ctx, mem, address, component)
	           : ConstantU32(ctx.state, 0);
}

uint32_t FloatBits(ValueEmitContext& ctx, uint32_t value) {
	return Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), value);
}

uint32_t SampledComponentBits(ValueEmitContext& ctx, uint32_t value,
                              Prospero::TextureNumericClass numeric_class) {
	if (numeric_class == Prospero::TextureNumericClass::Uint) {
		return value;
	}
	return Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), value);
}

uint32_t SampledComponentZero(EmitterState& state, Prospero::TextureNumericClass numeric_class) {
	switch (numeric_class) {
		case Prospero::TextureNumericClass::Float: return ZeroF32(state);
		case Prospero::TextureNumericClass::Uint: return ConstantU32(state, 0);
		case Prospero::TextureNumericClass::Sint: return ConstantI32(state, 0);
		case Prospero::TextureNumericClass::Unsupported: break;
	}
	EXIT("invalid sampled image numeric class");
}

uint32_t ResultVector(ValueEmitContext& ctx, uint32_t value,
                      Prospero::TextureNumericClass numeric_class, bool dref,
                      const IR::MemoryInfo& mem, bool gather = false) {
	auto value_class = numeric_class;
	const auto conversion = ImageResourceAt(ctx.state, mem.resource).conversion_format;
	if (conversion != Prospero::BufferFormat::kInvalid) {
		value_class = Prospero::SampledTextureNumericClass(conversion);
	}
	if (dref) {
		value_class = Prospero::TextureNumericClass::Float;
	}
	const bool integer = value_class == Prospero::TextureNumericClass::Uint ||
	                     value_class == Prospero::TextureNumericClass::Sint;
	if (mem.data_bits == 16u) {
		uint32_t   packed[4] = {ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0),
		                        ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0)};
		const auto scalar    = [&](uint32_t index) {
			if (dref) return value;
			const auto component = gather ? index : DmaskComponent(mem.dmask, index);
			const auto result    = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract,
			                              ImageScalarType(ctx.state, value_class), result, value,
			                              component);
			return result;
		};
		for (uint32_t word = 0; word < mem.data_dwords; word++) {
			const auto low_index  = word * 2u;
			const auto high_index = low_index + 1u;
			const auto low        = scalar(low_index);
			const auto high       = high_index < mem.component_count
			                            ? scalar(high_index)
			                            : SampledComponentZero(ctx.state, value_class);
			if (integer) {
				const auto low_bits  = SampledComponentBits(ctx, low, value_class);
				const auto high_bits = SampledComponentBits(ctx, high, value_class);
				const auto mask      = ConstantU32(ctx.state, 0xffffu);
				packed[word] =
				    Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state),
				           Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), low_bits, mask),
				           Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
				                  Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state),
				                         high_bits, mask),
				                  ConstantU32(ctx.state, 16u)));
			} else {
				const auto pair = ctx.state.builder.AllocateId();
				ctx.state.builder.AddFunction(spv::OpCompositeConstruct,
				                              TypeF32Vector(ctx.state, 2), pair, low, high);
				packed[word] = ctx.state.builder.AllocateId();
				ctx.state.builder.AddFunction(spv::OpExtInst, TypeU32(ctx.state), packed[word],
				                              GlslStd450(ctx.state), GLSLstd450PackHalf2x16, pair);
			}
		}
		const auto result = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4),
		                              result, packed[0], packed[1], packed[2], packed[3]);
		return result;
	}
	if (!dref)
		return value_class == Prospero::TextureNumericClass::Uint
		           ? value : Unary(ctx.state, spv::OpBitcast, TypeU32Vector(ctx.state, 4), value);
	const auto result = ctx.state.builder.AllocateId();
	const auto zero = ConstantU32(ctx.state, 0);
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4), result,
	                              FloatBits(ctx, value), zero, zero, zero);
	return result;
}

// The descriptor of a bindless table image the instruction selects: entry = key & mask;
// flattened_srt[mapping] holds the entry count, then each entry's slot among the table
// descriptors of the root's binding.
uint32_t ImageTableElement(ValueEmitContext& ctx, const IR::Inst& inst,
                           const IR::ImageResource& image) {
	auto&       state  = ctx.state;
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (handle == nullptr || handle->NumArgs() == 0u || state.flattened_srt_variable == 0 ||
	    image.table_descriptor_base == UINT32_MAX) {
		ctx.Fail(inst, "has no image table runtime mapping");
	}
	const auto LoadTableMapping = [&](uint32_t index) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, state.flattened_srt_variable, ConstantU32(state, 0),
		                          index);
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	};
	const auto entry    = Binary(state, spv::OpBitwiseAnd, TypeU32(state), ctx.Def(handle->Arg(0)),
	                             ConstantU32(state, image.table_entry_mask));
	const auto count    = LoadTableMapping(ConstantU32(state, image.table_mapping_offset));
	const auto in_range = Binary(state, spv::OpULessThan, TypeBool(state), entry, count);
	const auto clamped  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), clamped, in_range, entry,
	                          ConstantU32(state, 0));
	const auto mapped =
	    LoadTableMapping(Binary(state, spv::OpIAdd, TypeU32(state), clamped,
	                            ConstantU32(state, image.table_mapping_offset + 1u)));
	const auto slot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), slot, in_range, mapped,
	                          ConstantU32(state, 0));
	return Binary(state, spv::OpIAdd, TypeU32(state), slot,
	              ConstantU32(state, image.table_descriptor_base));
}

uint32_t QueryDimensions(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                         const IR::Inst& address, uint32_t resource, uint32_t array_index) {
	const auto  dimension = ImageResourceAt(ctx.state, resource).dimension;
	const auto& info      = ImageDimensionInfoFor(dimension);
	const auto  image     = LoadImageDescriptor(ctx.state, resource, 0u, array_index);
	const auto  size      = ctx.state.builder.AllocateId();
	if (info.multisampled != 0u) {
		ctx.state.builder.AddFunction(spv::OpImageQuerySize,
		                              ImageViewSizeType(ctx.state, dimension), size, image);
	} else {
		ctx.state.builder.AddFunction(spv::OpImageQuerySizeLod,
		                              ImageViewSizeType(ctx.state, dimension), size, image,
		                              AddressU32(ctx, mem, address, 0));
	}
	const auto components = info.coordinate_components;
	uint32_t   result[4]  = {ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0),
	                         ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0)};
	if (components == 1u) {
		result[0] = size;
	} else {
		for (uint32_t index = 0; index < components; index++) {
			result[index] = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state),
			                              result[index], size, index);
		}
	}
	if (info.multisampled == 0u) {
		result[3] = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpImageQueryLevels, TypeU32(ctx.state), result[3],
		                              image);
	}
	const auto vector = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4), vector,
	                              result[0], result[1], result[2], result[3]);
	return vector;
}

uint32_t PackedOffset(ValueEmitContext& ctx, IR::Value value, ImageDimension dimension,
                      uint32_t& mask) {
	const auto components = ImageDimensionInfoFor(dimension).spatial_components;
	uint32_t values[3] {};
	if (value.IsImmediate()) {
		// RDNA2 packs signed six-bit texel offsets into the low bits of each byte.
		for (uint32_t index = 0; index < components; index++) {
			const auto bits = (value.U32() >> (index * 8u)) & 0x3fu;
			values[index] = ConstantI32(ctx.state, static_cast<int32_t>(bits ^ 0x20u) - 0x20);
		}
		mask |= spv::ImageOperandsConstOffsetMask;
		return components == 1u
		           ? values[0]
		           : ctx.state.builder.Constant(spv::OpConstantComposite,
		                                        TypeI32Vector(ctx.state, components),
		                                        std::span(values).first(components));
	}
	ctx.state.builder.RequireCapability(spv::CapabilityImageGatherExtended);
	mask |= spv::ImageOperandsOffsetMask;
	const auto packed = Unary(ctx.state, spv::OpBitcast, TypeI32(ctx.state), ctx.Def(value));
	for (uint32_t index = 0; index < components; index++) {
		values[index] = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(ctx.state), values[index],
		                              packed, ConstantU32(ctx.state, index * 8u),
		                              ConstantU32(ctx.state, 6));
	}
	if (components == 1u) return values[0];
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeI32Vector(ctx.state, components),
	                              result, std::span(values).first(components));
	return result;
}

// IMAGE_SAMPLE*_O offsets: x in bits 5:0, y in 13:8 and z in 21:16, six-bit signed.
int32_t SignExtend6(uint32_t value) {
	return static_cast<int32_t>((value & 0x3fu) ^ 0x20u) - 0x20;
}

// Moves `coord` by the packed texel offsets divided by the size of the level `level_f32`
// (relative to the view's base, clamped to the view's levels). Exact for single-level reads up
// to float rounding; with linear mip filtering the coarser level moves by half the offset.
uint32_t ApplyDynamicSampleOffset(EmitterState& state, uint32_t resource, uint32_t coord,
                                  ImageDimension dimension, uint32_t packed, uint32_t level_f32,
                                  uint32_t array_index = 0u) {
	Common::HackNotice::ReportTranslation(
	    "IMAGE_SAMPLE*_O offset approximated as a coordinate shift", "SPIR-V image sample",
	    state.program.shader_hash, 0,
	    "a texel offset that is not constant or lies outside -8..7 moves the coordinate by offset / level size");
	state.builder.RequireCapability(spv::CapabilityImageQuery);
	const auto& info       = ImageDimensionInfoFor(dimension);
	const auto  components = info.spatial_components;
	const auto  image      = LoadImageDescriptor(state, resource, 0u, array_index);
	const auto  levels     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQueryLevels, TypeU32(state), levels, image);
	const auto floored = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), floored, GlslStd450(state),
	                          GLSLstd450FMax, level_f32, ConstantF32Value(state, 0.0f));
	const auto level_float = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), level_float, GlslStd450(state),
	                          GLSLstd450Floor, floored);
	const auto level_u32 = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertFToU, TypeU32(state), level_u32, level_float);
	const auto last  = Binary(state, spv::OpISub, TypeU32(state), levels, ConstantU32(state, 1));
	const auto level = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeU32(state), level, GlslStd450(state),
	                          GLSLstd450UMin, level_u32, last);
	const auto size = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQuerySizeLod, ImageViewSizeType(state, dimension), size,
	                          image, level);
	const auto packed_i32 = Unary(state, spv::OpBitcast, TypeI32(state), packed);
	uint32_t   moved[4]   = {};
	for (uint32_t index = 0; index < info.coordinate_components; index++) {
		auto component = coord;
		if (info.coordinate_components > 1u) {
			component = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, coord,
			                          index);
		}
		if (index < components) {
			const auto offset = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(state), offset, packed_i32,
			                          ConstantU32(state, index * 8u), ConstantU32(state, 6));
			const auto offset_f32 = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertSToF, TypeF32(state), offset_f32, offset);
			auto extent = size;
			if (info.coordinate_components > 1u) {
				extent = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), extent, size,
				                          index);
			}
			const auto extent_f32 = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), extent_f32, extent);
			component = Binary(state, spv::OpFAdd, TypeF32(state), component,
			                   Binary(state, spv::OpFDiv, TypeF32(state), offset_f32, extent_f32));
		}
		moved[index] = component;
	}
	if (info.coordinate_components == 1u) {
		return moved[0];
	}
	const auto result = state.builder.AllocateId();
	if (info.coordinate_components == 2u) {
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), result,
		                          moved[0], moved[1]);
	} else {
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 3), result,
		                          moved[0], moved[1], moved[2]);
	}
	return result;
}

uint32_t HorizontalOffsets(EmitterState& state, ImageDimension dimension) {
	const auto components   = ImageDimensionInfoFor(dimension).spatial_components;
	const auto count        = ConstantU32(state, 4);
	const auto element_type = components == 1u ? TypeI32(state) : TypeI32Vector(state, 2);
	const auto array_type   = state.builder.Type(spv::OpTypeArray, element_type, count);
	uint32_t   offsets[4] {};
	for (uint32_t index = 0; index < 4u; index++) {
		const auto x = ConstantI32(state, static_cast<int32_t>(index) - 1);
		if (components == 1u) {
			offsets[index] = x;
		} else {
			offsets[index] = state.builder.Constant(
			    spv::OpConstantComposite, TypeI32Vector(state, 2), x, ConstantI32(state, 0));
		}
	}
	return state.builder.Constant(spv::OpConstantComposite, array_type, offsets[0], offsets[1],
	                              offsets[2], offsets[3]);
}

uint32_t InverseSwizzle(uint32_t swizzle, uint32_t component) {
	for (uint32_t source = 0; source < 4u; source++) {
		if (((swizzle >> (source * 3u)) & 7u) == 4u + component) return source;
	}
	return UINT32_MAX;
}

Format::BufferFormatInfo ImageConversionFormat(const EmitterState&   state,
                                               const IR::MemoryInfo& mem) {
	const auto& image = ImageResourceAt(state, mem.resource);
	const auto format = image.conversion_format;
	if (format == Prospero::BufferFormat::kInvalid) return {};
	const auto info = Format::GetFormatInfo(format);
	if (IR::ReadsDepthBits(image)) return info;
	const bool raw_signed = IR::RawSignedStorageFormat(image, format) != Prospero::BufferFormat::kInvalid;
	EXIT_IF((Prospero::RemapTextureFormat(format) == format && !raw_signed) || info.component_count == 0u ||
	        info.component_count > 4u);
	EXIT_IF(!raw_signed && info.type != Format::ComponentType::Uscaled &&
	        info.type != Format::ComponentType::Sscaled &&
	        ((info.type != Format::ComponentType::Uint &&
	          info.type != Format::ComponentType::Unorm) || !info.packed_bitfield ||
	         info.byte_size != sizeof(uint32_t)));
	return info;
}

uint32_t ExtractPackedImageComponent(EmitterState& state, const Format::BufferFormatInfo& info,
                                     uint32_t packed, uint32_t type, uint32_t component) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitFieldUExtract, type, result, packed,
	                          ConstantU32(state, info.component_bit_offset[component]),
	                          ConstantU32(state, info.component_bits[component]));
	return result;
}

uint32_t ImageGatherSource(const EmitterState& state, const IR::MemoryInfo& mem) {
	const auto component = ImageGatherComponent(mem.dmask);
	const auto info      = ImageConversionFormat(state, mem);
	if (info.format == Prospero::BufferFormat::kInvalid) return component;
	if (info.packed_bitfield) return 0u;
	const auto selector =
	    (ImageResourceAt(state, mem.resource).shader_swizzle >> (component * 3u)) & 7u;
	return selector >= 4u ? (selector - 4u) % info.component_count : 0u;
}

// A *SCALED image is backed by the UNorm/SNorm format of the same width; this is the factor
// that turns its normalized texel back into the integer the guest sees (as a float).
float ScaledTexelFactor(const Format::BufferFormatInfo& info) {
	if (info.type != Format::ComponentType::Uscaled && info.type != Format::ComponentType::Sscaled) {
		return 255.0f;
	}
	const auto bits = info.component_bits[0];
	return static_cast<float>(info.type == Format::ComponentType::Sscaled ? (1u << (bits - 1u)) - 1u
	                                                                      : (1u << bits) - 1u);
}

// `exact` is for texel fetches, which return the stored integer exactly after rounding the
// product; filtered samples keep the interpolated value. 8-bit UScaled predates this and keeps
// its unrounded product.
uint32_t UnpackImageTexel(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t texel,
                          bool exact = false) {
	const auto info = ImageConversionFormat(ctx.state, mem);
	if (info.format == Prospero::BufferFormat::kInvalid) return texel;

	const auto numeric_class = ImageResourceAt(ctx.state, mem.resource).numeric_class;
	const auto scalar_type   = ImageScalarType(ctx.state, numeric_class);
	const bool integer       = info.packed_bitfield || info.type == Format::ComponentType::Uint;
	uint32_t   packed        = 0;
	if (info.packed_bitfield) {
		packed = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeExtract, scalar_type, packed, texel, 0u);
	}
	uint32_t components[4] {};
	for (uint32_t component = 0; component < info.component_count; component++) {
		if (info.packed_bitfield) {
			components[component] =
			    ExtractPackedImageComponent(ctx.state, info, packed, scalar_type, component);
			components[component] =
			    NormalizeFormatComponent(ctx.state, info, component, components[component]);
		} else {
			components[component] = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, scalar_type,
			                              components[component], texel, component);
			// UNorm/SNorm backing preserves the guest's filtering; scaling precedes swizzle
			// constants.
			const auto scale = info.type == Format::ComponentType::Sscaled
			                       ? ScaledTexelFactor(info)
			                       : static_cast<float>((1u << info.component_bits[component]) - 1u);
			components[component] = Binary(ctx.state, spv::OpFMul, scalar_type, components[component],
			                               ConstantF32Value(ctx.state, scale));
			if (integer) {
				// Depth bits: round back to the stored UNorm code.
				components[component] =
				    Unary(ctx.state, spv::OpConvertFToU, TypeU32(ctx.state),
				          Binary(ctx.state, spv::OpFAdd, scalar_type, components[component],
				                 ConstantF32Value(ctx.state, 0.5f)));
			} else if (exact && (info.type == Format::ComponentType::Sscaled ||
			                     (info.type == Format::ComponentType::Uscaled &&
			                      info.component_bits[0] > 8u))) {
				const auto rounded = ctx.state.builder.AllocateId();
				ctx.state.builder.AddFunction(spv::OpExtInst, scalar_type, rounded,
				                              GlslStd450(ctx.state), GLSLstd450RoundEven,
				                              components[component]);
				components[component] = rounded;
			}
		}
	}
	for (uint32_t component = info.component_count; component < 4u; component++) {
		components[component] = components[component % info.component_count];
	}

	const auto swizzle = ImageResourceAt(ctx.state, mem.resource).shader_swizzle;
	uint32_t   selected[4] {};
	for (uint32_t component = 0; component < 4u; component++) {
		const auto selector = (swizzle >> (component * 3u)) & 7u;
		if (selector == 1u) {
			selected[component] = integer
			    ? ConstantU32(ctx.state, Format::FormattedConstantBits(info, Format::FormattedSourceKind::One))
			    : ConstantF32Value(ctx.state, 1.0f);
		} else if (selector >= 4u) {
			selected[component] = components[selector - 4u];
		} else {
			selected[component] = integer ? ConstantU32(ctx.state, 0u) : ZeroF32(ctx.state);
		}
	}
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct,
	                              integer ? TypeU32Vector(ctx.state, 4) : TypeF32Vector(ctx.state, 4),
	                              result, selected[0], selected[1], selected[2], selected[3]);
	return info.type == Format::ComponentType::Unorm
	           ? Unary(ctx.state, spv::OpBitcast, TypeF32Vector(ctx.state, 4), result) : result;
}

uint32_t UnpackImageGather(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t gathered) {
	const auto info = ImageConversionFormat(ctx.state, mem);
	if (info.format == Prospero::BufferFormat::kInvalid) return gathered;

	const auto component = ImageGatherComponent(mem.dmask);
	const auto selector =
	    (ImageResourceAt(ctx.state, mem.resource).shader_swizzle >> (component * 3u)) & 7u;
	const auto numeric_class = ImageResourceAt(ctx.state, mem.resource).numeric_class;
	const auto vector_type   = ImageVectorType(ctx.state, numeric_class, 4);
	if (selector < 4u) {
		uint32_t value;
		switch (info.type) {
			case Format::ComponentType::Uscaled:
			case Format::ComponentType::Sscaled:
				value = ConstantF32Value(ctx.state, selector == 1u ? 1.0f : 0.0f);
				break;
			default: value = ConstantU32(ctx.state, selector == 1u ? 1u : 0u); break;
		}
		return ctx.state.builder.Constant(spv::OpConstantComposite, vector_type, value, value,
		                                  value, value);
	}
	if (!info.packed_bitfield) {
		const auto scale  = ConstantF32Value(ctx.state, ScaledTexelFactor(info));
		const auto scales = ctx.state.builder.Constant(spv::OpConstantComposite, vector_type,
		                                                scale, scale, scale, scale);
		return Binary(ctx.state, spv::OpFMul, vector_type, gathered, scales);
	}

	return ExtractPackedImageComponent(ctx.state, info, gathered, TypeU32Vector(ctx.state, 4),
	                                   (selector - 4u) % info.component_count);
}

uint32_t SamplePackedUnormLz(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t coord) {
	auto& state = ctx.state;
	const auto& image = ImageResourceAt(state, mem.resource);
	const auto info = ImageConversionFormat(state, mem);
	const auto sampled = MakeSampledImage(state, mem.resource,
	                                     LoadSamplerDescriptor(state, mem.sampler));
	const auto gathered = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageGather, TypeU32Vector(state, 4), gathered, sampled,
	                          coord, ConstantU32(state, 0u));
	state.builder.RequireCapability(spv::CapabilityImageQuery);
	const auto size = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQuerySizeLod, TypeU32Vector(state, 2), size,
	                          LoadImageDescriptor(state, mem.resource), ConstantU32(state, 0u));
	const auto size_f32 = Unary(state, spv::OpConvertUToF, TypeF32Vector(state, 2), size);
	const auto half = ConstantF32Value(state, 0.5f);
	const auto halves = state.builder.Constant(spv::OpConstantComposite,
	                                           TypeF32Vector(state, 2), half, half);
	const auto fractions = state.builder.AllocateId();
	state.builder.AddFunction(
	    spv::OpExtInst, TypeF32Vector(state, 2), fractions, GlslStd450(state), GLSLstd450Fract,
	    Binary(state, spv::OpFSub, TypeF32Vector(state, 2),
	           Binary(state, spv::OpFMul, TypeF32Vector(state, 2), coord, size_f32), halves));
	uint32_t fraction[2];
	uint32_t complement[2];
	for (uint32_t axis = 0; axis < 2u; axis++) {
		fraction[axis] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), fraction[axis],
		                          fractions, axis);
		complement[axis] = Binary(state, spv::OpFSub, TypeF32(state),
		                          ConstantF32Value(state, 1.0f), fraction[axis]);
	}
	// Gather order is (x0, y1), (x1, y1), (x1, y0), (x0, y0).
	const auto weights = state.builder.AllocateId();
	state.builder.AddFunction(
	    spv::OpCompositeConstruct, TypeF32Vector(state, 4), weights,
	    Binary(state, spv::OpFMul, TypeF32(state), complement[0], fraction[1]),
	    Binary(state, spv::OpFMul, TypeF32(state), fraction[0], fraction[1]),
	    Binary(state, spv::OpFMul, TypeF32(state), fraction[0], complement[1]),
	    Binary(state, spv::OpFMul, TypeF32(state), complement[0], complement[1]));
	uint32_t decoded[4] {};
	uint32_t selected[4] {ZeroF32(state), ZeroF32(state), ZeroF32(state), ZeroF32(state)};
	for (uint32_t component = 0; component < 4u; component++) {
		if ((mem.dmask & (1u << component)) == 0u) continue;
		const auto selector = (image.shader_swizzle >> (component * 3u)) & 7u;
		if (selector == 1u) {
			selected[component] = ConstantF32Value(state, 1.0f);
		} else if (selector >= 4u) {
			const auto physical = (selector - 4u) % info.component_count;
			if (decoded[physical] == 0u) {
				const auto texels = Unary(
				    state, spv::OpConvertUToF, TypeF32Vector(state, 4),
				    ExtractPackedImageComponent(state, info, gathered,
				                                TypeU32Vector(state, 4), physical));
				decoded[physical] = Binary(
				    state, spv::OpFMul, TypeF32(state),
				    Binary(state, spv::OpDot, TypeF32(state), texels, weights),
				    ConstantF32Value(state, 1.0f / ((1u << info.component_bits[physical]) - 1u)));
			}
			selected[component] = decoded[physical];
		}
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), result,
	                          selected[0], selected[1], selected[2], selected[3]);
	return result;
}

uint32_t EmitOneDimensionalGatherLz(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                    uint32_t coord, Prospero::TextureNumericClass numeric_class) {
	auto& state = ctx.state;
	state.builder.RequireCapability(spv::CapabilityImageQuery);
	const auto image = LoadImageDescriptor(state, mem.resource);
	const auto width = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQuerySizeLod, TypeU32(state), width, image,
	                          ConstantU32(state, 0));
	const auto width_f32 = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), width_f32, width);
	const auto left = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), left, GlslStd450(state),
	                          GLSLstd450Floor,
	                          Binary(state, spv::OpFSub, TypeF32(state),
	                                 Binary(state, spv::OpFMul, TypeF32(state), coord, width_f32),
	                                 ConstantF32(state, 0x3f000000u)));

	const auto sampled = MakeSampledImage(state, mem.resource,
	                                     LoadSamplerDescriptor(state, mem.sampler));
	const auto vector_type = ImageVectorType(state, numeric_class, 4);
	const auto scalar_type = ImageScalarType(state, numeric_class);
	const auto component = ImageGatherSource(state, mem);
	uint32_t values[2] {};
	for (uint32_t index = 0; index < 2u; index++) {
		const auto sample_coord =
		    Binary(state, spv::OpFDiv, TypeF32(state),
		           Binary(state, spv::OpFAdd, TypeF32(state), left,
		                  ConstantF32(state, index == 0u ? 0x3f000000u : 0x3fc00000u)),
		           width_f32);
		const auto texel = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpImageSampleExplicitLod, vector_type, texel, sampled,
		                          sample_coord, spv::ImageOperandsLodMask, ZeroF32(state));
		values[index] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, scalar_type, values[index], texel,
		                          component);
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, vector_type, result, values[0], values[1],
	                          values[1], values[0]);
	return result;
}

uint32_t StoreTexel(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t data, bool integer) {
	const auto swizzle = ImageResourceAt(ctx.state, mem.resource).shader_swizzle;
	const auto info = ImageConversionFormat(ctx.state, mem);
	const bool converted = info.format != Prospero::BufferFormat::kInvalid;
	const bool packed = info.packed_bitfield;
	EXIT_IF(converted && info.type != Format::ComponentType::Sint &&
	        (!packed || (info.type != Format::ComponentType::Uint &&
	                     info.type != Format::ComponentType::Unorm)));
	const auto zero = ConstantU32(ctx.state, 0);
	uint32_t values[4] {zero, zero, zero, zero};
	auto packed_value = zero;
	const auto dmask = mem.dmask != 0u ? mem.dmask : 1u;
	for (uint32_t component = 0; component < (converted ? info.component_count : 4u); component++) {
		const auto source = InverseSwizzle(swizzle, component);
		const bool selected = source < 4u && ((dmask >> source) & 1u) != 0u;
		uint32_t raw = zero;
		if (selected) {
			const auto packed_index = DmaskComponentIndex(dmask, source);
			raw                     = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), raw, data,
			                              mem.data_bits == 16u ? packed_index / 2u : packed_index);
			if (mem.data_bits == 16u) {
				if ((packed_index & 1u) != 0u) {
					raw = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), raw,
					             ConstantU32(ctx.state, 16u));
				}
				if (!packed) {
					raw = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), raw,
					             ConstantU32(ctx.state, 0xffffu));
				}
			}
		}
		if (converted) {
			if (!selected) continue;
			if (!packed) {
				values[component] =
				    EncodeFormattedStoreComponent(ctx.state, info, component, raw, mem.data_bits);
				continue;
			}
			const auto value =
			    EncodePackedColorComponent(ctx.state, info, component, raw, mem.data_bits);
			packed_value = packed_value == zero ? value
			    : Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state), packed_value, value);
			continue;
		}
		values[component] = integer ? raw
		                    : mem.data_bits == 16u
		                        ? EmitF16BitsToF32(ctx.state, raw)
		                        : Unary(ctx.state, spv::OpBitcast, TypeF32(ctx.state), raw);
	}
	if (packed) values[0] = packed_value;
	const auto texel = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct,
	                              integer ? TypeU32Vector(ctx.state, 4)
	                                      : TypeF32Vector(ctx.state, 4),
	                              texel, values[0], values[1], values[2], values[3]);
	return texel;
}

template <typename Emit>
auto EmitImageAccess(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t result_type,
                     const Emit& emit) {
	auto& state = ctx.state;
	const auto& mem = ctx.Memory(inst);
	const auto& image = ImageResourceAt(state, mem.resource);
	// A bindless image is its own root but selects its texture through the bindless table.
	if (image.indirect_root != mem.resource || image.bindless) return emit(mem.resource, 0u);
	const auto* handle = inst.Arg(0).ResolveInstruction();
	const auto* source = image.source < state.program.descriptor_sources.size()
	                         ? &state.program.descriptor_sources[image.source]
	                         : nullptr;
	if (handle == nullptr || source == nullptr || !source->indirect_descriptor.has_value() ||
	    handle->NumArgs() == 0u) {
		ctx.Fail(inst, "has invalid indirect image key provenance");
	}
	const auto key = ctx.Def(handle->Arg(0));
	if (state.flattened_srt_variable == 0 ||
	    image.indirect_resources.size() < 2u) {
		ctx.Fail(inst, "has no indirect image runtime mapping");
	}
	if (image.heap) {
		const auto entry = EmitImageHeapEntry(state, key, image.indirect_mapping_offset);
		const auto kinds = static_cast<uint32_t>(image.indirect_resources.size()) - 1u;
		// One kind leaves every slot word a bare element.
		if (kinds == 1u) return emit(image.indirect_resources[1], entry);
		const auto element = Binary(state, spv::OpBitwiseAnd, TypeU32(state), entry,
		                            ConstantU32(state, (1u << IR::ImageHeapKindShift) - 1u));
		const auto EmitKind = [&](uint32_t kind) {
			return emit(image.indirect_resources[kind + 1u], element);
		};
		return EmitIndexSwitch(state,
		                       Binary(state, spv::OpShiftRightLogical, TypeU32(state), entry,
		                              ConstantU32(state, IR::ImageHeapKindShift)),
		                       kinds, result_type, EmitKind);
	}
	const auto selected = EmitIndirectResourceIndex(
	    state, key, image.indirect_mapping_offset, image.indirect_search_iterations, 0u);
	const auto SameType = [&](uint32_t lhs, uint32_t rhs) {
		const auto& first = ImageResourceAt(state, lhs);
		const auto& second = ImageResourceAt(state, rhs);
		return first.dimension == second.dimension && first.cube == second.cube &&
		       IR::DescriptorBindingForImage(first) == IR::DescriptorBindingForImage(second) &&
		       (inst.GetOpcode() != IR::ValueOpcode::ImageWrite ||
		        first.shader_swizzle == second.shader_swizzle) &&
		       first.mip_count == 1u && second.mip_count == 1u;
	};
	// Descriptor order is unrelated to image type. Share each typed access and
	// select its descriptor slot independently, including gaps between native roots.
	struct ImageGroup { uint32_t resource; uint32_t count; };
	std::vector<ImageGroup> groups;
	std::vector<uint32_t> ordinal_groups;
	for (const auto resource: image.indirect_resources) {
		const auto found = std::ranges::find_if(groups, [&](const ImageGroup& group) {
			return SameType(resource, group.resource);
		});
		ordinal_groups.push_back(static_cast<uint32_t>(found - groups.begin()));
		if (found == groups.end()) groups.push_back({resource, 1u});
		else ++found->count;
	}
	const bool dynamic_slot = std::ranges::any_of(groups, [](const ImageGroup& group) {
		return group.count > 1u;
	});
	auto group_index = ConstantU32(state, 0u);
	auto slot_bias = ConstantU32(state, image.descriptor_index);
	auto previous_bias = image.descriptor_index;
	for (uint32_t ordinal = 1; ordinal < image.indirect_resources.size(); ++ordinal) {
		const auto& candidate = ImageResourceAt(state, image.indirect_resources[ordinal]);
		const auto bias = candidate.descriptor_index - ordinal;
		const bool select_group = ordinal_groups[ordinal] != ordinal_groups[ordinal - 1u];
		const bool select_slot = dynamic_slot && bias != previous_bias;
		previous_bias = bias;
		if (!select_group && !select_slot) continue;
		const auto at_or_after = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state),
		                               selected, ConstantU32(state, ordinal));
		if (select_group) group_index = Select(state, TypeU32(state), at_or_after,
		                                      ConstantU32(state, ordinal_groups[ordinal]), group_index);
		if (select_slot) slot_bias = Select(state, TypeU32(state), at_or_after,
		                                   ConstantU32(state, bias), slot_bias);
	}
	const auto slot = dynamic_slot
	    ? Binary(state, spv::OpIAdd, TypeU32(state), selected, slot_bias) : 0u;
	const auto EmitGroup = [&](uint32_t index) {
		const auto& group = groups[index];
		return emit(group.resource, group.count == 1u ? 0u : slot);
	};
	return groups.size() == 1u
	           ? EmitGroup(0u)
	           : EmitIndexSwitch(state, group_index, static_cast<uint32_t>(groups.size()),
	                             result_type, EmitGroup);
}

template <typename Emit>
uint32_t EmitImageResult(ValueEmitContext& ctx, const IR::Inst& inst, const Emit& emit) {
	auto& state = ctx.state;
	const auto& mem = ctx.Memory(inst);
	const auto& image = ImageResourceAt(state, mem.resource);
	const bool gather = inst.GetOpcode() == IR::ValueOpcode::ImageGatherRaw;
	const bool dref = HasFlag(mem, Decoder::ImageSampleFlagCompare);
	const bool scalar = dref && !gather;
	const bool mixed = !image.bindless && std::ranges::any_of(image.indirect_resources, [&](uint32_t resource) {
		return ImageResourceAt(state, resource).numeric_class != image.numeric_class;
	});
	const auto native_type = [&](uint32_t resource) {
		return scalar ? TypeF32(state) : ImageVectorType(
		    state, dref ? Prospero::TextureNumericClass::Float
		                : ImageResourceAt(state, resource).numeric_class, 4);
	};
	const auto result_bits = [&](uint32_t resource, uint32_t value) {
		auto candidate_mem = mem;
		candidate_mem.resource = resource;
		if (!scalar) {
			value = gather ? UnpackImageGather(ctx, candidate_mem, value)
			               : UnpackImageTexel(ctx, candidate_mem, value,
			                                  inst.GetOpcode() == IR::ValueOpcode::ImageRead);
		}
		return ResultVector(ctx, value,
		                    dref ? Prospero::TextureNumericClass::Float
		                         : ImageResourceAt(state, resource).numeric_class,
		                    scalar, candidate_mem, gather);
	};
	// RDNA returns register bits. Convert per typed run only when the join mixes
	// numeric classes; homogeneous accesses keep their single conversion after it.
	const auto value = EmitImageAccess(
	    ctx, inst, mixed ? TypeU32Vector(state, 4) : native_type(mem.resource),
	    [&](uint32_t resource, uint32_t array_index) {
		    const auto native = emit(resource, array_index, native_type(resource));
		    return mixed ? result_bits(resource, native) : native;
	    });
	return mixed ? value : result_bits(mem.resource, value);
}

// Ordinals outside the table select candidates[0]. Past ordinal 0, a consecutive table is an
// offset of ordinal - 1, which wraps at 0 so that one unsigned compare checks both ends.
uint32_t SamplerArrayIndex(EmitterState& state, uint32_t ordinal,
                           const std::vector<uint32_t>& candidates) {
	const auto count = static_cast<uint32_t>(candidates.size());
	const auto gap = std::adjacent_find(candidates.begin() + 1, candidates.end(),
	    [](uint32_t lhs, uint32_t rhs) { return rhs != lhs + 1u; });
	if (count == 1u || gap != candidates.end()) {
		return EmitIndexSwitch(state, ordinal, count, TypeU32(state),
		                       [&](uint32_t item) { return ConstantU32(state, candidates[item]); });
	}
	const auto offset = Binary(state, spv::OpISub, TypeU32(state), ordinal, ConstantU32(state, 1u));
	const auto in_table = Binary(state, spv::OpULessThan, TypeBool(state), offset,
	                             ConstantU32(state, count - 1u));
	const auto index = Binary(state, spv::OpIAdd, TypeU32(state), offset,
	                          ConstantU32(state, candidates[1]));
	return Select(state, TypeU32(state), in_table, index, ConstantU32(state, candidates[0]));
}

struct ImageSampler {
	uint32_t id = 0;
	bool dynamic = false;
};

std::array<ImageSampler, 2> LoadImageSamplers(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	const auto& mem = ctx.Memory(inst);
	const auto& samplers = state.program.info.samplers;
	const auto& root = samplers.at(mem.sampler);
	std::array<ImageSampler, 2> result {};
	const auto load = [&](const IR::ImageResource& image) {
		const bool integer = image.numeric_class != Prospero::TextureNumericClass::Float;
		auto& sampler = result[integer];
		if (sampler.id != 0u) return;
		uint32_t index = mem.sampler;
		if (root.integer_border != integer) {
			const auto found = std::ranges::find_if(samplers, [&](const auto& candidate) {
				return candidate.snapshot_index == root.snapshot_index &&
				       candidate.integer_border == integer;
			});
			if (found == samplers.end()) ctx.Fail(inst, "has no typed sampler variant");
			index = static_cast<uint32_t>(found - samplers.begin());
		}
		uint32_t array_index = 0;
		const auto& candidates = samplers[index].indirect_resources;
		if (!candidates.empty()) {
			const auto* handle = inst.Arg(1).ResolveInstruction();
			if (handle == nullptr || handle->GetOpcode() != IR::ValueOpcode::GetSamplerResource)
				ctx.Fail(inst, "has invalid finite sampler key provenance");
			auto selected = ctx.Def(handle->Arg(0));
			if (samplers[index].indirect_root != IR::SamplerResource::NoIndirectSampler)
				selected = EmitIndirectResourceIndex(state, selected,
				    samplers[index].indirect_mapping_offset,
				    samplers[index].indirect_search_iterations, 0u);
			array_index = SamplerArrayIndex(state, selected, candidates);
		}
		sampler = {LoadSamplerDescriptor(state, index, array_index), array_index != 0u};
	};
	const auto& image = ImageResourceAt(state, mem.resource);
	load(image);
	if (!image.bindless) {
		for (const auto candidate: image.indirect_resources) load(ImageResourceAt(state, candidate));
	}
	return result;
}

} // namespace

// The bindless slot of the image handle's key: translation[region + key] when the key is inside
// the heap's entry count (both from the flattened SRT, patched by the host), otherwise and while
// the texture is pending, slot 0 (the placeholder). A pending key is flagged in the feedback
// buffer for the host to resolve; every lane that reads an entry sees the same translation, so
// the unconditional store never races with a different value.
static uint32_t BindlessSlot(ValueEmitContext& ctx, const IR::ImageResource& image,
                             IR::Value image_arg) {
	auto&       state  = ctx.state;
	const auto* handle = image_arg.ResolveInstruction();
	EXIT_IF(handle == nullptr || handle->NumArgs() == 0u || state.flattened_srt_variable == 0 ||
	        state.bindless_translation_variable == 0);
	const auto key  = ctx.Def(handle->Arg(0));
	const auto Load = [&](uint32_t variable, uint32_t index) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, variable, ConstantU32(state, 0), index);
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	};
	const auto base     = Load(state.flattened_srt_variable,
	                           ConstantU32(state, image.indirect_mapping_offset));
	const auto count    = Load(state.flattened_srt_variable,
	                           ConstantU32(state, image.indirect_mapping_offset + 1u));
	const auto in_range = Binary(state, spv::OpULessThan, TypeBool(state), key, count);
	const auto entry    = Binary(state, spv::OpIAdd, TypeU32(state), base, key);
	const auto index    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), index, in_range, entry,
	                          ConstantU32(state, 0u));
	const auto raw     = Load(state.bindless_translation_variable, index);
	const auto pending = Binary(state, spv::OpIEqual, TypeBool(state), raw,
	                            ConstantU32(state, IR::BindlessPending));
	if (state.bindless_feedback_variable != 0) {
		const auto pending_flag = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), pending_flag, pending,
		                          ConstantU32(state, 1u), ConstantU32(state, 0u));
		// Diagnostics: a key outside its heap leaves itself (with the top bit set) in
		// feedback word 0, which belongs to no heap.
		const auto marked_key = Binary(state, spv::OpBitwiseOr, TypeU32(state), key,
		                               ConstantU32(state, 0x80000000u));
		const auto flag = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), flag, in_range, pending_flag,
		                          marked_key);
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, state.bindless_feedback_variable, ConstantU32(state, 0),
		                          index);
		state.builder.AddFunction(spv::OpStore, pointer, flag);
	}
	state.bindless_translation_raw = raw;
	const auto slot    = state.builder.AllocateId();
	// Pending keys sample slot 2 (the blue placeholder) while their texture is loaded.
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), slot, pending, ConstantU32(state, 2u),
	                          raw);
	state.builder.AddAnnotation(spv::OpDecorate, slot, spv::DecorationNonUniform);
	return slot;
}

// The bindless sampler array slot of the sampler handle's key: region + key while the key is
// inside the heap's entry count (both from the flattened SRT, patched by the host), otherwise
// slot 0, the default sampler.
static uint32_t BindlessSamplerSlot(ValueEmitContext& ctx, const IR::SamplerResource& sampler,
                                    IR::Value sampler_arg) {
	auto&       state  = ctx.state;
	const auto* handle = sampler_arg.ResolveInstruction();
	EXIT_IF(handle == nullptr || handle->NumArgs() == 0u || state.flattened_srt_variable == 0);
	const auto key  = ctx.Def(handle->Arg(0));
	const auto Load = [&](uint32_t index) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, state.flattened_srt_variable, ConstantU32(state, 0),
		                          ConstantU32(state, index));
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	};
	const auto base     = Load(sampler.bindless_mapping_offset);
	const auto count    = Load(sampler.bindless_mapping_offset + 1u);
	const auto in_range = Binary(state, spv::OpULessThan, TypeBool(state), key, count);
	const auto entry    = Binary(state, spv::OpIAdd, TypeU32(state), base, key);
	const auto slot     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), slot, in_range, entry,
	                          ConstantU32(state, 0u));
	state.builder.AddAnnotation(spv::OpDecorate, slot, spv::DecorationNonUniform);
	return slot;
}

static void EmitImageOperation(ValueEmitContext& ctx, const IR::Inst& inst);
static void EmitBindlessVolumeOrArray(ValueEmitContext& ctx, const IR::Inst& inst);

void EmitImage(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto op         = inst.GetOpcode();
	const auto image_info = IR::ImageOpcodeInfoOf(op);
	auto&       state     = ctx.state;
	const auto& mem       = ctx.Memory(inst);
	const auto  image_arg = inst.Arg(0);
	ctx.ResourceIndex(image_arg, IR::ValueOpcode::GetImageResource);
	const auto& image   = ImageResourceAt(state, mem.resource);
	state.bindless_slot = image.bindless ? BindlessSlot(ctx, image, image_arg) : 0u;
	state.bindless_sampler_slot = 0u;
	if (image_info.needs_sampler && inst.NumArgs() > 1u &&
	    mem.sampler < state.program.info.samplers.size() &&
	    state.program.info.samplers[mem.sampler].bindless) {
		state.bindless_sampler_slot =
		    BindlessSamplerSlot(ctx, state.program.info.samplers[mem.sampler], inst.Arg(1));
	}
	// Diagnostics: KYTY_KEY_CAPTURE=<hash> marks the keys this shader samples (BindlessCapture*).
	static const uint64_t capture_hash = [] {
		const char* value = std::getenv("KYTY_KEY_CAPTURE");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	}();
	if (capture_hash != 0 && state.program.shader_hash == capture_hash &&
	    state.bindless_feedback_variable != 0) {
		const auto Mark = [&](IR::Value handle_arg, uint32_t first, uint32_t value) {
			const auto* handle = handle_arg.ResolveInstruction();
			if (handle == nullptr || handle->NumArgs() == 0u) {
				return;
			}
			const auto key    = ctx.Def(handle->Arg(0));
			const auto low    = Binary(state, spv::OpBitwiseAnd, TypeU32(state), key,
			                           ConstantU32(state, 0xffffu));
			const auto index  = Binary(state, spv::OpIAdd, TypeU32(state), low, ConstantU32(state, first));
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
			                          pointer, state.bindless_feedback_variable, ConstantU32(state, 0),
			                          index);
			state.builder.AddFunction(spv::OpStore, pointer, value);
		};
		// Compile time: every image instruction, its sampler and what the sampler handle's key is.
		{
			std::string what = "none";
			if (inst.NumArgs() > 1u) {
				if (const auto* handle = inst.Arg(1).Resolve().TryInstruction();
				    handle != nullptr && handle->NumArgs() != 0u) {
					const auto key = handle->Arg(0).Resolve();
					if (key.IsImmediate()) {
						what = key.GetType() == IR::Type::U32 ? fmt::format("immediate 0x{:x}", key.U32()) : "immediate";
					} else if (const auto* key_inst = key.TryInstruction(); key_inst != nullptr) {
						what = std::string(IR::ValueOpcodeName(key_inst->GetOpcode()));
					}
				}
			}
			const bool sampler_bindless = image_info.needs_sampler && mem.sampler < state.program.info.samplers.size() &&
			                              state.program.info.samplers[mem.sampler].bindless;
			std::fprintf(stderr,
			             "[key-capture] emit pc=0x%x op=%s image=%u bindless_image=%d sampler=%u bindless_sampler=%d slot=%u key=%s\n",
			             inst.Flags<IR::MemoryFlags>().pc, std::string(IR::ValueOpcodeName(op)).c_str(), mem.resource,
			             image.bindless ? 1 : 0, mem.sampler, sampler_bindless ? 1 : 0, state.bindless_sampler_slot,
			             what.c_str());
		}
		uint32_t sampler_key = ConstantU32(state, 0xffffu);
		if (state.bindless_sampler_slot != 0u) {

			if (const auto* handle = inst.Arg(1).ResolveInstruction();
			    handle != nullptr && handle->NumArgs() != 0u) {
				sampler_key = Binary(state, spv::OpBitwiseAnd, TypeU32(state), ctx.Def(handle->Arg(0)),
				                     ConstantU32(state, 0xffffu));
			}
			// The raw dword the sampler key is cut from (before its shift or mask), at the image
			// key's word, to compare with the material data in memory.
			if (image.bindless) {
				const auto* sampler_handle = inst.Arg(1).ResolveInstruction();
				const auto* key_inst =
				    sampler_handle != nullptr && sampler_handle->NumArgs() != 0u
				        ? sampler_handle->Arg(0).Resolve().TryInstruction()
				        : nullptr;
				if (key_inst != nullptr && key_inst->NumArgs() == 2u &&
				    (key_inst->GetOpcode() == IR::ValueOpcode::ShiftRightLogical32 ||
				     key_inst->GetOpcode() == IR::ValueOpcode::BitwiseAnd32)) {
					const auto raw = ctx.Def(key_inst->Arg(0));
					Mark(image_arg, IR::BindlessCaptureSamplers, raw);
				}
			}
		}
		if (image.bindless) {
			const auto value = Binary(state, spv::OpBitwiseOr, TypeU32(state), sampler_key,
			                          ConstantU32(state, 0x40000000u | ((mem.resource & 0xffu) << 16u)));
			Mark(image_arg, IR::BindlessCaptureImages, value);
		}
	}
	if (BindlessVolumeMayBeArray(image) && op != IR::ValueOpcode::ImageWrite) {
		EmitBindlessVolumeOrArray(ctx, inst);
		return;
	}
	EmitImageOperation(ctx, inst);
}

// A bindless 3D image instruction whose key may name a 2D or 2D-array texture: the GPU addresses
// by the T# type, so such a texture is read as a 2D array at layer round(z) (Vulkan's array
// layer rule is the same). Both reads are emitted, each with the other view's placeholder slot,
// and the flag selects: no divergent branch, so implicit-LOD derivatives stay defined.
static void EmitBindlessVolumeOrArray(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state = ctx.state;
	const auto& mem   = ctx.Memory(inst);
	const auto  raw   = state.bindless_translation_raw;
	const auto  slot  = state.bindless_slot;
	const auto  flags = Binary(state, spv::OpBitwiseAnd, TypeU32(state), raw,
	                           ConstantU32(state, IR::BindlessSlotFlagMask));
	const auto  as_array = Binary(state, spv::OpIEqual, TypeBool(state), flags,
	                              ConstantU32(state, IR::BindlessSlotArrayFlag));
	const auto  array_index = Binary(state, spv::OpBitwiseAnd, TypeU32(state), raw,
	                                 ConstantU32(state, ~IR::BindlessSlotFlagMask));
	const auto  NonUniformSelect = [&](uint32_t condition, uint32_t yes, uint32_t no) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), value, condition, yes, no);
		state.builder.AddAnnotation(spv::OpDecorate, value, spv::DecorationNonUniform);
		return value;
	};
	const auto array_slot  = NonUniformSelect(as_array, array_index, ConstantU32(state, 0u));
	const auto volume_slot = NonUniformSelect(as_array, ConstantU32(state, 0u), slot);
	// An existing definition (a value used before this point) is re-pointed at the selection.
	const auto predefined = inst.Definition(ctx.half);
	const auto Emit = [&](bool array_view) {
		inst.SetDefinition(ctx.half, 0u);
		if (array_view) {
			state.image_override_index = mem.resource;
			state.image_override       = state.program.info.images.at(mem.resource);
			state.image_override.dimension = ImageDimension::Dim2DArray;
		}
		state.bindless_slot = array_view ? array_slot : volume_slot;
		EmitImageOperation(ctx, inst);
		state.image_override_index = UINT32_MAX;
		state.bindless_slot        = slot;
		const auto value = inst.Definition(ctx.half);
		EXIT_IF(value == 0u);
		inst.SetDefinition(ctx.half, 0u);
		return value;
	};
	const auto array_value  = Emit(true);
	const auto volume_value = Emit(false);
	const auto value        = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeId(state, inst.GetType()), value, as_array,
	                          array_value, volume_value);
	inst.SetDefinition(ctx.half, predefined);
	ctx.Define(inst, value);
}

static void EmitImageOperation(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto  op         = inst.GetOpcode();
	const auto  image_info = IR::ImageOpcodeInfoOf(op);
	auto&       state      = ctx.state;
	const auto& mem        = ctx.Memory(inst);
	const auto& image      = ImageResourceAt(state, mem.resource);
	const auto  image_arg  = inst.Arg(0);
	const auto* address = ctx.ImageAddress(inst.Arg(image_info.needs_sampler ? 2 : 1));
	if (op == IR::ValueOpcode::ImageQueryDimensions) {
		state.builder.RequireCapability(spv::CapabilityImageQuery);
		ctx.Define(inst, EmitImageAccess(ctx, inst, TypeU32Vector(state, 4),
		    [&](uint32_t resource, uint32_t array_index) {
			    return QueryDimensions(ctx, mem, *address, resource, array_index);
		    }));
		return;
	}
	if (image_info.needs_sampler &&
	    image.conversion_format == Prospero::BufferFormat::k11_11_10UNorm) {
		if (op != IR::ValueOpcode::ImageSampleRaw ||
		    image.dimension != ImageDimension::Dim2D ||
		    mem.image_dimension != ImageDimension::Dim2D || mem.data_bits != 32u ||
		    mem.image_sample_flags != Decoder::ImageSampleFlagLevelZero ||
		    image.indirect_root != IR::ImageResource::NoIndirectImage ||
		    !state.program.info.samplers.at(mem.sampler).indirect_resources.empty()) {
			ctx.Fail(inst, "packed UNorm sampling requires a direct 2D sample_lz");
			return;
		}
		const auto coord = CoordF32(ctx, mem, *address, Layout(mem).coord, 2u);
		ctx.Define(inst, ResultVector(ctx, SamplePackedUnormLz(ctx, mem, coord),
		                              Prospero::TextureNumericClass::Float, false, mem));
		return;
	}
	if (op == IR::ValueOpcode::ImageQueryLod) {
		state.builder.RequireCapability(spv::CapabilityImageQuery);
		const auto sampler = LoadSamplerDescriptor(state, mem.sampler);
		const auto lod = EmitImageAccess(ctx, inst, TypeF32Vector(state, 2),
		    [&](uint32_t resource, uint32_t array_index) {
			    const auto& candidate = state.program.info.images[resource];
			    const auto query = state.builder.AllocateId();
			    state.builder.AddFunction(
			        spv::OpImageQueryLod, TypeF32Vector(state, 2), query,
			        MakeSampledImage(state, resource, sampler, 0u, array_index),
			        CoordF32(ctx, mem, *address, 0,
			                 ImageDimensionInfoFor(candidate.dimension).spatial_components,
			                 candidate.cube));
			    return query;
		    });
		uint32_t values[4] = {ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0),
		                      ConstantU32(state, 0)};
		for (uint32_t index = 0; index < 2u; index++) {
			const auto component = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, lod,
			                          index);
			values[index] = FloatBits(ctx, component);
		}
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result,
		                          values[0], values[1], values[2], values[3]);
		ctx.Define(inst, result);
		return;
	}
	if (op == IR::ValueOpcode::ImageRead) {
		const auto condition = ctx.Arg(inst, 2);
		ctx.Define(
		    inst,
		    EmitValueOrDefaultIfCondition(
		        state, condition, TypeU32Vector(state, 4), ConstantU32CompositeZero(state, 4),
		        [&]() {
			        return EmitImageResult(
			            ctx, inst, [&](uint32_t resource, uint32_t array_index, uint32_t result_type) {
				        const auto& candidate = ImageResourceAt(state, resource);
				        const auto dimension = candidate.dimension;
				        const auto& dimension_info = ImageDimensionInfoFor(dimension);
				        const auto descriptor = LoadImageDescriptor(state, resource, 0u, array_index);
				        const auto fetched = state.builder.AllocateId();
				        const auto coord = CoordU32(ctx, mem, *address, dimension);
				        if (dimension_info.multisampled != 0u) {
					        state.builder.AddFunction(
					            spv::OpImageFetch, result_type, fetched,
					            descriptor, coord, spv::ImageOperandsSampleMask,
					            AddressU32(ctx, mem, *address, dimension_info.coordinate_components));
				        } else {
					        state.builder.AddFunction(
					            spv::OpImageFetch, result_type, fetched,
					            descriptor, coord, spv::ImageOperandsLodMask,
					            LodU32(ctx, mem, *address, mem.image_dimension));
				        }
				        return fetched;
			            });
		        }));
		return;
	}
	if (op == IR::ValueOpcode::ImageWrite) {
		EmitIfCondition(state, ctx.Arg(inst, 3), [&]() {
			EmitImageAccess(ctx, inst, 0u, [&](uint32_t resource, uint32_t array_index) {
				const auto& candidate = ImageResourceAt(state, resource);
				auto selected = mem;
				selected.resource = resource;
				const auto mip_lod = candidate.mip_mode == IR::ImageMipMode::Dynamic
				    ? LodU32(ctx, mem, *address, mem.image_dimension) : 0u;
				const auto coord = CoordU32(ctx, mem, *address, candidate.dimension);
				const auto& dimension_info = ImageDimensionInfoFor(candidate.dimension);
				const auto sample = dimension_info.multisampled != 0u
				    ? AddressU32(ctx, mem, *address, dimension_info.coordinate_components) : 0u;
				const auto texel = StoreTexel(ctx, selected, ctx.Arg(inst, 2),
				    candidate.numeric_class == Prospero::TextureNumericClass::Uint);
				EmitStorageImageWrite(state, resource, mip_lod, coord, texel, array_index, sample);
			});
		});
		return;
	}
	if (op == IR::ValueOpcode::ImageSampleRaw || op == IR::ValueOpcode::ImageGatherRaw) {
		const auto  dimension      = image.dimension;
		const auto& dimension_info = ImageDimensionInfoFor(dimension);
		const auto  layout         = Layout(mem);
		const auto  numeric_class  = image.numeric_class;
		const bool  dref           = HasFlag(mem, Decoder::ImageSampleFlagCompare);
		IR::Value offset_value;
		if (layout.offset != NoImageComponent) {
			const auto value = address->Arg(layout.offset).Resolve();
			offset_value = IR::ResolveActiveU32(value, inst.Arg(inst.NumArgs() - 1u).Resolve());
			if (offset_value.IsEmpty()) offset_value = value;
		}
		if (dref && ImageResourceAt(state, mem.resource).conversion_format !=
		                Prospero::BufferFormat::kInvalid) {
			ctx.Fail(inst, "uses depth comparison with a converted image");
			return;
		}
		if (op == IR::ValueOpcode::ImageGatherRaw) {
			if (dimension == ImageDimension::Dim1D) {
				if (image.indirect_root == mem.resource || dref ||
				    !HasFlag(mem, Decoder::ImageSampleFlagLevelZero) ||
				    HasFlag(mem, Decoder::ImageSampleFlagOffset) ||
				    HasFlag(mem, Decoder::ImageSampleFlagGatherHorizontal)) {
					ctx.Fail(inst, "has an unsupported 1D gather variant");
					return;
				}
				const auto coord = CoordF32(ctx, mem, *address, layout.coord,
				                            dimension_info.coordinate_components, image.cube);
				const auto sample = EmitOneDimensionalGatherLz(ctx, mem, coord, numeric_class);
				ctx.Define(inst, ResultVector(ctx, UnpackImageGather(ctx, mem, sample),
				                              numeric_class, false, mem, true));
				return;
			}
			if (dimension == ImageDimension::Dim1DArray) {
				ctx.Fail(inst, "has an unsupported 1D-array gather");
				return;
			}
			uint32_t component_or_dref;
			if (dref) {
				component_or_dref = layout.dref != NoImageComponent
				                        ? AddressF32(ctx, mem, *address, layout.dref)
				                        : ZeroF32(state);
			} else {
				component_or_dref = ConstantU32(state, ImageGatherSource(state, mem));
			}
			const auto samplers = LoadImageSamplers(ctx, inst);
			const auto EmitGather = [&](uint32_t resource, uint32_t array_index, uint32_t result_type) {
				const auto& candidate = ImageResourceAt(state, resource);
				const auto sampler = samplers[candidate.numeric_class != Prospero::TextureNumericClass::Float];
				const auto dimension = candidate.dimension;
				const auto coord = CoordF32(
				    ctx, mem, *address, layout.coord,
				    ImageDimensionInfoFor(dimension).coordinate_components, candidate.cube);
				uint32_t operand_mask = 0;
				uint32_t offset = 0;
				if (HasFlag(mem, Decoder::ImageSampleFlagGatherHorizontal)) {
					operand_mask = spv::ImageOperandsConstOffsetsMask;
					offset = HorizontalOffsets(state, dimension);
				} else if (layout.offset != NoImageComponent) {
					offset = PackedOffset(ctx, offset_value, dimension, operand_mask);
				}
				const auto EmitMip = [&](uint32_t mip) {
					const auto sampled = MakeSampledImage(state, resource, sampler.id, mip, array_index,
					                                      sampler.dynamic);
					const auto sample = state.builder.AllocateId();
					const std::array operands {operand_mask, offset};
					state.builder.AddFunction(
					    dref ? spv::OpImageDrefGather : spv::OpImageGather, result_type,
					    sample, sampled, coord, component_or_dref,
					    std::span(operands).first(operand_mask != 0u ? 2u : 0u));
					return sample;
				};
				return candidate.mip_mode == IR::ImageMipMode::Dynamic && candidate.mip_count > 1u
				           ? EmitIndexSwitch(
				                 state, GatherMip(ctx, inst, mem, *address, layout, candidate.mip_count),
				                 candidate.mip_count, result_type, EmitMip)
				           : EmitMip(0u);
			};
			ctx.Define(inst, EmitImageResult(ctx, inst, EmitGather));
			return;
		}
		const bool explicit_lod = HasFlag(mem, Decoder::ImageSampleFlagDerivative) ||
		                          HasFlag(mem, Decoder::ImageSampleFlagLod) ||
		                          HasFlag(mem, Decoder::ImageSampleFlagLevelZero) ||
		                          state.program.stage != ShaderType::Pixel;
		auto       opcode       = spv::OpImageSampleImplicitLod;
		if (explicit_lod) {
			opcode = dref ? spv::OpImageSampleDrefExplicitLod : spv::OpImageSampleExplicitLod;
		} else if (dref) {
			opcode = spv::OpImageSampleDrefImplicitLod;
		}
		uint32_t dref_value  = 0;
		if (dref) {
			dref_value  = ZeroF32(state);
			if (layout.dref != NoImageComponent) {
				dref_value = AddressF32(ctx, mem, *address, layout.dref);
			}
		}
		// IMAGE_SAMPLE*_CL: the last address component is a minimum LOD. Implicit and gradient
		// samples take it as the MinLod operand; explicit-LOD forms (every sample outside pixel
		// shaders) raise their LOD to it.
		uint32_t lod_clamp = 0;
		if (layout.clamp != NoImageComponent && mem.image_address_components > layout.clamp) {
			lod_clamp = AddressF32(ctx, mem, *address, layout.clamp);
		}
		const bool gradient   = HasFlag(mem, Decoder::ImageSampleFlagDerivative);
		const bool min_lod_op = lod_clamp != 0u && (gradient || !explicit_lod) &&
		                        GetHostImageFeatures().min_lod;
		uint32_t                explicit_lod_value = 0;
		std::array<uint32_t, 6> operands {};
		size_t                  operand_count = 0;
		if (gradient) {
			operands[0] = spv::ImageOperandsGradMask;
			operands[1] = CoordF32(ctx, mem, *address, layout.grad_x, dimension_info.spatial_components);
			operands[2] = CoordF32(ctx, mem, *address, layout.grad_y, dimension_info.spatial_components);
			operand_count = 3;
		} else if (explicit_lod) {
			auto lod = ZeroF32(state);
			if (HasFlag(mem, Decoder::ImageSampleFlagLod) && layout.lod != NoImageComponent) {
				lod = AddressF32(ctx, mem, *address, layout.lod);
			}
			if (lod_clamp != 0u) {
				const auto clamped = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpExtInst, TypeF32(state), clamped, GlslStd450(state),
				                          GLSLstd450FMax, lod, lod_clamp);
				lod = clamped;
			}
			explicit_lod_value = lod;
			operands[0] = spv::ImageOperandsLodMask;
			operands[1] = lod;
			operand_count = 2;
		} else if (layout.bias != NoImageComponent) {
			operands[0] = spv::ImageOperandsBiasMask;
			operands[1] = AddressF32(ctx, mem, *address, layout.bias);
			operand_count = 2;
		}
		// IMAGE_SAMPLE*_O: one dword of six-bit signed texel offsets. Vulkan
		// accepts a constant offset within [minTexelOffset, maxTexelOffset] (every implementation
		// supports -8..7) or, with maintenance8, any value; anything else moves the normalized
		// coordinate by offset / level size instead (ApplyDynamicSampleOffset).
		enum class SampleOffsetMode { None, Operand, Dynamic };
		auto offset_mode = SampleOffsetMode::None;
		if (layout.offset != NoImageComponent && !image.cube && !offset_value.IsEmpty()) {
			if (offset_value.IsImmediate() && offset_value.GetType() == IR::Type::U32) {
				bool in_range = true;
				bool any      = false;
				for (uint32_t index = 0; index < dimension_info.spatial_components; index++) {
					const auto offset = SignExtend6(offset_value.U32() >> (index * 8u));
					in_range          = in_range && offset >= -8 && offset <= 7;
					any               = any || offset != 0;
				}
				if (any) {
					offset_mode = in_range ? SampleOffsetMode::Operand : SampleOffsetMode::Dynamic;
				}
			} else {
				offset_mode = state.profile.maintenance8 ? SampleOffsetMode::Operand
				                                         : SampleOffsetMode::Dynamic;
			}
		} else if (layout.offset != NoImageComponent && image.cube) {
			Common::HackNotice::ReportTranslation(
			    "IMAGE_SAMPLE*_O offset ignored on a cube map", "SPIR-V image sample",
			    state.program.shader_hash, mem.resource);
		}
		if (offset_mode == SampleOffsetMode::Dynamic && image.bindless) {
			// The dynamic path queries the image's size through its descriptor, which a bindless
			// image does not have in this form.
			Common::HackNotice::ReportTranslation(
			    "non-constant IMAGE_SAMPLE*_O offset ignored on a bindless image",
			    "SPIR-V image sample", state.program.shader_hash, mem.resource);
			offset_mode = SampleOffsetMode::None;
		}
		if (lod_clamp != 0u && !min_lod_op && !explicit_lod) {
			Common::HackNotice::ReportTranslation(
			    "IMAGE_SAMPLE*_CL clamp ignored: the device lacks shaderResourceMinLod",
			    "SPIR-V image sample", state.program.shader_hash, mem.resource);
		}
		const auto samplers = LoadImageSamplers(ctx, inst);
		const auto EmitSample = [&](uint32_t resource, uint32_t array_index, uint32_t result_type) {
			const auto& candidate = ImageResourceAt(state, resource);
			const auto sampler = samplers[candidate.numeric_class != Prospero::TextureNumericClass::Float];
			auto sample_operands = operands;
			auto sample_operand_count = operand_count;
			if (offset_mode == SampleOffsetMode::Operand) {
				sample_operand_count = std::max(size_t{1}, sample_operand_count);
				sample_operands[sample_operand_count++] = PackedOffset(
				    ctx, offset_value, candidate.dimension, sample_operands[0]);
			}
			if (min_lod_op) {
				state.builder.RequireCapability(spv::CapabilityMinLod);
				sample_operand_count = std::max(size_t{1}, sample_operand_count);
				sample_operands[0] |= spv::ImageOperandsMinLodMask;
				sample_operands[sample_operand_count++] = lod_clamp;
			}
			auto coord =
			    CoordF32(ctx, mem, *address, layout.coord,
			             ImageDimensionInfoFor(candidate.dimension).coordinate_components,
			             candidate.cube);
			if (offset_mode == SampleOffsetMode::Dynamic) {
				uint32_t level = ZeroF32(state);
				if (explicit_lod_value != 0u) {
					level = explicit_lod_value;
				} else if (!HasFlag(mem, Decoder::ImageSampleFlagLevelZero) &&
				           state.program.stage == ShaderType::Pixel) {
					// The level an implicit or gradient sample reads, relative to the view base.
					state.builder.RequireCapability(spv::CapabilityImageQuery);
					const auto query = state.builder.AllocateId();
					state.builder.AddFunction(
					    spv::OpImageQueryLod, TypeF32Vector(state, 2), query,
					    MakeSampledImage(state, resource, sampler.id, 0u, array_index,
					                     sampler.dynamic),
					    CoordF32(ctx, mem, *address, layout.coord,
					             ImageDimensionInfoFor(candidate.dimension).spatial_components,
					             candidate.cube));
					level = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), level, query,
					                          0u);
				}
				coord = ApplyDynamicSampleOffset(state, resource, coord, candidate.dimension,
				                                 ctx.Def(offset_value), level, array_index);
			}
			const auto sampled = MakeSampledImage(state, resource, sampler.id, 0u, array_index,
			                                      sampler.dynamic);
			const auto sample = state.builder.AllocateId();
			state.builder.AddFunction(opcode, result_type, sample, sampled, coord,
			                          std::span(&dref_value, dref ? 1u : 0u),
			                          std::span<const uint32_t>(sample_operands).first(sample_operand_count));
			return sample;
		};
		ctx.Define(inst, EmitImageResult(ctx, inst, EmitSample));
		return;
	}
	if (image_info.access == IR::ImageAccess::Atomic) {
		const auto dimension = image.dimension;
		const auto result_type = TypeId(state, inst.GetType());
		const auto zero = image.atomic64 ? ConstantU64(state, 0) : ConstantU32(state, 0);
		ctx.Define(inst, EmitValueOrDefaultIfCondition(
		                     state, ctx.Arg(inst, inst.NumArgs() - 1), result_type, zero, [&]() {
			           const auto pointer      = state.builder.AllocateId();
			           const auto pointer_type =
			               state.builder.Type(spv::OpTypePointer, spv::StorageClassImage,
			                                  image.atomic64 ? TypeU64(state) : TypeU32(state));
			           state.builder.AddFunction(spv::OpImageTexelPointer, pointer_type, pointer,
			                                     ImageDescriptorPointer(state, mem.resource),
			                                     CoordU32(ctx, mem, *address, dimension),
			                                     ConstantU32(state, 0));
			           if (op == IR::ValueOpcode::ImageAtomicFMin32 ||
			               op == IR::ValueOpcode::ImageAtomicFMax32) {
				           return AtomicUpdate(state, pointer, IR::ResourceKind::Image,
				                               [&](uint32_t old) {
					                               return EmitFloatAtomicReplacement(
					                                   state, old, ctx.Arg(inst, 2),
					                                   op == IR::ValueOpcode::ImageAtomicFMax32);
				                               });
			           }
			           const auto old = EmitAtomicOperation(ctx, inst, pointer, spv::ScopeDevice);
			           EmitAtomicMemoryBarrier(state, IR::ResourceKind::Image);
			           return old;
		           }));
		return;
	}
	ctx.Fail(inst, "has no image SPIR-V emitter");
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
