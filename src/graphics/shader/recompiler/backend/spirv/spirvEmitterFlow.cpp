#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"

#include <algorithm>
#include <cstdlib>
#include <fmt/format.h>
#include <string>
#include <atomic>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

bool UserDataDwordIndex(const EmitterState& state, IR::ScalarReg reg, uint32_t& dword_index) {
	const auto register_index = IR::RegIndex(reg);
	const auto& registers = state.program.info.user_data_registers;
	const auto  found     = std::lower_bound(registers.begin(), registers.end(), register_index);
	if (found == registers.end() || *found != register_index) {
		return false;
	}
	dword_index = static_cast<uint32_t>(found - registers.begin());
	return true;
}

uint32_t EmitBuiltinU32(EmitterState& state, IR::StageInputKind kind, uint32_t component) {
	if (kind == IR::StageInputKind::DispatchThreadCount) {
		const auto dword = state.program.bindings.dispatch_thread_dword;
		if (!DispatchDimensionsIndirect(state)) {
			return EmitShaderDataDwordLoad(state, dword + component);
		}
		// The counts stay where the guest's indirect arguments are; the shader data holds
		// their device address.
		const auto u64     = TypeU64(state);
		const auto low     = Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, dword));
		const auto high    = Unary(state, spv::OpUConvert, u64,
		                           EmitShaderDataDwordLoad(state, dword + 1u));
		const auto base    = Binary(state, spv::OpBitwiseOr, u64, low,
		                            Binary(state, spv::OpShiftLeftLogical, u64, high,
		                                   state.builder.Constant(spv::OpConstant, u64, 32u, 0u)));
		const auto address = Binary(state, spv::OpIAdd, u64, base,
		                            state.builder.Constant(spv::OpConstant, u64,
		                                                   component * 4u, 0u));
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
		                          address);
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
		                          spv::MemoryAccessAlignedMask, 4u);
		return value;
	}
	if (kind == IR::StageInputKind::LocalInvocationIndex) {
		return EmitLocalInvocationIndex(state);
	}
	if (state.lane_count == 2 && (kind == IR::StageInputKind::LocalInvocationId ||
	                              kind == IR::StageInputKind::GlobalInvocationId)) {
		const auto* cs      = ShaderWorkgroupInput(state.program.stage, state.input_info);
		uint32_t    divisor = 1;
		for (uint32_t axis = 0; axis < component; axis++) {
			divisor *= std::max(cs->threads_num[axis], 1u);
		}
		const auto size    = std::max(cs->threads_num[component], 1u);
		const auto divided = EmitBinaryU32(state, spv::OpUDiv, EmitLocalInvocationIndex(state),
		                                   ConstantU32(state, divisor));
		const auto local   = EmitBinaryU32(state, spv::OpUMod, divided, ConstantU32(state, size));
		if (kind == IR::StageInputKind::LocalInvocationId) {
			return local;
		}
		const auto group = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, component);
		return EmitAddU32(state, local,
		                  EmitBinaryU32(state, spv::OpIMul, group, ConstantU32(state, size)));
	}
	const bool centroid = kind == IR::StageInputKind::BaryCoordSmoothCentroid;
	const bool sample = kind == IR::StageInputKind::BaryCoordSmoothSample;
	const auto variable = InputVariableForKind(
	    state, centroid || sample ? IR::StageInputKind::BaryCoordSmooth : kind);
	if (variable == 0) {
		return ConstantU32(state, 0);
	}
	if (kind == IR::StageInputKind::FrontFacing) {
		const auto value = state.builder.AllocateId();
		const auto bits  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeBool(state), value, variable);
		// SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS selects uint masks or float +/-1.
		const bool all_bits = state.input_info.pixel->ps_front_face_all_bits;
		const auto front = ConstantU32(state, all_bits ? 0xffffffffu : 0x3f800000u);
		const auto back  = ConstantU32(state, all_bits ? 0u : 0xbf800000u);
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), bits, value, front, back);
		return bits;
	}
	if (kind == IR::StageInputKind::VertexIndex || kind == IR::StageInputKind::InstanceIndex ||
	    kind == IR::StageInputKind::InvocationId || kind == IR::StageInputKind::PrimitiveId ||
	    kind == IR::StageInputKind::Layer || kind == IR::StageInputKind::SampleId) {
		const auto value = state.builder.AllocateId();
		const auto bits  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeI32(state), value, variable);
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	if (kind == IR::StageInputKind::FragCoord || kind == IR::StageInputKind::TessCoord) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		const auto bits    = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain,
		                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
		                          pointer, variable, ConstantU32(state, component));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	if (centroid || sample || kind == IR::StageInputKind::BaryCoordSmooth ||
	    kind == IR::StageInputKind::BaryCoordNoPerspective) {
		const auto value   = state.builder.AllocateId();
		const auto bits    = state.builder.AllocateId();
		if (centroid || sample) {
			const auto coordinates = state.builder.AllocateId();
			state.builder.RequireCapability(spv::CapabilityInterpolationFunction);
			if (sample) {
				const auto sample_id = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpLoad, TypeI32(state), sample_id,
				                          InputVariableForKind(state, IR::StageInputKind::SampleId));
				state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 3), coordinates,
				                          GlslStd450(state), GLSLstd450InterpolateAtSample,
				                          variable, sample_id);
			} else {
				state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 3), coordinates,
				                          GlslStd450(state), GLSLstd450InterpolateAtCentroid, variable);
			}
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), value,
			                          coordinates, component + 1u);
		} else {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain,
			                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
			                          pointer, variable, ConstantU32(state, component + 1u));
			state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		}
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	return EmitInputComponentU32(state, kind, component);
}

uint32_t EmitDppWriteCondition(ValueEmitContext& ctx, const IR::DppMoveFlags& flags,
                               uint32_t exec) {
	auto&    state    = ctx.state;
	uint32_t writable = exec;
	uint32_t lane     = 0;
	// Each mask bit enables writes to one bank (4 lanes) or row (16 lanes); a full mask enables all.
	const auto enable = [&](uint32_t mask, uint32_t group_shift) {
		if (mask == 0xfu) return;
		lane = lane != 0 ? lane : EmitSubgroupLocalInvocationId(state);
		const auto group = EmitNative<spv::OpBitFieldUExtract, IR::Type::U32>(
		    state, lane, ConstantU32(state, group_shift), ConstantU32(state, 2));
		const auto bit = EmitNative<spv::OpBitFieldUExtract, IR::Type::U32>(
		    state, ConstantU32(state, mask), group, ConstantU32(state, 1));
		writable = Binary(state, spv::OpLogicalAnd, TypeBool(state), writable,
		                  Binary(state, spv::OpINotEqual, TypeBool(state), bit, ConstantU32(state, 0)));
	};
	enable(flags.bank_mask, 2u);
	enable(flags.row_mask, 4u);
	// Only row shifts (0x101-0x10f, 0x111-0x11f) can vacate a lane; other controls always read one.
	const bool row_shift = !flags.dpp8 && flags.control > 0x100u && flags.control < 0x120u &&
	                       flags.control != 0x110u;
	if (!flags.bound_control && row_shift) {
		writable = Binary(state, spv::OpLogicalAnd, TypeBool(state), writable,
		                  EmitDppTargetLane(state, flags).valid);
	}
	return writable;
}

uint32_t LoadVertexParameter(EmitterState& state, const InputBinding& input, uint32_t vertex,
                             uint32_t chan) {
	const auto pointer = state.builder.AllocateId();
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain,
	                          TypePointer(state, spv::StorageClassInput, TypeF32(state)), pointer,
	                          input.variable_id, ConstantU32(state, vertex),
	                          ConstantU32(state, chan & 3u));
	state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
	return value;
}

// Interpolates a pixel input channel to f32. A half (0 low, 1 high) selects one f16 of a packed
// FP16_INTERP_MODE dword before blending; smooth inputs read by halves are always per-vertex.
uint32_t EmitPixelAttribute(EmitterState& state, const InputBinding& input, uint32_t attr,
                            uint32_t chan, std::optional<uint32_t> half) {
	const auto select = [&](uint32_t value) {
		if (!half) {
			return value;
		}
		const auto bits     = state.builder.AllocateId();
		const auto unpacked = state.builder.AllocateId();
		const auto word     = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 2), unpacked,
		                          GlslStd450(state), GLSLstd450UnpackHalf2x16, bits);
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), word, unpacked, *half);
		return word;
	};
	if (!input.per_vertex) {
		const auto vector    = state.builder.AllocateId();
		const auto component = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeF32Vector(state, 4), vector, input.variable_id);
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, vector,
		                          chan & 3u);
		return select(component);
	}
	if (PixelParameterIsFlat(state, attr)) {
		const auto vertex =
		    state.input_info.pixel->parameter_mode == ShaderPixelParameterMode::LastVertex ? 2u : 0u;
		return select(LoadVertexParameter(state, input, vertex, chan));
	}
	const auto barycentric_kind = state.input_info.pixel->ps_no_perspective
	                                  ? IR::StageInputKind::BaryCoordNoPerspective
	                                  : IR::StageInputKind::BaryCoordSmooth;
	const auto barycentric      = InputVariableForKind(state, barycentric_kind);
	uint32_t   sum              = 0;
	for (uint32_t vertex = 0; vertex < 3u; vertex++) {
		const auto pointer = state.builder.AllocateId();
		const auto weight  = state.builder.AllocateId();
		const auto product = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain,
		                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
		                          pointer, barycentric, ConstantU32(state, vertex));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), weight, pointer);
		state.builder.AddFunction(spv::OpFMul, TypeF32(state), product,
		                          select(LoadVertexParameter(state, input, vertex, chan)), weight);
		if (vertex == 0u) {
			sum = product;
		} else {
			const auto next = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpFAdd, TypeF32(state), next, sum, product);
			sum = next;
		}
	}
	return sum;
}

uint32_t EmitAttribute(EmitterState& state, uint32_t attr, uint32_t chan) {
	const auto* input = InputBindingForParameter(state, attr);
	if (input == nullptr || input->variable_id == 0) {
		return ConstantU32(state, 0);
	}
	if (state.program.stage == ShaderType::Vertex || state.program.stage == ShaderType::Local) {
		return EmitVertexParameterComponentU32(state, *input, chan & 3u);
	}
	const auto value = EmitPixelAttribute(state, *input, attr, chan, std::nullopt);
	const auto bits  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
	return bits;
}

uint32_t EmitInterpolationParameter(ValueEmitContext& ctx, uint32_t attr, uint32_t chan,
                                    uint32_t mode) {
	auto&       state = ctx.state;
	const auto* input = InputBindingForParameter(state, attr);
	if (!input->per_vertex) {
		return EmitAttribute(ctx.state, attr, chan);
	}
	uint32_t selected_vertex = (mode + 1u) % 3u;
	if (mode == 2u && PixelParameterIsFlat(state, attr)) {
		selected_vertex =
		    state.input_info.pixel->parameter_mode == ShaderPixelParameterMode::LastVertex ? 2u : 0u;
	}
	uint32_t value = LoadVertexParameter(state, *input, selected_vertex, chan);
	if (!PixelParameterIsCustom(state, attr) && mode < 2u) {
		const auto delta = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFSub, TypeF32(state), delta, value,
		                          LoadVertexParameter(state, *input, 0, chan));
		value = delta;
	}
	const auto bits = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
	return bits;
}

uint32_t MrtOutputMode(const EmitterState& state, const IR::ExportInfo& exp) {
	if (state.program.stage != ShaderType::Pixel || exp.kind != IR::ExportTargetKind::Mrt ||
	    exp.index >= std::size(state.input_info.pixel->target_output_mode)) {
		return 0;
	}
	return state.input_info.pixel->target_output_mode[exp.index];
}

uint32_t ExportRawComponent(ValueEmitContext& ctx, uint32_t vector, uint32_t component) {
	const auto value = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), value, vector,
	                              component);
	return value;
}

uint32_t ExportVector(ValueEmitContext& ctx, uint32_t data, const IR::ExportInfo& exp,
                      bool uint_output) {
	auto& state = ctx.state;
	if (exp.compr && !uint_output) {
		auto unpack = GLSLstd450UnpackHalf2x16;
		switch (MrtOutputMode(state, exp)) {
			case 5: unpack = GLSLstd450UnpackUnorm2x16; break;
			case 6: unpack = GLSLstd450UnpackSnorm2x16; break;
			default: break;
		}
		uint32_t f32[4] = {ConstantF32(state, 0), ConstantF32(state, 0), ConstantF32(state, 0),
		                   ConstantF32(state, 0x3f800000u)};
		for (uint32_t pair = 0; pair < 2u; pair++) {
			if ((exp.en & (3u << (pair * 2u))) == 0u) {
				continue;
			}
			const auto packed   = state.builder.AllocateId();
			const auto unpacked = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), packed, data, pair);
			state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 2), unpacked,
			                          GlslStd450(state), unpack, packed);
			for (uint32_t lane = 0; lane < 2u; lane++) {
				const auto component = pair * 2u + lane;
				if (((exp.en >> component) & 1u) != 0u) {
					f32[component] = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state),
					                          f32[component], unpacked, lane);
				}
			}
		}
		const auto vector = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), vector,
		                          f32[0], f32[1], f32[2], f32[3]);
		return vector;
	}
	if (exp.compr || exp.en != 0xfu) {
		uint32_t raw[4] = {
		    ConstantU32(state, 0),
		    ConstantU32(state, 0),
		    ConstantU32(state, 0),
		    ConstantU32(state, uint_output ? 1u : 0x3f800000u),
		};
		if (exp.compr) {
			for (uint32_t pair = 0; pair < 2u; pair++) {
				if ((exp.en & (3u << (pair * 2u))) == 0u) {
					continue;
				}
				const auto packed = ExportRawComponent(ctx, data, pair);
				for (uint32_t lane = 0; lane < 2u; lane++) {
					const auto component = pair * 2u + lane;
					if (((exp.en >> component) & 1u) == 0u) {
						continue;
					}
					raw[component] = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), raw[component],
					                          packed, ConstantU32(state, lane * 16u),
					                          ConstantU32(state, 16));
				}
			}
		} else {
			for (uint32_t component = 0; component < 4u; component++) {
				if (((exp.en >> component) & 1u) != 0u) {
					raw[component] = ExportRawComponent(ctx, data, component);
				}
			}
		}
		data = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), data,
		                          raw[0], raw[1], raw[2], raw[3]);
	}
	if (uint_output) {
		return data;
	}
	const auto vector = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeF32Vector(state, 4), vector, data);
	return vector;
}

uint32_t DistancePointer(EmitterState& state, IR::StageOutputKind kind, uint32_t index) {
	if (state.program.stage == ShaderType::Mesh) {
		return MeshOutputPointer(state, kind, index);
	}
	const auto variable = kind == IR::StageOutputKind::ClipDistance ? state.clip_distance_variable
	                                                                : state.cull_distance_variable;
	const auto pointer  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain,
	                          TypePointer(state, spv::StorageClassOutput, TypeF32(state)), pointer,
	                          variable, ConstantU32(state, index));
	return pointer;
}

// The clipper keeps the half space where dot(plane, position) >= 0 for each UCP_ENA plane.
void EmitUserClipDistances(EmitterState& state, uint32_t position) {
	uint32_t index = 0;
	for (auto mask = state.user_clip_planes; mask != 0; mask &= mask - 1u) {
		const auto dword = IR::PushData::UserClipPlaneDword + 4u * std::countr_zero(mask);
		uint32_t   components[4] {};
		for (uint32_t i = 0; i < 4; i++) {
			components[i] = Unary(state, spv::OpBitcast, TypeF32(state),
			                      EmitPushDataLoad(state, ConstantU32(state, dword + i)));
		}
		const auto plane = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), plane,
		                          components[0], components[1], components[2], components[3]);
		state.builder.AddFunction(
		    spv::OpStore, DistancePointer(state, IR::StageOutputKind::ClipDistance, index++),
		    Binary(state, spv::OpDot, TypeF32(state), plane, position));
	}
}

void EmitAuxPositionExport(ValueEmitContext& ctx, uint32_t data, const IR::ExportInfo& exp) {
	auto& state = ctx.state;
	for (uint32_t component = 0; component < 4; component++) {
		if ((exp.en & (1u << component)) == 0) {
			continue;
		}
		const auto output = IR::DecodePositionExportComponent(
		    state.input_info.vertex->pa_cl_vs_out_cntl, exp.index, component);
		if (output.layer || output.viewport) {
			const auto raw = ExportRawComponent(ctx, data, component);
			if (output.layer) {
				const auto layer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), layer, raw,
				                          ConstantU32(state, 0x7ffu));
				const auto pointer = state.program.stage == ShaderType::Mesh
				                         ? MeshOutputPointer(state, IR::StageOutputKind::Layer)
				                         : state.layer_variable;
				state.builder.AddFunction(spv::OpStore, pointer, layer);
			}
			if (output.viewport) {
				// GFX10 MISC.z packs the viewport index in bits 16..19 alongside the layer.
				const auto viewport = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), viewport, raw,
				                          ConstantU32(state, 16), ConstantU32(state, 4));
				state.builder.AddFunction(spv::OpStore, state.viewport_index_variable, viewport);
			}
			continue;
		}
		if (!output.point_size && output.clip_distance == UINT32_MAX &&
		    output.cull_distance == UINT32_MAX) {
			continue;
		}

		const auto raw = ExportRawComponent(ctx, data, component);
		const auto f32 = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeF32(state), f32, raw);
		if (output.point_size) {
			state.builder.AddFunction(spv::OpStore, state.point_size_variable, f32);
			continue;
		}
		auto StoreDistance = [&](IR::StageOutputKind kind, uint32_t index) {
			if (index != UINT32_MAX) {
				state.builder.AddFunction(spv::OpStore, DistancePointer(state, kind, index), f32);
			}
		};
		StoreDistance(IR::StageOutputKind::ClipDistance, output.clip_distance);
		StoreDistance(IR::StageOutputKind::CullDistance, output.cull_distance);
	}
}

uint32_t ConvertClipCoordinate(EmitterState& state, uint32_t coordinate, float scale,
                               float offset, float half_extent) {
	const auto window  = state.builder.AllocateId();
	const auto biased  = state.builder.AllocateId();
	const auto divided = state.builder.AllocateId();
	const auto ndc     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFMul, TypeF32(state), window, coordinate,
	                          ConstantF32Value(state, scale));
	state.builder.AddFunction(spv::OpFAdd, TypeF32(state), biased, window,
	                          ConstantF32Value(state, offset));
	state.builder.AddFunction(spv::OpFDiv, TypeF32(state), divided, biased,
	                          ConstantF32Value(state, half_extent));
	state.builder.AddFunction(spv::OpFSub, TypeF32(state), ndc, divided,
	                          ConstantF32Value(state, 1.0f));
	return ndc;
}

uint32_t ConvertPositionToClipSpace(EmitterState& state, uint32_t position) {
	const auto& transform = state.input_info.vertex->clip_space;
	uint32_t    components[4] {};
	for (uint32_t i = 0; i < 4; i++) {
		components[i] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), components[i], position,
		                          i);
	}
	components[0] = ConvertClipCoordinate(state, components[0], transform.scale[0],
	                                      transform.offset[0], transform.half_extent[0]);
	components[1] = ConvertClipCoordinate(state, components[1], transform.scale[1],
	                                      transform.offset[1], transform.half_extent[1]);
	const auto converted = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), converted,
	                          components[0], components[1], components[2], components[3]);
	return converted;
}

} // namespace
uint32_t EmitWqmU64(EmitterState& state, uint32_t value) {
	const auto shifted_one = state.builder.AllocateId();
	const auto merged_one  = state.builder.AllocateId();
	const auto shifted_two = state.builder.AllocateId();
	const auto merged_two  = state.builder.AllocateId();
	const auto quad_bits   = state.builder.AllocateId();
	const auto result      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU64(state), shifted_one, value,
	                          ConstantU32(state, 1));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU64(state), merged_one, value, shifted_one);
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU64(state), shifted_two, merged_one,
	                          ConstantU32(state, 2));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU64(state), merged_two, merged_one,
	                          shifted_two);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU64(state), quad_bits, merged_two,
	                          ConstantU64(state, 0x1111111111111111ull));
	state.builder.AddFunction(spv::OpIMul, TypeU64(state), result, quad_bits,
	                          ConstantU64(state, 15));
	return result;
}

// Diagnostics: KYTY_PROBE_RECT=<x>,<y>,<w>,<h> (w * h <= 16384) with KYTY_PROBE_PC=<hash>@...: that
// pixel shader stores its float MRT0 export (the probed registers) for each pixel of the screen
// rectangle in the bindless feedback buffer (BindlessCaptureProbe); the host writes it to
// KYTY_PROBE_RECT_FILE. Numbers, where the on-screen probe only shows post-processed colours.
static void EmitProbeRectStore(EmitterState& state, uint32_t value) {
	struct Rect {
		uint64_t hash = 0;
		uint32_t x = 0, y = 0, w = 0, h = 0;
	};
	static const Rect rect = [] {
		Rect        result;
		const char* probe = std::getenv("KYTY_PROBE_PC");
		const char* value = std::getenv("KYTY_PROBE_RECT");
		if (probe == nullptr || value == nullptr) {
			return result;
		}
		result.hash = std::strtoull(probe, nullptr, 16);
		uint32_t numbers[4] {};
		char*    cursor = const_cast<char*>(value);
		for (auto& number: numbers) {
			number = static_cast<uint32_t>(std::strtoul(cursor, &cursor, 10));
			if (*cursor == ',') {
				cursor++;
			}
		}
		result.x = numbers[0];
		result.y = numbers[1];
		result.w = numbers[2];
		result.h = numbers[3];
		if (result.w == 0u || result.h == 0u ||
		    result.w * result.h * 4u > IR::BindlessCaptureProbeWords) {
			result.hash = 0;
		}
		return result;
	}();
	if (rect.hash == 0 || rect.hash != state.program.shader_hash ||
	    state.program.stage != ShaderType::Pixel || state.bindless_feedback_variable == 0) {
		return;
	}
	const auto frag = InputVariableForKind(state, IR::StageInputKind::FragCoord);
	if (frag == 0) {
		std::fprintf(stderr, "[probe-rect] %016llx has no FragCoord input; nothing stored\n",
		             static_cast<unsigned long long>(state.program.shader_hash));
		return;
	}
	const auto Coordinate = [&](uint32_t component, uint32_t origin) {
		const auto pointer = state.builder.AllocateId();
		const auto f32     = state.builder.AllocateId();
		const auto u32     = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain,
		                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
		                          pointer, frag, ConstantU32(state, component));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), f32, pointer);
		state.builder.AddFunction(spv::OpConvertFToU, TypeU32(state), u32, f32);
		return std::pair {u32, Binary(state, spv::OpISub, TypeU32(state), u32,
		                              ConstantU32(state, origin))};
	};
	const auto [x, dx] = Coordinate(0, rect.x);
	const auto [y, dy] = Coordinate(1, rect.y);
	const auto inside  = Binary(state, spv::OpLogicalAnd, TypeBool(state),
	                            Binary(state, spv::OpULessThan, TypeBool(state), dx,
	                                   ConstantU32(state, rect.w)),
	                            Binary(state, spv::OpULessThan, TypeBool(state), dy,
	                                   ConstantU32(state, rect.h)));
	EmitIfCondition(state, inside, [&]() {
		const auto index = Binary(state, spv::OpIAdd, TypeU32(state),
		                          Binary(state, spv::OpIMul, TypeU32(state), dy,
		                                 ConstantU32(state, rect.w)),
		                          dx);
		const auto base  = Binary(state, spv::OpIAdd, TypeU32(state),
		                          Binary(state, spv::OpIMul, TypeU32(state), index,
		                                 ConstantU32(state, 4u)),
		                          ConstantU32(state, IR::BindlessCaptureProbe));
		const auto Store = [&](uint32_t word, uint32_t bits) {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
			                          pointer, state.bindless_feedback_variable,
			                          ConstantU32(state, 0),
			                          Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                 ConstantU32(state, word)));
			state.builder.AddFunction(spv::OpStore, pointer, bits);
		};
		Store(0, Binary(state, spv::OpBitwiseOr, TypeU32(state), x,
		                Binary(state, spv::OpShiftLeftLogical, TypeU32(state), y,
		                       ConstantU32(state, 16u))));
		for (uint32_t component = 0; component < 3u; component++) {
			const auto f32  = state.builder.AllocateId();
			const auto bits = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), f32, value,
			                          component);
			state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, f32);
			Store(1u + component, bits);
		}
	});
}

// Diagnostics: with KYTY_PROBE_RECT set, a probed shader of another stage appends its parameter-0
// export (the probed registers, Attribute.cpp) per vertex: an atomic counter at
// BindlessCaptureProbe, then records of 4 words (0x80000000 | sequence, three value bits).
static void EmitProbeAppend(EmitterState& state, uint32_t data) {
	const auto pointer = [&](uint32_t index) {
		const auto element = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          element, state.bindless_feedback_variable, ConstantU32(state, 0),
		                          index);
		return element;
	};
	const auto sequence = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), sequence,
	                          pointer(ConstantU32(state, IR::BindlessCaptureProbe)),
	                          ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone), ConstantU32(state, 1u));
	constexpr uint32_t Records = (IR::BindlessCaptureProbeWords - 4u) / 4u;
	const auto slot = Binary(state, spv::OpUMod, TypeU32(state), sequence, ConstantU32(state, Records));
	const auto base = Binary(state, spv::OpIAdd, TypeU32(state),
	                         Binary(state, spv::OpIMul, TypeU32(state), slot, ConstantU32(state, 4u)),
	                         ConstantU32(state, IR::BindlessCaptureProbe + 4u));
	const auto Store = [&](uint32_t word, uint32_t value) {
		state.builder.AddFunction(
		    spv::OpStore,
		    pointer(Binary(state, spv::OpIAdd, TypeU32(state), base, ConstantU32(state, word))), value);
	};
	Store(0, Binary(state, spv::OpBitwiseOr, TypeU32(state), sequence, ConstantU32(state, 0x80000000u)));
	for (uint32_t component = 0; component < 3u; component++) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, data, component);
		Store(1u + component, value);
	}
}

void EmitSetAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state = ctx.state;
	const auto& exp   = ctx.Export(inst);
	const auto  exec  = ctx.Arg(inst, 1);
	if (state.program.stage == ShaderType::Pixel && exp.vm && state.program.info.pixel_valid_mask &&
	    state.pixel_valid_mask_variable != 0) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), value, exec, ConstantU32(state, 1),
		                          ConstantU32(state, 0));
		state.builder.AddFunction(spv::OpStore, state.pixel_valid_mask_variable, value);
	}
	if (exp.kind == IR::ExportTargetKind::Null || exp.en == 0u) {
		return;
	}
	// Skip dormant color exports after their valid mask; MRT1 is reserved for logical alpha.
	if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
	    exp.index != 0 && state.input_info.pixel->alpha_blend_source != ShaderAlphaBlendSource::None) {
		return;
	}
	EmitIfCondition(state, exec, [&]() {
		const auto data = ctx.Arg(inst, 0);
		if (state.program.stage != ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Parameter &&
		    exp.index == 0 && ProbeAppendWanted(state) && state.bindless_feedback_variable != 0) {
			EmitProbeAppend(state, data);
		}
		if (exp.kind == IR::ExportTargetKind::Primitive) {
			if (state.program.stage == ShaderType::Mesh) {
				state.builder.AddFunction(spv::OpStore, MeshPrimitivePointer(state),
				                          ExportRawComponent(ctx, data, 0));
			}
			return;
		}
		if (exp.kind == IR::ExportTargetKind::Position && exp.index != 0) {
			EmitAuxPositionExport(ctx, data, exp);
			return;
		}
		if (exp.kind == IR::ExportTargetKind::MrtZ) {
			if ((exp.en & 1u) != 0u && state.depth_variable != 0) {
				const auto raw = ExportRawComponent(ctx, data, 0);
				const auto f32 = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitcast, TypeF32(state), f32, raw);
				state.builder.AddFunction(spv::OpStore, state.depth_variable, f32);
			}
			if ((exp.en & 4u) != 0u && state.sample_mask_variable != 0) {
				const auto raw     = ExportRawComponent(ctx, data, 2);
				const auto value   = state.builder.AllocateId();
				const auto pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitcast, TypeI32(state), value, raw);
				state.builder.AddFunction(
				    spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput, TypeI32(state)),
				    pointer, state.sample_mask_variable, ConstantU32(state, 0));
				state.builder.AddFunction(spv::OpStore, pointer, value);
			}
			return;
		}
		const auto variable =
		    state.program.stage == ShaderType::Mesh ? 0u : OutputVariableForExport(state, exp);
		if (state.program.stage != ShaderType::Mesh && variable == 0) {
			return;
		}
		const bool mrt =
		    state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt;
		const bool uint_output =
		    mrt && ((state.input_info.pixel->target_uint_mask & (1u << exp.index)) != 0 ||
		            MrtOutputMode(state, exp) == 7u);
		const bool sint_output =
		    mrt && (state.input_info.pixel->target_sint_mask & (1u << exp.index)) != 0;
		const auto vector_type = uint_output   ? TypeU32Vector(state, 4)
		                         : sint_output ? TypeI32Vector(state, 4)
		                                       : TypeF32Vector(state, 4);
		auto       value       = ExportVector(ctx, data, exp, uint_output || sint_output);
		if (mrt && exp.index == 0 && !uint_output && !sint_output) {
			EmitProbeRectStore(state, value);
		}
		// KYTY_DEBUG_MAGENTA_PS=<hash>: that pixel shader's float MRT0 export becomes magenta,
		// to see where its draws land (diagnostic).
		if (static const std::string magenta = [] {
			    const char* env = std::getenv("KYTY_DEBUG_MAGENTA_PS");
			    return env != nullptr ? std::string(env) : std::string();
		    }();
		    !magenta.empty() && mrt && !uint_output && !sint_output &&
		    [&] {
			    static std::atomic<uint32_t> printed {0};
			    if (printed.fetch_add(1) < 400)
				    std::fprintf(stderr, "[magenta] ps %016llx mrt%u uint=%d\n",
				                 static_cast<unsigned long long>(state.program.shader_hash),
				                 exp.index, uint_output ? 1 : 0);
			    return true;
		    }() &&
		    magenta.find(fmt::format("{:016x}", state.program.shader_hash)) != std::string::npos) {
			value = state.builder.AllocateId();
			if (uint_output || sint_output) {
				const auto all = ConstantU32(state, 0xffffffffu);
				state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), value,
				                          all, ConstantU32(state, 0), all, all);
			} else {
				// The n-th hash in the list gets the n-th color (magenta, green, black, blue, ...),
				// to tell several shaders apart in one run.
				const auto pos   = magenta.find(fmt::format("{:016x}", state.program.shader_hash));
				const auto index = static_cast<size_t>(
				    std::count(magenta.begin(), magenta.begin() + static_cast<ptrdiff_t>(pos), ','));
				static constexpr float Colors[][3] = {{1, 0, 1}, {0, 1, 0}, {0, 0, 0}, {0, 0, 1},
				                                      {1, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 0, 0}};
				const auto& c = Colors[index % 8];
				state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), value,
				                          ConstantF32(state, c[0]), ConstantF32(state, c[1]),
				                          ConstantF32(state, c[2]), ConstantF32(state, 1.0f));
			}
		}
		if (sint_output) {
			const auto signed_value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitcast, vector_type, signed_value, value);
			value = signed_value;
			if (exp.compr) {
				const auto shift = state.builder.AllocateId();
				const auto bits  = ConstantU32(state, 16);
				state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), shift,
				                          bits, bits, bits, bits);
				const auto high = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpShiftLeftLogical, vector_type, high, value, shift);
				value = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpShiftRightArithmetic, vector_type, value, high,
				                          shift);
			}
		}
		if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
		    exp.index == 0 && !uint_output &&
		    state.input_info.pixel->alpha_blend_source != ShaderAlphaBlendSource::None) {
			// Broadcast logical alpha before swizzling the primary output.
			const auto blend_output =
			    OutputVariableForExport(state, {.kind = IR::ExportTargetKind::Mrt, .index = 1});
			if (blend_output != 0) {
				auto alpha = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpVectorShuffle, vector_type, alpha, value, value,
				                          3u, 3u, 3u, 3u);
				uint32_t alpha_factor = 0;
				switch (state.input_info.pixel->alpha_blend_source) {
					case ShaderAlphaBlendSource::SourceAlphaOne:
						alpha_factor = ConstantF32Value(state, 1.0f);
						break;
					case ShaderAlphaBlendSource::SourceAlphaZero:
						alpha_factor = ConstantF32Value(state, 0.0f);
						break;
					default: break;
				}
				if (alpha_factor != 0) {
					const auto mapping = state.input_info.pixel->target_export_mapping[0];
					for (uint32_t component = 0; component < 4; component++) {
						if (mapping.Map(component) != 3u) continue;
						const auto factors = state.builder.AllocateId();
						state.builder.AddFunction(spv::OpCompositeInsert, vector_type, factors,
						                          alpha_factor, alpha, component);
						alpha = factors;
						break;
					}
				}
				state.builder.AddFunction(spv::OpStore, blend_output, alpha);
			}
		}
		if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
		    exp.index < state.input_info.pixel->target_export_mapping.size()) {
			const auto mapping = state.input_info.pixel->target_export_mapping[exp.index];
			if (!mapping.IsIdentity()) {
				const auto mapped = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpVectorShuffle, vector_type, mapped, value, value,
				                          mapping.Map(0), mapping.Map(1), mapping.Map(2),
				                          mapping.Map(3));
				value = mapped;
			}
			if ((state.input_info.pixel->target_packed_unorm_mask & (1u << exp.index)) != 0u) {
				if (exp.compr || MrtOutputMode(state, exp) != 9u ||
				    (mapping.ApplyMask(exp.en) & 7u) != 7u) {
					EXIT("unsupported packed UNorm EXP: export=%u mode=%u en=0x%x compr=%u\n",
					     exp.index, MrtOutputMode(state, exp), exp.en, static_cast<uint32_t>(exp.compr));
				}
				const auto info = Format::GetFormatInfo(Prospero::BufferFormat::k11_11_10UNorm);
				const auto rounding = (state.input_info.pixel->target_round_mode & (1u << exp.index)) != 0
				                          ? UnormRounding::Truncate : UnormRounding::HalfUp;
				const auto raw = Unary(state, spv::OpBitcast, TypeU32Vector(state, 4), value);
				auto packed = ConstantU32(state, 0);
				for (uint32_t component = 0; component < info.component_count; component++) {
					const auto encoded = EncodePackedColorComponent(
					    state, info, component, ExportRawComponent(ctx, raw, component), 32u, rounding);
					packed = component == 0u ? encoded
					    : Binary(state, spv::OpBitwiseOr, TypeU32(state), packed, encoded);
				}
				value = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), value,
				                          packed, ConstantU32(state, 0), ConstantU32(state, 0),
				                          ConstantU32(state, 0));
			}
		}
		if (exp.kind == IR::ExportTargetKind::Position &&
		    state.input_info.vertex->clip_space.enabled) {
			value = ConvertPositionToClipSpace(state, value);
		}
		if (exp.kind == IR::ExportTargetKind::Position && state.user_clip_planes != 0) {
			EmitUserClipDistances(state, value);
		}
		if (state.program.stage == ShaderType::Mesh) {
			const auto kind = exp.kind == IR::ExportTargetKind::Position
			                      ? IR::StageOutputKind::Position
			                      : IR::StageOutputKind::Parameter;
			state.builder.AddFunction(spv::OpStore, MeshOutputPointer(state, kind, exp.index),
			                          value);
		} else if (exp.kind == IR::ExportTargetKind::Position) {
			if (state.invalid_position_clip_distance != UINT32_MAX) {
				const auto zero = state.builder.Constant(spv::OpConstantNull, TypeF32Vector(state, 4));
				const auto equal = state.builder.AllocateId();
				const auto invalid = state.builder.AllocateId();
				const auto distance = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpFOrdEqual, TypeBoolVector(state, 4), equal,
				                          value, zero);
				state.builder.AddFunction(spv::OpAll, TypeBool(state), invalid, equal);
				// Zero at valid vertices makes a primitive containing an invalid position
				// collapse to its remaining edge, before the undefined 0/0 perspective divide.
				state.builder.AddFunction(spv::OpSelect, TypeF32(state), distance, invalid,
				                          ConstantF32Value(state, -1.0f),
				                          ConstantF32Value(state, 0.0f));
				state.builder.AddFunction(spv::OpStore,
				                          DistancePointer(state, IR::StageOutputKind::ClipDistance,
				                                          state.invalid_position_clip_distance),
				                          distance);
				static std::atomic_bool logged = false;
				if (!logged.exchange(true, std::memory_order_relaxed)) {
					Log::WriteToConsoleAndLog(
					    "Shader: emitted zero-position clip guard\n");
				}
			}
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(
			    spv::OpAccessChain,
			    TypePointer(state, spv::StorageClassOutput, TypeF32Vector(state, 4)), pointer,
			    variable, ConstantU32(state, 0));
			state.builder.AddFunction(spv::OpStore, pointer, value);
		} else {
			state.builder.AddFunction(spv::OpStore, variable, value);
		}
	});
}

uint32_t EmitIdentity(ValueEmitContext&, uint32_t value) {
	return value;
}

void EmitVoid(ValueEmitContext&) {}

void EmitStoreCompletion(EmitterState& state) {
	// A split wave spans two host subgroups of its workgroup.
	NoteWaveSync(state);
	state.builder.AddFunction(
	    spv::OpControlBarrier,
	    ConstantU32(state, state.split_wave ? spv::ScopeWorkgroup : spv::ScopeSubgroup),
	    ConstantU32(state, spv::ScopeDevice),
	    ConstantU32(state, spv::MemorySemanticsReleaseMask | spv::MemorySemanticsUniformMemoryMask |
	                          spv::MemorySemanticsImageMemoryMask));
}

void EmitDataCacheInvalidate(EmitterState& state) {
	// RDNA2 GL0/GL1 invalidation makes vector memory visible without a workgroup rendezvous.
	state.builder.AddFunction(
	    spv::OpMemoryBarrier, ConstantU32(state, spv::ScopeDevice),
	    ConstantU32(state, spv::MemorySemanticsAcquireMask | spv::MemorySemanticsUniformMemoryMask |
	                          spv::MemorySemanticsImageMemoryMask));
}

void EmitBarrier(EmitterState& state) {
	const auto tessellation = state.program.stage == ShaderType::TessellationControl;
	if (!tessellation && ShaderWorkgroupInput(state.program.stage, state.input_info) == nullptr) {
		// Independent graphics invocations have no native workgroup left to synchronize.
		return;
	}
	const auto memory_scope = tessellation ? spv::ScopeInvocation : spv::ScopeWorkgroup;
	const auto memory = state.lds_storage_class == spv::StorageClassStorageBuffer
	                        ? spv::MemorySemanticsUniformMemoryMask
	                        : spv::MemorySemanticsWorkgroupMemoryMask;
	const auto semantics = tessellation ? spv::MemorySemanticsMaskNone
	                                    : spv::MemorySemanticsAcquireReleaseMask | memory;
	NoteWaveSync(state);
	state.builder.AddFunction(spv::OpControlBarrier, ConstantU32(state, spv::ScopeWorkgroup),
	                          ConstantU32(state, memory_scope), ConstantU32(state, semantics));
}

// One guest wave's LDS accesses are in program order (all LDS reads and writes of a wavefront
// complete in order), so a lane reads LDS another lane of its wave just wrote without a barrier.
// Host invocations do not run in lockstep: they need one, at the scope WaveLdsScope chose.
void EmitWaveLdsBarrier(EmitterState& state) {
	const auto memory = state.lds_storage_class == spv::StorageClassStorageBuffer
	                        ? spv::MemorySemanticsUniformMemoryMask
	                        : spv::MemorySemanticsWorkgroupMemoryMask;
	state.builder.AddFunction(
	    spv::OpControlBarrier, ConstantU32(state, state.wave_lds_scope),
	    ConstantU32(state, spv::ScopeWorkgroup),
	    ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | memory));
}

uint32_t EmitLaneId(EmitterState& state) {
	return state.program.stage == ShaderType::TessellationControl
	           ? EmitBuiltinU32(state, IR::StageInputKind::InvocationId, 0)
	           : EmitSubgroupLocalInvocationId(state);
}

uint32_t EmitMeshDrawParameter(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state  = ctx.state;
	const auto index  = inst.Arg(0).U32();
	if (state.program.stage != ShaderType::Mesh || index >= IR::PushData::MeshDrawDwordCount) {
		ctx.Fail(inst, "invalid mesh draw parameter");
	}
	return EmitPushDataLoad(state, ConstantU32(state, index));
}

uint32_t EmitGetUserData(EmitterState& state, IR::ScalarReg reg) {

	uint32_t dword = 0;
	if (!UserDataDwordIndex(state, reg, dword)) {
		return ConstantU32(state, 0);
	} else {
		return EmitShaderDataDwordLoad(state, dword);
	}
}

uint32_t EmitGetBuiltin(ValueEmitContext& ctx, IR::Value kind, IR::Value index) {
	return EmitBuiltinU32(ctx.state, static_cast<IR::StageInputKind>(kind.U32()), index.U32());
}

uint32_t EmitGetDispatchThreadExtent(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto start = ctx.state.program.bindings.dispatch_thread_dword;
	if (start == IR::PushData::NoStart || !inst.Arg(0).IsImmediate() || inst.Arg(0).U32() >= 3u) {
		ctx.Fail(inst, "invalid dispatch thread extent");
	}
	if (DispatchDimensionsIndirect(ctx.state)) {
		return EmitBuiltinU32(ctx.state, IR::StageInputKind::DispatchThreadCount, inst.Arg(0).U32());
	}
	return EmitShaderDataDwordLoad(ctx.state, start + inst.Arg(0).U32());
}

uint32_t EmitUndefU1(EmitterState& state, const IR::Inst& inst) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpUndef, TypeId(state, inst.GetType()), result);
	return result;
}

// Reads `value` from `lane` of a split wave's 32-lane half: within the invocation's own host
// subgroup, for operations that stay within rows or halves of the wave.
uint32_t ReadHostSubgroupLane(EmitterState& state, uint32_t type, uint32_t value, uint32_t lane) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, result,
	                          ConstantU32(state, spv::ScopeSubgroup), value,
	                          EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31)));
	return result;
}

uint32_t EmitDppMoveU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state    = ctx.state;
	const auto flags    = inst.Flags<IR::DppMoveFlags>();
	const auto target   = EmitDppTargetLane(state, flags);
	// DPP reads within rows of 16 lanes, which never leave a split wave's half.
	const auto shuffled = state.split_wave
	                          ? ReadHostSubgroupLane(state, TypeU32(state), ctx.Arg(inst, 0),
	                                                 target.lane)
	                          : ctx.Shuffle(inst, 0, target.lane);
	if (flags.fetch_inactive) {
		return shuffled;
	}
	uint32_t ballot = 0, source_lane = target.lane;
	if (state.split_wave) {
		ballot = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot,
		                          ConstantU32(state, spv::ScopeSubgroup), ctx.Arg(inst, 1));
		source_lane = EmitBinaryU32(state, spv::OpBitwiseAnd, target.lane, ConstantU32(state, 31));
	} else {
		ballot = ctx.Ballot(inst.Arg(1));
	}
	const auto source_active = EmitBallotLaneActiveBool(state, ballot, source_lane);
	const auto can_fetch     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), can_fetch, target.valid,
	                          source_active);
	return EmitNative<spv::OpSelect, IR::Type::U32>(ctx.state, can_fetch, shuffled,
	                                                ConstantU32(state, 0));
}

uint32_t EmitDppUpdateU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto flags = inst.Flags<IR::DppMoveFlags>();
	auto       write = EmitDppWriteCondition(ctx, flags, ctx.Arg(inst, 2));
	if (!flags.bound_control && !flags.fetch_inactive && !flags.dpp8) {
		// A source lane EXEC disables is invalid like a vacated one, and without bound_ctrl the
		// receiving lane keeps its value; so does a lane the host subgroup
		// lacks. EmitDppMoveU32 read zero from it.
		const auto target        = EmitDppTargetLane(ctx.state, flags);
		const auto source_active = EmitBallotLaneActiveBool(ctx.state, ctx.Ballot(inst.Arg(2)),
		                                                    target.lane);
		write = Binary(ctx.state, spv::OpLogicalAnd, TypeBool(ctx.state), write, source_active);
	}
	return EmitNative<spv::OpSelect, IR::Type::U32>(ctx.state, write, ctx.Arg(inst, 0),
	                                                ctx.Arg(inst, 1));
}

uint32_t EmitConditionRef(ValueEmitContext& ctx, const IR::Inst& inst) {
	// Scalar branches include lanes disabled by EXEC, which the branch body may restore.
	if (ctx.other_half != nullptr && ctx.half != 0) return ctx.other_half->Result(inst);
	const auto kind = inst.Flags<CFG::BranchCondition>();
	if (kind == CFG::BranchCondition::ScalarInstruction) return ctx.Arg(inst, 0);
	const bool zero = kind == CFG::BranchCondition::ExecZero ||
	                  kind == CFG::BranchCondition::VccZero || kind == CFG::BranchCondition::SccZero;
	if (ctx.state.wave_local_insts.contains(&inst)) {
		// No block this branch controls synchronizes the halves, so a half may branch on its own
		// lanes: one whose lanes are all inactive skips code the guest ran with them masked.
		const auto word = EmitHostBallotWord(ctx.state, ctx.Arg(inst, 0));
		return Binary(ctx.state, zero ? spv::OpIEqual : spv::OpINotEqual, TypeBool(ctx.state),
		              word, ConstantU32(ctx.state, zero ? ~0u : 0u));
	}
	if (ctx.state.split_wave) {
		// A subgroup vote sees one half; the ballot covers the whole wave.
		const auto ballot = ctx.Ballot(inst.Arg(0));
		const auto reference = zero ? ctx.Ballot(IR::Value(true))
		                            : ConstantU32CompositeZero(ctx.state, 4);
		const auto compare = Binary(ctx.state, zero ? spv::OpIEqual : spv::OpINotEqual,
		                            TypeBoolVector(ctx.state, 4), ballot, reference);
		return Unary(ctx.state, zero ? spv::OpAll : spv::OpAny, TypeBool(ctx.state), compare);
	}
	// Zero-kind predicates are already inverted by translation.
	auto predicate = ctx.Arg(inst, 0);
	if (ctx.other_half != nullptr) {
		predicate = Binary(ctx.state, zero ? spv::OpLogicalAnd : spv::OpLogicalOr,
		                   TypeBool(ctx.state), predicate, ctx.HalfArg(inst, 0, 1));
	}
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(zero ? spv::OpGroupNonUniformAll : spv::OpGroupNonUniformAny,
	                              TypeBool(ctx.state), result,
	                              ConstantU32(ctx.state, spv::ScopeSubgroup), predicate);
	return result;
}

uint32_t EmitBallot(ValueEmitContext& ctx, IR::Value predicate) {
	return ctx.Ballot(predicate);
}

uint32_t EmitReadFirstLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (ctx.state.wave_local_insts.contains(&inst)) {
		// A waterfall loop handles the lanes whose value matches the one read and repeats for
		// the rest, so each half of a split wave can read its own first active lane.
		auto&      state = ctx.state;
		const auto first = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpExtInst, TypeU32(state), first, GlslStd450(state),
		                          GLSLstd450FindILsb, EmitHostBallotWord(state, ctx.Arg(inst, 1)));
		return ReadHostSubgroupLane(state, TypeU32(state), ctx.Arg(inst, 0), first);
	}
	// KYTY_RFL_IDENTITY_HASHES=<hash,...>: in those shaders each lane keeps its own value, for
	// values the game expects to be wave-uniform that the host's lane packing makes divergent
	// (experiment).
	if (static const std::string identity = [] {
		    const char* env = std::getenv("KYTY_RFL_IDENTITY_HASHES");
		    return env != nullptr ? std::string(env) : std::string();
	    }();
	    !identity.empty() &&
	    ((identity == "ps" && ctx.state.program.stage == ShaderType::Pixel) ||
	     identity.find(fmt::format("{:016x}", ctx.state.program.shader_hash)) != std::string::npos)) {
		return ctx.Def(inst.Arg(0));
	}
	const auto ballot = ctx.Ballot(inst.Arg(1));
	const auto lane   = ctx.FirstLane(ballot);
	return ctx.Shuffle(inst, 0, lane);
}

// The lanes' reduction, natively over the lanes the host subgroup has (IR::MatchLaneReduction).
static uint32_t EmitLaneReduction(ValueEmitContext& ctx, const IR::LaneReduction& reduction) {
	auto& state = ctx.state;
	// A wave64 on a 32-wide subgroup keeps lanes 32-63 in the second half.
	const auto host_lanes = state.lane_count == 2 ? 32u : state.program.wave_size;
	auto&      lane = reduction.first_lane / host_lanes == ctx.half ? ctx : *ctx.other_half;
	const auto subid = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), subid,
	                          state.subgroup_local_invocation_id_variable);
	const auto in_range =
	    Binary(state, spv::OpULessThan, TypeBool(state),
	           Binary(state, spv::OpISub, TypeU32(state), subid,
	                  ConstantU32(state, reduction.first_lane % host_lanes)),
	           ConstantU32(state, reduction.lanes));
	const auto contribution =
	    Select(state, TypeU32(state), in_range, lane.Def(reduction.source),
	           ConstantU32(state, IR::ReductionIdentity(reduction.operation)));
	spv::Op operation = spv::OpGroupNonUniformIAdd;
	switch (reduction.operation) {
		case IR::ValueOpcode::UMax32: operation = spv::OpGroupNonUniformUMax; break;
		case IR::ValueOpcode::UMin32: operation = spv::OpGroupNonUniformUMin; break;
		case IR::ValueOpcode::SMax32: operation = spv::OpGroupNonUniformSMax; break;
		case IR::ValueOpcode::SMin32: operation = spv::OpGroupNonUniformSMin; break;
		case IR::ValueOpcode::BitwiseOr32: operation = spv::OpGroupNonUniformBitwiseOr; break;
		case IR::ValueOpcode::BitwiseAnd32: operation = spv::OpGroupNonUniformBitwiseAnd; break;
		case IR::ValueOpcode::BitwiseXor32: operation = spv::OpGroupNonUniformBitwiseXor; break;
		default: break;
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(operation, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), spv::GroupOperationReduce,
	                          contribution);
	return result;
}

uint32_t EmitReadLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (const auto reduction = IR::MatchLaneReduction(inst, ctx.state.program.wave_size)) {
		return EmitLaneReduction(ctx, *reduction);
	}
	return ctx.Shuffle(inst, 0, ctx.Arg(inst, 1));
}

uint32_t EmitSubgroupUMinU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	auto value = ctx.Arg(inst, 0);
	if (ctx.other_half != nullptr) {
		value = EmitUMin32(state, ctx.HalfArg(inst, 0, 0), ctx.HalfArg(inst, 0, 1));
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformUMin, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), spv::GroupOperationReduce, value);
	if (state.split_wave) {
		// Each host subgroup reduces its half; the other half's result arrives through the
		// exchange, from the matching lane of that half.
		const auto other = ExchangeWaveValue(
		    state, result,
		    EmitBinaryU32(state, spv::OpBitwiseXor, EmitLocalInvocationIndex(state),
		                  ConstantU32(state, 32)));
		return EmitUMin32(state, result, other);
	}
	return result;
}

uint32_t EmitWriteLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto hit   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), hit,
	                          EmitSubgroupLocalInvocationId(state), ctx.Arg(inst, 2));
	return EmitNative<spv::OpSelect, IR::Type::U32>(ctx.state, hit, ctx.Arg(inst, 1),
	                                                ctx.Arg(inst, 0));
}

uint32_t EmitPermlane16U32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state     = ctx.state;
	const auto flags     = inst.Flags<IR::PermlaneFlags>();
	const auto subid     = EmitSubgroupLocalInvocationId(state);
	const auto row       = state.builder.AllocateId();
	const auto row_value = state.builder.AllocateId();
	const auto lane      = state.builder.AllocateId();
	const auto lane8     = state.builder.AllocateId();
	const auto shift     = state.builder.AllocateId();
	const auto upper     = state.builder.AllocateId();
	const auto selected  = state.builder.AllocateId();
	const auto shifted   = state.builder.AllocateId();
	const auto index     = state.builder.AllocateId();
	const auto target    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row, subid,
	                          ConstantU32(state, 0xfffffff0u));
	if (flags.x16) {
		state.builder.AddFunction(spv::OpBitwiseXor, TypeU32(state), row_value, row,
		                          ConstantU32(state, 16));
	} else {
		state.builder.AddFunction(spv::OpCopyObject, TypeU32(state), row_value, row);
	}
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane, subid,
	                          ConstantU32(state, 15));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane8, lane,
	                          ConstantU32(state, 7));
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), shift, lane8,
	                          ConstantU32(state, 2));
	state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), upper, lane,
	                          ConstantU32(state, 8));
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), selected, upper, ctx.Arg(inst, 2),
	                          ctx.Arg(inst, 1));
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), shifted, selected, shift);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), index, shifted,
	                          ConstantU32(state, 15));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, row_value, index);
	// Permlane16 reads within the caller's 32-lane half, a split wave's host subgroup.
	const auto read = [&](size_t argument) {
		return state.split_wave
		           ? ReadHostSubgroupLane(state, TypeId(state, inst.Arg(argument).GetType()),
		                                  ctx.Arg(inst, argument), target)
		           : ctx.Shuffle(inst, argument, target);
	};
	const auto shuffled = read(0);
	uint32_t   result   = shuffled;
	if (!flags.fetch_inactive) {
		const auto source_exec = read(3);
		result                 = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), result, source_exec, shuffled,
		                          ConstantU32(state, 0));
	}
	return result;
}

uint32_t EmitGetAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitAttribute(ctx.state, inst.Arg(0).U32(), inst.Arg(1).U32());
}

uint32_t EmitGetAttributeF16(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto attr = inst.Arg(0).U32();
	return EmitPixelAttribute(ctx.state, *InputBindingForParameter(ctx.state, attr), attr,
	                          inst.Arg(1).U32(), inst.Arg(2).U32());
}

uint32_t EmitGetInterpolationParameter(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitInterpolationParameter(ctx, inst.Arg(0).U32(), inst.Arg(1).U32(), inst.Arg(2).U32());
}

uint32_t EmitGetShaderBase(ValueEmitContext& ctx) {
	// Guest S_GETPC values stay shader-relative in SPIR-V, matching the runtime ABI. The
	// runtime descriptor evaluator supplies the mapped shader base for host-side planning.
	return ctx.Def(IR::Value(uint64_t {0}));
}

uint32_t EmitReadClockRealtime64(ValueEmitContext& ctx) {
	// S_MEMREALTIME is a free-running counter at 100 MHz. Read the device clock once as a
	// uvec2 so the two halves cannot tear, then divide it down to the guest's rate: by 10 for a
	// 1 GHz clock (NVIDIA), not at all for a 100 MHz one (AMD). Guest timeouts and frame pacers
	// compare differences of this counter against intervals in 100 MHz ticks, so a clock that ran
	// 25 % fast (shifting a 1 GHz clock by 3) made them expire early.
	auto&      state = ctx.state;
	const auto clock = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpReadClockKHR, TypeU32Vector(state, 2), clock,
	                          ConstantU32(state, spv::ScopeDevice));
	const auto low  = state.builder.AllocateId();
	const auto high = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, clock, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, clock, 1);
	const uint32_t divisor = GetDeviceClockDivisor();
	if (divisor == 1) {
		return PackU64(state, low, high);
	}
	if ((divisor & (divisor - 1u)) != 0u) {
		return Binary(state, spv::OpUDiv, TypeU64(state), PackU64(state, low, high),
		              ConstantU64(state, divisor));
	}
	uint32_t ClockShift = 0;
	while ((1u << ClockShift) != divisor) {
		ClockShift++;
	}
	const auto low_shifted = Binary(state, spv::OpShiftRightLogical, TypeU32(state), low,
	                                ConstantU32(state, ClockShift));
	const auto high_carried = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), high,
	                                 ConstantU32(state, 32u - ClockShift));
	const auto low_result =
	    Binary(state, spv::OpBitwiseOr, TypeU32(state), low_shifted, high_carried);
	const auto high_result = Binary(state, spv::OpShiftRightLogical, TypeU32(state), high,
	                                ConstantU32(state, ClockShift));
	return PackU64(state, low_result, high_result);
}

void EmitUnreachable(ValueEmitContext& ctx, const IR::Inst& inst) {
	ctx.Fail(inst, "must be lowered before SPIR-V emission");
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
