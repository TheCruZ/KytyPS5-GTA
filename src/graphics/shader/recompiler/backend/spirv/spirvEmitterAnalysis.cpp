#include "common/assert.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

uint32_t PixelParameterLocation(const EmitterState& state, uint32_t attr) {
	std::array<uint32_t, 32> active_inputs {};
	uint32_t                 active_count = 0;
	for (const auto& input: state.inputs) {
		if (input.kind == IR::StageInputKind::Parameter) {
			active_inputs[active_count++] = input.location;
		}
	}
	return state.program.stage == ShaderType::Pixel
	           ? ShaderPixelParameterLocation(*state.input_info.pixel,
	                                          {active_inputs.data(), active_count}, attr)
	           : attr;
}

bool PixelParameterIsFlat(const EmitterState& state, uint32_t attr) {
	return state.program.stage == ShaderType::Pixel &&
	       ShaderPixelParameterIsFlat(*state.input_info.pixel, attr);
}

bool PixelParameterIsCustom(const EmitterState& state, uint32_t attr) {
	return state.program.stage == ShaderType::Pixel &&
	       ShaderPixelParameterIsCustom(*state.input_info.pixel, attr);
}

uint32_t OutputVariableForExport(const EmitterState& state, const IR::ExportInfo& exp) {
	if (exp.kind == IR::ExportTargetKind::Position && exp.index == 0) {
		return state.per_vertex_variable;
	}
	if (exp.kind == IR::ExportTargetKind::MrtZ) {
		return state.depth_variable;
	}
	for (const auto& binding: state.outputs) {
		const auto expected_kind = exp.kind == IR::ExportTargetKind::Mrt
		                               ? IR::StageOutputKind::Mrt
		                               : IR::StageOutputKind::Parameter;
		if (binding.kind == expected_kind && binding.index == exp.index) {
			return binding.variable_id;
		}
	}
	return 0;
}

uint32_t          ConstantU32(EmitterState& state, uint32_t value);
[[noreturn]] void ExitDescriptorBindingFailure(const EmitterState&       state,
                                               IR::DescriptorBindingKind kind, uint32_t resource,
                                               const char* reason) {
	EXIT("shader binding resolution failed during SPIR-V emit: hash=0x%016" PRIx64
	     " stage=%u resource=%" PRIu32 " binding_kind=%u reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage), resource,
	     static_cast<unsigned>(kind), reason);
	std::abort();
}

uint32_t DescriptorElementPointer(EmitterState& state, uint32_t result_ptr_type,
                                  uint32_t variable_id, uint32_t array_index,
                                  IR::DescriptorBindingKind kind, uint32_t resource,
                                  const char* variable_name) {
	if (variable_id == 0) {
		ExitDescriptorBindingFailure(state, kind, resource, variable_name);
	}
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, result_ptr_type, pointer, variable_id,
	                          array_index);
	return pointer;
}

const ImageDimensionInfo& ImageDimensionInfoFor(ImageDimension dimension) {
	for (const auto& info: ImageDimensions) {
		if (info.dimension == dimension) {
			return info;
		}
	}
	EXIT("SPIR-V image dimension %u is invalid\n", static_cast<uint32_t>(dimension));
	std::abort();
}

uint32_t ImageScalarType(EmitterState& state, Prospero::TextureNumericClass numeric_class) {
	switch (numeric_class) {
		case Prospero::TextureNumericClass::Float: return TypeF32(state);
		case Prospero::TextureNumericClass::Uint: return TypeU32(state);
		case Prospero::TextureNumericClass::Sint: return TypeI32(state);
		case Prospero::TextureNumericClass::Unsupported: break;
	}
	EXIT("invalid image numeric class");
}

uint32_t ImageVectorType(EmitterState& state, Prospero::TextureNumericClass numeric_class,
                         uint32_t components) {
	return state.builder.Type(spv::OpTypeVector, ImageScalarType(state, numeric_class), components);
}

uint32_t ImageType(EmitterState& state, const IR::ImageResource& image) {
	uint32_t sampled = 0;
	uint32_t format  = spv::ImageFormatUnknown;
	if (image.resource_class == IR::ImageResourceClass::Sampled) {
		EXIT_IF(image.atomic);
		sampled = 1;
	} else if (image.resource_class == IR::ImageResourceClass::Storage) {
		EXIT_IF(image.numeric_class == Prospero::TextureNumericClass::Sint ||
		        image.numeric_class == Prospero::TextureNumericClass::Unsupported);
		sampled = 2;
		if (image.atomic) {
			EXIT_IF(image.numeric_class != Prospero::TextureNumericClass::Uint);
			format = image.atomic64 ? spv::ImageFormatR64ui : spv::ImageFormatR32ui;
			if (image.atomic64) {
				state.builder.RequireExtension("SPV_EXT_shader_image_int64");
				state.builder.RequireCapability(spv::CapabilityInt64);
				state.builder.RequireCapability(spv::CapabilityInt64Atomics);
				state.builder.RequireCapability(spv::CapabilityInt64ImageEXT);
			}
		}
	} else {
		EXIT("invalid image resource class");
	}
	const auto& info = ImageDimensionInfoFor(image.dimension);
	const auto  scalar_type =
	    image.atomic64 ? TypeU64(state) : ImageScalarType(state, image.numeric_class);
	return state.builder.Type(spv::OpTypeImage, scalar_type,
	                          info.spirv_dimension, image.depth_compare ? 1u : 0u, info.arrayed,
	                          info.multisampled, sampled, format);
}

uint32_t ImageViewSizeType(EmitterState& state, ImageDimension dimension) {
	switch (ImageDimensionInfoFor(dimension).coordinate_components) {
		case 1u: return TypeU32(state);
		case 2u: return TypeU32Vector(state, 2);
		case 3u: return TypeU32Vector(state, 3);
		default: return 0;
	}
}

namespace {
void DecorateNonUniformDescriptor(EmitterState& state, uint32_t id, spv::Capability capability) {
	state.builder.RequireExtension("SPV_EXT_descriptor_indexing");
	state.builder.RequireCapability(spv::CapabilityShaderNonUniform);
	state.builder.RequireCapability(capability);
	state.builder.AddAnnotation(spv::OpDecorate, id, spv::DecorationNonUniform);
}
} // namespace

uint32_t LoadImageDescriptor(EmitterState& state, uint32_t resource, uint32_t mip,
                             uint32_t array_index) {
	const auto pointer = ImageDescriptorPointer(state, resource, mip, array_index);
	const auto image = state.builder.AllocateId();
	const auto kind = *IR::DescriptorBindingForImage(state.program.info.images[resource]);
	state.builder.AddFunction(spv::OpLoad, state.images[IR::ImageBindingIndex(kind)].type,
	                          image, pointer);
	if (array_index != 0u) {
		DecorateNonUniformDescriptor(state, image,
		    IR::ImageBindingResourceClass(kind) == IR::ImageResourceClass::Storage
		        ? spv::CapabilityStorageImageArrayNonUniformIndexing
		        : spv::CapabilitySampledImageArrayNonUniformIndexing);
	}
	return image;
}

uint32_t LoadSamplerDescriptor(EmitterState& state, uint32_t sampler, uint32_t array_index) {
	if (sampler >= state.program.info.samplers.size()) {
		ExitDescriptorBindingFailure(state, IR::DescriptorBindingKind::Samplers, sampler,
		                             "sampler resource index is out of range");
	}
	const auto pointer = DescriptorElementPointer(
	    state, state.sampler_pointer_type, state.sampler_variable,
	    array_index == 0u ? ConstantU32(state, sampler) : array_index,
	    IR::DescriptorBindingKind::Samplers, sampler, "sampler descriptor array was not emitted");
	const auto sampler_id = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, state.sampler_type, sampler_id, pointer);
	if (array_index != 0u)
		DecorateNonUniformDescriptor(state, sampler_id, spv::CapabilitySampledImageArrayNonUniformIndexing);
	return sampler_id;
}

uint32_t MakeSampledImage(EmitterState& state, uint32_t resource, uint32_t sampler_id, uint32_t mip,
                          uint32_t array_index, bool sampler_dynamic) {
	const auto& image_resource = state.program.info.images.at(resource);
	EXIT_IF(image_resource.resource_class != IR::ImageResourceClass::Sampled);
	const auto  image          = LoadImageDescriptor(state, resource, mip, array_index);
	const auto  sampled_image = state.builder.AllocateId();
	const auto kind = *IR::DescriptorBindingForImage(image_resource);
	auto& sampled_type = state.images[IR::ImageBindingIndex(kind)].sampled_type;
	if (sampled_type == 0) {
		sampled_type = state.builder.Type(
		    spv::OpTypeSampledImage, state.images[IR::ImageBindingIndex(kind)].type);
	}
	state.builder.AddFunction(spv::OpSampledImage, sampled_type, sampled_image, image, sampler_id);
	if (array_index != 0u || sampler_dynamic) {
		DecorateNonUniformDescriptor(state, sampled_image, spv::CapabilitySampledImageArrayNonUniformIndexing);
	}
	return sampled_image;
}

uint32_t ImageDescriptorPointer(EmitterState& state, uint32_t resource, uint32_t mip,
                                uint32_t array_index) {
	const auto& image = state.program.info.images.at(resource);
	EXIT_IF(mip >= image.mip_count);
	const auto kind = IR::DescriptorBindingForImage(image);
	EXIT_IF(!kind.has_value());
	if (array_index == 0u) {
		array_index = ConstantU32(state, image.descriptor_index + mip);
	}
	const auto& definition = state.images[IR::ImageBindingIndex(*kind)];
	return DescriptorElementPointer(state, definition.pointer_type, definition.variable,
	                                array_index, *kind, resource,
	                                "image descriptor array was not emitted");
}

void EmitStorageImageWrite(EmitterState& state, uint32_t resource, uint32_t mip_lod, uint32_t coord,
                           uint32_t texel, uint32_t array_index) {
	const auto& image = state.program.info.images.at(resource);
	EXIT_IF(image.resource_class != IR::ImageResourceClass::Storage);
	if (!image.atomic) {
		state.builder.RequireCapability(spv::CapabilityStorageImageWriteWithoutFormat);
	}
	const auto EmitWrite = [&](uint32_t mip) {
		state.builder.AddFunction(spv::OpImageWrite, LoadImageDescriptor(state, resource, mip, array_index),
		                          coord, texel);
	};
	if (image.mip_mode != IR::ImageMipMode::Dynamic) {
		EmitWrite(0);
		return;
	}
	EmitIndexSwitch(state, mip_lod, image.mip_count, 0, EmitWrite);
}

spv::ExecutionModel ExecutionModelForStage(ShaderType stage) {
	switch (stage) {
		case ShaderType::Local:
		case ShaderType::Vertex: return spv::ExecutionModelVertex;
		case ShaderType::TessellationControl: return spv::ExecutionModelTessellationControl;
		case ShaderType::TessellationEvaluation: return spv::ExecutionModelTessellationEvaluation;
		case ShaderType::Mesh: return spv::ExecutionModelMeshEXT; // MeshEXT
		case ShaderType::Pixel: return spv::ExecutionModelFragment;
		default: return spv::ExecutionModelGLCompute;
	}
}

namespace {

// A branch on whether any lane's predicate is set, which the guest takes or skips for the whole
// wave. Scalar subvector loops branch on a uniform value instead.
bool IsWaveCondition(const IR::Inst& inst) {
	if (inst.GetOpcode() != IR::ValueOpcode::ConditionRef) {
		return false;
	}
	switch (inst.Flags<CFG::BranchCondition>()) {
		case CFG::BranchCondition::SccZero:
		case CFG::BranchCondition::SccNonZero:
		case CFG::BranchCondition::VccZero:
		case CFG::BranchCondition::VccNonZero:
		case CFG::BranchCondition::ExecZero:
		case CFG::BranchCondition::ExecNonZero: return true;
		default: return false;
	}
}

// A waterfall loop's read: the loop compares each lane's value with the one read, handles the
// lanes that match and repeats for the rest.
bool IsWaterfallRead(const IR::Inst& inst) {
	if (inst.GetOpcode() != IR::ValueOpcode::ReadFirstLane) {
		return false;
	}
	const auto source = inst.Arg(0).Resolve();
	return !source.IsImmediate() && std::ranges::any_of(inst.Uses(), [&](const IR::Use& use) {
		return use.user->GetOpcode() == IR::ValueOpcode::IEqual32 && use.operand < 2 &&
		       use.user->Arg(1 - use.operand).Resolve() == source;
	});
}

// A set of wave conditions, by index.
struct ConditionSet {
	std::vector<uint64_t> words;

	explicit ConditionSet(size_t count = 0): words((count + 63) / 64) {}

	void Set(size_t index) { words[index / 64] |= uint64_t {1} << (index % 64); }
	bool Test(size_t index) const { return (words[index / 64] >> (index % 64) & 1u) != 0; }
	bool Merge(const ConditionSet& other) {
		bool changed = false;
		for (size_t word = 0; word < words.size() && word < other.words.size(); word++) {
			changed |= (other.words[word] & ~words[word]) != 0;
			words[word] |= other.words[word];
		}
		return changed;
	}
};

// The immediate post-dominator of each block, with blocks.size() standing for the exit, or an
// empty result when some block never reaches the exit.
std::vector<size_t> PostDominators(const IR::BlockList&                            blocks,
                                   const std::unordered_map<const IR::Block*, size_t>& index) {
	const auto count = blocks.size();
	const auto exit  = count;
	constexpr auto none = SIZE_MAX;
	// The post-dominators are the dominators of the reversed graph, rooted at the exit.
	const auto successors = [&](size_t block) {
		std::vector<size_t> result;
		if (block == exit) {
			return result;
		}
		for (const auto* successor: blocks[block]->ImmSuccessors()) {
			result.push_back(index.at(successor));
		}
		if (result.empty()) {
			result.push_back(exit);
		}
		return result;
	};
	std::vector<std::vector<size_t>> reverse(count + 1);
	for (size_t block = 0; block < count; block++) {
		for (const auto successor: successors(block)) {
			reverse[successor].push_back(block);
		}
	}
	std::vector<size_t> order(count + 1, none);
	std::vector<size_t> postorder;
	std::vector<std::pair<size_t, size_t>> stack {{exit, 0}};
	order[exit] = 0;
	while (!stack.empty()) {
		auto& [node, next] = stack.back();
		if (next < reverse[node].size()) {
			const auto child = reverse[node][next++];
			if (order[child] == none) {
				order[child] = 0;
				stack.emplace_back(child, 0);
			}
			continue;
		}
		order[node] = postorder.size();
		postorder.push_back(node);
		stack.pop_back();
	}
	if (postorder.size() != count + 1) {
		return {};
	}
	std::vector<size_t> idom(count + 1, none);
	idom[exit] = exit;
	const auto intersect = [&](size_t lhs, size_t rhs) {
		while (lhs != rhs) {
			while (order[lhs] < order[rhs]) lhs = idom[lhs];
			while (order[rhs] < order[lhs]) rhs = idom[rhs];
		}
		return lhs;
	};
	for (bool changed = true; changed;) {
		changed = false;
		for (auto node = postorder.rbegin() + 1; node != postorder.rend(); ++node) {
			auto dominator = none;
			for (const auto successor: successors(*node)) {
				if (idom[successor] != none) {
					dominator = dominator == none ? successor : intersect(successor, dominator);
				}
			}
			if (idom[*node] != dominator) {
				idom[*node] = dominator;
				changed     = true;
			}
		}
	}
	return idom;
}

} // namespace

std::unordered_set<const IR::Inst*> SplitWaveLocalInstructions(
    const IR::Program& program, const std::unordered_set<const IR::Block*>& sync_blocks, bool all) {
	// Each half of a split wave may branch on its own lanes where the guest branched on all of
	// them: a half whose lanes are all inactive then skips code the guest ran with them masked,
	// which only changes what the guest's compiler already treats as undefined there. The halves
	// must still meet at every barrier together, so a branch stays whole-wave when a block it
	// controls synchronizes them, directly or through another whole-wave branch.
	std::unordered_set<const IR::Inst*> result;
	std::vector<const IR::Inst*>        conditions;
	std::unordered_map<const IR::Inst*, size_t> condition_index;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (IsWaterfallRead(inst)) {
				result.insert(&inst);
			} else if (IsWaveCondition(inst)) {
				condition_index.emplace(&inst, conditions.size());
				conditions.push_back(&inst);
			}
		}
	}
	if (all || program.dispatcher_fallback) {
		if (all) {
			result.insert(conditions.begin(), conditions.end());
		}
		return result;
	}

	const auto& blocks = program.blocks;
	const auto  count  = blocks.size();
	std::unordered_map<const IR::Block*, size_t> index;
	for (size_t block = 0; block < count; block++) {
		index.emplace(blocks[block], block);
	}
	const auto idom = PostDominators(blocks, index);
	if (idom.empty()) {
		return result;
	}
	// A block depends on a branch when one of the branch's successors leads to it but the
	// branch's block does not always reach it.
	std::vector<std::vector<size_t>> controllers(count);
	for (size_t block = 0; block < count; block++) {
		const auto successors = blocks[block]->ImmSuccessors();
		if (successors.size() < 2) {
			continue;
		}
		for (const auto* successor: successors) {
			for (auto runner = index.at(successor); runner != idom[block] && runner != count;
			     runner = idom[runner]) {
				if (std::ranges::find(controllers[runner], block) == controllers[runner].end()) {
					controllers[runner].push_back(block);
				}
			}
		}
	}

	std::vector<bool> local(conditions.size(), true);
	for (;;) {
		// The local conditions each branch condition and each block's execution depends on.
		std::unordered_map<const IR::Inst*, ConditionSet> values;
		std::vector<ConditionSet> control(count, ConditionSet(conditions.size()));
		const auto value_of = [&](IR::Value value) {
			value = value.Resolve();
			const auto* inst = value.IsImmediate() ? nullptr : value.TryInstruction();
			const auto  found = inst != nullptr ? values.find(inst) : values.end();
			return found != values.end() ? found->second : ConditionSet(conditions.size());
		};
		const auto branch_of = [&](size_t block) {
			const auto* source = blocks[block];
			return source->terminator.kind == CFG::TerminatorKind::ConditionalBranch &&
			               !source->condition.IsEmpty()
			           ? value_of(source->condition)
			           : ConditionSet(conditions.size());
		};
		for (bool changed = true; changed;) {
			changed = false;
			for (size_t block = 0; block < count; block++) {
				for (const auto controller: controllers[block]) {
					changed |= control[block].Merge(branch_of(controller));
					changed |= control[block].Merge(control[controller]);
				}
				for (const auto& inst: *blocks[block]) {
					ConditionSet set(conditions.size());
					switch (inst.GetOpcode()) {
						case IR::ValueOpcode::ConditionRef:
							if (const auto found = condition_index.find(&inst);
							    found != condition_index.end() && local[found->second]) {
								set.Set(found->second);
							}
							break;
						case IR::ValueOpcode::Phi:
							if (inst.GetType() != IR::Type::U1) {
								continue;
							}
							// A merged value also depends on which predecessor the wave came from.
							for (size_t arg = 0; arg < inst.NumArgs(); arg++) {
								set.Merge(value_of(inst.Arg(arg)));
								const auto predecessor = index.at(inst.PhiBlock(arg));
								set.Merge(control[predecessor]);
								set.Merge(branch_of(predecessor));
							}
							break;
						case IR::ValueOpcode::LogicalAnd:
						case IR::ValueOpcode::LogicalOr:
						case IR::ValueOpcode::LogicalXor:
						case IR::ValueOpcode::LogicalNot:
						case IR::ValueOpcode::SelectU1:
						case IR::ValueOpcode::Identity:
							for (size_t arg = 0; arg < inst.NumArgs(); arg++) {
								set.Merge(value_of(inst.Arg(arg)));
							}
							break;
						default: continue;
					}
					changed |= values.try_emplace(&inst, conditions.size()).first->second.Merge(set);
				}
			}
		}
		// Synchronizing blocks keep the branches they depend on whole-wave. A half that leaves
		// the program early would leave the other waiting at its next barrier.
		ConditionSet whole(conditions.size());
		for (size_t block = 0; block < count; block++) {
			bool syncs = sync_blocks.contains(blocks[block]) || blocks[block]->ImmSuccessors().empty();
			for (const auto& inst: *blocks[block]) {
				if (const auto found = condition_index.find(&inst);
				    found != condition_index.end() && !local[found->second]) {
					syncs = true;
				}
			}
			if (syncs) {
				whole.Merge(control[block]);
			}
		}
		bool removed = false;
		for (size_t condition = 0; condition < conditions.size(); condition++) {
			if (local[condition] && whole.Test(condition)) {
				local[condition] = false;
				removed          = true;
			}
		}
		if (!removed) {
			break;
		}
	}
	for (size_t condition = 0; condition < conditions.size(); condition++) {
		if (local[condition]) {
			result.insert(conditions[condition]);
		}
	}
	return result;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
