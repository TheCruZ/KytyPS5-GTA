#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

// KYTY_WAVE64_WARPS=1 runs eligible wave64 workgroups as two host subgroups (split waves).
bool SplitWavesEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_WAVE64_WARPS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	EXIT("SPIR-V validation failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     program.shader_hash, static_cast<unsigned>(program.stage), reason);
	std::abort();
}

void ValidateNativeProgram(const IR::Program& program, bool lds_storage) {
	std::vector<bool> live_buffers(program.info.buffers.size());
	bool uses_lds = false;
	bool uses_gds = false;
	const auto uses_mapping = [](const auto& resource) {
		return resource.indirect_root != UINT32_MAX;
	};
	bool uses_flattened_srt = std::ranges::any_of(program.info.buffers, uses_mapping) ||
	                          std::ranges::any_of(program.info.images, uses_mapping) ||
	                          std::ranges::any_of(program.info.images, [](const IR::ImageResource& image) {
		                          return image.table_capacity != 0u;
	                          });
	const auto planning_only_handle = [&](const IR::Inst& handle) {
		return !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       const auto op = use.user->GetOpcode();
			       if (op != IR::ValueOpcode::LoadAddressU32 &&
			           op != IR::ValueOpcode::ReadConstBuffer) {
				       return false;
			       }
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].planning_only;
		       });
	};
	const auto indirect_buffer_handle = [&](const IR::Inst& handle) {
		return program.info.uses_dma && handle.NumArgs() == 4u && !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       const auto access = IR::BufferAccessOf(use.user->GetOpcode());
			       if (access != IR::BufferAccess::Read && access != IR::BufferAccess::Write) {
				       return false;
			       }
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].kind == IR::ResourceKind::IndirectBuffer;
		       });
	};
	const auto local_flat_handle = [&](const IR::Inst& handle) {
		return !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       if (IR::AddressOpcodeInfoOf(use.user->GetOpcode()).access == IR::AddressAccess::None)
				       return false;
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].kind == IR::ResourceKind::FlatLocal;
		       });
	};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			uses_flattened_srt |= op == IR::ValueOpcode::ReadConst;
			if (IR::BufferAccessOf(op) != IR::BufferAccess::None ||
			    IR::SharedAccessOf(op) != IR::SharedAccess::None ||
			    IR::AddressOpcodeInfoOf(op).access != IR::AddressAccess::None) {
				const auto index = inst.Flags<IR::MemoryFlags>().index;
				if (index >= program.memory_info.size()) {
					Fail(program, "typed shader contains invalid memory metadata");
				}
				const auto& memory = program.memory_info[index];
				if (!memory.planning_only) {
					uses_lds |= memory.kind == IR::ResourceKind::FlatLocal;
					if (IR::SharedAccessOf(op) != IR::SharedAccess::None) {
						if (memory.kind != IR::ResourceKind::Lds && memory.kind != IR::ResourceKind::Gds) {
							Fail(program, "typed shader contains invalid shared-memory metadata");
						}
						uses_gds |= memory.kind == IR::ResourceKind::Gds;
						uses_lds |= memory.kind == IR::ResourceKind::Lds;
					} else if (memory.kind == IR::ResourceKind::Buffer ||
					           memory.kind == IR::ResourceKind::ScalarBuffer) {
						if (memory.resource >= program.info.buffers.size()) {
							Fail(program, "typed shader contains an invalid buffer resource");
						}
						live_buffers[memory.resource] = true;
						for (const auto child: program.info.buffers[memory.resource].indirect_resources) {
							if (child >= program.info.buffers.size()) {
								Fail(program, "typed shader contains an invalid indirect buffer resource");
							}
							live_buffers[child] = true;
						}
					}
				}
			}
			const auto dense = inst.Flags<uint32_t>();
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::GetBufferResource:
					if (planning_only_handle(inst) || indirect_buffer_handle(inst)) {
						break;
					}
					if (dense >= program.info.buffers.size()) {
						Fail(program, "typed buffer handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetBufferResourceIndex:
					if (dense >= program.info.buffers.size()) {
						Fail(program, "typed buffer selector has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetAddressResource:
					if (planning_only_handle(inst)) {
						break;
					}
					if (inst.NumArgs() != 2 || (!program.info.uses_dma && !local_flat_handle(inst))) {
						Fail(program, "typed address handle has invalid DMA metadata");
					}
					break;
				case IR::ValueOpcode::GetScratchResource:
					if (inst.NumArgs() != 0 || program.scratch_dwords == 0) {
						Fail(program, "typed scratch handle has invalid shader metadata");
					}
					break;
				case IR::ValueOpcode::GetImageResource:
					if (dense >= program.info.images.size()) {
						Fail(program, "typed image handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetSamplerResource:
					if (dense >= program.info.samplers.size()) {
						Fail(program, "typed sampler handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::ReadConst: {
					const auto slot = inst.Arg(1).Resolve();
					if (!slot.IsImmediate() || slot.GetType() != IR::Type::U32 ||
					    slot.U32() >= program.srt_reads.size()) {
						Fail(program, "flattened SRT read has an invalid dense slot");
					}
					break;
				}
				default: break;
			}
		}
	}
	using Kind = IR::DescriptorBindingKind;
	std::array<uint32_t, static_cast<size_t>(Kind::Count)> expected {};
	auto& buffer_count = expected[static_cast<size_t>(Kind::Buffers)];
	for (uint32_t index = 0; index < program.info.buffers.size(); ++index) {
		const auto slot = live_buffers[index] ? buffer_count++ : UINT32_MAX;
		if (program.info.buffers[index].live != live_buffers[index] ||
		    program.info.buffers[index].descriptor_index != slot) {
			Fail(program, "native buffer descriptor index does not match shader topology");
		}
	}
	auto& uniform_count = expected[static_cast<size_t>(Kind::UniformBuffers)];
	for (const auto& buffer: program.info.buffers) {
		if (buffer.uniform_index == UINT32_MAX) continue;
		if (buffer.uniform_index != uniform_count++ || !buffer.constant_only ||
		    buffer.descriptor_index == UINT32_MAX) {
			Fail(program, "native uniform buffer index does not match shader topology");
		}
	}
	for (const auto& image: program.info.images) {
		const auto kind = IR::DescriptorBindingForImage(image);
		if (!kind.has_value()) {
			Fail(program, "native shader plan has an invalid image class");
		}
		if (image.heap) continue;
		if (image.mip_count == 0u ||
		    (image.mip_mode != IR::ImageMipMode::Dynamic && image.mip_count != 1u)) {
			Fail(program, "native shader plan has an invalid image mip descriptor count");
		}
		auto& count = expected[static_cast<size_t>(*kind)];
		if (image.descriptor_index != count) {
			Fail(program, "native image descriptor index does not match shader topology");
		}
		count += image.mip_count;
	}
	const auto table_bases = IR::AllocateImageTableDescriptors(program.info.images, expected);
	for (size_t index = 0; index < program.info.images.size(); ++index) {
		if (program.info.images[index].table_descriptor_base != table_bases[index]) {
			Fail(program, "native image table descriptors do not match shader topology");
		}
	}
	expected[static_cast<size_t>(Kind::Samplers)] = static_cast<uint32_t>(program.info.samplers.size());
	expected[static_cast<size_t>(Kind::Gds)] = uses_gds;
	expected[static_cast<size_t>(Kind::SharedMemory)] = uses_lds && lds_storage;
	expected[static_cast<size_t>(Kind::BdaPagetable)] = program.info.uses_dma;
	expected[static_cast<size_t>(Kind::FaultBuffer)] = program.info.uses_dma;
	expected[static_cast<size_t>(Kind::FlattenedSrt)] =
	    uses_flattened_srt || program.bindings.srt_verify || Emitter::BindlessOrGpuSizedSrt(program);
	expected[static_cast<size_t>(Kind::ShaderData)] =
	    program.bindings.ShaderDataDwords() != 0 && !program.bindings.UsesPushData();
	uint64_t descriptor_mask = 0;
	for (uint32_t index = 0; index < expected.size(); ++index) {
		if (expected[index] != 0) descriptor_mask |= uint64_t{1} << index;
	}
	if (program.bindings.descriptor_counts != expected ||
	    program.bindings.descriptor_mask != descriptor_mask) {
		Fail(program, "native descriptor counts do not match shader topology");
	}
	const auto shader_data_dwords = program.bindings.ShaderDataDwords();
	const auto& registers = program.info.user_data_registers;
	const bool has_dispatch_threads = program.bindings.dispatch_thread_dword != IR::PushData::NoStart;
	if ((program.bindings.UsesPushData() &&
	     !IR::PushData::CanFit(program.bindings.push_data_start_dword, shader_data_dwords)) ||
	    (has_dispatch_threads && (program.stage != ShaderType::Compute ||
	                              program.bindings.dispatch_thread_dword != registers.size())) ||
	    program.bindings.memory_offset_dword != registers.size() + (has_dispatch_threads ? 3u : 0u) ||
	    !std::is_sorted(registers.begin(), registers.end()) ||
	    std::adjacent_find(registers.begin(), registers.end()) != registers.end()) {
		Fail(program, "native shader-data layout is inconsistent");
	}
}

// An upper bound of a U32 value built from constants, the lane id and simple integer arithmetic,
// or nothing. Phi cycles (a loop-carried address) are unbounded.
std::optional<uint64_t> UpperBoundU32(IR::Value value, uint32_t wave_size,
                                      std::vector<const IR::Inst*>& visiting) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		return value.GetType() == IR::Type::U32 ? std::optional<uint64_t>(value.U32())
		                                        : std::nullopt;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || visiting.size() > 64u || std::ranges::find(visiting, inst) != visiting.end()) {
		return std::nullopt;
	}
	visiting.push_back(inst);
	const auto arg = [&](size_t index) { return UpperBoundU32(inst->Arg(index), wave_size, visiting); };
	const auto immediate = [&](size_t index) -> std::optional<uint32_t> {
		const auto operand = inst->Arg(index).Resolve();
		return operand.IsImmediate() && operand.GetType() == IR::Type::U32
		           ? std::optional<uint32_t>(operand.U32())
		           : std::nullopt;
	};
	std::optional<uint64_t> result;
	switch (inst->GetOpcode()) {
		case IR::ValueOpcode::LaneId: result = wave_size - 1u; break;
		case IR::ValueOpcode::IAdd32:
			if (const auto a = arg(0), b = arg(1); a && b) {
				result = *a + *b;
			}
			break;
		case IR::ValueOpcode::IMul32:
			if (const auto a = arg(0), b = arg(1); a && b) {
				result = *a * *b;
			}
			break;
		case IR::ValueOpcode::ShiftLeftLogical32:
		case IR::ValueOpcode::ShiftRightLogical32: {
			const auto a = arg(0);
			const auto b = immediate(1);
			if (a && b && *b < 32u) {
				result = inst->GetOpcode() == IR::ValueOpcode::ShiftLeftLogical32 ? *a << *b : *a >> *b;
			}
			break;
		}
		case IR::ValueOpcode::BitwiseAnd32: {
			const auto a = arg(0), b = arg(1);
			if (a || b) {
				result = std::min(a.value_or(UINT32_MAX), b.value_or(UINT32_MAX));
			}
			break;
		}
		case IR::ValueOpcode::BitwiseOr32:
		case IR::ValueOpcode::BitwiseXor32:
			if (const auto a = arg(0), b = arg(1); a && b) {
				result = std::bit_ceil(std::max(*a, *b) + 1u) - 1u;
			}
			break;
		case IR::ValueOpcode::SelectU32:
			if (const auto a = arg(1), b = arg(2); a && b) {
				result = std::max(*a, *b);
			}
			break;
		case IR::ValueOpcode::Phi: {
			uint64_t bound = 0;
			bool     known = inst->NumArgs() != 0;
			for (size_t index = 0; known && index < inst->NumArgs(); index++) {
				const auto incoming = arg(index);
				known               = incoming.has_value();
				bound               = known ? std::max(bound, *incoming) : bound;
			}
			if (known) {
				result = bound;
			}
			break;
		}
		default: break;
	}
	visiting.pop_back();
	return result && *result <= UINT32_MAX ? result : std::nullopt;
}

} // namespace

namespace {
// Requirements the shader info does not carry: the shader clock, wave reductions matched by
// IR::MatchLaneReduction, and the size of a non-compute stage's function-scope LDS array.
// A non-compute stage keeps LDS in an array of its own, which NVIDIA places in local memory: at
// the 8192-dword default that is 32 KB per invocation, several GB across the GPU. Size it from the
// addresses the shader can form instead.
void AnalyzeExtraRequirements(Emitter::EmitterState& state) {
	auto&    program                = state.program;
	uint64_t function_lds_bytes     = 0;
	bool     function_lds_unbounded = false;
	const bool function_stage = program.stage != ShaderType::Compute && program.stage != ShaderType::Mesh;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (op == IR::ValueOpcode::ReadClockRealtime64) {
				state.shader_clock = true;
			}
			if (op == IR::ValueOpcode::ReadLane && IR::MatchLaneReduction(inst, program.wave_size)) {
				program.info.subgroup_arithmetic          = true;
				program.info.subgroup_local_invocation_id = true;
			}
			if (!function_stage) {
				continue;
			}
			if (IR::AddressOpcodeInfoOf(op).access != IR::AddressAccess::None) {
				const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
				if (memory_index < program.memory_info.size() &&
				    program.memory_info[memory_index].kind == IR::ResourceKind::FlatLocal) {
					function_lds_unbounded = true; // FLAT addresses into LDS are not bounded here
				}
			}
			if (IR::SharedAccessOf(op) == IR::SharedAccess::None) {
				continue;
			}
			const auto index = inst.Flags<IR::MemoryFlags>().index;
			if (index >= program.memory_info.size() ||
			    program.memory_info[index].kind != IR::ResourceKind::Lds) {
				continue;
			}
			std::vector<const IR::Inst*> visiting;
			const auto address = UpperBoundU32(inst.Arg(0), program.wave_size, visiting);
			if (address.has_value()) {
				const auto& memory = program.memory_info[index];
				function_lds_bytes = std::max(
				    function_lds_bytes, *address + std::max(memory.offset, memory.secondary_offset) +
				                            4u * IR::SharedComponentCount(op));
			} else {
				function_lds_unbounded = true;
			}
		}
	}
	if (program.info.function_lds && !function_lds_unbounded &&
	    function_lds_bytes < 8192u * sizeof(uint32_t)) {
		state.function_lds_dwords =
		    static_cast<uint32_t>((function_lds_bytes + sizeof(uint32_t) - 1u) / sizeof(uint32_t));
	}
}
} // namespace

namespace {
// The scope of the barriers that keep one guest wave's LDS accesses in order (EmitBlock). The
// guest's lanes run in lockstep and its LDS accesses complete in program order; host invocations
// need a barrier. It can be issued wherever the guest wave's control flow is uniform across the
// barrier's scope, which holds everywhere at block level (divergence is emulated with EXEC):
// - a single-wave workgroup gets the workgroup scope (on NVIDIA a subgroup-scope barrier did not
//   make one lane's LDS writes visible to the others, as AnyPS5 found);
// - otherwise the host subgroup scope, when one host subgroup holds exactly one guest wave;
// - otherwise none (a barrier that other waves of the scope may not reach could hang).
// Compute shaders that write LDS only. KYTY_WAVE_LDS_ORDER=0 turns it off.
uint32_t WaveLdsScope(const IR::Program& program, const ShaderWorkgroupInputInfo* workgroup,
                      uint32_t lane_count) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_WAVE_LDS_ORDER");
		return value == nullptr || value[0] != '0';
	}();
	if (!enabled || program.stage != ShaderType::Compute || workgroup == nullptr) {
		return 0;
	}
	bool writes = false;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto access = IR::SharedAccessOf(inst.GetOpcode());
			writes |= access != IR::SharedAccess::None && access != IR::SharedAccess::Read;
		}
	}
	if (!writes) {
		return 0;
	}
	const auto threads = std::max(workgroup->threads_num[0], 1u) *
	                     std::max(workgroup->threads_num[1], 1u) *
	                     std::max(workgroup->threads_num[2], 1u);
	if (threads <= program.wave_size) {
		return spv::ScopeWorkgroup;
	}
	if (lane_count == 2u || (program.wave_size == workgroup->host_subgroup_size &&
	                         workgroup->host_subgroup_size <= 32u)) {
		return spv::ScopeSubgroup;
	}
	return 0;
}

std::atomic_bool     g_image_min_lod {false};
std::atomic_uint32_t g_device_clock_divisor {10};
std::atomic_bool     g_coherent_load_acquire {false};
std::atomic_bool     g_denorm_ftz32 {false};
} // namespace

void SetCoherentLoadAcquire(bool enabled) {
	g_coherent_load_acquire.store(enabled, std::memory_order_relaxed);
}

void SetDenormFlushToZero32(bool supported) {
	g_denorm_ftz32.store(supported, std::memory_order_relaxed);
}

bool GetDenormFlushToZero32() {
	static const bool allowed = [] {
		const char* value = std::getenv("KYTY_DENORM_FTZ_MODE");
		return value == nullptr || value[0] != '0';
	}();
	return allowed && g_denorm_ftz32.load(std::memory_order_relaxed);
}

bool GetCoherentLoadAcquire() {
	static const int forced = [] {
		const char* value = std::getenv("KYTY_COHERENT_LOAD_ACQUIRE");
		return value == nullptr ? -1 : (std::strcmp(value, "0") != 0 ? 1 : 0);
	}();
	return forced >= 0 ? forced == 1 : g_coherent_load_acquire.load(std::memory_order_relaxed);
}

void SetDeviceClockDivisor(uint32_t divisor) {
	g_device_clock_divisor.store(divisor != 0 ? divisor : 1u, std::memory_order_relaxed);
}

uint32_t GetDeviceClockDivisor() {
	return g_device_clock_divisor.load(std::memory_order_relaxed);
}

void SetHostImageFeatures(const HostImageFeatures& features) {
	g_image_min_lod.store(features.min_lod, std::memory_order_relaxed);
}

HostImageFeatures GetHostImageFeatures() {
	return {.min_lod = g_image_min_lod.load(std::memory_order_relaxed)};
}

std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t push_data_start_dword, const Profile& profile,
                                  bool emit_debug_info, bool robust_buffer_access2) {
	using namespace Emitter;

	if (program.stage != ShaderType::Compute && program.stage != ShaderType::Vertex &&
	    program.stage != ShaderType::Pixel && program.stage != ShaderType::Mesh &&
	    program.stage != ShaderType::Local && program.stage != ShaderType::TessellationControl &&
	    program.stage != ShaderType::TessellationEvaluation) {
		Fail(program, "binary SPIR-V emitter received an unsupported shader stage");
	}
	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	if (IR::IrValidationEnabled()) {
		IR::ValidateProgram(program, true);
	}
	program.bindings = {.push_data_start_dword = push_data_start_dword};
	EmitterState state(program, input_info, profile);
	if (emit_debug_info) {
		char name[64];
		std::snprintf(name, sizeof(name), "rdna2-%016" PRIx64 "-block-byte-pc-plus-one",
		              program.shader_hash);
		state.debug_source = state.builder.AddString(name);
		state.builder.EnableNames();
	}
	state.robust_buffer_access2 = robust_buffer_access2;
	const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
	state.lane_count =
	    workgroup != nullptr && program.wave_size == 64u && workgroup->host_subgroup_size == 32u
	        ? 2u
	        : 1u;
	if (program.stage == ShaderType::Mesh) {
		state.mesh_passes = std::max(input_info.vertex->mesh.passes, 1u);
	}
	state.wave_lds_scope = WaveLdsScope(program, workgroup, state.lane_count);
	AnalyzeExtraRequirements(state);
	DefineModule(state);
	ValidateNativeProgram(program,
	                      program.stage == ShaderType::Compute && input_info.compute->lds_storage);
	EmitProgram(state);
	state.builder.AddEntryPoint(ExecutionModelForStage(state.program.stage), state.main_func,
	                            "main", state.interface_variables);

	return state.builder.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
