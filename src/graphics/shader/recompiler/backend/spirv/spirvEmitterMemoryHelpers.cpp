#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include <algorithm>
#include <bit>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

uint32_t EmitPushDataLoad(EmitterState& state, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state),
	                          pointer, state.push_constant_variable, ConstantU32(state, 0), index);
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	return value;
}

static uint32_t EmitShaderDataDwordLoadAt(EmitterState& state, uint32_t dword_index) {
	if (state.program.bindings.UsesPushData()) return EmitPushDataLoad(state, dword_index);
	if (state.shader_data_storage_variable != 0) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, state.shader_data_storage_variable,
		                          ConstantU32(state, 0), dword_index);
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	}
	return ConstantU32(state, 0);
}

uint32_t EmitShaderDataDwordLoad(EmitterState& state, uint32_t dword_index) {
	if (state.program.bindings.UsesPushData()) {
		dword_index += state.program.bindings.push_data_start_dword;
	}
	return EmitShaderDataDwordLoadAt(state, ConstantU32(state, dword_index));
}

uint32_t EmitBinaryU32(EmitterState& state, spv::Op opcode, uint32_t lhs, uint32_t rhs) {
	const auto ret = state.builder.AllocateId();
	state.builder.AddFunction(opcode, TypeU32(state), ret, lhs, rhs);
	return ret;
}

uint32_t EmitIndirectResourceIndex(EmitterState& state, uint32_t key, uint32_t mapping_offset,
                                   uint32_t search_iterations, uint32_t default_resource) {
	const auto LoadMapping = [&](uint32_t index) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, state.flattened_srt_variable, ConstantU32(state, 0),
		                          index);
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	};
	const auto mapping = ConstantU32(state, mapping_offset);
	const auto count = LoadMapping(mapping);
	if (search_iterations == 0u) {
		const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), key,
		                          ConstantU32(state, 2u));
		const auto valid = Binary(state, spv::OpULessThan, TypeBool(state), index, count);
		const auto safe = Select(state, TypeU32(state), valid, index, ConstantU32(state, 0u));
		const auto entry = Binary(state, spv::OpIAdd, TypeU32(state), safe,
		                          ConstantU32(state, mapping_offset + 1u));
		return Select(state, TypeU32(state), valid, LoadMapping(entry),
		              ConstantU32(state, default_resource));
	}
	const auto entry_at = [&](uint32_t index) {
		return Binary(state, spv::OpIAdd, TypeU32(state), mapping,
		              Binary(state, spv::OpIAdd, TypeU32(state),
		                     Binary(state, spv::OpShiftLeftLogical, TypeU32(state), index,
		                            ConstantU32(state, 1u)), ConstantU32(state, 1u)));
	};
	auto low = ConstantU32(state, 0u);
	auto high = count;
	for (uint32_t iteration = 0; iteration < search_iterations; ++iteration) {
		const auto active = Binary(state, spv::OpULessThan, TypeBool(state), low, high);
		const auto mid = Binary(state, spv::OpShiftRightLogical, TypeU32(state),
		                        Binary(state, spv::OpIAdd, TypeU32(state), low, high),
		                        ConstantU32(state, 1u));
		const auto probe = Select(state, TypeU32(state), active, mid, ConstantU32(state, 0u));
		const auto less = Binary(state, spv::OpULessThan, TypeBool(state),
		                         LoadMapping(entry_at(probe)), key);
		low = Select(state, TypeU32(state),
		             Binary(state, spv::OpLogicalAnd, TypeBool(state), active, less),
		             Binary(state, spv::OpIAdd, TypeU32(state), mid, ConstantU32(state, 1u)), low);
		high = Select(state, TypeU32(state),
		              Binary(state, spv::OpLogicalAnd, TypeBool(state), active,
		                     Unary(state, spv::OpLogicalNot, TypeBool(state), less)), mid, high);
	}
	const auto in_range = Binary(state, spv::OpULessThan, TypeBool(state), low, count);
	const auto entry = entry_at(Select(state, TypeU32(state), in_range, low, ConstantU32(state, 0u)));
	const auto match = Binary(state, spv::OpLogicalAnd, TypeBool(state), in_range,
	                          Binary(state, spv::OpIEqual, TypeBool(state), LoadMapping(entry), key));
	const auto resource = LoadMapping(Binary(state, spv::OpIAdd, TypeU32(state), entry,
	                                         ConstantU32(state, 1u)));
	return Select(state, TypeU32(state), match, resource, ConstantU32(state, default_resource));
}

uint32_t StorageBufferPackedStride(const EmitterState& state, const IR::MemoryInfo& mem) {
	if (mem.resource >= state.program.info.buffers.size()) {
		ExitDescriptorBindingFailure(state, IR::DescriptorBindingKind::Buffers, mem.resource,
		                             "buffer specialization is missing");
	}
	return state.program.info.buffers[mem.resource].packed_stride;
}

Prospero::BufferFormat StorageBufferFormat(const EmitterState& state, const IR::MemoryInfo& mem) {
	if (mem.resource >= state.program.info.buffers.size()) {
		ExitDescriptorBindingFailure(state, IR::DescriptorBindingKind::Buffers, mem.resource,
		                             "buffer specialization is missing");
	}
	return state.program.info.buffers[mem.resource].descriptor_format;
}

uint32_t SharedIndirectBufferResource(const EmitterState& state, uint32_t resource) {
	const auto& buffers = state.program.info.buffers;
	const auto& root = buffers[resource];
	if (root.indirect_root != resource || root.descriptor_index == UINT32_MAX ||
	    root.indirect_resources.size() < 2u) {
		return UINT32_MAX;
	}
	const auto first = root.packed_stride == 0u ? 1u : 0u;
	const auto representative = root.indirect_resources[first];
	const auto& shared = buffers[representative];
	if (shared.packed_stride == 0u) return UINT32_MAX;
	uint32_t next_child_slot = UINT32_MAX;
	for (uint32_t ordinal = first; ordinal < root.indirect_resources.size(); ++ordinal) {
		const auto& candidate = buffers[root.indirect_resources[ordinal]];
		if (candidate.packed_stride != shared.packed_stride ||
		    candidate.descriptor_format != shared.descriptor_format ||
		    candidate.descriptor_swizzle != shared.descriptor_swizzle ||
		    candidate.descriptor_index == UINT32_MAX ||
		    (ordinal > 1u && candidate.descriptor_index != next_child_slot)) return UINT32_MAX;
		if (ordinal != 0u) next_child_slot = candidate.descriptor_index + 1u;
	}
	return representative;
}

void EmitMemoryOffsets(EmitterState& state) {
	const auto count = state.program.bindings.descriptor_counts[static_cast<size_t>(IR::DescriptorBindingKind::Buffers)];
	state.memory_byte_offsets.resize(count);
	state.memory_dword_lengths.resize(count);
	std::vector<bool> dynamic_offsets(count);
	for (uint32_t resource = 0; resource < state.program.info.buffers.size(); ++resource) {
		if (SharedIndirectBufferResource(state, resource) == UINT32_MAX) continue;
		for (const auto candidate: state.program.info.buffers[resource].indirect_resources) {
			dynamic_offsets[state.program.info.buffers[candidate].descriptor_index] = true;
		}
	}
	for (uint32_t i = 0; i < count; i += 4u) {
		const auto lanes = std::min(4u, count - i);
		if (std::all_of(dynamic_offsets.begin() + i, dynamic_offsets.begin() + i + lanes,
		                [](bool dynamic) { return dynamic; })) continue;
		const auto word =
		    EmitShaderDataDwordLoad(state, state.program.bindings.memory_offset_dword + i / 4u);
		for (uint32_t lane = 0; lane < lanes; ++lane) {
			if (dynamic_offsets[i + lane]) continue;
			state.memory_byte_offsets[i + lane] = EmitNative<spv::OpBitFieldUExtract, IR::Type::U32>(
			    state, word, ConstantU32(state, lane * 8u), ConstantU32(state, 8u));
			// The bound range's length in bytes (OpArrayLength is zero for ranges over 2 GiB on
			// some NVIDIA drivers), from the shader data.
			state.memory_dword_lengths[i + lane] = EmitShaderDataDwordLoad(
			    state, state.program.bindings.BufferLengthDword() + i + lane);
		}
	}
	// A buffer bound with a conservative record count (KYTY_GPU_SIZED_DYNAMIC): its length is the
	// smaller of the bound range and the exact count's, read from the flattened SRT dword the GPU
	// filled before the dispatch (BufferResource::gpu_records_flat). Lengths count bytes from the
	// aligned binding start, so the byte offset of the guest base is added first; sums saturate.
	if (state.flattened_srt_variable != 0) {
		const auto bool_type = TypeBool(state);
		const auto u32       = TypeU32(state);
		const auto max_u32   = ConstantU32(state, UINT32_MAX);
		for (const auto& buffer: state.program.info.buffers) {
			if (!buffer.live || buffer.descriptor_index >= count ||
			    dynamic_offsets[buffer.descriptor_index] ||
			    buffer.gpu_records_flat == IR::BufferResource::NoGpuRecords) {
				continue;
			}
			const auto i       = buffer.descriptor_index;
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
			                          pointer, state.flattened_srt_variable, ConstantU32(state, 0),
			                          ConstantU32(state, buffer.gpu_records_flat));
			uint32_t records = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, u32, records, pointer);
			if (buffer.gpu_records_add_flat != IR::BufferResource::NoGpuRecords) {
				const auto add_pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
				                          add_pointer, state.flattened_srt_variable,
				                          ConstantU32(state, 0),
				                          ConstantU32(state, buffer.gpu_records_add_flat));
				const auto added = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpLoad, u32, added, add_pointer);
				records = EmitBinaryU32(state, spv::OpIAdd, records, added);
			}
			if (buffer.gpu_records_add != 0u) {
				records = EmitBinaryU32(state, spv::OpIAdd, records,
				                        ConstantU32(state, buffer.gpu_records_add));
			}
			const uint32_t stride = buffer.packed_stride & 0x3fffu;
			uint32_t       bytes  = records;
			if (stride > 1u) {
				const auto overflow = Binary(state, spv::OpUGreaterThan, bool_type, records,
				                             ConstantU32(state, UINT32_MAX / stride));
				bytes = Select(state, u32, overflow, max_u32,
				               EmitBinaryU32(state, spv::OpIMul, records, ConstantU32(state, stride)));
			}
			const auto offset    = state.memory_byte_offsets[i];
			const auto saturated = Binary(state, spv::OpUGreaterThan, bool_type, bytes,
			                              EmitBinaryU32(state, spv::OpISub, max_u32, offset));
			const auto end       = Select(state, u32, saturated, max_u32,
			                              EmitBinaryU32(state, spv::OpIAdd, bytes, offset));
			const auto bound     = state.memory_dword_lengths[i];
			state.memory_dword_lengths[i] =
			    Select(state, u32, Binary(state, spv::OpULessThan, bool_type, end, bound), end,
			           bound);
		}
	}
	if (state.program.bindings.write_tracking) {
		const auto load64 = [&](uint32_t dword) {
			const auto u64 = TypeU64(state);
			const auto lo  = Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, dword));
			const auto hi = Unary(state, spv::OpUConvert, u64,
			                      EmitShaderDataDwordLoad(state, dword + 1u));
			return Binary(state, spv::OpBitwiseOr, u64, lo,
			              Binary(state, spv::OpShiftLeftLogical, u64, hi,
			                     ConstantU64(state, 32)));
		};
		const auto start           = state.program.bindings.WriteTrackDword();
		state.write_bitmap_address = load64(start);
		state.write_track_bases.resize(count);
		for (uint32_t i = 0; i < count; i++) {
			state.write_track_bases[i] = load64(start + 2u + 2u * i);
		}
	}
}

// KYTY_SRT_VERIFY (ResourceSnapshot.h, SrtVerify): before anything else, compare the pairs the
// renderer appended to the flattened SRT (the value the host used, the value the GPU copied
// behind the work that wrote it). When any pair differs, report the shader in the buffer at the
// address the shader data holds (count, hash low, hash high) and return: the dispatch has no
// effect, and the host stops using such values for this shader.
void EmitSrtVerify(EmitterState& state) {
	if (!state.program.bindings.srt_verify || state.flattened_srt_variable == 0) {
		return;
	}
	const auto u32       = TypeU32(state);
	const auto bool_type = TypeBool(state);
	const auto first     = state.program.bindings.VerifyDword();
	const auto base      = EmitShaderDataDwordLoad(state, first);
	const auto count     = EmitShaderDataDwordLoad(state, first + 1u);
	const auto load_flat = [&](uint32_t index) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, state.flattened_srt_variable, ConstantU32(state, 0),
		                          index);
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, u32, value, pointer);
		return value;
	};
	// for (i = 0; i < count; i++) miss |= flat[base + 2i] != flat[base + 2i + 1];
	const auto entry_label  = state.builder.AllocateId();
	const auto header_label = state.builder.AllocateId();
	const auto body_label   = state.builder.AllocateId();
	const auto cont_label   = state.builder.AllocateId();
	const auto merge_label  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBranch, entry_label);
	EmitLabel(state, entry_label);
	state.builder.AddFunction(spv::OpBranch, header_label);
	EmitLabel(state, header_label);
	const auto index    = state.builder.AllocateId();
	const auto miss     = state.builder.AllocateId();
	const auto next     = state.builder.AllocateId();
	const auto miss_out = state.builder.AllocateId();
	const auto bad      = state.builder.AllocateId(); // the first pair that differed
	const auto bad_out  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, u32, index, ConstantU32(state, 0), entry_label, next,
	                          cont_label);
	state.builder.AddFunction(spv::OpPhi, bool_type, miss, ConstantBool(state, false), entry_label,
	                          miss_out, cont_label);
	state.builder.AddFunction(spv::OpPhi, u32, bad, ConstantU32(state, 0), entry_label, bad_out,
	                          cont_label);
	state.builder.AddFunction(spv::OpLoopMerge, merge_label, cont_label,
	                          spv::LoopControlMaskNone);
	const auto more = Binary(state, spv::OpULessThan, bool_type, index, count);
	state.builder.AddFunction(spv::OpBranchConditional, more, body_label, merge_label);
	EmitLabel(state, body_label);
	const auto at       = EmitBinaryU32(state, spv::OpIAdd, base,
	                                    EmitBinaryU32(state, spv::OpShiftLeftLogical, index,
	                                                  ConstantU32(state, 1)));
	const auto expected = load_flat(at);
	const auto actual   = load_flat(EmitBinaryU32(state, spv::OpIAdd, at, ConstantU32(state, 1)));
	const auto differs  = Binary(state, spv::OpINotEqual, bool_type, expected, actual);
	state.builder.AddFunction(spv::OpLogicalOr, bool_type, miss_out, miss, differs);
	const auto first_bad = Binary(state, spv::OpLogicalAnd, bool_type, differs,
	                              Unary(state, spv::OpLogicalNot, bool_type, miss));
	state.builder.AddFunction(spv::OpSelect, u32, bad_out, first_bad, index, bad);
	state.builder.AddFunction(spv::OpBranch, cont_label);
	EmitLabel(state, cont_label);
	state.builder.AddFunction(spv::OpIAdd, u32, next, index, ConstantU32(state, 1));
	state.builder.AddFunction(spv::OpBranch, header_label);
	EmitLabel(state, merge_label);
	// At the merge the loop's miss is the header's value (the only edge into the merge).
	const auto report_label = state.builder.AllocateId();
	const auto after_label  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, after_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, miss, report_label, after_label);
	EmitLabel(state, report_label);
	{
		const auto u64     = TypeU64(state);
		const auto lo      = Unary(state, spv::OpUConvert, u64,
		                           EmitShaderDataDwordLoad(state, first + 2u));
		const auto hi      = Unary(state, spv::OpUConvert, u64,
		                           EmitShaderDataDwordLoad(state, first + 3u));
		const auto address = Binary(state, spv::OpBitwiseOr, u64, lo,
		                            Binary(state, spv::OpShiftLeftLogical, u64, hi,
		                                   ConstantU64(state, 32)));
		const auto word    = [&](uint32_t dword) {
			const auto at_address = dword == 0u ? address
			                                    : Binary(state, spv::OpIAdd, u64, address,
			                                             ConstantU64(state, dword * 4u));
			return Unary(state, spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), at_address);
		};
		const auto hash  = state.program.shader_hash;
		const auto atom  = [&](spv::Op op, uint32_t pointer, uint32_t value) {
			const auto result = state.builder.AllocateId();
			state.builder.AddFunction(op, u32, result, pointer, ConstantU32(state, spv::ScopeDevice),
			                          ConstantU32(state, 0), value);
		};
		const auto bad_at = EmitBinaryU32(state, spv::OpIAdd, base,
		                                  EmitBinaryU32(state, spv::OpShiftLeftLogical, bad,
		                                                ConstantU32(state, 1)));
		atom(spv::OpAtomicExchange, word(1u), ConstantU32(state, static_cast<uint32_t>(hash)));
		atom(spv::OpAtomicExchange, word(2u), ConstantU32(state, static_cast<uint32_t>(hash >> 32u)));
		atom(spv::OpAtomicExchange, word(3u), bad);
		atom(spv::OpAtomicExchange, word(4u), load_flat(bad_at));
		atom(spv::OpAtomicExchange, word(5u),
		     load_flat(EmitBinaryU32(state, spv::OpIAdd, bad_at, ConstantU32(state, 1))));
		atom(spv::OpAtomicIAdd, word(0u), ConstantU32(state, 1));
	}
	state.builder.AddFunction(spv::OpReturn);
	EmitLabel(state, after_label);
}

// Rework Phase 2, first step: a store or atomic through a write-tracked binding sets the bit of
// its guest page (4 KiB) in the GPU write bitmap; the host then downloads only pages that were
// written. A binding the host does not track in this draw has a base of 0 and skips it.
void EmitWriteBitmapMark(EmitterState& state, const MemoryResourceAccess& access, uint32_t index) {
	const auto u64     = TypeU64(state);
	const auto tracked = Binary(state, spv::OpINotEqual, TypeBool(state), access.write_base,
	                            ConstantU64(state, 0));
	(void)EmitValueOrZeroIfCondition(state, tracked, [&]() {
		const auto element = Unary(state, spv::OpUConvert, u64, index);
		const auto shift   = std::countr_zero(std::max(access.element_bits / 8u, 1u));
		const auto offset  = shift == 0 ? element
		                                : Binary(state, spv::OpShiftLeftLogical, u64, element,
		                                         ConstantU64(state, shift));
		const auto address = Binary(state, spv::OpIAdd, u64, access.write_base, offset);
		const auto page    = Binary(state, spv::OpShiftRightLogical, u64, address,
		                            ConstantU64(state, 12));
		const auto word    = Binary(state, spv::OpShiftRightLogical, u64, page,
		                            ConstantU64(state, 5));
		const auto bit     = Binary(
		    state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), Unary(state, spv::OpUConvert, TypeU32(state), page),
		           ConstantU32(state, 31)));
		const auto word_address =
		    Binary(state, spv::OpIAdd, u64, state.write_bitmap_address,
		           Binary(state, spv::OpShiftLeftLogical, u64, word, ConstantU64(state, 2)));
		const auto pointer = Unary(state, spv::OpConvertUToPtr, TypePhysicalU32Pointer(state),
		                           word_address);
		const auto value   = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicOr, TypeU32(state), value, pointer,
		                          ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0), bit);
		return value;
	});
}

uint32_t LdsDwordCount(const EmitterState& state) {
	const auto* workgroup = ShaderWorkgroupInput(state.program.stage, state.input_info);
	if (workgroup != nullptr) {
		return workgroup->lds_size_dwords;
	}
	return state.function_lds_dwords != 0 ? state.function_lds_dwords
	                                                   : 8192u;
}

// KYTY_LDS_PAD_DWORDS=<n>, optionally with KYTY_LDS_PAD_HASHES="hash,hash,...": declare the
// workgroup LDS array with at least n dwords. Accesses are still bounds-checked against the guest
// LDS size. A diagnostic: NVIDIA's compiler spills registers into workgroup memory the shader
// leaves unused, and a larger array pushes those spills to local memory instead.
static uint32_t LdsPadDwords(uint64_t shader_hash) {
	static const auto setting = [] {
		struct Setting {
			uint32_t              dwords = 0;
			std::vector<uint64_t> hashes;
		} result;
		if (const char* value = std::getenv("KYTY_LDS_PAD_DWORDS"); value != nullptr) {
			result.dwords = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
		}
		if (const char* value = std::getenv("KYTY_LDS_PAD_HASHES"); value != nullptr) {
			const std::string list(value);
			size_t            start = 0;
			while (start < list.size()) {
				const auto end = std::min(list.find(',', start), list.size());
				if (end > start) {
					result.hashes.push_back(
					    std::strtoull(list.substr(start, end - start).c_str(), nullptr, 16));
				}
				start = end + 1;
			}
		}
		return result;
	}();
	if (!setting.hashes.empty() &&
	    std::find(setting.hashes.begin(), setting.hashes.end(), shader_hash) == setting.hashes.end()) {
		return 0;
	}
	return setting.dwords;
}

void EnsureLdsStorage(EmitterState& state) {
	if (state.lds_variable != 0) {
		return;
	}
	if (state.lds_storage_class != spv::StorageClassWorkgroup) {
		EXIT("LDS backing was not prepared before SPIR-V function emission\n");
	}
	const auto dwords =
	    std::max({1u, LdsDwordCount(state), LdsPadDwords(state.program.shader_hash)});
	const auto define = [&](uint32_t type, uint32_t bytes) {
		const auto array = state.builder.DecoratedType(
		    spv::OpTypeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, bytes}}}, type,
		    ConstantU32(state, std::max(dwords * 4u / bytes, 1u)));
		const auto block = state.builder.DecoratedType(
		    spv::OpTypeStruct, {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
		                        {spv::OpDecorate, {spv::DecorationBlock}}}, array);
		const auto variable = state.builder.DefineGlobalVariable(
		    TypePointer(state, spv::StorageClassWorkgroup, block), spv::StorageClassWorkgroup);
		state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationAliased);
		return variable;
	};
	if (state.program.info.shared_int64_atomics) {
		state.lds_variable = define(TypeU32(state), 4u);
		state.lds_u64_variable = define(TypeU64(state), 8u);
		state.builder.AddName(state.lds_u64_variable, "lds_qwords");
	} else {
		state.lds_variable = state.builder.DefineGlobalVariable(
		    TypeU32ArrayPointer(state, spv::StorageClassWorkgroup, dwords),
		    spv::StorageClassWorkgroup);
	}
	state.builder.AddName(state.lds_variable, "lds_dwords");
}

MemoryResourceAccess PrepareStorageBufferResourceAccess(EmitterState& state,
                                                         const IR::MemoryInfo& mem,
                                                         BufferDefinition& buffer,
                                                         uint32_t array_index,
                                                         uint32_t element_bits) {
	if (buffer.variable == 0) {
		ExitDescriptorBindingFailure(state, IR::DescriptorBindingKind::Buffers, mem.resource,
		                             "storage buffer descriptor array was not emitted");
	}
	if (buffer.pointer_type == 0) {
		buffer.pointer_type = TypePointer(state, spv::StorageClassStorageBuffer, buffer.type);
	}
	const bool non_uniform = array_index != 0u;
	const auto static_index = state.program.info.buffers.at(mem.resource).descriptor_index;
	if (!non_uniform) array_index = ConstantU32(state, static_index);
	MemoryResourceAccess access {.kind             = mem.kind,
	                             .descriptor_index = array_index,
	                             .element_bits     = element_bits,
	                             .non_uniform      = non_uniform,
	                             .memory_access    = mem.coherent ? spv::MemoryAccessVolatileMask
	                                                              : spv::MemoryAccessMaskNone};
	access.object_pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, buffer.pointer_type, access.object_pointer,
	                          buffer.variable, array_index);
	const bool write_tracked = state.emitting_write && mem.kind == IR::ResourceKind::Buffer &&
	                           state.program.bindings.write_tracking &&
	                           state.program.info.buffers[mem.resource].write_tracked;
	uint32_t bytes = 0;
	if (non_uniform) {
		state.builder.RequireExtension("SPV_EXT_descriptor_indexing");
		state.builder.RequireCapability(spv::CapabilityShaderNonUniform);
		state.builder.RequireCapability(spv::CapabilityStorageBufferArrayNonUniformIndexing);
		state.builder.AddAnnotation(spv::OpDecorate, access.object_pointer, spv::DecorationNonUniform);
		const auto& bindings = state.program.bindings;
		const auto push_start = bindings.UsesPushData() ? bindings.push_data_start_dword : 0u;
		const auto offsets_start = bindings.memory_offset_dword + push_start;
		const auto word = EmitShaderDataDwordLoadAt(state, EmitAddU32(
		    state, ConstantU32(state, offsets_start),
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state), array_index, ConstantU32(state, 2))));
		const auto shift = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), array_index, ConstantU32(state, 3)),
		    ConstantU32(state, 3));
		access.byte_offset = EmitNative<spv::OpBitFieldUExtract, IR::Type::U32>(
		    state, word, shift, ConstantU32(state, 8));
		bytes = EmitShaderDataDwordLoadAt(state, EmitAddU32(
		    state, ConstantU32(state, bindings.BufferLengthDword() + push_start), array_index));
		if (write_tracked) {
			// The guest base of this binding (2 dwords) in the write-tracking table.
			const auto at = EmitAddU32(
			    state, ConstantU32(state, bindings.WriteTrackDword() + push_start + 2u),
			    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), array_index,
			           ConstantU32(state, 1)));
			const auto u64 = TypeU64(state);
			const auto lo  = Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoadAt(state, at));
			const auto hi  = Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoadAt(
			    state, EmitAddU32(state, at, ConstantU32(state, 1))));
			access.write_base = Binary(state, spv::OpBitwiseOr, u64, lo,
			    Binary(state, spv::OpShiftLeftLogical, u64, hi, ConstantU64(state, 32)));
		}
	} else {
		access.byte_offset = state.memory_byte_offsets[static_index];
		bytes              = state.memory_dword_lengths[static_index];
		if (write_tracked) {
			access.write_base = state.write_track_bases[static_index];
		}
	}
	// Some NVIDIA drivers return zero from OpArrayLength for ranges over 2 GiB: the length is
	// the actual bound range, including alignment, without truncating guest memory.
	// The length is in bytes; the bounds checks index elements of the access's own width (narrow
	// storage indexes bytes or halfwords, 64-bit atomics qwords). A byte length keeps the tail of
	// a range that ends inside a dword (an 8-bit record count of 20 bound at byte offset 2), which
	// a dword length dropped.
	// gpu_records_ssa: also within the count of the descriptor this access names (the handle's
	// dword 2, which the shader computes), from the guest base; sums saturate.
	if ((mem.kind == IR::ResourceKind::Buffer || mem.kind == IR::ResourceKind::ScalarBuffer) &&
	    mem.resource < state.program.info.buffers.size() &&
	    state.program.info.buffers[mem.resource].gpu_records_ssa && state.emitting_inst != nullptr &&
	    state.emitting_ctx != nullptr && state.emitting_inst->NumArgs() != 0) {
		const auto* handle = state.emitting_inst->Arg(0).Resolve().TryInstruction();
		if (handle != nullptr && handle->GetOpcode() == IR::ValueOpcode::GetBufferResource &&
		    handle->Flags<uint32_t>() == mem.resource && handle->NumArgs() > 2u) {
			const auto bool_type = TypeBool(state);
			const auto u32       = TypeU32(state);
			const auto max_u32   = ConstantU32(state, UINT32_MAX);
			const auto records   = state.emitting_ctx->Arg(*handle, 2);
			const uint32_t stride = state.program.info.buffers[mem.resource].packed_stride & 0x3fffu;
			uint32_t       record_bytes = records;
			if (stride > 1u) {
				const auto overflow = Binary(state, spv::OpUGreaterThan, bool_type, records,
				                             ConstantU32(state, UINT32_MAX / stride));
				record_bytes = Select(state, u32, overflow, max_u32,
				                      EmitBinaryU32(state, spv::OpIMul, records,
				                                    ConstantU32(state, stride)));
			}
			const auto offset    = access.byte_offset;
			const auto saturated = Binary(state, spv::OpUGreaterThan, bool_type, record_bytes,
			                              EmitBinaryU32(state, spv::OpISub, max_u32, offset));
			const auto end       = Select(state, u32, saturated, max_u32,
			                              EmitBinaryU32(state, spv::OpIAdd, record_bytes, offset));
			bytes = Select(state, u32, Binary(state, spv::OpULessThan, bool_type, end, bytes), end,
			               bytes);
		}
	}
	const uint32_t element_shift = element_bits == 8u    ? 0u
	                               : element_bits == 16u ? 1u
	                               : element_bits == 64u ? 3u
	                                                     : 2u;
	access.length = element_shift == 0u
	                    ? bytes
	                    : Binary(state, spv::OpShiftRightLogical, TypeU32(state), bytes,
	                             ConstantU32(state, element_shift));
	return access;
}

MemoryResourceAccess PrepareMemoryResourceAccess(EmitterState& state, const IR::MemoryInfo& mem,
                                                 uint32_t array_index) {
	MemoryResourceAccess access {.kind = mem.kind};
	switch (mem.kind) {
		case IR::ResourceKind::Lds:
			EnsureLdsStorage(state);
			access.object_pointer = state.lds_variable;
			access.length         = ConstantU32(state, LdsDwordCount(state));
			return access;
		case IR::ResourceKind::Gds:
			if (state.gds_variable == 0) {
				ExitDescriptorBindingFailure(state, IR::DescriptorBindingKind::Gds, mem.resource,
				                             "GDS binding was not emitted");
			}
			access.object_pointer = state.gds_variable;
			if (state.gds_length == 0) {
				EXIT("GDS length was not prepared at function entry\n");
			}
			access.length = state.gds_length;
			return access;
		case IR::ResourceKind::Scratch:
			if (state.scratch_variable[state.lane_half] == 0) {
				EXIT("scratch storage was not prepared before SPIR-V function emission\n");
			}
			access.object_pointer = state.scratch_variable[state.lane_half];
			access.length         = ConstantU32(state, state.program.scratch_dwords);
			return access;
		case IR::ResourceKind::ScalarAddress:
		case IR::ResourceKind::Flat:
		case IR::ResourceKind::Global:
			EXIT("physical address memory must use the BDA emitter\n");
		case IR::ResourceKind::ScalarBuffer:
		case IR::ResourceKind::Buffer: {
			const auto bits = mem.kind == IR::ResourceKind::Buffer
			                      ? IR::StorageBufferElementBits(state.program, mem) : 32u;
			auto* buffer = &state.storage_buffers[2];
			if (bits == 8u) {
				buffer = &state.storage_buffers[0];
			} else if (bits == 16u) {
				buffer = &state.storage_buffers[1];
			}
			access = PrepareStorageBufferResourceAccess(state, mem, *buffer, array_index, bits);
			access.element_bits = bits;
			return access;
		}
		default: EXIT("unsupported memory resource kind: %u\n", static_cast<unsigned>(mem.kind));
	}
}

uint32_t EmitMemoryElementInBounds(EmitterState& state, const MemoryResourceAccess& access,
                                   uint32_t index) {
	const auto in_bounds = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpULessThan, TypeBool(state), in_bounds, index, access.length);
	return in_bounds;
}

uint32_t EmitMemoryElementPointer(EmitterState& state, const MemoryResourceAccess& access,
                                  uint32_t index) {
	if (access.kind == IR::ResourceKind::Lds || access.kind == IR::ResourceKind::Scratch) {
		const auto storage_class = access.kind == IR::ResourceKind::Scratch
		                               ? spv::StorageClassFunction : state.lds_storage_class;
		if (storage_class == spv::StorageClassStorageBuffer) {
			return EmitStorageBufferElementPointer(
			    state, access, EmitAddU32(state, state.lds_base_dwords, index),
			    TypeStorageBufferElementPointer(state, 32));
		}
		const auto pointer = state.builder.AllocateId();
		if (access.kind == IR::ResourceKind::Lds && state.program.info.shared_int64_atomics) {
			state.builder.AddFunction(spv::OpAccessChain, TypeU32ElementPointer(state, storage_class),
			                          pointer, access.object_pointer, ConstantU32(state, 0), index);
			return pointer;
		}
		state.builder.AddFunction(spv::OpAccessChain, TypeU32ElementPointer(state, storage_class),
		                          pointer, access.object_pointer, index);
		return pointer;
	}
	return EmitStorageBufferElementPointer(state, access, index,
	                                       TypeStorageBufferElementPointer(state, access.element_bits));
}

uint32_t EmitStorageBufferElementPointer(EmitterState& state,
                                         const MemoryResourceAccess& access, uint32_t index,
                                         uint32_t pointer_type) {
	if (access.write_base != 0) {
		EmitWriteBitmapMark(state, access, index);
	}
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, pointer_type, pointer, access.object_pointer,
	                          ConstantU32(state, 0), index);
	if (access.non_uniform) {
		state.builder.AddAnnotation(spv::OpDecorate, pointer, spv::DecorationNonUniform);
	}
	return pointer;
}

uint32_t EmitTBufferBitcastU32ToI32(EmitterState& state, uint32_t value) {
	const auto ret = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeI32(state), ret, value);
	return ret;
}

bool IsSignedFormatComponent(Format::ComponentType type) {
	return type == Format::ComponentType::Sint || type == Format::ComponentType::Snorm ||
	       type == Format::ComponentType::Sscaled;
}

uint32_t EmitUFloatToF32Bits(EmitterState& state, uint32_t raw, uint32_t bits) {
	const auto mantissa_bits = bits == 11u ? 6u : 5u;
	const auto mantissa_mask = (1u << mantissa_bits) - 1u;
	const auto mantissa =
	    EmitBinaryU32(state, spv::OpBitwiseAnd, raw, ConstantU32(state, mantissa_mask));
	const auto exponent = EmitBinaryU32(
	    state, spv::OpBitwiseAnd,
	    EmitBinaryU32(state, spv::OpShiftRightLogical, raw, ConstantU32(state, mantissa_bits)),
	    ConstantU32(state, 0x1fu));

	const auto exponent_32 = EmitAddU32(state, exponent, ConstantU32(state, 127u - 15u));
	const auto exponent_bits =
	    EmitBinaryU32(state, spv::OpShiftLeftLogical, exponent_32, ConstantU32(state, 23));
	const auto mantissa_bits_32 = EmitBinaryU32(state, spv::OpShiftLeftLogical, mantissa,
	                                            ConstantU32(state, 23u - mantissa_bits));
	const auto normal_bits =
	    EmitBinaryU32(state, spv::OpBitwiseOr, exponent_bits, mantissa_bits_32);
	const auto normal      = EmitBitcastU32ToF32(state, normal_bits);

	const auto special_bits =
	    EmitBinaryU32(state, spv::OpBitwiseOr, ConstantU32(state, 0x7f800000u), mantissa_bits_32);
	const auto special = EmitBitcastU32ToF32(state, special_bits);

	const auto mantissa_f32 = state.builder.AllocateId();
	const auto subnormal    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), mantissa_f32, mantissa);
	state.builder.AddFunction(
	    spv::OpFMul, TypeF32(state), subnormal, mantissa_f32,
	    ConstantF32Value(state, std::ldexp(1.0f, 1 - 15 - static_cast<int>(mantissa_bits))));

	const auto zero_exp    = EmitCompareU32Constant(state, spv::OpIEqual, exponent, 0);
	const auto special_exp = EmitCompareU32Constant(state, spv::OpIEqual, exponent, 31);
	const auto finite      = EmitTBufferSelectF32(state, zero_exp, subnormal, normal);
	const auto result      = EmitTBufferSelectF32(state, special_exp, special, finite);
	return EmitBitcastF32ToU32(state, result);
}

uint32_t NormalizeFormatComponent(EmitterState& state, const Format::BufferFormatInfo& info,
                                  uint32_t component, uint32_t raw) {
	const auto bits = info.component_bits[component];
	switch (info.type) {
		case Format::ComponentType::Uint:
		case Format::ComponentType::Sint: return raw;
		case Format::ComponentType::Uscaled: {
			const auto value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), value, raw);
			return EmitBitcastF32ToU32(state, value);
		}
		case Format::ComponentType::Sscaled: {
			const auto signed_raw = EmitTBufferBitcastU32ToI32(state, raw);
			const auto value      = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertSToF, TypeF32(state), value, signed_raw);
			return EmitBitcastF32ToU32(state, value);
		}
		case Format::ComponentType::Unorm: {
			const auto value      = state.builder.AllocateId();
			const auto normalized = state.builder.AllocateId();
			const auto max_value  = static_cast<float>((1u << bits) - 1u);
			state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), value, raw);
			state.builder.AddFunction(spv::OpFDiv, TypeF32(state), normalized, value,
			                          ConstantF32Value(state, max_value));
			return EmitBitcastF32ToU32(state, normalized);
		}
		case Format::ComponentType::Snorm: {
			const auto signed_raw = EmitTBufferBitcastU32ToI32(state, raw);
			const auto value      = state.builder.AllocateId();
			const auto normalized = state.builder.AllocateId();
			const auto clamped    = state.builder.AllocateId();
			const auto max_value  = static_cast<float>((1u << (bits - 1u)) - 1u);
			state.builder.AddFunction(spv::OpConvertSToF, TypeF32(state), value, signed_raw);
			state.builder.AddFunction(spv::OpFDiv, TypeF32(state), normalized, value,
			                          ConstantF32Value(state, max_value));
			state.builder.AddFunction(spv::OpExtInst, TypeF32(state), clamped, GlslStd450(state),
			                          GLSLstd450FMax, normalized, ConstantF32Value(state, -1.0f));
			return EmitBitcastF32ToU32(state, clamped);
		}
		case Format::ComponentType::Float:
			if (bits == 32u) {
				return raw;
			}
			if (bits == 16u) {
				return EmitBitcastF32ToU32(state, EmitF16BitsToF32(state, raw));
			}
			return EmitUFloatToF32Bits(state, raw, bits);
		default: return raw;
	}
}

uint32_t EncodeFormattedStoreComponent(EmitterState& state, const Format::BufferFormatInfo& info,
                                       uint32_t component, uint32_t data, uint32_t data_bits,
                                       UnormRounding rounding) {
	EXIT_IF(rounding != UnormRounding::RoundEven && info.type != Format::ComponentType::Unorm);
	const auto bits = info.component_bits[component];
	if (data_bits == 16u) {
		if (info.type == Format::ComponentType::Uint) {
			data = EmitBitFieldUExtract(state, data, ConstantU32(state, 0), ConstantU32(state, 16));
		} else if (info.type == Format::ComponentType::Sint) {
			data = EmitBitFieldSExtract(state, data, ConstantU32(state, 0), ConstantU32(state, 16));
		} else {
			data = EmitBitCastU32F32(state, EmitF16BitsToF32(state, data));
		}
	}
	if (info.type == Format::ComponentType::Sint && bits < 32u) {
		const auto maximum = (1u << (bits - 1u)) - 1u;
		data = EmitSMax32(state, data, ConstantU32(state, ~maximum));
		data = EmitSMin32(state, data, ConstantU32(state, maximum));
		return Binary(state, spv::OpBitwiseAnd, TypeU32(state), data,
		              ConstantU32(state, (1u << bits) - 1u));
	}
	const bool normalized = info.type == Format::ComponentType::Unorm ||
	                        info.type == Format::ComponentType::Snorm;
	const bool scaled = info.type == Format::ComponentType::Uscaled ||
	                    info.type == Format::ComponentType::Sscaled;
	if (normalized || scaled) {
		// Two-bit packed channels are unsigned even in signed formats.
		const bool is_signed = bits != 2u && (info.type == Format::ComponentType::Snorm ||
		                                      info.type == Format::ComponentType::Sscaled);
		const auto max_value = static_cast<float>((1u << (bits - (is_signed ? 1u : 0u))) - 1u);
		const auto lower = normalized ? (is_signed ? -1.0f : 0.0f)
		                              : (is_signed ? -max_value - 1.0f : 0.0f);
		const auto upper = normalized ? 1.0f : max_value;
		auto value = Select(state, TypeF32(state), EmitClassifyF32Bits(state, data).nan,
		                    ConstantF32Value(state, 0.0f), EmitBitCastF32U32(state, data));
		value = EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(
		    state, value, ConstantF32Value(state, lower), ConstantF32Value(state, upper));
		if (normalized) {
			value = EmitFPMul32(state, value, ConstantF32Value(state, max_value));
			if (rounding == UnormRounding::RoundEven) {
				value = EmitFPRoundEven32(state, value);
			} else if (rounding == UnormRounding::HalfUp) {
				value = EmitFPAdd32(state, value, ConstantF32Value(state, 0.5f));
			}
		}
		// Float-to-integer conversion truncates scaled values toward zero.
		value = Unary(state, is_signed ? spv::OpConvertFToS : spv::OpConvertFToU,
		              is_signed ? TypeI32(state) : TypeU32(state), value);
		return is_signed ? Unary(state, spv::OpBitcast, TypeU32(state), value) : value;
	}
	if (bits == 16u && info.type == Format::ComponentType::Float) {
		const auto pair = EmitCompositeConstructF32x2(
		    state, EmitBitCastF32U32(state, data), ConstantF32Value(state, 0.0f));
		return EmitPackHalf2x16(state, pair);
	}
	return data;
}

uint32_t EncodePackedColorComponent(EmitterState& state, const Format::BufferFormatInfo& info,
                                    uint32_t component, uint32_t data, uint32_t data_bits,
                                    UnormRounding rounding) {
	auto value = EncodeFormattedStoreComponent(state, info, component, data, data_bits, rounding);
	if (info.type == Format::ComponentType::Uint) {
		value = EmitUMin32(state, value,
		                  ConstantU32(state, (1u << info.component_bits[component]) - 1u));
	}
	if (info.component_bit_offset[component] != 0u) {
		value = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), value,
		               ConstantU32(state, info.component_bit_offset[component]));
	}
	return value;
}

spv::Op SpirvAtomicOpcode(IR::ValueOpcode opcode) {
	switch (opcode) {
		case IR::ValueOpcode::SharedAtomicCmpSwap32:
		case IR::ValueOpcode::SharedAtomicCmpSwap64:
		case IR::ValueOpcode::ImageAtomicCompareSwap32:
		case IR::ValueOpcode::BufferAtomicCmpSwap32: return spv::OpAtomicCompareExchange;
		case IR::ValueOpcode::ImageAtomicSwap32:
		case IR::ValueOpcode::ImageAtomicSwap64:
		case IR::ValueOpcode::BufferAtomicSwap32:
		case IR::ValueOpcode::BufferAtomicSwap64:
		case IR::ValueOpcode::SharedAtomicSwap32: return spv::OpAtomicExchange;
		case IR::ValueOpcode::ImageAtomicIAdd32:
		case IR::ValueOpcode::ImageAtomicIAdd64:
		case IR::ValueOpcode::BufferAtomicIAdd32:
		case IR::ValueOpcode::SharedAtomicIAdd64:
		case IR::ValueOpcode::BufferAtomicIAdd64:
		case IR::ValueOpcode::SharedAtomicIAdd32: return spv::OpAtomicIAdd;
		case IR::ValueOpcode::BufferAtomicISub32:
		case IR::ValueOpcode::SharedAtomicISub32: return spv::OpAtomicISub;
		case IR::ValueOpcode::ImageAtomicSMin32:
		case IR::ValueOpcode::BufferAtomicSMin32:
		case IR::ValueOpcode::SharedAtomicSMin32: return spv::OpAtomicSMin;
		case IR::ValueOpcode::ImageAtomicUMin32:
		case IR::ValueOpcode::ImageAtomicUMin64:
		case IR::ValueOpcode::BufferAtomicUMin32:
		case IR::ValueOpcode::SharedAtomicUMin32: return spv::OpAtomicUMin;
		case IR::ValueOpcode::ImageAtomicSMax32:
		case IR::ValueOpcode::BufferAtomicSMax32:
		case IR::ValueOpcode::SharedAtomicSMax32: return spv::OpAtomicSMax;
		case IR::ValueOpcode::ImageAtomicUMax32:
		case IR::ValueOpcode::ImageAtomicUMax64:
		case IR::ValueOpcode::BufferAtomicUMax32:
		case IR::ValueOpcode::SharedAtomicUMax32:
		case IR::ValueOpcode::SharedAtomicUMax64: return spv::OpAtomicUMax;
		case IR::ValueOpcode::ImageAtomicAnd32:
		case IR::ValueOpcode::ImageAtomicAnd64:
		case IR::ValueOpcode::BufferAtomicAnd32:
		case IR::ValueOpcode::BufferAtomicAnd64:
		case IR::ValueOpcode::SharedAtomicAnd32: return spv::OpAtomicAnd;
		case IR::ValueOpcode::ImageAtomicOr32:
		case IR::ValueOpcode::ImageAtomicOr64:
		case IR::ValueOpcode::BufferAtomicOr32:
		case IR::ValueOpcode::BufferAtomicOr64:
		case IR::ValueOpcode::SharedAtomicOr64:
		case IR::ValueOpcode::SharedAtomicOr32: return spv::OpAtomicOr;
		case IR::ValueOpcode::ImageAtomicXor32:
		case IR::ValueOpcode::ImageAtomicXor64:
		case IR::ValueOpcode::BufferAtomicXor32:
		case IR::ValueOpcode::SharedAtomicXor32: return spv::OpAtomicXor;
		default: return spv::OpNop;
	}
}

uint32_t EmitAtomicOperation(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t pointer,
                             uint32_t scope) {
	const auto opcode = SpirvAtomicOpcode(inst.GetOpcode());
	const auto old    = ctx.state.builder.AllocateId();
	const bool wide   = inst.GetType() == IR::Type::U64;
	const auto type   = wide ? TypeU64(ctx.state) : TypeU32(ctx.state);
	if (opcode == spv::OpAtomicCompareExchange) {
		auto desired    = ctx.Arg(inst, inst.NumArgs() - 3);
		auto comparator = ctx.Arg(inst, inst.NumArgs() - 2);
		if (inst.GetOpcode() == IR::ValueOpcode::SharedAtomicCmpSwap32) {
			// RDNA2 DS compares DATA and stores DATA2, opposite to BUFFER atomics.
			std::swap(desired, comparator);
		}
		ctx.state.builder.AddFunction(
		    spv::OpAtomicCompareExchange, TypeU32(ctx.state), old, pointer,
		    ConstantU32(ctx.state, scope), ConstantU32(ctx.state, spv::MemorySemanticsMaskNone),
		    ConstantU32(ctx.state, spv::MemorySemanticsMaskNone), desired, comparator);
	} else {
		const auto value = ctx.Arg(inst, inst.NumArgs() - 2);
		ctx.state.builder.AddFunction(opcode, type, old, pointer,
		                              ConstantU32(ctx.state, scope),
		                              ConstantU32(ctx.state, spv::MemorySemanticsMaskNone), value);
	}
	return old;
}

void EmitAtomicMemoryBarrier(EmitterState& state, IR::ResourceKind kind) {
	const auto scope  = kind == IR::ResourceKind::Lds ? spv::ScopeWorkgroup : spv::ScopeDevice;
	const auto memory = [&] {
		switch (kind) {
			case IR::ResourceKind::Lds:
				return state.lds_storage_class == spv::StorageClassStorageBuffer
				           ? spv::MemorySemanticsUniformMemoryMask
				           : spv::MemorySemanticsWorkgroupMemoryMask;
			case IR::ResourceKind::Image: return spv::MemorySemanticsImageMemoryMask;
			default: return spv::MemorySemanticsUniformMemoryMask;
		}
	}();
	state.builder.AddFunction(spv::OpMemoryBarrier, ConstantU32(state, scope),
	                          ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | memory));
}

namespace {

uint32_t FloatOrderKey(EmitterState& state, uint32_t bits) {
	const auto negative = EmitCompareU32Constant(state, spv::OpINotEqual,
	                                             EmitAndConstant(state, bits, 0x80000000u), 0u);
	const auto negative_key = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpNot, TypeU32(state), negative_key, bits);
	const auto positive_key =
	    EmitBinaryU32(state, spv::OpBitwiseXor, bits, ConstantU32(state, 0x80000000u));
	return EmitSelectValueU32(state, negative, negative_key, positive_key);
}

} // namespace

uint32_t EmitFloatAtomicReplacement(EmitterState& state, uint32_t old, uint32_t source,
                                    bool max_value) {
	struct OrderedBits {
		uint32_t nan;
		uint32_t zero;
		uint32_t key;
	};
	const auto classify = [&](uint32_t bits) {
		const auto cls = EmitClassifyF32Bits(state, bits);
		return OrderedBits {cls.nan, cls.zero, FloatOrderKey(state, bits)};
	};
	const auto source_class = classify(source);
	const auto old_class    = classify(old);
	const auto unordered = EmitLogicalOrBool(
	    state, EmitLogicalOrBool(state, source_class.nan, old_class.nan),
	    EmitLogicalAndBool(state, source_class.zero, old_class.zero));
	const auto compare = state.builder.AllocateId();
	state.builder.AddFunction(max_value ? spv::OpUGreaterThan : spv::OpULessThan, TypeBool(state),
	                          compare, source_class.key, old_class.key);
	return EmitSelectValueU32(
	    state, EmitLogicalAndBool(state, EmitLogicalNotBool(state, unordered), compare), source,
	    old);
}

uint32_t EmitDsFloatAtomicReplacement(EmitterState& state, uint32_t old, uint32_t source,
                                      bool max_value) {
	struct OrderedBits {
		uint32_t nan;
		uint32_t signaling_nan;
		uint32_t key;
	};
	const auto classify = [&](uint32_t bits) {
		const auto abs = EmitAndConstant(state, bits, 0x7fffffffu);
		const auto nan = EmitCompareU32Constant(state, spv::OpUGreaterThan, abs, 0x7f800000u);
		const auto signaling_nan = EmitLogicalAndBool(
		    state, nan, EmitCompareU32Constant(state, spv::OpIEqual,
		                                      EmitAndConstant(state, bits, 0x00400000u), 0u));
		const auto small = EmitCompareU32Constant(state, spv::OpULessThan, abs, 0x00800000u);
		const auto sign = EmitAndConstant(state, bits, 0x80000000u);
		const auto positive_denorm = EmitLogicalAndBool(
		    state, EmitCompareU32Constant(state, spv::OpIEqual, sign, 0u),
		    EmitCompareU32Constant(state, spv::OpINotEqual, abs, 0u));
		// Negative denorms tie with -0; positive denorms tie only with each other.
		const auto small_bits = EmitSelectValueU32(
		    state, positive_denorm, ConstantU32(state, 1u), sign);
		const auto key_bits = EmitSelectValueU32(state, small, small_bits, bits);
		return OrderedBits {nan, signaling_nan, FloatOrderKey(state, key_bits)};
	};
	const auto lhs = classify(old);
	const auto rhs = classify(source);
	const auto choose_source = state.builder.AllocateId();
	state.builder.AddFunction(max_value ? spv::OpUGreaterThan : spv::OpULessThanEqual,
	                          TypeBool(state), choose_source, rhs.key, lhs.key);
	// Keep original bits: ordering denorms does not flush the selected operand.
	auto result = EmitSelectValueU32(state, choose_source, source, old);
	result = EmitSelectValueU32(state, lhs.nan, source, result);
	result = EmitSelectValueU32(state, rhs.nan, old, result);
	result = EmitSelectValueU32(state, rhs.signaling_nan,
	                            EmitOrU32(state, source, ConstantU32(state, 0x00400000u)), result);
	return EmitSelectValueU32(state, lhs.signaling_nan,
	                          EmitOrU32(state, old, ConstantU32(state, 0x00400000u)), result);
}

uint32_t EmitDsSwizzleTargetLane(EmitterState& state, uint32_t subid, uint32_t control) {
	if ((control & 0xc000u) == 0xc000u) {
		const uint32_t mask         = control & 0x1fu;
		const uint32_t rotate       = (control >> 5u) & 0x1fu;
		const uint32_t rotate_delta = (control & 0x400u) != 0u ? ((32u - rotate) & 0x1fu) : rotate;
		const auto     lane         = EmitAndConstant(state, subid, 31);
		const auto     rotated_sum  = EmitAddU32(state, lane, ConstantU32(state, rotate_delta));
		const auto     rotated      = EmitAndConstant(state, rotated_sum, 31);
		const auto     kept         = EmitAndConstant(state, lane, mask);
		const auto     moved        = EmitAndConstant(state, rotated, (~mask) & 31u);
		const auto     combined     = EmitOrU32(state, kept, moved);
		const auto     base         = EmitAndConstant(state, subid, 0xffffffe0u);
		return EmitOrU32(state, base, combined);
	}

	if ((control & 0x8000u) != 0) {
		const auto lane2  = state.builder.AllocateId();
		const auto shift  = state.builder.AllocateId();
		const auto perm0  = state.builder.AllocateId();
		const auto perm   = state.builder.AllocateId();
		const auto base   = state.builder.AllocateId();
		const auto target = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane2, subid,
		                          ConstantU32(state, 3));
		state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), shift, lane2,
		                          ConstantU32(state, 1));
		state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), perm0,
		                          ConstantU32(state, control), shift);
		state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), perm, perm0,
		                          ConstantU32(state, 3));
		state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), base, subid,
		                          ConstantU32(state, 0xfffffffcu));
		state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, base, perm);
		return target;
	}

	const auto lane   = state.builder.AllocateId();
	const auto masked = state.builder.AllocateId();
	const auto ored   = state.builder.AllocateId();
	const auto xored  = state.builder.AllocateId();
	const auto base   = state.builder.AllocateId();
	const auto target = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane, subid,
	                          ConstantU32(state, 31));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), masked, lane,
	                          ConstantU32(state, control & 0x1fu));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), ored, masked,
	                          ConstantU32(state, (control >> 5u) & 0x1fu));
	state.builder.AddFunction(spv::OpBitwiseXor, TypeU32(state), xored, ored,
	                          ConstantU32(state, (control >> 10u) & 0x1fu));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), base, subid,
	                          ConstantU32(state, 0xffffffe0u));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, base, xored);
	return target;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
