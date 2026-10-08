#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/abi.h"
#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

SrtRuntime SrtReadCapture::ObservedRuntime() {
	auto runtime = m_source;
	runtime.userdata = this;
	runtime.read_specialization_memory = m_source.read_specialization_memory != nullptr
	                                         ? ReadStrict : nullptr;
	runtime.read_memory = ReadOrdinary;
	if (m_slots != nullptr) {
		runtime.read_slot = &m_slot;
	}
	return runtime;
}

void SrtReadCapture::Record(uint64_t address, uint64_t size) {
	m_ranges.emplace_back(address, size);
	if (m_slots != nullptr) {
		m_slots->push_back(m_slot);
	}
}

bool SrtReadCapture::ReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (!capture.m_source.read_specialization_memory(capture.m_source.userdata, address, values)) {
		return false;
	}
	capture.Record(address, values.size_bytes());
	return true;
}

bool SrtReadCapture::ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (capture.m_source.read_memory != nullptr) {
		if (!capture.m_source.read_memory(capture.m_source.userdata, address, values)) return false;
	} else {
		std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	}
	capture.Record(address, values.size_bytes());
	return true;
}

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	runtime.read_memory = runtime.read_specialization_memory != nullptr
	                          ? runtime.read_specialization_memory
	                          : +[](void*, uint64_t, std::span<uint32_t>) { return false; };
	return runtime;
}

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer &&
	    op != ValueOpcode::LoadBufferU32) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto& memory = values.memory_info[index];
	if (op != ValueOpcode::LoadBufferU32) {
		return (op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarAddress) ||
		       (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarBuffer);
	}
	if (inst.NumArgs() != 5u || memory.kind != ResourceKind::Buffer || memory.typed ||
	    memory.formatted || memory.coherent || memory.data_bits != 32u ||
	    memory.data_dwords != 1u || memory.idxen || memory.offen || memory.offset != 0u ||
	    inst.Arg(4).GetType() != Type::U1) {
		return false;
	}
	for (uint32_t arg = 1; arg <= 3; ++arg) {
		if (inst.Arg(arg).Resolve() != Value(0u)) return false;
	}
	return true;
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ConditionRef:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type,
	                          bool wave_uniform = false)
	    : m_program(program), m_type(type), m_wave_uniform(wave_uniform) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type != RuntimeValueType::Any &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return false;
			}
		}
		if (require_uniform && !m_active_mask.IsEmpty() && value == m_active_mask) return true;
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (m_type == RuntimeValueType::ImmutableInteger &&
		    (op == ValueOpcode::ReadConst || BufferAccessOf(op) != BufferAccess::None ||
		     AddressOpcodeInfoOf(op).access != AddressAccess::None ||
		     ImageOpcodeInfoOf(op).access != ImageAccess::None ||
		     SharedAccessOf(op) != SharedAccess::None)) {
			return finish(false);
		}
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU8 ||
		    op == ValueOpcode::UndefU16 || op == ValueOpcode::UndefU32 ||
		    op == ValueOpcode::UndefU64 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (m_wave_uniform && op == ValueOpcode::GetBuiltin) {
			const auto axis = inst->NumArgs() == 2u ? inst->Arg(1).Resolve() : Value {};
			return finish(m_program.stage == ShaderType::Compute && inst->NumArgs() == 2u &&
			              inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId)) &&
			              axis.IsImmediate() && axis.GetType() == Type::U32 && axis.U32() < 3u);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type != RuntimeValueType::Any && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer ||
		    op == ValueOpcode::LoadBufferU32) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
			if (op == ValueOpcode::LoadBufferU32) {
				const auto guard = inst->Arg(4).Resolve();
				return finish(Validate(inst->Arg(0)) &&
				              ((!m_active_mask.IsEmpty() && guard == m_active_mask) || Validate(guard)));
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u ||
			    (source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 &&
			     source->GetOpcode() != ValueOpcode::IAddCarry32)) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	bool                            m_wave_uniform;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
};


} // namespace

void CompileSrtWalker(ResourcePlan& program) {
	if (!program.resource_tracking_complete || !program.srt_plan_complete ||
	    program.capture_specialization_reads || !program.control_flow.empty() ||
	    program.srt_reads.empty()) return;

	using Reads = std::map<int32_t, const SrtRead*>;
	std::map<uint32_t, Reads> roots;
	std::unordered_map<const Inst*, Reads> pointer_uses;
	std::unordered_map<const Inst*, int32_t> offsets;
	for (const auto& read: program.srt_reads) {
		const auto* inst = read.value.Resolve().TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::LoadAddressU32 ||
		    inst->NumArgs() < 2 || !IsRawRead(program, *inst) ||
		    read.flat_offset >= program.srt_reads.size()) return;
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32) return;
		const auto& memory = program.memory_info[inst->Flags<SrtReadFlags>().index];
		const auto relative =
		    (static_cast<int64_t>(static_cast<int32_t>(memory.offset)) & ~int64_t {3}) +
		    static_cast<int64_t>(offset.U32() & ~3u);
		if (relative < INT32_MIN || relative > INT32_MAX ||
		    !offsets.emplace(inst, static_cast<int32_t>(relative)).second) return;
	}
	uint32_t user_data_count = 0;
	for (const auto& read: program.srt_reads) {
		const auto* inst = read.value.ResolveInstruction();
		const auto* handle = inst->Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource ||
		    handle->NumArgs() != 2) return;
		const auto* low = handle->Arg(0).Resolve().TryInstruction();
		const auto* high = handle->Arg(1).Resolve().TryInstruction();
		if (low == nullptr || high == nullptr) return;
		Reads* uses = nullptr;
		if (low->GetOpcode() == ValueOpcode::GetUserData &&
		    high->GetOpcode() == ValueOpcode::GetUserData && low->NumArgs() == 1 &&
		    high->NumArgs() == 1 && low->Arg(0).GetType() == Type::ScalarReg &&
		    high->Arg(0).GetType() == Type::ScalarReg) {
			const auto reg = RegIndex(low->Arg(0).ScalarRegister());
			if (reg < program.user_data_base ||
			    RegIndex(high->Arg(0).ScalarRegister()) != reg + 1u) return;
			const auto base = reg - program.user_data_base;
			user_data_count = std::max(user_data_count, base + 2u);
			uses = &roots[base];
		} else {
			const auto lo = offsets.find(low);
			const auto hi = offsets.find(high);
			if (lo == offsets.end() || hi == offsets.end() ||
			    static_cast<int64_t>(lo->second) + 4 != hi->second ||
			    !EquivalentValue(program, low->Arg(0), high->Arg(0))) return;
			uses = &pointer_uses[low];
		}
		if (!uses->emplace(offsets.at(inst), &read).second) return;
	}
	std::unordered_set<const Inst*> visited;
	const auto reachable = [&](auto&& self, const Reads& reads) -> bool {
		for (const auto& [offset, read]: reads) {
			const auto* inst = read->value.ResolveInstruction();
			const auto child = pointer_uses.find(inst);
			if (child != pointer_uses.end() &&
			    (!visited.insert(inst).second || !self(self, child->second))) return false;
		}
		return true;
	};
	for (const auto& [reg, reads]: roots) {
		if (!reachable(reachable, reads)) return;
	}
	if (visited.size() != pointer_uses.size()) return;

	program.srt_code = std::make_unique<Xbyak::CodeGenerator>(4096, Xbyak::AutoGrow);
	auto& code = *program.srt_code;
	Xbyak::Label failed, finished;
	code.push(code.rbp);
	code.mov(code.rbp, code.rsp);
	code.cmp(code.r8, user_data_count);
	code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
	code.mov(code.r9, AddressMask & ~uint64_t {3});
	// Follow shadPS4's VisitPointer: copy one table level in offset order, then
	// descend through its child pointers. Seed the descriptor evaluator's memo.
	const auto visit = [&](auto&& self, int32_t pointer_offset, const Reads& reads) -> void {
		code.push(code.rdi);
		code.mov(code.rdi, code.qword[code.rdi + pointer_offset]);
		code.and_(code.rdi, code.r9);
		const auto first = reads.begin()->first;
		const auto last = reads.rbegin()->first;
		if (first < 0) {
			code.mov(code.r10, -static_cast<int64_t>(first));
			code.cmp(code.rdi, code.r10);
			code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
		}
		if (last > 0) {
			code.mov(code.r10, AddressMask - last);
			code.cmp(code.rdi, code.r10);
			code.ja(failed, Xbyak::CodeGenerator::T_NEAR);
		}
		for (const auto& [offset, read]: reads) {
			const auto index = read->value.ResolveInstruction()->EvaluationIndex(program.evaluation_value_count);
			using Entry = ResourcePlan::EvaluationContext::Entry;
			const auto slot = code.rdx + index * sizeof(Entry);
			code.mov(code.r10d, code.dword[code.rdi + offset]);
			code.mov(code.dword[code.rsi + read->flat_offset * sizeof(uint32_t)], code.r10d);
			code.mov(code.qword[slot + offsetof(Entry, value)], code.r10);
			code.mov(code.qword[slot + offsetof(Entry, generation)], code.rcx);
		}
		for (const auto& [offset, read]: reads) {
			const auto child = pointer_uses.find(read->value.ResolveInstruction());
			if (child == pointer_uses.end()) continue;
			self(self, offset, child->second);
		}
		code.pop(code.rdi);
	};
	for (const auto& [reg, reads]: roots) {
		visit(visit, reg * sizeof(uint32_t), reads);
	}
	code.mov(code.eax, 1);
	code.jmp(finished, Xbyak::CodeGenerator::T_NEAR);
	code.L(failed);
	code.xor_(code.eax, code.eax);
	code.L(finished);
	code.mov(code.rsp, code.rbp);
	code.pop(code.rbp);
	code.ret();
	code.readyRE();
	EXIT_IF(Xbyak::GetError() != 0);
}

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     SrtWalker* clean_evaluator)
    : m_program(program), m_runtime(runtime), m_clean_evaluator(clean_evaluator),
      m_session(this),
      m_read_policy(clean_evaluator != nullptr ? 2u : 0u), m_context(AcquireContext()) {}

SrtWalker::SrtWalker(SrtWalker& parent, Value active_mask, bool clean,
                     SrtWalker* clean_evaluator)
    : m_program(parent.m_program), m_runtime(clean ? CleanRuntime(parent.m_runtime) : parent.m_runtime),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()),
      m_reachable_blocks(parent.m_reachable_blocks), m_session(parent.m_session),
      m_read_policy((parent.m_read_policy & 1u) | (clean ? 1u : 0u) |
                    (clean_evaluator != nullptr ? 2u : 0u)), m_context(AcquireContext()) {}

SrtWalker::~SrtWalker() {
	if (m_session != this) return;
	for (auto& context: m_program.evaluation_contexts)
		if (context.session == this) context.session = nullptr;
}

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateWide(value, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

ResourcePlan::EvaluationContext& SrtWalker::AcquireContext() {
	ResourcePlan::EvaluationContext* unused = nullptr;
	for (auto& context: m_program.evaluation_contexts) {
		if (context.session == m_session && context.active_mask == m_active_mask &&
		    context.read_policy == m_read_policy) return context;
		if (context.session == nullptr && unused == nullptr) unused = &context;
	}
	auto& context = unused != nullptr ? *unused : m_program.evaluation_contexts.emplace_back();
	context.session = m_session;
	context.active_mask = m_active_mask;
	context.read_policy = m_read_policy;
	context.generation += 4;
	return context;
}

void SrtWalker::ResetUnknowns() {
	for (auto& context: m_program.evaluation_contexts) {
		if (context.session != m_session) continue;
		for (auto& entry: context.values)
			if (entry.generation == (context.generation | 2u)) entry.generation = 0;
	}
}

float SrtWalker::Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

bool SrtWalker::EvaluateWide(Value value, uint64_t& result) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case Type::U1: result = value.U1(); return true;
			case Type::U8: result = value.U8(); return true;
			case Type::U16: result = value.U16(); return true;
			case Type::U32: result = value.U32(); return true;
			case Type::U64: result = value.U64(); return true;
			case Type::F32: result = std::bit_cast<uint32_t>(value.F32Value()); return true;
			default: return false;
		}
	}
	if (!m_active_mask.IsEmpty() && value == m_active_mask) {
		result = 1u;
		return true;
	}
	auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) &&
	    inst->NumArgs() == 3 && inst->Arg(0).Resolve() == m_active_mask) {
		return EvaluateWide(inst->Arg(1), result);
	}
	const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	if (m_context.values[index].generation == m_context.generation) {
		result = m_context.values[index].value;
		return true;
	}
	// Visiting and completed-unknown entries both stop repeated evaluation in this
	// context. Only completed unknowns are retried when reachability becomes tighter.
	if (m_context.values[index].generation == (m_context.generation | 1u) ||
	    m_context.values[index].generation == (m_context.generation | 2u)) {
		return false;
	}
	m_context.values[index].generation = m_context.generation | 1u;
	uint64_t out = 0;
	const bool evaluated = EvaluateInst(*inst, out);
	// Recursive evaluation may grow the dense memo vector.
	auto& memo = m_context.values[index];
	if (!evaluated) {
		memo.generation = m_context.generation | 2u;
		return false;
	}
	memo.value      = out;
	memo.generation = m_context.generation;
	result = out;
	return true;
}

bool SrtWalker::Arg(const Inst& inst, size_t index, uint64_t& result) {
	return EvaluateWide(inst.Arg(index), result);
}

bool SrtWalker::EvaluatePhi(const Inst& inst, uint64_t& result) {
	if (const auto control = m_program.control_phi_predecessors.find(&inst);
	    control != m_program.control_phi_predecessors.end()) {
		bool found = false;
		for (size_t index = 0; index < inst.NumArgs(); ++index) {
			if (m_reachable_blocks != nullptr &&
			    !m_reachable_blocks->at(control->second[index])) continue;
			uint64_t incoming = 0;
			if (!Arg(inst, index, incoming) || (found && incoming != result)) return false;
			result = incoming;
			found = true;
		}
		return found;
	}
	const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
	return !value.IsEmpty() && EvaluateWide(value, result);
}

bool SrtWalker::EvaluateExtract(const Inst& inst, uint64_t& result) {
	const auto index = inst.Arg(1).Resolve();
	if (!index.IsImmediate() || index.GetType() != Type::U32) {
		return false;
	}
	const auto component = index.U32();
	if (component >= 2u) {
		return false;
	}
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
		uint64_t packed = 0;
		if (!Arg(inst, 0, packed)) {
			return false;
		}
		result = static_cast<uint32_t>(packed >> (component * 32u));
		return true;
	}
	const auto* source = inst.Arg(0).ResolveInstruction();
	if (source == nullptr) {
		return false;
	}
	if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
		return EvaluateWide(source->Arg(component), result);
	}
	if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
		uint64_t lhs = 0;
		uint64_t rhs = 0;
		if (!Arg(*source, 0, lhs) || !Arg(*source, 1, rhs)) {
			return false;
		}
		const auto sum =
		    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
		result =
		    component == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
		return true;
	}
	return false;
}

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
	const auto flags = inst.Flags<SrtReadFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const auto& mem    = m_program.memory_info[flags.index];
	const bool vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	const auto guard = vector ? inst.Arg(4).Resolve() : Value {};
	if (vector && (guard.IsImmediate() || guard != m_active_mask)) {
		uint64_t enabled = 0;
		if (!Arg(inst, 4, enabled)) return false;
		if (enabled == 0u) {
			result = 0u;
			return true;
		}
	}
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (handle == nullptr) {
		return false;
	}
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!Arg(*handle, 0, low) || !Arg(*handle, 1, high) || !Arg(inst, 1, offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
	uint64_t   address   = 0;
	if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer || vector) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset =
		    (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
		const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		if (vector) {
			// Only the uniform, unswizzled structured DWORD address is evaluated on the host.
			if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u)
				return false;
			if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
				result = 0u;
				return true;
			}
		}
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			result = 0u;
			return true;
		}
		address = (base & ~uint64_t {3}) + byte_offset;
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	uint32_t word = 0;
	if (m_runtime.read_slot != nullptr) {
		// A load shared by several slots has no single slot.
		const auto roots = std::ranges::count_if(m_program.srt_reads, [&](const SrtRead& read) {
			return read.value.Resolve().TryInstruction() == &inst;
		});
		const auto root = std::ranges::find_if(m_program.srt_reads, [&](const SrtRead& read) {
			return read.value.Resolve().TryInstruction() == &inst;
		});
		*m_runtime.read_slot = roots == 1   ? root->flat_offset
		                       : roots == 0 ? RawReadSlot(inst.Flags<SrtReadFlags>().index)
		                                    : NoSrtSlot;
	}
	bool       read   = true;
	const auto reader = vector ? m_runtime.read_specialization_memory : m_runtime.read_memory;
	if (reader != nullptr) {
		read = reader(m_runtime.userdata, address, {&word, 1});
	} else {
		if (vector) return false;
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	if (m_runtime.read_slot != nullptr) {
		*m_runtime.read_slot = NoSrtSlot;
	}
	result = word;
	return read;
}

bool SrtWalker::EvaluateInst(const Inst& inst, uint64_t& result) {
	uint64_t   a       = 0;
	uint64_t   b       = 0;
	uint64_t   c       = 0;
	const auto binary  = [&]() { return Arg(inst, 0, a) && Arg(inst, 1, b); };
	const auto ternary = [&]() {
		return Arg(inst, 0, a) && Arg(inst, 1, b) && Arg(inst, 2, c);
	};
	switch (inst.GetOpcode()) {
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[reg - m_program.user_data_base];
			return true;
		}
		case ValueOpcode::GetShaderBase: result = m_runtime.shader_base; return true;
		case ValueOpcode::Phi: return EvaluatePhi(inst, result);
		case ValueOpcode::ReadFirstLane: {
			if (!m_active_mask.IsEmpty() && inst.Arg(1).Resolve() == m_active_mask) {
				return EvaluateWide(inst.Arg(0), result);
			}
			SrtWalker clean_active(*this, inst.Arg(1), true);
			SrtWalker active(*this, inst.Arg(1), false, &clean_active);
			return active.EvaluateWide(inst.Arg(0), result);
		}
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: return Arg(inst, 0, result);
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result);
		case ValueOpcode::CompositeConstructU64:
			if (!binary()) {
				return false;
			}
			result = static_cast<uint32_t>(a) |
			         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32:
			if (IsRawRead(m_program, inst)) {
				if (m_clean_evaluator != nullptr &&
				    (inst.Flags<SrtReadFlags>().clean != 0u ||
				     (inst.GetOpcode() == ValueOpcode::LoadBufferU32 &&
				      m_clean_evaluator->m_active_mask == m_active_mask))) {
					return m_clean_evaluator->EvaluateWide(Value(const_cast<Inst*>(&inst)), result);
				}
				return EvaluateRawRead(inst, result);
			}
			break;
		case ValueOpcode::IAdd32:
			if (binary()) {
				result = static_cast<uint32_t>(a + b);
				return true;
			}
			return false;
		case ValueOpcode::IAdd64:
			if (binary()) {
				result = a + b;
				return true;
			}
			return false;
		case ValueOpcode::ISub32:
			if (binary()) {
				result = static_cast<uint32_t>(a - b);
				return true;
			}
			return false;
		case ValueOpcode::ISub64:
			if (binary()) {
				result = a - b;
				return true;
			}
			return false;
		case ValueOpcode::IMul32:
			if (binary()) {
				result = static_cast<uint32_t>(a * b);
				return true;
			}
			return false;
		case ValueOpcode::IMul64:
			if (binary()) {
				result = a * b;
				return true;
			}
			return false;
		case ValueOpcode::UMulHi:
			if (binary()) {
				result = (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
				return true;
			}
			return false;
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
			if (binary()) {
				result = inst.GetOpcode() == ValueOpcode::UMin32
				    ? std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b))
				    : std::max(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::ConvertF32U32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
				return true;
			}
			return false;
		case ValueOpcode::ConvertU32F32:
			if (Arg(inst, 0, a)) {
				const auto value = Float32(a);
				if (!std::isfinite(value) || value < 0.0f ||
				    static_cast<double>(value) > UINT32_MAX) {
					return false;
				}
				result = static_cast<uint32_t>(value);
				return true;
			}
			return false;
		case ValueOpcode::FPMul32:
			if (binary()) {
				result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
				return true;
			}
			return false;
		case ValueOpcode::FPTrunc32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
				return true;
			}
			return false;
		case ValueOpcode::FPRecipIFlag32:
			if (Arg(inst, 0, a)) {
				const auto exponent = (a >> 23u) & 0xffu;
				// Normal positive powers of two have exact normal reciprocals in every FP mode.
				if ((a & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
				result = (254u - exponent) << 23u;
				return true;
			}
			return false;
		case ValueOpcode::FPIsNan32:
			if (Arg(inst, 0, a)) {
				result = std::isnan(Float32(a));
				return true;
			}
			return false;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (binary()) {
				const auto operand = [&](uint64_t bits) {
					if (inst.Flags<FPCompareFlags>().flush_input_denorms &&
					    (bits & 0x7fffffffu) < 0x00800000u) {
						bits &= 0x80000000u;
					}
					return Float32(bits);
				};
				result = inst.GetOpcode() == ValueOpcode::FPOrdLessThanEqual32
				             ? operand(a) <= operand(b)
				             : operand(a) >= operand(b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd32:
			if (binary()) {
				result = static_cast<uint32_t>(a & b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd64:
			if (binary()) {
				result = a & b;
				return true;
			}
			return false;
		case ValueOpcode::BitwiseOr32:
			if (binary()) {
				result = static_cast<uint32_t>(a | b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseXor32:
			if (binary()) {
				result = static_cast<uint32_t>(a ^ b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseNot32:
			if (Arg(inst, 0, a)) {
				result = ~static_cast<uint32_t>(a);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) << (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical64:
			if (binary()) {
				result = a << (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) >> (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical64:
			if (binary()) {
				result = a >> (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic32:
			if (binary()) {
				result = static_cast<uint32_t>(
				    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic64:
			if (binary()) {
				result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
				return true;
			}
			return false;
		case ValueOpcode::BitFieldUExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				const auto mask = width == 32u  ? UINT32_MAX
				                  : width == 0u ? 0u
				                                : (uint32_t {1} << width) - 1u;
				result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldSExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					result = 0;
					return true;
				}
				const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
				auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
				if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
					bits |= ~mask;
				}
				result = bits;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldInsert: {
			uint64_t d = 0;
			if (!ternary() || !Arg(inst, 3, d)) {
				return false;
			}
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask =
			    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) |
			         ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (predicate.EvaluateWide(inst.Arg(0), a)) {
				return Arg(inst, a != 0u ? 1u : 2u, result);
			}
			return false;
		}
		case ValueOpcode::IEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::INotEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThanEqual32:
			// A zero lower bound accepts every U32, including a varying lane value.
			if (!Arg(inst, 0, a)) return false;
			if (a == 0u) {
				result = 1u;
				return true;
			}
			if (Arg(inst, 1, b)) {
				result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::UGreaterThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::SGreaterThanEqual32:
			if (binary()) {
				result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
				         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::LogicalAnd: {
			const bool left = Arg(inst, 0, a);
			if (left && a == 0u) {
				result = 0u;
				return true;
			}
			if (Arg(inst, 1, b) && (b == 0u || left)) {
				result = b != 0u;
				return true;
			}
			if (!left) {
				// Only lanes satisfying the left EXEC factor can make the conjunction
				// true. Use a separate memo so this assumption cannot escape its RHS.
				SrtWalker active(m_clean_evaluator != nullptr ? *m_clean_evaluator : *this,
				                 inst.Arg(0));
				if (active.EvaluateWide(inst.Arg(1), b) && b == 0u) {
					result = 0u;
					return true;
				}
			}
			return false;
		}
		case ValueOpcode::LogicalOr: {
			const bool left = Arg(inst, 0, a);
			if (left && a != 0u) {
				result = 1u;
				return true;
			}
			if (!Arg(inst, 1, b) || (b == 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case ValueOpcode::LogicalXor:
			if (binary()) {
				result = (a != 0u) != (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::ConditionRef: return Arg(inst, 0, result);
		case ValueOpcode::LogicalNot:
			if (Arg(inst, 0, a)) {
				result = a == 0u;
				return true;
			}
			return false;
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU8:
		case ValueOpcode::UndefU16:
		case ValueOpcode::UndefU32:
		case ValueOpcode::UndefU64: return false;
		default: break;
	}
	return false;
}
bool SrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	const auto& descriptor = m_program.descriptor_sources[source];
	result = {};
	result.dword_count = descriptor.dword_count;
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		if (!Evaluate(descriptor.dwords[index], result.dwords[index])) {
			return false;
		}
	}
	return true;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	if (!m_program.srt_plan_complete) return false;
	if (m_program.srt_code && m_runtime.read_memory == nullptr) {
		flat.resize(m_program.srt_reads.size());
		m_program.Scratch().active_sources.clear();
		m_context.values.resize(m_program.evaluation_value_count);
		using Walker = bool KYTY_SYSV_ABI (*)(const uint32_t*, uint32_t*,
		    ResourcePlan::EvaluationContext::Entry*, uint64_t, size_t);
		return m_program.srt_code->getCode<Walker>()(m_runtime.user_data.data(), flat.data(),
		    m_context.values.data(), m_context.generation, m_runtime.user_data.size());
	}
	const auto refresh = [&](uint32_t slot) {
		if (slot >= m_program.srt_reads.size()) return false;
		const auto& read = m_program.srt_reads[slot];
		const auto* inst = read.value.ResolveInstruction();
		const bool clean = inst != nullptr && inst->Flags<SrtReadFlags>().clean != 0u;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
			return false;
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		return read.flat_offset < flat.size() && evaluator.Evaluate(read.value, flat[read.flat_offset]);
	};
	auto& active = m_program.Scratch().active_sources;
	if (m_program.control_flow.empty()) {
		active.clear();
		flat.resize(m_program.srt_reads.size());
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); ++slot) {
			if (!refresh(slot)) return false;
		}
		return true;
	}
	auto& visited = m_program.Scratch().visited_blocks;
	auto& reachable = m_program.Scratch().reachable_blocks;
	auto& pending = m_program.Scratch().pending_blocks;
	visited.assign(m_program.control_flow.size(), 1u);
	ResetUnknowns();
	m_reachable_blocks = &visited;
	if (m_clean_evaluator != nullptr) {
		m_clean_evaluator->ResetUnknowns();
		m_clean_evaluator->m_reachable_blocks = &visited;
	}
	for (;;) {
		reachable.assign(m_program.control_flow.size(), 0u);
		pending.clear();
		pending.push_back(0u);
		while (!pending.empty()) {
			const auto index = pending.back();
			pending.pop_back();
			if (reachable.at(index)) continue;
			reachable[index] = 1u;
			const auto& block = m_program.control_flow[index];
			uint32_t condition = 0;
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (!block.condition.IsEmpty() &&
			    (m_runtime.read_specialization_memory != nullptr || !m_program.capture_specialization_reads) &&
			    predicate.Evaluate(block.condition, condition)) {
				pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
			} else {
				pending.insert(pending.end(), block.successors.begin(), block.successors.end());
			}
		}
		if (reachable == visited) break;
		visited.swap(reachable);
		if (m_program.control_phi_predecessors.empty()) break;
		// Only unknown values need another evaluation: successful Phi facts remain
		// true as their possible predecessors shrink. Keep uniform-read memos stable.
		ResetUnknowns();
		if (m_clean_evaluator != nullptr) m_clean_evaluator->ResetUnknowns();
	}
	flat.assign(m_program.srt_reads.size(), 0u);
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow)
		for (const auto source: block.sources) active.at(source) = 0u;
	for (uint32_t index = 0; index < visited.size(); ++index) {
		if (!visited[index]) continue;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) active[source] = 1u;
		for (const auto slot: block.srt_reads)
			if (!refresh(slot)) return false;
	}
	return true;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}

bool ValidateWaveUniformValue(const ResourcePlan& program, Value value) {
	return RuntimeValidator(program, RuntimeValueType::Any, true).Run(value);
}

namespace {

class SrtPlanCompiler {
public:
	using Node  = CompiledSrtPlan::Node;
	using Shape = CompiledSrtPlan::Shape;

	SrtPlanCompiler(const ResourcePlan& program, CompiledSrtPlan& plan)
	    : m_program(program), m_plan(plan) {}

	bool Valid() const { return m_valid; }

	uint32_t NodeOf(Value value) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			return ImmediateNode(value);
		}
		const auto* inst = value.TryInstruction();
		if (const auto found = m_insts.find(inst); found != m_insts.end()) {
			return found->second;
		}
		const auto id = static_cast<uint32_t>(m_plan.nodes.size());
		m_plan.nodes.emplace_back();
		m_insts.emplace(inst, id);
		// Operands are built first: the node vector may grow meanwhile.
		const auto node = BuildInst(*inst);
		m_plan.nodes[id] = node;
		return id;
	}

private:
	// Nodes compare equal exactly when their resolved Values do, so an EXEC-mask select is
	// recognized by node identity.
	uint32_t ImmediateNode(Value value) {
		Node     node;
		uint64_t payload   = 0;
		bool     equatable = true;
		switch (value.GetType()) {
			case Type::Void: break;
			case Type::ScalarReg: payload = RegIndex(value.ScalarRegister()); break;
			case Type::VectorReg: payload = RegIndex(value.VectorRegister()); break;
			case Type::U1:
				payload   = value.U1();
				node.kind = CompiledSrtPlan::NodeKind::Constant;
				break;
			case Type::U8:
				payload   = value.U8();
				node.kind = CompiledSrtPlan::NodeKind::Constant;
				break;
			case Type::U16:
				payload   = value.U16();
				node.kind = CompiledSrtPlan::NodeKind::Constant;
				break;
			case Type::F16: payload = value.F16Bits(); break;
			case Type::U32:
				payload   = value.U32();
				node.kind = CompiledSrtPlan::NodeKind::Constant;
				break;
			case Type::F32:
				payload   = std::bit_cast<uint32_t>(value.F32Value());
				node.kind = CompiledSrtPlan::NodeKind::Constant;
				break;
			case Type::U64:
				payload   = value.U64();
				node.kind = CompiledSrtPlan::NodeKind::Constant;
				break;
			default: equatable = false; break;
		}
		node.imm = payload;
		const auto key = std::pair {value.GetType(), payload};
		if (equatable) {
			if (const auto found = std::ranges::find(m_immediates, key,
			                                         &std::pair<std::pair<Type, uint64_t>, uint32_t>::first);
			    found != m_immediates.end()) {
				return found->second;
			}
		}
		const auto id = static_cast<uint32_t>(m_plan.nodes.size());
		m_plan.nodes.push_back(node);
		if (equatable) {
			m_immediates.emplace_back(key, id);
		}
		return id;
	}

	// SrtWalker reads these operands without bounds checks beyond Inst::Arg; a plan whose
	// operand lists are shorter keeps the reference walker.
	bool Require(const Inst& inst, size_t count) {
		m_valid &= inst.NumArgs() >= count;
		return m_valid;
	}

	void Operands(const Inst& inst, Node& node, size_t count) {
		if (!Require(inst, count)) {
			return;
		}
		for (size_t index = 0; index < count; index++) {
			node.args[index] = NodeOf(inst.Arg(index));
		}
	}

	Node BuildInst(const Inst& inst) {
		Node node;
		node.op     = inst.GetOpcode();
		node.kind   = CompiledSrtPlan::NodeKind::Instruction;
		node.select = IsRuntimeSelect(node.op) && inst.NumArgs() == 3;
		switch (node.op) {
			case ValueOpcode::GetUserData: {
				if (!Require(inst, 1)) break;
				const auto reg = inst.Arg(0);
				if (reg.TryInstruction() != nullptr || reg.GetType() != Type::ScalarReg) {
					m_valid = false;
					break;
				}
				// A register below the user-data base never evaluates.
				const auto index = RegIndex(reg.ScalarRegister());
				node.kind        = index < m_program.user_data_base
				                       ? CompiledSrtPlan::NodeKind::Invalid
				                       : CompiledSrtPlan::NodeKind::UserData;
				node.imm = index - m_program.user_data_base;
				break;
			}
			case ValueOpcode::GetShaderBase: break;
			case ValueOpcode::Phi: {
				if (const auto control = m_program.control_phi_predecessors.find(&inst);
				    control != m_program.control_phi_predecessors.end()) {
					if (control->second.size() != inst.NumArgs()) {
						m_valid = false;
						break;
					}
					std::vector<std::pair<uint32_t, uint32_t>> incoming;
					for (size_t index = 0; index < inst.NumArgs(); ++index) {
						incoming.emplace_back(NodeOf(inst.Arg(index)), control->second[index]);
					}
					node.shape = Shape::ControlPhi;
					node.imm   = m_plan.control_phis.size();
					m_plan.control_phis.push_back(std::move(incoming));
					break;
				}
				node.shape         = Shape::Phi;
				const auto invariant =
				    ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
				node.args[0] = invariant.IsEmpty() ? CompiledSrtPlan::NoNode : NodeOf(invariant);
				break;
			}
			case ValueOpcode::ReadFirstLane: {
				node.shape = Shape::ReadFirstLane;
				if (!Require(inst, 2)) break;
				node.args[0]    = NodeOf(inst.Arg(0));
				const auto mask = inst.Arg(1).Resolve();
				node.args[1]    = mask.IsEmpty() ? CompiledSrtPlan::NoNode : NodeOf(mask);
				break;
			}
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2: {
				node.shape = node.op == ValueOpcode::CompositeExtractU64 ? Shape::ExtractU64
				                                                         : Shape::ExtractU32x2;
				if (!Require(inst, 2)) break;
				const auto index = inst.Arg(1).Resolve();
				if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
					break; // aux2 == 0: evaluation fails.
				}
				node.aux = static_cast<uint8_t>(index.U32());
				if (node.op == ValueOpcode::CompositeExtractU64) {
					node.aux2    = 1;
					node.args[0] = NodeOf(inst.Arg(0));
					break;
				}
				const auto source_value = inst.Arg(0);
				if (source_value.TryInstruction() == nullptr) {
					m_valid = false; // Value::ResolveInstruction rejects an immediate.
					break;
				}
				const auto* source = source_value.ResolveInstruction();
				if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
					node.aux2 = 1;
					if (Require(*source, node.aux + 1u)) {
						node.args[node.aux] = NodeOf(source->Arg(node.aux));
					}
				} else if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
					node.aux2 = 2;
					Operands(*source, node, 2);
				}
				break;
			}
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer: {
				if (!IsRawRead(m_program, inst)) {
					node.shape = Shape::NotRawRead;
					break;
				}
				node.shape = Shape::RawRead;
				node.clean = inst.Flags<SrtReadFlags>().clean != 0u;
				node.memory = inst.Flags<SrtReadFlags>().index;
				if (!Require(inst, 2)) break;
				const auto handle_value = inst.Arg(0);
				if (handle_value.TryInstruction() == nullptr) {
					m_valid = false; // Value::ResolveInstruction rejects an immediate.
					break;
				}
				const auto* handle = handle_value.ResolveInstruction();
				if (!Require(*handle, 2)) break;
				node.handle_args = static_cast<uint8_t>(std::min<size_t>(handle->NumArgs(), 255u));
				node.args[0]     = NodeOf(handle->Arg(0));
				node.args[1]     = NodeOf(handle->Arg(1));
				node.args[4]     = NodeOf(inst.Arg(1));
				if (node.op == ValueOpcode::ReadConstBuffer && handle->NumArgs() == 4u) {
					node.args[2] = NodeOf(handle->Arg(2));
					node.args[3] = NodeOf(handle->Arg(3));
				}
				const auto& memory = m_program.memory_info[inst.Flags<SrtReadFlags>().index];
				node.imm           = static_cast<uint64_t>(
                    static_cast<int64_t>(static_cast<int32_t>(memory.offset)));
				const auto leaf = [&](uint32_t id) {
					return m_plan.nodes[id].kind == CompiledSrtPlan::NodeKind::Constant ||
					       m_plan.nodes[id].kind == CompiledSrtPlan::NodeKind::UserData;
				};
				if (node.op == ValueOpcode::LoadAddressU32 && leaf(node.args[0]) &&
				    leaf(node.args[1]) &&
				    m_plan.nodes[node.args[4]].kind == CompiledSrtPlan::NodeKind::Constant) {
					// The relative offset of EvaluateRawRead, folded once.
					const auto dynamic = m_plan.nodes[node.args[4]].imm;
					node.shape         = Shape::LeafAddressRead;
					node.imm           = static_cast<uint64_t>(
                        (static_cast<int64_t>(node.imm) & ~int64_t {3}) +
                        static_cast<int64_t>(static_cast<uint32_t>(dynamic) & ~3u));
				}
				break;
			}
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::ConvertF32U32:
			case ValueOpcode::ConvertU32F32:
			case ValueOpcode::FPTrunc32:
			case ValueOpcode::FPRecipIFlag32:
			case ValueOpcode::FPIsNan32:
			case ValueOpcode::BitwiseNot32:
			case ValueOpcode::ConditionRef:
			case ValueOpcode::LogicalNot: Operands(inst, node, 1); break;
			case ValueOpcode::CompositeConstructU64:
			case ValueOpcode::IAdd32:
			case ValueOpcode::IAdd64:
			case ValueOpcode::ISub32:
			case ValueOpcode::ISub64:
			case ValueOpcode::IMul32:
			case ValueOpcode::IMul64:
			case ValueOpcode::UMulHi:
			case ValueOpcode::UMin32:
			case ValueOpcode::UMax32:
			case ValueOpcode::FPMul32:
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseAnd64:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32:
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftLeftLogical64:
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftRightLogical64:
			case ValueOpcode::ShiftRightArithmetic32:
			case ValueOpcode::ShiftRightArithmetic64:
			case ValueOpcode::IEqual32:
			case ValueOpcode::INotEqual32:
			case ValueOpcode::ULessThan32:
			case ValueOpcode::ULessThanEqual32:
			case ValueOpcode::UGreaterThan32:
			case ValueOpcode::SGreaterThanEqual32:
			case ValueOpcode::LogicalAnd:
			case ValueOpcode::LogicalOr:
			case ValueOpcode::LogicalXor: Operands(inst, node, 2); break;
			case ValueOpcode::FPOrdLessThanEqual32:
			case ValueOpcode::FPOrdGreaterThanEqual32:
				node.aux = inst.Flags<FPCompareFlags>().flush_input_denorms ? 1u : 0u;
				Operands(inst, node, 2);
				break;
			case ValueOpcode::BitFieldUExtract:
			case ValueOpcode::BitFieldSExtract:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectF32: Operands(inst, node, 3); break;
			case ValueOpcode::BitFieldInsert: Operands(inst, node, 4); break;
			case ValueOpcode::LoadBufferU32:
				// Raw vector reads depend on the EXEC guard and the clean evaluator: keep the
				// reference walker for plans that have them.
				m_valid &= !IsRawRead(m_program, inst);
				break;
			default: break; // SrtWalker::EvaluateInst fails for every other opcode.
		}
		return node;
	}

	const ResourcePlan&                                        m_program;
	CompiledSrtPlan&                                           m_plan;
	std::unordered_map<const Inst*, uint32_t>                  m_insts;
	std::vector<std::pair<std::pair<Type, uint64_t>, uint32_t>> m_immediates;
	bool                                                       m_valid = true;
};

} // namespace

const CompiledSrtPlan* CompileSrtPlan(const ResourcePlan& program) {
	if (program.compiled_srt_built) {
		return program.compiled_srt.get();
	}
	program.compiled_srt_built = true;
	auto            plan = std::make_shared<CompiledSrtPlan>();
	SrtPlanCompiler compiler(program, *plan);
	const auto      add_root = [&](Value value) {
		value = value.Resolve();
		if (std::ranges::none_of(plan->roots, [&](const auto& root) { return root.first == value; })) {
			plan->roots.emplace_back(value, compiler.NodeOf(value));
		}
	};
	plan->srt_plan_complete = program.srt_plan_complete;
	plan->srt_reads.reserve(program.srt_reads.size());
	for (const auto& read: program.srt_reads) {
		const auto* inst = read.value.Resolve().TryInstruction();
		plan->srt_reads.push_back({compiler.NodeOf(read.value), read.flat_offset,
		                           inst != nullptr && inst->Flags<SrtReadFlags>().clean != 0u});
	}
	plan->descriptors.resize(program.descriptor_sources.size());
	for (size_t source = 0; source < program.descriptor_sources.size(); source++) {
		const auto& descriptor = program.descriptor_sources[source];
		auto&       compiled   = plan->descriptors[source];
		compiled.dwords.fill(CompiledSrtPlan::NoNode);
		compiled.dword_count = std::min<uint32_t>(descriptor.dword_count, 8u);
		for (uint32_t index = 0; index < compiled.dword_count; index++) {
			compiled.dwords[index] = compiler.NodeOf(descriptor.dwords[index]);
		}
		if (descriptor.indirect_descriptor.has_value()) {
			add_root(descriptor.indirect_descriptor->key_count);
			add_root(descriptor.indirect_descriptor->selector_first);
			add_root(descriptor.indirect_descriptor->selector_mask);
		}
	}
	for (const auto& block: program.control_flow) {
		plan->conditions.push_back(block.condition.IsEmpty() ? CompiledSrtPlan::NoNode
		                                                     : compiler.NodeOf(block.condition));
	}
	for (uint32_t index = 0; index < program.uniform_fill.fill.words && index < 4u; index++) {
		add_root(program.uniform_fill.values[index]);
	}
	if (!compiler.Valid()) {
		return nullptr;
	}
	// A load of no slot is one the shader issues itself; a load shared by several slots has no
	// single slot.
	plan->read_slots.assign(plan->nodes.size(), NoSrtSlot);
	std::vector<uint32_t> roots(plan->nodes.size());
	for (const auto& read: plan->srt_reads) {
		if (read.node < plan->nodes.size()) {
			roots[read.node]++;
			plan->read_slots[read.node] = read.flat_offset;
		}
	}
	for (size_t node = 0; node < plan->nodes.size(); node++) {
		if (roots[node] == 0u) {
			plan->read_slots[node] = plan->nodes[node].memory != UINT32_MAX
			                             ? RawReadSlot(plan->nodes[node].memory)
			                             : NoSrtSlot;
		} else if (roots[node] > 1u) {
			plan->read_slots[node] = NoSrtSlot;
		}
	}
	program.compiled_srt = std::move(plan);
	return program.compiled_srt.get();
}

namespace {

CompiledSrtPlan::Context& AcquireCompiledContext(const CompiledSrtPlan& plan, const SrtRuntime& runtime,
                                                 bool clean_evaluator, uint32_t mask, bool nested) {
	CompiledSrtPlan::Context* context = nullptr;
	if (!nested) {
		// The refresh's clean and ordinary walkers; the first one starts a session.
		const auto depth = plan.lanes[t_srt_scratch_lane].depth++;
		if (depth == 0) {
			++plan.lanes[t_srt_scratch_lane].session;
		}
		EXIT_IF(depth >= plan.lanes[t_srt_scratch_lane].top_contexts.size());
		context = &plan.lanes[t_srt_scratch_lane].top_contexts[depth];
	} else {
		const auto* read   = reinterpret_cast<const void*>(runtime.read_memory);
		const auto* strict = reinterpret_cast<const void*>(runtime.read_specialization_memory);
		CompiledSrtPlan::Context* unused = nullptr;
		for (auto& candidate: plan.lanes[t_srt_scratch_lane].nested_contexts) {
			if (candidate.session == plan.lanes[t_srt_scratch_lane].session && candidate.mask == mask &&
			    candidate.read == read && candidate.strict == strict &&
			    candidate.clean_evaluator == clean_evaluator) {
				return candidate;
			}
			if (candidate.session != plan.lanes[t_srt_scratch_lane].session && unused == nullptr) {
				unused = &candidate;
			}
		}
		context                  = unused != nullptr ? unused : &plan.lanes[t_srt_scratch_lane].nested_contexts.emplace_back();
		context->session         = plan.lanes[t_srt_scratch_lane].session;
		context->mask            = mask;
		context->read            = read;
		context->strict          = strict;
		context->clean_evaluator = clean_evaluator;
	}
	// Sized once: recursive evaluation keeps references into the memo.
	if (context->values.size() != plan.nodes.size()) {
		context->values.resize(plan.nodes.size());
	}
	// Bit 0 marks a node being evaluated, bit 1 a completed unknown.
	context->generation += 4;
	return *context;
}

} // namespace

CompiledSrtWalker::CompiledSrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                                     CompiledSrtWalker* clean_evaluator)
    : CompiledSrtWalker(program, runtime, clean_evaluator, NoNode, false) {}

CompiledSrtWalker::CompiledSrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                                     CompiledSrtWalker* clean_evaluator, uint32_t active_mask,
                                     bool nested)
    : m_program(program), m_plan(*program.compiled_srt), m_runtime(runtime),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask), m_nested(nested),
      m_context(AcquireCompiledContext(m_plan, runtime, clean_evaluator != nullptr, active_mask,
                                       nested)) {}

CompiledSrtWalker::~CompiledSrtWalker() {
	if (!m_nested) {
		--m_plan.lanes[t_srt_scratch_lane].depth;
	}
}

bool CompiledSrtWalker::Evaluate(Value value, uint32_t& result) {
	value = value.Resolve();
	const auto root = std::ranges::find_if(m_plan.roots,
	                                       [&](const auto& candidate) { return candidate.first == value; });
	uint64_t wide = 0;
	// Unregistered values have types SrtWalker cannot evaluate either.
	if (root == m_plan.roots.end() || !EvaluateNode(root->second, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

bool CompiledSrtWalker::EvaluateNode(uint32_t id, uint64_t& result) {
	const auto& node = m_plan.nodes[id];
	if (node.kind == CompiledSrtPlan::NodeKind::Constant) {
		result = node.imm;
		return true;
	}
	if (m_active_mask != NoNode && id == m_active_mask) {
		result = 1u;
		return true;
	}
	if (node.kind == CompiledSrtPlan::NodeKind::Invalid) {
		return false;
	}
	if (node.kind == CompiledSrtPlan::NodeKind::UserData) {
		if (node.imm >= m_runtime.user_data.size()) {
			return false;
		}
		result = m_runtime.user_data[node.imm];
		return true;
	}
	if (m_active_mask != NoNode && node.select && node.args[0] == m_active_mask) {
		return EvaluateNode(node.args[1], result);
	}
	auto&      memo       = m_context.values[id];
	const auto generation = m_context.generation;
	if (memo.generation == generation) {
		result = memo.value;
		return true;
	}
	// Visiting and completed-unknown entries both stop repeated evaluation, as in SrtWalker.
	if (memo.generation == (generation | 1u) || memo.generation == (generation | 2u)) {
		return false;
	}
	memo.generation = generation | 1u;
	uint64_t out    = 0;
	bool evaluated = false;
	if (node.clean && m_clean_evaluator != nullptr) {
		// SrtWalker evaluates clean scalar reads in the clean evaluator.
		evaluated = m_clean_evaluator->EvaluateNode(id, out);
	} else if (node.shape == CompiledSrtPlan::Shape::LeafAddressRead) {
		uint64_t low     = 0;
		uint64_t high    = 0;
		uint64_t address = 0;
		evaluated        = EvaluateLeaf(node.args[0], low) && EvaluateLeaf(node.args[1], high) &&
		            AddSignedAddress((((high << 32u) | static_cast<uint32_t>(low)) & AddressMask) &
		                                 ~uint64_t {3},
		                             static_cast<int64_t>(node.imm), address) &&
		            ReadWord(id, address, out);
	} else {
		evaluated = EvaluateInst(id, out);
	}
	if (!evaluated) {
		memo.generation = generation | 2u;
		return false;
	}
	memo.value      = out;
	memo.generation = generation;
	result          = out;
	return true;
}

bool CompiledSrtWalker::EvaluateRawRead(uint32_t id, uint64_t& result) {
	const auto& node   = m_plan.nodes[id];
	uint64_t    low    = 0;
	uint64_t    high   = 0;
	uint64_t    offset = 0;
	if (!EvaluateNode(node.args[0], low) || !EvaluateNode(node.args[1], high) ||
	    !EvaluateNode(node.args[4], offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(node.imm);
	uint64_t   address   = 0;
	if (node.op == ValueOpcode::ReadConstBuffer) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (node.handle_args != 4u || !EvaluateNode(node.args[2], records) ||
		    !EvaluateNode(node.args[3], word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset =
		    (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
		const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		const auto size   = stride == 0u
		                        ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                        : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			// Out-of-range scalar descriptor reads return zero.
			result = 0u;
			return true;
		}
		address = (base & ~uint64_t {3}) + byte_offset;
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	return ReadWord(id, address, result);
}

bool CompiledSrtWalker::EvaluateLeaf(uint32_t id, uint64_t& result) const {
	const auto& node = m_plan.nodes[id];
	if (node.kind == CompiledSrtPlan::NodeKind::Constant) {
		result = node.imm;
		return true;
	}
	if (node.imm >= m_runtime.user_data.size()) {
		return false;
	}
	result = m_runtime.user_data[node.imm];
	return true;
}

bool CompiledSrtWalker::ReadWord(uint32_t node, uint64_t address, uint64_t& result) {
	uint32_t word = 0;
	if (m_runtime.read_slot != nullptr) {
		*m_runtime.read_slot =
		    node < m_plan.read_slots.size() ? m_plan.read_slots[node] : NoSrtSlot;
	}
	bool read = true;
	if (m_runtime.read_memory != nullptr) {
		read = m_runtime.read_memory(m_runtime.userdata, address, {&word, 1});
	} else {
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	if (m_runtime.read_slot != nullptr) {
		*m_runtime.read_slot = NoSrtSlot;
	}
	result = word;
	return read;
}

// Mirrors SrtWalker::EvaluateInst on resolved node operands.
bool CompiledSrtWalker::EvaluateInst(uint32_t id, uint64_t& result) {
	const auto& node    = m_plan.nodes[id];
	uint64_t    a       = 0;
	uint64_t    b       = 0;
	uint64_t    c       = 0;
	const auto  binary  = [&]() { return EvaluateNode(node.args[0], a) && EvaluateNode(node.args[1], b); };
	const auto  ternary = [&]() {
        return EvaluateNode(node.args[0], a) && EvaluateNode(node.args[1], b) &&
               EvaluateNode(node.args[2], c);
	};
	const auto float32 = [](uint64_t bits) { return std::bit_cast<float>(static_cast<uint32_t>(bits)); };
	switch (node.op) {
		case ValueOpcode::GetShaderBase: result = m_runtime.shader_base; return true;
		case ValueOpcode::Phi: {
			if (node.shape != CompiledSrtPlan::Shape::ControlPhi) {
				return node.args[0] != NoNode && EvaluateNode(node.args[0], result);
			}
			bool found = false;
			for (const auto& [incoming, block]: m_plan.control_phis[node.imm]) {
				if (m_reachable_blocks != nullptr && !m_reachable_blocks->at(block)) {
					continue;
				}
				uint64_t value = 0;
				if (!EvaluateNode(incoming, value) || (found && value != result)) {
					return false;
				}
				result = value;
				found  = true;
			}
			return found;
		}
		case ValueOpcode::ReadFirstLane: {
			if (m_active_mask != NoNode && node.args[1] == m_active_mask) {
				return EvaluateNode(node.args[0], result);
			}
			const auto        clean_runtime = CleanRuntime(m_runtime);
			CompiledSrtWalker clean_active(m_program, clean_runtime, nullptr, node.args[1], true);
			CompiledSrtWalker active(m_program, m_runtime, &clean_active, node.args[1], true);
			clean_active.m_reachable_blocks = m_reachable_blocks;
			active.m_reachable_blocks       = m_reachable_blocks;
			return active.EvaluateNode(node.args[0], result);
		}
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: return EvaluateNode(node.args[0], result);
		case ValueOpcode::CompositeExtractU64: {
			uint64_t packed = 0;
			if (node.aux2 == 0u || !EvaluateNode(node.args[0], packed)) {
				return false;
			}
			result = static_cast<uint32_t>(packed >> (node.aux * 32u));
			return true;
		}
		case ValueOpcode::CompositeExtractU32x2:
			if (node.aux2 == 1u) {
				return EvaluateNode(node.args[node.aux], result);
			}
			if (node.aux2 == 2u) {
				if (!binary()) {
					return false;
				}
				const auto sum =
				    static_cast<uint64_t>(static_cast<uint32_t>(a)) + static_cast<uint32_t>(b);
				result = node.aux == 0u ? static_cast<uint32_t>(sum)
				                        : static_cast<uint32_t>(sum >> 32u);
				return true;
			}
			return false;
		case ValueOpcode::CompositeConstructU64:
			if (!binary()) {
				return false;
			}
			result = static_cast<uint32_t>(a) | (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
			return node.shape == CompiledSrtPlan::Shape::RawRead && EvaluateRawRead(id, result);
		case ValueOpcode::IAdd32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a + b);
			return true;
		case ValueOpcode::IAdd64:
			if (!binary()) return false;
			result = a + b;
			return true;
		case ValueOpcode::ISub32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a - b);
			return true;
		case ValueOpcode::ISub64:
			if (!binary()) return false;
			result = a - b;
			return true;
		case ValueOpcode::IMul32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a * b);
			return true;
		case ValueOpcode::IMul64:
			if (!binary()) return false;
			result = a * b;
			return true;
		case ValueOpcode::UMulHi:
			if (!binary()) return false;
			result = (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
			return true;
		case ValueOpcode::UMin32:
			if (!binary()) return false;
			result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
			return true;
		case ValueOpcode::UMax32:
			if (!binary()) return false;
			result = std::max(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
			return true;
		case ValueOpcode::ConvertF32U32:
			if (!EvaluateNode(node.args[0], a)) return false;
			result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
			return true;
		case ValueOpcode::ConvertU32F32: {
			if (!EvaluateNode(node.args[0], a)) return false;
			const auto value = float32(a);
			if (!std::isfinite(value) || value < 0.0f || static_cast<double>(value) > UINT32_MAX) {
				return false;
			}
			result = static_cast<uint32_t>(value);
			return true;
		}
		case ValueOpcode::FPMul32:
			if (!binary()) return false;
			result = std::bit_cast<uint32_t>(float32(a) * float32(b));
			return true;
		case ValueOpcode::FPTrunc32:
			if (!EvaluateNode(node.args[0], a)) return false;
			result = std::bit_cast<uint32_t>(std::trunc(float32(a)));
			return true;
		case ValueOpcode::FPRecipIFlag32: {
			if (!EvaluateNode(node.args[0], a)) return false;
			const auto exponent = (a >> 23u) & 0xffu;
			// Normal positive powers of two have exact normal reciprocals in every FP mode.
			if ((a & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
			result = (254u - exponent) << 23u;
			return true;
		}
		case ValueOpcode::FPIsNan32:
			if (!EvaluateNode(node.args[0], a)) return false;
			result = std::isnan(float32(a));
			return true;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32: {
			if (!binary()) return false;
			const auto operand = [&](uint64_t bits) {
				if (node.aux != 0u && (bits & 0x7fffffffu) < 0x00800000u) {
					bits &= 0x80000000u;
				}
				return float32(bits);
			};
			result = node.op == ValueOpcode::FPOrdLessThanEqual32 ? operand(a) <= operand(b)
			                                                      : operand(a) >= operand(b);
			return true;
		}
		case ValueOpcode::BitwiseAnd32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a & b);
			return true;
		case ValueOpcode::BitwiseAnd64:
			if (!binary()) return false;
			result = a & b;
			return true;
		case ValueOpcode::BitwiseOr32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a | b);
			return true;
		case ValueOpcode::BitwiseXor32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a ^ b);
			return true;
		case ValueOpcode::BitwiseNot32:
			if (!EvaluateNode(node.args[0], a)) return false;
			result = ~static_cast<uint32_t>(a);
			return true;
		case ValueOpcode::ShiftLeftLogical32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) << (b & 31u);
			return true;
		case ValueOpcode::ShiftLeftLogical64:
			if (!binary()) return false;
			result = a << (b & 63u);
			return true;
		case ValueOpcode::ShiftRightLogical32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) >> (b & 31u);
			return true;
		case ValueOpcode::ShiftRightLogical64:
			if (!binary()) return false;
			result = a >> (b & 63u);
			return true;
		case ValueOpcode::ShiftRightArithmetic32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
			return true;
		case ValueOpcode::ShiftRightArithmetic64:
			if (!binary()) return false;
			result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
			return true;
		case ValueOpcode::BitFieldUExtract: {
			if (!ternary()) return false;
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			const auto mask = width == 32u  ? UINT32_MAX
			                  : width == 0u ? 0u
			                                : (uint32_t {1} << width) - 1u;
			result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
			return true;
		}
		case ValueOpcode::BitFieldSExtract: {
			if (!ternary()) return false;
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = 0;
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
			auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
			if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
				bits |= ~mask;
			}
			result = bits;
			return true;
		}
		case ValueOpcode::BitFieldInsert: {
			uint64_t d = 0;
			if (!ternary() || !EvaluateNode(node.args[3], d)) {
				return false;
			}
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) | ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (!predicate.EvaluateNode(node.args[0], a)) {
				return false;
			}
			return EvaluateNode(node.args[a != 0u ? 1u : 2u], result);
		}
		case ValueOpcode::IEqual32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::INotEqual32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::ULessThan32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::ULessThanEqual32:
			// A zero lower bound accepts every U32, including a varying lane value.
			if (!EvaluateNode(node.args[0], a)) return false;
			if (a == 0u) {
				result = 1u;
				return true;
			}
			if (!EvaluateNode(node.args[1], b)) return false;
			result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::UGreaterThan32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::SGreaterThanEqual32:
			if (!binary()) return false;
			result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
			         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
			return true;
		// Short-circuit like SrtWalker: a known false (true) operand decides an And (Or) whose
		// other operand is unknown.
		case ValueOpcode::LogicalAnd: {
			const bool left = EvaluateNode(node.args[0], a);
			if (left && a == 0u) {
				result = 0u;
				return true;
			}
			if (EvaluateNode(node.args[1], b) && (b == 0u || left)) {
				result = b != 0u;
				return true;
			}
			if (!left) {
				// Only lanes satisfying the left EXEC factor can make the conjunction true;
				// evaluate the right operand in a separate memo under that assumption.
				auto&             parent = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
				CompiledSrtWalker active(m_program, parent.m_runtime, nullptr, node.args[0], true);
				active.m_reachable_blocks = parent.m_reachable_blocks;
				if (active.EvaluateNode(node.args[1], b) && b == 0u) {
					result = 0u;
					return true;
				}
			}
			return false;
		}
		case ValueOpcode::LogicalOr: {
			const bool left = EvaluateNode(node.args[0], a);
			if (left && a != 0u) {
				result = 1u;
				return true;
			}
			if (!EvaluateNode(node.args[1], b) || (b == 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case ValueOpcode::LogicalXor:
			if (!binary()) return false;
			result = (a != 0u) != (b != 0u);
			return true;
		case ValueOpcode::ConditionRef: return EvaluateNode(node.args[0], result);
		case ValueOpcode::LogicalNot:
			if (!EvaluateNode(node.args[0], a)) return false;
			result = a == 0u;
			return true;
		default: return false;
	}
}

void CompiledSrtWalker::ResetUnknowns() const {
	if (m_plan.control_phis.empty()) {
		return;
	}
	const auto reset = [](CompiledSrtPlan::Context& context) {
		for (auto& entry: context.values) {
			if (entry.generation == (context.generation | 2u)) {
				entry.generation = 0;
			}
		}
	};
	for (auto& context: m_plan.lanes[t_srt_scratch_lane].top_contexts) {
		reset(context);
	}
	for (auto& context: m_plan.lanes[t_srt_scratch_lane].nested_contexts) {
		if (context.session == m_plan.lanes[t_srt_scratch_lane].session) {
			reset(context);
		}
	}
}

bool CompiledSrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_plan.descriptors.size()) {
		return false;
	}
	const auto& descriptor = m_plan.descriptors[source];
	result                 = {};
	result.dword_count     = descriptor.dword_count;
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		uint64_t wide = 0;
		if (!EvaluateNode(descriptor.dwords[index], wide)) {
			return false;
		}
		result.dwords[index] = static_cast<uint32_t>(wide);
	}
	return true;
}

bool CompiledSrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	// Same walk as SrtWalker::RefreshFlatBuffer: only reads of reachable blocks are evaluated
	// (unreachable ones may dereference pointers the shader never follows), and the active
	// descriptor sources are found on the way.
	if (!m_plan.srt_plan_complete) {
		return false;
	}
	const auto refresh = [&](uint32_t slot) {
		if (slot >= m_plan.srt_reads.size()) {
			return false;
		}
		const auto& read  = m_plan.srt_reads[slot];
		const bool  clean = read.clean;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr)) {
			return false;
		}
		auto&    evaluator = clean ? *m_clean_evaluator : *this;
		uint64_t wide      = 0;
		if (read.flat_offset >= flat.size() || !evaluator.EvaluateNode(read.node, wide)) {
			return false;
		}
		flat[read.flat_offset] = static_cast<uint32_t>(wide);
		return true;
	};
	auto& active = m_program.Scratch().active_sources;
	if (m_program.control_flow.empty()) {
		active.clear();
		flat.resize(m_plan.srt_reads.size());
		for (uint32_t slot = 0; slot < m_plan.srt_reads.size(); ++slot) {
			if (!refresh(slot)) {
				return false;
			}
		}
		return true;
	}
	// As SrtWalker: find the reachable blocks first (to a fixed point when control Phis depend on
	// them), then refresh their reads in block order.
	auto& visited   = m_program.Scratch().visited_blocks;
	auto& reachable = m_program.Scratch().reachable_blocks;
	auto& pending   = m_program.Scratch().pending_blocks;
	visited.assign(m_program.control_flow.size(), 1u);
	ResetUnknowns();
	m_reachable_blocks = &visited;
	if (m_clean_evaluator != nullptr) {
		m_clean_evaluator->m_reachable_blocks = &visited;
	}
	for (;;) {
		reachable.assign(m_program.control_flow.size(), 0u);
		pending.clear();
		pending.push_back(0u);
		while (!pending.empty()) {
			const auto index = pending.back();
			pending.pop_back();
			if (reachable.at(index)) {
				continue;
			}
			reachable[index]  = 1u;
			const auto& block = m_program.control_flow[index];
			uint64_t    condition = 0;
			const auto  node = index < m_plan.conditions.size() ? m_plan.conditions[index] : NoNode;
			auto&       predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (node != NoNode &&
			    (m_runtime.read_specialization_memory != nullptr ||
			     !m_program.capture_specialization_reads) &&
			    predicate.EvaluateNode(node, condition)) {
				pending.push_back(block.successors[static_cast<uint32_t>(condition) != 0u ? 0u : 1u]);
			} else {
				pending.insert(pending.end(), block.successors.begin(), block.successors.end());
			}
		}
		if (reachable == visited) {
			break;
		}
		visited.swap(reachable);
		if (m_plan.control_phis.empty()) {
			break;
		}
		// Only unknown values need another evaluation: successful Phi facts remain true as their
		// possible predecessors shrink.
		ResetUnknowns();
	}
	flat.assign(m_plan.srt_reads.size(), 0u);
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow) {
		for (const auto source: block.sources) {
			active.at(source) = 0u;
		}
	}
	for (uint32_t index = 0; index < visited.size(); ++index) {
		if (!visited[index]) {
			continue;
		}
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) {
			active[source] = 1u;
		}
		for (const auto slot: block.srt_reads) {
			if (!refresh(slot)) {
				return false;
			}
		}
	}
	return true;
}


} // namespace Libs::Graphics::ShaderRecompiler::IR
