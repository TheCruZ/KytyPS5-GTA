#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/abi.h"
#include "common/assert.h"
#include "common/hackNotice.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

// Guest memory starts at HOST_SYSTEM_MANAGED_MIN (memoryAddressSpace.inc), so a table pointer or an
// ordinary read below it comes from a null table pointer. The walker reaches such reads when it
// follows both sides of a branch it cannot decide, so read zero there instead of faulting the host.
constexpr uint64_t GUEST_MEMORY_MIN = 0x40000;

} // namespace

SrtRuntime SrtReadCapture::ObservedRuntime() {
	auto runtime = m_source;
	runtime.userdata = this;
	runtime.read_specialization_memory = m_source.read_specialization_memory != nullptr
	                                         ? Read : nullptr;
	return runtime;
}

bool SrtReadCapture::Read(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (!capture.m_source.read_specialization_memory(capture.m_source.userdata, address, values)) {
		return false;
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool SrtReadCapture::ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (capture.m_source.read_memory != nullptr) {
		if (!capture.m_source.read_memory(capture.m_source.userdata, address, values)) return false;
		if (address < GUEST_MEMORY_MIN) {
			// The reader answered a null table with zeros: nothing is mapped there, so the bytes
			// cannot change and need no entry in the read set that is validated later.
			return true;
		}
	} else if (address < GUEST_MEMORY_MIN) {
		// A null table pointer (see ReadRawWord): zero, and nothing to validate later.
		std::ranges::fill(values, 0u);
		return true;
	} else {
		std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	}
	return true;
}

bool SrtReadCapture::ReadsHostMemory(const SrtRuntime& runtime) {
	return runtime.read_memory == ReadOrdinary &&
	       static_cast<const SrtReadCapture*>(runtime.userdata)->m_source.read_memory == nullptr;
}

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	if (runtime.read_specialization_memory == nullptr) {
		runtime.map_clean_page = nullptr;
	}
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

// LoadAddressU32 / ReadConstBuffer after its operands (EvaluateRawRead's address computation):
// Read with the address to read, Zero for a constant buffer read past the end (reads zero), or
// Fail.
enum class RawAddress : uint8_t { Fail, Read, Zero };
RawAddress ComputeRawAddress(bool const_buffer, int64_t immediate, uint64_t low, uint64_t high,
                             uint64_t offset, uint64_t records, uint64_t& base, uint64_t& address) {
	// A raw scalar address and a constant buffer's base are 48 bits.
	base = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	if (const_buffer) {
		if (immediate < 0) {
			return RawAddress::Fail;
		}
		const auto byte_offset = (static_cast<uint64_t>(immediate) & ~uint64_t {3}) +
		                         (static_cast<uint32_t>(offset) & ~3u);
		const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		const auto size   = stride == 0u
		                        ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                        : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			// A scalar buffer read past the end returns zero,
			// as the shader's own load does (EmitReadConstBuffer). Failing skipped the draw.
			return RawAddress::Zero;
		}
		address = (base & ~uint64_t {3}) + byte_offset;
		return RawAddress::Read;
	}
	const auto relative =
	    (immediate & ~int64_t {3}) + static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
	return AddSignedAddress(base & ~uint64_t {3}, relative, address) ? RawAddress::Read
	                                                                  : RawAddress::Fail;
}

thread_local bool     t_evaluating_condition = false;
thread_local uint64_t t_condition_refusals   = 0;

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
		case ValueOpcode::ConvertS32F32:
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
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
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
		case ValueOpcode::IEqual64:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::INotEqual64:
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

bool ImmediateBits(Value value, uint64_t& bits) {
	switch (value.GetType()) {
		case Type::U1: bits = value.U1(); return true;
		case Type::U8: bits = value.U8(); return true;
		case Type::U16: bits = value.U16(); return true;
		case Type::U32: bits = value.U32(); return true;
		case Type::U64: bits = value.U64(); return true;
		case Type::F32: bits = std::bit_cast<uint32_t>(value.F32Value()); return true;
		default: return false;
	}
}

// Scratch layout in bytes: per descriptor source its eight DWORDs and a known DWORD, one
// activity byte per source, then per image, then one qword per walked value (all ones when
// unknown). The walker's value slots, captured loads (address qword, value dword, pad),
// reachability and states follow.
constexpr uint32_t DescriptorStride = 9u * sizeof(uint32_t);

uint32_t ActiveOffset(const ResourcePlan& plan) {
	return static_cast<uint32_t>(plan.descriptor_sources.size()) * DescriptorStride;
}

uint32_t OutputOffset(const ResourcePlan& plan) {
	const auto active = plan.descriptor_sources.size() + plan.info.images.size();
	return (ActiveOffset(plan) + static_cast<uint32_t>(active) + 7u) & ~7u;
}

const uint8_t* Scratch(const ResourcePlan& plan) {
	return reinterpret_cast<const uint8_t*>(plan.srt_scratch.data());
}

thread_local const Xbyak::CodeGenerator* t_running_walker = nullptr;
thread_local bool                         t_walker_ahead   = false;
thread_local bool                         t_walker_faulted = false;
// Guest memory starts here: a table pointer below it is null.
constexpr uint64_t GuestMemoryStart = 0x40000;

// Every guest load is mov eax, [r10] (41 8b 02), so a fault completes it as a zero read.
void LoadGuest(Xbyak::CodeGenerator& code, const Xbyak::RegExp& address) {
	code.lea(code.r10, code.ptr[address]);
	code.mov(code.eax, code.dword[code.r10]);
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
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU32 || op == ValueOpcode::Void) {
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
			    index.U32() >= 2u || source->GetOpcode() != ValueOpcode::CompositeConstructU32x2) {
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


// Emits one x86 function per plan. Each value, per EXEC-mask context, is a memoized thunk with a
// state byte (0 unevaluated, 1 unknown or in progress, 2 known, 3 a control Phi holding its first
// value) and a qword slot; a reference is `call thunk; cmp state, 2`. Operands that are unknown,
// constant or plain user data are folded at emission. Registers: rdi user data, rsi flat buffer,
// rdx scratch, r8 capture cursor.
class WalkerCompiler {
public:
	WalkerCompiler(ResourcePlan& plan, Xbyak::CodeGenerator& code): m_plan(plan), m_code(code) {}

bool SrtEvaluatingCondition() {
	return t_evaluating_condition;
}

void SrtSetEvaluatingCondition(bool evaluating) {
	t_evaluating_condition = evaluating;
}

uint64_t SrtConditionRefusals() {
	return t_condition_refusals;
}

void SrtNoteConditionRefusal() {
	t_condition_refusals++;
}

bool SrtConditionsNoWait() {
	static const bool no_wait = [] {
		const char* value = std::getenv("KYTY_SRT_CONDITIONS_WAIT");
		return value == nullptr || value[0] != '1';
	}();
	return no_wait;
}

void CompileSrtWalker(ResourcePlan& program) {
	if (!program.resource_tracking_complete || !program.srt_plan_complete ||
	    program.capture_specialization_reads || !program.control_flow.empty() ||
	    program.srt_reads.empty()) return;

private:
	enum class Kind : uint8_t { Unknown, Constant, UserData, ShaderBase, Node };
	struct Operand {
		Kind     kind  = Kind::Unknown;
		uint64_t value = 0;

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
		// Tables below guest memory and underflowing reads fall back to the interpreted walk.
		code.mov(code.r10, GUEST_MEMORY_MIN + static_cast<uint64_t>(-std::min<int64_t>(first, 0)));
		code.cmp(code.rdi, code.r10);
		code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
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
	// masked: the operand depends on its context's EXEC mask, so it is not the root context's.
	struct Ref {
		Operand operand;
		bool    masked = false;
	};
	struct Node {
		const Inst*          inst    = nullptr;
		uint32_t             context = 0;
		Operand              result;
		std::vector<Operand> args;
		uint32_t             slot = 0, state = 0;
		Xbyak::Label         label;
		bool                 pending = true, referenced = false, masked = false;
	};

	static constexpr auto Near = Xbyak::CodeGenerator::T_NEAR;

	bool Thunk(uint32_t index) const {
		const auto& node = m_nodes[index];
		return node.referenced || node.result == Operand {Kind::Node, index};
	}
	static bool MayBeUnknown(const Operand& operand) {
		return operand.kind == Kind::Unknown || operand.kind == Kind::Node;
	}

	uint32_t Context(Value mask) {
		mask = mask.Resolve();
		const auto found = std::ranges::find(m_masks, mask);
		if (found != m_masks.end()) return static_cast<uint32_t>(found - m_masks.begin());
		m_masks.push_back(mask);
		return static_cast<uint32_t>(m_masks.size() - 1u);
	}

	void AddOutput(Value value, uint32_t gate);
	void Consume(std::vector<std::vector<uint32_t>>& gates);
	Ref Evaluate(Value value, uint32_t context);
	Operand Analyze(uint32_t index);
	Operand AnalyzeRead(Node& node, const auto& arg, Operand self);
	void Layout();
	void Gate(std::vector<uint32_t>& gates, const Xbyak::Label& skip);
	void Zero(uint32_t offset, uint32_t size);
	void Branch(const Operand& operand, const Xbyak::Label& unknown);
	void Load(const Xbyak::Reg64& reg, const Operand& operand);
	void Fetch(const Operand& operand, const Xbyak::Label& unknown);
	void Set(void (Xbyak::CodeGenerator::*condition)(const Xbyak::Operand&));
	void Store(const Node& node);
	void EmitNode(uint32_t index);
	void EmitRead(const Node& node);
	void EmitStrict(const Node& node);

	ResourcePlan&                                     m_plan;
	Xbyak::CodeGenerator&                             m_code;
	std::vector<Value>                                m_masks {Value {}};
	std::deque<Node>                                  m_nodes;
	std::map<std::pair<const Inst*, uint32_t>, uint32_t> m_memo;
	std::vector<Value>                                m_outputs;
	std::vector<std::vector<uint32_t>>                m_output_gates;
	Xbyak::Label                                      m_ret;
	uint32_t m_shader_base = 0, m_reach = 0, m_states = 0, m_states_end = 0;
	uint32_t m_user_data = 0, m_captures = 0;
};

void WalkerCompiler::AddOutput(Value value, uint32_t gate) {
	const auto* inst = value.Resolve().TryInstruction();
	if (inst == nullptr) return;
	auto index = inst->WalkOutput();
	if (index >= m_outputs.size() || m_outputs[index].TryInstruction() != inst) {
		index = static_cast<uint32_t>(m_outputs.size());
		inst->SetWalkOutput(index);
		m_outputs.push_back(value.Resolve());
		m_output_gates.emplace_back();
	}
	m_output_gates[index].push_back(gate);
}

// Mirrors MaterializeResources: each descriptor it reads back is gated by the activity of the
// resource root that needs it, and so are an indirect root's selector values.
void WalkerCompiler::Consume(std::vector<std::vector<uint32_t>>& gates) {
	const auto& sources = m_plan.descriptor_sources;
	const auto  consume = [&](uint32_t root) {
		if (root >= sources.size()) return;
		const auto& indirect = sources[root].indirect_descriptor;
		if (!indirect) {
			gates[root].push_back(root);
			return;
		}
		const auto add = [&](uint32_t source) {
			if (source < sources.size()) gates[source].push_back(root);
		};
		if (indirect->sources.empty()) {
			add(indirect->table_source);
			if (indirect->selector) {
				add(indirect->selector->source);
				if (indirect->selector->rows) add(indirect->selector->rows->source);
			}
		}
		for (const auto source: indirect->sources) add(source);
		AddOutput(indirect->key_count, root);
		AddOutput(indirect->selector_first, root);
		AddOutput(indirect->selector_mask, root);
	};
	for (const auto& buffer: m_plan.info.buffers) consume(buffer.source);
	for (const auto& image: m_plan.info.images) consume(image.source);
	for (const auto& sampler: m_plan.info.samplers) {
		const auto* indirect = sampler.source < sources.size() && sources[sampler.source].indirect_descriptor
		    ? &*sources[sampler.source].indirect_descriptor : nullptr;
		if (indirect == nullptr || indirect->sources.empty()) {
			consume(sampler.source);
		} else if (indirect->sources[0] < sources.size()) {
			gates[indirect->sources[0]].push_back(indirect->sources[0]);
		}
	}
	const auto& fill = m_plan.uniform_fill;
	if (fill.fill.words != 0) {
		AddOutput(fill.pattern_size, UINT32_MAX);
		AddOutput(fill.element_count, UINT32_MAX);
		for (const auto value: fill.values) AddOutput(value, UINT32_MAX);
	}
}

WalkerCompiler::Ref WalkerCompiler::Evaluate(Value value, uint32_t context) {
	value = value.Resolve();
	if (value.IsEmpty()) return {};
	if (value.IsImmediate()) {
		uint64_t bits = 0;
		return ImmediateBits(value, bits) ? Ref {{Kind::Constant, bits}} : Ref {};
	}
	const auto mask = m_masks[context];
	if (context != 0 && value == mask) return {{Kind::Constant, 1u}, true};
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return {};
	const auto op = inst->GetOpcode();
	if (context != 0 && IsRuntimeSelect(op) && inst->NumArgs() == 3 && inst->Arg(0).Resolve() == mask) {
		return {Evaluate(inst->Arg(1), context).operand, true};
	}
	switch (op) {
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConditionRef: return Evaluate(inst->Arg(0), context);
		case ValueOpcode::CompositeExtractU32x2: {
			const auto* source = inst->Arg(0).ResolveInstruction();
			const auto  index  = inst->Arg(1).Resolve();
			if (source == nullptr || source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 ||
			    !index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) return {};
			return Evaluate(source->Arg(index.U32()), context);
		}
		case ValueOpcode::ReadFirstLane:
			// The lane read does not depend on the EXEC mask it is read under.
			return {Evaluate(inst->Arg(0), Context(inst->Arg(1))).operand};
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_plan.user_data_base || reg - m_plan.user_data_base >= m_plan.user_data_count) return {};
			return {{Kind::UserData, reg - m_plan.user_data_base}};
		}
		case ValueOpcode::GetShaderBase: return {{Kind::ShaderBase}};
		case ValueOpcode::Phi:
			if (!m_plan.control_phi_predecessors.contains(inst)) {
				const auto invariant = ResolveInvariantPhi(m_plan, value);
				return invariant.IsEmpty() ? Ref {} : Evaluate(invariant, context);
			}
			break;
		default: break;
	}
	const auto [found, inserted] = m_memo.try_emplace({inst, context}, static_cast<uint32_t>(m_nodes.size()));
	const auto index = found->second;
	if (!inserted) {
		auto& node = m_nodes[index];
		if (!node.pending) return {node.result, node.masked};
		// A cycle: the thunk reads as unknown while it runs.
		node.referenced = true;
		return {{Kind::Node, index}, context != 0};
	}
	auto& node   = m_nodes.emplace_back();
	node.inst    = inst;
	node.context = context;
	node.result  = Analyze(index);
	node.pending = false;
	if (context != 0 && !node.masked && !node.referenced) {
		node.result = Evaluate(value, 0).operand;
		return {node.result};
	}
	return {node.result, node.masked};
}

WalkerCompiler::Operand WalkerCompiler::Analyze(uint32_t index) {
	auto&       node = m_nodes[index];
	const auto& inst = *node.inst;
	const auto  self = Operand {Kind::Node, index};
	const auto  arg  = [&](Value value) {
		const auto ref = Evaluate(value, node.context);
		node.masked |= ref.masked;
		return ref.operand;
	};
	const auto constant = [](uint64_t value) { return Operand {Kind::Constant, value}; };
	switch (inst.GetOpcode()) {
		case ValueOpcode::Phi: {
			bool known = false;
			for (size_t i = 0; i < inst.NumArgs(); ++i) {
				node.args.push_back(arg(inst.Arg(i)));
				known |= node.args.back().kind != Kind::Unknown;
			}
			return known ? self : Operand {};
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32: return AnalyzeRead(node, arg, self);
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32: {
			const auto predicate = arg(inst.Arg(0));
			if (predicate.kind == Kind::Unknown) return {};
			if (predicate.kind == Kind::Constant) return arg(inst.Arg(predicate.value != 0u ? 1 : 2));
			node.args = {predicate, arg(inst.Arg(1)), arg(inst.Arg(2))};
			return node.args[1].kind == Kind::Unknown && node.args[2].kind == Kind::Unknown ? Operand {} : self;
		}
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32: {
			// No U32 is below zero and every U32 is at least zero, including an unknown loop-carried
			// or varying lane value. args[0] is that zero-tested bound.
			const bool less  = inst.GetOpcode() == ValueOpcode::ULessThan32;
			const auto bound = arg(inst.Arg(less ? 1 : 0));
			if (bound.kind == Kind::Unknown) return {};
			if (bound.kind == Kind::Constant && static_cast<uint32_t>(bound.value) == 0u) return constant(!less);
			node.args = {bound, arg(inst.Arg(less ? 0 : 1))};
			return node.args[1].kind == Kind::Unknown && bound.kind == Kind::Constant ? Operand {} : self;
		}
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd: {
			// A deciding operand, nonzero for Or and zero for And, fixes the result.
			const bool conjunction = inst.GetOpcode() == ValueOpcode::LogicalAnd;
			const auto decides     = [&](const Operand& x) {
				return x.kind == Kind::Constant && (x.value == 0u) == conjunction;
			};
			const auto a = arg(inst.Arg(0));
			if (decides(a)) return constant(!conjunction);
			const auto b = arg(inst.Arg(1));
			if (decides(b)) return constant(!conjunction);
			if (a.kind == Kind::Constant && b.kind == Kind::Constant) return constant(conjunction);
			if (a.kind == Kind::Constant && b.kind == Kind::Unknown) return {};
			// Only lanes satisfying the left EXEC factor can make a conjunction true, so an
			// unknown left side retries the right one under that factor as its mask.
			Operand refined;
			if (conjunction && MayBeUnknown(a)) {
				refined = Evaluate(inst.Arg(1), Context(inst.Arg(0))).operand;
				if (refined == b || (refined.kind == Kind::Constant && refined.value != 0u)) refined = {};
				if (a.kind == Kind::Unknown && refined.kind == Kind::Constant) return constant(0u);
			}
			if (a.kind == Kind::Unknown && refined.kind == Kind::Unknown &&
			    (b.kind == Kind::Unknown || b.kind == Kind::Constant)) return {};
			node.args = {a, b, refined};
			return self;
		}
		case ValueOpcode::CompositeExtractU64: {
			const auto component = inst.Arg(1).Resolve();
			if (!component.IsImmediate() || component.GetType() != Type::U32 || component.U32() >= 2u) return {};
			[[fallthrough]];
		}
		default:
			// The other runtime-uniform ops are strict; a U32x2 pair is only read through its extracts.
			if (!IsRuntimeUniformOp(inst.GetOpcode()) || inst.GetOpcode() == ValueOpcode::CompositeConstructU32x2)
				return {};
			for (size_t i = 0; i < inst.NumArgs(); ++i) {
				node.args.push_back(arg(inst.Arg(i)));
				if (node.args.back().kind == Kind::Unknown) return {};
			}
			return self;
	}
}

WalkerCompiler::Operand WalkerCompiler::AnalyzeRead(Node& node, const auto& arg, Operand self) {
	const auto& inst = *node.inst;
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (!IsRawRead(m_plan, inst) || handle == nullptr) return {};
	const bool vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	if (vector) {
		const auto guard = arg(inst.Arg(4));
		if (guard.kind == Kind::Unknown) return {};
		if (guard.kind == Kind::Constant && guard.value == 0u) return {Kind::Constant, 0u};
		node.args.push_back(guard);
	}
	const bool buffer = inst.GetOpcode() != ValueOpcode::LoadAddressU32;
	const auto offset = static_cast<int32_t>(m_plan.memory_info[inst.Flags<MemoryFlags>().index].offset);
	bool failed = buffer && (handle->NumArgs() != 4u || offset < 0);
	for (const auto& word: {handle->Arg(0), handle->Arg(1), inst.Arg(1)}) {
		node.args.push_back(arg(word));
		failed |= node.args.back().kind == Kind::Unknown;
	}
	for (uint32_t word = 2; buffer && !failed && word < 4u; ++word) {
		node.args.push_back(arg(handle->Arg(word)));
		failed |= node.args.back().kind == Kind::Unknown;
	}
	if (!failed) return self;
	// A disabled vector lane still reads zero.
	if (!vector || node.args[0].kind == Kind::Constant) return {};
	node.args.resize(1);
	return self;
}

void WalkerCompiler::Layout() {
	uint32_t offset = OutputOffset(m_plan) + static_cast<uint32_t>(m_outputs.size()) * 8u;
	m_shader_base = offset;
	offset += 8u;
	for (uint32_t index = 0; index < m_nodes.size(); ++index) {
		if (!Thunk(index)) continue;
		m_nodes[index].slot = offset;
		offset += 8u;
	}
	m_reach = offset;
	offset += static_cast<uint32_t>(m_plan.control_flow.size());
	m_states = offset;
	for (uint32_t index = 0; index < m_nodes.size(); ++index) {
		if (Thunk(index)) m_nodes[index].state = offset++;
	}
	m_states_end = offset;
	// Captured addresses come last; EmitRead counts them.
	m_plan.srt_captures = (offset + 7u) & ~7u;
}

// Falls through when any gating source is active. The list is not empty.
void WalkerCompiler::Gate(std::vector<uint32_t>& gates, const Xbyak::Label& skip) {
	auto& c = m_code;
	std::ranges::sort(gates);
	if (gates.back() == UINT32_MAX) return;
	gates.erase(std::unique(gates.begin(), gates.end()), gates.end());
	Xbyak::Label active;
	for (size_t i = 0; i < gates.size(); ++i) {
		c.cmp(c.byte[c.rdx + ActiveOffset(m_plan) + gates[i]], 0);
		if (i + 1u < gates.size()) {
			c.jne(active, Near);
		} else {
			c.je(skip, Near);
		}
	}
	c.L(active);
}

void WalkerCompiler::Zero(uint32_t offset, uint32_t size) {
	auto& c = m_code;
	if (size == 0) return;
	c.mov(c.r11, c.rdi);
	c.lea(c.rdi, c.ptr[c.rdx + offset]);
	c.mov(c.ecx, size);
	c.xor_(c.eax, c.eax);
	c.rep();
	c.stosb();
	c.mov(c.rdi, c.r11);
}

// Jumps to unknown unless the operand has a value at run time.
void WalkerCompiler::Branch(const Operand& operand, const Xbyak::Label& unknown) {
	auto& c = m_code;
	if (operand.kind == Kind::Unknown) {
		c.jmp(unknown, Near);
	} else if (operand.kind == Kind::Node) {
		auto& node = m_nodes[operand.value];
		c.call(node.label);
		c.cmp(c.byte[c.rdx + node.state], 2);
		c.jne(unknown, Near);
	}
}

void WalkerCompiler::Load(const Xbyak::Reg64& reg, const Operand& operand) {
	auto& c = m_code;
	switch (operand.kind) {
		case Kind::Constant: c.mov(reg, operand.value); break;
		case Kind::UserData:
			m_user_data = std::max(m_user_data, static_cast<uint32_t>(operand.value) + 1u);
			c.mov(reg.cvt32(), c.dword[c.rdi + operand.value * sizeof(uint32_t)]);
			break;
		case Kind::ShaderBase: c.mov(reg, c.qword[c.rdx + m_shader_base]); break;
		case Kind::Node: c.mov(reg, c.qword[c.rdx + m_nodes[operand.value].slot]); break;
		case Kind::Unknown: break;
	}
}

// Loads the operand into rax, or jumps to unknown when it has no value at run time.
void WalkerCompiler::Fetch(const Operand& operand, const Xbyak::Label& unknown) {
	Branch(operand, unknown);
	Load(m_code.rax, operand);
}

// Sets eax to 1 when the condition holds, else 0.
void WalkerCompiler::Set(void (Xbyak::CodeGenerator::*condition)(const Xbyak::Operand&)) {
	(m_code.*condition)(m_code.al);
	m_code.movzx(m_code.eax, m_code.al);
}

void WalkerCompiler::Store(const Node& node) {
	auto& c = m_code;
	c.mov(c.qword[c.rdx + node.slot], c.rax);
	c.mov(c.byte[c.rdx + node.state], 2);
	c.ret();
}

void WalkerCompiler::EmitNode(uint32_t index) {
	auto&       c    = m_code;
	auto&       node = m_nodes[index];
	const auto& args = node.args;
	c.L(node.label);
	c.cmp(c.byte[c.rdx + node.state], 0);
	c.jne(m_ret, Near);
	c.mov(c.byte[c.rdx + node.state], 1);
	if (node.result != Operand {Kind::Node, index}) {
		// A cycle referenced this value before it folded to another operand.
		Fetch(node.result, m_ret);
		Store(node);
		return;
	}
	switch (node.inst->GetOpcode()) {
		case ValueOpcode::Phi: {
			// Only predecessors reached by this walk contribute, and they must agree. The state is 3
			// once one of them has stored the value; callers read it like 1, as unknown.
			const auto& blocks = m_plan.control_phi_predecessors.at(node.inst);
			const auto  state  = c.byte[c.rdx + node.state];
			for (size_t i = 0; i < args.size(); ++i) {
				Xbyak::Label next, first;
				c.cmp(c.byte[c.rdx + m_reach + blocks[i]], 0);
				c.je(next, Near);
				Fetch(args[i], m_ret);
				c.cmp(state, 1);
				c.je(first, Near);
				c.cmp(c.rax, c.qword[c.rdx + node.slot]);
				c.jne(m_ret, Near);
				c.jmp(next, Near);
				c.L(first);
				c.mov(c.qword[c.rdx + node.slot], c.rax);
				c.mov(state, 3);
				c.L(next);
			}
			c.cmp(state, 3);
			c.jne(m_ret, Near);
			c.mov(state, 2);
			c.ret();
			return;
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32: EmitRead(node); return;
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32: {
			Xbyak::Label other;
			Fetch(args[0], m_ret);
			c.test(c.rax, c.rax);
			c.jz(other, Near);
			Fetch(args[1], m_ret);
			Store(node);
			c.L(other);
			Fetch(args[2], m_ret);
			Store(node);
			return;
		}
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32: {
			const bool less = node.inst->GetOpcode() == ValueOpcode::ULessThan32;
			Xbyak::Label bound_zero;
			Fetch(args[0], m_ret);
			c.test(c.eax, c.eax);
			c.jz(bound_zero, Near);
			Branch(args[1], m_ret);
			Load(c.rax, args[0]);
			Load(c.rcx, args[1]);
			// ULT tests other < bound, ULE bound <= other.
			c.cmp(c.ecx, c.eax);
			Set(less ? &Xbyak::CodeGenerator::setb : &Xbyak::CodeGenerator::setae);
			Store(node);
			c.L(bound_zero);
			c.mov(c.eax, uint32_t {!less});
			Store(node);
			return;
		}
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd: {
			const bool conjunction = node.inst->GetOpcode() == ValueOpcode::LogicalAnd;
			Xbyak::Label unknown_left, refine, decided;
			const auto decide = [&] {
				if (conjunction) {
					c.jz(decided, Near);
				} else {
					c.jnz(decided, Near);
				}
			};
			Branch(args[0], unknown_left);
			if (args[0].kind != Kind::Unknown) {
				Load(c.rax, args[0]);
				c.test(c.rax, c.rax);
				decide();
				Fetch(args[1], m_ret);
				c.test(c.rax, c.rax);
				Set(&Xbyak::CodeGenerator::setne);
				Store(node);
			}
			c.L(unknown_left);
			if (MayBeUnknown(args[0])) {
				// Without the left side only a deciding right side, or a conjunction's zero
				// refined right side, is known.
				if (args[1].kind != Kind::Unknown) {
					Fetch(args[1], refine);
					c.test(c.rax, c.rax);
					decide();
				}
				c.L(refine);
				Fetch(args[2], m_ret);
				c.test(c.rax, c.rax);
				c.jnz(m_ret, Near);
			}
			c.L(decided);
			c.mov(c.eax, uint32_t {!conjunction});
			Store(node);
			return;
		}
		default: EmitStrict(node); return;
	}
}

// Raw scalar and uniform DWORD buffer loads; out-of-bounds reads are zero.
void WalkerCompiler::EmitRead(const Node& node) {
	auto&      c    = m_code;
	const auto& inst = *node.inst;
	const auto op = inst.GetOpcode();
	const bool vector = op == ValueOpcode::LoadBufferU32;
	auto args = std::span(node.args);
	Xbyak::Label zero;
	if (vector) {
		Fetch(args[0], m_ret);
		c.test(c.rax, c.rax);
		c.jz(zero, Near);
		args = args.subspan(1);
		if (args.empty()) c.jmp(m_ret, Near);
	}
	if (!args.empty()) {
		for (const auto& arg: args) Branch(arg, m_ret);
		const auto immediate =
		    static_cast<int64_t>(static_cast<int32_t>(m_plan.memory_info[inst.Flags<MemoryFlags>().index].offset)) &
		    ~int64_t {3};
		// rax = the 48-bit base (high << 32 | low) with its low two bits clear.
		Load(c.rax, args[1]);
		c.shl(c.rax, 48);
		c.shr(c.rax, 16);
		Load(c.rcx, args[0]);
		c.and_(c.ecx, ~3u);
		c.or_(c.rax, c.rcx);
		Load(c.r9, args[2]);
		c.and_(c.r9d, ~3u);
		if (op == ValueOpcode::LoadAddressU32) {
			c.lea(c.rax, c.ptr[c.rax + c.r9 + static_cast<size_t>(immediate)]);
			c.mov(c.rcx, c.rax);
			c.shr(c.rcx, 48);
			c.jnz(m_ret, Near);
			LoadGuest(c, c.rax);
		} else {
			Load(c.r11, args[1]);
			c.shr(c.r11d, 16);
			if (vector) {
				// Only the uniform, unswizzled structured DWORD address is evaluated on the host.
				c.test(c.r11d, 1u << 15u);
				c.jnz(m_ret, Near);
				Load(c.rcx, args[4]);
				c.test(c.ecx, (1u << 23u) | 0xf0000000u);
				c.jnz(m_ret, Near);
				c.test(c.r11d, 0x3fffu);
				c.jz(zero, Near);
				c.test(c.ecx, 0x7fu << 12u);
				c.jz(zero, Near);
			}
			c.and_(c.r11d, 0x3fffu);
			Load(c.rcx, args[3]);
			if (vector) {
				c.test(c.ecx, c.ecx);
				c.jz(zero, Near);
			}
			Xbyak::Label sized;
			c.test(c.r11d, c.r11d);
			c.jz(sized, Near);
			c.imul(c.rcx, c.r11);
			c.L(sized);
			c.lea(c.r11, c.ptr[c.r9 + static_cast<size_t>(immediate) + 4u]);
			c.cmp(c.r11, c.rcx);
			c.ja(zero, Near);
			LoadGuest(c, c.rax + c.r9 + static_cast<size_t>(immediate));
		}
		++m_captures;
		// The slot the load fills (see WalkSrt): one flat SRT slot, or a load the shader also
		// issues itself; a load shared by several slots has no single slot.
		const auto roots = std::ranges::count_if(m_plan.srt_reads, [&](const SrtRead& read) {
			return read.value.Resolve().TryInstruction() == &inst;
		});
		const auto root = std::ranges::find_if(m_plan.srt_reads, [&](const SrtRead& read) {
			return read.value.Resolve().TryInstruction() == &inst;
		});
		const uint32_t slot = roots == 1   ? root->flat_offset
		                      : roots == 0 ? RawReadSlot(inst.Flags<MemoryFlags>().index)
		                                   : NoSrtSlot;
		c.mov(c.qword[c.r8], c.r10);
		c.mov(c.dword[c.r8 + 8], c.eax);
		c.mov(c.dword[c.r8 + 12], slot);
		c.add(c.r8, 16);
		Store(node);
	}
	c.L(zero);
	c.xor_(c.eax, c.eax);
	Store(node);
}

void WalkerCompiler::EmitStrict(const Node& node) {
	auto&       c    = m_code;
	const auto& inst = *node.inst;
	for (const auto& arg: node.args) Branch(arg, m_ret);
	const std::array regs {c.rax, c.rcx, c.r9, c.r11};
	for (size_t i = 0; i < node.args.size() && i < regs.size(); ++i) Load(regs[i], node.args[i]);
	// Field ops require offset + width <= 32. Leaves (1 << width) - 1 in r11, the offset in
	// cl and the former rcx (the offset or the inserted value) in r10.
	const auto field = [&](const Xbyak::Reg32& offset, const Xbyak::Reg32& width) {
		c.lea(c.r10, c.ptr[offset.cvt64() + width.cvt64()]);
		c.cmp(c.r10, 32);
		c.ja(m_ret, Near);
		c.mov(c.r10, c.rcx);
		c.mov(c.ecx, width);
		c.mov(c.r11d, 1);
		c.shl(c.r11, c.cl);
		c.dec(c.r11);
		c.mov(c.ecx, offset.getIdx() == Xbyak::Operand::RCX ? c.r10d : offset);
	};
	switch (inst.GetOpcode()) {
		case ValueOpcode::CompositeConstructU64:
			c.shl(c.rcx, 32);
			c.or_(c.rax, c.rcx);
			break;
		case ValueOpcode::CompositeExtractU64:
			if (inst.Arg(1).Resolve().U32() != 0u) c.shr(c.rax, 32);
			c.mov(c.eax, c.eax);
			break;
		case ValueOpcode::IAdd32: c.add(c.eax, c.ecx); break;
		case ValueOpcode::IAdd64: c.add(c.rax, c.rcx); break;
		case ValueOpcode::ISub32: c.sub(c.eax, c.ecx); break;
		case ValueOpcode::ISub64: c.sub(c.rax, c.rcx); break;
		case ValueOpcode::IMul32: c.imul(c.eax, c.ecx); break;
		case ValueOpcode::UMulHi:
			c.imul(c.rax, c.rcx);
			c.shr(c.rax, 32);
			break;
		case ValueOpcode::UMin32:
			c.cmp(c.eax, c.ecx);
			c.cmova(c.eax, c.ecx);
			break;
		case ValueOpcode::UMax32:
			c.cmp(c.eax, c.ecx);
			c.cmovb(c.eax, c.ecx);
			break;
		case ValueOpcode::ConvertF32U32:
			c.cvtsi2ss(c.xmm0, c.rax);
			c.movd(c.eax, c.xmm0);
			break;
		case ValueOpcode::ConvertU32F32: {
			// The conversion saturates and maps NaN to zero.
			Xbyak::Label done;
			c.movd(c.xmm0, c.eax);
			c.xorps(c.xmm1, c.xmm1);
			c.xor_(c.eax, c.eax);
			c.ucomiss(c.xmm0, c.xmm1);
			c.jbe(done, Near);
			c.mov(c.ecx, 0x4f800000u);
			c.movd(c.xmm1, c.ecx);
			c.mov(c.eax, UINT32_MAX);
			c.ucomiss(c.xmm0, c.xmm1);
			c.jae(done, Near);
			c.cvttss2si(c.rax, c.xmm0);
			c.L(done);
			break;
		}
		case ValueOpcode::ConvertS32F32: {
			// The conversion saturates and maps NaN to zero. cvttss2si's out-of-range result,
			// INT32_MIN, is already right for negative overflow.
			Xbyak::Label done;
			c.movd(c.xmm0, c.eax);
			c.xor_(c.eax, c.eax);
			c.ucomiss(c.xmm0, c.xmm0);
			c.jp(done, Near);
			c.mov(c.ecx, 0x4f000000u);
			c.movd(c.xmm1, c.ecx);
			c.mov(c.eax, INT32_MAX);
			c.ucomiss(c.xmm0, c.xmm1);
			c.jae(done, Near);
			c.cvttss2si(c.eax, c.xmm0);
			c.L(done);
			break;
		}
		case ValueOpcode::FPMul32:
			c.movd(c.xmm0, c.eax);
			c.movd(c.xmm1, c.ecx);
			c.mulss(c.xmm0, c.xmm1);
			c.movd(c.eax, c.xmm0);
			break;
		case ValueOpcode::FPTrunc32:
			c.movd(c.xmm0, c.eax);
			c.roundss(c.xmm0, c.xmm0, 3);
			c.movd(c.eax, c.xmm0);
			break;
		case ValueOpcode::FPRecipIFlag32:
			// Normal positive powers of two have exact normal reciprocals in every FP mode.
			c.test(c.eax, 0x807fffffu);
			c.jnz(m_ret, Near);
			c.shr(c.eax, 23);
			c.jz(m_ret, Near);
			c.cmp(c.eax, 254);
			c.jae(m_ret, Near);
			c.neg(c.eax);
			c.add(c.eax, 254);
			c.shl(c.eax, 23);
			break;
		case ValueOpcode::FPIsNan32:
			c.and_(c.eax, 0x7fffffffu);
			c.cmp(c.eax, 0x7f800000u);
			Set(&Xbyak::CodeGenerator::seta);
			break;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (inst.Flags<FPCompareFlags>().flush_input_denorms) {
				for (const auto& reg: {c.eax, c.ecx}) {
					Xbyak::Label normal;
					c.mov(c.r11d, reg);
					c.and_(c.r11d, 0x7fffffffu);
					c.cmp(c.r11d, 0x00800000u);
					c.jae(normal, Near);
					c.and_(reg, 0x80000000u);
					c.L(normal);
				}
			}
			c.movd(c.xmm0, c.eax);
			c.movd(c.xmm1, c.ecx);
			// Unordered operands set CF, so AE is false for NaN.
			if (inst.GetOpcode() == ValueOpcode::FPOrdLessThanEqual32) {
				c.ucomiss(c.xmm1, c.xmm0);
			} else {
				c.ucomiss(c.xmm0, c.xmm1);
			}
			Set(&Xbyak::CodeGenerator::setae);
			break;
		case ValueOpcode::BitwiseAnd32: c.and_(c.eax, c.ecx); break;
		case ValueOpcode::BitwiseAnd64: c.and_(c.rax, c.rcx); break;
		case ValueOpcode::BitwiseOr32: c.or_(c.eax, c.ecx); break;
		case ValueOpcode::BitwiseXor32: c.xor_(c.eax, c.ecx); break;
		case ValueOpcode::BitwiseNot32: c.not_(c.eax); break;
		case ValueOpcode::ShiftLeftLogical32: c.shl(c.eax, c.cl); break;
		case ValueOpcode::ShiftLeftLogical64: c.shl(c.rax, c.cl); break;
		case ValueOpcode::ShiftRightLogical32: c.shr(c.eax, c.cl); break;
		case ValueOpcode::ShiftRightLogical64: c.shr(c.rax, c.cl); break;
		case ValueOpcode::ShiftRightArithmetic32: c.sar(c.eax, c.cl); break;
		case ValueOpcode::ShiftRightArithmetic64: c.sar(c.rax, c.cl); break;
		case ValueOpcode::BitFieldUExtract:
			field(c.ecx, c.r9d);
			c.shr(c.rax, c.cl);
			c.and_(c.rax, c.r11);
			break;
		case ValueOpcode::BitFieldSExtract: {
			Xbyak::Label done;
			field(c.ecx, c.r9d);
			c.shr(c.rax, c.cl);
			c.and_(c.rax, c.r11);
			// Width zero tests bit 63, which is clear.
			c.lea(c.ecx, c.ptr[c.r9 - 1]);
			c.bt(c.rax, c.rcx);
			c.jnc(done, Near);
			c.not_(c.r11);
			c.or_(c.rax, c.r11);
			c.mov(c.eax, c.eax);
			c.L(done);
			break;
		}
		case ValueOpcode::BitFieldInsert:
			field(c.r9d, c.r11d);
			c.shl(c.r11, c.cl);
			c.shl(c.r10, c.cl);
			c.and_(c.r10, c.r11);
			c.not_(c.r11);
			c.and_(c.rax, c.r11);
			c.or_(c.rax, c.r10);
			c.mov(c.eax, c.eax);
			break;
		case ValueOpcode::IEqual32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::sete);
			break;
		case ValueOpcode::INotEqual32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::setne);
			break;
		case ValueOpcode::IEqual64:
			c.cmp(c.rax, c.rcx);
			Set(&Xbyak::CodeGenerator::sete);
			break;
		case ValueOpcode::INotEqual64:
			c.cmp(c.rax, c.rcx);
			Set(&Xbyak::CodeGenerator::setne);
			break;
		case ValueOpcode::UGreaterThan32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::seta);
			break;
		case ValueOpcode::SGreaterThanEqual32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::setge);
			break;
		case ValueOpcode::LogicalXor:
			c.test(c.rax, c.rax);
			c.setne(c.al);
			c.test(c.rcx, c.rcx);
			c.setne(c.cl);
			c.xor_(c.al, c.cl);
			c.movzx(c.eax, c.al);
			break;
		case ValueOpcode::LogicalNot:
			c.test(c.rax, c.rax);
			Set(&Xbyak::CodeGenerator::sete);
			break;
		default: EXIT("unexpected SRT walker opcode\n");
	}
	Store(node);
}

void WalkerCompiler::Compile(std::span<const Value> values) {
	auto&       c         = m_code;
	const auto& sources   = m_plan.descriptor_sources;
	const auto& blocks    = m_plan.control_flow;
	const bool  resources = values.empty();
	std::vector<std::vector<uint32_t>> gates(sources.size());
	if (resources) Consume(gates);
	for (const auto value: values) AddOutput(value, UINT32_MAX);
	// Analyze every root first, so thunk bodies only reuse finished operands.
	std::vector<Operand>              conditions, reads, outputs;
	std::vector<std::vector<Operand>> descriptors(sources.size());
	if (resources) {
		for (const auto& block: blocks) conditions.push_back(Evaluate(block.condition, 0).operand);
		for (const auto& read: m_plan.srt_reads) reads.push_back(Evaluate(read.value, 0).operand);
		for (uint32_t source = 0; source < sources.size(); ++source) {
			if (gates[source].empty()) continue;
			for (uint32_t word = 0; word < sources[source].dword_count; ++word)
				descriptors[source].push_back(Evaluate(sources[source].dwords[word], 0).operand);
		}
	}
	for (const auto value: m_outputs) outputs.push_back(Evaluate(value, 0).operand);
	Layout();

	Xbyak::Label pass, fail, reset;
	c.mov(c.qword[c.rdx + m_shader_base], c.rcx);
	Zero(m_reach, m_states_end - m_reach);
	c.lea(c.r8, c.ptr[c.rdx + m_plan.srt_captures]);
	if (!resources) {
		for (uint32_t block = 0; block < blocks.size(); ++block) c.mov(c.byte[c.rdx + m_reach + block], 1);
	} else if (!blocks.empty()) {
		// Reachability grows monotonically. A newly reached predecessor of an evaluated
		// control Phi restarts the walk with fresh values; a retreating edge resumes at its target.
		std::vector<std::vector<uint32_t>> rechecks(blocks.size());
		for (uint32_t index = 0; index < m_nodes.size(); ++index) {
			const auto& node = m_nodes[index];
			if (node.inst->GetOpcode() != ValueOpcode::Phi || node.result != Operand {Kind::Node, index}) continue;
			for (const auto block: m_plan.control_phi_predecessors.at(node.inst)) rechecks[block].push_back(node.state);
		}
		std::vector<Xbyak::Label> starts(blocks.size());
		c.mov(c.byte[c.rdx + m_reach], 1);
		c.L(pass);
		for (uint32_t block = 0; block < blocks.size(); ++block) {
			Xbyak::Label next;
			const auto mark = [&](uint32_t target) {
				Xbyak::Label reached;
				c.cmp(c.byte[c.rdx + m_reach + target], 0);
				c.jne(reached, Near);
				c.mov(c.byte[c.rdx + m_reach + target], 1);
				for (const auto state: rechecks[target]) {
					c.cmp(c.byte[c.rdx + state], 0);
					c.jne(reset, Near);
				}
				if (target <= block) c.jmp(starts[target], Near);
				c.L(reached);
			};
			const auto& successors = blocks[block].successors;
			const auto  condition  = conditions[block];
			c.L(starts[block]);
			c.cmp(c.byte[c.rdx + m_reach + block], 0);
			c.je(next, Near);
			if (condition.kind == Kind::Constant) {
				mark(successors[condition.value != 0u ? 0 : 1]);
			} else {
				if (condition.kind != Kind::Unknown) {
					Xbyak::Label both, other;
					Fetch(condition, both);
					c.test(c.rax, c.rax);
					c.jz(other, Near);
					mark(successors[0]);
					c.jmp(next, Near);
					c.L(other);
					mark(successors[1]);
					c.jmp(next, Near);
					c.L(both);
				}
				for (const auto successor: successors) mark(successor);
			}
			c.L(next);
		}
	}
	if (resources) {
		const auto flat = [&](uint32_t slot) {
			const auto offset = slot < reads.size() ? m_plan.srt_reads[slot].flat_offset : UINT32_MAX;
			if (offset >= reads.size()) {
				c.jmp(fail, Near);
				return;
			}
			Fetch(reads[slot], fail);
			c.mov(c.dword[c.rsi + offset * sizeof(uint32_t)], c.eax);
		};
		if (blocks.empty()) {
			for (uint32_t slot = 0; slot < reads.size(); ++slot) flat(slot);
		}
		for (uint32_t block = 0; block < blocks.size(); ++block) {
			if (blocks[block].srt_reads.empty()) continue;
			Xbyak::Label skip;
			c.cmp(c.byte[c.rdx + m_reach + block], 0);
			c.je(skip, Near);
			for (const auto slot: blocks[block].srt_reads) flat(slot);
			c.L(skip);
		}
		// A source or image no block lists is always active.
		for (uint32_t index = 0; index < sources.size() + m_plan.info.images.size(); ++index) {
			const bool     image = index >= sources.size();
			const uint32_t id    = image ? index - static_cast<uint32_t>(sources.size()) : index;
			bool listed = false;
			for (uint32_t block = 0; block < blocks.size(); ++block) {
				const auto& list = image ? blocks[block].images : blocks[block].sources;
				if (std::ranges::find(list, id) == list.end()) continue;
				if (listed) {
					c.or_(c.al, c.byte[c.rdx + m_reach + block]);
				} else {
					c.mov(c.al, c.byte[c.rdx + m_reach + block]);
				}
				listed = true;
			}
			const auto active = c.byte[c.rdx + ActiveOffset(m_plan) + index];
			if (listed) {
				c.mov(active, c.al);
			} else {
				c.mov(active, 1);
			}
		}
		for (uint32_t source = 0; source < sources.size(); ++source) {
			if (gates[source].empty()) continue;
			Xbyak::Label skip;
			const auto known = c.dword[c.rdx + source * DescriptorStride + 8u * sizeof(uint32_t)];
			c.mov(known, 0);
			Gate(gates[source], skip);
			for (uint32_t word = 0; word < descriptors[source].size(); ++word) {
				Fetch(descriptors[source][word], skip);
				c.mov(c.dword[c.rdx + source * DescriptorStride + word * sizeof(uint32_t)], c.eax);
			}
			c.mov(known, 1);
			c.L(skip);
		}
	}
	for (uint32_t index = 0; index < outputs.size(); ++index) {
		Xbyak::Label skip;
		const auto output = c.qword[c.rdx + OutputOffset(m_plan) + index * 8u];
		c.mov(output, -1);
		Gate(m_output_gates[index], skip);
		Fetch(outputs[index], skip);
		c.mov(c.eax, c.eax);
		c.mov(output, c.rax);
		c.L(skip);
	}
	c.mov(c.rax, c.r8);
	c.L(m_ret);
	c.ret();
	c.L(fail);
	c.xor_(c.eax, c.eax);
	c.ret();
	if (resources && !blocks.empty()) {
		c.L(reset);
		Zero(m_states, m_states_end - m_states);
		c.lea(c.r8, c.ptr[c.rdx + m_plan.srt_captures]);
		c.jmp(pass, Near);
	}
	for (uint32_t index = 0; index < m_nodes.size(); ++index) {
		if (Thunk(index)) EmitNode(index);
	}
	m_plan.srt_scratch.assign(m_plan.srt_captures / 8u + m_captures * 2u, 0u);
	m_plan.srt_user_data = m_user_data;
	c.readyRE();
	EXIT_IF(Xbyak::GetError() != 0);
}

} // namespace

void CompileSrtWalker(ResourcePlan& plan, std::span<const Value> values) {
	plan.srt_code = std::make_unique<Xbyak::CodeGenerator>(4096, Xbyak::AutoGrow);
	WalkerCompiler(plan, *plan.srt_code).Compile(values);
}

bool WalkSrt(const ResourcePlan& plan, const SrtRuntime& runtime, std::vector<uint32_t>& flat,
             std::vector<std::pair<uint64_t, uint64_t>>* reads,
             std::vector<uint32_t>* read_slots) {
	if (plan.srt_code == nullptr || runtime.user_data.size() < plan.srt_user_data) return false;
	flat.assign(plan.srt_reads.size(), 0u);
	using Walker = const uint64_t* KYTY_SYSV_ABI (*)(const uint32_t*, uint32_t*, uint64_t*, uint64_t);
	t_running_walker = plan.srt_code.get();
	t_walker_ahead   = runtime.record_load != nullptr;
	t_walker_faulted = false;
	const auto* end = plan.srt_code->getCode<Walker>()(runtime.user_data.data(), flat.data(),
	                                                     plan.srt_scratch.data(), runtime.shader_base);
	t_running_walker = nullptr;
	t_walker_ahead   = false;
	if (end == nullptr || t_walker_faulted) return false;
	const auto* first = plan.srt_scratch.data() + plan.srt_captures / 8u;
	if (reads != nullptr) {
		for (auto* read = first; read != end; read += 2) reads->emplace_back(*read, sizeof(uint32_t));
	}
	if (read_slots != nullptr) {
		for (auto* read = first; read != end; read += 2)
			read_slots->push_back(static_cast<uint32_t>(read[1] >> 32u));
	}
	if (runtime.record_load != nullptr) {
		for (auto* read = first; read != end; read += 2) {
			// Nothing is mapped below guest memory: those bytes read zero on every thread.
			if (read[0] >= GuestMemoryStart)
				runtime.record_load(runtime.userdata, read[0], static_cast<uint32_t>(read[1]));
		}
	}
	return true;
}

bool WalkedValue(const ResourcePlan& plan, Value value, uint32_t& result) {
	value = value.Resolve();
	uint64_t bits = 0;
	if (value.IsImmediate()) {
		if (!ImmediateBits(value, bits)) return false;
		result = static_cast<uint32_t>(bits);
		return true;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return false;
	EXIT_IF(inst->WalkOutput() == UINT32_MAX);
	std::memcpy(&bits, Scratch(plan) + OutputOffset(plan) + inst->WalkOutput() * 8u, sizeof(bits));
	result = static_cast<uint32_t>(bits);
	return bits <= UINT32_MAX;
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
	return false;
}

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
	const auto flags = inst.Flags<SrtReadFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const auto& mem    = m_program.memory_info[flags.index];
	const bool  vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	const auto  guard  = vector ? inst.Arg(4).Resolve() : Value {};
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
	const bool const_buffer = inst.GetOpcode() == ValueOpcode::ReadConstBuffer;
	const auto immediate    = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
	uint64_t   records      = 0;
	uint64_t   word3        = 0;
	if (const_buffer || vector) {
		if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
			return false;
		}
	}
	if (vector) {
		// Only the uniform, unswizzled structured DWORD address is evaluated on the host, through
		// the strict (specialization) reader.
		if (immediate < 0) {
			return false;
		}
		const auto base = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
		const auto byte_offset =
		    (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
		const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u)
			return false;
		if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
			result = 0u;
			return true;
		}
		const auto size = static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			result = 0;
			return true;
		}
		const auto address = (base & ~uint64_t {3}) + byte_offset;
		uint32_t   word    = 0;
		if (m_runtime.read_specialization_memory == nullptr) {
			return false;
		}
		if (!m_runtime.read_specialization_memory(m_runtime.userdata, address, {&word, 1})) {
			if (base == 0 && m_program.control_flow.empty()) {
				Common::HackNotice::Report("failed SRT read through a null base read as zero",
				                           "SrtWalker::EvaluateRawRead (vector)", m_program.shader_hash,
				                           0, "a plan without control flow cannot tell a guarded read");
				result = 0;
				return true;
			}
			m_read_failure         = "guest memory unreadable";
			m_read_failure_address = address;
			m_read_failure_offset  = 0;
			m_read_failure_size    = 0;
			return false;
		}
		result = word;
		return true;
	}
	uint64_t base    = 0;
	uint64_t address = 0;
	switch (ComputeRawAddress(const_buffer, immediate, low, high, offset, records, base, address)) {
		case RawAddress::Fail: return false;
		case RawAddress::Zero: result = 0; return true;
		case RawAddress::Read: break;
	}
	m_reading_inst = &inst;
	const bool ok  = ReadRawWord(address, base, result);
	m_reading_inst = nullptr;
	return ok;
}

bool SrtWalker::ReadRawWord(uint64_t address, uint64_t base, uint64_t& result) {
	// The walk evaluates both sides of a branch it cannot decide (the exec mask, a count the GPU
	// writes), so it reads through table pointers that only the side the shader never takes
	// leaves null: the shader guards those loads itself (it tests the pointer or has no lane
	// active) and, taken, they would fault on the console. Guest memory starts at 0x40000, so a
	// host-memory read through a table below it reads zero, whatever its offset: the values of a
	// side that does not run cannot reach a descriptor the draw uses.
	if (base < GUEST_MEMORY_MIN &&
	    (m_runtime.read_memory == nullptr || SrtReadCapture::ReadsHostMemory(m_runtime))) {
		result = 0;
		return true;
	}
	if (m_fill_active && (m_fill_any || m_reading_inst == m_fill_inst) &&
	    m_runtime.gpu_written(address, sizeof(uint32_t))) {
		// Only the shader uses this slot: the GPU copies the bytes into the flat buffer before
		// the command (SrtGpuFill), where reading them here would wait for the GPU.
		m_runtime.gpu_fills->push_back({m_fill_offset, address});
		result = 0;
		return true;
	}
	uint32_t word = 0;
	if (m_runtime.map_clean_page != nullptr) {
		// The reader's own first step (a GPU-clean page is read from its backing), without the
		// callback chain and its page search for every dword. Raw reads are dword aligned.
		const auto page = address >> m_runtime.page_shift;
		if (page != m_mapped_page) {
			m_mapped_page  = page;
			m_mapped_bytes = m_runtime.map_clean_page(m_runtime.page_userdata,
			                                          page << m_runtime.page_shift);
		}
		if (m_mapped_bytes != nullptr && (address & 3u) == 0u) {
			std::memcpy(&word,
			            m_mapped_bytes + (address & ((uint64_t {1} << m_runtime.page_shift) - 1u)),
			            sizeof(word));
			if (m_runtime.capture_ranges != nullptr) {
				m_runtime.capture_ranges->emplace_back(address, sizeof(word));
			}
			result = word;
			return true;
		}
	}
	if (m_runtime.read_memory != nullptr) {
		// Names the shader for the notices of the memory callbacks.
		const Common::HackNotice::ScopedShader shader(m_program.shader_hash);
		if (!m_runtime.read_memory(m_runtime.userdata, address, {&word, 1})) {
			// A plan without control flow cannot tell a guarded read from an active one: a
			// failed read through a null base there reads zero (CS 0x0b4b91abfed42248 tests
			// s[4:5] against zero before its s_load; failing skipped its dispatch every frame).
			// With control flow, the reachability walk skips guarded reads and a null read in a
			// reachable block fails (ResourceTrackingTests, guarded scalar descriptor reads).
			if (base == 0 && m_program.control_flow.empty()) {
				Common::HackNotice::Report("failed SRT read through a null base read as zero",
				                           "SrtWalker::ReadRawWord", m_program.shader_hash, 0,
				                           "a plan without control flow cannot tell a guarded read");
				result = 0;
				return true;
			}
			m_read_failure         = "guest memory unreadable";
			m_read_failure_address = address;
			m_read_failure_offset  = 0;
			m_read_failure_size    = 0;
			return false;
		}
	} else {
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	result = word;
	return true;
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
				// The conversion saturates and maps NaN to zero.
				const auto value = Float32(a);
				result           = std::isnan(value) || value <= 0.0f ? 0u
				                   : value >= 4294967296.0f ? UINT32_MAX
				                                            : static_cast<uint32_t>(value);
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
		case ValueOpcode::IEqual64:
		case ValueOpcode::INotEqual64:
			if (!binary()) return false;
			result = inst.GetOpcode() == ValueOpcode::IEqual64 ? a == b : a != b;
			return true;
		case ValueOpcode::ULessThan32:
			// No U32 is below zero, including an unknown loop-carried value.
			if (!Arg(inst, 1, b)) return false;
			if (static_cast<uint32_t>(b) == 0u) {
				result = 0u;
				return true;
			}
			if (Arg(inst, 0, a)) {
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
		case ValueOpcode::UndefU32: return false;
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
	result.dword_count = plan.descriptor_sources[source].dword_count;
	std::copy_n(words.begin(), result.dword_count, result.dwords.begin());
	return true;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	if (!m_program.srt_plan_complete) return false;
	if (m_program.srt_code && m_runtime.read_memory == nullptr) {
		flat.resize(m_program.srt_reads.size());
		m_program.active_sources.clear();
		m_context.values.resize(m_program.evaluation_value_count);
		using Walker = bool KYTY_SYSV_ABI (*)(const uint32_t*, uint32_t*,
		    ResourcePlan::EvaluationContext::Entry*, uint64_t, size_t);
		if (m_program.srt_code->getCode<Walker>()(m_runtime.user_data.data(), flat.data(),
		                                          m_context.values.data(), m_context.generation,
		                                          m_runtime.user_data.size())) {
			return true;
		}
	}
	const auto refresh = [&](uint32_t slot) {
		if (slot >= m_program.srt_reads.size()) return false;
		const auto& read = m_program.srt_reads[slot];
		const auto* inst = read.value.ResolveInstruction();
		const bool clean = inst != nullptr && inst->Flags<SrtReadFlags>().clean != 0u;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
			return false;
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		if (read.flat_offset >= flat.size()) return false;
		if (!clean && slot < m_program.gpu_fill_slots.size() && m_program.gpu_fill_slots[slot] != 0u &&
		    m_runtime.gpu_written != nullptr && m_runtime.gpu_fills != nullptr) {
			// ReadRawWord leaves GPU-written bytes to the GPU (SrtGpuFill) instead of waiting for it.
			m_fill_active = true;
			m_fill_offset = read.flat_offset;
			m_fill_any    = m_program.gpu_fill_slots[slot] == 1u;
			m_fill_inst   = inst;
			const auto fills_before = m_runtime.gpu_fills->size();
			bool       ok           = false;
			if (!m_fill_any && m_fill_inst != nullptr &&
			    (m_fill_inst->GetOpcode() == ValueOpcode::LoadAddressU32 ||
			     m_fill_inst->GetOpcode() == ValueOpcode::ReadConstBuffer)) {
				// Its own read: the reads that make its address are done on the host.
				uint64_t wide = 0;
				ok            = EvaluateRawRead(*m_fill_inst, wide);
				flat[read.flat_offset] = static_cast<uint32_t>(wide);
			} else {
				ok = Evaluate(read.value, flat[read.flat_offset]);
			}
			m_fill_active = false;
			m_fill_any    = true;
			if (m_runtime.gpu_fills->size() != fills_before && m_fill_inst != nullptr) {
				// The read left a placeholder (0) for the GPU to fill: forget it, so a host use of the
				// same value in this walk (a branch condition) reads it again instead of taking 0.
				const auto index = m_fill_inst->EvaluationIndex(m_program.evaluation_value_count);
				if (index < m_context.values.size()) {
					m_context.values[index].generation = 0;
				}
			}
			m_fill_inst = nullptr;
			return ok;
		}
		return evaluator.Evaluate(read.value, flat[read.flat_offset]);
	};
	auto& active = m_program.active_sources;
	if (m_program.control_flow.empty()) {
		active.clear();
		flat.resize(m_program.srt_reads.size());
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); ++slot) {
			if (!refresh(slot)) return false;
		}
		return true;
	}
	auto& visited = m_program.visited_blocks;
	auto& reachable = m_program.reachable_blocks;
	auto& pending = m_program.pending_blocks;
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
			// A condition whose bytes the GPU wrote is "not known" instead of a wait for the GPU:
			// both successors are walked.
			bool known = false;
			if (!block.condition.IsEmpty() &&
			    (m_runtime.read_specialization_memory != nullptr || !m_program.capture_specialization_reads)) {
				t_evaluating_condition = true;
				known                  = predicate.Evaluate(block.condition, condition);
				t_evaluating_condition = false;
			}
			if (known) {
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

} // namespace Libs::Graphics::ShaderRecompiler::IR
