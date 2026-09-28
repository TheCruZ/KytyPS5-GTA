#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

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
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto kind = values.memory_info[index].kind;
	return (op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
	       (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer);
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

} // namespace

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
		case ValueOpcode::UMin32:
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
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

namespace {

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type)
	    : m_program(program), m_type(type) {}

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
		if (m_type == RuntimeValueType::Integer &&
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
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer) {
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
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(2), false)) {
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
		if (op == ValueOpcode::Phi) {
			if (m_type == RuntimeValueType::Integer && !ValidateArguments(*inst, false)) {
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
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(1), false)) {
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
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
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
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
};


} // namespace

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     std::span<const uint8_t> clean_flat_slots, SrtWalker* clean_evaluator,
                     Value active_mask)
    : m_program(program), m_runtime(runtime), m_clean_flat_slots(clean_flat_slots),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()),
      m_context(AcquireContext(program)) {}

SrtWalker::~SrtWalker() { --m_program.evaluation_depth; }

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateWide(value, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

ResourcePlan::EvaluationContext& SrtWalker::AcquireContext(const ResourcePlan& program) {
	if (program.evaluation_depth == program.evaluation_contexts.size()) {
		program.evaluation_contexts.emplace_back();
	}
	auto& context = program.evaluation_contexts[program.evaluation_depth++];
	context.generation += 2;
	return context;
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
	// The low generation bit marks an instruction that is still being evaluated.
	if (m_context.values[index].generation == (m_context.generation | 1u)) {
		return false;
	}
	m_context.values[index].generation = m_context.generation | 1u;
	uint64_t out = 0;
	const bool evaluated = EvaluateInst(*inst, out);
	// Recursive evaluation may grow the dense memo vector.
	auto& memo = m_context.values[index];
	if (!evaluated) {
		memo.generation = 0;
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
	const auto flags = inst.Flags<MemoryFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const auto& mem    = m_program.memory_info[flags.index];
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
	if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer) {
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
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			return false;
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
	if (m_runtime.read_memory != nullptr) {
		if (!m_runtime.read_memory(m_runtime.userdata, address, {&word, 1})) {
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
			const auto clean_runtime = CleanRuntime(m_runtime);
			SrtWalker  clean_active(m_program, clean_runtime, {}, nullptr, inst.Arg(1));
			SrtWalker  active(m_program, m_runtime, m_clean_flat_slots, &clean_active,
			                  inst.Arg(1));
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
		case ValueOpcode::ReadConst: {
			const auto slot = inst.Arg(1).Resolve();
			if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return false;
			}
			if (slot.U32() < m_clean_flat_slots.size() &&
			    m_clean_flat_slots[slot.U32()] != 0u && m_clean_evaluator != nullptr) {
				return m_clean_evaluator->EvaluateWide(m_program.srt_reads[slot.U32()].value,
				                                       result);
			}
			return EvaluateWide(m_program.srt_reads[slot.U32()].value, result);
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
			if (IsRawRead(m_program, inst)) {
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
		case ValueOpcode::UMin32:
			if (binary()) {
				result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
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
			if (!Arg(inst, 1, b) || (b != 0u && !left)) return false;
			result = b != 0u;
			return true;
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
	const auto refresh = [&](uint32_t slot) {
		if (slot >= m_program.srt_reads.size()) return false;
		const auto& read = m_program.srt_reads[slot];
		const bool clean = read.flat_offset < m_clean_flat_slots.size() &&
		                   m_clean_flat_slots[read.flat_offset] != 0u;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
			return false;
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		return read.flat_offset < flat.size() && evaluator.Evaluate(read.value, flat[read.flat_offset]);
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
	flat.assign(m_program.srt_reads.size(), 0u);
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow) {
		for (const auto source: block.sources) active.at(source) = 0u;
	}
	auto& visited = m_program.visited_blocks;
	auto& pending = m_program.pending_blocks;
	visited.assign(m_program.control_flow.size(), 0u);
	pending.clear();
	pending.push_back(0u);
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		if (visited.at(index)) continue;
		visited[index] = 1u;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) active[source] = 1u;
		for (const auto slot: block.srt_reads) {
			if (!refresh(slot)) return false;
		}
		uint32_t condition = 0;
		auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
		if (!block.condition.IsEmpty() && m_runtime.read_specialization_memory != nullptr &&
		    predicate.Evaluate(block.condition, condition)) {
			pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
		} else {
			pending.insert(pending.end(), block.successors.begin(), block.successors.end());
		}
	}
	return true;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
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
			case ValueOpcode::ReadConst: {
				if (!Require(inst, 2)) break;
				const auto slot = inst.Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
				    slot.U32() >= m_program.srt_reads.size()) {
					node.kind = CompiledSrtPlan::NodeKind::Invalid;
					break;
				}
				node.kind    = CompiledSrtPlan::NodeKind::ReadConst;
				node.imm     = slot.U32();
				node.args[0] = NodeOf(m_program.srt_reads[slot.U32()].value);
				break;
			}
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer: {
				if (!IsRawRead(m_program, inst)) {
					node.shape = Shape::NotRawRead;
					break;
				}
				node.shape = Shape::RawRead;
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
				const auto& memory = m_program.memory_info[inst.Flags<MemoryFlags>().index];
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
			case ValueOpcode::UMin32:
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
		plan->srt_reads.push_back({compiler.NodeOf(read.value), read.flat_offset});
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
		if (descriptor.indirect_image.has_value()) {
			add_root(descriptor.indirect_image->key_count);
			add_root(descriptor.indirect_image->selector_mask);
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
	program.compiled_srt = std::move(plan);
	return program.compiled_srt.get();
}

namespace {

CompiledSrtPlan::Context& AcquireCompiledContext(const CompiledSrtPlan& plan) {
	const auto depth = plan.depth++;
	if (depth >= plan.top_contexts.size() &&
	    depth - plan.top_contexts.size() == plan.nested_contexts.size()) {
		plan.nested_contexts.emplace_back();
	}
	auto& context = depth < plan.top_contexts.size()
	                    ? plan.top_contexts[depth]
	                    : plan.nested_contexts[depth - plan.top_contexts.size()];
	// Sized once: recursive evaluation keeps references into the memo.
	if (context.values.size() != plan.nodes.size()) {
		context.values.resize(plan.nodes.size());
	}
	context.generation += 2;
	return context;
}

} // namespace

CompiledSrtWalker::CompiledSrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                                     std::span<const uint8_t> clean_flat_slots,
                                     CompiledSrtWalker* clean_evaluator, uint32_t active_mask)
    : m_program(program), m_plan(*program.compiled_srt), m_runtime(runtime),
      m_clean_flat_slots(clean_flat_slots), m_clean_evaluator(clean_evaluator),
      m_active_mask(active_mask), m_context(AcquireCompiledContext(m_plan)) {}

CompiledSrtWalker::~CompiledSrtWalker() { --m_plan.depth; }

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
	if (node.kind == CompiledSrtPlan::NodeKind::ReadConst) {
		// SrtWalker memoizes the ReadConst itself too; its value is always the memoized value
		// of the SRT read in the same evaluator, so the extra entry is not observable.
		if (node.imm < m_clean_flat_slots.size() && m_clean_flat_slots[node.imm] != 0u &&
		    m_clean_evaluator != nullptr) {
			return m_clean_evaluator->EvaluateNode(node.args[0], result);
		}
		return EvaluateNode(node.args[0], result);
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
	// The low generation bit marks a node that is still being evaluated.
	if (memo.generation == (generation | 1u)) {
		return false;
	}
	memo.generation = generation | 1u;
	uint64_t out    = 0;
	bool evaluated = false;
	if (node.shape == CompiledSrtPlan::Shape::LeafAddressRead) {
		uint64_t low     = 0;
		uint64_t high    = 0;
		uint64_t address = 0;
		evaluated        = EvaluateLeaf(node.args[0], low) && EvaluateLeaf(node.args[1], high) &&
		            AddSignedAddress((((high << 32u) | static_cast<uint32_t>(low)) & AddressMask) &
		                                 ~uint64_t {3},
		                             static_cast<int64_t>(node.imm), address) &&
		            ReadWord(address, out);
	} else {
		evaluated = EvaluateInst(id, out);
	}
	if (!evaluated) {
		memo.generation = 0;
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
			return false;
		}
		address = (base & ~uint64_t {3}) + byte_offset;
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	return ReadWord(address, result);
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

bool CompiledSrtWalker::ReadWord(uint64_t address, uint64_t& result) {
	uint32_t word = 0;
	if (m_runtime.read_memory != nullptr) {
		if (!m_runtime.read_memory(m_runtime.userdata, address, {&word, 1})) {
			return false;
		}
	} else {
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	result = word;
	return true;
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
		case ValueOpcode::Phi:
			return node.args[0] != NoNode && EvaluateNode(node.args[0], result);
		case ValueOpcode::ReadFirstLane: {
			const auto        clean_runtime = CleanRuntime(m_runtime);
			CompiledSrtWalker clean_active(m_program, clean_runtime, {}, nullptr, node.args[1]);
			CompiledSrtWalker active(m_program, m_runtime, m_clean_flat_slots, &clean_active,
			                         node.args[1]);
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
		case ValueOpcode::UMin32:
			if (!binary()) return false;
			result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
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
			if (!EvaluateNode(node.args[1], b) || (b != 0u && !left)) return false;
			result = b != 0u;
			return true;
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
		const bool  clean = read.flat_offset < m_clean_flat_slots.size() &&
		                   m_clean_flat_slots[read.flat_offset] != 0u;
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
	auto& active = m_program.active_sources;
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
	flat.assign(m_plan.srt_reads.size(), 0u);
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow) {
		for (const auto source: block.sources) {
			active.at(source) = 0u;
		}
	}
	auto& visited = m_program.visited_blocks;
	auto& pending = m_program.pending_blocks;
	visited.assign(m_program.control_flow.size(), 0u);
	pending.clear();
	pending.push_back(0u);
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		if (visited.at(index)) {
			continue;
		}
		visited[index]    = 1u;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) {
			active[source] = 1u;
		}
		for (const auto slot: block.srt_reads) {
			if (!refresh(slot)) {
				return false;
			}
		}
		uint64_t   condition = 0;
		const auto node      = index < m_plan.conditions.size() ? m_plan.conditions[index] : NoNode;
		auto&      predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
		if (node != NoNode && m_runtime.read_specialization_memory != nullptr &&
		    predicate.EvaluateNode(node, condition)) {
			pending.push_back(block.successors[static_cast<uint32_t>(condition) != 0u ? 0u : 1u]);
		} else {
			pending.insert(pending.end(), block.successors.begin(), block.successors.end());
		}
	}
	return true;
}


} // namespace Libs::Graphics::ShaderRecompiler::IR
