#include "graphics/shader/recompiler/frontend/decode/RayTracingPatch.h"
#include "graphics/shader/recompiler/frontend/translate/Translator.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"

// The pseudo instructions of the native ray tracing patch (see RayTracingPatch.h).

namespace Libs::Graphics::ShaderRecompiler::Frontend {

namespace {

Decoder::Operand VectorOperand(uint32_t reg) {
	Decoder::Operand operand;
	operand.kind = Decoder::OperandKind::Vgpr;
	operand.reg  = reg;
	return operand;
}

Decoder::Operand ExecOperand() {
	Decoder::Operand operand;
	operand.kind = Decoder::OperandKind::ExecLo;
	return operand;
}

} // namespace

void Translator::EmitRayTracing(const Decoder::Instruction& inst) {
	using Flag = IR::RayQueryFlagKind;
	using Mark = IR::RayQueryMarkKind;
	const auto site = inst.offset;
	if (site >= IR::MaxRayQuerySites) {
		EXIT("ray tracing patch site %u out of range at pc 0x%08x", site, inst.pc);
	}
	const auto flags = [site](auto variant) {
		return IR::RayQueryFlags(site, static_cast<uint32_t>(variant));
	};
	const auto flag = [&](Flag kind) {
		return IR::U1(ir.Emit(IR::ValueOpcode::RayQueryFlag, {}, flags(kind)));
	};
	const auto mark = [&](Mark kind, IR::U1 value) {
		ir.Emit(IR::ValueOpcode::RayQueryMark, {value}, flags(kind));
	};
	const auto write_mask = [&](const Decoder::Operand& operand, IR::U1 value) {
		WriteMask(operand, value, BallotMask(value));
	};
	const auto any_lane = [&](IR::U1 lanes) {
		const auto mask = BallotMask(lanes);
		return ir.INotEqual(ir.BitwiseOr(mask[0], mask[1]), IR::U32(IR::Value(0u)));
	};

	switch (inst.opcode) {
		case Decoder::Opcode::RT_INIT: {
			std::array<IR::U32, 8> ray;
			for (uint32_t i = 0; i < 8; i++) {
				// A site without a mask register traces against every instance.
				ray[i] = i == 7 && inst.image_nsa_addr[i] == Decoder::NoRegister
				             ? IR::U32(IR::Value(0xffu))
				             : ReadU32(VectorOperand(inst.image_nsa_addr[i]));
			}
			const auto header    = ReadU32Pair(inst.src0);
			const auto alpha_off = ReadU32(inst.src1);
			const auto native    = IR::U1(ir.Emit(
                IR::ValueOpcode::RayQueryInit,
                {ray[0], ray[1], ray[2], ray[3], ray[4], ray[5], ray[6], ray[7], header[0], header[1],
                 alpha_off, ir.GetExec()},
                flags(inst.dmask)));
			// The same in every lane: the host has (or not) the TLAS of the header.
			ir.SetScc(any_lane(native));
			return;
		}
		case Decoder::Opcode::RT_SELECT: {
			// The wave's smallest (instance, leaf) key: uniform values.
			const auto next     = ir.Emit(IR::ValueOpcode::RayQueryNext, {}, flags(0u));
			const auto instance = IR::U32(ir.CompositeExtract(next, 0));
			const auto has      = ir.INotEqual(IR::U32(ir.CompositeExtract(next, 2)), IR::U32(IR::Value(0u)));
			const auto match    = IR::U1(ir.Emit(IR::ValueOpcode::RayQueryMatch, {}, flags(0u)));
			WriteOperand(inst.dst, instance);
			write_mask(inst.dst2, match);
			write_mask(ExecOperand(), match);
			ir.SetScc(has);
			return;
		}
		case Decoder::Opcode::RT_REUSE: ir.SetScc(flag(Flag::Reuse)); return;
		case Decoder::Opcode::RT_SETUP: {
			// The setup runs for every lane with a key of the instance, so that later keys of the
			// instance can reuse its registers.
			ir.SetScc(flag(Flag::NeedRestore));
			const auto lanes = flag(Flag::InstanceMatch);
			mark(Mark::SetupBegin, lanes);
			write_mask(ExecOperand(), lanes);
			return;
		}
		case Decoder::Opcode::RT_LEAF: {
			// The leaf runs for the lanes with its key that the instance setup kept.
			const auto lanes = ir.LogicalAnd(flag(Flag::Matched), ReadMask(inst.src0));
			WriteOperand(inst.dst,
			             IR::U32(ir.Emit(IR::ValueOpcode::RayQuerySelectedLeaf, {}, flags(0u))));
			write_mask(inst.src0, lanes);
			write_mask(ExecOperand(), lanes);
			write_mask(inst.dst2, lanes);
			mark(Mark::Enter, lanes);
			return;
		}
		case Decoder::Opcode::RT_RESOLVE:
			mark(inst.dmask != 0 ? Mark::CullResolve : Mark::Resolve, ReadMask(inst.src0));
			return;
		case Decoder::Opcode::RT_EXIT_CHECK: ir.SetScc(flag(Flag::NeedRestore)); return;
		case Decoder::Opcode::RT_COMMIT: {
			// Lanes whose result is an opaque hit the hardware found: the registers the guest
			// leaf code writes for an accepted hit.
			using Result      = IR::RayQueryResultKind;
			const auto result = [&](Result kind) {
				return IR::U32(ir.Emit(IR::ValueOpcode::RayQueryResult, {}, flags(kind)));
			};
			auto has = ir.INotEqual(result(Result::Has), IR::U32(IR::Value(0u)));
			if (inst.dmask != 0) {
				// Closest hit: only when nearer than the hit the registers hold (or the t max).
				const auto as_float = [&](IR::U32 value) {
					return IR::F32(ir.Emit(IR::ValueOpcode::BitCastF32U32, {value}));
				};
				const auto current = IR::U32(ir.GetVectorReg(
				    static_cast<IR::VectorReg>(inst.image_nsa_addr[0])));
				has = ir.LogicalAnd(has, IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThanEqual32,
				                                         {as_float(result(Result::T)),
				                                          as_float(current)})));
			}
			const std::array<std::pair<uint32_t, Result>, 7> outputs {{
			    {inst.image_nsa_addr[0], Result::T},
			    {inst.image_nsa_addr[1], Result::Word},
			    {inst.image_nsa_addr[2], Result::InstanceLo},
			    {inst.image_nsa_addr[3], Result::InstanceHi},
			    {inst.image_nsa_addr[4], Result::U},
			    {inst.image_nsa_addr[5], Result::V},
			    {inst.image_nsa_addr[6], Result::Node},
			}};
			for (const auto& [reg, kind]: outputs) {
				if (reg == Decoder::NoRegister) {
					continue;
				}
				const auto vreg = static_cast<IR::VectorReg>(reg);
				ir.SetVectorReg(vreg, ir.Select(has, result(kind), ir.GetVectorReg(vreg)));
			}
			return;
		}
		case Decoder::Opcode::RT_EXIT:
			write_mask(ExecOperand(), flag(Flag::InTraversal));
			mark(Mark::Exit, IR::U1(IR::Value(true)));
			return;
		default: FailMissingTranslation(inst);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
