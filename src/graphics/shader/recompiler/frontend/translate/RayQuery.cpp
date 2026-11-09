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
				// An origin component the same for every lane is in an SGPR.
				if (i < 3 && inst.image_nsa_addr[i] == Decoder::NoRegister) {
					Decoder::Operand origin;
					origin.kind = Decoder::OperandKind::Sgpr;
					origin.reg  = inst.image_nsa_addr[9 + i];
					ray[i]      = ReadU32(origin);
					continue;
				}
				// A site without a mask VGPR has a wave-wide mask (src2) or a constant one.
				if (i == 7 && inst.image_nsa_addr[i] == Decoder::NoRegister) {
					ray[i] = inst.src2.kind == Decoder::OperandKind::Sgpr ? ReadU32(inst.src2)
					                                                      : IR::U32(IR::Value(inst.image_nsa_addr[8]));
					continue;
				}
				ray[i] = ReadU32(VectorOperand(inst.image_nsa_addr[i]));
			}
			const auto header    = ReadU32Pair(inst.src0);
			const auto alpha_off = ReadU32(inst.src1);
			// The lanes that cull facings: every lane, none when an SGPR is 0, or those of a mask.
			auto cull = IR::U1(IR::Value(true));
			if ((inst.dmask & 4u) != 0) {
				cull = ir.INotEqual(ReadU32(inst.src3), IR::U32(IR::Value(0u)));
			} else if ((inst.dmask & 8u) != 0) {
				cull = ReadMask(inst.src3);
			}
			// A site without guest leaf code (dmask 16): the SGPR that flips the culled facing and
			// the one without which the traversal is the guest's (0xffff: none).
			auto flip = IR::U1(IR::Value(false)), allow = IR::U1(IR::Value(true));
			if ((inst.dmask & 16u) != 0) {
				const auto sgpr = [&](uint32_t reg) {
					Decoder::Operand operand;
					operand.kind = Decoder::OperandKind::Sgpr;
					operand.reg  = reg;
					return ir.INotEqual(ReadU32(operand), IR::U32(IR::Value(0u)));
				};
				if (const auto reg = inst.secondary_offset & 0xffffu; reg != 0xffffu) {
					flip = sgpr(reg);
				}
				if (const auto reg = inst.secondary_offset >> 16u; reg != 0xffffu) {
					allow = sgpr(reg);
				}
			}
			const auto variant = (inst.dmask & 3u) | ((inst.dmask & 16u) != 0 ? IR::RayQueryInitNoLeaf : 0u) |
			                     ((inst.dmask & 32u) != 0 ? IR::RayQueryInitAllCandidates : 0u);
			const auto native  = IR::U1(ir.Emit(
			     IR::ValueOpcode::RayQueryInit,
			     {ray[0], ray[1], ray[2], ray[3], ray[4], ray[5], ray[6], ray[7], header[0], header[1], alpha_off,
			      ir.GetExec(), cull, flip, allow},
			     flags(variant)));
			// The same in every lane: the host has (or not) the TLAS of the header.
			ir.SetScc(any_lane(native));
			return;
		}
		case Decoder::Opcode::RT_SELECT: {
			// The wave's smallest (instance, leaf) key: uniform values (dmask 1: the smallest
			// instance; each lane takes its own leaf of it).
			const auto keys     = (inst.dmask & 1u) != 0 ? IR::RayQueryKeyLaneLeaf : 0u;
			const auto next     = ir.Emit(IR::ValueOpcode::RayQueryNext, {}, flags(keys));
			const auto instance = IR::U32(ir.CompositeExtract(next, 0));
			const auto has      = ir.INotEqual(IR::U32(ir.CompositeExtract(next, 2)), IR::U32(IR::Value(0u)));
			const auto match    = IR::U1(ir.Emit(IR::ValueOpcode::RayQueryMatch, {}, flags(keys)));
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
			const auto keys  = (inst.dmask & 1u) != 0 ? IR::RayQueryKeyLaneLeaf : 0u;
			WriteOperand(inst.dst,
			             IR::U32(ir.Emit(IR::ValueOpcode::RayQuerySelectedLeaf, {}, flags(keys))));
			write_mask(inst.src0, lanes);
			write_mask(ExecOperand(), lanes);
			write_mask(inst.dst2, lanes);
			mark(Mark::Enter, lanes);
			return;
		}
		case Decoder::Opcode::RT_RESOLVE:
			if (inst.dmask == 2u) {
				// Closest hit: the leaf code keeps the nearest hit, every lane keeps searching.
				mark(Mark::Resolve, IR::U1(IR::Value(true)));
				return;
			}
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
		case Decoder::Opcode::RT_ALPHA: {
			// The values the per-lane alpha test reads (see RayTracingAlphaTest): the level of
			// detail term, the cone spread, the threshold factor (1.0 without one), the V#s of the
			// material table and of the buffer table, which the guest leaf code loads from user
			// data, the threshold (the lane's reference when the site has one) and the format of
			// the guest data (Decoder::RayTracingAlphaFields::Pack()).
			// A V# the guest loads before the traversal is in its SGPRs (RayTracingAlphaDirect).
			const auto vsharp = [&](uint32_t base, uint32_t offset) {
				if (offset != Decoder::RayTracingAlphaDirect) {
					return LoadScalarAddressDwords(base, offset, 4, inst.pc);
				}
				std::array<IR::U32, 4> dwords;
				for (uint32_t i = 0; i < 4; i++) {
					Decoder::Operand operand;
					operand.kind = Decoder::OperandKind::Sgpr;
					operand.reg  = base + i;
					dwords[i]    = ReadU32(operand);
				}
				return dwords;
			};
			const auto material = vsharp(inst.src2.reg, inst.image_nsa_addr[1]);
			const auto buffers  = vsharp(inst.src3.reg, inst.image_nsa_addr[2]);
			// dmask 4: src1 is a reference the same for every lane (no factor); dmask 8: the
			// reference is threshold * factor for every material.
			const bool uniform  = (inst.dmask & 4u) != 0;
			auto       factor   = inst.src1.kind == Decoder::OperandKind::Sgpr && !uniform
			                          ? ReadU32(inst.src1)
			                          : IR::U32(IR::Value(0x3f800000u));
			auto       threshold = uniform ? ReadU32(inst.src1)
			                       : inst.image_nsa_addr[5] != Decoder::NoRegister
			                           ? ReadU32(VectorOperand(inst.image_nsa_addr[5]))
			                           : IR::U32(IR::Value(inst.image_nsa_addr[3]));
			if ((inst.dmask & 8u) != 0) {
				const auto as_float = [&](IR::U32 value) {
					return IR::F32(ir.Emit(IR::ValueOpcode::BitCastF32U32, {value}));
				};
				threshold = IR::U32(ir.Emit(IR::ValueOpcode::BitCastU32F32,
				                            {ir.Emit(IR::ValueOpcode::FPMul32, {as_float(threshold), as_float(factor)})}));
				factor    = IR::U32(IR::Value(0x3f800000u));
			}
			ir.Emit(IR::ValueOpcode::RayQueryAlpha,
			        {ReadU32(VectorOperand(inst.image_nsa_addr[0])), ReadU32(inst.src0), factor, material[0],
			         material[1], material[2], material[3], buffers[0], buffers[1], buffers[2], buffers[3], threshold,
			         IR::U32(IR::Value(inst.image_nsa_addr[4])), IR::U32(IR::Value(inst.image_nsa_addr[6]))},
			        flags(inst.dmask & 3u));
			return;
		}
		default: FailMissingTranslation(inst);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
