#include "graphics/shader/recompiler/frontend/decode/RayTracingPatch.h"

#include "common/logging/log.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstddef>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Decoder {

namespace {

Operand Sgpr(uint32_t reg) {
	Operand operand;
	operand.kind  = OperandKind::Sgpr;
	operand.reg   = reg;
	operand.value = reg;
	return operand;
}

uint32_t EndPc(const Instruction& inst) {
	return inst.pc + inst.word_count * 4u;
}

const Instruction* FindInstruction(const Program& program, uint32_t pc) {
	const auto found = std::ranges::lower_bound(program.instructions, pc, {}, &Instruction::pc);
	return found != program.instructions.end() && found->pc == pc ? &*found : nullptr;
}

bool CheckSite(const Program& program, const RayTracingSite& site) {
	const auto expect = [&](uint32_t pc, Opcode opcode) {
		const auto* inst = FindInstruction(program, pc);
		return inst != nullptr && (opcode == Opcode::UNKNOWN || inst->opcode == opcode);
	};
	const auto* entry = FindInstruction(program, site.entry_pc);
	if (site.kind == RayTracingKind::ClosestOpaque) {
		return entry != nullptr && EndPc(*entry) == site.loop_pc && !IsDirectBranch(entry->opcode) &&
		       expect(site.loop_pc, Opcode::UNKNOWN) && expect(site.exit_pc, Opcode::UNKNOWN);
	}
	return entry != nullptr && EndPc(*entry) == site.loop_pc && !IsDirectBranch(entry->opcode) &&
	       expect(site.loop_pc, Opcode::S_AND_B64) && expect(site.setup_pc, Opcode::UNKNOWN) &&
	       expect(site.cull_exit_pc, Opcode::S_CMP_EQ_U32) &&
	       expect(site.blas_head_pc, Opcode::S_CMP_LG_U32) && expect(site.leaf_pc, Opcode::UNKNOWN) &&
	       expect(site.tail_pc, Opcode::S_CMP_LG_U64) && expect(site.exit_pc, Opcode::UNKNOWN) &&
	       (site.after_leaf_pc == 0 || expect(site.after_leaf_pc, Opcode::V_READLANE_B32));
}

// The rewritten program before its pcs are assigned: branch targets are guest pcs (resolved
// through the new position of the guest instruction) or labels of inserted instructions.
class Layout {
public:
	enum class Target { None, Guest, Label };

	struct Item {
		Instruction inst;
		Target      target = Target::None;
		uint32_t    value  = 0;
	};

	uint32_t NewLabel() { return m_next_label++; }
	// The label refers to the next item added.
	void Bind(uint32_t label) { m_bindings.emplace_back(label, m_items.size()); }

	Instruction& AddGuest(const Instruction& inst) {
		Item item {inst};
		// The guest code stays where it is in memory: PC-relative addresses (S_GETPC_B64 reads
		// of data after the program) keep their guest pc.
		if (item.inst.guest_pc == UINT32_MAX) {
			item.inst.guest_pc = inst.pc;
		}
		if (IsDirectBranch(inst.opcode)) {
			item.target = Target::Guest;
			item.value  = inst.branch_target;
		}
		m_guest_items.emplace(inst.pc, m_items.size());
		m_items.push_back(std::move(item));
		return m_items.back().inst;
	}
	// A guest instruction placed again elsewhere (its branches set by the caller).
	Instruction& AddCopy(const Instruction& inst) {
		m_items.push_back({inst});
		return m_items.back().inst;
	}
	void SetBranch(Target target, uint32_t value) {
		m_items.back().target = target;
		m_items.back().value  = value;
	}
	void AddBranch(Opcode opcode, Target target, uint32_t value) {
		Instruction inst;
		inst.family     = Family::SOPP;
		inst.opcode     = opcode;
		inst.word_count = 1;
		m_items.push_back({inst, target, value});
	}
	Instruction& AddPseudo(Opcode opcode, uint32_t site) {
		Instruction inst;
		inst.family     = Family::Unknown;
		inst.opcode     = opcode;
		inst.word_count = 1;
		inst.offset     = site;
		m_items.push_back({inst});
		return m_items.back().inst;
	}

	// Assigns the pcs from `first_pc`; false when a target is unknown.
	bool Finish(uint32_t first_pc, uint32_t guest_end_pc, std::vector<Instruction>& out) {
		std::vector<uint32_t> pcs(m_items.size() + 1);
		uint32_t              pc = first_pc;
		for (size_t i = 0; i < m_items.size(); i++) {
			pcs[i] = pc;
			pc += m_items[i].inst.word_count * 4u;
		}
		pcs[m_items.size()] = pc;
		std::unordered_map<uint32_t, uint32_t> labels;
		for (const auto& [label, index]: m_bindings) {
			labels.emplace(label, pcs[index]);
		}
		out.clear();
		out.reserve(m_items.size());
		for (size_t i = 0; i < m_items.size(); i++) {
			auto inst = m_items[i].inst;
			inst.pc   = pcs[i];
			if (m_items[i].target == Target::Guest) {
				const auto target = m_items[i].value;
				if (target == guest_end_pc) {
					inst.branch_target = pc;
				} else if (const auto found = m_guest_items.find(target); found != m_guest_items.end()) {
					inst.branch_target = pcs[found->second];
				} else {
					return false;
				}
			} else if (m_items[i].target == Target::Label) {
				const auto found = labels.find(m_items[i].value);
				if (found == labels.end()) {
					return false;
				}
				inst.branch_target = found->second;
			}
			out.push_back(std::move(inst));
		}
		return true;
	}

private:
	std::vector<Item>                                m_items;
	std::unordered_map<uint32_t, size_t>             m_guest_items;
	std::vector<std::pair<uint32_t, size_t>>         m_bindings;
	uint32_t                                         m_next_label = 0;
};

// Places a copy of the guest instructions in [begin, end): branches within the range go to the
// copy; `outside` gives the label of any other target and of the fall through past the end.
// With `backward_to_end`, branches back within the range leave it instead (as to `end`).
template <typename Outside>
bool AddRangeCopy(Layout& layout, const Program& program, uint32_t begin, uint32_t end,
                  Outside&& outside, bool backward_to_end = false) {
	const auto first = std::ranges::lower_bound(program.instructions, begin, {}, &Instruction::pc);
	if (first == program.instructions.end() || first->pc != begin) {
		return false;
	}
	std::unordered_map<uint32_t, uint32_t> copy_labels;
	std::vector<const Instruction*>        range;
	for (auto it = first; it != program.instructions.end() && it->pc < end; ++it) {
		if (it->opcode == Opcode::S_SETPC_B64 || it->opcode == Opcode::S_GETPC_B64 ||
		    it->opcode == Opcode::S_ENDPGM || it->opcode == Opcode::UNSUPPORTED) {
			return false;
		}
		copy_labels.emplace(it->pc, layout.NewLabel());
		range.push_back(&*it);
	}
	if (range.empty() || EndPc(*range.back()) != end) {
		return false;
	}
	for (const auto* inst: range) {
		layout.Bind(copy_labels.at(inst->pc));
		layout.AddCopy(*inst);
		if (!IsDirectBranch(inst->opcode)) {
			continue;
		}
		const auto target = inst->branch_target;
		if (backward_to_end && target <= inst->pc && copy_labels.contains(target)) {
			const auto label = outside(end);
			if (label == UINT32_MAX) {
				return false;
			}
			layout.SetBranch(Layout::Target::Label, label);
			continue;
		}
		if (const auto found = copy_labels.find(target); found != copy_labels.end()) {
			layout.SetBranch(Layout::Target::Label, found->second);
			continue;
		}
		const auto label = outside(target);
		if (label == UINT32_MAX) {
			return false;
		}
		layout.SetBranch(Layout::Target::Label, label);
	}
	if (range.back()->opcode != Opcode::S_BRANCH) {
		const auto label = outside(end);
		if (label == UINT32_MAX) {
			return false;
		}
		layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Label, label);
	}
	return true;
}

Operand Constant(uint32_t value) {
	Operand operand;
	operand.kind       = OperandKind::IntegerInlineConstant;
	operand.value      = value;
	operand.signed_val = static_cast<int32_t>(value);
	return operand;
}

Operand Vgpr(uint32_t reg) {
	Operand operand;
	operand.kind  = OperandKind::Vgpr;
	operand.reg   = reg;
	operand.value = reg;
	return operand;
}

// What the per-lane alpha test reads (see RayTracingAlphaTest).
void AddAlphaTest(Layout& layout, const RayTracingSite& site, uint32_t index) {
	const auto& test        = site.alpha;
	const bool  lane        = test.layout == RayTracingAlphaLayout::Lane;
	auto&       alpha       = layout.AddPseudo(Opcode::RT_ALPHA, index);
	alpha.image_nsa_addr[0] = test.lod_vgpr;
	alpha.image_nsa_addr[1] = test.material_offset;
	alpha.image_nsa_addr[2] = test.buffer_offset;
	alpha.image_nsa_addr[3] = test.threshold;
	alpha.image_nsa_addr[4] = test.lod_offset;
	alpha.image_nsa_addr[5] = lane ? test.reference_vgpr : NoRegister;
	alpha.src0              = lane ? Vgpr(test.spread_reg) : Sgpr(test.spread_reg);
	if (!lane) {
		alpha.src1 = Sgpr(test.factor_sgpr);
	}
	alpha.src2  = Sgpr(test.material_user);
	alpha.src3  = Sgpr(test.buffer_user);
	alpha.dmask = static_cast<uint32_t>(test.layout);
}

// The native traversal of a closest hit site: the hardware finds the hit (every triangle opaque,
// both facings); no guest code runs for it.
void AddClosestTraversal(Layout& layout, const RayTracingSite& site, uint32_t index) {
	if (site.alpha.valid) {
		AddAlphaTest(layout, site, index);
	}
	auto& init = layout.AddPseudo(Opcode::RT_INIT, index);
	for (uint32_t i = 0; i < 3; i++) {
		init.image_nsa_addr[i]     = site.origin[i];
		init.image_nsa_addr[3 + i] = site.direction[i];
	}
	init.image_nsa_addr[6] = site.t_max;
	init.image_nsa_addr[7] = site.cull_mask;
	init.image_nsa_addr[8] = site.cull_mask_constant;
	init.src0              = Sgpr(site.header_sgpr);
	// Every triangle opaque, or the opaque ones by their primitive word and the alpha test.
	init.src1  = Constant(site.alpha.valid ? 0 : 1);
	init.dmask = 1 | (site.alpha.valid ? 2 : 0); // Both facings (and opacity from the geometry).
	layout.AddBranch(Opcode::S_CBRANCH_SCC0, Layout::Target::Guest, site.loop_pc);
	const auto loop_label = layout.NewLabel(), exit_label = layout.NewLabel();
	// Runs the queries to their end (an opaque only query has no candidates; one would be dropped).
	layout.Bind(loop_label);
	auto& select = layout.AddPseudo(Opcode::RT_SELECT, index);
	select.dst   = Sgpr(site.instance_sgpr);
	select.dst2  = Sgpr(site.mask_sgpr);
	layout.AddBranch(Opcode::S_CBRANCH_SCC0, Layout::Target::Label, exit_label);
	auto& skip = layout.AddPseudo(Opcode::RT_RESOLVE, index);
	skip.src0  = Sgpr(site.mask_sgpr);
	skip.dmask = 1;
	layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Label, loop_label);
	layout.Bind(exit_label);
	auto& commit = layout.AddPseudo(Opcode::RT_COMMIT, index);
	commit.image_nsa_addr[0] = site.result_t;
	commit.image_nsa_addr[1] = site.result_word;
	commit.image_nsa_addr[2] = site.result_instance;
	commit.image_nsa_addr[3] = site.result_instance + 1;
	commit.image_nsa_addr[4] = site.result_u;
	commit.image_nsa_addr[5] = site.result_v;
	commit.image_nsa_addr[6] = site.result_node;
	commit.dmask             = 1; // Only when nearer than the hit in the registers.
	layout.AddPseudo(Opcode::RT_EXIT, index);
	layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Guest, site.exit_pc);
}

// The native traversal of a site, placed between its entry instruction and the loop header:
// RT_INIT falls back to the loop header, else the native loop runs and exits after the loop.
bool AddNativeTraversal(Layout& layout, const Program& program, const RayTracingSite& site,
                        uint32_t index) {
	if (site.kind == RayTracingKind::ClosestOpaque) {
		AddClosestTraversal(layout, site, index);
		return true;
	}
	if (site.alpha.valid) {
		AddAlphaTest(layout, site, index);
	}
	auto& init = layout.AddPseudo(Opcode::RT_INIT, index);
	for (uint32_t i = 0; i < 3; i++) {
		init.image_nsa_addr[i]     = site.origin[i];
		init.image_nsa_addr[3 + i] = site.direction[i];
	}
	init.image_nsa_addr[6] = site.t_max;
	init.image_nsa_addr[7] = site.cull_mask;
	init.image_nsa_addr[8] = site.cull_mask_constant;
	init.src0              = Sgpr(site.header_sgpr);
	init.src1              = site.alpha_off_sgpr != NoAlphaOff ? Sgpr(site.alpha_off_sgpr) : Constant(0);
	layout.AddBranch(Opcode::S_CBRANCH_SCC0, Layout::Target::Guest, site.loop_pc);

	const auto loop_label = layout.NewLabel(), exit_label = layout.NewLabel();
	const auto setup_label = layout.NewLabel(), leaf_label = layout.NewLabel();
	const auto tail_label = layout.NewLabel(), cull_label = layout.NewLabel();
	const auto exit_check_label = layout.NewLabel();
	// The instruction that undoes the instance setup's spills (none: nothing to undo).
	const auto restore = [&] {
		if (site.after_leaf_pc != 0) {
			layout.AddCopy(*FindInstruction(program, site.after_leaf_pc));
		}
	};

	// Loop header: the wave's next (instance, leaf), with the lanes that have it, or the exit.
	layout.Bind(loop_label);
	auto& select = layout.AddPseudo(Opcode::RT_SELECT, index);
	select.dst   = Sgpr(site.instance_sgpr);
	select.dst2  = Sgpr(site.mask_sgpr);
	layout.AddBranch(Opcode::S_CBRANCH_SCC0, Layout::Target::Label, exit_check_label);
	// The instance whose setup ran last: its registers still hold the object space ray.
	layout.AddPseudo(Opcode::RT_REUSE, index);
	layout.AddBranch(Opcode::S_CBRANCH_SCC1, Layout::Target::Label, leaf_label);
	// Another instance: undo the previous setup's spills, then set this one up.
	layout.AddPseudo(Opcode::RT_SETUP, index);
	layout.AddBranch(Opcode::S_CBRANCH_SCC0, Layout::Target::Label, setup_label);
	restore();
	layout.Bind(setup_label);
	// The guest instance setup; it continues with the selected leaf instead of the BLAS stack.
	if (!AddRangeCopy(layout, program, site.setup_pc, site.blas_head_pc, [&](uint32_t target) {
		    return target == site.blas_head_pc  ? leaf_label
		           : target == site.cull_exit_pc ? cull_label
		                                         : UINT32_MAX;
	    })) {
		return false;
	}
	layout.Bind(leaf_label);
	auto& leaf = layout.AddPseudo(Opcode::RT_LEAF, index);
	leaf.dst   = Sgpr(site.leaf_sgpr);
	leaf.dst2  = Sgpr(site.saved_exec_sgpr);
	leaf.src0  = Sgpr(site.mask_sgpr);
	// The guest leaf code; it ends at the BLAS loop latch. The guest loops over the rest of the
	// leaf list until an entry marked last; here it tests only the selected entry (the hardware
	// reports the other triangles as candidates of their own). A selected entry from a list the
	// guest rewrote meanwhile could otherwise run past the end of the BLAS, where the loads read
	// zeros and no entry is ever the last one.
	if (!AddRangeCopy(
	        layout, program, site.leaf_pc, site.tail_pc,
	        [&](uint32_t target) { return target == site.tail_pc ? tail_label : UINT32_MAX; },
	        true)) {
		return false;
	}
	// The end of an iteration: lanes that accepted a hit in the leaf stop.
	layout.Bind(tail_label);
	auto& resolve = layout.AddPseudo(Opcode::RT_RESOLVE, index);
	resolve.src0  = Sgpr(site.mask_sgpr);
	layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Label, loop_label);
	// The instance mask culled every selected lane (before the setup spilled anything).
	layout.Bind(cull_label);
	auto& skip = layout.AddPseudo(Opcode::RT_RESOLVE, index);
	skip.src0  = Sgpr(site.mask_sgpr);
	skip.dmask = 1;
	layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Label, loop_label);

	// The exit: opaque hits the hardware found, then undo the last setup's spills.
	layout.Bind(exit_check_label);
	auto& commit = layout.AddPseudo(Opcode::RT_COMMIT, index);
	commit.image_nsa_addr[0] = site.result_t;
	commit.image_nsa_addr[1] = site.result_word;
	commit.image_nsa_addr[2] = site.result_instance;
	commit.image_nsa_addr[3] = site.result_instance + 1;
	commit.image_nsa_addr[4] = NoRegister;
	commit.image_nsa_addr[5] = NoRegister;
	commit.image_nsa_addr[6] = NoRegister;
	layout.AddPseudo(Opcode::RT_EXIT_CHECK, index);
	layout.AddBranch(Opcode::S_CBRANCH_SCC0, Layout::Target::Label, exit_label);
	restore();
	layout.Bind(exit_label);
	layout.AddPseudo(Opcode::RT_EXIT, index);
	layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Guest, site.exit_pc);
	return true;
}

} // namespace

// ---- Structural derivation of the traversal sites (see the top of RayTracingPatch.h). ----
//
// The traversal loops come from a template of the BVH library (sce::Psr's kBase format: root
// node 0x25, node type in the low 3 bits, a wave-wide stack of node pointers in VGPR lanes capped
// at 60 entries with S_MIN_U32, the T# of a BVH built from its address with
// S_BFE_U64 dst, addr, 0x280008); the shader's own code is the instance setup and the leaf code.
// The sites are found by that structure:
//
// Packet sites (a wave-wide traversal, the guest leaf code tests each leaf):
//   loop header   S_AND_B64 (the target of the loop's backward branch), S_AND_SAVEEXEC_B64,
//                 S_AND_B32 vcc_lo, <node>, 7, S_ADD_I32 vcc_lo, vcc_lo, -4 ... S_CMP_LE_U32 vcc_lo, 2,
//                 S_CBRANCH_SCC0 <instance setup>
//   box test      the first IMAGE_BVH_INTERSECT_RAY after it: t max, origin and direction
//                 VGPRs; its T# comes from the S_BFE_U64 of the TLAS header address
//   stack         the S_MIN_U32 <sp>, <sp>, 60 after the box test
//   setup         S_LOAD_DWORD of the instance record +0x3c, V_AND_B32 of its mask byte with the
//                 ray's cull mask VGPR, S_CBRANCH_EXECZ <cull exit>
//   BLAS head     S_CMP_LG_U32 <sp>, ...; V_READLANE_B32 <leaf>, ...; S_AND_B64 vcc, <mask>, vcc;
//                 S_AND_SAVEEXEC_B64 <saved>, vcc; ... S_CMP_GT_U32 vcc_lo, 3; S_CBRANCH_SCC0 <leaf>
//   tail          S_CMP_LG_U64 <mask>, 0 after the leaf code
//   accepted hit  the V_MOV_B32s before S_ANDN2_B64 <mask>, <mask>, exec in the leaf code: the
//                 distance (from a VGPR), the primitive word and the instance record pair (SGPRs)
//   exit          the S_CBRANCH_VCCNZ before the loop's backward branch
//   alpha switch  S_CMP_LG_U32 <s>, 0 then S_CSELECT_B32 vcc_lo, 1, vcc_lo in the leaf code
//   after leaf    a V_READLANE_B32 after the tail restoring an SGPR the setup saved with
//                 V_WRITELANE_B32
// Closest hit sites (a traversal per lane, nearest opaque hit):
//   loop header   S_MOV_B64 <s>, exec (the target of the loop's backward branch), S_AND_B32
//                 vcc_lo, <node>, 7, S_ADD_I32 vcc_lo, vcc_lo, -4, S_CMP_LE_U32 vcc_lo, 2
//   box test      the IMAGE_BVH_INTERSECT_RAY whose T# is the TLAS's
//   exit          the S_CBRANCH_SCC0 just before the loop's backward branch that skips it
//   nearest hit   after V_CMP_GT_F32 vcc_lo, <t>, <t max> and S_AND_SAVEEXEC_B64 vcc, vcc: two
//                 V_MUL_F32 (barycentrics) and V_MOV_B32s of the distance (into t max), the
//                 instance record pair, and the two words of the list entry (BUFFER_LOAD_DWORDX2:
//                 triangle node, primitive word)
// The TLAS header address is a 64-bit load from a user data pointer (S_LOAD_DWORDX2 of an SGPR
// pair nothing writes before) and the root is the S_MOV_B32 <node>, 0x25 before the loop.
namespace {

bool IsSgpr(const Operand& operand, uint32_t reg) {
	return operand.kind == OperandKind::Sgpr && operand.reg == reg;
}

bool IsVccLo(const Operand& operand) {
	return operand.kind == OperandKind::VccLo;
}

bool IsConstant(const Operand& operand, uint32_t value) {
	return (operand.kind == OperandKind::IntegerInlineConstant ||
	        operand.kind == OperandKind::LiteralConstant) &&
	       operand.value == value;
}

bool SameOperand(const Operand& a, const Operand& b) {
	if (a.kind != b.kind) {
		return false;
	}
	switch (a.kind) {
		case OperandKind::Sgpr:
		case OperandKind::Vgpr: return a.reg == b.reg;
		case OperandKind::IntegerInlineConstant:
		case OperandKind::LiteralConstant: return a.value == b.value;
		default: return true;
	}
}

class SiteFinder {
public:
	explicit SiteFinder(const Program& program): m_inst(program.instructions) {}

	// The sites of the program and its TLAS source; false when a traversal loop does not match.
	bool Find(RayTracingPatch& patch) {
		bool any = false;
		for (size_t i = 0; i < m_inst.size(); i++) {
			if (m_inst[i].opcode != Opcode::IMAGE_BVH_INTERSECT_RAY) {
				continue;
			}
			any = true;
		}
		if (!any) {
			return false;
		}
		std::vector<RayTracingSite> sites;
		uint32_t                    user_sgpr = NoRegister, header_offset = 0, root = 0;
		for (size_t header = 1; header < m_inst.size(); header++) {
			const auto back = BackwardBranchTo(header);
			if (back == npos) {
				continue;
			}
			RayTracingSite site;
			size_t         node_index = npos;
			if (IsPacketHeader(header)) {
				if (!DerivePacket(header, back, site)) {
					return false;
				}
				node_index = header + 2;
			} else if (IsClosestHeader(header)) {
				if (!DeriveClosest(header, back, site)) {
					return false;
				}
				node_index = header + 1;
			} else {
				continue;
			}
			// The TLAS source: the header pair is loaded from a user data pointer, the root is
			// the node register's first value.
			uint32_t site_user = NoRegister, site_offset = 0, site_root = 0;
			if (!FindTlasSource(header, site.header_sgpr, site_user, site_offset) ||
			    !FindRoot(header, m_inst[node_index].src0, site_root)) {
				return false;
			}
			if (user_sgpr != NoRegister &&
			    (site_user != user_sgpr || site_offset != header_offset || site_root != root)) {
				return false;
			}
			user_sgpr     = site_user;
			header_offset = site_offset;
			root          = site_root;
			sites.push_back(site);
		}
		if (sites.empty() || sites.size() > 8) {
			return false;
		}
		patch.sites         = std::move(sites);
		patch.user_sgpr     = user_sgpr;
		patch.header_offset = header_offset;
		patch.root          = root;
		return true;
	}

private:
	static constexpr size_t npos = SIZE_MAX;

	const Instruction& At(size_t i) const { return m_inst[i]; }

	size_t IndexOf(uint32_t pc) const {
		const auto found = std::ranges::lower_bound(m_inst, pc, {}, &Instruction::pc);
		return found != m_inst.end() && found->pc == pc ? static_cast<size_t>(found - m_inst.begin()) : npos;
	}

	// The last backward S_BRANCH to instruction `target`, or npos.
	size_t BackwardBranchTo(size_t target) const {
		size_t found = npos;
		for (size_t i = target + 1; i < m_inst.size(); i++) {
			if (m_inst[i].opcode == Opcode::S_BRANCH && m_inst[i].branch_target == m_inst[target].pc) {
				found = i;
			}
		}
		return found;
	}

	// The next instruction from `from` (included) with `opcode`, before `limit`.
	size_t Next(size_t from, Opcode opcode, size_t limit = SIZE_MAX) const {
		for (size_t i = from; i < m_inst.size() && i < limit; i++) {
			if (m_inst[i].opcode == opcode) {
				return i;
			}
		}
		return npos;
	}

	// The next instruction after `i` that is not S_NOP.
	size_t NextReal(size_t i) const {
		for (i++; i < m_inst.size() && m_inst[i].opcode == Opcode::S_NOP; i++) {
		}
		return i < m_inst.size() ? i : npos;
	}

	// The address VGPR of component `k` of an IMAGE_BVH_INTERSECT_RAY (node, t max, origin,
	// direction, inverse direction).
	static uint32_t AddressReg(const Instruction& inst, uint32_t k) {
		if (inst.src0.kind != OperandKind::Vgpr) {
			return NoRegister;
		}
		if (k == 0) {
			return inst.src0.reg;
		}
		return inst.image_nsa_dwords != 0 ? inst.image_nsa_addr[k - 1] : inst.src0.reg + k;
	}

	// S_AND_B32 vcc_lo, <node>, 7 at `i` followed by S_ADD_I32 vcc_lo, vcc_lo, -4.
	bool IsNodeType(size_t i) const {
		if (i + 1 >= m_inst.size()) {
			return false;
		}
		const auto& a = m_inst[i];
		const auto& b = m_inst[i + 1];
		return a.opcode == Opcode::S_AND_B32 && IsVccLo(a.dst) && a.src0.kind == OperandKind::Sgpr &&
		       IsConstant(a.src1, 7) && b.opcode == Opcode::S_ADD_I32 && IsVccLo(b.dst) &&
		       IsVccLo(b.src0) && b.src1.kind == OperandKind::IntegerInlineConstant && b.src1.signed_val == -4;
	}

	bool IsPacketHeader(size_t i) const {
		return i + 3 < m_inst.size() && m_inst[i].opcode == Opcode::S_AND_B64 &&
		       m_inst[i + 1].opcode == Opcode::S_AND_SAVEEXEC_B64 && IsNodeType(i + 2);
	}

	bool IsClosestHeader(size_t i) const {
		return i + 2 < m_inst.size() && m_inst[i].opcode == Opcode::S_MOV_B64 &&
		       m_inst[i].src0.kind == OperandKind::ExecLo && IsNodeType(i + 1);
	}

	// The box test after `from`: the first IMAGE_BVH_INTERSECT_RAY, its ray registers, and the
	// TLAS header SGPR pair its T# was built from (before `header`).
	bool BoxTest(size_t header, size_t box, RayTracingSite& site) const {
		const auto& test = m_inst[box];
		if (test.src1.kind != OperandKind::Sgpr) {
			return false;
		}
		site.t_max = AddressReg(test, 1);
		for (uint32_t c = 0; c < 3; c++) {
			site.origin[c]    = AddressReg(test, 2 + c);
			site.direction[c] = AddressReg(test, 5 + c);
		}
		for (size_t i = header; i-- > 0;) {
			const auto& bfe = m_inst[i];
			if (bfe.opcode == Opcode::S_BFE_U64 && IsSgpr(bfe.dst, test.src1.reg) &&
			    bfe.src0.kind == OperandKind::Sgpr && IsConstant(bfe.src1, 0x280008)) {
				site.header_sgpr = bfe.src0.reg;
				return site.t_max != NoRegister;
			}
		}
		return false;
	}

	bool DerivePacket(size_t header, size_t back, RayTracingSite& site) const {
		site.kind     = RayTracingKind::Packet;
		site.entry_pc = m_inst[header - 1].pc;
		site.loop_pc  = m_inst[header].pc;
		const auto& node_type = m_inst[header + 2];
		site.instance_sgpr    = node_type.src0.reg;
		// The instance setup.
		const auto le = Next(header + 3, Opcode::S_CMP_LE_U32, back);
		if (le == npos || !IsVccLo(m_inst[le].src0) || !IsConstant(m_inst[le].src1, 2) ||
		    m_inst[le + 1].opcode != Opcode::S_CBRANCH_SCC0) {
			return false;
		}
		site.setup_pc = m_inst[le + 1].branch_target;
		const auto box = Next(le, Opcode::IMAGE_BVH_INTERSECT_RAY, back);
		if (box == npos || !BoxTest(header, box, site)) {
			return false;
		}
		// The stack pointer.
		const auto cap = Next(box, Opcode::S_MIN_U32, back);
		if (cap == npos || m_inst[cap].dst.kind != OperandKind::Sgpr || !IsConstant(m_inst[cap].src1, 60)) {
			return false;
		}
		const auto sp = m_inst[cap].dst.reg;
		// The instance mask test of the setup.
		const auto setup = IndexOf(site.setup_pc);
		if (setup == npos) {
			return false;
		}
		size_t mask_load = npos;
		for (size_t i = setup; i < back && mask_load == npos; i++) {
			if (m_inst[i].opcode == Opcode::S_LOAD_DWORD && m_inst[i].offset == 0x3c) {
				mask_load = i;
			}
		}
		if (mask_load == npos) {
			return false;
		}
		const auto mask_and = Next(mask_load, Opcode::V_AND_B32, back);
		if (mask_and == npos || !SameOperand(m_inst[mask_and].src0, m_inst[mask_load].dst) ||
		    m_inst[mask_and].src1.kind != OperandKind::Vgpr) {
			return false;
		}
		site.cull_mask   = m_inst[mask_and].src1.reg;
		const auto culled = Next(mask_and, Opcode::S_CBRANCH_EXECZ, back);
		if (culled == npos) {
			return false;
		}
		site.cull_exit_pc = m_inst[culled].branch_target;
		// The BLAS loop head.
		size_t blas_head = npos;
		for (size_t i = culled; i < back && blas_head == npos; i++) {
			if (m_inst[i].opcode == Opcode::S_CMP_LG_U32 && IsSgpr(m_inst[i].src0, sp)) {
				blas_head = i;
			}
		}
		if (blas_head == npos) {
			return false;
		}
		site.blas_head_pc = m_inst[blas_head].pc;
		const auto saveexec = Next(blas_head, Opcode::S_AND_SAVEEXEC_B64, back);
		if (saveexec == npos || m_inst[saveexec].dst.kind != OperandKind::Sgpr ||
		    m_inst[saveexec - 1].opcode != Opcode::S_AND_B64 ||
		    m_inst[saveexec - 1].src0.kind != OperandKind::Sgpr) {
			return false;
		}
		site.saved_exec_sgpr = m_inst[saveexec].dst.reg;
		site.mask_sgpr       = m_inst[saveexec - 1].src0.reg;
		site.leaf_sgpr       = NoRegister;
		for (size_t i = blas_head; i < saveexec && site.leaf_sgpr == NoRegister; i++) {
			if (m_inst[i].opcode == Opcode::V_READLANE_B32 && m_inst[i].dst.kind == OperandKind::Sgpr) {
				site.leaf_sgpr = m_inst[i].dst.reg;
			}
		}
		const auto gt = Next(saveexec, Opcode::S_CMP_GT_U32, back);
		if (site.leaf_sgpr == NoRegister || gt == npos || !IsConstant(m_inst[gt].src1, 3) ||
		    m_inst[gt + 1].opcode != Opcode::S_CBRANCH_SCC0) {
			return false;
		}
		site.leaf_pc    = m_inst[gt + 1].branch_target;
		const auto leaf = IndexOf(site.leaf_pc);
		if (leaf == npos) {
			return false;
		}
		// The BLAS loop latch.
		size_t tail = npos;
		for (size_t i = leaf; i < back && tail == npos; i++) {
			if (m_inst[i].opcode == Opcode::S_CMP_LG_U64 && IsSgpr(m_inst[i].src0, site.mask_sgpr) &&
			    IsConstant(m_inst[i].src1, 0)) {
				tail = i;
			}
		}
		if (tail == npos) {
			return false;
		}
		site.tail_pc = m_inst[tail].pc;
		// The accepted hit: the moves before the lanes leave the mask.
		size_t accept = npos;
		for (size_t i = tail; i-- > leaf && accept == npos;) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::S_ANDN2_B64 && IsSgpr(inst.dst, site.mask_sgpr) &&
			    IsSgpr(inst.src0, site.mask_sgpr) && inst.src1.kind == OperandKind::ExecLo) {
				accept = i;
			}
		}
		if (accept == npos || accept < leaf + 4) {
			return false;
		}
		const auto& t        = m_inst[accept - 4];
		const auto& word     = m_inst[accept - 3];
		const auto& instance = m_inst[accept - 2];
		const auto& high     = m_inst[accept - 1];
		for (const auto* move: {&t, &word, &instance, &high}) {
			if (move->opcode != Opcode::V_MOV_B32 || move->dst.kind != OperandKind::Vgpr) {
				return false;
			}
		}
		if (t.src0.kind != OperandKind::Vgpr || word.src0.kind != OperandKind::Sgpr ||
		    instance.src0.kind != OperandKind::Sgpr || !IsSgpr(high.src0, instance.src0.reg + 1) ||
		    high.dst.reg != instance.dst.reg + 1) {
			return false;
		}
		site.result_t        = t.dst.reg;
		site.result_word     = word.dst.reg;
		site.result_instance = instance.dst.reg;
		// The exit of the traversal.
		size_t exit = npos;
		for (size_t i = back; i-- > tail && exit == npos;) {
			if (m_inst[i].opcode == Opcode::S_CBRANCH_VCCNZ) {
				exit = i;
			}
		}
		if (exit == npos) {
			return false;
		}
		site.exit_pc = m_inst[exit].branch_target;
		// The alpha test switch (none: the leaf code decides per lane).
		site.alpha_off_sgpr = NoAlphaOff;
		for (size_t i = leaf; i < tail; i++) {
			const auto& cmp = m_inst[i];
			if (cmp.opcode != Opcode::S_CMP_LG_U32 || cmp.src0.kind != OperandKind::Sgpr ||
			    !IsConstant(cmp.src1, 0)) {
				continue;
			}
			const auto sel = NextReal(i);
			if (sel != npos && m_inst[sel].opcode == Opcode::S_CSELECT_B32 && IsVccLo(m_inst[sel].dst) &&
			    IsConstant(m_inst[sel].src0, 1) && IsVccLo(m_inst[sel].src1)) {
				if (site.alpha_off_sgpr != NoAlphaOff) {
					return false;
				}
				site.alpha_off_sgpr = cmp.src0.reg;
			}
		}
		// What the setup spilled into lanes and the traversal restores after the BLAS loop.
		site.after_leaf_pc = 0;
		for (size_t i = tail; i < exit; i++) {
			const auto& restore = m_inst[i];
			if (restore.opcode != Opcode::V_READLANE_B32 || restore.dst.kind != OperandKind::Sgpr) {
				continue;
			}
			for (size_t j = setup; j < blas_head; j++) {
				const auto& save = m_inst[j];
				if (save.opcode == Opcode::V_WRITELANE_B32 && SameOperand(save.dst, restore.src0) &&
				    IsSgpr(save.src0, restore.dst.reg) && SameOperand(save.src1, restore.src1)) {
					if (site.after_leaf_pc != 0) {
						return false;
					}
					site.after_leaf_pc = restore.pc;
					break;
				}
			}
		}
		// The alpha test (none: the native traversal runs the leaf code for each candidate).
		site.alpha = {};
		RayTracingAlphaTest alpha;
		if (DeriveAlphaTest(header, back, setup, leaf, tail, t.src0.reg, word.src0.reg,
		                    instance.src0.reg, alpha)) {
			site.alpha = alpha;
		}
		return site.cull_mask != NoRegister;
	}

	// ---- The alpha test of GTA V's leaf code (see RayTracingAlphaTest). ----
	//
	// The leaf code tests one entry of a leaf list; past its opaque check, the alpha test is
	// straight-line code. It is recognized by the dataflow of the values it computes: each
	// register a step reads is traced to the last instruction before it that writes it (within
	// the leaf code), which must be the expected one with the expected constants.

	static uint32_t ScalarIndex(const Operand& operand) {
		switch (operand.kind) {
			case OperandKind::Sgpr: return operand.reg;
			case OperandKind::VccLo: return 106u;
			case OperandKind::VccHi: return 107u;
			default: return NoRegister;
		}
	}

	// Whether the instruction writes SGPR `reg` (lane masks and 64-bit results are pairs; loads
	// write their dwords).
	static bool WritesScalar(const Instruction& inst, uint32_t reg) {
		const auto name  = magic_enum::enum_name(inst.opcode);
		const bool wide  = name.find("64") != std::string_view::npos || name.starts_with("V_CMP");
		const auto count = inst.family == Family::SMEM ? std::max(inst.data_dwords, 1u) : (wide ? 2u : 1u);
		const auto covers = [&](const Operand& dst, uint32_t n) {
			const auto first = ScalarIndex(dst);
			return first != NoRegister && reg >= first && reg < first + n;
		};
		return covers(inst.dst, count) || covers(inst.dst2, 2u);
	}

	// Whether the instruction writes VGPR `reg` (conservatively wide for image results).
	static bool WritesVector(const Instruction& inst, uint32_t reg) {
		if (inst.dst.kind != OperandKind::Vgpr) {
			return false;
		}
		uint32_t count = 1;
		switch (inst.family) {
			case Family::MIMG: count = 8; break;
			case Family::MUBUF:
			case Family::MTBUF:
			case Family::FLAT: count = std::max(inst.data_dwords, 1u); break;
			case Family::DS: count = std::max(inst.data_dwords, 1u) * 2u; break;
			default:
				count = magic_enum::enum_name(inst.opcode).find("64") != std::string_view::npos ? 2u : 1u;
				break;
		}
		return reg >= inst.dst.reg && reg < inst.dst.reg + count;
	}

	// The last instruction in [from, i) that writes SGPR `reg` (npos: none).
	size_t ScalarWriter(size_t i, uint32_t reg, size_t from) const {
		if (reg == NoRegister) {
			return npos;
		}
		for (size_t j = i; j-- > from;) {
			if (WritesScalar(m_inst[j], reg)) {
				return j;
			}
		}
		return npos;
	}

	// The writer of SGPR operand `operand` of instruction `i`, when it has `opcode` (npos: other).
	size_t ScalarSource(size_t i, const Operand& operand, Opcode opcode, size_t from) const {
		const auto j = ScalarWriter(i, ScalarIndex(operand), from);
		return j != npos && m_inst[j].opcode == opcode ? j : npos;
	}

	// The writer of the `count` SGPRs from `reg` read by instruction `i`: one instruction with
	// `opcode` writing them all (npos: other).
	size_t ScalarRangeSource(size_t i, uint32_t reg, uint32_t count, Opcode opcode, size_t from) const {
		const auto j = ScalarWriter(i, reg, from);
		if (j == npos || m_inst[j].opcode != opcode || ScalarIndex(m_inst[j].dst) != reg) {
			return npos;
		}
		for (uint32_t k = 1; k < count; k++) {
			if (ScalarWriter(i, reg + k, from) != j) {
				return npos;
			}
		}
		return j;
	}

	size_t VectorSource(size_t i, const Operand& operand, Opcode opcode, size_t from) const {
		if (operand.kind != OperandKind::Vgpr) {
			return npos;
		}
		for (size_t j = i; j-- > from;) {
			if (WritesVector(m_inst[j], operand.reg)) {
				return m_inst[j].opcode == opcode && m_inst[j].dst.reg == operand.reg ? j : npos;
			}
		}
		return npos;
	}

	static bool Plain(const Operand& operand) {
		return !operand.negate && !operand.absolute && !operand.clamp && operand.omod == 0 &&
		       operand.sdwa_sel == 6 && !operand.dpp && !operand.dpp8;
	}

	// S_<op> dst, src, <constant> with the source register `source` (NoRegister: any).
	bool IsScalarConstantOp(size_t i, Opcode opcode, uint32_t constant) const {
		return i != npos && m_inst[i].opcode == opcode && IsConstant(m_inst[i].src1, constant);
	}

	// A user data V#: S_LOAD_DWORDX4 from an SGPR pair no instruction before `limit` writes.
	bool UserVsharp(size_t load, size_t limit, uint32_t& user, uint32_t& offset) const {
		if (load == npos || m_inst[load].opcode != Opcode::S_LOAD_DWORDX4 ||
		    m_inst[load].src0.kind != OperandKind::Sgpr ||
		    (m_inst[load].src1.kind != OperandKind::Unknown && m_inst[load].src1.kind != OperandKind::Null)) {
			return false;
		}
		const auto base = m_inst[load].src0.reg;
		for (size_t j = 0; j < limit; j++) {
			if (WritesScalar(m_inst[j], base) || WritesScalar(m_inst[j], base + 1)) {
				return false;
			}
		}
		user   = base;
		offset = m_inst[load].offset;
		return true;
	}

	// The S_LOAD_DWORD <dst>, s[base:base+1], <constant offset> that writes `operand` before `i`.
	size_t ConstantFieldLoad(size_t i, const Operand& operand, uint32_t base, uint32_t offset, size_t from) const {
		const auto j = ScalarSource(i, operand, Opcode::S_LOAD_DWORD, from);
		return j != npos && IsSgpr(m_inst[j].src0, base) && m_inst[j].offset == offset &&
		               (m_inst[j].src1.kind == OperandKind::Unknown || m_inst[j].src1.kind == OperandKind::Null)
		           ? j
		           : npos;
	}

	bool DeriveAlphaTest(size_t header, size_t back, size_t setup, size_t leaf, size_t tail, uint32_t t_vgpr,
	                     uint32_t word_sgpr, uint32_t record_sgpr, RayTracingAlphaTest& alpha) const {
		constexpr uint32_t One = 0x3f800000u, NoTypeBits = 0xc0000000u;
		// The sample: the leaf code's only IMAGE_SAMPLE, an explicit level of the W channel, then
		// V_CMP_GT_F32 <sampled>, <reference>.
		size_t sample = npos;
		for (size_t i = leaf; i < tail; i++) {
			if (m_inst[i].opcode == Opcode::IMAGE_SAMPLE) {
				if (sample != npos) {
					return false;
				}
				sample = i;
			}
		}
		if (sample == npos) {
			return false;
		}
		const auto& sampled = m_inst[sample];
		if (sampled.dmask != 8 || sampled.image_sample_flags != ImageSampleFlagLod ||
		    sampled.image_dimension != ImageDimension::Dim2D || sampled.src1.kind != OperandKind::Sgpr ||
		    sampled.dst.kind != OperandKind::Vgpr) {
			return false;
		}
		// The T#: 32 bytes of the texture table at (M2 >> 4) * 32.
		const auto tsharp = ScalarRangeSource(sample, sampled.src1.reg, 8, Opcode::S_BUFFER_LOAD_DWORDX8, leaf);
		if (tsharp == npos || m_inst[tsharp].offset != 0 || m_inst[tsharp].src0.kind != OperandKind::Sgpr) {
			return false;
		}
		const auto key_scale = ScalarSource(tsharp, m_inst[tsharp].src1, Opcode::S_LSHL_B32, leaf);
		if (!IsScalarConstantOp(key_scale, Opcode::S_LSHL_B32, 5)) {
			return false;
		}
		const auto key = ScalarSource(key_scale, m_inst[key_scale].src0, Opcode::S_LSHR_B32, leaf);
		if (!IsScalarConstantOp(key, Opcode::S_LSHR_B32, 4)) {
			return false;
		}
		uint32_t texture_user = 0, texture_offset = 0;
		if (!UserVsharp(ScalarRangeSource(tsharp, m_inst[tsharp].src0.reg, 4, Opcode::S_LOAD_DWORDX4, leaf), tail,
		                texture_user, texture_offset)) {
			return false;
		}
		// M2: the third dword of the material record.
		const auto m2 = ScalarSource(key, m_inst[key].src0, Opcode::S_BUFFER_LOAD_DWORD, leaf);
		if (m2 == npos || m_inst[m2].offset != 8 || m_inst[m2].src0.kind != OperandKind::Sgpr) {
			return false;
		}
		const auto material_vsharp = m_inst[m2].src0.reg;
		const auto record_offset   = ScalarIndex(m_inst[m2].src1);
		if (!UserVsharp(ScalarRangeSource(m2, material_vsharp, 4, Opcode::S_LOAD_DWORDX4, leaf), tail,
		                alpha.material_user, alpha.material_offset)) {
			return false;
		}
		// Its offset: (geometry word + R[0x78]) * 12.
		const auto scale4 = ScalarSource(m2, m_inst[m2].src1, Opcode::S_LSHL_B32, leaf);
		if (!IsScalarConstantOp(scale4, Opcode::S_LSHL_B32, 2)) {
			return false;
		}
		const auto scale3 = ScalarSource(scale4, m_inst[scale4].src0, Opcode::S_MUL_I32, leaf);
		if (!IsScalarConstantOp(scale3, Opcode::S_MUL_I32, 3)) {
			return false;
		}
		const auto sum = ScalarSource(scale3, m_inst[scale3].src0, Opcode::S_ADD_I32, leaf);
		if (sum == npos) {
			return false;
		}
		const auto blas_head = [&](size_t i) {
			// The BLAS the leaf code's triangle test reads: the S_BFE_U64 of its T# in the setup.
			for (size_t j = leaf; j < tail; j++) {
				if (m_inst[j].opcode != Opcode::IMAGE_BVH_INTERSECT_RAY || m_inst[j].src1.kind != OperandKind::Sgpr) {
					continue;
				}
				const auto bfe = ScalarSource(j, m_inst[j].src1, Opcode::S_BFE_U64, setup);
				return bfe != npos && IsConstant(m_inst[bfe].src1, 0x280008) && m_inst[bfe].src0.kind == OperandKind::Sgpr &&
				       m_inst[bfe].src0.reg == i;
			}
			return false;
		};
		bool geometry_ok = false, base_ok = false;
		for (const auto* operand: {&m_inst[sum].src0, &m_inst[sum].src1}) {
			if (ConstantFieldLoad(sum, *operand, record_sgpr, 0x78, leaf) != npos) {
				base_ok = true;
				continue;
			}
			// The geometry word: B[B[0x28] + ((N >> 3) - B[0x7c]) * 4] of the triangle node N.
			const auto load = ScalarSource(sum, *operand, Opcode::S_LOAD_DWORD, leaf);
			if (load == npos || m_inst[load].offset != 0 || m_inst[load].src0.kind != OperandKind::Sgpr ||
			    !blas_head(m_inst[load].src0.reg)) {
				continue;
			}
			const auto blas  = m_inst[load].src0.reg;
			const auto index = ScalarSource(load, m_inst[load].src1, Opcode::S_LSHL2_ADD_U32, leaf);
			if (index == npos || ConstantFieldLoad(index, m_inst[index].src1, blas, 0x28, leaf) == npos) {
				continue;
			}
			const auto local = ScalarSource(index, m_inst[index].src0, Opcode::S_SUB_U32, leaf);
			if (local == npos || ConstantFieldLoad(local, m_inst[local].src1, blas, 0x7c, leaf) == npos) {
				continue;
			}
			const auto node = ScalarSource(local, m_inst[local].src0, Opcode::S_LSHR_B32, leaf);
			if (!IsScalarConstantOp(node, Opcode::S_LSHR_B32, 3)) {
				continue;
			}
			// N is the node of the triangle test.
			const auto node_sgpr = ScalarIndex(m_inst[node].src0);
			for (size_t j = leaf; j < tail; j++) {
				if (m_inst[j].opcode != Opcode::IMAGE_BVH_INTERSECT_RAY) {
					continue;
				}
				const auto move = VectorSource(j, m_inst[j].src0, Opcode::V_MOV_B32, leaf);
				geometry_ok = move != npos && ScalarIndex(m_inst[move].src0) == node_sgpr &&
				              ScalarWriter(node, node_sgpr, move) == npos;
				break;
			}
		}
		if (!geometry_ok || !base_ok) {
			return false;
		}
		// M0, M1: the first two dwords of the record.
		size_t m01 = npos;
		for (size_t i = leaf; i < sample && m01 == npos; i++) {
			const auto& load = m_inst[i];
			if (load.opcode == Opcode::S_BUFFER_LOAD_DWORDX2 && load.offset == 0 && IsSgpr(load.src0, material_vsharp) &&
			    ScalarIndex(load.src1) == record_offset && ScalarWriter(i, record_offset, leaf) == scale4 &&
			    ScalarRangeSource(i, material_vsharp, 4, Opcode::S_LOAD_DWORDX4, leaf) ==
			        ScalarRangeSource(m2, material_vsharp, 4, Opcode::S_LOAD_DWORDX4, leaf)) {
				m01 = i;
			}
		}
		if (m01 == npos || m_inst[m01].dst.kind != OperandKind::Sgpr) {
			return false;
		}
		const auto m0_sgpr = m_inst[m01].dst.reg, m1_sgpr = m0_sgpr + 1;
		const auto is_m0 = [&](size_t i, const Operand& operand) {
			return ScalarIndex(operand) == m0_sgpr && ScalarWriter(i, m0_sgpr, leaf) == m01;
		};
		const auto is_m1 = [&](size_t i, const Operand& operand) {
			return ScalarIndex(operand) == m1_sgpr && ScalarWriter(i, m1_sgpr, leaf) == m01;
		};
		// The reference: threshold * (M0 bit 0 ? factor : 1.0), compared with the sample.
		size_t compare = npos;
		for (size_t i = sample + 1; i < tail && compare == npos; i++) {
			if (m_inst[i].opcode == Opcode::V_CMP_GT_F32 && m_inst[i].src0.kind == OperandKind::Vgpr &&
			    m_inst[i].src0.reg == sampled.dst.reg && VectorSource(i, m_inst[i].src0, Opcode::IMAGE_SAMPLE, sample) == sample) {
				compare = i;
			}
		}
		if (compare == npos) {
			return false;
		}
		const auto reference = VectorSource(compare, m_inst[compare].src1, Opcode::V_MUL_F32, leaf);
		if (reference == npos || !Plain(m_inst[reference].src0) || !Plain(m_inst[reference].src1) ||
		    !Plain(m_inst[reference].dst)) {
			return false;
		}
		bool threshold_ok = false, factor_ok = false;
		for (const auto* operand: {&m_inst[reference].src0, &m_inst[reference].src1}) {
			const auto j = ScalarWriter(reference, ScalarIndex(*operand), leaf);
			if (j == npos) {
				continue;
			}
			const auto& inst = m_inst[j];
			if (inst.opcode == Opcode::S_MOV_B32 && inst.src0.kind == OperandKind::LiteralConstant) {
				alpha.threshold = inst.src0.value;
				threshold_ok    = true;
			} else if (inst.opcode == Opcode::S_CSELECT_B32 && inst.src0.kind == OperandKind::Sgpr &&
			           inst.src1.kind == OperandKind::FloatInlineConstant && inst.src1.value == One) {
				// S_BITCMP1_B32 M0, 0 sets SCC.
				size_t bitcmp = j;
				while (bitcmp-- > leaf && m_inst[bitcmp].opcode == Opcode::S_NOP) {
				}
				if (bitcmp > leaf && m_inst[bitcmp].opcode == Opcode::S_BITCMP1_B32 &&
				    IsConstant(m_inst[bitcmp].src1, 0) && is_m0(bitcmp, m_inst[bitcmp].src0)) {
					alpha.factor_sgpr = inst.src0.reg;
					factor_ok         = true;
				}
			}
		}
		if (!threshold_ok || !factor_ok) {
			return false;
		}
		// The level: min(M2 & 15, log2(2 * (spread * t + lod)) + offset).
		const auto level = VectorSource(sample, Operand {.kind = OperandKind::Vgpr, .reg = AddressReg(sampled, 2)},
		                                Opcode::V_MIN_F32, leaf);
		if (level == npos || !Plain(m_inst[level].src0) || !Plain(m_inst[level].src1) || !Plain(m_inst[level].dst)) {
			return false;
		}
		bool max_ok = false, lod_ok = false;
		for (const auto* operand: {&m_inst[level].src0, &m_inst[level].src1}) {
			if (const auto convert = VectorSource(level, *operand, Opcode::V_CVT_F32_U32, leaf); convert != npos) {
				const auto mask = ScalarSource(convert, m_inst[convert].src0, Opcode::S_AND_B32, leaf);
				max_ok = IsScalarConstantOp(mask, Opcode::S_AND_B32, 15) &&
				         ScalarWriter(mask, ScalarIndex(m_inst[mask].src0), leaf) == m2;
				continue;
			}
			const auto offset = VectorSource(level, *operand, Opcode::V_ADD_F32, leaf);
			if (offset == npos || m_inst[offset].src0.kind != OperandKind::LiteralConstant || !Plain(m_inst[offset].src1) ||
			    !Plain(m_inst[offset].dst)) {
				continue;
			}
			const auto log = VectorSource(offset, m_inst[offset].src1, Opcode::V_LOG_F32, leaf);
			if (log == npos || !Plain(m_inst[log].src0) || !Plain(m_inst[log].dst)) {
				continue;
			}
			const auto cone = VectorSource(log, m_inst[log].src0, Opcode::V_MAD_F32, leaf);
			if (cone == npos) {
				continue;
			}
			const auto& mad = m_inst[cone];
			// omod 1: the result times 2.
			if (mad.dst.omod != 1 || mad.dst.clamp || !Plain(mad.src0) || !Plain(mad.src1) || !Plain(mad.src2) ||
			    mad.src0.kind != OperandKind::Sgpr || mad.src1.kind != OperandKind::Vgpr || mad.src1.reg != t_vgpr ||
			    mad.src2.kind != OperandKind::Vgpr) {
				continue;
			}
			alpha.spread_reg = mad.src0.reg;
			alpha.lod_vgpr    = mad.src2.reg;
			alpha.lod_offset  = m_inst[offset].src0.value;
			lod_ok            = true;
		}
		if (!max_ok || !lod_ok) {
			return false;
		}
		// The index buffer: three BUFFER_LOAD_FORMAT_X idxen through V# (M1 >> 8) * 16 of the buffer
		// table; the vertices: three S_BUFFER_LOAD_DWORDX2 through V# (M0 >> 8) * 16.
		const auto table_vsharp = [&](size_t load, bool index_buffer, uint32_t& vsharp) {
			const auto& inst = m_inst[load];
			const auto scale = ScalarSource(load, inst.src1, Opcode::S_LSHL_B32, leaf);
			if (inst.offset != 0 || inst.src0.kind != OperandKind::Sgpr || !IsScalarConstantOp(scale, Opcode::S_LSHL_B32, 4)) {
				return false;
			}
			const auto shift = ScalarSource(scale, m_inst[scale].src0, Opcode::S_LSHR_B32, leaf);
			if (!IsScalarConstantOp(shift, Opcode::S_LSHR_B32, 8) ||
			    !(index_buffer ? is_m1(shift, m_inst[shift].src0) : is_m0(shift, m_inst[shift].src0))) {
				return false;
			}
			vsharp = inst.src0.reg;
			return true;
		};
		uint32_t formatted = 0, uvs = 0, buffer_table = NoRegister;
		size_t   table_load = npos;
		for (size_t i = leaf; i < sample; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::BUFFER_LOAD_FORMAT_X) {
				const auto vsharp = ScalarRangeSource(i, inst.src1.reg, 4, Opcode::S_BUFFER_LOAD_DWORDX4, leaf);
				uint32_t   table  = 0;
				if (!inst.idxen || inst.offen || inst.offset != 0 || inst.src1.kind != OperandKind::Sgpr || vsharp == npos ||
				    !table_vsharp(vsharp, true, table) || (buffer_table != NoRegister && table != buffer_table)) {
					return false;
				}
				buffer_table = table;
				table_load   = vsharp;
				formatted++;
			} else if (inst.opcode == Opcode::S_BUFFER_LOAD_DWORDX2 && i != m01) {
				const auto vsharp = ScalarRangeSource(i, inst.src0.reg, 4, Opcode::S_BUFFER_LOAD_DWORDX4, leaf);
				uint32_t   table  = 0;
				if (inst.src0.kind != OperandKind::Sgpr || vsharp == npos || !table_vsharp(vsharp, false, table) ||
				    (buffer_table != NoRegister && table != buffer_table)) {
					return false;
				}
				buffer_table = table;
				uvs++;
			}
		}
		if (formatted != 3 || uvs != 3 ||
		    !UserVsharp(ScalarRangeSource(table_load, buffer_table, 4, Opcode::S_LOAD_DWORDX4, leaf), tail,
		                alpha.buffer_user, alpha.buffer_offset)) {
			return false;
		}
		// The primitive index (W & 0x3fffffff) * 3, the vertex stride and uv offset (& 0xfc).
		bool primitive_ok = false;
		uint32_t strides  = 0;
		for (size_t i = leaf; i < sample; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::S_ANDN2_B32 && inst.src1.kind == OperandKind::FloatInlineConstant &&
			    inst.src1.value == NoTypeBits && ScalarIndex(inst.src0) == word_sgpr) {
				for (size_t j = i + 1; j < sample && !primitive_ok; j++) {
					primitive_ok = IsScalarConstantOp(j, Opcode::S_MUL_I32, 3) &&
					               ScalarWriter(j, ScalarIndex(m_inst[j].src0), leaf) == i;
				}
			}
			if (IsScalarConstantOp(i, Opcode::S_AND_B32, 0xfc) && (is_m0(i, inst.src0) || is_m1(i, inst.src0))) {
				strides |= is_m0(i, inst.src0) ? 1u : 2u;
			}
		}
		if (!primitive_ok || strides != 3) {
			return false;
		}
		// The registers the native code reads at the site entry keep their value in the traversal.
		for (size_t i = header; i <= back; i++) {
			if (WritesScalar(m_inst[i], alpha.spread_reg) || WritesScalar(m_inst[i], alpha.factor_sgpr) ||
			    WritesVector(m_inst[i], alpha.lod_vgpr)) {
				return false;
			}
		}
		(void)texture_user;
		(void)texture_offset;
		alpha.valid = true;
		return true;
	}

	// A user data descriptor: the 4 SGPRs from `reg` read by instruction `i` come from one
	// S_LOAD_DWORDX4/X8/X16 of an SGPR pair no instruction before `limit` writes.
	bool UserDescriptor(size_t i, uint32_t reg, size_t limit, uint32_t& user, uint32_t& offset) const {
		const auto load = ScalarWriter(i, reg, 0);
		if (load == npos) {
			return false;
		}
		const auto& inst = m_inst[load];
		if ((inst.opcode != Opcode::S_LOAD_DWORDX4 && inst.opcode != Opcode::S_LOAD_DWORDX8 &&
		     inst.opcode != Opcode::S_LOAD_DWORDX16) ||
		    inst.src0.kind != OperandKind::Sgpr || inst.dst.kind != OperandKind::Sgpr || reg < inst.dst.reg ||
		    (reg - inst.dst.reg) % 4 != 0 || reg + 4 > inst.dst.reg + inst.data_dwords ||
		    (inst.src1.kind != OperandKind::Unknown && inst.src1.kind != OperandKind::Null)) {
			return false;
		}
		for (uint32_t k = 1; k < 4; k++) {
			if (ScalarWriter(i, reg + k, 0) != load) {
				return false;
			}
		}
		const auto base = inst.src0.reg;
		for (size_t j = 0; j < limit; j++) {
			if (WritesScalar(m_inst[j], base) || WritesScalar(m_inst[j], base + 1)) {
				return false;
			}
		}
		user   = base;
		offset = inst.offset + (reg - inst.dst.reg) * 4;
		return true;
	}

	// The BLAS of a closest hit site: the SGPR pair whose S_BFE_U64 <t#>, <pair>, 0x280008 builds
	// the T# of an IMAGE_BVH_INTERSECT_RAY of the loop other than the TLAS's.
	bool IsBlasPair(size_t header, size_t back, uint32_t pair, uint32_t tlas_header) const {
		for (size_t i = header; i < back; i++) {
			const auto& test = m_inst[i];
			if (test.opcode != Opcode::IMAGE_BVH_INTERSECT_RAY || test.src1.kind != OperandKind::Sgpr) {
				continue;
			}
			const auto bfe = ScalarSource(i, test.src1, Opcode::S_BFE_U64, header);
			if (bfe != npos && IsConstant(m_inst[bfe].src1, 0x280008) && IsSgpr(m_inst[bfe].src0, pair) &&
			    pair != tlas_header) {
				return true;
			}
		}
		return false;
	}

	// ---- The alpha test of GTA V's closest hit leaf code (RayTracingAlphaLayout::Lane). ----
	//
	// The test of the packet sites (see DeriveAlphaTest) with vector loads in [begin, end): the
	// material dwords m * 3 + k with BUFFER_LOAD_DWORD idxen, the V#s and the T# of the tables in
	// waterfall loops (V_READFIRSTLANE_B32 of the key), the vertices with FLAT loads at the base
	// address of the vertex V# (BUFFER_LOAD_DWORDX2 idxen of the buffer table). The reference
	// and the cone spread are VGPRs.
	bool DeriveLaneAlphaTest(size_t header, size_t back, size_t begin, size_t end, uint32_t t_vgpr,
	                         uint32_t node_vgpr, uint32_t word_vgpr, uint32_t flag_vgpr, uint32_t record_sgpr,
	                         RayTracingAlphaTest& alpha) const {
		const auto vgpr = [](uint32_t reg) {
			Operand operand;
			operand.kind = OperandKind::Vgpr;
			operand.reg  = reg;
			return operand;
		};
		// The writer (with `opcode`) of the value a waterfall key is made of:
		// S_LSHL_B32 <key>, <k>, scale of V_READFIRSTLANE_B32 <k>, <v> of V_LSHRREV_B32 <v>, shift, <value>.
		const auto waterfall_value = [&](size_t i, const Operand& key, uint32_t scale, uint32_t shift,
		                                 Opcode opcode) -> size_t {
			const auto lsl = ScalarSource(i, key, Opcode::S_LSHL_B32, begin);
			if (!IsScalarConstantOp(lsl, Opcode::S_LSHL_B32, scale)) {
				return npos;
			}
			const auto first = ScalarSource(lsl, m_inst[lsl].src0, Opcode::V_READFIRSTLANE_B32, begin);
			if (first == npos) {
				return npos;
			}
			const auto lsr = VectorSource(first, m_inst[first].src0, Opcode::V_LSHRREV_B32, begin);
			if (lsr == npos || !IsConstant(m_inst[lsr].src0, shift)) {
				return npos;
			}
			return VectorSource(lsr, m_inst[lsr].src1, opcode, begin);
		};
		// The sample, the comparison with the reference and the flag it sets.
		size_t sample = npos;
		for (size_t i = begin; i < end; i++) {
			if (m_inst[i].opcode == Opcode::IMAGE_SAMPLE) {
				if (sample != npos) {
					return false;
				}
				sample = i;
			}
		}
		if (sample == npos) {
			return false;
		}
		const auto& sampled = m_inst[sample];
		if (sampled.dmask != 8 || sampled.image_sample_flags != ImageSampleFlagLod ||
		    sampled.image_dimension != ImageDimension::Dim2D || sampled.src1.kind != OperandKind::Sgpr ||
		    sampled.dst.kind != OperandKind::Vgpr || sampled.image_nsa_dwords != 0 ||
		    sampled.src0.kind != OperandKind::Vgpr) {
			return false;
		}
		size_t compare = npos;
		for (size_t i = sample + 1; i < end && compare == npos; i++) {
			if (m_inst[i].opcode == Opcode::V_CMP_GT_F32 && IsVccLo(m_inst[i].dst) &&
			    VectorSource(i, m_inst[i].src0, Opcode::IMAGE_SAMPLE, sample) == sample) {
				compare = i;
			}
		}
		if (compare == npos || m_inst[compare].src1.kind != OperandKind::Vgpr ||
		    VectorSource(end, vgpr(flag_vgpr), Opcode::V_CNDMASK_B32, compare) == npos) {
			return false;
		}
		alpha.reference_vgpr = m_inst[compare].src1.reg;
		// The material dwords: BUFFER_LOAD_DWORD idxen of one V# at m * 3 + k, with m * 3 from
		// V_LSHL_ADD_U32 <m3>, <m>, 1, <m> and V_ADD_NC_U32 <i>, k, <m3>.
		std::array<size_t, 3> material {npos, npos, npos};
		size_t                scale3          = npos;
		uint32_t              material_vsharp = NoRegister;
		for (size_t i = begin; i < end; i++) {
			const auto& load = m_inst[i];
			if (load.opcode != Opcode::BUFFER_LOAD_DWORD || !load.idxen || load.offen || load.offset != 0 ||
			    load.src1.kind != OperandKind::Sgpr || !IsConstant(load.src2, 0) ||
			    (material_vsharp != NoRegister && load.src1.reg != material_vsharp)) {
				continue;
			}
			uint32_t k     = 0;
			auto     three = VectorSource(i, load.src0, Opcode::V_LSHL_ADD_U32, begin);
			if (three == npos) {
				const auto add = VectorSource(i, load.src0, Opcode::V_ADD_NC_U32, begin);
				if (add == npos || m_inst[add].src0.kind != OperandKind::IntegerInlineConstant ||
				    (m_inst[add].src0.value != 1 && m_inst[add].src0.value != 2)) {
					return false;
				}
				k     = m_inst[add].src0.value;
				three = VectorSource(add, m_inst[add].src1, Opcode::V_LSHL_ADD_U32, begin);
			}
			const auto& mad3 = three != npos ? m_inst[three] : load;
			if (three == npos || (scale3 != npos && three != scale3) || material[k] != npos ||
			    mad3.src0.kind != OperandKind::Vgpr || !SameOperand(mad3.src0, mad3.src2) ||
			    !IsConstant(mad3.src1, 1)) {
				return false;
			}
			scale3          = three;
			material[k]     = i;
			material_vsharp = load.src1.reg;
		}
		if (material[0] == npos || material[1] == npos || material[2] == npos ||
		    !UserDescriptor(material[0], material_vsharp, header, alpha.material_user, alpha.material_offset)) {
			return false;
		}
		// m = B[B[0x28] + ((N >> 3) - B[0x7c]) * 4] + R[0x78] of the triangle node N.
		const auto sum = VectorSource(scale3, m_inst[scale3].src0, Opcode::V_ADD_NC_U32, begin);
		if (sum == npos || ConstantFieldLoad(sum, m_inst[sum].src0, record_sgpr, 0x78, begin) == npos) {
			return false;
		}
		const auto geometry = VectorSource(sum, m_inst[sum].src1, Opcode::FLAT_LOAD_DWORD, begin);
		if (geometry == npos || m_inst[geometry].memory_segment != 2 || m_inst[geometry].offset != 0 ||
		    m_inst[geometry].src1.kind != OperandKind::Sgpr) {
			return false;
		}
		const auto blas = m_inst[geometry].src1.reg;
		if (!IsBlasPair(header, back, blas, NoRegister)) {
			return false;
		}
		const auto table = VectorSource(geometry, m_inst[geometry].src0, Opcode::V_ADD_NC_U32, begin);
		if (table == npos || ConstantFieldLoad(table, m_inst[table].src0, blas, 0x28, begin) == npos) {
			return false;
		}
		const auto local = VectorSource(table, m_inst[table].src1, Opcode::V_ADD_LSHL_U32, begin);
		if (local == npos || !IsConstant(m_inst[local].src2, 2)) {
			return false;
		}
		const auto negate = ScalarSource(local, m_inst[local].src0, Opcode::S_SUB_I32, begin);
		if (negate == npos || !IsConstant(m_inst[negate].src0, 0) ||
		    ConstantFieldLoad(negate, m_inst[negate].src1, blas, 0x7c, begin) == npos) {
			return false;
		}
		const auto node = VectorSource(local, m_inst[local].src1, Opcode::V_LSHRREV_B32, begin);
		if (node == npos || !IsConstant(m_inst[node].src0, 3) || !SameOperand(m_inst[node].src1, vgpr(node_vgpr))) {
			return false;
		}
		// The T#: (M2 >> 4) * 32 of the texture table.
		const auto tsharp = ScalarRangeSource(sample, sampled.src1.reg, 8, Opcode::S_BUFFER_LOAD_DWORDX8, begin);
		uint32_t   texture_user = 0, texture_offset = 0;
		if (tsharp == npos || m_inst[tsharp].offset != 0 || m_inst[tsharp].src0.kind != OperandKind::Sgpr ||
		    waterfall_value(tsharp, m_inst[tsharp].src1, 5, 4, Opcode::BUFFER_LOAD_DWORD) != material[2] ||
		    !UserDescriptor(tsharp, m_inst[tsharp].src0.reg, header, texture_user, texture_offset)) {
			return false;
		}
		// The indices: BUFFER_LOAD_FORMAT_X idxen at p * 3 + k, p = W & 0x3fffffff, through V#
		// (M1 >> 8) * 16 of the buffer table.
		uint32_t buffer_table = NoRegister, indices = 0;
		size_t   triangle     = npos;
		for (size_t i = begin; i < end; i++) {
			const auto& load = m_inst[i];
			if (load.opcode != Opcode::BUFFER_LOAD_FORMAT_X) {
				continue;
			}
			const auto vsharp = ScalarRangeSource(i, load.src1.reg, 4, Opcode::S_BUFFER_LOAD_DWORDX4, begin);
			if (!load.idxen || load.offen || load.offset != 0 || load.src1.kind != OperandKind::Sgpr ||
			    vsharp == npos || m_inst[vsharp].offset != 0 || m_inst[vsharp].src0.kind != OperandKind::Sgpr ||
			    waterfall_value(vsharp, m_inst[vsharp].src1, 4, 8, Opcode::BUFFER_LOAD_DWORD) != material[1] ||
			    (buffer_table != NoRegister && m_inst[vsharp].src0.reg != buffer_table)) {
				return false;
			}
			buffer_table = m_inst[vsharp].src0.reg;
			auto three   = VectorSource(i, load.src0, Opcode::V_LSHL_ADD_U32, begin);
			if (three == npos) {
				const auto add = VectorSource(i, load.src0, Opcode::V_ADD_NC_U32, begin);
				three = add != npos && m_inst[add].src0.kind == OperandKind::IntegerInlineConstant &&
				                (m_inst[add].src0.value == 1 || m_inst[add].src0.value == 2)
				            ? VectorSource(add, m_inst[add].src1, Opcode::V_LSHL_ADD_U32, begin)
				            : npos;
			}
			if (three == npos || (triangle != npos && three != triangle)) {
				return false;
			}
			triangle = three;
			indices++;
		}
		if (indices != 3 || !SameOperand(m_inst[triangle].src0, m_inst[triangle].src2) ||
		    !IsConstant(m_inst[triangle].src1, 1)) {
			return false;
		}
		const auto primitive = VectorSource(triangle, m_inst[triangle].src0, Opcode::V_AND_B32, begin);
		if (primitive == npos || !IsConstant(m_inst[primitive].src0, 0x3fffffff) ||
		    !SameOperand(m_inst[primitive].src1, vgpr(word_vgpr)) ||
		    !UserDescriptor(triangle, buffer_table, header, alpha.buffer_user, alpha.buffer_offset)) {
			return false;
		}
		// The vertices: the base address of V# M0 >> 8 of the buffer table (BUFFER_LOAD_DWORDX2
		// idxen) plus 32-bit offsets, three times two dwords; the stride and the uv offset (& 0xfc).
		size_t vertex_vsharp = npos;
		for (size_t i = begin; i < end && vertex_vsharp == npos; i++) {
			const auto& load = m_inst[i];
			if (load.opcode == Opcode::BUFFER_LOAD_DWORDX2 && load.idxen && !load.offen && load.offset == 0 &&
			    IsSgpr(load.src1, buffer_table) && load.dst.kind == OperandKind::Vgpr) {
				const auto shift = VectorSource(i, load.src0, Opcode::V_LSHRREV_B32, begin);
				if (shift == npos || !IsConstant(m_inst[shift].src0, 8) ||
				    VectorSource(shift, m_inst[shift].src1, Opcode::BUFFER_LOAD_DWORD, begin) != material[0]) {
					return false;
				}
				vertex_vsharp = i;
			}
		}
		if (vertex_vsharp == npos) {
			return false;
		}
		uint32_t vertices = 0, strides = 0;
		for (size_t i = begin; i < end; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::FLAT_LOAD_DWORD && i != geometry) {
				const auto add = VectorSource(i, inst.src0, Opcode::V_ADD_I32, begin);
				if (inst.memory_segment != 2 || inst.offset != 0 || inst.src1.kind != OperandKind::Vgpr || add == npos ||
				    (VectorSource(add, m_inst[add].src0, Opcode::BUFFER_LOAD_DWORDX2, begin) != vertex_vsharp &&
				     VectorSource(add, m_inst[add].src1, Opcode::BUFFER_LOAD_DWORDX2, begin) != vertex_vsharp)) {
					return false;
				}
				vertices++;
			} else if (inst.opcode == Opcode::V_AND_B32 && IsConstant(inst.src0, 0xfc)) {
				const auto source = VectorSource(i, inst.src1, Opcode::BUFFER_LOAD_DWORD, begin);
				strides |= source == material[0] ? 1u : source == material[1] ? 2u : 4u;
			}
		}
		if (vertices != 6 || strides != 3) {
			return false;
		}
		// The level: min(M2 & 15, log2(2 * (spread * t + lod)) + offset).
		const auto level = VectorSource(sample, vgpr(sampled.src0.reg + 2), Opcode::V_MIN_F32, begin);
		if (level == npos || !Plain(m_inst[level].src0) || !Plain(m_inst[level].src1) || !Plain(m_inst[level].dst)) {
			return false;
		}
		bool max_ok = false, lod_ok = false;
		for (const auto* operand: {&m_inst[level].src0, &m_inst[level].src1}) {
			if (const auto convert = VectorSource(level, *operand, Opcode::V_CVT_F32_U32, begin); convert != npos) {
				const auto mask = VectorSource(convert, m_inst[convert].src0, Opcode::V_AND_B32, begin);
				max_ok          = mask != npos && IsConstant(m_inst[mask].src0, 15) &&
				         VectorSource(mask, m_inst[mask].src1, Opcode::BUFFER_LOAD_DWORD, begin) == material[2];
				continue;
			}
			const auto offset = VectorSource(level, *operand, Opcode::V_ADD_F32, begin);
			if (offset == npos || m_inst[offset].src0.kind != OperandKind::LiteralConstant ||
			    !Plain(m_inst[offset].src1) || !Plain(m_inst[offset].dst)) {
				continue;
			}
			const auto log = VectorSource(offset, m_inst[offset].src1, Opcode::V_LOG_F32, begin);
			if (log == npos || !Plain(m_inst[log].src0) || !Plain(m_inst[log].dst)) {
				continue;
			}
			const auto cone = VectorSource(log, m_inst[log].src0, Opcode::V_MAD_F32, begin);
			if (cone == npos) {
				continue;
			}
			const auto& mad = m_inst[cone];
			// omod 1: the result times 2; t times the spread, either order.
			if (mad.dst.omod != 1 || mad.dst.clamp || !Plain(mad.src0) || !Plain(mad.src1) || !Plain(mad.src2) ||
			    mad.src0.kind != OperandKind::Vgpr || mad.src1.kind != OperandKind::Vgpr ||
			    mad.src2.kind != OperandKind::Vgpr || (mad.src0.reg != t_vgpr && mad.src1.reg != t_vgpr)) {
				continue;
			}
			alpha.spread_reg = mad.src0.reg == t_vgpr ? mad.src1.reg : mad.src0.reg;
			alpha.lod_vgpr   = mad.src2.reg;
			alpha.lod_offset = m_inst[offset].src0.value;
			lod_ok           = true;
		}
		if (!max_ok || !lod_ok) {
			return false;
		}
		// The registers the native code reads at the site entry keep their value in the traversal.
		for (size_t i = header; i <= back; i++) {
			if (WritesVector(m_inst[i], alpha.spread_reg) || WritesVector(m_inst[i], alpha.lod_vgpr) ||
			    WritesVector(m_inst[i], alpha.reference_vgpr)) {
				return false;
			}
		}
		(void)texture_user;
		(void)texture_offset;
		alpha.layout = RayTracingAlphaLayout::Lane;
		alpha.valid  = true;
		return true;
	}

	bool DeriveClosest(size_t header, size_t back, RayTracingSite& site) const {
		site.kind           = RayTracingKind::ClosestOpaque;
		site.entry_pc       = m_inst[header - 1].pc;
		site.loop_pc        = m_inst[header].pc;
		site.cull_mask      = NoRegister;
		site.alpha_off_sgpr = NoAlphaOff;
		const auto le       = header + 3;
		if (le >= back || m_inst[le].opcode != Opcode::S_CMP_LE_U32 || !IsConstant(m_inst[le].src1, 2)) {
			return false;
		}
		// The box test of the TLAS: the intersection whose T# is built from the header before the
		// loop.
		size_t box = npos;
		for (size_t i = header; i < back && box == npos; i++) {
			if (m_inst[i].opcode != Opcode::IMAGE_BVH_INTERSECT_RAY) {
				continue;
			}
			RayTracingSite probe;
			if (BoxTest(header, i, probe)) {
				box = i;
				site.t_max       = probe.t_max;
				site.origin[0]   = probe.origin[0];
				site.origin[1]   = probe.origin[1];
				site.origin[2]   = probe.origin[2];
				site.direction[0] = probe.direction[0];
				site.direction[1] = probe.direction[1];
				site.direction[2] = probe.direction[2];
				site.header_sgpr = probe.header_sgpr;
			}
		}
		if (box == npos) {
			return false;
		}
		// The exit: the branch that skips the loop's backward branch.
		size_t exit = npos;
		for (size_t i = back; i-- > header && i + 4 > back && exit == npos;) {
			if (m_inst[i].opcode == Opcode::S_CBRANCH_SCC0 && m_inst[i].branch_target == EndPc(m_inst[back])) {
				exit = i;
			}
		}
		if (exit == npos) {
			return false;
		}
		site.exit_pc = m_inst[exit].branch_target;
		// The nearest hit: the distance test, then the moves of the hit, right after it (every
		// triangle opaque) or after the alpha test of the triangles not marked opaque.
		size_t compare = npos, accept = npos;
		for (size_t i = header; i + 3 < back && accept == npos; i++) {
			const auto& cmp = m_inst[i];
			if (cmp.opcode != Opcode::V_CMP_GT_F32 || !IsVccLo(cmp.dst) ||
			    cmp.src1.kind != OperandKind::Vgpr || cmp.src1.reg != site.t_max) {
				continue;
			}
			const auto save = Next(i, Opcode::S_AND_SAVEEXEC_B64, i + 4);
			if (save != npos && m_inst[save + 1].opcode == Opcode::S_CBRANCH_EXECZ) {
				compare = i;
				accept  = save + 2;
			}
		}
		if (accept == npos || m_inst[compare].src0.kind != OperandKind::Vgpr) {
			return false;
		}
		size_t moves_end = accept;
		while (moves_end < back && m_inst[moves_end].opcode != Opcode::S_MOV_B64) {
			moves_end++;
		}
		site.alpha          = {};
		uint32_t entry_base = NoRegister;
		if (!HitMoves(header, back, accept, moves_end, {NoRegister, NoRegister}, site, entry_base) &&
		    !ClosestAlphaHit(header, back, compare, accept, site)) {
			return false;
		}
		// The instance mask: the guest skips the instances whose mask (record +0x3c bits 24-31)
		// has none of the bits of S_AND_B32 <s>, <mask dword>, <bits << 24>; S_CBRANCH_SCC0.
		for (size_t i = header; i + 1 < back; i++) {
			const auto& test = m_inst[i];
			if (test.opcode != Opcode::S_AND_B32 || test.src0.kind != OperandKind::Sgpr ||
			    test.src1.kind != OperandKind::LiteralConstant || (test.src1.value & 0xffffffu) != 0 ||
			    m_inst[NextReal(i)].opcode != Opcode::S_CBRANCH_SCC0) {
				continue;
			}
			const auto load = ScalarWriter(i, test.src0.reg, header);
			if (load != npos && m_inst[load].family == Family::SMEM && IsSgpr(m_inst[load].src0, site.instance_sgpr) &&
			    m_inst[load].dst.kind == OperandKind::Sgpr && test.src0.reg >= m_inst[load].dst.reg &&
			    m_inst[load].offset + (test.src0.reg - m_inst[load].dst.reg) * 4u == 0x3cu) {
				site.cull_mask_constant = test.src1.value >> 24u;
				break;
			}
		}
		// The pair is free at the exit (the selected key goes there).
		site.mask_sgpr = site.instance_sgpr;
		return true;
	}

	// The registers a closest hit site's leaf code writes for an accepted hit, from the V_MOV_B32s
	// in [begin, end): the distance (into t max), the instance record pair (SGPRs), the
	// barycentrics (copies of the products `bary`, or V_MUL_F32s among the moves) and the two words
	// of the list entry (BUFFER_LOAD_DWORDX2 to `entry_base`: triangle node, primitive word).
	bool HitMoves(size_t header, size_t back, size_t begin, size_t end, const std::array<uint32_t, 2>& bary,
	              RayTracingSite& site, uint32_t& entry_base) const {
		uint32_t muls = 0, moves = 0;
		uint32_t vgpr_moves[2] {NoRegister, NoRegister}, vgpr_sources[2] {};
		site.result_t = site.result_instance = site.result_u = site.result_v = NoRegister;
		for (size_t i = begin; i < end; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::V_MUL_F32 && bary[0] == NoRegister && muls < 2) {
				(muls++ == 0 ? site.result_u : site.result_v) = inst.dst.reg;
			} else if (inst.opcode == Opcode::V_MOV_B32 && inst.dst.kind == OperandKind::Vgpr) {
				if (inst.dst.reg == site.t_max && inst.src0.kind == OperandKind::Vgpr) {
					site.result_t = site.t_max;
				} else if (inst.src0.kind == OperandKind::Sgpr && site.result_instance == NoRegister) {
					site.result_instance = inst.dst.reg;
					site.instance_sgpr   = inst.src0.reg;
				} else if (inst.src0.kind == OperandKind::Sgpr && inst.dst.reg == site.result_instance + 1 &&
				           inst.src0.reg == site.instance_sgpr + 1) {
				} else if (inst.src0.kind == OperandKind::Vgpr && bary[0] != NoRegister &&
				           (inst.src0.reg == bary[0] || inst.src0.reg == bary[1])) {
					(inst.src0.reg == bary[0] ? site.result_u : site.result_v) = inst.dst.reg;
				} else if (inst.src0.kind == OperandKind::Vgpr && moves < 2) {
					vgpr_moves[moves]     = inst.dst.reg;
					vgpr_sources[moves++] = inst.src0.reg;
				} else {
					return false;
				}
			} else {
				return false;
			}
		}
		if (moves != 2 || site.result_t == NoRegister || site.result_instance == NoRegister ||
		    site.result_u == NoRegister || site.result_v == NoRegister) {
			return false;
		}
		// The two words of the list entry: the triangle node, then the primitive word.
		const auto base = std::min(vgpr_sources[0], vgpr_sources[1]);
		if (std::max(vgpr_sources[0], vgpr_sources[1]) != base + 1) {
			return false;
		}
		bool loaded = false;
		for (size_t j = header; j < back && !loaded; j++) {
			loaded = m_inst[j].opcode == Opcode::BUFFER_LOAD_DWORDX2 && m_inst[j].dst.kind == OperandKind::Vgpr &&
			         m_inst[j].dst.reg == base;
		}
		if (!loaded) {
			return false;
		}
		site.result_node = vgpr_sources[0] == base ? vgpr_moves[0] : vgpr_moves[1];
		site.result_word = vgpr_sources[0] == base ? vgpr_moves[1] : vgpr_moves[0];
		entry_base       = base;
		return true;
	}

	// The nearest hit of a closest hit site whose leaf code alpha tests the triangles not marked
	// opaque (primitive word bit 30 clear). After the distance test:
	//   V_MUL_F32 x2 (barycentrics), V_LSHLREV_B32 <x>, 1, <word>, V_MOV_B32 <flag>, 1,
	//   S_MOV_B64 <saved>, exec; V_CMPX_LE_I32 0, <x>; S_CBRANCH_EXECZ <join>
	//   the alpha test, which sets the flag
	//   join: S_MOV_B64 exec, <saved>; V_CMPX_NE_U32 0, <flag>; S_CBRANCH_EXECZ <end>
	//   the V_MOV_B32s of the hit
	bool ClosestAlphaHit(size_t header, size_t back, size_t compare, size_t accept, RayTracingSite& site) const {
		size_t split = npos;
		for (size_t i = accept; i < back && split == npos && !IsDirectBranch(m_inst[i].opcode); i++) {
			if (m_inst[i].opcode == Opcode::V_CMPX_LE_I32) {
				split = i;
			}
		}
		if (split == npos || split == accept || split + 2 >= back) {
			return false;
		}
		const auto& opaque_test = m_inst[split];
		const auto  shift       = VectorSource(split, opaque_test.src1, Opcode::V_LSHLREV_B32, accept);
		if (!IsConstant(opaque_test.src0, 0) || shift == npos || !IsConstant(m_inst[shift].src0, 1) ||
		    m_inst[shift].src1.kind != OperandKind::Vgpr) {
			return false;
		}
		const auto  word_vgpr = m_inst[shift].src1.reg;
		const auto& saved     = m_inst[split - 1];
		if (saved.opcode != Opcode::S_MOV_B64 || saved.src0.kind != OperandKind::ExecLo ||
		    saved.dst.kind != OperandKind::Sgpr || m_inst[split + 1].opcode != Opcode::S_CBRANCH_EXECZ) {
			return false;
		}
		const auto join = IndexOf(m_inst[split + 1].branch_target);
		if (join == npos || join <= split + 2 || join + 3 >= back) {
			return false;
		}
		const auto& restore   = m_inst[join];
		const auto& flag_test = m_inst[join + 1];
		if (restore.opcode != Opcode::S_MOV_B64 || restore.dst.kind != OperandKind::ExecLo ||
		    !IsSgpr(restore.src0, saved.dst.reg) || flag_test.opcode != Opcode::V_CMPX_NE_U32 ||
		    !IsConstant(flag_test.src0, 0) || flag_test.src1.kind != OperandKind::Vgpr ||
		    m_inst[join + 2].opcode != Opcode::S_CBRANCH_EXECZ) {
			return false;
		}
		const auto end = IndexOf(m_inst[join + 2].branch_target);
		if (end == npos || end <= join + 3 || end > back) {
			return false;
		}
		// Opaque triangles keep the flag at 1.
		const auto opaque = VectorSource(split, flag_test.src1, Opcode::V_MOV_B32, accept);
		if (opaque == npos || !IsConstant(m_inst[opaque].src0, 1)) {
			return false;
		}
		std::array<uint32_t, 2> bary {NoRegister, NoRegister};
		uint32_t                muls = 0;
		for (size_t i = accept; i < split; i++) {
			if (m_inst[i].opcode == Opcode::V_MUL_F32 && m_inst[i].dst.kind == OperandKind::Vgpr && muls < 2) {
				bary[muls++] = m_inst[i].dst.reg;
			}
		}
		uint32_t entry_base = NoRegister;
		if (muls != 2 || !HitMoves(header, back, join + 3, end, bary, site, entry_base) ||
		    word_vgpr != entry_base + 1) {
			return false;
		}
		RayTracingAlphaTest alpha;
		if (!DeriveLaneAlphaTest(header, back, split + 2, join, m_inst[compare].src0.reg, entry_base, word_vgpr,
		                         flag_test.src1.reg, site.instance_sgpr, alpha)) {
			return false;
		}
		site.alpha = alpha;
		return true;
	}

	// The TLAS header pair: a S_LOAD_DWORDX2 from a pair nothing wrote before (user data).
	bool FindTlasSource(size_t header, uint32_t header_sgpr, uint32_t& user_sgpr, uint32_t& offset) const {
		for (size_t i = header; i-- > 0;) {
			const auto& load = m_inst[i];
			if (!IsSgpr(load.dst, header_sgpr)) {
				continue;
			}
			// An immediate offset only (no offset register).
			if (load.opcode != Opcode::S_LOAD_DWORDX2 || load.src0.kind != OperandKind::Sgpr ||
			    (load.src1.kind != OperandKind::Unknown && load.src1.kind != OperandKind::Null)) {
				return false;
			}
			const auto base = load.src0.reg;
			for (size_t j = 0; j < i; j++) {
				const auto& dst = m_inst[j].dst;
				if (dst.kind == OperandKind::Sgpr && (dst.reg == base || dst.reg == base + 1 ||
				                                      (dst.reg + 1 == base && WritesPair(m_inst[j])))) {
					return false;
				}
			}
			user_sgpr = base;
			offset    = static_cast<uint32_t>(load.offset);
			return true;
		}
		return false;
	}

	static bool WritesPair(const Instruction& inst) {
		switch (inst.opcode) {
			case Opcode::S_MOV_B64:
			case Opcode::S_LOAD_DWORDX2:
			case Opcode::S_AND_B64:
			case Opcode::S_AND_SAVEEXEC_B64: return true;
			default: return false;
		}
	}

	// The root: the value the node register gets last before the loop.
	bool FindRoot(size_t header, const Operand& node, uint32_t& root) const {
		for (size_t i = header; i-- > 0;) {
			const auto& inst = m_inst[i];
			if (!SameOperand(inst.dst, node)) {
				continue;
			}
			if (inst.opcode != Opcode::S_MOV_B32 || (inst.src0.kind != OperandKind::IntegerInlineConstant &&
			                                         inst.src0.kind != OperandKind::LiteralConstant)) {
				return false;
			}
			root = inst.src0.value;
			return true;
		}
		return false;
	}

	const std::vector<Instruction>& m_inst;
};

} // namespace

bool DeriveRayTracingPatch(const Program& program, RayTracingPatch& patch) {
	if (program.instructions.empty()) {
		return false;
	}
	SiteFinder finder(program);
	if (!finder.Find(patch)) {
		return false;
	}
	for (const auto& site: patch.sites) {
		if (!CheckSite(program, site)) {
			return false;
		}
	}
	return true;
}

bool ApplyRayTracingPatch(const RayTracingPatch& patch, Program& program) {
	if (program.instructions.empty()) {
		return false;
	}
	const auto fail = [&](uint32_t pc, const char* reason) {
		LOGF("RayTracingPatch: site at 0x%x: %s; not patched\n", pc, reason);
		return false;
	};
	for (const auto& site: patch.sites) {
		if (!CheckSite(program, site)) {
			return fail(site.entry_pc, "the code does not match");
		}
	}
	// The native code goes right after each entry instruction, in program order, so the
	// structured control flow sees it as a region of the shader; the guest instructions move.
	Layout layout;
	for (const auto& inst: program.instructions) {
		layout.AddGuest(inst);
		for (uint32_t index = 0; index < patch.sites.size(); index++) {
			if (patch.sites[index].entry_pc == inst.pc &&
			    !AddNativeTraversal(layout, program, patch.sites[index], index)) {
				return fail(inst.pc, "the instance setup or the leaf code cannot be copied");
			}
		}
	}
	std::vector<Instruction> patched;
	if (!layout.Finish(program.instructions.front().pc, EndPc(program.instructions.back()), patched)) {
		return fail(0, "a branch target is unknown");
	}
	program.instructions       = std::move(patched);
	program.ray_tracing_source = {.user_sgpr     = patch.user_sgpr,
	                              .header_offset = patch.header_offset,
	                              .root          = patch.root,
	                              .valid         = true};
	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler::Decoder
