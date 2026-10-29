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

// The native traversal of a closest hit site: the hardware finds the hit (every triangle opaque,
// both facings); no guest code runs for it.
void AddClosestTraversal(Layout& layout, const RayTracingSite& site, uint32_t index) {
	auto& init = layout.AddPseudo(Opcode::RT_INIT, index);
	for (uint32_t i = 0; i < 3; i++) {
		init.image_nsa_addr[i]     = site.origin[i];
		init.image_nsa_addr[3 + i] = site.direction[i];
	}
	init.image_nsa_addr[6] = site.t_max;
	init.image_nsa_addr[7] = site.cull_mask;
	init.src0              = Sgpr(site.header_sgpr);
	init.src1              = Constant(1);
	init.dmask             = 1; // Both facings.
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
	auto& init = layout.AddPseudo(Opcode::RT_INIT, index);
	for (uint32_t i = 0; i < 3; i++) {
		init.image_nsa_addr[i]     = site.origin[i];
		init.image_nsa_addr[3 + i] = site.direction[i];
	}
	init.image_nsa_addr[6] = site.t_max;
	init.image_nsa_addr[7] = site.cull_mask;
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
		return site.cull_mask != NoRegister;
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
		// The nearest hit.
		size_t accept = npos;
		for (size_t i = header; i + 3 < back && accept == npos; i++) {
			const auto& cmp = m_inst[i];
			if (cmp.opcode != Opcode::V_CMP_GT_F32 || !IsVccLo(cmp.dst) ||
			    cmp.src1.kind != OperandKind::Vgpr || cmp.src1.reg != site.t_max) {
				continue;
			}
			const auto save = Next(i, Opcode::S_AND_SAVEEXEC_B64, i + 4);
			if (save != npos && m_inst[save + 1].opcode == Opcode::S_CBRANCH_EXECZ) {
				accept = save + 2;
			}
		}
		if (accept == npos) {
			return false;
		}
		uint32_t muls = 0;
		uint32_t vgpr_moves[2] {NoRegister, NoRegister}, vgpr_sources[2] {};
		uint32_t moves = 0;
		site.result_t = site.result_instance = NoRegister;
		size_t i = accept;
		for (; i < back && m_inst[i].opcode != Opcode::S_MOV_B64; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::V_MUL_F32 && muls < 2) {
				(muls++ == 0 ? site.result_u : site.result_v) = inst.dst.reg;
			} else if (inst.opcode == Opcode::V_MOV_B32 && inst.dst.kind == OperandKind::Vgpr) {
				if (inst.dst.reg == site.t_max && inst.src0.kind == OperandKind::Vgpr) {
					site.result_t = site.t_max;
				} else if (inst.src0.kind == OperandKind::Sgpr && site.result_instance == NoRegister) {
					site.result_instance = inst.dst.reg;
					site.instance_sgpr   = inst.src0.reg;
				} else if (inst.src0.kind == OperandKind::Sgpr && inst.dst.reg == site.result_instance + 1 &&
				           inst.src0.reg == site.instance_sgpr + 1) {
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
		if (muls != 2 || moves != 2 || site.result_t == NoRegister || site.result_instance == NoRegister) {
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
		// The pair is free at the exit (the selected key goes there).
		site.mask_sgpr = site.instance_sgpr;
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
