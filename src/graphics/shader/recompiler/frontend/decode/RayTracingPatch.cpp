#include "graphics/shader/recompiler/frontend/decode/RayTracingPatch.h"

#include "common/logging/log.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Decoder {

namespace {

// GTA V (PPSA04263) RT shadows: two traversals of the TLAS at [s[4:5] + 16].
constexpr std::array<RayTracingSite, 2> ShadowSites {{
    {
        .entry_pc        = 0xa20,
        .loop_pc         = 0xa28,
        .origin          = {20, 19, 18},
        .direction       = {14, 16, 15},
        .t_max           = 21,
        .cull_mask       = 29,
        .header_sgpr     = 2,
        .alpha_off_sgpr  = 51,
        .result_t        = 28,
        .result_word     = 27,
        .result_instance = 10,
        .setup_pc        = 0xb4c,
        .instance_sgpr   = 26,
        .cull_exit_pc    = 0x10f8,
        .blas_head_pc    = 0xc94,
        .leaf_pc         = 0xde4,
        .leaf_sgpr       = 24,
        .mask_sgpr       = 88,
        .saved_exec_sgpr = 90,
        .tail_pc         = 0x10dc,
        .after_leaf_pc   = 0x10ec,
        .exit_pc         = 0x1138,
    },
    {
        .entry_pc        = 0x1ad4,
        .loop_pc         = 0x1adc,
        .origin          = {17, 16, 18},
        .direction       = {10, 15, 11},
        .t_max           = 19,
        .cull_mask       = 27,
        .header_sgpr     = 10,
        .alpha_off_sgpr  = 76,
        .result_t        = 26,
        .result_word     = 25,
        .result_instance = 8,
        .setup_pc        = 0x1c00,
        .instance_sgpr   = 12,
        .cull_exit_pc    = 0x2194,
        .blas_head_pc    = 0x1d40,
        .leaf_pc         = 0x1e90,
        .leaf_sgpr       = 8,
        .mask_sgpr       = 66,
        .saved_exec_sgpr = 70,
        .tail_pc         = 0x2180,
        .after_leaf_pc   = 0,
        .exit_pc         = 0x21d4,
    },
}};

// GTA V (PPSA04263) RT shadows of the second shadow pass (6db33a2bcb077bc3): the same two
// traversals of the TLAS at [s[4:5] + 16], over other registers. The leaf code decides per lane
// (v9, v13) whether triangles that are not opaque count; there is no switch for the alpha test.
constexpr std::array<RayTracingSite, 2> SecondShadowSites {{
    {
        .entry_pc        = 0x16e8,
        .loop_pc         = 0x16ec,
        .origin          = {44, 41, 40},
        .direction       = {37, 39, 38},
        .t_max           = 35,
        .cull_mask       = 33,
        .header_sgpr     = 52,
        .alpha_off_sgpr  = NoAlphaOff,
        .result_t        = 28,
        .result_word     = 32,
        .result_instance = 6,
        .setup_pc        = 0x1810,
        .instance_sgpr   = 16,
        .cull_exit_pc    = 0x1c1c,
        .blas_head_pc    = 0x1970,
        .leaf_pc         = 0x1ac0,
        .leaf_sgpr       = 30,
        .mask_sgpr       = 64,
        .saved_exec_sgpr = 28,
        .tail_pc         = 0x1c08,
        .after_leaf_pc   = 0,
        .exit_pc         = 0x1c5c,
    },
    {
        .entry_pc        = 0x2438,
        .loop_pc         = 0x243c,
        .origin          = {44, 45, 46},
        .direction       = {47, 48, 49},
        .t_max           = 43,
        .cull_mask       = 16,
        .header_sgpr     = 52,
        .alpha_off_sgpr  = NoAlphaOff,
        .result_t        = 14,
        .result_word     = 15,
        .result_instance = 6,
        .setup_pc        = 0x2550,
        .instance_sgpr   = 8,
        .cull_exit_pc    = 0x295c,
        .blas_head_pc    = 0x26b0,
        .leaf_pc         = 0x2800,
        .leaf_sgpr       = 22,
        .mask_sgpr       = 36,
        .saved_exec_sgpr = 20,
        .tail_pc         = 0x2948,
        .after_leaf_pc   = 0,
        .exit_pc         = 0x299c,
    },
}};

constexpr std::array<RayTracingPatch, 2> Patches {{
    {.shader_hash   = 0xfa8b1616fddc203bull,
     .sites         = ShadowSites,
     .user_sgpr     = 4,
     .header_offset = 16,
     .root          = 37},
    {.shader_hash   = 0x6db33a2bcb077bc3ull,
     .sites         = SecondShadowSites,
     .user_sgpr     = 4,
     .header_offset = 16,
     .root          = 37},
}};

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

// The native traversal of a site, placed between its entry instruction and the loop header:
// RT_INIT falls back to the loop header, else the native loop runs and exits after the loop.
bool AddNativeTraversal(Layout& layout, const Program& program, const RayTracingSite& site,
                        uint32_t index) {
	auto& init = layout.AddPseudo(Opcode::RT_INIT, index);
	for (uint32_t i = 0; i < 3; i++) {
		init.image_nsa_addr[i]     = site.origin[i];
		init.image_nsa_addr[3 + i] = site.direction[i];
	}
	init.image_nsa_addr[6] = site.t_max;
	init.image_nsa_addr[7] = site.cull_mask;
	init.src0              = Sgpr(site.header_sgpr);
	if (site.alpha_off_sgpr != NoAlphaOff) {
		init.src1 = Sgpr(site.alpha_off_sgpr);
	} else {
		init.src1.kind = OperandKind::IntegerInlineConstant;
	}
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
	layout.AddPseudo(Opcode::RT_EXIT_CHECK, index);
	layout.AddBranch(Opcode::S_CBRANCH_SCC0, Layout::Target::Label, exit_label);
	restore();
	layout.Bind(exit_label);
	layout.AddPseudo(Opcode::RT_EXIT, index);
	layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Guest, site.exit_pc);
	return true;
}

} // namespace

const RayTracingPatch* FindRayTracingPatch(uint64_t shader_hash) noexcept {
	for (const auto& patch: Patches) {
		if (patch.shader_hash == shader_hash) {
			return &patch;
		}
	}
	return nullptr;
}

bool ApplyRayTracingPatch(const RayTracingPatch& patch, Program& program) {
	if (program.instructions.empty()) {
		return false;
	}
	const auto fail = [&](uint32_t pc, const char* reason) {
		LOGF("RayTracingPatch: hash=0x%016" PRIx64 " site at 0x%x: %s; not patched\n",
		     patch.shader_hash, pc, reason);
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
	program.instructions = std::move(patched);
	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler::Decoder
