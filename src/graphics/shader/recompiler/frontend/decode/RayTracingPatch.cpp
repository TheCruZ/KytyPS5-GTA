#include "graphics/shader/recompiler/frontend/decode/RayTracingPatch.h"

#include "common/logging/log.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cinttypes>
#include <cstddef>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Decoder {

// The words of the layout (LoadGuestInstance in rt_common.inc). Record dwords take 6 bits: the low
// 4 in their field and the high 2 in word 3; the transform's are relative to a base dword.
//   0: transform[0..7] - base, 4 bits each
//   1: transform[8..11] - base, 4 bits each; instance_offset / 8 << 16; blas_dword << 20;
//      blas_packed << 24; root_dword << 25; flip_facing << 29; (record_stride - 128) / 32 << 30
//   2: mask_dword; mask_shift << 4; flags_dword << 9; has flags << 13; flags_shift << 14;
//      has mask << 19; base << 20; mask_width << 25
//   3: inline_offset / 8; inline_root << 8; inline_type << 16; has an inline type << 19;
//      blas_dword >> 4 << 20; root_dword >> 4 << 22; mask_dword >> 4 << 24; flags_dword >> 4 << 26;
//      inline_offset bit 2 << 28; node_count_80 << 29
// A layout of the first 16 dwords of a 128-byte record, without the newer fields, packs as
// before they existed.
std::array<uint32_t, 4> RayTracingLayout::Pack() const {
	std::array<uint32_t, 4> words {};
	const uint32_t          base = TransformBase();
	for (uint32_t i = 0; i < 12; i++) {
		words[i / 8] |= ((transform[i] - base) & 15u) << ((i % 8) * 4);
	}
	words[1] |= ((instance_offset / 8u) & 15u) << 16u | (blas_dword & 15u) << 20u | (blas_packed ? 1u : 0u) << 24u |
	            (root_dword & 15u) << 25u | (flip_facing ? 1u : 0u) << 29u | ((record_stride - 128u) / 32u & 3u) << 30u;
	const bool flags = flags_dword != NoRegister;
	const bool mask  = mask_dword != NoRegister;
	words[2] = (mask ? mask_dword & 15u : 0u) | (mask_shift & 31u) << 4u | (flags ? flags_dword & 15u : 0u) << 9u |
	           (flags ? 1u : 0u) << 13u | (flags_shift & 31u) << 14u | (mask ? 1u : 0u) << 19u | (base & 31u) << 20u |
	           (mask ? mask_width & 31u : 0u) << 25u;
	if (inline_type != NoRegister) {
		words[3] = ((inline_offset / 8u) & 0xffu) | (inline_root & 0xffu) << 8u | (inline_type & 7u) << 16u | 1u << 19u |
		           ((inline_offset >> 2u) & 1u) << 28u;
	}
	words[3] |= (blas_dword >> 4u & 3u) << 20u | (root_dword >> 4u & 3u) << 22u | (mask ? mask_dword >> 4u & 3u : 0u) << 24u |
	            (flags ? flags_dword >> 4u & 3u : 0u) << 26u | (node_count_80 ? 1u : 0u) << 29u;
	return words;
}

// The dword the transform's are counted from: 0 when they all are in the first 16 dwords.
uint32_t RayTracingLayout::TransformBase() const {
	const auto [low, high] = std::ranges::minmax(transform);
	return high < 16 ? 0u : low;
}

// The format word of RayQueryAlpha: material_dword, table_dword << 6, first_dword << 12,
// vertex_align << 18, vertex_mask << 19, blas_dword << 27, blas_packed << 31.
uint32_t RayTracingAlphaFields::Pack(const RayTracingLayout& layout) const {
	return (material_dword & 63u) | (table_dword & 63u) << 6u | (first_dword & 63u) << 12u |
	       (vertex_align ? 1u : 0u) << 18u | (vertex_mask & 0xffu) << 19u | (layout.blas_dword & 15u) << 27u |
	       (layout.blas_packed ? 1u : 0u) << 31u;
}

namespace {

// Whether the layout fits the packed words (record fields in the first 160 bytes of the record).
bool Packable(const RayTracingLayout& layout) {
	const auto small = [&](uint32_t dword) { return dword < std::min(40u, layout.record_stride / 4u); };
	const auto base  = layout.TransformBase();
	return std::ranges::all_of(layout.transform, [&](uint8_t dword) { return dword >= base && dword - base < 16 && small(dword); }) &&
	       base < 32 && layout.instance_offset % 8 == 0 && layout.instance_offset < 128 &&
	       layout.record_stride % 32 == 0 && layout.record_stride >= 128 && layout.record_stride <= 224 &&
	       small(layout.blas_dword + 1) && small(layout.root_dword) &&
	       (layout.mask_dword == NoRegister ||
	        (small(layout.mask_dword) && layout.mask_shift < 32 && layout.mask_width < 32 &&
	         layout.mask_shift + layout.mask_width <= 32)) &&
	       (layout.inline_type == NoRegister ||
	        (layout.inline_offset % 4 == 0 && layout.inline_offset < 0x800 && layout.inline_root < 0x100)) &&
	       (layout.flags_dword == NoRegister || (small(layout.flags_dword) && layout.flags_shift < 32));
}

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
	alpha.image_nsa_addr[6] = test.fields.Pack(site.layout);
	alpha.src0              = test.spread_vector ? Vgpr(test.spread_reg) : Sgpr(test.spread_reg);
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
	if (site.cull_mask_sgpr != NoRegister) {
		init.src2 = Sgpr(site.cull_mask_sgpr);
	}
	// Every triangle opaque, or the opaque ones by their primitive word and the alpha test.
	init.src1  = Constant(site.alpha.valid ? 0 : 1);
	// Both facings, or the facing of the instance flags (and opacity from the geometry).
	init.dmask = (site.both_facings ? 1u : 0u) | (site.alpha.valid ? 2u : 0u);
	if (site.facing_sgpr != NoRegister) {
		init.src3 = Sgpr(site.facing_sgpr);
		init.dmask |= 4u;
	} else if (site.facing_mask != NoRegister) {
		init.src3 = Sgpr(site.facing_mask);
		init.dmask |= 8u;
	}
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
	commit.image_nsa_addr[3] = site.result_instance != NoRegister ? site.result_instance + 1 : NoRegister;
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
	if (site.cull_mask_sgpr != NoRegister) {
		init.src2 = Sgpr(site.cull_mask_sgpr);
	}
	if (site.facing_sgpr != NoRegister) {
		init.src3 = Sgpr(site.facing_sgpr);
		init.dmask |= 4u;
	} else if (site.facing_mask != NoRegister) {
		init.src3 = Sgpr(site.facing_mask);
		init.dmask |= 8u;
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
	SiteFinder(const Program& program, uint32_t wave_size): m_inst(program.instructions), m_wave32(wave_size == 32) {}

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
		RayTracingLayout            layout;
		// The end of the last site's loop: loops inside it (its BLAS walk) are part of it.
		size_t site_end = 0;
		for (size_t header = 1; header < m_inst.size(); header++) {
			const auto back = BackwardBranchTo(header);
			if (back == npos || header < site_end) {
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
			// One layout for the program: the sites read the same records (a site that does not
			// read the mask or the flags takes those of one that does).
			auto merged = site.layout;
			if (!sites.empty()) {
				const auto& other = layout;
				if (merged.flags_dword == NoRegister) {
					merged.flags_dword = other.flags_dword;
					merged.flags_shift = other.flags_shift;
					merged.flip_facing = other.flip_facing;
				} else if (other.flags_dword != NoRegister &&
				           (other.flags_dword != merged.flags_dword || other.flags_shift != merged.flags_shift ||
				            other.flip_facing != merged.flip_facing)) {
					return false;
				}
				if (merged.mask_dword == NoRegister) {
					merged.mask_dword = other.mask_dword;
					merged.mask_shift = other.mask_shift;
					merged.mask_width = other.mask_width;
				} else if (other.mask_dword != NoRegister &&
				           (other.mask_dword != merged.mask_dword || other.mask_shift != merged.mask_shift ||
				            other.mask_width != merged.mask_width)) {
					return false;
				}
				auto check        = other;
				check.flags_dword = merged.flags_dword;
				check.flags_shift = merged.flags_shift;
				check.flip_facing = merged.flip_facing;
				check.mask_dword  = merged.mask_dword;
				check.mask_shift  = merged.mask_shift;
				check.mask_width  = merged.mask_width;
				if (!(check == merged)) {
					return false;
				}
			}
			layout        = merged;
			site_end      = back;
			user_sgpr     = site_user;
			header_offset = site_offset;
			root          = site_root;
			sites.push_back(site);
		}
		if (sites.empty() || sites.size() > 8 || !layout.valid || !Packable(layout)) {
			return false;
		}
		patch.sites         = std::move(sites);
		patch.user_sgpr     = user_sgpr;
		patch.header_offset = header_offset;
		patch.root          = root;
		patch.layout        = layout;
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
		return !m_wave32 && i + 3 < m_inst.size() && m_inst[i].opcode == Opcode::S_AND_B64 &&
		       m_inst[i + 1].opcode == Opcode::S_AND_SAVEEXEC_B64 && IsNodeType(i + 2);
	}

	// The node type test of a loop header at `i`: S_AND_B32 <t>, <node>, 7, S_ADD_I32 vcc_lo, <t>, -4
	// and S_CMP_LE_U32 vcc_lo, 2 (box nodes 4 to 6) or 1 (4 and 5), with other instructions between
	// them that write neither; `cmp` is the compare.
	bool NodeTypeTest(size_t i, size_t& cmp) const {
		if (i >= m_inst.size()) {
			return false;
		}
		const auto& type = m_inst[i];
		const auto  t    = ScalarIndex(type.dst);
		if (type.opcode != Opcode::S_AND_B32 || t == NoRegister || type.src0.kind != OperandKind::Sgpr ||
		    !IsConstant(type.src1, 7)) {
			return false;
		}
		size_t add = npos;
		for (size_t j = i + 1; j < m_inst.size() && j <= i + 3 && add == npos; j++) {
			const auto& inst = m_inst[j];
			if (inst.opcode == Opcode::S_ADD_I32 && IsVccLo(inst.dst) && ScalarIndex(inst.src0) == t &&
			    inst.src1.kind == OperandKind::IntegerInlineConstant && inst.src1.signed_val == -4) {
				add = j;
			} else if (WritesScalar(inst, t) || WritesScalar(inst, 106u) || IsDirectBranch(inst.opcode)) {
				return false;
			}
		}
		if (add == npos || add + 1 >= m_inst.size()) {
			return false;
		}
		cmp              = add + 1;
		const auto& test = m_inst[cmp];
		return test.opcode == Opcode::S_CMP_LE_U32 && IsVccLo(test.src0) &&
		       (IsConstant(test.src1, 2) || IsConstant(test.src1, 1));
	}

	// A closest hit loop header: the copy of EXEC (S_MOV_B64 <s>, exec, or S_MOV_B32 <s>, exec_lo
	// in waves of 32 lanes), then the node type test.
	bool IsClosestHeader(size_t i) const {
		size_t cmp = npos;
		return i + 2 < m_inst.size() && m_inst[i].opcode == Lane(Opcode::S_MOV_B64) &&
		       m_inst[i].src0.kind == OperandKind::ExecLo && NodeTypeTest(i + 1, cmp);
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
		// The instance records (see DeriveLayout).
		Field record;
		if (!DeriveLayout(header, back, site, site.instance_sgpr, site.layout, record)) {
			return false;
		}
		// The instance cull: the setup's first S_CBRANCH_EXECZ, after tests that narrow EXEC, one
		// of which (V_CMP_NE_U32 or V_CMPX_NE_U32 0, x) tests the instance mask.
		const auto culled = Next(setup, Opcode::S_CBRANCH_EXECZ, back);
		if (culled == npos) {
			return false;
		}
		site.cull_exit_pc   = m_inst[culled].branch_target;
		site.cull_mask      = NoRegister;
		site.cull_mask_sgpr = NoRegister;
		bool mask           = false;
		for (size_t i = setup; i < culled && !mask; i++) {
			const auto& cmp = m_inst[i];
			if ((cmp.opcode == Opcode::V_CMPX_NE_U32 || cmp.opcode == Opcode::V_CMP_NE_U32) && IsConstant(cmp.src0, 0)) {
				mask = InstanceMaskTest(i, cmp.src1, header, back, record, site);
			}
		}
		if (!mask) {
			return false;
		}
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
		// The instance flags and the facing the triangle test accepts.
		return LeafFlags(leaf, tail, header, word.src0.reg, record, site.layout) &&
		       LeafFacing(leaf, tail, header, back, site);
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

	// Whether the instruction writes SGPR `reg` (64-bit results are pairs, lane masks pairs in
	// waves of 64; loads write their dwords).
	bool WritesScalar(const Instruction& inst, uint32_t reg) const {
		const auto name      = magic_enum::enum_name(inst.opcode);
		const bool lane_mask = name.starts_with("V_CMP");
		const bool wide      = lane_mask ? !m_wave32 : name.find("64") != std::string_view::npos;
		const auto count     = inst.family == Family::SMEM ? std::max(inst.data_dwords, 1u) : (wide ? 2u : 1u);
		const auto covers    = [&](const Operand& dst, uint32_t n) {
			const auto first = ScalarIndex(dst);
			return first != NoRegister && reg >= first && reg < first + n;
		};
		return covers(inst.dst, count) || covers(inst.dst2, m_wave32 ? 1u : 2u);
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

	// The S_LOAD_DWORD <dst>, s[base:base+1], <constant offset> that writes `operand` before `i`;
	// `dword` is its offset in dwords.
	size_t FieldLoad(size_t i, const Operand& operand, uint32_t base, size_t from, uint32_t& dword) const {
		const auto j = ScalarSource(i, operand, Opcode::S_LOAD_DWORD, from);
		if (j == npos || !IsSgpr(m_inst[j].src0, base) || (m_inst[j].offset & 3u) != 0 || m_inst[j].offset >= 0x100 ||
		    (m_inst[j].src1.kind != OperandKind::Unknown && m_inst[j].src1.kind != OperandKind::Null)) {
			return npos;
		}
		dword = static_cast<uint32_t>(m_inst[j].offset) / 4u;
		return j;
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
			if (FieldLoad(sum, *operand, record_sgpr, leaf, alpha.fields.material_dword) != npos) {
				base_ok = true;
				continue;
			}
			// The geometry word: B[B[table] + ((N >> 3) - B[first]) * 4] of the triangle node N.
			const auto load = ScalarSource(sum, *operand, Opcode::S_LOAD_DWORD, leaf);
			if (load == npos || m_inst[load].offset != 0 || m_inst[load].src0.kind != OperandKind::Sgpr ||
			    !blas_head(m_inst[load].src0.reg)) {
				continue;
			}
			const auto blas  = m_inst[load].src0.reg;
			const auto index = ScalarSource(load, m_inst[load].src1, Opcode::S_LSHL2_ADD_U32, leaf);
			if (index == npos || FieldLoad(index, m_inst[index].src1, blas, leaf, alpha.fields.table_dword) == npos) {
				continue;
			}
			const auto local = ScalarSource(index, m_inst[index].src0, Opcode::S_SUB_U32, leaf);
			if (local == npos || FieldLoad(local, m_inst[local].src1, blas, leaf, alpha.fields.first_dword) == npos) {
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
		if (sum == npos || FieldLoad(sum, m_inst[sum].src0, record_sgpr, begin, alpha.fields.material_dword) == npos) {
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
		if (table == npos || FieldLoad(table, m_inst[table].src0, blas, begin, alpha.fields.table_dword) == npos) {
			return false;
		}
		const auto local = VectorSource(table, m_inst[table].src1, Opcode::V_ADD_LSHL_U32, begin);
		if (local == npos || !IsConstant(m_inst[local].src2, 2)) {
			return false;
		}
		const auto negate = ScalarSource(local, m_inst[local].src0, Opcode::S_SUB_I32, begin);
		if (negate == npos || !IsConstant(m_inst[negate].src0, 0) ||
		    FieldLoad(negate, m_inst[negate].src1, blas, begin, alpha.fields.first_dword) == npos) {
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
		// Each vertex dword at base + ((M1 & mask) + index * (M0 & mask)), aligned down to 4 bytes
		// or not, + 4 for the second one; the mask is a V_AND_B32 constant, or 0xff for a byte
		// read through SDWA.
		const auto masked = [&](size_t at, const Operand& operand, size_t word, uint32_t& mask) {
			if (operand.kind != OperandKind::Vgpr || operand.negate || operand.absolute) {
				return false;
			}
			if (operand.sdwa_sel == 0) {
				mask = 0xffu;
				return VectorSource(at, operand, Opcode::BUFFER_LOAD_DWORD, begin) == material[word];
			}
			const auto bits = VectorSource(at, operand, Opcode::V_AND_B32, begin);
			if (!Plain(operand) || bits == npos ||
			    (m_inst[bits].src0.kind != OperandKind::IntegerInlineConstant &&
			     m_inst[bits].src0.kind != OperandKind::LiteralConstant)) {
				return false;
			}
			mask = m_inst[bits].src0.value;
			return (mask == 0xfcu || mask == 0xffu) &&
			       VectorSource(bits, m_inst[bits].src1, Opcode::BUFFER_LOAD_DWORD, begin) == material[word];
		};
		const auto vertex_offset = [&](size_t at, const Operand& operand, bool& second, bool& align, uint32_t& mask) {
			size_t w = operand.kind == OperandKind::Vgpr && Plain(operand) ? VectorWriter(at, operand.reg, begin) : npos;
			second   = w != npos && m_inst[w].opcode == Opcode::V_ADD_NC_U32 && IsConstant(m_inst[w].src0, 4);
			if (second) {
				w = m_inst[w].src1.kind == OperandKind::Vgpr ? VectorWriter(w, m_inst[w].src1.reg, begin) : npos;
			}
			align = w != npos && m_inst[w].opcode == Opcode::V_AND_B32 && m_inst[w].src0.signed_val == -4 &&
			        m_inst[w].src0.kind == OperandKind::IntegerInlineConstant;
			if (align) {
				w = m_inst[w].src1.kind == OperandKind::Vgpr ? VectorWriter(w, m_inst[w].src1.reg, begin) : npos;
			}
			if (w == npos || m_inst[w].opcode != Opcode::V_ADD_NC_U32) {
				return false;
			}
			for (const auto& [uv, product]: {std::pair {&m_inst[w].src0, &m_inst[w].src1},
			                                 std::pair {&m_inst[w].src1, &m_inst[w].src0}}) {
				uint32_t uv_mask = 0, stride_mask = 0;
				if (!masked(w, *uv, 1, uv_mask)) {
					continue;
				}
				const auto mul = VectorSource(w, *product, Opcode::V_MUL_LO_U32, begin);
				if (mul == npos || !Plain(*product) ||
				    (!masked(mul, m_inst[mul].src0, 0, stride_mask) && !masked(mul, m_inst[mul].src1, 0, stride_mask)) ||
				    stride_mask != uv_mask) {
					continue;
				}
				mask = uv_mask;
				return true;
			}
			return false;
		};
		uint32_t vertices = 0, seconds = 0, vertex_mask = 0;
		bool     vertex_align = false;
		for (size_t i = begin; i < end; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode != Opcode::FLAT_LOAD_DWORD || i == geometry) {
				continue;
			}
			const auto add = VectorSource(i, inst.src0, Opcode::V_ADD_I32, begin);
			if (inst.memory_segment != 2 || inst.offset != 0 || inst.src1.kind != OperandKind::Vgpr || add == npos) {
				return false;
			}
			const bool base0  = VectorSource(add, m_inst[add].src0, Opcode::BUFFER_LOAD_DWORDX2, begin) == vertex_vsharp;
			const bool base1  = VectorSource(add, m_inst[add].src1, Opcode::BUFFER_LOAD_DWORDX2, begin) == vertex_vsharp;
			bool       second = false, align = false;
			uint32_t   mask   = 0;
			if (base0 == base1 || !vertex_offset(add, base0 ? m_inst[add].src1 : m_inst[add].src0, second, align, mask) ||
			    (vertices != 0 && (mask != vertex_mask || align != vertex_align))) {
				return false;
			}
			vertex_mask  = mask;
			vertex_align = align;
			seconds += second ? 1u : 0u;
			vertices++;
		}
		if (vertices != 6 || seconds != 3) {
			return false;
		}
		alpha.fields.vertex_mask  = vertex_mask;
		alpha.fields.vertex_align = vertex_align;
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
			// omod 1: the result times 2; t times the spread (a VGPR or an SGPR), either order.
			const bool t0 = mad.src0.kind == OperandKind::Vgpr && mad.src0.reg == t_vgpr;
			const bool t1 = mad.src1.kind == OperandKind::Vgpr && mad.src1.reg == t_vgpr;
			const auto& spread = t1 ? mad.src0 : mad.src1;
			if (mad.dst.omod != 1 || mad.dst.clamp || !Plain(mad.src0) || !Plain(mad.src1) || !Plain(mad.src2) ||
			    mad.src2.kind != OperandKind::Vgpr || !(t0 || t1) ||
			    (spread.kind != OperandKind::Vgpr && spread.kind != OperandKind::Sgpr)) {
				continue;
			}
			alpha.spread_reg    = spread.reg;
			alpha.spread_vector = spread.kind == OperandKind::Vgpr;
			alpha.lod_vgpr      = mad.src2.reg;
			alpha.lod_offset = m_inst[offset].src0.value;
			lod_ok           = true;
		}
		if (!max_ok || !lod_ok) {
			return false;
		}
		// The registers the native code reads at the site entry keep their value in the traversal.
		for (size_t i = header; i <= back; i++) {
			if ((alpha.spread_vector ? WritesVector(m_inst[i], alpha.spread_reg) : WritesScalar(m_inst[i], alpha.spread_reg)) ||
			    WritesVector(m_inst[i], alpha.lod_vgpr) || WritesVector(m_inst[i], alpha.reference_vgpr)) {
				return false;
			}
		}
		if (alpha.fields.material_dword >= 32) {
			return false;
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
		size_t le           = npos;
		if (!NodeTypeTest(header + 1, le) || le >= back) {
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
		// The exit: the branch that skips the loop's backward branch (at the end of the code that pops
		// the next node, which may read it from one of two stack VGPRs).
		size_t exit = npos;
		for (size_t i = back; i-- > header && i + 16 > back && exit == npos;) {
			if (m_inst[i].opcode == Opcode::S_CBRANCH_SCC0 && m_inst[i].branch_target == EndPc(m_inst[back])) {
				exit = i;
			}
		}
		if (exit == npos) {
			return false;
		}
		site.exit_pc = m_inst[exit].branch_target;
		// The instance records (see DeriveLayout).
		Field record;
		if (!DeriveLayout(header, back, site, m_inst[header + 1].src0.reg, site.layout, record)) {
			return false;
		}
		// The nearest hit: the distance test, then the moves of the hit, right after it (every
		// triangle opaque) or after the alpha test of the triangles not marked opaque; or, in a
		// wave-wide walk, V_CNDMASK_B32 of the hit's registers (see ClosestSelectHit).
		size_t compare = npos, accept = npos;
		for (size_t i = header; i + 3 < back && accept == npos; i++) {
			const auto& cmp = m_inst[i];
			if (cmp.opcode != Opcode::V_CMP_GT_F32 || !IsVccLo(cmp.dst) ||
			    cmp.src1.kind != OperandKind::Vgpr || cmp.src1.reg != site.t_max) {
				continue;
			}
			const auto save = Next(i, Lane(Opcode::S_AND_SAVEEXEC_B64), i + 4);
			if (save != npos && m_inst[save + 1].opcode == Opcode::S_CBRANCH_EXECZ) {
				compare = i;
				accept  = save + 2;
			}
		}
		site.alpha        = {};
		site.both_facings = true;
		if (accept == npos) {
			// The selected key goes to an SGPR pair the program does not use.
			const auto scratch = UnusedPair();
			if (!ClosestSelectHit(header, back, record.pair, site) || scratch == NoRegister) {
				return false;
			}
			site.both_facings  = false;
			site.instance_sgpr = scratch;
		} else if (m_inst[compare].src0.kind != OperandKind::Vgpr) {
			return false;
		}
		size_t moves_end = accept;
		while (moves_end < back && m_inst[moves_end].opcode != Lane(Opcode::S_MOV_B64)) {
			moves_end++;
		}
		uint32_t entry_base = NoRegister;
		if (accept != npos &&
		    ((!HitMoves(header, back, accept, moves_end, {NoRegister, NoRegister}, site, entry_base) &&
		      !ClosestAlphaHit(header, back, compare, accept, site)) ||
		     record.pair != site.instance_sgpr)) {
			// The hit moves copy the record pair.
			return false;
		}
		// The instance mask: the guest skips the instances whose mask (record dword >> shift) has
		// none of the bits of S_AND_B32 <s>, <mask dword>, <bits << shift>; S_CBRANCH_SCC0.
		bool mask = false;
		for (size_t i = header; i + 1 < back && !mask; i++) {
			Field    field;
			uint32_t shift = 0, bits = 0, width = 0;
			if (ConstantMaskTest(i, header, field, shift, bits, width) && field.pair == record.pair &&
			    ScalarWriter(field.load, record.pair, 0) == ScalarWriter(record.load, record.pair, 0)) {
				site.cull_mask_constant = bits;
				site.layout.mask_dword  = field.dword;
				site.layout.mask_shift  = shift;
				site.layout.mask_width  = width;
				mask                    = true;
			}
		}
		// Without one every instance counts.
		if (!mask) {
			site.layout.mask_dword = NoRegister;
			site.layout.mask_shift = 0;
			site.layout.mask_width = 0;
		}
		// Where the BVH headers hold their node count.
		site.layout.node_count_80 = NodeCountAt80(box, site.header_sgpr);
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
		if (saved.opcode != Lane(Opcode::S_MOV_B64) || saved.src0.kind != OperandKind::ExecLo ||
		    saved.dst.kind != OperandKind::Sgpr || m_inst[split + 1].opcode != Opcode::S_CBRANCH_EXECZ) {
			return false;
		}
		const auto join = IndexOf(m_inst[split + 1].branch_target);
		if (join == npos || join <= split + 2 || join + 3 >= back) {
			return false;
		}
		const auto& restore   = m_inst[join];
		const auto& flag_test = m_inst[join + 1];
		if (restore.opcode != Lane(Opcode::S_MOV_B64) || restore.dst.kind != OperandKind::ExecLo ||
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

	// The TLAS header pair: a S_LOAD_DWORDX2 from a pair nothing wrote before (user data), the
	// value the T# of the TLAS is built from (the pair may hold something else by the loop).
	bool FindTlasSource(size_t header, uint32_t header_sgpr, uint32_t& user_sgpr, uint32_t& offset) const {
		size_t start = header;
		for (size_t i = header; i-- > 0;) {
			const auto& bfe = m_inst[i];
			if (bfe.opcode == Opcode::S_BFE_U64 && IsSgpr(bfe.src0, header_sgpr) && IsConstant(bfe.src1, 0x280008)) {
				start = i;
				break;
			}
		}
		for (size_t i = start; i-- > 0;) {
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

	// ---- The layout of the guest TLAS (see RayTracingLayout), from the instance setup. ----
	//
	// The instance setup loads fields of the instance record (SMEM loads from the record's SGPR
	// pair, base + (node >> 3) << 7 with base = header + the 64-bit offset at a header field) and
	// computes the object space ray with V_MAD/V_MAC/V_MUL/V_ADD of record fields and the world
	// ray, then builds the T# of the BLAS from an address made of record fields. Each value is
	// traced to the instructions that make it.

	// The last instruction in [from, i) that writes VGPR `reg` (npos: none).
	size_t VectorWriter(size_t i, uint32_t reg, size_t from) const {
		for (size_t j = i; j-- > from;) {
			if (WritesVector(m_inst[j], reg)) {
				return j;
			}
		}
		return npos;
	}

	// A record field: SGPR `reg` at instruction `i` holds dword `dword` of the record at SGPR pair
	// `pair`, loaded by instruction `load` (through S_MOV_B32 copies).
	struct Field {
		uint32_t pair  = NoRegister;
		size_t   load  = npos;
		uint32_t dword = 0;
	};

	bool RecordField(size_t i, uint32_t reg, size_t from, Field& field) const {
		auto j = ScalarWriter(i, reg, from);
		while (j != npos && m_inst[j].opcode == Opcode::S_MOV_B32 && ScalarIndex(m_inst[j].src0) != NoRegister &&
		       ScalarIndex(m_inst[j].dst) == reg) {
			reg = ScalarIndex(m_inst[j].src0);
			j   = ScalarWriter(j, reg, from);
		}
		if (j == npos) {
			return false;
		}
		const auto& load = m_inst[j];
		switch (load.opcode) {
			case Opcode::S_LOAD_DWORD:
			case Opcode::S_LOAD_DWORDX2:
			case Opcode::S_LOAD_DWORDX4:
			case Opcode::S_LOAD_DWORDX8:
			case Opcode::S_LOAD_DWORDX16: break;
			default: return false;
		}
		const auto first = ScalarIndex(load.dst);
		if (load.src0.kind != OperandKind::Sgpr || first == NoRegister || reg < first ||
		    (load.src1.kind != OperandKind::Unknown && load.src1.kind != OperandKind::Null) || (load.offset & 3u) != 0) {
			return false;
		}
		field = {load.src0.reg, j, (static_cast<uint32_t>(load.offset) >> 2u) + (reg - first)};
		return field.dword < 64;
	}

	// The record field operand `operand` of instruction `i` reads (an SGPR, VCC included).
	bool OperandField(size_t i, const Operand& operand, size_t from, Field& field) const {
		const auto reg = ScalarIndex(operand);
		return reg != NoRegister && Plain(operand) && RecordField(i, reg, from, field);
	}

	// A record field operand read through S_AND_B32s with constants; `mask` is their AND (every bit
	// without one).
	bool MaskedField(size_t i, const Operand& operand, size_t from, Field& field, uint32_t& mask) const {
		mask             = ~0u;
		const Operand* x = &operand;
		for (uint32_t depth = 0; depth < 4; depth++) {
			if (OperandField(i, *x, from, field)) {
				return true;
			}
			const auto a = ScalarSource(i, *x, Opcode::S_AND_B32, from);
			if (a == npos || !Plain(*x)) {
				return false;
			}
			const auto& inst = m_inst[a];
			const bool  c0   = inst.src0.kind == OperandKind::IntegerInlineConstant ||
			                inst.src0.kind == OperandKind::LiteralConstant;
			const bool c1 = inst.src1.kind == OperandKind::IntegerInlineConstant ||
			                inst.src1.kind == OperandKind::LiteralConstant;
			if (c0 == c1) {
				return false;
			}
			mask &= (c0 ? inst.src0 : inst.src1).value;
			x = c0 ? &inst.src1 : &inst.src0;
			i = a;
		}
		return false;
	}

	// Whether the T# of the box test `box` (built from the TLAS header pair `header_sgpr`) takes its
	// size from the 64-bit node count at header + 80, less one: S_LOAD_DWORDX2 <c>, <header>, 0x50;
	// S_ADD_U32 / S_ADDC_U32 <c>, -1; S_BFE_U64 <size>, <c>, 0x2a0000. Otherwise the size is at
	// header + 88.
	bool NodeCountAt80(size_t box, uint32_t header_sgpr) const {
		const auto& test = m_inst[box];
		const auto  size = test.src1.kind == OperandKind::Sgpr ? test.src1.reg + 2u : NoRegister;
		const auto  bfe  = ScalarWriter(box, size, 0);
		if (bfe == npos || m_inst[bfe].opcode != Opcode::S_BFE_U64 || ScalarIndex(m_inst[bfe].dst) != size ||
		    !IsConstant(m_inst[bfe].src1, 0x2a0000u)) {
			return false;
		}
		const auto pair = ScalarIndex(m_inst[bfe].src0);
		const auto lo   = ScalarWriter(bfe, pair, 0);
		const auto hi   = pair != NoRegister ? ScalarWriter(bfe, pair + 1, 0) : npos;
		if (lo == npos || hi == npos || m_inst[lo].opcode != Opcode::S_ADD_U32 ||
		    m_inst[hi].opcode != Opcode::S_ADDC_U32 || !IsConstant(m_inst[lo].src1, ~0u) ||
		    !IsConstant(m_inst[hi].src1, ~0u)) {
			return false;
		}
		const auto count_lo = ScalarIndex(m_inst[lo].src0);
		const auto count    = ScalarRangeSource(lo, count_lo, 2, Opcode::S_LOAD_DWORDX2, 0);
		return count != npos && count_lo != NoRegister && ScalarIndex(m_inst[hi].src0) == count_lo + 1 &&
		       ScalarWriter(hi, count_lo + 1, 0) == count && IsSgpr(m_inst[count].src0, header_sgpr) &&
		       m_inst[count].offset == 0x50 &&
		       (m_inst[count].src1.kind == OperandKind::Unknown || m_inst[count].src1.kind == OperandKind::Null);
	}

	// The instance record address: the pair `field.pair` (as `field.load` reads it) holds
	// base + (node >> 3) * stride with base = header + the 64-bit value at header + offset, header
	// the TLAS header pair `header_sgpr`. Gives that offset, the stride and the node register.
	bool RecordAddress(const Field& field, uint32_t header_sgpr, uint32_t& instance_offset, uint32_t& stride,
	                   uint32_t& node) const {
		const auto lo = ScalarWriter(field.load, field.pair, 0);
		const auto hi = ScalarWriter(field.load, field.pair + 1, 0);
		if (lo == npos || hi == npos || m_inst[lo].opcode != Opcode::S_ADD_U32 || m_inst[hi].opcode != Opcode::S_ADDC_U32 ||
		    ScalarIndex(m_inst[lo].dst) != field.pair || ScalarIndex(m_inst[hi].dst) != field.pair + 1) {
			return false;
		}
		// Whether operands k of the low and high adds are `is_a` and the others `is_b`.
		const auto pair_operands = [&](size_t add_lo, size_t add_hi, auto&& is_a, auto&& is_b) {
			const std::array<const Operand*, 2> a {&m_inst[add_lo].src0, &m_inst[add_lo].src1};
			const std::array<const Operand*, 2> b {&m_inst[add_hi].src0, &m_inst[add_hi].src1};
			for (uint32_t k = 0; k < 2; k++) {
				if (is_a(add_lo, *a[k], add_hi, *b[k]) && is_b(add_lo, *a[1 - k], add_hi, *b[1 - k])) {
					return true;
				}
			}
			return false;
		};
		// (node >> 3) << 7: S_LSHL_B64 <pair>, <n>, 7 with n = (S_LSHR_B32 node, 3; 0); or
		// (node >> 3) * stride: S_MUL_HI_U32 <hi>, <n>, stride and S_MULK_I32 <n>, stride with
		// n = S_LSHR_B32 node, 3.
		const auto is_index = [&](size_t add_lo, const Operand& a, size_t add_hi, const Operand& b) {
			const auto shift = ScalarWriter(add_lo, ScalarIndex(a), 0);
			if (shift != npos && m_inst[shift].opcode == Opcode::S_MULK_I32) {
				return StrideIndex(shift, add_hi, a, b, stride, node);
			}
			stride = 128;
			if (shift == npos || m_inst[shift].opcode != Opcode::S_LSHL_B64 || ScalarIndex(m_inst[shift].dst) != ScalarIndex(a) ||
			    ScalarIndex(b) != ScalarIndex(a) + 1 || ScalarWriter(add_hi, ScalarIndex(b), 0) != shift ||
			    !IsConstant(m_inst[shift].src1, 7)) {
				return false;
			}
			const auto n    = ScalarIndex(m_inst[shift].src0);
			const auto lsr  = n != NoRegister ? ScalarWriter(shift, n, 0) : npos;
			const auto zero = n != NoRegister ? ScalarWriter(shift, n + 1, 0) : npos;
			if (lsr == npos || zero == npos || !IsScalarConstantOp(lsr, Opcode::S_LSHR_B32, 3) ||
			    m_inst[zero].opcode != Opcode::S_MOV_B32 || !IsConstant(m_inst[zero].src0, 0)) {
				return false;
			}
			node = ScalarIndex(m_inst[lsr].src0);
			return node != NoRegister;
		};
		// header + [header + offset]: S_ADD_U32 / S_ADDC_U32 of the header pair and of a
		// S_LOAD_DWORDX2 from it.
		const auto is_base = [&](size_t add_lo, const Operand& a, size_t add_hi, const Operand& b) {
			const auto base_lo = ScalarWriter(add_lo, ScalarIndex(a), 0);
			const auto base_hi = ScalarWriter(add_hi, ScalarIndex(b), 0);
			if (base_lo == npos || base_hi == npos || m_inst[base_lo].opcode != Opcode::S_ADD_U32 ||
			    m_inst[base_hi].opcode != Opcode::S_ADDC_U32) {
				return false;
			}
			const auto is_header = [&](size_t, const Operand& x, size_t, const Operand& y) {
				return ScalarIndex(x) == header_sgpr && ScalarIndex(y) == header_sgpr + 1;
			};
			const auto is_offset = [&](size_t at_lo, const Operand& x, size_t at_hi, const Operand& y) {
				const auto load = ScalarWriter(at_lo, ScalarIndex(x), 0);
				if (load == npos || m_inst[load].opcode != Opcode::S_LOAD_DWORDX2 || ScalarWriter(at_hi, ScalarIndex(y), 0) != load ||
				    ScalarIndex(m_inst[load].dst) != ScalarIndex(x) || ScalarIndex(y) != ScalarIndex(x) + 1 ||
				    !IsSgpr(m_inst[load].src0, header_sgpr) ||
				    (m_inst[load].src1.kind != OperandKind::Unknown && m_inst[load].src1.kind != OperandKind::Null)) {
					return false;
				}
				instance_offset = static_cast<uint32_t>(m_inst[load].offset);
				return true;
			};
			return pair_operands(base_lo, base_hi, is_offset, is_header);
		};
		return pair_operands(lo, hi, is_index, is_base);
	}

	// The index of a record of `stride` bytes (see RecordAddress): S_MULK_I32 at `mul_lo` writes the
	// low dword `a`, S_MUL_HI_U32 the high dword `b` that `add_hi` reads.
	bool StrideIndex(size_t mul_lo, size_t add_hi, const Operand& a, const Operand& b, uint32_t& stride,
	                 uint32_t& node) const {
		const auto& lo = m_inst[mul_lo];
		const auto  n  = ScalarIndex(a);
		const auto  hi = ScalarWriter(add_hi, ScalarIndex(b), 0);
		if (n == NoRegister || ScalarIndex(lo.dst) != n || ScalarIndex(lo.src0) != n || hi == npos ||
		    m_inst[hi].opcode != Opcode::S_MUL_HI_U32 || ScalarIndex(m_inst[hi].dst) != ScalarIndex(b)) {
			return false;
		}
		const auto& high = m_inst[hi];
		const auto* by   = ScalarIndex(high.src0) == n ? &high.src1 : ScalarIndex(high.src1) == n ? &high.src0 : nullptr;
		const auto  lsr  = ScalarWriter(mul_lo, n, 0);
		if (by == nullptr || !IsConstant(*by, lo.src1.value) || lsr == npos || ScalarWriter(hi, n, 0) != lsr ||
		    !IsScalarConstantOp(lsr, Opcode::S_LSHR_B32, 3) || ScalarIndex(m_inst[lsr].dst) != n) {
			return false;
		}
		stride = lo.src1.value;
		node   = ScalarIndex(m_inst[lsr].src0);
		return node != NoRegister;
	}

	// The affine form of a value of the object space ray: per world component (the VGPRs
	// `inputs`, unchanged since `from`), the record dword it is multiplied by, and the dword
	// added (NoRegister: none).
	struct Form {
		std::array<uint32_t, 3> factor {NoRegister, NoRegister, NoRegister};
		uint32_t                constant = NoRegister;
		// A load of the record (the first field found).
		size_t                  load     = npos;
	};

	bool FormTerm(size_t i, const Operand& a, const Operand& b, size_t from, const std::array<uint32_t, 3>& inputs,
	              uint32_t& pair, Form& form) const {
		for (const auto& [scalar, vector]: {std::pair {&a, &b}, std::pair {&b, &a}}) {
			Field field;
			if (vector->kind != OperandKind::Vgpr || !Plain(*vector) || !OperandField(i, *scalar, from, field)) {
				continue;
			}
			const auto input = std::ranges::find(inputs, vector->reg);
			if (input == inputs.end() || VectorWriter(i, vector->reg, from) != npos ||
			    (pair != NoRegister && pair != field.pair)) {
				return false;
			}
			auto& factor = form.factor[input - inputs.begin()];
			if (factor != NoRegister) {
				return false;
			}
			pair   = field.pair;
			factor = field.dword;
			form.load = form.load == npos ? field.load : form.load;
			return true;
		}
		return false;
	}

	bool FormConstant(size_t i, const Operand& operand, size_t from, uint32_t& pair, Form& form) const {
		Field field;
		if (form.constant != NoRegister || !OperandField(i, operand, from, field) ||
		    (pair != NoRegister && pair != field.pair)) {
			return false;
		}
		pair          = field.pair;
		form.constant = field.dword;
		form.load     = form.load == npos ? field.load : form.load;
		return true;
	}

	bool LinearForm(size_t i, uint32_t reg, size_t from, const std::array<uint32_t, 3>& inputs, uint32_t& pair, Form& form,
	                uint32_t depth = 0) const {
		const auto w = VectorWriter(i, reg, from);
		if (w == npos || depth > 8) {
			return false;
		}
		const auto& inst = m_inst[w];
		if (inst.dst.kind != OperandKind::Vgpr || inst.dst.reg != reg || !Plain(inst.dst) || !Plain(inst.src0) ||
		    !Plain(inst.src1)) {
			return false;
		}
		const auto add = [&](const Operand& operand) {
			if (operand.kind == OperandKind::Vgpr) {
				return LinearForm(w, operand.reg, from, inputs, pair, form, depth + 1);
			}
			return FormConstant(w, operand, from, pair, form);
		};
		switch (inst.opcode) {
			case Opcode::V_MAD_F32:
			case Opcode::V_FMA_F32:
				return Plain(inst.src2) && FormTerm(w, inst.src0, inst.src1, from, inputs, pair, form) && add(inst.src2);
			case Opcode::V_MAC_F32:
			case Opcode::V_FMAC_F32:
				return FormTerm(w, inst.src0, inst.src1, from, inputs, pair, form) &&
				       LinearForm(w, reg, from, inputs, pair, form, depth + 1);
			case Opcode::V_MUL_F32: return FormTerm(w, inst.src0, inst.src1, from, inputs, pair, form);
			case Opcode::V_ADD_F32: return add(inst.src0) && add(inst.src1);
			case Opcode::V_MOV_B32: return add(inst.src0);
			default: return false;
		}
	}

	// The first IMAGE_BVH_INTERSECT_RAY in [from, to) whose T# is built (S_BFE_U64 <t#>, <pair>,
	// 0x280008) in the loop, from the BLAS address (the T# of the TLAS is built before it), and that
	// S_BFE_U64.
	bool BlasTest(size_t from, size_t to, size_t& test, size_t& bfe) const {
		for (size_t i = from; i < to; i++) {
			if (m_inst[i].opcode != Opcode::IMAGE_BVH_INTERSECT_RAY || m_inst[i].src1.kind != OperandKind::Sgpr) {
				continue;
			}
			const auto b = ScalarSource(i, m_inst[i].src1, Opcode::S_BFE_U64, from);
			if (b != npos && IsConstant(m_inst[b].src1, 0x280008) && m_inst[b].src0.kind == OperandKind::Sgpr &&
			    ScalarIndex(m_inst[b].dst) == m_inst[i].src1.reg) {
				test = i;
				bfe  = b;
				return true;
			}
		}
		return false;
	}

	// A record field read through pair `pair` in [from, i): the value of the last load from it.
	bool RecordFieldOf(size_t i, uint32_t pair, size_t from, Field& field) const {
		for (size_t j = i; j-- > from;) {
			const auto& load = m_inst[j];
			if (load.family == Family::SMEM && IsSgpr(load.src0, pair) && ScalarIndex(load.dst) != NoRegister &&
			    RecordField(j + 1, ScalarIndex(load.dst), from, field)) {
				return field.pair == pair && field.load == j;
			}
		}
		return false;
	}

	// The layout of the instance records the traversal from `header` to `back` reads (see
	// RayTracingLayout); `node` is the TLAS node register. The caller adds the mask (from the
	// site's instance mask test) and the flags (from a Packet site's leaf code).
	bool DeriveLayout(size_t header, size_t back, const RayTracingSite& site, uint32_t node, RayTracingLayout& layout,
	                  Field& record) const {
		size_t test = npos, bfe = npos;
		if (!BlasTest(header, back, test, bfe)) {
			return false;
		}
		// The object space ray: origin = M * world origin + t, direction = M * world direction.
		const std::array<uint32_t, 3> origin {site.origin[0], site.origin[1], site.origin[2]};
		const std::array<uint32_t, 3> direction {site.direction[0], site.direction[1], site.direction[2]};
		uint32_t                      pair = NoRegister;
		size_t                        load = npos;
		for (uint32_t r = 0; r < 3; r++) {
			Form o, d;
			if (!LinearForm(test, AddressReg(m_inst[test], 2 + r), header, origin, pair, o) ||
			    !LinearForm(test, AddressReg(m_inst[test], 5 + r), header, direction, pair, d) || o.factor != d.factor ||
			    o.constant == NoRegister || d.constant != NoRegister ||
			    std::ranges::find(o.factor, NoRegister) != o.factor.end()) {
				return false;
			}
			for (uint32_t c = 0; c < 3; c++) {
				layout.transform[r * 4 + c] = static_cast<uint8_t>(o.factor[c]);
			}
			layout.transform[r * 4 + 3] = static_cast<uint8_t>(o.constant);
			load                        = load == npos ? o.load : load;
		}
		// The record address (from one of the loads of the transform).
		record = {pair, load, 0};
		if (load == npos) {
			return false;
		}
		uint32_t record_node = NoRegister;
		if (!RecordAddress(record, site.header_sgpr, layout.instance_offset, layout.record_stride, record_node) ||
		    record_node != node) {
			return false;
		}
		const auto same_record = [&](const Field& field) {
			return field.pair == pair && ScalarWriter(field.load, pair, 0) == ScalarWriter(record.load, pair, 0);
		};
		// The BLAS: the pair the T# is built from, two record dwords as they are or packed.
		const auto blas  = m_inst[bfe].src0.reg;
		const auto shift = ScalarWriter(bfe, blas, header);
		Field      lo, hi;
		if (shift != npos && m_inst[shift].opcode == Opcode::S_LSHL_B64) {
			// (lo | (hi & 0x3fffff) << 32) << 3.
			const auto packed = ScalarIndex(m_inst[shift].src0);
			const auto mask   = packed != NoRegister ? ScalarWriter(shift, packed + 1, header) : npos;
			if (ScalarIndex(m_inst[shift].dst) != blas || ScalarWriter(bfe, blas + 1, header) != shift ||
			    !IsConstant(m_inst[shift].src1, 3) || mask == npos || !RecordField(shift, packed, header, lo) ||
			    m_inst[mask].opcode != Opcode::S_AND_B32 || ScalarIndex(m_inst[mask].dst) != packed + 1) {
				return false;
			}
			const auto& and_mask = m_inst[mask];
			const auto* value    = IsConstant(and_mask.src1, 0x3fffff)   ? &and_mask.src0
			                       : IsConstant(and_mask.src0, 0x3fffff) ? &and_mask.src1
			                                                               : nullptr;
			if (value == nullptr || !OperandField(mask, *value, header, hi)) {
				return false;
			}
			layout.blas_packed = true;
		} else if (RecordField(bfe, blas, header, lo) && RecordField(bfe, blas + 1, header, hi)) {
			layout.blas_packed = false;
		} else {
			return false;
		}
		if (!same_record(lo) || !same_record(hi) || hi.dword != lo.dword + 1) {
			return false;
		}
		layout.blas_dword = lo.dword;
		// The BLAS root: the node the BLAS walk starts at, a record dword moved into the node VGPR
		// of a test of the BLAS (per lane walks; through S_CSELECT_B32 of a constant root, for
		// walks that also take a BLAS without a record) or pushed on the node stack the BLAS loop
		// pops it from with V_READLANE_B32 (wave-wide walks).
		// The root of the record, or a constant one (the S_CSELECT_B32 of the constant and the
		// record field; `inline_root` gets the constant).
		uint32_t   inline_root = NoRegister;
		const auto root_field  = [&](size_t i, const Operand& operand, Field& field) {
			if (OperandField(i, operand, header, field)) {
				if (!same_record(field)) {
					return false;
				}
				// A constant root for the leaves whose record a branch skips the load of:
				// S_MOV_B32 <root>, <constant>, then the branch, then S_LOAD_DWORD <root>.
				const auto  reg  = ScalarIndex(operand);
				const auto& load = m_inst[field.load];
				const auto  move = load.opcode == Opcode::S_LOAD_DWORD && ScalarIndex(load.dst) == reg
				                       ? ScalarWriter(field.load, reg, header)
				                       : npos;
				if (move != npos && m_inst[move].opcode == Opcode::S_MOV_B32 &&
				    (m_inst[move].src0.kind == OperandKind::IntegerInlineConstant ||
				     m_inst[move].src0.kind == OperandKind::LiteralConstant)) {
					for (size_t j = move + 1; j < field.load; j++) {
						if (IsConditionalBranch(m_inst[j].opcode) && m_inst[j].branch_target > load.pc) {
							inline_root = m_inst[move].src0.value;
						}
					}
				}
				return true;
			}
			const auto select = ScalarSource(i, operand, Opcode::S_CSELECT_B32, header);
			if (select == npos) {
				return false;
			}
			const auto& sel = m_inst[select];
			const bool  c0  = sel.src0.kind == OperandKind::IntegerInlineConstant || sel.src0.kind == OperandKind::LiteralConstant;
			const bool  c1  = sel.src1.kind == OperandKind::IntegerInlineConstant || sel.src1.kind == OperandKind::LiteralConstant;
			if (c0 == c1 || !OperandField(select, c0 ? sel.src1 : sel.src0, header, field) || !same_record(field)) {
				return false;
			}
			inline_root = (c0 ? sel.src0 : sel.src1).value;
			return true;
		};
		bool root = false;
		// A BLAS walked wave-wide: the value of the node SGPR of the inner loop (S_AND_B32 <t>,
		// <node>, 7 ...; its backward branch goes to the copy of EXEC before it or to it) before it.
		for (size_t inner = header + 3; inner + 1 < back && !root; inner++) {
			Field  field;
			size_t cmp = npos;
			if (NodeTypeTest(inner, cmp) && m_inst[inner].src0.kind == OperandKind::Sgpr &&
			    root_field(inner, m_inst[inner].src0, field) && ScalarWriter(inner, m_inst[inner].src0.reg, header) != npos &&
			    (BackwardBranchTo(inner - 1) != npos || BackwardBranchTo(inner) != npos)) {
				layout.root_dword = field.dword;
				root              = true;
			}
		}
		for (size_t blas_test = header; blas_test < back && !root; blas_test++) {
			const auto& candidate = m_inst[blas_test];
			if (candidate.opcode != Opcode::IMAGE_BVH_INTERSECT_RAY || candidate.src1.kind != OperandKind::Sgpr) {
				continue;
			}
			const auto blas_bfe = ScalarSource(blas_test, candidate.src1, Opcode::S_BFE_U64, header);
			if (blas_bfe == npos || !IsConstant(m_inst[blas_bfe].src1, 0x280008)) {
				continue;
			}
			const auto node_vgpr = AddressReg(candidate, 0);
			for (size_t i = header; i < blas_test && !root; i++) {
				const auto& inst = m_inst[i];
				Field       field;
				if (inst.opcode == Opcode::V_MOV_B32 && inst.dst.kind == OperandKind::Vgpr && inst.dst.reg == node_vgpr &&
				    root_field(i, inst.src0, field)) {
					layout.root_dword = field.dword;
					root              = true;
				} else if (inst.opcode == Opcode::V_WRITELANE_B32 && inst.dst.kind == OperandKind::Vgpr &&
				           OperandField(i, inst.src0, header, field) && same_record(field)) {
					const auto move = VectorWriter(blas_test, node_vgpr, header);
					if (move == npos || m_inst[move].opcode != Opcode::V_MOV_B32 || ScalarIndex(m_inst[move].src0) == NoRegister) {
						continue;
					}
					const auto pop = ScalarWriter(move, ScalarIndex(m_inst[move].src0), header);
					if (pop != npos && m_inst[pop].opcode == Opcode::V_READLANE_B32 &&
					    m_inst[pop].src0.kind == OperandKind::Vgpr && m_inst[pop].src0.reg == inst.dst.reg) {
						layout.root_dword = field.dword;
						root              = true;
					}
				}
			}
		}
		// Instance leaves of another node type whose record is at a fixed offset of the header,
		// with a constant root: S_CMP_LG/EQ_U32 <type>, <node & 7> selecting
		// S_MOV_B64 <record pair>, <header + offset> (S_ADD_U32 / S_ADDC_U32 of the header pair).
		layout.inline_type = NoRegister;
		for (size_t i = header; i < record.load && root && inline_root != NoRegister; i++) {
			const auto& move = m_inst[i];
			if (move.opcode != Opcode::S_MOV_B64 || ScalarIndex(move.dst) != pair || ScalarIndex(move.src0) == NoRegister) {
				continue;
			}
			const auto source = ScalarIndex(move.src0);
			const auto lo     = ScalarWriter(i, source, 0);
			const auto hi     = ScalarWriter(i, source + 1, 0);
			if (lo == npos || hi == npos || m_inst[lo].opcode != Opcode::S_ADD_U32 || m_inst[hi].opcode != Opcode::S_ADDC_U32 ||
			    !IsSgpr(m_inst[lo].src0, site.header_sgpr) || m_inst[lo].src1.kind != OperandKind::LiteralConstant ||
			    !IsSgpr(m_inst[hi].src0, site.header_sgpr + 1) || !IsConstant(m_inst[hi].src1, 0)) {
				continue;
			}
			// The type compare before it.
			for (size_t c = i; c-- > header;) {
				const auto& cmp = m_inst[c];
				if (cmp.opcode != Opcode::S_CMP_LG_U32 && cmp.opcode != Opcode::S_CMP_EQ_U32) {
					continue;
				}
				const auto* type  = cmp.src0.kind == OperandKind::IntegerInlineConstant ? &cmp.src0 : &cmp.src1;
				const auto* value = type == &cmp.src0 ? &cmp.src1 : &cmp.src0;
				const auto  bits  = ScalarSource(c, *value, Opcode::S_AND_B32, header);
				if (type->kind == OperandKind::IntegerInlineConstant && type->value < 8 && bits != npos &&
				    IsConstant(m_inst[bits].src1, 7) && IsSgpr(m_inst[bits].src0, node)) {
					layout.inline_type   = type->value;
					layout.inline_offset = m_inst[lo].src1.value;
					layout.inline_root   = inline_root;
				}
				break;
			}
		}
		layout.flags_dword = NoRegister;
		layout.flags_shift = 0;
		layout.valid       = root;
		return root;
	}

	// The instance mask test of a per lane walk: (dword >> shift) & mask, mask a constant:
	// S_AND_B32 <s>, <dword>, <mask << shift>; S_CBRANCH_SCC0. A contiguous mask of more than 8 bits
	// gives its `width` (the instance counts when any of them is set) and mask 0xff.
	bool ConstantMaskTest(size_t i, size_t from, Field& field, uint32_t& shift, uint32_t& mask, uint32_t& width) const {
		width = 0;
		const auto& test = m_inst[i];
		const auto  next = NextReal(i);
		// Any bit from `shift` up: S_CMP_GT_U32 <dword>, (1 << shift) - 1; S_CBRANCH_SCC0, or
		// S_CMP_LE_U32 (S_CMPK_LE_U32) <dword>, (1 << shift) - 1; S_CBRANCH_SCC1. Below the top byte
		// that is a mask of 32 - shift bits.
		const bool constant = test.src1.kind == OperandKind::LiteralConstant ||
		                      test.src1.kind == OperandKind::IntegerInlineConstant;
		if (constant && next != npos &&
		    ((test.opcode == Opcode::S_CMP_GT_U32 && m_inst[next].opcode == Opcode::S_CBRANCH_SCC0) ||
		     (test.opcode == Opcode::S_CMP_LE_U32 && m_inst[next].opcode == Opcode::S_CBRANCH_SCC1)) &&
		    OperandField(i, test.src0, from, field)) {
			const auto below = test.src1.value;
			shift            = static_cast<uint32_t>(std::popcount(below));
			mask             = 0xffu;
			width            = shift < 24u ? 32u - shift : 0u;
			return (below & (below + 1u)) == 0 && shift >= 1u && shift < 32u;
		}
		if (test.opcode != Opcode::S_AND_B32 || test.src1.kind != OperandKind::LiteralConstant || test.src1.value == 0 ||
		    next == npos || m_inst[next].opcode != Opcode::S_CBRANCH_SCC0 || !OperandField(i, test.src0, from, field)) {
			return false;
		}
		shift = static_cast<uint32_t>(std::countr_zero(test.src1.value));
		mask  = test.src1.value >> shift;
		if (mask <= 0xffu) {
			return true;
		}
		width                 = static_cast<uint32_t>(std::popcount(mask));
		const bool contiguous = (mask & (mask + 1u)) == 0;
		mask                  = 0xffu;
		return contiguous;
	}

	bool SameRecord(const Field& field, const Field& record) const {
		return field.pair == record.pair &&
		       ScalarWriter(field.load, record.pair, 0) == ScalarWriter(record.load, record.pair, 0);
	}

	// Whether no instruction in [from, to] writes register `reg` (a VGPR or an SGPR).
	bool Keeps(size_t from, size_t to, uint32_t reg, bool vector) const {
		for (size_t i = from; i <= to && i < m_inst.size(); i++) {
			if (vector ? WritesVector(m_inst[i], reg) : WritesScalar(m_inst[i], reg)) {
				return false;
			}
		}
		return true;
	}

	// The instance mask test of a wave-wide walk's setup, V_CMPX_NE_U32 0, x at `i`, x the mask of
	// a record dword and of the ray: V_AND_B32 x, <dword (or byte k of it)>, <VGPR mask>, or
	// x = S_AND_B32 <S_LSHR_B32 dword, shift (or the dword)>, <SGPR mask or constant>. The ray's
	// mask keeps its value from the site entry.
	bool InstanceMaskTest(size_t i, const Operand& x, size_t header, size_t back, const Field& record,
	                      RayTracingSite& site) const {
		auto& layout = site.layout;
		if (x.kind == OperandKind::Vgpr) {
			const auto w = VectorWriter(i, x.reg, header);
			if (w == npos || m_inst[w].opcode != Opcode::V_AND_B32 || !Plain(m_inst[w].dst)) {
				return false;
			}
			for (const auto& [a, b]: {std::pair {&m_inst[w].src0, &m_inst[w].src1},
			                          std::pair {&m_inst[w].src1, &m_inst[w].src0}}) {
				Field      field;
				const auto reg = ScalarIndex(*a);
				if (b->kind != OperandKind::Vgpr || !Plain(*b) || reg == NoRegister || a->negate || a->absolute ||
				    (a->sdwa_sel > 3 && a->sdwa_sel != 6) || !RecordField(w, reg, header, field) ||
				    !SameRecord(field, record) || !Keeps(header, back, b->reg, true)) {
					continue;
				}
				site.cull_mask    = b->reg;
				layout.mask_dword = field.dword;
				layout.mask_shift = a->sdwa_sel == 6 ? 0u : a->sdwa_sel * 8u;
				return true;
			}
			return false;
		}
		const auto y = ScalarSource(i, x, Opcode::S_AND_B32, header);
		if (y == npos || !Plain(x)) {
			return false;
		}
		for (const auto& [a, b]: {std::pair {&m_inst[y].src0, &m_inst[y].src1}, std::pair {&m_inst[y].src1, &m_inst[y].src0}}) {
			Field    field;
			uint32_t shift = 0;
			if (!OperandField(y, *a, header, field)) {
				const auto lsr = ScalarSource(y, *a, Opcode::S_LSHR_B32, header);
				if (lsr == npos || m_inst[lsr].src1.kind != OperandKind::IntegerInlineConstant ||
				    !OperandField(lsr, m_inst[lsr].src0, header, field)) {
					continue;
				}
				shift = m_inst[lsr].src1.value;
			}
			if (!SameRecord(field, record) || shift > 24) {
				continue;
			}
			if (b->kind == OperandKind::Sgpr && Plain(*b) && Keeps(header, back, b->reg, false)) {
				site.cull_mask_sgpr = b->reg;
			} else if ((b->kind == OperandKind::IntegerInlineConstant || b->kind == OperandKind::LiteralConstant) &&
			           b->value <= 0xffu) {
				site.cull_mask_constant = b->value;
			} else {
				continue;
			}
			layout.mask_dword = field.dword;
			layout.mask_shift = shift;
			return true;
		}
		return false;
	}

	// ---- The facing a traversal's leaf code accepts. ----
	//
	// After the triangle test the leaf code keeps the lanes with a hit (first result not +inf)
	// whose facing is the accepted one, or with a hit when they do not cull:
	//   accept = hit & (facing | !cull)  with cull = !(flags bit 2) & enable
	// as the EXEC of the code that takes the hit (wave-wide walks of a Packet site) or as the
	// negated mask of the V_CNDMASK_B32s that keep the nearest hit, with the lanes beyond t max:
	//   reject = !exec | !accept | t > t max.
	// facing: V_CMP_EQ/NE of flags bit 3 (S_BFE_U32) and the sign of the first result
	// (V_LSHRREV_B32 31); NE accepts, with the bit clear, the triangles whose sign is set (the host's
	// back faces). The selector of the two cases is a lane mask: S_CSELECT_B64 exec, 0 of SCC, or
	// V_CMP_NE 0, x; it is cull or its negation (flags bit 2, both facings). The enable is every
	// lane, a wave-wide boolean (S_CSELECT_B32 1, 0) or a lane mask (V_CNDMASK_B32 0, 1, <mask>)
	// that keep their values from the site entry.

	static constexpr size_t ExecAtom = SIZE_MAX - 1;

	// A lane mask (an SGPR pair) as a truth table over up to five atoms, the instructions that
	// write them or EXEC (bit c of the table: its value with atom k = bit k of c), made of
	// S_AND_B64, S_ANDN2_B64 and S_OR_B64.
	bool MaskTable(size_t i, const Operand& operand, size_t from, std::array<size_t, 5>& atoms, uint32_t& table,
	               uint32_t depth = 0) const {
		static constexpr std::array<uint32_t, 5> AtomTables {0xaaaaaaaau, 0xccccccccu, 0xf0f0f0f0u, 0xff00ff00u,
		                                                     0xffff0000u};
		const auto reg = ScalarIndex(operand);
		size_t     w   = operand.kind == OperandKind::ExecLo ? ExecAtom : reg != NoRegister ? ScalarWriter(i, reg, from) : npos;
		if (w == npos || depth > 8) {
			return false;
		}
		if (w != ExecAtom) {
			const auto& inst = m_inst[w];
			const auto op = inst.opcode;
			if (op == Lane(Opcode::S_NOT_B64) && ScalarIndex(inst.dst) == reg) {
				uint32_t a = 0;
				if (!MaskTable(w, inst.src0, from, atoms, a, depth + 1)) {
					return false;
				}
				table = ~a;
				return true;
			}
			if ((op == Lane(Opcode::S_AND_B64) || op == Lane(Opcode::S_ANDN2_B64) || op == Lane(Opcode::S_OR_B64) ||
			     op == Lane(Opcode::S_ORN2_B64) || op == Lane(Opcode::S_NOR_B64) || op == Lane(Opcode::S_NAND_B64)) &&
			    ScalarIndex(inst.dst) == reg) {
				uint32_t a = 0, b = 0;
				if (!MaskTable(w, inst.src0, from, atoms, a, depth + 1) || !MaskTable(w, inst.src1, from, atoms, b, depth + 1)) {
					return false;
				}
				table = op == Lane(Opcode::S_AND_B64)     ? a & b
				        : op == Lane(Opcode::S_ANDN2_B64) ? a & ~b
				        : op == Lane(Opcode::S_OR_B64)    ? a | b
				        : op == Lane(Opcode::S_ORN2_B64)  ? a | ~b
				        : op == Lane(Opcode::S_NOR_B64)   ? ~(a | b)
				                                          : ~(a & b);
				return true;
			}
		}
		for (uint32_t k = 0; k < atoms.size(); k++) {
			if (atoms[k] == w || atoms[k] == npos) {
				atoms[k] = w;
				table    = AtomTables[k];
				return true;
			}
		}
		return false;
	}

	// The facing compare of a triangle test's leaf code (see above) at `i`: whether it is one, and
	// whether it accepts with NE (flip). Gives the flags dword and shift of its bit 3.
	bool FacingCompare(size_t i, size_t test, size_t from, Field& flags, uint32_t& bit, bool& ne) const {
		const auto& cmp = m_inst[i];
		ne              = cmp.opcode == Opcode::V_CMP_NE_U32 || cmp.opcode == Opcode::V_CMP_NE_I32;
		if (!ne && cmp.opcode != Opcode::V_CMP_EQ_U32 && cmp.opcode != Opcode::V_CMP_EQ_I32) {
			return false;
		}
		for (const auto& [a, b]: {std::pair {&cmp.src0, &cmp.src1}, std::pair {&cmp.src1, &cmp.src0}}) {
			if (b->kind != OperandKind::Vgpr || !Plain(*a) || !Plain(*b)) {
				continue;
			}
			const auto sign = VectorSource(i, *b, Opcode::V_LSHRREV_B32, from);
			if (sign == npos || !IsConstant(m_inst[sign].src0, 31) || m_inst[sign].src1.kind != OperandKind::Vgpr ||
			    VectorWriter(sign, m_inst[sign].src1.reg, from) != test || m_inst[test].dst.reg != m_inst[sign].src1.reg) {
				continue;
			}
			const auto bfe = ScalarSource(i, *a, Opcode::S_BFE_U32, from);
			if (bfe != npos && m_inst[bfe].src1.kind == OperandKind::LiteralConstant &&
			    (m_inst[bfe].src1.value >> 16u) == 1u && (m_inst[bfe].src1.value & 0xffffu) >= 3u &&
			    OperandField(bfe, m_inst[bfe].src0, from, flags)) {
				bit = m_inst[bfe].src1.value & 0xffffu;
				return true;
			}
			// Or S_LSHR_B32 <flags dword & mask>, <bit>, with no bit of the mask above it.
			const auto lsr  = ScalarSource(i, *a, Opcode::S_LSHR_B32, from);
			uint32_t   mask = 0;
			if (lsr != npos && m_inst[lsr].src1.kind == OperandKind::IntegerInlineConstant &&
			    m_inst[lsr].src1.value >= 3u && m_inst[lsr].src1.value < 32u &&
			    MaskedField(lsr, m_inst[lsr].src0, from, flags, mask) && (mask >> m_inst[lsr].src1.value) == 1u) {
				bit = m_inst[lsr].src1.value;
				return true;
			}
		}
		return false;
	}

	// The facing condition of the mask SGPR `reg` read at `at` (see above), from the triangle
	// test `test`; `reject`: the mask is the reject form. Sets the layout's flags (when the site
	// has none yet) and flip, and the site's facing enable.
	bool LeafFacingTable(size_t test, size_t at, size_t from, size_t back, uint32_t reg, bool reject, RayTracingSite& site) const {
		Operand mask;
		mask.kind = OperandKind::Sgpr;
		mask.reg  = reg;
		if (reg == 106u) {
			mask.kind = OperandKind::VccLo;
		}
		std::array<size_t, 5> atoms {npos, npos, npos, npos, npos};
		uint32_t              table = 0;
		if (!MaskTable(at, mask, from, atoms, table)) {
			return false;
		}
		// What each atom is.
		uint32_t facing = 5, hit = 5, selector = 5, far = 5, exec = 5;
		Field    flags;
		uint32_t bit = 0;
		bool     ne  = false;
		for (uint32_t k = 0; k < atoms.size() && atoms[k] != npos; k++) {
			if (atoms[k] == ExecAtom) {
				exec = k;
				continue;
			}
			const auto& c = m_inst[atoms[k]];
			Field       f;
			uint32_t    b = 0;
			bool        n = false;
			if (facing == 5 && FacingCompare(atoms[k], test, from, f, b, n)) {
				facing = k, flags = f, bit = b, ne = n;
				continue;
			}
			const bool compare = c.family == Family::VOPC || magic_enum::enum_name(c.opcode).starts_with("V_CMP_");
			const auto reads   = [&](uint32_t vgpr) {
				return (c.src0.kind == OperandKind::Vgpr && c.src0.reg == vgpr) ||
				       (c.src1.kind == OperandKind::Vgpr && c.src1.reg == vgpr);
			};
			if (compare && hit == 5 && reads(m_inst[test].dst.reg) && VectorWriter(atoms[k], m_inst[test].dst.reg, from) == test) {
				hit = k;
			} else if (compare && far == 5 && c.opcode == Opcode::V_CMP_GT_F32 && reads(site.t_max)) {
				far = k;
			} else if (selector == 5) {
				selector = k;
			} else {
				return false;
			}
		}
		if (facing == 5 || hit == 5 || selector == 5 || (!reject && exec != 5)) {
			return false;
		}
		// Lanes outside EXEC do not count (the V_CNDMASK_B32s do not write them).
		const auto value = [&](uint32_t k, uint32_t c) { return k < 5 && ((c >> k) & 1u) != 0; };
		uint32_t   both_table = 0, cull_table = 0, care = 0;
		for (uint32_t c = 0; c < 32; c++) {
			const bool f = value(facing, c), h = value(hit, c), s = value(selector, c);
			const bool g = value(far, c);
			const bool both = h && (f || s), cull = h && (f || !s);
			care |= exec == 5 || value(exec, c) ? 1u << c : 0u;
			both_table |= (reject ? (!both || g) : (both && !g)) ? 1u << c : 0u;
			cull_table |= (reject ? (!cull || g) : (cull && !g)) ? 1u << c : 0u;
		}
		const bool is_both = ((table ^ both_table) & care) == 0, is_cull = ((table ^ cull_table) & care) == 0;
		if (is_both == is_cull) {
			return false;
		}
		// The flags (the opacity test may have given them already).
		auto& layout = site.layout;
		if (layout.flags_dword == NoRegister) {
			layout.flags_dword = flags.dword;
			layout.flags_shift = bit - 3u;
		} else if (flags.dword != layout.flags_dword || bit != layout.flags_shift + 3u) {
			return false;
		}
		// The selector's condition, as cull = !(flags bit 2) & enable.
		const auto& sel = m_inst[atoms[selector]];
		Condition   cond;
		if (sel.opcode == Lane(Opcode::S_CSELECT_B64) && sel.src0.kind == OperandKind::ExecLo && IsConstant(sel.src1, 0)) {
			if (!SccCondition(atoms[selector], from, layout, back, cond, 0)) {
				return false;
			}
		} else if (sel.opcode == Opcode::V_CMP_NE_U32 || sel.opcode == Opcode::V_CMP_NE_I32) {
			const auto* x = IsConstant(sel.src0, 0) ? &sel.src1 : IsConstant(sel.src1, 0) ? &sel.src0 : nullptr;
			if (x == nullptr || !LaneCondition(atoms[selector], *x, from, layout, back, cond)) {
				return false;
			}
		} else {
			return false;
		}
		if (is_both) {
			// The selector is "both facings": only flags bit 2 negates.
			if (cond.both != 1 || cond.uniform != NoRegister || cond.lanes != NoRegister) {
				return false;
			}
			cond.both = -1;
		}
		if (cond.both != -1) {
			return false;
		}
		site.facing_sgpr   = cond.uniform;
		site.facing_mask   = cond.lanes;
		layout.flip_facing = ne;
		return true;
	}

	// A condition that is a conjunction of: flags bit 2 (or its negation), a wave-wide boolean
	// SGPR, a lane mask SGPR pair.
	struct Condition {
		int      both    = 0; // 1: flags bit 2 set, -1: clear, 0: not a term.
		uint32_t uniform = NoRegister;
		uint32_t lanes   = NoRegister;
	};

	static bool Conjoin(Condition& a, const Condition& b) {
		if ((a.both != 0 && b.both != 0 && a.both != b.both) || (a.uniform != NoRegister && b.uniform != NoRegister) ||
		    (a.lanes != NoRegister && b.lanes != NoRegister)) {
			return false;
		}
		a.both    = a.both != 0 ? a.both : b.both;
		a.uniform = a.uniform != NoRegister ? a.uniform : b.uniform;
		a.lanes   = a.lanes != NoRegister ? a.lanes : b.lanes;
		return true;
	}

	// Whether the instruction may write SCC.
	static bool WritesScc(const Instruction& inst) {
		switch (inst.family) {
			case Family::SOPC: return true;
			case Family::SOP2: return inst.opcode != Opcode::S_CSELECT_B32 && inst.opcode != Opcode::S_CSELECT_B64;
			case Family::SOP1: return inst.opcode != Opcode::S_MOV_B32 && inst.opcode != Opcode::S_MOV_B64;
			case Family::SOPK: return inst.opcode != Opcode::S_MOVK_I32;
			default: return false;
		}
	}

	// SCC at instruction `i`, from the S_CMP that last wrote it: x != 0 or x == 0 of a value
	// `Bits` takes.
	bool SccCondition(size_t i, size_t from, const RayTracingLayout& layout, size_t back, Condition& cond,
	                  uint32_t depth) const {
		size_t cmp = npos;
		for (size_t j = i; j-- > from && cmp == npos;) {
			cmp = WritesScc(m_inst[j]) ? j : npos;
		}
		if (cmp == npos) {
			return false;
		}
		const auto& inst = m_inst[cmp];
		const bool  ne   = inst.opcode == Opcode::S_CMP_LG_U32;
		if (!ne && inst.opcode != Opcode::S_CMP_EQ_U32) {
			return false;
		}
		const auto* x = IsConstant(inst.src0, 0) ? &inst.src1 : IsConstant(inst.src1, 0) ? &inst.src0 : nullptr;
		if (x == nullptr || !BitsCondition(cmp, *x, from, layout, back, cond, depth + 1)) {
			return false;
		}
		if (!ne) {
			// Only flags bit 2 alone negates.
			if (cond.uniform != NoRegister || cond.lanes != NoRegister || cond.both == 0) {
				return false;
			}
			cond.both = -cond.both;
		}
		return true;
	}

	// The condition `x != 0` of an SGPR (or SCC) value `x` read by instruction `i`.
	bool BitsCondition(size_t i, const Operand& x, size_t from, const RayTracingLayout& layout, size_t back,
	                   Condition& cond, uint32_t depth) const {
		if (depth > 6) {
			return false;
		}
		if (x.kind == OperandKind::Scc) {
			return SccCondition(i, from, layout, back, cond, depth + 1);
		}
		const auto reg = ScalarIndex(x);
		const auto w   = reg != NoRegister ? ScalarWriter(i, reg, from) : npos;
		if (w == npos) {
			// A value from before the loop: a wave-wide boolean (S_CSELECT_B32 1, 0).
			const auto before = reg != NoRegister ? ScalarWriter(i, reg, 0) : npos;
			if (before == npos || m_inst[before].opcode != Opcode::S_CSELECT_B32 || !IsConstant(m_inst[before].src0, 1) ||
			    !IsConstant(m_inst[before].src1, 0) || !Keeps(from, back, reg, false)) {
				return false;
			}
			cond = {};
			cond.uniform = reg;
			return true;
		}
		const auto& inst = m_inst[w];
		if (inst.opcode != Opcode::S_AND_B32 || ScalarIndex(inst.dst) != reg) {
			return false;
		}
		// Flags bit 2: S_AND_B32 <x>, <flags dword (masked)>, 1 << (shift + 2).
		const uint32_t bit = 1u << (layout.flags_shift + 2u);
		for (const auto& [a, b]: {std::pair {&inst.src0, &inst.src1}, std::pair {&inst.src1, &inst.src0}}) {
			Field    field;
			uint32_t mask = 0;
			if (IsConstant(*b, bit) && MaskedField(w, *a, from, field, mask) && (mask & bit) != 0 &&
			    field.dword == layout.flags_dword) {
				cond      = {};
				cond.both = 1;
				return true;
			}
		}
		Condition a, b;
		if (!BitsCondition(w, inst.src0, from, layout, back, a, depth + 1) ||
		    !BitsCondition(w, inst.src1, from, layout, back, b, depth + 1) || !Conjoin(a, b)) {
			return false;
		}
		cond = a;
		return true;
	}

	// The condition of lanes whose VGPR `x` (read by instruction `i`) is not 0.
	bool LaneCondition(size_t i, const Operand& x, size_t from, const RayTracingLayout& layout, size_t back,
	                   Condition& cond) const {
		const auto w = x.kind == OperandKind::Vgpr && Plain(x) ? VectorWriter(i, x.reg, from) : npos;
		if (w == npos) {
			return false;
		}
		const auto& inst = m_inst[w];
		if (inst.opcode == Opcode::V_CNDMASK_B32 && IsConstant(inst.src0, 0) && IsConstant(inst.src1, 1) &&
		    inst.src2.kind == OperandKind::Sgpr && Keeps(from, back, inst.src2.reg, false) &&
		    Keeps(from, back, inst.src2.reg + 1, false)) {
			// A lane mask from before the loop.
			if (ScalarWriter(w, inst.src2.reg, from) != npos) {
				return false;
			}
			cond       = {};
			cond.lanes = inst.src2.reg;
			return true;
		}
		if (inst.opcode != Opcode::V_AND_B32 || !Plain(inst.dst)) {
			return false;
		}
		Condition a, b;
		const auto part = [&](const Operand& operand, Condition& c) {
			return operand.kind == OperandKind::Vgpr ? LaneCondition(w, operand, from, layout, back, c)
			                                         : BitsCondition(w, operand, from, layout, back, c, 0);
		};
		if (!part(inst.src0, a) || !part(inst.src1, b) || !Conjoin(a, b)) {
			return false;
		}
		cond = a;
		return true;
	}

	// The facing of a Packet site's leaf code: the EXEC the code that takes a hit runs with (the
	// first S_AND_SAVEEXEC_B64 after a triangle test that depends on a facing compare).
	bool LeafFacing(size_t leaf, size_t tail, size_t from, size_t back, RayTracingSite& site) const {
		for (size_t test = leaf; test < tail; test++) {
			if (m_inst[test].opcode != Opcode::IMAGE_BVH_INTERSECT_RAY) {
				continue;
			}
			const auto save = Next(test, Opcode::S_AND_SAVEEXEC_B64, tail);
			if (save != npos && ScalarIndex(m_inst[save].src0) != NoRegister &&
			    LeafFacingTable(test, save, from, back, ScalarIndex(m_inst[save].src0), false, site)) {
				return true;
			}
		}
		return false;
	}
	// The flags of the instance a Packet site's leaf code reads: the opacity test
	// S_OR_B32 <x>, <flags>, <word >> 30>; S_BITCMP1_B32 <x>, 0 with flags a record dword, shifted
	// right or not.
	bool LeafFlags(size_t leaf, size_t tail, size_t from, uint32_t word_sgpr, const Field& record,
	               RayTracingLayout& layout) const {
		auto& dword = layout.flags_dword;
		auto& shift = layout.flags_shift;
		for (size_t i = leaf; i < tail; i++) {
			const auto& bit = m_inst[i];
			if (bit.opcode != Opcode::S_BITCMP1_B32 || !IsConstant(bit.src1, 0)) {
				continue;
			}
			const auto bits = ScalarSource(i, bit.src0, Opcode::S_OR_B32, leaf);
			if (bits == npos) {
				continue;
			}
			for (const auto& [a, b]: {std::pair {&m_inst[bits].src0, &m_inst[bits].src1},
			                          std::pair {&m_inst[bits].src1, &m_inst[bits].src0}}) {
				const auto word = ScalarSource(bits, *a, Opcode::S_LSHR_B32, leaf);
				if (!IsScalarConstantOp(word, Opcode::S_LSHR_B32, 30) || ScalarIndex(m_inst[word].src0) != word_sgpr) {
					continue;
				}
				Field field;
				if (OperandField(bits, *b, from, field) && SameRecord(field, record)) {
					dword = field.dword;
					shift = 0;
					return true;
				}
				const auto shifted = ScalarSource(bits, *b, Opcode::S_LSHR_B32, from);
				if (shifted != npos && m_inst[shifted].src1.kind == OperandKind::IntegerInlineConstant &&
				    OperandField(shifted, m_inst[shifted].src0, from, field) && SameRecord(field, record)) {
					dword = field.dword;
					shift = m_inst[shifted].src1.value;
					return true;
				}
			}
		}
		return false;
	}

	// The mask operand of V_CNDMASK_B32 (VCC without a third operand).
	static bool IsConditionalBranch(Opcode opcode) {
		switch (opcode) {
			case Opcode::S_CBRANCH_SCC0:
			case Opcode::S_CBRANCH_SCC1:
			case Opcode::S_CBRANCH_VCCZ:
			case Opcode::S_CBRANCH_VCCNZ:
			case Opcode::S_CBRANCH_EXECZ:
			case Opcode::S_CBRANCH_EXECNZ: return true;
			default: return false;
		}
	}

	static Operand CndmaskMask(const Instruction& inst) {
		if (inst.src_count >= 3u) {
			return inst.src2;
		}
		Operand operand;
		operand.kind = OperandKind::VccLo;
		return operand;
	}

	// The opcode of a lane mask operation in the program's waves: `op64`, or its 32-bit form in
	// waves of 32 lanes.
	Opcode Lane(Opcode op64) const {
		if (!m_wave32) {
			return op64;
		}
		switch (op64) {
			case Opcode::S_AND_B64: return Opcode::S_AND_B32;
			case Opcode::S_ANDN2_B64: return Opcode::S_ANDN2_B32;
			case Opcode::S_OR_B64: return Opcode::S_OR_B32;
			case Opcode::S_ORN2_B64: return Opcode::S_ORN2_B32;
			case Opcode::S_NOR_B64: return Opcode::S_NOR_B32;
			case Opcode::S_NAND_B64: return Opcode::S_NAND_B32;
			case Opcode::S_NOT_B64: return Opcode::S_NOT_B32;
			case Opcode::S_MOV_B64: return Opcode::S_MOV_B32;
			case Opcode::S_CSELECT_B64: return Opcode::S_CSELECT_B32;
			case Opcode::S_AND_SAVEEXEC_B64: return Opcode::S_AND_SAVEEXEC_B32;
			default: return op64;
		}
	}

	// An SGPR pair the program never reads or writes (NoRegister: none).
	uint32_t UnusedPair() const {
		std::array<bool, 106> used {};
		const auto mark = [&](const Operand& operand, uint32_t count) {
			const auto reg = operand.kind == OperandKind::Sgpr ? operand.reg : NoRegister;
			for (uint32_t k = 0; reg != NoRegister && k < count && reg + k < used.size(); k++) {
				used[reg + k] = true;
			}
		};
		for (const auto& inst: m_inst) {
			// Conservatively wide: whole descriptors and pairs. A scalar memory access reads its
			// base pair or buffer descriptor and an offset, an image access its resource (eight
			// SGPRs) and sampler (four).
			if (inst.family == Family::SMEM) {
				mark(inst.src0, 4u);
				mark(inst.src1, 1u);
			} else if (inst.family == Family::MIMG) {
				mark(inst.src0, 2u);
				mark(inst.src1, 8u);
				mark(inst.src2, 4u);
				mark(inst.src3, 2u);
			} else {
				const uint32_t width = inst.family == Family::MUBUF || inst.family == Family::MTBUF ? 4u : 2u;
				for (const auto* operand: {&inst.src0, &inst.src1, &inst.src2, &inst.src3}) {
					mark(*operand, width);
				}
			}
			mark(inst.dst, inst.family == Family::SMEM ? std::max(inst.data_dwords, 1u) : 2u);
			mark(inst.dst2, 2u);
		}
		for (uint32_t pair = 104; pair >= 2; pair -= 2) {
			if (!used[pair] && !used[pair + 1]) {
				return pair;
			}
		}
		return NoRegister;
	}

	// The nearest hit of a closest hit traversal whose box and leaf walks are wave-wide (scalar
	// nodes; Astro Bot): the leaf code keeps the nearest hit in VGPRs,
	//   V_CNDMASK_B32 <t max>, <t>, <t max>, <reject>   V_CNDMASK_B32 <word>, <entry word>, <word>, <reject>
	// with reject = !exec | !(hit & (facing | !cull)) | t > t max (facing and cull as in LeafFacing;
	// every triangle opaque). The site traces with the hardware and commits t and the word.
	bool ClosestSelectHit(size_t header, size_t back, uint32_t record_pair, RayTracingSite& site) const {
		for (size_t i = header; i < back; i++) {
			const auto& keep_t = m_inst[i];
			const auto  mask_t = CndmaskMask(keep_t);
			if (keep_t.opcode != Opcode::V_CNDMASK_B32 || keep_t.dst.kind != OperandKind::Vgpr ||
			    keep_t.dst.reg != site.t_max || keep_t.src1.kind != OperandKind::Vgpr || keep_t.src1.reg != site.t_max ||
			    keep_t.src0.kind != OperandKind::Vgpr || !Plain(keep_t.src0) || !Plain(keep_t.src1) ||
			    ScalarIndex(mask_t) == NoRegister) {
				continue;
			}
			const auto reject = ScalarWriter(i, ScalarIndex(mask_t), header);
			// The word kept with the same mask (before or after t).
			size_t keep_word = npos;
			for (size_t j = header + 1; j < back && keep_word == npos; j++) {
				const auto& keep = m_inst[j];
				const auto  mask = CndmaskMask(keep);
				if (j != i && keep.opcode == Opcode::V_CNDMASK_B32 && keep.dst.kind == OperandKind::Vgpr &&
				    keep.src1.kind == OperandKind::Vgpr && keep.src1.reg == keep.dst.reg && ScalarIndex(mask) != NoRegister &&
				    ScalarWriter(j, ScalarIndex(mask), header) == reject && keep.dst.reg != site.t_max) {
					keep_word = j;
				}
			}
			if (keep_word == npos) {
				continue;
			}
			// The triangle test of the leaf (its first result is the hit's numerator).
			const auto t_value = keep_t.src0.reg;
			const auto mul     = VectorWriter(i, t_value, header);
			if (mul == npos || m_inst[mul].opcode != Opcode::V_MUL_F32) {
				return false;
			}
			size_t test = npos;
			for (const auto* operand: {&m_inst[mul].src0, &m_inst[mul].src1}) {
				const auto w = operand->kind == OperandKind::Vgpr ? VectorWriter(mul, operand->reg, header) : npos;
				if (w != npos && m_inst[w].opcode == Opcode::IMAGE_BVH_INTERSECT_RAY && m_inst[w].dst.reg == operand->reg) {
					test = w;
				}
			}
			if (test == npos) {
				return false;
			}
			if (!LeafFacingTable(test, i, header, back, ScalarIndex(mask_t), true, site)) {
				return false;
			}
			site.result_t        = site.t_max;
			site.result_word     = m_inst[keep_word].dst.reg;
			site.result_instance = NoRegister;
			site.result_u = site.result_v = site.result_node = NoRegister;
			return true;
		}
		// Or the moves of the nearest hit under EXEC = accept & !(t > t max):
		//   S_AND_SAVEEXEC_B64 <s>, <accept>; S_CBRANCH_EXECZ; V_MOV_B32s of t into t max, of the
		//   instance record pair and of the list entry (triangle node, primitive word).
		// The far test can also come after the S_AND_SAVEEXEC_B64, with t computed there:
		//   V_RCP_F32 / V_MUL_F32 of t; V_CMPX_LE_F32 <t>, <t max>; S_CBRANCH_EXECZ; the moves.
		for (size_t i = header; i + 2 < back; i++) {
			const auto& save = m_inst[i];
			if (save.opcode != Lane(Opcode::S_AND_SAVEEXEC_B64) || m_inst[i + 1].opcode != Opcode::S_CBRANCH_EXECZ ||
			    ScalarIndex(save.src0) == NoRegister) {
				continue;
			}
			size_t first = i + 2, far_test = npos;
			for (size_t j = first; j + 1 < back && j < i + 6; j++) {
				const auto& c = m_inst[j];
				if (c.opcode == Opcode::V_CMPX_LE_F32 && m_inst[j + 1].opcode == Opcode::S_CBRANCH_EXECZ &&
				    c.src0.kind == OperandKind::Vgpr && c.src1.kind == OperandKind::Vgpr && c.src1.reg == site.t_max &&
				    c.src0.reg != site.t_max && Plain(c.src0) && Plain(c.src1)) {
					far_test = j;
					first    = j + 2;
					break;
				}
				if ((c.opcode != Opcode::V_RCP_F32 && c.opcode != Opcode::V_MUL_F32) || c.dst.kind != OperandKind::Vgpr ||
				    c.dst.reg == site.t_max) {
					break;
				}
			}
			uint32_t t_value = NoRegister, instance = NoRegister, entry = NoRegister, node = NoRegister, word = NoRegister;
			bool     ok      = true;
			const auto waits = [&](size_t j) {
				return m_inst[j].opcode == Opcode::S_WAITCNT || m_inst[j].opcode == Opcode::S_NOP;
			};
			for (size_t j = first; j < back && (m_inst[j].opcode == Opcode::V_MOV_B32 || waits(j)) && ok; j++) {
				const auto& move = m_inst[j];
				if (waits(j)) {
					continue;
				}
				if (move.dst.kind != OperandKind::Vgpr || !Plain(move.src0)) {
					ok = false;
				} else if (move.dst.reg == site.t_max && move.src0.kind == OperandKind::Vgpr) {
					t_value = move.src0.reg;
				} else if (move.src0.kind == OperandKind::Sgpr && move.src0.reg == record_pair) {
					instance = instance == NoRegister ? move.dst.reg : NoRegister;
				} else if (move.src0.kind == OperandKind::Sgpr && move.src0.reg == record_pair + 1) {
					ok = move.dst.reg == instance + 1;
				} else if (move.src0.kind == OperandKind::Sgpr && (entry == NoRegister || (move.src0.reg | 1u) == (entry | 1u))) {
					// The two dwords of the list entry: the triangle node, then the word.
					entry = std::min(entry, move.src0.reg);
					((move.src0.reg & 1u) == 0 ? node : word) = move.dst.reg;
				} else {
					ok = false;
				}
			}
			if (!ok || t_value == NoRegister || instance == NoRegister || node == NoRegister || word == NoRegister ||
			    (far_test != npos && m_inst[far_test].src0.reg != t_value)) {
				continue;
			}
			const auto mul = VectorWriter(far_test != npos ? far_test : i, t_value, header);
			size_t     test = npos;
			for (const auto* operand: {&m_inst[mul].src0, &m_inst[mul].src1}) {
				const auto w = mul != npos && operand->kind == OperandKind::Vgpr ? VectorWriter(mul, operand->reg, header) : npos;
				if (w != npos && m_inst[w].opcode == Opcode::IMAGE_BVH_INTERSECT_RAY && m_inst[w].dst.reg == operand->reg) {
					test = w;
				}
			}
			if (mul == npos || m_inst[mul].opcode != Opcode::V_MUL_F32 || test == npos ||
			    !LeafFacingTable(test, i, header, back, ScalarIndex(save.src0), false, site)) {
				return false;
			}
			site.result_t        = site.t_max;
			site.result_word     = word;
			site.result_instance = instance;
			site.result_node     = node;
			site.result_u = site.result_v = NoRegister;
			return true;
		}
		return false;
	}

	const std::vector<Instruction>& m_inst;
	// Lane masks are one SGPR (EXEC_LO, VCC_LO) instead of a pair.
	const bool m_wave32;
};

} // namespace

bool DeriveRayTracingPatch(const Program& program, RayTracingPatch& patch, uint32_t wave_size) {
	if (program.instructions.empty() || (wave_size != 32 && wave_size != 64)) {
		return false;
	}
	SiteFinder finder(program, wave_size);
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
	                              .layout        = patch.layout.Pack(),
	                              .valid         = true};
	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler::Decoder
