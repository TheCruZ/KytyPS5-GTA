#include "graphics/shader/recompiler/frontend/decode/RayTracingPatch.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cinttypes>
#include <cstddef>
#include <ranges>
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
	if (site.kind == RayTracingKind::ClosestOpaque || site.kind == RayTracingKind::PacketLane) {
		return entry != nullptr && EndPc(*entry) == site.loop_pc && !IsDirectBranch(entry->opcode) &&
		       expect(site.loop_pc, Opcode::UNKNOWN) && expect(site.exit_pc, Opcode::UNKNOWN) &&
		       (site.header_load_pc == 0 || expect(site.header_load_pc, Opcode::S_LOAD_DWORDX2)) &&
		       (site.alpha_off_load_pc == 0 || expect(site.alpha_off_load_pc, Opcode::V_READLANE_B32)) &&
		       (site.cull_mask_load_pc == 0 || expect(site.cull_mask_load_pc, Opcode::V_READLANE_B32));
	}
	if (site.kind == RayTracingKind::ClosestLane &&
	    (site.leaf_vgpr == NoRegister || !expect(site.leaf_load_pc, Opcode::V_LSHRREV_B32) ||
	     site.leaf_load_end_pc <= site.leaf_load_pc)) {
		return false;
	}
	if (site.kind == RayTracingKind::AnyLane) {
		return entry != nullptr && EndPc(*entry) == site.loop_pc && !IsDirectBranch(entry->opcode) &&
		       expect(site.loop_pc, Opcode::UNKNOWN) && expect(site.setup_pc, Opcode::UNKNOWN) &&
		       expect(site.cull_exit_pc, Opcode::UNKNOWN) && expect(site.blas_head_pc, Opcode::UNKNOWN) &&
		       site.leaf_pc == site.blas_head_pc && expect(site.tail_pc, Opcode::UNKNOWN) &&
		       site.blas_end_pc > site.tail_pc && expect(site.exit_pc, Opcode::UNKNOWN) &&
		       site.leaf_vgpr != NoRegister && expect(site.leaf_load_pc, Opcode::V_LSHRREV_B32) &&
		       site.leaf_load_end_pc > site.leaf_load_pc && expect(site.mask_copy_pc, Opcode::UNKNOWN) &&
		       (site.after_leaf_pc == 0 || expect(site.after_leaf_pc, Opcode::V_READLANE_B32));
	}
	if (site.kind == RayTracingKind::ClosestLeaf || site.kind == RayTracingKind::ClosestLane) {
		return entry != nullptr && EndPc(*entry) == site.loop_pc && !IsDirectBranch(entry->opcode) &&
		       site.setup_pc == site.loop_pc && expect(site.loop_pc, Opcode::UNKNOWN) &&
		       expect(site.cull_exit_pc, Opcode::UNKNOWN) && expect(site.blas_head_pc, Opcode::UNKNOWN) &&
		       site.leaf_pc == site.blas_head_pc && expect(site.tail_pc, Opcode::UNKNOWN) &&
		       site.blas_end_pc > site.tail_pc && expect(site.exit_pc, Opcode::UNKNOWN) &&
		       (site.after_leaf_pc == 0 || expect(site.after_leaf_pc, Opcode::V_READLANE_B32));
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
		// The pc of the guest instruction it places (UINT32_MAX: an inserted one).
		uint32_t    origin = UINT32_MAX;
	};

	uint32_t NewLabel() { return m_next_label++; }
	// The label refers to the next item added.
	void Bind(uint32_t label) { m_bindings.emplace_back(label, m_items.size()); }

	Instruction& AddGuest(const Instruction& inst) {
		Item item {inst};
		item.origin = inst.pc;
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
		m_items.back().origin = inst.pc;
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

	// Assigns the pcs from `first_pc`; false when a target is unknown. `origins`: the guest pc each
	// output instruction places (UINT32_MAX: inserted).
	bool Finish(uint32_t first_pc, uint32_t guest_end_pc, std::vector<Instruction>& out, std::vector<uint32_t>& origins) {
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
		origins.clear();
		origins.reserve(m_items.size());
		for (size_t i = 0; i < m_items.size(); i++) {
			origins.push_back(m_items[i].origin);
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
		// The end of an inlined call whose target is none of the expanded ones (S_TRAP; S_ENDPGM,
		// see ShaderFunctionExpander) is copied as it is.
		const bool call_miss = it->opcode == Opcode::S_ENDPGM && it != first && std::prev(it)->opcode == Opcode::S_TRAP;
		if (it->opcode == Opcode::S_SETPC_B64 || it->opcode == Opcode::S_GETPC_B64 ||
		    (it->opcode == Opcode::S_ENDPGM && !call_miss) || it->opcode == Opcode::UNSUPPORTED) {
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
		// A loop of an inlined call stays a loop.
		const bool in_call = std::ranges::any_of(program.call_bodies, [&](const InlinedCallBody& body) {
			return body.begin <= target && inst->pc < body.end;
		});
		if (backward_to_end && target <= inst->pc && copy_labels.contains(target) && !in_call) {
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

// The ray registers of RT_INIT: the origin (VGPRs, or SGPRs in [9..11]) and the direction.
void SetRay(Instruction& init, const RayTracingSite& site) {
	for (uint32_t i = 0; i < 3; i++) {
		init.image_nsa_addr[i]     = site.origin[i];
		init.image_nsa_addr[3 + i] = site.direction[i];
		if (site.origin_sgpr[i] != NoRegister) {
			init.image_nsa_addr[i]     = NoRegister;
			init.image_nsa_addr[9 + i] = site.origin_sgpr[i];
		}
	}
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
	if (test.layout == RayTracingAlphaLayout::Entry) {
		// No level of detail term, spread, factor nor reference; the V#s are SGPRs of the program.
		alpha.image_nsa_addr[0] = NoRegister;
		alpha.image_nsa_addr[5] = NoRegister;
		alpha.src0              = Constant(0);
		alpha.src2              = Sgpr(test.material_user);
		alpha.src3              = Sgpr(test.buffer_user);
		alpha.dmask             = 16u;
		return;
	}
	alpha.src0              = test.spread_vector ? Vgpr(test.spread_reg) : Sgpr(test.spread_reg);
	// The threshold and its factor (the Lane layout without a reference: dmask 2), or the
	// reference SGPR (dmask 4).
	const bool uniform = lane && test.reference_sgpr != NoRegister;
	const bool factor  = !lane || (test.reference_vgpr == NoRegister && !uniform);
	if (factor) {
		alpha.src1 = Sgpr(test.factor_sgpr);
	} else if (uniform) {
		alpha.src1 = Sgpr(test.reference_sgpr);
	}
	alpha.src2  = Sgpr(test.material_user);
	alpha.src3  = Sgpr(test.buffer_user);
	alpha.dmask = static_cast<uint32_t>(test.layout) | (lane && factor ? 2u : 0u) | (uniform ? 4u : 0u) |
	              (factor && test.factor_always ? 8u : 0u);
}

// The native traversal of a closest hit site: the hardware finds the hit (every triangle opaque,
// both facings); no guest code runs for it. A PacketLane site's (the opacity and the facing of
// the instance flags, any hit) is the same with the alpha test of its leaf code.
void AddClosestTraversal(Layout& layout, const Program& program, const RayTracingSite& site, uint32_t index) {
	const bool lane = site.kind == RayTracingKind::PacketLane;
	// A ClosestLane site with an Entry alpha test runs no guest leaf code either; its facing is
	// both and its opacity comes from the geometry and the instance flags, as for a PacketLane site
	// (but a hit is the nearest, not any).
	const bool entry = site.kind == RayTracingKind::ClosestLane;
	if (site.alpha.valid) {
		AddAlphaTest(layout, site, index);
	}
	// The TLAS header address, loaded again into a pair nothing reads before.
	auto header = Sgpr(site.header_sgpr);
	if (site.header_load_pc != 0) {
		header    = Sgpr(site.header_load_sgpr);
		auto& load = layout.AddCopy(*FindInstruction(program, site.header_load_pc));
		load.dst   = header;
	}
	// The alpha test switch and the instance mask the program keeps in VGPR lanes, read back into
	// free SGPRs.
	for (const auto& [pc, sgpr]: {std::pair {site.alpha_off_load_pc, site.alpha_off_sgpr},
	                              std::pair {site.cull_mask_load_pc, site.cull_mask_sgpr}}) {
		if (pc != 0) {
			auto& read = layout.AddCopy(*FindInstruction(program, pc));
			read.dst   = Sgpr(sgpr);
		}
	}
	auto& init = layout.AddPseudo(Opcode::RT_INIT, index);
	SetRay(init, site);
	init.image_nsa_addr[6] = site.t_max;
	init.image_nsa_addr[7] = site.cull_mask;
	init.image_nsa_addr[8] = site.cull_mask_constant;
	init.src0              = header;
	if (site.cull_mask_sgpr != NoRegister) {
		init.src2 = Sgpr(site.cull_mask_sgpr);
	}
	// Every triangle opaque, or the opaque ones by their primitive word and the alpha test.
	init.src1  = Constant(site.alpha.valid ? 0 : 1);
	// Both facings, or the facing of the instance flags (and opacity from the geometry).
	init.dmask = (site.both_facings ? 1u : 0u) | (site.alpha.valid ? 2u : 0u);
	if (lane) {
		// The opacity of the instance flags, which the alpha test switch overrides; the facing
		// flip and the condition of the alpha test's level of detail.
		init.src1 = site.alpha_off_sgpr != NoAlphaOff ? Sgpr(site.alpha_off_sgpr) : Constant(0);
		init.dmask = 16u;
		const auto reg = [](uint32_t sgpr) { return sgpr != NoRegister ? sgpr : 0xffffu; };
		init.secondary_offset = reg(site.flip_sgpr) | reg(site.alpha.level_sgpr) << 16u;
	} else if (entry) {
		// Both facings, the opacity of the geometry and the instance flags; no flip, always allowed.
		init.src1             = Constant(0);
		init.dmask            = 1u | 16u;
		init.secondary_offset = 0xffffffffu;
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
	commit.image_nsa_addr[3] = site.result_instance_hi != NoRegister ? site.result_instance_hi
	                           : site.result_instance != NoRegister  ? site.result_instance + 1
	                                                                 : NoRegister;
	commit.image_nsa_addr[4] = site.result_u;
	commit.image_nsa_addr[5] = site.result_v;
	commit.image_nsa_addr[6] = site.result_node;
	// Only when nearer than the hit in the registers (any hit: the registers hold none).
	commit.dmask = lane ? 0u : 1u;
	layout.AddPseudo(Opcode::RT_EXIT, index);
	layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Guest, site.exit_pc);
}

// The native traversal of a site, placed between its entry instruction and the loop header:
// RT_INIT falls back to the loop header, else the native loop runs and exits after the loop.
bool AddNativeTraversal(Layout& layout, const Program& program, const RayTracingSite& site,
                        uint32_t index) {
	if (site.kind == RayTracingKind::ClosestOpaque || site.kind == RayTracingKind::PacketLane ||
	    (site.kind == RayTracingKind::ClosestLane && site.alpha.valid &&
	     site.alpha.layout == RayTracingAlphaLayout::Entry)) {
		AddClosestTraversal(layout, program, site, index);
		return true;
	}
	if (site.alpha.valid) {
		AddAlphaTest(layout, site, index);
	}
	// A site with a per lane BLAS walk (closest hit, or any hit with lane masks): every triangle is
	// a candidate of the lane's own leaf code.
	const bool any_lane  = site.kind == RayTracingKind::AnyLane;
	const bool lane_leaf = site.kind == RayTracingKind::ClosestLane || any_lane;
	auto& init = layout.AddPseudo(Opcode::RT_INIT, index);
	SetRay(init, site);
	init.image_nsa_addr[6] = site.t_max;
	for (uint32_t c = 0; c < 3; c++) {
		if (site.direction_sgpr[c] != NoRegister) {
			init.image_nsa_addr[3 + c]  = NoRegister;
			init.image_nsa_addr[12 + c] = site.direction_sgpr[c];
		}
	}
	if (site.t_max_is_constant) {
		init.image_nsa_addr[6] = NoRegister;
		init.secondary_offset  = site.t_max_constant;
	}
	init.image_nsa_addr[7] = site.cull_mask;
	init.image_nsa_addr[8] = site.cull_mask_constant;
	init.src0              = Sgpr(site.header_sgpr);
	if (lane_leaf) {
		init.dmask = 1u | 32u;
	}
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
		if (site.after_leaf_pc == 0) {
			return;
		}
		const auto end = site.after_leaf_end_pc != 0 ? site.after_leaf_end_pc : EndPc(*FindInstruction(program, site.after_leaf_pc));
		for (auto pc = site.after_leaf_pc; pc < end;) {
			const auto* inst = FindInstruction(program, pc);
			layout.AddCopy(*inst);
			pc = EndPc(*inst);
		}
	};
	// A closest hit site's leaf code keeps the nearest hit itself: lanes that accept one keep
	// searching. It has no lane mask of its own (EXEC is the lanes that enter the leaf).
	const bool closest = site.kind == RayTracingKind::ClosestLeaf || lane_leaf;
	const auto mask    = [&] {
		if (!closest) {
			return Sgpr(site.mask_sgpr);
		}
		Operand exec;
		exec.kind = OperandKind::ExecLo;
		return exec;
	};
	// An AnyLane site's leaf starts from the lanes of EXEC; the guest's mask copy (run after it)
	// is what its leaf code clears.
	const auto leaf_mask = [&] {
		if (!any_lane) {
			return mask();
		}
		Operand exec;
		exec.kind = OperandKind::ExecLo;
		return exec;
	};

	// Loop header: the wave's next (instance, leaf), with the lanes that have it, or the exit.
	layout.Bind(loop_label);
	auto& select = layout.AddPseudo(Opcode::RT_SELECT, index);
	select.dst   = Sgpr(site.instance_sgpr);
	select.dst2  = Sgpr(any_lane ? site.saved_exec_sgpr : site.mask_sgpr);
	select.dmask = lane_leaf ? 1u : 0u;
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
	// A closest hit site's setup starts at the loop header (it copies EXEC there); the header's
	// branch to the box code is never taken for an instance node.
	if (!AddRangeCopy(layout, program, site.setup_pc, site.blas_head_pc, [&](uint32_t target) {
		    return target == site.blas_head_pc            ? leaf_label
		           : target == site.cull_exit_pc || closest ? cull_label
		                                                   : UINT32_MAX;
	    })) {
		return false;
	}
	layout.Bind(leaf_label);
	auto& leaf = layout.AddPseudo(Opcode::RT_LEAF, index);
	leaf.dst   = lane_leaf ? Vgpr(site.leaf_vgpr) : Sgpr(site.leaf_sgpr);
	leaf.dst2  = Sgpr(site.saved_exec_sgpr);
	leaf.src0  = leaf_mask();
	leaf.dmask = lane_leaf ? 1u : 0u;
	if (any_lane) {
		const auto* copy = FindInstruction(program, site.mask_copy_pc);
		if (copy == nullptr) {
			return false;
		}
		layout.AddCopy(*copy);
	}
	// Each lane's own leaf list entry, loaded as the guest walk loads it.
	if (lane_leaf) {
		for (auto pc = site.leaf_load_pc; pc < site.leaf_load_end_pc;) {
			const auto* inst = FindInstruction(program, pc);
			if (inst == nullptr) {
				return false;
			}
			// The guest walk's restore of its lanes: here EXEC is the lanes RT_LEAF selected.
			if (inst->dst.kind == OperandKind::ExecLo || inst->dst.kind == OperandKind::ExecHi) {
				pc = EndPc(*inst);
				continue;
			}
			layout.AddCopy(*inst);
			pc = EndPc(*inst);
		}
	}
	// The guest leaf code; it ends at the BLAS loop latch. The guest loops over the rest of the
	// leaf list until an entry marked last; here it tests only the selected entry (the hardware
	// reports the other triangles as candidates of their own). A selected entry from a list the
	// guest rewrote meanwhile could otherwise run past the end of the BLAS, where the loads read
	// zeros and no entry is ever the last one.
	// A closest hit site's leaf code leaves through the rest of the BLAS loop (its box code and
	// the pop of its next node).
	// A ClosestLane site's copy leaves out the box code of its per lane walk: the box stack is
	// never pushed (and its stores are not resources of the program).
	// The code before the box code continues with the code after it.
	const bool skip_box  = lane_leaf && site.leaf_box_pc != 0;
	const auto box_label = layout.NewLabel();
	const auto leaf_end  = [&](uint32_t target) {
		if (skip_box && target == site.leaf_box_pc) {
			return box_label;
		}
		return target == site.tail_pc || ((closest || any_lane) && target > site.tail_pc && target < site.blas_end_pc)
		           ? tail_label
		           : UINT32_MAX;
	};
	if (!AddRangeCopy(layout, program, site.leaf_pc, skip_box ? site.leaf_box_pc : site.tail_pc, leaf_end,
	                  !lane_leaf)) {
		return false;
	}
	if (skip_box) {
		layout.Bind(box_label);
		if (!AddRangeCopy(layout, program, site.leaf_box_end_pc, site.tail_pc, leaf_end, false)) {
			return false;
		}
	}
	// The end of an iteration: lanes that accepted a hit in the leaf stop (or, closest hit, keep
	// searching).
	layout.Bind(tail_label);
	auto& resolve = layout.AddPseudo(Opcode::RT_RESOLVE, index);
	resolve.src0  = mask();
	resolve.dmask = closest ? 2u : 0u;
	layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Label, loop_label);
	// The instance mask culled every selected lane (before the setup spilled anything).
	layout.Bind(cull_label);
	auto& skip = layout.AddPseudo(Opcode::RT_RESOLVE, index);
	skip.src0  = leaf_mask();
	skip.dmask = 1;
	layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Label, loop_label);

	// The exit: opaque hits the hardware found (closest hit: when nearer than the leaf code's),
	// then undo the last setup's spills.
	layout.Bind(exit_check_label);
	// (An AnyLane site has no opaque hits: every triangle is a candidate of the leaf code.)
	if (!any_lane) {
		auto& commit = layout.AddPseudo(Opcode::RT_COMMIT, index);
		commit.image_nsa_addr[0] = site.result_t;
		commit.image_nsa_addr[1] = site.result_word;
		commit.image_nsa_addr[2] = site.result_instance;
		commit.image_nsa_addr[3] =
		    site.result_instance_hi != NoRegister ? site.result_instance_hi : site.result_instance + 1;
		commit.image_nsa_addr[4] = closest ? site.result_u : NoRegister;
		commit.image_nsa_addr[5] = closest ? site.result_v : NoRegister;
		commit.image_nsa_addr[6] = closest ? site.result_node : NoRegister;
		commit.dmask             = closest ? 1u : 0u;
	}
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
	SiteFinder(const Program& program, uint32_t wave_size)
	    : m_inst(program.instructions), m_bodies(program.call_bodies), m_wave32(wave_size == 32) {}
	// Over other instructions of the program (see LevelBranch).
	SiteFinder(const std::vector<Instruction>& inst, const std::vector<InlinedCallBody>& bodies, bool wave32)
	    : m_inst(inst), m_bodies(bodies), m_wave32(wave32) {}

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
		uint32_t                    user_sgpr = NoRegister, header_offset = 0, root = 0, header_vsharp = NoRegister;
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
			if (const auto type = PacketNodeType(header); type != npos) {
				if (m_wave32) {
					// Waves of 32 lanes: only an AnyLane site (Marvel's Wolverine); other loops of this form
					// keep the guest's walk, as before.
					if (!DerivePacket(header, back, type, site) || site.kind != RayTracingKind::AnyLane) {
						continue;
					}
				} else if (!DerivePacket(header, back, type, site)) {
					// A header with spills that is not a site keeps its loop, as other loops.
					if (HeaderSpills(header, type)) {
						continue;
					}
					return false;
				}
				node_index = type;
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
			uint32_t site_user = NoRegister, site_offset = 0, site_root = 0, site_vsharp = NoRegister;
			if (!FindTlasSource(header, site.header_sgpr, site_user, site_offset, nullptr,
			                    (site.kind == RayTracingKind::ClosestLane || site.kind == RayTracingKind::AnyLane) ? &site_vsharp : nullptr) ||
			    !FindRoot(header, m_inst[node_index].src0, site_root)) {
				return false;
			}
			if (user_sgpr != NoRegister && (site_user != user_sgpr || site_offset != header_offset ||
			                                site_root != root || site_vsharp != header_vsharp)) {
				return false;
			}
			// One layout for the program: the sites read the same records (a site that does not
			// read the mask or the flags takes those of one that does).
			auto merged = site.layout;
			if (!sites.empty()) {
				// Sites whose ray masks are constants that select the instance mask's bits from another
				// shift of the same record dword (Marvel's Wolverine: 0x68000000 and 0x30000000): the
				// lower shift serves both, with the constants moved up.
				const auto constant_mask = [](const RayTracingSite& x) {
					return x.cull_mask == NoRegister && x.cull_mask_sgpr == NoRegister && x.cull_mask_load_pc == 0 &&
					       x.layout.mask_dword != NoRegister && x.layout.mask_width == 0;
				};
				if (layout.mask_dword == merged.mask_dword && layout.mask_dword != NoRegister && layout.mask_shift != merged.mask_shift &&
				    constant_mask(site) && std::ranges::all_of(sites, constant_mask)) {
					const auto common = std::min(layout.mask_shift, merged.mask_shift);
					const auto rebase = [&](RayTracingSite& x) {
						x.cull_mask_constant <<= x.layout.mask_shift - common;
						x.layout.mask_shift = common;
					};
					rebase(site);
					for (auto& other_site: sites) {
						rebase(other_site);
					}
					layout.mask_shift = common;
					merged            = site.layout;
				}
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
			header_vsharp = site_vsharp;
			root          = site_root;
			sites.push_back(site);
		}
		// Each site has its ray query state in the translated program (IR::MaxRayQuerySites).
		if (sites.empty() || sites.size() > IR::MaxRayQuerySites || !layout.valid || !Packable(layout)) {
			return false;
		}
		patch.sites         = std::move(sites);
		patch.user_sgpr     = user_sgpr;
		patch.header_offset = header_offset;
		patch.header_vsharp = header_vsharp;
		patch.root          = root;
		patch.layout        = layout;
		return true;
	}

private:
	static constexpr size_t npos = SIZE_MAX;
	struct Field;
	struct Condition;


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

	// S_AND_B32 vcc_lo, <node>, 7 at `i` followed by S_ADD_I32 vcc_lo, vcc_lo, -4 (with `spills`,
	// after V_WRITELANE_B32s).
	bool IsNodeType(size_t i, bool spills = false) const {
		size_t next = i + 1;
		while (spills && next < m_inst.size() && m_inst[next].opcode == Opcode::V_WRITELANE_B32) {
			next++;
		}
		if (next >= m_inst.size()) {
			return false;
		}
		const auto& a = m_inst[i];
		const auto& b = m_inst[next];
		return a.opcode == Opcode::S_AND_B32 && IsVccLo(a.dst) && a.src0.kind == OperandKind::Sgpr &&
		       IsConstant(a.src1, 7) && b.opcode == Opcode::S_ADD_I32 && IsVccLo(b.dst) &&
		       IsVccLo(b.src0) && b.src1.kind == OperandKind::IntegerInlineConstant && b.src1.signed_val == -4;
	}

	// A packet loop header at `i`: S_AND_B64, S_AND_SAVEEXEC_B64 and the node type test, or the same
	// after V_READLANE_B32s (the lane mask kept in VGPR lanes) and/or with V_WRITELANE_B32s in the
	// node type test (the saved EXEC spilled; GTA V 01.005 spills it with the lane mask in SGPRs).
	// The index of the node type's S_AND_B32 (npos: none).
	size_t PacketNodeType(size_t i) const {
		size_t first = i;
		while (first < m_inst.size() && m_inst[first].opcode == Opcode::V_READLANE_B32) {
			first++;
		}
		return first + 3 < m_inst.size() && m_inst[first].opcode == Lane(Opcode::S_AND_B64) &&
		               m_inst[first + 1].opcode == Lane(Opcode::S_AND_SAVEEXEC_B64) && IsNodeType(first + 2, true)
		           ? first + 2
		           : npos;
	}

	// Whether the packet loop header at `i` (node type test at `type`) keeps registers in VGPR
	// lanes: V_READLANE_B32s before it or V_WRITELANE_B32s in its node type test.
	bool HeaderSpills(size_t i, size_t type) const {
		return type != i + 2 || m_inst[type + 1].opcode == Opcode::V_WRITELANE_B32;
	}

	bool IsPacketHeader(size_t i) const { return PacketNodeType(i) != npos; }

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
	bool BoxTest(size_t header, size_t back, size_t box, RayTracingSite& site) const {
		const auto& test = m_inst[box];
		if (test.src1.kind != OperandKind::Sgpr) {
			return false;
		}
		site.t_max = AddressReg(test, 1);
		for (uint32_t c = 0; c < 3; c++) {
			site.origin[c]      = AddressReg(test, 2 + c);
			site.origin_sgpr[c] = NoRegister;
			site.direction[c]   = AddressReg(test, 5 + c);
		}
		for (size_t i = header; i-- > 0;) {
			const auto& bfe = m_inst[i];
			if (bfe.opcode == Opcode::S_BFE_U64 && IsSgpr(bfe.dst, test.src1.reg) &&
			    bfe.src0.kind == OperandKind::Sgpr && IsConstant(bfe.src1, 0x280008)) {
				site.header_sgpr = bfe.src0.reg;
				ScalarOrigin(header, back, box, site);
				return site.t_max != NoRegister;
			}
		}
		// Or the T#'s base was kept in VGPR lanes (V_WRITELANE_B32, read back with V_READLANE_B32
		// before the test; Marvel's Wolverine).
		const auto writer = ScalarWriter(box, test.src1.reg, 0);
		if (writer != npos && writer < header && m_inst[writer].opcode == Opcode::S_BFE_U64 &&
		    m_inst[writer].src0.kind == OperandKind::Sgpr && IsConstant(m_inst[writer].src1, 0x280008)) {
			site.header_sgpr = m_inst[writer].src0.reg;
			ScalarOrigin(header, back, box, site);
			return site.t_max != NoRegister;
		}
		return false;
	}

	// A ray origin the same for every lane (camera rays): the loop copies it into the box test's
	// VGPRs with V_MOV_B32 <vgpr>, <sgpr> right before the test (no branch in between, nor into it),
	// from SGPRs nothing in the loop writes. The origin is then those SGPRs, which hold it from the
	// loop's entry on.
	void ScalarOrigin(size_t header, size_t back, size_t box, RayTracingSite& site) const {
		for (uint32_t c = 0; c < 3; c++) {
			const auto w = site.origin[c] != NoRegister ? VectorWriter(box, site.origin[c], header) : npos;
			if (w == npos) {
				continue;
			}
			const auto& move   = m_inst[w];
			const auto  source = move.src0.kind == OperandKind::Sgpr ? move.src0.reg : NoRegister;
			if (move.opcode != Opcode::V_MOV_B32 || !Plain(move.dst) || !Plain(move.src0) || source == NoRegister ||
			    ScalarWriter(back, source, header) != npos) {
				continue;
			}
			bool straight = true;
			for (size_t j = w + 1; j < box && straight; j++) {
				straight = !IsDirectBranch(m_inst[j].opcode) && m_inst[j].opcode != Opcode::S_SETPC_B64;
			}
			for (const auto& inst: m_inst) {
				if (IsDirectBranch(inst.opcode) && inst.branch_target > move.pc && inst.branch_target <= m_inst[box].pc) {
					straight = false;
				}
			}
			if (straight) {
				site.origin[c]      = NoRegister;
				site.origin_sgpr[c] = source;
			}
		}
	}

	bool DerivePacket(size_t header, size_t back, size_t type, RayTracingSite& site) const {
		site.kind     = RayTracingKind::Packet;
		site.entry_pc = m_inst[header - 1].pc;
		site.loop_pc  = m_inst[header].pc;
		const auto& node_type = m_inst[type];
		site.instance_sgpr    = node_type.src0.reg;
		// The instance setup.
		const auto le = Next(type + 1, Opcode::S_CMP_LE_U32, back);
		if (le == npos || !IsVccLo(m_inst[le].src0) || !IsConstant(m_inst[le].src1, 2) ||
		    m_inst[le + 1].opcode != Opcode::S_CBRANCH_SCC0) {
			return false;
		}
		site.setup_pc = m_inst[le + 1].branch_target;
		const auto box = Next(le, Opcode::IMAGE_BVH_INTERSECT_RAY, back);
		if (box == npos || !BoxTest(header, back, box, site)) {
			return false;
		}
		// Waves of 32 lanes: a direction the loop reads from VGPR lanes each iteration (AnyLane).
		for (uint32_t c = 0; m_wave32 && c < 3; c++) {
			const auto w = VectorWriter(box, site.direction[c], header);
			if (w == npos || m_inst[w].opcode != Opcode::V_MOV_B32 || m_inst[w].src0.kind != OperandKind::Sgpr) {
				continue;
			}
			for (size_t j = w; j-- > header;) {
				if (!WritesScalar(m_inst[j], m_inst[w].src0.reg)) {
					continue;
				}
				if (m_inst[j].opcode == Opcode::V_READLANE_B32 && m_inst[j].src0.kind == OperandKind::Vgpr &&
				    IsLaneConstant(m_inst[j].src1)) {
					site.direction_lane_vgpr[c] = m_inst[j].src0.reg;
					site.direction_lane[c]      = m_inst[j].src1.value;
				}
				break;
			}
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
		// of which (V_CMP_NE_U32 or V_CMPX_NE_U32 0, x) tests the instance mask; or, for a ray mask
		// the same for every lane, an S_CBRANCH_SCC0 before it of the S_AND_B32 that tests the
		// mask, to where the node type test sends the wave without lanes (the next node), with no
		// test before it that narrows EXEC.
		auto       culled = Next(setup, Opcode::S_CBRANCH_EXECZ, back);
		const auto none   = Next(type, Opcode::S_CBRANCH_EXECZ, le);
		if (culled == npos) {
			return false;
		}
		site.cull_mask         = NoRegister;
		site.cull_mask_sgpr    = NoRegister;
		site.cull_mask_load_pc = 0;
		bool mask              = false;
		for (size_t i = setup; i < culled && !mask; i++) {
			if (WritesExec(m_inst[i])) {
				break;
			}
			if (m_inst[i].opcode == Opcode::S_CBRANCH_SCC0 && none != npos &&
			    m_inst[i].branch_target == m_inst[none].branch_target) {
				mask   = ScalarMaskTest(i, header, back, record, site);
				culled = mask ? i : culled;
				break;
			}
		}
		for (size_t i = setup; i < culled && !mask; i++) {
			const auto& cmp = m_inst[i];
			if ((cmp.opcode == Opcode::V_CMPX_NE_U32 || cmp.opcode == Opcode::V_CMP_NE_U32) && IsConstant(cmp.src0, 0)) {
				mask = InstanceMaskTest(i, cmp.src1, header, back, record, site);
			}
		}
		if (!mask) {
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
			// Or each lane walks the BLAS (and, in waves of 32 lanes, runs the guest leaf code).
			if (m_wave32) {
				return DeriveAnyLane(header, back, type, le, culled, record, site);
			}
			return DerivePacketLane(header, back, type, culled, record, site);
		}
		// The native traversal of a Packet site reads the ray's mask from its SGPR.
		if (site.cull_mask_load_pc != 0) {
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

	// ---- A Packet site whose lanes each walk the BLAS and run the guest leaf code (AnyLane). ----
	//
	// Marvel's Wolverine's shadow rays (waves of 32 lanes, sce::Psr BVH version 10): the TLAS loop
	// keeps a wave-wide stack of (node, lane mask) pairs and the lanes of the mask that find an
	// accepted hit drop out of it; the instance setup (the instance record's mask test, the object
	// space ray, the BLAS T#, the root in the node VGPR) is followed by a per lane BLAS walk of the
	// form of ClosestLaneWalk: the loop's only IMAGE_BVH_INTERSECT_RAY has the lane's node VGPR as
	// its address, the leaf code runs for the lanes with a triangle node, the lane list entry of
	// the next triangle is loaded just before the backward S_CBRANCH_EXECNZ, and an accepted hit
	// moves the primitive word into a VGPR and clears the lane from the lane mask the setup copied
	// from EXEC (S_ANDN2 <mask>, <mask>, exec, then S_MOV exec, <saved> which ends the leaf code).
	// The ray's direction is the same for every lane and the loop reads it from VGPR lanes each
	// iteration (V_READLANE_B32; V_MOV_B32 <direction VGPR>, <s>); its t max is a constant the loop
	// moves into its VGPR.
	bool DeriveAnyLane(size_t header, size_t back, size_t type, size_t le, size_t culled, const Field& record,
	                   RayTracingSite& site) const {
		(void)record;
		const auto setup = IndexOf(site.setup_pc);
		if (setup == npos || culled == npos) {
			return false;
		}
		// The BLAS walk: the backward S_CBRANCH_EXECNZ into the instance code with the earliest target.
		size_t blas = npos, blas_back = npos;
		for (size_t i = culled + 1; i < back; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::S_CBRANCH_EXECNZ && inst.branch_target > m_inst[culled].pc &&
			    inst.branch_target < inst.pc && (blas == npos || inst.branch_target <= m_inst[blas].pc)) {
				blas      = IndexOf(inst.branch_target);
				blas_back = i;
			}
		}
		if (blas == npos || blas_back == npos) {
			return false;
		}
		size_t test = npos;
		for (size_t i = blas; i < blas_back; i++) {
			if (m_inst[i].opcode == Opcode::IMAGE_BVH_INTERSECT_RAY) {
				if (test != npos) {
					return false;
				}
				test = i;
			}
		}
		if (test == npos || m_inst[test].src0.kind != OperandKind::Vgpr) {
			return false;
		}
		const auto node = m_inst[test].src0.reg;
		// The load of the lane's next leaf list entry.
		size_t load = npos;
		for (size_t i = blas_back; i-- > test && load == npos;) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::BUFFER_LOAD_DWORDX2 && IsVgpr(inst.dst, node)) {
				load = i;
			}
		}
		const auto shift = load != npos && m_inst[load].src0.kind == OperandKind::Vgpr
		                       ? VectorWriter(load, m_inst[load].src0.reg, test)
		                       : npos;
		if (shift == npos || m_inst[shift].opcode != Opcode::V_LSHRREV_B32 || !IsConstant(m_inst[shift].src0, 3) ||
		    !IsVgpr(m_inst[shift].src1, node)) {
			return false;
		}
		for (size_t i = shift + 1; i < load; i++) {
			if (IsDirectBranch(m_inst[i].opcode) || WritesExec(m_inst[i]) ||
			    WritesVector(m_inst[i], m_inst[load].src0.reg) || WritesVector(m_inst[i], node)) {
				return false;
			}
		}
		// The setup: no spills, its branches stay inside it or leave through the cull exit; the
		// copy of EXEC into the lane mask the leaf code clears.
		size_t mask_copy = npos;
		for (size_t i = setup; i < blas; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::V_WRITELANE_B32 && inst.src0.kind == OperandKind::Sgpr) {
				return false;
			}
			if (inst.opcode == Lane(Opcode::S_MOV_B64) && inst.dst.kind == OperandKind::Sgpr &&
			    inst.src0.kind == OperandKind::ExecLo && mask_copy == npos) {
				mask_copy = i;
			}
			if (!IsDirectBranch(inst.opcode) ||
			    (inst.branch_target > m_inst[setup].pc && inst.branch_target <= m_inst[blas].pc) ||
			    inst.branch_target == site.cull_exit_pc) {
				continue;
			}
			return false;
		}
		if (mask_copy == npos || VectorWriter(blas, node, setup) == npos) {
			return false;
		}
		const auto mask = m_inst[mask_copy].dst.reg;
		// The accepted hit: V_MOV_B32 <result word>, <the entry's word (node + 1)>, then the lane
		// leaves the mask; S_MOV exec, <saved> ends the leaf code.
		size_t accept = npos;
		for (size_t i = test + 1; i < shift && accept == npos; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Lane(Opcode::S_ANDN2_B64) && IsSgpr(inst.dst, mask) && IsSgpr(inst.src0, mask) &&
			    inst.src1.kind == OperandKind::ExecLo) {
				accept = i;
			}
		}
		if (accept == npos || accept < test + 2 || m_inst[accept - 1].opcode != Opcode::V_MOV_B32 ||
		    m_inst[accept - 1].dst.kind != OperandKind::Vgpr || !IsVgpr(m_inst[accept - 1].src0, node + 1) ||
		    !Plain(m_inst[accept - 1].src0) || accept + 2 >= shift ||
		    m_inst[accept + 1].opcode != Lane(Opcode::S_MOV_B64) || m_inst[accept + 1].dst.kind != OperandKind::ExecLo ||
		    m_inst[accept + 1].src0.kind != OperandKind::Sgpr) {
			return false;
		}
		const auto tail = accept + 2;
		// The box code: from the first S_CBRANCH_EXECZ after the test that skips forward to the
		// S_ANDN2 exec, <lanes>, exec of the lanes with a triangle node.
		size_t box = npos;
		for (size_t i = test + 1; i < tail && box == npos; i++) {
			const auto& inst = m_inst[i];
			if (IsDirectBranch(inst.opcode)) {
				const auto target = IndexOf(inst.branch_target);
				if (inst.opcode == Opcode::S_CBRANCH_EXECZ && target != npos && target > i && target < tail &&
				    m_inst[target].opcode == Lane(Opcode::S_ANDN2_B64) && m_inst[target].dst.kind == OperandKind::ExecLo &&
				    m_inst[target].src1.kind == OperandKind::ExecLo) {
					box = i;
				}
				break;
			}
		}
		// The leaf code leaves only into the rest of the walk.
		for (size_t i = blas; i < tail; i++) {
			const auto& inst = m_inst[i];
			if (IsDirectBranch(inst.opcode) &&
			    (inst.branch_target < m_inst[blas].pc || inst.branch_target > m_inst[blas_back].pc)) {
				return false;
			}
		}
		// The exit of the traversal: the last S_CBRANCH_VCCNZ of the loop.
		size_t exit = npos;
		for (size_t i = back; i-- > blas_back && exit == npos;) {
			if (m_inst[i].opcode == Opcode::S_CBRANCH_VCCNZ) {
				exit = i;
			}
		}
		if (exit == npos) {
			return false;
		}
		// The ray: the TLAS box test's registers. A direction component the loop reads from a
		// VGPR lane before the test, and a constant t max, are not in their VGPRs at the entry.
		const auto box_test = Next(le, Opcode::IMAGE_BVH_INTERSECT_RAY, back);
		if (box_test == npos) {
			return false;
		}
		const auto free_a = UnusedPair();
		if (free_a == NoRegister) {
			return false;
		}
		for (uint32_t c = 0; c < 3; c++) {
			const auto w = VectorWriter(box_test, site.direction[c], header);
			if (w == npos) {
				continue;
			}
			const auto& move = m_inst[w];
			size_t      read = npos;
			if (move.opcode == Opcode::V_MOV_B32 && Plain(move.dst) && Plain(move.src0) &&
			    move.src0.kind == OperandKind::Sgpr) {
				for (size_t j = w; j-- > header && read == npos;) {
					if (WritesScalar(m_inst[j], move.src0.reg)) {
						read = j;
					}
				}
			}
			if (read == npos || !KeptLane(read, header, back)) {
				return false;
			}
			// RT_INIT reads the lane itself: 0x80000000 | lane << 8 | VGPR.
			site.direction_sgpr[c] = 0x80000000u | m_inst[read].src1.value << 8u | m_inst[read].src0.reg;
		}
		if (const auto w = VectorWriter(box_test, site.t_max, header); w != npos) {
			const auto& move = m_inst[w];
			if (move.opcode != Opcode::V_MOV_B32 || !Plain(move.dst) || move.src0.kind != OperandKind::LiteralConstant) {
				return false;
			}
			site.t_max_is_constant = true;
			site.t_max_constant    = move.src0.value;
		}
		site.kind              = RayTracingKind::AnyLane;
		site.instance_sgpr     = m_inst[type].src0.reg;
		site.blas_head_pc      = m_inst[blas].pc;
		site.leaf_pc           = m_inst[blas].pc;
		site.tail_pc           = m_inst[tail].pc;
		site.blas_end_pc       = EndPc(m_inst[blas_back]);
		site.leaf_vgpr         = node;
		site.leaf_load_pc      = m_inst[shift].pc;
		site.leaf_load_end_pc  = EndPc(m_inst[load]);
		site.leaf_box_pc       = box != npos ? m_inst[box].pc : 0;
		site.leaf_box_end_pc   = box != npos ? m_inst[box].branch_target : 0;
		site.leaf_sgpr         = 0;
		site.mask_sgpr         = mask;
		site.mask_copy_pc      = m_inst[mask_copy].pc;
		site.saved_exec_sgpr   = free_a;
		site.alpha_off_sgpr    = NoAlphaOff;
		site.after_leaf_pc     = 0;
		site.after_leaf_end_pc = 0;
		site.both_facings      = true;
		site.result_t          = NoRegister;
		site.result_word       = m_inst[accept - 1].dst.reg;
		site.result_instance   = NoRegister;
		site.exit_pc           = m_inst[exit].branch_target;
		site.alpha             = {};
		return true;
	}

	// ---- A Packet site with a per lane BLAS walk (RayTracingKind::PacketLane). ----
	//
	// After the instance setup (its cull at `culled`), each lane walks the instance's BLAS: one
	// IMAGE_BVH_INTERSECT_RAY of a VGPR node in a loop closed by S_AND_B64 exec, <x>, <mask>;
	// S_CBRANCH_EXECNZ, the node starting at the record's root. Box nodes push their children on
	// a stack of the lane; the triangle nodes' lanes (after S_ANDN2_B64 exec, <s>, exec) run the
	// leaf code:
	//   facing    S_AND_SAVEEXEC_B64 of the triangle test's hit and facing (see LaneLeafFacing)
	//   t         V_MUL_F32 <t>, <first result>, V_RCP_F32 <second result>; V_CMPX_LE_F32 <t>, <t max>
	//   opacity   the alpha test runs for the lanes that are neither opaque nor switched off, the
	//             others accept (see LaneLeafOpacity)
	//   alpha     the per lane alpha test (DeriveLaneAlphaTest) sets the flag V_CMPX_NE_U32 0, <flag>
	//   accept    V_MOV_B32 <result t>, <t>; <result word>, <word>; <instance pair>, <record pair>;
	//             S_ANDN2_B64 <mask>, <mask>, exec
	// The leaf entry (node, word) is a BUFFER_LOAD_DWORDX2 idxen of the walk. The TLAS walk drops
	// the lanes with a hit (V_CMP_NE_U32 vcc, -1, <result word>; S_ANDN2_B64 <lanes>, <lanes>, vcc).
	bool DerivePacketLane(size_t header, size_t back, size_t type, size_t culled, const Field& record,
	                      RayTracingSite& site) const {
		site.kind = RayTracingKind::PacketLane;
		// The BLAS walk.
		const auto test = Next(culled, Opcode::IMAGE_BVH_INTERSECT_RAY, back);
		if (test == npos || m_inst[test].src0.kind != OperandKind::Vgpr || m_inst[test].dst.kind != OperandKind::Vgpr) {
			return false;
		}
		const auto node   = AddressReg(m_inst[test], 0);
		const auto result = m_inst[test].dst.reg;
		size_t     latch  = npos;
		for (size_t i = test + 1; i < back && latch == npos; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::S_CBRANCH_EXECNZ && inst.branch_target <= m_inst[test].pc &&
			    inst.branch_target > m_inst[culled].pc) {
				latch = i;
			}
		}
		const auto walk = latch != npos ? IndexOf(m_inst[latch].branch_target) : npos;
		if (walk == npos || latch < 1 || !IsBlasPair(header, back, BlasPair(test), NoRegister)) {
			return false;
		}
		const auto& again = m_inst[latch - 1];
		if (again.opcode != Lane(Opcode::S_AND_B64) || again.dst.kind != OperandKind::ExecLo) {
			return false;
		}
		const auto mask = again.src1.kind == OperandKind::Sgpr ? again.src1.reg
		                  : again.src0.kind == OperandKind::Sgpr ? again.src0.reg
		                                                         : NoRegister;
		if (mask == NoRegister) {
			return false;
		}
		// The root: V_MOV_B32 <node>, <record root dword> before the walk.
		const auto root = VectorSource(walk, Vgpr(node), Opcode::V_MOV_B32, culled);
		Field      root_field;
		if (root == npos || !OperandField(root, m_inst[root].src0, culled, root_field) ||
		    !SameRecord(root_field, record) || root_field.dword != site.layout.root_dword) {
			return false;
		}
		// The leaf entries: <node, word> of BUFFER_LOAD_DWORDX2 idxen in the walk.
		bool entries = false;
		for (size_t i = walk; i < latch && !entries; i++) {
			const auto& load = m_inst[i];
			entries = load.opcode == Opcode::BUFFER_LOAD_DWORDX2 && load.idxen && load.dst.kind == OperandKind::Vgpr &&
			          load.dst.reg == node;
		}
		const auto word = node + 1;
		// The triangle lanes.
		size_t leaf = npos;
		for (size_t i = test + 1; i < latch && leaf == npos; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Lane(Opcode::S_ANDN2_B64) && inst.dst.kind == OperandKind::ExecLo &&
			    inst.src1.kind == OperandKind::ExecLo) {
				leaf = i + 1;
			}
		}
		if (!entries || leaf == npos) {
			return false;
		}
		// The facing: the first S_AND_SAVEEXEC_B64 of the leaf code.
		const auto facing = Next(leaf, Lane(Opcode::S_AND_SAVEEXEC_B64), latch);
		if (facing == npos) {
			return false;
		}
		// The distance, within the t max.
		size_t t_test = npos;
		for (size_t i = facing + 1; i < latch && t_test == npos; i++) {
			const auto& cmp = m_inst[i];
			if (cmp.opcode == Opcode::V_CMPX_LE_F32 && cmp.src0.kind == OperandKind::Vgpr && Plain(cmp.src0) &&
			    Plain(cmp.src1) && IsVgpr(cmp.src1, site.t_max)) {
				t_test = i;
			}
		}
		const auto t   = t_test != npos ? VectorSource(t_test, m_inst[t_test].src0, Opcode::V_MUL_F32, leaf) : npos;
		const auto rcp = t != npos ? VectorSource(t, m_inst[t].src1, Opcode::V_RCP_F32, leaf) : npos;
		if (rcp == npos || !Plain(m_inst[t].src0) || !Plain(m_inst[t].src1) || !Plain(m_inst[t].dst) ||
		    !IsVgpr(m_inst[t].src0, result) || ResultWriter(t, result, test) != test || !IsVgpr(m_inst[rcp].src0, result + 1) ||
		    !Plain(m_inst[rcp].src0) || ResultWriter(rcp, result + 1, test) != test) {
			return false;
		}
		const auto t_vgpr = m_inst[t].dst.reg;
		// The opacity and the alpha test switch, then the facing (with the flags they read).
		size_t   alpha_path = npos;
		uint32_t flag       = NoRegister;
		if (!LaneLeafOpacity(header, back, leaf, t_test, latch, word, record, site, alpha_path, flag) ||
		    !LaneLeafFacing(header, back, test, facing, site)) {
			return false;
		}
		// The accepted hit.
		size_t accept = npos;
		for (size_t i = alpha_path; i < latch && accept == npos; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Lane(Opcode::S_ANDN2_B64) && IsSgpr(inst.dst, mask) && IsSgpr(inst.src0, mask) &&
			    inst.src1.kind == OperandKind::ExecLo) {
				accept = i;
			}
		}
		if (accept == npos || accept < alpha_path + 6) {
			return false;
		}
		const auto& t_move    = m_inst[accept - 4];
		const auto& word_move = m_inst[accept - 3];
		const auto& low       = m_inst[accept - 2];
		const auto& high      = m_inst[accept - 1];
		for (const auto* move: {&t_move, &word_move, &low, &high}) {
			if (move->opcode != Opcode::V_MOV_B32 || move->dst.kind != OperandKind::Vgpr || !Plain(move->src0) ||
			    !Plain(move->dst)) {
				return false;
			}
		}
		if (!IsVgpr(t_move.src0, t_vgpr) || !IsVgpr(word_move.src0, word) || !IsSgpr(low.src0, record.pair) ||
		    !IsSgpr(high.src0, record.pair + 1) || high.dst.reg != low.dst.reg + 1) {
			return false;
		}
		// The flag: V_CMPX_NE_U32 0, <flag> (then S_CBRANCH_EXECZ) before the moves, after the alpha
		// test restored EXEC.
		const auto& flag_test = m_inst[accept - 6];
		if ((flag_test.opcode != Opcode::V_CMPX_NE_U32 && flag_test.opcode != Opcode::V_CMPX_NE_I32) ||
		    !IsConstant(flag_test.src0, 0) || !IsVgpr(flag_test.src1, flag) || !Plain(flag_test.src1) ||
		    m_inst[accept - 5].opcode != Opcode::S_CBRANCH_EXECZ || m_inst[accept - 7].opcode != Lane(Opcode::S_MOV_B64) ||
		    m_inst[accept - 7].dst.kind != OperandKind::ExecLo ||
		    !IsSgpr(m_inst[accept - 7].src0, m_inst[alpha_path].dst.reg)) {
			return false;
		}
		site.result_t        = t_move.dst.reg;
		site.result_word     = word_move.dst.reg;
		site.result_instance = low.dst.reg;
		// The TLAS walk drops the lanes with a hit (the SGPRs of its lane mask restored from VGPR lanes
		// or not), then exits (the last S_CBRANCH_VCCNZ).
		const auto lanes = ScalarIndex(m_inst[type - 2].src1);
		bool       drop  = false;
		for (size_t i = latch + 1; i + 1 < back && !drop; i++) {
			const auto& cmp = m_inst[i];
			auto        j   = i + 1;
			while (j + 1 < back && m_inst[j].opcode == Opcode::V_READLANE_B32) {
				j++;
			}
			const auto& andn = m_inst[j];
			drop = cmp.opcode == Opcode::V_CMP_NE_U32 && IsVccLo(cmp.dst) && cmp.src0.kind == OperandKind::IntegerInlineConstant &&
			       cmp.src0.signed_val == -1 && IsVgpr(cmp.src1, site.result_word) && andn.opcode == Lane(Opcode::S_ANDN2_B64) &&
			       lanes != NoRegister && IsSgpr(andn.dst, lanes) && IsSgpr(andn.src0, lanes) && IsVccLo(andn.src1);
		}
		size_t exit = npos;
		for (size_t i = back; i-- > latch && exit == npos;) {
			if (m_inst[i].opcode == Opcode::S_CBRANCH_VCCNZ) {
				exit = i;
			}
		}
		if (!drop || exit == npos) {
			return false;
		}
		site.exit_pc = m_inst[exit].branch_target;
		// The alpha test, on the path of a wave-wide condition of its level of detail if it has one.
		size_t sample = npos;
		for (size_t i = alpha_path; i < accept && sample == npos; i++) {
			if (m_inst[i].opcode == Opcode::IMAGE_SAMPLE) {
				sample = i;
			}
		}
		if (sample == npos) {
			return false;
		}
		std::vector<Instruction> linear;
		uint32_t                 condition = NoRegister;
		if (!LevelBranch(header, back, alpha_path, sample, linear, condition)) {
			return false;
		}
		const SiteFinder path(linear.empty() ? m_inst : linear, m_bodies, m_wave32);
		for (size_t i = leaf; i < accept - 6; i++) {
			const auto opcode = path.m_inst[i].opcode;
			if (opcode == Opcode::S_BRANCH || ((IsConditionalBranch(opcode) && opcode != Opcode::S_CBRANCH_EXECZ &&
			                                     opcode != Opcode::S_CBRANCH_EXECNZ))) {
				return false;
			}
		}
		RayTracingAlphaTest alpha;
		if (!path.DeriveLaneAlphaTest(header, back, alpha_path, accept - 6, t_vgpr, node, word, flag, record.pair, alpha)) {
			return false;
		}
		alpha.level_sgpr = condition;
		site.alpha       = alpha;
		// RT_SELECT's registers: a pair the program does not use. The TLAS header address at the
		// entry, when loaded again, goes there before RT_SELECT writes it.
		site.instance_sgpr = UnusedPair();
		site.mask_sgpr     = site.instance_sgpr;
		if (site.instance_sgpr == NoRegister || !HeaderAtEntry(header, site.instance_sgpr, site)) {
			return false;
		}
		// The values RT_INIT reads back from VGPR lanes (see LaneLeafOpacity and MaskAnd) go to a
		// free pair too: RT_SELECT's when the header address is not loaded there, else another one.
		if (site.alpha_off_load_pc != 0 || site.cull_mask_load_pc != 0) {
			const auto values = site.header_load_pc == 0 ? site.instance_sgpr : UnusedPair(site.instance_sgpr);
			if (values == NoRegister) {
				return false;
			}
			if (site.cull_mask_load_pc != 0) {
				site.cull_mask_sgpr = values;
			}
			if (site.alpha_off_load_pc != 0) {
				site.alpha_off_sgpr = values + 1;
			}
		}
		return true;
	}

	// The accepted hit of a closest hit walk's per lane leaf code (DerivePacketLane with `closest`):
	// after V_CMPX_NE_U32 0, <flag>; S_CBRANCH_EXECZ, a run of V_MOV_B32s of the distance (into t
	// max), the primitive word and the triangle node (the leaf entry), the instance record pair (its
	// SGPRs) and the barycentrics (V_MUL_F32s of the triangle test's third and fourth results).
	// Gives the index of the flag test.
	bool ClosestLaneHit(size_t begin_at, size_t latch, size_t test, uint32_t t_vgpr, uint32_t node,
	                    const Field& record, RayTracingSite& site, size_t& flag_test) const {
		const auto move = [&](size_t i) {
			const auto& inst = m_inst[i];
			return inst.opcode == Opcode::V_MOV_B32 && inst.dst.kind == OperandKind::Vgpr && Plain(inst.dst) &&
			       Plain(inst.src0) && (inst.src0.kind == OperandKind::Vgpr || inst.src0.kind == OperandKind::Sgpr);
		};
		size_t t_move = npos;
		for (size_t i = begin_at; i < latch && t_move == npos; i++) {
			if (move(i) && m_inst[i].dst.reg == site.t_max && m_inst[i].src0.kind == OperandKind::Vgpr &&
			    (t_vgpr == NoRegister || IsVgpr(m_inst[i].src0, t_vgpr))) {
				t_move = i;
			}
		}
		if (t_move == npos) {
			return false;
		}
		size_t begin = t_move, end = t_move + 1;
		while (begin > begin_at && move(begin - 1)) {
			begin--;
		}
		while (end < latch && move(end)) {
			end++;
		}
		if (begin < begin_at + 3 || m_inst[begin - 1].opcode != Opcode::S_CBRANCH_EXECZ) {
			return false;
		}
		flag_test         = begin - 2;
		t_vgpr            = m_inst[t_move].src0.reg;
		const auto result = m_inst[test].dst.reg;
		uint32_t   high   = NoRegister;
		site.result_t = site.result_word = site.result_instance = NoRegister;
		site.result_u = site.result_v = site.result_node = NoRegister;
		for (size_t i = begin; i < end; i++) {
			const auto& inst = m_inst[i];
			const auto  dst  = inst.dst.reg;
			if (inst.src0.kind == OperandKind::Sgpr) {
				if (inst.src0.reg == record.pair) {
					site.result_instance = dst;
				} else if (inst.src0.reg == record.pair + 1) {
					high = dst;
				} else {
					return false;
				}
				continue;
			}
			const auto src = inst.src0.reg;
			if (src == t_vgpr && dst == site.t_max) {
				site.result_t = dst;
			} else if (src == node + 1) {
				site.result_word = dst;
			} else if (src == node) {
				site.result_node = dst;
			} else {
				// A barycentric: the product of the test's third or fourth result.
				const auto mul = VectorWriter(i, src, test + 1);
				if (mul == npos || m_inst[mul].opcode != Opcode::V_MUL_F32) {
					return false;
				}
				// (The walk's box code, for other lanes, may write those VGPRs in between.)
				const auto reads = [&](uint32_t reg) {
					const auto& m = m_inst[mul];
					return IsVgpr(m.src0, reg) || IsVgpr(m.src1, reg);
				};
				const bool u = reads(result + 2), v = !u && reads(result + 3);
				auto&      bary = u ? site.result_u : site.result_v;
				if ((!u && !v) || bary != NoRegister) {
					return false;
				}
				bary = dst;
			}
		}
		return site.result_t != NoRegister && site.result_word != NoRegister && site.result_node != NoRegister &&
		       site.result_instance != NoRegister && high == site.result_instance + 1 && site.result_u != NoRegister &&
		       site.result_v != NoRegister;
	}

	// The TLAS header address at the entry of a site that RT_INIT reads (AddClosestTraversal): the
	// header pair, or the guest's load of it from its user data pointer, unchanged, again into the
	// SGPR pair `pair` (a pair nothing reads before; the compiler may reuse the header pair before
	// the loop, as GTA V 01.005's 0x84ed00053b3771da does for the root node).
	bool HeaderAtEntry(size_t header, uint32_t pair, RayTracingSite& site) const {
		uint32_t user = NoRegister, offset = 0;
		size_t   load = npos;
		if (!FindTlasSource(header, site.header_sgpr, user, offset, &load)) {
			return false;
		}
		site.header_load_pc = 0;
		if (ScalarWriter(header, site.header_sgpr, 0) != load || ScalarWriter(header, site.header_sgpr + 1, 0) != load) {
			if (pair == NoRegister || ScalarWriter(header, user, 0) != npos || ScalarWriter(header, user + 1, 0) != npos) {
				return false;
			}
			site.header_load_pc   = m_inst[load].pc;
			site.header_load_sgpr = pair;
		}
		return true;
	}

	// The last instruction before `i`, from `from`, that writes VGPR `reg` (stores, whose data
	// WritesVector counts, do not).
	size_t ResultWriter(size_t i, uint32_t reg, size_t from) const {
		for (size_t j = i; j-- > from;) {
			const auto name = magic_enum::enum_name(m_inst[j].opcode);
			if (WritesVector(m_inst[j], reg) && name.find("_STORE") == std::string_view::npos &&
			    !name.starts_with("DS_WRITE")) {
				return j;
			}
		}
		return npos;
	}

	static Operand Vgpr(uint32_t reg) {
		Operand operand;
		operand.kind = OperandKind::Vgpr;
		operand.reg  = reg;
		return operand;
	}

	static bool IsVgpr(const Operand& operand, uint32_t reg) {
		return operand.kind == OperandKind::Vgpr && operand.reg == reg;
	}

	// The SGPR pair the T# of an IMAGE_BVH_INTERSECT_RAY was built from (S_BFE_U64 <t#>, <pair>,
	// 0x280008), or NoRegister.
	uint32_t BlasPair(size_t test) const {
		const auto& inst = m_inst[test];
		const auto  bfe  = inst.src1.kind == OperandKind::Sgpr ? ScalarSource(test, inst.src1, Opcode::S_BFE_U64, 0) : npos;
		return bfe != npos && IsConstant(m_inst[bfe].src1, 0x280008) && m_inst[bfe].src0.kind == OperandKind::Sgpr
		           ? m_inst[bfe].src0.reg
		           : NoRegister;
	}

	// The opacity of a PacketLane site's leaf code (lanes from `leaf`, after the distance test
	// `from`): S_AND_SAVEEXEC_B64 <saved>, !(off | (opaque & !alpha tested)) starts the alpha test
	// (`path`), after V_CNDMASK_B32 <flag>, 0, 1, (off | (opaque & !alpha tested)), where
	//   opaque        V_CMP_NE_U32 0, V_AND_B32 1, V_OR_B32 <flags>, V_LSHRREV_B32 30, <word>
	//   alpha tested  S_CSELECT_B64 exec, 0 of S_CMP_LG_U32 0, S_AND_B32 <flags>, 2
	//   off           S_CSELECT_B64 exec, 0 of S_CMP_LG_U32 0, <SGPR the traversal keeps, or that it
	//                 reads back from a VGPR lane it keeps (KeptLane)>
	// (the compares may be their I32 forms).
	// with <flags> a record dword, shifted right or not: the layout's flags (bit 0 opaque, bit 1
	// alpha tested anyway).
	bool LaneLeafOpacity(size_t header, size_t back, size_t leaf, size_t from, size_t latch, uint32_t word,
	                     const Field& record, RayTracingSite& site, size_t& path, uint32_t& flag) const {
		site.alpha_off_load_pc = 0;
		path                   = Next(from, Lane(Opcode::S_AND_SAVEEXEC_B64), latch);
		if (path == npos || ScalarIndex(m_inst[path].src0) == NoRegister || m_inst[path].dst.kind != OperandKind::Sgpr) {
			return false;
		}
		size_t direct = npos;
		for (size_t i = path; i-- > from && direct == npos;) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::V_CNDMASK_B32 && IsConstant(inst.src0, 0) && IsConstant(inst.src1, 1) &&
			    inst.dst.kind == OperandKind::Vgpr && Plain(inst.dst) && ScalarIndex(CndmaskMask(inst)) != NoRegister) {
				direct = i;
			}
		}
		std::array<size_t, 5> atoms {npos, npos, npos, npos, npos};
		uint32_t              path_table = 0, direct_table = 0;
		if (direct == npos || !MaskTable(path, m_inst[path].src0, leaf, atoms, path_table) ||
		    !MaskTable(direct, CndmaskMask(m_inst[direct]), leaf, atoms, direct_table)) {
			return false;
		}
		flag = m_inst[direct].dst.reg;
		uint32_t opaque = 5, tested = 5, off = 5;
		auto&    layout = site.layout;
		for (uint32_t k = 0; k < atoms.size() && atoms[k] != npos; k++) {
			if (atoms[k] == ExecAtom) {
				return false;
			}
			const auto& atom = m_inst[atoms[k]];
			if ((atom.opcode == Opcode::V_CMP_NE_U32 || atom.opcode == Opcode::V_CMP_NE_I32) && opaque == 5) {
				// opaque: (flags | word >> 30) & 1.
				const auto* x   = IsConstant(atom.src0, 0) ? &atom.src1 : IsConstant(atom.src1, 0) ? &atom.src0 : nullptr;
				const auto  bit = x != nullptr ? VectorSource(atoms[k], *x, Opcode::V_AND_B32, leaf) : npos;
				const auto  any = bit != npos && IsConstant(m_inst[bit].src0, 1) && Plain(*x)
				                      ? VectorSource(bit, m_inst[bit].src1, Opcode::V_OR_B32, leaf)
				                      : npos;
				if (any == npos) {
					return false;
				}
				const auto& either = m_inst[any];
				const auto* shifted = either.src1.kind == OperandKind::Vgpr ? &either.src1 : &either.src0;
				const auto* flags   = either.src1.kind == OperandKind::Vgpr ? &either.src0 : &either.src1;
				const auto  lsr     = VectorSource(any, *shifted, Opcode::V_LSHRREV_B32, leaf);
				if (lsr == npos || !IsConstant(m_inst[lsr].src0, 30) || !IsVgpr(m_inst[lsr].src1, word) ||
				    !Plain(*shifted) || !Plain(*flags) || !LaneFlags(any, *flags, header, record, layout)) {
					return false;
				}
				opaque = k;
				continue;
			}
			// S_CSELECT_B64 exec, 0 of S_CMP_LG_U32 0, <x>.
			uint32_t x = NoRegister;
			if (!SelectedLanes(atoms[k], leaf, x)) {
				return false;
			}
			const auto w = ScalarWriter(atoms[k], x, header);
			if (w == npos) {
				// A value from before the loop: the switch.
				if (off != 5 || ScalarWriter(header, x, 0) == npos || !Keeps(header, back, x, false)) {
					return false;
				}
				site.alpha_off_sgpr = x;
				off                 = k;
				continue;
			}
			if (m_inst[w].opcode == Opcode::V_READLANE_B32 && KeptLane(w, header, back)) {
				// Or one the program keeps in a VGPR lane (GTA V 01.005): RT_INIT reads it back into
				// a free SGPR (see DerivePacketLane).
				if (off != 5) {
					return false;
				}
				site.alpha_off_load_pc = m_inst[w].pc;
				off                    = k;
				continue;
			}
			// flags & 2.
			const auto& bits = m_inst[w];
			const auto* f    = IsConstant(bits.src1, 2) ? &bits.src0 : IsConstant(bits.src0, 2) ? &bits.src1 : nullptr;
			if (tested != 5 || bits.opcode != Opcode::S_AND_B32 || f == nullptr || !LaneFlags(w, *f, header, record, layout)) {
				return false;
			}
			tested = k;
		}
		if (opaque == 5 || tested == 5 || off == 5) {
			return false;
		}
		uint32_t expected = 0;
		for (uint32_t c = 0; c < 32; c++) {
			const bool o = ((c >> opaque) & 1u) != 0, a = ((c >> tested) & 1u) != 0, z = ((c >> off) & 1u) != 0;
			expected |= (z || (o && !a)) ? 1u << c : 0u;
		}
		return direct_table == expected && path_table == ~expected;
	}

	// A lane mask S_CSELECT_B64 <m>, exec, 0 at `i` of S_CMP_LG_U32 (or S_CMP_LG_I32) 0, <x> (or
	// <x>, 0): x.
	bool SelectedLanes(size_t i, size_t from, uint32_t& x) const {
		const auto& sel = m_inst[i];
		if (sel.opcode != Lane(Opcode::S_CSELECT_B64) || sel.src0.kind != OperandKind::ExecLo || !IsConstant(sel.src1, 0)) {
			return false;
		}
		size_t cmp = npos;
		for (size_t j = i; j-- > from && cmp == npos;) {
			cmp = WritesScc(m_inst[j]) ? j : npos;
		}
		if (cmp == npos || (m_inst[cmp].opcode != Opcode::S_CMP_LG_U32 && m_inst[cmp].opcode != Opcode::S_CMP_LG_I32)) {
			return false;
		}
		const auto& inst  = m_inst[cmp];
		const auto* value = IsConstant(inst.src0, 0) ? &inst.src1 : IsConstant(inst.src1, 0) ? &inst.src0 : nullptr;
		x                 = value != nullptr ? ScalarIndex(*value) : NoRegister;
		return x != NoRegister;
	}

	// The instance flags SGPR operand `flags` of instruction `i`: a record dword, shifted right
	// (S_LSHR_B32 <dword>, <shift>) or not. Sets the layout's flags, or checks them.
	bool LaneFlags(size_t i, const Operand& flags, size_t from, const Field& record, RayTracingLayout& layout) const {
		Field    field;
		uint32_t shift = 0;
		if (!OperandField(i, flags, from, field)) {
			const auto lsr = ScalarSource(i, flags, Opcode::S_LSHR_B32, from);
			if (lsr == npos || m_inst[lsr].src1.kind != OperandKind::IntegerInlineConstant || m_inst[lsr].src1.value >= 29u ||
			    !OperandField(lsr, m_inst[lsr].src0, from, field)) {
				return false;
			}
			shift = m_inst[lsr].src1.value;
		}
		if (!SameRecord(field, record)) {
			return false;
		}
		if (layout.flags_dword == NoRegister) {
			layout.flags_dword = field.dword;
			layout.flags_shift = shift;
			return true;
		}
		return layout.flags_dword == field.dword && layout.flags_shift == shift;
	}

	// The facing of a PacketLane site's leaf code: S_AND_SAVEEXEC_B64 at `save` of
	//   hit & (!A | accept A) & (!B | accept B)
	// with hit and facing the compares of the triangle test `test` (see LeafFacingTable), and A and
	// B lane masks (S_CSELECT_B64 exec, 0) of "flags bit 2 clear and x == 0" and "... and x != 0"
	// for a wave-wide value x; one accepts the facing compare, the other its negation. The layout
	// takes the first's facing; when the second's lanes cull, the site's flip SGPR is not 0.
	bool LaneLeafFacing(size_t header, size_t back, size_t test, size_t save, RayTracingSite& site) const {
		std::array<size_t, 5> atoms {npos, npos, npos, npos, npos};
		uint32_t              table = 0;
		if (ScalarIndex(m_inst[save].src0) == NoRegister || !MaskTable(save, m_inst[save].src0, header, atoms, table)) {
			return false;
		}
		uint32_t facing = 5, hit = 5;
		std::array<uint32_t, 2> selector {5, 5};
		Field    flags;
		uint32_t bit = 0;
		bool     ne  = false;
		for (uint32_t k = 0; k < atoms.size() && atoms[k] != npos; k++) {
			if (atoms[k] == ExecAtom) {
				return false;
			}
			const auto& c = m_inst[atoms[k]];
			Field       f;
			uint32_t    b = 0;
			bool        n = false;
			if (facing == 5 && FacingCompare(atoms[k], test, header, f, b, n)) {
				facing = k, flags = f, bit = b, ne = n;
				continue;
			}
			const bool compare = c.family == Family::VOPC || magic_enum::enum_name(c.opcode).starts_with("V_CMP_");
			const auto reads   = [&](uint32_t vgpr) { return IsVgpr(c.src0, vgpr) || IsVgpr(c.src1, vgpr); };
			if (compare && hit == 5 && reads(m_inst[test].dst.reg) && VectorWriter(atoms[k], m_inst[test].dst.reg, header) == test) {
				hit = k;
			} else if (c.opcode == Lane(Opcode::S_CSELECT_B64) && (selector[0] == 5 || selector[1] == 5)) {
				selector[selector[0] == 5 ? 0 : 1] = k;
			} else {
				return false;
			}
		}
		if (facing == 5 || hit == 5 || selector[1] == 5) {
			return false;
		}
		// Which selector accepts the compare (`first`).
		const auto value = [&](uint32_t k, uint32_t c) { return ((c >> k) & 1u) != 0; };
		uint32_t   first = 2;
		for (uint32_t order = 0; order < 2; order++) {
			uint32_t expected = 0;
			for (uint32_t c = 0; c < 32; c++) {
				const bool f = value(facing, c), h = value(hit, c);
				const bool a = value(selector[order], c), b = value(selector[1 - order], c);
				expected |= h && (!a || f) && (!b || !f) ? 1u << c : 0u;
			}
			if (expected == table) {
				first = order;
			}
		}
		if (first == 2) {
			return false;
		}
		auto& layout = site.layout;
		if (layout.flags_dword == NoRegister) {
			layout.flags_dword = flags.dword;
			layout.flags_shift = bit - 3u;
		} else if (flags.dword != layout.flags_dword || bit != layout.flags_shift + 3u) {
			return false;
		}
		// The selectors' conditions: cull (flags bit 2 clear) and complementary wave-wide booleans.
		std::array<Condition, 2> cond;
		for (uint32_t s = 0; s < 2; s++) {
			uint32_t x = NoRegister;
			if (!SelectedLanes(atoms[selector[s]], header, x)) {
				return false;
			}
			Operand operand;
			operand.kind = OperandKind::Sgpr;
			operand.reg  = x;
			if (!SelectBits(atoms[selector[s]], operand, header, layout, back, cond[s], 0) || cond[s].both != -1 ||
			    cond[s].uniform == NoRegister || cond[s].lanes != NoRegister) {
				return false;
			}
		}
		if (!Complementary(header, cond[0].uniform, cond[1].uniform)) {
			return false;
		}
		site.facing_sgpr   = NoRegister;
		site.facing_mask   = NoRegister;
		site.both_facings  = false;
		layout.flip_facing = ne;
		site.flip_sgpr     = cond[1 - first].uniform;
		return true;
	}

	// BitsCondition, or also (in the loop) S_CSELECT_B32 <x>, 1, 0 of an SCC condition and the
	// S_AND_B32 of such values.
	bool SelectBits(size_t i, const Operand& x, size_t from, const RayTracingLayout& layout, size_t back, Condition& cond,
	                uint32_t depth) const {
		if (BitsCondition(i, x, from, layout, back, cond, 0)) {
			return true;
		}
		const auto reg = ScalarIndex(x);
		const auto w   = reg != NoRegister && depth < 4 ? ScalarWriter(i, reg, from) : npos;
		if (w == npos || ScalarIndex(m_inst[w].dst) != reg) {
			return false;
		}
		const auto& inst = m_inst[w];
		if (inst.opcode == Opcode::S_CSELECT_B32 && IsConstant(inst.src0, 1) && IsConstant(inst.src1, 0)) {
			return SccCondition(w, from, layout, back, cond, 0);
		}
		Condition a, b;
		if (inst.opcode != Opcode::S_AND_B32 || !SelectBits(w, inst.src0, from, layout, back, a, depth + 1) ||
		    !SelectBits(w, inst.src1, from, layout, back, b, depth + 1) || !Conjoin(a, b)) {
			return false;
		}
		cond = a;
		return true;
	}

	// Whether the wave-wide booleans `a` and `b` (S_CSELECT_B32 <r>, 1, 0 before `header`) are
	// x == 0 and x != 0 (S_CMP_EQ_U32 / S_CMP_LG_U32 with 0) of the same value of an SGPR x.
	bool Complementary(size_t header, uint32_t a, uint32_t b) const {
		std::array<size_t, 2>   source {npos, npos};
		std::array<uint32_t, 2> value {NoRegister, NoRegister};
		std::array<bool, 2>     equal {};
		for (uint32_t k = 0; k < 2; k++) {
			const auto w = ScalarWriter(header, k == 0 ? a : b, 0);
			if (w == npos || m_inst[w].opcode != Opcode::S_CSELECT_B32 || !IsConstant(m_inst[w].src0, 1) ||
			    !IsConstant(m_inst[w].src1, 0)) {
				return false;
			}
			size_t cmp = npos;
			for (size_t j = w; j-- > 0 && cmp == npos;) {
				cmp = WritesScc(m_inst[j]) ? j : npos;
			}
			if (cmp == npos || (m_inst[cmp].opcode != Opcode::S_CMP_EQ_U32 && m_inst[cmp].opcode != Opcode::S_CMP_LG_U32)) {
				return false;
			}
			const auto& inst = m_inst[cmp];
			const auto* x    = IsConstant(inst.src0, 0) ? &inst.src1 : IsConstant(inst.src1, 0) ? &inst.src0 : nullptr;
			value[k]         = x != nullptr ? ScalarIndex(*x) : NoRegister;
			if (value[k] == NoRegister) {
				return false;
			}
			source[k] = ScalarWriter(cmp, value[k], 0);
			equal[k]  = inst.opcode == Opcode::S_CMP_EQ_U32;
		}
		return value[0] == value[1] && source[0] == source[1] && source[0] != npos && equal[0] != equal[1];
	}

	// The leaf code on the path of a wave-wide condition: [begin, sample) may hold S_CMP_LG_U32 0, <c>
	// (or <c>, 0), later S_CBRANCH_SCC0 <else> (SCC unchanged), straight code, S_BRANCH <join>, then
	// <else> up to <join> (at or before the sample). Where c is not 0 the leaf code skips <else>:
	// `linear` is the program with the two branches and <else> replaced by S_NOP, `condition` c (an
	// SGPR the traversal keeps; NoRegister and `linear` empty without such a branch).
	bool LevelBranch(size_t header, size_t back, size_t begin, size_t sample, std::vector<Instruction>& linear,
	                 uint32_t& condition) const {
		linear.clear();
		condition = NoRegister;
		for (size_t i = begin; i < sample; i++) {
			if (m_inst[i].opcode != Opcode::S_CBRANCH_SCC0) {
				continue;
			}
			const auto other = IndexOf(m_inst[i].branch_target);
			if (other == npos || other <= i + 1 || other > sample || m_inst[other - 1].opcode != Opcode::S_BRANCH) {
				return false;
			}
			const auto join = IndexOf(m_inst[other - 1].branch_target);
			if (join == npos || join < other || join > sample) {
				return false;
			}
			for (size_t j = i + 1; j + 1 < other; j++) {
				if (IsDirectBranch(m_inst[j].opcode) || IsConditionalBranch(m_inst[j].opcode)) {
					return false;
				}
			}
			size_t cmp = npos;
			for (size_t j = i; j-- > begin && cmp == npos;) {
				cmp = WritesScc(m_inst[j]) ? j : npos;
			}
			if (cmp == npos || m_inst[cmp].opcode != Opcode::S_CMP_LG_U32) {
				return false;
			}
			const auto& inst = m_inst[cmp];
			const auto* c    = IsConstant(inst.src0, 0) ? &inst.src1 : IsConstant(inst.src1, 0) ? &inst.src0 : nullptr;
			condition        = c != nullptr && c->kind == OperandKind::Sgpr ? c->reg : NoRegister;
			if (condition == NoRegister || ScalarWriter(header, condition, 0) == npos || !Keeps(header, back, condition, false)) {
				return false;
			}
			linear = m_inst;
			for (size_t j = i; j < join; j++) {
				if (j == i || j + 1 >= other) {
					Instruction nop;
					nop.pc         = linear[j].pc;
					nop.word_count = linear[j].word_count;
					nop.family     = Family::SOPP;
					nop.opcode     = Opcode::S_NOP;
					linear[j]      = nop;
				}
			}
			return true;
		}
		return true;
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
		// 64-bit results, but not the 32-bit ones of 64-bit sources (S_FF1_I32_B64, S_BCNT1_I32_B64...).
		const bool wide      = lane_mask ? !m_wave32
		                                 : name.find("64") != std::string_view::npos && name.find("_I32_") == std::string_view::npos;
		const auto count     = inst.family == Family::SMEM ? std::max(inst.data_dwords, 1u) : (wide ? 2u : 1u);
		const auto covers    = [&](const Operand& dst, uint32_t n) {
			const auto first = ScalarIndex(dst);
			return first != NoRegister && reg >= first && reg < first + n;
		};
		return covers(inst.dst, count) || covers(inst.dst2, m_wave32 ? 1u : 2u);
	}

	// Whether the instruction writes EXEC (a destination, V_CMPX_* or S_*_SAVEEXEC_*).
	static bool WritesExec(const Instruction& inst) {
		const auto name = magic_enum::enum_name(inst.opcode);
		return inst.dst.kind == OperandKind::ExecLo || inst.dst.kind == OperandKind::ExecHi || name.starts_with("V_CMPX") ||
		       name.find("SAVEEXEC") != std::string_view::npos;
	}

	// Whether the instruction writes VGPR `reg`.
	static bool WritesVector(const Instruction& inst, uint32_t reg) {
		if (inst.dst.kind != OperandKind::Vgpr) {
			return false;
		}
		uint32_t count = 1;
		switch (inst.family) {
			// The dwords of the data, and the one that TFE or LWE (bits 16 and 17) add.
			case Family::MIMG: count = std::max(inst.data_dwords, 1u) + (((inst.raw[0] >> 16u) & 3u) != 0 ? 1u : 0u); break;
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
	// An SGPR the compiler spilled into a lane of a VGPR (V_WRITELANE_B32 <v>, <s>, lane) and read back
	// (V_READLANE_B32 <reg>, <v>, lane, both lanes constants) counts as written where the spilled
	// SGPR was, by the instruction that wrote that one.
	size_t ScalarWriter(size_t i, uint32_t reg, size_t from) const {
		if (reg == NoRegister) {
			return npos;
		}
		for (size_t j = i; j-- > from;) {
			if (WritesScalar(m_inst[j], reg)) {
				const auto spill = m_inst[j].opcode == Opcode::V_READLANE_B32 ? SpilledFrom(j, from) : npos;
				return spill != npos ? ScalarWriter(spill, m_inst[spill].src0.reg, from) : j;
			}
		}
		return npos;
	}

	// The V_WRITELANE_B32 of the SGPR that V_READLANE_B32 `read` reads back from its constant lane
	// (npos: none, or the VGPR may have changed otherwise).
	size_t SpilledFrom(size_t read, size_t from) const {
		const auto& inst = m_inst[read];
		if (inst.src0.kind != OperandKind::Vgpr || !IsLaneConstant(inst.src1)) {
			return npos;
		}
		for (size_t j = read; j-- > from;) {
			const auto& write = m_inst[j];
			if (!WritesVector(write, inst.src0.reg)) {
				continue;
			}
			if (write.opcode != Opcode::V_WRITELANE_B32 || write.dst.reg != inst.src0.reg || !IsLaneConstant(write.src1)) {
				return npos;
			}
			if (write.src1.value == inst.src1.value) {
				return write.src0.kind == OperandKind::Sgpr ? j : npos;
			}
		}
		return npos;
	}

	static bool IsLaneConstant(const Operand& operand) {
		return operand.kind == OperandKind::IntegerInlineConstant || operand.kind == OperandKind::LiteralConstant;
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
		// M2: the third dword of the material record, at offset 8 of the record's offset or at
		// offset 0 of S_ADD_I32 <record offset>, 8 (GTA V 01.005).
		const auto m2 = ScalarSource(key, m_inst[key].src0, Opcode::S_BUFFER_LOAD_DWORD, leaf);
		if (m2 == npos || m_inst[m2].src0.kind != OperandKind::Sgpr) {
			return false;
		}
		size_t         offset_at = m2;
		const Operand* offset    = &m_inst[m2].src1;
		if (m_inst[m2].offset != 8) {
			const auto add = m_inst[m2].offset == 0 ? ScalarSource(m2, m_inst[m2].src1, Opcode::S_ADD_I32, leaf) : npos;
			offset         = add == npos                       ? nullptr
			                 : IsConstant(m_inst[add].src1, 8) ? &m_inst[add].src0
			                 : IsConstant(m_inst[add].src0, 8) ? &m_inst[add].src1
			                                                   : nullptr;
			if (offset == nullptr || ScalarIndex(*offset) == NoRegister || !Plain(*offset)) {
				return false;
			}
			offset_at = add;
		}
		const auto material_vsharp = m_inst[m2].src0.reg;
		const auto record_offset   = ScalarIndex(*offset);
		if (!UserVsharp(ScalarRangeSource(m2, material_vsharp, 4, Opcode::S_LOAD_DWORDX4, leaf), tail,
		                alpha.material_user, alpha.material_offset)) {
			return false;
		}
		// Its offset: (geometry word + R[0x78]) * 12.
		const auto scale4 = ScalarSource(offset_at, *offset, Opcode::S_LSHL_B32, leaf);
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
		// A pair of the BLAS address at `i`: the pair of the triangle test's T#, or the same record
		// dwords loaded again (GTA V 01.005 loads them into other pairs in the leaf code).
		const auto blas_of = [&](size_t i, uint32_t pair) {
			if (blas_head(pair)) {
				return pair;
			}
			const auto blas = BlasPairOfRecordLoad(i, pair, header, back, record_sgpr);
			return blas != NoRegister && blas_head(blas) ? blas : NoRegister;
		};
		// Whether `operand` of `i` holds a field of the BLAS `blas`.
		const auto blas_field = [&](size_t i, const Operand& operand, uint32_t blas, uint32_t& dword) {
			const auto load = ScalarSource(i, operand, Opcode::S_LOAD_DWORD, leaf);
			return load != npos && m_inst[load].src0.kind == OperandKind::Sgpr &&
			       blas_of(load, m_inst[load].src0.reg) == blas &&
			       FieldLoad(i, operand, m_inst[load].src0.reg, leaf, dword) != npos;
		};
		bool geometry_ok = false, base_ok = false;
		for (const auto* operand: {&m_inst[sum].src0, &m_inst[sum].src1}) {
			if (FieldLoad(sum, *operand, record_sgpr, leaf, alpha.fields.material_dword) != npos) {
				base_ok = true;
				continue;
			}
			// The geometry word: B[B[table] + ((N >> 3) - B[first]) * 4] of the triangle node N.
			const auto load = ScalarSource(sum, *operand, Opcode::S_LOAD_DWORD, leaf);
			const auto base = load != npos && m_inst[load].offset == 0 ? ScalarIndex(m_inst[load].src0) : NoRegister;
			const auto blas = base != NoRegister ? blas_of(load, base) : NoRegister;
			if (blas == NoRegister) {
				continue;
			}
			const auto index = ScalarSource(load, m_inst[load].src1, Opcode::S_LSHL2_ADD_U32, leaf);
			if (index == npos || !blas_field(index, m_inst[index].src1, blas, alpha.fields.table_dword)) {
				continue;
			}
			const auto local = ScalarSource(index, m_inst[index].src0, Opcode::S_SUB_U32, leaf);
			if (local == npos || !blas_field(local, m_inst[local].src1, blas, alpha.fields.first_dword)) {
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
		// The reference: threshold * (M0 bit 0 ? factor : 1.0), compared with the sample
		// (V_CMP_GT_F32 <sampled>, <reference>); or threshold * factor for every material, the same
		// for every lane: V_CMP_LT_F32 <reference SGPR>, <sampled> of a V_READFIRSTLANE_B32 of it
		// (GTA V 01.005).
		size_t         compare = npos;
		const Operand* ref     = nullptr;
		for (size_t i = sample + 1; i < tail && compare == npos; i++) {
			const auto& cmp = m_inst[i];
			if (cmp.opcode == Opcode::V_CMP_GT_F32 && cmp.src0.kind == OperandKind::Vgpr && cmp.src0.reg == sampled.dst.reg &&
			    VectorSource(i, cmp.src0, Opcode::IMAGE_SAMPLE, sample) == sample) {
				compare = i;
				ref     = &cmp.src1;
			} else if (cmp.opcode == Opcode::V_CMP_LT_F32 && cmp.src0.kind == OperandKind::Sgpr &&
			           cmp.src1.kind == OperandKind::Vgpr && cmp.src1.reg == sampled.dst.reg && Plain(cmp.src0) &&
			           Plain(cmp.src1) && VectorSource(i, cmp.src1, Opcode::IMAGE_SAMPLE, sample) == sample) {
				compare = i;
				ref     = &cmp.src0;
			}
		}
		if (compare == npos) {
			return false;
		}
		size_t compared = compare;
		alpha.factor_always = ref->kind == OperandKind::Sgpr;
		if (alpha.factor_always) {
			compared = ScalarSource(compare, *ref, Opcode::V_READFIRSTLANE_B32, leaf);
			if (compared == npos || !Plain(m_inst[compared].src0)) {
				return false;
			}
			ref = &m_inst[compared].src0;
		}
		const auto reference = VectorSource(compared, *ref, Opcode::V_MUL_F32, leaf);
		if (reference == npos || !Plain(m_inst[reference].src0) || !Plain(m_inst[reference].src1) ||
		    !Plain(m_inst[reference].dst)) {
			return false;
		}
		bool threshold_ok = false, factor_ok = false;
		for (const auto* operand: {&m_inst[reference].src0, &m_inst[reference].src1}) {
			const auto j = ScalarWriter(reference, ScalarIndex(*operand), leaf);
			if (j == npos) {
				// For every material: the factor in an SGPR from before the loop that the traversal keeps.
				if (alpha.factor_always && operand->kind == OperandKind::Sgpr &&
				    ScalarWriter(header, operand->reg, 0) != npos && Keeps(header, back, operand->reg, false)) {
					alpha.factor_sgpr = operand->reg;
					factor_ok         = true;
				}
				continue;
			}
			const auto& inst = m_inst[j];
			if (inst.opcode == Opcode::S_MOV_B32 && inst.src0.kind == OperandKind::LiteralConstant) {
				alpha.threshold = inst.src0.value;
				threshold_ok    = true;
			} else if (!alpha.factor_always && inst.opcode == Opcode::S_CSELECT_B32 && inst.src0.kind == OperandKind::Sgpr &&
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
		uint32_t formatted = 0, uvs = 0, aligned = 0, buffer_table = NoRegister;
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
				// Its offset aligned down to 4 bytes: S_AND_B32 <offset>, -4, <x> (GTA V 01.005).
				const auto align  = ScalarSource(i, inst.src1, Opcode::S_AND_B32, leaf);
				const auto minus4 = [](const Operand& operand) {
					return operand.kind == OperandKind::IntegerInlineConstant && operand.signed_val == -4;
				};
				if (align != npos && (minus4(m_inst[align].src0) || minus4(m_inst[align].src1))) {
					aligned++;
				}
			}
		}
		if (formatted != 3 || uvs != 3 ||
		    !UserVsharp(ScalarRangeSource(table_load, buffer_table, 4, Opcode::S_LOAD_DWORDX4, leaf), tail,
		                alpha.buffer_user, alpha.buffer_offset)) {
			return false;
		}
		// The primitive index (W & 0x3fffffff) * 3, the vertex stride and uv offset (& 0xfc, or & 0xff
		// with the vertex offsets aligned down to 4 bytes).
		bool primitive_ok = false;
		uint32_t strides  = 0, wide_strides = 0;
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
			if (IsScalarConstantOp(i, Opcode::S_AND_B32, 0xff) && (is_m0(i, inst.src0) || is_m1(i, inst.src0))) {
				wide_strides |= is_m0(i, inst.src0) ? 1u : 2u;
			}
		}
		if (strides != 3) {
			if (strides != 0 || wide_strides != 3 || aligned != 3) {
				return false;
			}
			alpha.fields.vertex_mask  = 0xffu;
			alpha.fields.vertex_align = true;
		}
		if (!primitive_ok) {
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

	// A V# the traversal reads at the site entry: SGPRs `reg`..`reg` + 3 that one S_LOAD_DWORDX4 (or
	// wider) before `header` loaded and no instruction in [header, back] writes.
	bool EntryDescriptor(size_t i, uint32_t reg, size_t header, size_t back, uint32_t& user, uint32_t& offset) const {
		const auto load = ScalarWriter(i, reg, 0);
		if (load == npos || load >= header) {
			return false;
		}
		const auto& inst = m_inst[load];
		if ((inst.opcode != Opcode::S_LOAD_DWORDX4 && inst.opcode != Opcode::S_LOAD_DWORDX8 &&
		     inst.opcode != Opcode::S_LOAD_DWORDX16) ||
		    inst.dst.kind != OperandKind::Sgpr || reg < inst.dst.reg || (reg - inst.dst.reg) % 4 != 0 ||
		    reg + 4 > inst.dst.reg + inst.data_dwords) {
			return false;
		}
		for (uint32_t k = 0; k < 4; k++) {
			if (ScalarWriter(i, reg + k, 0) != load || !Keeps(header, back, reg + k, false)) {
				return false;
			}
		}
		user   = reg;
		offset = RayTracingAlphaDirect;
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

	// The pair of a BLAS T# (IsBlasPair) that holds what SGPR pair `pair` holds at `at`: both written
	// by S_LOAD_DWORDXn of the same dwords of the instance record pair `record`, with the same value
	// of that pair (NoRegister: none).
	uint32_t BlasPairOfRecordLoad(size_t at, uint32_t pair, size_t header, size_t back, uint32_t record) const {
		// The record byte offset that the load writing SGPR pair `reg` at `i` reads into it.
		const auto source = [&](size_t i, uint32_t reg, size_t& load) -> int64_t {
			load = ScalarWriter(i, reg, header);
			if (load == npos || ScalarWriter(i, reg + 1, header) != load) {
				return -1;
			}
			const auto& inst = m_inst[load];
			const auto  dst  = ScalarIndex(inst.dst);
			const bool  smem = inst.opcode == Opcode::S_LOAD_DWORDX2 || inst.opcode == Opcode::S_LOAD_DWORDX4 ||
			                  inst.opcode == Opcode::S_LOAD_DWORDX8 || inst.opcode == Opcode::S_LOAD_DWORDX16;
			if (!smem || dst == NoRegister || reg < dst || !IsSgpr(inst.src0, record) ||
			    (inst.src1.kind != OperandKind::Unknown && inst.src1.kind != OperandKind::Null)) {
				return -1;
			}
			return static_cast<int64_t>(inst.offset) + (reg - dst) * 4;
		};
		size_t     load   = npos;
		const auto offset = source(at, pair, load);
		if (offset < 0) {
			return NoRegister;
		}
		for (size_t i = header; i < back; i++) {
			const auto blas = m_inst[i].opcode == Opcode::IMAGE_BVH_INTERSECT_RAY ? BlasPair(i) : NoRegister;
			if (blas == NoRegister || blas == pair) {
				continue;
			}
			const auto bfe   = ScalarSource(i, m_inst[i].src1, Opcode::S_BFE_U64, header);
			size_t     other = npos;
			if (bfe != npos && source(bfe, blas, other) == offset &&
			    ScalarWriter(load, record, header) == ScalarWriter(other, record, header) &&
			    ScalarWriter(load, record + 1, header) == ScalarWriter(other, record + 1, header)) {
				return blas;
			}
		}
		return NoRegister;
	}

	// ---- The alpha test of GTA V's per lane leaf code (RayTracingAlphaLayout::Lane). ----
	//
	// The test of the packet sites (see DeriveAlphaTest) with vector loads in [begin, end): the
	// material dwords m * 3 + k with BUFFER_LOAD_DWORD idxen, the V#s and the T# of the tables in
	// waterfall loops (V_READFIRSTLANE_B32 of the key), the vertices with FLAT loads at the base
	// address of the vertex V# (BUFFER_LOAD_DWORDX2 idxen of the buffer table). The cone spread is
	// a VGPR or an SGPR, the reference a VGPR (closest hit sites), the leaf code's
	// threshold * (M0 bit 0 ? factor : 1) (PacketLane sites, whose tables may be loaded before the
	// traversal) or an SGPR the traversal keeps (GTA V 01.005's PacketLane sites).
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
		    sampled.dst.kind != OperandKind::Vgpr ||
		    sampled.src0.kind != OperandKind::Vgpr) {
			return false;
		}
		// sampled > reference: V_CMP_GT_F32 <sampled>, <reference> or V_CMP_LT_F32 <reference>, <sampled>.
		size_t         compare   = npos;
		const Operand* reference = nullptr;
		for (size_t i = sample + 1; i < end && compare == npos; i++) {
			const auto& cmp = m_inst[i];
			if (!IsVccLo(cmp.dst)) {
				continue;
			}
			if (cmp.opcode == Opcode::V_CMP_GT_F32 && VectorSource(i, cmp.src0, Opcode::IMAGE_SAMPLE, sample) == sample) {
				compare   = i;
				reference = &cmp.src1;
			} else if (cmp.opcode == Opcode::V_CMP_LT_F32 &&
			           VectorSource(i, cmp.src1, Opcode::IMAGE_SAMPLE, sample) == sample) {
				compare   = i;
				reference = &cmp.src0;
			}
		}
		if (compare == npos || !Plain(m_inst[compare].src0) || !Plain(m_inst[compare].src1) ||
		    (reference->kind != OperandKind::Vgpr && reference->kind != OperandKind::Sgpr) ||
		    VectorSource(end, vgpr(flag_vgpr), Opcode::V_CNDMASK_B32, compare) == npos) {
			return false;
		}
		// A reference the same for every lane: an SGPR from before the loop that the traversal keeps
		// (GTA V 01.005's performance mode computes threshold * factor there).
		alpha.reference_sgpr = NoRegister;
		if (reference->kind == OperandKind::Sgpr) {
			if (ScalarWriter(header, reference->reg, 0) == npos || !Keeps(header, back, reference->reg, false)) {
				return false;
			}
			alpha.reference_sgpr = reference->reg;
		}
		alpha.reference_vgpr = reference->kind == OperandKind::Vgpr ? reference->reg : NoRegister;
		// Or the leaf code computes the reference: threshold * (M0 bit 0 ? factor : 1) (see below).
		const auto product = VectorSource(compare, *reference, Opcode::V_MUL_F32, begin);
		if (reference->kind == OperandKind::Vgpr && product == npos &&
		    VectorWriter(compare, alpha.reference_vgpr, begin) != npos) {
			return false;
		}
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
		    !(UserDescriptor(material[0], material_vsharp, header, alpha.material_user, alpha.material_offset) ||
		      EntryDescriptor(material[0], material_vsharp, header, back, alpha.material_user, alpha.material_offset))) {
			return false;
		}
		// The computed reference: V_MUL_F32 <reference>, <threshold>, <x> with
		// V_CNDMASK_B32 <x>, 1.0, <factor SGPR>, (V_CMP_NE_U32 0, V_AND_B32 1, M0).
		if (product != npos) {
			const auto& mul       = m_inst[product];
			const bool  literal   = mul.src0.kind == OperandKind::LiteralConstant;
			const auto& threshold = literal ? mul.src0 : mul.src1;
			const auto  choice    = VectorSource(product, literal ? mul.src1 : mul.src0, Opcode::V_CNDMASK_B32, begin);
			if (threshold.kind != OperandKind::LiteralConstant || !Plain(mul.src0) || !Plain(mul.src1) ||
			    !Plain(mul.dst) || choice == npos) {
				return false;
			}
			const auto& select = m_inst[choice];
			const bool  one    = (select.src0.kind == OperandKind::FloatInlineConstant ||
			                      select.src0.kind == OperandKind::LiteralConstant) &&
			                     select.src0.value == 0x3f800000u;
			const auto lanes = ScalarIndex(CndmaskMask(select));
			const auto set   = lanes != NoRegister ? ScalarWriter(choice, lanes, begin) : npos;
			if (!one || select.src1.kind != OperandKind::Sgpr || !Plain(select.src0) || !Plain(select.src1) ||
			    !Plain(select.dst) || set == npos) {
				return false;
			}
			const auto& ne  = m_inst[set];
			const auto* m0  = IsConstant(ne.src0, 0) ? &ne.src1 : IsConstant(ne.src1, 0) ? &ne.src0 : nullptr;
			const auto  low = m0 != nullptr && Plain(*m0) ? VectorSource(set, *m0, Opcode::V_AND_B32, begin) : npos;
			if ((ne.opcode != Opcode::V_CMP_NE_U32 && ne.opcode != Opcode::V_CMP_NE_I32) || ScalarIndex(ne.dst) != lanes ||
			    low == npos || !IsConstant(m_inst[low].src0, 1) ||
			    VectorSource(low, m_inst[low].src1, Opcode::BUFFER_LOAD_DWORD, begin) != material[0]) {
				return false;
			}
			alpha.threshold      = threshold.value;
			alpha.factor_sgpr    = select.src1.reg;
			alpha.reference_vgpr = NoRegister;
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
		// The BLAS address: the pair of a BLAS T#, or the same record dwords loaded again (GTA V
		// 01.005 loads them into another pair in the leaf code).
		auto blas = m_inst[geometry].src1.reg;
		if (!IsBlasPair(header, back, blas, NoRegister)) {
			blas = BlasPairOfRecordLoad(geometry, blas, header, back, record_sgpr);
			if (blas == NoRegister) {
				return false;
			}
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
		    !(UserDescriptor(tsharp, m_inst[tsharp].src0.reg, header, texture_user, texture_offset) ||
		      EntryDescriptor(tsharp, m_inst[tsharp].src0.reg, header, back, texture_user, texture_offset))) {
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
		    !(UserDescriptor(triangle, buffer_table, header, alpha.buffer_user, alpha.buffer_offset) ||
		      EntryDescriptor(triangle, buffer_table, header, back, alpha.buffer_user, alpha.buffer_offset))) {
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
		const auto level = VectorSource(sample, vgpr(AddressReg(sampled, 2)), Opcode::V_MIN_F32, begin);
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
			    WritesVector(m_inst[i], alpha.lod_vgpr) || WritesVector(m_inst[i], alpha.reference_vgpr) ||
			    (product != npos && WritesScalar(m_inst[i], alpha.factor_sgpr))) {
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
			if (BoxTest(header, back, i, probe)) {
				box = i;
				site.t_max       = probe.t_max;
				for (uint32_t c = 0; c < 3; c++) {
					site.origin[c]      = probe.origin[c];
					site.origin_sgpr[c] = probe.origin_sgpr[c];
					site.direction[c]   = probe.direction[c];
				}
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
		bool leaf_walk    = false;
		if (accept == npos) {
			// A wave-wide walk whose leaf code keeps the nearest hit (ClosestLeafWalk, or with a
			// per lane BLAS walk ClosestLaneWalk), or one whose leaf code selects it
			// (ClosestSelectHit).
			auto walk = site;
			if (ClosestLeafWalk(header, back, record, walk)) {
				site      = walk;
				leaf_walk = true;
			} else if (walk = site; ClosestLaneWalk(header, back, record, walk)) {
				site      = walk;
				leaf_walk = true;
			} else {
				// The selected key goes to an SGPR pair the program does not use.
				const auto scratch = UnusedPair();
				if (!ClosestSelectHit(header, back, record.pair, site) || scratch == NoRegister) {
					return false;
				}
				site.both_facings  = false;
				site.instance_sgpr = scratch;
			}
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
		if (!leaf_walk) {
			site.mask_sgpr = site.instance_sgpr;
		}
		// The TLAS header address RT_INIT reads at the entry (ClosestOpaque sites, whose native
		// traversal can load it again: AddClosestTraversal).
		return leaf_walk || HeaderAtEntry(header, UnusedPair(), site);
	}

	// The next leaf list entry of a per lane BLAS walk that loads it into a pair other than the
	// node VGPR (Resident Evil Village, sce::Psr version 1: be0b848d2e06ec9b, f969c192323c215b):
	//   V_LSHRREV_B32 <x>, 3, <entry>; BUFFER_LOAD_DWORDX2 <entry>, <x> idxen
	//   V_ADD_NC_U32 <next>, 1, <x>                 the index of the entry after it
	//   S_MOV_B64 exec, <lanes>                     (the walk's lanes; not run in the native leaf)
	//   V_MOV_B32 <node>, <entry>                   the triangle node, last write of the node VGPR
	// before the walk's backward branch, with no other branch or write of those VGPRs in between.
	struct LaneEntry {
		size_t   load  = npos;
		size_t   move  = npos;
		uint32_t entry = NoRegister;
		uint32_t next  = NoRegister;
	};
	bool LaneEntryCopy(size_t test, size_t back, uint32_t node, LaneEntry& copy) const {
		const auto move = VectorWriter(back, node, test + 1);
		if (move == npos || m_inst[move].opcode != Opcode::V_MOV_B32 || !Plain(m_inst[move].dst) ||
		    !Plain(m_inst[move].src0) || m_inst[move].src0.kind != OperandKind::Vgpr) {
			return false;
		}
		const auto entry = m_inst[move].src0.reg;
		const auto load  = VectorWriter(move, entry, test + 1);
		if (load == npos || m_inst[load].opcode != Opcode::BUFFER_LOAD_DWORDX2 || !IsVgpr(m_inst[load].dst, entry) ||
		    m_inst[load].src0.kind != OperandKind::Vgpr || entry + 1u == node) {
			return false;
		}
		const auto index = m_inst[load].src0.reg;
		uint32_t   next  = NoRegister;
		for (size_t i = load + 1; i < move; i++) {
			const auto& inst = m_inst[i];
			if (IsDirectBranch(inst.opcode) || WritesVector(inst, entry) || WritesVector(inst, entry + 1u) ||
			    WritesVector(inst, node) || WritesVector(inst, index)) {
				return false;
			}
			if (WritesExec(inst)) {
				// Only a restore of the walk's lanes.
				if (inst.opcode != Lane(Opcode::S_MOV_B64) || inst.dst.kind != OperandKind::ExecLo ||
				    inst.src0.kind == OperandKind::Vgpr) {
					return false;
				}
				continue;
			}
			if (inst.opcode == Opcode::V_ADD_NC_U32 && inst.dst.kind == OperandKind::Vgpr && Plain(inst.dst) &&
			    IsConstant(inst.src0, 1) && IsVgpr(inst.src1, index) && next == NoRegister) {
				next = inst.dst.reg;
			}
		}
		if (next == NoRegister) {
			return false;
		}
		copy = {load, move, entry, next};
		return true;
	}

	// The accepted hit of the leaf code of LaneEntryCopy() walks: after V_CMPX_NE_U32 0, <flag>;
	// S_CBRANCH_EXECZ, a run of V_MOV_B32s of the distance (into t max), the primitive word (the
	// entry's second DWORD), the index of the next entry and the instance record pair (its SGPRs,
	// in any VGPRs). It has no barycentrics, and the guest's node result is that index, which the
	// hardware does not give: the native commit leaves those registers to the leaf code.
	bool ClosestLaneIndexHit(size_t begin_at, size_t latch, const Field& record, const LaneEntry& copy,
	                         RayTracingSite& site, size_t& flag_test) const {
		const auto move = [&](size_t i) {
			const auto& inst = m_inst[i];
			return inst.opcode == Opcode::V_MOV_B32 && inst.dst.kind == OperandKind::Vgpr && Plain(inst.dst) &&
			       Plain(inst.src0) && (inst.src0.kind == OperandKind::Vgpr || inst.src0.kind == OperandKind::Sgpr);
		};
		size_t t_move = npos;
		for (size_t i = begin_at; i < latch && t_move == npos; i++) {
			if (move(i) && m_inst[i].dst.reg == site.t_max && m_inst[i].src0.kind == OperandKind::Vgpr) {
				t_move = i;
			}
		}
		if (t_move == npos) {
			return false;
		}
		size_t begin = t_move, end = t_move + 1;
		while (begin > begin_at && move(begin - 1)) {
			begin--;
		}
		while (end < latch && move(end)) {
			end++;
		}
		if (begin < begin_at + 3 || m_inst[begin - 1].opcode != Opcode::S_CBRANCH_EXECZ) {
			return false;
		}
		flag_test     = begin - 2;
		uint32_t next = NoRegister;
		site.result_t = site.result_word = site.result_instance = site.result_instance_hi = NoRegister;
		site.result_u = site.result_v = site.result_node = NoRegister;
		for (size_t i = begin; i < end; i++) {
			const auto& inst = m_inst[i];
			const auto  dst  = inst.dst.reg;
			auto*       slot = &next;
			if (inst.src0.kind == OperandKind::Sgpr) {
				slot = inst.src0.reg == record.pair       ? &site.result_instance
				       : inst.src0.reg == record.pair + 1 ? &site.result_instance_hi
				                                          : nullptr;
			} else if (i == t_move) {
				slot = &site.result_t;
			} else if (inst.src0.reg == copy.entry + 1u) {
				slot = &site.result_word;
			} else if (inst.src0.reg != copy.next) {
				slot = nullptr;
			}
			if (slot == nullptr || *slot != NoRegister) {
				return false;
			}
			*slot = dst;
		}
		site.result_node = next;
		return site.result_t != NoRegister && site.result_word != NoRegister && next != NoRegister &&
		       site.result_instance != NoRegister && site.result_instance_hi != NoRegister;
	}

	// The alpha test of the leaf code of LaneEntryCopy() walks (Resident Evil Village, sce::Psr
	// version 1: be0b848d2e06ec9b, f969c192323c215b). The leaf code is the compiler's output for the
	// library's template, so it is recognized by its machine code: the words of [begin, end) (from
	// the code that follows the walk's box code, the triangle test, up to the flag test of the
	// accepted hit) are those of the template. What the template does with an accepted triangle
	// candidate (primitive word W, instance record R, the barycentrics i and j of the triangle test):
	//   opaque  = ((W >> 30) | R[0x3c] & 3) & 1, taken when (R[0x3c] & 2) == 0; every other
	//             candidate is alpha tested (the host's instance flags: bit 0 force opaque, bit 1
	//             force not opaque)
	//   entry   = e0..e3, the 16 bytes at R[0x70] * 16 of the V# `entries` (S_BUFFER_LOAD_DWORDX4)
	//   B       = the V# `buffers` (S_BUFFER_LOAD_*), read with offsets in bytes:
	//   stride  = B[((e1 >> 4) + 4) * 16]: byte 0 and byte 3 scale e2 and e3 (24 bit signed)
	//   buffers = B[e1 & ~12 ..] x4: dword 0 and 3 are the low addresses of the index and the vertex
	//             buffers (their high dword is 31, of the FLAT loads)
	//   indices = 3 16 bit indices at index buffer + byte0 * e2 + (W & 0x3fffffff) * 6 (aligned
	//             down to 4 bytes, the odd triangles taking the upper half-words)
	//   uv_k    = packed half floats at vertex buffer + byte3 * e3 + 4 * index k
	//   uv      = uv_0 * (1 - i - j) + uv_1 * i + uv_2 * j
	//   material at m = e0 & ~14 (B[m + 16 ..] x8, B[m + 0x48] x2, B[((e0 >> 4) + 3) * 16], B[(e0 & ~12) + 12]):
	//     texture 1 index m1 = dword 0, scale (dwords 5 and 6), offset u = dword 7; offset v = the dword
	//     B[((e0 >> 4) + 3) * 16]; m2 = B[(e0 & ~12) + 12]
	//     m1 != -1: sample the texture (m1 * 48 + 16 into the heap of T#s) at uv, level 2: its
	//     x channel > 0.5 selects the first of B[m + 0x48] for both scales, its y channel > 0.5
	//     the second one (y wins)
	//     uv' = (u * scale u + offset u, v * scale v + offset v)
	//     m2 != -1: alpha = x channel of the texture m2 at uv', level 2; else 1
	//   accepted when alpha > 0.5
	// The tables are SGPRs of the program, loaded from user data before the traversal.
	bool DeriveEntryAlphaTest(size_t begin, size_t end, const Field& record, size_t header,
	                          RayTracingSite& site) const {
		constexpr uint64_t TemplateHash = 0x4856271b2855201bull;
		uint64_t           hash         = 1469598103934665603ull;
		for (size_t i = begin; i < end; i++) {
			const auto& inst = m_inst[i];
			for (uint32_t k = 0; k < inst.word_count && k < MaxInstructionRawWords; k++) {
				hash = (hash ^ inst.raw[k]) * 1099511628211ull;
			}
		}
		if (hash != TemplateHash) {
			return false;
		}
		// The V#s of the entry table (the first S_BUFFER_LOAD_DWORDX4) and of the buffers (the
		// first S_BUFFER_LOAD_DWORD), loaded from user data and unchanged until the site.
		uint32_t entries = NoRegister, buffers = NoRegister, flags_reg = NoRegister, tested_reg = NoRegister;
		bool     material = false;
		for (size_t i = begin; i < end; i++) {
			const auto& inst = m_inst[i];
			if (entries == NoRegister && inst.opcode == Opcode::S_BUFFER_LOAD_DWORDX4 && inst.src0.kind == OperandKind::Sgpr) {
				entries = inst.src0.reg;
			} else if (buffers == NoRegister && inst.opcode == Opcode::S_BUFFER_LOAD_DWORD && inst.src0.kind == OperandKind::Sgpr) {
				buffers = inst.src0.reg;
			} else if (flags_reg == NoRegister && inst.opcode == Opcode::V_OR_B32 && inst.src0.kind == OperandKind::Sgpr) {
				flags_reg = inst.src0.reg;
			} else if (tested_reg == NoRegister && inst.opcode == Opcode::S_CMP_LG_I32 && inst.src1.kind == OperandKind::Sgpr) {
				tested_reg = inst.src1.reg;
			} else if (!material && inst.opcode == Opcode::S_LOAD_DWORD && IsSgpr(inst.src0, record.pair) && inst.offset == 0x70) {
				material = true;
			}
		}
		if (entries == NoRegister || buffers == NoRegister || flags_reg == NoRegister || tested_reg == NoRegister || !material) {
			return false;
		}
		const auto user_vsharp = [&](uint32_t first) {
			Field base;
			for (uint32_t k = 0; k < 4; k++) {
				Field field;
				if (!RecordField(header, first + k, 0, field) || (k != 0 && (field.pair != base.pair || field.load != base.load ||
				                                                         field.dword != base.dword + k))) {
					return false;
				}
				if (k == 0) {
					base = field;
				}
			}
			for (size_t j = 0; j < base.load; j++) {
				if (WritesScalar(m_inst[j], base.pair) || WritesScalar(m_inst[j], base.pair + 1)) {
					return false;
				}
			}
			return true;
		};
		if (!user_vsharp(entries) || !user_vsharp(buffers)) {
			return false;
		}
		// The instance flags: the registers the template tests are R[0x3c] & 3 and that & 2.
		Operand flags_operand;
		flags_operand.kind = OperandKind::Sgpr;
		flags_operand.reg  = flags_reg;
		Operand tested_operand = flags_operand;
		tested_operand.reg     = tested_reg;
		Field    flags;
		uint32_t mask = 0;
		const auto tested = ScalarSource(begin, tested_operand, Opcode::S_AND_B32, header);
		if (!MaskedField(begin, flags_operand, header, flags, mask) || flags.pair != record.pair || flags.dword != 15 ||
		    mask != 3 || tested == npos || !IsSgpr(m_inst[tested].src0, flags_reg) || !IsConstant(m_inst[tested].src1, 2)) {
			return false;
		}
		auto& alpha                  = site.alpha;
		alpha                        = {};
		alpha.valid                  = true;
		alpha.layout                 = RayTracingAlphaLayout::Entry;
		alpha.fields.material_dword  = 0x70 / 4;
		alpha.material_user          = entries;
		alpha.material_offset        = RayTracingAlphaDirect;
		alpha.buffer_user            = buffers;
		alpha.buffer_offset          = RayTracingAlphaDirect;
		alpha.threshold              = std::bit_cast<uint32_t>(0.5f);
		alpha.lod_offset             = std::bit_cast<uint32_t>(2.0f);
		site.layout.flags_dword      = flags.dword;
		site.layout.flags_shift      = 0;
		return true;
	}

	// ---- Closest hit walks whose instance code has a per lane BLAS walk (ClosestLane). ----
	//
	// Marvel's Wolverine: past the loop header's node type test (whose S_CBRANCH_SCC1 goes to the
	// box code) comes the instance setup, then a per lane BLAS walk, a loop of the form
	//   setup      [loop header, BLAS head): the instance record and its tests, which leave for
	//              the cull exit; the object space ray, the BLAS T#, the BLAS root in the node VGPR
	//   BLAS head  the target of the walk's backward S_CBRANCH_EXECNZ; then its only
	//              IMAGE_BVH_INTERSECT_RAY, with the node VGPR as its first address (src0)
	//   leaf code  for the lanes whose node is a triangle node: the triangle test's results, the
	//              lane's alpha test and, for an accepted hit, the V_MOV_B32s of ClosestLaneHit()
	//              and an S_MOV_B32 exec, <saved> that ends it (the tail: the lane's next node)
	//   load       V_LSHRREV_B32 <x>, 3, <node>; BUFFER_LOAD_DWORDX2 <node>, <x> idxen before the
	//              backward branch: the lane's next leaf list entry, from its pointer
	//              (or the form of LaneEntryCopy(), Resident Evil Village: the entry goes to another
	//              pair, and the accepted hit is that of ClosestLaneIndexHit())
	// The setup spills no SGPR (V_WRITELANE_B32) that would need restoring.
	bool ClosestLaneWalk(size_t header, size_t back, const Field& record, RayTracingSite& site) const {
		size_t tlas_test = npos;
		if (!NodeTypeTest(header + 1, tlas_test) || tlas_test + 2 >= back ||
		    m_inst[tlas_test + 1].opcode != Opcode::S_CBRANCH_SCC1) {
			return false;
		}
		// The BLAS walk: the backward S_CBRANCH_EXECNZ into the instance code with the earliest
		// target (the leaf code's own loops are inside it).
		size_t blas = npos, blas_back = npos;
		for (size_t i = tlas_test + 2; i < back; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::S_CBRANCH_EXECNZ && inst.branch_target > m_inst[tlas_test + 1].pc &&
			    inst.branch_target < inst.pc && (blas == npos || inst.branch_target <= m_inst[blas].pc)) {
				blas      = IndexOf(inst.branch_target);
				blas_back = i;
			}
		}
		if (blas == npos || blas_back == npos) {
			return false;
		}
		size_t test = npos;
		for (size_t i = blas; i < blas_back; i++) {
			if (m_inst[i].opcode == Opcode::IMAGE_BVH_INTERSECT_RAY) {
				if (test != npos) {
					return false;
				}
				test = i;
			}
		}
		if (test == npos || m_inst[test].src0.kind != OperandKind::Vgpr) {
			return false;
		}
		const auto node = m_inst[test].src0.reg;
		// The load of the next leaf list entry.
		size_t load = npos;
		for (size_t i = blas_back; i-- > test && load == npos;) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::BUFFER_LOAD_DWORDX2 && IsVgpr(inst.dst, node)) {
				load = i;
			}
		}
		LaneEntry  copy;
		const bool copied = load == npos && LaneEntryCopy(test, blas_back, node, copy);
		if (copied) {
			load = copy.load;
		}
		// The VGPR of the entry pointer the load reads (and the entry it writes).
		const auto entry = copied ? copy.entry : node;
		const auto shift = load != npos && m_inst[load].src0.kind == OperandKind::Vgpr
		                       ? VectorWriter(load, m_inst[load].src0.reg, test)
		                       : npos;
		if (shift == npos || m_inst[shift].opcode != Opcode::V_LSHRREV_B32 || !IsConstant(m_inst[shift].src0, 3) ||
		    !IsVgpr(m_inst[shift].src1, entry)) {
			return false;
		}
		for (size_t i = shift + 1; i < load; i++) {
			if (IsDirectBranch(m_inst[i].opcode) || WritesExec(m_inst[i]) ||
			    WritesVector(m_inst[i], m_inst[load].src0.reg) || WritesVector(m_inst[i], entry)) {
				return false;
			}
		}
		// The setup: no spills, and its exits other than the walk are the box code (the header's
		// branch) and the cull exit.
		size_t cull = npos;
		for (size_t i = header; i < blas; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::V_WRITELANE_B32 && inst.src0.kind == OperandKind::Sgpr) {
				return false;
			}
			if (!IsDirectBranch(inst.opcode) || i == tlas_test + 1 ||
			    (inst.branch_target > m_inst[header].pc && inst.branch_target <= m_inst[blas].pc)) {
				continue;
			}
			const auto target = IndexOf(inst.branch_target);
			if (inst.opcode == Opcode::S_BRANCH || target == npos || (cull != npos && target != cull)) {
				return false;
			}
			cull = target;
		}
		if (cull == npos || VectorWriter(blas, node, header) == npos) {
			return false;
		}
		// The accepted hit, then the end of the leaf code.
		size_t flag_test = npos;
		if (copied ? !ClosestLaneIndexHit(test + 1, blas_back, record, copy, site, flag_test)
		           : !ClosestLaneHit(test + 1, blas_back, test, NoRegister, node, record, site, flag_test)) {
			return false;
		}
		size_t tail = npos;
		for (size_t i = flag_test + 2; i < blas_back && tail == npos; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Lane(Opcode::S_MOV_B64) && inst.dst.kind == OperandKind::ExecLo &&
			    inst.src0.kind == OperandKind::Sgpr) {
				tail = i + 1;
			}
		}
		if (tail == npos || tail >= shift) {
			return false;
		}
		// The box code: from the first S_CBRANCH_EXECZ after the test that skips forward to
		// S_ANDN2 exec, <lanes>, exec (the lanes with a triangle node), with no branch before it.
		size_t box = npos;
		for (size_t i = test + 1; i < tail && box == npos; i++) {
			const auto& inst = m_inst[i];
			if (IsDirectBranch(inst.opcode)) {
				const auto target = IndexOf(inst.branch_target);
				if (inst.opcode == Opcode::S_CBRANCH_EXECZ && target != npos && target > i && target < tail &&
				    m_inst[target].opcode == Lane(Opcode::S_ANDN2_B64) &&
				    m_inst[target].dst.kind == OperandKind::ExecLo && m_inst[target].src1.kind == OperandKind::ExecLo) {
					box = i;
				}
				break;
			}
		}
		// The leaf code leaves only into the rest of the walk.
		for (size_t i = blas; i < tail; i++) {
			const auto& inst = m_inst[i];
			if (IsDirectBranch(inst.opcode) &&
			    (inst.branch_target < m_inst[blas].pc || inst.branch_target > m_inst[blas_back].pc)) {
				return false;
			}
		}
		const auto scratch = UnusedPair();
		if (scratch == NoRegister) {
			return false;
		}
		// Resident Evil Village's alpha test: the traversal then runs no guest leaf code.
		const auto leaf_start = box != npos ? IndexOf(m_inst[box].branch_target) : npos;
		if (copied && (leaf_start == npos || !DeriveEntryAlphaTest(leaf_start, flag_test, record, header, site))) {
			site.alpha       = {};
			site.result_node = NoRegister;
		}
		site.kind              = RayTracingKind::ClosestLane;
		site.setup_pc          = m_inst[header].pc;
		site.cull_exit_pc      = m_inst[cull].pc;
		site.blas_head_pc      = m_inst[blas].pc;
		site.leaf_pc           = m_inst[blas].pc;
		site.tail_pc           = m_inst[tail].pc;
		site.blas_end_pc       = EndPc(m_inst[blas_back]);
		site.leaf_vgpr         = entry;
		site.leaf_load_pc      = m_inst[shift].pc;
		site.leaf_load_end_pc  = EndPc(m_inst[copied ? copy.move : load]);
		site.leaf_box_pc       = box != npos ? m_inst[box].pc : 0;
		site.leaf_box_end_pc   = box != npos ? m_inst[box].branch_target : 0;
		site.leaf_sgpr         = 0;
		site.instance_sgpr     = m_inst[header + 1].src0.reg;
		site.mask_sgpr         = scratch;
		site.saved_exec_sgpr   = scratch;
		site.alpha_off_sgpr    = NoAlphaOff;
		site.after_leaf_pc     = 0;
		site.after_leaf_end_pc = 0;
		site.both_facings      = true;
		return true;
	}

	// ---- Closest hit walks whose instance code walks the BLAS itself (ClosestLeaf). ----
	//
	// Past the loop header's node type test (whose S_CBRANCH_SCC1 goes to the box code) comes the
	// instance code, which holds a second loop of the same form, the BLAS walk:
	//   setup      [loop header, BLAS header): the instance record and its mask, whose tests leave
	//              for the cull exit; the object space ray, the BLAS T#, SGPRs it spills with
	//              V_WRITELANE_B32, and the BLAS root in the BLAS loop's node register
	//   leaf code  [BLAS header, the target of its node type test's S_CBRANCH_SCC1): one triangle
	//              test, the opacity test, the call of the any-hit function, then V_CMPX_* and
	//              S_CBRANCH_EXECZ before the V_MOV_B32s of an accepted hit (see LeafHit); it
	//              leaves only into the rest of the BLAS loop
	//   restores   after the BLAS loop (the S_CBRANCH_SCC0 just before its backward branch that
	//              skips it): V_READLANE_B32s of every SGPR the setup spilled
	bool ClosestLeafWalk(size_t header, size_t back, const Field& record, RayTracingSite& site) const {
		size_t tlas_test = npos;
		if (!NodeTypeTest(header + 1, tlas_test) || tlas_test + 1 >= back ||
		    m_inst[tlas_test + 1].opcode != Opcode::S_CBRANCH_SCC1) {
			return false;
		}
		// The BLAS loop.
		size_t blas = npos, blas_back = npos;
		for (size_t i = tlas_test + 2; i < back && blas == npos; i++) {
			const auto b = IsClosestHeader(i) ? BackwardBranchTo(i) : npos;
			if (b != npos && b < back) {
				blas      = i;
				blas_back = b;
			}
		}
		size_t blas_test = npos;
		if (blas == npos || !NodeTypeTest(blas + 1, blas_test) ||
		    m_inst[blas_test + 1].opcode != Opcode::S_CBRANCH_SCC1) {
			return false;
		}
		const auto tail = IndexOf(m_inst[blas_test + 1].branch_target);
		if (tail == npos || tail <= blas_test + 1 || tail >= blas_back) {
			return false;
		}
		const auto blas_end  = EndPc(m_inst[blas_back]);
		const auto leaf_sgpr = m_inst[blas + 1].src0.reg;
		// The setup falls through into the BLAS walk; its other exits are the box code (the
		// header's branch) and the cull exit.
		size_t cull = npos;
		for (size_t i = header; i < blas; i++) {
			const auto& inst = m_inst[i];
			if (!IsDirectBranch(inst.opcode) || i == tlas_test + 1 ||
			    (inst.branch_target > m_inst[header].pc && inst.branch_target < m_inst[blas].pc)) {
				continue;
			}
			const auto target = IndexOf(inst.branch_target);
			if (inst.opcode == Opcode::S_BRANCH || target == npos || (cull != npos && target != cull)) {
				return false;
			}
			cull = target;
		}
		if (cull == npos || ScalarWriter(blas, leaf_sgpr, header) == npos) {
			return false;
		}
		// The leaf code: one triangle test; it leaves only into the rest of the BLAS loop.
		size_t test = npos;
		for (size_t i = blas; i < tail; i++) {
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::IMAGE_BVH_INTERSECT_RAY) {
				if (test != npos) {
					return false;
				}
				test = i;
			}
			if (IsDirectBranch(inst.opcode) &&
			    (inst.branch_target < m_inst[blas].pc || inst.branch_target >= blas_end)) {
				return false;
			}
		}
		if (test == npos) {
			return false;
		}
		// The restores after the BLAS loop.
		size_t blas_exit = npos;
		for (size_t i = blas_back; i-- > tail && i + 16 > blas_back && blas_exit == npos;) {
			if (m_inst[i].opcode == Opcode::S_CBRANCH_SCC0 && m_inst[i].branch_target == blas_end) {
				blas_exit = i;
			}
		}
		if (blas_exit == npos) {
			return false;
		}
		const auto spill_of = [&](const Instruction& save, const Instruction& restore) {
			return save.opcode == Opcode::V_WRITELANE_B32 && save.src0.kind == OperandKind::Sgpr &&
			       restore.opcode == Opcode::V_READLANE_B32 && IsSgpr(restore.dst, save.src0.reg) &&
			       SameOperand(save.dst, restore.src0) && SameOperand(save.src1, restore.src1);
		};
		const auto spilled = [&](size_t k) {
			for (size_t j = header; j < blas; j++) {
				if (spill_of(m_inst[j], m_inst[k])) {
					return true;
				}
			}
			return false;
		};
		size_t restores_end = blas_back + 1;
		while (restores_end < m_inst.size() && spilled(restores_end)) {
			restores_end++;
		}
		for (size_t j = header; j < blas; j++) {
			const auto& save = m_inst[j];
			if (save.opcode != Opcode::V_WRITELANE_B32 || save.src0.kind != OperandKind::Sgpr) {
				continue;
			}
			bool restored = false;
			for (size_t k = blas_back + 1; k < restores_end && !restored; k++) {
				restored = spill_of(save, m_inst[k]);
			}
			if (!restored) {
				return false;
			}
		}
		site.after_leaf_pc     = restores_end > blas_back + 1 ? m_inst[blas_back + 1].pc : 0;
		site.after_leaf_end_pc = restores_end > blas_back + 1 ? EndPc(m_inst[restores_end - 1]) : 0;
		// The accepted hit, the instance flags and the facing the triangle test accepts.
		uint32_t   word_sgpr = NoRegister;
		const auto scratch   = UnusedPair();
		if (scratch == NoRegister || !LeafHit(blas, tail, test, record, site, word_sgpr) ||
		    !LeafFlags(blas, tail, header, word_sgpr, record, site.layout) ||
		    !LeafFacing(blas, tail, header, back, site)) {
			return false;
		}
		site.kind            = RayTracingKind::ClosestLeaf;
		site.setup_pc        = m_inst[header].pc;
		site.cull_exit_pc    = m_inst[cull].pc;
		site.blas_head_pc    = m_inst[blas].pc;
		site.leaf_pc         = m_inst[blas].pc;
		site.tail_pc         = m_inst[tail].pc;
		site.blas_end_pc     = blas_end;
		site.leaf_sgpr       = leaf_sgpr;
		site.instance_sgpr   = m_inst[header + 1].src0.reg;
		site.mask_sgpr       = scratch;
		site.saved_exec_sgpr = scratch;
		site.alpha_off_sgpr  = NoAlphaOff;
		site.both_facings    = false;
		return true;
	}

	// Whether instruction `i` is in an inlined call (the function's own code).
	bool InCall(size_t i) const {
		return std::ranges::any_of(m_bodies, [&](const InlinedCallBody& body) {
			return body.begin <= m_inst[i].pc && m_inst[i].pc < body.end;
		});
	}

	// The last instruction in [from, i) outside inlined calls that writes VGPR `reg` (a called
	// function keeps the registers its caller reads after the call).
	size_t CallerVectorWriter(size_t i, uint32_t reg, size_t from) const {
		for (size_t j = i; j-- > from;) {
			if (!InCall(j) && WritesVector(m_inst[j], reg)) {
				return j;
			}
		}
		return npos;
	}

	// The accepted hit of a ClosestLeaf site's leaf code: after V_CMPX_*; S_CBRANCH_EXECZ, a run
	// of V_MOV_B32s of the distance (into t max), the instance record pair (the record's SGPRs),
	// the triangle node (the SGPR the triangle test's node VGPR was moved from), the primitive
	// word (the other SGPR) and the barycentrics (V_MUL_F32s of the test's second and third
	// results). Gives the SGPR of the primitive word.
	bool LeafHit(size_t leaf, size_t tail, size_t test, const Field& record, RayTracingSite& site,
	             uint32_t& word_sgpr) const {
		const auto move = [&](size_t i) {
			const auto& inst = m_inst[i];
			return inst.opcode == Opcode::V_MOV_B32 && inst.dst.kind == OperandKind::Vgpr && Plain(inst.src0) &&
			       (inst.src0.kind == OperandKind::Vgpr || inst.src0.kind == OperandKind::Sgpr);
		};
		size_t t_move = npos;
		for (size_t i = test + 1; i < tail && t_move == npos; i++) {
			if (move(i) && m_inst[i].dst.reg == site.t_max && m_inst[i].src0.kind == OperandKind::Vgpr && !InCall(i)) {
				t_move = i;
			}
		}
		if (t_move == npos) {
			return false;
		}
		size_t begin = t_move, end = t_move + 1;
		while (begin > test + 1 && move(begin - 1)) {
			begin--;
		}
		while (end < tail && move(end)) {
			end++;
		}
		if (begin < test + 3 || m_inst[begin - 1].opcode != Opcode::S_CBRANCH_EXECZ ||
		    !magic_enum::enum_name(m_inst[begin - 2].opcode).starts_with("V_CMPX_")) {
			return false;
		}
		const auto node_move = CallerVectorWriter(test, AddressReg(m_inst[test], 0), leaf);
		if (node_move == npos || m_inst[node_move].opcode != Opcode::V_MOV_B32 ||
		    m_inst[node_move].src0.kind != OperandKind::Sgpr) {
			return false;
		}
		const auto node_sgpr = m_inst[node_move].src0.reg;
		const auto result    = m_inst[test].dst.reg;
		uint32_t   high      = NoRegister;
		site.result_t = site.result_word = site.result_instance = NoRegister;
		site.result_u = site.result_v = site.result_node = NoRegister;
		word_sgpr = NoRegister;
		for (size_t i = begin; i < end; i++) {
			const auto& inst = m_inst[i];
			const auto  dst  = inst.dst.reg;
			if (inst.src0.kind == OperandKind::Vgpr) {
				if (dst == site.t_max) {
					site.result_t = dst;
					continue;
				}
				// A barycentric: the product of the test's second or third result.
				const auto mul = CallerVectorWriter(i, inst.src0.reg, test + 1);
				if (mul == npos || m_inst[mul].opcode != Opcode::V_MUL_F32) {
					return false;
				}
				const auto reads = [&](uint32_t reg) {
					const auto& m = m_inst[mul];
					return ((m.src0.kind == OperandKind::Vgpr && m.src0.reg == reg) ||
					        (m.src1.kind == OperandKind::Vgpr && m.src1.reg == reg)) &&
					       CallerVectorWriter(mul, reg, test) == test;
				};
				const bool u = reads(result + 2), v = !u && reads(result + 3);
				auto&      bary = u ? site.result_u : site.result_v;
				if ((!u && !v) || bary != NoRegister) {
					return false;
				}
				bary = dst;
			} else if (inst.src0.reg == record.pair) {
				site.result_instance = dst;
			} else if (inst.src0.reg == record.pair + 1) {
				high = dst;
			} else if (inst.src0.reg == node_sgpr) {
				site.result_node = dst;
			} else if (word_sgpr == NoRegister) {
				word_sgpr        = inst.src0.reg;
				site.result_word = dst;
			} else {
				return false;
			}
		}
		return site.result_t != NoRegister && site.result_instance != NoRegister && high == site.result_instance + 1 &&
		       site.result_u != NoRegister && site.result_v != NoRegister && site.result_node != NoRegister &&
		       site.result_word != NoRegister;
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
	// With `vsharp` (sce::Psr BVH version 10, Marvel's Wolverine), the header pair may also be an
	// S_BUFFER_LOAD_DWORDX2 <header>, <V#>, <offset> of a V# the program loads from its user data
	// pointer (S_LOAD_DWORDX4): then `vsharp` is that V#'s offset from the pointer (else
	// NoRegister) and the header address is at the V#'s base + `offset`.
	bool FindTlasSource(size_t header, uint32_t header_sgpr, uint32_t& user_sgpr, uint32_t& offset,
	                    size_t* load_index = nullptr, uint32_t* vsharp = nullptr) const {
		if (vsharp != nullptr) {
			*vsharp = NoRegister;
		}
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
			if (load.src0.kind != OperandKind::Sgpr ||
			    (load.src1.kind != OperandKind::Unknown && load.src1.kind != OperandKind::Null)) {
				return false;
			}
			size_t pointer_load = i;
			if (load.opcode == Opcode::S_BUFFER_LOAD_DWORDX2 && vsharp != nullptr) {
				// The V#: S_LOAD_DWORDX4 from the user data pointer, unchanged up to the load.
				const auto v = ScalarWriter(i, load.src0.reg, 0);
				if (v == npos || m_inst[v].opcode != Opcode::S_LOAD_DWORDX4 || !IsSgpr(m_inst[v].dst, load.src0.reg) ||
				    m_inst[v].src0.kind != OperandKind::Sgpr ||
				    (m_inst[v].src1.kind != OperandKind::Unknown && m_inst[v].src1.kind != OperandKind::Null)) {
					return false;
				}
				for (uint32_t k = 1; k < 4; k++) {
					if (ScalarWriter(i, load.src0.reg + k, 0) != v) {
						return false;
					}
				}
				pointer_load = v;
				*vsharp      = static_cast<uint32_t>(m_inst[v].offset);
			} else if (load.opcode != Opcode::S_LOAD_DWORDX2) {
				return false;
			}
			// Nothing before writes the pair on a path to the load (the other cases of a switch
			// may reuse it).
			const auto base = m_inst[pointer_load].src0.reg;
			i               = pointer_load;
			for (size_t j = 0; j < i; j++) {
				const auto& dst = m_inst[j].dst;
				if (dst.kind == OperandKind::Sgpr &&
				    (dst.reg == base || dst.reg == base + 1 || (dst.reg + 1 == base && WritesPair(m_inst[j]))) &&
				    Reaches(j + 1, i)) {
					return false;
				}
			}
			user_sgpr = base;
			offset    = static_cast<uint32_t>(load.offset);
			if (load_index != nullptr) {
				*load_index = i;
			}
			return true;
		}
		return false;
	}

	// Whether instruction `to` can run after instruction `from` runs (through fall through and
	// direct branches; an indirect branch can reach any instruction).
	bool Reaches(size_t from, size_t to) const {
		std::vector<bool>   seen(m_inst.size());
		std::vector<size_t> work {from};
		while (!work.empty()) {
			const auto i = work.back();
			work.pop_back();
			if (i >= m_inst.size() || seen[i]) {
				continue;
			}
			if (i == to) {
				return true;
			}
			seen[i]          = true;
			const auto& inst = m_inst[i];
			if (inst.opcode == Opcode::S_SETPC_B64 || inst.opcode == Opcode::S_SWAPPC_B64) {
				return true;
			}
			if (inst.opcode == Opcode::S_ENDPGM) {
				continue;
			}
			if (IsDirectBranch(inst.opcode)) {
				const auto target = IndexOf(inst.branch_target);
				if (target == npos) {
					return true;
				}
				work.push_back(target);
				if (inst.opcode == Opcode::S_BRANCH) {
					continue;
				}
			}
			work.push_back(i + 1);
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

	// The world components of the ray: per component its VGPR, or (a component the same for every
	// lane) its SGPR in `scalar`.
	struct Inputs {
		std::array<uint32_t, 3> vector {};
		std::array<uint32_t, 3> scalar {NoRegister, NoRegister, NoRegister};
		// A component the program reads into an SGPR from a lane of a VGPR it keeps (V_READLANE_B32
		// <s>, <lane_vgpr>, <lane>) wherever it needs it (NoRegister: none).
		std::array<uint32_t, 3> lane_vgpr {NoRegister, NoRegister, NoRegister};
		std::array<uint32_t, 3> lane {};
	};

	// The component of `inputs` that SGPR `reg` holds at instruction `i`: the last write since `from`
	// is a V_READLANE_B32 of a lane input (-1: none).
	int LaneInput(size_t i, uint32_t reg, size_t from, const Inputs& inputs) const {
		for (size_t j = i; j-- > from;) {
			if (!WritesScalar(m_inst[j], reg)) {
				continue;
			}
			const auto& read = m_inst[j];
			if (read.opcode != Opcode::V_READLANE_B32 || read.src0.kind != OperandKind::Vgpr || !IsLaneConstant(read.src1)) {
				return -1;
			}
			for (int c = 0; c < 3; c++) {
				if (inputs.lane_vgpr[c] == read.src0.reg && inputs.lane[c] == read.src1.value) {
					return c;
				}
			}
			return -1;
		}
		return -1;
	}

	// The affine form of a value of the object space ray: per world component (the registers
	// `inputs`, unchanged since `from`), the record dword it is multiplied by, and the dword
	// added (NoRegister: none).
	struct Form {
		std::array<uint32_t, 3> factor {NoRegister, NoRegister, NoRegister};
		uint32_t                constant = NoRegister;
		// A load of the record (the first field found).
		size_t                  load     = npos;
	};

	bool FormTerm(size_t i, const Operand& a, const Operand& b, size_t from, const Inputs& inputs, uint32_t& pair,
	              Form& form) const {
		for (const auto& [scalar, vector]: {std::pair {&a, &b}, std::pair {&b, &a}}) {
			Field      field;
			const bool uniform = vector->kind == OperandKind::Sgpr && std::ranges::find(inputs.scalar, vector->reg) != inputs.scalar.end();
			const int  lane    = vector->kind == OperandKind::Sgpr && !uniform ? LaneInput(i, vector->reg, from, inputs) : -1;
			if ((vector->kind != OperandKind::Vgpr && !uniform && lane < 0) || !Plain(*vector) ||
			    !OperandField(i, *scalar, from, field)) {
				continue;
			}
			const auto& regs    = uniform ? inputs.scalar : inputs.vector;
			const auto  input   = std::ranges::find(regs, vector->reg);
			const auto  changed = lane >= 0 ? npos : uniform ? ScalarWriter(i, vector->reg, from) : VectorWriter(i, vector->reg, from);
			if ((lane < 0 && input == regs.end()) || changed != npos || (pair != NoRegister && pair != field.pair)) {
				return false;
			}
			auto& factor = form.factor[lane >= 0 ? static_cast<size_t>(lane) : static_cast<size_t>(input - regs.begin())];
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

	bool LinearForm(size_t i, uint32_t reg, size_t from, const Inputs& inputs, uint32_t& pair, Form& form,
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
		const Inputs origin {{site.origin[0], site.origin[1], site.origin[2]},
		                     {site.origin_sgpr[0], site.origin_sgpr[1], site.origin_sgpr[2]}};
		Inputs direction {{site.direction[0], site.direction[1], site.direction[2]}};
		for (uint32_t c = 0; c < 3; c++) {
			direction.lane_vgpr[c] = site.direction_lane_vgpr[c];
			direction.lane[c]      = site.direction_lane[c];
		}
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
		// One bit: S_BITCMP1_B32 <dword>, <bit>; S_CBRANCH_SCC0 (mask 1).
		if (test.opcode == Opcode::S_BITCMP1_B32 && test.src1.kind == OperandKind::IntegerInlineConstant &&
		    test.src1.value < 32u && next != npos && m_inst[next].opcode == Opcode::S_CBRANCH_SCC0 &&
		    OperandField(i, test.src0, from, field)) {
			shift = test.src1.value;
			mask  = 1u;
			return true;
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
		return MaskAnd(y, header, back, record, site);
	}

	// The wave-wide instance mask test of a setup whose ray mask is the same for every lane:
	// S_CBRANCH_SCC0 <next node> at `i` of S_AND_B32 <s>, <S_LSHR_B32 dword, shift (or the dword)>,
	// <SGPR mask or constant> (see MaskAnd), the last instruction before it that writes SCC; or,
	// right before the branch, a test of the record dword with the ray's mask in a constant
	// (S_AND_B32 <s>, <dword>, <bits << shift>, as in GTA V 01.005's 0xa31ec6a25be5e017; see
	// ConstantMaskTest).
	bool ScalarMaskTest(size_t i, size_t header, size_t back, const Field& record, RayTracingSite& site) const {
		size_t test = npos;
		for (size_t j = i; j-- > header && test == npos;) {
			test = WritesScc(m_inst[j]) ? j : npos;
		}
		if (test == npos) {
			return false;
		}
		if (m_inst[test].opcode == Opcode::S_AND_B32 && MaskAnd(test, header, back, record, site)) {
			return true;
		}
		Field    field;
		uint32_t shift = 0, bits = 0, width = 0;
		if (NextReal(test) != i || !ConstantMaskTest(test, header, field, shift, bits, width) || !SameRecord(field, record)) {
			return false;
		}
		site.cull_mask_constant = bits;
		site.layout.mask_dword  = field.dword;
		site.layout.mask_shift  = shift;
		site.layout.mask_width  = width;
		return true;
	}

	// S_AND_B32 at `y` of a record dword (S_LSHR_B32 <dword>, shift, or the dword) and of the ray's
	// mask: an SGPR the traversal keeps, a value it reads back from a VGPR lane it keeps (GTA V
	// 01.005, whose ray mask the compiler spills; see KeptLane) or a constant of 8 bits.
	bool MaskAnd(size_t y, size_t header, size_t back, const Field& record, RayTracingSite& site) const {
		auto& layout = site.layout;
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
			const auto read = b->kind == OperandKind::Sgpr && Plain(*b) ? ScalarWriter(y, b->reg, header) : npos;
			if (b->kind == OperandKind::Sgpr && Plain(*b) && Keeps(header, back, b->reg, false)) {
				site.cull_mask_sgpr = b->reg;
			} else if (read != npos && KeptLane(read, header, back) && ByteValue(y, *b, 0)) {
				// RT_INIT reads it back into a free SGPR (DerivePacketLane).
				site.cull_mask_load_pc = m_inst[read].pc;
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

	// Whether the SGPR operand `operand` of instruction `i` holds a value of 8 bits (as the host's
	// instance masks): a constant, or copies and selections (S_MOV_B32, S_MOVK_I32, S_CSELECT_B32)
	// of such values, through spills.
	bool ByteValue(size_t i, const Operand& operand, uint32_t depth) const {
		if (operand.kind == OperandKind::IntegerInlineConstant || operand.kind == OperandKind::LiteralConstant) {
			return operand.value <= 0xffu;
		}
		const auto reg = ScalarIndex(operand);
		// The writer of the value (of the SGPR spilled, for a value read back from a lane).
		const auto w = reg != NoRegister && depth < 4 ? ScalarWriter(i, reg, 0) : npos;
		if (w == npos) {
			return false;
		}
		const auto& inst = m_inst[w];
		switch (inst.opcode) {
			case Opcode::S_MOV_B32:
			case Opcode::S_MOVK_I32: return ByteValue(w, inst.src0, depth + 1);
			case Opcode::S_CSELECT_B32: return ByteValue(w, inst.src0, depth + 1) && ByteValue(w, inst.src1, depth + 1);
			default: return false;
		}
	}

	// Whether V_READLANE_B32 `read` (in the loop [header, back]) reads back, from a constant lane, a
	// value spilled there before the loop that the lane keeps through it: no instruction of the loop
	// writes that VGPR but V_WRITELANE_B32s of other constant lanes. Run again at the entry, it
	// reads the same value.
	bool KeptLane(size_t read, size_t header, size_t back) const {
		const auto& inst  = m_inst[read];
		const auto  spill = inst.opcode == Opcode::V_READLANE_B32 ? SpilledFrom(read, 0) : npos;
		if (spill == npos || spill >= header || read < header || read > back || inst.dst.kind != OperandKind::Sgpr) {
			return false;
		}
		for (size_t i = header; i <= back; i++) {
			const auto& write = m_inst[i];
			if (WritesVector(write, inst.src0.reg) &&
			    (write.opcode != Opcode::V_WRITELANE_B32 || write.dst.reg != inst.src0.reg ||
			     !IsLaneConstant(write.src1) || write.src1.value == inst.src1.value)) {
				return false;
			}
		}
		return true;
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
	// `Bits` takes (the U32 or I32 compares).
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
		const bool  ne   = inst.opcode == Opcode::S_CMP_LG_U32 || inst.opcode == Opcode::S_CMP_LG_I32;
		if (!ne && inst.opcode != Opcode::S_CMP_EQ_U32 && inst.opcode != Opcode::S_CMP_EQ_I32) {
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

	// An SGPR pair the program never reads or writes, other than `skip` (NoRegister: none).
	uint32_t UnusedPair(uint32_t skip = NoRegister) const {
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
			if (!used[pair] && !used[pair + 1] && pair != skip) {
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

	const std::vector<Instruction>&     m_inst;
	// The functions the shader calls, inlined (expanded programs).
	const std::vector<InlinedCallBody>& m_bodies;
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
	std::vector<uint32_t>    origins;
	if (!layout.Finish(program.instructions.front().pc, EndPc(program.instructions.back()), patched, origins)) {
		return fail(0, "a branch target is unknown");
	}
	// An expanded program's literals read at run time and the code of its inlined calls follow
	// their instructions (and their copies).
	if (!program.runtime_literals.empty() || !program.call_bodies.empty()) {
		std::vector<RuntimeLiteral>  literals;
		std::vector<InlinedCallBody> bodies;
		for (size_t i = 0; i < patched.size(); i++) {
			const auto origin = origins[i];
			const auto literal = std::ranges::lower_bound(program.runtime_literals, origin, {}, &RuntimeLiteral::pc);
			if (origin != UINT32_MAX && literal != program.runtime_literals.end() && literal->pc == origin) {
				literals.push_back({patched[i].pc, literal->word});
			}
			// A body's code is placed in order from its first instruction (in the guest code and
			// in each copy).
			for (const auto& body: program.call_bodies) {
				if (origin != body.begin) {
					continue;
				}
				size_t end = i;
				while (end < patched.size() && origins[end] >= body.begin && origins[end] < body.end) {
					end++;
				}
				bodies.push_back(
				    {patched[i].pc, end < patched.size() ? patched[end].pc : EndPc(patched.back()), body.address, body.aliases});
			}
		}
		std::ranges::sort(literals, {}, &RuntimeLiteral::pc);
		program.runtime_literals = std::move(literals);
		program.call_bodies      = std::move(bodies);
	}
	program.instructions       = std::move(patched);
	program.ray_tracing_source = {.user_sgpr     = patch.user_sgpr,
	                              .header_offset = patch.header_offset,
	                              .header_vsharp = patch.header_vsharp,
	                              .root          = patch.root,
	                              .layout        = patch.layout.Pack(),
	                              .valid         = true};
	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler::Decoder
