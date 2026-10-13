#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"
// windows.h defines these.
#undef near
#undef far

// Native ray tracing (see RayTracingPatch.h and NativeRayTracing). Each patched traversal (site)
// keeps, per lane: a host ray query over the TLAS the host built from the guest's BVH, a list of
// up to RayQueryListSize candidate (instance node, leaf) keys the query found and not yet
// processed, the index of the lane's smallest key and a state word. The wave processes the
// smallest key any lane has, with every lane that has it, so each leaf runs once per wave like
// in the guest's own packet traversal; keys of one instance come together, so its setup runs
// once. The host publishes its acceleration structures in a context table whose device address
// is the entry after the last page of the BDA page table (see NativeRayTracing).

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

constexpr uint32_t DoneBit       = 1u << 0u; // The query has no more candidates (or never ran).
constexpr uint32_t TerminatedBit = 1u << 1u; // The lane accepted a hit.
constexpr uint32_t EnteredBit    = 1u << 2u; // The lane entered the leaf code of this round.
constexpr uint32_t TraversalBit  = 1u << 3u; // The lane started the native traversal.
constexpr uint32_t MatchedBit    = 1u << 4u; // The lane has the key of this round.
constexpr uint32_t PreparedBit   = 1u << 5u; // The last instance setup ran for the lane.
constexpr uint32_t CommittedBit  = 1u << 6u; // The hardware found an opaque hit (the result).
constexpr uint32_t None          = 0xffffffffu;
constexpr uint32_t ListSize      = IR::RayQueryListSize;

// The context entry (rt_tlas_walk.comp ContextEntry) and slot info (rt_common.inc SlotInfo).
constexpr uint32_t ContextHeader        = 0;
constexpr uint32_t ContextTlas          = 8;
constexpr uint32_t ContextInstanceNodes = 16;
constexpr uint32_t ContextSlots         = 24;
constexpr uint32_t ContextValid         = 36;
constexpr uint32_t ContextInstanceBase  = 40;
constexpr uint32_t SlotStride           = 64;
constexpr uint32_t SlotEntryTable       = 8;
constexpr uint32_t SlotCapacity0        = 16;
constexpr uint32_t SlotCapacity1        = 20;

constexpr uint32_t CandidateIntersection = 0; // RayQueryCandidateIntersectionKHR
constexpr uint32_t CommittedIntersection = 1; // RayQueryCommittedIntersectionKHR

// gl_RayFlags: back faces (see the instance flags of the TLAS walk) do not count; with the
// guest's alpha test off, every triangle is opaque.
constexpr uint32_t RayFlagOpaque              = 0x1;
constexpr uint32_t RayFlagCullBackFacing      = 0x10;

struct LaneVariables {
	uint32_t query   = 0;
	uint32_t state   = 0;
	uint32_t nodes   = 0;
	uint32_t slots   = 0;
	uint32_t list    = 0;
	uint32_t count   = 0;
	uint32_t current = 0;
	uint32_t base    = 0;
	uint32_t result_t        = 0;
	uint32_t result_word     = 0;
	uint32_t result_instance = 0;
	uint32_t result_u        = 0;
	uint32_t result_v        = 0;
	uint32_t result_node     = 0;
};

uint32_t SiteOf(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto site = IR::RayQuerySite(inst.Flags<uint32_t>());
	if (site >= IR::MaxRayQuerySites || ctx.state.ray_query_variables[site][0] == 0) {
		ctx.Fail(inst, "ray query site without variables");
	}
	return site;
}

LaneVariables Lane(EmitterState& s, uint32_t site, uint32_t half) {
	return {s.ray_query_variables[site][half], s.ray_query_states[site][half],
	        s.ray_query_nodes[site][half],     s.ray_query_slots[site][half],
	        s.ray_query_lists[site][half],     s.ray_query_counts[site][half],
	        s.ray_query_currents[site][half],  s.ray_query_bases[site][half],
	        s.ray_query_result_t[site][half],  s.ray_query_result_word[site][half],
	        s.ray_query_result_instance[site][half], s.ray_query_result_u[site][half],
	        s.ray_query_result_v[site][half],  s.ray_query_result_node[site][half]};
}

uint32_t Load(EmitterState& s, uint32_t type, uint32_t pointer) {
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpLoad, type, result, pointer);
	return result;
}

void Store(EmitterState& s, uint32_t pointer, uint32_t value) {
	s.builder.AddFunction(spv::OpStore, pointer, value);
}

uint32_t U32(EmitterState& s, uint32_t value) {
	return ConstantU32(s, value);
}

uint32_t And(EmitterState& s, uint32_t value, uint32_t mask) {
	return Binary(s, spv::OpBitwiseAnd, TypeU32(s), value, U32(s, mask));
}

uint32_t Or(EmitterState& s, uint32_t value, uint32_t bits) {
	return Binary(s, spv::OpBitwiseOr, TypeU32(s), value, bits);
}

uint32_t BitSet(EmitterState& s, uint32_t value, uint32_t bit) {
	return Binary(s, spv::OpINotEqual, TypeBool(s), And(s, value, bit), U32(s, 0));
}

uint32_t Equal(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpIEqual, TypeBool(s), a, b);
}

uint32_t Less(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpULessThan, TypeBool(s), a, b);
}

uint32_t LogicalAnd(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpLogicalAnd, TypeBool(s), a, b);
}

uint32_t LogicalOr(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpLogicalOr, TypeBool(s), a, b);
}

uint32_t LogicalNot(EmitterState& s, uint32_t a) {
	return Unary(s, spv::OpLogicalNot, TypeBool(s), a);
}

uint32_t Add(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpIAdd, TypeU32(s), a, b);
}

// Whether key (a_instance, a_leaf) sorts before (b_instance, b_leaf).
uint32_t KeyLess(EmitterState& s, uint32_t a_instance, uint32_t a_leaf, uint32_t b_instance,
                 uint32_t b_leaf) {
	return LogicalOr(s, Less(s, a_instance, b_instance),
	                 LogicalAnd(s, Equal(s, a_instance, b_instance), Less(s, a_leaf, b_leaf)));
}

// The pointer to word `index` of a lane's key list.
uint32_t ListWord(EmitterState& s, const LaneVariables& lane, uint32_t index) {
	const auto pointer = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpAccessChain, TypePointer(s, spv::StorageClassFunction, TypeU32(s)),
	                      pointer, lane.list, index);
	return pointer;
}

// A 4-byte aligned load through a device address.
uint32_t LoadAddressU32(EmitterState& s, uint32_t address) {
	const auto pointer = Unary(s, spv::OpConvertUToPtr, TypePhysicalU32Pointer(s), address);
	const auto result  = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpLoad, TypeU32(s), result, pointer, spv::MemoryAccessAlignedMask, 4u);
	return result;
}

uint32_t LoadAddressU64(EmitterState& s, uint32_t address) {
	const auto low  = LoadAddressU32(s, address);
	const auto high = LoadAddressU32(s, Binary(s, spv::OpIAdd, TypeU64(s), address, ConstantU64(s, 4)));
	return PackU64(s, low, high);
}

uint32_t AddressPlus(EmitterState& s, uint32_t address, uint32_t offset) {
	return Binary(s, spv::OpIAdd, TypeU64(s), address, ConstantU64(s, offset));
}

uint32_t ScaledAddress(EmitterState& s, uint32_t base, uint32_t index, uint32_t scale) {
	const auto wide = TypeU64(s);
	return Binary(s, spv::OpIAdd, wide, base,
	              Binary(s, spv::OpIMul, wide, Unary(s, spv::OpUConvert, wide, index),
	                     ConstantU64(s, scale)));
}

uint32_t Finite(EmitterState& s, uint32_t value) {
	const auto b = TypeBool(s);
	return LogicalAnd(s, LogicalNot(s, Unary(s, spv::OpIsNan, b, value)),
	                  LogicalNot(s, Unary(s, spv::OpIsInf, b, value)));
}

// A structured loop: `body(merge, cont)` emits the body from the current label and ends with a
// branch to one of them (the continue target goes back to the header).
template <typename Body>
void EmitLoop(EmitterState& s, Body&& body) {
	const auto header = s.builder.AllocateId(), start = s.builder.AllocateId();
	const auto cont = s.builder.AllocateId(), merge = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(s, header);
	s.builder.AddFunction(spv::OpLoopMerge, merge, cont, spv::LoopControlMaskNone);
	s.builder.AddFunction(spv::OpBranch, start);
	EmitLabel(s, start);
	body(merge, cont);
	EmitLabel(s, cont);
	s.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(s, merge);
}

// A lane's keys: list[current .. count) holds the unprocessed ones, ascending (an array in
// function memory indexed at run time).
uint32_t LoadWord(EmitterState& s, const LaneVariables& lane, uint32_t index) {
	return Load(s, TypeU32(s), ListWord(s, lane, index));
}

// Inserts a key into the sorted list (whose current index is 0 while it fills).
void InsertKey(EmitterState& s, const LaneVariables& lane, uint32_t count, uint32_t instance,
               uint32_t leaf) {
	const auto u = TypeU32(s), b = TypeBool(s);
	// The position moves down while the key before it is larger.
	const auto position = ListWord(s, lane, U32(s, 2 * ListSize));
	Store(s, position, count);
	EmitLoop(s, [&](uint32_t merge, uint32_t cont) {
		const auto i     = Load(s, u, position);
		const auto above = Binary(s, spv::OpUGreaterThan, b, i, U32(s, 0));
		const auto move  = EmitValueOrDefaultIfCondition(s, above, b, ConstantBool(s, false), [&] {
			const auto at = Binary(s, spv::OpShiftLeftLogical, u, Binary(s, spv::OpISub, u, i, U32(s, 1)),
			                       U32(s, 1));
			const auto previous_instance = LoadWord(s, lane, at);
			const auto previous_leaf     = LoadWord(s, lane, Add(s, at, U32(s, 1)));
			const auto larger = KeyLess(s, instance, leaf, previous_instance, previous_leaf);
			EmitIfCondition(s, larger, [&] {
				const auto to = Binary(s, spv::OpShiftLeftLogical, u, i, U32(s, 1));
				Store(s, ListWord(s, lane, to), previous_instance);
				Store(s, ListWord(s, lane, Add(s, to, U32(s, 1))), previous_leaf);
				Store(s, position, Binary(s, spv::OpISub, u, i, U32(s, 1)));
			});
			return larger;
		});
		s.builder.AddFunction(spv::OpBranchConditional, move, cont, merge);
	});
	const auto at = Binary(s, spv::OpShiftLeftLogical, u, Load(s, u, position), U32(s, 1));
	Store(s, ListWord(s, lane, at), instance);
	Store(s, ListWord(s, lane, Add(s, at, U32(s, 1))), leaf);
}

// Removes the smallest key of a lane's list.
void PopKey(EmitterState& s, const LaneVariables& lane) {
	Store(s, lane.current, Add(s, Load(s, TypeU32(s), lane.current), U32(s, 1)));
}

void ClearKeys(EmitterState& s, const LaneVariables& lane) {
	Store(s, lane.count, U32(s, 0));
	Store(s, lane.current, U32(s, 0));
}

// The guest list entry of a hit or candidate triangle: (leaf pointer of the entry, primitive
// word, guest triangle node, barycentric remap of the node), from the entry table of its host
// BLAS (rt_blas_walk.comp). Also whether the entry is one of the table (a slot rebuilt meanwhile
// may have another table).
struct GuestEntry {
	uint32_t entry = 0;
	uint32_t word  = 0;
	uint32_t node  = 0;
	uint32_t remap = 0;
	uint32_t valid = 0;
};

GuestEntry TriangleEntry(EmitterState& s, const LaneVariables& lane, uint32_t intersection) {
	const auto u = TypeU32(s), wide = TypeU64(s);
	const auto query = [&](spv::Op op) {
		const auto result = s.builder.AllocateId();
		s.builder.AddFunction(op, u, result, lane.query, U32(s, intersection));
		return result;
	};
	const auto slot      = query(spv::OpRayQueryGetIntersectionInstanceCustomIndexKHR);
	const auto geometry  = query(spv::OpRayQueryGetIntersectionGeometryIndexKHR);
	const auto primitive = query(spv::OpRayQueryGetIntersectionPrimitiveIndexKHR);
	const auto info      = ScaledAddress(s, Load(s, wide, lane.slots), slot, SlotStride);
	const auto table     = LoadAddressU64(s, AddressPlus(s, info, SlotEntryTable));
	const auto capacity0 = LoadAddressU32(s, AddressPlus(s, info, SlotCapacity0));
	const auto capacity1 = LoadAddressU32(s, AddressPlus(s, info, SlotCapacity1));
	const auto opaque    = Equal(s, geometry, U32(s, 0));
	const auto index     = Select(s, u, opaque, primitive, Add(s, capacity0, primitive));
	const auto valid     = LogicalAnd(
	    s, Binary(s, spv::OpINotEqual, TypeBool(s), table, ConstantU64(s, 0)),
	    Less(s, primitive, Select(s, u, opaque, capacity0, capacity1)));
	const auto entry = ScaledAddress(s, table, Select(s, u, valid, index, U32(s, 0)), 16);
	const auto field = [&](uint32_t offset) {
		return EmitValueOrZeroIfCondition(
		    s, valid, [&] { return LoadAddressU32(s, AddressPlus(s, entry, offset)); });
	};
	return {field(0), field(4), field(8), field(12), valid};
}

// The guest TLAS leaf pointer of the instance of a hit or candidate.
uint32_t InstanceNode(EmitterState& s, const LaneVariables& lane, uint32_t intersection) {
	const auto instance = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpRayQueryGetIntersectionInstanceIdKHR, TypeU32(s), instance, lane.query,
	                      U32(s, intersection));
	return LoadAddressU32(s, ScaledAddress(s, Load(s, TypeU64(s), lane.nodes), instance, 4));
}

// Collects the alpha tested triangles a lane's ray hits (the candidates the hardware reports),
// for a lane whose list is empty, until the list is full or the query is done. A query is done
// at its closest opaque hit, kept as the lane's result.
void Refill(EmitterState& s, const LaneVariables& lane) {
	const auto u = TypeU32(s), b = TypeBool(s), wide = TypeU64(s), f = TypeF32(s);
	const auto state  = Load(s, u, lane.state);
	const auto empty  = Equal(s, Load(s, u, lane.current), Load(s, u, lane.count));
	const auto refill = LogicalAnd(s, empty, LogicalAnd(s, BitSet(s, state, TraversalBit),
	                                                    LogicalNot(s, BitSet(s, state, DoneBit | TerminatedBit))));
	EmitIfCondition(s, refill, [&] {
		ClearKeys(s, lane);
		EmitLoop(s, [&](uint32_t merge, uint32_t cont) {
			const auto count = Load(s, u, lane.count);
			const auto room  = Less(s, count, U32(s, ListSize));
			const auto proceed = EmitValueOrDefaultIfCondition(s, room, b, ConstantBool(s, false), [&] {
				const auto result = s.builder.AllocateId();
				s.builder.AddFunction(spv::OpRayQueryProceedKHR, b, result, lane.query);
				return result;
			});
			// The query is done: an opaque hit is the lane's result (its alpha tested candidates
			// do not matter any more).
			EmitIfCondition(s, LogicalAnd(s, room, LogicalNot(s, proceed)), [&] {
				const auto committed = s.builder.AllocateId();
				s.builder.AddFunction(spv::OpRayQueryGetIntersectionTypeKHR, u, committed, lane.query,
				                      U32(s, CommittedIntersection));
				const auto hit = Equal(
				    s, committed,
				    U32(s, spv::RayQueryCommittedIntersectionTypeRayQueryCommittedIntersectionTriangleKHR));
				EmitIfCondition(s, hit, [&] {
					const auto t = s.builder.AllocateId();
					s.builder.AddFunction(spv::OpRayQueryGetIntersectionTKHR, f, t, lane.query,
					                      U32(s, CommittedIntersection));
					const auto hit_entry = TriangleEntry(s, lane, CommittedIntersection);
					const auto node = InstanceNode(s, lane, CommittedIntersection);
					Store(s, lane.result_t, Unary(s, spv::OpBitcast, u, t));
					Store(s, lane.result_word, hit_entry.word);
					Store(s, lane.result_node, hit_entry.node);
					// The guest triangle test reports the weights of the node vertices its remap
					// bits select (two bits each); the host triangle has the node's vertices in order.
					const auto bary = s.builder.AllocateId();
					s.builder.AddFunction(spv::OpRayQueryGetIntersectionBarycentricsKHR,
					                      TypeF32Vector(s, 2), bary, lane.query,
					                      U32(s, CommittedIntersection));
					std::array<uint32_t, 3> weights {};
					for (uint32_t i = 0; i < 2; i++) {
						weights[i + 1] = s.builder.AllocateId();
						s.builder.AddFunction(spv::OpCompositeExtract, f, weights[i + 1], bary, i);
					}
					weights[0] = Binary(s, spv::OpFSub, f,
					                    Binary(s, spv::OpFSub, f, ConstantF32Value(s, 1.0f), weights[1]),
					                    weights[2]);
					const auto weight = [&](uint32_t shift) {
						const auto index = And(s, Binary(s, spv::OpShiftRightLogical, u, hit_entry.remap,
						                                 U32(s, shift)),
						                       3u);
						return Unary(s, spv::OpBitcast, u,
						             Select(s, f, Equal(s, index, U32(s, 1)), weights[1],
						                    Select(s, f, Equal(s, index, U32(s, 2)), weights[2], weights[0])));
					};
					Store(s, lane.result_u, weight(0));
					Store(s, lane.result_v, weight(2));
					Store(s, lane.result_instance,
					      Binary(s, spv::OpIAdd, wide, Load(s, wide, lane.base),
					             Binary(s, spv::OpShiftLeftLogical, wide,
					                    Unary(s, spv::OpUConvert, wide,
					                          Binary(s, spv::OpShiftRightLogical, u, node, U32(s, 3))),
					                    ConstantU64(s, 7))));
					ClearKeys(s, lane);
					Store(s, lane.state, Or(s, Load(s, u, lane.state), U32(s, CommittedBit)));
				});
				Store(s, lane.state, Or(s, Load(s, u, lane.state), U32(s, DoneBit)));
			});
			// A candidate: an alpha tested triangle the ray hits. The wave runs the guest code of
			// its list entry (triangle and alpha test) later.
			EmitIfCondition(s, proceed, [&] {
				const auto type = s.builder.AllocateId();
				s.builder.AddFunction(spv::OpRayQueryGetIntersectionTypeKHR, u, type, lane.query,
				                      U32(s, CandidateIntersection));
				const auto triangle = Equal(
				    s, type,
				    U32(s, spv::RayQueryCandidateIntersectionTypeRayQueryCandidateIntersectionTriangleKHR));
				EmitIfCondition(s, triangle, [&] {
					const auto candidate = TriangleEntry(s, lane, CandidateIntersection);
					// An entry outside its table is not tested (the candidate is a miss).
					EmitIfCondition(s, candidate.valid, [&] {
						InsertKey(s, lane, count, InstanceNode(s, lane, CandidateIntersection),
						          candidate.entry);
						Store(s, lane.count, Add(s, count, U32(s, 1)));
					});
				});
			});
			s.builder.AddFunction(spv::OpBranchConditional, proceed, cont, merge);
		});
	});
}

// The smallest unprocessed key of a lane's list (None, None when there is none).
std::pair<uint32_t, uint32_t> LaneMinimum(EmitterState& s, const LaneVariables& lane) {
	const auto u       = TypeU32(s), b = TypeBool(s);
	const auto current = Load(s, u, lane.current);
	const auto has     = Less(s, current, Load(s, u, lane.count));
	const auto at      = Binary(s, spv::OpShiftLeftLogical, u, current, U32(s, 1));
	const auto instance = EmitValueOrDefaultIfCondition(s, has, u, U32(s, None),
	                                                    [&] { return LoadWord(s, lane, at); });
	const auto leaf = EmitValueOrDefaultIfCondition(s, has, u, U32(s, None),
	                                                [&] { return LoadWord(s, lane, Add(s, at, U32(s, 1))); });
	(void)b;
	return {instance, leaf};
}

uint32_t GroupMin(EmitterState& s, uint32_t value) {
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpGroupNonUniformUMin, TypeU32(s), result, U32(s, spv::ScopeSubgroup),
	                      spv::GroupOperationReduce, value);
	return result;
}

} // namespace

void DeclareRayQueryVariables(EmitterState& s) {
	if (!s.program.info.ray_query) {
		return;
	}
	const auto u          = TypeU32(s);
	const auto query_type = s.builder.Type(spv::OpTypeRayQueryKHR);
	const auto list_type = s.builder.Type(spv::OpTypeArray, u, ConstantU32(s, 2 * ListSize + 1));
	const auto declare   = [&](uint32_t pointee, uint32_t& id) {
		id = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpVariable, TypePointer(s, spv::StorageClassFunction, pointee), id,
		                      spv::StorageClassFunction);
	};
	for (uint32_t site = 0; site < IR::MaxRayQuerySites; site++) {
		if ((s.program.info.ray_query_sites & (1u << site)) == 0) {
			continue;
		}
		for (uint32_t half = 0; half < s.lane_count; half++) {
			declare(query_type, s.ray_query_variables[site][half]);
			declare(u, s.ray_query_states[site][half]);
			declare(TypeU64(s), s.ray_query_nodes[site][half]);
			declare(TypeU64(s), s.ray_query_slots[site][half]);
			declare(list_type, s.ray_query_lists[site][half]);
			declare(u, s.ray_query_counts[site][half]);
			declare(u, s.ray_query_currents[site][half]);
			declare(TypeU64(s), s.ray_query_bases[site][half]);
			declare(u, s.ray_query_result_t[site][half]);
			declare(u, s.ray_query_result_word[site][half]);
			declare(TypeU64(s), s.ray_query_result_instance[site][half]);
			declare(u, s.ray_query_result_u[site][half]);
			declare(u, s.ray_query_result_v[site][half]);
			declare(u, s.ray_query_result_node[site][half]);
		}
		declare(u, s.ray_query_selected_instance[site]);
		declare(u, s.ray_query_selected_leaf[site]);
		declare(u, s.ray_query_setup_instance[site]);
		declare(u, s.ray_query_reuse[site]);
		declare(u, s.ray_query_rounds[site]);
	}
}

void InitializeRayQueryVariables(EmitterState& s) {
	for (uint32_t site = 0; site < IR::MaxRayQuerySites; site++) {
		for (uint32_t half = 0; half < s.lane_count; half++) {
			if (s.ray_query_states[site][half] != 0) {
				Store(s, s.ray_query_states[site][half], U32(s, DoneBit));
				ClearKeys(s, Lane(s, site, half));
			}
		}
		if (s.ray_query_setup_instance[site] != 0) {
			Store(s, s.ray_query_setup_instance[site], U32(s, None));
			Store(s, s.ray_query_reuse[site], U32(s, 0));
			Store(s, s.ray_query_selected_instance[site], U32(s, None));
			Store(s, s.ray_query_selected_leaf[site], U32(s, None));
		}
	}
}

uint32_t EmitRayQueryInit(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = Lane(s, site, ctx.half);
	const auto u = TypeU32(s), f = TypeF32(s), b = TypeBool(s), wide = TypeU64(s);
	std::array<uint32_t, 8> ray;
	for (uint32_t i = 0; i < 8; i++) {
		ray[i] = i < 7 ? Unary(s, spv::OpBitcast, f, ctx.Arg(inst, i)) : ctx.Arg(inst, i);
	}
	const auto header    = PackU64(s, ctx.Arg(inst, 8), ctx.Arg(inst, 9));
	const auto alpha_off = Binary(s, spv::OpINotEqual, b, ctx.Arg(inst, 10), U32(s, 0));
	const auto exec      = ctx.Arg(inst, 11);
	// The context the host published (the page table entry after the last page).
	const auto entry = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(s, 64), entry,
	                      s.bda_pagetable_variable, U32(s, 0),
	                      U32(s, static_cast<uint32_t>(BufferCache::NATIVE_RT_CONTEXT_PAGE)));
	const auto context     = Load(s, wide, entry);
	const auto has_context = Binary(s, spv::OpINotEqual, b, context, ConstantU64(s, 0));
	const auto native = EmitValueOrDefaultIfCondition(s, has_context, b, ConstantBool(s, false), [&] {
		const auto valid = Binary(s, spv::OpINotEqual, b,
		                          LoadAddressU32(s, AddressPlus(s, context, ContextValid)), U32(s, 0));
		const auto same = Equal(s, LoadAddressU64(s, AddressPlus(s, context, ContextHeader)), header);
		return LogicalAnd(s, valid, same);
	});
	// Lanes whose ray the hardware cannot take (NaN, infinite or negative extents) find nothing,
	// as the guest's box tests would not.
	auto lane_ok = LogicalAnd(s, native, exec);
	for (uint32_t i = 0; i < 6; i++) {
		lane_ok = LogicalAnd(s, lane_ok, Finite(s, ray[i]));
	}
	lane_ok = LogicalAnd(s, lane_ok,
	                     Binary(s, spv::OpFOrdGreaterThanEqual, b, ray[6], ConstantF32Value(s, 0.0f)));
	const auto traversal = Select(s, u, LogicalAnd(s, native, exec), U32(s, TraversalBit), U32(s, 0));
	Store(s, lane.state, Or(s, traversal, Select(s, u, lane_ok, U32(s, 0), U32(s, DoneBit))));
	ClearKeys(s, lane);
	Store(s, s.ray_query_setup_instance[site], U32(s, None));
	if (ctx.half == 0) {
		Store(s, s.ray_query_rounds[site], U32(s, 0));
	}
	if (ctx.half == 0) {
	}
	EmitIfCondition(s, lane_ok, [&] {
		const auto tlas = LoadAddressU64(s, AddressPlus(s, context, ContextTlas));
		Store(s, lane.nodes, LoadAddressU64(s, AddressPlus(s, context, ContextInstanceNodes)));
		Store(s, lane.slots, LoadAddressU64(s, AddressPlus(s, context, ContextSlots)));
		Store(s, lane.base, LoadAddressU64(s, AddressPlus(s, context, ContextInstanceBase)));
		const auto structure = Unary(s, spv::OpConvertUToAccelerationStructureKHR,
		                             s.builder.Type(spv::OpTypeAccelerationStructureKHR), tlas);
		const auto origin = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(s, 3), origin, ray[0], ray[1],
		                      ray[2]);
		const auto direction = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(s, 3), direction, ray[3],
		                      ray[4], ray[5]);
		// The closest opaque hit, not the first one found: the game's traversal visits boxes front
		// to back, and its shadow denoiser keeps no history where the hit instance is dynamic
		// (instance record +60 bit 31). An arbitrary first hit (often static geometry farther
		// along the ray) left moving vehicles without that flag: trails of their shadows.
		// Closest hit sites (RayQueryInitBothFacings) count both facings.
		const uint32_t facing =
		    (IR::RayQueryVariant(inst.Flags<uint32_t>()) & IR::RayQueryInitBothFacings) != 0
		        ? 0u
		        : RayFlagCullBackFacing;
		const auto flags = Select(s, u, alpha_off, U32(s, facing | RayFlagOpaque), U32(s, facing));
		s.builder.AddFunction(spv::OpRayQueryInitializeKHR, lane.query, structure, flags,
		                      And(s, ray[7], 0xffu), origin, ConstantF32Value(s, 0.0f), direction,
		                      ray[6]);
	});
	return native;
}

// Wave operation (emitted once for both emulated halves): refills empty lane lists and selects
// the smallest key of the wave. Result: instance node, leaf, 1 when there is one, 1 when its
// instance is the one whose setup ran last.
uint32_t EmitRayQueryNext(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto u = TypeU32(s), b = TypeBool(s);
	uint32_t   local_instance = 0, local_leaf = 0;
	std::array<std::pair<uint32_t, uint32_t>, 2> heads {};
	for (uint32_t half = 0; half < s.lane_count; half++) {
		const auto lane = Lane(s, site, half);
		Refill(s, lane);
		const auto [instance, leaf] = LaneMinimum(s, lane);
		heads[half] = {instance, leaf};
		if (half == 0) {
			local_instance = instance;
			local_leaf     = leaf;
		} else {
			const auto smaller = KeyLess(s, instance, leaf, local_instance, local_leaf);
			local_instance     = Select(s, u, smaller, instance, local_instance);
			local_leaf         = Select(s, u, smaller, leaf, local_leaf);
		}
	}
	// A wave that keeps selecting keys (a lane whose key the leaf code never retires) stops
	// after MaxRounds instead of hanging the GPU: its remaining candidates are dropped.
	constexpr uint32_t MaxRounds = 256;
	const auto rounds  = Add(s, Load(s, u, s.ray_query_rounds[site]), U32(s, 1));
	Store(s, s.ray_query_rounds[site], rounds);
	const auto running = Less(s, rounds, U32(s, MaxRounds));
	const auto instance =
	    Select(s, u, running, GroupMin(s, local_instance), U32(s, None));
	const auto leaf =
	    GroupMin(s, Select(s, u, Equal(s, local_instance, instance), local_leaf, U32(s, None)));
	const auto has   = Binary(s, spv::OpINotEqual, b, instance, U32(s, None));
	// The last setup can serve this key only if it ran for every lane that has the key (its
	// object space ray is in the lanes the setup was active in).
	uint32_t unprepared = U32(s, 0);
	for (uint32_t half = 0; half < s.lane_count; half++) {
		const auto lane  = Lane(s, site, half);
		const auto mine  = LogicalAnd(s, Equal(s, heads[half].first, instance), Equal(s, heads[half].second, leaf));
		const auto ready = BitSet(s, Load(s, u, lane.state), PreparedBit);
		unprepared = Or(s, unprepared, Select(s, u, LogicalAnd(s, mine, LogicalNot(s, ready)), U32(s, 1), U32(s, 0)));
	}
	const auto max_unprepared = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpGroupNonUniformUMax, u, max_unprepared, U32(s, spv::ScopeSubgroup),
	                      spv::GroupOperationReduce, unprepared);
	const auto setup = Load(s, u, s.ray_query_setup_instance[site]);
	const auto reuse = LogicalAnd(s, LogicalAnd(s, has, Equal(s, instance, setup)),
	                              Equal(s, max_unprepared, U32(s, 0)));
	Store(s, s.ray_query_reuse[site], Select(s, u, reuse, U32(s, 1), U32(s, 0)));
	Store(s, s.ray_query_selected_instance[site], instance);
	Store(s, s.ray_query_selected_leaf[site], leaf);
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(s, 4), result, instance, leaf,
	                      Select(s, u, has, U32(s, 1), U32(s, 0)),
	                      Select(s, u, reuse, U32(s, 1), U32(s, 0)));
	return result;
}

// Whether the lane has the selected key (its smallest one); marks the lane for this round.
uint32_t EmitRayQueryMatch(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = Lane(s, site, ctx.half);
	const auto u    = TypeU32(s);
	const auto [instance, leaf] = LaneMinimum(s, lane);
	const auto matched = LogicalAnd(
	    s, LogicalAnd(s, Equal(s, instance, Load(s, u, s.ray_query_selected_instance[site])),
	                  Equal(s, leaf, Load(s, u, s.ray_query_selected_leaf[site]))),
	    Binary(s, spv::OpINotEqual, TypeBool(s), instance, U32(s, None)));
	const auto state = And(s, Load(s, u, lane.state), ~(MatchedBit | EnteredBit));
	Store(s, lane.state, Or(s, state, Select(s, u, matched, U32(s, MatchedBit), U32(s, 0))));
	return matched;
}

// The leaf of the key the wave processes (uniform).
uint32_t EmitRayQuerySelectedLeaf(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	return Load(s, TypeU32(s), s.ray_query_selected_leaf[site]);
}

// The lane's opaque hit, unless it has none or its guest code accepted another one.
uint32_t EmitRayQueryResult(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s     = ctx.state;
	const auto site  = SiteOf(ctx, inst);
	const auto lane  = Lane(s, site, ctx.half);
	const auto u     = TypeU32(s), wide = TypeU64(s);
	const auto state = Load(s, u, lane.state);
	const auto has   = LogicalAnd(s, BitSet(s, state, CommittedBit), LogicalNot(s, BitSet(s, state, TerminatedBit)));
	const auto instance = Load(s, wide, lane.result_instance);
	switch (static_cast<IR::RayQueryResultKind>(IR::RayQueryVariant(inst.Flags<uint32_t>()))) {
		case IR::RayQueryResultKind::Has: return Select(s, u, has, U32(s, 1), U32(s, 0));
		case IR::RayQueryResultKind::T: return Load(s, u, lane.result_t);
		case IR::RayQueryResultKind::Word: return Load(s, u, lane.result_word);
		case IR::RayQueryResultKind::InstanceLo: return Unary(s, spv::OpUConvert, u, instance);
		case IR::RayQueryResultKind::InstanceHi:
			return Unary(s, spv::OpUConvert, u, Binary(s, spv::OpShiftRightLogical, wide, instance, ConstantU64(s, 32)));
		case IR::RayQueryResultKind::U: return Load(s, u, lane.result_u);
		case IR::RayQueryResultKind::V: return Load(s, u, lane.result_v);
		case IR::RayQueryResultKind::Node: return Load(s, u, lane.result_node);
	}
	ctx.Fail(inst, "unknown ray query result");
}

uint32_t EmitRayQueryFlag(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s     = ctx.state;
	const auto site  = SiteOf(ctx, inst);
	const auto lane  = Lane(s, site, ctx.half);
	const auto u     = TypeU32(s);
	const auto setup = Load(s, u, s.ray_query_setup_instance[site]);
	switch (static_cast<IR::RayQueryFlagKind>(IR::RayQueryVariant(inst.Flags<uint32_t>()))) {
		case IR::RayQueryFlagKind::InTraversal: return BitSet(s, Load(s, u, lane.state), TraversalBit);
		case IR::RayQueryFlagKind::Reuse:
			return Binary(s, spv::OpINotEqual, TypeBool(s), Load(s, u, s.ray_query_reuse[site]), U32(s, 0));
		case IR::RayQueryFlagKind::InstanceMatch: {
			const auto head = LaneMinimum(s, lane).first;
			return LogicalAnd(s, Equal(s, head, Load(s, u, s.ray_query_selected_instance[site])),
			                  Binary(s, spv::OpINotEqual, TypeBool(s), head, U32(s, None)));
		}
		case IR::RayQueryFlagKind::Matched: return BitSet(s, Load(s, u, lane.state), MatchedBit);
		case IR::RayQueryFlagKind::NeedRestore:
			return Binary(s, spv::OpINotEqual, TypeBool(s), setup, U32(s, None));
	}
	ctx.Fail(inst, "unknown ray query flag");
}

void EmitRayQueryMark(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s       = ctx.state;
	const auto site    = SiteOf(ctx, inst);
	const auto lane    = Lane(s, site, ctx.half);
	const auto u       = TypeU32(s);
	const auto value   = ctx.Arg(inst, 0);
	const auto state   = Load(s, u, lane.state);
	const auto matched = BitSet(s, state, MatchedBit);
	switch (static_cast<IR::RayQueryMarkKind>(IR::RayQueryVariant(inst.Flags<uint32_t>()))) {
		case IR::RayQueryMarkKind::Enter:
			Store(s, lane.state,
			      Or(s, And(s, state, ~EnteredBit),
			         Select(s, u, LogicalAnd(s, matched, value), U32(s, EnteredBit), U32(s, 0))));
			return;
		case IR::RayQueryMarkKind::Resolve: {
			// Matched lanes that entered the leaf and left the searching lanes accepted a hit.
			const auto stop =
			    LogicalAnd(s, LogicalAnd(s, matched, BitSet(s, state, EnteredBit)), LogicalNot(s, value));
			EmitIfCondition(s, stop, [&] {
				s.builder.AddFunction(spv::OpRayQueryTerminateKHR, lane.query);
				ClearKeys(s, lane);
			});
			EmitIfCondition(s, LogicalAnd(s, matched, LogicalNot(s, stop)), [&] { PopKey(s, lane); });
			const auto cleared = And(s, state, ~(MatchedBit | EnteredBit));
			Store(s, lane.state, Or(s, cleared, Select(s, u, stop, U32(s, TerminatedBit), U32(s, 0))));
			return;
		}
		case IR::RayQueryMarkKind::CullResolve:
			EmitIfCondition(s, matched, [&] { PopKey(s, lane); });
			Store(s, lane.state, And(s, state, ~(MatchedBit | EnteredBit | PreparedBit)));
			Store(s, s.ray_query_setup_instance[site], U32(s, None));
			return;
		case IR::RayQueryMarkKind::SetupBegin:
			if (ctx.half == 0) {
			}
			Store(s, s.ray_query_setup_instance[site], Load(s, u, s.ray_query_selected_instance[site]));
			Store(s, lane.state, Or(s, And(s, state, ~PreparedBit),
			                        Select(s, u, value, U32(s, PreparedBit), U32(s, 0))));
			return;
		case IR::RayQueryMarkKind::Exit:
			Store(s, lane.state, U32(s, DoneBit));
			ClearKeys(s, lane);
			Store(s, s.ray_query_setup_instance[site], U32(s, None));
			return;
	}
	ctx.Fail(inst, "unknown ray query mark");
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
