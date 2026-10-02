#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	std::span<const uint32_t> workgroup_counts;
	uint64_t                  indirect_args_address = 0;
	// When set, holds the flat SRT slot whose scalar load is reading memory, or NoSrtSlot.
	uint32_t*                 read_slot                  = nullptr;
};

class SrtReadCapture {
public:
	// slots (optional) receives, for each read, the slot the walkers set (SrtRuntime::read_slot).
	SrtReadCapture(SrtRuntime source, std::vector<std::pair<uint64_t, uint64_t>>& ranges,
	               std::vector<uint32_t>* slots = nullptr):
	    m_source(source), m_ranges(ranges), m_slots(slots) {}
	SrtRuntime ObservedRuntime();

private:
	static bool ReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values);
	static bool ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values);
	void        Record(uint64_t address, uint64_t size);

	SrtRuntime                                  m_source;
	std::vector<std::pair<uint64_t, uint64_t>>& m_ranges;
	std::vector<uint32_t>*                      m_slots = nullptr;
	// Set by the walkers around the load of a flattened scalar read.
	uint32_t                                    m_slot = NoSrtSlot;
};

// Retained reads no longer need the guest instruction PC. A clean read evaluates
// its address and bounds through the strict reader as well as the final DWORD.
struct SrtReadFlags {
	uint32_t index = 0;
	uint32_t clean = 0;
};

enum class RuntimeValueType { Any, Integer, ImmutableInteger };

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Uniform across a compute wave, including values derived from its workgroup ID.
// This does not imply that the host can evaluate the value.
bool ValidateWaveUniformValue(const ResourcePlan& program, Value value);
void CompileSrtWalker(ResourcePlan& program);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);

// Nonnegative affine coefficients: element zero is constant, remaining elements are
// caller-selected leaves. Runtime range proofs must additionally exclude U32 wrap.
template <size_t Size>
std::optional<std::array<uint64_t, Size>> AffineIndex(
    Value value, const auto& resolve, const auto& leaf, uint32_t depth = 0) {
	value = resolve(value);
	if (depth > 32u || value.GetType() != Type::U32) return {};
	std::array<uint64_t, Size> result {};
	if (value.IsImmediate()) {
		result[0] = value.U32();
		return result;
	}
	if (const auto index = leaf(value)) {
		result[*index] = 1u;
		return result;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return {};
	const auto op = inst->GetOpcode();
	if (op != ValueOpcode::IAdd32 && op != ValueOpcode::IMul32 &&
	    op != ValueOpcode::ShiftLeftLogical32) return {};
	auto left = AffineIndex<Size>(inst->Arg(0), resolve, leaf, depth + 1u);
	auto right = AffineIndex<Size>(inst->Arg(1), resolve, leaf, depth + 1u);
	if (!left || !right) return {};
	const auto constant = [](const auto& coefficients) {
		return std::ranges::all_of(std::span(coefficients).subspan(1u),
		                          [](uint64_t coefficient) { return coefficient == 0u; });
	};
	if (op == ValueOpcode::IMul32 && !constant(*right)) std::swap(left, right);
	if (op != ValueOpcode::IAdd32 && !constant(*right)) return {};
	if (op == ValueOpcode::ShiftLeftLogical32) {
		if ((*right)[0] >= 32u) return {};
		(*right)[0] = uint64_t {1} << (*right)[0];
	}
	for (size_t index = 0; index < Size; ++index) {
		(*left)[index] = op == ValueOpcode::IAdd32
		    ? (*left)[index] + (*right)[index] : (*left)[index] * (*right)[0];
		if ((*left)[index] > UINT32_MAX) return {};
	}
	return left;
}

struct CompiledSrtPlan {
	static constexpr uint32_t NoNode = UINT32_MAX;

	// User data is a side-effect-free leaf (imm = dword index) that needs no memo entry.
	enum class NodeKind : uint8_t { Constant, Invalid, UserData, Instruction };
	// Per-opcode operand shapes that the walker resolves at build time.
	enum class Shape : uint8_t {
		Generic,
		Phi,           // args[0] = invariant value or NoNode
		ReadFirstLane, // args[0] = value, args[1] = EXEC mask
		ExtractU64,    // args[0] = packed value, aux = component, aux2 = valid index
		ExtractU32x2,  // args[0..1] = source operands, aux = component, aux2 = source kind
		RawRead,       // args[0..3] = handle operands, args[4] = dynamic offset
		// A LoadAddressU32 whose address operands are user data or constants:
		// args[0..1] = address, imm = byte offset relative to the aligned address.
		LeafAddressRead,
		NotRawRead,    // LoadAddressU32/ReadConstBuffer that is not a scalar read
		// A Phi decided by its reachable predecessors: imm = index into control_phis.
		ControlPhi,
	};
	struct Node {
		ValueOpcode op    = ValueOpcode::Void;
		NodeKind    kind  = NodeKind::Invalid;
		Shape       shape = Shape::Generic;
		uint8_t     aux   = 0;
		uint8_t     aux2  = 0;
		// A select whose predicate may equal an EXEC mask.
		bool        select = false;
		// ReadConstBuffer handle operand count.
		uint8_t     handle_args = 0;
		// A scalar read that the clean evaluator reads (SrtReadFlags::clean).
		bool        clean = false;
		uint32_t    args[5] {};
		// The memory-info index of a raw read.
		uint32_t    memory = UINT32_MAX;
		uint64_t    imm = 0;
	};
	struct Context {
		struct Entry {
			uint64_t value      = 0;
			uint64_t generation = 0;
		};
		std::vector<Entry> values;
		uint64_t           generation = 0;
		// Nested EXEC-mask walkers of one refresh share the memo of their mask and read policy,
		// as SrtWalker's evaluation contexts do.
		uint64_t    session         = 0;
		uint32_t    mask            = UINT32_MAX;
		const void* read            = nullptr;
		const void* strict          = nullptr;
		bool        clean_evaluator = false;
	};

	struct SrtRead {
		uint32_t node        = NoNode;
		uint32_t flat_offset = 0;
		bool     clean       = false;
	};
	struct Descriptor {
		std::array<uint32_t, 8> dwords {};
		uint32_t                dword_count = 0;
	};

	// Copies of the plan state that every refresh reads, kept next to the nodes.
	std::vector<Node>       nodes;
	std::vector<SrtRead>    srt_reads;
	// The flat SRT slot whose scalar load each node is, or NoSrtSlot.
	std::vector<uint32_t>   read_slots;
	std::vector<Descriptor> descriptors;
	std::vector<uint32_t>   conditions;
	// The (incoming node, predecessor block) pairs of each control Phi.
	std::vector<std::vector<std::pair<uint32_t, uint32_t>>> control_phis;
	bool                    srt_plan_complete = false;
	// Other values that materialization evaluates by Value (fill words, indirect image keys).
	std::vector<std::pair<Value, uint32_t>> roots;
	// GPU-thread scratch: the memos of a refresh's clean and ordinary walkers, then of nested
	// EXEC-mask walkers.
	mutable std::array<Context, 2> top_contexts;
	mutable std::deque<Context>    nested_contexts;
	mutable uint32_t               depth   = 0;
	mutable uint64_t               session = 0;
};

// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          SrtWalker* clean_evaluator = nullptr);
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// Refreshes reachable scalar reads and active descriptor sources in one walk.
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

private:
	SrtWalker(SrtWalker& parent, Value active_mask, bool clean = false,
	          SrtWalker* clean_evaluator = nullptr);
	ResourcePlan::EvaluationContext& AcquireContext();
	void ResetUnknowns();
	static float Float32(uint64_t bits);
	bool EvaluateWide(Value value, uint64_t& result);
	bool Arg(const Inst& inst, size_t index, uint64_t& result);
	bool EvaluatePhi(const Inst& inst, uint64_t& result);
	bool EvaluateExtract(const Inst& inst, uint64_t& result);
	bool EvaluateRawRead(const Inst& inst, uint64_t& result);
	bool EvaluateInst(const Inst& inst, uint64_t& result);

	const ResourcePlan&              m_program;
	SrtRuntime                      m_runtime;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
	const std::vector<uint8_t>*      m_reachable_blocks = nullptr;
	const SrtWalker*                m_session;
	uint8_t                         m_read_policy;
	ResourcePlan::EvaluationContext& m_context;
};

// The value graph of a resource plan flattened into dense nodes whose operands are resolved
// node indices. Built once per plan; null when the plan uses a construct the flat walker
// does not model, in which case SrtWalker evaluates the plan.
const CompiledSrtPlan* CompileSrtPlan(const ResourcePlan& program);

// SrtWalker over a compiled plan: the same demand-driven evaluation, memoization, strict-read
// and EXEC-mask rules, without walking Value/Inst operand lists on every refresh.
class CompiledSrtWalker {
public:
	static constexpr uint32_t NoNode = CompiledSrtPlan::NoNode;

	CompiledSrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	                  CompiledSrtWalker* clean_evaluator = nullptr);
	~CompiledSrtWalker();
	CompiledSrtWalker(const CompiledSrtWalker&)            = delete;
	CompiledSrtWalker& operator=(const CompiledSrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// An empty span means that all sources are active.
	// Refreshes reachable scalar reads and active descriptor sources in one walk.
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

private:
	// A nested walker that evaluates under an EXEC mask (NoNode: none).
	CompiledSrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	                  CompiledSrtWalker* clean_evaluator, uint32_t active_mask, bool nested);

	bool EvaluateNode(uint32_t node, uint64_t& result);
	bool EvaluateInst(uint32_t node, uint64_t& result);
	// Forgets the completed unknowns of this refresh's contexts (see SrtWalker::ResetUnknowns()).
	void ResetUnknowns() const;
	bool EvaluateRawRead(uint32_t node, uint64_t& result);
	bool EvaluateLeaf(uint32_t node, uint64_t& result) const;
	bool ReadWord(uint32_t node, uint64_t address, uint64_t& result);

	const ResourcePlan&       m_program;
	const CompiledSrtPlan&    m_plan;
	SrtRuntime                m_runtime;
	CompiledSrtWalker*        m_clean_evaluator = nullptr;
	uint32_t                  m_active_mask     = NoNode;
	bool                      m_nested          = false;
	// The blocks the refresh found reachable (null: all), for control Phis.
	const std::vector<uint8_t>* m_reachable_blocks = nullptr;
	CompiledSrtPlan::Context& m_context;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
