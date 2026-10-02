#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

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
	// When set, holds the flat SRT slot whose scalar load is reading memory, or NoSrtSlot.
	uint32_t*                 read_slot                  = nullptr;
};

enum class RuntimeValueType { Any, Integer };

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Side-effect-free operations that runtime values may combine.
bool IsRuntimeUniformOp(ValueOpcode op);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);

struct CompiledSrtPlan {
	static constexpr uint32_t NoNode = UINT32_MAX;

	// User data is a side-effect-free leaf (imm = dword index) and a valid ReadConst forwards
	// to its SRT read (imm = slot, args[0] = read); neither needs a memo entry.
	enum class NodeKind : uint8_t { Constant, Invalid, UserData, ReadConst, Instruction };
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
		uint32_t    args[5] {};
		uint64_t    imm = 0;
	};
	struct Context {
		struct Entry {
			uint64_t value      = 0;
			uint64_t generation = 0;
		};
		std::vector<Entry> values;
		uint64_t           generation = 0;
	};

	struct SrtRead {
		uint32_t node        = NoNode;
		uint32_t flat_offset = 0;
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
	bool                    srt_plan_complete = false;
	// Other values that materialization evaluates by Value (fill words, indirect image keys).
	std::vector<std::pair<Value, uint32_t>> roots;
	// GPU-thread scratch: the memos of a refresh's clean and ordinary walkers, then of nested
	// EXEC-mask walkers.
	mutable std::array<Context, 2> top_contexts;
	mutable std::deque<Context>    nested_contexts;
	mutable uint32_t               depth = 0;
};

// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots = {}, SrtWalker* clean_evaluator = nullptr,
	          Value active_mask = {});
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// Refreshes reachable scalar reads and active descriptor sources in one walk.
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

private:
	static ResourcePlan::EvaluationContext& AcquireContext(const ResourcePlan& program);
	static float Float32(uint64_t bits);
	bool EvaluateWide(Value value, uint64_t& result);
	bool Arg(const Inst& inst, size_t index, uint64_t& result);
	bool EvaluatePhi(const Inst& inst, uint64_t& result);
	bool EvaluateExtract(const Inst& inst, uint64_t& result);
	bool EvaluateRawRead(const Inst& inst, uint64_t& result);
	bool EvaluateInst(const Inst& inst, uint64_t& result);

	const ResourcePlan&              m_program;
	SrtRuntime                      m_runtime;
	std::span<const uint8_t>         m_clean_flat_slots;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
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
	                  std::span<const uint8_t> clean_flat_slots = {},
	                  CompiledSrtWalker* clean_evaluator = nullptr, uint32_t active_mask = NoNode);
	~CompiledSrtWalker();
	CompiledSrtWalker(const CompiledSrtWalker&)            = delete;
	CompiledSrtWalker& operator=(const CompiledSrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// An empty span means that all sources are active.
	// Refreshes reachable scalar reads and active descriptor sources in one walk.
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

private:
	bool EvaluateNode(uint32_t node, uint64_t& result);
	bool EvaluateInst(uint32_t node, uint64_t& result);
	bool EvaluateRawRead(uint32_t node, uint64_t& result);
	bool EvaluateLeaf(uint32_t node, uint64_t& result) const;
	bool ReadWord(uint32_t node, uint64_t address, uint64_t& result);

	const ResourcePlan&       m_program;
	const CompiledSrtPlan&    m_plan;
	SrtRuntime                m_runtime;
	std::span<const uint8_t>  m_clean_flat_slots;
	CompiledSrtWalker*        m_clean_evaluator = nullptr;
	uint32_t                  m_active_mask     = NoNode;
	CompiledSrtPlan::Context& m_context;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
