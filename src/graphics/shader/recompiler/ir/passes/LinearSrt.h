#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_LINEARSRT_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_LINEARSRT_H_

// Native evaluation of resource plans (chenxiao07's KYTY_SRT_NATIVE and KYTY_SRT_PREDICATES, LinearSrt.h and
// src/local/native-resource-aot.h), over our compiled SRT plans (CompiledSrtPlan).
//
// The SRT interpreter (CompiledSrtWalker) evaluates a plan's flat SRT and descriptors on demand, node by node, through
// memos. A linear plan is the same evaluation as one straight-line program: every node a materialization needs, in
// dependency order, once per read policy (a node under a clean, specialization-affecting read is evaluated with the
// strict reader, as the interpreter's clean evaluator does). It runs as native code generated with Xbyak, or as a
// function of a library compiled ahead of time from the plans' exported sources (SrtAotAbi.h). A plan whose values
// branch on control flow evaluates its conditions natively first (KYTY_SRT_PREDICATES), then runs the linear plan of
// the blocks they reach. Plans with what the linear form does not model (control Phis, EXEC-mask lane reads, more
// than two conditional blocks) keep the interpreter; so does any evaluation that fails natively (the interpreter
// then evaluates it again: it may succeed where an eager evaluation fails, short-circuits and selects being lazy).

#include "common/abi.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Xbyak {
class CodeGenerator;
}

namespace Libs::Graphics::ShaderRecompiler::IR {

struct LinearSrtPlan {
	static constexpr uint32_t Invalid = UINT32_MAX;
	enum class Kind : uint8_t { Immediate, User, Base, LeafRead, RawRead, Select, Op };
	struct Node {
		Kind        kind  = Kind::Immediate;
		ValueOpcode op    = ValueOpcode::Void;
		uint8_t     aux   = 0;
		uint8_t     aux2  = 0;
		bool        clean = false; // a read through the strict reader
		uint32_t    args[5] {Invalid, Invalid, Invalid, Invalid, Invalid};
		uint64_t    imm  = 0;
		uint32_t    slot = NoSrtSlot; // a read's flat SRT slot (CompiledSrtPlan::read_slots)
		uint32_t    plan_node = Invalid; // the compiled plan's node it is (Invalid: a constant or user data)
	};
	struct Leaf {
		std::vector<Node>     nodes;
		std::vector<uint32_t> flat_words;       // per flat slot: its node, or Invalid (an unreached read: 0)
		std::vector<uint32_t> descriptor_words; // per descriptor source, 8 words: their nodes, or Invalid
		std::vector<uint8_t>  descriptor_built; // per descriptor source
		std::vector<uint8_t>  active_sources;   // empty: all
		bool                  complete = true;  // every directly materialized active source is built
		using Function = bool KYTY_SYSV_ABI (*)(const void* runtime, uint64_t* values);
		Function                              function = nullptr;
		std::shared_ptr<Xbyak::CodeGenerator> code;
		bool (*aot)(const void* runtime, uint64_t* values) = nullptr;
	};
	// Plans without control flow: one leaf. With it: a leaf per combination of the conditions' outcomes (true, false,
	// unknown), and a program per condition (its flat word: the condition), in `conditionals` order.
	std::vector<std::shared_ptr<Leaf>> leaves;
	std::vector<uint32_t>              conditionals; // the conditional blocks
	std::vector<std::shared_ptr<Leaf>> predicates;
	std::vector<uint32_t>              leaf_of_choice; // per combination (base 3, conditionals[0] lowest): a leaf
	bool                               clean_evaluator = false; // built for a walker with a clean evaluator
	size_t                             plan_nodes      = 0;     // the compiled plan's node count
	bool                               seed            = false; // the interpreter evaluates more afterwards
};

// The plan's linear form, built on first use (null: the interpreter evaluates it). Callers hold the program lock, as
// for CompileSrtPlan().
const LinearSrtPlan* LinearSrtOf(const ResourcePlan& program, bool clean_evaluator);

// Evaluates the flat SRT and the directly materialized descriptors natively (descriptor_valid: which). Failed: the
// evaluation failed where the interpreter's fails too (nothing the interpreter may skip can fail natively: see
// Builder); NotApplicable: the interpreter evaluates them. Sets program.active_sources as
// CompiledSrtWalker::RefreshFlatBuffer() does. The guest reads are those the interpreter makes, in another order.
// `seed` receives each computed node's value (with its read policy) for the interpreter's memos: what the interpreter
// evaluates next (fill values, indirect descriptors) reuses them as it reuses its own, without reading them again.
enum class LinearSrtResult { NotApplicable, Failed, Evaluated };
using LinearSrtSeed = void (*)(void* context, bool clean, uint32_t node, uint64_t value);
LinearSrtResult EvaluateLinearSrt(const ResourcePlan& program, const LinearSrtPlan& plan, const SrtRuntime& runtime,
                                  std::vector<uint32_t>& flat, std::vector<DescriptorValue>& descriptors,
                                  std::vector<uint8_t>& descriptor_valid, LinearSrtSeed seed, void* seed_context);

// KYTY_SRT_NATIVE=0: plans always take the interpreter.
bool LinearSrtEnabled();
void SetLinearSrtEnabledForTesting(bool enabled);
// KYTY_SRT_LINEAR_VERIFY=1: native evaluations are checked against the interpreter (mismatches printed).
bool LinearSrtVerifyEnabled();

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_LINEARSRT_H_
