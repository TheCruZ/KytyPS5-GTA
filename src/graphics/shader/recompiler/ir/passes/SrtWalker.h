#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <span>
#include <vector>

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
	// Accept image atomics on k32Float descriptors: upstream's float atomics, and integer atomics
	// run as uint atomics on the raw bits. On by default, as in the emulator; the emulator passes
	// --no-float-image-atomics through here.
	bool                      float_image_atomics        = true;
	// Compute: the workgroup size; with workgroup_counts (zero when the host does not know them,
	// as for an indirect dispatch) it bounds the invocation IDs in buffer write extents.
	std::array<uint32_t, 3>   workgroup_size             = {};
	// Optional fast path for raw reads: the host bytes of a guest page whose contents are current
	// (no GPU-written byte), or null; such a page is read directly instead of through
	// read_memory. Valid for one refresh. A reader that cannot read (CleanRuntime without a
	// strict reader) has none.
	const uint8_t* (*map_clean_page)(void* userdata, uint64_t page) = nullptr;
	void*          page_userdata                                     = nullptr;
	uint32_t       page_shift                                        = 12;
	// When the refresh captures its reads (specialization_reads), reads taken through the fast
	// path are recorded here, as read_memory's capture records them.
	std::vector<std::pair<uint64_t, uint64_t>>* capture_ranges = nullptr;
	// GPU-filled flat slots (SrtGpuFill): whether bytes are GPU-written, and where the refresh
	// lists the slots it left to the GPU. Both null: every slot is read on the host.
	bool (*gpu_written)(uint64_t address, uint64_t size) = nullptr;
	std::vector<SrtGpuFill>* gpu_fills                    = nullptr;
	// KYTY_GPU_SIZED_DYNAMIC: the last value of a GPU-written dword the host has seen, from an
	// earlier copy that did not make the host wait (0 when none landed yet). Bounds a binding
	// whose exact record count the shader reads on the GPU.
	uint32_t (*peek_gpu_dword)(uint64_t address) = nullptr;
	uint64_t                  indirect_args_address = 0;
};

class SrtReadCapture {
public:
	SrtReadCapture(SrtRuntime source, std::vector<std::pair<uint64_t, uint64_t>>& ranges):
	    m_source(source), m_ranges(ranges) {}
	SrtRuntime ObservedRuntime();

private:
	static bool ReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values);
	static bool ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values);

	SrtRuntime                                  m_source;
	std::vector<std::pair<uint64_t, uint64_t>>& m_ranges;
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

// True while the walker evaluates a control-flow condition (which blocks, and so which
// descriptor sources, are active). The renderer's readers then fail a read of GPU-written bytes
// instead of waiting for the GPU: the condition becomes "not known" and both successors are
// walked. KYTY_SRT_CONDITIONS_WAIT=1 restores waiting.
bool SrtEvaluatingCondition();
bool SrtConditionsNoWait();
// The same for other values the host only uses to narrow what it does (buffer write extents):
// a failed evaluation falls back to the conservative answer.
void SrtSetEvaluatingCondition(bool evaluating);
// Reads refused under SrtSetEvaluatingCondition on this thread so far: a refused read may still
// let an evaluation succeed (a condition's unknown value), so callers that need the exact value
// compare the count before and after.
uint64_t SrtConditionRefusals();
void     SrtNoteConditionRefusal();

class SrtWalker;

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
	// The last raw read that failed (diagnostics).
	[[nodiscard]] const char* ReadFailure() const { return m_read_failure; }
	[[nodiscard]] uint64_t    ReadFailureAddress() const { return m_read_failure_address; }

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
	// EvaluateRawRead's read once the address is known.
	bool ReadRawWord(uint64_t address, uint64_t base, uint64_t& result);
	bool EvaluateInst(const Inst& inst, uint64_t& result);

	const ResourcePlan&              m_program;
	SrtRuntime                      m_runtime;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
	const std::vector<uint8_t>*      m_reachable_blocks = nullptr;
	const SrtWalker*                m_session;
	uint8_t                         m_read_policy;
	ResourcePlan::EvaluationContext& m_context;
	// ReadRawWord's fast path: the last page mapped (SrtRuntime::map_clean_page).
	uint64_t       m_mapped_page  = UINT64_MAX;
	// While a GPU-fillable slot is refreshed: its flat offset (its own read is the only raw read).
	bool           m_fill_active  = false;
	uint32_t       m_fill_offset  = 0;
	// A slot whose address comes from other reads (gpu_fill_slots 2): only its own read (this
	// instruction) is left to the GPU; the reads that make its address are done on the host.
	bool           m_fill_any     = true;
	const Inst*    m_fill_inst    = nullptr;
	// The read instruction ReadRawWord serves (EvaluateRawRead).
	const Inst*    m_reading_inst = nullptr;
	const uint8_t* m_mapped_bytes = nullptr;
	// The last raw read that failed, for RefreshFlatBuffer's report.
	const char* m_read_failure         = nullptr;
	uint64_t    m_read_failure_address = 0;
	uint64_t    m_read_failure_offset  = 0;
	uint64_t    m_read_failure_size    = 0;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
