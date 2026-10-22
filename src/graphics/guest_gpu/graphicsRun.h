#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/spscQueue.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <vector>
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

class RenderContext;
class DrawPrepScanner;

class GuestGpu final {
public:
	explicit GuestGpu(RenderContext& renderer);
	~GuestGpu();
	KYTY_CLASS_NO_COPY(GuestGpu);

	void               Shutdown();
	[[nodiscard]] bool IsStopping();
	void               SendCommand(Common::UniqueFunction<void>&& command);
	// Like SendCommand, but false instead of exiting once the GPU no longer accepts commands.
	bool               TrySendCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);
	// Runs a host command on the execution thread before the operations queued so far, and
	// waits for it. Only for work that no queued operation depends on (memory the guest unmaps
	// after the operations that used it executed).
	void SendUrgentCommandSync(Common::UniqueFunction<void>&& command);

	// End-of-pipe labels (RELEASE_MEM without an interrupt) become visible to the CPU once the
	// host GPU has executed the work recorded before them, as on hardware, instead of when the
	// packet is recorded. Until then the GPU thread's own label checks (WAIT_REG_MEM,
	// COND_INDIRECT_BUFFER) see the recorded value through ReadLabel.
	// KYTY_LABELS_AT_COMPLETION=1 enables it.
	[[nodiscard]] static bool LabelsAtCompletion();
	// Whether any label write has been deferred (so ReadLabel must consult the pending ones).
	[[nodiscard]] static bool LabelsDeferred() noexcept;
	// GPU thread.
	void DeferLabelWrite(uint64_t address, uint64_t value, uint32_t size);
	template <typename T>
	[[nodiscard]] T ReadLabel(const volatile T* address) const;

	// Submitted command memory is borrowed and must remain valid until GPU execution completes.
	void Submit(std::span<const uint32_t> draw_commands,
	            std::span<const uint32_t> constant_commands);
	void SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void SubmitFlipPreparation(uint64_t request_id);
	// Insert an ordered graphics drain; only a previous suspend point can block the caller.
	void SuspendPoint();
	// Wait for guest command processing, including all compute queues (not native GPU completion).
	void              WaitForIdle();
	[[nodiscard]] int GetFrameNum() const;

	// The thread that owns the renderer: the execution thread of a pipelined GPU.
	[[nodiscard]] static bool IsGpuThread() noexcept;

	// A pipelined GPU parses command buffers on its command processor thread and executes the
	// operations they produce, in order, on its execution thread.
	[[nodiscard]] bool Pipelined() const noexcept { return m_pipelined; }
	// Command processor thread.
	[[nodiscard]] uint64_t NextSequence() const noexcept { return m_emitted + 1; }
	// Returns the sequence of the operation.
	uint64_t               EmitOperation(GpuOperation&& operation);
	// The guest memory an emitted operation writes, and the value it writes when that is known
	// when the operation is emitted (`value` null otherwise, or larger than 8 bytes).
	void NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size, const void* value);
	// Reads guest memory the command processor decides on as a serial GPU would at this point
	// of the stream: from the value of a pending write when it is known, otherwise from memory
	// once the operations that write it executed.
	void ReadDecisionMemory(const volatile void* address, void* value, size_t size);
	void                   WaitForExecution(uint64_t sequence) { m_executed.WaitFor(sequence); }
	// Waits until every operation emitted so far executed.
	void WaitForExecution() { m_executed.WaitFor(m_emitted); }
	[[nodiscard]] const GpuRegisterState& NeutralState() const noexcept { return m_neutral_state; }

private:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;

	enum class SubmissionType { Graphics, Compute, FlipPreparation, SuspendPoint };

	struct Submission {
		SubmissionType            type     = SubmissionType::Graphics;
		uint32_t                  queue_id = 0;
		std::span<const uint32_t> commands;
		std::span<const uint32_t> constant_commands;
		Pm4Execution              command_execution;
		Pm4Execution              constant_execution;
		bool                      started           = false;
		bool                      command_complete  = false;
		bool                      constant_complete = false;
		bool                      blocked           = false;
		uint64_t                  flip_request_id   = 0;
		uint64_t                  prep_seq          = 0; // draw prep: graphics submission number
		uint64_t                  srt_wait_tick     = 0; // KYTY_SRT_DEFER: run again once free
		// KYTY_COPY_SUBMISSIONS=1: the command words as they were at submission (the spans point
		// here), in case the game rewrites the buffer before the GPU thread reaches it.
		std::shared_ptr<std::vector<uint32_t>> owned_words;
	};

	void              Enqueue(Submission submission);
	void              ProcessCommands();
	void              ExecutionThread();
	void              ResolveThread();
	void              PrepareAhead(GpuOperation& operation);
	void              ResolveAhead(GpuOperation& operation);
	void              StopExecution();
	// Command processor thread of a pipelined GPU: runs a command on the execution thread and
	// waits for it.
	void              ExecuteSync(Common::UniqueFunction<void>&& command);
	bool              Process(Submission& submission);
	// A submission waits in a queue and is not blocked (KYTY_COALESCE_SUBMITS).
	bool              HasReadySubmission();
	static void       ThreadRun(void* data);
	CommandProcessor& GetProcessor(uint32_t queue_id);

	RenderContext&                                 m_renderer;
	Common::Mutex                                  m_queue_mutex;
	std::mutex                                     m_shutdown_mutex;
	Common::CondVar                                m_work_available;
	Common::CondVar                                m_idle;
	std::array<std::deque<Submission>, QueueCount> m_queues;
	std::deque<Common::UniqueFunction<void>>       m_commands;
	std::atomic_uint32_t                           m_pending_commands {0};
	uint32_t                                       m_next_queue        = 0;
	uint32_t                                       m_submission_count  = 0;
	bool                                           m_processing        = false;
	bool                                           m_accepting         = true;
	bool                                           m_stopping          = false;
	bool                                           m_shutdown_complete = false;
	// Completion callbacks can outlive GuestGpu during renderer shutdown.
	std::shared_ptr<std::binary_semaphore> m_suspend_point_ready =
	    std::make_shared<std::binary_semaphore>(1);

	struct PendingLabel {
		uint64_t value    = 0;
		uint32_t size     = 0;
		uint64_t sequence = 0;
	};
	// Label writes recorded but not yet visible to the CPU. GPU thread only.
	std::unordered_map<uint64_t, PendingLabel> m_pending_labels;
	uint64_t                                   m_label_sequence = 0;

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;
	// KYTY_DRAW_PREP: the scanner, and the number of the last graphics submission queued
	// (under m_queue_mutex).
	std::unique_ptr<DrawPrepScanner> m_draw_prep;
	uint64_t                         m_prep_seq = 0;

	uint64_t        m_submit_id = 0;
	std::atomic_int m_done_num  = 0;

	// Pipelined execution. m_emitted is only used by the command processor thread.
	struct GuestWrite {
		uint64_t sequence = 0;
		uint64_t address  = 0;
		uint64_t size     = 0;
		bool     known    = false;
		uint8_t  value[8] = {};
	};

	const bool                      m_pipelined;
	// With a resolve stage, the command processor queues operations for the resolve thread,
	// which forwards them to the execution thread.
	const bool                      m_resolving;
	Common::SpscQueue<GpuOperation> m_operations;
	Common::SpscQueue<GpuOperation> m_resolved;
	// KYTY_VALIDATE_STAGE: operations pass a validation thread between the resolve and the
	// execution threads, which checks their resolutions' guest reads a few operations ahead of
	// execution.
	const bool                      m_validating;
	Common::SpscQueue<GpuOperation> m_validated;
	Common::ProgressCounter         m_executed;
	uint64_t                        m_emitted = 0;
	// Command processor thread: a ring of the writes of operations that may not have executed
	// yet, oldest first. It holds more writes than operations can be queued.
	static constexpr size_t PendingWriteCapacity = 16384;
	std::unique_ptr<GuestWrite[]> m_pending_writes;
	size_t                        m_pending_first = 0;
	size_t                        m_pending_count = 0;
	void                          ForgetExecutedWrites();
	// Commands that run before the next operation the execution thread takes.
	void                                      RunUrgentCommands();
	std::mutex                                m_urgent_mutex;
	std::vector<Common::UniqueFunction<void>> m_urgent_commands;
	std::atomic_uint32_t                      m_urgent_pending {0};
	// Game and host threads bump it when they queue work, under m_queue_mutex; the command
	// processor thread spins on it before it sleeps.
	std::atomic_uint64_t m_work_epoch {0};
	HW::Context                     m_neutral_context;
	HW::UserConfig                  m_neutral_user_config;
	HW::Shader                      m_neutral_shaders;
	GpuRegisterState                m_neutral_state;
	std::jthread                    m_executor;
	std::jthread                    m_resolver;
	std::jthread                    m_validator;
	void                            ValidateThread();

	std::jthread m_thread;

	friend class CommandProcessor;
};

// Diagnostics (KYTY_TRACE_GS_SLOT with KYTY_TRACE_GS_SLOT_SEQ): the last write to the traced GS
// user-data slot on this thread. source: 0 none, 1 SET_SH_REG, 2 SET_SH_REG_INDIRECT table entry,
// 3 other register-list write. where: the dword in the command buffer or table.
struct GsSlotWriter {
	uint32_t        source = 0;
	const uint32_t* where  = nullptr;
	uint32_t        value  = 0;
	uint64_t        seq    = 0;
	uint32_t        start  = 0; // SET_SH_REG: first slot and register count
	uint32_t        count  = 0;
};
GsSlotWriter GetGsSlotLastWriter();
// Lines left for the [gs-seq] trace (0 when it is off); counts one down.
bool TakeGsSlotSeqLine();
} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */
