#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <condition_variable>
#include <mutex>

#include <queue>

#include <thread>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	// With deferred_recording, command buffers are recorded into a CommandStream and a thread of
	// the scheduler replays them into Vulkan command buffers and submits them in order.
	CommandScheduler(RenderContext& context, GraphicContext& graphics,
	                 bool deferred_recording = false);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(const HW::Context& registers, const HW::UserConfig& user_config,
	                     const HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();
		// A command buffer for the recording thread, busy until Retire() gives it the tick of
		// its submission.
		[[nodiscard]] std::pair<vk::CommandBuffer, size_t> CommitDeferred();
		void                                               Retire(size_t index, uint64_t tick);

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();
		// A command buffer the GPU finished with, marked busy until `busy_until`.
		size_t Acquire(uint64_t busy_until);

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	// A submission recorded into the stream: the recording thread submits it in order.
	struct RecordedSubmit {
		SubmitInfo info;
		uint64_t   tick            = 0;
		uint64_t   debug_submit_id = 0;
		uint64_t   debug_arg4      = 0;
		uint32_t   debug_op        = 0;
		uint32_t   debug_arg0      = 0;
		uint32_t   debug_arg1      = 0;
		uint32_t   debug_arg2      = 0;
		uint32_t   debug_arg3      = 0;
	};

	void BeginNext();
	void RecordingThread();
	void SubmitRecorded(vk::CommandBuffer& buffer, size_t buffer_index,
	                    const RecordedSubmit& recorded);
	void QueueSubmit(vk::CommandBuffer buffer, SubmitInfo& submit, uint64_t tick,
	                 const RecordedSubmit& debug);
	void PriorityOperationsThread(std::stop_token stop);
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	const bool                   m_deferred;
	CommandChunkQueue            m_chunk_queue;
	CommandStream                m_stream;
	CommandBuffer                m_command;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;
	// Replays m_stream when recording is deferred; declared last so it stops first.
	std::jthread m_recording_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
