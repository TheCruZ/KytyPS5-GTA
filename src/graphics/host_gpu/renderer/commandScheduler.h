#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/spscQueue.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/hostCopyQueue.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <emmintrin.h>
#include <source_location>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>

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
	void           EndRendering(std::source_location site = std::source_location::current());
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	// Bounds how many no-interrupt RELEASE_MEM fence writes accumulate in one command buffer
	// before it is submitted, so consecutive fence updates that nothing is blocked waiting on
	// don't each pay a host vkQueueSubmit. Only safe for a RELEASE_MEM whose guest-visible write
	// already happened synchronously and that scheduled no interrupt callback -- see the call
	// site in pm4Handlers.cpp CpOpReleaseMem for the exact condition. Never waits itself.
	void           CompleteReleaseMemWrite();
	// Same idea, but for a RELEASE_MEM that DOES request a guest interrupt/event (a guest thread
	// may be waiting on it via an event queue) -- deferring the flush delays real event delivery,
	// so this uses a much smaller batch bound than CompleteReleaseMemWrite as a hedge, trading
	// some of the possible win for less added latency. Never waits itself.
	void           CompleteReleaseMemInterrupt();
	// Called after every DrawIndex/DrawAuto. Long chains of draws with no intervening
	// RELEASE_MEM/wait can otherwise sit fully recorded but unsubmitted for a long time, leaving
	// the GPU idle until something else finally forces a flush -- this periodically calls Flush()
	// (non-blocking: Submit() + BeginNext(), same as CompleteReleaseMemWrite) every
	// KYTY_DRAW_FLUSH_INTERVAL draws to keep the queue fed instead. Submitting while the GPU is
	// still executing earlier work is always legal (vkQueueSubmit never waits on prior submissions
	// completing), so this never blocks the CPU. Defaults to 16, validated against real gameplay --
	// override via KYTY_DRAW_FLUSH_INTERVAL (0 disables) if a different workload needs retuning:
	// too small reintroduces per-submit overhead, too large leaves the same idle bubbles this
	// exists to remove.
	void           CompleteDraw();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	// Operations deferred to a tick the GPU has not finished are waiting (cheap, lock-free).
	[[nodiscard]] bool        HasPendingOperations() const noexcept {
		return m_pending_front_tick.load(std::memory_order_acquire) != UINT64_MAX;
	}
	// PopPendingOperations() for a thread with nothing else to do: asks the GPU timeline every call
	// instead of every few draws.
	void                      PollPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Whether a deferred or priority operation that may write guest memory in
	// [address, address + size) is queued or running.
	[[nodiscard]] bool        HasPendingGuestOperations(uint64_t address, uint64_t size);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	// A completion that may act on guest memory makes an unmap of memory it may write drain the
	// GPU while it is pending. They write at most [address, address + size), by default anywhere.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation,
	                                         uint64_t address = 0, uint64_t size = UINT64_MAX);
	// As DeferOperation, also between guest command buffers (no active command buffer): runs
	// once the GPU has finished the current tick, so work submitted earlier, still running, can
	// keep using what the operation releases. Acts on host state only.
	void                      DeferRelease(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation,
	                                                 uint64_t address = 0,
	                                                 uint64_t size    = UINT64_MAX);
	// Completions that only act on host state (cache bookkeeping, releases, flips, interrupts):
	// they never make an unmap drain the GPU.
	void                      DeferHostOperation(Common::UniqueFunction<void>&& operation,
	                                             bool priority = false);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	[[nodiscard]] static bool InDeferredOperation() noexcept;
	// Hands later submissions to a dedicated queue thread, which submits them in tick order, so
	// vkQueueSubmit (and the queue lock that present also takes) leaves the recording thread.
	// Submit() then returns once the tick is allocated; host waits on a timeline value may precede
	// its signal operation. Enable before the first Submit().
	void EnableAsyncSubmit();
	[[nodiscard]] bool AsyncSubmit() const noexcept { return m_async_submit; }
	// A Vulkan recording thread (KYTY_RECORD_THREAD=0 turns it off; the pattern and the
	// CommandRecorder are IDXTRI's, d9958e28). The GPU thread queues the current command
	// buffer's Vulkan commands (CommandBuffer::Recorder(), Record()); the recording thread begins,
	// records, ends and submits the command buffers in order, so the driver's recording cost
	// leaves the GPU thread. Ticks are still allocated here, in order, at Submit(). Code that
	// records through the raw CommandBuffer::Handle() first waits for the thread to catch up
	// (DrainRecording), so it stays correct. Replaces the queue thread (EnableAsyncSubmit).
	// Enable before the first command buffer.
	void EnableRecordingThread();
	// The current command buffer's commands go through the recording thread.
	[[nodiscard]] bool Threaded() const noexcept { return m_threaded; }
	// A host copy thread for the stream-buffer writes of the pipeline's execution thread (see
	// HostCopyQueue). Submissions wait for the copies queued before them. Enable it before the
	// execution thread starts; null until then.
	void                         EnableHostCopies();
	[[nodiscard]] HostCopyQueue* HostCopies() noexcept { return m_host_copies.get(); }
	// The most one Stash() or Record() can take: a larger copy is made another way (a bindless
	// image table's descriptor array is megabytes).
	static constexpr size_t MaxStashBytes = 128 * 1024;
	// Queues fn(vk::CommandBuffer) for the recording thread (threaded only). Captures must be
	// values; arrays go through Stash().
	template <typename F>
	void Record(F&& fn) {
		using Command = TypedCommand<std::decay_t<F>>;
		void* memory  = Allocate(sizeof(Command), alignof(Command));
		Link(new (memory) Command(std::forward<F>(fn)));
	}
	// Queues state.Commit(graphics, command) for the recording thread with the state copied once,
	// straight into the chunk (a closure would copy its capture twice more).
	void RecordDynamicState(const DynamicState& state, const GraphicContext& graphics) {
		struct Command final: RecordedCommand {
			Command(const DynamicState& source, const GraphicContext* context)
			    : state(source), graphics(context) {}
			void Execute(vk::CommandBuffer command) override { state.Commit(*graphics, command); }
			DynamicState          state;
			const GraphicContext* graphics;
		};
		void* memory = Allocate(sizeof(Command), alignof(Command));
		Link(new (memory) Command(state, &graphics));
	}
	// Queues vkCmdPushDescriptorSetKHR with the writes, and the info arrays they point to, copied
	// right behind the command in one reservation (Stash() of each array would reserve once for
	// the writes and once more for every info array, ~10 reservations a draw). False (nothing is
	// queued) when the copy would exceed MaxStashBytes.
	[[nodiscard]] bool RecordPushDescriptors(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                           uint32_t set, uint32_t count, const vk::WriteDescriptorSet* writes) {
		struct Command final: RecordedCommand {
			Command(vk::PipelineBindPoint point, vk::PipelineLayout pipeline_layout,
			        uint32_t set_index, uint32_t write_count)
			    : bind_point(point), layout(pipeline_layout), set(set_index), count(write_count) {}
			void Execute(vk::CommandBuffer command) override {
				command.pushDescriptorSetKHR(
				    bind_point, layout, set, count,
				    reinterpret_cast<const vk::WriteDescriptorSet*>(this + 1));
			}
			vk::PipelineBindPoint bind_point;
			vk::PipelineLayout    layout;
			uint32_t              set;
			uint32_t              count;
		};
		static_assert(sizeof(Command) % alignof(vk::WriteDescriptorSet) == 0);
		const size_t bytes = DeepWritesBytes(writes, count);
		if (sizeof(Command) + bytes > MaxStashBytes) {
			return false;
		}
		void* memory  = Allocate(sizeof(Command) + bytes, alignof(Command));
		auto* command = new (memory) Command(bind_point, layout, set, count);
		CopyDeepWrites(reinterpret_cast<std::byte*>(command + 1), writes, count);
		Link(command);
		return true;
	}
	// Copies count elements into memory that stays valid until the recording thread has run the
	// next command recorded after this call (threaded only).
	template <typename T>
	[[nodiscard]] const T* Stash(const T* data, size_t count) {
		if (data == nullptr || count == 0) {
			return nullptr;
		}
		auto* copy = static_cast<T*>(Allocate(sizeof(T) * count, alignof(T)));
		CopyStash(copy, data, sizeof(T) * count);
		return copy;
	}
	// Waits until the recording thread has run every queued command, and returns the command
	// buffer it records into, for code that records directly.
	vk::CommandBuffer DrainRecording();

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	// Waits at most timeout_ns for the GPU to reach a submitted tick; true when it has.
	bool                           WaitFor(uint64_t tick, uint64_t timeout_ns);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	// The last tick whose command buffer reached vkQueueSubmit: with deferred recording, a tick
	// the execution thread already submitted may still wait for the recording thread.
	[[nodiscard]] uint64_t SubmittedTick() const noexcept {
		return m_submitted_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }
	// Copies of guest data into the stream buffer that the execution thread hands off, with
	// deferred recording (null otherwise). Submissions wait for the copies recorded before them;
	// the owner of the scheduler drains them before any guest-visible effect.
	[[nodiscard]] HostCopyQueue* HostCopies() noexcept { return m_host_copies.get(); }
	// Called before each submission, while its command buffer still takes commands.
	using SubmitHook = void (*)(void* userdata, CommandBuffer& command);
	void SetSubmitHook(SubmitHook hook, void* userdata) noexcept {
		m_submit_hook      = hook;
		m_submit_hook_data = userdata;
	}

private:
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority);
	Common::UniqueFunction<void> TrackGuestOperation(Common::UniqueFunction<void>&& operation,
	                                                 uint64_t address, uint64_t size);

	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		// tick: the tick the command buffer will be submitted with.
		vk::CommandBuffer Commit(uint64_t tick);

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

	// A finished command buffer and everything its vkQueueSubmit needs, including the recording
	// state that a fatal submit report prints.
	struct SubmitJob {
		vk::CommandBuffer buffer = nullptr;
		SubmitInfo        submit;
		uint64_t          tick         = 0;
		// Host copies queued before the submission: the GPU reads their destinations.
		uint64_t          host_copies  = 0;
		uint32_t          debug_op     = 0;
		uint64_t          debug_submit = 0;
		uint32_t          debug_arg0   = 0;
		uint32_t          debug_arg1   = 0;
		uint32_t          debug_arg2   = 0;
		uint32_t          debug_arg3   = 0;
		uint64_t          debug_arg4   = 0;
	};

	struct RecordedCommand {
		virtual ~RecordedCommand()                  = default;
		virtual void     Execute(vk::CommandBuffer) = 0;
		RecordedCommand* next                       = nullptr;
	};
	template <typename F>
	struct TypedCommand final: RecordedCommand {
		explicit TypedCommand(F&& function): fn(std::move(function)) {}
		explicit TypedCommand(const F& function): fn(function) {}
		void Execute(vk::CommandBuffer command) override { fn(command); }
		F    fn;
	};
	struct Chunk {
		static constexpr size_t Size = 256 * 1024;
		alignas(64) std::byte storage[Size];
		size_t           used  = 0;
		RecordedCommand* first = nullptr;
		RecordedCommand* last  = nullptr;
	};
	// What a draw stashes is a few dozen descriptor writes and infos of 24 or 64 bytes: copied
	// inline, as a call into memcpy costs more than the copy.
	static void CopyStash(void* destination, const void* source, size_t bytes) {
		if (bytes > 256) {
			std::memcpy(destination, source, bytes);
			return;
		}
		auto*       to   = static_cast<uint8_t*>(destination);
		const auto* from = static_cast<const uint8_t*>(source);
		size_t      done = 0;
		for (; done + 16 <= bytes; done += 16) {
			_mm_storeu_si128(reinterpret_cast<__m128i*>(to + done),
			                 _mm_loadu_si128(reinterpret_cast<const __m128i*>(from + done)));
		}
		if (done + 8 <= bytes) {
			uint64_t word = 0;
			std::memcpy(&word, from + done, sizeof(word));
			std::memcpy(to + done, &word, sizeof(word));
			done += 8;
		}
		if (done < bytes) {
			std::memcpy(to + done, from + done, bytes - done);
		}
	}
	// The bytes of the writes and of the image, buffer and texel buffer view arrays they point to.
	static size_t DeepWritesBytes(const vk::WriteDescriptorSet* writes, uint32_t count) {
		size_t bytes = sizeof(vk::WriteDescriptorSet) * count;
		for (uint32_t i = 0; i < count; i++) {
			const auto& write = writes[i];
			if (write.pImageInfo != nullptr) {
				bytes += sizeof(vk::DescriptorImageInfo) * write.descriptorCount;
			}
			if (write.pBufferInfo != nullptr) {
				bytes += sizeof(vk::DescriptorBufferInfo) * write.descriptorCount;
			}
			if (write.pTexelBufferView != nullptr) {
				bytes += sizeof(vk::BufferView) * write.descriptorCount;
			}
		}
		return bytes;
	}
	// Writes the copy of the writes at block (DeepWritesBytes() bytes, 8-aligned); the copy's
	// arrays follow the writes.
	static void CopyDeepWrites(std::byte* block, const vk::WriteDescriptorSet* writes,
	                           uint32_t count) {
		static_assert(sizeof(vk::DescriptorImageInfo) % 8 == 0 &&
		              sizeof(vk::DescriptorBufferInfo) % 8 == 0 && sizeof(vk::BufferView) % 8 == 0);
		auto* copy = reinterpret_cast<vk::WriteDescriptorSet*>(block);
		CopyStash(copy, writes, sizeof(vk::WriteDescriptorSet) * count);
		std::byte* cursor = block + sizeof(vk::WriteDescriptorSet) * count;
		const auto take = [&cursor](const auto* source, size_t elements) {
			using T = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
			if (source == nullptr || elements == 0) {
				return static_cast<const T*>(nullptr);
			}
			auto* destination = reinterpret_cast<T*>(cursor);
			CopyStash(destination, source, sizeof(T) * elements);
			cursor += sizeof(T) * elements;
			return static_cast<const T*>(destination);
		};
		for (uint32_t i = 0; i < count; i++) {
			auto& write = copy[i];
			write.pImageInfo       = take(write.pImageInfo, write.descriptorCount);
			write.pBufferInfo      = take(write.pBufferInfo, write.descriptorCount);
			write.pTexelBufferView = take(write.pTexelBufferView, write.descriptorCount);
		}
	}
	// Memory in the current chunk for the recording thread; the common case is a pointer bump.
	void* Allocate(size_t size, size_t alignment) {
		if (m_chunk != nullptr) {
			const auto offset = (m_chunk->used + alignment - 1) & ~(alignment - 1);
			if (offset + size <= Chunk::Size) {
				m_chunk->used = offset + size;
				return m_chunk->storage + offset;
			}
		}
		return AllocateInNewChunk(size, alignment);
	}
	void* AllocateInNewChunk(size_t size, size_t alignment);
	void  Link(RecordedCommand* command) {
		if (m_chunk->last != nullptr) {
			m_chunk->last->next = command;
		} else {
			m_chunk->first = command;
		}
		m_chunk->last = command;
	}
	void  DispatchChunk();
	void  RecordingThread(std::stop_token stop);
	void  StopRecordingThread();
	void  ReportRecording();

	void BeginNext();
	void QueueSubmit(SubmitJob& job);
	void SubmitThread(std::stop_token stop);
	void StopSubmitThread();
	void PriorityOperationsThread(std::stop_token stop);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	// KYTY_GPU_TIME (see commandScheduler.cpp).
	struct GpuTimer;
	std::unique_ptr<GpuTimer> m_gpu_timer;
	uint32_t                  m_timer_slot = UINT32_MAX;

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	const bool                   m_deferred;
	CommandChunkQueue            m_chunk_queue;
	CommandStream                m_stream;
	CommandBuffer                m_command;
	uint32_t                     m_recorded_release_mem_writes     = 0;
	uint32_t                     m_recorded_release_mem_interrupts = 0;
	uint32_t                     m_recorded_draws                  = 0;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations;
	// The tick of the oldest pending operation (UINT64_MAX without any), written under
	// m_operation_mutex: draws check it without the lock.
	std::atomic<uint64_t>        m_pending_front_tick {UINT64_MAX};
	// Calls since draws last queried the GPU timeline for pending operations.
	std::atomic<uint32_t>        m_pending_skips {0};
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	// Guest memory that queued or running guest operations may write, [begin, end) by id.
	struct GuestWrite {
		uint64_t id    = 0;
		uint64_t begin = 0;
		uint64_t end   = 0;
	};
	std::mutex                   m_guest_writes_mutex;
	std::vector<GuestWrite>      m_guest_writes;
	uint64_t                     m_next_guest_write     = 0;
	OperationState               m_operation_state      = OperationState::Open;
	bool                         m_async_submit         = false;
	std::deque<SubmitJob>        m_submit_jobs;
	std::mutex                   m_submit_mutex;
	std::condition_variable_any  m_submit_available;
	std::jthread                 m_submit_thread;
	// Recording thread. m_threaded, m_chunk, m_chunks_dispatched and m_drains belong to the
	// producer (the GPU thread); m_worker_buffer to the recording thread (read by the producer only
	// after a drain). The chunks go to the recording thread through a lock-free ring and come back
	// through another: neither side makes a system call while the other one is busy.
	using ChunkQueue = Common::SpscQueue<std::unique_ptr<Chunk>>;
	bool                                m_threaded = false;
	std::unique_ptr<HostCopyQueue>      m_host_copies;
	std::unique_ptr<Chunk>              m_chunk;
	std::unique_ptr<ChunkQueue>         m_record_queue;
	std::unique_ptr<ChunkQueue>         m_free_chunks;
	Common::ProgressCounter             m_chunks_executed;
	uint64_t                            m_chunks_dispatched = 0;
	uint64_t                            m_drains            = 0;
	vk::CommandBuffer                   m_worker_buffer     = nullptr;
	uint64_t                            m_record_report_buffers = 0;
	uint64_t                            m_record_report_chunks  = 0;
	uint64_t                            m_record_report_drains  = 0;
	std::chrono::steady_clock::time_point m_record_report_time {};
	std::jthread                        m_record_thread;
};

} // namespace Libs::Graphics

namespace Libs::Graphics {
// Diagnostics (KYTY_FPS_LOG): what the next submission on this thread is for, counted by Submit().
void SetSubmitReason(const char* reason);
void SetSubmitReasonIfUnset(const char* reason);
} // namespace Libs::Graphics
#define KYTY_SUBMIT_REASON_STR2(x) #x
#define KYTY_SUBMIT_REASON_STR(x)  KYTY_SUBMIT_REASON_STR2(x)
#define KYTY_SUBMIT_REASON()       ::Libs::Graphics::SetSubmitReason(__FILE__ ":" KYTY_SUBMIT_REASON_STR(__LINE__))

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
