#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/emulatorConfig.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <optional>
#include <tuple>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	return m_buffers[Acquire(m_master.CurrentTick())];
}

std::pair<vk::CommandBuffer, size_t> CommandScheduler::CommandPool::CommitDeferred() {
	const auto index = Acquire(std::numeric_limits<uint64_t>::max());
	return {m_buffers[index], index};
}

void CommandScheduler::CommandPool::Retire(size_t index, uint64_t tick) {
	EXIT_IF(index >= m_ticks.size() || m_ticks[index] != std::numeric_limits<uint64_t>::max());
	m_ticks[index] = tick;
}

size_t CommandScheduler::CommandPool::Acquire(uint64_t busy_until) {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick, busy_until](size_t begin,
	                                                  size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = busy_until;
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = busy_until;
	}

	m_hint = (*found + 1) % m_ticks.size();
	return *found;
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics,
                                   bool deferred_recording)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_deferred(deferred_recording),
      m_stream(&m_chunk_queue), m_command(*this, deferred_recording ? &m_stream : nullptr),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {
	if (m_deferred) {
		m_recording_thread = std::jthread([this] { RecordingThread(); });
	}
}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	if (m_recording_thread.joinable()) {
		// Every recorded submission has completed.
		m_chunk_queue.Stop();
		m_recording_thread.join();
	}
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(const HW::Context& registers, const HW::UserConfig& user_config,
                             const HW::Shader& shaders) {
	// Every GPU operation binds its register state. A shutdown submits the open command buffer
	// before it closes the scheduler, so an open command buffer means an open scheduler.
	if (!m_command.IsInvalid()) {
		m_command.Bind(registers, user_config, shaders);
		return;
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);
	BeginNext();
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && !m_command.IsInvalid()) {
		Current().EndRendering();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait() {
	const auto tick = Submit();
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	// Every draw and dispatch comes here: only query the timeline semaphore (a driver call) when
	// the oldest operation is not already known to be free.
	bool refreshed = false;
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty()) {
				return;
			}
			if (!m_master.IsFree(m_pending_operations.front().tick)) {
				if (refreshed) {
					return;
				}
				m_master.Refresh();
				refreshed = true;
				if (!m_master.IsFree(m_pending_operations.front().tick)) {
					return;
				}
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_pending_operations.push({std::move(operation), CurrentTick()});
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_priority_operations.push({std::move(operation), CurrentTick()});
		lock.unlock();
		m_operation_available.notify_one();
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	CheckActive();
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	if (m_deferred) {
		m_command.m_open = true;
	} else {
		m_command.m_buffer = m_command_pool.Commit();
	}
	m_command.Begin();
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);

	m_command.End();
	// Only this scheduler's owner allocates its ticks, in submission order.
	const auto     tick = m_master.NextTick();
	RecordedSubmit debug {};
	debug.debug_op        = m_command.m_debug_op;
	debug.debug_submit_id = m_command.m_debug_submit_id;
	debug.debug_arg0      = m_command.m_debug_arg0;
	debug.debug_arg1      = m_command.m_debug_arg1;
	debug.debug_arg2      = m_command.m_debug_arg2;
	debug.debug_arg3      = m_command.m_debug_arg3;
	debug.debug_arg4      = m_command.m_debug_arg4;
	if (m_deferred) {
		// Waiting for the tick waits for the recording thread too: the timeline semaphore only
		// reaches it after the thread submitted the commands recorded until now.
		auto* recorded = m_stream.RecordSpecial<RecordedSubmit>(CommandStream::RecordKind::Submit);
		*recorded      = debug;
		recorded->info = submit;
		recorded->tick = tick;
		m_stream.Publish();
		m_command.m_open = false;
		return tick;
	}

	QueueSubmit(m_command.m_buffer, submit, tick, debug);
	m_command.m_buffer = nullptr;
	return tick;
}

void CommandScheduler::QueueSubmit(vk::CommandBuffer buffer, SubmitInfo& submit, uint64_t tick,
                                   const RecordedSubmit& debug) {
	auto& graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);
	submit.AddSignal(m_master.Handle(), tick);

	vk::TimelineSemaphoreSubmitInfo timeline_info {};
	timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
	timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
	timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
	timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

	vk::SubmitInfo submit_info {};
	submit_info.pNext                = &timeline_info;
	submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
	submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
	submit_info.pWaitDstStageMask    = submit.wait_stages.data();
	submit_info.commandBufferCount   = buffer != nullptr ? 1 : 0;
	submit_info.pCommandBuffers      = buffer != nullptr ? &buffer : nullptr;
	submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
	submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

	vk::Result result;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		result = graphics.queue.submit(1, &submit_info, nullptr);
	}

	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, tick, debug.debug_op, debug.debug_submit_id,
		                  debug.debug_arg0, debug.debug_arg1, debug.debug_arg2, debug.debug_arg3,
		                  debug.debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandScheduler::RecordingThread() {
	KYTY_PROFILER_THREAD("Thread_GpuRecord");
	Config::ConfigureGpuStageThread(Config::GpuStageThread::Recording);
	vk::CommandBuffer buffer       = nullptr;
	size_t            buffer_index = 0;
	const auto        acquire      = [this, &buffer, &buffer_index] {
		if (buffer == nullptr) {
			std::tie(buffer, buffer_index) = m_command_pool.CommitDeferred();
			vk::CommandBufferBeginInfo begin_info {};
			begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
			EXIT_NOT_IMPLEMENTED(buffer.begin(&begin_info) != vk::Result::eSuccess);
		}
		return buffer;
	};
	for (;;) {
		auto chunk = m_chunk_queue.Pop();
		if (chunk == nullptr) {
			break;
		}
		CommandStream::Replay(*chunk, acquire, [&](const CommandStream::Entry& entry) {
			EXIT_IF(entry.kind != CommandStream::RecordKind::Submit);
			SubmitRecorded(buffer, buffer_index,
			               *static_cast<const RecordedSubmit*>(entry.Payload()));
		});
		m_chunk_queue.Recycle(std::move(chunk));
	}
	// A scheduler only stops after its last submission.
	EXIT_IF(buffer != nullptr);
}

void CommandScheduler::SubmitRecorded(vk::CommandBuffer& buffer, size_t buffer_index,
                                      const RecordedSubmit& recorded) {
	if (buffer != nullptr) {
		EXIT_NOT_IMPLEMENTED(buffer.end() != vk::Result::eSuccess);
	}
	// A submission without commands still signals its tick.
	auto submit = recorded.info;
	QueueSubmit(buffer, submit, recorded.tick, recorded);
	if (buffer != nullptr) {
		m_command_pool.Retire(buffer_index, recorded.tick);
		buffer = nullptr;
	}
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
