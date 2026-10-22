#include "common/sampler.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/cpuAffinity.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/timer.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/timeline.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
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

// KYTY_DRAW_FLUSH_INTERVAL=N overrides CompleteDraw()'s periodic non-blocking flush interval.
// Defaults to 128 draws (about 46 submissions a frame in GTA V; 16 meant about 370, whose
// submission boundaries cost the host GPU ~3 ms a frame and the recording thread ~4 ms). Setting
// it to 0 disables the periodic flush; submissions then still follow end-of-pipe events.
uint32_t DrawFlushInterval() {
	static const uint32_t interval = [] {
		const char* v = std::getenv("KYTY_DRAW_FLUSH_INTERVAL");
		if (v == nullptr) {
			return 128u;
		}
		return static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
	}();
	return interval;
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

vk::CommandBuffer CommandScheduler::CommandPool::Commit(uint64_t tick) {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search = [this, &gpu_tick, tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = tick;
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
		m_ticks[*found] = tick;
	}

	m_hint = (*found + 1) % m_ticks.size();
	return *found;
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

// KYTY_GPU_TIME=1 (ported from idxtri/wolverine-perf 4f42b798): every command buffer gets a
// timestamp at its start and end. When its tick completes, the GPU time it covered (overlap with
// the previous command buffer removed) is added up, and every 5 s the log reports how busy the GPU
// was: whether the GPU or the GPU thread limits the frame rate. Begin/End run on the thread that
// records (the recording thread when KYTY_RECORD_THREAD is on), Collect on the GPU thread.
struct CommandScheduler::GpuTimer {
	static constexpr uint32_t Slots = 4096;

	explicit GpuTimer(GraphicContext& graphics): graphics(graphics) {
		if (!Enabled()) return;
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = Slots * 2;
		if (graphics.device.createQueryPool(&info, nullptr, &pool) != vk::Result::eSuccess) {
			pool = nullptr;
		}
		period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	}
	~GpuTimer() {
		if (pool) graphics.device.destroyQueryPool(pool, nullptr);
	}
	KYTY_CLASS_NO_COPY(GpuTimer);

	static bool Enabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_GPU_TIME");
			return value != nullptr && value[0] != '0';
		}();
		return enabled;
	}

	// The recording thread, a new command buffer: its slot, or UINT32_MAX.
	uint32_t Begin(vk::CommandBuffer command) {
		if (!pool) return UINT32_MAX;
		{
			std::lock_guard lock(mutex);
			if (pending.size() + open >= Slots) return UINT32_MAX;
			open++;
		}
		const auto slot = next_slot++ % Slots;
		command.resetQueryPool(pool, slot * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, pool, slot * 2);
		return slot;
	}

	// The recording thread, before the command buffer ends.
	void End(vk::CommandBuffer command, uint32_t slot) {
		if (slot != UINT32_MAX) {
			command.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, pool, slot * 2 + 1);
		}
	}

	void Submitted(uint64_t tick, uint32_t slot) {
		if (slot == UINT32_MAX) return;
		std::lock_guard lock(mutex);
		open--;
		pending.push_back({tick, slot});
	}

	template <typename IsFree>
	void Collect(IsFree&& is_free) {
		if (!pool) return;
		std::lock_guard lock(mutex);
		while (!pending.empty() && is_free(pending.front().tick)) {
			const auto              slot = pending.front().slot;
			std::array<uint64_t, 2> stamps {};
			if (graphics.device.getQueryPoolResults(pool, slot * 2, 2, sizeof(stamps), stamps.data(),
			                                        sizeof(uint64_t), vk::QueryResultFlagBits::e64) ==
			        vk::Result::eSuccess &&
			    stamps[1] >= stamps[0]) {
				const auto begin = std::max(stamps[0], last_end);
				if (stamps[1] > begin) {
					busy_ns += static_cast<double>(stamps[1] - begin) * period_ns;
				}
				last_end = std::max(last_end, stamps[1]);
				buffers++;
			}
			pending.pop_front();
		}
		const auto now     = std::chrono::steady_clock::now();
		const auto elapsed = std::chrono::duration<double>(now - report).count();
		if (elapsed >= 5.0) {
			if (buffers != 0) {
				const double busy_ms_per_s = busy_ns / 1e6 / elapsed;
				std::fprintf(stderr, "[gpu-time] %.1f s: GPU busy %.1f ms/s (%.0f%%), %llu command buffers\n",
				             elapsed, busy_ms_per_s, busy_ms_per_s / 10.0,
				             static_cast<unsigned long long>(buffers));
				std::fflush(stderr);
			}
			busy_ns = 0;
			buffers = 0;
			report  = now;
		}
	}

	struct Pending {
		uint64_t tick = 0;
		uint32_t slot = 0;
	};

	GraphicContext&                       graphics;
	vk::QueryPool                         pool      = nullptr;
	float                                 period_ns = 1.0f;
	uint32_t                              next_slot = 0; // recording thread only
	uint32_t                              open      = 0;
	std::mutex                            mutex;
	std::deque<Pending>                   pending;
	uint64_t                              last_end = 0;
	double                                busy_ns  = 0;
	uint64_t                              buffers  = 0;
	std::chrono::steady_clock::time_point report   = std::chrono::steady_clock::now();
};

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_gpu_timer(std::make_unique<GpuTimer>(graphics)), m_master(graphics), m_context(context),
      m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {}

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
	// Every tick was waited on above, so the queue and recording threads have nothing left to
	// submit.
	StopRecordingThread();
	StopSubmitThread();
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

void CommandScheduler::EndRendering(std::source_location site) {
	if (Active() && !m_command.IsInvalid()) {
		Current().EndRendering(site);
	}
}

static thread_local const char* t_submit_reason = nullptr;

void SetSubmitReason(const char* reason) {
	t_submit_reason = reason;
}

// Sets the reason unless the caller already named one.
void SetSubmitReasonIfUnset(const char* reason) {
	if (t_submit_reason == nullptr) {
		t_submit_reason = reason;
	}
}
static void DefaultSubmitReason(const char* reason) {
	SetSubmitReasonIfUnset(reason);
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::CompleteReleaseMemWrite() {
	constexpr uint32_t WritesPerSubmission = 32;
	if (++m_recorded_release_mem_writes < WritesPerSubmission) {
		return;
	}
	CheckActive();
	DefaultSubmitReason("eop writes");
	Flush();
}

void CommandScheduler::CompleteReleaseMemInterrupt() {
	// Deliberately smaller than CompleteReleaseMemWrite's 32: this event has already been queued
	// for guest delivery once its tick completes (see CommandProcessor::TriggerEopEventAtEndOfPipe ->
	// DeferPriorityOperation), and a guest thread may be blocked waiting on it via an event queue.
	// Batching still defers only the vkQueueSubmit -- the event fires once that (now slightly
	// larger) submission's tick completes, same as before, just a handful of RELEASE_MEM events
	// later instead of immediately.
	constexpr uint32_t InterruptsPerSubmission = 8;
	if (++m_recorded_release_mem_interrupts < InterruptsPerSubmission) {
		return;
	}
	CheckActive();
	DefaultSubmitReason("eop interrupts");
	Flush();
}

void CommandScheduler::CompleteDraw() {
	const auto interval = DrawFlushInterval();
	if (interval == 0u || ++m_recorded_draws < interval) {
		return;
	}
	CheckActive();
	DefaultSubmitReason("draw interval");
	Flush();
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait() {
	KYTY_PROFILER_FUNCTION();
	DefaultSubmitReason("flush and wait");
	const auto tick = Submit();
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	KYTY_PROFILER_FUNCTION();
	CheckActive();
	if (!m_command.IsInvalid()) {
		DefaultSubmitReason("finish");
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

// How long the GPU thread spends drained on the host GPU, and how often. A drain that
// submits the current tick first (a CPU read of GPU-owned memory mid-frame) is counted apart
// from one that waits on already submitted work.
namespace {
struct WaitStats {
	uint64_t count       = 0;
	uint64_t drain_count = 0;
	double   total_s     = 0.0;
	double   drain_s     = 0.0;
	double   max_s       = 0.0;
};
WaitStats g_wait_stats;

void RecordWait(double seconds, bool drained) {
	if (!GuestGpu::IsGpuThread()) {
		return;
	}
	auto& stats = g_wait_stats;
	stats.count++;
	stats.total_s += seconds;
	stats.max_s = std::max(stats.max_s, seconds);
	if (drained) {
		stats.drain_count++;
		stats.drain_s += seconds;
	}
	if (stats.count % 256 == 0) {
		LOGF("GpuWaits: count=%" PRIu64 " total=%.1fms max=%.1fms drains=%" PRIu64
		     " drain_total=%.1fms\n",
		     stats.count, stats.total_s * 1000.0, stats.max_s * 1000.0, stats.drain_count,
		     stats.drain_s * 1000.0);
		stats = {};
	}
}
} // namespace

void CommandScheduler::Wait(uint64_t tick) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(tick > CurrentTick());
	// KYTY_WAIT_STACKS=1: a host stack for every 23rd wait that has to block (up to 60), to name
	// the paths that make the command thread wait for the GPU.
	if (static const bool stacks = std::getenv("KYTY_WAIT_STACKS") != nullptr;
	    stacks && !m_master.IsFree(tick)) {
		static std::atomic<uint32_t> waits {0};
		const auto                   n = waits.fetch_add(1);
		if (n % 23 == 0 && n / 23 < 60) {
			std::fprintf(stderr, "[wait-stack] #%u tick=%llu current=%llu\n%s\n", n,
			             static_cast<unsigned long long>(tick),
			             static_cast<unsigned long long>(CurrentTick()),
			             Common::HostBacktrace().c_str());
			std::fflush(stderr);
		}
	}
	Common::Timer timer;
	timer.Start();
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		{
			KYTY_PROFILER_BLOCK("CommandScheduler::Wait (forced submit-then-wait)");
			// KYTY_DRAIN_STACKS=1: a host stack for every 37th forced drain (up to 48), to name
			// the paths that make the command thread wait for the whole GPU.
			if (static const bool stacks = std::getenv("KYTY_DRAIN_STACKS") != nullptr; stacks) {
				static std::atomic<uint32_t> drains {0};
				const auto                   n = drains.fetch_add(1);
				if (n % 37 == 0 && n / 37 < 48) {
					std::fprintf(stderr, "[drain] #%u\n%s\n", n, Common::HostBacktrace().c_str());
					std::fflush(stderr);
				}
			}
			DefaultSubmitReason("wait for the open tick");
			const auto submitted_tick = Submit();
			EXIT_IF(submitted_tick != tick);
			Timeline::Mark("drain-begin", tick);
			m_master.Wait(tick);
			Timeline::Mark("drain-end", tick);
			BeginNext();
		}
		RecordWait(timer.GetTimeS(), true);
	} else {
		Timeline::Mark("wait-begin", tick);
		m_master.Wait(tick);
		Timeline::Mark("wait-end", tick);
		RecordWait(timer.GetTimeS(), false);
	}
}

void CommandScheduler::PopPendingOperations() {
	// Every draw and dispatch comes here: only query the timeline semaphore (a driver call) when
	// the oldest operation is not already known to be free.
	// Without the lock: nothing pending, or the oldest operation is not done and the timeline
	// was queried a few calls ago. Pending operations only free resources and complete host
	// work, so finding them some microseconds later costs nothing, while a query per draw is a
	// driver call per draw.
	if (const auto front = m_pending_front_tick.load(std::memory_order_acquire);
	    front == UINT64_MAX) {
		return;
	} else if (!m_master.IsFree(front)) {
		constexpr uint32_t RefreshInterval = 8;
		// Draws come from the execution thread: no locked instruction needed (a racing caller at
		// worst queries once more or once less).
		if (const auto skips = m_pending_skips.load(std::memory_order_relaxed) + 1;
		    skips < RefreshInterval) {
			m_pending_skips.store(skips, std::memory_order_relaxed);
			return;
		}
		m_pending_skips.store(0, std::memory_order_relaxed);
		m_master.Refresh();
		if (!m_master.IsFree(front)) {
			return;
		}
	}
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
			m_pending_front_tick.store(m_pending_operations.empty()
			                               ? UINT64_MAX
			                               : m_pending_operations.front().tick,
			                           std::memory_order_release);
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::PollPendingOperations() {
	const auto front = m_pending_front_tick.load(std::memory_order_acquire);
	if (front == UINT64_MAX) {
		return;
	}
	if (!m_master.IsFree(front)) {
		m_master.Refresh();
		if (!m_master.IsFree(front)) {
			return;
		}
	}
	PopPendingOperations();
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation, uint64_t address,
                                      uint64_t size) {
	CheckActive();
	QueueOperation(TrackGuestOperation(std::move(operation), address, size), false);
}

void CommandScheduler::DeferRelease(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), false);
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation,
                                              uint64_t address, uint64_t size) {
	CheckActive();
	QueueOperation(TrackGuestOperation(std::move(operation), address, size), true);
}

void CommandScheduler::DeferHostOperation(Common::UniqueFunction<void>&& operation,
                                          bool                           priority) {
	CheckActive();
	QueueOperation(std::move(operation), priority);
}

Common::UniqueFunction<void>
CommandScheduler::TrackGuestOperation(Common::UniqueFunction<void>&& operation, uint64_t address,
                                      uint64_t size) {
	EXIT_IF(!operation);
	uint64_t id = 0;
	{
		std::lock_guard lock(m_guest_writes_mutex);
		id = ++m_next_guest_write;
		m_guest_writes.push_back(
		    {id, address, size > UINT64_MAX - address ? UINT64_MAX : address + size});
	}
	return [this, id, operation = std::move(operation)]() mutable {
		operation();
		std::lock_guard lock(m_guest_writes_mutex);
		// Operations mostly complete in the order they were queued.
		const auto it = std::find_if(m_guest_writes.begin(), m_guest_writes.end(),
		                             [id](const GuestWrite& write) { return write.id == id; });
		EXIT_IF(it == m_guest_writes.end());
		m_guest_writes.erase(it);
	};
}

bool CommandScheduler::HasPendingGuestOperations(uint64_t address, uint64_t size) {
	const auto      end = size > UINT64_MAX - address ? UINT64_MAX : address + size;
	std::lock_guard lock(m_guest_writes_mutex);
	return std::any_of(m_guest_writes.begin(), m_guest_writes.end(), [&](const GuestWrite& write) {
		return write.begin < end && address < write.end;
	});
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation, bool priority) {
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		auto& queue = priority ? m_priority_operations : m_pending_operations;
		queue.push({std::move(operation), CurrentTick()});
		if (!priority && m_pending_operations.size() == 1) {
			m_pending_front_tick.store(m_pending_operations.front().tick,
			                           std::memory_order_release);
		}
		lock.unlock();
		if (priority) {
			m_operation_available.notify_one();
		}
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

void CommandScheduler::QueuePriorityOperation(Common::UniqueFunction<void>&& operation) {
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
		Timeline::Mark("tick-done", operation.tick);
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

bool CommandScheduler::WaitFor(uint64_t tick, uint64_t timeout_ns) {
	return IsFree(tick) || (tick < CurrentTick() && m_master.WaitFor(tick, timeout_ns));
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	EXIT_IF(m_command.IsInvalid());
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	m_command.m_open = true;
	// Ticks are allocated in submit order, so this command buffer will be submitted with the
	// current one.
	const auto tick = m_master.CurrentTick();
	m_gpu_timer->Collect([this](uint64_t done) { return m_master.IsFree(done); });
	if (m_threaded) {
		m_command.m_buffer = nullptr;
		Record([this, tick](vk::CommandBuffer) {
			m_worker_buffer = m_command_pool.Commit(tick);
			vk::CommandBufferBeginInfo begin_info {};
			begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
			EXIT_NOT_IMPLEMENTED(m_worker_buffer.begin(&begin_info) != vk::Result::eSuccess);
			m_timer_slot = m_gpu_timer->Begin(m_worker_buffer);
		});
		m_dynamic_state.Invalidate();
		return m_command;
	}
	m_command.m_buffer = m_command_pool.Commit(tick);
	m_command.Begin();
	m_timer_slot = m_gpu_timer->Begin(m_command.Handle());
	m_dynamic_state.Invalidate();
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(m_command.IsInvalid());
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);
	if (m_submit_hook != nullptr) {
		m_submit_hook(m_submit_hook_data, m_command);
	}

	if (!m_threaded) {
		m_gpu_timer->End(m_command.Handle(), m_timer_slot);
	}
	m_command.End();
	EXIT_IF(m_graphics.queue == nullptr);
	Profiler::Add(Profiler::Counter::CommandBuffersSubmitted);
	// KYTY_SUBMIT_STACKS=1: a host stack for every 97th submission (up to 40), to name what splits
	// the frame into so many submissions. KYTY_FPS_LOG: submissions per second every 5 s.
	if (static const bool stacks = std::getenv("KYTY_SUBMIT_STACKS") != nullptr; stacks) {
		static std::atomic<uint32_t> submits {0};
		const auto                   n = submits.fetch_add(1);
		if (n % 97 == 0 && n / 97 < 40) {
			std::fprintf(stderr, "[submit-stack] #%u\n%s\n", n, Common::HostBacktrace().c_str());
			std::fflush(stderr);
		}
	}
	if (static const bool fps_log = std::getenv("KYTY_FPS_LOG") != nullptr; fps_log) {
		static std::atomic<uint64_t> count {0};
		static std::atomic<int64_t>  window_start {0};
		const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
		                     std::chrono::steady_clock::now().time_since_epoch())
		                     .count();
		auto start = window_start.load(std::memory_order_relaxed);
		if (start == 0) {
			window_start.compare_exchange_strong(start, now);
			start = now;
		}
		const auto n = count.fetch_add(1) + 1;
		static std::mutex                                  reasons_mutex;
		static std::vector<std::pair<const char*, uint64_t>> reasons;
		{
			std::lock_guard lock(reasons_mutex);
			const char*     reason = t_submit_reason != nullptr ? t_submit_reason : "untagged";
			auto it = std::find_if(reasons.begin(), reasons.end(), [reason](const auto& r) { return r.first == reason; });
			if (it == reasons.end()) {
				reasons.emplace_back(reason, 1);
			} else {
				it->second++;
			}
		}
		if (now - start >= 5000 && window_start.compare_exchange_strong(start, now)) {
			count.store(0);
			std::string text;
			{
				std::lock_guard lock(reasons_mutex);
				std::sort(reasons.begin(), reasons.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
				for (const auto& [reason, k]: reasons) {
					const char* base = std::strrchr(reason, '/');
					const char* base2 = std::strrchr(reason, '\\');
					base = base2 != nullptr && (base == nullptr || base2 > base) ? base2 : base;
					text += fmt::format(" {}={:.0f}", base != nullptr ? base + 1 : reason,
					                    static_cast<double>(k) * 1000.0 / static_cast<double>(now - start));
				}
				reasons.clear();
			}
			std::fprintf(stderr, "[submits] %.0f per second:%s\n",
			             static_cast<double>(n) * 1000.0 / static_cast<double>(now - start), text.c_str());
		}
	}
	t_submit_reason = nullptr;

	SubmitJob job {
	    .buffer       = m_command.m_buffer,
	    .submit       = submit,
	    .debug_op     = m_command.m_debug_op,
	    .debug_submit = m_command.m_debug_submit_id,
	    .debug_arg0   = m_command.m_debug_arg0,
	    .debug_arg1   = m_command.m_debug_arg1,
	    .debug_arg2   = m_command.m_debug_arg2,
	    .debug_arg3   = m_command.m_debug_arg3,
	    .debug_arg4   = m_command.m_debug_arg4,
	};
	// The command buffer reads the stream data the execution thread's copies write.
	job.host_copies                   = m_host_copies != nullptr ? m_host_copies->Flush() : 0;
	m_command.m_buffer                = nullptr;
	m_command.m_open                  = false;
	m_recorded_release_mem_writes     = 0;
	m_recorded_release_mem_interrupts = 0;
	m_recorded_draws                  = 0;

	if (m_threaded) {
		// The tick is allocated now, in order; the recording thread ends the command buffer and
		// submits it once it has recorded everything queued before.
		job.tick = m_master.NextTick();
		job.submit.AddSignal(m_master.Handle(), job.tick);
		Record([this, job](vk::CommandBuffer command) mutable {
			m_gpu_timer->End(command, m_timer_slot);
			EXIT_NOT_IMPLEMENTED(command.end() != vk::Result::eSuccess);
			job.buffer = command;
			QueueSubmit(job);
			m_gpu_timer->Submitted(job.tick, m_timer_slot);
			m_timer_slot    = UINT32_MAX;
			m_worker_buffer = nullptr;
		});
		DispatchChunk();
		ReportRecording();
		return job.tick;
	}
	if (!m_async_submit) {
		QueueSubmit(job);
		m_gpu_timer->Submitted(job.tick, m_timer_slot);
		m_timer_slot = UINT32_MAX;
		return job.tick;
	}
	{
		// Ticks are allocated in queue order and the queue thread submits in that order, so the
		// timeline is signaled in order.
		std::lock_guard lock(m_submit_mutex);
		job.tick = m_master.NextTick();
		job.submit.AddSignal(m_master.Handle(), job.tick);
		m_submit_jobs.push_back(job);
	}
	m_gpu_timer->Submitted(job.tick, m_timer_slot);
	m_timer_slot = UINT32_MAX;
	m_submit_available.notify_one();
	return job.tick;
}

void CommandScheduler::EnableHostCopies() {
	EXIT_IF(m_host_copies != nullptr || !m_threaded);
	m_host_copies = std::make_unique<HostCopyQueue>(
	    [] { Config::ConfigureGpuStageThread(Config::GpuStageThread::HostCopy); });
}

void CommandScheduler::QueueSubmit(SubmitJob& job) {
	if (m_host_copies != nullptr && job.host_copies != 0) {
		m_host_copies->WaitFor(job.host_copies);
	}
	auto&      graphics = m_graphics;
	vk::Result result;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		if (!m_async_submit) {
			job.tick = m_master.NextTick();
			job.submit.AddSignal(m_master.Handle(), job.tick);
		}
		const auto& submit = job.submit;

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
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &job.buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.queue.submit(1, &submit_info, nullptr);
	}
	Timeline::Mark("vk-submit", job.tick);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result == vk::Result::eErrorDeviceLost) {
		DumpDeviceLossDiagnostics(graphics);
	}
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, job.tick, job.debug_op, job.debug_submit,
		                  job.debug_arg0, job.debug_arg1, job.debug_arg2, job.debug_arg3,
		                  job.debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandScheduler::EnableAsyncSubmit() {
	// The owner enables it while constructing, before any other thread can submit.
	EXIT_IF(m_async_submit);
	m_async_submit  = true;
	m_submit_thread = std::jthread([this](std::stop_token stop) { SubmitThread(stop); });
}

void CommandScheduler::SubmitThread(std::stop_token stop) {
	KYTY_PROFILER_THREAD("GpuQueueSubmit");
	for (;;) {
		SubmitJob job;
		{
			std::unique_lock lock(m_submit_mutex);
			// A stop request still submits the queued jobs: their ticks may already be waited on.
			if (!m_submit_available.wait(lock, stop, [this] { return !m_submit_jobs.empty(); })) {
				return;
			}
			job = m_submit_jobs.front();
			m_submit_jobs.pop_front();
		}
		QueueSubmit(job);
	}
}

void CommandScheduler::StopSubmitThread() {
	if (m_submit_thread.joinable()) {
		m_submit_thread.request_stop();
		m_submit_thread.join();
	}
	std::lock_guard lock(m_submit_mutex);
	EXIT_IF(!m_submit_jobs.empty());
}

void CommandScheduler::EnableRecordingThread() {
	// The owner enables it while constructing, before the first command buffer.
	EXIT_IF(m_record_thread.joinable() || m_async_submit || !m_command.IsInvalid());
	// QueueSubmit then takes the tick Submit() allocated, as for the queue thread.
	m_async_submit  = true;
	m_threaded      = true;
	m_record_queue  = std::make_unique<ChunkQueue>(64);
	m_free_chunks   = std::make_unique<ChunkQueue>(64);
	m_record_thread = std::jthread([this](std::stop_token stop) { RecordingThread(stop); });
}

void* CommandScheduler::AllocateInNewChunk(size_t size, size_t alignment) {
	EXIT_IF(!m_threaded || size > Chunk::Size);
	const auto take = [this] {
		// An executed chunk the recording thread handed back, else a new one (not zeroed: its
		// memory is written before it is read).
		if (auto recycled = m_free_chunks->TryPop()) {
			return std::move(*recycled);
		}
		return std::unique_ptr<Chunk>(new Chunk);
	};
	if (m_chunk == nullptr) {
		m_chunk = take();
	}
	auto offset = (m_chunk->used + alignment - 1) & ~(alignment - 1);
	if (offset + size > Chunk::Size) {
		// Also a chunk that holds only Stash() copies: the next command reads them, and the
		// recording thread keeps the last chunks alive for it (dropping it here freed them).
		if (m_chunk->used != 0) {
			m_chunks_dispatched++;
			m_record_queue->Push(std::move(m_chunk));
		}
		m_chunk = take();
		offset  = 0;
	}
	m_chunk->used = offset + size;
	return m_chunk->storage + offset;
}

void CommandScheduler::DispatchChunk() {
	if (m_chunk == nullptr || m_chunk->first == nullptr) {
		return;
	}
	m_chunks_dispatched++;
	m_record_queue->Push(std::move(m_chunk));
}

vk::CommandBuffer CommandScheduler::DrainRecording() {
	if (!m_threaded) {
		return m_command.m_buffer;
	}
	KYTY_PROFILER_FUNCTION();
	DispatchChunk();
	m_drains++;
	Profiler::Add(Profiler::Counter::RecordingDrains);
	// KYTY_RECORD_DRAIN_STACKS=1: which raw Handle() sites make the GPU thread wait (a host stack
	// for every 4,999th drain, up to 64).
	static const bool stacks = std::getenv("KYTY_RECORD_DRAIN_STACKS") != nullptr;
	if (stacks && m_drains % 4999 == 0 && m_drains / 4999 <= 64) {
		LOGF("Recording thread drain %" PRIu64 ":\n%s", m_drains,
		     Common::HostBacktrace().c_str());
	}
	m_chunks_executed.WaitFor(m_chunks_dispatched);
	return m_worker_buffer;
}

void CommandScheduler::RecordingThread(std::stop_token /*stop*/) {
	KYTY_PROFILER_THREAD("GpuRecord");
	Common::Sampler::RegisterCurrentThread("record");
	// Its own physical core, like the other stages: the thread now spins between chunks, and on
	// the SMT sibling of the execution thread's core it would slow it down.
	Config::ConfigureGpuStageThread(Config::GpuStageThread::Recording);
	// A command's stashed arrays may sit in up to two chunks before its own: executed chunks are
	// recycled only after the next two have run.
	std::deque<std::unique_ptr<Chunk>> retained;
	uint64_t                           executed = 0;
	// Pop() returns nothing once the ring was stopped and holds no more chunks: stopping still
	// runs the queued chunks, whose ticks may already be waited on.
	while (auto popped = m_record_queue->Pop()) {
		auto chunk = std::move(*popped);
		for (auto* command = chunk->first; command != nullptr;) {
			auto* next = command->next;
			command->Execute(m_worker_buffer);
			command->~RecordedCommand();
			command = next;
		}
		chunk->first = nullptr;
		chunk->last  = nullptr;
		chunk->used  = 0;
		retained.push_back(std::move(chunk));
		if (retained.size() > 2) {
			// A full ring of free chunks frees this one instead.
			auto recycled = std::move(retained.front());
			retained.pop_front();
			(void)m_free_chunks->TryPush(recycled);
		}
		m_chunks_executed.Advance(++executed);
	}
}

// Every 5 s: command buffers submitted, chunks handed over, and drains (raw Handle() uses, which
// make the GPU thread wait for the recording thread).
void CommandScheduler::ReportRecording() {
	m_record_report_buffers++;
	const auto now = std::chrono::steady_clock::now();
	if (now - m_record_report_time < std::chrono::seconds(5)) {
		return;
	}
	const uint64_t chunks = m_chunks_dispatched;
	const uint64_t drains = m_drains;
	LOGF("Recording thread (5 s): %" PRIu64 " command buffers, %" PRIu64 " chunks, %" PRIu64
	     " drains\n",
	     m_record_report_buffers, chunks - m_record_report_chunks, drains - m_record_report_drains);
	m_record_report_buffers = 0;
	m_record_report_chunks  = chunks;
	m_record_report_drains  = drains;
	m_record_report_time    = now;
}

void CommandScheduler::StopRecordingThread() {
	if (!m_record_thread.joinable()) {
		return;
	}
	(void)DrainRecording();
	m_record_queue->Stop();
	m_record_thread.join();
	LOGF("Recording thread: %" PRIu64 " chunks, %" PRIu64 " drains\n", m_chunks_dispatched,
	     m_drains);
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
