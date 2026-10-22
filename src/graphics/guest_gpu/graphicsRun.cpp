#include "common/hackNotice.h"
#include "graphics/guest_gpu/graphicsRun.h"

#include "common/assert.h"
#include "common/cpuAffinity.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/drawPrepScanner.h"
#include "graphics/host_gpu/renderer/drawPrep.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "common/sampler.h"
#include "graphics/host_gpu/renderer/barrierSink.h"
#include "graphics/host_gpu/renderer/dispatchRun.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/timeline.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fmt/format.h>
#include <map>
#include <string>
#include <memory>
#include <mutex>
#include <semaphore>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

static std::atomic<uint64_t> g_cmd_from_send {0};
static std::atomic<uint64_t> g_cmd_from_try {0};
static std::atomic<uint64_t> g_cmd_from_sync {0};

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static const uint32_t*                g_run_expected_next = nullptr; // KYTY_DISPATCH_RUNS
static std::atomic<uint64_t> g_compute_submissions {0};
static std::atomic<uint64_t> g_compute_submission_dw {0};
static thread_local bool              g_gpu_thread        = false;
// The command processor thread of a pipelined GPU.
static thread_local bool              g_cp_thread         = false;
static thread_local GuestGpu*         g_gpu_state         = nullptr;

// The Tracy source locations of the PM4 packets that get a command zone (Profiler::CommandZone):
// draws, dispatches, copies, memory writes and synchronization; not register writes and NOPs.
static const tracy::SourceLocationData* Pm4ZoneLocation(uint32_t opcode, uint32_t r) {
	struct Table {
		std::array<tracy::SourceLocationData, 256>        ops {};
		std::array<tracy::SourceLocationData, Pm4::R_NUM> custom {};
	};
	static const Table table = [] {
		Table      result;
		const auto op = [&](uint32_t code, const char* name) {
			result.ops[code] = {name, "CommandProcessor::ProcessPm4", __FILE__, 0, 0};
		};
		const auto custom = [&](uint32_t code, const char* name) {
			result.custom[code] = {name, "CommandProcessor::ProcessPm4", __FILE__, 0, 0};
		};
		op(Pm4::IT_DISPATCH_DIRECT, "PM4 DISPATCH_DIRECT");
		op(Pm4::IT_DISPATCH_INDIRECT, "PM4 DISPATCH_INDIRECT");
		op(Pm4::IT_DRAW_INDIRECT, "PM4 DRAW_INDIRECT");
		op(Pm4::IT_DRAW_INDEX_INDIRECT, "PM4 DRAW_INDEX_INDIRECT");
		op(Pm4::IT_DRAW_INDEX_2, "PM4 DRAW_INDEX_2");
		op(Pm4::IT_DRAW_INDIRECT_MULTI, "PM4 DRAW_INDIRECT_MULTI");
		op(Pm4::IT_DRAW_INDEX_AUTO, "PM4 DRAW_INDEX_AUTO");
		op(Pm4::IT_DRAW_INDEX_OFFSET_2, "PM4 DRAW_INDEX_OFFSET_2");
		op(Pm4::IT_DRAW_INDEX_INDIRECT_MULTI, "PM4 DRAW_INDEX_INDIRECT_MULTI");
		op(Pm4::IT_DISPATCH_DRAW, "PM4 DISPATCH_DRAW");
		op(Pm4::IT_WRITE_DATA, "PM4 WRITE_DATA");
		op(Pm4::IT_MEM_SEMAPHORE, "PM4 MEM_SEMAPHORE");
		op(Pm4::IT_WAIT_REG_MEM, "PM4 WAIT_REG_MEM");
		op(Pm4::IT_WAIT_REG_MEM_64, "PM4 WAIT_REG_MEM_64");
		op(Pm4::IT_COPY_DATA, "PM4 COPY_DATA");
		op(Pm4::IT_CP_DMA, "PM4 CP_DMA");
		op(Pm4::IT_DMA_DATA, "PM4 DMA_DATA");
		op(Pm4::IT_SURFACE_SYNC, "PM4 SURFACE_SYNC");
		op(Pm4::IT_EVENT_WRITE, "PM4 EVENT_WRITE");
		op(Pm4::IT_EVENT_WRITE_EOP, "PM4 EVENT_WRITE_EOP");
		op(Pm4::IT_EVENT_WRITE_EOS, "PM4 EVENT_WRITE_EOS");
		op(Pm4::IT_RELEASE_MEM, "PM4 RELEASE_MEM");
		op(Pm4::IT_ACQUIRE_MEM, "PM4 ACQUIRE_MEM");
		op(Pm4::IT_DUMP_CONST_RAM, "PM4 DUMP_CONST_RAM");
		custom(Pm4::R_DRAW_RESET, "PM4 NOP DRAW_RESET");
		custom(Pm4::R_WAIT_FLIP_DONE, "PM4 NOP WAIT_FLIP_DONE");
		custom(Pm4::R_DISPATCH_RESET, "PM4 NOP DISPATCH_RESET");
		custom(Pm4::R_WRITE_DATA, "PM4 NOP WRITE_DATA");
		custom(Pm4::R_FLIP, "PM4 NOP FLIP");
		custom(Pm4::R_RELEASE_MEM, "PM4 NOP RELEASE_MEM");
		custom(Pm4::R_DMA_DATA, "PM4 NOP DMA_DATA");
		custom(Pm4::R_CONTEXT_STATE, "PM4 NOP CONTEXT_STATE");
		return result;
	}();
	const auto& location = opcode == Pm4::IT_NOP ? table.custom[r & (Pm4::R_NUM - 1u)]
	                                             : table.ops[opcode & 0xffu];
	return location.name != nullptr ? &location : nullptr;
}

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::LogDirection::Silent;
}

// KYTY_LOG_INDIRECT_DRAWS=<n>: log the arguments of the first n indirect draws, including
// multi-draws whose GPU-written count is zero. A diagnostic for draws that GPU work generation
// (culling passes) should have filled.
// KYTY_GPU_INDIRECT_DRAWS=0 reads every DRAW_INDIRECT's arguments on the CPU, as before.
static bool GpuIndirectDraws() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_INDIRECT_DRAWS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

static bool LogIndirectDraw() {
	static const uint32_t budget = [] {
		const char* value = std::getenv("KYTY_LOG_INDIRECT_DRAWS");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	if (budget == 0) {
		return false;
	}
	// With KYTY_LOG_DRAWS_TRIGGER=<file>, start once that file exists (checked every 256 draws).
	static const char*           trigger   = std::getenv("KYTY_LOG_DRAWS_TRIGGER");
	static std::atomic_bool      triggered = trigger == nullptr;
	static std::atomic<uint32_t> polls {0};
	if (!triggered) {
		if ((polls.fetch_add(1, std::memory_order_relaxed) & 255u) != 0u) {
			return false;
		}
		FILE* file = std::fopen(trigger, "rb");
		if (file == nullptr) {
			return false;
		}
		std::fclose(file);
		triggered = true;
	}
	static std::atomic<uint32_t> used {0};
	return used.fetch_add(1, std::memory_order_relaxed) < budget;
}

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GraphicsInitJmpTables();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_gfx_cp->Reset();
	if (DrawPrep::Enabled()) {
		m_draw_prep = std::make_unique<DrawPrepScanner>(renderer);
		LOGF("DrawPrep: the scanner prepares draws ahead of the GPU thread (KYTY_DRAW_PREP=0 turns it "
		     "off; verify %d)\n",
		     DrawPrep::VerifyMode());
	}
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_epoch.fetch_add(1, std::memory_order_release);
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_draw_prep.reset();
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	g_cmd_from_send.fetch_add(1, std::memory_order_relaxed);
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_epoch.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::Wake() {
	Common::LockGuard lock(m_queue_mutex);
	m_wake_pending = true;
	m_work_available.Signal();
}

bool GuestGpu::TrySendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return true;
	}
	g_cmd_from_try.fetch_add(1, std::memory_order_relaxed);
	Common::LockGuard lock(m_queue_mutex);
	if (!m_accepting) {
		return false;
	}
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
	return true;
}

namespace {
std::atomic<bool> g_labels_deferred {false};
} // namespace

bool GuestGpu::LabelsDeferred() noexcept {
	return g_labels_deferred.load(std::memory_order_relaxed);
}

bool GuestGpu::LabelsAtCompletion() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LABELS_AT_COMPLETION");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void GuestGpu::DeferLabelWrite(uint64_t address, uint64_t value, uint32_t size) {
	EXIT_IF(!IsGpuThread() || (size != 4 && size != 8));
	g_labels_deferred.store(true, std::memory_order_relaxed);
	const auto sequence        = ++m_label_sequence;
	m_pending_labels[address] = {value, size, sequence};
	// The completion runs on the scheduler's priority thread, which must not fault on guest
	// memory the GPU wrote to; the write itself goes back to the GPU thread.
	m_renderer.GetCommandScheduler().DeferPriorityOperation([this, address, value, size,
	                                                         sequence] {
		(void)TrySendCommand([this, address, value, size, sequence] {
			Timeline::Mark("label", address, value);
			std::memcpy(reinterpret_cast<void*>(address), &value, size);
			const auto pending = m_pending_labels.find(address);
			if (pending != m_pending_labels.end() && pending->second.sequence == sequence) {
				m_pending_labels.erase(pending);
			}
		});
	});
}

template <typename T>
T GuestGpu::ReadLabel(const volatile T* address) const {
	if (!m_pending_labels.empty()) {
		const auto pending = m_pending_labels.find(reinterpret_cast<uint64_t>(address));
		if (pending != m_pending_labels.end() && pending->second.size >= sizeof(T)) {
			return static_cast<T>(pending->second.value);
		}
	}
	return *address;
}

template uint32_t GuestGpu::ReadLabel<uint32_t>(const volatile uint32_t*) const;
template uint64_t GuestGpu::ReadLabel<uint64_t>(const volatile uint64_t*) const;

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread() && !g_cp_thread);
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		if (m_pipelined) {
			// Host commands run on the execution thread, after the operations emitted so far.
			m_gfx_cp->EmitHostCommand(std::move(command));
		} else {
			command();
		}
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	g_cmd_from_sync.fetch_add(1, std::memory_order_relaxed);
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

static bool CopySubmissions() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_COPY_SUBMISSIONS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void GuestGpu::Submit(std::span<const uint32_t> draw_commands,
                      std::span<const uint32_t> constant_commands) {
	if (draw_commands.empty()) {
		return;
	}
	Timeline::Mark("submit-gfx", draw_commands.size(), constant_commands.size());
	Submission submission;
	submission.type              = SubmissionType::Graphics;
	submission.queue_id          = 0;
	submission.commands          = draw_commands;
	submission.constant_commands = constant_commands;
	if (CopySubmissions()) {
		auto words = std::make_shared<std::vector<uint32_t>>();
		words->reserve(draw_commands.size() + constant_commands.size());
		words->insert(words->end(), draw_commands.begin(), draw_commands.end());
		words->insert(words->end(), constant_commands.begin(), constant_commands.end());
		submission.commands = std::span<const uint32_t>(words->data(), draw_commands.size());
		submission.constant_commands =
		    constant_commands.empty()
		        ? std::span<const uint32_t> {}
		        : std::span<const uint32_t>(words->data() + draw_commands.size(),
		                                    constant_commands.size());
		submission.owned_words = std::move(words);
	}
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Timeline::Mark("submit-cq", compute_queue, commands.size());
	Submission submission;
	submission.type     = SubmissionType::Compute;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	if (CopySubmissions()) {
		auto words = std::make_shared<std::vector<uint32_t>>(commands.begin(), commands.end());
		submission.commands    = std::span<const uint32_t>(words->data(), words->size());
		submission.owned_words = std::move(words);
	}
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Timeline::Mark("submit-flip", request_id);
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Enqueue(std::move(submission));
}

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	// Do not hold a queue lock while waiting: asynchronous work may be needed to
	// finish the preceding graphics frame. The first point returns immediately.
	m_suspend_point_ready->acquire();
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
		if (m_pipelined) {
			processor->AttachPipeline(this);
		}
	}
	return *processor;
}

uint64_t GuestGpu::EmitOperation(GpuOperation&& operation) {
	const auto sequence = ++m_emitted;
	operation.sequence  = sequence;
	m_operations.Push(std::move(operation));
	return sequence;
}

void GuestGpu::ForgetExecutedWrites() {
	// Executed writes are in guest memory now.
	const auto executed = m_executed.Load();
	while (m_pending_count != 0 && m_pending_writes[m_pending_first].sequence <= executed) {
		m_pending_first = (m_pending_first + 1) % PendingWriteCapacity;
		m_pending_count--;
	}
}

void GuestGpu::NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size,
                              const void* value) {
	ForgetExecutedWrites();
	if (sequence <= m_executed.Load()) {
		return;
	}
	if (m_pending_count == PendingWriteCapacity) {
		WaitForExecution(m_pending_writes[m_pending_first].sequence);
		ForgetExecutedWrites();
	}
	auto& write = m_pending_writes[(m_pending_first + m_pending_count) % PendingWriteCapacity];
	write       = {.sequence = sequence, .address = address, .size = size};
	if (value != nullptr && size <= sizeof(write.value)) {
		write.known = true;
		std::memcpy(write.value, value, size);
	}
	m_pending_count++;
}

void GuestGpu::ReadDecisionMemory(const volatile void* address, void* value, size_t size) {
	ForgetExecutedWrites();
	const auto begin = reinterpret_cast<uint64_t>(address);
	const auto end   = begin + size;
	// The newest pending write that overlaps the range decides; older ones execute before it.
	for (size_t i = m_pending_count; i-- > 0;) {
		const auto* write = &m_pending_writes[(m_pending_first + i) % PendingWriteCapacity];
		if (write->address >= end || write->address + write->size <= begin) {
			continue;
		}
		if (write->known && write->address <= begin && end <= write->address + write->size) {
			std::memcpy(value, write->value + (begin - write->address), size);
			return;
		}
		// The value is only known once the write executed; waiting acquires it.
		WaitForExecution(write->sequence);
		break;
	}
	CopyFromGuest(value, address, size);
}

void GuestGpu::ExecuteSync(Common::UniqueFunction<void>&& command) {
	m_gfx_cp->EmitHostCommand(std::move(command));
	WaitForExecution();
}

void GuestGpu::ExecutionThread() {
	KYTY_PROFILER_THREAD("Thread_GpuExecute");
	Config::ConfigureGpuStageThread(Config::GpuStageThread::Execution);
	g_gpu_thread = true;
	// Waking the command processor costs a fence: publish every operation, but only look for a
	// waiter now and then and before sleeping.
	constexpr uint32_t NotifyInterval = 16;
	uint32_t           unnotified     = 0;
	auto&              operations     = m_resolving ? m_resolved : m_operations;
	for (;;) {
		if (m_urgent_pending.load(std::memory_order_relaxed) != 0) {
			RunUrgentCommands();
		}
		auto operation = operations.TryPop();
		if (!operation) {
			m_executed.Notify();
			unnotified = 0;
			operation  = operations.Pop();
			if (!operation) {
				break;
			}
		}
		operation->processor->Execute(*operation);
		const auto sequence = operation->sequence;
		// Destroy the callback before the command processor may reuse what it references.
		operation.reset();
		m_executed.Publish(sequence);
		if (++unnotified >= NotifyInterval) {
			m_executed.Notify();
			unnotified = 0;
		}
	}
	m_executed.Notify();
	g_gpu_thread = false;
}

static bool IndirectDrawOffsets(IndirectDrawRegisters registers, ShaderIndirectOffsets& offsets);

void GuestGpu::ResolveThread() {
	KYTY_PROFILER_THREAD("Thread_GpuResolve");
	Config::ConfigureGpuStageThread(Config::GpuStageThread::Resolve);
	auto& executor = m_renderer.GetRenderExecutor();
	for (;;) {
		auto operation = m_operations.TryPop();
		if (!operation) {
			operation = m_operations.Pop();
			if (!operation) {
				break;
			}
		}
		if (operation->kind == GpuOperationKind::DrawIndex ||
		    operation->kind == GpuOperationKind::DrawAuto) {
			// The register snapshots stay valid until the operation executed.
			operation->resolved =
			    executor.ResolveDrawAhead(*operation->state.context, *operation->state.user_config,
			                              *operation->state.shaders);
		} else if (operation->kind == GpuOperationKind::DrawIndirect) {
			// Resolved for the offsets the GPU-read records give the shaders; draws that read
			// their records on the CPU patch the SGPRs instead and resolve themselves.
			ShaderIndirectOffsets offsets;
			if (IndirectDrawOffsets(operation->draw_indirect.registers, offsets)) {
				operation->resolved = executor.ResolveDrawAhead(
				    *operation->state.context, *operation->state.user_config,
				    *operation->state.shaders, offsets);
			}
		} else if (operation->kind == GpuOperationKind::DispatchDirect) {
			const uint32_t groups[] = {operation->dispatch.thread_group_x,
			                           operation->dispatch.thread_group_y,
			                           operation->dispatch.thread_group_z};
			operation->resolved = executor.ResolveDispatchAhead(
			    *operation->state.context, *operation->state.shaders, groups,
			    operation->dispatch.mode, operation->processor->IsAsyncComputeQueue());
		} else if (operation->kind == GpuOperationKind::DispatchIndirect) {
			// The program does not depend on the GPU-written arguments, only on the registers
			// and the guest memory its resources read.
			operation->resolved = executor.ResolveIndirectDispatchAhead(
			    *operation->state.context, *operation->state.shaders, operation->dispatch.mode,
			    operation->dispatch.args_addr,
			    operation->processor->IsAsyncComputeQueue());
		}
		m_resolved.Push(std::move(*operation));
	}
	m_resolved.Stop();
}

void GuestGpu::StopExecution() {
	WaitForExecution();
	m_operations.Stop();
	if (m_resolver.joinable()) {
		m_resolver.join();
	}
	if (m_executor.joinable()) {
		m_executor.join();
	}
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_ctx_dirty    = true;
	m_ucfg_dirty   = true;
	m_sh_ctx_dirty = true;
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_index_base_addr                  = 0;
	m_num_instances.value              = 1;
	m_num_instances.sequence++;
	m_predicate_skip                   = false;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	m_ctx_dirty = true;
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_context_state_pushed);
			m_ctx                  = m_saved_ctx;
			m_saved_ctx            = {};
			m_context_state_pushed = false;
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			m_ctx.Reset();
			break;
		default: EXIT("unknown context state operation: %u\n", static_cast<uint32_t>(operation));
	}
}

GpuOperation CommandProcessor::MakeOperation(GpuOperationKind kind) {
	GpuOperation operation;
	operation.kind          = kind;
	operation.processor     = this;
	operation.submit_id     = m_submit_id;
	operation.num_instances = m_num_instances;
	operation.after_wait    = std::exchange(m_wait_since_operation, false);
	operation.guest_sync    = std::exchange(m_acquire_since_operation, false) || operation.after_wait;
	if (m_pipeline == nullptr) {
		operation.state = {&m_ctx, &m_ucfg, &m_sh_ctx};
	} else if (kind == GpuOperationKind::Callback) {
		// Only draws and dispatches read registers.
		operation.state = m_pipeline->NeutralState();
	} else {
		operation.state = SnapshotState(m_pipeline->NextSequence());
	}
	return operation;
}

template <typename T>
const T* CommandProcessor::Snapshot(SnapshotRing<T>& ring, const T& live, bool& dirty,
                                    uint64_t sequence, uint64_t* id) {
	static_assert(std::is_trivially_copyable_v<T>);
	if (dirty || ring.current == SnapshotRing<T>::Size) {
		if (ring.slots == nullptr) {
			ring.slots = std::make_unique<typename SnapshotRing<T>::Slot[]>(SnapshotRing<T>::Size);
			ring.last_use = std::make_unique<uint64_t[]>(SnapshotRing<T>::Size);
		}
		auto& slot = ring.slots[ring.next];
		// The execution thread publishes the sequence of an operation after it stops reading the
		// operation's snapshots; the wait acquires that.
		m_pipeline->WaitForExecution(ring.last_use[ring.next]);
		slot.value   = live;
		slot.id      = s_snapshot_ids.fetch_add(1, std::memory_order_relaxed) + 1;
		ring.current = ring.next;
		ring.next    = (ring.next + 1) % SnapshotRing<T>::Size;
		dirty        = false;
	}
	// The operation built from this state is the next one the command processor emits.
	ring.last_use[ring.current] = sequence;
	if (id != nullptr) {
		*id = ring.slots[ring.current].id;
	}
	return &ring.slots[ring.current].value;
}

GpuRegisterState CommandProcessor::SnapshotState(uint64_t sequence) {
	GpuRegisterState state;
	state.context     = Snapshot(m_ctx_snapshots, m_ctx, m_ctx_dirty, sequence, &state.context_id);
	state.user_config = Snapshot(m_ucfg_snapshots, m_ucfg, m_ucfg_dirty, sequence);
	state.shaders     = Snapshot(m_sh_ctx_snapshots, m_sh_ctx, m_sh_ctx_dirty, sequence);
	return state;
}

uint64_t CommandProcessor::Emit(GpuOperation&& operation) {
	if (m_pipeline != nullptr) {
		EXIT_IF(operation.kind != GpuOperationKind::Callback &&
		        m_pipeline->NextSequence() !=
		            m_ctx_snapshots.last_use[m_ctx_snapshots.current]);
		return m_pipeline->EmitOperation(std::move(operation));
	}
	Execute(operation);
	return 0;
}

void CommandProcessor::NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size,
                                      const void* value) {
	if (m_pipeline != nullptr && size != 0) {
		m_pipeline->NoteGuestWrite(sequence, address, size, value);
	}
}

void CommandProcessor::EmitHostCommand(Common::UniqueFunction<void>&& command) {
	EmitCallback(std::move(command));
}

void CommandProcessor::ReadDecisionMemory(const volatile void* address, void* value,
                                          size_t size) {
	if (m_pipeline != nullptr) {
		m_pipeline->ReadDecisionMemory(address, value, size);
		return;
	}
	CopyFromGuest(value, address, size);
}

uint64_t CommandProcessor::EmitCallback(GpuCallback&& callback) {
	auto operation     = MakeOperation(GpuOperationKind::Callback);
	operation.callback = std::move(callback);
	return Emit(std::move(operation));
}

void CommandProcessor::Execute(GpuOperation& operation) {
	EXIT_IF(operation.processor != this);
	BindState(operation.state);
	m_renderer.NoteOperation(operation.submit_id, operation.after_wait);
	auto& executor = m_renderer.GetRenderExecutor();
	executor.UseContextId(operation.state.context_id);
	executor.BeginGuestOperation(static_cast<uint32_t>(m_interrupt_event_id),
	                             operation.kind == GpuOperationKind::DispatchDirect,
	                             operation.guest_sync);
	if (operation.after_wait) {
		executor.InvalidateRenderTargetMemo();
	}
	if (operation.kind != GpuOperationKind::DrawIndex &&
	    operation.kind != GpuOperationKind::DrawAuto &&
	    operation.kind != GpuOperationKind::DrawIndirect) {
		executor.InvalidateRenderTargetMemo();
		// Guest-visible effects (labels, interrupts, flips, host commands) let the guest overwrite
		// what earlier draws read: their copies complete first.
		if (auto* copies = GetScheduler().HostCopies(); copies != nullptr) {
			copies->Drain();
		}
	}
	struct CommitWriteTicks {
		RenderContext& renderer;
		~CommitWriteTicks() {
			if (renderer.GetBufferCache().CommitWriteTicks()) {
				renderer.GetCommandScheduler().Flush();
			}
		}
	} commit_write_ticks {m_renderer};
	switch (operation.kind) {
		case GpuOperationKind::DrawIndex:
		case GpuOperationKind::DrawAuto: {
			executor.UseResolvedDraw(operation.resolved);
			if (operation.kind == GpuOperationKind::DrawIndex) {
				ExecuteDrawIndex(operation, operation.draw_index);
			} else {
				ExecuteDrawAuto(operation, operation.draw_auto);
			}
			executor.UseResolvedDraw(nullptr);
			if (operation.resolved != nullptr) {
				RenderExecutor::ReleaseResolvedDraw(operation.resolved);
			}
			break;
		}
		case GpuOperationKind::DrawIndirect:
			// The resolution serves the first draw of the packet (its programs do not depend on
			// the arguments); later draws resolve their own.
			executor.UseResolvedDraw(operation.resolved);
			ExecuteDrawIndirect(operation);
			executor.UseResolvedDraw(nullptr);
			if (operation.resolved != nullptr) {
				RenderExecutor::ReleaseResolvedDraw(operation.resolved);
			}
			break;
		case GpuOperationKind::DispatchDirect: {
			const auto& dispatch = operation.dispatch;
			executor.UseResolvedDraw(operation.resolved);
			ExecuteDispatchDirect(operation, dispatch.thread_group_x, dispatch.thread_group_y,
			                      dispatch.thread_group_z, dispatch.mode);
			executor.UseResolvedDraw(nullptr);
			if (operation.resolved != nullptr) {
				RenderExecutor::ReleaseResolvedDraw(operation.resolved);
			}
			break;
		}
		case GpuOperationKind::DispatchIndirect:
			executor.UseResolvedDraw(operation.resolved);
			ExecuteDispatchIndirect(operation);
			executor.UseResolvedDraw(nullptr);
			if (operation.resolved != nullptr) {
				RenderExecutor::ReleaseResolvedDraw(operation.resolved);
			}
			break;
		case GpuOperationKind::Callback:
			if (operation.callback) {
				operation.callback();
			}
			break;
	}
	// Starts the copies of the operation while the next ones are prepared.
	if (auto* copies = GetScheduler().HostCopies(); copies != nullptr) {
		(void)copies->Flush();
	}
}

void CommandProcessor::BindState(const GpuRegisterState& state) {
	GetScheduler().Begin(*state.context, *state.user_config, *state.shaders);
}

void CommandProcessor::BufferInit() {
	// Executing any operation binds its register state and begins a command buffer.
	if (m_pipeline == nullptr) {
		Emit(MakeOperation(GpuOperationKind::Callback));
	}
}

void CommandProcessor::BufferFlush() {
	// Downloads of the windows the game threads keep reading back, behind this submission's
	// writes, in the recording about to be submitted (BufferCache::EagerDownloads).
	m_renderer.GetBufferCache().EagerDownloads();
	SetSubmitReasonIfUnset("BufferFlush (other)");
	(KYTY_SUBMIT_REASON(), GetScheduler().Flush());
}

void CommandProcessor::CompleteReleaseMemWrite() {
	GetScheduler().CompleteReleaseMemWrite();
}

void CommandProcessor::CompleteReleaseMemInterrupt() {
	GetScheduler().CompleteReleaseMemInterrupt();
}

void CommandProcessor::CompleteDraw() {
	GetScheduler().CompleteDraw();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	(KYTY_SUBMIT_REASON(), GetScheduler().Finish());
}

void CommandProcessor::RunGarbageCollector() {
	EmitCallback([this] { m_renderer.RunGarbageCollector(); });
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4();
	}
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

template <typename T>
T CommandProcessor::ReadLabel(const volatile T* addr) const {
	// A label the CPU wrote (its page is CPU-dirty) may follow data it wrote (KYTY_HOT_PAGES);
	// the GPU's own labels (end-of-pipe writes) do not start a generation.
	if (HotPagesEnabled() && m_renderer.GetBufferCache().IsRegionCpuModified(
	                             reinterpret_cast<uint64_t>(addr) & ~uint64_t {3}, sizeof(T))) {
		BumpHotGeneration();
	}
	return GuestGpu::LabelsDeferred() ? m_renderer.GetGpu().ReadLabel(addr) : *addr;
}

template uint32_t CommandProcessor::ReadLabel<uint32_t>(const volatile uint32_t*) const;
template uint64_t CommandProcessor::ReadLabel<uint64_t>(const volatile uint64_t*) const;

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	(void)poll;
	if (!TestWaitRegMemValue(ReadLabel(addr), ref, mask, func)) {
		SuspendPm4();
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

// Diagnostics: KYTY_TRACE_WRITERS=<hex address>[:<hex size>]: CP writes (DMA_DATA, WRITE_DATA)
// overlapping that range are printed with their source ([cp-writers] on stderr).
static bool CpWriteTraced(uint64_t dst, uint64_t bytes) {
	static const std::pair<uint64_t, uint64_t> target = [] {
		std::pair<uint64_t, uint64_t> range {0, 0};
		if (const char* value = std::getenv("KYTY_TRACE_WRITERS"); value != nullptr) {
			char* end    = nullptr;
			range.first  = std::strtoull(value, &end, 16);
			range.second = (end != nullptr && *end == ':') ? std::strtoull(end + 1, nullptr, 16) : 4u;
		}
		return range;
	}();
	static std::atomic<uint32_t> printed {0};
	return target.first != 0 && dst < target.first + target.second && target.first < dst + bytes &&
	       printed.fetch_add(1) < 200u;
}

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}
	if (CpWriteTraced(reinterpret_cast<uint64_t>(dst), static_cast<uint64_t>(dw_num) * 4u)) {
		std::fprintf(stderr, "[cp-writers] WRITE_DATA dst=0x%012" PRIx64 " dwords=%u first=%08x one_addr=%d\n",
		             reinterpret_cast<uint64_t>(dst), dw_num, src[0], write_one_address ? 1 : 0);
	}

	// Labels often share a page with bytes the GPU wrote. A plain write then faults and downloads
	// the page, draining the GPU; WriteClean writes both copies instead.
	auto& cache = m_renderer.GetBufferCache();
	if (write_one_address) {
		if (cache.WriteClean(reinterpret_cast<uint64_t>(dst), &src[dw_num - 1], sizeof(uint32_t))) {
			return;
		}
	} else {
		const auto bytes = static_cast<size_t>(dw_num) * sizeof(uint32_t);
		if (cache.WriteClean(reinterpret_cast<uint64_t>(dst), src, bytes)) {
			return;
		}
	}
	// Completion labels can be reused immediately; memcpy may write a destination twice.
	auto* output = static_cast<volatile uint32_t*>(dst);
	for (uint32_t i = 0; i < dw_num; i++) {
		*output = src[i];
		output += !write_one_address;
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto sequence = EmitCallback([dst_address, num_bytes] {
		const auto value = Sync::ReadReferenceClock();
		std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
		static std::atomic<uint32_t> clock_log_count {0};
		if (clock_log_count.fetch_add(1) < 64) {
			LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
			     " size=%u\n",
			     dst_address, value, num_bytes);
		}
	});
	NoteGuestWrite(sequence, dst_address, num_bytes);
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	bool src_gds = false;
	if (src_sel != 2) {
		if (!decode_gds(src_sel, src_gds)) {
			EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
		}
		if (src_gds && dst_gds) {
			EXIT("unsupported dmaData GDS-to-GDS copy\n");
		}
	}
	const auto sequence = EmitCallback([=, this] {
		ExecuteDmaData(dst_gds, dst_address_or_offset, src_sel, src_gds,
		               src_address_or_offset_or_immediate, num_bytes);
	});
	if (!dst_gds) {
		NoteGuestWrite(sequence, dst_address_or_offset, num_bytes);
	}
}

void CommandProcessor::ExecuteDmaData(bool dst_gds, uint64_t dst_address_or_offset,
                                      uint8_t src_sel, bool src_gds,
                                      uint64_t src_address_or_offset_or_immediate,
                                      uint32_t num_bytes) {
	auto& buffer_cache = m_renderer.GetBufferCache();
	if (!dst_gds && CpWriteTraced(dst_address_or_offset, num_bytes)) {
		uint32_t words[8] {};
		const bool src_mem = src_sel == 0 || src_sel == 3;
		const bool read = src_mem && Libs::LibKernel::Memory::TryReadBacking(
		                                 src_address_or_offset_or_immediate, words, std::min<uint32_t>(num_bytes, 32u));
		std::fprintf(stderr,
		             "[cp-writers] DMA_DATA dst=0x%012" PRIx64 " bytes=0x%x src_sel=%u src=0x%012" PRIx64
		             " src_gpu_dirty=%d src_words(%d)=%08x %08x %08x %08x %08x %08x %08x %08x\n",
		             dst_address_or_offset, num_bytes, src_sel, src_address_or_offset_or_immediate,
		             src_mem && buffer_cache.HasGpuDirtyBytes(src_address_or_offset_or_immediate, num_bytes) ? 1 : 0,
		             read ? 1 : 0, words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
	}
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	if (m_draw_prep != nullptr) {
		// In queue order, as the GPU thread will run them.
		if (submission.type == SubmissionType::Graphics ||
		    submission.type == SubmissionType::Compute) {
			submission.prep_seq = ++m_prep_seq;
			m_draw_prep->EnqueueSubmission(submission.queue_id, submission.prep_seq,
			                               submission.commands);
		} else if (submission.type == SubmissionType::SuspendPoint) {
			m_draw_prep->EnqueueReset(submission.queue_id);
		}
	}
	if (submission.owned_words != nullptr) {
		// The draw-prep scanner can still be reading a submission the GPU thread has finished:
		// keep the copies of the last ones alive.
		static std::deque<std::shared_ptr<std::vector<uint32_t>>> recent;
		recent.push_back(submission.owned_words);
		if (recent.size() > 1024) {
			recent.pop_front();
		}
	}
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_epoch.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

// KYTY_SRT_DEFER: the tick the dispatch just suspended waits for; GuestGpu::ThreadRun moves it
// to the submission (GPU thread only).
static uint64_t g_srt_wait_tick = 0;

struct SrtDeferStats {
	uint64_t deferred   = 0; // dispatches suspended
	uint64_t ran_other  = 0; // submissions of other queues run while one waited
	uint64_t idle_waits = 0; // nothing else to run: the GPU thread waited for the tick
	uint64_t idle_us    = 0;
	// GPU thread time by state (KYTY_FPS_LOG): no work queued, every queue blocked, processing
	// submissions, running commands.
	uint64_t empty_us   = 0;
	uint64_t blocked_us = 0;
	uint64_t process_us = 0;
	uint64_t command_us = 0;
	uint64_t lock_us    = 0; // acquiring the queue lock
	uint64_t loop_us    = 0; // whole loop iterations
	uint64_t iters      = 0;
	uint64_t prog_ns = 0, prog_calls = 0, noprog_ns = 0, noprog_calls = 0, completes = 0;
	uint64_t blocked_waits = 0, cmd_count = 0;
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};
static SrtDeferStats g_srt_defer_stats;
static uint32_t      g_stat_queue = 0;
struct SuspendSite {
	uint64_t count = 0;
	uint32_t queue = 0;
	uint32_t dw[8] {};
};
static std::unordered_map<const uint32_t*, SuspendSite> g_suspend_sites;
static void NoteSuspendSite(const uint32_t* packet, uint32_t remaining_dw) {
	static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr;
	if (!log) return;
	auto& site = g_suspend_sites[packet];
	if (site.count++ == 0) {
		site.queue = g_stat_queue;
		for (uint32_t i = 0; i < 8 && i < remaining_dw; i++) site.dw[i] = packet[i];
	}
}

static void ReportSrtDeferStats() {
	auto&      st  = g_srt_defer_stats;
	const auto now = std::chrono::steady_clock::now();
	if (now - st.last < std::chrono::seconds(5)) {
		return;
	}
	static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr;
	if (log) {
		const double s = std::chrono::duration<double>(now - st.last).count();
		std::fprintf(stderr,
		             "[gpu-thread] window=%.2fs iters=%llu ms/s: loop=%.0f lock=%.0f empty=%.0f blocked=%.0f process=%.0f commands=%.0f | "
		             "srt-defer deferred=%.0f/s other-queue slices=%.0f/s idle=%.0f ms/s\n",
		             s, static_cast<unsigned long long>(st.iters),
		             static_cast<double>(st.loop_us) / 1e6 / s,
		             static_cast<double>(st.lock_us) / 1e6 / s,
		             static_cast<double>(st.empty_us) / 1e6 / s,
		             static_cast<double>(st.blocked_us) / 1e6 / s,
		             static_cast<double>(st.process_us) / 1e6 / s,
		             static_cast<double>(st.command_us) / 1e6 / s,
		             static_cast<double>(st.deferred) / s, static_cast<double>(st.ran_other) / s,
		             static_cast<double>(st.idle_us) / 1e6 / s);
	}
	if (log) {
		const double s = std::chrono::duration<double>(now - st.last).count();
		std::fprintf(stderr,
		             "[gpu-thread2] progress: %.0f calls/s %.0f ms/s (completes %.0f/s) | no-progress: %.0f calls/s %.0f ms/s | all-blocked waits %.0f/s | loop commands %.0f/s (send %.0f, labels %.0f, sync %.0f)\n",
		             static_cast<double>(st.prog_calls) / s, static_cast<double>(st.prog_ns) / 1e6 / s,
		             static_cast<double>(st.completes) / s,
		             static_cast<double>(st.noprog_calls) / s, static_cast<double>(st.noprog_ns) / 1e6 / s,
		             static_cast<double>(st.blocked_waits) / s, static_cast<double>(st.cmd_count) / s,
		             static_cast<double>(g_cmd_from_send.exchange(0)) / s,
		             static_cast<double>(g_cmd_from_try.exchange(0)) / s,
		             static_cast<double>(g_cmd_from_sync.exchange(0)) / s);
		std::vector<std::pair<const uint32_t*, SuspendSite>> sites(g_suspend_sites.begin(),
		                                                           g_suspend_sites.end());
		std::sort(sites.begin(), sites.end(),
		          [](const auto& x, const auto& y) { return x.second.count > y.second.count; });
		for (size_t i = 0; i < sites.size() && i < 8; i++) {
			const auto& t = sites[i].second;
			std::fprintf(stderr,
			             "[suspend] %.0f/s q=%u op=0x%02x packet=%p dw: %08x %08x %08x %08x %08x %08x %08x %08x\n",
			             static_cast<double>(t.count) / s, t.queue, (t.dw[0] >> 8u) & 0xffu,
			             static_cast<const void*>(sites[i].first), t.dw[0], t.dw[1], t.dw[2], t.dw[3],
			             t.dw[4], t.dw[5], t.dw[6], t.dw[7]);
		}
		std::fprintf(stderr, "[suspend] distinct sites=%zu\n", sites.size());
	}
	g_suspend_sites.clear();
	st = {};
	st.last = now;
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	Common::PreferPerformanceCores("GPU thread");
	Common::RaiseGpuThreadPriority();
	Profiler::t_gpu_thread = true;
	g_gpu_thread = true;
	g_gpu_state  = gpu;
	Common::Sampler::RegisterCurrentThread("gpu");

	auto loop_mark = std::chrono::steady_clock::now();
	for (;;) {
		{
			const auto now = std::chrono::steady_clock::now();
			g_srt_defer_stats.loop_us += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(now - loop_mark).count());
			g_srt_defer_stats.iters++;
			loop_mark = now;
		}
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		if (spin_blocked) {
			spin_blocked      = false;
			const auto epoch  = gpu->m_work_epoch.load(std::memory_order_acquire);
			const auto target = std::chrono::steady_clock::now() + BlockedRetryInterval;
			while (gpu->m_work_epoch.load(std::memory_order_acquire) == epoch &&
			       std::chrono::steady_clock::now() < target) {
				Common::SpinPause();
			}
		}
		if (gpu->m_pipelined) {
			// The GPU is idle once the execution thread ran every operation too. Only this
			// thread emits operations, so none can follow the wait until it finds work.
			bool     no_work = false;
			uint64_t epoch   = 0;
			{
				Common::LockGuard lock(gpu->m_queue_mutex);
				no_work = gpu->m_commands.empty() && gpu->m_submission_count == 0 &&
				          !gpu->m_stopping;
			}
			if (no_work) {
				gpu->WaitForExecution();
				{
					Common::LockGuard lock(gpu->m_queue_mutex);
					epoch = gpu->m_work_epoch.load(std::memory_order_acquire);
					if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
						gpu->m_processing = false;
						gpu->m_idle.SignalAll();
					}
				}
				// Spin for the next submission; sleep below only when none came.
				Common::SpinWait spin;
				while (gpu->m_work_epoch.load(std::memory_order_acquire) == epoch && spin.Spin()) {
				}
			}
		}
		{
			const auto lock_begin = std::chrono::steady_clock::now();
			Common::LockGuard lock(gpu->m_queue_mutex);
			g_srt_defer_stats.lock_us += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(
			        std::chrono::steady_clock::now() - lock_begin)
			        .count());
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				Timeline::Mark("gpu-idle");
				const auto empty_begin = std::chrono::steady_clock::now();
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
				g_srt_defer_stats.empty_us += static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(
				        std::chrono::steady_clock::now() - empty_begin)
				        .count());
				Timeline::Mark("gpu-wake");
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				int      selected_queue = -1;
				bool     tick_pending   = false;
				uint64_t pending_tick   = UINT64_MAX;
				auto& scheduler     = gpu->m_renderer.GetCommandScheduler();
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (gpu->m_queues[id].empty()) {
						continue;
					}
					auto& front = gpu->m_queues[id].front();
					if (front.srt_wait_tick != 0) {
						// A dispatch suspended for SRT bytes to land (KYTY_SRT_DEFER).
						if (!scheduler.IsFree(front.srt_wait_tick)) {
							tick_pending = true;
							pending_tick = std::min(pending_tick, front.srt_wait_tick);
							continue;
						}
						front.blocked = false;
					}
					if (!front.blocked && selected_queue < 0) {
						selected_queue = static_cast<int>(id);
						if (tick_pending || !PipelineCache::SrtDeferEnabled()) {
							break;
						}
					}
				}
				if (selected_queue < 0 && tick_pending) {
					// Nothing else to run: poll for the downloads (and new submissions).
					gpu->m_processing = false;
					const auto wait_begin = std::chrono::steady_clock::now();
					// Wait for the GPU itself (2 ms slices, the queue unlocked so the guest can
					// submit), not for a new submission: nothing signals m_work_available when a
					// download lands, so a 50 ms wait sat out most of it when no other work came
					// (KYTY_SRT_DEFER_IDLE_LEGACY=1: the old wait).
					static const bool legacy_idle = [] {
						const char* value = std::getenv("KYTY_SRT_DEFER_IDLE_LEGACY");
						return value != nullptr && value[0] == '1';
					}();
					if (legacy_idle || pending_tick == UINT64_MAX) {
						gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 50);
					} else {
						gpu->m_queue_mutex.Unlock();
						if (scheduler.Active() && pending_tick >= scheduler.CurrentTick()) {
							(KYTY_SUBMIT_REASON(), scheduler.Flush()); // the download is still in the open command buffer
						}
						(void)scheduler.WaitFor(pending_tick, 2'000'000);
						gpu->m_queue_mutex.Lock();
					}
					g_srt_defer_stats.idle_waits++;
					g_srt_defer_stats.idle_us += static_cast<uint64_t>(
					    std::chrono::duration_cast<std::chrono::nanoseconds>(
					        std::chrono::steady_clock::now() - wait_begin)
					        .count());
					continue;
				}
				if (selected_queue >= 0 && tick_pending) {
					g_srt_defer_stats.ran_other++;
				}
				if (selected_queue < 0) {
					gpu->m_processing = false;
					Timeline::Mark("gpu-blocked");
					const auto blocked_begin = std::chrono::steady_clock::now();
					g_srt_defer_stats.blocked_waits++;
					gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					g_srt_defer_stats.blocked_us += static_cast<uint64_t>(
					    std::chrono::duration_cast<std::chrono::nanoseconds>(
					        std::chrono::steady_clock::now() - blocked_begin)
					        .count());
					Timeline::Mark("gpu-unblocked");
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
				has_submission    = true;
			}
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			if (gpu->m_pipelined) {
				gpu->StopExecution();
			}
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			g_cp_thread  = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			const auto command_begin = std::chrono::steady_clock::now();
			command();
			g_srt_defer_stats.cmd_count++;
			g_srt_defer_stats.command_us += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(
			        std::chrono::steady_clock::now() - command_begin)
			        .count());
			ReportSrtDeferStats();

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		if (submission.srt_wait_tick != 0) {
			// Its downloads are complete; their publication runs as a priority operation.
			gpu->m_renderer.GetCommandScheduler().WaitPriorityOperations(
			    std::exchange(submission.srt_wait_tick, 0));
		}
		g_srt_wait_tick     = 0;
		const auto process_begin = std::chrono::steady_clock::now();
		g_stat_queue = submission.queue_id;
		const bool complete = gpu->Process(submission);
		{
			const auto ns = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(
			        std::chrono::steady_clock::now() - process_begin)
			        .count());
			g_srt_defer_stats.process_us += ns;
			const bool progress = submission.command_execution.MadeProgress() ||
			                      submission.constant_execution.MadeProgress();
			if (progress) {
				g_srt_defer_stats.prog_ns += ns;
				g_srt_defer_stats.prog_calls++;
				if (complete) g_srt_defer_stats.completes++;
			} else {
				g_srt_defer_stats.noprog_ns += ns;
				g_srt_defer_stats.noprog_calls++;
			}
		}
		const auto srt_tick = std::exchange(g_srt_wait_tick, 0);
		ReportSrtDeferStats();

		const auto lock2_begin = std::chrono::steady_clock::now();
		Common::LockGuard lock(gpu->m_queue_mutex);
		g_srt_defer_stats.lock_us += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(
		        std::chrono::steady_clock::now() - lock2_begin)
		        .count());
		if (!complete) {
			submission.blocked       = true;
			submission.srt_wait_tick = srt_tick;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		if (gpu->m_pipelined) {
			// Stays processing until the operations executed; see the idle wait above.
			continue;
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

bool GuestGpu::HasReadySubmission() {
	Common::LockGuard lock(m_queue_mutex);
	for (const auto& queue: m_queues) {
		if (!queue.empty() && !queue.front().blocked) {
			return true;
		}
	}
	return false;
}

// KYTY_COALESCE_SUBMITS=1: a complete guest submission with another one ready behind it is not
// submitted to the host GPU on its own; the next one records into the same command buffer (the
// periodic draw flush still submits every 16 draws). ~108 of the ~160 submissions per frame in
// the jungle were these, 2026-10-18.
static bool CoalesceSubmits() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_COALESCE_SUBMITS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// KYTY_HOT_PAGES: a new generation when a submission starts (the game submits after writing);
// every 60 s every hot page goes back to faulting once (KYTY_FPS_LOG prints [hot-pages]).
static void HotPagesSlice(bool first_slice) {
	if (!HotPagesEnabled()) {
		return;
	}
	if (first_slice) {
		BumpHotGeneration();
	}
	static auto last       = std::chrono::steady_clock::now();
	static auto last_epoch = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	const auto  s    = std::chrono::duration<double>(now - last).count();
	if (s < 5.0) {
		return;
	}
	last = now;
	if (now - last_epoch > std::chrono::seconds(60)) {
		last_epoch = now;
		g_hot_epoch.fetch_add(1, std::memory_order_relaxed);
	}
	if (static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr; log) {
		std::fprintf(stderr,
		             "[hot-pages] faults %.0f/s, promoted %.0f/s, uploads %.0f/s, repeats skipped "
		             "%.0f/s, demoted by GPU writes %.0f/s, generations %.0f/s\n",
		             g_hot_stats.faults.exchange(0) / s, g_hot_stats.promoted.exchange(0) / s,
		             g_hot_stats.uploads.exchange(0) / s, g_hot_stats.repeats.exchange(0) / s,
		             g_hot_stats.demoted_by_gpu.exchange(0) / s,
		             [&] {
			             static uint32_t previous = 0;
			             const auto      current  = g_hot_generation.load();
			             const auto      delta    = current - previous;
			             previous                 = current;
			             return delta / s;
		             }());
	}
}

bool GuestGpu::Process(Submission& submission) {
	HotPagesSlice(!submission.started);
	const bool first_slice = !submission.started;
	g_run_expected_next    = nullptr; // a run continues only within one slice of one submission
	auto&      cp          = GetProcessor(submission.queue_id);
	Timeline::Mark("proc-begin", submission.queue_id, static_cast<uint64_t>(submission.type));
	struct EndMark {
		const Submission& submission;
		~EndMark() {
			Timeline::Mark("proc-end", submission.queue_id,
			               submission.command_complete ? 1u : 0u);
		}
	} end_mark {submission};

	if (first_slice) {
		g_guest_submission_seq.fetch_add(1, std::memory_order_relaxed);
		// Texture-heap entries the guest wrote before submitting are visible to all of this
		// submission's draws, as on the console.
		m_renderer.GetRenderExecutor().SyncBindlessHeaps();
	}

	if (first_slice) {
		submission.started = true;
		cp.SetSubmitId(++m_submit_id);
	}

	cp.BufferInit();
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Graphics: {
			submission.command_execution.SetPrepSeq(submission.prep_seq, submission.queue_id);
			bool progressed = false;
			submission.constant_complete |= submission.constant_commands.empty();
			for (;;) {
				bool round_progress = false;
				if (!submission.constant_complete) {
					submission.constant_complete =
					    cp.Process(submission.constant_execution, submission.constant_commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.constant_execution.MadeProgress();
				}
				cp.SetCeComplete(submission.constant_complete);
				if (!submission.command_complete) {
					submission.command_complete =
					    cp.Process(submission.command_execution, submission.commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.command_execution.MadeProgress();
				}
				progressed |= round_progress;
				complete = submission.command_complete && submission.constant_complete;
				if (complete || !round_progress) {
					break;
				}
			}
			// Caches age at the frame boundary (SuspendPoint), not once per command list.
			if (progressed && !(complete && CoalesceSubmits() && HasReadySubmission())) {
				SetSubmitReason("end of graphics submission");
				cp.BufferFlush();
			}
			break;
		}
		case SubmissionType::Compute: {
			submission.command_execution.SetPrepSeq(submission.prep_seq, submission.queue_id);
			const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
			const auto*     buffer = submission.commands.data();
			static uint32_t compute_batch_log_count = 0;
			if (first_slice && num_dw <= 128 && compute_batch_log_count++ < 32) {
				LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(buffer), num_dw);
				for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
					LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
				}
			}
			if (first_slice) {
				const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
				const auto*     buffer = submission.commands.data();
				static uint32_t compute_batch_log_count = 0;
				if (num_dw <= 128 && compute_batch_log_count++ < 32) {
					LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
					     reinterpret_cast<uint64_t>(buffer), num_dw);
					for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
						LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
					}
				}
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
				g_compute_submissions.fetch_add(1, std::memory_order_relaxed);
				g_compute_submission_dw.fetch_add(num_dw, std::memory_order_relaxed);
			}
			complete = cp.Process(submission.command_execution, submission.commands) ==
			           Pm4ProcessResult::Complete;
			if (submission.command_execution.MadeProgress() &&
			    !(complete && CoalesceSubmits() && HasReadySubmission())) {
				SetSubmitReason("end of compute submission");
				cp.BufferFlush();
			}
			break;
		case SubmissionType::FlipPreparation:
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::SuspendPoint:
			// Age caches at the frame boundary, not once per command list.
			m_renderer.RunGarbageCollector();
			m_renderer.GetCommandScheduler().DeferPriorityOperation(
			    [ready = m_suspend_point_ready] { ready->release(); });
			SetSubmitReason("suspend point");
			cp.BufferFlush();
			cp.Reset();
			break;
	}

	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~ExecutionScope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} execution_scope(*this, execution);

	ProcessPm4(execution);
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

bool CommandProcessor::BeginDispatchDefer() {
	if (!PipelineCache::SrtDeferEnabled() || g_current_execution == nullptr ||
	    g_current_execution->m_buffer_stack.empty()) {
		PipelineCache::BeginSrtDefer(false);
		return false;
	}
	auto&       execution = *g_current_execution;
	const auto& cursor    = execution.m_buffer_stack.back();
	const auto* packet    = cursor.commands.data() + cursor.offset_dw;
	if (execution.m_deferred_packet != packet) {
		execution.m_deferred_packet = nullptr;
		execution.m_defer_count     = 0;
		execution.m_deferred_tick   = 0;
	} else if (execution.m_deferred_tick != 0 && !GetScheduler().IsFree(execution.m_deferred_tick)) {
		// Run again before its downloads landed (the constant engine made progress): still waiting.
		g_srt_wait_tick = std::max(g_srt_wait_tick, execution.m_deferred_tick);
		SuspendPm4();
		return true;
	}
	// A packet run again after its downloads landed may find newer GPU writes (another queue
	// ran meanwhile); after a few rounds it waits as before rather than chase them.
	PipelineCache::BeginSrtDefer(execution.m_defer_count < 4);
	return false;
}

void CommandProcessor::EndDispatchDefer() {
	const auto tick = PipelineCache::EndSrtDefer();
	if (tick == 0 || g_current_execution == nullptr || g_current_execution->m_buffer_stack.empty()) {
		return;
	}
	auto&       execution       = *g_current_execution;
	const auto& cursor          = execution.m_buffer_stack.back();
	execution.m_deferred_packet = cursor.commands.data() + cursor.offset_dw;
	execution.m_deferred_tick   = tick;
	execution.m_defer_count++;
	g_srt_wait_tick = std::max(g_srt_wait_tick, tick);
	g_srt_defer_stats.deferred++;
	SuspendPm4();
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	const bool draw_prep = execution.m_prep_seq != 0 && DrawPrep::Enabled();
	if (DrawPrep::Enabled() && !draw_prep) {
		DrawPrep::ClearCurrent();
	}
	while (!execution.m_buffer_stack.empty()) {
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
		}
		auto& cursor = execution.m_buffer_stack.back();
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.offset_dw == cursor.commands.size()) {
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;
		EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);

		if (packet_header == 0x80000000u || Pm4::PacketSizeDw(packet_header) == 1u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if ((packet_header >> 30u) == 0u) {
			// A type-0 packet writes COUNT+1 registers from a base index; the command processor
			// ignores them, and games leave reserved dwords as zeros, which decode to this
			// (a zero header is one register write of 0 to index 0).
			const auto packet_dw = ((packet_header >> 16u) & 0x3fffu) + 2u;
			static std::atomic<uint32_t> type0_log_count {0};
			if (type0_log_count.fetch_add(1) < 16) {
				LOGF("CP: skipping type-0 packet header=0x%08" PRIx32 " len=%" PRIu32
				     " at offset=0x%05" PRIx32 "\n",
				     packet_header, packet_dw, total_dw - remaining_dw);
			}
			if (static std::atomic<uint64_t> skipped {0}; Common::HackNotice::PowerOfTwoTick(skipped)) {
				Common::HackNotice::Report("PM4 type-0 packet skipped", "CommandProcessor::Process", 0,
				                           packet_header, "the command processor ignores its registers writes");
			}
			EXIT_NOT_IMPLEMENTED(packet_dw > remaining_dw);
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		// Draw prep: number the draw and dispatch packets as the scanner does (predicated ones too;
		// a packet suspended and run again keeps its number).
		if (draw_prep && DrawPrep::CountsAsDrawOrDispatch(opcode)) {
			if (execution.m_prep_packet != packet) {
				execution.m_prep_packet = packet;
				execution.m_prep_ordinal++;
			}
			DrawPrep::SetCurrent({.queue   = execution.m_prep_queue,
			                      .seq     = execution.m_prep_seq,
			                      .ordinal = execution.m_prep_ordinal - 1u,
			                      .packet  = packet});
		}

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];
		// KYTY_PM4_TRACE="seconds:count": after `seconds`, log `count` compute-queue packets.
		{
			static const auto trace = [] {
				std::pair<double, uint64_t> t {0.0, 0};
				if (const char* v = std::getenv("KYTY_PM4_TRACE"); v != nullptr) {
					char* end = nullptr;
					t.first   = std::strtod(v, &end);
					t.second  = (end != nullptr && *end == ':') ? std::strtoull(end + 1, nullptr, 10) : 2000;
				}
				return t;
			}();
			static const auto trace_start = std::chrono::steady_clock::now();
			static uint64_t   traced      = 0;
			if (trace.second != 0 && traced < trace.second && g_stat_queue != 0 &&
			    std::chrono::duration<double>(std::chrono::steady_clock::now() - trace_start).count() >=
			        trace.first) {
				traced++;
				std::fprintf(stderr,
				             "[pm4] q=%u op=0x%02x r=0x%02x len=%u pkt=%p dw: %08x %08x %08x %08x %08x %08x\n",
				             g_stat_queue, opcode, KYTY_PM4_R(packet_header), KYTY_PM4_LEN(packet_header),
				             static_cast<const void*>(packet), remaining_dw > 1 ? packet[1] : 0,
				             remaining_dw > 2 ? packet[2] : 0, remaining_dw > 3 ? packet[3] : 0,
				             remaining_dw > 4 ? packet[4] : 0, remaining_dw > 5 ? packet[5] : 0,
				             remaining_dw > 6 ? packet[6] : 0);
			}
		}

		if (handler == nullptr) {
			const auto offset = total_dw - remaining_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
			     total_dw - remaining_dw, packet_header);
		}

		std::optional<Profiler::CommandZone> command_zone;
		if (Profiler::g_counters) {
			if (const auto* location = Pm4ZoneLocation(opcode, KYTY_PM4_R(packet_header));
			    location != nullptr) {
				command_zone.emplace(location);
			}
		}
		const auto packet_dw =
		    handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			NoteSuspendSite(packet, remaining_dw);
			return;
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances.value = num_instances;
	m_num_instances.sequence++;
}

uint32_t CommandProcessor::ResolveNumInstances(const GpuOperation& operation) {
	// An indirect draw executed after the last NUM_INSTANCES packet of the operation's stream
	// position set the persistent count.
	if (m_has_indirect_instances &&
	    m_indirect_instances_sequence == operation.num_instances.sequence) {
		const auto& gpu = m_gpu_indirect_instances;
		if (gpu.address != 0) {
			// The GPU read the records: read the last draw's instance count when a later draw
			// inherits it (the second dword of both record layouts), waiting for the GPU work that
			// wrote it.
			uint32_t count = gpu.count;
			if (gpu.count_address != 0) {
				const uint32_t written =
				    *reinterpret_cast<const volatile uint32_t*>(gpu.count_address);
				count = std::min(written, count);
			}
			m_indirect_instances =
			    count != 0 ? *reinterpret_cast<const volatile uint32_t*>(
			                     gpu.address + uint64_t {count - 1} * gpu.stride +
			                     offsetof(vk::DrawIndirectCommand, instanceCount))
			               : m_indirect_instances;
			m_gpu_indirect_instances = {};
		}
		return m_indirect_instances;
	}
	return operation.num_instances.value;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	(void)count_in_dwords;
	BumpHotGeneration();
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t      ready_bit = 1ull << 63u;
			std::array<uint64_t, 32> results {};
			ReadDecisionMemory(address, results.data(), sizeof(results));
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						// Host query results reach guest memory from the execution thread once
						// the GPU finished them: make sure it does and they are published.
						if (!m_occlusion_wait_requested) {
							m_occlusion_wait_requested = true;
							EmitCallback([this] {
								auto& queries = m_renderer.GetOcclusionQueries();
								if (queries.HasPendingDumps()) {
									GetScheduler().Finish();
									queries.PublishCompleted(true);
								}
							});
						}
						SuspendPm4();
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
			m_occlusion_wait_requested = false;
		} break;
		case 0x03:
			// The wait selector applies only to Z-pass query readiness.
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			value = ReadDecisionValue<uint64_t>(address);
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
	if (op == 0x03) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 128) {
			LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
			     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
			     reinterpret_cast<uint64_t>(address), value, condition,
			     m_predicate_skip ? 1u : 0u, wait_op);
		}
	}
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	args.index_type_and_size = m_index_type_and_size;
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	if (GraphicsRunDebugDumpEnabled() && (args.base_vertex != 0 || args.first_instance != 0)) {
		LOGF("\t draw indexed offsets: base_vertex = %" PRId32 ", first_instance = %" PRIu32 "\n",
		     args.base_vertex, args.first_instance);
	}
	m_renderer.GetRenderExecutor().DrawIndex(m_submit_id, CurrentBuffer(), args);
	CompleteDraw();
}

void CommandProcessor::ExecuteDrawIndex(const GpuOperation& operation, DrawIndexArgs args) {
	NoteRecordedWork();
	if (args.instance_count == 0) {
		args.instance_count = ResolveNumInstances(operation);
	}
	if (GraphicsRunDebugDumpEnabled() && (args.base_vertex != 0 || args.first_instance != 0)) {
		LOGF("\t draw indexed offsets: base_vertex = %" PRId32 ", first_instance = %" PRIu32 "\n",
		     args.base_vertex, args.first_instance);
	}
	m_renderer.GetRenderExecutor().DrawIndex(operation.submit_id, CurrentBuffer(), args);
}

const void* CommandProcessor::IndexAddress(uint64_t base, uint32_t type_and_size,
                                           uint32_t first_index) {
	uint64_t index_size = 0;
	switch (type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", type_and_size);
	}
	return reinterpret_cast<const void*>(base + first_index * index_size);
}

const void* CommandProcessor::IndexAddress(uint32_t first_index) const {
	return IndexAddress(m_index_base_addr, m_index_type_and_size, first_index);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	DrawIndex({.index_count = index_count, .index_addr = IndexAddress(index_offset)});
}

// DRAW_INDIRECT(_MULTI) START_VTX_LOC/START_INST_LOC/START_INDEX_LOC: the CP writes each draw's
// base vertex, start instance and first index into the SH registers the packet names before the
// draw, as hardware does. Fast-launch mesh shaders add the start-instance SGPR to their instance
// VGPR themselves (GPU-driven draws pick per-instance records, materials among them, with it).
// KYTY_START_INSTANCE_SGPR=0 restores the old behaviour (the start instance register keeps the
// guest's value).
static void PatchIndirectDrawOffsets(CommandProcessor& cp, IndirectDrawRegisters registers,
                                     uint32_t vertex_offset, uint32_t instance_offset,
                                     uint32_t first_index) {
	static const bool start_instance_enabled = [] {
		const char* value = std::getenv("KYTY_START_INSTANCE_SGPR");
		return value == nullptr || value[0] != '0';
	}();
	const auto write = [&](uint32_t location, uint32_t value) {
		if (location == Pm4::SH_NOP) {
			continue;
		}
		EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM ||
		                     g_hw_sh_indirect_func[location] == nullptr);
		cp.SetUserDataMarker(HW::UserSgprType::Unknown);
		g_hw_sh_indirect_func[location](cp, location, value);
	};
	if (static const bool log = std::getenv("KYTY_LOG_START_INST") != nullptr; log) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 40) {
			std::fprintf(stderr, "[start-inst] cpu reg=0x%x value=%u\n", registers.instance_offset,
			             instance_offset);
		}
	}
	write(registers.vertex_offset, vertex_offset);
	if (start_instance_enabled) {
		write(registers.instance_offset, instance_offset);
	}
	write(registers.index_offset, first_index);
}

// KYTY_MESH_ARGS_RUNS: packet, end: the DRAW_INDIRECT packet and the end of its command buffer.
// Collects the arguments of the non-indexed DRAW_INDIRECTs from this one up to the first packet
// that may write memory, synchronize, change the arguments' base or leave this command buffer.
static void PrepareMeshArgsRun(const CommandProcessor& cp, const uint32_t* packet,
                               const uint32_t* end) {
	auto& run  = MeshArgsRun();
	run.packet = nullptr;
	run.upcoming.clear();
	// KYTY_FPS_LOG: why runs do (not) form, every 5 s.
	static uint64_t calls = 0, no_packet = 0, bad_header = 0, contained = 0, scanned = 0,
	                collected = 0, multi = 0;
	static uint32_t last_header = 0, last_prev = 0, last_stop = 0;
	static auto     last_log    = std::chrono::steady_clock::now();
	if (static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr; log && MeshArgsRunsEnabled()) {
		calls++;
		const auto now = std::chrono::steady_clock::now();
		if (now - last_log >= std::chrono::seconds(5)) {
			last_log = now;
			std::fprintf(stderr,
			             "[mesh-args-scan] calls %llu no-packet %llu bad-header %llu (0x%08x, next "
			             "0x%08x) contained %llu scanned %llu collected %llu runs>1 %llu last-stop "
			             "0x%08x\n",
			             static_cast<unsigned long long>(calls),
			             static_cast<unsigned long long>(no_packet),
			             static_cast<unsigned long long>(bad_header), last_header, last_prev,
			             static_cast<unsigned long long>(contained),
			             static_cast<unsigned long long>(scanned),
			             static_cast<unsigned long long>(collected),
			             static_cast<unsigned long long>(multi), last_stop);
			calls = no_packet = bad_header = contained = scanned = collected = multi = 0;
		}
	}
	if (!MeshArgsRunsEnabled()) {
		return;
	}
	if (packet == nullptr || end == nullptr || packet >= end) {
		no_packet++;
		return;
	}
	if ((packet[0] & ~1u) != 0xc0032400u) {
		bad_header++;
		last_header = packet[0];
		last_prev   = packet + 1 < end ? packet[1] : 0;
		return;
	}
	run.packet = packet;
	if (run.Contains(packet)) {
		contained++;
		return;
	}
	uint64_t base = cp.GetDrawIndirectArgsBaseAddress();
	if (base == 0) {
		return;
	}
	scanned++;
	for (const uint32_t* p = packet; p < end && run.upcoming.size() < 1024;) {
		const uint32_t header = p[0];
		if ((header >> 30u) != 3u || (header & 1u) != 0) {
			break; // not a type-3 packet, or predicated
		}
		const uint32_t op  = (header >> 8u) & 0xffu;
		const uint32_t len = KYTY_PM4_LEN(header);
		if (len < 2 || len > static_cast<uint64_t>(end - p)) {
			break;
		}
		bool go_on = true;
		switch (op) {
			case Pm4::IT_SET_BASE:
				// The arguments' base for the DRAW_INDIRECTs after it (as CpOpSetBase).
				if (len != 4u || (p[1] & 0xfu) != 1u || ((header >> 1u) & 0x3u) > 1u) {
					go_on = false;
				} else if (((header >> 1u) & 0x3u) == 0u) {
					base = (p[2] & ~0x7ull) | (static_cast<uint64_t>(p[3] & 0xffffu) << 32u);
					go_on = base != 0;
				}
				break;
			case Pm4::IT_DRAW_INDIRECT:
				if (header != 0xc0032400u || (p[1] & 3u) != 0) {
					go_on = false;
				} else {
					run.upcoming.push_back({p, base + p[1], {}, 0});
				}
				break;
			// Register state and draws: none writes memory the arguments live in without a guest
			// sync packet before the arguments are read.
			case Pm4::IT_SET_CONTEXT_REG:
			case Pm4::IT_SET_SH_REG:
			case Pm4::IT_SET_UCONFIG_REG:
			case Pm4::IT_SET_UCONFIG_REG_INDEX:
			case Pm4::IT_INDEX_TYPE:
			case Pm4::IT_NUM_INSTANCES:
			case Pm4::IT_INDEX_BASE:
			case Pm4::IT_INDEX_BUFFER_SIZE:
			case Pm4::IT_DRAW_INDEX_AUTO:
			case Pm4::IT_DRAW_INDEX_2:
			case Pm4::IT_DRAW_INDEX_OFFSET_2:
			case Pm4::IT_DRAW_INDEX_INDIRECT:
			// Register loads from memory (they write registers only).
			case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
			case Pm4::IT_SET_SH_REG_INDIRECT:
			case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
			// An end-of-pipe label or interrupt: it orders nothing after it on the console (a
			// later read of what it waited for needs WAIT_REG_MEM or a cache sync, which stop the
			// scan), and its label is not a draw's arguments.
			case Pm4::IT_RELEASE_MEM: break;
			case Pm4::IT_NOP: {
				const auto r = KYTY_PM4_R(header);
				go_on = r == Pm4::R_ZERO || r == Pm4::R_PUSH_MARKER || r == Pm4::R_POP_MARKER ||
				        r == Pm4::R_RELEASE_MEM || r == Pm4::R_CONTEXT_STATE;
				break;
			}
			default: go_on = false; break;
		}
		if (!go_on) {
			last_stop = header;
			if (op == Pm4::IT_EVENT_WRITE && len >= 2) {
				static std::map<uint32_t, uint64_t> events;
				static auto                         events_log = std::chrono::steady_clock::now();
				events[p[1]]++;
				if (const auto now = std::chrono::steady_clock::now();
				    now - events_log >= std::chrono::seconds(5)) {
					events_log = now;
					std::string line = "[mesh-args-scan] EVENT_WRITE stops (payload:count)";
					for (const auto& [payload, count]: events) {
						line += fmt::format(" 0x{:x}:{}", payload, count);
					}
					std::fprintf(stderr, "%s\n", line.c_str());
					events.clear();
				}
			}
			break;
		}
		p += len;
	}
	collected += run.upcoming.size();
	multi += run.upcoming.size() > 1 ? 1 : 0;
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, IndirectDrawRegisters registers,
                                    uint32_t draw_initiator, bool indexed) {
	const auto args_size =
	    indexed ? sizeof(vk::DrawIndexedIndirectCommand) : sizeof(vk::DrawIndirectCommand);
	DrawIndirectMulti(data_offset, 1, nullptr, args_size, registers, draw_initiator, indexed);
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, IndirectDrawRegisters registers,
                                         uint32_t draw_initiator, bool indexed) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != (indexed ? 0u : 2u));
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);
	// The guest records match Vulkan's indirect commands byte for byte.
	EXIT_NOT_IMPLEMENTED((stride_in_bytes & 3u) != 0 ||
	                     stride_in_bytes < (indexed ? sizeof(vk::DrawIndexedIndirectCommand)
	                                                : sizeof(vk::DrawIndirectCommand)));

	auto operation          = MakeOperation(GpuOperationKind::DrawIndirect);
	operation.draw_indirect = {.args_base         = m_draw_indirect_args_base_addr,
	                           .data_offset       = data_offset,
	                           .max_count         = max_count_or_count,
	                           .count_addr        = count_addr,
	                           .stride            = stride_in_bytes,
	                           .draw_initiator    = draw_initiator,
	                           .index_type        = m_index_type_and_size,
	                           .index_buffer_size = m_index_buffer_size,
	                           .index_base        = m_index_base_addr,
	                           .indexed           = indexed,
	                           .multi             = true,
	                           .registers         = registers};
	Emit(std::move(operation));
}

static bool PatchUserSgpr(HW::Shader& shaders, uint32_t location, uint32_t value) {
	const auto in  = [location](uint32_t base) { return location >= base && location < base + 32u; };
	const auto set = [value](const HW::UserSgprInfo& sgprs, uint32_t id, auto&& setter) {
		if (sgprs.value[id] == value) {
			return false;
		}
		setter(id, value);
		return true;
	};
	if (in(Pm4::SPI_SHADER_USER_DATA_GS_0)) {
		return set(shaders.GetVs().gs_user_sgpr, location - Pm4::SPI_SHADER_USER_DATA_GS_0,
		           [&](uint32_t id, uint32_t v) { shaders.SetGsUserSgpr(id, v); });
	}
	if (in(Pm4::SPI_SHADER_USER_DATA_HS_0)) {
		return set(shaders.GetVs().hs_user_sgpr, location - Pm4::SPI_SHADER_USER_DATA_HS_0,
		           [&](uint32_t id, uint32_t v) { shaders.SetHsUserSgpr(id, v); });
	}
	if (in(Pm4::SPI_SHADER_USER_DATA_PS_0)) {
		return set(shaders.GetPs().ps_user_sgpr, location - Pm4::SPI_SHADER_USER_DATA_PS_0,
		           [&](uint32_t id, uint32_t v) { shaders.SetPsUserSgpr(id, v); });
	}
	EXIT("unsupported indirect draw offset register 0x%" PRIx32 "\n", location);
	return false;
}

// Writes the offsets of one draw of an indirect packet to the user SGPRs the packet names, as
// the CP does before the draw. Without a pipeline the registers are the live ones. A pipelined
// operation carries a register snapshot shared with other operations: a copy is patched and
// bound for this draw, and the ahead resolution (made from the unpatched registers) is dropped
// when a value changed. Returns whether the packet names any register.
bool CommandProcessor::PatchIndirectDrawOffsets(const GpuOperation& operation,
                                                uint32_t vertex_offset, uint32_t instance_offset,
                                                uint32_t first_index, uint32_t draw_index) {
	const auto& registers = operation.draw_indirect.registers;
	if (registers.vertex_offset == Pm4::SH_NOP && registers.instance_offset == Pm4::SH_NOP &&
	    registers.index_offset == Pm4::SH_NOP && registers.draw_index == Pm4::SH_NOP) {
		return false;
	}
	const std::pair<uint32_t, uint32_t> writes[] = {{registers.vertex_offset, vertex_offset},
	                                                {registers.instance_offset, instance_offset},
	                                                {registers.index_offset, first_index},
	                                                {registers.draw_index, draw_index}};
	if (m_pipeline == nullptr) {
		for (const auto& [location, value]: writes) {
			if (location == Pm4::SH_NOP) {
				continue;
			}
			EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM ||
			                     g_hw_sh_indirect_func[location] == nullptr);
			g_hw_sh_indirect_func[location](*this, location, value);
		}
		return true;
	}
	if (m_indirect_shaders == nullptr) {
		m_indirect_shaders = std::make_unique<HW::Shader>();
	}
	*m_indirect_shaders = *operation.state.shaders;
	bool changed        = false;
	for (const auto& [location, value]: writes) {
		if (location != Pm4::SH_NOP) {
			changed |= PatchUserSgpr(*m_indirect_shaders, location, value);
		}
	}
	if (changed) {
		m_renderer.GetRenderExecutor().UseResolvedDraw(nullptr);
	}
	GetScheduler().Begin(*operation.state.context, *operation.state.user_config,
	                     *m_indirect_shaders);
	return true;
}

// The arguments are read when the draw executes: earlier GPU work may produce them.
void CommandProcessor::ExecuteDrawIndirect(const GpuOperation& operation) {
	const auto& indirect = operation.draw_indirect;
	if (indirect.max_count == 0) {
		return;
	}
	const auto  args_addr = indirect.args_base + indirect.data_offset;
	const auto& registers = indirect.registers;
	// The GPU reads the records if the shaders allow; the VS prolog loads the patched SGPRs from
	// Vulkan's bases. With no SGPR patched, Vulkan applies the offsets, which matches while they
	// are zero. GPU work (culling shaders) writes the records and the draw count: the host GPU
	// reads the count too (vkCmdDraw*IndirectCount), since reading it here would wait for all the
	// GPU work recorded so far.
	const auto count_address = reinterpret_cast<uint64_t>(indirect.count_addr);
	DrawIndirectArgs records {args_addr, indirect.max_count, indirect.stride};
	records.count_address = count_address;
	if ((count_address == 0 || (m_renderer.GetGraphics().indirect_draws_enabled &&
	                            (count_address & 3u) == 0)) &&
	    IndirectDrawOffsets(registers, records.offsets) &&
	    (!indirect.indexed || indirect.index_buffer_size != 0)) {
		// NUM_INSTANCES keeps the last draw's instance count.
		m_gpu_indirect_instances      = records;
		m_indirect_instances_sequence = operation.num_instances.sequence;
		m_has_indirect_instances      = true;
		NoteRecordedWork();
		auto& executor = m_renderer.GetRenderExecutor();
		if (indirect.indexed
		        ? executor.DrawIndex(operation.submit_id, CurrentBuffer(),
		                             {.index_count         = indirect.index_buffer_size,
		                              .index_addr          = IndexAddress(indirect.index_base,
		                                                                  indirect.index_type, 0),
		                              .index_type_and_size = indirect.index_type,
		                              .indirect            = records})
		        : executor.DrawAuto(operation.submit_id, CurrentBuffer(), {.indirect = records})) {
			Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws,
			                           indirect.max_count);
			return;
		}
	}

	uint32_t draw_count = indirect.max_count;
	if (indirect.count_addr != nullptr) {
		draw_count = *indirect.count_addr;
		if (draw_count > indirect.max_count) {
			draw_count = indirect.max_count;
		}
	}
	m_gpu_indirect_instances = {};
	if (draw_count == 0) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws, draw_count);

	auto&      executor  = m_renderer.GetRenderExecutor();
	const auto args_addr = m_draw_indirect_args_base_addr + data_offset;
	// NUM_INSTANCES keeps the last draw's instance count.
	m_num_instances_addr = args_addr + static_cast<uint64_t>(draw_count - 1) * stride_in_bytes +
	                       offsetof(vk::DrawIndirectCommand, instanceCount);
	// The GPU path below can only hand the start instance to a GS user SGPR; a draw whose packet
	// names a START_VTX_LOC destination, or a start-instance destination outside the GS user
	// data, needs the CP to write those registers from the arguments (PatchIndirectDrawOffsets).
	const bool gpu_args_supported =
	    registers.vertex_offset == Pm4::SH_NOP &&
	    (registers.instance_offset == Pm4::SH_NOP ||
	     (registers.instance_offset >= Pm4::SPI_SHADER_USER_DATA_GS_0 &&
	      registers.instance_offset <= Pm4::SPI_SHADER_USER_DATA_GS_31));
	if (!indexed && count_addr == nullptr && draw_count == 1 && GpuIndirectDraws() &&
	    gpu_args_supported) {
		// The GPU reads the counts (RenderExecutor::DrawAuto and ExecutePreparedDraw): fast-launch
		// mesh draws and vertex-pipeline draws; other mesh draws read them there on the CPU. The
		// arguments are usually written by a dispatch just before; reading them here waited for
		// the whole GPU (6.7 % of the GPU thread in the jungle for the fast-launch draws,
		// 2026-10-03, and 2.9 % for the rest, 2026-10-05).
		m_num_instances_addr = args_addr + offsetof(vk::DrawIndirectCommand, instanceCount);
		// START_INST_LOC: the shader loads the start instance into that user SGPR itself.
		uint32_t start_instance_sgpr = 0;
		if (static const bool enabled = [] {
			    const char* value = std::getenv("KYTY_START_INSTANCE_SGPR");
			    return value == nullptr || value[0] != '0';
		    }();
		    enabled && registers.instance_offset >= Pm4::SPI_SHADER_USER_DATA_GS_0 &&
		    registers.instance_offset <= Pm4::SPI_SHADER_USER_DATA_GS_31) {
			start_instance_sgpr = registers.instance_offset - Pm4::SPI_SHADER_USER_DATA_GS_0 + 1u;
		}
		{
			const uint32_t* run_packet = nullptr;
			const uint32_t* run_end    = nullptr;
			if (g_current_execution != nullptr && !g_current_execution->m_buffer_stack.empty()) {
				const auto& cursor = g_current_execution->m_buffer_stack.back();
				run_packet         = cursor.commands.data() + cursor.offset_dw;
				run_end            = cursor.commands.data() + cursor.commands.size();
			}
			PrepareMeshArgsRun(*this, run_packet, run_end);
		}
		DrawIndexAuto({.offset_source       = DrawOffsetSource::IndirectArgs,
		               .gpu_args_address    = args_addr,
		               .start_instance_sgpr = start_instance_sgpr});
		MeshArgsRun().packet = nullptr;
		return;
	}
	// With no SGPRs to patch, the GPU reads the records if the shaders allow. Shaders apply
	// the records' offsets only through those SGPRs; Vulkan applies them, which matches
	// while they are zero.
	const DrawIndirectArgs records {args_addr, draw_count, stride_in_bytes};
	if (registers.vertex_offset == Pm4::SH_NOP && registers.instance_offset == Pm4::SH_NOP &&
	    registers.index_offset == Pm4::SH_NOP && (!indexed || m_index_buffer_size != 0) &&
	    (indexed ? executor.DrawIndex(m_submit_id, CurrentBuffer(),
	                                  {.index_count         = m_index_buffer_size,
	                                   .index_addr          = IndexAddress(0),
	                                   .index_type_and_size = m_index_type_and_size,
	                                   .indirect            = records})
	             : executor.DrawAuto(m_submit_id, CurrentBuffer(), {.indirect = records}))) {
		return;
	}

	const uint32_t max_indices = m_index_buffer_size != 0 ? m_index_buffer_size : UINT32_MAX;
	for (uint32_t i = 0; i < draw_count; i++) {
		const auto* record_addr = reinterpret_cast<const void*>(
		    args_addr + static_cast<uint64_t>(i) * indirect.stride);

		if (!indirect.indexed) {
			vk::DrawIndirectCommand args {};
			std::memcpy(&args, record_addr, sizeof(args));
			m_indirect_instances          = args.instanceCount;
			m_indirect_instances_sequence = operation.num_instances.sequence;
			m_has_indirect_instances      = true;
			PatchIndirectDrawOffsets(operation, args.firstVertex, args.firstInstance, 0, i);
			ExecuteDrawAuto(operation, {.vertex_count   = args.vertexCount,
			                            .instance_count = args.instanceCount,
			                            .first_vertex   = args.firstVertex,
			                            .first_instance = args.firstInstance,
			                            .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		vk::DrawIndexedIndirectCommand args {};
		std::memcpy(&args, record_addr, sizeof(args));
		PatchIndirectDrawOffsets(operation, static_cast<uint32_t>(args.vertexOffset),
		                         args.firstInstance, args.firstIndex, i);

		const auto*    index_addr = IndexAddress(indirect.index_base, indirect.index_type, args.firstIndex);
		const uint32_t index_count = std::min(args.indexCount, max_indices);
		if (GraphicsRunDebugDumpEnabled() && index_count != args.indexCount) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("\t DrawIndexIndirect: clamped index_count from %" PRIu32 " to %" PRIu32
				     " using INDEX_BUFFER_SIZE\n",
				     args.indexCount, index_count);
			}
		}

		m_indirect_instances          = args.instanceCount;
		m_indirect_instances_sequence = operation.num_instances.sequence;
		m_has_indirect_instances      = true;
		ExecuteDrawIndex(operation, {.index_count         = index_count,
		                             .index_addr          = index_addr,
		                             .instance_count      = args.instanceCount,
		                             .index_type_and_size = indirect.index_type,
		                             .base_vertex    = args.vertexOffset,
		                             .first_instance = args.firstInstance,
		                             .offset_source  = DrawOffsetSource::IndirectArgs});
	}
}

// Profiling (KYTY_FPS_LOG): which queue each compute shader is dispatched from, with its first
// group counts, and the dispatch rate per queue.
static uint64_t g_dispatch_cpu_us[2] {};
// KYTY_PACKET_STATS=1: which dispatch packets run how often (async compute queues), and how many
// compute submissions arrive, every 5 s: a packet that runs hundreds of times a frame is a loop
// in the command stream (a chained indirect buffer that re-runs until a label changes).
static void NotePacket(const uint32_t* packet, bool async, uint64_t shader) {
	static const bool enabled = std::getenv("KYTY_PACKET_STATS") != nullptr;
	if (!enabled || packet == nullptr) {
		return;
	}
	struct Entry {
		uint64_t count  = 0;
		uint64_t shader = 0;
		bool     async  = false;
	};
	static std::unordered_map<const uint32_t*, Entry> counts;
	static auto start = std::chrono::steady_clock::now();
	auto&       e     = counts[packet];
	e.count++;
	e.shader = shader;
	e.async  = async;
	const auto now = std::chrono::steady_clock::now();
	const auto s   = std::chrono::duration<double>(now - start).count();
	if (s < 5.0) {
		return;
	}
	std::vector<std::pair<const uint32_t*, Entry>> list(counts.begin(), counts.end());
	std::sort(list.begin(), list.end(),
	          [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
	std::fprintf(stderr,
	             "[packets] %zu distinct dispatch packets; compute submissions %.0f/s (%.0f dw/s)\n",
	             list.size(), static_cast<double>(g_compute_submissions.exchange(0)) / s,
	             static_cast<double>(g_compute_submission_dw.exchange(0)) / s);
	for (size_t i = 0; i < list.size() && i < 10; i++) {
		std::fprintf(stderr, "[packets]   %p %s shader %016llx %.0f/s\n",
		             static_cast<const void*>(list[i].first), list[i].second.async ? "async" : "gfx",
		             static_cast<unsigned long long>(list[i].second.shader),
		             static_cast<double>(list[i].second.count) / s);
	}
	counts.clear();
	start = now;
}

static void NoteDispatchQueue(uint64_t addr, bool async, uint32_t x, uint32_t y, uint32_t z) {
	static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr;
	if (!log) {
		return;
	}
	static std::mutex                             mutex;
	static std::unordered_map<uint64_t, uint32_t> seen;
	static auto                                   start = std::chrono::steady_clock::now();
	static uint64_t                               counts[2] {};
	std::lock_guard lock(mutex);
	if (x == UINT32_MAX) {
		// CPU time of the previous dispatch's recording, in microseconds (z), on queue `async`.
		g_dispatch_cpu_us[async ? 1 : 0] += z;
		return;
	}
	counts[async ? 1 : 0]++;
	auto& n = seen[addr ^ (async ? 1ull : 0ull)];
	if (n++ < 2) {
		std::fprintf(stderr, "[dq] addr=0x%016" PRIx64 " async=%d groups=%ux%ux%u\n", addr,
		             async ? 1 : 0, x, y, z);
	}
	const auto now = std::chrono::steady_clock::now();
	const auto s   = std::chrono::duration<double>(now - start).count();
	if (s >= 5.0) {
		std::fprintf(stderr, "[dq-rate] gfx=%.0f/s async=%.0f/s cpu_ms_per_s gfx=%.0f async=%.0f\n",
		             counts[0] / s, counts[1] / s, g_dispatch_cpu_us[0] / 1000.0 / s,
		             g_dispatch_cpu_us[1] / 1000.0 / s);
		counts[0] = counts[1] = 0;
		g_dispatch_cpu_us[0] = g_dispatch_cpu_us[1] = 0;
		start                 = now;
	}
}

// KYTY_DISPATCH_RUNS (see dispatchRun.h): the dispatch packet the previous dispatch's run continues
// with, if the stream reaches it with nothing but SET_SH_REG and debug markers in between
// (g_run_expected_next, declared at the top).

static const uint32_t* NextRunDispatch(const uint32_t* p, const uint32_t* end,
                                       std::array<uint32_t, 3>* local) {
	while (p < end) {
		const uint32_t header = p[0];
		if (header == 0x80000000u) {
			p++;
			continue;
		}
		if ((header >> 30u) != 3u || (header & 1u) != 0) {
			return nullptr; // not a type-3 packet, or predicated
		}
		const uint32_t op  = (header >> 8u) & 0xffu;
		const uint32_t len = KYTY_PM4_LEN(header);
		if (len < 2 || len > static_cast<uint64_t>(end - p)) {
			return nullptr;
		}
		if (op == Pm4::IT_DISPATCH_DIRECT || op == Pm4::IT_DISPATCH_INDIRECT) {
			return p;
		}
		if (op == Pm4::IT_SET_SH_REG) {
			if (local != nullptr) {
				const uint32_t reg = p[1] & 0xffffu;
				for (uint32_t i = 0; i + 2 < len; i++) {
					const uint32_t r = reg + i;
					if (r >= 0x207u && r <= 0x209u) {
						(*local)[r - 0x207u] = std::max(p[2 + i], 1u);
					}
				}
			}
			p += len;
			continue;
		}
		if (op == Pm4::IT_NOP) {
			const auto r = KYTY_PM4_R(header);
			if (r == Pm4::R_ZERO || r == Pm4::R_PUSH_MARKER || r == Pm4::R_POP_MARKER) {
				p += len;
				continue;
			}
		}
		return nullptr;
	}
	return nullptr;
}

// packet, end: the dispatch packet and the end of its command buffer (nullptr: unknown).
static void PrepareDispatchRun(CommandProcessor& cp, bool indirect, const uint32_t* packet,
                               const uint32_t* end) {
	auto& run     = DispatchRun();
	run.packet    = nullptr;
	run.continues = false;
	run.more      = false;
	if (!DispatchRunsEnabled() || packet == nullptr) {
		g_run_expected_next = nullptr;
		return;
	}
	run.packet             = packet;
	run.continues          = packet == g_run_expected_next;
	if (!run.continues && !run.converted.empty()) {
		// Converted counts belong to the run instance that converted them. A command buffer the
		// game submits again (or a run resumed after KYTY_SRT_DEFER) reaches the same packets
		// later: their entries then name ring slots later conversions rewrote, and counts of
		// arguments since rewritten (a refit then ran with the wrong number of groups).
		// KYTY_RUN_CONVERTED_KEEP=1: keep them, as before (the count of such starts is still logged).
		static const bool keep = [] {
			const char* value = std::getenv("KYTY_RUN_CONVERTED_KEEP");
			return value != nullptr && value[0] == '1';
		}();
		if (run.converted.contains(packet)) {
			run.stale_hits++;
		}
		if (!keep) {
			run.converted.clear();
		}
	}
	// KYTY_FPS_LOG: run lengths by the shader of each dispatch (addr -> {dispatches, runs started}).
	// (getenv once: on Windows each call locks and scans the environment, ~1 us a dispatch.)
	static const bool run_log = std::getenv("KYTY_FPS_LOG") != nullptr;
	if (run_log) {
		static std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> by_shader;
		static uint64_t                                                   run_len = 0;
		static auto last = std::chrono::steady_clock::now();
		auto&       e    = by_shader[cp.GetShCtx().GetCs().cs_regs.data_addr];
		e.first++;
		if (!run.continues) e.second++;
		const auto now = std::chrono::steady_clock::now();
		if (now - last > std::chrono::seconds(5)) {
			last = now;
			std::vector<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> list(by_shader.begin(),
			                                                                      by_shader.end());
			std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) {
				return a.second.first > b.second.first;
			});
			for (size_t i = 0; i < list.size() && i < 8; i++) {
				std::fprintf(stderr, "[dispatch-runs]   cs 0x%llx dispatches %llu, runs started %llu\n",
				             static_cast<unsigned long long>(list[i].first),
				             static_cast<unsigned long long>(list[i].second.first),
				             static_cast<unsigned long long>(list[i].second.second));
			}
			by_shader.clear();
		}
		(void)run_len;
	}
	run.runs += run.continues ? 0 : 1;
	run.members++;
	const uint32_t* next = NextRunDispatch(packet + KYTY_PM4_LEN(packet[0]), end, nullptr);
	run.more             = next != nullptr;
	g_run_expected_next  = next;
	if (!indirect || run.converted.contains(packet)) {
		return;
	}
	run.upcoming.clear();
	std::array<uint32_t, 3> local {0, 0, 0};
	for (const uint32_t* p = packet; p != nullptr && run.upcoming.size() < 1024;) {
		const uint32_t header = p[0] & ~1u;
		if (((header >> 8u) & 0xffu) == Pm4::IT_DISPATCH_INDIRECT) {
			uint64_t args = 0;
			uint32_t mode = 0;
			if (header == 0xc0021600u) {
				args = p[1] | (static_cast<uint64_t>(p[2]) << 32u);
				mode = p[3];
			} else if (header == 0xc0011600u) {
				const auto base = cp.GetDispatchIndirectArgsBaseAddress();
				if (base == 0) break;
				args = base + p[1];
				mode = p[2];
			} else {
				break;
			}
			if ((mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0 && args != 0 &&
			    (args & 3u) == 0) {
				run.upcoming.push_back({p, args, local});
			}
		}
		p = NextRunDispatch(p + KYTY_PM4_LEN(header), end, &local);
	}
}

static void ReportDispatchRuns() {
	static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr;
	if (!log || !DispatchRunsEnabled()) return;
	static auto last = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) return;
	const double s   = std::chrono::duration<double>(now - last).count();
	last             = now;
	auto& run        = DispatchRun();
	std::fprintf(stderr,
	             "[dispatch-runs] runs %.0f/s dispatches %.0f/s barriers left out %.0f/s indirect "
	             "counts converted in batches %.0f/s, stale conversions dropped %llu\n",
	             static_cast<double>(run.runs) / s, static_cast<double>(run.members) / s,
	             static_cast<double>(run.skipped_barriers) / s, static_cast<double>(run.batched) / s,
	             static_cast<unsigned long long>(run.stale_hits));
	run.runs = run.members = run.skipped_barriers = run.batched = 0;
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	NoteDispatchQueue(m_sh_ctx.GetCs().cs_regs.data_addr, IsAsyncComputeQueue(), thread_group_x,
	                  thread_group_y, thread_group_z);
	NotePacket((g_current_execution != nullptr && !g_current_execution->m_buffer_stack.empty()
	               ? g_current_execution->m_buffer_stack.back().commands.data() +
	                     g_current_execution->m_buffer_stack.back().offset_dw
	               : nullptr), IsAsyncComputeQueue(), m_sh_ctx.GetCs().cs_regs.data_addr);

void CommandProcessor::ExecuteDispatchDirect(const GpuOperation& operation, uint32_t thread_group_x,
                                             uint32_t thread_group_y, uint32_t thread_group_z,
                                             uint32_t mode) {
	NoteRecordedWork();

	if (GraphicsRunDebugDumpEnabled()) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
			const auto  frame_num = m_renderer.GetGpu().GetFrameNum();
			const auto& ucfg      = *operation.state.user_config;
			const auto& cs        = operation.state.shaders->GetCs().cs_regs;
			const auto& oa        = ucfg.GetGdsOaCounter(ucfg.GetGdsOaState().GetIndex());
			LOGF("QueuePoint DispatchDirect: frame=%u submit=%" PRIu64
			     " groups=%ux%ux%u local=%ux%ux%u mode=0x%08" PRIx32 " wave=%u cs=0x%016" PRIx64
			     " oa_index=%u oa_enabled=%s oa_addr=0x%04" PRIx32 " oa_space=0x%08" PRIx32 "\n",
			     frame_num, operation.submit_id, thread_group_x, thread_group_y, thread_group_z,
			     std::max(cs.num_thread_x, 1u), std::max(cs.num_thread_y, 1u),
			     std::max(cs.num_thread_z, 1u), mode, static_cast<uint32_t>(cs.wave_size),
			     cs.data_addr, ucfg.GetGdsOaState().GetIndex(),
			     oa.IsCounterEnabled() ? "true" : "false", oa.GetAddressBytes(),
			     oa.GetSpaceAvailable());
		}

		const auto cpu_begin = std::chrono::steady_clock::now();
		if (BeginDispatchDefer()) {
			return;
		}
		{
			const uint32_t* run_packet = nullptr;
			const uint32_t* run_end    = nullptr;
			if (g_current_execution != nullptr && !g_current_execution->m_buffer_stack.empty()) {
				const auto& cursor = g_current_execution->m_buffer_stack.back();
				run_packet         = cursor.commands.data() + cursor.offset_dw;
				run_end            = cursor.commands.data() + cursor.commands.size();
			}
			PrepareDispatchRun(*this, false, run_packet, run_end);
		}
		ReportDispatchRuns();
		m_renderer.GetRenderExecutor().DispatchDirect(m_submit_id, CurrentBuffer(), thread_group_x,
		                                              thread_group_y, thread_group_z, mode, IsAsyncComputeQueue());
		EndDispatchDefer();
		NoteDispatchQueue(0, IsAsyncComputeQueue(), UINT32_MAX, 0,
		                  static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(
		                                            std::chrono::steady_clock::now() - cpu_begin)
		                                            .count()));
	}

	m_renderer.GetRenderExecutor().DispatchDirect(operation.submit_id, CurrentBuffer(),
	                                              thread_group_x, thread_group_y, thread_group_z,
	                                              mode, IsAsyncComputeQueue());
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	// Thread counts are converted on the GPU. Reading them here drained the GPU for every
	// dispatch, since the previous dispatch writes them; KYTY_INDIRECT_THREADS_ON_CPU=1 restores
	// that path for A/B runs.
	static const bool threads_on_cpu = std::getenv("KYTY_INDIRECT_THREADS_ON_CPU") != nullptr;
	if (threads_on_cpu && (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		const auto* args = reinterpret_cast<const vk::DispatchIndirectCommand*>(args_addr);
		DispatchDirect(args->x, args->y, args->z, mode);
		return;
	}
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	NoteDispatchQueue(m_sh_ctx.GetCs().cs_regs.data_addr, IsAsyncComputeQueue(), 0, 0, 0);
	// KYTY_INDIRECT_STATS=1: the group counts indirect dispatches had, per shader, from copies of
	// their arguments that never make the GPU thread wait (a frame or more old), every 5 s.
	if (static const bool stats = std::getenv("KYTY_INDIRECT_STATS") != nullptr; stats) {
		struct Stat {
			uint64_t n = 0, zero = 0, groups = 0, seen = 0, max = 0;
		};
		static std::unordered_map<uint64_t, Stat> by_shader;
		static auto start = std::chrono::steady_clock::now();
		auto&       st    = by_shader[m_sh_ctx.GetCs().cs_regs.data_addr];
		st.n++;
		const uint32_t x = Libs::LibKernel::Memory::PeekGpuDword(args_addr);
		const uint32_t y = Libs::LibKernel::Memory::PeekGpuDword(args_addr + 4);
		const uint32_t z = Libs::LibKernel::Memory::PeekGpuDword(args_addr + 8);
		if (x != 0 || y != 0 || z != 0 || st.seen != 0) {
			st.seen++;
			const uint64_t g = uint64_t {x} * std::max(y, 1u) * std::max(z, 1u);
			st.zero += (x == 0 || y == 0 || z == 0) ? 1 : 0;
			st.groups += (x == 0 || y == 0 || z == 0) ? 0 : g;
			st.max = std::max(st.max, (x == 0 || y == 0 || z == 0) ? 0 : g);
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - start > std::chrono::seconds(5)) {
			start = now;
			for (const auto& [addr, e]: by_shader) {
				if (e.n < 100) continue;
				std::fprintf(stderr,
				             "[indirect] cs 0x%llx dispatches=%llu sampled=%llu zero-group=%llu "
				             "avg-groups=%.1f max=%llu (mode 0x%x)\n",
				             static_cast<unsigned long long>(addr),
				             static_cast<unsigned long long>(e.n),
				             static_cast<unsigned long long>(e.seen),
				             static_cast<unsigned long long>(e.zero),
				             e.seen > e.zero ? static_cast<double>(e.groups) / (e.seen - e.zero) : 0.0,
				             static_cast<unsigned long long>(e.max), mode);
			}
			by_shader.clear();
		}
	}
	NotePacket((g_current_execution != nullptr && !g_current_execution->m_buffer_stack.empty()
	               ? g_current_execution->m_buffer_stack.back().commands.data() +
	                     g_current_execution->m_buffer_stack.back().offset_dw
	               : nullptr), IsAsyncComputeQueue(), m_sh_ctx.GetCs().cs_regs.data_addr);
	const auto cpu_begin = std::chrono::steady_clock::now();
	if (BeginDispatchDefer()) {
		return;
	}
	{
		const uint32_t* run_packet = nullptr;
		const uint32_t* run_end    = nullptr;
		if (g_current_execution != nullptr && !g_current_execution->m_buffer_stack.empty()) {
			const auto& cursor = g_current_execution->m_buffer_stack.back();
			run_packet         = cursor.commands.data() + cursor.offset_dw;
			run_end            = cursor.commands.data() + cursor.commands.size();
		}
		PrepareDispatchRun(*this, (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0,
		                   run_packet, run_end);
	}
	ReportDispatchRuns();
	m_renderer.GetRenderExecutor().DispatchIndirect(m_submit_id, CurrentBuffer(), args_addr, mode,
	                                                IsAsyncComputeQueue());
	EndDispatchDefer();
	NoteDispatchQueue(0, IsAsyncComputeQueue(), UINT32_MAX, 0,
	                  static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                            std::chrono::steady_clock::now() - cpu_begin)
	                                            .count()));
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	if (args.instance_count == 0 && args.gpu_args_address == 0) {
		if (m_num_instances_addr != 0 && GpuIndirectDraws()) {
			// The instance count a DRAW_INDIRECT read on the GPU left: the GPU reads it too
			// (RenderExecutor::DrawAuto). Reading it here drained the GPU (2.35 % of the GPU
			// thread at the jungle's Leap Attack prompt, 2026-10-05). Read at this draw, not
			// latched at the indirect draw: the arguments are not rewritten in between.
			args.gpu_instance_count_address = m_num_instances_addr;
		} else {
			args.instance_count = NumInstances();
		}
	}
	m_renderer.GetRenderExecutor().DrawAuto(m_submit_id, CurrentBuffer(), args);
	CompleteDraw();
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	// Stall only this command processor, not the host command lane or other queues.
	if (!m_renderer.GetVideoOut().IsSafeForRendering(static_cast<int>(video_out_handle),
	                                                 static_cast<int>(display_buffer_index))) {
		SuspendPm4();
		return;
	}
	// Flips before this packet that the execution thread has not run yet can still make the
	// buffer busy, so the execution thread waits for it too, in operation order.
	EmitCallback([this, video_out_handle, display_buffer_index] {
		GetScheduler().Flush();
		auto& video_out = m_renderer.GetVideoOut();
		for (Common::SpinWait wait; !video_out.IsSafeForRendering(static_cast<int>(video_out_handle),
		                                                          static_cast<int>(display_buffer_index));) {
			if (!wait.Spin()) {
				std::this_thread::yield();
			}
		}
	});
}

void CommandProcessor::ReleaseMem(uint32_t data_sel, uint32_t interrupt_selector, void* address,
                                  uint64_t data, uint32_t interrupt_context_id) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::ReleaseMem()\n"
		     "\t data_sel           = %" PRIu32 "\n"
		     "\t interrupt_selector = %" PRIu32 "\n"
		     "\t address            = 0x%016" PRIx64 "\n"
		     "\t data               = 0x%016" PRIx64 "\n"
		     "\t interrupt_context  = 0x%08" PRIx32 "\n",
		     data_sel, interrupt_selector, reinterpret_cast<uint64_t>(address), data,
		     interrupt_context_id);
	}

	// QueuedInterrupt: kNone, kOnly, kAfterWriteConfirm, kDataAfterWriteConfirm, kSendIntCtx.
	EXIT_NOT_IMPLEMENTED(interrupt_selector > 4);
	const bool interrupt = (interrupt_selector != 0 && interrupt_selector != 3);
	// kSendIntCtx has no label, and kOnly writes none on graphics queues.
	if (address == nullptr || interrupt_selector == 4 ||
	    (interrupt_selector == 1 && !IsAsyncComputeQueue())) {
		data_sel = 0;
	}
	if (data_sel == 0 && !interrupt) {
		return;
	}

	const auto sequence = EmitCallback([=, this] {
		ExecuteReleaseMem(data_sel, interrupt, address, data, interrupt_context_id);
	});
	// What ExecuteReleaseMem() writes: the value, or the clock and GDS dwords, known only then.
	const auto destination = reinterpret_cast<uint64_t>(address);
	switch (data_sel) {
		case 1: {
			const auto value = static_cast<uint32_t>(data);
			NoteGuestWrite(sequence, destination, sizeof(value), &value);
			break;
		case 0x02: with_interrupt = true; break;
		default: EXIT("unknown interrupt_selector\n");
	}

	const bool at_completion = !with_interrupt && GuestGpu::LabelsAtCompletion();

	auto write32 = [&] {
		auto* dst  = static_cast<uint32_t*>(dst_gpu_addr);
		auto  data = static_cast<uint32_t>(value);
		EXIT_IF(dst == nullptr);
		(void)command.Handle();
		if (at_completion) {
			m_renderer.GetGpu().DeferLabelWrite(reinterpret_cast<uint64_t>(dst), data,
			                                    sizeof(data));
		} else {
			std::memcpy(dst, &data, sizeof(data));
		}

		if (with_interrupt) {
			TriggerEopEventAtEndOfPipe(interrupt_context_id);
		}
	};

	switch (event_write_source) {
		case 0x01:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && cache_action == 0x00 && event_index == 0x06) {
					auto* dst = static_cast<uint32_t*>(dst_gpu_addr);
					SynchronizeGpu();
					Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(), dst, value & 0xffffu,
					              value >> 16u);
					if (with_interrupt) {
						m_renderer.TriggerInterrupt(m_interrupt_event_id, interrupt_context_id);
					}
					return;
				}
			} else if (eop_event_type == 0x04 && cache_action == 0x00 && event_index == 0x05) {
				write32();
				return;
			}
			break;
		case 0x02:
		case 0x04:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (event_write_source == 0x02 && eop_event_type == 0x2f && event_index == 0x06) {
					switch (cache_action) {
						case 0x00:
						case 0x38: write32(); return;
						default: break;
					}
				}
			} else {
				if (event_write_source == 0x04) {
					value = Sync::ReadReferenceClock();
				}
				auto write64 = [&] {
					auto* dst = static_cast<uint64_t*>(dst_gpu_addr);
					EXIT_IF(dst == nullptr);
					(void)command.Handle();
					if (at_completion) {
						m_renderer.GetGpu().DeferLabelWrite(reinterpret_cast<uint64_t>(dst),
						                                    value, sizeof(value));
					} else {
						std::memcpy(dst, &value, sizeof(value));
					}

					if (with_interrupt) {
						TriggerEopEventAtEndOfPipe(interrupt_context_id);
					}
				};

				switch (cache_action) {
					case 0x00:
						switch (eop_event_type) {
							case 0x04:
								if (event_index == 0x05) {
									write64();
									return;
								}
								break;
							case 0x14:
							case 0x28:
							case 0x2f:
								if (event_index == 0x00) {
									write64();
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
							case 0x30:
								if (event_index == 0x00 && !with_interrupt) {
									write64();
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x38:
						switch (eop_event_type) {
							case 0x04:
							case 0x14:
							case 0x28:
								if (((eop_event_type == 0x04 || eop_event_type == 0x28) &&
								     event_index == 0x05) ||
								    (event_index == 0x00)) {
									write64();
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
								if (event_index == 0x00 && !with_interrupt) {
									write64();
									return;
								}
								break;
							case 0x2f:
								if (event_index == 0x06 && !with_interrupt) {
									write64();
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x3b:
						if (eop_event_type == 0x04 && event_index == 0x05 && with_interrupt) {
							write64();
							return;
						}
						break;
					default: break;
				}
			}
			break;
		default: break;
	}
}

void CommandProcessor::WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint32_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint64_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::EmitGlobalBarrier(std::source_location site) {
	Common::LockGuard lock(m_renderer.GetMutex());

	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	if (BarrierSinkEnabled()) {
		// Rendering ends there only if the barrier cannot stay pending (barrierSink.h).
		CurrentBuffer().Recorder().guestPipelineBarrier2(dependency, site);
		return;
	}
	GetScheduler().EndRendering();
	CurrentBuffer().Recorder().pipelineBarrier2(dependency, site);
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	(void)CurrentBuffer().Handle();

	// A label beside GPU-written bytes goes to the backing store, so they are not downloaded.
	// Other labels go through the guest mapping: its fault keeps a clean cached copy current and
	// downloads GPU-written bytes under the label before the store.
	const auto write_label = [this, address](const void* value, uint64_t size) {
		const auto vaddr   = reinterpret_cast<uint64_t>(address);
		auto&      buffers = m_renderer.GetBufferCache();
		if (!GuestRange {vaddr, size}.Valid() || !buffers.IsRegionGpuModified(vaddr, size) ||
		    buffers.HasGpuDirtyBytes(vaddr, size) ||
		    !LibKernel::Memory::TryWriteBacking(vaddr, value, size)) {
			m_renderer.PrepareGpuWrite(address, size);
			std::memcpy(address, value, size);
		}
	};
	// EndOfPipeData: kNone, kImmediate32b, kImmediate64b, kRefClk; 5 reads GDS.
	switch (data_sel) {
		case 0: break;
		case 1: {
			const auto value = static_cast<uint32_t>(data);
			write_label(&value, sizeof(value));
			break;
		}
		case 3: data = Sync::ReadReferenceClock(); [[fallthrough]];
		case 2: write_label(&data, sizeof(data)); break;
		case 5: {
			GetScheduler().Finish();
			const auto gds = static_cast<uint32_t>(data);
			Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(),
			              static_cast<uint32_t*>(address), gds & 0xffffu, gds >> 16u);
			break;
		}
		default: EXIT("unknown release_mem data selector: %" PRIu32 "\n", data_sel);
	}

	// GuestGpu::Process's end-of-slice flush submits the work this interrupt waits on.
	if (interrupt) {
		DeferEopInterrupt(interrupt_context_id);
	}
}

void CommandProcessor::DeferEopInterrupt(uint32_t interrupt_context_id) {
	(void)CurrentBuffer().Handle();
	GetScheduler().DeferHostOperation(
	    [&renderer = m_renderer, event_id = m_interrupt_event_id, interrupt_context_id] {
		    renderer.TriggerInterrupt(event_id, interrupt_context_id);
	    },
	    true);
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto sequence = EmitCallback([this, event_type, event_index, event_address] {
		ExecuteTriggerEvent(event_type, event_index, event_address);
	});
	if (event_type == 0x00000039) {
		// Occlusion counter pairs of 16 DBs.
		NoteGuestWrite(sequence, event_address, 16u * 2u * sizeof(uint64_t));
	}
}

void CommandProcessor::ExecuteTriggerEvent(uint32_t event_type, uint32_t event_index,
                                           uint64_t event_address) {
	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	// Partial flushes and cache writebacks only order GPU work, which the renderer's resource
	// barriers already order.
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate, DbDataWritebackInvalidate,
		// DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			auto& queries = m_renderer.GetOcclusionQueries();
			if (queries.Enabled()) {
				queries.Dump(GetScheduler(), event_address);
				break;
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are treated as always "
				            "visible without --occlusion-queries; GPU usage may be higher and FPS "
				            "may be lower.\n");
			});

			// Without host occlusion queries, publish an always-visible result. The PS5 layout
			// contains one interleaved begin/end pair per DB, and bit 63 marks a result ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			// Profiling: KYTY_OCCLUSION_ZERO=1 reports every query as occluded (no samples), the
			// upper bound of what real occlusion results could cull. KYTY_FPS_LOG prints the rate.
			static const bool occlusion_zero = [] {
				const char* v = std::getenv("KYTY_OCCLUSION_ZERO");
				return v != nullptr && v[0] == '1';
			}();
			if (!occlusion_zero) {
				m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			}
			if (static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr; log) {
				static auto     start = std::chrono::steady_clock::now();
				static uint64_t dumps = 0;
				dumps++;
				const auto now = std::chrono::steady_clock::now();
				const auto s   = std::chrono::duration<double>(now - start).count();
				if (s >= 5.0) {
					std::fprintf(stderr, "[occl] %.1f dumps/s\n", static_cast<double>(dumps) / s);
					dumps = 0;
					start = now;
				}
			}
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::WriteLodStats(void* dst, uint32_t size) {
	const auto sequence = EmitCallback([dst, size] {
		memset(dst, 0, size);
		// Hack?
		if (size >= sizeof(uint32_t)) {
			auto* label = static_cast<uint32_t*>(dst);
			*label      = 1;
		}
	});
	NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst), size);
}

void CommandProcessor::Flip(int handle, int index, int flip_mode, int64_t flip_arg) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}

	auto& command = CurrentBuffer();
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	(void)command.Handle();
	GetScheduler().DeferPriorityOperation(
	    [&renderer = m_renderer, request] { renderer.GetVideoOut().CompleteFlip(request); });
	(KYTY_SUBMIT_REASON(), GetScheduler().Flush());
}

void CommandProcessor::Flip(void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n"
		     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
		     "\t value        = 0x%08" PRIx32 "\n",
		     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	EXIT_IF(dst_gpu_addr == nullptr);
	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	(void)command.Handle();
	GetScheduler().DeferPriorityOperation(
	    [&renderer = m_renderer, request] { renderer.GetVideoOut().CompleteFlip(request); });
	(KYTY_SUBMIT_REASON(), GetScheduler().Flush());
}

void CommandProcessor::FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action,
                                         void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::FlipWithInterrupt()\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%08" PRIx32 "\n",
		     eop_event_type, cache_action, reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	if (eop_event_type != 0x00000004 || cache_action != 0x00000038) {
		EXIT("unknown event type\n");
	}
	EXIT_IF(dst_gpu_addr == nullptr);
	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	(void)command.Handle();
	GetScheduler().DeferPriorityOperation(
	    [&renderer = m_renderer, event_id = m_interrupt_event_id, request] {
		    renderer.GetVideoOut().CompleteFlip(request);
		    renderer.TriggerInterrupt(event_id, 0);
	    });
	(KYTY_SUBMIT_REASON(), GetScheduler().Flush());
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	EmitCallback([this, request_id] {
		struct ProcessorScope {
			explicit ProcessorScope(CommandProcessor& processor) {
				g_current_processor = &processor;
			}
			~ProcessorScope() { g_current_processor = nullptr; }
		};
		ProcessorScope processor_scope(*this);

	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	GetScheduler().DeferPriorityOperation(
	    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
	(KYTY_SUBMIT_REASON(), GetScheduler().Flush());
}

void CommandProcessor::SynchronizeGpu() {
	(KYTY_SUBMIT_REASON(), GetScheduler().Finish());
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}

} // namespace Libs::Graphics
