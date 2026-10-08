#ifndef GRAPHICS_GUEST_GPU_EXEC_TRACE_H
#define GRAPHICS_GUEST_GPU_EXEC_TRACE_H

// Debug (KYTY_EXEC_TRACE=<path prefix>): a timeline of the operations the execution thread runs,
// written every 5 s to <prefix>.<n % 8>.bin, for offline analysis of how much the queues could
// overlap.

#include <chrono>
#include <cstdint>

namespace Libs::Graphics::ExecTrace {

struct Record {
	uint64_t sequence;
	uint64_t emit_ns;   // command processor emitted it
	uint64_t start_ns;  // execution thread started it
	uint64_t end_ns;    // ... and finished it
	uint64_t gpu_wait_ns;   // inside: waits for GPU work (semaphores, readback fences)
	uint64_t other_wait_ns; // inside: spins for the recording thread, priority operations
	uint64_t dep_sequence;  // a WAIT_REG_MEM before it was satisfied by this operation's write
	uint64_t cs_addr;       // dispatches: shader address
	uint32_t queue;
	uint8_t  kind;
	uint8_t  after_wait;
	uint16_t flushes;       // command buffers submitted inside
	uint64_t wait_tick_behind; // largest (current tick - waited tick) of a GPU wait inside
	uint64_t shader_hash;      // dispatches: program hash
	uint64_t resolve_start_ns; // resolve thread popped it
	uint64_t resolve_end_ns;   // ... and forwarded it
};

bool Enabled() noexcept;
uint64_t NowNs() noexcept;

// Execution thread accumulators.
extern thread_local uint64_t g_gpu_wait_ns;
extern thread_local uint64_t g_other_wait_ns;
extern thread_local uint32_t g_flushes;
extern thread_local uint64_t g_wait_tick_behind;

void Add(const Record& record) noexcept;

// The operation the execution thread runs, and the label writes it makes (for the command
// processor's WAIT_REG_MEM that read memory instead of the journal).
extern thread_local uint64_t g_current_sequence;
extern thread_local uint64_t g_shader_hash;
void     NoteLabelWrite(uint64_t address) noexcept;
// Phase profiler hook: time of each phase inside the current operation (slow operations are
// written to <prefix>.slow.txt with their phases).
void OnPhase(const char* name, uint64_t ns) noexcept;
void BeginOperation() noexcept;
void EndOperation(const Record& record) noexcept;
uint64_t LastLabelWriter(uint64_t address) noexcept;

class GpuWaitScope {
public:
	GpuWaitScope() noexcept: m_start(Enabled() ? NowNs() : 0) {}
	~GpuWaitScope() {
		if (m_start != 0) {
			g_gpu_wait_ns += NowNs() - m_start;
		}
	}
	GpuWaitScope(const GpuWaitScope&)            = delete;
	GpuWaitScope& operator=(const GpuWaitScope&) = delete;

private:
	uint64_t m_start;
};

class OtherWaitScope {
public:
	OtherWaitScope() noexcept: m_start(Enabled() ? NowNs() : 0) {}
	~OtherWaitScope() {
		if (m_start != 0) {
			g_other_wait_ns += NowNs() - m_start;
		}
	}
	OtherWaitScope(const OtherWaitScope&)            = delete;
	OtherWaitScope& operator=(const OtherWaitScope&) = delete;

private:
	uint64_t m_start;
};

} // namespace Libs::Graphics::ExecTrace

#endif
