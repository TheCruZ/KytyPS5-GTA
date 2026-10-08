#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUPROFILER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUPROFILER_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace Libs::Graphics {

struct GraphicContext;

// Opt-in GPU/CPU-sync profiler (KYTY_GPU_PROFILE=1). Records GPU timestamps around each
// draw/dispatch and around each submitted command buffer, and accumulates the time host
// threads block on the GPU. Every few seconds it prints GPU busy time, the most expensive
// shaders and the CPU waits to stdout.
namespace GpuProfiler {

bool Enabled() noexcept;
// Command-buffer timestamps only (KYTY_GPU_PROFILE=cbonly turns on these but not Enabled()).
bool CbEnabled() noexcept;

// Command-buffer lifetime hooks (scheduler command buffers only).
void BeginCommandBuffer(GraphicContext& graphics, vk::CommandBuffer command);
// Returns an opaque slot id to pass to Resolve() once the command buffer completed, or
// UINT32_MAX when nothing was recorded.
uint32_t EndCommandBuffer(vk::CommandBuffer command);
void     Resolve(GraphicContext& graphics, uint32_t slot, uint64_t tick = 0, uint64_t scheduler = 0);

// Per-operation marks, called next to the device-lost checkpoints.
void Mark(vk::CommandBuffer command, GpuCheckpointKind kind, uint64_t hash_a, uint64_t hash_b);
// Other GPU work: a named begin/end region (transfers, detiling, clears).
void Region(vk::CommandBuffer command, bool begin, const char* name);
// Command recorders: only direct recording is profiled.
template <typename Recorder>
    requires(!std::is_same_v<Recorder, vk::CommandBuffer>)
void Mark(const Recorder& command, GpuCheckpointKind kind, uint64_t hash_a, uint64_t hash_b) {
	if (command.IsDirect()) {
		Mark(command.Direct(), kind, hash_a, hash_b);
	}
}
template <typename Recorder>
    requires(!std::is_same_v<Recorder, vk::CommandBuffer>)
void Region(const Recorder& command, bool begin, const char* name) {
	if (command.IsDirect()) {
		Region(command.Direct(), begin, name);
	}
}

// Host waits.
// Tags are string literals; AddWaitString copies a formatted tag.
void AddWait(const char* tag, uint64_t ns) noexcept;
void AddWaitString(const char* tag, uint64_t ns) noexcept;
void AddCount(const char* tag, uint64_t count = 1) noexcept;
void AddFrame() noexcept;
// Attribution of GPU-written guest ranges to the shader that wrote them.
void     SetCurrentShader(uint64_t hash) noexcept;
void     NoteGpuWrite(uint64_t address, uint64_t size) noexcept;
uint64_t LastGpuWriter(uint64_t address) noexcept;
// Histogram of hot guest addresses (printed as the top entries per tag).
void AddAddress(const char* tag, uint64_t address, uint64_t value = 1) noexcept;

const char* CurrentWaitTag() noexcept;

// KYTY_PHASE_PROFILE=1: linear phase timing of the calling thread. Each mark closes the
// previous phase; a report of ms/frame, calls/frame and average us per phase prints every 5 s,
// with frame-time percentiles and the host events (waits, compiles, uploads, faults). Frames
// slower than KYTY_HITCH_MS (default 50) print their own phases, events and cache counters.
inline const bool g_phase_profile_enabled = std::getenv("KYTY_PHASE_PROFILE") != nullptr;
inline bool       PhaseEnabled() noexcept {
    return g_phase_profile_enabled;
}
void PhaseMarkImpl(const char* name) noexcept;
// The open phase of the calling thread (to resume it after a nested mark).
const char* CurrentPhase() noexcept;
inline void PhaseMark(const char* name) noexcept {
	if (PhaseEnabled()) {
		PhaseMarkImpl(name);
	}
}

class WaitTagScope {
public:
	explicit WaitTagScope(const char* tag) noexcept;
	~WaitTagScope();
	WaitTagScope(const WaitTagScope&)            = delete;
	WaitTagScope& operator=(const WaitTagScope&) = delete;

private:
	const char* m_previous;
};

class TimedWait {
public:
	explicit TimedWait(const char* tag) noexcept
	    : m_tag(Enabled() || PhaseEnabled() ? tag : nullptr), m_start(std::chrono::steady_clock::now()) {}
	~TimedWait() {
		if (m_tag != nullptr) {
			AddWait(m_tag, static_cast<uint64_t>(
			                   std::chrono::duration_cast<std::chrono::nanoseconds>(
			                       std::chrono::steady_clock::now() - m_start)
			                       .count()));
		}
	}
	TimedWait(const TimedWait&)            = delete;
	TimedWait& operator=(const TimedWait&) = delete;

private:
	const char*                           m_tag;
	std::chrono::steady_clock::time_point m_start;
};

} // namespace GpuProfiler
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUPROFILER_H_
