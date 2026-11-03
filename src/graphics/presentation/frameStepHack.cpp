#include "graphics/presentation/frameStepHack.h"

#include "common/hackNotice.h"
#include "common/logging/log.h"
#include "common/timer.h"
#include "common/virtualMemory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fmt/format.h>
#include <string>
#include <utility>

namespace Libs::VideoOut::FrameStepHack {

// [HACK] ASTRO BOT (PPSA21564) runs in slow motion when the emulator presents fewer frames than
// its frame rate: at 30 fps the game moves at half speed. Its simulation never measures time: at
// start-up it sets a frame rate (60) and stores it in three globals of its executable, the rate (a
// double), the frame in seconds (a float) and the frame in microseconds (a 64-bit integer). Each
// frame the main update gets 1 / rate as its time step, the physics world steps by the float, and
// most objects convert the microseconds into their smoothing factors. About 180 functions also
// read a 1/60 float constant of the executable as the step of their own per-frame updates (timers,
// velocities, the targets of keyframed physics bodies). Guest clocks play no part.
//
// While the game presents fewer frames than its rate, the three globals now hold the time between
// presented frames (averaged over about eight frames, at most MaxStep), as if the game had set
// that frame rate, and the constant grows in the same ratio; at its rate they keep the game's own
// values. The game also paces its frames by sleeping until 90 % of 1 / rate has passed; that one
// reader is redirected to the game's own frame time, so that the pacing does not hold the game at
// the slower rate.
//
// To be removed when the game runs at its frame rate in the emulator.
namespace {

// The game's executable (01.007.000): offsets from its load address. All the bytes below are
// checked before anything is written.
constexpr uint64_t RateValue   = 0x8fd4040; // double, frames per second
constexpr uint64_t StepValue   = 0x8fd4048; // float, seconds per frame
constexpr uint64_t PeriodValue = 0x8fd4050; // uint64_t, microseconds per frame
// float 1/60 in the read-only data: the step of fixed-step updates (219 reads in the code).
constexpr uint64_t FrameConstant    = 0x834f34c;
constexpr float    OwnFrameConstant = 1.0f / 60.0f;

struct Code {
	uint64_t                offset;
	std::array<uint8_t, 17> bytes;
	size_t                  size;
};

constexpr std::array<Code, 6> CheckedCode {{
    // mov rax, [period]; ret
    {0x3520, {0x48, 0x8b, 0x05, 0x29, 0x0b, 0xfd, 0x08, 0xc3}, 8},
    // vmovsd xmm0, [1.0]; vdivsd xmm0, xmm0, [rate]; ret
    {0x3530,
     {0xc5, 0xfb, 0x10, 0x05, 0x48, 0xd0, 0x34, 0x08, 0xc5, 0xfb, 0x5e, 0x05, 0x00, 0x0b, 0xfd,
      0x08, 0xc3},
     17},
    // vmovss xmm0, [step]; ret
    {0x3550, {0xc5, 0xfa, 0x10, 0x05, 0xf0, 0x0a, 0xfd, 0x08, 0xc3}, 9},
    // The main update: call 1 / rate; mov rax, [r14]; vcvtsd2ss xmm0, xmm0, xmm0
    {0x10afe2d, {0xe8, 0xfe, 0x36, 0xf5, 0xfe, 0x49, 0x8b, 0x06, 0xc5, 0xfb, 0x5a, 0xc0}, 12},
    // Unused padding (int3) before the main frame function: holds the pacing's frame time.
    {0x2a34c8, {0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc}, 8},
    // The 1/60 constant (0x3c888889).
    {FrameConstant, {0x89, 0x88, 0x88, 0x3c}, 4},
}};

// The frame pacing: call 1 / rate; vmulsd xmm0, xmm0, [1000.0]. Its result (the frame in
// milliseconds) then goes to the sleep until 90 % of the frame.
constexpr uint64_t                PacingSite = 0x2a470b;
constexpr std::array<uint8_t, 13> PacingCode {0xe8, 0x20, 0xee, 0xd5, 0xff, 0xc5, 0xfb,
                                              0x59, 0x05, 0xb8, 0xc4, 0x0a, 0x08};
// Replaced by vmovsd xmm0, [PacingFrame] and a 5-byte nop.
constexpr uint64_t PacingFrame = 0x2a34c8; // double, milliseconds per frame at the game's own rate

// Longest step: below that frame rate the game slows down again instead of taking larger steps
// (or a step across a pause or a load).
constexpr double MaxStep = 1.0 / 15.0;
// The presented frame time is averaged over about this many frames (presentation is quantized to
// vblanks: at 24 fps the frames alternate 2 and 3 vblanks).
constexpr double AverageFrames = 8.0;
// Within this ratio of the game's frame the game keeps its own values.
constexpr double FullRateSlack = 1.08;

struct State {
	bool     active       = false;
	uint64_t base         = 0;
	double   own_rate     = 0.0; // the rate the game set, 0 until seen
	float    own_step     = 0.0f;
	uint64_t own_period   = 0;
	double   written_rate = 0.0; // the rate in the game's global after our last look at it
	uint64_t last_counter = 0;
	double   frequency    = 0.0;
	double   average      = 0.0;
	bool     scaled       = false;
};

State& GetState() {
	static State state;
	return state;
}

template <typename T>
T Read(uint64_t address) {
	return std::atomic_ref<T>(*reinterpret_cast<T*>(address)).load(std::memory_order_relaxed);
}

template <typename T>
void Write(uint64_t address, T value) {
	std::atomic_ref<T>(*reinterpret_cast<T*>(address)).store(value, std::memory_order_relaxed);
}

bool IsAstroBot() {
	std::string title_id;
	return Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) && title_id == "PPSA21564";
}

bool IsKnownVersion() {
	std::string version;
	return Loader::SystemContentParamSfoGetString("APP_VER", &version) && version == "01.007.000";
}

bool Matches(uint64_t base, uint64_t offset, const uint8_t* bytes, size_t size) {
	return std::memcmp(reinterpret_cast<const void*>(base + offset), bytes, size) == 0;
}

// What SetFrameRate stores: the rate, 1 / rate as a float and 1000000 / rate truncated.
bool IsOwnFrameRate(double rate, float step, uint64_t period) {
	return rate >= 20.0 && rate <= 240.0 &&
	       std::fabs(static_cast<double>(step) * rate - 1.0) < 1e-4 &&
	       period == static_cast<uint64_t>(1000000.0 / rate);
}

// The game's frame lasts `frame` seconds: what SetFrameRate(1 / frame) would store, and the 1/60
// constant in the same ratio to the game's own frame.
void WriteFrame(State* state, double frame) {
	const double rate = 1.0 / frame;
	Write<double>(state->base + RateValue, rate);
	Write<float>(state->base + StepValue, static_cast<float>(1.0 / rate));
	Write<uint64_t>(state->base + PeriodValue, static_cast<uint64_t>(1000000.0 / rate));
	Write<float>(state->base + FrameConstant,
	             static_cast<float>(OwnFrameConstant * frame * state->own_rate));
	state->written_rate = rate;
}

void WriteOwnFrame(State* state) {
	Write<double>(state->base + RateValue, state->own_rate);
	Write<float>(state->base + StepValue, state->own_step);
	Write<uint64_t>(state->base + PeriodValue, state->own_period);
	Write<float>(state->base + FrameConstant, OwnFrameConstant);
	state->written_rate = state->own_rate;
}

} // namespace

void PatchMainProgram(uint64_t base_vaddr, uint64_t size) noexcept {
	if (!IsAstroBot()) {
		return;
	}
	const bool known = IsKnownVersion() && base_vaddr != 0 &&
	                   size > PeriodValue + sizeof(uint64_t) &&
	                   Matches(base_vaddr, PacingSite, PacingCode.data(), PacingCode.size()) &&
	                   std::all_of(CheckedCode.begin(), CheckedCode.end(), [&](const Code& code) {
		                   return Matches(base_vaddr, code.offset, code.bytes.data(), code.size);
	                   });
	if (!known) {
		Log::WriteToConsoleAndLog(
		    "[HACK] ASTRO BOT frame time: not applied (unknown executable version)\n");
		return;
	}

	// The pacing reads the game's own frame time from the padding instead of 1 / rate:
	// vmovsd xmm0, [rip + disp32] (8 bytes) and a 5-byte nop.
	constexpr std::array<uint8_t, 4> load_frame {0xc5, 0xfb, 0x10, 0x05};
	constexpr std::array<uint8_t, 5> nop5 {0x0f, 0x1f, 0x44, 0x00, 0x00};
	const int32_t           disp = static_cast<int32_t>(static_cast<int64_t>(PacingFrame) -
	                                                    static_cast<int64_t>(PacingSite + 8));
	std::array<uint8_t, 13> pacing {};
	std::memcpy(pacing.data(), load_frame.data(), load_frame.size());
	std::memcpy(pacing.data() + 4, &disp, sizeof(disp));
	std::memcpy(pacing.data() + 8, nop5.data(), nop5.size());
	Write<double>(base_vaddr + PacingFrame, (1.0 / 60.0) * 1000.0);
	std::memcpy(reinterpret_cast<void*>(base_vaddr + PacingSite), pacing.data(), pacing.size());
	if (!Common::VirtualMemory::FlushInstructionCache(base_vaddr + PacingSite, pacing.size())) {
		std::memcpy(reinterpret_cast<void*>(base_vaddr + PacingSite), PacingCode.data(),
		            PacingCode.size());
		Log::WriteToConsoleAndLog("[HACK] ASTRO BOT frame time: not applied (code write failed)\n");
		return;
	}

	auto& state     = GetState();
	state.base      = base_vaddr;
	state.frequency = static_cast<double>(Common::Timer::QueryPerformanceFrequency());
	state.active    = true;
	Log::WriteToConsoleAndLog(fmt::format(
	    "[HACK] ASTRO BOT frame time: below the game's frame rate its frame time follows the "
	    "presented frames (at most {:.0f} ms), so that it does not run in slow motion\n",
	    MaxStep * 1000.0));
}

void OnPresentedFrame() noexcept {
	auto& state = GetState();
	if (!state.active) {
		return;
	}
	const auto counter = Common::Timer::QueryPerformanceCounter();
	const auto last    = std::exchange(state.last_counter, counter);

	const auto rate = Read<double>(state.base + RateValue);
	if (rate != state.written_rate) {
		// The game set its frame rate (at start-up, or for another video mode): it owns the three
		// globals again, and the pacing follows its rate.
		const auto step   = Read<float>(state.base + StepValue);
		const auto period = Read<uint64_t>(state.base + PeriodValue);
		if (!IsOwnFrameRate(rate, step, period)) {
			return; // Not set yet, or being set.
		}
		if (rate != state.own_rate) {
			Log::WriteToConsoleAndLog(
			    fmt::format("[HACK] ASTRO BOT frame time: the game runs at {:.2f} fps\n", rate));
		}
		state.own_rate     = rate;
		state.own_step     = step;
		state.own_period   = period;
		state.written_rate = rate;
		state.average      = 0.0;
		state.scaled       = false;
		Write<float>(state.base + FrameConstant, OwnFrameConstant);
		Write<double>(state.base + PacingFrame, (1.0 / rate) * 1000.0);
	}
	if (last == 0 || state.own_rate == 0.0) {
		return;
	}

	const double own_frame = 1.0 / state.own_rate;
	const double frame = std::min(static_cast<double>(counter - last) / state.frequency, MaxStep);
	state.average =
	    state.average == 0.0 ? frame : state.average + (frame - state.average) / AverageFrames;
	if (state.average <= own_frame * FullRateSlack) {
		if (state.scaled) {
			WriteOwnFrame(&state);
			state.scaled = false;
		}
		return;
	}

	WriteFrame(&state, state.average);
	state.scaled = true;
	Common::HackNotice::Report("ASTRO BOT frame time follows the presented frames",
	                           "FrameStepHack::OnPresentedFrame", 0, 0,
	                           fmt::format("frame {:.1f} ms instead of {:.1f} ms",
	                                       state.average * 1000.0, own_frame * 1000.0));
}

} // namespace Libs::VideoOut::FrameStepHack
