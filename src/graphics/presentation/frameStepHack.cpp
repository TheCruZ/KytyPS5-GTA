#include "graphics/presentation/frameStepHack.h"

#include "common/hackNotice.h"
#include "common/logging/log.h"
#include "common/timer.h"
#include "common/virtualMemory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Libs::VideoOut::FrameStepHack {

// [HACK] ASTRO BOT runs in slow motion when the emulator presents fewer frames than its frame
// rate: at 30 fps the game moves at half speed. Its simulation never measures time: at start-up it
// sets a frame rate (60) and stores it in three globals of its executable, the rate (a double),
// the frame in seconds (a float) and the frame in microseconds (a 64-bit integer). Each frame the
// main update gets 1 / rate as its time step, the physics world steps by the float, and most
// objects convert the microseconds into their smoothing factors. About 180 functions also read a
// 1/60 float constant of the executable as the step of their own per-frame updates (timers,
// velocities, the targets of keyframed physics bodies). Guest clocks play no part.
//
// While the game presents fewer frames than its rate, the three globals now hold the time between
// presented frames (averaged over about eight frames, at most MaxStep), as if the game had set
// that frame rate, and the constant grows in the same ratio; at its rate they keep the game's own
// values. ASTRO BOT also paces its frames by sleeping until 90 % of 1 / rate has passed; that one
// reader is redirected to the game's own frame time, so that the pacing does not hold the game at
// the slower rate.
//
// ASTRO's PLAYROOM (PPSA01325) runs on the same engine, with the same three globals, getters and
// main update, and about 50 readers of the 1/60 constant. It does not pace its frames by 1 / rate
// (the rate is only read by the 1 / rate getter, which only its updates call), so it has no pacing
// site to redirect.
//
// The disc, digital and Japanese releases and their updates place this code and data at different
// addresses, so FindSites finds them by their code; nothing is written unless each one is found
// exactly once.
//
// To be removed when the game runs at its frame rate in the emulator.
namespace {

struct Title {
	std::string_view id;
	const char*      name;
	bool             pacing; // paces its frames by 1 / rate
};

constexpr std::array<Title, 5> Titles {{
    {"PPSA21559", "ASTRO BOT", true},
    {"PPSA21564", "ASTRO BOT", true},
    {"PPSA21567", "ASTRO BOT", true},
    {"PPSA21568", "ASTRO BOT", true},
    {"PPSA01325", "ASTRO's PLAYROOM", false},
}};

constexpr float OwnFrameConstant = 1.0f / 60.0f;

// Longest step: below that frame rate the game slows down again instead of taking larger steps
// (or a step across a pause or a load).
constexpr double MaxStep = 1.0 / 15.0;
// The presented frame time is averaged over about this many frames (presentation is quantized to
// vblanks: at 24 fps the frames alternate 2 and 3 vblanks).
constexpr double AverageFrames = 8.0;
// Within this ratio of the game's frame the game keeps its own values.
constexpr double FullRateSlack = 1.08;

// Offsets from the load address.
struct Sites {
	uint64_t rate           = 0; // double, frames per second
	uint64_t step           = 0; // float, seconds per frame (rate + 8)
	uint64_t period         = 0; // uint64_t, microseconds per frame (rate + 16)
	uint64_t frame_constant = 0; // float 1/60, the step of fixed-step updates
	uint64_t pacing_site    = 0; // call <1 / rate>; vmulsd xmm0, xmm0, [1000.0] (13 bytes), or 0
	uint64_t pacing_frame   = 0; // 8 bytes of int3 padding: the pacing's frame time (double, ms)
};

struct State {
	bool        active       = false;
	const char* name         = nullptr;
	uint64_t    base         = 0;
	Sites       sites;
	double      own_rate     = 0.0; // the rate the game set, 0 until seen
	float       own_step     = 0.0f;
	uint64_t    own_period   = 0;
	double      written_rate = 0.0; // the rate in the game's global after our last look at it
	uint64_t    last_counter = 0;
	double      frequency    = 0.0;
	double      average      = 0.0;
	bool        scaled       = false;
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

std::string ParamString(const char* name) {
	std::string value;
	return Loader::SystemContentParamSfoGetString(name, &value) ? value : std::string("?");
}

// The loaded executable, read through offsets from its load address.
class Image {
public:
	Image(uint64_t base, uint64_t size)
	    : m_data(reinterpret_cast<const uint8_t*>(base)), m_size(size) {}

	[[nodiscard]] bool Has(uint64_t offset, uint64_t size) const {
		return offset <= m_size && size <= m_size - offset;
	}
	[[nodiscard]] bool Is(uint64_t offset, std::span<const uint8_t> bytes) const {
		return Has(offset, bytes.size()) &&
		       std::memcmp(m_data + offset, bytes.data(), bytes.size()) == 0;
	}
	template <typename T>
	[[nodiscard]] T Get(uint64_t offset) const {
		T value {};
		std::memcpy(&value, m_data + offset, sizeof(T));
		return value;
	}
	// The target of the rip-relative disp32 at `disp` of an instruction that ends at `end`.
	[[nodiscard]] uint64_t Target(uint64_t disp, uint64_t end) const {
		return end + static_cast<uint64_t>(static_cast<int64_t>(Get<int32_t>(disp)));
	}
	[[nodiscard]] bool IsDouble(uint64_t offset, double value) const {
		return Has(offset, sizeof(double)) && Get<double>(offset) == value;
	}
	// Calls f with every offset in [begin, end) where `bytes` starts.
	template <typename F>
	void ForEach(uint64_t begin, uint64_t end, std::span<const uint8_t> bytes, F&& f) const {
		end = std::min(end, m_size);
		if (begin >= end || end - begin < bytes.size()) {
			return;
		}
		const auto* first = m_data + begin;
		const auto* last  = m_data + end;
		const std::boyer_moore_horspool_searcher searcher(bytes.begin(), bytes.end());
		for (auto it = std::search(first, last, searcher); it != last;
		     it      = std::search(it + 1, last, searcher)) {
			f(static_cast<uint64_t>(it - m_data));
		}
	}

private:
	const uint8_t* m_data;
	uint64_t       m_size;
};

constexpr std::array<uint8_t, 4> VmovsdLoad {0xc5, 0xfb, 0x10, 0x05}; // vmovsd xmm0, [rip + d]
constexpr std::array<uint8_t, 4> VdivsdLoad {0xc5, 0xfb, 0x5e, 0x05}; // vdivsd xmm0, xmm0, [rip + d]
constexpr std::array<uint8_t, 4> VmulsdLoad {0xc5, 0xfb, 0x59, 0x05}; // vmulsd xmm0, xmm0, [rip + d]
constexpr std::array<uint8_t, 4> VmovssLoad {0xc5, 0xfa, 0x10, 0x05}; // vmovss xmm0, [rip + d]
constexpr std::array<uint8_t, 3> MovRaxLoad {0x48, 0x8b, 0x05};       // mov rax, [rip + d]
constexpr std::array<uint8_t, 4> Vcvtsd2ss {0xc5, 0xfb, 0x5a, 0xc0};  // vcvtsd2ss xmm0, xmm0, xmm0
constexpr std::array<uint8_t, 1> Call {0xe8};                         // call rel32
constexpr std::array<uint8_t, 8> Int3x8 {0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc};
constexpr uint8_t                Ret = 0xc3;

// Fills `sites` from the code of the executable, or returns why it cannot. Without `pacing` the
// game must have no pacing site.
std::string_view FindSites(const Image& image, std::span<const CodeRange> code, bool pacing,
                           Sites* sites) {
	// The frame-rate getters, next to each other: mov rax, [period]; ret, then
	// vmovsd xmm0, [1.0]; vdivsd xmm0, xmm0, [rate]; ret, then vmovss xmm0, [step]; ret, with
	// step = rate + 8 and period = rate + 16.
	uint64_t getter  = 0;
	uint32_t getters = 0;
	for (const auto& range: code) {
		image.ForEach(range.begin, range.end, VmovsdLoad, [&](uint64_t p) {
			if (!image.Has(p, 17) || !image.Is(p + 8, VdivsdLoad) ||
			    image.Get<uint8_t>(p + 16) != Ret || !image.IsDouble(image.Target(p + 4, p + 8), 1.0)) {
				return;
			}
			const auto rate          = image.Target(p + 12, p + 16);
			bool       period_getter = false;
			for (uint64_t q = p >= 32 ? p - 32 : 0; q < p && !period_getter; q++) {
				period_getter = image.Is(q, MovRaxLoad) && image.Get<uint8_t>(q + 7) == Ret &&
				                image.Target(q + 3, q + 7) == rate + 16;
			}
			bool step_getter = false;
			for (uint64_t q = p + 17; q < p + 64 && image.Has(q, 9) && !step_getter; q++) {
				step_getter = image.Is(q, VmovssLoad) && image.Get<uint8_t>(q + 8) == Ret &&
				              image.Target(q + 4, q + 8) == rate + 8;
			}
			if (period_getter && step_getter && image.Has(rate, 24)) {
				getter      = p;
				sites->rate = rate;
				getters++;
			}
		});
	}
	if (getters != 1) {
		return getters == 0 ? "frame-rate getters not found" : "frame-rate getters found twice";
	}
	sites->step   = sites->rate + 8;
	sites->period = sites->rate + 16;

	// The calls of the 1 / rate getter: the pacing multiplies its result by 1000.0 and the main
	// update converts it to a float.
	uint32_t pacing_sites = 0;
	bool     update       = false;
	for (const auto& range: code) {
		image.ForEach(range.begin, range.end, Call, [&](uint64_t p) {
			if (!image.Has(p, 21) || image.Target(p + 1, p + 5) != getter) {
				return;
			}
			if (image.Is(p + 5, VmulsdLoad) && image.IsDouble(image.Target(p + 9, p + 13), 1000.0)) {
				sites->pacing_site = p;
				pacing_sites++;
			}
			for (uint64_t q = p + 5; q + Vcvtsd2ss.size() <= p + 21 && !update; q++) {
				update = image.Is(q, Vcvtsd2ss);
			}
		});
	}
	if (!pacing && pacing_sites != 0) {
		return "unexpected frame pacing found";
	}
	if (pacing && pacing_sites != 1) {
		return pacing_sites == 0 ? "frame pacing not found" : "frame pacing found twice";
	}
	if (!update) {
		return "main update not found";
	}

	// The int3 padding closest before the pacing site, within 64 KB of it in its code range.
	if (pacing) {
		const auto range = std::find_if(code.begin(), code.end(), [&](const CodeRange& r) {
			return sites->pacing_site >= r.begin && sites->pacing_site < r.end;
		});
		const uint64_t start =
		    sites->pacing_site >= 8 ? (sites->pacing_site - 8) & ~uint64_t {7} : 0;
		for (uint64_t a = start; a >= range->begin && a + 0x10000 > sites->pacing_site && a >= 8;
		     a -= 8) {
			if (image.Is(a, Int3x8)) {
				sites->pacing_frame = a;
				break;
			}
		}
		if (sites->pacing_frame == 0) {
			return "padding for the pacing frame time not found";
		}
	}

	// The 1/60 float that the code reads the most (instructions that end with their rip-relative
	// disp32, as the loads and arithmetic of the fixed-step updates do), other than the step.
	std::vector<uint64_t> constants;
	image.ForEach(0, UINT64_MAX, std::bit_cast<std::array<uint8_t, 4>>(OwnFrameConstant),
	              [&](uint64_t a) {
		              if (a % 4 == 0 && (a < sites->rate || a >= sites->rate + 24)) {
			              constants.push_back(a);
		              }
	              });
	if (constants.empty()) {
		return "1/60 constant not found";
	}
	std::vector<uint32_t> reads(constants.size());
	for (const auto& r: code) {
		for (uint64_t p = r.begin; p + 4 <= r.end && image.Has(p, 4); p++) {
			const auto target = image.Target(p, p + 4);
			if (target < constants.front() || target > constants.back()) {
				continue;
			}
			const auto it = std::lower_bound(constants.begin(), constants.end(), target);
			if (it != constants.end() && *it == target) {
				reads[static_cast<size_t>(it - constants.begin())]++;
			}
		}
	}
	const auto most   = std::max_element(reads.begin(), reads.end());
	uint32_t   second = 0;
	for (auto it = reads.begin(); it != reads.end(); ++it) {
		second = it != most ? std::max(second, *it) : second;
	}
	// 01.007.000 of PPSA21564: 218 reads, 01.905.000 of PPSA01325: 47; no other 1/60 float is read.
	if (*most < 32 || *most < 8 * second) {
		return "1/60 constant of the fixed-step updates not found";
	}
	sites->frame_constant = constants[static_cast<size_t>(most - reads.begin())];
	return {};
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
	const double rate  = 1.0 / frame;
	const auto&  sites = state->sites;
	Write<double>(state->base + sites.rate, rate);
	Write<float>(state->base + sites.step, static_cast<float>(1.0 / rate));
	Write<uint64_t>(state->base + sites.period, static_cast<uint64_t>(1000000.0 / rate));
	Write<float>(state->base + sites.frame_constant,
	             static_cast<float>(OwnFrameConstant * frame * state->own_rate));
	state->written_rate = rate;
}

void WriteOwnFrame(State* state) {
	const auto& sites = state->sites;
	Write<double>(state->base + sites.rate, state->own_rate);
	Write<float>(state->base + sites.step, state->own_step);
	Write<uint64_t>(state->base + sites.period, state->own_period);
	Write<float>(state->base + sites.frame_constant, OwnFrameConstant);
	state->written_rate = state->own_rate;
}

} // namespace

void PatchMainProgram(uint64_t base_vaddr, uint64_t size, std::span<const CodeRange> code) noexcept {
	const auto id    = ParamString("TITLE_ID");
	const auto title = std::find_if(Titles.begin(), Titles.end(),
	                                [&](const Title& t) { return t.id == id; });
	if (title == Titles.end()) {
		return;
	}
	const auto version = ParamString("APP_VER");
	Sites      sites;
	const auto failure = base_vaddr == 0
	                         ? std::string_view("no executable")
	                         : FindSites(Image(base_vaddr, size), code, title->pacing, &sites);
	if (!failure.empty()) {
		Log::WriteToConsoleAndLog(fmt::format("[HACK] {} frame time: not applied to {} {} ({})\n",
		                                      title->name, id, version, failure));
		return;
	}

	if (sites.pacing_site != 0) {
		// The pacing reads the game's own frame time from the padding instead of 1 / rate:
		// vmovsd xmm0, [rip + disp32] (8 bytes) and a 5-byte nop.
		constexpr std::array<uint8_t, 5> nop5 {0x0f, 0x1f, 0x44, 0x00, 0x00};
		const auto                       site = base_vaddr + sites.pacing_site;
		const int32_t disp = static_cast<int32_t>(static_cast<int64_t>(sites.pacing_frame) -
		                                          static_cast<int64_t>(sites.pacing_site + 8));
		std::array<uint8_t, 13> pacing {};
		std::memcpy(pacing.data(), VmovsdLoad.data(), VmovsdLoad.size());
		std::memcpy(pacing.data() + 4, &disp, sizeof(disp));
		std::memcpy(pacing.data() + 8, nop5.data(), nop5.size());
		std::array<uint8_t, 13> original {};
		std::memcpy(original.data(), reinterpret_cast<const void*>(site), original.size());
		Write<double>(base_vaddr + sites.pacing_frame, (1.0 / 60.0) * 1000.0);
		std::memcpy(reinterpret_cast<void*>(site), pacing.data(), pacing.size());
		if (!Common::VirtualMemory::FlushInstructionCache(site, pacing.size())) {
			std::memcpy(reinterpret_cast<void*>(site), original.data(), original.size());
			Log::WriteToConsoleAndLog(
			    fmt::format("[HACK] {} frame time: not applied to {} {} (code write failed)\n",
			                title->name, id, version));
			return;
		}
	}

	auto& state     = GetState();
	state.base      = base_vaddr;
	state.sites     = sites;
	state.frequency = 1e9; // Common::SteadyNs() counts nanoseconds.
	state.name      = title->name;
	state.active    = true;
	Log::WriteToConsoleAndLog(fmt::format(
	    "[HACK] {} frame time ({} {}: rate {:#x}, constant {:#x}, pacing {:#x}): below the game's "
	    "frame rate its frame time follows the presented frames (at most {:.0f} ms), so that it "
	    "does not run in slow motion\n",
	    title->name, id, version, sites.rate, sites.frame_constant, sites.pacing_site,
	    MaxStep * 1000.0));
}

void OnPresentedFrame() noexcept {
	auto& state = GetState();
	if (!state.active) {
		return;
	}
	const auto counter = Common::SteadyNs();
	const auto last    = std::exchange(state.last_counter, counter);

	const auto& sites = state.sites;
	const auto  rate  = Read<double>(state.base + sites.rate);
	if (rate != state.written_rate) {
		// The game set its frame rate (at start-up, or for another video mode): it owns the three
		// globals again, and the pacing follows its rate.
		const auto step   = Read<float>(state.base + sites.step);
		const auto period = Read<uint64_t>(state.base + sites.period);
		if (!IsOwnFrameRate(rate, step, period)) {
			return; // Not set yet, or being set.
		}
		if (rate != state.own_rate) {
			Log::WriteToConsoleAndLog(fmt::format("[HACK] {} frame time: the game runs at {:.2f} fps\n",
			                                      state.name, rate));
		}
		state.own_rate     = rate;
		state.own_step     = step;
		state.own_period   = period;
		state.written_rate = rate;
		state.average      = 0.0;
		state.scaled       = false;
		Write<float>(state.base + sites.frame_constant, OwnFrameConstant);
		if (sites.pacing_frame != 0) {
			Write<double>(state.base + sites.pacing_frame, (1.0 / rate) * 1000.0);
		}
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
	Common::HackNotice::Report("Frame time follows the presented frames",
	                           "FrameStepHack::OnPresentedFrame", 0, 0,
	                           fmt::format("{}: frame {:.1f} ms instead of {:.1f} ms", state.name,
	                                       state.average * 1000.0, own_frame * 1000.0));
}

} // namespace Libs::VideoOut::FrameStepHack
