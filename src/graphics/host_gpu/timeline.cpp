#include "graphics/host_gpu/timeline.h"

#include "common/logging/log.h"
#include "kernel/pthread.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::Graphics::Timeline {

namespace {

// KYTY_TIMELINE=<first frame>:<frames>, or t<seconds>:<frames> to open at the first present
// that many seconds after the emulator started.
struct Window {
	uint64_t first   = 0;
	uint64_t count   = 0;
	uint64_t seconds = 0;
};

const auto g_process_start = std::chrono::steady_clock::now();

const Window& GetWindow() {
	static const Window window = [] {
		Window result;
		const char* value = std::getenv("KYTY_TIMELINE");
		if (value != nullptr) {
			char* end = nullptr;
			if (value[0] == 't') {
				result.seconds = std::strtoull(value + 1, &end, 10);
				result.first   = UINT64_MAX;
			} else {
				result.first = std::strtoull(value, &end, 10);
			}
			result.count = end != nullptr && *end == ':' ? std::strtoull(end + 1, nullptr, 10) : 1;
		}
		return result;
	}();
	return window;
}

uint64_t NowMicros() {
	static const auto start = std::chrono::steady_clock::now();
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                 std::chrono::steady_clock::now() - start)
	                                 .count());
}

uint32_t ThreadId() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return static_cast<uint32_t>(GetCurrentThreadId());
#else
	return 0;
#endif
}

// Events are kept in memory while the window is open and written out when it closes: written as
// they happened (unbuffered stderr), the marks slowed the frames they measured ~10x.
struct Event {
	uint64_t    micros;
	uint32_t    thread;
	const char* name;
	uint64_t    a;
	uint64_t    b;
};

constexpr size_t MaxEvents = size_t {1} << 21u;

struct Store {
	std::unique_ptr<Event[]>                       events {new Event[MaxEvents]};
	std::atomic<uint64_t>                          next {0};
	std::mutex                                     names_mutex;
	std::vector<std::pair<uint32_t, std::string>> names;
};

Store& GetStore() {
	static Store store;
	return store;
}

void Flush() {
	auto&      store = GetStore();
	const auto count = std::min<uint64_t>(store.next.exchange(0), MaxEvents);
	std::string text;
	text.reserve(1u << 20u);
	{
		std::lock_guard lock(store.names_mutex);
		for (const auto& [thread, name]: store.names) {
			text += "TLN " + std::to_string(thread) + " " + name + "\n";
		}
	}
	char line[192];
	for (uint64_t i = 0; i < count; i++) {
		const auto& e = store.events[i];
		std::snprintf(line, sizeof(line), "TL %" PRIu64 " %u %s 0x%" PRIx64 " 0x%" PRIx64 "\n",
		              e.micros, e.thread, e.name, e.a, e.b);
		text += line;
		if (text.size() > (1u << 20u)) {
			std::fwrite(text.data(), 1, text.size(), stderr);
			text.clear();
		}
	}
	std::fwrite(text.data(), 1, text.size(), stderr);
	std::fprintf(stderr, "TLEND %" PRIu64 " events\n", count);
	std::fflush(stderr);
}

} // namespace

namespace Detail {

void MarkSlow(const char* event, uint64_t a, uint64_t b) {
	auto& store = GetStore();
	// Each thread names itself once (guest thread name, or host).
	thread_local bool named = false;
	if (!named) {
		named         = true;
		char name[64] = "(host)";
		if (auto self = Libs::LibKernel::PthreadSelfOrNull(); self != nullptr) {
			if (Libs::LibKernel::PthreadGetname(self, name) != 0) {
				std::snprintf(name, sizeof(name), "(guest)");
			}
		}
		std::lock_guard lock(store.names_mutex);
		store.names.emplace_back(ThreadId(), name);
	}
	const auto index = store.next.fetch_add(1, std::memory_order_relaxed);
	if (index < MaxEvents) {
		store.events[index] = {NowMicros(), ThreadId(), event, a, b};
	}
}

} // namespace Detail

void Present(uint64_t frame) {
	const auto& window = GetWindow();
	if (window.count == 0) {
		return;
	}
	static uint64_t first = window.first;
	if (first == UINT64_MAX &&
	    std::chrono::steady_clock::now() - g_process_start >= std::chrono::seconds(window.seconds)) {
		first = frame;
	}
	const bool active      = first != UINT64_MAX && frame >= first && frame - first < window.count;
	const bool was_active  = Detail::g_active.load(std::memory_order_relaxed);
	if (active || was_active) {
		Detail::MarkSlow("present", frame, 0);
	}
	Detail::g_active.store(active, std::memory_order_relaxed);
	if (was_active && !active) {
		Flush();
	}
}

} // namespace Libs::Graphics::Timeline
