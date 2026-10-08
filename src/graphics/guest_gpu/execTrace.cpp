#include "common/syncAcct.h"
#include "common/expSwitch.h"
#include "graphics/guest_gpu/execTrace.h"

#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace Libs::Graphics::ExecTrace {

thread_local uint64_t g_gpu_wait_ns      = 0;
thread_local uint64_t g_other_wait_ns    = 0;
thread_local uint32_t g_flushes          = 0;
thread_local uint64_t g_wait_tick_behind = 0;
thread_local uint64_t g_current_sequence = 0;
thread_local uint64_t g_shader_hash      = 0;

static std::mutex                             g_label_mutex;
static std::unordered_map<uint64_t, uint64_t> g_label_writers;

void NoteLabelWrite(uint64_t address) noexcept {
	{
		// Debug (KYTY_SYNC_ACCT): every label the execution thread writes (arg = address).
		KYTY_SYNC_SITE(s_label, "exec: label write", Point);
		Common::SyncAcct::Point(s_label, address, g_current_sequence);
	}
	if (!(Enabled() || KYTY_XO(LANE_R)) || g_current_sequence == 0) {
		return;
	}
	std::lock_guard lock(g_label_mutex);
	g_label_writers[address & ~7ull] = g_current_sequence;
}

uint64_t LastLabelWriter(uint64_t address) noexcept {
	if (!(Enabled() || KYTY_XO(LANE_R))) {
		return 0;
	}
	std::lock_guard lock(g_label_mutex);
	const auto it = g_label_writers.find(address & ~7ull);
	return it != g_label_writers.end() ? it->second : 0;
}

static const char* Prefix() noexcept {
	static const char* prefix = std::getenv("KYTY_EXEC_TRACE");
	return prefix;
}

namespace {
struct PhaseTime {
	const char* name;
	uint64_t    ns;
};
thread_local PhaseTime t_op_phases[64];
thread_local uint32_t  t_op_phase_count = 0;
thread_local bool      t_in_operation   = false;
FILE*                  g_slow_file      = nullptr;
} // namespace

void OnPhase(const char* name, uint64_t ns) noexcept {
	if (!t_in_operation) {
		return;
	}
	for (uint32_t i = 0; i < t_op_phase_count; i++) {
		if (t_op_phases[i].name == name) {
			t_op_phases[i].ns += ns;
			return;
		}
	}
	if (t_op_phase_count < 64) {
		t_op_phases[t_op_phase_count++] = {name, ns};
	}
}

void BeginOperation() noexcept {
	t_op_phase_count = 0;
	t_in_operation   = true;
}

void EndOperation(const Record& record) noexcept {
	t_in_operation = false;
	const auto ns  = record.end_ns - record.start_ns;
	static const uint64_t threshold = [] {
		const char* value = std::getenv("KYTY_EXEC_TRACE_SLOW_US");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 1000ull) * 1000ull;
	}();
	if (ns < threshold) {
		return;
	}
	if (g_slow_file == nullptr) {
		g_slow_file = std::fopen((std::string(Prefix()) + ".slow.txt").c_str(), "w");
		if (g_slow_file == nullptr) {
			return;
		}
	}
	std::fprintf(g_slow_file, "t=%llu q%u kind=%u hash=%016llx seq=%llu total=%.0f gpuwait=%.0f |",
	             static_cast<unsigned long long>(record.start_ns / 1000000ull), record.queue,
	             record.kind, static_cast<unsigned long long>(record.shader_hash),
	             static_cast<unsigned long long>(record.sequence), static_cast<double>(ns) / 1e3,
	             static_cast<double>(record.gpu_wait_ns) / 1e3);
	for (uint32_t i = 0; i < t_op_phase_count; i++) {
		static const uint64_t phase_threshold = [] {
			const char* value = std::getenv("KYTY_EXEC_TRACE_PHASE_NS");
			return value != nullptr ? std::strtoull(value, nullptr, 10) : 20000ull;
		}();
		if (t_op_phases[i].ns >= phase_threshold) {
			std::fprintf(g_slow_file, " %s=%.0f", t_op_phases[i].name,
			             static_cast<double>(t_op_phases[i].ns) / 1e3);
		}
	}
	std::fputc('\n', g_slow_file);
	std::fflush(g_slow_file);
}

bool Enabled() noexcept {
	static const bool enabled = Prefix() != nullptr && Prefix()[0] != 0;
	return enabled;
}

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

namespace {
struct Window {
	std::vector<Record> records;
	uint64_t            start = 0;
	uint32_t            index = 0;
};
Window g_window; // execution thread only
} // namespace

void Add(const Record& record) noexcept {
	auto& window = g_window;
	if (window.start == 0) {
		window.start = record.start_ns;
		window.records.reserve(1u << 18u);
	}
	window.records.push_back(record);
	if (record.end_ns - window.start < 5'000'000'000ull) {
		return;
	}
	std::vector<Record> records;
	records.swap(window.records);
	window.records.reserve(1u << 18u);
	window.start     = 0;
	const auto index = window.index++;
	std::thread([records = std::move(records), index] {
		const auto path = std::string(Prefix()) + "." + std::to_string(index % 8u) + ".bin";
		if (FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
			std::fwrite(records.data(), sizeof(Record), records.size(), file);
			std::fclose(file);
		}
	}).detach();
}

} // namespace Libs::Graphics::ExecTrace
