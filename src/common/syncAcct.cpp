#include "common/syncAcct.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#include <psapi.h>

#include <array>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Common::SyncAcct {
namespace {

struct RecordData {
	uint64_t t0;
	uint64_t t1;
	uint64_t arg;
	uint64_t arg2;
	uint64_t ret;
	uint32_t tid;
	uint16_t site;
	uint16_t parent;
};
static_assert(sizeof(RecordData) == 48);

constexpr uint16_t SiteSummary = 1; // arg = ticks, arg2 = count, ret = site | parent << 16
constexpr uint16_t SiteCpu     = 2; // arg = cycle time, arg2 = kernel+user (100 ns), ret = 1 if new
constexpr uint16_t SiteDrops   = 3; // arg = dropped records
constexpr size_t   MaxSites    = 2048;

struct SummaryEntry {
	std::atomic<uint32_t> key {0}; // site | parent << 16, 0 = free
	std::atomic<uint64_t> ticks {0};
	std::atomic<uint64_t> count {0};
};

struct ThreadState {
	static constexpr size_t RingSize = 32768;
	uint32_t                tid      = 0;
	std::unique_ptr<RecordData[]> ring = std::make_unique<RecordData[]>(RingSize);
	alignas(64) std::atomic<uint64_t> head {0}; // writer
	alignas(64) std::atomic<uint64_t> tail {0}; // owner
	std::atomic<uint64_t>             drops {0};
	std::array<SummaryEntry, 512>     summary;
	std::array<uint16_t, 16>          stack {};
	uint32_t                          depth = 0;
};

std::mutex                                g_threads_mutex;
std::vector<std::shared_ptr<ThreadState>> g_threads;
thread_local std::shared_ptr<ThreadState> t_state;
thread_local uint32_t                     t_tid = 0;

std::mutex                       g_sites_mutex;
std::vector<std::string>         g_site_names;
std::atomic<uint16_t>            g_next_site {16};
std::mutex                       g_meta_mutex;
FILE*                            g_meta = nullptr;
FILE*                            g_bin  = nullptr;
uint64_t                         g_min_ticks = 0;
std::once_flag                   g_start_once;
std::atomic<Hook>                g_hook {nullptr};
std::atomic<void*>               g_hook_data {nullptr};
uintptr_t                        g_exe_base = 0;
uintptr_t                        g_exe_end  = 0;
// Only waits and taxes are summed when short; points, GPU and work intervals are always kept.
std::array<std::atomic<uint8_t>, MaxSites> g_site_short {};

double CalibrateTsc() {
	const auto clock0 = std::chrono::steady_clock::now();
	const auto tsc0   = __rdtsc();
	while (std::chrono::steady_clock::now() - clock0 < std::chrono::milliseconds(50)) {
	}
	const auto tsc1 = __rdtsc();
	const auto ns   = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - clock0)
	                    .count();
	return ns / static_cast<double>(tsc1 - tsc0);
}

void WriterThread();

void Start() {
	std::call_once(g_start_once, [] {
		wchar_t path[MAX_PATH] {};
		GetModuleFileNameW(nullptr, path, MAX_PATH);
		std::wstring dir(path);
		dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
		const auto pid = GetCurrentProcessId();
		wchar_t    bin[MAX_PATH + 64];
		wchar_t    meta[MAX_PATH + 64];
		swprintf(bin, MAX_PATH + 64, L"%lssyncacct_%lu.bin", dir.c_str(), pid);
		swprintf(meta, MAX_PATH + 64, L"%lssyncacct_%lu.meta.txt", dir.c_str(), pid);
		g_bin  = _wfopen(bin, L"wb");
		g_meta = _wfopen(meta, L"w");
		const auto* module = reinterpret_cast<const uint8_t*>(GetModuleHandleW(nullptr));
		g_exe_base         = reinterpret_cast<uintptr_t>(module);
		const auto* dos    = reinterpret_cast<const IMAGE_DOS_HEADER*>(module);
		const auto* nt     = reinterpret_cast<const IMAGE_NT_HEADERS*>(module + dos->e_lfanew);
		g_exe_end          = g_exe_base + nt->OptionalHeader.SizeOfImage;
		const double ns_per_tick = CalibrateTsc();
		const char*  min_env     = std::getenv("KYTY_SYNC_MIN_NS");
		const double min_ns      = min_env != nullptr ? std::atof(min_env) : 3000.0;
		g_min_ticks              = static_cast<uint64_t>(min_ns / ns_per_tick);
		LARGE_INTEGER qpc {};
		QueryPerformanceCounter(&qpc);
		Meta("tsc_ns_per_tick %.9f", ns_per_tick);
		{
			FILETIME now {}, creation {}, exit {}, kernel {}, user {};
			GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user);
			const auto tsc = __rdtsc();
			GetSystemTimePreciseAsFileTime(&now);
			const auto to64 = [](const FILETIME& t) {
				return (static_cast<unsigned long long>(t.dwHighDateTime) << 32u) | t.dwLowDateTime;
			};
			Meta("tsc_now %llu filetime_now %llu process_creation %llu", static_cast<unsigned long long>(tsc),
			     to64(now), to64(creation));
		}
		Meta("exe_base 0x%llx", static_cast<unsigned long long>(g_exe_base));
		Meta("min_ns %.0f", min_ns);
		Meta("site 1 3 summary");
		Meta("site 2 3 cpu");
		Meta("site 3 3 drops");
		std::printf("SYNC_ACCT: tracing to syncacct_%lu.bin (%.4f ns/tick, min %.0f ns)\n", pid,
		            ns_per_tick, min_ns);
		std::fflush(stdout);
		std::thread(WriterThread).detach();
	});
}

ThreadState& State() {
	if (t_state == nullptr) {
		Start();
		auto state = std::make_shared<ThreadState>();
		state->tid = GetCurrentThreadId();
		t_tid      = state->tid;
		{
			std::lock_guard lock(g_threads_mutex);
			g_threads.push_back(state);
		}
		t_state = std::move(state);
	}
	return *t_state;
}

void AddSummary(ThreadState& state, uint16_t site, uint16_t parent, uint64_t ticks) {
	const uint32_t key  = site | (static_cast<uint32_t>(parent) << 16u);
	auto           slot = (key * 0x9e3779b1u) >> 23u;
	for (size_t probe = 0; probe < state.summary.size(); probe++, slot = (slot + 1) % state.summary.size()) {
		auto&      entry = state.summary[slot];
		const auto k     = entry.key.load(std::memory_order_relaxed);
		if (k == key || k == 0) {
			if (k == 0) {
				entry.key.store(key, std::memory_order_release);
			}
			// Single writer; the writer thread exchanges them (a racing exchange may lose one).
			entry.ticks.store(entry.ticks.load(std::memory_order_relaxed) + ticks, std::memory_order_relaxed);
			entry.count.store(entry.count.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
			return;
		}
	}
}

void Append(ThreadState& state, const RecordData& record) {
	const auto tail = state.tail.load(std::memory_order_relaxed);
	if (tail - state.head.load(std::memory_order_acquire) >= ThreadState::RingSize) {
		state.drops.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	state.ring[tail % ThreadState::RingSize] = record;
	state.tail.store(tail + 1, std::memory_order_release);
}

struct ThreadCpu {
	HANDLE   handle = nullptr;
	bool     named  = false;
	uint32_t misses = 0;
};

void DrainAll(std::vector<RecordData>& out) {
	std::vector<std::shared_ptr<ThreadState>> threads;
	{
		std::lock_guard lock(g_threads_mutex);
		threads = g_threads;
	}
	for (auto& state: threads) {
		const auto head = state->head.load(std::memory_order_relaxed);
		const auto tail = state->tail.load(std::memory_order_acquire);
		for (auto i = head; i != tail; i++) {
			out.push_back(state->ring[i % ThreadState::RingSize]);
		}
		state->head.store(tail, std::memory_order_release);
	}
}

void FlushSummaries(std::vector<RecordData>& out) {
	std::vector<std::shared_ptr<ThreadState>> threads;
	{
		std::lock_guard lock(g_threads_mutex);
		threads = g_threads;
	}
	const auto now = __rdtsc();
	for (auto& state: threads) {
		for (auto& entry: state->summary) {
			const auto key = entry.key.load(std::memory_order_acquire);
			if (key == 0 || entry.count.load(std::memory_order_relaxed) == 0) {
				continue;
			}
			RecordData record {};
			record.t0   = now;
			record.t1   = now;
			record.arg  = entry.ticks.exchange(0, std::memory_order_relaxed);
			record.arg2 = entry.count.exchange(0, std::memory_order_relaxed);
			record.ret  = key;
			record.tid  = state->tid;
			record.site = SiteSummary;
			out.push_back(record);
		}
		if (const auto drops = state->drops.exchange(0, std::memory_order_relaxed); drops != 0) {
			RecordData record {};
			record.t0   = now;
			record.t1   = now;
			record.arg  = drops;
			record.tid  = state->tid;
			record.site = SiteDrops;
			out.push_back(record);
		}
	}
}

void SampleCpu(std::unordered_map<uint32_t, ThreadCpu>& cpus, bool discover,
               std::vector<RecordData>& out) {
	const auto pid = GetCurrentProcessId();
	if (discover) {
		HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
		if (snapshot != INVALID_HANDLE_VALUE) {
			THREADENTRY32 entry {};
			entry.dwSize = sizeof(entry);
			for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
				if (entry.th32OwnerProcessID != pid || cpus.contains(entry.th32ThreadID)) {
					continue;
				}
				ThreadCpu cpu;
				cpu.handle = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
				if (cpu.handle != nullptr) {
					cpus.emplace(entry.th32ThreadID, cpu);
				}
			}
			CloseHandle(snapshot);
		}
	}
	const auto now = __rdtsc();
	{
		// Process-wide OS page faults (tid 0): arg = PageFaultCount.
		PROCESS_MEMORY_COUNTERS counters {};
		counters.cb = sizeof(counters);
		if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
			RecordData record {};
			record.t0   = now;
			record.t1   = now;
			record.arg  = counters.PageFaultCount;
			record.arg2 = counters.WorkingSetSize;
			record.tid  = 0;
			record.site = SiteCpu;
			out.push_back(record);
		}
	}
	for (auto it = cpus.begin(); it != cpus.end();) {
		auto& [tid, cpu] = *it;
		if (!cpu.named || discover) {
			PWSTR description = nullptr;
			if (SUCCEEDED(GetThreadDescription(cpu.handle, &description)) && description != nullptr) {
				char name[256] {};
				WideCharToMultiByte(CP_UTF8, 0, description, -1, name, sizeof(name) - 1, nullptr, nullptr);
				LocalFree(description);
				if (name[0] != 0) {
					static std::unordered_map<uint32_t, std::string> known;
					if (known[tid] != name) {
						known[tid] = name;
						Meta("thread %u %s", tid, name);
					}
					cpu.named = true;
				}
			}
		}
		ULONG64 cycles = 0;
		FILETIME creation {}, exit {}, kernel {}, user {};
		DWORD    code = 0;
		if (!GetExitCodeThread(cpu.handle, &code) || code != STILL_ACTIVE) {
			CloseHandle(cpu.handle);
			it = cpus.erase(it);
			continue;
		}
		QueryThreadCycleTime(cpu.handle, &cycles);
		GetThreadTimes(cpu.handle, &creation, &exit, &kernel, &user);
		const auto to64 = [](const FILETIME& t) {
			return (static_cast<uint64_t>(t.dwHighDateTime) << 32u) | t.dwLowDateTime;
		};
		RecordData record {};
		record.t0   = now;
		record.t1   = now;
		record.arg  = cycles;
		record.arg2 = to64(kernel) + to64(user);
		record.ret  = to64(kernel);
		record.tid  = tid;
		record.site = SiteCpu;
		out.push_back(record);
		++it;
	}
}

void WriterThread() {
	SetThreadDescription(GetCurrentThread(), L"SyncAcct writer");
	std::vector<RecordData>                 out;
	std::unordered_map<uint32_t, ThreadCpu> cpus;
	uint32_t                                iteration = 0;
	HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
	                                      TIMER_ALL_ACCESS);
	for (;;) {
		if (timer != nullptr) {
			LARGE_INTEGER due {};
			due.QuadPart = -20000; // 2 ms
			SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0);
			WaitForSingleObject(timer, INFINITE);
		} else {
			Sleep(2);
		}
		iteration++;
		if (auto hook = g_hook.load(std::memory_order_acquire); hook != nullptr) {
			hook(g_hook_data.load(std::memory_order_relaxed));
		}
		out.clear();
		DrainAll(out);
		if (iteration % 25 == 0) {
			SampleCpu(cpus, iteration % 500 == 25, out);
			FlushSummaries(out);
		}
		if (!out.empty() && g_bin != nullptr) {
			std::fwrite(out.data(), sizeof(RecordData), out.size(), g_bin);
		}
		if (iteration % 50 == 0 && g_bin != nullptr) {
			std::fflush(g_bin);
		}
	}
}

} // namespace

uint16_t RegisterSite(const char* name, Kind kind) noexcept {
	Start();
	std::lock_guard lock(g_sites_mutex);
	for (size_t i = 0; i < g_site_names.size(); i++) {
		if (g_site_names[i] == name) {
			return static_cast<uint16_t>(16 + i);
		}
	}
	const auto id = static_cast<uint16_t>(16 + g_site_names.size());
	if (id >= MaxSites) {
		return 0;
	}
	g_site_names.emplace_back(name);
	g_site_short[id].store(kind == Kind::Spin || kind == Kind::Block || kind == Kind::Tax ? 1 : 0,
	                       std::memory_order_release);
	Meta("site %u %u %s", id, static_cast<unsigned>(kind), name);
	return id;
}

uint16_t SiteId(Site& site) noexcept {
	auto id = site.id.load(std::memory_order_acquire);
	if (id == 0) {
		id = RegisterSite(site.name, site.kind);
		site.id.store(id, std::memory_order_release);
	}
	return id;
}

uint16_t PushScope(uint16_t site) noexcept {
	auto& state = State();
	const uint16_t parent = state.depth != 0 ? state.stack[std::min<uint32_t>(state.depth, 16) - 1] : uint16_t {0};
	if (state.depth < state.stack.size()) {
		state.stack[state.depth] = site;
	}
	state.depth++;
	return parent;
}

bool InScope() noexcept {
	return t_state != nullptr && t_state->depth != 0;
}

void PopScope() noexcept {
	auto& state = State();
	if (state.depth != 0) {
		state.depth--;
	}
}

void Record(uint16_t site, uint64_t t0, uint64_t t1, uint64_t arg, uint64_t arg2,
            uint64_t ret) noexcept {
	auto&      state  = State();
	const uint16_t parent = state.depth != 0 ? state.stack[std::min<uint32_t>(state.depth, 16) - 1] : uint16_t {0};
	if (t1 - t0 < g_min_ticks && t1 > t0 && site < MaxSites &&
	    g_site_short[site].load(std::memory_order_relaxed) != 0) {
		AddSummary(state, site, parent, t1 - t0);
		return;
	}
	Append(state, RecordData {t0, t1, arg, arg2, ret, state.tid, site, parent});
}

uint32_t ExeOffset(const void* address) noexcept {
	const auto a = reinterpret_cast<uintptr_t>(address);
	return a >= g_exe_base && a < g_exe_end ? static_cast<uint32_t>(a - g_exe_base) : 0;
}

uint64_t CallerPair() noexcept {
	void*      frames[6] {};
	const auto n = RtlCaptureStackBackTrace(2, 6, frames, nullptr);
	uint64_t   value  = 0;
	int        stored = 0;
	for (USHORT i = 0; i < n && stored < 2; i++) {
		const auto offset = ExeOffset(frames[i]);
		if (offset != 0) {
			value |= static_cast<uint64_t>(offset) << (32u * stored);
			stored++;
		}
	}
	return value;
}

void Meta(const char* format, ...) noexcept {
	std::lock_guard lock(g_meta_mutex);
	if (g_meta == nullptr) {
		return;
	}
	va_list args;
	va_start(args, format);
	std::vfprintf(g_meta, format, args);
	va_end(args);
	std::fputc('\n', g_meta);
	std::fflush(g_meta);
}

namespace {
struct WakeEntry {
	std::atomic<uint64_t> key {0};
	std::atomic<uint32_t> tid {0};
};
std::array<WakeEntry, 8192> g_wakes;
size_t WakeSlot(const void* object) {
	return static_cast<size_t>((reinterpret_cast<uintptr_t>(object) >> 3u) * 0x9e3779b97f4a7c15ull >> 51u);
}
} // namespace

void NoteWake(const void* object) noexcept {
	if (!Enabled() || object == nullptr) {
		return;
	}
	auto& entry = g_wakes[WakeSlot(object)];
	entry.tid.store(CurrentTid(), std::memory_order_relaxed);
	entry.key.store(reinterpret_cast<uint64_t>(object), std::memory_order_release);
}

uint32_t LastWaker(const void* object) noexcept {
	const auto& entry = g_wakes[WakeSlot(object)];
	return entry.key.load(std::memory_order_acquire) == reinterpret_cast<uint64_t>(object)
	           ? entry.tid.load(std::memory_order_relaxed)
	           : 0;
}

uint64_t GuestCaller() noexcept {
	void*      frames[12] {};
	const auto n = RtlCaptureStackBackTrace(1, 12, frames, nullptr);
	for (USHORT i = 0; i < n; i++) {
		const auto a = reinterpret_cast<uintptr_t>(frames[i]);
		if (ExeOffset(frames[i]) == 0 && a < 0x7f0000000000ull) {
			return a;
		}
	}
	return 0;
}

void DumpCodeOnce(const void* address) noexcept {
	if (!Enabled()) {
		return;
	}
	static std::mutex                      mutex;
	static std::vector<const void*>        seen;
	std::lock_guard                        lock(mutex);
	for (const auto* s: seen) {
		if (s == address) {
			return;
		}
	}
	seen.push_back(address);
	const auto* begin = static_cast<const uint8_t*>(address) - 160;
	std::string hex;
	char        buf[4];
	for (int i = 0; i < 192; i++) {
		std::snprintf(buf, sizeof(buf), "%02x", begin[i]);
		hex += buf;
	}
	Meta("code 0x%llx %s", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(begin)), hex.c_str());
}

void SetPeriodicHook(Hook hook, void* data) noexcept {
	g_hook_data.store(data, std::memory_order_relaxed);
	g_hook.store(hook, std::memory_order_release);
}

uint32_t CurrentTid() noexcept {
	if (t_tid == 0) {
		t_tid = GetCurrentThreadId();
	}
	return t_tid;
}

} // namespace Common::SyncAcct
