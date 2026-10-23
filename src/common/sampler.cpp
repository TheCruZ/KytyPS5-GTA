#include "common/sampler.h"

#include "common/assert.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <timeapi.h>
#endif

namespace Common::Sampler {

#ifdef _WIN32

namespace {

const auto g_process_start = std::chrono::steady_clock::now();

struct Config {
	bool     enabled = false;
	uint64_t start   = 170;
	uint64_t seconds = 10;
};

const Config& GetConfig() {
	static const Config config = [] {
		Config c;
		const char* value = std::getenv("KYTY_SAMPLE");
		if (value == nullptr) {
			return c;
		}
		c.enabled = true;
		char* end = nullptr;
		c.start   = std::strtoull(value, &end, 10);
		if (end != nullptr && *end == ':') {
			c.seconds = std::strtoull(end + 1, nullptr, 10);
		}
		return c;
	}();
	return config;
}

struct Target {
	std::string name;
	HANDLE      handle     = nullptr;
	uintptr_t   stack_base = 0;
};

constexpr int MaxCallers = 4;

struct Sample {
	uint32_t target;
	DWORD64  rip;
	DWORD64  callers[MaxCallers]; // return addresses into the executable found on the stack
};

uintptr_t g_image_begin = 0, g_image_end = 0;

// A plausible return address: inside the executable, right after a call instruction.
bool IsReturnAddress(uintptr_t a) {
	if (a < g_image_begin + 16 || a >= g_image_end) return false;
	const auto* code = reinterpret_cast<const uint8_t*>(a);
	if (code[-5] == 0xE8) return true;                                // call rel32
	if (code[-6] == 0xFF && (code[-5] & 0x38) == 0x10) return true;   // call [rip+disp32] / [r+disp32]
	if (code[-2] == 0xFF && (code[-1] & 0x38) == 0x10) return true;   // call r / [r]
	if (code[-3] == 0xFF && (code[-2] & 0x38) == 0x10) return true;   // call [r+disp8]
	if (code[-7] == 0xFF && (code[-6] & 0x38) == 0x10) return true;   // call [r+disp32] with SIB
	return false;
}

// Scans the top of a suspended thread's stack (up to its base) for return addresses.
void ScanStack(DWORD64 rsp, uintptr_t stack_base, DWORD64* callers) {
	const auto* stack = reinterpret_cast<const uintptr_t*>(rsp);
	int         found = 0;
	for (int k = 0; k < 512 && found < MaxCallers; k++) {
		if (rsp + (k + 1) * sizeof(uintptr_t) > stack_base) break;
		if (IsReturnAddress(stack[k])) callers[found++] = stack[k];
	}
}

std::mutex          g_mutex;
std::vector<Target> g_targets;
bool                g_started = false;

std::string FunctionOf(const std::string& symbol) {
	// " name+0xoffset [file:line]" -> "name"
	auto s = symbol;
	if (!s.empty() && s[0] == ' ') s.erase(0, 1);
	if (const auto plus = s.find("+0x"); plus != std::string::npos) s.resize(plus);
	return s.empty() ? "?" : s;
}

std::string FileOf(const std::string& symbol) {
	const auto open  = symbol.find(" [");
	const auto colon = symbol.rfind(':');
	if (open == std::string::npos || colon == std::string::npos || colon < open) return "?";
	auto file = symbol.substr(open + 2, colon - open - 2);
	if (const auto slash = file.find_last_of("\\/"); slash != std::string::npos) file.erase(0, slash + 1);
	return file;
}

void Run() {
	const auto& config = GetConfig();
	std::this_thread::sleep_until(g_process_start + std::chrono::seconds(config.start));
	std::vector<Target> targets;
	{
		std::scoped_lock lock(g_mutex);
		targets = g_targets;
	}
	if (targets.empty()) {
		return;
	}
	{
		MODULEINFO info {};
		if (GetModuleInformation(GetCurrentProcess(), GetModuleHandleW(nullptr), &info, sizeof(info)) != FALSE) {
			g_image_begin = reinterpret_cast<uintptr_t>(info.lpBaseOfDll);
			g_image_end   = g_image_begin + info.SizeOfImage;
		}
	}
	timeBeginPeriod(1);
	// No allocation while a target is suspended (it may hold the heap lock).
	std::vector<Sample> samples;
	samples.reserve(static_cast<size_t>(config.seconds) * 1200u * targets.size() + 16u);
	const auto stop = std::chrono::steady_clock::now() + std::chrono::seconds(config.seconds);
	while (std::chrono::steady_clock::now() < stop && samples.size() + targets.size() < samples.capacity()) {
		for (uint32_t i = 0; i < targets.size(); i++) {
			if (SuspendThread(targets[i].handle) == static_cast<DWORD>(-1)) continue;
			CONTEXT context {};
			context.ContextFlags = CONTEXT_CONTROL;
			const bool ok        = GetThreadContext(targets[i].handle, &context) != FALSE;
			Sample     sample {i, context.Rip, {}};
			if (ok) {
				ScanStack(context.Rsp, targets[i].stack_base, sample.callers);
			}
			ResumeThread(targets[i].handle);
			if (ok) samples.push_back(sample);
		}
		Sleep(1);
	}
	timeEndPeriod(1);
	// Aggregate by function.
	std::unordered_map<DWORD64, std::string> names;
	for (uint32_t t = 0; t < targets.size(); t++) {
		std::map<std::string, uint64_t> by_function, by_file;
		uint64_t                        total = 0;
		for (const auto& s: samples) {
			if (s.target != t) continue;
			auto it = names.find(s.rip);
			if (it == names.end()) it = names.emplace(s.rip, Common::HostAddressName(reinterpret_cast<const void*>(s.rip))).first;
			by_function[FunctionOf(it->second)]++;
			by_file[FileOf(it->second)]++;
			total++;
		}
		std::vector<std::pair<std::string, uint64_t>> f(by_function.begin(), by_function.end()), g(by_file.begin(), by_file.end());
		std::sort(f.begin(), f.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
		std::sort(g.begin(), g.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
		std::string out = "[sample] thread " + targets[t].name + ": " + std::to_string(total) + " samples\n";
		char        line[512];
		for (size_t i = 0; i < f.size() && i < 45; i++) {
			std::snprintf(line, sizeof(line), "[sample]   %5.1f%%  %s\n", 100.0 * static_cast<double>(f[i].second) / static_cast<double>(std::max<uint64_t>(total, 1)), f[i].first.c_str());
			out += line;
		}
		// Callers of the hottest leaf functions (heuristic stack scan: may include stale frames).
		for (size_t i = 0; i < f.size() && i < 12; i++) {
			std::map<std::string, uint64_t> chains;
			uint64_t                        n = 0;
			for (const auto& s: samples) {
				if (s.target != t || FunctionOf(names[s.rip]) != f[i].first) continue;
				std::string chain;
				for (const auto c: s.callers) {
					if (c == 0) break;
					auto it = names.find(c);
					if (it == names.end()) it = names.emplace(c, Common::HostAddressName(reinterpret_cast<const void*>(c))).first;
					if (!chain.empty()) chain += " < ";
					chain += FunctionOf(it->second);
				}
				chains[chain.empty() ? "?" : chain]++;
				n++;
			}
			std::vector<std::pair<std::string, uint64_t>> c(chains.begin(), chains.end());
			std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
			out += "[sample]  callers of " + f[i].first + ":\n";
			for (size_t j = 0; j < c.size() && j < 5; j++) {
				std::snprintf(line, sizeof(line), "[sample]      %5.1f%%  %s\n", 100.0 * static_cast<double>(c[j].second) / static_cast<double>(std::max<uint64_t>(n, 1)), c[j].first.substr(0, 400).c_str());
				out += line;
			}
		}
		out += "[sample]  by file:";
		for (size_t i = 0; i < g.size() && i < 20; i++) {
			std::snprintf(line, sizeof(line), " %s %.1f%%", g[i].first.c_str(), 100.0 * static_cast<double>(g[i].second) / static_cast<double>(std::max<uint64_t>(total, 1)));
			out += line;
		}
		out += "\n";
		std::fputs(out.c_str(), stderr);
	}
	std::fflush(stderr);
}

} // namespace

void RegisterCurrentThread(const char* name) {
	if (!GetConfig().enabled) {
		return;
	}
	HANDLE handle = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
	                    THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE,
	                    0) == FALSE) {
		return;
	}
	std::scoped_lock lock(g_mutex);
	const auto* tib = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
	g_targets.push_back({name, handle, reinterpret_cast<uintptr_t>(tib->StackBase)});
	if (!g_started) {
		g_started = true;
		std::thread(Run).detach();
	}
}

#else

void RegisterCurrentThread(const char* /*name*/) {}

#endif

} // namespace Common::Sampler
