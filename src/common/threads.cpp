#include "common/threads.h"

#include "common/assert.h"

#include <algorithm>
#include <bit>
#include <atomic>
#include <cerrno>
#include <chrono>             // IWYU pragma: keep
#include <condition_variable> // IWYU pragma: keep
#include <mutex>

// macOS has no clock_nanosleep.
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS && !defined(__APPLE__)
#define KYTY_POSIX_HIGH_RES_SLEEP
#include <ctime>
#endif

#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#include <tlhelp32.h>
// IWYU pragma: no_include <winbase.h>
constexpr DWORD KYTY_CS_SPIN_COUNT = 4000;

// The logical processors other than those of the cores Thread::ReserveCores() keeps for pinned
// threads (0 while none are reserved), and the next ideal processor to hand out among them.
static std::atomic<uint64_t> g_unreserved_processors {0};
static std::atomic<uint32_t> g_next_ideal_processor {0};
// Set by Thread::PinCurrent() on a processor of a reserved core.
static thread_local bool t_reserved_core = false;

// A default CPU set alone does not keep a thread off the reserved cores: Windows still gives the
// thread an ideal processor there, and a thread made ready while every allowed processor is busy
// waits in that processor's queue, which the thread pinned there never yields to. Astro Bot's
// game threads waited there 2-6 s at a time. A hard affinity, with the ideal processor inside it,
// keeps such threads among the processors they may use. Threads that chose their own processors
// keep them.
static void KeepThreadOffReservedCores(HANDLE thread) noexcept {
	const auto unreserved = g_unreserved_processors.load(std::memory_order_acquire);
	if (unreserved == 0) {
		return;
	}
	DWORD_PTR process_mask = 0;
	DWORD_PTR system_mask  = 0;
	if (!::GetProcessAffinityMask(::GetCurrentProcess(), &process_mask, &system_mask)) {
		return;
	}
	const auto previous = ::SetThreadAffinityMask(thread, static_cast<DWORD_PTR>(unreserved));
	if (previous == 0) {
		return;
	}
	if (previous != process_mask) {
		(void)::SetThreadAffinityMask(thread, previous);
		return;
	}
	auto       allowed = unreserved;
	const auto count   = static_cast<uint32_t>(std::popcount(allowed));
	for (auto skip = g_next_ideal_processor.fetch_add(1, std::memory_order_relaxed) % count;
	     skip != 0; skip--) {
		allowed &= allowed - 1;
	}
	(void)::SetThreadIdealProcessor(thread, static_cast<DWORD>(std::countr_zero(allowed)));
}

// Runs in every thread of the process when it starts, also in those that other libraries and the
// graphics driver create.
static void NTAPI KeepNewThreadOffReservedCores(PVOID /*module*/, DWORD reason, PVOID /*reserved*/) {
	if (reason == DLL_THREAD_ATTACH) {
		KeepThreadOffReservedCores(::GetCurrentThread());
	}
}

#pragma section(".CRT$XLK", read)
extern "C" __declspec(allocate(".CRT$XLK")) const PIMAGE_TLS_CALLBACK kyty_keep_new_thread_off_reserved_cores =
    KeepNewThreadOffReservedCores;
#pragma comment(linker, "/INCLUDE:_tls_used")
#pragma comment(linker, "/INCLUDE:kyty_keep_new_thread_off_reserved_cores")

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

static void SleepHighResolution100ns(uint64_t units_100ns) {
	if (units_100ns == 0) {
		return;
	}

	thread_local HANDLE timer = CreateWaitableTimerExW(
	    nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
	if (timer == nullptr) {
		thread_local HANDLE fallback_timer = CreateWaitableTimerW(nullptr, TRUE, nullptr);
		timer                              = fallback_timer;
	}

	if (timer != nullptr) {
		LARGE_INTEGER due_time {};
		due_time.QuadPart = -static_cast<LONGLONG>(units_100ns);
		if (SetWaitableTimerEx(timer, &due_time, 0, nullptr, nullptr, nullptr, 0) != 0) {
			WaitForSingleObject(timer, INFINITE);
			return;
		}
	}

	std::this_thread::sleep_for(std::chrono::nanoseconds(units_100ns * 100));
}

// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <synchapi.h>
// IWYU pragma: no_include <minwinbase.h>
// IWYU pragma: no_include <__mutex_base>
// IWYU pragma: no_include <__threading_support>
// IWYU pragma: no_include <errhandlingapi.h>
// IWYU pragma: no_include <winerror.h>

using InitializeConditionVariable_func_t = /*WINBASEAPI*/ VOID WINAPI (*)(PCONDITION_VARIABLE);
using WakeConditionVariable_func_t    = /*WINBASEAPI*/ VOID       WINAPI (*)(PCONDITION_VARIABLE);
using WakeAllConditionVariable_func_t = /*WINBASEAPI*/ VOID    WINAPI (*)(PCONDITION_VARIABLE);
using SleepConditionVariableCS_func_t = /*WINBASEAPI*/ BOOL    WINAPI (*)(PCONDITION_VARIABLE,
                                                                          PCRITICAL_SECTION, DWORD);

static InitializeConditionVariable_func_t ResolveInitializeConditionVariable() {
	if (HMODULE h = GetModuleHandle("KernelBase"); h != nullptr) // @suppress("Invalid arguments")
	{
		return reinterpret_cast<InitializeConditionVariable_func_t>(
		    GetProcAddress(h, "InitializeConditionVariable"));
	}
	return nullptr;
}
static WakeConditionVariable_func_t ResolveWakeConditionVariable() {
	if (HMODULE h = GetModuleHandle("KernelBase"); h != nullptr) // @suppress("Invalid arguments")
	{
		return reinterpret_cast<WakeConditionVariable_func_t>(
		    GetProcAddress(h, "WakeConditionVariable"));
	}
	return nullptr;
}
static WakeAllConditionVariable_func_t ResolveWakeAllConditionVariable() {
	if (HMODULE h = GetModuleHandle("KernelBase"); h != nullptr) // @suppress("Invalid arguments")
	{
		return reinterpret_cast<WakeAllConditionVariable_func_t>(
		    GetProcAddress(h, "WakeAllConditionVariable"));
	}
	return nullptr;
}
static SleepConditionVariableCS_func_t ResolveSleepConditionVariableCS() {
	if (HMODULE h = GetModuleHandle("KernelBase"); h != nullptr) // @suppress("Invalid arguments")
	{
		return reinterpret_cast<SleepConditionVariableCS_func_t>(
		    GetProcAddress(h, "SleepConditionVariableCS"));
	}
	return nullptr;
}

#endif

#ifdef KYTY_POSIX_HIGH_RES_SLEEP
// An absolute deadline preserves the requested sleep across signal interruptions.
static void SleepHighResolutionNanos(uint64_t nanos) {
	if (nanos == 0) {
		return;
	}

	constexpr uint64_t NANOS_PER_SEC = 1000000000;

	timespec deadline {};
	if (clock_gettime(CLOCK_MONOTONIC, &deadline) != 0) {
		std::this_thread::sleep_for(std::chrono::nanoseconds(nanos));
		return;
	}

	auto target_nsec = static_cast<uint64_t>(deadline.tv_nsec) + nanos;
	deadline.tv_sec += static_cast<time_t>(target_nsec / NANOS_PER_SEC);
	deadline.tv_nsec = static_cast<long>(target_nsec % NANOS_PER_SEC);

	while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, nullptr) == EINTR) {
	}
}
#endif

namespace Common {

using thread_id_t = std::thread::id;

struct MutexPrivate {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	MutexPrivate() { InitializeCriticalSectionAndSpinCount(&m_cs, KYTY_CS_SPIN_COUNT); }
	~MutexPrivate() { DeleteCriticalSection(&m_cs); }
	KYTY_CLASS_NO_COPY(MutexPrivate);
	CRITICAL_SECTION m_cs {};
#else
	std::recursive_mutex m_mutex;
#endif
};

struct CondVarPrivate {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	CondVarPrivate() {
		static auto func = ResolveInitializeConditionVariable();
		EXIT_NOT_IMPLEMENTED(func == nullptr);
		func(&m_cv);
	}
	~CondVarPrivate() = default;
	KYTY_CLASS_NO_COPY(CondVarPrivate);
	CONDITION_VARIABLE m_cv {};
#else
	std::condition_variable_any m_cv;
#endif
};

static wait_poll_func_t g_cond_wait_poll_callback = nullptr;

struct ThreadPrivate {
	ThreadPrivate(thread_func_t f, void* a): func(f), arg(a), m_thread(&Run, this) {}

	static void Run(ThreadPrivate* t) {
		t->unique_id = Thread::GetThreadIdUnique();
		t->started   = true;
		t->func(t->arg);
	}

	thread_func_t    func;
	void*            arg;
	std::atomic_bool finished    = false;
	std::atomic_bool auto_delete = false;
	std::atomic_bool started     = false;
	int              unique_id   = 0;
	std::thread      m_thread;
};

static thread_id_t      g_main_thread;
static int              g_main_thread_int;
static std::atomic<int> g_thread_counter = 0;

void InitializeThreads() {
	g_main_thread     = std::this_thread::get_id();
	g_main_thread_int = Thread::GetThreadIdUnique();
}

Thread::Thread(thread_func_t func, void* arg)
    : m_thread(std::make_unique<ThreadPrivate>(func, arg)) {
	while (!m_thread->started) {
		Common::Thread::SleepMicro(1000);
	}
}

Thread::~Thread() {
	EXIT_IF(!m_thread->finished && !m_thread->auto_delete);

	m_thread.reset();
}

void Thread::Join() {
	EXIT_IF(m_thread->finished || m_thread->auto_delete);

	m_thread->m_thread.join();

	m_thread->finished = true;
}

void Thread::Detach() {
	EXIT_IF(m_thread->finished || m_thread->auto_delete);

	m_thread->auto_delete = true;
	m_thread->m_thread.detach();
}

void Thread::RaiseCurrentPriority() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif
}

void Thread::LowerCurrentPriority() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
}

void Thread::PinCurrent(int logical_cpu) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (logical_cpu >= 0 && logical_cpu < 64) {
		const auto mask = DWORD_PTR {1} << static_cast<uint32_t>(logical_cpu);
		::SetThreadAffinityMask(::GetCurrentThread(), mask);
		::SetThreadIdealProcessor(::GetCurrentThread(), static_cast<DWORD>(logical_cpu));
		const auto unreserved = g_unreserved_processors.load(std::memory_order_acquire);
		t_reserved_core = unreserved != 0 && (unreserved & mask) == 0;
	}
#else
	(void)logical_cpu;
#endif
}

bool Thread::HasReservedCore() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return t_reserved_core;
#else
	return false;
#endif
}

void Thread::ReserveCores(std::span<const int> logical_cpus) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	ULONG length = 0;
	(void)::GetSystemCpuSetInformation(nullptr, 0, &length, ::GetCurrentProcess(), 0);
	if (length == 0) {
		return;
	}
	std::vector<uint8_t> buffer(length);
	auto* const          first = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.data());
	if (!::GetSystemCpuSetInformation(first, length, &length, ::GetCurrentProcess(), 0)) {
		return;
	}
	const auto for_each_set = [&](auto&& func) {
		for (ULONG offset = 0; offset + sizeof(SYSTEM_CPU_SET_INFORMATION) <= length;) {
			const auto* info =
			    reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
			if (info->Type == CpuSetInformation) {
				func(info->CpuSet);
			}
			offset += info->Size;
		}
	};
	// The physical cores (group, core index) of the reserved logical processors.
	std::vector<std::pair<WORD, BYTE>> reserved;
	for_each_set([&](const auto& set) {
		for (const auto cpu: logical_cpus) {
			if (set.Group == 0 && cpu >= 0 && set.LogicalProcessorIndex == cpu) {
				reserved.emplace_back(set.Group, set.CoreIndex);
			}
		}
	});
	std::vector<ULONG>                 others;
	std::vector<std::pair<WORD, BYTE>> other_cores;
	for_each_set([&](const auto& set) {
		const std::pair core {set.Group, set.CoreIndex};
		if (std::ranges::find(reserved, core) == reserved.end()) {
			others.push_back(set.Id);
			if (std::ranges::find(other_cores, core) == other_cores.end()) {
				other_cores.push_back(core);
			}
		}
	});
	// Smaller processors keep every core for the guest, which expects about seven.
	constexpr size_t MinOtherCores = 6;
	if (reserved.empty() || other_cores.size() < MinOtherCores) {
		return;
	}
	(void)::SetProcessDefaultCpuSets(::GetCurrentProcess(), others.data(),
	                                 static_cast<ULONG>(others.size()));
	// The default CPU sets stay for threads that start without the callback above; every other
	// thread, those already running included, gets a hard affinity off the reserved cores. The
	// calling thread pins itself next.
	uint64_t unreserved = 0;
	for_each_set([&](const auto& set) {
		if (set.Group == 0 && set.LogicalProcessorIndex < 64 &&
		    std::ranges::find(reserved, std::pair {set.Group, set.CoreIndex}) == reserved.end()) {
			unreserved |= uint64_t {1} << set.LogicalProcessorIndex;
		}
	});
	g_unreserved_processors.store(unreserved, std::memory_order_release);
	if (HANDLE snapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	    snapshot != INVALID_HANDLE_VALUE) {
		THREADENTRY32 entry {};
		entry.dwSize = sizeof(entry);
		for (BOOL ok = ::Thread32First(snapshot, &entry); ok; ok = ::Thread32Next(snapshot, &entry)) {
			if (entry.th32OwnerProcessID != ::GetCurrentProcessId() ||
			    entry.th32ThreadID == ::GetCurrentThreadId()) {
				continue;
			}
			if (HANDLE thread = ::OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION,
			                                 FALSE, entry.th32ThreadID);
			    thread != nullptr) {
				KeepThreadOffReservedCores(thread);
				::CloseHandle(thread);
			}
		}
		::CloseHandle(snapshot);
	}
#else
	(void)logical_cpus;
#endif
}

void Thread::SleepMicro(uint32_t micros) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SleepHighResolution100ns(static_cast<uint64_t>(micros) * 10);
#elif defined(KYTY_POSIX_HIGH_RES_SLEEP)
	SleepHighResolutionNanos(static_cast<uint64_t>(micros) * 1000);
#else
	std::this_thread::sleep_for(std::chrono::microseconds(micros));
#endif
}

void Thread::SleepNano(uint64_t nanos) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SleepHighResolution100ns((nanos + 99) / 100);
#elif defined(KYTY_POSIX_HIGH_RES_SLEEP)
	SleepHighResolutionNanos(nanos);
#else
	std::this_thread::sleep_for(std::chrono::nanoseconds(nanos));
#endif
}

bool Thread::IsMainThread() {
	return g_main_thread == std::this_thread::get_id();
}

std::string Thread::GetId() const {
	std::stringstream ss;
	ss << m_thread->m_thread.get_id();
	return ss.str();
}

int Thread::GetUniqueId() const {
	return m_thread->unique_id;
}

std::string Thread::GetThreadId() {
	std::stringstream ss;
	ss << std::this_thread::get_id();
	return ss.str();
}

Mutex::Mutex(): m_mutex(std::make_unique<MutexPrivate>()) {}

Mutex::~Mutex() {
	m_mutex.reset();
}

void Mutex::Lock() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	EnterCriticalSection(&m_mutex->m_cs);
#else
	m_mutex->m_mutex.lock();
#endif
}

void Mutex::Unlock() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	LeaveCriticalSection(&m_mutex->m_cs);
#else
	m_mutex->m_mutex.unlock();
#endif
}

bool Mutex::TryLock() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return (TryEnterCriticalSection(&m_mutex->m_cs) != 0);
#else
	return m_mutex->m_mutex.try_lock();
#endif
}

CondVar::CondVar(): m_cond_var(std::make_unique<CondVarPrivate>()) {}

CondVar::~CondVar() {
	m_cond_var.reset();
}

void CondVar::Wait(Mutex* mutex) {
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	std::unique_lock<std::recursive_mutex> cpp_lock(mutex->m_mutex->m_mutex, std::adopt_lock_t());
#endif
	auto poll_callback = [&] {
		auto* callback = g_cond_wait_poll_callback;
		if (callback == nullptr) {
			return;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		LeaveCriticalSection(&mutex->m_mutex->m_cs);
		callback();
		EnterCriticalSection(&mutex->m_mutex->m_cs);
#else
		cpp_lock.unlock();
		callback();
		cpp_lock.lock();
#endif
	};
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static auto func = ResolveSleepConditionVariableCS();
	EXIT_NOT_IMPLEMENTED(func == nullptr);
	if (g_cond_wait_poll_callback == nullptr) {
		func(&m_cond_var->m_cv, &mutex->m_mutex->m_cs, INFINITE);
	} else {
		if (func(&m_cond_var->m_cv, &mutex->m_mutex->m_cs, 10) == 0 &&
		    GetLastError() == ERROR_TIMEOUT) {
			poll_callback();
		}
	}
#else
	if (g_cond_wait_poll_callback == nullptr) {
		m_cond_var->m_cv.wait(cpp_lock);
	} else {
		if (m_cond_var->m_cv.wait_for(cpp_lock, std::chrono::microseconds(10000)) ==
		    std::cv_status::timeout) {
			poll_callback();
		}
	}
	cpp_lock.release();
#endif
}

void CondVar::SetWaitPollCallback(wait_poll_func_t callback) {
	g_cond_wait_poll_callback = callback;
}

bool CondVar::WaitFor(Mutex* mutex, uint32_t micros) {
	bool ok = false;
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	std::unique_lock<std::recursive_mutex> cpp_lock(mutex->m_mutex->m_mutex, std::adopt_lock_t());
#endif
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static auto func = ResolveSleepConditionVariableCS();
	EXIT_NOT_IMPLEMENTED(func == nullptr);
	ok = !(func(&m_cond_var->m_cv, &mutex->m_mutex->m_cs, (micros < 1000 ? 1 : micros / 1000)) ==
	           0 &&
	       GetLastError() == ERROR_TIMEOUT);
#else
	ok = (m_cond_var->m_cv.wait_for(cpp_lock, std::chrono::microseconds(micros)) ==
	      std::cv_status::no_timeout);
	cpp_lock.release();
#endif
	return ok;
}

void CondVar::Signal() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static auto func = ResolveWakeConditionVariable();
	EXIT_NOT_IMPLEMENTED(func == nullptr);
	func(&m_cond_var->m_cv);
#else
	m_cond_var->m_cv.notify_one();
#endif
}

void CondVar::SignalAll() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static auto func = ResolveWakeAllConditionVariable();
	EXIT_NOT_IMPLEMENTED(func == nullptr);
	func(&m_cond_var->m_cv);
#else
	m_cond_var->m_cv.notify_all();
#endif
}

int Thread::GetThreadIdUnique() {
	static thread_local int tid = ++g_thread_counter;
	return tid;
}

} // namespace Common
