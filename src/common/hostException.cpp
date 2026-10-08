#include "common/syncAcct.h"
#include "common/hostException.h"

#include <atomic>
#include <cstdio>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
// windows.h first
#include <tlhelp32.h>

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <thread>
#include <vector>
#else
#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <initializer_list>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/ucontext.h>
#else
#include <ucontext.h> // IWYU pragma: keep
#endif
#endif

// IWYU pragma: no_include <errhandlingapi.h>
// IWYU pragma: no_include <excpt.h>
// IWYU pragma: no_include <minwinbase.h>
// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <wtypes.h>

namespace Common::HostException {

#if !defined(__APPLE__)

static std::atomic<Handler> g_handler {nullptr};
static std::atomic_uint32_t g_install_state {0};

static_assert(decltype(g_handler)::is_always_lock_free);
static_assert(decltype(g_install_state)::is_always_lock_free);
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX

// macOS uses the same POSIX platform setting and needs a signal stack too.
class ThreadSignalStack {
public:
	ThreadSignalStack() {
		const auto page_size = static_cast<size_t>(::getpagesize());
		const auto stack_size =
		    (std::max<size_t>(64 * 1024, MINSIGSTKSZ) + page_size - 1) & ~(page_size - 1);
		if (::posix_memalign(&m_memory, page_size, stack_size) != 0) {
			return;
		}

		stack_t stack {};
		stack.ss_sp   = m_memory;
		stack.ss_size = stack_size;
		if (::sigaltstack(&stack, &m_previous) != 0) {
			std::free(m_memory);
			m_memory = nullptr;
		}
	}

	~ThreadSignalStack() {
		if (m_memory != nullptr && ::sigaltstack(&m_previous, nullptr) == 0) {
			std::free(m_memory);
		}
	}

	[[nodiscard]] bool IsInitialized() const { return m_memory != nullptr; }

	KYTY_CLASS_NO_COPY(ThreadSignalStack)

private:
	void*   m_memory = nullptr;
	stack_t m_previous {};
};

bool InitializeThreadSignalStack() {
	// Keep fault handling off guest stacks, which GPU tracking can make read-only.
	thread_local ThreadSignalStack signal_stack;
	return signal_stack.IsInitialized();
}

#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

// Local debugging: KYTY_WATCH_WRITE=<hex address> arms a hardware write watchpoint on the
// 8 bytes at that address in every thread of the process (rearmed as threads appear) and logs
// each distinct writer, plus every write that stores KYTY_WATCH_VALUE (hex, default all ones).
static uint64_t             g_watch_address = 0;
static uint64_t             g_watch_value   = UINT64_MAX;
static std::atomic_uint64_t g_watch_hits {0};
static std::atomic_uint32_t g_watch_logged {0};
static std::atomic<uint64_t> g_watch_writers[64] {};

static LONG WINAPI WatchFilter(PEXCEPTION_POINTERS exception) noexcept {
	if (exception->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP ||
	    (exception->ContextRecord->Dr6 & 0xfu) == 0) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	auto* context = exception->ContextRecord;
	context->Dr6  = 0;
	const auto value = *reinterpret_cast<volatile uint64_t*>(g_watch_address);
	const auto hits  = g_watch_hits.fetch_add(1, std::memory_order_relaxed) + 1;
	bool       fresh = false;
	for (auto& writer: g_watch_writers) {
		uint64_t current = writer.load(std::memory_order_relaxed);
		if (current == context->Rip) {
			break;
		}
		if (current == 0 && writer.compare_exchange_strong(current, context->Rip)) {
			fresh = true;
			break;
		}
		if (current == context->Rip) {
			break;
		}
	}
	if ((fresh || value == g_watch_value) &&
	    g_watch_logged.fetch_add(1, std::memory_order_relaxed) < 400) {
		HMODULE module = nullptr;
		GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
		                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		                   reinterpret_cast<LPCSTR>(context->Rip), &module);
		std::printf("WATCH write 0x%016" PRIx64 " = 0x%016" PRIx64 " hit=%" PRIu64
		            " tid=%lu rip=0x%016" PRIx64 " host_module=%p %s\n",
		            g_watch_address, value, hits, GetCurrentThreadId(), context->Rip,
		            static_cast<void*>(module), fresh ? "new-writer" : "WATCHED-VALUE");
		PWSTR thread_name = nullptr;
		if (SUCCEEDED(GetThreadDescription(GetCurrentThread(), &thread_name)) && thread_name != nullptr) {
			std::printf("  thread: %ls\n", thread_name);
			LocalFree(thread_name);
		}
		std::printf("  rax=%016" PRIx64 " rbx=%016" PRIx64 " rcx=%016" PRIx64 " rdx=%016" PRIx64
		            " rsi=%016" PRIx64 " rdi=%016" PRIx64 " rbp=%016" PRIx64 " rsp=%016" PRIx64 "\n",
		            context->Rax, context->Rbx, context->Rcx, context->Rdx, context->Rsi,
		            context->Rdi, context->Rbp, context->Rsp);
		std::printf("  r8=%016" PRIx64 " r9=%016" PRIx64 " r10=%016" PRIx64 " r11=%016" PRIx64
		            " r12=%016" PRIx64 " r13=%016" PRIx64 " r14=%016" PRIx64 " r15=%016" PRIx64 "\n",
		            context->R8, context->R9, context->R10, context->R11, context->R12,
		            context->R13, context->R14, context->R15);
		const auto* stack = reinterpret_cast<const uint64_t*>(context->Rsp);
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(stack, &info, sizeof(info)) != 0 && info.State == MEM_COMMIT &&
		    reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize >=
		        context->Rsp + 48 * sizeof(uint64_t)) {
			std::printf("  stack:");
			for (int i = 0; i < 48; i++) {
				std::printf("%s %016" PRIx64, (i % 6 == 0) ? "\n   " : "", stack[i]);
			}
			std::printf("\n");
		}
		std::fflush(stdout);
	}
	return EXCEPTION_CONTINUE_EXECUTION;
}

static void ArmWatch(DWORD thread_id) {
	HANDLE thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME,
	                           FALSE, thread_id);
	if (thread == nullptr) {
		return;
	}
	if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
		CONTEXT context {};
		context.ContextFlags = CONTEXT_DEBUG_REGISTERS;
		if (GetThreadContext(thread, &context) != 0) {
			context.Dr0 = g_watch_address;
			// L0, R/W0 = 01 (writes), LEN0 = 10 (8 bytes).
			context.Dr7 = (context.Dr7 & ~uint64_t {0xf0003}) | 1u | (1u << 16u) | (2u << 18u);
			SetThreadContext(thread, &context);
		}
		ResumeThread(thread);
	}
	CloseHandle(thread);
}

static void ArmWatchThreads();

static void StartWriteWatch() {
	const char* address = std::getenv("KYTY_WATCH_WRITE");
	if (address == nullptr) {
		return;
	}
	g_watch_address = std::strtoull(address, nullptr, 16) & ~uint64_t {7};
	if (const char* value = std::getenv("KYTY_WATCH_VALUE"); value != nullptr) {
		g_watch_value = std::strtoull(value, nullptr, 16);
	}
	ArmWatchThreads();
}

// Debug: arms the write watchpoint at runtime on an address known only then (once).
void ArmWriteWatch(uint64_t address) {
	static std::atomic_bool armed {false};
	if (address == 0 || armed.exchange(true)) {
		return;
	}
	g_watch_address = address & ~uint64_t {7};
	ArmWatchThreads();
}

static void ArmWatchThreads() {
	if (g_watch_address == 0 || AddVectoredExceptionHandler(1, WatchFilter) == nullptr) {
		return;
	}
	std::printf("WATCH armed on 0x%016" PRIx64 " (value 0x%016" PRIx64 ")\n", g_watch_address,
	            g_watch_value);
	std::thread([] {
		const auto self = GetCurrentThreadId();
		const auto pid  = GetCurrentProcessId();
		std::vector<DWORD> armed;
		for (;;) {
			HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
			if (snapshot != INVALID_HANDLE_VALUE) {
				THREADENTRY32 entry {};
				entry.dwSize = sizeof(entry);
				std::vector<DWORD> current;
				for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
					if (entry.th32OwnerProcessID == pid && entry.th32ThreadID != self) {
						current.push_back(entry.th32ThreadID);
					}
				}
				CloseHandle(snapshot);
				for (const auto id: current) {
					if (std::find(armed.begin(), armed.end(), id) == armed.end()) {
						ArmWatch(id);
					}
				}
				armed = std::move(current);
			}
			Sleep(20);
		}
	}).detach();
}

static LONG WINAPI ExceptionFilter(PEXCEPTION_POINTERS exception) noexcept {
	auto* exception_record = exception->ExceptionRecord;

	if (exception_record->ExceptionCode == DBG_PRINTEXCEPTION_C ||
	    exception_record->ExceptionCode == DBG_PRINTEXCEPTION_WIDE_C) {
		return EXCEPTION_CONTINUE_SEARCH;
	}

	if (exception_record->ExceptionCode == 0x406D1388) {
		// Set a thread name.
		return EXCEPTION_CONTINUE_EXECUTION;
	}

	ExceptionInfo info {};
	info.exception_address = reinterpret_cast<uint64_t>(exception_record->ExceptionAddress);
	info.native_code       = exception_record->ExceptionCode;
	info.native_context    = exception->ContextRecord;

	if (exception_record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
		info.type = ExceptionType::AccessViolation;
		switch (exception_record->ExceptionInformation[0]) {
			case 0: info.access_violation_type = AccessViolationType::Read; break;
			case 1: info.access_violation_type = AccessViolationType::Write; break;
			case 8: info.access_violation_type = AccessViolationType::Execute; break;
			default: info.access_violation_type = AccessViolationType::Unknown; break;
		}
		info.access_violation_vaddr = exception_record->ExceptionInformation[1];
	} else if (exception_record->ExceptionCode == EXCEPTION_ILLEGAL_INSTRUCTION) {
		info.type = ExceptionType::IllegalInstruction;
	} else {
		return EXCEPTION_CONTINUE_SEARCH;
	}

	info.rax = exception->ContextRecord->Rax;
	info.rbx = exception->ContextRecord->Rbx;
	info.rcx = exception->ContextRecord->Rcx;
	info.rdx = exception->ContextRecord->Rdx;
	info.rsi = exception->ContextRecord->Rsi;
	info.rdi = exception->ContextRecord->Rdi;
	info.rbp = exception->ContextRecord->Rbp;
	info.rsp = exception->ContextRecord->Rsp;
	info.r8  = exception->ContextRecord->R8;
	info.r9  = exception->ContextRecord->R9;
	info.r10 = exception->ContextRecord->R10;
	info.r11 = exception->ContextRecord->R11;
	info.r12 = exception->ContextRecord->R12;
	info.r13 = exception->ContextRecord->R13;
	info.r14 = exception->ContextRecord->R14;
	info.r15 = exception->ContextRecord->R15;

	const auto handler = g_handler.load(std::memory_order_acquire);
	// Debug (KYTY_SYNC_ACCT): the emulator's exception handling (arg = vaddr, arg2 = type).
	KYTY_SYNC_SITE(s_av, "fault: access violation handler", Tax);
	KYTY_SYNC_SITE(s_ill, "fault: illegal instruction handler", Tax);
	SyncAcct::Scope sync_scope(info.type == ExceptionType::AccessViolation ? s_av : s_ill,
	                           info.access_violation_vaddr,
	                           static_cast<uint64_t>(info.access_violation_type),
	                           info.exception_address);
	if (handler != nullptr && handler(info)) {
		return EXCEPTION_CONTINUE_EXECUTION;
	}
	return EXCEPTION_CONTINUE_SEARCH;
}

#elif defined(__APPLE__)

static std::atomic<Handler> g_handler {nullptr};
static std::atomic_uint32_t g_install_state {0};

static_assert(decltype(g_handler)::is_always_lock_free);
static_assert(decltype(g_install_state)::is_always_lock_free);

// Translate the x86-64 page-fault error code (mcontext __es.__err) into an access type.
// bit 1 (0x2) = write, bit 4 (0x10) = instruction fetch, otherwise a read.
static AccessViolationType DecodeAccess(uint64_t err) {
	if ((err & 0x10u) != 0) {
		return AccessViolationType::Execute;
	}
	if ((err & 0x2u) != 0) {
		return AccessViolationType::Write;
	}
	return AccessViolationType::Read;
}

// POSIX signal handler that mirrors the Windows vectored handler: build an ExceptionInfo
// from the mcontext and dispatch. A resolved fault (handler returns true) simply returns,
// re-executing the faulting instruction against the now-fixed protection. An unresolved
// fault restores the default disposition so the retry terminates the process.
static void SignalHandler(int sig, siginfo_t* si, void* uctx) {
	auto*       uc = static_cast<ucontext_t*>(uctx);
	const auto* mc = uc->uc_mcontext;
	const auto& ss = mc->__ss;

	ExceptionInfo info {};
	info.exception_address = ss.__rip;
	info.native_code       = static_cast<uint32_t>(si->si_code);
	info.native_context    = uctx;

	if (sig == SIGILL) {
		info.type = ExceptionType::IllegalInstruction;
	} else {
		info.type                   = ExceptionType::AccessViolation;
		info.access_violation_type  = DecodeAccess(mc->__es.__err);
		info.access_violation_vaddr = reinterpret_cast<uint64_t>(si->si_addr);
	}

	info.rax = ss.__rax;
	info.rbx = ss.__rbx;
	info.rcx = ss.__rcx;
	info.rdx = ss.__rdx;
	info.rsi = ss.__rsi;
	info.rdi = ss.__rdi;
	info.rbp = ss.__rbp;
	info.rsp = ss.__rsp;
	info.r8  = ss.__r8;
	info.r9  = ss.__r9;
	info.r10 = ss.__r10;
	info.r11 = ss.__r11;
	info.r12 = ss.__r12;
	info.r13 = ss.__r13;
	info.r14 = ss.__r14;
	info.r15 = ss.__r15;

	const auto handler = g_handler.load(std::memory_order_acquire);
	if (handler != nullptr && handler(info)) {
		return; // retry the faulting instruction against the fixed mapping
	}

	// Unresolved: restore the default action so the re-executed instruction terminates.
	struct sigaction dfl {};
	dfl.sa_handler = SIG_DFL;
	sigemptyset(&dfl.sa_mask);
	sigaction(sig, &dfl, nullptr);
}

#else

// x86-64 page-fault error bits.
constexpr uint64_t PAGE_FAULT_ERROR_WRITE       = 0x02;
constexpr uint64_t PAGE_FAULT_ERROR_INSTRUCTION = 0x10;

// Let the kernel handle an unresolved fault on retry.
static void ChainToDefault(int signal_number) noexcept {
	struct sigaction restore {};
	restore.sa_handler = SIG_DFL;
	sigemptyset(&restore.sa_mask);
	restore.sa_flags = 0;
	::sigaction(signal_number, &restore, nullptr);
}

static void SignalHandler(int signal_number, siginfo_t* signal_info, void* native_context) {
	auto* context = static_cast<ucontext_t*>(native_context);
	auto* gregs   = context->uc_mcontext.gregs;

	ExceptionInfo info {};
	info.exception_address = static_cast<uint64_t>(gregs[REG_RIP]);
	info.native_code       = static_cast<uint32_t>(signal_number);
	info.native_context    = context;

	if (signal_number == SIGSEGV || signal_number == SIGBUS) {
		info.type             = ExceptionType::AccessViolation;
		const auto error_code = static_cast<uint64_t>(gregs[REG_ERR]);
		if ((error_code & PAGE_FAULT_ERROR_INSTRUCTION) != 0) {
			info.access_violation_type = AccessViolationType::Execute;
		} else if ((error_code & PAGE_FAULT_ERROR_WRITE) != 0) {
			info.access_violation_type = AccessViolationType::Write;
		} else {
			info.access_violation_type = AccessViolationType::Read;
		}
		info.access_violation_vaddr = reinterpret_cast<uint64_t>(signal_info->si_addr);
	} else if (signal_number == SIGILL) {
		info.type = ExceptionType::IllegalInstruction;
	} else {
		ChainToDefault(signal_number);
		return;
	}

	info.rax = static_cast<uint64_t>(gregs[REG_RAX]);
	info.rbx = static_cast<uint64_t>(gregs[REG_RBX]);
	info.rcx = static_cast<uint64_t>(gregs[REG_RCX]);
	info.rdx = static_cast<uint64_t>(gregs[REG_RDX]);
	info.rsi = static_cast<uint64_t>(gregs[REG_RSI]);
	info.rdi = static_cast<uint64_t>(gregs[REG_RDI]);
	info.rbp = static_cast<uint64_t>(gregs[REG_RBP]);
	info.rsp = static_cast<uint64_t>(gregs[REG_RSP]);
	info.r8  = static_cast<uint64_t>(gregs[REG_R8]);
	info.r9  = static_cast<uint64_t>(gregs[REG_R9]);
	info.r10 = static_cast<uint64_t>(gregs[REG_R10]);
	info.r11 = static_cast<uint64_t>(gregs[REG_R11]);
	info.r12 = static_cast<uint64_t>(gregs[REG_R12]);
	info.r13 = static_cast<uint64_t>(gregs[REG_R13]);
	info.r14 = static_cast<uint64_t>(gregs[REG_R14]);
	info.r15 = static_cast<uint64_t>(gregs[REG_R15]);

	const auto handler = g_handler.load(std::memory_order_acquire);
	if (handler != nullptr && handler(info)) {
		return;
	}

	ChainToDefault(signal_number);
}

#endif

bool InstallHandler(Handler handler) {
	if (handler == nullptr) {
		return false;
	}

	uint32_t expected_state = 0;
	if (!g_install_state.compare_exchange_strong(expected_state, 1, std::memory_order_acq_rel)) {
		return expected_state == 2 && g_handler.load(std::memory_order_acquire) == handler;
	}

	g_handler.store(handler, std::memory_order_release);

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (AddVectoredExceptionHandler(0, ExceptionFilter) == nullptr) {
		g_handler.store(nullptr, std::memory_order_release);
		g_install_state.store(0, std::memory_order_release);
		printf("AddVectoredExceptionHandler() failed\n");
		return false;
	}
	StartWriteWatch();
#elif defined(__APPLE__)
	struct sigaction sa {};
	sa.sa_sigaction = SignalHandler;
	sa.sa_flags     = SA_SIGINFO | SA_ONSTACK;
	sigemptyset(&sa.sa_mask);
	// The guest signal-dispatch path (KernelRaiseException) interrupts threads with
	// SIGUSR1; block it while a fault is being resolved so a stop-the-world request
	// cannot preempt the handler between the protection fix and the retry.
	sigaddset(&sa.sa_mask, SIGUSR1);

	// macOS raises SIGBUS for protection faults on some paths and SIGSEGV on others;
	// SIGILL covers instructions the host cannot execute (routed to the x64 emulator).
	bool ok = sigaction(SIGSEGV, &sa, nullptr) == 0 && sigaction(SIGBUS, &sa, nullptr) == 0 &&
	          sigaction(SIGILL, &sa, nullptr) == 0;
	if (!ok) {
		g_handler.store(nullptr, std::memory_order_release);
		g_install_state.store(0, std::memory_order_release);
		printf("sigaction() failed to install the host fault handler\n");
		return false;
	}
#else
	struct sigaction action {};
	action.sa_sigaction = SignalHandler;
	sigemptyset(&action.sa_mask);
	action.sa_flags = SA_SIGINFO | SA_RESTART | SA_ONSTACK;

	for (const int signal_number: {SIGSEGV, SIGBUS, SIGILL}) {
		if (::sigaction(signal_number, &action, nullptr) != 0) {
			g_handler.store(nullptr, std::memory_order_release);
			g_install_state.store(0, std::memory_order_release);
			printf("sigaction(%d) failed\n", signal_number);
			return false;
		}
	}
#endif

	g_install_state.store(2, std::memory_order_release);
	return true;
}

} // namespace Common::HostException
