#ifndef KYTY_COMMON_SYNCACCT_H_
#define KYTY_COMMON_SYNCACCT_H_

// Local debugging (sync-wait audit): accounting of every place a thread waits.
//
// KYTY_SYNC_ACCT=1 records, per thread and without locks, each wait (spin or kernel), each
// emulation tax (fault handling, urgent work run for another thread) and point events (flips,
// GPU command buffer timestamps, submits) into a binary trace next to the executable
// (syncacct_<pid>.bin + .meta.txt), plus the CPU time of every thread of the process every
// 50 ms. _Build/sw/sa.py closes the accounting offline (wall = work + waits per thread) and
// builds the frame critical path. Waits shorter than KYTY_SYNC_MIN_NS (default 3000) are only
// summed per site.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <intrin.h>

namespace Common::SyncAcct {

enum class Kind : uint8_t {
	Spin  = 0, // waits on the CPU (busy-wait)
	Block = 1, // waits in the kernel (off the CPU)
	Tax   = 2, // emulation work on the CPU done for someone else (faults, urgent commands)
	Point = 3, // instant events
	Gpu   = 4, // GPU timestamps (t0/t1 in GPU ticks)
	Work  = 5, // named work intervals (to split busy time)
};

inline const bool g_enabled = std::getenv("KYTY_SYNC_ACCT") != nullptr;
[[nodiscard]] inline bool Enabled() noexcept {
	return g_enabled;
}

struct Site {
	const char*           name;
	Kind                  kind;
	std::atomic<uint16_t> id {0};
	constexpr Site(const char* n, Kind k) noexcept: name(n), kind(k) {}
};

// Registers lazily; names are copied.
uint16_t SiteId(Site& site) noexcept;
uint16_t RegisterSite(const char* name, Kind kind) noexcept;

[[nodiscard]] inline uint64_t Now() noexcept {
	return __rdtsc();
}

void Record(uint16_t site, uint64_t t0, uint64_t t1, uint64_t arg, uint64_t arg2,
            uint64_t ret) noexcept;
uint16_t PushScope(uint16_t site) noexcept;
// Whether the calling thread is inside a recorded scope.
bool     InScope() noexcept;
void     PopScope() noexcept;
// Offset of a code address in the executable (0 outside it).
uint32_t ExeOffset(const void* address) noexcept;
// Two return addresses of the caller's callers as exe offsets (for waits deep in helpers).
uint64_t CallerPair() noexcept;
void     Meta(const char* format, ...) noexcept;
// Called every ~2 ms from the writer thread.
using Hook = void (*)(void* data);
void SetPeriodicHook(Hook hook, void* data) noexcept;

uint32_t LastWaker(const void* object) noexcept;

class Scope {
public:
	explicit Scope(Site& site, uint64_t arg = 0, uint64_t arg2 = 0, uint64_t ret = 0) noexcept {
		if (Enabled()) {
			Begin(SiteId(site), arg, arg2, ret);
		}
	}
	Scope(uint16_t site_id, uint64_t arg, uint64_t arg2 = 0, uint64_t ret = 0) noexcept {
		if (Enabled() && site_id != 0) {
			Begin(site_id, arg, arg2, ret);
		}
	}
	~Scope() { End(); }
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

	void SetArg(uint64_t value) noexcept { m_arg = value; }
	void SetArg2(uint64_t value) noexcept { m_arg2 = value; }
	void SetRet(uint64_t value) noexcept { m_ret = value; }
	// arg2 becomes the last thread that signaled `object` (NoteWake) when the scope ends.
	void SetWakerObject(const void* object) noexcept { m_waker_object = object; }
	void End() noexcept {
		if (m_site != 0) {
			const auto t1 = Now();
			PopScope();
			if (m_waker_object != nullptr) {
				m_arg2 = LastWaker(m_waker_object);
			}
			Record(m_site, m_t0, t1, m_arg, m_arg2, m_ret);
			m_site = 0;
		}
	}
	// The wait did not happen after all.
	void Cancel() noexcept {
		if (m_site != 0) {
			PopScope();
			m_site = 0;
		}
	}

private:
	void Begin(uint16_t site, uint64_t arg, uint64_t arg2, uint64_t ret) noexcept {
		m_site = site;
		m_arg  = arg;
		m_arg2 = arg2;
		m_ret  = ret;
		PushScope(site);
		m_t0 = Now();
	}
	uint16_t    m_site         = 0;
	const void* m_waker_object = nullptr;
	uint64_t    m_t0           = 0;
	uint64_t m_arg  = 0;
	uint64_t m_arg2 = 0;
	uint64_t m_ret  = 0;
};

inline void Point(Site& site, uint64_t arg = 0, uint64_t arg2 = 0, uint64_t ret = 0) noexcept {
	if (Enabled()) {
		const auto now = Now();
		Record(SiteId(site), now, now, arg, arg2, ret);
	}
}
inline void Interval(Site& site, uint64_t t0, uint64_t t1, uint64_t arg = 0, uint64_t arg2 = 0,
                     uint64_t ret = 0) noexcept {
	if (Enabled()) {
		Record(SiteId(site), t0, t1, arg, arg2, ret);
	}
}

// The waker of a wait: the host thread id that released it.
uint32_t CurrentTid() noexcept;
// Signal sites note who signaled an object; waits read it back when they end.
void     NoteWake(const void* object) noexcept;
uint32_t LastWaker(const void* object) noexcept;
// The first return address of the stack outside the executable (the guest caller).
uint64_t GuestCaller() noexcept;
// Writes the guest code around a return address to the meta file, once per address.
void     DumpCodeOnce(const void* address) noexcept;

} // namespace Common::SyncAcct

#define KYTY_SYNC_SITE(var, name, kind)                                                            \
	static ::Common::SyncAcct::Site var {name, ::Common::SyncAcct::Kind::kind}

// A guest-facing wait API: arg = object, arg2 = the thread that last signaled `object`,
// ret = the guest return address.
#define KYTY_SYNC_GAME_WAIT(name, object)                                                          	KYTY_SYNC_SITE(sync_game_site_, name, Block);                                                  	::Common::SyncAcct::Scope sync_game_scope_(sync_game_site_, reinterpret_cast<uint64_t>(object), 0, 	                                           reinterpret_cast<uint64_t>(__builtin_return_address(0))); 	sync_game_scope_.SetWakerObject(reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(object)))

#endif // KYTY_COMMON_SYNCACCT_H_
