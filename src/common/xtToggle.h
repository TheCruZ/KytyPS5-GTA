#ifndef KYTY_COMMON_XTTOGGLE_H_
#define KYTY_COMMON_XTTOGGLE_H_

// Local A/B harness (perf-scene47, never commit): KYTY_XT_A=<s> turns experiment A on and off
// every s seconds; the window loop prints the presented frames of each period ("XTF").
// Without it the experiment is on. KYTY_XT_A=-1 keeps it off.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Common {
inline bool       g_xt_a    = true;
// KYTY_XT_HOST=1: experiment A is the cached host stream ring (instead of the copy-thread path).
inline const bool g_xt_host = std::getenv("KYTY_XT_HOST") != nullptr;
// Stage waits in TSC cycles (exec waiting for input, resolve waiting for input / for space).
inline std::atomic<uint64_t> g_xt_exec_idle {0};
inline std::atomic<uint64_t> g_xt_resolve_idle {0};
inline std::atomic<uint64_t> g_xt_resolve_full {0};
// KYTY_XT_EXP=<NAME>[,<NAME>...]: the experiments that follow the toggle (KYTY_XA(NAME) is
// g_xt_a for them, false for the others).
inline bool XtSelected(const char* name) noexcept {
	const char* list = std::getenv("KYTY_XT_EXP");
	return list != nullptr && std::strstr(list, name) != nullptr;
}
// Experiments that went to the clean branch are on unless selected for the toggle.
inline bool XtAdopted(const char* name) noexcept {
	return std::strcmp(name, "TBLPATCH") == 0 || std::strcmp(name, "FASTBUILD") == 0 ||
	       std::strcmp(name, "GATHER") == 0;
}
} // namespace Common

#define KYTY_XA(NAME)                                                                              	([]() noexcept {                                                                               		static const bool kyty_xt_sel     = ::Common::XtSelected(#NAME);                           		static const bool kyty_xt_adopted = ::Common::XtAdopted(#NAME);                            		return kyty_xt_sel ? ::Common::g_xt_a : kyty_xt_adopted;                                   	}())

#endif // KYTY_COMMON_XTTOGGLE_H_
