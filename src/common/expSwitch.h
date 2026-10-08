#ifndef KYTY_COMMON_EXPSWITCH_H_
#define KYTY_COMMON_EXPSWITCH_H_

// Local debugging (exec-batch experiments): every experimental optimization sits behind
// KYTY_X_<NAME>. Default on; KYTY_X_<NAME>=0 turns one off, KYTY_X_ALL=0 turns all off (an
// explicit KYTY_X_<NAME>=1 keeps that one on).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

namespace Common {

inline bool ExpSwitch(const char* name) noexcept {
	if (const char* value = std::getenv(name); value != nullptr) {
		return value[0] != '0';
	}
	if (const char* all = std::getenv("KYTY_X_ALL"); all != nullptr) {
		return all[0] != '0';
	}
	return true;
}

} // namespace Common

#define KYTY_X(NAME)                                                                               \
	([]() noexcept {                                                                               \
		static const bool kyty_x_on = [] {                                                         \
			const bool on = ::Common::ExpSwitch("KYTY_X_" #NAME);                                  \
			std::printf("KYTY_X_%s=%d\n", #NAME, on ? 1 : 0);                                      \
			return on;                                                                             \
		}();                                                                                       \
		return kyty_x_on;                                                                          \
	}())

// Opt-in experiments: on only with KYTY_X_<NAME>=1.
#define KYTY_XO(NAME)                                                                              \
	([]() noexcept {                                                                               \
		static const bool kyty_x_on = [] {                                                         \
			const char* value = std::getenv("KYTY_X_" #NAME);                                      \
			const bool  on    = value != nullptr && value[0] == '1';                               \
			std::printf("KYTY_X_%s=%d (opt-in)\n", #NAME, on ? 1 : 0);                             \
			return on;                                                                             \
		}();                                                                                       \
		return kyty_x_on;                                                                          \
	}())

// Debug (rain-perf): interleaved A/B within one run. KYTY_XT_<NAME>=<s> turns the experiment on
// and off every s seconds (on in odd periods since the process started); without it the
// experiment is on. Each change prints "XT <NAME> on/off at HH:MM:SS" (see _Build/xt.py).
namespace Common {
inline bool ExpToggle(const char* name, int64_t period_s) noexcept {
	if (period_s <= 0) {
		return true;
	}
	static const auto start = std::chrono::steady_clock::now();
	const auto        s     = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::steady_clock::now() - start)
	                   .count();
	(void)name;
	return ((s / period_s) & 1) != 0;
}
} // namespace Common

#define KYTY_XT(NAME)                                                                              \
	([]() noexcept {                                                                               \
		static const int64_t kyty_xt_period = [] {                                                 \
			const char* value = std::getenv("KYTY_XT_" #NAME);                                     \
			const int64_t p   = value != nullptr ? std::atoll(value) : 0;                          \
			std::printf("KYTY_XT_%s=%lld\n", #NAME, static_cast<long long>(p));                    \
			return p;                                                                              \
		}();                                                                                       \
		const bool kyty_xt_on   = ::Common::ExpToggle(#NAME, kyty_xt_period);                      \
		static int kyty_xt_last = -1;                                                              \
		if (kyty_xt_period > 0 && kyty_xt_last != static_cast<int>(kyty_xt_on)) {                  \
			kyty_xt_last   = static_cast<int>(kyty_xt_on);                                         \
			const auto now = std::time(nullptr);                                                   \
			std::tm    local {};                                                                   \
			localtime_s(&local, &now);                                                             \
			std::printf("XT %s %s at %02d:%02d:%02d\n", #NAME, kyty_xt_on ? "on" : "off",          \
			            local.tm_hour, local.tm_min, local.tm_sec);                                \
		}                                                                                          \
		return kyty_xt_on;                                                                         \
	}())

#endif // KYTY_COMMON_EXPSWITCH_H_
