#ifndef KYTY_COMMON_ABLATION_H_
#define KYTY_COMMON_ABLATION_H_

// Local debugging (fidelity-diff agent): ablation switches that drop or neuter work to measure
// what it costs. They break the picture on purpose. Every switch only acts once
// KYTY_SKIP_AFTER_SECONDS have passed since the process started (the boot needs the work).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Common::Ablation {

inline const auto g_process_start = std::chrono::steady_clock::now();

inline bool Started() noexcept {
	static const int64_t after = [] {
		const char* value = std::getenv("KYTY_SKIP_AFTER_SECONDS");
		return value != nullptr ? std::atoll(value) : 0ll;
	}();
	static bool started = false;
	if (started) {
		return true;
	}
	if (std::chrono::steady_clock::now() - g_process_start >= std::chrono::seconds(after)) {
		started = true;
		std::printf("ABLATION: switches active (%lld s)\n", static_cast<long long>(after));
	}
	return started;
}

// A list of hex values ("*" = everything) from an environment variable.
struct HashList {
	explicit HashList(const char* env) {
		const char* value = std::getenv(env);
		if (value == nullptr || *value == 0) {
			return;
		}
		present = true;
		if (value[0] == '*') {
			all = true;
			return;
		}
		while (*value != 0) {
			char* end = nullptr;
			values.push_back(std::strtoull(value, &end, 16));
			if (end == value) break;
			value = *end == ',' ? end + 1 : end;
		}
		std::printf("ABLATION: %s = %zu values\n", env, values.size());
	}
	[[nodiscard]] bool Contains(uint64_t value) const noexcept {
		if (!present) return false;
		if (all) return true;
		for (auto v: values) {
			if (v == value) return true;
		}
		return false;
	}
	bool                  present = false;
	bool                  all     = false;
	std::vector<uint64_t> values;
};

inline bool Flag(const char* env) noexcept {
	const char* value = std::getenv(env);
	return value != nullptr && value[0] != '0' && value[0] != 0;
}

} // namespace Common::Ablation

// KYTY_ABL(NAME): flag KYTY_ABL_<NAME> set and the ablation started.
#define KYTY_ABL(NAME)                                                                             \
	([]() noexcept {                                                                               \
		static const bool kyty_abl_on = [] {                                                       \
			const bool on = ::Common::Ablation::Flag("KYTY_ABL_" #NAME);                           \
			if (on) std::printf("ABLATION: KYTY_ABL_%s=1\n", #NAME);                               \
			return on;                                                                             \
		}();                                                                                       \
		return kyty_abl_on && ::Common::Ablation::Started();                                       \
	}())

namespace Common::Ablation {
// Debug (rain-perf): KYTY_ABL_<NAME>=<s> with s >= 2 starts that switch s seconds after the process
// started (cumulative ablations in one run); "1" follows KYTY_SKIP_AFTER_SECONDS.
inline bool StartedAt(int64_t seconds) noexcept {
	if (seconds <= 1) {
		return Started();
	}
	return std::chrono::steady_clock::now() - g_process_start >= std::chrono::seconds(seconds);
}
} // namespace Common::Ablation

#define KYTY_ABL_AT(NAME)                                                                          \
	([]() noexcept {                                                                               \
		static const int64_t kyty_abl_at = [] {                                                    \
			const char* value = std::getenv("KYTY_ABL_" #NAME);                                    \
			const int64_t at  = value != nullptr ? std::atoll(value) : 0;                          \
			if (at > 0) std::printf("ABLATION: KYTY_ABL_%s at %lld s\n", #NAME,                    \
			                        static_cast<long long>(at));                                   \
			return at;                                                                             \
		}();                                                                                       \
		static bool kyty_abl_logged = false;                                                       \
		if (kyty_abl_at <= 0 || !::Common::Ablation::StartedAt(kyty_abl_at)) return false;         \
		if (!kyty_abl_logged) {                                                                    \
			kyty_abl_logged = true;                                                                \
			std::printf("ABLATION: KYTY_ABL_%s active\n", #NAME);                                  \
		}                                                                                          \
		return true;                                                                               \
	}())

#endif // KYTY_COMMON_ABLATION_H_
