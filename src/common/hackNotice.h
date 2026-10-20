#ifndef KYTY_COMMON_HACK_NOTICE_H_
#define KYTY_COMMON_HACK_NOTICE_H_

#include <cstdint>
#include <atomic>
#include <string>
#include <string_view>

// Visible notices for the workarounds ("hacks") that change what a game computes. They go to the
// normal log (stdout / emulator.log) with the fixed prefix "[HACK]", without any environment
// variable, so a session shows whether a hack was applied. Report() is safe for hot paths: it takes
// no locks, only atomics. The first report of each distinct (hack, site, shader, where) key prints a
// line; afterwards the keys that kept firing are summarised every 10 seconds with their totals.
namespace Common::HackNotice {

// `hack` and `site` must be string literals (they are compared by address). `shader_hash` 0 means
// "the shader of the enclosing ScopedShader, if any". `where` is part of the key (an address, an
// offset or a limit). `detail` is only used for the first line of a key.
void Report(const char* hack, const char* site, uint64_t shader_hash, uint64_t where,
            std::string_view detail = {});

// Defaults of this fork that change what a game computes are noted at startup, before the log exists
// (main.cpp), and reported once it does (ReportStartupDefaults).
void NoteStartupDefault(std::string text);
void ReportStartupDefaults();

inline uint64_t& CurrentShaderSlot() {
	static thread_local uint64_t shader_hash = 0;
	return shader_hash;
}

// Names the shader whose resources the calling thread is reading, for the notices of readers that
// do not know it (memory callbacks).
class ScopedShader {
public:
	explicit ScopedShader(uint64_t shader_hash): m_previous(CurrentShaderSlot()) {
		CurrentShaderSlot() = shader_hash;
	}
	~ScopedShader() { CurrentShaderSlot() = m_previous; }
	ScopedShader(const ScopedShader&)            = delete;
	ScopedShader& operator=(const ScopedShader&) = delete;

private:
	uint64_t m_previous;
};

// For sites on hot paths: true on the 1st, 2nd, 4th, 8th... application, so a site reports itself
// at powers of two with one relaxed atomic increment in between.
[[nodiscard]] inline bool PowerOfTwoTick(std::atomic<uint64_t>& counter) {
	const auto n = counter.fetch_add(1, std::memory_order_relaxed) + 1;
	return (n & (n - 1)) == 0;
}

// Workarounds applied while a guest shader is translated are reported when the translation runs, but
// a translation kept in the shader disk cache is not run again. ReportTranslation() reports the
// workaround and notes it on the calling thread; the translation takes the notes with
// TakeTranslationHacks() and stores them with its cached result, and ReplayTranslationHacks()
// reports them again when that result is loaded. `hack` and `site` must be string literals.
void ReportTranslation(const char* hack, const char* site, uint64_t shader_hash, uint64_t where,
                       std::string_view detail = {});
[[nodiscard]] std::string TakeTranslationHacks();
void                      ReplayTranslationHacks(std::string_view notes, uint64_t shader_hash);

} // namespace Common::HackNotice

#endif /* KYTY_COMMON_HACK_NOTICE_H_ */
