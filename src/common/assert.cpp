#include "common/assert.h"

#include "common/logging/log.h"
#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>

namespace Common {

static thread_local bool t_soft_exit = false;

SoftExitScope::SoftExitScope(): m_previous(t_soft_exit) {
	t_soft_exit = true;
}

SoftExitScope::~SoftExitScope() {
	t_soft_exit = m_previous;
}

// Throws instead of reporting while the thread has a SoftExitScope.
static void ThrowIfSoft([[maybe_unused]] const std::string& text) {
#if defined(__cpp_exceptions)
	if (t_soft_exit) {
		throw SoftExitError(text);
	}
#endif
}

static std::string BuildFatalReport(const char* title, std::string_view text, const char* file,
                                    int line) {
	return fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n", KYTY_BUILD_LABEL, title, text, file,
	                   line);
}

static int DbgReport(const char* title, std::string_view text, const char* file, int line) {
	Log::WriteFatal(BuildFatalReport(title, text, file, line));
	Subsystems::EmergencyShutdownActive();
	return 1;
}

int DbgExitIfHandler(const char* expr, const char* file, int line) {
	ThrowIfSoft(fmt::format("condition ({}) is true at {}:{}", expr, file, line));
	return DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr),
	                 file, line);
}

int DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	ThrowIfSoft(fmt::format("not implemented ({}) at {}:{}", expr, file, line));
	return DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

int DbgExitHandler(const char* file, int line, std::string_view text) {
	ThrowIfSoft(fmt::format("{} at {}:{}", text, file, line));
	Log::WriteFatal(BuildFatalReport("--- Error ---", text, file, line));
	return 1;
}

int DbgExitHandler(const char* file, int line, fmt::text_style style, std::string_view text) {
	ThrowIfSoft(fmt::format("{} at {}:{}", text, file, line));
	Log::WriteFatal(style, BuildFatalReport("--- Error ---", text, file, line));
	return 1;
}

void DbgExit(int status) {
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
	std::_Exit(status);
}

} // namespace Common
