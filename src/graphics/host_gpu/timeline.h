#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_TIMELINE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_TIMELINE_H_

#include <atomic>
#include <cstdint>

// Diagnostics: KYTY_TIMELINE=<first frame>:<frames> logs "TL <microseconds> <thread> <event>"
// lines for guest submissions, GPU-thread processing, host GPU submissions and completions,
// drains and presents, for that window of presented frames. Off by default.
namespace Libs::Graphics::Timeline {

namespace Detail {
inline std::atomic<bool> g_active {false};
void MarkSlow(const char* event, uint64_t a, uint64_t b);
} // namespace Detail

[[nodiscard]] inline bool Active() {
	return Detail::g_active.load(std::memory_order_relaxed);
}

inline void Mark(const char* event, uint64_t a = 0, uint64_t b = 0) {
	if (Active()) {
		Detail::MarkSlow(event, a, b);
	}
}

// Marks `begin` now and `end` when the scope ends (both with `a`).
class Scope {
public:
	Scope(const char* begin, const char* end, uint64_t a): m_end(end), m_a(a), m_active(Active()) {
		if (m_active) {
			Detail::MarkSlow(begin, a, 0);
		}
	}
	~Scope() {
		if (m_active) {
			Detail::MarkSlow(m_end, m_a, 0);
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	const char* m_end;
	uint64_t    m_a;
	bool        m_active;
};

// Called once per presented frame.
void Present(uint64_t frame);

} // namespace Libs::Graphics::Timeline

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_TIMELINE_H_
