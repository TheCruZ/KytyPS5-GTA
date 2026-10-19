#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_SPINYIELD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_SPINYIELD_H_

#include "common/common.h"

#include <cstdlib>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#undef MemoryBarrier
#else
#include <sched.h>
#endif

namespace Libs::Graphics {

// chenxiao07's tracking locks (KYTY_SPIN_YIELD, default on; 0: spin with test_and_set and _mm_pause, never yielding):
// a waiter reads the lock with plain loads and, past 512 spins, gives its CPU to the owner. In Demon's Souls the guest
// threads streaming textures fault into the texture cache and the page manager at once, and pure spinners starve an
// owner that lost its CPU (frames of 100+ ms with the owner parked inside its critical section).
[[nodiscard]] inline bool SpinYieldEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SPIN_YIELD");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

inline void YieldCpu() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SwitchToThread();
#else
	sched_yield();
#endif
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_SPINYIELD_H_
