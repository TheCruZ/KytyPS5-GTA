#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMESTEPHACK_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMESTEPHACK_H_

#include <cstdint>

namespace Libs::VideoOut::FrameStepHack {

// [HACK] ASTRO BOT (PPSA21564, 01.007.000) advances its simulation by one frame of its frame rate
// for each frame, whatever the time between frames. Below that rate the game's frame time follows
// the presented frames instead. No effect on other titles or versions.

// Called once with the main executable, after relocation and before the game runs.
void PatchMainProgram(uint64_t base_vaddr, uint64_t size) noexcept;

// Called on the presentation thread for each presented main-bus guest flip.
void OnPresentedFrame() noexcept;

} // namespace Libs::VideoOut::FrameStepHack

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMESTEPHACK_H_
