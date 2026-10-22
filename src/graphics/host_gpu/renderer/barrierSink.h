#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERSINK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERSINK_H_

namespace Libs::Graphics {

// Guest barrier sinking (KYTY_BARRIER_SINK=1). The game's partial flushes (EVENT_WRITE
// CS/PS_PARTIAL_FLUSH, ACQUIRE/RELEASE_MEM cache actions) become full ALL_COMMANDS barriers, and a
// barrier cannot be recorded inside a rendering instance, so each one ended rendering: in the
// jungle a PS_PARTIAL_FLUSH follows every GPU-driven mesh draw, ~350 render pass restarts and
// drains a frame. With sinking, a guest barrier requested inside a rendering instance stays
// pending across the draws that follow it in that instance while the draws are provably
// independent of it, and is recorded before the first command that is not.
//
// Implemented at the Vulkan dispatch level (VULKAN_HPP_DEFAULT_DISPATCHER), which sees every
// command of a command buffer in recording order, whichever thread or path records it.
//
// A pending barrier B (requested after commands P, before a draw D2 of the same instance R) may
// stay pending past D2 only when:
//   (a) a full barrier F (ALL_COMMANDS/MEMORY_WRITE -> ALL_COMMANDS/MEMORY_READ|WRITE) was
//       recorded in this command buffer, and since F only state commands, the begin of R (the
//       first instance after F) and safe draws in R were recorded (the "epoch");
//   (b) D2 is safe: it writes only its own color/depth attachments (no storage buffer or image
//       writes, no address writes) and samples none of them.
// Then every write since F is an attachment write of R, which D2 reads only as an attachment,
// ordered by rasterization order; everything before F is ordered against D2 by F; and D2 writes
// nothing but attachments, so no later reader misses B (B is recorded before any command that is
// not a safe draw in R: at the end of R, before any other command, at the latest before the
// command buffer ends). An unsafe draw (or a clear) in R with B pending ends R, records B and
// begins R again with its attachments loaded.
[[nodiscard]] bool BarrierSinkEnabled();
void               InstallBarrierSink();

// Recording side: the renderer marks the draw it records next as safe (see (b)) on its own thread;
// CommandRecorder carries the mark to the thread that records the Vulkan command.
void SetNextDrawSafe(bool safe) noexcept;
[[nodiscard]] bool NextDrawSafe() noexcept;
// On the thread recording the Vulkan command, right before it.
void SetCommandDrawSafe(bool safe) noexcept;
// Around the guest's barrier (CommandProcessor::EmitGlobalBarrier) on the recording thread.
void SetCommandGuestBarrier(bool guest) noexcept;

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERSINK_H_ */
