#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_GPUDIRTYSEQ_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_GPUDIRTYSEQ_H_

#include <atomic>
#include <cstdint>

namespace Libs::Graphics {

// Advances whenever guest bytes become GPU-written or start to download (the BufferCache adds a
// range to its GPU-modified or downloading set). Bytes read as clean stay clean while it does not
// advance, which lets a check of "these recorded reads are still clean" run ahead of the thread
// that needs the verdict (GuestReadLog::PreValidate).
inline std::atomic<uint64_t> g_gpu_dirty_seq {1};

inline void NoteGpuDirtyAdded() noexcept {
	g_gpu_dirty_seq.fetch_add(1, std::memory_order_release);
}

[[nodiscard]] inline uint64_t GpuDirtySeq() noexcept {
	return g_gpu_dirty_seq.load(std::memory_order_acquire);
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_GPUDIRTYSEQ_H_
