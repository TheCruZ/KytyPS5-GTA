#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_CACHESTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_CACHESTATS_H_

#include "graphics/host_gpu/rangeSet.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>

// Diagnostic counters for cache synchronization costs, printed periodically when
// KYTY_CACHE_STATS is set.
namespace Libs::Graphics::CacheStats {

enum Counter : uint32_t {
	WriteFaults,
	ReadFaults,
	ReadbackDownloads,
	ReadbackWaitUs,
	BdaSyncs,
	BdaSyncUs,
	BdaSyncBuffers,
	TextureUploads,
	TextureUploadBytes,
	TextureGcFrees,
	TextureGcGpuDataDropped,
	BufferGcFrees,
	IndirectTargets,
	IndirectPrepareUs,
	IndirectDispatches,
	ReadbackIndirect,
	TextureUploadsFromBuffer,
	TextureUploadsInIndirect,
	WriteFaultsInIndirect,
	ImagesMarkedFill,
	ImagesMarkedCopy,
	ImagesMarkedStorage,
	ImagesMarkedStorageDynamic,
	ImagesMarkedIndirect,
	ReadbackStorage,
	ReadbackDynamic,
	ReadbackDma,
	StorageWriteOverGpuImage,
	StorageRmwOverGpuImage,
	Submissions,
	Count
};

inline std::atomic<uint64_t> g_counters[Count] {};
// Range attribution below is only kept when KYTY_CACHE_STATS is set.
inline const bool g_enabled = std::getenv("KYTY_CACHE_STATS") != nullptr;
// GPU thread only: guest ranges prepared as V#-table store targets.
inline RangeSet g_indirect_targets;
inline RangeSet g_storage_written;
inline RangeSet g_dynamic_written;
inline RangeSet g_dma_written;
// Which InvalidateMemoryFromGPU caller is running (a Counter from ImagesMarkedFill on).
inline thread_local Counter t_invalidate_source = ImagesMarkedFill;

inline void Add(Counter counter, uint64_t value = 1) {
	g_counters[counter].fetch_add(value, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::CacheStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_CACHESTATS_H_
