#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute, Unknown };

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	[[nodiscard]] uint64_t GetPageSize() const;
	// Whether cache tracking currently restricts access to the page holding vaddr.
	[[nodiscard]] bool IsWatched(uint64_t vaddr) const;

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

	// While a sink is set on a thread, the write protections that thread's watcher updates add are not applied: their
	// ranges are appended to the sink (and the pages keep their access until ReapplyProtection()). Null: applied.
	static void SetDeferredProtectionSink(std::vector<std::pair<uint64_t, uint64_t>>* sink) noexcept;
	// Any thread: protects the pages of the range as their watchers say now.
	void ReapplyProtection(uint64_t vaddr, uint64_t size);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
