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
	// While set on this thread, watchers released inside [begin, end) update the page state without making the pages
	// writable on the host: the range is about to be unmapped, which drops its protection anyway (a texture pool made
	// of 64 KiB mappings cost one VirtualProtect per mapping for each image the unmap deleted). Protections are still
	// applied: a skipped one would leave a watched page writable and writes to it unseen. If the unmap fails, the
	// caller reapplies the range (ReapplyProtection). end == 0 clears it.
	static void SetUnmappingRange(uint64_t begin, uint64_t end) noexcept;
	// A write fault on a page nothing watches: the protection outlived its watchers and would fault forever. The page
	// gets the guest's own protection back; false when it is not such a page.
	[[nodiscard]] bool RestoreIfUnwatched(uint64_t vaddr) noexcept;

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
