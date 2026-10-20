#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// A/B switch for the clean-region skip (KYTY_TRACKER_SKIP_CLEAN=0 disables it).
inline const bool g_tracker_skip_clean =
    std::getenv("KYTY_TRACKER_SKIP_CLEAN") == nullptr || std::getenv("KYTY_TRACKER_SKIP_CLEAN")[0] == '1';

// A/B switch for the per-region bitmap fast path (KYTY_TRACKER_BITMAP=0 disables it).
inline const bool g_tracker_bitmap =
    std::getenv("KYTY_TRACKER_BITMAP") == nullptr || std::getenv("KYTY_TRACKER_BITMAP")[0] == '1';

// A/B switch for the dirty-region walk of read-only uploads (KYTY_TRACKER_DIRTY_WALK=0 disables
// it): a read-only sync whose regions all exist visits only the regions the bitmap marks
// CPU-dirty, instead of loading every region of the range. The game binds heap-wide V#s (up to
// 2.7 GB, ~700 regions) many times a draw; one dirty region anywhere sent each of those through
// every region (~0.3 us a sync, 27 syncs a draw at the Leap Attack prompt).
inline const bool g_tracker_dirty_walk = std::getenv("KYTY_TRACKER_DIRTY_WALK") == nullptr ||
                                         std::getenv("KYTY_TRACKER_DIRTY_WALK")[0] == '1';

class MemoryTracker final {
public:
	explicit MemoryTracker(PageManager& page_manager);
	~MemoryTracker();

	KYTY_CLASS_NO_COPY(MemoryTracker);

	// Lock-free and without touching the regions: every tracker region of the range exists and
	// none may hold CPU-dirty pages. The regions' own summaries, mirrored one bit per region, so a
	// buffer spanning hundreds of clean regions is answered from a few words. False is
	// conservative.
	[[nodiscard]] bool IsRangeCpuCleanHint(uint64_t vaddr, uint64_t size) const noexcept;
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// Whether a page of the tracking regions the range touches may be GPU dirty, without taking
	// their locks: false means the range can be read directly.
	[[nodiscard]] bool MayBeGpuModified(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || size > TRACKER_ADDRESS_SIZE || vaddr > TRACKER_ADDRESS_SIZE - size) {
			return true;
		}
		const auto last = (vaddr + size - 1) / TRACKER_REGION_SIZE;
		for (auto index = vaddr / TRACKER_REGION_SIZE; index <= last && index < REGION_COUNT;
		     index++) {
			const auto* manager = m_regions[index].load(std::memory_order_acquire);
			if (manager != nullptr && manager->HasGpuDirty()) {
				return true;
			}
		}
		return false;
	}
	// The CPU-dirty epoch of the tracking region `index` (address / TRACKER_REGION_SIZE), which
	// advances whenever its pages become CPU dirty; 0 while the region is untracked.
	[[nodiscard]] uint64_t RegionCpuDirtyEpoch(uint64_t index) const {
		if (index >= REGION_COUNT) {
			EXIT("invalid memory tracker region\n");
		}
		const auto* manager = m_regions[index].load(std::memory_order_acquire);
		return manager != nullptr ? manager->RegionCpuDirtyEpoch() : 0;
	}
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	// KYTY_READ_SNAPSHOT (RegionManager::SetReadLift).
	void               LiftReads(uint64_t vaddr, uint64_t size);
	void               RestoreReads(uint64_t vaddr, uint64_t size);
	// MarkRegionAsGpuModified, atomically skipping the tracker regions that hold CPU-dirty pages
	// there; false when one did. Pass a page for an all-or-nothing answer.
	[[nodiscard]] bool TryMarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UntrackMemory(uint64_t vaddr, uint64_t size);
	// Calls func(address, bytes) for each part of the range, one per tracker region, that may
	// hold CPU-dirty pages (lock-free summaries: false is exact, true is conservative).
	template <typename Func>
	void ForEachMaybeCpuDirtyRegion(uint64_t vaddr, uint64_t size, Func&& func) {
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			if (manager->MaybeModified<DirtySource::Cpu>()) {
				func(manager->GetCpuAddr() + offset, bytes);
			}
		});
	}
	// Queue after changing ownership; a concurrent drain must not consume an event
	// before the dirty pages it describes become visible.
	void                   QueueBdaSync(uint64_t vaddr, uint64_t size);
	[[nodiscard]] RangeSet TakeBdaSync();
	// As TakeBdaSync(), for the parts of the queued ranges inside [first, last) of each span;
	// the rest stays queued.
	[[nodiscard]] RangeSet TakeBdaSyncIn(std::span<const std::pair<uint64_t, uint64_t>> spans);
	// Removes protection from a range and flushes GPU-owned data when required.
	template <typename Flush>
	void InvalidateRegion(uint64_t vaddr, uint64_t size, Flush&& on_flush) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		CheckNotInUploadCallback();

		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				// Perform both the GPU modification check and CPU state change with the lock in
				// case the GPU thread is racing to mark the page modified. If a flush is needed,
				// on_flush performs the CPU state change.
				std::scoped_lock lock(manager->lock);
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				manager->NoteCpuWrite(manager->GetCpuAddr() + offset, bytes);
				manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
				return false;
			}();
			if (should_flush) {
				on_flush();
			}
		});
	}
#if KYTY_BUILD == KYTY_BUILD_DEBUG
	void ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                           const char* operation) const noexcept;
	void ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                               const char* operation);
#else
	void ValidateGpuDirtyPages(const RangeSet&, uint64_t, uint64_t, const char*) const noexcept {}
	void ValidateGpuDirtyOwnership(const RangeSet&, uint64_t, uint64_t, const char*) {}
#endif

	template <bool clear, typename Func>
	void ForEachDownloadRange(uint64_t vaddr, uint64_t size, Func&& func) {
		static_assert(std::is_nothrow_invocable_v<Func&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			const auto       address = manager->GetCpuAddr() + offset;
			manager->template ForEachModifiedRange<DirtySource::Gpu, false>(address, bytes, func);
			if constexpr (clear) {
				manager->template ChangeState<DirtySource::Gpu, false>(address, bytes);
			}
		});
	}

	// Calls func(address, size) for the runs of pages of [vaddr, vaddr + size) that the CPU may
	// have written since the previous call over them: pages that became CPU dirty since then, or
	// are still CPU dirty. Untracked pages start tracking as written. A single consumer owns the
	// written state.
	template <typename Func>
	void ConsumeCpuWrites(uint64_t vaddr, uint64_t size, Func&& func) {
		CheckNotInUploadCallback();
		std::vector<std::pair<uint64_t, uint64_t>> runs;
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			manager->ConsumeCpuWritten(manager->GetCpuAddr() + offset, bytes,
			                           [&](uint64_t address, uint64_t length) {
				                           if (!runs.empty() &&
				                               runs.back().first + runs.back().second == address) {
					                           runs.back().second += length;
				                           } else {
					                           runs.emplace_back(address, length);
				                           }
			                           });
		});
		const auto end = vaddr + size;
		for (const auto& [address, length]: runs) {
			const auto begin  = std::max(address, vaddr);
			const auto finish = std::min(address + length, end);
			if (begin < finish) {
				func(begin, finish - begin);
			}
		}
	}

	// Marks the pages of [vaddr, vaddr + size) GPU modified, except those that are CPU dirty, and
	// calls func(address, size) for the page runs it marked.
	template <typename Func>
	void MarkRegionAsGpuModifiedUnlessCpuDirty(uint64_t vaddr, uint64_t size, Func&& func) {
		CheckNotInUploadCallback();
		std::vector<std::pair<uint64_t, uint64_t>> runs;
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			manager->MarkGpuDirtyUnlessCpuDirty(
			    manager->GetCpuAddr() + offset, bytes,
			    [&](uint64_t address, uint64_t length) { runs.emplace_back(address, length); });
		});
		for (const auto& [address, length]: runs) {
			func(address, length);
		}
	}

	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRange(uint64_t vaddr, uint64_t size, bool is_written, RangeFunc&& range_func,
	                        UploadFunc&& upload_func) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, uint64_t, uint64_t>);
		static_assert(std::is_nothrow_invocable_v<UploadFunc&>);
		CheckNotInUploadCallback();
		if (!is_written && g_tracker_skip_clean && g_tracker_bitmap &&
		    IsRangeCpuCleanHint(vaddr, size)) {
			// What the walk below finds for a read-only range whose regions all exist and are
			// clean: nothing to copy.
			const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
			upload_func();
			s_upload_owner = previous_upload_owner;
			return;
		}
		if (!is_written && g_tracker_skip_clean && g_tracker_bitmap && g_tracker_dirty_walk &&
		    AllRegionsPresent(vaddr, size)) {
			// The loop below for a read-only range, with the clean regions skipped by their bitmap
			// bits rather than by loading each region (the bits mirror the regions' summaries).
			const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
			ForEachCpuDirtyRegion(vaddr, size,
			                      [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				                      manager->lock.lock();
				                      manager->ForEachModifiedRange<DirtySource::Cpu, true>(
				                          manager->GetCpuAddr() + offset, bytes, range_func);
				                      manager->lock.unlock();
			                      });
			upload_func();
			s_upload_owner = previous_upload_owner;
			return;
		}
		const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool summary_clean = !manager->MaybeModified<DirtySource::Cpu>();
			if (!is_written && g_tracker_skip_clean && summary_clean) {
				return;
			}
			manager->lock.lock();
			if (summary_clean && manager->IsModified<DirtySource::Cpu>(offset, bytes)) {
				static std::atomic<uint32_t> misses = 0;
				if (misses.fetch_add(1) < 32) {
					LOGF("Tracker: CPU summary miss region=0x%016" PRIx64 " offset=0x%" PRIx64
					     " bytes=0x%" PRIx64 " written=%d\n",
					     manager->GetCpuAddr(), offset, bytes, static_cast<int>(is_written));
				}
			}
			manager->ForEachModifiedRange<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset,
			                                                      bytes, range_func);
			if (!is_written) {
				manager->lock.unlock();
			}
		});
		upload_func();
		if (is_written) {
			Iterate<false>(vaddr, size,
			               [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				               manager->template ChangeState<DirtySource::Gpu, true>(
				                   manager->GetCpuAddr() + offset, bytes);
				               manager->lock.unlock();
			               });
		}
		s_upload_owner = previous_upload_owner;
	}

private:
	static constexpr size_t REGION_COUNT = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	inline static thread_local const MemoryTracker* s_upload_owner = nullptr;

	void CheckNotInUploadCallback() const noexcept {
		if (s_upload_owner == this) {
			EXIT("memory tracker re-entered from upload callback\n");
		}
	}

	template <bool create, typename Func>
	bool Iterate(uint64_t vaddr, uint64_t size, Func&& func) {
		ValidateRange(vaddr, size);
		using Result = std::invoke_result_t<Func, RegionManager*, uint64_t, uint64_t>;
		constexpr bool returns_bool = std::is_same_v<Result, bool>;
		uint64_t       remaining    = size;
		uint64_t       index        = vaddr / TRACKER_REGION_SIZE;
		uint64_t       offset       = vaddr % TRACKER_REGION_SIZE;
		while (remaining != 0) {
			const auto bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr && create) {
				manager = GetOrCreateRegion(index);
			}
			if (manager != nullptr) {
				if constexpr (returns_bool) {
					if (func(manager, offset, bytes)) {
						return true;
					}
				} else {
					func(manager, offset, bytes);
				}
			}
			remaining -= bytes;
			offset = 0;
			index++;
		}
		return false;
	}

	// Bits [low, high] of a bitmap word.
	static uint64_t WordMask(uint64_t low, uint64_t high) noexcept {
		return high - low == 63 ? ~uint64_t {0} : ((uint64_t {1} << (high - low + 1)) - 1) << low;
	}

	// Every tracker region of the range exists (regions are never removed once created).
	[[nodiscard]] bool AllRegionsPresent(uint64_t vaddr, uint64_t size) const noexcept {
		if (size == 0 || vaddr >= TRACKER_ADDRESS_SIZE || size > TRACKER_ADDRESS_SIZE - vaddr) {
			return false;
		}
		const uint64_t first = vaddr / TRACKER_REGION_SIZE;
		const uint64_t last  = (vaddr + size - 1) / TRACKER_REGION_SIZE;
		for (uint64_t word = first / 64; word <= last / 64; word++) {
			const auto mask = WordMask(word == first / 64 ? first % 64 : 0,
			                           word == last / 64 ? last % 64 : 63);
			if ((m_present_regions[word].load(std::memory_order_acquire) & mask) != mask) {
				return false;
			}
		}
		return true;
	}

	// func(manager, offset, bytes) for the part of the range in each region whose CPU summary
	// bit is set; the range's regions must all exist (AllRegionsPresent).
	template <typename Func>
	void ForEachCpuDirtyRegion(uint64_t vaddr, uint64_t size, Func&& func) {
		const uint64_t first = vaddr / TRACKER_REGION_SIZE;
		const uint64_t last  = (vaddr + size - 1) / TRACKER_REGION_SIZE;
		const uint64_t end   = vaddr + size;
		for (uint64_t word = first / 64; word <= last / 64; word++) {
			auto dirty = m_cpu_dirty_regions[word].load(std::memory_order_acquire) &
			             WordMask(word == first / 64 ? first % 64 : 0,
			                      word == last / 64 ? last % 64 : 63);
			while (dirty != 0) {
				const uint64_t index = word * 64 + static_cast<uint64_t>(std::countr_zero(dirty));
				dirty &= dirty - 1;
				const uint64_t region_begin = index * TRACKER_REGION_SIZE;
				const uint64_t begin        = std::max(vaddr, region_begin);
				const uint64_t finish       = std::min(end, region_begin + TRACKER_REGION_SIZE);
				func(m_regions[index].load(std::memory_order_acquire), begin - region_begin,
				     finish - begin);
			}
		}
	}

	static void    ValidateRange(uint64_t vaddr, uint64_t size);
	RegionManager* GetOrCreateRegion(uint64_t index);

	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	// One bit per region: the region exists / its CPU summary is set (RegionManager::StoreSummary).
	std::unique_ptr<std::atomic<uint64_t>[]> m_present_regions;
	std::unique_ptr<std::atomic<uint64_t>[]> m_cpu_dirty_regions;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	std::mutex                                     m_region_mutex;
	PageManager&                                   m_page_manager;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
