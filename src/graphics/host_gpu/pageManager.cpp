#include "graphics/host_gpu/pageManager.h"

#include "common/alignment.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <unordered_map>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <unistd.h>
#endif

namespace Libs::Graphics {
namespace {

constexpr uint64_t PAGE_SIZE    = TRACKER_PAGE_SIZE;
constexpr uint64_t REGION_SIZE  = TRACKER_REGION_SIZE;
constexpr uint64_t ADDRESS_SIZE = TRACKER_ADDRESS_SIZE;
constexpr uint64_t REGION_COUNT = ADDRESS_SIZE / REGION_SIZE;

constexpr uint64_t REGION_PAGES = REGION_SIZE / PAGE_SIZE;

[[noreturn]] void FailFast(const char* reason = nullptr) noexcept {
	std::fputs("PageManager fail-fast: ", stderr);
	std::fputs(reason != nullptr ? reason : "invalid page state", stderr);
	std::fputc('\n', stderr);
	std::fflush(stderr);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(EXCEPTION_NONCONTINUABLE_EXCEPTION));
#endif
	std::_Exit(322);
}

[[noreturn]] void Fatal(const char* format, ...) {
	std::fputs("PageManager fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	std::_Exit(322);
}

class SpinGuard final {
public:
	explicit SpinGuard(std::atomic_flag& lock): m_lock(lock) {
		while (m_lock.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
			_mm_pause();
#elif defined(_M_ARM64)
			__yield();
#elif defined(__aarch64__)
			asm volatile("yield");
#endif
		}
	}
	~SpinGuard() { m_lock.clear(std::memory_order_release); }
	KYTY_CLASS_NO_COPY(SpinGuard);

private:
	std::atomic_flag& m_lock;
};

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

} // namespace

struct PageManager::Impl {
	struct PageState {
		uint8_t write_watchers  : 7 = 0;
		uint8_t access_watchers : 1 = 0;

		[[nodiscard]] Common::VirtualMemory::Mode Perms() const noexcept {
			if (access_watchers != 0) {
				return Common::VirtualMemory::Mode::NoAccess;
			}
			if (write_watchers != 0) {
				return Common::VirtualMemory::Mode::Read;
			}
			return Common::VirtualMemory::Mode::ReadWrite;
		}

		template <int delta, bool is_read>
		uint32_t AddDelta(uint64_t address) {
			static_assert(delta >= -1 && delta <= 1);
			if constexpr (is_read) {
				if constexpr (delta == 1) {
					if (access_watchers != 0) {
						Fatal("read-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++access_watchers;
				} else if constexpr (delta == -1) {
					if (access_watchers == 0) {
						Fatal("read-watcher underflow at 0x%016" PRIx64, address);
					}
					return --access_watchers;
				} else {
					return access_watchers;
				}
			} else {
				if constexpr (delta == 1) {
					if (write_watchers == 0x7f) {
						Fatal("write-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++write_watchers;
				} else if constexpr (delta == -1) {
					if (write_watchers == 0) {
						Fatal("write-watcher underflow at 0x%016" PRIx64, address);
					}
					return --write_watchers;
				} else {
					return write_watchers;
				}
			}
		}
	};
	static_assert(sizeof(PageState) == 1);

	struct Region {
		std::atomic_flag                    lock = ATOMIC_FLAG_INIT;
		std::array<PageState, REGION_PAGES> pages;
	};

	Impl() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		if (info.dwPageSize != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32,
			      static_cast<uint32_t>(info.dwPageSize));
		}
#elif defined(__APPLE__)
		// Under Rosetta the host page size is 4 KB, matching TRACKER_PAGE_SIZE.
		if (static_cast<uint64_t>(getpagesize()) != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32, static_cast<uint32_t>(getpagesize()));
		}
#else
		const auto host_page_size = ::sysconf(_SC_PAGESIZE);
		if (host_page_size < 0 || static_cast<uint64_t>(host_page_size) != PAGE_SIZE) {
			Fatal("unsupported host page size %ld", static_cast<long>(host_page_size));
		}
#endif
		regions = std::make_unique<std::atomic<Region*>[]>(REGION_COUNT);
	}

	~Impl() {
		for (const auto& region: region_storage) {
			SpinGuard lock(region->lock);
			for (auto& page: region->pages) {
				if (page.write_watchers != 0 || page.access_watchers != 0) {
					FailFast("PageManager destroyed with live page state");
				}
			}
		}
	}

	Region* FindRegion(uint64_t vaddr) const noexcept {
		return vaddr < ADDRESS_SIZE ? regions[vaddr / REGION_SIZE].load(std::memory_order_acquire)
		                            : nullptr;
	}

	Region* GetOrCreateRegion(uint64_t vaddr) {
		const auto index = vaddr / REGION_SIZE;
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		std::lock_guard lock(region_mutex);
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		auto  region = std::make_unique<Region>();
		auto* ptr    = region.get();
		region_storage.push_back(std::move(region));
		regions[index].store(ptr, std::memory_order_release);
		return ptr;
	}

	// KYTY_FPS_LOG: host protection changes per second, arming (to read-only or no access) and
	// disarming (back to read-write), and how many pages are armed again and again ([page-protect]).
	struct ProtectStats {
		std::mutex                             mutex;
		std::chrono::steady_clock::time_point  start = std::chrono::steady_clock::now();
		uint64_t                               arm_calls = 0, arm_pages = 0, disarm_calls = 0,
		                                       disarm_pages = 0, noaccess_pages = 0;
		std::unordered_map<uint64_t, std::array<uint64_t, 3>> regions; // 2 MB -> read, none, rw
		double                                 arm_us = 0, disarm_us = 0;
		std::unordered_map<uint64_t, uint32_t> armed; // page -> arm count in the window
	};
	static ProtectStats* Stats() {
		static ProtectStats* stats =
		    std::getenv("KYTY_FPS_LOG") != nullptr ? new ProtectStats : nullptr;
		return stats;
	}
	static void Note(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode, double us) {
		auto* stats = Stats();
		if (stats == nullptr) {
			return;
		}
		const bool       arm   = mode != Common::VirtualMemory::Mode::ReadWrite;
		const uint64_t   pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
		std::scoped_lock lock(stats->mutex);
		auto& region = stats->regions[vaddr >> 21u];
		region[mode == Common::VirtualMemory::Mode::Read ? 0 : arm ? 1 : 2] += pages;
		if (mode == Common::VirtualMemory::Mode::NoAccess) {
			stats->noaccess_pages += pages;
		}
		if (arm) {
			stats->arm_calls++;
			stats->arm_pages += pages;
			stats->arm_us += us;
			for (uint64_t p = 0; p < pages && p < 4096; p++) {
				stats->armed[vaddr / PAGE_SIZE + p]++;
			}
		} else {
			stats->disarm_calls++;
			stats->disarm_pages += pages;
			stats->disarm_us += us;
		}
		const auto now = std::chrono::steady_clock::now();
		const auto s   = std::chrono::duration<double>(now - stats->start).count();
		if (s < 5.0) {
			return;
		}
		uint64_t hot = 0, hot_arms = 0, total_arms = 0;
		for (const auto& [page, count]: stats->armed) {
			total_arms += count;
			if (count >= 5) {
				hot++;
				hot_arms += count;
			}
		}
		std::fprintf(stderr,
		             "[page-protect] arm %.0f calls/s %.0f pages/s %.1f ms/s | disarm %.0f calls/s "
		             "%.0f pages/s %.1f ms/s | pages armed %zu, armed >=5x: %llu pages = %.0f%% of "
		             "arms\n",
		             stats->arm_calls / s, stats->arm_pages / s, stats->arm_us / 1000.0 / s,
		             stats->disarm_calls / s, stats->disarm_pages / s,
		             stats->disarm_us / 1000.0 / s, stats->armed.size(),
		             static_cast<unsigned long long>(hot),
		             total_arms != 0 ? 100.0 * static_cast<double>(hot_arms) / total_arms : 0.0);
		std::vector<std::pair<uint64_t, std::array<uint64_t, 3>>> top(stats->regions.begin(),
		                                                               stats->regions.end());
		std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) {
			return a.second[0] + a.second[1] + a.second[2] > b.second[0] + b.second[1] + b.second[2];
		});
		std::string text;
		for (size_t i = 0; i < top.size() && i < 10; i++) {
			char line[160];
			std::snprintf(line, sizeof(line), " 0x%llx:r%.0f/n%.0f/w%.0f",
			              static_cast<unsigned long long>(top[i].first << 21u),
			              top[i].second[0] / s, top[i].second[1] / s, top[i].second[2] / s);
			text += line;
		}
		std::fprintf(stderr, "[page-protect] no-access %.0f pages/s; 2MB regions (pages/s read-only/no-access/rw):%s\n",
		             stats->noaccess_pages / s, text.c_str());
		stats->regions.clear();
		stats->noaccess_pages = 0;
		stats->arm_calls = stats->arm_pages = stats->disarm_calls = stats->disarm_pages = 0;
		stats->arm_us = stats->disarm_us = 0;
		stats->armed.clear();
		stats->start = now;
	}

	void Protect(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) noexcept {
		const auto t0 = Stats() != nullptr ? std::chrono::steady_clock::now()
		                                   : std::chrono::steady_clock::time_point {};
		const bool ok = Libs::LibKernel::Memory::ProtectGuestHostMemory(vaddr, size, mode);
		if (Stats() != nullptr) {
			Note(vaddr, size, mode,
			     std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
			         .count());
		}
		if (!ok) {
			Fatal("address-space protection failed at 0x%016" PRIx64 ", mode=0x%08" PRIx32, vaddr,
			      static_cast<uint32_t>(mode));
		}
	}

	template <bool track, bool is_read, bool masked>
	void UpdateRegionWatchers(Region& region, uint64_t base_addr, size_t first, size_t last,
	                          const RegionBits* mask = nullptr) {
		SpinGuard lock(region.lock);
		auto      perms                 = region.pages[first].Perms();
		uint64_t  range_begin           = 0;
		uint64_t  range_bytes           = 0;
		uint64_t  potential_range_bytes = 0;

		const auto release_pending = [&] {
			if (range_bytes != 0) {
				Protect(base_addr + range_begin * PAGE_SIZE, range_bytes, perms);
				range_bytes           = 0;
				potential_range_bytes = 0;
			}
		};

		for (size_t page_index = first; page_index < last; page_index++) {
			auto&      page    = region.pages[page_index];
			const auto address = base_addr + page_index * PAGE_SIZE;
			const bool update  = !masked || mask->Get(page_index);

			const auto old_perms = page.Perms();
			const auto new_count = update ? page.AddDelta<track ? 1 : -1, is_read>(address)
			                              : page.AddDelta<0, is_read>(address);
			const auto new_perms = page.Perms();

			if (new_perms != perms) [[unlikely]] {
				release_pending();
				perms = new_perms;
			} else if (range_bytes != 0) {
				potential_range_bytes += PAGE_SIZE;
			}

			if (!update) {
				continue;
			}

			const bool watcher_edge = (track && new_count == 1) || (!track && new_count == 0);
			if (watcher_edge && old_perms != new_perms) {
				if (range_bytes == 0) {
					range_begin           = page_index;
					potential_range_bytes = PAGE_SIZE;
				}
				range_bytes = potential_range_bytes;
			}
		}

		release_pending();
	}

	template <bool track, bool is_read>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
		ValidateRange(vaddr, size);
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			auto*      region = track ? GetOrCreateRegion(chunk_begin) : FindRegion(chunk_begin);
			if (region == nullptr) {
				Fatal("untracking unknown page 0x%016" PRIx64, chunk_begin);
			}
			const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			UpdateRegionWatchers<track, is_read, false>(*region, region_base, first, last);
			chunk_begin = chunk_end;
		}
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	std::mutex                              region_mutex;
};

static_assert(std::atomic<void*>::is_always_lock_free);

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {}

PageManager::~PageManager() = default;

uint64_t PageManager::GetPageSize() const {
	return PAGE_SIZE;
}

template <bool track>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	m_impl->UpdatePageWatchers<track, false>(vaddr, size);
}

template void PageManager::UpdatePageWatchers<true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false>(uint64_t, uint64_t);

bool PageManager::IsReadWatched(uint64_t vaddr) const noexcept {
	const auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	const auto index = static_cast<size_t>((vaddr % REGION_SIZE) / PAGE_SIZE);
	return region->pages[index].access_watchers != 0;
}

void PageManager::ResyncHostProtection(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid() || vaddr >= ADDRESS_SIZE) {
		return;
	}
	const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
	const auto end   = std::min(Common::AlignUp(vaddr + size, PAGE_SIZE), ADDRESS_SIZE);
	for (auto chunk_begin = begin; chunk_begin < end;) {
		const auto chunk_end   = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
		const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
		const auto first       = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
		const auto last        = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
		auto*      region      = m_impl->FindRegion(chunk_begin);
		chunk_begin            = chunk_end;
		if (region == nullptr) {
			continue; // never watched: the host protection is not the tracker's
		}
		SpinGuard lock(region->lock);
		// Runs of watched pages with the same protection, one host call each.
		for (size_t page = first; page < last;) {
			const auto perms = region->pages[page].Perms();
			size_t     run   = page + 1;
			while (run < last && region->pages[run].Perms() == perms) {
				run++;
			}
			if (perms != Common::VirtualMemory::Mode::ReadWrite) {
				m_impl->Protect(region_base + page * PAGE_SIZE, (run - page) * PAGE_SIZE, perms);
			}
			page = run;
		}
	}
}

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	if (base_addr % REGION_SIZE != 0 || base_addr >= ADDRESS_SIZE ||
	    REGION_SIZE > ADDRESS_SIZE - base_addr) {
		Fatal("invalid tracking region base 0x%016" PRIx64, base_addr);
	}

	const auto start_range = mask.FirstRange();
	const auto end_range   = mask.LastRange();
	if (start_range.first == REGION_PAGES) {
		FailFast("empty region watcher mask");
	}
	const auto first = start_range.first;
	const auto last  = end_range.second;
	if (start_range.second == end_range.second) {
		m_impl->UpdatePageWatchers<track, is_read>(base_addr + first * PAGE_SIZE,
		                                           (last - first) * PAGE_SIZE);
		return;
	}

	auto* region = track ? m_impl->GetOrCreateRegion(base_addr) : m_impl->FindRegion(base_addr);
	if (region == nullptr) {
		Fatal("untracking unknown region 0x%016" PRIx64, base_addr);
	}
	m_impl->UpdateRegionWatchers<track, is_read, true>(*region, base_addr, first, last, &mask);
}

template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);

} // namespace Libs::Graphics
