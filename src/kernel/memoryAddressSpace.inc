// Internal memory address-space helpers included by memory.cpp.
// Kept in the same translation unit because they depend on memory.cpp-local helpers.

static constexpr uint64_t HOST_SYSTEM_MANAGED_MIN  = 0x0000040000ull;
static constexpr uint64_t HOST_SYSTEM_MANAGED_MAX  = 0x07ffffbfffull;
#if defined(__APPLE__) && (defined(__x86_64__) || defined(_M_X64))
static constexpr uint64_t HOST_SYSTEM_RESERVED_MAX = 0x0fbfffffffull;
static constexpr uint64_t HOST_USER_MIN            = 0x7000000000ull;
#else
static constexpr uint64_t HOST_SYSTEM_RESERVED_MAX = 0x0fffffffffull;
static constexpr uint64_t HOST_USER_MIN            = 0x1000000000ull;
#endif
static constexpr uint64_t HOST_USER_MAX = 0xfbffffffffull;

static bool GuestFreeRangeContains(uint64_t base, uint64_t range_size, uint64_t candidate,
                                   uint64_t size) {
	if (UINT64_MAX - base < range_size || candidate < base) {
		return false;
	}
	const auto range_end = base + range_size;
	return candidate <= range_end && size <= range_end - candidate;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
static bool CoalescePlaceholders(uint64_t vaddr, uint64_t size) {
	return size != 0 && VirtualFree(reinterpret_cast<void*>(vaddr), size,
	                                MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) != 0;
}

// chenxiao07's parked partial unmaps (KYTY_PARK_UNMAPS, on unless it is 0). A Windows file view
// cannot be split: unmapping part of one used to unmap the whole view, cut its placeholder and map
// the other parts again, all on the guest's thread (dozens of system calls per unmap, and the
// texture streaming of Demon's Souls unmaps parts of views every frame). With the switch on the
// unmapped range stays inside its view, "parked": the guest has no mapping there and a map over it
// splits the view first (GuestBackingStore::Materialize). Off: the view is split on every unmap.
static bool ParkUnmapsEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PARK_UNMAPS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// chenxiao07's maps and unmaps outside the address-space lock (KYTY_MAP_OUTSIDE_LOCK, on unless it is 0).
// GuestAddressSpace::MapBacking and UnmapBacking held the lock exclusively during their system calls (cutting
// placeholders, mapping and unmapping views, unmapping a whole view to split it): the page protections of the GPU
// trackers (ProtectTransient, a shared lock; every write fault of every guest thread goes through one) all waited
// behind it, up to milliseconds for a big view. With the switch on the range is marked busy instead, the system calls
// run without the lock, and only what overlaps the busy range waits. Off: the lock is held throughout.
static bool MapOutsideLockEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_MAP_OUTSIDE_LOCK");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// Ranges parked in their views, taken out of the working set by a thread of their own: mapped,
// their pages stayed counted in use, and dropping them (VirtualUnlock on pages that are not locked)
// costs ~0.5 us per 4 KiB page, which the guest's threads no longer pay. Harmless when late: on a
// range unmapped meanwhile it fails, on one mapped again its pages are faulted back in.
class ParkedRangeTrimmer {
public:
	static void Trim(uint64_t vaddr, uint64_t size) {
		// (Never destroyed: its thread may outlive the static destructors.)
		static auto* trimmer = new ParkedRangeTrimmer;
		{
			std::lock_guard lock(trimmer->m_mutex);
			trimmer->m_ranges.emplace_back(vaddr, size);
		}
		trimmer->m_ready.notify_one();
	}

private:
	ParkedRangeTrimmer() {
		std::thread([this] {
			SetThreadDescription(GetCurrentThread(), L"Kyty.Trim");
			SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
			std::vector<std::pair<uint64_t, uint64_t>> ranges;
			for (;;) {
				{
					std::unique_lock lock(m_mutex);
					m_ready.wait(lock, [this] { return !m_ranges.empty(); });
					ranges.swap(m_ranges);
				}
				for (const auto& [vaddr, size]: ranges) {
					VirtualUnlock(reinterpret_cast<void*>(vaddr), size);
				}
				ranges.clear();
			}
		}).detach();
	}

	std::mutex                                 m_mutex;
	std::condition_variable                    m_ready;
	std::vector<std::pair<uint64_t, uint64_t>> m_ranges;
};
#endif

template <typename Regions, typename Source, typename Gap>
static void CopySparseRange(const Regions& regions, uint64_t vaddr, void* data, uint64_t size,
                            Source&& source, Gap&& gap) {
	const auto end     = vaddr + size;
	auto       mapping = regions.lower_bound(vaddr);
	if (mapping != regions.begin()) {
		auto previous = std::prev(mapping);
		if (previous->second.vaddr + previous->second.size > vaddr) {
			mapping = previous;
		}
	}

	auto* destination = static_cast<uint8_t*>(data);
	auto  cursor      = vaddr;
	for (; mapping != regions.end() && mapping->second.vaddr < end; ++mapping) {
		const auto& entry = mapping->second;
		const auto  start = std::max(vaddr, entry.vaddr);
		const auto  bytes = std::min(end, entry.vaddr + entry.size) - start;
		if (cursor < start) {
			gap(cursor, destination + (cursor - vaddr), start - cursor);
		}
		std::memcpy(destination + (start - vaddr), source(entry, start), bytes);
		cursor = start + bytes;
	}
	if (cursor < end) {
		gap(cursor, destination + (cursor - vaddr), end - cursor);
	}
}

// Ranges whose guest mapping is briefly absent while the emulator replaces it: a guest mmap with
// MAP_FIXED over an existing mapping (unmapped, then mapped again), and, on Windows, the parts of a
// file view that stay mapped when part of it is unmapped (Windows splits a view only by unmapping
// all of it). On the console these replacements are atomic, so guest threads keep using the
// memory: GTA V's streamer decompresses into buffers whose 64 KiB pages are mapped again or
// neighbour an unmap. A fault inside such a range waits until the mapping is back and is retried
// instead of being reported as an access violation.
struct TransientUnmapWindow {
	std::atomic<uint64_t> id {0};
	std::atomic<uint64_t> start {0};
	std::atomic<uint64_t> end {0};
	std::atomic<int64_t>  closed_ns {0}; // 0 while the range is being remapped
};

class TransientUnmapWindows {
public:
	using Window = TransientUnmapWindow;

	class Scope {
	public:
		Scope(uint64_t start, uint64_t end) noexcept: m_window(Open(start, end)) {}
		~Scope() { m_window->closed_ns.store(Now(), std::memory_order_release); }
		KYTY_CLASS_NO_COPY(Scope);

	private:
		Window* m_window;
	};

	// The newest range being remapped recently that holds vaddr (0: none). Waits while it still
	// is being remapped.
	static uint64_t Wait(uint64_t vaddr) noexcept {
		constexpr int64_t Second  = 1000000000;
		uint64_t          matched = 0;
		for (auto& window: s_windows) {
			if (vaddr < window.start.load(std::memory_order_acquire) ||
			    vaddr >= window.end.load(std::memory_order_acquire)) {
				continue;
			}
			const auto begin = Now();
			for (uint32_t spins = 0; window.closed_ns.load(std::memory_order_acquire) == 0; spins++) {
				if (Now() - begin > Second * 10) {
					return 0; // never closed: report the fault
				}
				if (spins < 4096) {
#if defined(_M_X64) || defined(__x86_64__)
					_mm_pause();
#endif
				} else {
					std::this_thread::yield();
				}
			}
			// The fault may be handled after the mapping is already back.
			if (Now() - window.closed_ns.load(std::memory_order_acquire) < Second) {
				matched = std::max(matched, window.id.load(std::memory_order_acquire));
			}
		}
		return matched;
	}

private:
	static Window* Open(uint64_t start, uint64_t end) noexcept {
		auto& window = s_windows[s_next.fetch_add(1, std::memory_order_relaxed) % s_windows.size()];
		window.closed_ns.store(0, std::memory_order_relaxed);
		window.id.store(s_opened.fetch_add(1, std::memory_order_relaxed) + 1,
		                std::memory_order_relaxed);
		window.start.store(start, std::memory_order_relaxed);
		window.end.store(end, std::memory_order_seq_cst);
		return &window;
	}

	static int64_t Now() noexcept {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(
		           std::chrono::steady_clock::now().time_since_epoch())
		    .count();
	}

	inline static std::array<Window, 16> s_windows {};
	inline static std::atomic<uint32_t>  s_next {0};
	inline static std::atomic<uint64_t>  s_opened {0};
};

class GuestBackingStore {
public:
	enum class FailureReason {
		None,
		BackingUnavailable,
		PhysicalRangeOutOfBounds,
		FixedMapViewFailed,
		PlaceholderMapFailed,
		ProtectFailed,
		WrongHostAddress,
		AddressRangeUnavailable,
	};

	// What an unmap did with a range: it is a free placeholder now, it stays parked inside its view
	// (KYTY_PARK_UNMAPS), or its view was unmapped whole (the parked ranges of the view are placeholders too).
	enum class UnmapKind { Placeholder, Parked, Released };

	explicit GuestBackingStore(uint64_t size): m_size(size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		m_handle = CreateFileMappingA(
		    INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | SEC_COMMIT,
		    static_cast<DWORD>(m_size >> 32u), static_cast<DWORD>(m_size & 0xffffffffu), nullptr);
		if (m_handle == nullptr) {
			LOGF_COLOR(Log::Color::Red,
			           "\t direct-memory backing: CreateFileMapping failed: 0x%08" PRIx32 "\n",
			           static_cast<uint32_t>(GetLastError()));
			return;
		}
		m_backing_base = static_cast<uint8_t*>(
		    MapViewOfFile(m_handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, m_size));
		if (m_backing_base == nullptr) {
			LOGF_COLOR(Log::Color::Red,
			           "\t direct-memory backing: writable alias failed: 0x%08" PRIx32 "\n",
			           static_cast<uint32_t>(GetLastError()));
			CloseHandle(m_handle);
			m_handle = nullptr;
		}
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#if defined(__APPLE__)
		{
			char shm_name[64];
			snprintf(shm_name, sizeof(shm_name), "/kyty-dm-%d-%llx", static_cast<int>(getpid()),
			         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(this)));
			m_fd = shm_open(shm_name, O_RDWR | O_CREAT | O_EXCL, 0600);
			if (m_fd >= 0) {
				shm_unlink(shm_name);
			}
		}
#else
		m_fd = static_cast<int>(syscall(SYS_memfd_create, "KytyDirectMemory", 0));
#endif
		if (m_fd < 0) {
			LOGF_COLOR(Log::Color::Red, "\t direct-memory backing: shared fd creation failed: %d\n",
			           errno);
			return;
		}
		if (ftruncate(m_fd, static_cast<off_t>(m_size)) != 0) {
			LOGF_COLOR(Log::Color::Red, "\t direct-memory backing: ftruncate failed: %d\n", errno);
			close(m_fd);
			m_fd = -1;
			return;
		}
		auto* backing = mmap(nullptr, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);
		if (backing == MAP_FAILED) {
			LOGF_COLOR(Log::Color::Red, "\t direct-memory backing: writable alias failed: %d\n",
			           errno);
			close(m_fd);
			m_fd = -1;
			return;
		}
		m_backing_base = static_cast<uint8_t*>(backing);
#endif
	}

	~GuestBackingStore() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& [base, size]: m_views) {
			UnmapViewPreservePlaceholder(reinterpret_cast<void*>(base), size);
		}
		m_views.clear();
		m_parked.clear();
		m_maps.clear();
		MapsChangedUnlocked();
		if (m_backing_base != nullptr) {
			UnmapViewOfFile(m_backing_base);
		}
		if (m_handle != nullptr) {
			CloseHandle(m_handle);
		}
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		for (const auto& [_, entry]: m_maps) {
			EXIT_IF(
			    !UnmapViewPreservePlaceholder(reinterpret_cast<void*>(entry.vaddr), entry.size));
		}
		m_maps.clear();
		MapsChangedUnlocked();
		if (m_backing_base != nullptr) {
			munmap(m_backing_base, m_size);
		}
		if (m_fd >= 0) {
			close(m_fd);
		}
#endif
	}

	KYTY_CLASS_NO_COPY(GuestBackingStore);

	static const char* GetFailureReasonName(FailureReason reason) {
		switch (reason) {
			case FailureReason::None: return "n/a";
			case FailureReason::BackingUnavailable: return "backing-unavailable";
			case FailureReason::PhysicalRangeOutOfBounds: return "physical-range-out-of-bounds";
			case FailureReason::FixedMapViewFailed: return "fixed-map-view-failed";
			case FailureReason::PlaceholderMapFailed: return "placeholder-map-failed";
			case FailureReason::ProtectFailed: return "protect-failed";
			case FailureReason::WrongHostAddress: return "wrong-host-address";
			case FailureReason::AddressRangeUnavailable: return "address-range-unavailable";
			default: return "unknown";
		}
	}

	[[nodiscard]] bool IsAvailable() const {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return m_handle != nullptr && m_backing_base != nullptr;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		return m_fd >= 0 && m_backing_base != nullptr;
#else
		return false;
#endif
	}

	[[nodiscard]] uint64_t GetBackingBase() const {
		return reinterpret_cast<uint64_t>(m_backing_base);
	}

	[[nodiscard]] uint64_t GetBackingSize() const { return m_size; }

	bool ZeroRange(uint64_t backing_offset, uint64_t size) {
		if (!IsAvailable() || !RangeOk(backing_offset, size)) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		std::memset(m_backing_base + backing_offset, 0, size);
		return true;
	}

	bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
		if (!IsAvailable() || data == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		const auto* source = static_cast<const uint8_t*>(data);
		if (TryCachedTransfer(vaddr, size, [source](uint8_t* backing, size_t bytes) {
			    std::memcpy(backing, source, bytes);
		    })) {
			return true;
		}

		std::lock_guard lock(m_mutex);
		const auto      end   = vaddr + size;
		auto            first = m_maps.upper_bound(vaddr);
		if (first == m_maps.begin()) {
			return false;
		}
		--first;
		auto current = vaddr;
		for (auto mapping = first; current < end; ++mapping) {
			if (mapping == m_maps.end() || current < mapping->second.vaddr ||
			    current >= mapping->second.vaddr + mapping->second.size) {
				return false;
			}
			const auto& entry = mapping->second;
			const auto  bytes = std::min(end, entry.vaddr + entry.size) - current;
			if (!RangeOk(entry.backing_offset + current - entry.vaddr, bytes)) {
				return false;
			}
			current += bytes;
		}

		RememberMapping(first->second, vaddr, end);
		current = vaddr;
		for (auto mapping = first; current < end; ++mapping) {
			const auto& entry  = mapping->second;
			const auto  bytes  = std::min(end, entry.vaddr + entry.size) - current;
			const auto  offset = entry.backing_offset + current - entry.vaddr;
			std::memcpy(m_backing_base + offset, source + (current - vaddr), bytes);
			current += bytes;
		}
		return true;
	}

	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
		if (!IsAvailable() || data == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		auto* destination = static_cast<uint8_t*>(data);
		if (TryCachedTransfer(vaddr, size, [destination](uint8_t* backing, size_t bytes) {
			    std::memcpy(destination, backing, bytes);
		    })) {
			return true;
		}

		std::shared_lock lock(m_mutex);
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_backing_read != nullptr) {
			g_test_backing_read(vaddr, size);
		}
#endif
		const auto end   = vaddr + size;
		auto       first = m_maps.upper_bound(vaddr);
		if (first == m_maps.begin()) {
			return false;
		}
		--first;
		auto current = vaddr;
		for (auto mapping = first; current < end; ++mapping) {
			if (mapping == m_maps.end() || current < mapping->second.vaddr ||
			    current >= mapping->second.vaddr + mapping->second.size) {
				return false;
			}
			const auto& entry = mapping->second;
			const auto  bytes = std::min(end, entry.vaddr + entry.size) - current;
			if (!RangeOk(entry.backing_offset + current - entry.vaddr, bytes)) {
				return false;
			}
			current += bytes;
		}

		RememberMapping(first->second, vaddr, end);
		current = vaddr;
		for (auto mapping = first; current < end; ++mapping) {
			const auto& entry  = mapping->second;
			const auto  bytes  = std::min(end, entry.vaddr + entry.size) - current;
			const auto  offset = entry.backing_offset + current - entry.vaddr;
			std::memcpy(destination + (current - vaddr), m_backing_base + offset, bytes);
			current += bytes;
		}
		return true;
	}

	// The backing alias of a range that one mapping holds, or null. The alias stays mapped for
	// the store's lifetime, so the pointer can be read later, even after the guest range changed.
	const uint8_t* FindBackingPointer(uint64_t vaddr, uint64_t size) {
		if (!IsAvailable() || size == 0 || UINT64_MAX - vaddr < size) {
			return nullptr;
		}
		const uint8_t* result = nullptr;
		if (TryCachedTransfer(vaddr, size, [&](uint8_t* backing, size_t) { result = backing; })) {
			return result;
		}

		std::shared_lock lock(m_mutex);
		const auto       end     = vaddr + size;
		auto             mapping = m_maps.upper_bound(vaddr);
		if (mapping == m_maps.begin()) {
			return nullptr;
		}
		const auto& entry = std::prev(mapping)->second;
		if (vaddr < entry.vaddr || end > entry.vaddr + entry.size ||
		    !RangeOk(entry.backing_offset + vaddr - entry.vaddr, size)) {
			return nullptr;
		}
		RememberMapping(entry, vaddr, end);
		return m_backing_base + entry.backing_offset + (vaddr - entry.vaddr);
	}

	template <typename Gap>
	bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size, Gap&& gap) {
		if (!IsAvailable() || data == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		std::shared_lock lock(m_mutex);
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_backing_read != nullptr) {
			g_test_backing_read(vaddr, size);
		}
#endif
		CopySparseRange(m_maps, vaddr, data, size,
		                [this](const auto& entry, uint64_t start) {
			                return m_backing_base + (entry.backing_offset + start - entry.vaddr);
		                },
		                gap);
		return true;
	}

	bool MapFixed(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	              VirtualMemory::Mode mode, FailureReason* failure_reason = nullptr) {
		if (!IsAvailable() || !RangeOk(backing_offset, size)) {
			SetFailure(failure_reason, !IsAvailable() ? FailureReason::BackingUnavailable
			                                          : FailureReason::PhysicalRangeOutOfBounds);
			return false;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		void* mapped = ReplacePlaceholderView(vaddr, size, backing_offset, mode, failure_reason);
		if (mapped == nullptr || reinterpret_cast<uint64_t>(mapped) != vaddr) {
			if (mapped != nullptr) {
				UnmapViewPreservePlaceholder(mapped, size);
				SetFailure(failure_reason, FailureReason::WrongHostAddress);
			}
			return false;
		}

		{
			std::lock_guard lock(m_mutex);
			EXIT_IF(!m_maps.emplace(vaddr, MapEntry {vaddr, size, backing_offset, mode}).second);
			EXIT_IF(!m_views.emplace(vaddr, size).second);
			MapsChangedUnlocked();
		}
		BumpBackingEpoch(vaddr, size);
		SetFailure(failure_reason, FailureReason::None);
		return true;
#else
		std::lock_guard lock(m_mutex);
		void* mapped = MapView(vaddr, size, backing_offset, mode, failure_reason);
		if (mapped == nullptr || reinterpret_cast<uint64_t>(mapped) != vaddr) {
			return false;
		}
		RemoveMappingsUnlocked(vaddr, size);
		EXIT_IF(!m_maps.emplace(vaddr, MapEntry {vaddr, size, backing_offset, mode}).second);
		MapsChangedUnlocked();
		BumpBackingEpoch(vaddr, size);
		SetFailure(failure_reason, FailureReason::None);
		return true;
#endif
	}

	// [vaddr, vaddr + size) unmapped. The ranges that became free placeholders are appended to `freed` (when given):
	// the range itself, except on Windows with KYTY_PARK_UNMAPS, where a range left parked in its view is none and a
	// view released whole frees the parked ranges of earlier unmaps with it.
	bool Unmap(uint64_t vaddr, uint64_t size, bool* placeholder_preserved = nullptr,
	           std::vector<std::pair<uint64_t, uint64_t>>* freed = nullptr) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		BumpBackingEpoch(vaddr, size);

		if (placeholder_preserved != nullptr) {
			*placeholder_preserved = false;
		}

		const auto            end = vaddr + size;
		std::vector<MapEntry> targets;
		{
			std::lock_guard lock(m_mutex);
			auto            current = vaddr;
			while (current < end) {
				MapEntry entry {};
				if (!FindContainingUnlocked(current, 1, &entry)) {
					return false;
				}
				const auto part_size = std::min(end, entry.vaddr + entry.size) - current;
				targets.push_back(
				    {current, part_size, entry.backing_offset + current - entry.vaddr, entry.mode});
				current += part_size;
			}
		}

		struct Removed {
			MapEntry  entry;
			UnmapKind kind;
		};
		std::vector<Removed>                       removed;
		std::vector<std::pair<uint64_t, uint64_t>> part_freed;
		removed.reserve(targets.size());
		for (const auto& entry: targets) {
			bool      part_placeholder_preserved = false;
			UnmapKind kind                       = UnmapKind::Placeholder;
			if (!UnmapOne(entry.vaddr, entry.size, &part_placeholder_preserved, &part_freed, &kind)) {
				for (auto it = removed.rbegin(); it != removed.rend(); ++it) {
					RemapUnmapped(it->entry, it->kind);
				}
				if (placeholder_preserved != nullptr) {
					*placeholder_preserved = false;
				}
				return false;
			}
			EXIT_IF(!part_placeholder_preserved);
			removed.push_back({entry, kind});
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (!ParkUnmapsEnabled()) {
			// Each unmapped file view becomes a separate Windows placeholder. The address-space owner
			// records this contiguous guest span as one free range, so normalize fragmented backing
			// views into one placeholder before publishing it there.
			if (removed.size() > 1 && !CoalescePlaceholders(vaddr, size)) {
				for (auto it = removed.rbegin(); it != removed.rend(); ++it) {
					RemapUnmapped(it->entry, it->kind);
				}
				return false;
			}
			if (freed != nullptr) {
				freed->emplace_back(vaddr, size);
			}
		} else if (freed != nullptr) {
			// (Placeholders next to each other are coalesced when the owner adds them to its free list.)
			freed->insert(freed->end(), part_freed.begin(), part_freed.end());
		}
#else
		if (freed != nullptr) {
			freed->emplace_back(vaddr, size);
		}
#endif

		if (placeholder_preserved != nullptr) {
			*placeholder_preserved = true;
		}
		return true;
	}

	bool UnmapOne(uint64_t vaddr, uint64_t size, bool* placeholder_preserved = nullptr,
	              std::vector<std::pair<uint64_t, uint64_t>>* freed = nullptr,
	              UnmapKind*                                  kind  = nullptr) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_backing_store_unmaps_before_failure == 0) {
			g_test_backing_store_unmaps_before_failure = UINT32_MAX;
			return false;
		}
		if (g_test_backing_store_unmaps_before_failure != UINT32_MAX) {
			g_test_backing_store_unmaps_before_failure--;
		}
#endif
		if (kind != nullptr) {
			*kind = UnmapKind::Placeholder;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// Whole: the range is its view. Split: the view is unmapped and its other parts mapped again
		// (a view cannot be split). Park: the range stays in its view (m_parked). Release: the view
		// is unmapped whole, nothing of it stays mapped.
		enum class How { Whole, Split, Park, Release };
		const bool park = ParkUnmapsEnabled();
		MapEntry   old {};
		uint64_t   view_base = 0;
		uint64_t   view_size = 0;
		How        how       = How::Whole;
		{
			std::lock_guard lock(m_mutex);
			const auto      next = m_maps.upper_bound(vaddr);
			if (next == m_maps.begin()) {
				return false;
			}
			const auto entry = std::prev(next);
			if (vaddr + size > entry->second.vaddr + entry->second.size) {
				return false;
			}
			old             = entry->second;
			const auto view = FindViewUnlocked(old.vaddr);
			EXIT_IF(view == m_views.end() || old.vaddr + old.size > view->first + view->second);
			view_base             = view->first;
			view_size             = view->second;
			const bool whole_view = old.vaddr == view_base && old.size == view_size;
			const bool exact      = vaddr == old.vaddr && size == old.size;
			how = !whole_view ? How::Park : exact ? How::Whole : park ? How::Park : How::Split;
			m_maps.erase(entry);
			if (how != How::Park) {
				m_views.erase(view);
			} else {
				if (vaddr != old.vaddr) {
					EXIT_IF(!m_maps
					             .emplace(old.vaddr, MapEntry {old.vaddr, vaddr - old.vaddr,
					                                           old.backing_offset, old.mode})
					             .second);
				}
				if (const auto right = vaddr + size; right != old.vaddr + old.size) {
					EXIT_IF(!m_maps
					             .emplace(right, MapEntry {right, old.vaddr + old.size - right,
					                                       old.backing_offset + (right - old.vaddr),
					                                       old.mode})
					             .second);
				}
				const auto view_end = view_base + view_size;
				if (const auto live = m_maps.lower_bound(view_base);
				    live == m_maps.end() || live->first >= view_end) {
					how = How::Release;
					m_parked.erase(m_parked.lower_bound(view_base), m_parked.lower_bound(view_end));
					m_views.erase(view);
				} else {
					ParkUnlocked(vaddr, size, view_base, view_end);
				}
			}
			MapsChangedUnlocked();
		}

		if (how == How::Park) {
			// The range keeps its protection and its pages: the guest has no mapping there (a map over
			// it splits the view first, Materialize) and the emulator's reads and writes of guest
			// memory find none either (m_maps). Its pages leave the working set on the trimming thread.
			ParkedRangeTrimmer::Trim(vaddr, size);
			if (kind != nullptr) {
				*kind = UnmapKind::Parked;
			}
			if (placeholder_preserved != nullptr) {
				*placeholder_preserved = true;
			}
			return true;
		}
		if (how == How::Release) {
			EXIT_IF(!UnmapViewPreservePlaceholder(reinterpret_cast<void*>(view_base), view_size));
			if (freed != nullptr) {
				freed->emplace_back(view_base, view_size);
			}
			if (kind != nullptr) {
				*kind = UnmapKind::Released;
			}
			if (placeholder_preserved != nullptr) {
				*placeholder_preserved = true;
			}
			return true;
		}
		if (how == How::Whole) {
			const bool unmapped =
			    UnmapViewPreservePlaceholder(reinterpret_cast<void*>(old.vaddr), old.size);
			if (!unmapped) {
				std::lock_guard lock(m_mutex);
				m_maps.emplace(old.vaddr, old);
				m_views.emplace(old.vaddr, old.size);
				MapsChangedUnlocked();
				return false;
			}
			if (freed != nullptr) {
				freed->emplace_back(vaddr, size);
			}
			if (placeholder_preserved != nullptr) {
				*placeholder_preserved = true;
			}
			return true;
		}

		const uint64_t left_size  = vaddr - old.vaddr;
		const uint64_t right_addr = vaddr + size;
		const uint64_t right_size = old.vaddr + old.size - right_addr;

		// The parts that stay mapped vanish until they are mapped again (see
		// TransientUnmapWindows), and come back with the view's protection: give them back their
		// own (guest mprotect, GPU write and read tracking) before faults there are retried.
		const auto                   kept_protections = SnapshotProtections(old, vaddr, size);
		const uint64_t               old_end          = old.vaddr + old.size;
		TransientUnmapWindows::Scope transient(old.vaddr, old_end);
		if (UnmapViewPreservePlaceholder(reinterpret_cast<void*>(old.vaddr), old.size)) {
			bool ok = true;
			if (left_size != 0 && left_size != old.size) {
				ok = SplitPlaceholder(old.vaddr, left_size) && ok;
			}
			if (right_size != 0) {
				ok = SplitPlaceholder(vaddr, size) && ok;
			}
			if (left_size != 0) {
				ok = MapFixed(old.vaddr, left_size, old.backing_offset, old.mode) && ok;
			}
			if (right_size != 0) {
				ok = MapFixed(right_addr, right_size,
				              old.backing_offset + (right_addr - old.vaddr), old.mode) &&
				     ok;
			}
			if (ok) {
				RestoreProtections(kept_protections);
				if (freed != nullptr) {
					freed->emplace_back(vaddr, size);
				}
				if (placeholder_preserved != nullptr) {
					*placeholder_preserved = true;
				}
				return true;
			}

			if (left_size != 0 && Contains(old.vaddr, left_size)) {
				EXIT_IF(!UnmapOne(old.vaddr, left_size));
			}
			if (right_size != 0 && Contains(right_addr, right_size)) {
				EXIT_IF(!UnmapOne(right_addr, right_size));
			}
			EXIT_IF(VirtualFree(reinterpret_cast<void*>(old.vaddr), old.size,
			                    MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) == 0);
			EXIT_IF(!MapFixed(old.vaddr, old.size, old.backing_offset, old.mode));
			RestoreProtections(kept_protections);
			return false;
		}
		std::lock_guard lock(m_mutex);
		m_maps.emplace(old.vaddr, old);
		m_views.emplace(old.vaddr, old.size);
		MapsChangedUnlocked();
		return false;
#else
		MapEntry old {};
		{
			std::lock_guard lock(m_mutex);
			auto            next = m_maps.upper_bound(vaddr);
			if (next != m_maps.begin()) {
				auto entry = std::prev(next);
				if (vaddr + size <= entry->second.vaddr + entry->second.size) {
					old = entry->second;
				}
			}
		}

		if (old.size == 0) {
			return false;
		}

		// Replace only the requested range so concurrent accesses and GPU watch
		// protections on the surviving mappings remain intact.
		if (!UnmapViewPreservePlaceholder(reinterpret_cast<void*>(vaddr), size)) {
			return false;
		}

		std::lock_guard lock(m_mutex);
		RemoveMappingsUnlocked(vaddr, size);
		MapsChangedUnlocked();

		if (placeholder_preserved != nullptr) {
			*placeholder_preserved = true;
		}
		return true;
#endif
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Whether ranges parked in views (UnmapOne) overlap [vaddr, vaddr + size).
	bool HasParked(uint64_t vaddr, uint64_t size) {
		std::shared_lock lock(m_mutex);
		return FirstParkedUnlocked(vaddr, size) != m_parked.end();
	}

	// Whether ranges parked in views overlap [vaddr, vaddr + size); the span of their views widens
	// [*begin, *end) (Materialize unmaps those views and maps their other parts again).
	bool ParkedViews(uint64_t vaddr, uint64_t size, uint64_t* begin, uint64_t* end) {
		std::shared_lock lock(m_mutex);
		bool             found = false;
		for (auto it = FirstParkedUnlocked(vaddr, size); it != m_parked.end() && it->first < vaddr + size; ++it) {
			const auto view = FindViewUnlocked(it->first);
			EXIT_IF(view == m_views.end());
			*begin = std::min(*begin, view->first);
			*end   = std::max(*end, view->first + view->second);
			found  = true;
		}
		return found;
	}

	// The span of the views over [vaddr, vaddr + size), widening [*begin, *end): an unmap there unmaps
	// the first and the last view whole to map their other parts again, or releases them.
	void ViewSpan(uint64_t vaddr, uint64_t size, uint64_t* begin, uint64_t* end) {
		std::shared_lock lock(m_mutex);
		if (const auto view = FindViewUnlocked(vaddr); view != m_views.end()) {
			*begin = std::min(*begin, view->first);
		}
		if (const auto view = FindViewUnlocked(vaddr + size - 1); view != m_views.end()) {
			*end = std::max(*end, view->first + view->second);
		}
	}

	// The end of the parked range holding `vaddr`, 0 when none.
	uint64_t ParkedEnd(uint64_t vaddr) {
		std::shared_lock lock(m_mutex);
		const auto       it = FirstParkedUnlocked(vaddr, 1);
		return it != m_parked.end() ? it->first + it->second : 0;
	}

	// The views holding ranges parked over [vaddr, vaddr + size) split for real, before something is
	// mapped there: each is unmapped whole and its mapped parts are mapped again as views of their own
	// (as an unmap without KYTY_PARK_UNMAPS does, with their protections given back); its parked
	// ranges, free placeholders now, are appended to `freed`.
	void Materialize(uint64_t vaddr, uint64_t size, std::vector<std::pair<uint64_t, uint64_t>>* freed) {
		for (;;) {
			uint64_t                                   view_base = 0;
			uint64_t                                   view_size = 0;
			std::vector<MapEntry>                      mapped;
			std::vector<std::pair<uint64_t, uint64_t>> parts;
			{
				std::lock_guard lock(m_mutex);
				const auto      parked = FirstParkedUnlocked(vaddr, size);
				if (parked == m_parked.end()) {
					return;
				}
				const auto view = FindViewUnlocked(parked->first);
				EXIT_IF(view == m_views.end());
				view_base           = view->first;
				view_size           = view->second;
				const auto view_end = view_base + view_size;
				for (auto it = m_maps.lower_bound(view_base); it != m_maps.end() && it->first < view_end;) {
					mapped.push_back(it->second);
					parts.emplace_back(it->second.vaddr, it->second.size);
					it = m_maps.erase(it);
				}
				for (auto it = m_parked.lower_bound(view_base);
				     it != m_parked.end() && it->first < view_end;) {
					parts.emplace_back(it->first, it->second);
					freed->emplace_back(it->first, it->second);
					it = m_parked.erase(it);
				}
				m_views.erase(view);
				MapsChangedUnlocked();
			}

			std::vector<KeptProtection> kept_protections;
			for (const auto& entry: mapped) {
				SnapshotRange(entry.vaddr, entry.vaddr + entry.size, GetProtect(entry.mode),
				              kept_protections);
			}
			TransientUnmapWindows::Scope transient(view_base, view_base + view_size);
			EXIT_IF(!UnmapViewPreservePlaceholder(reinterpret_cast<void*>(view_base), view_size));
			// The view's placeholder is cut at the parts (they tile it), then the mapped ones are mapped again.
			std::sort(parts.begin(), parts.end());
			uint64_t at = view_base;
			for (size_t i = 0; i < parts.size(); ++i) {
				EXIT_IF(parts[i].first != at);
				at += parts[i].second;
				if (i + 1 < parts.size()) {
					EXIT_IF(!SplitPlaceholder(parts[i].first, parts[i].second));
				}
			}
			EXIT_IF(at != view_base + view_size);
			for (const auto& entry: mapped) {
				EXIT_IF(!MapFixed(entry.vaddr, entry.size, entry.backing_offset, entry.mode));
			}
			RestoreProtections(kept_protections);
		}
	}
#endif

	bool Contains(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		std::shared_lock lock(m_mutex);
		const auto      end     = vaddr + size;
		auto            current = vaddr;

		while (current < end) {
			MapEntry entry {};
			if (!FindContainingUnlocked(current, 1, &entry)) {
				return false;
			}
			const auto entry_end = entry.vaddr + entry.size;
			if (entry_end <= current) {
				return false;
			}
			current = std::min(end, entry_end);
		}

		return true;
	}

private:
	struct MapEntry {
		uint64_t            vaddr          = 0;
		uint64_t            size           = 0;
		uint64_t            backing_offset = 0;
		VirtualMemory::Mode mode           = VirtualMemory::Mode::NoAccess;
	};

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	void RemoveMappingsUnlocked(uint64_t vaddr, uint64_t size) {
		const auto end = vaddr + size;
		auto       it = m_maps.upper_bound(vaddr);
		if (it != m_maps.begin()) {
			--it;
		}
		while (it != m_maps.end() && it->first < end) {
			const auto old     = it->second;
			const auto old_end = old.vaddr + old.size;
			if (old_end <= vaddr) {
				++it;
				continue;
			}
			it = m_maps.erase(it);
			if (old.vaddr < vaddr) {
				m_maps.emplace(old.vaddr,
				               MapEntry {old.vaddr, vaddr - old.vaddr, old.backing_offset, old.mode});
			}
			if (end < old_end) {
				m_maps.emplace(end, MapEntry {end, old_end - end,
				                             old.backing_offset + end - old.vaddr, old.mode});
			}
		}
	}
#endif
	// The mapping of each 64 KB of guest addresses that a transfer started in, direct mapped:
	// GPU threads stream thousands of small buffers per frame from rings spread over more
	// mappings than the round-robin entries hold.
	static constexpr uint32_t SlotBits  = 16;
	static constexpr size_t   SlotCount = 1024;
	struct LookupCache {
		const GuestBackingStore*         owner      = nullptr;
		uint64_t                         generation = 0;
		uint32_t                         next       = 0;
		std::array<MapEntry, 4>          entries {};
		std::array<MapEntry, SlotCount>  slots {};
	};

	[[nodiscard]] static size_t SlotOf(uint64_t vaddr) noexcept {
		return static_cast<size_t>(vaddr >> SlotBits) % SlotCount;
	}

	static LookupCache& ThreadLookupCache() noexcept {
		thread_local LookupCache cache;
		return cache;
	}

	// Called after every change of m_maps, under m_mutex. The generation is shared by all stores,
	// so a store recreated at the address of a destroyed one never matches a stale lookup.
	static void MapsChangedUnlocked() noexcept {
		s_maps_generation.fetch_add(1, std::memory_order_release);
	}

	inline static std::atomic<uint64_t> s_maps_generation {0};

	// The GPU thread streams thousands of small buffers per frame from a few mappings: reuse the
	// lookups it made last, without the lock, while no mapping changed. The entries and the
	// generation were read together under m_mutex, and every change of m_maps advances the
	// generation under m_mutex, so observing it unchanged (acquire) means no change completed
	// since. A change racing with this transfer orders as if it followed it: the backing alias
	// stays mapped for the store's lifetime, and host views are never unmapped under m_mutex
	// either.
	template <typename Transfer>
	bool TryCachedTransfer(uint64_t vaddr, uint64_t size, Transfer transfer) {
		const auto& cached = ThreadLookupCache();
		if (cached.owner != this ||
		    cached.generation != s_maps_generation.load(std::memory_order_acquire)) {
			return false;
		}
		const auto end  = vaddr + size;
		const auto& slot = cached.slots[SlotOf(vaddr)];
		if (vaddr >= slot.vaddr && end <= slot.vaddr + slot.size && slot.size != 0) {
			transfer(m_backing_base + slot.backing_offset + (vaddr - slot.vaddr),
			         static_cast<size_t>(size));
			return true;
		}
		for (const auto& entry: cached.entries) {
			if (vaddr >= entry.vaddr && end <= entry.vaddr + entry.size) {
				transfer(m_backing_base + entry.backing_offset + (vaddr - entry.vaddr),
				         static_cast<size_t>(size));
				return true;
			}
		}
		return false;
	}

	// Called under m_mutex (shared or exclusive) after a lookup: remembers a mapping that holds
	// the whole transfer [entry.vaddr <= vaddr, end].
	void RememberMapping(const MapEntry& entry, uint64_t vaddr, uint64_t end) {
		if (end > entry.vaddr + entry.size || !RangeOk(entry.backing_offset, entry.size)) {
			return;
		}
		auto&      cached     = ThreadLookupCache();
		const auto generation = s_maps_generation.load(std::memory_order_relaxed);
		if (cached.owner != this || cached.generation != generation) {
			cached.owner      = this;
			cached.generation = generation;
			cached.next       = 0;
			cached.entries.fill({});
			cached.slots.fill({});
		}
		cached.entries[cached.next++ % cached.entries.size()] = entry;
		cached.slots[SlotOf(vaddr)]                           = entry;
	}

	bool FindContainingUnlocked(uint64_t vaddr, uint64_t size, MapEntry* out) const {
		EXIT_IF(out == nullptr);

		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		auto next = m_maps.upper_bound(vaddr);
		if (next == m_maps.begin()) {
			return false;
		}
		const auto& entry = std::prev(next)->second;
		if (vaddr + size <= entry.vaddr + entry.size) {
			*out = entry;
			return true;
		}

		return false;
	}

	[[nodiscard]] bool RangeOk(uint64_t backing_offset, uint64_t size) const {
		return size != 0 && backing_offset < m_size && size <= m_size - backing_offset;
	}

	static void SetFailure(FailureReason* out, FailureReason reason) {
		if (out != nullptr) {
			*out = reason;
		}
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static DWORD GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PAGE_READONLY;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PAGE_READWRITE;
			case VirtualMemory::Mode::Execute: return PAGE_EXECUTE;
			case VirtualMemory::Mode::ExecuteRead: return PAGE_EXECUTE_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PAGE_EXECUTE_READWRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PAGE_NOACCESS;
		}
	}

	static bool ApplyProtection(void* ptr, uint64_t size, VirtualMemory::Mode mode) {
		DWORD old_protect = 0;
		return VirtualProtect(ptr, size, GetProtect(mode), &old_protect) != 0;
	}

	struct KeptProtection {
		uint64_t vaddr   = 0;
		uint64_t size    = 0;
		DWORD    protect = 0;
	};

	// The protections other than `view_protect` of [start, end), appended to `kept`.
	static void SnapshotRange(uint64_t start, uint64_t end, DWORD view_protect,
	                          std::vector<KeptProtection>& kept) {
		for (auto current = start; current < end;) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<const void*>(current), &info, sizeof(info)) == 0) {
				break;
			}
			const auto region_end =
			    std::min(end, reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize);
			if (region_end <= current) {
				break;
			}
			if (info.State == MEM_COMMIT && info.Protect != view_protect) {
				kept.push_back({current, region_end - current, info.Protect});
			}
			current = region_end;
		}
	}

	// The protections, other than the view's own, of the parts of a view that stay mapped when
	// [vaddr, vaddr + size) is unmapped from it.
	static std::vector<KeptProtection> SnapshotProtections(const MapEntry& view, uint64_t vaddr,
	                                                       uint64_t size) {
		std::vector<KeptProtection> kept;
		const auto                  view_protect = GetProtect(view.mode);
		for (const auto& [start, end]:
		     {std::pair {view.vaddr, vaddr}, std::pair {vaddr + size, view.vaddr + view.size}}) {
			SnapshotRange(start, end, view_protect, kept);
		}
		return kept;
	}

	static void RestoreProtections(const std::vector<KeptProtection>& kept) {
		for (const auto& part: kept) {
			DWORD old_protect = 0;
			if (VirtualProtect(reinterpret_cast<void*>(part.vaddr), part.size, part.protect,
			                   &old_protect) == 0) {
				LOGF_COLOR(Log::Color::Red,
				           "\t view split: protection 0x%lx of [0x%016" PRIx64 ", 0x%016" PRIx64
				           ") not restored, error=0x%08" PRIx32 "\n",
				           static_cast<unsigned long>(part.protect), part.vaddr,
				           part.vaddr + part.size, static_cast<uint32_t>(GetLastError()));
			}
		}
	}

	static bool SplitPlaceholder(uint64_t vaddr, uint64_t size) {
		if (size == 0) {
			return true;
		}
		return VirtualFree(reinterpret_cast<void*>(vaddr), size,
		                   MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) != 0;
	}

	void* ReplacePlaceholderView(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                             VirtualMemory::Mode mode, FailureReason* failure_reason) {
		const DWORD map_protect =
		    mode == VirtualMemory::Mode::NoAccess ? PAGE_READWRITE : GetProtect(mode);
		void* ptr =
		    MapViewOfFile3(m_handle, GetCurrentProcess(), reinterpret_cast<void*>(vaddr),
		                   backing_offset, size, MEM_REPLACE_PLACEHOLDER, map_protect, nullptr, 0);
		if (ptr == nullptr || reinterpret_cast<uint64_t>(ptr) != vaddr) {
			if (ptr != nullptr) {
				UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			}
			SetFailure(failure_reason, FailureReason::PlaceholderMapFailed);
			return nullptr;
		}

		if (mode == VirtualMemory::Mode::NoAccess && !ApplyProtection(ptr, size, mode)) {
			UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			SetFailure(failure_reason, FailureReason::ProtectFailed);
			return nullptr;
		}
		return ptr;
	}

	static bool UnmapViewPreservePlaceholder(void* vaddr, uint64_t /*size*/) {
		return UnmapViewOfFile2(GetCurrentProcess(), vaddr, MEM_PRESERVE_PLACEHOLDER) != 0;
	}

	HANDLE m_handle = nullptr;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	[[nodiscard]] static uint64_t GetMappableTestSize() { return 0x4000; }

	static int GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PROT_READ;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PROT_READ | PROT_WRITE;
			case VirtualMemory::Mode::Execute: return PROT_EXEC;
			case VirtualMemory::Mode::ExecuteRead: return PROT_EXEC | PROT_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PROT_EXEC | PROT_READ | PROT_WRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PROT_NONE;
		}
	}

	void* MapView(uint64_t vaddr, uint64_t size, uint64_t backing_offset, VirtualMemory::Mode mode,
	              FailureReason* failure_reason) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (const auto callback = g_test_before_backing_map) {
			g_test_before_backing_map = nullptr;
			callback(vaddr, size);
		}
#endif
		void* ptr = mmap(reinterpret_cast<void*>(vaddr), size, GetProtect(mode),
		                 MAP_SHARED | MAP_FIXED, m_fd, static_cast<off_t>(backing_offset));
		if (ptr == MAP_FAILED) {
			SetFailure(failure_reason, FailureReason::FixedMapViewFailed);
			return nullptr;
		}
		return ptr;
	}

	// Replace the file view atomically instead of exposing an unmapped hole in the
	// process-lifetime guest reservation.
	static bool UnmapViewPreservePlaceholder(void* vaddr, uint64_t size) {
		void* ptr = mmap(vaddr, size, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
		return ptr != MAP_FAILED && ptr == vaddr;
	}

	int m_fd = -1;
#endif

	// Gives back to the guest a range that Unmap took out, because a later part of the same unmap
	// failed.
	void RemapUnmapped(const MapEntry& entry, UnmapKind kind) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (kind == UnmapKind::Parked) {
			// (It never left its view.)
			std::lock_guard lock(m_mutex);
			UnparkUnlocked(entry);
			return;
		}
		// (A view released whole can only be given back by a failure injected by the tests, and
		// none of them unmaps part of a view first.)
		EXIT_IF(kind == UnmapKind::Released);
#else
		(void)kind;
#endif
		EXIT_IF(!MapFixed(entry.vaddr, entry.size, entry.backing_offset, entry.mode));
	}

	uint8_t*                     m_backing_base = nullptr;
	uint64_t                     m_size         = 0;
	std::shared_mutex            m_mutex;
	std::map<uint64_t, MapEntry> m_maps;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// The file views of the guest mappings (base -> size): each view's mapped parts (m_maps) and
	// parked ranges (m_parked) tile it. A view is one mapping until an unmap parks a range in it.
	std::map<uint64_t, uint64_t> m_views;
	// Ranges the guest unmapped from views that stay mapped (base -> size, each inside one view,
	// KYTY_PARK_UNMAPS). They are split off for real when something is mapped over one
	// (Materialize), or unmapped with their view once nothing of it stays mapped.
	std::map<uint64_t, uint64_t> m_parked;

	std::map<uint64_t, uint64_t>::iterator FindViewUnlocked(uint64_t vaddr) {
		const auto next = m_views.upper_bound(vaddr);
		if (next == m_views.begin()) {
			return m_views.end();
		}
		const auto view = std::prev(next);
		return vaddr < view->first + view->second ? view : m_views.end();
	}

	// The first parked range that overlaps [vaddr, vaddr + size), or end.
	std::map<uint64_t, uint64_t>::iterator FirstParkedUnlocked(uint64_t vaddr, uint64_t size) {
		auto it = m_parked.upper_bound(vaddr);
		if (it != m_parked.begin() && std::prev(it)->first + std::prev(it)->second > vaddr) {
			--it;
		}
		return it != m_parked.end() && it->first < vaddr + size ? it : m_parked.end();
	}

	// [vaddr, vaddr + size) parked, joined to the parked ranges next to it in the same view
	// [view_begin, view_end).
	void ParkUnlocked(uint64_t vaddr, uint64_t size, uint64_t view_begin, uint64_t view_end) {
		uint64_t begin = vaddr;
		uint64_t end   = vaddr + size;
		auto     next  = m_parked.lower_bound(vaddr);
		if (next != m_parked.begin()) {
			const auto previous = std::prev(next);
			if (previous->first >= view_begin && previous->first + previous->second == begin) {
				begin = previous->first;
				m_parked.erase(previous);
			}
		}
		if (next != m_parked.end() && next->first == end && next->first + next->second <= view_end) {
			end = next->first + next->second;
			m_parked.erase(next);
		}
		EXIT_IF(!m_parked.emplace(begin, end - begin).second);
	}

	// A parked range mapped for the guest again (an unmap failed after parking it).
	void UnparkUnlocked(const MapEntry& entry) {
		auto it = m_parked.upper_bound(entry.vaddr);
		EXIT_IF(it == m_parked.begin());
		--it;
		const auto begin = it->first;
		const auto end   = it->first + it->second;
		const auto stop  = entry.vaddr + entry.size;
		EXIT_IF(entry.vaddr < begin || stop > end);
		m_parked.erase(it);
		if (begin < entry.vaddr) {
			EXIT_IF(!m_parked.emplace(begin, entry.vaddr - begin).second);
		}
		if (stop < end) {
			EXIT_IF(!m_parked.emplace(stop, end - stop).second);
		}
		EXIT_IF(!m_maps.emplace(entry.vaddr, entry).second);
		MapsChangedUnlocked();
	}
#endif
};

class GuestAddressSpace {
public:
	explicit GuestAddressSpace(uint64_t backing_size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		m_granularity = info.dwAllocationGranularity;
#endif
		ReserveGuestRegions();
		m_backing = new GuestBackingStore(backing_size);
		if (!m_backing->IsAvailable()) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			EXIT("could not reserve %" PRIu64 " MB for guest direct memory. Windows commits\n"
			     "this up front, so the paging file is usually what needs to be larger.\n",
			     backing_size / (1024U * 1024U));
#else
			EXIT("could not reserve %" PRIu64 " MB for guest direct memory\n",
			     backing_size / (1024U * 1024U));
#endif
		}
	}

	~GuestAddressSpace() {
		delete m_backing;
		m_backing = nullptr;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& [start, size]: m_owned) {
			const auto end     = start + size;
			auto       current = start;
			while (current < end) {
				MEMORY_BASIC_INFORMATION info {};
				EXIT_IF(VirtualQuery(reinterpret_cast<const void*>(current), &info, sizeof(info)) ==
				        0);
				const auto region_start = reinterpret_cast<uint64_t>(info.BaseAddress);
				const auto region_end   = region_start + static_cast<uint64_t>(info.RegionSize);
				if (info.State == MEM_COMMIT) {
					EXIT_IF(VirtualFree(reinterpret_cast<void*>(region_start), info.RegionSize,
					                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0);
					continue;
				}
				if (info.State == MEM_RESERVE) {
					EXIT_IF(VirtualFree(info.AllocationBase, 0, MEM_RELEASE) == 0);
				}
				current = region_end;
			}
		}
#else
		for (const auto& [start, size]: m_owned) {
			munmap(reinterpret_cast<void*>(start), size);
		}
#endif
	}

	KYTY_CLASS_NO_COPY(GuestAddressSpace);

	[[nodiscard]] uint64_t GetGranularity() const { return m_granularity; }
	[[nodiscard]] uint64_t GetBackingBase() const { return m_backing->GetBackingBase(); }
	[[nodiscard]] uint64_t GetBackingSize() const { return m_backing->GetBackingSize(); }

	bool ZeroBacking(uint64_t backing_offset, uint64_t size) {
		return m_backing->ZeroRange(backing_offset, size);
	}

	[[nodiscard]] bool Owns(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		std::shared_lock lock(m_mutex);
		return std::any_of(m_owned.begin(), m_owned.end(), [vaddr, size](const auto& range) {
			return vaddr >= range.first && vaddr + size <= range.first + range.second;
		});
	}

	[[nodiscard]] bool OverlapsOwned(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		return std::any_of(m_owned.begin(), m_owned.end(), [vaddr, size](const auto& range) {
			return VirtualRangesOverlap(vaddr, size, range.first, range.second);
		});
	}

	bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
		return m_backing->TryWriteBacking(vaddr, data, size);
	}

	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
		return m_backing->TryReadBacking(vaddr, data, size);
	}

	const uint8_t* FindBackingPointer(uint64_t vaddr, uint64_t size) {
		return m_backing->FindBackingPointer(vaddr, size);
	}

	template <bool copy_private>
	bool CopySparseMemory(uint64_t vaddr, void* data, uint64_t size) {
		if (data == nullptr || !Owns(vaddr, size)) {
			return false;
		}
		std::shared_lock lock(m_mutex);
		WaitNotBusy(lock, vaddr, size);
		if constexpr (!copy_private) {
			auto mapping = m_mapped.upper_bound(vaddr);
			if (mapping != m_mapped.begin()) {
				--mapping;
			}
			for (; mapping != m_mapped.end() && mapping->second.vaddr < vaddr + size; ++mapping) {
				const auto& region = mapping->second;
				if (region.type == MappingType::Private && region.vaddr + region.size > vaddr) {
					return false;
				}
			}
		}
		return m_backing->TryReadSparseBacking(
		    vaddr, data, size, [&](uint64_t start, uint8_t* destination, uint64_t bytes) {
			    const auto zero = [](uint64_t, uint8_t* output, uint64_t length) {
				    std::memset(output, 0, length);
			    };
			    if constexpr (copy_private) {
				    // Backing gaps can contain private mappings or reservations only.
				    CopySparseRange(m_mapped, start, destination, bytes,
				                    [](const auto& region, uint64_t address) {
					                    EXIT_IF(region.type != MappingType::Private);
					                    return reinterpret_cast<const void*>(address);
				                    },
				                    zero);
			    } else {
				    zero(start, destination, bytes);
			    }
		    });
	}

	bool BackingContains(uint64_t vaddr, uint64_t size) { return m_backing->Contains(vaddr, size); }

	bool Protect(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		std::unique_lock lock(m_mutex);
		WaitNotBusy(lock, vaddr, size);
		return ProtectMappedUnlocked(vaddr, size, mode);
	}

	// Only the PageManager calls this, with the SpinGuard of the page's region held for the whole
	// call, so every change of one page's protection through here is serialized per region, and a
	// change never crosses regions. The mapping table is only read, so this takes m_mutex shared
	// and the many watcher updates of the GPU threads and of the game's write faults run in
	// parallel. Protect() (the game's mprotect) still takes m_mutex exclusive and so excludes it.
	bool ProtectTransient(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		std::shared_lock lock(m_mutex);
		// (Not while a map or unmap is changing the range outside the lock: MapBackingOutsideLock.)
		WaitNotBusy(lock, vaddr, size);
		return ProtectMappedUnlocked(vaddr, size, mode);
	}

	bool MapBacking(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                VirtualMemory::Mode               mode,
	                GuestBackingStore::FailureReason* failure_reason = nullptr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (MapOutsideLockEnabled()) {
			return MapBackingOutsideLock(vaddr, size, backing_offset, mode, failure_reason);
		}
#endif
		std::lock_guard lock(m_mutex);
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		const bool replacing = ContainsMappedUnlocked(vaddr, size, MappingType::Backing);
#else
		constexpr bool replacing = false;
		MaterializeParkedUnlocked(vaddr, size);
#endif
		if (!replacing && !ConsumeUnlocked(vaddr, size)) {
			if (failure_reason != nullptr) {
				*failure_reason = GuestBackingStore::FailureReason::AddressRangeUnavailable;
			}
			return false;
		}
		const auto host_mode = VirtualMemory::IsExecute(mode)
		                           ? VirtualMemory::Mode::ExecuteReadWrite
		                           : VirtualMemory::Mode::ReadWrite;
		if (m_backing->MapFixed(vaddr, size, backing_offset, host_mode, failure_reason)) {
			if (!replacing) {
				EXIT_IF(!AddMappedUnlocked(vaddr, size, MappingType::Backing));
			}
			return true;
		}
		if (!replacing) {
			AddFreeUnlocked(vaddr, size);
		}
		return false;
	}

	bool UnmapBacking(uint64_t vaddr, uint64_t size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (MapOutsideLockEnabled()) {
			return UnmapBackingOutsideLock(vaddr, size);
		}
#endif
		std::lock_guard lock(m_mutex);
		if (!ContainsMappedUnlocked(vaddr, size, MappingType::Backing)) {
			return false;
		}
		bool                                       placeholder_preserved = false;
		std::vector<std::pair<uint64_t, uint64_t>> freed;
		if (!m_backing->Unmap(vaddr, size, &placeholder_preserved, &freed)) {
			return false;
		}
		if (!placeholder_preserved) {
			return false;
		}
		EXIT_IF(!RemoveMappedUnlocked(vaddr, size, MappingType::Backing));
		// (The range itself, unless it stayed parked in its view: KYTY_PARK_UNMAPS.)
		for (const auto& [freed_vaddr, freed_size]: freed) {
			AddFreeUnlocked(freed_vaddr, freed_size);
		}
		return true;
	}

	uint64_t ReserveAligned(uint64_t search_addr, uint64_t size, uint64_t alignment) {
		if (size == 0) {
			return 0;
		}
		alignment = std::max<uint64_t>(alignment != 0 ? alignment : PageSize(), PageSize());
		std::lock_guard lock(m_mutex);
		if (search_addr == 0) {
			auto addr = FindFreeAlignedUnlocked(HOST_SYSTEM_MANAGED_MIN,
			                                    HOST_SYSTEM_MANAGED_MAX + 1u, size, alignment);
			return addr != 0 ? addr
			                 : FindFreeAlignedUnlocked(HOST_USER_MIN, HOST_USER_MAX + 1u, size,
			                                           alignment);
		}
		return FindFreeAlignedUnlocked(search_addr, HOST_USER_MAX + 1u, size, alignment);
	}

	uint64_t FindFreeAligned(uint64_t search_start, uint64_t search_end, uint64_t size,
	                         uint64_t alignment) {
		if (size == 0 || alignment == 0) {
			return 0;
		}
		alignment = std::max<uint64_t>(alignment, PageSize());
		std::lock_guard lock(m_mutex);
		return FindFreeAlignedUnlocked(search_start, search_end, size, alignment);
	}

	bool ReserveFixed(uint64_t vaddr, uint64_t size) {
		if (vaddr == 0 || size == 0 || !IsPageAligned(vaddr) || !IsPageAligned(size)) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		return FreeOrParkedUnlocked(vaddr, size);
	}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	bool TestContainsFree(uint64_t vaddr, uint64_t size) {
		std::lock_guard lock(m_mutex);
		return FreeOrParkedUnlocked(vaddr, size);
	}
#endif

	bool ReleaseFree(uint64_t vaddr, uint64_t size) {
		std::lock_guard lock(m_mutex);
		return FreeOrParkedUnlocked(vaddr, size);
	}

	bool Commit(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (MapOutsideLockEnabled()) {
			MaterializeParkedOutsideLock(vaddr, size);
		}
#endif
		std::unique_lock lock(m_mutex);
		WaitNotBusy(lock, vaddr, size);
		MaterializeParkedUnlocked(vaddr, size);
		if (!ConsumeUnlocked(vaddr, size)) {
			return false;
		}
		const auto host_mode = VirtualMemory::IsExecute(mode)
		                           ? VirtualMemory::Mode::ExecuteReadWrite
		                           : VirtualMemory::Mode::ReadWrite;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		auto* ptr = VirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(vaddr), size,
		                          MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
		                          GetProtect(host_mode), nullptr, 0);
		if (ptr == nullptr || reinterpret_cast<uint64_t>(ptr) != vaddr) {
			if (ptr != nullptr) {
				VirtualFree(ptr, 0, MEM_RELEASE);
			}
			AddFreeUnlocked(vaddr, size);
			return false;
		}
		EXIT_IF(!AddMappedUnlocked(vaddr, size, MappingType::Private));
		return true;
#else
		if (VirtualMemory::Protect(vaddr, size, host_mode)) {
			EXIT_IF(!AddMappedUnlocked(vaddr, size, MappingType::Private));
			return true;
		}
		AddFreeUnlocked(vaddr, size);
		return false;
#endif
	}

	bool ReleaseCommitted(uint64_t vaddr, uint64_t size) {
		if (vaddr == 0 || size == 0 || !IsPageAligned(vaddr) || !IsPageAligned(size)) {
			return false;
		}
		std::unique_lock lock(m_mutex);
		WaitNotBusy(lock, vaddr, size);
		if (!ContainsMappedUnlocked(vaddr, size, MappingType::Private)) {
			return false;
		}
		const auto end     = vaddr + size;
		auto       current = vaddr;
		std::vector<std::pair<uint64_t, uint64_t>> regions;
		while (current < end) {
			const auto it       = FindMappedContainingUnlocked(current);
			const auto part_end = std::min(end, it->second.vaddr + it->second.size);
			regions.emplace_back(current, part_end - current);
			current = part_end;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& [region_vaddr, region_size]: regions) {
			EXIT_IF(VirtualFree(reinterpret_cast<void*>(region_vaddr), region_size,
			                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0);
		}
		// ReleaseCommitted also publishes the aggregate span as one logical free range.
		EXIT_IF(regions.size() > 1 && !CoalescePlaceholders(vaddr, size));
#else
		for (const auto& [region_vaddr, region_size]: regions) {
			RestorePlaceholder(region_vaddr, region_size);
		}
#endif
		EXIT_IF(!RemoveMappedUnlocked(vaddr, size, MappingType::Private));
		AddFreeUnlocked(vaddr, size);
		return true;
	}

	bool SelfTest() {
		const auto addr = ReserveAligned(0, PageSize(), PageSize());
		if (addr == 0) {
			LOGF_COLOR(Log::Color::Red,
			           "\t guest address-space self-test: 16 KiB range unavailable\n");
			return false;
		}
		if (!Commit(addr, PageSize(), VirtualMemory::Mode::ReadWrite)) {
			LOGF_COLOR(Log::Color::Red, "\t guest address-space self-test: commit failed\n");
			return false;
		}

		auto* ptr = reinterpret_cast<uint64_t*>(addr);
		*ptr      = 0x4b59545950484f4cull; // "KYTYPHOL"

		if (!ReleaseCommitted(addr, PageSize())) {
			LOGF_COLOR(Log::Color::Red,
			           "\t guest address-space self-test: release-committed failed\n");
			return false;
		}

		LOGF_COLOR(Log::Color::Green, "\t guest address-space self-test: ok\n");
		return true;
	}

private:
	enum class MappingType { Backing, Private };

	struct MappedRegion {
		uint64_t    vaddr = 0;
		uint64_t    size  = 0;
		MappingType type  = MappingType::Private;
	};

	static constexpr uint64_t PageSize() { return 0x4000; }

	static uint64_t AlignUp(uint64_t value, uint64_t alignment) {
		if (alignment == 0) {
			return 0;
		}
		const auto remainder = value % alignment;
		const auto increment = (remainder != 0 ? alignment - remainder : 0);
		return increment <= UINT64_MAX - value ? value + increment : 0;
	}

	static uint64_t AlignDown(uint64_t value, uint64_t alignment) {
		return value & ~(alignment - 1u);
	}

	[[nodiscard]] static bool IsPageAligned(uint64_t value) {
		return (value & (PageSize() - 1u)) == 0;
	}

	[[nodiscard]] bool IsGranularityAligned(uint64_t value) const {
		return (value & (m_granularity - 1u)) == 0;
	}

	bool ProtectMappedUnlocked(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		const auto end = vaddr + size;
		auto       it  = m_mapped.upper_bound(vaddr);
		if (it != m_mapped.begin()) {
			--it;
		}
		for (; it != m_mapped.end() && it->second.vaddr < end; ++it) {
			const auto& region      = it->second;
			const auto  range_start = std::max(vaddr, region.vaddr);
			const auto  range_end   = std::min(end, region.vaddr + region.size);
			if (range_start < range_end &&
			    !VirtualMemory::Protect(range_start, range_end - range_start, mode)) {
				return false;
			}
		}
		return true;
	}

	// Whether a backing map or unmap is changing a range that overlaps [vaddr, vaddr + size) outside the lock
	// (MapBackingOutsideLock, UnmapBackingOutsideLock). Called with m_mutex held.
	[[nodiscard]] bool BusyUnlocked(uint64_t vaddr, uint64_t size) const {
		for (const auto& [begin, end]: m_busy) {
			if (vaddr < end && begin < vaddr + size) {
				return true;
			}
		}
		return false;
	}

	// Returns once no map or unmap is changing [vaddr, vaddr + size) (the lock is released while it waits).
	template <typename Lock>
	void WaitNotBusy(Lock& lock, uint64_t vaddr, uint64_t size) {
		if (!BusyUnlocked(vaddr, size)) {
			return;
		}
		m_busy_done.wait(lock, [&] { return !BusyUnlocked(vaddr, size); });
	}

	void EraseBusyUnlocked(uint64_t begin, uint64_t end) {
		const auto it =
		    std::find(m_busy.begin(), m_busy.end(), std::pair<uint64_t, uint64_t> {begin, end});
		EXIT_IF(it == m_busy.end());
		m_busy.erase(it);
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// A free entry consumed with its placeholder split later (SplitPlaceholders, outside the lock).
	struct FreeSplit {
		uint64_t base = 0;
		uint64_t end  = 0; // the free entry
	};

	// ConsumeUnlocked's free-list change without its system calls: the remainders become free entries now.
	bool ConsumeDeferredUnlocked(uint64_t vaddr, uint64_t size, FreeSplit* split) {
		const auto it = FindContaining(vaddr, size);
		if (it == m_free.end()) {
			return false;
		}
		split->base = it->first;
		split->end  = it->first + it->second;
		m_free.erase(it);
		if (split->base != vaddr) {
			m_free.emplace(split->base, vaddr - split->base);
		}
		if (vaddr + size != split->end) {
			m_free.emplace(vaddr + size, split->end - (vaddr + size));
		}
		return true;
	}

	// The placeholder of a consumed free entry split around [vaddr, vaddr + size) (SplitUnlocked's calls).
	// False when a split failed (RestoreSplitUnlocked).
	static bool SplitPlaceholders(const FreeSplit& split, uint64_t vaddr, uint64_t size) {
		if (split.base != vaddr &&
		    VirtualFree(reinterpret_cast<void*>(split.base), vaddr - split.base,
		                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0) {
			LOGF_COLOR(Log::Color::Red,
			           "\t guest placeholder split failed: base=0x%016" PRIx64 " size=0x%016" PRIx64
			           " requested=0x%016" PRIx64 " request_size=0x%016" PRIx64 " error=0x%08" PRIx32 "\n",
			           split.base, split.end - split.base, vaddr, size,
			           static_cast<uint32_t>(GetLastError()));
			return false;
		}
		if (vaddr + size != split.end &&
		    VirtualFree(reinterpret_cast<void*>(vaddr), size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0) {
			LOGF_COLOR(Log::Color::Red,
			           "\t guest placeholder split failed: base=0x%016" PRIx64 " size=0x%016" PRIx64
			           " requested=0x%016" PRIx64 " request_size=0x%016" PRIx64 " error=0x%08" PRIx32 "\n",
			           vaddr, split.end - vaddr, vaddr, size, static_cast<uint32_t>(GetLastError()));
			return false;
		}
		return true;
	}

	// A map that failed gives its free entry back whole, its placeholders (the ones split) made one again.
	void RestoreSplitUnlocked(const FreeSplit& split) {
		m_free.erase(m_free.lower_bound(split.base), m_free.lower_bound(split.end));
		// (Coalescing one placeholder fails: nothing was split then.)
		(void)CoalescePlaceholders(split.base, split.end - split.base);
		m_free.emplace(split.base, split.end - split.base);
	}

	// MapBacking with its system calls outside the lock (KYTY_MAP_OUTSIDE_LOCK): the free-list change and the
	// bookkeeping of the mapping under the lock, the range marked busy in between. Guest maps and unmaps run one at a
	// time (g_memory_operation_mutex): another change over the range is only waited for here, and what waits for
	// the busy range is the protection of the GPU trackers' pages and the readers of the mappings.
	bool MapBackingOutsideLock(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                           VirtualMemory::Mode               mode,
	                           GuestBackingStore::FailureReason* failure_reason) {
		MaterializeParkedOutsideLock(vaddr, size);
		FreeSplit split;
		{
			std::unique_lock lock(m_mutex);
			WaitNotBusy(lock, vaddr, size);
			if (!ConsumeDeferredUnlocked(vaddr, size, &split)) {
				if (failure_reason != nullptr) {
					*failure_reason = GuestBackingStore::FailureReason::AddressRangeUnavailable;
				}
				return false;
			}
			m_busy.emplace_back(vaddr, vaddr + size);
		}
		const auto host_mode = VirtualMemory::IsExecute(mode) ? VirtualMemory::Mode::ExecuteReadWrite
		                                                      : VirtualMemory::Mode::ReadWrite;
		const bool split_done = SplitPlaceholders(split, vaddr, size);
		const bool mapped =
		    split_done && m_backing->MapFixed(vaddr, size, backing_offset, host_mode, failure_reason);
		if (!split_done && failure_reason != nullptr) {
			*failure_reason = GuestBackingStore::FailureReason::AddressRangeUnavailable;
		}
		{
			std::unique_lock lock(m_mutex);
			if (mapped) {
				EXIT_IF(!AddMappedUnlocked(vaddr, size, MappingType::Backing));
			} else {
				RestoreSplitUnlocked(split);
			}
			EraseBusyUnlocked(vaddr, vaddr + size);
		}
		m_busy_done.notify_all();
		return mapped;
	}

	// UnmapBacking with its system calls outside the lock (KYTY_MAP_OUTSIDE_LOCK): the views over the range are busy
	// (an unmap of part of a view unmaps the view whole to map the rest again: the pages around the range must
	// keep their protections meanwhile).
	bool UnmapBackingOutsideLock(uint64_t vaddr, uint64_t size) {
		uint64_t busy_begin = vaddr;
		uint64_t busy_end   = vaddr + size;
		{
			std::unique_lock lock(m_mutex);
			for (;;) {
				if (!ContainsMappedUnlocked(vaddr, size, MappingType::Backing)) {
					return false;
				}
				busy_begin = vaddr;
				busy_end   = vaddr + size;
				m_backing->ViewSpan(vaddr, size, &busy_begin, &busy_end);
				if (!BusyUnlocked(busy_begin, busy_end - busy_begin)) {
					m_busy.emplace_back(busy_begin, busy_end);
					break;
				}
				m_busy_done.wait(lock);
			}
		}
		bool                                       placeholder_preserved = false;
		std::vector<std::pair<uint64_t, uint64_t>> freed;
		const bool                                 done =
		    m_backing->Unmap(vaddr, size, &placeholder_preserved, &freed) && placeholder_preserved;
		{
			std::unique_lock lock(m_mutex);
			if (done) {
				EXIT_IF(!RemoveMappedUnlocked(vaddr, size, MappingType::Backing));
				for (const auto& [freed_vaddr, freed_size]: freed) {
					AddFreeUnlocked(freed_vaddr, freed_size);
				}
			}
			EraseBusyUnlocked(busy_begin, busy_end);
		}
		m_busy_done.notify_all();
		return done;
	}

	// MaterializeParkedUnlocked without the lock held during its system calls: the views are busy meanwhile.
	void MaterializeParkedOutsideLock(uint64_t vaddr, uint64_t size) {
		uint64_t begin = UINT64_MAX;
		uint64_t end   = 0;
		if (!m_backing->ParkedViews(vaddr, size, &begin, &end)) {
			return;
		}
		{
			std::unique_lock lock(m_mutex);
			WaitNotBusy(lock, begin, end - begin);
			m_busy.emplace_back(begin, end);
		}
		std::vector<std::pair<uint64_t, uint64_t>> freed;
		m_backing->Materialize(vaddr, size, &freed);
		{
			std::unique_lock lock(m_mutex);
			for (const auto& [freed_vaddr, freed_size]: freed) {
				AddFreeUnlocked(freed_vaddr, freed_size);
			}
			EraseBusyUnlocked(begin, end);
		}
		m_busy_done.notify_all();
	}
#endif

	// Whether [vaddr, vaddr + size) is free: free placeholders, or ranges parked in views
	// (GuestBackingStore::UnmapOne, KYTY_PARK_UNMAPS), which a map or commit there makes free
	// placeholders first (MaterializeParkedUnlocked).
	bool FreeOrParkedUnlocked(uint64_t vaddr, uint64_t size) {
		if (FindContaining(vaddr, size) != m_free.end()) {
			return true;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (!ParkUnmapsEnabled()) {
			return false;
		}
		const auto end = vaddr + size;
		for (auto at = vaddr; at < end;) {
			if (const auto it = FindContaining(at, 1); it != m_free.end()) {
				at = it->first + it->second;
			} else if (const auto parked_end = m_backing->ParkedEnd(at); parked_end != 0) {
				at = parked_end;
			} else {
				return false;
			}
		}
		return true;
#else
		return false;
#endif
	}

	// The ranges parked in views over [vaddr, vaddr + size) made free placeholders, before a map
	// or commit there: their views are split for real (GuestBackingStore::Materialize).
	void MaterializeParkedUnlocked([[maybe_unused]] uint64_t vaddr, [[maybe_unused]] uint64_t size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (!m_backing->HasParked(vaddr, size)) {
			return;
		}
		std::vector<std::pair<uint64_t, uint64_t>> freed;
		m_backing->Materialize(vaddr, size, &freed);
		for (const auto& [freed_vaddr, freed_size]: freed) {
			AddFreeUnlocked(freed_vaddr, freed_size);
		}
#endif
	}

	bool ConsumeUnlocked(uint64_t vaddr, uint64_t size) {
		auto it = FindContaining(vaddr, size);
		if (it == m_free.end() || !SplitUnlocked(it, vaddr, size)) {
			return false;
		}
		it = m_free.find(vaddr);
		if (it == m_free.end() || it->second != size) {
			return false;
		}
		m_free.erase(it);
		return true;
	}

	std::map<uint64_t, MappedRegion>::iterator FindMappedContainingUnlocked(uint64_t vaddr) {
		auto next = m_mapped.upper_bound(vaddr);
		if (next == m_mapped.begin()) {
			return m_mapped.end();
		}
		auto it = std::prev(next);
		return vaddr < it->second.vaddr + it->second.size ? it : m_mapped.end();
	}

	bool ContainsMappedUnlocked(uint64_t vaddr, uint64_t size, MappingType type) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		const auto end     = vaddr + size;
		auto       current = vaddr;
		while (current < end) {
			const auto it = FindMappedContainingUnlocked(current);
			if (it == m_mapped.end() || it->second.type != type) {
				return false;
			}
			current = std::min(end, it->second.vaddr + it->second.size);
		}
		return true;
	}

	bool AddMappedUnlocked(uint64_t vaddr, uint64_t size, MappingType type) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		const auto end  = vaddr + size;
		auto       next = m_mapped.lower_bound(vaddr);
		if (next != m_mapped.end() && next->second.vaddr < end) {
			return false;
		}
		if (next != m_mapped.begin()) {
			const auto& previous = std::prev(next)->second;
			if (previous.vaddr + previous.size > vaddr) {
				return false;
			}
		}
		return m_mapped.emplace(vaddr, MappedRegion {vaddr, size, type}).second;
	}

	bool RemoveMappedUnlocked(uint64_t vaddr, uint64_t size, MappingType type) {
		if (!ContainsMappedUnlocked(vaddr, size, type)) {
			return false;
		}
		const auto end     = vaddr + size;
		auto       current = vaddr;
		while (current < end) {
			auto       it       = FindMappedContainingUnlocked(current);
			const auto original = it->second;
			const auto part_end = std::min(end, original.vaddr + original.size);
			m_mapped.erase(it);
			if (original.vaddr < current) {
				const auto left_size = current - original.vaddr;
				EXIT_IF(!AddMappedUnlocked(original.vaddr, left_size, original.type));
			}
			if (part_end < original.vaddr + original.size) {
				const auto right_size = original.vaddr + original.size - part_end;
				EXIT_IF(!AddMappedUnlocked(part_end, right_size, original.type));
			}
			current = part_end;
		}
		return true;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static DWORD GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PAGE_READONLY;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PAGE_READWRITE;
			case VirtualMemory::Mode::Execute: return PAGE_EXECUTE;
			case VirtualMemory::Mode::ExecuteRead: return PAGE_EXECUTE_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PAGE_EXECUTE_READWRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PAGE_NOACCESS;
		}
	}
#endif

	void ReserveGuestRegions() {
		const std::array<std::pair<uint64_t, uint64_t>, 3> regions {{
		    {HOST_SYSTEM_MANAGED_MIN, HOST_SYSTEM_RESERVED_MAX + 1u},
		    {HOST_USER_MIN, HOST_USER_MAX + 1u},
		    {kExtendedMemoryBase, kExtendedMemoryBase + kExtendedMemorySize},
		}};

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		constexpr uint64_t MinimumRegionSize = 0x1000000;
		const auto         process           = GetCurrentProcess();
		for (const auto& [guest_start, guest_end]: regions) {
			auto next_addr = guest_start;
			while (next_addr < guest_end) {
				MEMORY_BASIC_INFORMATION info {};
				if (VirtualQuery(reinterpret_cast<const void*>(next_addr), &info, sizeof(info)) == 0) {
					EXIT("failed to inspect guest address space at 0x%016" PRIx64 "\n", next_addr);
				}

				const auto region_base = reinterpret_cast<uint64_t>(info.BaseAddress);
				const auto region_end = std::min<uint64_t>(
				    guest_end, region_base + static_cast<uint64_t>(info.RegionSize));
				const auto reserve_start =
				    AlignUp(std::max(region_base, guest_start), m_granularity);
				const auto reserve_end = AlignDown(region_end, m_granularity);

				if (info.State == MEM_FREE && reserve_end > reserve_start &&
				    reserve_end - reserve_start > MinimumRegionSize) {
					auto* ptr = VirtualAlloc2(
					    process, reinterpret_cast<void*>(reserve_start), reserve_end - reserve_start,
					    MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
					if (reinterpret_cast<uint64_t>(ptr) != reserve_start) {
						EXIT("failed to reserve guest address space at 0x%016" PRIx64
						     ", size 0x%016" PRIx64 "\n",
						     reserve_start, reserve_end - reserve_start);
					}
					AddFreeUnlocked(reserve_start, reserve_end - reserve_start);
					m_owned.emplace_back(reserve_start, reserve_end - reserve_start);
				}

				if (region_end <= next_addr) {
					EXIT("guest address-space scan did not advance at 0x%016" PRIx64 "\n", next_addr);
				}
				next_addr = region_end;
			}
		}
#else
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
		std::vector<std::pair<uint64_t, uint64_t>> occupied;
		FILE*                                      maps = fopen("/proc/self/maps", "r");
		EXIT_IF(maps == nullptr);

		char line[512];
		while (fgets(line, sizeof(line), maps) != nullptr) {
			unsigned long long mapping_start = 0;
			unsigned long long mapping_end   = 0;
			if (sscanf(line, "%llx-%llx", &mapping_start, &mapping_end) == 2) {
				occupied.emplace_back(static_cast<uint64_t>(mapping_start),
				                      static_cast<uint64_t>(mapping_end));
			}
		}
		fclose(maps);

		auto reserve_range = [this](uint64_t start, uint64_t end) {
			start = AlignUp(start, PageSize());
			end   = AlignDown(end, PageSize());
			if (start == 0 || end <= start) {
				return;
			}

			const auto size = end - start;
			int        flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#if defined(KYTY_LINKED_GUEST_ADDRESS_SPACE)
			flags |= MAP_FIXED;
#elif defined(MAP_FIXED_NOREPLACE)
			flags |= MAP_FIXED_NOREPLACE;
#endif

			void* ptr = mmap(reinterpret_cast<void*>(start), size, PROT_NONE, flags, -1, 0);
			if (ptr == MAP_FAILED || reinterpret_cast<uint64_t>(ptr) != start) {
				if (ptr != MAP_FAILED) {
					munmap(ptr, size);
				}
				return;
			}

			AddFreeUnlocked(start, size);
			m_owned.emplace_back(start, size);
		};

		for (const auto& [region_start, region_end]: regions) {
			auto current = region_start;

			for (const auto& [mapping_start, mapping_end]: occupied) {
				if (mapping_end <= current) {
					continue;
				}
				if (mapping_start >= region_end) {
					break;
				}

				if (mapping_start > current) {
					reserve_range(current, std::min(mapping_start, region_end));
				}

				current = std::max(current, mapping_end);
				if (current >= region_end) {
					break;
				}
			}

			if (current < region_end) {
				reserve_range(current, region_end);
			}
		}
#else
		for (const auto& [start, end]: regions) {
			int flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#if defined(KYTY_LINKED_GUEST_ADDRESS_SPACE)
			flags |= MAP_FIXED;
#elif defined(MAP_FIXED_NOREPLACE)
			flags |= MAP_FIXED_NOREPLACE;
#endif
			void* ptr = mmap(reinterpret_cast<void*>(start), end - start, PROT_NONE, flags, -1, 0);
			if (ptr == MAP_FAILED || reinterpret_cast<uint64_t>(ptr) != start) {
				if (ptr != MAP_FAILED) {
					munmap(ptr, end - start);
				}
				EXIT("failed to reserve guest address space at 0x%016" PRIx64
				     ", size 0x%016" PRIx64 "\n",
				     start, end - start);
			}
			AddFreeUnlocked(start, end - start);
			m_owned.emplace_back(start, end - start);
		}
#endif
#endif
	}

	uint64_t FindFreeAlignedUnlocked(uint64_t search_start, uint64_t search_end, uint64_t size,
	                                 uint64_t alignment) {
		if (search_start >= search_end || size > search_end - search_start) {
			return 0;
		}
		for (const auto& [base, range_size]: m_free) {
			const auto candidate = AlignUp(std::max(base, search_start), alignment);
			if (candidate < search_end && size <= search_end - candidate &&
			    GuestFreeRangeContains(base, range_size, candidate, size)) {
				return candidate;
			}
		}
		return 0;
	}

	std::map<uint64_t, uint64_t>::iterator FindContaining(uint64_t vaddr, uint64_t size) {
		auto next = m_free.upper_bound(vaddr);
		if (next == m_free.begin()) {
			return m_free.end();
		}
		auto it  = std::prev(next);
		auto end = it->first + it->second;
		return (vaddr >= it->first && vaddr + size <= end ? it : m_free.end());
	}

	bool SplitUnlocked(std::map<uint64_t, uint64_t>::iterator it, uint64_t vaddr, uint64_t size) {
		const auto base = it->first;
		const auto end  = it->first + it->second;
		if (base == vaddr && end == vaddr + size) {
			return true;
		}

		if (base != vaddr) {
			const auto left_size = vaddr - base;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (VirtualFree(reinterpret_cast<void*>(base), left_size,
			                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0) {
				LOGF_COLOR(Log::Color::Red,
				           "\t guest placeholder split failed: base=0x%016" PRIx64
				           " size=0x%016" PRIx64 " requested=0x%016" PRIx64
				           " request_size=0x%016" PRIx64 " error=0x%08" PRIx32 "\n",
				           base, end - base, vaddr, size, static_cast<uint32_t>(GetLastError()));
				return false;
			}
#endif
			it->second = left_size;
			it         = m_free.emplace(vaddr, end - vaddr).first;
		}

		if (it->second != size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (VirtualFree(reinterpret_cast<void*>(it->first), size,
			                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0) {
				LOGF_COLOR(Log::Color::Red,
				           "\t guest placeholder split failed: base=0x%016" PRIx64
				           " size=0x%016" PRIx64 " requested=0x%016" PRIx64
				           " request_size=0x%016" PRIx64 " error=0x%08" PRIx32 "\n",
				           it->first, it->second, vaddr, size,
				           static_cast<uint32_t>(GetLastError()));
				return false;
			}
#endif
			const auto right_start = it->first + size;
			const auto right_size  = it->second - size;
			it->second             = size;
			m_free.emplace(right_start, right_size);
		}

		return true;
	}

	void AddFreeUnlocked(uint64_t vaddr, uint64_t size) {
		if (vaddr == 0 || size == 0) {
			return;
		}
		auto [it, inserted] = m_free.emplace(vaddr, size);
		if (!inserted) {
			it->second = size;
		}

		auto base = it->first;
		auto end  = it->first + it->second;

		auto first = it;
		while (first != m_free.begin()) {
			auto prev = std::prev(first);
			if (prev->first + prev->second != base) {
				break;
			}
			base  = prev->first;
			first = prev;
		}

		auto last = std::next(it);
		while (last != m_free.end() && end == last->first) {
			end  = last->first + last->second;
			last = std::next(last);
		}

		if (base != it->first || end != it->first + it->second) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (CoalescePlaceholders(base, end - base)) {
				m_free.erase(first, last);
				m_free.emplace(base, end - base);
			}
#else
			m_free.erase(first, last);
			m_free.emplace(base, end - base);
#endif
		}
	}

	void RestorePlaceholder(uint64_t vaddr, uint64_t size) {
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
		// The address remains owned for the process lifetime; MAP_FIXED only replaces one of our
		// mappings and never acquires a new guest address.
		void* ptr = mmap(reinterpret_cast<void*>(vaddr), size, PROT_NONE,
		                 MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
		EXIT_IF(ptr == MAP_FAILED || reinterpret_cast<uint64_t>(ptr) != vaddr);
#else
		(void)vaddr;
		(void)size;
#endif
	}

	std::shared_mutex                          m_mutex;
	// Ranges a backing map or unmap is changing outside m_mutex (KYTY_MAP_OUTSIDE_LOCK), under it; what overlaps one
	// waits for it (WaitNotBusy).
	std::vector<std::pair<uint64_t, uint64_t>> m_busy;
	std::condition_variable_any                m_busy_done;
	std::map<uint64_t, uint64_t>               m_free;
	std::map<uint64_t, MappedRegion>            m_mapped;
	std::vector<std::pair<uint64_t, uint64_t>> m_owned;
	uint64_t                                   m_granularity = 0x10000;
	GuestBackingStore*                         m_backing     = nullptr;
};
