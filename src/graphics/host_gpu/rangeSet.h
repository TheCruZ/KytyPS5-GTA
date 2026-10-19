#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_

#include "common/assert.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>

namespace Libs::Graphics {

class RangeSet final {
public:
	// Queries keep a hint of the range they found (see HintCovers()): only for a set whose users serialize all their
	// accesses, queries included.
	void AllowFastPath() { m_fast = true; }

	void Add(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		if (HintCovers(address, end)) {
			return;
		}
		// Written ranges are added again for every draw that binds them: extend a range that
		// begins at or before the address in place, and allocate a node only for a new range.
		auto it = m_ranges.upper_bound(address);
		if (it != m_ranges.begin()) {
			const auto previous = std::prev(it);
			if (previous->second >= address) {
				if (previous->second >= end) {
					return;
				}
				uint64_t last = end;
				while (it != m_ranges.end() && it->first <= last) {
					last = std::max(last, it->second);
					it   = m_ranges.erase(it);
				}
				previous->second = last;
				return;
			}
		}
		uint64_t last = end;
		while (it != m_ranges.end() && it->first <= last) {
			last = std::max(last, it->second);
			it   = m_ranges.erase(it);
		}
		m_ranges.emplace_hint(it, address, last);
	}

	void Subtract(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		m_hint_end     = 0;
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			it = std::prev(it);
		}
		while (it != m_ranges.end() && it->first < end) {
			const auto begin = it->first;
			const auto last  = it->second;
			it               = m_ranges.erase(it);
			m_removals++;
			if (begin < address) {
				m_ranges.emplace(begin, address);
			}
			if (last > end) {
				m_ranges.emplace(end, last);
				break;
			}
		}
	}

	void Clear() {
		m_ranges.clear();
		m_hint_end = 0;
		m_removals++;
	}

	// Advances whenever a Subtract() or Clear() took bytes out of the set: a range it held whole at some value holds
	// whole until it changes (Add() only grows the ranges).
	[[nodiscard]] uint64_t RemovalEpoch() const { return m_removals; }

	template <typename Func>
	void ForEach(Func&& func) const {
		for (const auto& [begin, end]: m_ranges) {
			func(begin, end);
		}
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			return true;
		}
		return it != m_ranges.end() && it->first < end;
	}

	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		if (HintCovers(address, end)) {
			return true;
		}
		auto it = m_ranges.upper_bound(address);
		if (it == m_ranges.begin()) {
			return false;
		}
		--it;
		if (it->first <= address && it->second >= end) {
			m_hint_begin = it->first;
			m_hint_end   = it->second;
			return true;
		}
		return false;
	}

	template <typename Func>
	void ForEachInRange(uint64_t address, uint64_t size, Func&& func) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (; it != m_ranges.end() && it->first < end; ++it) {
			const auto begin = std::max(address, it->first);
			const auto last  = std::min(end, it->second);
			if (begin < last) {
				func(begin, last);
			}
		}
	}

	[[nodiscard]] bool Empty() const { return m_ranges.empty(); }

private:
	// The range the last Contains() found (chenxiao07's KYTY_RANGE_SET_FAST): buffers are queried again and again
	// for the same ranges. Add() only grows the ranges, so the hint holds until Subtract() or Clear().
	static bool FastEnabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_RANGE_SET_FAST");
			return value == nullptr || value[0] != '0';
		}();
		return enabled;
	}
	[[nodiscard]] bool HintCovers(uint64_t address, uint64_t end) const {
		return m_fast && m_hint_begin <= address && end <= m_hint_end && FastEnabled();
	}

	static uint64_t End(uint64_t address, uint64_t size) {
		if (size == 0 || size > UINT64_MAX - address) {
			EXIT("invalid range-set address or size\n");
		}
		return address + size;
	}

	std::map<uint64_t, uint64_t> m_ranges;
	uint64_t                     m_removals = 0;
	mutable uint64_t             m_hint_begin = 0;
	mutable uint64_t             m_hint_end   = 0;
	bool                         m_fast       = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
