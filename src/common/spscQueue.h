#ifndef KYTY_COMMON_SPSCQUEUE_H_
#define KYTY_COMMON_SPSCQUEUE_H_

#include "common/assert.h"
#include "common/common.h"
#include "common/syncAcct.h"

#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Common {

// Local debug: spin iterations and kernel waits per stage (0 other, 1 CP, 2 exec, 3 record,
// 4 resolve, 5 host copies).
namespace WaitStats {
inline std::atomic_uint64_t     spins[6];
inline std::atomic_uint64_t     kernel_waits[6];
inline std::atomic_uint64_t     gpu_waits[6];
inline thread_local int         t_stage = 0;
inline void Spins(uint64_t n) noexcept { spins[t_stage].fetch_add(n, std::memory_order_relaxed); }
inline void Kernel() noexcept { kernel_waits[t_stage].fetch_add(1, std::memory_order_relaxed); }
} // namespace WaitStats

inline void SpinPause() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#endif
}

// How long a pipeline thread spins on an empty or full handoff before it falls back to a kernel
// wait: KYTY_GPU_SPIN_US microseconds, 2 ms by default. Within a busy frame the stages never
// wait that long, so they never enter the kernel; an idle stage stops burning its core.
inline std::chrono::nanoseconds SpinBudget() noexcept {
	static const auto budget = [] {
		uint64_t micros = 2000;
		if (const char* value = std::getenv("KYTY_GPU_SPIN_US"); value != nullptr) {
			micros = std::strtoull(value, nullptr, 10);
		}
		return std::chrono::nanoseconds(micros * 1000u);
	}();
	return budget;
}

// Local debugging (KYTY_X_LONG_SPIN): a spin budget of this thread's own (nanoseconds; 0 = the
// global one).
inline thread_local int64_t t_spin_budget_ns = 0;

// Busy-waiting with an exponential _mm_pause backoff (1, 2, 4 ... 64 pauses per round) that
// gives up once the spin budget is spent. The clock is read every few rounds only.
class SpinWait {
public:
	SpinWait() noexcept
	    : m_budget(t_spin_budget_ns > 0 ? std::chrono::nanoseconds(t_spin_budget_ns) : SpinBudget()) {}
	~SpinWait() { WaitStats::Spins(m_iterations); m_iterations = 0; }

	// Pauses once more; false once the budget is spent and the caller should sleep.
	bool Spin() noexcept {
		for (uint32_t i = 0; i < m_pauses; i++) {
			SpinPause();
		}
		m_iterations += m_pauses;
		if (m_pauses < MaxPauses) {
			m_pauses <<= 1u;
		}
		if ((++m_rounds & ClockMask) != 0) {
			return true;
		}
		const auto now = std::chrono::steady_clock::now();
		if (m_rounds == ClockMask + 1) {
			m_start = now;
			return true;
		}
		return now - m_start < m_budget;
	}

	[[nodiscard]] uint64_t Iterations() const noexcept { return m_iterations; }

private:
	static constexpr uint32_t MaxPauses = 64;
	static constexpr uint32_t ClockMask = 15;

	std::chrono::nanoseconds              m_budget;
	std::chrono::steady_clock::time_point m_start {};
	uint32_t                              m_pauses     = 1;
	uint32_t                              m_rounds     = 0;
	uint64_t                              m_iterations = 0;
};

// A monotonically increasing counter one thread advances and one other thread waits on.
//
// The waiter spins (SpinWait) and only then sleeps. A wake-up is only issued when the value
// reaches the sleeping thread's target. WaitFor() stores the target and then reads the value,
// seq_cst; Notify() issues a seq_cst fence and then reads the target. Either the waiter sees a
// value published before the fence, or Notify() sees the target and wakes the waiter;
// atomic::wait() also returns at once when the value already differs from the one the waiter
// read. Publish() alone is a plain release store: a sleeper it satisfies wakes at the next
// Notify().
class ProgressCounter {
public:
	ProgressCounter() = default;
	KYTY_CLASS_NO_COPY(ProgressCounter);

	// Debug (KYTY_SYNC_ACCT): names the waits on this counter.
	void SetDebugName(const char* name) {
		if (SyncAcct::Enabled()) {
			m_site_spin  = SyncAcct::RegisterSite((std::string(name) + " wait spin").c_str(), SyncAcct::Kind::Spin);
			m_site_block = SyncAcct::RegisterSite((std::string(name) + " wait block").c_str(), SyncAcct::Kind::Block);
		}
	}

	[[nodiscard]] uint64_t Load() const noexcept { return m_value.load(std::memory_order_acquire); }

	void Publish(uint64_t value) noexcept { m_value.store(value, std::memory_order_release); }

	void Notify() noexcept {
		std::atomic_thread_fence(std::memory_order_seq_cst);
		if (m_wait_target.load(std::memory_order_relaxed) <=
		    m_value.load(std::memory_order_relaxed)) {
			m_value.notify_all();
		}
	}

	void Advance(uint64_t value) noexcept {
		Publish(value);
		Notify();
	}

	// Waits until the counter reaches `target`; the wait acquires every write the advancing
	// thread made before it reached `target`.
	void WaitFor(uint64_t target) noexcept {
		const auto start = m_value.load(std::memory_order_acquire);
		if (start >= target) {
			return;
		}
		SyncAcct::Scope spin_scope(m_site_spin != 0 ? m_site_spin : DefaultSite(false), target, target - start);
		SpinWait spin;
		while (spin.Spin()) {
			if (m_value.load(std::memory_order_acquire) >= target) {
				return;
			}
		}
		spin_scope.End();
		SyncAcct::Scope block_scope(m_site_block != 0 ? m_site_block : DefaultSite(true), target, target - start);
		m_wait_target.store(target, std::memory_order_seq_cst);
		for (;;) {
			const auto value = m_value.load(std::memory_order_seq_cst);
			if (value >= target) {
				break;
			}
			WaitStats::Kernel();
			m_value.wait(value, std::memory_order_seq_cst);
		}
		m_wait_target.store(NoTarget, std::memory_order_relaxed);
	}

private:
	static constexpr uint64_t NoTarget = ~uint64_t {0};
	static uint16_t DefaultSite(bool block) {
		if (!SyncAcct::Enabled()) {
			return 0;
		}
		static const uint16_t spin_site  = SyncAcct::RegisterSite("progress? wait spin", SyncAcct::Kind::Spin);
		static const uint16_t block_site = SyncAcct::RegisterSite("progress? wait block", SyncAcct::Kind::Block);
		return block ? block_site : spin_site;
	}
	uint16_t m_site_spin  = 0;
	uint16_t m_site_block = 0;

	alignas(64) std::atomic_uint64_t m_value {0};
	alignas(64) std::atomic_uint64_t m_wait_target {NoTarget};
};

// A bounded lock-free single-producer single-consumer ring. Push() waits while the ring is full
// and Pop() waits while it is empty: they spin (SpinWait) and only then sleep.
//
// The producer constructs an element in its slot and then publishes it with a release store of
// m_tail; the consumer acquires m_tail before it reads the slot. The consumer moves the element
// out, destroys it and then releases the slot with a release store of m_head; the producer
// acquires m_head before it reuses the slot. The indices live on separate cache lines, and each
// side keeps a cached copy of the other side's index so it only reads the shared line when the
// cached one says the ring is full or empty.
//
// A side that sleeps announces itself and then re-reads the index it waits on, seq_cst. The
// producer looks for a sleeping consumer behind a seq_cst fence after every push. The consumer
// does so every CheckInterval pops and before it sleeps itself: a producer only sleeps on a full
// ring, so a consumer that keeps popping reaches a check soon, and one that stops popping is
// about to sleep. Either way one side sees the other.
template <typename T>
class SpscQueue {
public:
	explicit SpscQueue(size_t capacity, const char* debug_name = "spsc?")
	    : m_capacity(std::bit_ceil(capacity)), m_mask(m_capacity - 1),
	      m_slots(std::make_unique<Slot[]>(m_capacity)) {
		EXIT_IF(capacity == 0);
		if (SyncAcct::Enabled()) {
			const std::string name(debug_name);
			m_site_pop_spin   = SyncAcct::RegisterSite((name + " pop spin").c_str(), SyncAcct::Kind::Spin);
			m_site_pop_block  = SyncAcct::RegisterSite((name + " pop block").c_str(), SyncAcct::Kind::Block);
			m_site_push_spin  = SyncAcct::RegisterSite((name + " push spin").c_str(), SyncAcct::Kind::Spin);
			m_site_push_block = SyncAcct::RegisterSite((name + " push block").c_str(), SyncAcct::Kind::Block);
		}
	}
	~SpscQueue() {
		const auto tail = m_tail.load(std::memory_order_acquire);
		for (auto head = m_head.load(std::memory_order_acquire); head != tail; head++) {
			Element(head)->~T();
		}
	}
	KYTY_CLASS_NO_COPY(SpscQueue);

	void Push(T&& value) {
		const auto tail = m_tail.load(std::memory_order_relaxed);
		if (tail - m_producer_head >= m_capacity) {
			m_producer_head = m_head.load(std::memory_order_acquire);
			if (tail - m_producer_head >= m_capacity) {
				WaitForSpace(tail);
			}
		}
		Construct(tail, std::move(value));
	}

	// Returns false (and keeps `value`) when the ring is full.
	bool TryPush(T& value) {
		const auto tail = m_tail.load(std::memory_order_relaxed);
		if (tail - m_producer_head >= m_capacity) {
			m_producer_head = m_head.load(std::memory_order_acquire);
			if (tail - m_producer_head >= m_capacity) {
				return false;
			}
		}
		Construct(tail, std::move(value));
		return true;
	}

	// Returns nothing once Stop() was called and the ring is empty.
	[[nodiscard]] std::optional<T> Pop() {
		const auto head = m_head.load(std::memory_order_relaxed);
		if (m_consumer_tail == head) {
			m_consumer_tail = m_tail.load(std::memory_order_acquire);
			if (m_consumer_tail == head && !WaitForElement(head)) {
				return std::nullopt;
			}
		}
		return Take(head);
	}

	// Returns nothing when the ring is empty.
	[[nodiscard]] std::optional<T> TryPop() {
		const auto head = m_head.load(std::memory_order_relaxed);
		if (m_consumer_tail == head) {
			m_consumer_tail = m_tail.load(std::memory_order_acquire);
			if (m_consumer_tail == head) {
				return std::nullopt;
			}
		}
		return Take(head);
	}

	// Any thread: wakes the consumer, whose Pop() returns nothing once the ring is empty.
	void Stop() {
		m_stopped.store(true, std::memory_order_seq_cst);
		m_consumer_signal.fetch_add(1, std::memory_order_seq_cst);
		m_consumer_signal.notify_all();
	}

	[[nodiscard]] size_t Capacity() const noexcept { return m_capacity; }

private:
	static constexpr uint32_t CheckInterval = 16;

	// A slot per cache line at least: the producer writes one while the consumer reads another.
	struct alignas(alignof(T) > 64 ? alignof(T) : 64) Slot {
		std::byte storage[sizeof(T)];
	};

	T* Element(uint64_t index) {
		return std::launder(reinterpret_cast<T*>(m_slots[index & m_mask].storage));
	}

	void Construct(uint64_t tail, T&& value) {
		::new (static_cast<void*>(m_slots[tail & m_mask].storage)) T(std::move(value));
		m_tail.store(tail + 1, std::memory_order_release);
		std::atomic_thread_fence(std::memory_order_seq_cst);
		if (m_consumer_sleeping.load(std::memory_order_relaxed)) {
			m_consumer_signal.fetch_add(1, std::memory_order_seq_cst);
			m_consumer_signal.notify_one();
		}
	}

	std::optional<T> Take(uint64_t head) {
		auto*            element = Element(head);
		std::optional<T> value(std::move(*element));
		element->~T();
		m_head.store(head + 1, std::memory_order_release);
		if (++m_unchecked_pops >= CheckInterval) {
			WakeProducer();
		}
		return value;
	}

	void WakeProducer() {
		m_unchecked_pops = 0;
		std::atomic_thread_fence(std::memory_order_seq_cst);
		if (m_producer_sleeping.load(std::memory_order_relaxed)) {
			m_producer_signal.fetch_add(1, std::memory_order_seq_cst);
			m_producer_signal.notify_one();
		}
	}

	void WaitForSpace(uint64_t tail) {
		SyncAcct::Scope spin_scope(m_site_push_spin, 0);
		SpinWait spin;
		while (spin.Spin()) {
			m_producer_head = m_head.load(std::memory_order_acquire);
			if (tail - m_producer_head < m_capacity) {
				return;
			}
		}
		spin_scope.End();
		SyncAcct::Scope block_scope(m_site_push_block, 0);
		for (;;) {
			m_producer_sleeping.store(true, std::memory_order_seq_cst);
			const auto signal = m_producer_signal.load(std::memory_order_seq_cst);
			m_producer_head   = m_head.load(std::memory_order_seq_cst);
			if (tail - m_producer_head < m_capacity) {
				break;
			}
			WaitStats::Kernel();
			m_producer_signal.wait(signal, std::memory_order_seq_cst);
		}
		m_producer_sleeping.store(false, std::memory_order_relaxed);
	}

	// False when stopped with nothing left to pop.
	bool WaitForElement(uint64_t head) {
		// The producer may be asleep on a full ring the consumer drained without a check.
		WakeProducer();
		SyncAcct::Scope spin_scope(m_site_pop_spin, 0);
		SpinWait spin;
		while (spin.Spin()) {
			m_consumer_tail = m_tail.load(std::memory_order_acquire);
			if (m_consumer_tail != head) {
				return true;
			}
			if (m_stopped.load(std::memory_order_relaxed)) {
				break;
			}
		}
		spin_scope.End();
		SyncAcct::Scope block_scope(m_site_pop_block, 0);
		bool available = false;
		for (;;) {
			m_consumer_sleeping.store(true, std::memory_order_seq_cst);
			const auto signal = m_consumer_signal.load(std::memory_order_seq_cst);
			m_consumer_tail   = m_tail.load(std::memory_order_seq_cst);
			if (m_consumer_tail != head) {
				available = true;
				break;
			}
			if (m_stopped.load(std::memory_order_seq_cst)) {
				break;
			}
			WaitStats::Kernel();
			m_consumer_signal.wait(signal, std::memory_order_seq_cst);
		}
		m_consumer_sleeping.store(false, std::memory_order_relaxed);
		return available;
	}

	uint16_t m_site_pop_spin   = 0;
	uint16_t m_site_pop_block  = 0;
	uint16_t m_site_push_spin  = 0;
	uint16_t m_site_push_block = 0;

	const uint64_t          m_capacity;
	const uint64_t          m_mask;
	std::unique_ptr<Slot[]> m_slots;

	// Consumer side.
	alignas(64) std::atomic_uint64_t m_head {0};
	uint64_t             m_consumer_tail  = 0;
	uint32_t             m_unchecked_pops = 0;
	std::atomic_bool     m_consumer_sleeping {false};
	std::atomic_uint32_t m_consumer_signal {0};
	std::atomic_bool     m_stopped {false};

	// Producer side.
	alignas(64) std::atomic_uint64_t m_tail {0};
	uint64_t             m_producer_head = 0;
	std::atomic_bool     m_producer_sleeping {false};
	std::atomic_uint32_t m_producer_signal {0};
};

} // namespace Common

#endif // KYTY_COMMON_SPSCQUEUE_H_
