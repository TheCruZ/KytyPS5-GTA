#include "common/inlineFunction.h"
#include "common/spscQueue.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "GpuPipelineTests: failed: %s\n", text);
		std::abort();
	}
}

// Busy-waits longer than the spin budget, so the other side falls back to its kernel wait.
void Stall(std::chrono::nanoseconds duration) {
	const auto end = std::chrono::steady_clock::now() + duration;
	while (std::chrono::steady_clock::now() < end) {
		Common::SpinPause();
	}
}

// Move-only elements pass through a small queue in order while both sides sleep and wake.
void TestSpscQueue() {
	constexpr uint64_t              Count = 1000000;
	Common::SpscQueue<std::unique_ptr<uint64_t>> queue(8);
	Check(queue.Capacity() == 8, "capacity is a power of two");
	uint64_t     received = 0;
	bool         ordered  = true;
	std::jthread consumer([&] {
		for (;;) {
			auto value = queue.Pop();
			if (!value) {
				return;
			}
			ordered &= **value == received;
			received++;
			// Let the producer fill the queue now and then.
			if (received % 100000 == 0) {
				Stall(std::chrono::milliseconds(3));
			}
		}
	});
	for (uint64_t i = 0; i < Count; i++) {
		queue.Push(std::make_unique<uint64_t>(i));
		// Let the consumer sleep on an empty queue now and then.
		if (i % 250000 == 0) {
			Stall(std::chrono::milliseconds(3));
		}
	}
	queue.Stop();
	consumer.join();
	Check(ordered && received == Count, "every element arrives once, in order");

	// Elements left in a stopped queue are still popped, then destroyed with the queue.
	Common::SpscQueue<std::unique_ptr<uint64_t>> leftover(4);
	leftover.Push(std::make_unique<uint64_t>(1));
	leftover.Push(std::make_unique<uint64_t>(2));
	leftover.Stop();
	auto first = leftover.Pop();
	Check(first && **first == 1, "a stopped queue drains before it reports the stop");
}

// A waiter sees the writes made before the counter reached its target.
void TestProgressCounter() {
	constexpr uint64_t     Rounds = 200000;
	Common::ProgressCounter counter;
	std::vector<uint64_t>  data(Rounds + 1, 0);
	std::jthread           advancer([&] {
		for (uint64_t i = 1; i <= Rounds; i++) {
			data[i] = i * 3;
			counter.Advance(i);
			if (i % 50000 == 0) {
				Stall(std::chrono::milliseconds(3));
			}
		}
	});
	bool visible = true;
	for (uint64_t target = 1; target <= Rounds; target += 7) {
		counter.WaitFor(target);
		visible &= data[target] == target * 3;
	}
	counter.WaitFor(Rounds);
	advancer.join();
	Check(visible, "writes before Advance() are visible after WaitFor()");
	Check(counter.Load() == Rounds, "final value");

	// Published values wake a waiter at the next Notify(), as the execution thread does.
	Common::ProgressCounter lazy;
	std::jthread            publisher([&] {
		for (uint64_t i = 1; i <= Rounds; i++) {
			lazy.Publish(i);
			if (i % 16 == 0) {
				lazy.Notify();
			}
			if (i % 40000 == 0) {
				Stall(std::chrono::milliseconds(3));
			}
		}
		lazy.Notify();
	});
	for (uint64_t target = 1; target <= Rounds; target += 997) {
		lazy.WaitFor(target);
	}
	lazy.WaitFor(Rounds);
	publisher.join();
	Check(lazy.Load() == Rounds, "published final value");
}

// Small callables live in place, large ones on the heap; both move and destroy once.
void TestInlineFunction() {
	static int destroyed = 0;
	struct Probe {
		int* calls;
		Probe(int* c): calls(c) {}
		Probe(Probe&& other) noexcept: calls(other.calls) { other.calls = nullptr; }
		~Probe() {
			if (calls != nullptr) {
				destroyed++;
			}
		}
		void operator()() const { (*calls)++; }
	};
	int calls = 0;
	{
		Common::InlineFunction<32> small(Probe {&calls});
		auto moved = std::move(small);
		Check(!small && moved, "moving empties the source");
		moved();
		std::array<uint64_t, 16> big {};
		big[15] = 7;
		Common::InlineFunction<32> large([big, &calls] { calls += static_cast<int>(big[15]); });
		Common::InlineFunction<32> large_moved;
		large_moved = std::move(large);
		large_moved();
	}
	Check(calls == 8, "calls reach the callables");
	Check(destroyed == 1, "the small callable is destroyed once");
}

} // namespace

int main() {
	TestSpscQueue();
	TestProgressCounter();
	TestInlineFunction();
	std::printf("GpuPipelineTests: all tests passed\n");
	return 0;
}
