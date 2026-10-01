#include "common/inlineFunction.h"
#include "common/spscQueue.h"
#include "graphics/host_gpu/renderer/occlusionQueries.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
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


namespace Occlusion = Libs::Graphics::OcclusionCounters;

// Plays the role of OcclusionQueries over a fake GPU: each allocated slot counts `samples`, and
// dumps publish in order once "completed".
struct OcclusionModel {
	explicit OcclusionModel(uint32_t capacity): segments(capacity) {}

	Occlusion::Segments                   segments;
	std::map<uint64_t, uint64_t>          memory;
	std::map<uint32_t, uint64_t>          slot_samples;
	std::vector<Occlusion::Segments::Dump> pending;
	uint64_t                              total = 0;

	// Draws that pass `samples` samples, recorded inside one render pass instance.
	void Draw(uint64_t samples) {
		if (!segments.Counting()) {
			return;
		}
		uint32_t slot = 0;
		if (segments.AllocateSlot(slot)) {
			slot_samples[slot] = samples;
		}
	}
	void Dump(uint64_t address) { pending.push_back(segments.OnDump(address)); }
	void Complete() {
		for (const auto& dump: pending) {
			for (uint64_t index = dump.first_slot; index < dump.end_slot; index++) {
				total += slot_samples[static_cast<uint32_t>(index % segments.Capacity())];
			}
			if (dump.uncounted) {
				total += Occlusion::UncountedSamples;
			}
			segments.Release(dump.end_slot);
			for (uint32_t db = 0; db < Occlusion::DbCount; db++) {
				memory[dump.address + db * Occlusion::DbStride] = Occlusion::DbValue(db, total);
			}
		}
		pending.clear();
	}
	// What SET_PREDICATION and the guest compute: the sum of end - begin over the DBs, if ready.
	bool Read(uint64_t address, uint64_t& samples) {
		samples = 0;
		for (uint32_t db = 0; db < Occlusion::DbCount; db++) {
			const auto begin = memory[address + db * Occlusion::DbStride];
			const auto end   = memory[address + db * Occlusion::DbStride + 8u];
			if ((begin & end & Occlusion::ReadyBit) == 0) {
				return false;
			}
			samples += end - begin;
		}
		return true;
	}
};

void TestOcclusionCounters() {
	Check(Occlusion::DbValue(0, 5) == (Occlusion::ReadyBit | 5u), "DB 0 holds the count");
	Check(Occlusion::DbValue(3, 5) == Occlusion::ReadyBit, "other DBs stay zero");
	Check(Occlusion::DumpSize == 15u * 16u + 8u, "dump size");

	// Draws outside queries are not counted; a query reads the samples between its dumps.
	OcclusionModel model(64);
	model.Draw(1000);
	Check(!model.segments.Counting(), "no query open");
	model.Dump(0x1000);
	Check(model.segments.Counting(), "begin opens a query");
	model.Draw(7);
	model.Draw(3);
	model.Dump(0x1008);
	Check(!model.segments.Counting(), "end closes it");
	uint64_t samples = 0;
	Check(!model.Read(0x1000, samples), "not ready before the GPU finished");
	model.Complete();
	Check(model.Read(0x1000, samples) && samples == 10, "query counts its draws");

	// An occluded query reads zero.
	model.Dump(0x2000);
	model.Draw(0);
	model.Dump(0x2008);
	model.Complete();
	Check(model.Read(0x2000, samples) && samples == 0, "occluded query");

	// Nested and interleaved queries share the running counter.
	model.Dump(0x3000);
	model.Draw(5);
	model.Dump(0x4000);
	model.Draw(11);
	model.Dump(0x3008);
	model.Draw(13);
	model.Dump(0x4008);
	model.Complete();
	Check(model.Read(0x3000, samples) && samples == 16, "outer query");
	Check(model.Read(0x4000, samples) && samples == 24, "interleaved query");
	Check(!model.segments.Counting(), "all queries closed");

	// A full slot ring makes the segment read visible instead of occluded.
	OcclusionModel small(2);
	small.Dump(0x5000);
	small.Draw(0);
	small.Draw(0);
	small.Draw(0);
	small.Dump(0x5008);
	small.Complete();
	Check(small.Read(0x5000, samples) && samples >= Occlusion::UncountedSamples,
	      "uncounted samples read visible");
	// Released slots are allocated again.
	small.Dump(0x6000);
	small.Draw(4);
	small.Draw(2);
	small.Dump(0x6008);
	small.Complete();
	Check(small.Read(0x6000, samples) && samples == 6, "slots reused after release");

	// A begin without its end stops keeping the counters running after a few frames.
	Occlusion::Segments ageing(16, 2);
	(void)ageing.OnDump(0x7000);
	Check(ageing.Counting(), "open");
	ageing.OnFrame();
	ageing.OnFrame();
	Check(ageing.Counting(), "still open within the limit");
	ageing.OnFrame();
	Check(!ageing.Counting(), "dropped after the limit");
}

int main() {
	TestSpscQueue();
	TestProgressCounter();
	TestInlineFunction();
	TestOcclusionCounters();
	std::printf("GpuPipelineTests: all tests passed\n");
	return 0;
}
