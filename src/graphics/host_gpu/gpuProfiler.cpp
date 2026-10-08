#include "graphics/guest_gpu/execTrace.h"
#include "graphics/host_gpu/gpuProfiler.h"

#include "common/spscQueue.h"
#include "common/syncAcct.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/cacheStats.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <intrin.h>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::GpuProfiler {
namespace {

constexpr uint32_t NumSlots       = 128;
constexpr uint32_t QueriesPerSlot = 4096;
constexpr double   ReportSeconds  = 5.0;

enum class RecordKind : uint8_t { Dispatch, Draw, Region };

struct Record {
	RecordKind  kind    = RecordKind::Dispatch;
	uint64_t    hash_a  = 0;
	uint64_t    hash_b  = 0;
	const char* name    = nullptr;
	uint32_t    begin_q = 0;
	uint32_t    end_q   = UINT32_MAX;
};

struct Slot {
	bool                busy    = false;
	vk::CommandBuffer   command = nullptr;
	uint32_t            used    = 0;
	std::vector<Record> records;
	std::vector<size_t> open;
};

struct OpStats {
	uint64_t ticks = 0;
	uint64_t count = 0;
	uint64_t max   = 0;
};

struct WaitStats {
	uint64_t ns    = 0;
	uint64_t count = 0;
};

using OpKey = std::tuple<uint8_t, uint64_t, uint64_t, std::string>;

struct State {
	std::mutex                   mutex;
	bool                         initialized = false;
	bool                         failed      = false;
	vk::Device                   device      = nullptr;
	vk::QueryPool                pool        = nullptr;
	double                       ns_per_tick = 1.0;
	uint64_t                     valid_mask  = ~uint64_t {0};
	std::array<Slot, NumSlots>   slots;
	uint32_t                     next_slot  = 0;
	std::unordered_map<VkCommandBuffer, uint32_t> open_slots;
	uint64_t                     dropped_cb = 0;
	std::map<OpKey, OpStats>     ops;
	uint64_t                     busy_ticks  = 0;
	uint64_t                     idle_ticks  = 0;
	uint64_t                     last_end    = 0;
	uint64_t                     cb_count    = 0;
	uint64_t                     op_count    = 0;
	std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
};

State& GetState() {
	static State state;
	return state;
}

std::mutex& WaitMutex() {
	static std::mutex mutex;
	return mutex;
}

std::map<std::string, WaitStats>& Waits() {
	static std::map<std::string, WaitStats> waits;
	return waits;
}

std::atomic<uint64_t> g_frames {0};
// The GPU profile report resets g_frames; the phase report keeps its own count.
std::atomic<uint64_t> g_phase_frames {0};

std::map<std::pair<std::string, uint64_t>, uint64_t>& Addresses() {
	static std::map<std::pair<std::string, uint64_t>, uint64_t> addresses;
	return addresses;
}

thread_local const char* g_wait_tag = nullptr;
thread_local uint64_t    g_current_shader = 0;

std::map<uint64_t, std::pair<uint64_t, uint64_t>>& Writers() {
	static std::map<uint64_t, std::pair<uint64_t, uint64_t>> writers;
	return writers;
}

// Per-frame attribution (KYTY_PHASE_PROFILE): host events (waits, compiles, uploads, faults)
// accumulated since the last flip and since the last periodic report.
struct EventTotals {
	uint64_t ns    = 0;
	uint64_t count = 0;
};
using EventList = std::vector<std::pair<std::string, EventTotals>>;
std::mutex         g_event_mutex;
EventList          g_frame_events;
EventList          g_period_events;
std::vector<float> g_frame_times_ms;

void AddEvent(EventList& list, const char* tag, uint64_t ns, uint64_t count) {
	auto it = std::find_if(list.begin(), list.end(), [&](const auto& e) { return e.first == tag; });
	if (it == list.end()) {
		list.emplace_back(tag, EventTotals {});
		it = list.end() - 1;
	}
	it->second.ns += ns;
	it->second.count += count;
}

// Events with string-literal tags accumulate per thread without locks (the hot paths count
// several per draw); TakeEvents merges them by name.
struct ThreadEvents {
	static constexpr size_t Capacity = 256;
	struct Entry {
		std::atomic<uint64_t> ns {0};
		std::atomic<uint64_t> count {0};
	};
	std::array<const char*, Capacity> names {};
	std::array<Entry, Capacity>       frame;
	std::array<Entry, Capacity>       period;
	std::atomic<uint32_t>             size {0};
	std::array<const char*, 512>      keys {};
	std::array<uint16_t, 512>         index {};

	size_t Find(const char* name) {
		auto slot = static_cast<size_t>((reinterpret_cast<uintptr_t>(name) >> 3u) * 0x9e3779b97f4a7c15ull >> 55u);
		for (;; slot = (slot + 1) % keys.size()) {
			if (keys[slot] == name) {
				return index[slot];
			}
			if (keys[slot] == nullptr) {
				const auto n = size.load(std::memory_order_relaxed);
				if (n == Capacity) {
					return Capacity;
				}
				keys[slot]  = name;
				index[slot] = static_cast<uint16_t>(n);
				names[n]    = name;
				size.store(n + 1, std::memory_order_release);
				return n;
			}
		}
	}
};
std::mutex                                 g_event_threads_mutex;
std::vector<std::shared_ptr<ThreadEvents>> g_event_threads;
thread_local std::shared_ptr<ThreadEvents> t_events;

void AddFrameEventString(const char* tag, uint64_t ns, uint64_t count) noexcept {
	std::lock_guard lock(g_event_mutex);
	AddEvent(g_frame_events, tag, ns, count);
	AddEvent(g_period_events, tag, ns, count);
}

// The tag must be a string literal (or otherwise outlive the process).
void AddFrameEvent(const char* tag, uint64_t ns, uint64_t count) noexcept {
	if (tag == nullptr) {
		tag = "untagged";
	}
	if (t_events == nullptr) {
		t_events = std::make_shared<ThreadEvents>();
		std::lock_guard lock(g_event_threads_mutex);
		g_event_threads.push_back(t_events);
	}
	const auto index = t_events->Find(tag);
	if (index == ThreadEvents::Capacity) {
		AddFrameEventString(tag, ns, count);
		return;
	}
	// Single writer: no locked instructions (see PhaseMarkImpl).
	auto& frame = t_events->frame[index];
	frame.ns.store(frame.ns.load(std::memory_order_relaxed) + ns, std::memory_order_relaxed);
	frame.count.store(frame.count.load(std::memory_order_relaxed) + count, std::memory_order_relaxed);
	auto& period = t_events->period[index];
	period.ns.store(period.ns.load(std::memory_order_relaxed) + ns, std::memory_order_relaxed);
	period.count.store(period.count.load(std::memory_order_relaxed) + count, std::memory_order_relaxed);
}

// Moves the per-thread totals of one kind into the list (called with g_event_mutex held).
void MergeThreadEvents(EventList& list, bool frame) {
	std::vector<std::shared_ptr<ThreadEvents>> threads;
	{
		std::lock_guard lock(g_event_threads_mutex);
		threads = g_event_threads;
	}
	for (auto& thread: threads) {
		const auto size = thread->size.load(std::memory_order_acquire);
		for (uint32_t i = 0; i < size; i++) {
			auto& entry = frame ? thread->frame[i] : thread->period[i];
			if (entry.count.load(std::memory_order_relaxed) == 0) {
				continue;
			}
			const auto count = entry.count.exchange(0, std::memory_order_relaxed);
			const auto ns    = entry.ns.exchange(0, std::memory_order_relaxed);
			AddEvent(list, thread->names[i], ns, count);
		}
	}
}

// Moves the non-zero entries out and zeroes them in place (keeps the strings allocated).
EventList TakeEvents(EventList& list) {
	MergeThreadEvents(list, &list == &g_frame_events);
	EventList taken;
	for (auto& entry: list) {
		if (entry.second.ns != 0 || entry.second.count != 0) {
			taken.push_back(entry);
			entry.second = {};
		}
	}
	std::sort(taken.begin(), taken.end(), [](const auto& a, const auto& b) {
		return a.second.ns != b.second.ns ? a.second.ns > b.second.ns
		                                  : a.second.count > b.second.count;
	});
	return taken;
}

void FrameBoundary() noexcept;

bool Initialize(State& state, GraphicContext& graphics) {
	if (state.initialized) {
		return !state.failed;
	}
	state.initialized = true;
	state.device      = graphics.device;
	vk::QueryPoolCreateInfo info {};
	info.queryType  = vk::QueryType::eTimestamp;
	info.queryCount = NumSlots * QueriesPerSlot;
	if (graphics.device.createQueryPool(&info, nullptr, &state.pool) != vk::Result::eSuccess) {
		state.failed = true;
		std::printf("GPU profiler: cannot create the timestamp query pool\n");
		return false;
	}
	state.ns_per_tick = graphics.physical_device_properties.limits.timestampPeriod;
	uint32_t count    = 0;
	graphics.physical_device.getQueueFamilyProperties(&count, nullptr);
	std::vector<vk::QueueFamilyProperties> families(count);
	graphics.physical_device.getQueueFamilyProperties(&count, families.data());
	if (graphics.queue_family < count) {
		const auto bits = families[graphics.queue_family].timestampValidBits;
		state.valid_mask = bits >= 64 ? ~uint64_t {0} : ((uint64_t {1} << bits) - 1);
	}
	std::printf("GPU profiler: enabled, %.3f ns/tick\n", state.ns_per_tick);
	Common::SyncAcct::Meta("gpu_ns_per_tick %.6f", state.ns_per_tick);
	Common::SyncAcct::Meta("gpu_valid_mask 0x%llx", static_cast<unsigned long long>(state.valid_mask));
	std::fflush(stdout);
	return true;
}

std::string OpName(const OpKey& key) {
	char buffer[160];
	switch (static_cast<RecordKind>(std::get<0>(key))) {
		case RecordKind::Dispatch:
			std::snprintf(buffer, sizeof(buffer), "CS %016" PRIx64, std::get<1>(key));
			break;
		case RecordKind::Draw:
			std::snprintf(buffer, sizeof(buffer), "VS %016" PRIx64 " PS %016" PRIx64,
			              std::get<1>(key), std::get<2>(key));
			break;
		case RecordKind::Region:
			std::snprintf(buffer, sizeof(buffer), "%s", std::get<3>(key).c_str());
			break;
	}
	return buffer;
}

void Report(State& state, double seconds) {
	const auto frames = g_frames.exchange(0);
	const auto ms     = [&state](uint64_t ticks) { return ticks * state.ns_per_tick / 1.0e6; };
	const double fps  = frames / seconds;
	std::printf("==== GPU profile: %.2f s, %" PRIu64 " frames (%.2f fps), %" PRIu64
	            " command buffers, %" PRIu64 " ops, dropped cb=%" PRIu64 "\n",
	            seconds, frames, fps, state.cb_count, state.op_count, state.dropped_cb);
	const double busy = ms(state.busy_ticks);
	const double idle = ms(state.idle_ticks);
	std::printf("  GPU busy %.1f ms/s (%.1f%%), idle gaps %.1f ms/s, busy per frame %.2f ms\n",
	            busy / seconds, busy / (seconds * 10.0), idle / seconds,
	            frames != 0 ? busy / static_cast<double>(frames) : 0.0);
	std::vector<std::pair<OpKey, OpStats>> sorted(state.ops.begin(), state.ops.end());
	std::sort(sorted.begin(), sorted.end(),
	          [](const auto& a, const auto& b) { return a.second.ticks > b.second.ticks; });
	std::array<double, 3> by_kind {};
	for (const auto& [key, stats]: sorted) {
		by_kind[std::get<0>(key)] += ms(stats.ticks);
	}
	std::printf("  totals: dispatch %.1f ms/s, draw %.1f ms/s, regions %.1f ms/s\n",
	            by_kind[0] / seconds, by_kind[1] / seconds, by_kind[2] / seconds);
	const size_t top = std::min<size_t>(sorted.size(), 40);
	for (size_t i = 0; i < top; i++) {
		const auto& [key, stats] = sorted[i];
		const double total       = ms(stats.ticks);
		std::printf("  %6.2f ms/frame %5.1f%% calls/frame %7.1f avg %8.1f us max %8.1f us  %s\n",
		            frames != 0 ? total / static_cast<double>(frames) : 0.0,
		            busy > 0 ? 100.0 * total / busy : 0.0,
		            frames != 0 ? static_cast<double>(stats.count) / static_cast<double>(frames)
		                        : 0.0,
		            stats.count != 0 ? 1000.0 * total / static_cast<double>(stats.count) : 0.0,
		            ms(stats.max) * 1000.0, OpName(key).c_str());
	}
	{
		std::lock_guard lock(WaitMutex());
		std::vector<std::pair<std::string, WaitStats>> waits(Waits().begin(), Waits().end());
		std::sort(waits.begin(), waits.end(),
		          [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
		for (const auto& [tag, stats]: waits) {
			std::printf("  wait %-44s %8.1f ms/s  %8.1f /s  avg %8.1f us\n", tag.c_str(),
			            stats.ns / 1.0e6 / seconds, static_cast<double>(stats.count) / seconds,
			            stats.count != 0 ? stats.ns / 1.0e3 / static_cast<double>(stats.count)
			                             : 0.0);
		}
		Waits().clear();
		std::map<std::string, std::vector<std::pair<uint64_t, uint64_t>>> by_tag;
		for (const auto& [key, value]: Addresses()) {
			by_tag[key.first].emplace_back(value, key.second);
		}
		for (auto& [tag, entries]: by_tag) {
			std::sort(entries.begin(), entries.end(), std::greater<>());
			uint64_t total = 0;
			for (const auto& entry: entries) {
				total += entry.first;
			}
			std::printf("  addr %s: %zu distinct, total %" PRIu64 "\n", tag.c_str(),
			            entries.size(), total);
			for (size_t i = 0; i < std::min<size_t>(entries.size(), 12); i++) {
				std::printf("    0x%012" PRIx64 " %8" PRIu64 "\n", entries[i].second,
				            entries[i].first);
			}
		}
		Addresses().clear();
	}
	std::fflush(stdout);
	state.ops.clear();
	state.busy_ticks = 0;
	state.idle_ticks = 0;
	state.cb_count   = 0;
	state.op_count   = 0;
	state.dropped_cb = 0;
}

void MaybeReport(State& state) {
	const auto now     = std::chrono::steady_clock::now();
	const auto seconds = std::chrono::duration<double>(now - state.last_report).count();
	if (seconds >= ReportSeconds) {
		Report(state, seconds);
		state.last_report = now;
	}
}

} // namespace

// Debug (sync-wait audit): KYTY_GPU_PROFILE=cbonly records only the command-buffer timestamps;
// Enabled() stays false so that the debug bookkeeping it gates (GPU writer maps, readback
// comparisons, fault attribution with stack walks) does not slow the measured run.
bool CbEnabled() noexcept {
	static const bool enabled = std::getenv("KYTY_GPU_PROFILE") != nullptr;
	return enabled;
}

bool Enabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_PROFILE");
		return value != nullptr && std::string(value) != "cbonly";
	}();
	return enabled;
}

// KYTY_GPU_PROFILE=cb records only command-buffer boundaries (no per-operation timestamps,
// which serialize the GPU), for a low-overhead GPU busy measurement.
static bool OpsEnabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_PROFILE");
		return value != nullptr && std::string(value) != "cb" && std::string(value) != "cbonly";
	}();
	return enabled;
}

void BeginCommandBuffer(GraphicContext& graphics, vk::CommandBuffer command) {
	if (!CbEnabled()) {
		return;
	}
	auto&           state = GetState();
	std::lock_guard lock(state.mutex);
	if (const auto it = state.open_slots.find(command); it != state.open_slots.end()) {
		state.slots[it->second].busy = false;
		state.open_slots.erase(it);
	}
	if (!Initialize(state, graphics)) {
		return;
	}
	for (uint32_t i = 0; i < NumSlots; i++) {
		const auto index = (state.next_slot + i) % NumSlots;
		auto&      slot  = state.slots[index];
		if (slot.busy) {
			continue;
		}
		state.next_slot = (index + 1) % NumSlots;
		slot.busy       = true;
		slot.command    = command;
		slot.used       = 1;
		slot.records.clear();
		slot.open.clear();
		const auto base = index * QueriesPerSlot;
		command.resetQueryPool(state.pool, base, QueriesPerSlot);
		command.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, state.pool, base);
		state.open_slots[command] = index;
		return;
	}
	state.dropped_cb++;
}

uint32_t EndCommandBuffer(vk::CommandBuffer command) {
	if (!CbEnabled()) {
		return UINT32_MAX;
	}
	auto&           state = GetState();
	std::lock_guard lock(state.mutex);
	const auto it = state.open_slots.find(command);
	if (it == state.open_slots.end()) {
		return UINT32_MAX;
	}
	const auto index = it->second;
	auto&      slot  = state.slots[index];
	state.open_slots.erase(it);
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, state.pool,
	                       index * QueriesPerSlot + slot.used);
	slot.used++;
	return index;
}

void Resolve(GraphicContext& graphics, uint32_t index, uint64_t tick, uint64_t scheduler) {
	(void)graphics;
	if (index >= NumSlots) {
		return;
	}
	auto&           state = GetState();
	std::lock_guard lock(state.mutex);
	auto&           slot = state.slots[index];
	if (!slot.busy) {
		return;
	}
	std::vector<uint64_t> data(slot.used);
	const auto            result = state.device.getQueryPoolResults(
        state.pool, index * QueriesPerSlot, slot.used, data.size() * sizeof(uint64_t), data.data(),
        sizeof(uint64_t), vk::QueryResultFlagBits::e64);
	slot.busy = false;
	if (result != vk::Result::eSuccess) {
		return;
	}
	for (auto& value: data) {
		value &= state.valid_mask;
	}
	const auto start = data.front();
	const auto end   = data.back();
	{
		// Debug (KYTY_SYNC_ACCT): the GPU interval of each command buffer, by tick.
		KYTY_SYNC_SITE(s_gpu_cb, "gpu cb", Gpu);
		Common::SyncAcct::Interval(s_gpu_cb, start, end, tick, scheduler);
	}
	if (end > start) {
		state.busy_ticks += end - start;
	}
	if (state.last_end != 0 && start > state.last_end) {
		state.idle_ticks += start - state.last_end;
	}
	state.last_end = std::max(state.last_end, end);
	state.cb_count++;
	for (const auto& record: slot.records) {
		if (record.end_q == UINT32_MAX || record.end_q >= data.size()) {
			continue;
		}
		const auto begin_ts = data[record.begin_q];
		const auto end_ts   = data[record.end_q];
		const auto duration = end_ts > begin_ts ? end_ts - begin_ts : 0;
		auto& stats = state.ops[OpKey {static_cast<uint8_t>(record.kind), record.hash_a,
		                               record.hash_b, record.name != nullptr ? record.name : ""}];
		stats.ticks += duration;
		stats.count++;
		stats.max = std::max(stats.max, duration);
		state.op_count++;
	}
	MaybeReport(state);
}

static void MarkImpl(vk::CommandBuffer command, bool begin, RecordKind kind, uint64_t hash_a,
                     uint64_t hash_b, const char* name) {
	auto&           state = GetState();
	std::lock_guard lock(state.mutex);
	const auto      it = state.open_slots.find(command);
	if (it == state.open_slots.end()) {
		return;
	}
	const auto index = it->second;
	auto&      slot  = state.slots[index];
	if (begin) {
		if (slot.used + 3 >= QueriesPerSlot) {
			return;
		}
		Record record;
		record.kind    = kind;
		record.hash_a  = hash_a;
		record.hash_b  = hash_b;
		record.name    = name;
		record.begin_q = slot.used;
		command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, state.pool,
		                       index * QueriesPerSlot + slot.used);
		slot.used++;
		slot.open.push_back(slot.records.size());
		slot.records.push_back(record);
		return;
	}
	if (slot.open.empty()) {
		return;
	}
	auto& record = slot.records[slot.open.back()];
	slot.open.pop_back();
	record.end_q = slot.used;
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, state.pool,
	                       index * QueriesPerSlot + slot.used);
	slot.used++;
}

void Mark(vk::CommandBuffer command, GpuCheckpointKind kind, uint64_t hash_a, uint64_t hash_b) {
	if (!OpsEnabled()) {
		return;
	}
	switch (kind) {
		case GpuCheckpointKind::DispatchBegin:
			MarkImpl(command, true, RecordKind::Dispatch, hash_a, hash_b, nullptr);
			break;
		case GpuCheckpointKind::DispatchEnd:
			MarkImpl(command, false, RecordKind::Dispatch, hash_a, hash_b, nullptr);
			break;
		case GpuCheckpointKind::DrawBegin:
			MarkImpl(command, true, RecordKind::Draw, hash_a, hash_b, nullptr);
			break;
		case GpuCheckpointKind::DrawEnd:
			MarkImpl(command, false, RecordKind::Draw, hash_a, hash_b, nullptr);
			break;
	}
}

void Region(vk::CommandBuffer command, bool begin, const char* name) {
	if (!OpsEnabled()) {
		return;
	}
	MarkImpl(command, begin, RecordKind::Region, 0, 0, name);
}

void AddWait(const char* tag, uint64_t ns) noexcept {
	if (PhaseEnabled()) {
		AddFrameEvent(tag, ns, 1);
	}
	if (!Enabled()) {
		return;
	}
	std::lock_guard lock(WaitMutex());
	auto&           stats = Waits()[tag != nullptr ? tag : "untagged"];
	stats.ns += ns;
	stats.count++;
}

void AddWaitString(const char* tag, uint64_t ns) noexcept {
	if (PhaseEnabled()) {
		AddFrameEventString(tag, ns, 1);
	}
	if (!Enabled()) {
		return;
	}
	std::lock_guard lock(WaitMutex());
	auto&           stats = Waits()[tag];
	stats.ns += ns;
	stats.count++;
}

void AddCount(const char* tag, uint64_t count) noexcept {
	if (PhaseEnabled()) {
		AddFrameEvent(tag, 0, count);
	}
	if (!Enabled()) {
		return;
	}
	std::lock_guard lock(WaitMutex());
	auto&           stats = Waits()[std::string("count ") + tag];
	stats.count += count;
}

void AddAddress(const char* tag, uint64_t address, uint64_t value) noexcept {
	if (!Enabled()) {
		return;
	}
	std::lock_guard lock(WaitMutex());
	Addresses()[{tag, address}] += value;
}

void SetCurrentShader(uint64_t hash) noexcept {
	g_current_shader = hash;
	ExecTrace::g_shader_hash = hash;
}

void NoteGpuWrite(uint64_t address, uint64_t size) noexcept {
	if (!Enabled()) {
		return;
	}
	std::lock_guard lock(WaitMutex());
	Writers()[address] = {address + size, g_current_shader};
}

uint64_t LastGpuWriter(uint64_t address) noexcept {
	if (!Enabled()) {
		return 0;
	}
	std::lock_guard lock(WaitMutex());
	auto& writers = Writers();
	auto  it      = writers.upper_bound(address);
	while (it != writers.begin()) {
		--it;
		if (it->second.first > address) {
			return it->second.second;
		}
		if (address - it->first > (uint64_t {64} << 20u)) {
			break;
		}
	}
	return 0;
}

void AddFrame() noexcept {
	g_frames.fetch_add(1, std::memory_order_relaxed);
	g_phase_frames.fetch_add(1, std::memory_order_relaxed);
	if (PhaseEnabled()) {
		FrameBoundary();
	}
}


namespace {
struct PhaseTotals {
	uint64_t ns    = 0;
	uint64_t count = 0;
};
// Per-thread totals written only by their thread without locks (relaxed atomics: a mark costs two
// uncontended adds); the reporter takes them with exchanges. Entries are never removed, so a small
// pointer-keyed table finds a phase's index without scanning.
struct ThreadPhases {
	static constexpr size_t Capacity = 256;
	struct Entry {
		std::atomic<uint64_t> ns {0};
		std::atomic<uint64_t> count {0};
	};
	std::array<const char*, Capacity> names {};
	std::array<Entry, Capacity>       totals;
	// Since the last flip (hitch attribution), indexed like totals.
	std::array<Entry, Capacity>       frame;
	// Entries [0, size) have their names set (release by the owner, acquire by readers).
	std::atomic<uint32_t>             size {0};
	std::array<const char*, 512>      keys {};
	std::array<uint16_t, 512>         index {};

	size_t Find(const char* name) {
		auto slot = static_cast<size_t>((reinterpret_cast<uintptr_t>(name) >> 3u) * 0x9e3779b97f4a7c15ull >> 55u);
		for (;; slot = (slot + 1) % keys.size()) {
			if (keys[slot] == name) {
				return index[slot];
			}
			if (keys[slot] == nullptr) {
				const auto n = size.load(std::memory_order_relaxed);
				if (n == Capacity) {
					return Capacity - 1;
				}
				keys[slot]  = name;
				index[slot] = static_cast<uint16_t>(n);
				names[n]    = name;
				size.store(n + 1, std::memory_order_release);
				return n;
			}
		}
	}
};

void AddPhase(std::vector<std::pair<const char*, PhaseTotals>>& list, const char* name,
              uint64_t ns, uint64_t count) {
	// Phase names are string literals: pointer identity is enough on the hot path.
	auto it = std::find_if(list.begin(), list.end(), [&](const auto& entry) { return entry.first == name; });
	if (it == list.end()) {
		list.emplace_back(name, PhaseTotals {});
		it = list.end() - 1;
	}
	it->second.ns += ns;
	it->second.count += count;
}
std::mutex                                 g_phase_mutex;
std::vector<std::shared_ptr<ThreadPhases>> g_phase_threads;
thread_local std::shared_ptr<ThreadPhases> t_phases;
thread_local const char*                   t_phase = nullptr;
thread_local uint64_t                      t_phase_start = 0;
// Phase marks read the TSC (invariant on the supported CPUs): a QueryPerformanceCounter call per
// mark was a large part of the profiler's own cost.
std::atomic<double> g_ns_per_tick {0.0};

double CalibrateTsc() {
	const auto clock0 = std::chrono::steady_clock::now();
	const auto tsc0   = __rdtsc();
	while (std::chrono::steady_clock::now() - clock0 < std::chrono::milliseconds(20)) {
	}
	const auto ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - clock0)
	                    .count();
	return ns / static_cast<double>(__rdtsc() - tsc0);
}

void PhaseReport() {
	using Clock   = std::chrono::steady_clock;
	auto previous = Clock::now();
	auto frames0  = g_phase_frames.load();
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(5));
		const auto now     = Clock::now();
		const auto seconds = std::chrono::duration<double>(now - previous).count();
		const auto frames  = g_phase_frames.load();
		const auto nframes = static_cast<double>(std::max<uint64_t>(frames - frames0, 1));
		previous = now;
		frames0  = frames;
		std::vector<std::pair<const char*, PhaseTotals>> totals;
		std::vector<std::shared_ptr<ThreadPhases>>       threads;
		{
			std::lock_guard lock(g_phase_mutex);
			threads = g_phase_threads;
		}
		for (auto& thread: threads) {
			std::vector<std::pair<const char*, PhaseTotals>> part;
			const auto size = thread->size.load(std::memory_order_acquire);
			for (uint32_t i = 0; i < size; i++) {
				auto& entry = thread->totals[i];
				if (entry.count.load(std::memory_order_relaxed) != 0) {
					PhaseTotals t;
					t.count = entry.count.exchange(0, std::memory_order_relaxed);
					t.ns    = static_cast<uint64_t>(
                        static_cast<double>(entry.ns.exchange(0, std::memory_order_relaxed)) *
                        g_ns_per_tick.load(std::memory_order_relaxed));
					part.emplace_back(thread->names[i], t);
				}
			}
			for (const auto& [name, t]: part) {
				auto it = std::find_if(totals.begin(), totals.end(),
				                       [&](const auto& entry) { return entry.first == name; });
				if (it == totals.end()) {
					totals.emplace_back(name, t);
				} else {
					it->second.ns += t.ns;
					it->second.count += t.count;
				}
			}
		}
		std::sort(totals.begin(), totals.end(),
		          [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
		double all_ms = 0;
		for (const auto& [name, t]: totals) {
			all_ms += static_cast<double>(t.ns) / 1e6;
		}
		{
			const auto wall = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			std::tm    local {};
			localtime_s(&local, &wall);
			std::printf("==== phase profile %02d:%02d:%02d: %.1f s, %.0f frames (%.2f fps), GPU thread "
			            "marked %.1f ms/frame\n",
			            local.tm_hour, local.tm_min, local.tm_sec, seconds, nframes, nframes / seconds,
			            all_ms / nframes);
		}
		{
			std::vector<float> times;
			EventList          events;
			{
				std::lock_guard lock(g_event_mutex);
				times.swap(g_frame_times_ms);
				events = TakeEvents(g_period_events);
			}
			if (!times.empty()) {
				std::sort(times.begin(), times.end());
				const auto pct = [&](double p) {
					return times[std::min(times.size() - 1,
					                      static_cast<size_t>(p * static_cast<double>(times.size() - 1) + 0.5))];
				};
				double sum    = 0;
				size_t over33 = 0, over50 = 0, over100 = 0;
				for (const auto t: times) {
					sum += t;
					over33 += t > 33.4f ? 1 : 0;
					over50 += t > 50.0f ? 1 : 0;
					over100 += t > 100.0f ? 1 : 0;
				}
				std::printf("  frame times: n %zu avg %.1f p50 %.1f p90 %.1f p99 %.1f max %.1f ms;"
				            " >33ms %zu >50ms %zu >100ms %zu\n",
				            times.size(), sum / static_cast<double>(times.size()), pct(0.5), pct(0.9),
				            pct(0.99), times.back(), over33, over50, over100);
			}
			for (const auto& [tag, t]: events) {
				std::printf("  event %-48s %8.2f ms/frame %8.1f /frame  avg %8.1f us\n", tag.c_str(),
				            static_cast<double>(t.ns) / 1e6 / nframes,
				            static_cast<double>(t.count) / nframes,
				            t.count != 0 ? static_cast<double>(t.ns) / 1e3 / static_cast<double>(t.count)
				                         : 0.0);
			}
		}
		{
			static const char* names[6] = {"other", "cp", "exec", "rec", "res", "copy"};
			std::printf("  waits/frame:");
			for (int i = 0; i < 6; i++) {
				const auto spins  = Common::WaitStats::spins[i].exchange(0);
				const auto kernel = Common::WaitStats::kernel_waits[i].exchange(0);
				const auto gpu    = Common::WaitStats::gpu_waits[i].exchange(0);
				std::printf(" %s spin %.0f kernel %.1f gpu %.1f;", names[i], spins / nframes,
				            kernel / nframes, gpu / nframes);
			}
			std::printf("\n");
		}
		for (const auto& [name, t]: totals) {
			const auto ms = static_cast<double>(t.ns) / 1e6;
			std::printf("  %-28s %8.2f ms/frame %5.1f%%  calls/frame %8.1f  avg %8.2f us\n", name,
			            ms / nframes, 100.0 * ms / std::max(all_ms, 1e-9),
			            static_cast<double>(t.count) / nframes,
			            t.count != 0 ? static_cast<double>(t.ns) / 1e3 / static_cast<double>(t.count)
			                         : 0.0);
		}
		std::fflush(stdout);
	}
}
} // namespace

const char* CurrentPhase() noexcept {
	return t_phase;
}

void PhaseMarkImpl(const char* name) noexcept {
	// std::call_once enters InitOnceBeginInitialize on every call: check a flag first.
	static std::atomic_bool started {false};
	if (!started.load(std::memory_order_acquire)) {
		static std::once_flag reporter;
		std::call_once(reporter, [] {
			g_ns_per_tick.store(CalibrateTsc());
			std::thread(PhaseReport).detach();
		});
		started.store(true, std::memory_order_release);
	}
	const auto now = __rdtsc();
	if (t_phases == nullptr) {
		t_phases = std::make_shared<ThreadPhases>();
		std::lock_guard lock(g_phase_mutex);
		g_phase_threads.push_back(t_phases);
	}
	if (t_phase != nullptr) {
		// Only this thread writes its entries: plain read-modify-write without a locked
		// instruction (an exchange of the reporter racing with it may lose one mark). The entries
		// hold TSC ticks; reports convert them.
		const auto ticks = now - t_phase_start;
		const auto index = t_phases->Find(t_phase);
		auto&      total = t_phases->totals[index];
		total.ns.store(total.ns.load(std::memory_order_relaxed) + ticks, std::memory_order_relaxed);
		total.count.store(total.count.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
		auto& frame = t_phases->frame[index];
		frame.ns.store(frame.ns.load(std::memory_order_relaxed) + ticks, std::memory_order_relaxed);
		frame.count.store(frame.count.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
		if (ExecTrace::Enabled()) {
			ExecTrace::OnPhase(t_phase, static_cast<uint64_t>(
			                                static_cast<double>(ticks) *
			                                g_ns_per_tick.load(std::memory_order_relaxed)));
		}
	}
	t_phase       = name;
	t_phase_start = now;
}

namespace {
// Frames slower than KYTY_HITCH_MS (default 50) print what every profiled thread did since the
// previous flip, the host events and the cache counters of that frame.
double HitchMs() {
	static const double value = [] {
		const char* env = std::getenv("KYTY_HITCH_MS");
		return env != nullptr ? std::atof(env) : 50.0;
	}();
	return value;
}

void FrameBoundary() noexcept {
	using Clock = std::chrono::steady_clock;
	static Clock::time_point last;
	static uint64_t          frame = 0;
	static uint64_t          counters[CacheStats::Count] {};
	const auto               now = Clock::now();

	std::vector<std::pair<const char*, PhaseTotals>> phases;
	std::vector<std::shared_ptr<ThreadPhases>>       threads;
	{
		std::lock_guard lock(g_phase_mutex);
		threads = g_phase_threads;
	}
	for (auto& thread: threads) {
		const auto size = thread->size.load(std::memory_order_acquire);
		for (uint32_t i = 0; i < size; i++) {
			auto& t = thread->frame[i];
			if (t.count.load(std::memory_order_relaxed) != 0) {
				const auto count = t.count.exchange(0, std::memory_order_relaxed);
				const auto ns    = static_cast<uint64_t>(
                    static_cast<double>(t.ns.exchange(0, std::memory_order_relaxed)) *
                    g_ns_per_tick.load(std::memory_order_relaxed));
				AddPhase(phases, thread->names[i], ns, count);
			}
		}
	}
	EventList events;
	{
		std::lock_guard lock(g_event_mutex);
		events = TakeEvents(g_frame_events);
	}
	uint64_t delta[CacheStats::Count] {};
	for (uint32_t i = 0; i < CacheStats::Count; i++) {
		const auto value = CacheStats::g_counters[i].load(std::memory_order_relaxed);
		delta[i]         = value - counters[i];
		counters[i]      = value;
	}

	frame++;
	if (last == Clock::time_point {}) {
		last = now;
		return;
	}
	const double ms = std::chrono::duration<double, std::milli>(now - last).count();
	last            = now;
	{
		std::lock_guard lock(g_event_mutex);
		g_frame_times_ms.push_back(static_cast<float>(ms));
	}
	if (ms < HitchMs()) {
		return;
	}

	std::sort(phases.begin(), phases.end(),
	          [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
	const auto wall   = std::chrono::system_clock::now();
	const auto time   = std::chrono::system_clock::to_time_t(wall);
	const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(wall.time_since_epoch()).count() %
	                    1000;
	std::tm local {};
	localtime_s(&local, &time);
	std::printf("==== hitch %02d:%02d:%02d.%03d frame %" PRIu64 ": %.1f ms\n", local.tm_hour,
	            local.tm_min, local.tm_sec, static_cast<int>(millis), frame, ms);
	std::printf("  phases:");
	for (size_t i = 0; i < phases.size() && i < 14; i++) {
		if (phases[i].second.ns < 500000) {
			break;
		}
		std::printf(" %s %.1f ms/%" PRIu64 ";", phases[i].first,
		            static_cast<double>(phases[i].second.ns) / 1e6, phases[i].second.count);
	}
	std::printf("\n  events:");
	for (size_t i = 0; i < events.size() && i < 14; i++) {
		const auto& [tag, t] = events[i];
		if (t.ns != 0) {
			std::printf(" %s %.1f ms/%" PRIu64 ";", tag.c_str(), static_cast<double>(t.ns) / 1e6,
			            t.count);
		} else {
			std::printf(" %s x%" PRIu64 ";", tag.c_str(), t.count);
		}
	}
	std::printf("\n  cache: submits=%" PRIu64 " wfaults=%" PRIu64 " rfaults=%" PRIu64
	            " readbacks=%" PRIu64 " readback_ms=%.1f tex_uploads=%" PRIu64
	            " tex_upload_mb=%.1f bda_syncs=%" PRIu64 " bda_ms=%.1f tex_gc=%" PRIu64
	            " buf_gc=%" PRIu64 "\n",
	            delta[CacheStats::Submissions], delta[CacheStats::WriteFaults],
	            delta[CacheStats::ReadFaults], delta[CacheStats::ReadbackDownloads],
	            static_cast<double>(delta[CacheStats::ReadbackWaitUs]) / 1e3,
	            delta[CacheStats::TextureUploads],
	            static_cast<double>(delta[CacheStats::TextureUploadBytes]) / (1024.0 * 1024.0),
	            delta[CacheStats::BdaSyncs], static_cast<double>(delta[CacheStats::BdaSyncUs]) / 1e3,
	            delta[CacheStats::TextureGcFrees], delta[CacheStats::BufferGcFrees]);
	std::fflush(stdout);
}
} // namespace

const char* CurrentWaitTag() noexcept {
	return g_wait_tag;
}

WaitTagScope::WaitTagScope(const char* tag) noexcept: m_previous(g_wait_tag) {
	g_wait_tag = tag;
}

WaitTagScope::~WaitTagScope() {
	g_wait_tag = m_previous;
}

} // namespace Libs::Graphics::GpuProfiler
