#include "common/syncAcct.h"
#include <map>
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/gpuProfiler.h"
#include "graphics/guest_gpu/execTrace.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/expSwitch.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/spscQueue.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/cacheStats.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write,
                                        bool wait = true) {
	const auto [mapped, offset] = staging.Map(size, alignment, wait);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferHostOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_register_epoch++;
		m_buffer_set_epoch++;
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		m_buffer_set_epoch++;
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

// Debug: sizes of recently deleted buffers (would a size-keyed pool find one to reuse?).
static std::array<uint64_t, 256> g_dbg_deleted_sizes {};
static size_t                    g_dbg_deleted_next = 0;

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	if (GpuProfiler::PhaseEnabled()) {
		g_dbg_deleted_sizes[g_dbg_deleted_next++ % g_dbg_deleted_sizes.size()] = m_slot_buffers[id].Size();
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferHostOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

std::map<uint64_t, std::pair<uint64_t, uint64_t>>& DebugWriteTicks();
uint64_t DebugLastWriteTick(uint64_t vaddr);

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	GpuProfiler::AddAddress("download bytes by buffer", buffer_address, total_size);

	if constexpr (!async) {
		if (DownloadOnReadbackQueue(buffer, copies, total_size)) {
			return true;
		}
	}
	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	GpuProfiler::Region(native, true, "buffer download copy");
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());
	GpuProfiler::Region(native, false, "buffer download copy");

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	uint64_t written_begin = UINT64_MAX;
	uint64_t written_end   = 0;
	for (const auto& copy: copies) {
		written_begin = std::min(written_begin, buffer_address + copy.srcOffset);
		written_end   = std::max(written_end, buffer_address + copy.srcOffset + copy.size);
	}
	auto publish = [this, mapped, offset, total_size, buffer_address,
	                copies = std::move(copies), owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish), written_begin,
		                                   written_end - written_begin);
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	m_scheduler.SetSubmitHook(
	    [](void* self, CommandBuffer& command) {
		    static_cast<BufferCache*>(self)->RecordMirrorCopies(command);
	    },
	    this);
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	m_scheduler.SetSubmitHook(nullptr, nullptr);
	if (m_readback.pool) {
		m_graphics.device.destroyFence(m_readback.fence);
		m_graphics.device.destroyCommandPool(m_readback.pool);
	}
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

static bool KnownFillsEnabled() {
	static const bool enabled = std::getenv("KYTY_NO_KNOWN_FILL") == nullptr;
	return enabled;
}

void BufferCache::NoteKnownFill(uint64_t vaddr, uint64_t size, uint32_t value) {
	if (!KnownFillsEnabled() || size == 0) {
		return;
	}
	ForgetKnownFills(vaddr, size);
	m_known_fills[vaddr] = {vaddr + size, value};
	GpuProfiler::AddCount("known fill noted");
}

bool BufferCache::KnownFill(uint64_t vaddr, uint64_t size, uint32_t& value) const {
	auto it = m_known_fills.upper_bound(vaddr);
	if (it == m_known_fills.begin()) {
		return false;
	}
	--it;
	if (it->first > vaddr || it->second.first < vaddr + size) {
		return false;
	}
	value = it->second.second;
	return true;
}

std::string BufferCache::DescribeKnownFills(uint64_t vaddr, uint64_t size) const {
	std::string text;
	auto        it = m_known_fills.upper_bound(vaddr);
	if (it != m_known_fills.begin()) {
		--it;
	}
	for (; it != m_known_fills.end() && it->first < vaddr + size; ++it) {
		if (it->second.first <= vaddr) continue;
		char line[96];
		std::snprintf(line, sizeof(line), " [0x%llx,0x%llx)=0x%x", static_cast<unsigned long long>(it->first),
		              static_cast<unsigned long long>(it->second.first), it->second.second);
		text += line;
	}
	return text.empty() ? std::string(" none") : text;
}

void BufferCache::ForgetKnownFills(uint64_t vaddr, uint64_t size) {
	if (m_known_fills.empty()) {
		return;
	}
	auto it = m_known_fills.upper_bound(vaddr);
	if (it != m_known_fills.begin() && std::prev(it)->second.first > vaddr) {
		--it;
	}
	while (it != m_known_fills.end() && it->first < vaddr + size) {
		it = m_known_fills.erase(it);
	}
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	ForgetKnownFills(vaddr, size);
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
}

void BufferCache::InvalidateFaultedPage(uint64_t vaddr) {
	// The guest fills its streaming memory (vertex and constant rings) upwards, and every page it
	// writes under a cached buffer takes a fault: thousands per frame, each a system call to drop
	// the protection. When the page below is already writable the guest is filling the range
	// upwards, so release the pages ahead up to the next 64 KB boundary with this fault; they are
	// only uploaded again at their next synchronization. A range with pages the GPU wrote must be
	// read back first, so it keeps to the faulting page.
	static constexpr uint64_t FaultBlockSize = 64 * 1024;
	static_assert(TRACKER_REGION_SIZE % FaultBlockSize == 0);
	const auto page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end  = Common::AlignDown(page, FaultBlockSize) + FaultBlockSize;
	if (page >= TRACKER_PAGE_SIZE && GuestRange {page - TRACKER_PAGE_SIZE, end - page}.Valid() &&
	    m_memory_tracker.IsRegionCpuModified(page - TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE) &&
	    !m_memory_tracker.IsRegionGpuModified(page, end - page)) {
		InvalidateMemory(page, end - page);
		return;
	}
	InvalidateMemory(vaddr, 1);
}

static bool AsyncReadbackEnabled() {
	// Experimental, opt-in: measured slower than the synchronous readback in GTA V.
	static const bool enabled = std::getenv("KYTY_ASYNC_READBACK") != nullptr;
	return enabled;
}

uint64_t BufferCache::PendingReadbackTick(uint64_t vaddr, uint64_t size) const {
	uint64_t tick = 0;
	for (const auto& pending: m_pending_readbacks) {
		if (vaddr < pending.end && pending.begin < vaddr + size) {
			tick = std::max(tick, pending.tick);
		}
	}
	return tick;
}

void BufferCache::FinalizeReadbacks() {
	EXIT_IF(!GuestGpu::IsGpuThread());
	for (auto it = m_pending_readbacks.begin(); it != m_pending_readbacks.end();) {
		if (!m_scheduler.IsFree(it->tick)) {
			++it;
			continue;
		}
		// The priority runner publishes the backing once the tick has completed.
		m_scheduler.WaitPriorityOperations(it->tick);
		for (const auto& [start, end]: it->ranges) {
			if (!m_readback_rewritten.Intersects(start, end - start)) {
				m_gpu_modified_ranges.Subtract(start, end - start);
				continue;
			}
			RangeSet clean;
			clean.Add(start, end - start);
			m_readback_rewritten.ForEachInRange(start, end - start, [&](uint64_t b, uint64_t e) {
				clean.Subtract(b, e - b);
			});
			clean.ForEach([&](uint64_t b, uint64_t e) { m_gpu_modified_ranges.Subtract(b, e - b); });
		}
		// Publish the pages without GPU-dirty bytes, as the synchronous path does for its whole
		// window.
		const auto begin = Common::AlignDown(it->begin, TRACKER_PAGE_SIZE);
		const auto end   = Common::AlignUp(it->end, TRACKER_PAGE_SIZE);
		uint64_t   run   = 0;
		bool       open  = false;
		for (auto page = begin; page <= end; page += TRACKER_PAGE_SIZE) {
			const bool clean =
			    page < end && !m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE);
			if (clean && !open) {
				run  = page;
				open = true;
			} else if (!clean && open) {
				m_memory_tracker.UnmarkRegionAsGpuModified(run, page - run);
				open = false;
			}
		}
		it = m_pending_readbacks.erase(it);
	}
	if (m_pending_readbacks.empty()) {
		m_readback_rewritten.Clear();
	}
}

void BufferCache::ReadMemoryAsync(uint64_t vaddr, uint64_t size) {
	GpuProfiler::TimedWait timed_readback("game: CPU read of GPU-dirty page (async)");
	auto&    gpu       = m_scheduler.Context().GetGpu();
	uint64_t wait_tick = 0;
	gpu.SendCommandSync([this, vaddr, size, &wait_tick] {
		GpuProfiler::WaitTagScope tag("gpu-thread: readback drain");
		FinalizeReadbacks();
		if (const auto tick = PendingReadbackTick(vaddr, size); tick != 0) {
			wait_tick = tick;
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
		if (GpuProfiler::Enabled()) {
			GpuProfiler::AddAddress("cpu read fault page", vaddr & ~uint64_t {0xfff});
			GpuProfiler::AddAddress("fault last writer shader", GpuProfiler::LastGpuWriter(vaddr));
		}
		constexpr uint64_t WindowSize   = 512 * 1024;
		const auto         buffer_begin = buffer.CpuAddress();
		const auto         buffer_end   = buffer_begin + buffer.Size();
		const auto window_begin = std::max(Common::AlignDown(vaddr, WindowSize), buffer_begin);
		const auto window_end =
		    std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);
		PendingReadback pending;
		EXIT("KYTY_ASYNC_READBACK is not available on this branch\n");
		pending.begin = window_begin;
		pending.end   = window_end;
		pending.tick  = m_scheduler.CurrentTick();
		wait_tick     = pending.tick;
		m_pending_readbacks.push_back(std::move(pending));
		// Submit without waiting: the GPU thread keeps processing commands meanwhile.
		m_scheduler.Flush();
	});
	if (wait_tick == 0) {
		return;
	}
	{
		GpuProfiler::TimedWait timed("game: async readback GPU wait");
		m_scheduler.GetMasterSemaphore().Wait(wait_tick);
		m_scheduler.WaitPriorityOperations(wait_tick);
	}
	gpu.SendCommandSync([this] { FinalizeReadbacks(); });
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	if (!is_write && !GuestGpu::IsGpuThread() && AsyncReadbackEnabled()) {
		ReadMemoryAsync(vaddr, size);
		return;
	}
	if (GuestGpu::IsGpuThread() && GpuProfiler::PhaseEnabled()) {
		// Debug: the execution-thread phase that read GPU-written memory.
		static std::mutex                                   names_mutex;
		static std::unordered_map<const void*, std::string> names;
		const char*                                         phase = GpuProfiler::CurrentPhase();
		std::lock_guard                                     lock(names_mutex);
		auto& name = names[phase];
		if (name.empty()) {
			name = std::string("readback in ") + (phase != nullptr ? phase : "?");
		}
		GpuProfiler::AddCount(name.c_str());
	}
	GpuProfiler::TimedWait timed_readback(
	    GuestGpu::IsGpuThread()
	        ? (is_write ? "gpu-thread: write to GPU-dirty page (sync)"
	                    : "gpu-thread: read of GPU-dirty page (sync)")
	        : (is_write ? "game: CPU write to GPU-dirty page (sync)"
	                    : "game: CPU read of GPU-dirty page (sync)"));
	// The guest reads (or writes) memory the GPU wrote only after a label the execution thread
	// writes once the work before it executed: the operations queued since cannot be what the
	// access needs, so the readback runs before them instead of after the whole backlog (a
	// frame of draws, 100-200 ms while an area streams in).
	m_scheduler.Context().GetGpu().SendUrgentCommandSync([this, vaddr, size, is_write] {
		GpuProfiler::WaitTagScope tag("gpu-thread: readback drain");
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
		if (GpuProfiler::Enabled() && GuestGpu::IsGpuThread()) {
			const auto last = DebugLastWriteTick(vaddr);
			const auto now  = m_scheduler.CurrentTick();
			const auto done = m_scheduler.GetMasterSemaphore().KnownGpuTick();
			GpuProfiler::AddCount(last == 0 ? "rb tick: unknown" : last <= done ? "rb tick: writer done" : last < now ? "rb tick: writer submitted" : "rb tick: writer in current");
			static uint32_t printed = 0;
			if (printed++ < 40) {
				std::printf("rb 0x%llx writer tick %llu now %llu gpu done %llu\n", static_cast<unsigned long long>(vaddr),
				            static_cast<unsigned long long>(last), static_cast<unsigned long long>(now), static_cast<unsigned long long>(done));
			}
		}
		if (GpuProfiler::Enabled()) {
			GpuProfiler::AddAddress(is_write ? "cpu write fault page" : "cpu read fault page",
			                        vaddr & ~uint64_t {0xfff});
			GpuProfiler::AddAddress("fault buffer base (count)", buffer.CpuAddress());
			GpuProfiler::AddAddress("fault last writer shader",
			                        GpuProfiler::LastGpuWriter(vaddr));
			GpuProfiler::AddAddress("fault buffer size by base", buffer.CpuAddress(),
			                        buffer.Size());
		}

		// KYTY_RB_WINDOW_KB: widen nearby CPU reads so they share one GPU drain (A/B).
		static const uint64_t WindowSize = [] {
			const char* value = std::getenv("KYTY_RB_WINDOW_KB");
			return value != nullptr ? std::strtoull(value, nullptr, 10) * 1024 : uint64_t {0};
		}();
		const auto buffer_begin = buffer.CpuAddress();
		const auto buffer_end   = buffer_begin + buffer.Size();
		auto       window_begin = vaddr;
		auto       window_end   = vaddr + size;
		if (WindowSize != 0) {
			window_begin = std::max(Common::AlignDown(vaddr, WindowSize), buffer_begin);
			window_end   = std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);
		}

		const auto begin = std::chrono::steady_clock::now();
		if (DownloadBufferMemory<false>(buffer, window_begin, window_end - window_begin)) {
			CacheStats::Add(CacheStats::ReadbackDownloads);
			const bool indirect =
			    CacheStats::g_enabled && CacheStats::g_indirect_targets.Intersects(vaddr, size);
			if (indirect) {
				CacheStats::Add(CacheStats::ReadbackIndirect);
			}
			if (CacheStats::g_enabled && CacheStats::g_storage_written.Intersects(vaddr, size)) {
				CacheStats::Add(CacheStats::ReadbackStorage);
			}
			if (CacheStats::g_enabled && CacheStats::g_dynamic_written.Intersects(vaddr, size)) {
				CacheStats::Add(CacheStats::ReadbackDynamic);
			}
			if (CacheStats::g_enabled && CacheStats::g_dma_written.Intersects(vaddr, size)) {
				CacheStats::Add(CacheStats::ReadbackDma);
			}
			static uint32_t logged = 0;
			static uint32_t seen = 0;
			if (CacheStats::g_enabled && (seen++ % 300) == 0 && logged < 60) {
				logged++;
				std::printf("CacheStats readback addr=0x%016" PRIx64 " write=%d buffer=0x%016" PRIx64
				            "+0x%" PRIx64 " window=0x%016" PRIx64 "+0x%" PRIx64
				            " indirect=%d storage=%d dynamic=%d dma=%d\n",
				            vaddr, is_write, buffer_begin, buffer.Size(), window_begin,
				            window_end - window_begin, indirect,
				            CacheStats::g_storage_written.Intersects(vaddr, size),
				            CacheStats::g_dynamic_written.Intersects(vaddr, size),
				            CacheStats::g_dma_written.Intersects(vaddr, size));
			}
			CacheStats::Add(CacheStats::ReadbackWaitUs,
			                static_cast<uint64_t>(
			                    std::chrono::duration_cast<std::chrono::microseconds>(
			                        std::chrono::steady_clock::now() - begin)
			                        .count()));
			m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_end - window_begin);
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
	});
}

// Debug: the recording tick of the last GPU write that started at each address.
std::map<uint64_t, std::pair<uint64_t, uint64_t>>& DebugWriteTicks() {
	static std::map<uint64_t, std::pair<uint64_t, uint64_t>> ticks;
	return ticks;
}

uint64_t DebugLastWriteTick(uint64_t vaddr) {
	auto& ticks = DebugWriteTicks();
	uint64_t best = 0;
	auto it = ticks.upper_bound(vaddr);
	for (int n = 0; it != ticks.begin() && n < 64; n++) {
		--it;
		if (it->second.first > vaddr) {
			best = std::max(best, it->second.second);
		}
	}
	return best;
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

void BufferCache::CacheFaultedMemory(uint64_t vaddr, uint64_t size) {
	if (const auto id = FindBuffer(vaddr, size)) {
		m_slot_buffers[id].device_address_pinned = true;
	}
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.device_address_pinned |= overlap.device_address_pinned;
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	NoteWrite(overlap.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	GpuProfiler::TimedWait timed("ev: buffer create");
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);
	if (GpuProfiler::PhaseEnabled()) {
		const auto want = overlap.end - overlap.begin;
		const auto it   = std::find(g_dbg_deleted_sizes.begin(), g_dbg_deleted_sizes.end(), want);
		if (it != g_dbg_deleted_sizes.end()) {
			*it = 0;
			GpuProfiler::AddCount("bufpool: same size deleted recently");
		} else {
			GpuProfiler::AddCount("bufpool: no match");
		}
	}

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (source && m_upload_batching && source == m_staging_buffer.Handle()) {
		GpuProfiler::AddCount("sync: upload", 1);
		GpuProfiler::AddCount("sync: upload bytes", total_size);
		GpuProfiler::AddCount("sync: batched upload", 1);
		m_upload_batch_dsts.emplace_back(buffer.Handle(), static_cast<uint32_t>(copies.size()));
		m_upload_batch_copies.insert(m_upload_batch_copies.end(), copies.begin(), copies.end());
	} else if (source) {
		GpuProfiler::AddCount("sync: upload", 1);
		GpuProfiler::AddCount("sync: upload bytes", total_size);
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		GpuProfiler::Region(native, true, "buffer upload copy");
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		GpuProfiler::Region(native, false, "buffer upload copy");
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}

	// Within a batch of uploads (see FlushUploadBatch()) the staging buffer must not wait for
	// the current command buffer: that would submit it without the batched copies. A copy that
	// does not fit then goes through a buffer of its own.
	// As StreamGuestData(): larger copies go to the host copy thread, from the backing store
	// (never protected). The submission of the command buffer waits for them.
	auto*      host_copies = m_scheduler.HostCopies();
	const bool async       = KYTY_XO(ASYNC_UPLOAD) && host_copies != nullptr &&
	                   m_staging_buffer.IsCoherent() && GuestGpu::IsGpuThread();
	// Debug (rain-perf ablation): no guest bytes copied to the staging buffer.
	const bool no_copy = KYTY_ABL_AT(NO_UPLOAD_COPY);
	const auto source = UploadMemory(m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		for (auto& copy: copies) {
			if (no_copy) {
				copy.srcOffset += base_offset;
				continue;
			}
			const auto  address = buffer.CpuAddress() + copy.dstOffset;
			const void* backing = async && copy.size >= 1024
			                          ? Libs::LibKernel::Memory::FindBackingPointer(address, copy.size)
			                          : nullptr;
			if (backing != nullptr) {
				host_copies->Copy(mapped + copy.srcOffset, backing, copy.size);
			} else if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset,
			                                                copy.size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			copy.srcOffset += base_offset;
		}
	}, !m_upload_batching);
	return source.first->Handle();
}

void BufferCache::StreamGuestData(uint8_t* destination, uint64_t vaddr, uint64_t size) {
	// The execution thread hands its copies to the host copy thread (see HostCopyQueue: writing
	// the stream buffer itself would stall it on its next fence). The copy reads the backing
	// store, which holds the same bytes as the guest range and is never protected, so a copy that
	// runs after a later draw protected the range for GPU writes does not fault.
	auto* copies = m_scheduler.HostCopies();
	if (copies != nullptr && m_stream_buffer.IsCoherent() && GuestGpu::IsGpuThread()) {
		if (const void* source = Libs::LibKernel::Memory::FindBackingPointer(vaddr, size)) {
			copies->Copy(destination, source, size);
			return;
		}
	}
	// Backing reads mostly hit the per-thread mapping cache; sparse ranges with holes or private
	// mappings take the full copy.
	if (!LibKernel::Memory::TryReadBacking(vaddr, destination, size) &&
	    !LibKernel::Memory::CopySparseMemory(vaddr, destination, size)) {
		EXIT("BufferCache: failed to read sparse guest buffer memory\n");
	}
}

uint64_t BufferCache::StreamUpload(const void* data, uint64_t size, uint64_t alignment) {
	auto* copies = m_scheduler.HostCopies();
	if (copies == nullptr || !m_stream_buffer.IsCoherent() || !GuestGpu::IsGpuThread()) {
		return m_stream_buffer.Copy(data, size, alignment);
	}
	const auto [mapped, offset] = m_stream_buffer.Map(size, alignment);
	EXIT_IF(mapped == nullptr);
	copies->CopyInline(mapped, data, size);
	m_stream_buffer.Commit();
	return offset;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	// Draws stream the same small buffers (frame and pass constants) again and again. Within a
	// draw window the guest cannot have written them since the last copy, so draws share it while
	// it is in the stream buffer and no work wrote cached buffers. A copy in the window answers
	// before the dirty checks: it was made from CPU-dirty pages, and within the window the guest
	// neither wrote them nor did they become GPU dirty (that ends the window or advances the GPU
	// write generation, both checked).
	const bool  small_read = !is_written && size <= CACHING_PAGESIZE;
	StreamMemo* stream     = nullptr;
	uint64_t    window     = 0;
	if (small_read && !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		window = m_scheduler.Context().GetRenderExecutor().DrawWindow();
		stream = &(*m_stream_memo)[((vaddr ^ (size << 40u)) * 0x9e3779b97f4a7c15ull >> 32u) %
		                           StreamMemoSlots];
		if (KYTY_X(RP_STREAMFIRST) && stream->vaddr == vaddr && stream->size == size &&
		    stream->window == window && stream->lap == m_stream_buffer.Lap() &&
		    stream->write_generation == m_gpu_write_generation) {
			GpuProfiler::AddCount("stream: reused");
			return {&m_stream_buffer, stream->offset};
		}
	}

	// Draws bind the same read-only ranges over and over. While the range stays in the same
	// buffer and none of its pages became CPU dirty since a lookup found it clean (or uploaded
	// it), the lookup finds the same buffer and has nothing to upload. The epoch is read before
	// the dirty check: a page dirtied after it advances it.
	const bool  plain_read = !is_written && !is_texel_buffer;
	ObtainMemo* memo       = nullptr;
	uint64_t    cpu_epoch  = 0;
	if (plain_read) {
		const auto hash = (vaddr ^ (size << 40u)) * 0x9e3779b97f4a7c15ull;
		memo            = &(*m_obtain_memo)[(hash >> 32u) % ObtainMemoSlots];
		cpu_epoch       = CpuDirtyEpoch(vaddr, size);
		if (memo->vaddr == vaddr && memo->size == size &&
		    memo->buffer_epoch == m_buffer_set_epoch && memo->cpu_epoch == cpu_epoch) {
			auto& buffer = m_slot_buffers[memo->id];
			TouchBuffer(buffer);
			return {&buffer, memo->offset};
		}
	}

	if (stream != nullptr && m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		// Debug (rain-perf A/B, KYTY_X_RP_STREAMFIRST=0): the reuse check after the dirty checks.
		if (!KYTY_X(RP_STREAMFIRST) && stream->vaddr == vaddr && stream->size == size &&
		    stream->window == window && stream->lap == m_stream_buffer.Lap() &&
		    stream->write_generation == m_gpu_write_generation) {
			GpuProfiler::AddCount("stream: reused");
			return {&m_stream_buffer, stream->offset};
		}
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped == nullptr) {
			GpuProfiler::AddCount("obtain: stream map fail");
		}
		if (mapped != nullptr) {
			GpuProfiler::AddCount("stream: copied");
			GpuProfiler::AddCount("stream: copied bytes", size);
			StreamGuestData(mapped, vaddr, size);
			m_stream_buffer.Commit();
			*stream = {.vaddr            = vaddr,
			           .size             = size,
			           .window           = window,
			           .lap              = m_stream_buffer.Lap(),
			           .write_generation = m_gpu_write_generation,
			           .offset           = offset};
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (plain_read) {
		*memo = {.vaddr        = vaddr,
		         .size         = size,
		         .buffer_epoch = m_buffer_set_epoch,
		         .cpu_epoch    = cpu_epoch,
		         .offset       = buffer.Offset(vaddr),
		         .id           = id};
	}
	if (is_written) {
		m_gpu_write_generation++;
		m_gpu_modified_ranges.Add(vaddr, size);
		NoteWrite(vaddr, size);
		ForgetKnownFills(vaddr, size);
		if (!m_pending_readbacks.empty()) {
			m_readback_rewritten.Add(vaddr, size);
		}
		GpuProfiler::NoteGpuWrite(vaddr, size);
		if (GpuProfiler::Enabled()) {
			DebugWriteTicks()[vaddr] = {vaddr + size, m_scheduler.CurrentTick()};
		}
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::FillInternalMemory(uint64_t vaddr, uint64_t size, uint32_t value) {
	if (vaddr == 0 || (vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 ||
	    !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: internal fill range must be dword aligned\n");
	}
	if (!IsRegionGpuModified(vaddr, size)) {
		const std::vector<uint32_t> words(size / sizeof(uint32_t), value);
		InvalidateMemory(vaddr, size);
		Libs::LibKernel::Memory::WriteBacking(vaddr, words.data(), size);
		return;
	}
	// Publishing through the backing would first read the GPU-owned bytes back.
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	m_gpu_write_generation++;
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Invalidate cached buffers and images for the whole range up front, as the write
		// faults would page by page, so the fill below does not fault on every tracked page.
		InvalidateMemory(vaddr, size);
		m_texture_cache.InvalidateMemory(vaddr, size);
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	CacheStats::t_invalidate_source = CacheStats::ImagesMarkedFill;
	if (CacheStats::g_enabled) {
		CacheStats::g_dma_written.Add(vaddr, size);
	}
	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	m_gpu_write_generation++;
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		CacheStats::t_invalidate_source = CacheStats::ImagesMarkedCopy;
		if (CacheStats::g_enabled) {
			CacheStats::g_dma_written.Add(dst_vaddr, size);
		}
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

void BufferCache::WriteTicks::Assign(uint64_t begin, uint64_t end, uint64_t tick) {
	if (begin >= end) {
		return;
	}
	auto it = m_ranges.lower_bound(begin);
	// Most writes repeat a range of an earlier operation.
	if (it != m_ranges.end() && it->first == begin && it->second.end == end) {
		it->second.tick = tick;
		return;
	}
	if (it != m_ranges.begin()) {
		const auto previous = std::prev(it);
		if (previous->second.end > begin) {
			const auto right       = previous->second;
			previous->second.end = begin;
			it                     = m_ranges.emplace_hint(it, begin, right);
		}
	}
	while (it != m_ranges.end() && it->first < end) {
		if (it->second.end > end) {
			const auto tail = it->second;
			m_ranges.erase(it);
			m_ranges.emplace(end, tail);
			break;
		}
		it = m_ranges.erase(it);
	}
	m_ranges.emplace(begin, Range {end, tick});
}

uint64_t BufferCache::WriteTicks::Latest(uint64_t begin, uint64_t end) const {
	auto it = m_ranges.upper_bound(begin);
	if (it != m_ranges.begin()) {
		--it;
	}
	uint64_t latest = 0;
	uint64_t cursor = begin;
	for (; it != m_ranges.end() && it->first < end && cursor < end; ++it) {
		if (it->second.end <= cursor) {
			continue;
		}
		if (it->first > cursor) {
			return UINT64_MAX;
		}
		latest = std::max(latest, it->second.tick);
		cursor = it->second.end;
	}
	return cursor >= end ? latest : UINT64_MAX;
}

void BufferCache::NoteReadbackWait(uint64_t begin, uint64_t end, bool device_address,
                                   bool drained) {
	const auto count = ++m_readback_count;
	if (device_address && (drained || m_device_address_hot != 0)) {
		m_device_address_hot = count;
	}
	std::erase_if(m_readback_hot, [count](const HotReadback& hot) {
		return count - hot.last_hit >= HotReadbackLifetime;
	});
	begin = Common::AlignDown(begin, TRACKER_PAGE_SIZE);
	end   = Common::AlignUp(end, TRACKER_PAGE_SIZE);
	for (auto& hot: m_readback_hot) {
		if (begin < hot.end && hot.begin < end) {
			hot.begin    = std::min(hot.begin, begin);
			hot.end      = std::max(hot.end, end);
			hot.last_hit = count;
			return;
		}
	}
	if (drained && !device_address) {
		if (m_readback_hot.size() == MaxHotReadbacks) {
			m_readback_hot.erase(std::ranges::min_element(m_readback_hot, {}, &HotReadback::last_hit));
		}
		m_readback_hot.push_back({begin, end, count});
	}
}

bool BufferCache::CommitWriteTicks() {
	if (m_pending_writes.empty() && !m_device_address_pending) {
		return std::exchange(m_flush_requested, false);
	}
	// The operation's commands are all recorded: they run in the current command buffer or in
	// one submitted while it was recorded.
	const auto tick = m_scheduler.CurrentTick();
	for (const auto& [vaddr, size]: m_pending_writes) {
		m_write_ticks.Assign(vaddr, vaddr + size, tick);
	}
	// Device-address stores name no range: a mirror copy older than them is not used (see
	// TryReadMirroredGpuWrites()).
	if (!m_mirror_lines.empty()) {
		for (const auto& [vaddr, size]: m_pending_writes) {
			MarkMirrorLinesWritten(vaddr, size, tick);
		}
	}
	m_pending_writes.clear();
	if (m_device_address_pending) {
		if (GpuProfiler::PhaseEnabled()) {
			char tag[64];
			std::snprintf(tag, sizeof(tag), "da writer %016llx",
			              static_cast<unsigned long long>(ExecTrace::g_shader_hash));
			GpuProfiler::AddWaitString(tag, 0);
			static uint32_t calls = 0;
			if ((calls++ & 63u) == 0) {
				uint64_t hot = 0;
				for (const auto& [address, line]: m_mirror_lines) {
					hot += m_mirror_uses - line.last_use < MirrorFlushLifetime ? 1u : 0u;
				}
				GpuProfiler::AddCount("da: hot mirror lines (x64 sample)", hot);
				GpuProfiler::AddCount("da: mirror lines (x64 sample)", m_mirror_lines.size());
			}
		}
		m_device_address_pending = false;
		m_device_address_tick    = tick;
		// Device-address stores name no range and may have written any mirrored line: copy the
		// lines read recently again at the end of this command buffer, so that reads after the
		// stores still find them in the mirror (see TryReadMirroredGpuWrites()).
		static const bool mirror_da = [] {
			const char* value = std::getenv("KYTY_MIRROR_DA");
			return value == nullptr || value[0] != '0';
		}();
		if (mirror_da) {
			for (auto& [address, line]: m_mirror_lines) {
				if (m_mirror_uses - line.last_use >= MirrorFlushLifetime) {
					continue;
				}
				line.tick = 0;
				// Read again soon: let the GPU start on the stores now.
				m_flush_requested = true;
				if (line.dirty_tick != tick) {
					line.dirty_tick = tick;
					m_mirror_dirty.push_back(address);
				}
			}
		}
	}
	return std::exchange(m_flush_requested, false);
}

void BufferCache::MarkMirrorLinesWritten(uint64_t vaddr, uint64_t size, uint64_t tick) {
	const auto end = size > UINT64_MAX - vaddr ? UINT64_MAX : vaddr + size;
	for (auto it = m_mirror_lines.lower_bound(Common::AlignDown(vaddr, MirrorLineSize));
	     it != m_mirror_lines.end() && it->first < end; ++it) {
		auto& line = it->second;
		line.tick  = 0;
		// A line read recently is read again soon: let the GPU start on its writer now.
		if (m_mirror_uses - line.last_use < MirrorFlushLifetime) {
			m_flush_requested = true;
		}
		if (line.dirty_tick != tick) {
			line.dirty_tick = tick;
			m_mirror_dirty.push_back(it->first);
		}
	}
}

void BufferCache::AddMirrorLine(uint64_t line) {
	if (m_mirror_free_slots.empty()) {
		if (m_mirror_lines.size() < MaxMirrorLines) {
			m_mirror_free_slots.push_back(static_cast<uint32_t>(m_mirror_lines.size()));
		} else {
			// Forget the half used least recently; their queued copies are skipped.
			std::vector<uint64_t> uses;
			uses.reserve(m_mirror_lines.size());
			for (const auto& [address, entry]: m_mirror_lines) {
				uses.push_back(entry.last_use);
			}
			std::nth_element(uses.begin(), uses.begin() + uses.size() / 2, uses.end());
			const auto oldest = uses[uses.size() / 2];
			for (auto it = m_mirror_lines.begin(); it != m_mirror_lines.end();) {
				if (it->second.last_use <= oldest) {
					m_mirror_free_slots.push_back(it->second.slot);
					it = m_mirror_lines.erase(it);
				} else {
					++it;
				}
			}
		}
	}
	MirrorLine entry;
	entry.slot = m_mirror_free_slots.back();
	m_mirror_free_slots.pop_back();
	entry.last_use = ++m_mirror_uses;
	// Nothing recorded the line's bytes yet: copy them at the end of this command buffer.
	entry.dirty_tick = m_scheduler.CurrentTick();
	m_mirror_lines.emplace(line, entry);
	m_mirror_dirty.push_back(line);
}

bool BufferCache::TryReadMirroredGpuWrites(uint64_t vaddr, void* data, uint64_t size) {
	if (size == 0 || size > MirrorMaxRead || m_device_address_pending ||
	    !GuestGpu::IsGpuThread()) {
		GpuProfiler::AddCount(size > MirrorMaxRead ? "mirror no: size" : m_device_address_pending ? "mirror no: da pending" : "mirror no: other");
		return false;
	}
	// Writes of the operation being recorded have no tick yet.
	for (const auto& [address, bytes]: m_pending_writes) {
		if (address < vaddr + size && vaddr < address + bytes) {
			GpuProfiler::AddCount("mirror no: pending write");
			return false;
		}
	}
	// As DownloadOnReadbackQueue(): the submissions that last wrote the bytes, and those that
	// wrote through device addresses (which name no range).
	uint64_t needed  = m_device_address_tick;
	bool     unknown = false;
	m_gpu_modified_ranges.ForEachInRange(vaddr, size, [&](uint64_t start, uint64_t end) {
		const auto latest = m_write_ticks.Latest(start, end);
		unknown |= latest == UINT64_MAX;
		needed = std::max(needed, latest);
	});
	if (unknown) {
		GpuProfiler::AddCount("mirror no: unknown tick");
		return false;
	}
	const auto first = Common::AlignDown(vaddr, MirrorLineSize);
	const auto last  = Common::AlignUp(vaddr + size, MirrorLineSize);
	uint64_t   wait  = 0;
	bool       held  = true;
	for (auto line = first; line < last; line += MirrorLineSize) {
		const auto it = m_mirror_lines.find(line);
		if (it == m_mirror_lines.end()) {
			AddMirrorLine(line);
			held = false;
			continue;
		}
		auto& entry    = it->second;
		entry.last_use = ++m_mirror_uses;
		if (entry.tick == 0 || entry.tick < needed) {
			// Copy it at the end of this command buffer, for the next read.
			held = false;
			if (entry.dirty_tick != m_scheduler.CurrentTick()) {
				entry.dirty_tick = m_scheduler.CurrentTick();
				m_mirror_dirty.push_back(line);
			}
		}
		wait = std::max(wait, entry.tick);
	}
	if (!held || wait >= m_scheduler.CurrentTick()) {
		GpuProfiler::AddCount(!held ? (needed == m_device_address_tick ? "mirror no: not held (da tick)" : "mirror no: not held") : "mirror no: copy in current");
		return false;
	}
	GpuProfiler::AddCount("mirror: hit");
	m_scheduler.GetMasterSemaphore().Wait(wait);
	m_mirror_flush_use = m_mirror_uses;
	// As a readback: earlier asynchronous publications of these bytes run first, and the bytes
	// become clean, so later reads take them from the backing store until the GPU writes them
	// again.
	m_scheduler.WaitPriorityOperations(needed);
	if (!Libs::LibKernel::Memory::TryReadBacking(vaddr, data, size)) {
		return false;
	}
	const auto* mapped = m_mirror->Mapped().data();
	m_gpu_modified_ranges.ForEachInRange(vaddr, size, [&](uint64_t start, uint64_t end) {
		while (start < end) {
			const auto line   = Common::AlignDown(start, MirrorLineSize);
			const auto chunk  = std::min(end, line + MirrorLineSize) - start;
			const auto slot   = m_mirror_lines.find(line)->second.slot;
			const auto offset = uint64_t {slot} * MirrorLineSize + (start - line);
			m_mirror->Invalidate(offset, chunk);
			std::memcpy(static_cast<uint8_t*>(data) + (start - vaddr), mapped + offset, chunk);
			Libs::LibKernel::Memory::WriteBacking(start, mapped + offset, chunk);
			start += chunk;
		}
	});
	m_gpu_modified_ranges.Subtract(vaddr, size);
	return true;
}

void BufferCache::RecordMirrorCopies(CommandBuffer& command) {
	if (m_mirror_dirty.empty() || !GuestGpu::IsGpuThread()) {
		return;
	}
	if (!m_mirror) {
		m_mirror = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                    vk::BufferUsageFlagBits::eTransferDst,
		                                    uint64_t {MaxMirrorLines} * MirrorLineSize);
	}
	std::sort(m_mirror_dirty.begin(), m_mirror_dirty.end());
	m_mirror_dirty.erase(std::unique(m_mirror_dirty.begin(), m_mirror_dirty.end()),
	                     m_mirror_dirty.end());
	const auto tick = m_scheduler.CurrentTick();
	command.EndRendering();
	const auto native = command.Handle();
	// Every earlier write of the command buffer, and the earlier copies into the mirror slots.
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                       nullptr);
	std::vector<vk::BufferCopy> copies;
	const Buffer*               source = nullptr;
	const auto                  flush  = [&] {
		if (!copies.empty()) {
			native.copyBuffer(source->Handle(), m_mirror->Handle(),
			                  static_cast<uint32_t>(copies.size()), copies.data());
			copies.clear();
		}
	};
	for (const auto line: m_mirror_dirty) {
		const auto it = m_mirror_lines.find(line);
		if (it == m_mirror_lines.end()) {
			continue;
		}
		// Lines outside one cached buffer stay unmirrored (their reads take the readback path).
		const auto* owner = m_page_table.Find(line >> PageTable::kPageBits);
		if (owner == nullptr || !*owner) {
			continue;
		}
		const auto& buffer = m_slot_buffers[*owner];
		if (buffer.is_deleted || !buffer.IsInBounds(line, MirrorLineSize)) {
			continue;
		}
		if (&buffer != source) {
			flush();
			source = &buffer;
		}
		copies.push_back({buffer.Offset(line), uint64_t {it->second.slot} * MirrorLineSize,
		                  MirrorLineSize});
		it->second.tick = tick;
	}
	flush();
	m_mirror_dirty.clear();
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                       {}, 1, &after, 0, nullptr, 0, nullptr);
}

bool BufferCache::DownloadOnReadbackQueue(Buffer& buffer, std::span<const vk::BufferCopy> copies,
                                          uint64_t total_size) {
	if (!m_graphics.readback_queue || m_device_address_pending) {
		GpuProfiler::AddCount("rbq no: device address pending");
		return false;
	}
	const auto buffer_address = buffer.CpuAddress();
	// Writes of the operation being recorded, or through device addresses (which name no range)
	// since the needed submissions, may still change the bytes.
	uint64_t needed = m_device_address_tick;
	const auto readback_begin = buffer_address + copies.front().srcOffset;
	const auto readback_end   = buffer_address + copies.back().srcOffset + copies.back().size;
	for (const auto& copy: copies) {
		const auto begin = buffer_address + copy.srcOffset;
		const auto end   = begin + copy.size;
		for (const auto& [vaddr, size]: m_pending_writes) {
			if (vaddr < end && begin < vaddr + size) {
				NoteReadbackWait(readback_begin, readback_end, false, true);
				GpuProfiler::AddCount("rbq no: pending write");
				return false;
			}
		}
		needed = std::max(needed, m_write_ticks.Latest(begin, end));
	}
	const bool device_address = needed != 0 && needed == m_device_address_tick;
	if (needed >= m_scheduler.CurrentTick()) {
		NoteReadbackWait(readback_begin, readback_end, device_address, true);
		GpuProfiler::AddCount(needed == UINT64_MAX ? "rbq no: unknown tick" : needed == m_device_address_tick ? "rbq no: device address tick" : "rbq no: writer in current");
		return false;
	}
	NoteReadbackWait(readback_begin, readback_end, device_address, false);
	if (GpuProfiler::Enabled() && !m_scheduler.GetMasterSemaphore().IsFree(needed)) {
		// Debug: readbacks that wait for GPU work still running.
		extern thread_local uint64_t g_dbg_cs_addr;
		GpuProfiler::AddAddress("rbq busy wait writer shader", GpuProfiler::LastGpuWriter(readback_begin));
		GpuProfiler::AddAddress("rbq busy wait reader cs", g_dbg_cs_addr);
		GpuProfiler::AddAddress("rbq busy wait addr", readback_begin);
		GpuProfiler::AddAddress("rbq busy wait ticks behind", m_scheduler.CurrentTick() - needed);
	}
	// The copy waits for the needed submission on the GPU, but must not be submitted before it:
	// the driver may then block inside vkQueueSubmit, holding locks a submission of the
	// recording thread (or a present) needs.
	{
		ExecTrace::OtherWaitScope trace_wait;
		KYTY_SYNC_SITE(s_rbq_submit, "readback queue: wait rec submit of writer tick", Spin);
		Common::SyncAcct::Scope sync_scope(
		    Common::SyncAcct::Enabled() && m_scheduler.SubmittedTick() < needed ? Common::SyncAcct::SiteId(s_rbq_submit) : uint16_t {0},
		    needed, m_scheduler.SubmittedTick());
		while (m_scheduler.SubmittedTick() < needed) {
			Common::SpinPause();
		}
	}
	auto& device = m_graphics.device;
	if (!m_readback.pool) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
		                             vk::CommandPoolCreateFlagBits::eTransient;
		pool_info.queueFamilyIndex = m_graphics.readback_family;
		vk::CommandBufferAllocateInfo allocate {};
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		const vk::FenceCreateInfo fence_info {};
		if (device.createCommandPool(&pool_info, nullptr, &m_readback.pool) != vk::Result::eSuccess ||
		    (allocate.commandPool = m_readback.pool,
		     device.allocateCommandBuffers(&allocate, &m_readback.command) != vk::Result::eSuccess) ||
		    device.createFence(&fence_info, nullptr, &m_readback.fence) != vk::Result::eSuccess) {
			EXIT("BufferCache: could not create the readback queue resources\n");
		}
	}
	if (!m_readback.staging || m_readback.staging->Size() < total_size) {
		m_readback.staging = std::make_unique<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::Download, 0,
		    vk::BufferUsageFlagBits::eTransferDst,
		    std::max<uint64_t>(Common::AlignUp(total_size, 1024 * 1024), 4 * MiB));
	}
	auto&                       staging = *m_readback.staging;
	const auto command = m_readback.command;
	(void)command.reset({});
	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	(void)command.begin(&begin_info);
	command.copyBuffer(buffer.Handle(), staging.Handle(), static_cast<uint32_t>(copies.size()),
	                   copies.data());
	vk::BufferMemoryBarrier after {};
	after.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.buffer              = staging.Handle();
	after.offset              = 0;
	after.size                = total_size;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                        {}, 0, nullptr, 1, &after, 0, nullptr);
	(void)command.end();
	// The semaphore wait makes the writes of the needed submissions visible to the copy.
	const auto                      semaphore = m_scheduler.GetMasterSemaphore().Handle();
	const vk::PipelineStageFlags    wait_stage = vk::PipelineStageFlagBits::eTransfer;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount = 1;
	timeline.pWaitSemaphoreValues    = &needed;
	vk::SubmitInfo submit {};
	submit.pNext              = &timeline;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores    = &semaphore;
	submit.pWaitDstStageMask  = &wait_stage;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	GpuProfiler::TimedWait timed_submit("rbq: submit");
	const auto submit_result = m_graphics.readback_queue.submit(1, &submit, m_readback.fence);
	timed_submit.~TimedWait();
	const bool busy_writer = !m_scheduler.GetMasterSemaphore().IsFree(needed);
	new (&timed_submit) GpuProfiler::TimedWait(!busy_writer ? "rbq: fence spin (writer done)" : "rbq: fence spin (writer busy)");
	if (submit_result != vk::Result::eSuccess) {
		EXIT("BufferCache: readback queue submission failed\n");
	}
	GpuProfiler::AddCount("readback: on readback queue");
	ExecTrace::GpuWaitScope trace_fence;
	KYTY_SYNC_SITE(s_rbq_fence, "readback queue: wait GPU copy fence", Spin);
	Common::SyncAcct::Scope rbq_scope(s_rbq_fence, needed, m_scheduler.GetMasterSemaphore().KnownGpuTick());
	for (;;) {
		const auto status = device.getFenceStatus(m_readback.fence);
		if (status == vk::Result::eSuccess) {
			break;
		}
		if (status != vk::Result::eNotReady) {
			EXIT("BufferCache: readback queue wait failed: %s\n", vk::to_string(status).c_str());
		}
		Common::SpinPause();
	}
	rbq_scope.End();
	(void)device.resetFences(1, &m_readback.fence);
	// Earlier asynchronous publications of these bytes run first, as in the queued path.
	m_scheduler.WaitPriorityOperations(needed);
	staging.Invalidate(0, total_size);
	const auto* mapped = staging.Mapped().data();
	if (GpuProfiler::Enabled()) {
		// Debug: would the stale backing have given the GPU's bytes?
		bool same = true;
		for (const auto& copy: copies) {
			std::vector<uint8_t> old(copy.size);
			if (!Libs::LibKernel::Memory::TryReadBacking(buffer_address + copy.srcOffset, old.data(),
			                                             copy.size) ||
			    std::memcmp(old.data(), mapped + copy.dstOffset, copy.size) != 0) {
				same = false;
			}
		}
		GpuProfiler::AddCount(busy_writer ? (same ? "rbq busy: backing had same bytes" : "rbq busy: bytes changed")
		                                  : (same ? "rbq done: backing had same bytes" : "rbq done: bytes changed"));
	}
	for (const auto& copy: copies) {
		Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
		                                      mapped + copy.dstOffset, copy.size);
	}
	return true;
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::WriteAroundGpuWrites(uint64_t vaddr, const void* data, uint64_t size) {
	constexpr uint64_t MaxBytes = 4096;
	if (size == 0 || size > MaxBytes || (vaddr & 3u) != 0 || (size & 3u) != 0 ||
	    !GuestRange {vaddr, size}.Valid() || !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	Buffer*    buffer = nullptr;
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner && m_slot_buffers[*owner].IsInBounds(vaddr, size)) {
		buffer = &m_slot_buffers[*owner];
	} else if (IsRegionRegistered(vaddr, size)) {
		// Partly cached: the faulting path handles it.
		return false;
	}
	// Bytes the GPU wrote stay GPU-dirty: the copy below orders the new value after the GPU's
	// writes, and a later read of them reads the buffer back. Without a buffer to copy into,
	// the GPU's bytes would be lost.
	if (buffer == nullptr && HasGpuDirtyBytes(vaddr, size)) {
		return false;
	}
	if (!Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		return false;
	}
	if (buffer == nullptr) {
		return true;
	}
	// The cached buffer is what the GPU and later readbacks of the page use: give it the bytes
	// at this point of the command stream.
	auto&      stream  = GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset  = StreamUpload(data, size, 4);
	auto&      command = m_scheduler.Current();
	command.EndRendering();
	const auto native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer->Handle();
	before.offset              = buffer->Offset(vaddr);
	before.size                = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 1, &before, 0, nullptr);
	const vk::BufferCopy copy {offset, buffer->Offset(vaddr), size};
	native.copyBuffer(stream.Handle(), buffer->Handle(), 1, &copy);
	NoteWrite(vaddr, size);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	m_gpu_write_generation++;
	return true;
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

// Debug: counters of the age collection, printed with StaleGC lines.
struct StaleStats {
	uint64_t freed = 0, kept_pinned = 0, kept_gpu = 0, use_scans = 0, use_pages = 0, use_touched = 0;
};
static StaleStats g_stale_stats;

void BufferCache::CollectStaleBuffers(uint64_t tick) {
	// Buffers outlive the draws that created them, and once the guest moves elsewhere the memory
	// they cover is reused for other data. Every device-address synchronization keeps uploading the
	// guest's writes there and protecting the pages again, so the guest takes write faults on
	// memory no draw reads, more of it the longer the session runs (the collection under memory
	// pressure below rarely runs). Release the buffers no lookup used for a long time; the next use
	// creates them again. Buffers with GPU writes stay, and so do the ones shaders reached through
	// device addresses the host could not see coming.
	static constexpr uint64_t StaleAge      = 2048; // garbage-collection ticks: about half a minute
	static constexpr uint64_t StalePeriod   = 8;
	static constexpr size_t   StaleVisitMax = 64;
	static constexpr size_t   StaleFreeMax  = 16;
	// Debug: what the age collection releases and keeps, every 1024 collections.
	if (tick % 1024 == 0) {
		std::printf("StaleGC tick=%llu buffers=%zu freed=%llu kept_pinned=%llu kept_gpu=%llu "
		            "use_scans=%llu use_pages=%llu use_touched=%llu\n",
		            static_cast<unsigned long long>(tick), m_buffers.size(),
		            static_cast<unsigned long long>(g_stale_stats.freed),
		            static_cast<unsigned long long>(g_stale_stats.kept_pinned),
		            static_cast<unsigned long long>(g_stale_stats.kept_gpu),
		            static_cast<unsigned long long>(g_stale_stats.use_scans),
		            static_cast<unsigned long long>(g_stale_stats.use_pages),
		            static_cast<unsigned long long>(g_stale_stats.use_touched));
		std::fflush(stdout);
		g_stale_stats = {};
	}
	if (tick < StaleAge || tick % StalePeriod != 0) {
		return;
	}
	std::vector<BufferId> stale;
	std::vector<BufferId> kept;
	m_lru_cache.ForEachItemBelow(tick - StaleAge, [&](BufferId id) {
		const auto& buffer = m_slot_buffers[id];
		const bool  keep   = buffer.device_address_pinned ||
		                  m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size()) ||
		                  m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		(keep ? kept : stale).push_back(id);
		if (keep) {
			(buffer.device_address_pinned ? g_stale_stats.kept_pinned : g_stale_stats.kept_gpu)++;
		}
		return stale.size() == StaleFreeMax || stale.size() + kept.size() == StaleVisitMax;
	});
	// The buffers that stay move to the back of the queue, so the next passes reach the others.
	for (const auto id: kept) {
		m_lru_cache.Touch(m_slot_buffers[id].lru_id, tick);
	}
	for (const auto id: stale) {
		const auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		if (m_record_released) {
			m_released_ranges.Add(buffer.CpuAddress(), buffer.Size());
		}
		DeleteBuffer(id);
		g_stale_stats.freed++;
	}
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	CollectStaleBuffers(tick);
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			if (m_record_released) {
				m_released_ranges.Add(buffer.CpuAddress(), buffer.Size());
			}
			DeleteBuffer(id);
			CacheStats::Add(CacheStats::BufferGcFrees);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	GpuProfiler::WaitTagScope tag("gpu-thread: buffer GC dirty download");
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		if (m_record_released) {
			m_released_ranges.Add(buffer.CpuAddress(), buffer.Size());
		}
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::ProcessUseBuffer() {
	m_fault_manager.ProcessUseBuffer([this](std::span<const uint64_t> pages) {
		g_stale_stats.use_scans++;
		g_stale_stats.use_pages += pages.size();
		for (const auto page: pages) {
			const auto* owner = m_page_table.Find(page >> PageTable::kPageBits);
			if (owner != nullptr && *owner) {
				g_stale_stats.use_touched++;
				TouchBuffer(m_slot_buffers[*owner]);
			}
		}
	});
}

void BufferCache::ObtainUnsharedWrite(uint64_t vaddr, uint64_t size) {
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	TouchBuffer(buffer);
	// The guest's bytes reach the buffer first (the writes change part of them).
	(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
	m_gpu_write_generation++;
	NoteWrite(vaddr, size);
	ForgetKnownFills(vaddr, size);
}

void BufferCache::NoteGpuWrites(uint64_t vaddr, uint64_t size) {
	m_gpu_write_generation++;
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid GPU write range\n");
	}
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		const auto  start  = std::max(buffer.CpuAddress(), vaddr);
		const auto  finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start >= finish) {
			continue;
		}
		TouchBuffer(buffer);
		ForgetKnownFills(start, finish - start);
		NoteWrite(start, finish - start);
		m_memory_tracker.MarkRegionAsGpuModifiedUnlessCpuDirty(
		    start, finish - start, [&](uint64_t page, uint64_t bytes) {
			    const auto first = std::max(page, start);
			    const auto last  = std::min(page + bytes, finish);
			    m_gpu_modified_ranges.Add(first, last - first);
		    });
	}
}

void BufferCache::FlushUploadBatch() {
	m_upload_batching = false;
	if (m_upload_batch_dsts.empty()) {
		return;
	}
	// The staging ranges belong to the command buffer they were mapped for.
	EXIT_IF(m_scheduler.CurrentTick() != m_upload_batch_tick);
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto native = command.Handle();
	// As SynchronizeBuffer() records around each upload, for all the buffers at once.
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
	                       vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion, 1,
	                       &before, 0, nullptr, 0, nullptr);
	const auto staging = m_staging_buffer.Handle();
	size_t     first   = 0;
	for (const auto& [destination, count]: m_upload_batch_dsts) {
		native.copyBuffer(staging, destination, count, m_upload_batch_copies.data() + first);
		first += count;
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlagBits::eByRegion,
	                       1, &after, 0, nullptr, 0, nullptr);
	m_upload_batch_dsts.clear();
	m_upload_batch_copies.clear();
}

// Each buffer's upload would record its own pair of barriers: record them once, around the
// copies of every buffer (hundreds per frame in GTA V).
struct BufferCache::UploadBatch {
	BufferCache& cache;
	explicit UploadBatch(BufferCache& owner): cache(owner) {
		if (KYTY_X(BDA_BATCH)) {
			cache.m_upload_batching   = true;
			cache.m_upload_batch_tick = cache.m_scheduler.CurrentTick();
		}
	}
	~UploadBatch() { cache.FlushUploadBatch(); }
	UploadBatch(const UploadBatch&)            = delete;
	UploadBatch& operator=(const UploadBatch&) = delete;
};

void BufferCache::SynchronizeBuffersInSpans(std::span<const std::pair<uint64_t, uint64_t>> spans) {
	const UploadBatch batch {*this};
	for (const auto& [first, last]: spans) {
		SynchronizeBuffersInRange(first, last - first);
	}
}

void BufferCache::SynchronizeBuffersInRanges(const RangeSet& ranges, bool all) {
	const UploadBatch batch {*this};
	// Read each region's epoch before scanning it: pages dirtied during or after the scan advance
	// it again, and the next call visits the region again.
	m_sync_regions.clear();
	GpuProfiler::PhaseMark("bda: scan");
	ranges.ForEach([&](uint64_t start, uint64_t end) {
		for (auto index = start / TRACKER_REGION_SIZE; index * TRACKER_REGION_SIZE < end; index++) {
			if (!m_sync_regions.empty() && m_sync_regions.back().first == index) {
				continue;
			}
			const auto epoch = m_memory_tracker.RegionCpuDirtyEpoch(index);
			if (!all) {
				// Synchronizing a buffer tracks its pages, so an untracked region holds no buffer
				// since the last full pass.
				if (epoch == 0) {
					continue;
				}
				const auto synced = m_synced_region_epochs.find(index);
				if (synced != m_synced_region_epochs.end() && synced->second == epoch) {
					continue;
				}
			}
			m_sync_regions.emplace_back(index, epoch);
		}
	});
	GpuProfiler::PhaseMark("bda: sync");
	GpuProfiler::AddCount("bda: regions to sync", m_sync_regions.size());
	if (all) {
		m_synced_region_epochs.clear();
		ranges.ForEach([this](uint64_t start, uint64_t end) {
			SynchronizeBuffersInRange(start, end - start);
		});
	} else {
		for (const auto& [index, epoch]: m_sync_regions) {
			ranges.ForEachInRange(index * TRACKER_REGION_SIZE, TRACKER_REGION_SIZE,
			                      [this](uint64_t start, uint64_t end) {
				                      SynchronizeBuffersInRange(start, end - start);
			                      });
		}
	}
	for (const auto& [index, epoch]: m_sync_regions) {
		m_synced_region_epochs[index] = epoch;
	}
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	// Only tracking regions with pending CPU writes can have anything to upload.
	m_memory_tracker.ForEachCpuDirtySpan(vaddr, size, [&](uint64_t span, uint64_t span_size) {
		GpuProfiler::AddCount("bda: dirty spans");
		const auto end = span + span_size;
		auto       it  = m_buffers.upper_bound(span);
		if (it != m_buffers.begin()) {
			--it;
		}
		for (; it != m_buffers.end() && it->first < end; ++it) {
			auto&      buffer = m_slot_buffers[it->second];
			const auto start  = std::max(buffer.CpuAddress(), span);
			const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
			if (start < finish) {
				GpuProfiler::AddCount("bda: span buffer syncs");
				(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
				CacheStats::Add(CacheStats::BdaSyncBuffers);
			}
		}
	});
}

} // namespace Libs::Graphics
