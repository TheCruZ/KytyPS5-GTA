#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/spscQueue.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>
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
		NoteBufferRegions(buffer, true);
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
		NoteBufferRegions(buffer, false);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

bool BufferCache::TouchLiveBuffer(BufferId id, vk::Buffer handle, uint64_t vaddr, uint64_t size) {
	auto* buffer = GetBufferIfLive(id);
	if (buffer == nullptr || buffer->Handle() != handle || !buffer->IsInBounds(vaddr, size)) {
		return false;
	}
	TouchBuffer(*buffer);
	return true;
}

void BufferCache::EnsureBufferContents(uint64_t vaddr, uint64_t size) {
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	// A copy in flight on the readback queue may still read it.
	if (std::ranges::any_of(m_async_readbacks, [](const auto& slot) { return slot.request != nullptr; })) {
		DrainAsyncReadbacks(0, 0, m_slot_buffers[id].Handle());
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferHostOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

// Downloads in progress on this thread: a query they make reentrantly can see ranges not yet subtracted from the
// GPU-modified set whose pages are no longer GPU-owned (see SynchronizeBuffer()).
static thread_local uint32_t t_download_depth = 0;

// KYTY_BUFFER_RESIDENCY=0: a written range the GPU already owns whole is synchronized as any other.
static bool BufferResidencyEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BUFFER_RESIDENCY");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	struct DepthScope {
		DepthScope() { ++t_download_depth; }
		~DepthScope() { --t_download_depth; }
	} depth;
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
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

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
      // (Device addresses: table dispatches read the small ranges ObtainBuffer() streams through it.)
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB,
                      AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	m_page_manager = &page_manager;
	// Queried and changed under the renderer's lock only.
	m_gpu_modified_ranges.AllowFastPath();
	// chenxiao07's KYTY_STREAM_UPLOAD (5c4da93aa): the small ranges draws stream go through a ring in host memory
	// (written by the CPU without the device-visible memory's write combining) instead of the stream buffer.
	if (const char* value = std::getenv("KYTY_STREAM_UPLOAD"); value != nullptr && value[0] == '1') {
		m_host_upload = std::make_unique<StreamBuffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 64 * MiB,
		                                               AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress);
	}
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

// KYTY_COPY_FEEDBACK=0: no snapshots of copy destinations.
static bool CopyFeedbackEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_COPY_FEEDBACK");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// Snapshots of the destinations of copies the GPU makes (ScheduleCopyFeedback()), each in its
// slot of a host-visible buffer, copied there right after the copy in the same command buffer.
// Execution thread only. A slot is reused once its copy executed; a GPU write to its range drops
// it (NoteWrite()). Ported from chenxiao07's KytyPS5 fork (KYTY_COPY_FEEDBACK).
struct BufferCache::CopyFeedback {
	static constexpr uint64_t SlotSize  = 64 * 1024;
	static constexpr size_t   SlotCount = 512;
	struct Slot {
		uint64_t   address = 0, size = 0, tick = 0;
		vk::Buffer owner   = nullptr;
	};
	Buffer                      download;
	std::array<Slot, SlotCount> slots {};
	// The slot of each snapshot by its guest address, sorted by address.
	std::vector<std::pair<uint64_t, size_t>> index;
	size_t                                   cursor = 0;
	// Snapshots per 16 MiB granule, hashed: a range whose granules count none overlaps none.
	std::array<uint16_t, 4096> granules {};

	CopyFeedback(GraphicContext& graphics, CommandScheduler& scheduler)
	    : download(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst,
	               SlotCount * SlotSize) {}

	[[nodiscard]] auto LowerBound(uint64_t address) {
		return std::lower_bound(index.begin(), index.end(), address,
		                        [](const auto& entry, uint64_t value) { return entry.first < value; });
	}
	[[nodiscard]] auto UpperBound(uint64_t address) {
		return std::upper_bound(index.begin(), index.end(), address,
		                        [](uint64_t value, const auto& entry) { return value < entry.first; });
	}
	void Count(const Slot& slot, int delta) {
		for (uint64_t g = slot.address >> 24u; g <= (slot.address + slot.size - 1) >> 24u; ++g) {
			granules[g & 4095u] = static_cast<uint16_t>(granules[g & 4095u] + delta);
		}
	}
	[[nodiscard]] bool MayOverlap(uint64_t vaddr, uint64_t size) const {
		const uint64_t first = vaddr >> 24u, last = (vaddr + size - 1) >> 24u;
		if (size == 0 || last - first >= granules.size()) {
			return true;
		}
		for (uint64_t g = first; g <= last; ++g) {
			if (granules[g & 4095u] != 0) {
				return true;
			}
		}
		return false;
	}
	void Erase(size_t slot) {
		auto& entry = slots[slot];
		if (entry.address == 0) {
			return;
		}
		if (const auto it = LowerBound(entry.address); it != index.end() && it->second == slot) {
			index.erase(it);
		}
		Count(entry, -1);
		entry.address = 0;
	}
};

void BufferCache::InvalidateCopyFeedback(uint64_t vaddr, uint64_t size) {
	if (!m_copy_feedback || m_copy_feedback->index.empty()) {
		return;
	}
	auto& feedback = *m_copy_feedback;
	if (!feedback.MayOverlap(vaddr, size)) {
		return;
	}
	auto it = feedback.LowerBound(vaddr);
	if (it != feedback.index.begin()) {
		const auto& prior = feedback.slots[std::prev(it)->second];
		if (prior.address + prior.size > vaddr) {
			--it;
		}
	}
	std::vector<size_t> erased;
	for (; it != feedback.index.end() && it->first < vaddr + size; ++it) {
		erased.push_back(it->second);
	}
	for (const auto slot: erased) {
		feedback.Erase(slot);
	}
}

// Demon's Souls copies memory the GPU wrote (its cs_memset32 copies, as buffer copies) and then
// reads the destination on the CPU: each such read was a readback that waited for the GPU.
void BufferCache::ScheduleCopyFeedback(uint64_t vaddr, uint64_t size) {
	if (!CopyFeedbackEnabled() || size == 0 || size > CopyFeedback::SlotSize || ((vaddr | size) & 3u) != 0 ||
	    !GuestRange {vaddr, size}.Valid() || !m_gpu_modified_ranges.Contains(vaddr, size)) {
		return;
	}
	const auto* owner_id = m_page_table.Find(vaddr >> PageTable::kPageBits);
	auto*       owner    = owner_id != nullptr && *owner_id ? m_slot_buffers.try_get(*owner_id) : nullptr;
	if (owner == nullptr || owner->is_deleted || !owner->IsInBounds(vaddr, size)) {
		return;
	}
	if (!m_copy_feedback) {
		m_copy_feedback = std::make_unique<CopyFeedback>(m_graphics, m_scheduler);
	}
	auto&  feedback = *m_copy_feedback;
	auto&  master   = m_scheduler.GetMasterSemaphore();
	size_t selected = CopyFeedback::SlotCount;
	for (int refresh = 0; refresh < 2 && selected == CopyFeedback::SlotCount; ++refresh) {
		if (refresh != 0) {
			master.Refresh();
		}
		const auto completed = master.KnownGpuTick();
		// An empty slot first, then the oldest finished snapshot. Never a wait for storage.
		for (int evict = 0; evict < 2 && selected == CopyFeedback::SlotCount; ++evict) {
			for (size_t n = 0; n < CopyFeedback::SlotCount; ++n) {
				const auto  i    = (feedback.cursor + n) % CopyFeedback::SlotCount;
				const auto& slot = feedback.slots[i];
				if (slot.tick <= completed && (evict != 0 || slot.address == 0)) {
					selected = i;
					break;
				}
			}
		}
	}
	if (selected == CopyFeedback::SlotCount) {
		return;
	}
	feedback.Erase(selected);
	InvalidateCopyFeedback(vaddr, size);
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = owner->Handle();
	before.offset              = owner->Offset(vaddr);
	before.size                = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 0,
	                       nullptr, 1, &before, 0, nullptr);
	const vk::BufferCopy copy {owner->Offset(vaddr), selected * CopyFeedback::SlotSize, size};
	native.copyBuffer(owner->Handle(), feedback.download.Handle(), 1, &copy);
	vk::BufferMemoryBarrier after {};
	after.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.buffer              = feedback.download.Handle();
	after.offset              = selected * CopyFeedback::SlotSize;
	after.size                = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                       1, &after, 0, nullptr);
	auto& slot = feedback.slots[selected];
	slot       = {vaddr, size, m_scheduler.CurrentTick(), owner->Handle()};
	feedback.index.insert(feedback.LowerBound(vaddr), {vaddr, selected});
	feedback.Count(slot, 1);
	feedback.cursor = (selected + 1) % CopyFeedback::SlotCount;
}

// Execution thread: the GPU-written bytes of the range served from finished snapshots, when they
// hold all of them. They leave the GPU-modified set; the caller makes the range CPU-readable.
bool BufferCache::TryReadCopyFeedback(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	if (!m_copy_feedback || m_copy_feedback->index.empty() || !m_copy_feedback->MayOverlap(vaddr, size)) {
		return false;
	}
	auto& feedback = *m_copy_feedback;
	struct Part {
		uint64_t address, size, offset;
	};
	std::vector<Part>                          parts;
	std::vector<std::pair<uint64_t, uint64_t>> covered;
	uint64_t                                   latest = 0;
	bool                                       served = true;
	m_memory_tracker.ForEachDownloadRange<false>(vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			for (auto cursor = start; served && cursor < end;) {
				auto it = feedback.UpperBound(cursor);
				if (it == feedback.index.begin()) {
					served = false;
					break;
				}
				--it;
				const auto& slot = feedback.slots[it->second];
				if (cursor >= slot.address + slot.size || slot.owner != buffer.Handle()) {
					served = false;
					break;
				}
				const auto part = std::min(end, slot.address + slot.size) - cursor;
				parts.push_back({cursor, part, it->second * CopyFeedback::SlotSize + (cursor - slot.address)});
				latest = std::max(latest, slot.tick);
				cursor += part;
			}
			covered.emplace_back(start, end - start);
		});
	});
	if (!served || parts.empty() || !m_scheduler.IsFree(latest)) {
		return false;
	}
	// Earlier asynchronous publications of these bytes run first, as on the readback paths.
	m_scheduler.WaitPriorityOperations(latest);
	for (const auto& part: parts) {
		feedback.download.Invalidate(part.offset, part.size);
		Libs::LibKernel::Memory::WriteBacking(part.address, feedback.download.Mapped().data() + part.offset,
		                                      part.size);
	}
	for (const auto& [address, bytes]: covered) {
		m_gpu_modified_ranges.Subtract(address, bytes);
	}
	return true;
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

void BufferCache::NoteKnownFill(uint64_t vaddr, uint64_t size, uint32_t value) {
	if (size == 0) {
		return;
	}
	ForgetKnownFills(vaddr, size);
	m_known_fills[vaddr] = {vaddr + size, value};
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

// Demon's Souls streams files (AMPR reads) and moves megabytes (its memmove, DemonsSoulsCopy) over
// memory the GPU wrote: each page faulted and was read back first, for bytes the write then
// replaced. Ported from chenxiao07's KytyPS5 fork.
void BufferCache::InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	ForgetKnownFills(vaddr, size);
	bool requested = false; // (once for all of the range's regions)
	m_memory_tracker.InvalidateRegion(vaddr, size, [&] {
		if (std::exchange(requested, true)) {
			return;
		}
		if (GuestGpu::IsGpuThread()) {
			DiscardGpuWrites(vaddr, size);
			return;
		}
		m_scheduler.Context().GetGpu().SendUrgentCommandSync([this, vaddr, size] { DiscardGpuWrites(vaddr, size); });
	});
}

// Execution thread: as a write fault's readback, but only the GPU-written bytes of the first and
// last page outside the range are read back; the others are dropped, the write replaces them.
void BufferCache::DiscardGpuWrites(uint64_t vaddr, uint64_t size) {
	const auto end   = vaddr + size;
	const auto first = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto last  = Common::AlignUp(end, TRACKER_PAGE_SIZE);
	DrainAsyncReadbacks(first, last - first);
	if (!IsRegionRegistered(vaddr, size)) {
		return;
	}
	for (const auto& [begin, stop]: {std::pair {first, vaddr}, std::pair {end, last}}) {
		if (begin >= stop || !m_gpu_modified_ranges.Intersects(begin, stop - begin) ||
		    !IsRegionRegistered(begin, stop - begin)) {
			continue;
		}
		// (The buffer of that page alone: one over the whole range would join every buffer in it.)
		auto& buffer = m_slot_buffers[FindBuffer(begin, stop - begin)];
		(void)DownloadBufferMemory<false>(buffer, begin, stop - begin);
	}
	m_gpu_modified_ranges.Subtract(first, last - first);
	m_memory_tracker.UnmarkRegionAsGpuModified(first, last - first);
	m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
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

// A guest read of GPU-written memory in flight on the readback queue (see ReadMemory()). Its
// ranges stay GPU-modified until it is finished: any other access of them waits for it first
// (DrainAsyncReadbacks()).
struct BufferCache::AsyncReadback {
	uint64_t                    vaddr = 0, size = 0;
	uint64_t                    needed = 0; // the submission whose writes the copy waits for
	uint64_t                    buffer_address = 0;
	vk::Buffer                  source;     // the cache buffer copied from
	std::vector<vk::BufferCopy> copies;     // dstOffset: into the slot's staging buffer
	size_t                      slot = 0;
	vk::Device                  device;
	vk::Fence                   fence;
	Buffer*                     staging = nullptr;
	std::atomic_bool            finished {false};
};

static bool AsyncReadbacksEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GUEST_READBACK_ASYNC");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// Writes fault inside host code too (file reads into GPU memory, streaming), which may hold locks
// the execution thread needs between the two urgent commands.
static bool AsyncWriteReadbacksEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GUEST_WRITE_READBACK_ASYNC");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

std::shared_ptr<BufferCache::AsyncReadback> BufferCache::BeginAsyncReadback(uint64_t vaddr, uint64_t size) {
	if (!m_graphics.readback_queue || m_device_address_pending || !IsRegionRegistered(vaddr, size)) {
		return nullptr;
	}
	for (auto& slot: m_async_readbacks) {
		const auto request = slot.request;
		if (!request || request->finished.load(std::memory_order_acquire)) {
			continue;
		}
		// Another guest thread already waits for these bytes.
		if (vaddr >= request->vaddr && vaddr + size <= request->vaddr + request->size) {
			return request;
		}
	}
	// A copy of some of them goes first.
	DrainAsyncReadbacks(vaddr, size);
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	if (const auto window = ReadbackWindow(buffer, vaddr, size); window.second != size) {
		std::tie(vaddr, size) = window;
		DrainAsyncReadbacks(vaddr, size);
	}
	const auto buffer_address = buffer.CpuAddress();
	// What DownloadBufferMemory() would copy.
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	m_memory_tracker.ForEachDownloadRange<false>(vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			copies.emplace_back(start - buffer_address, total_size, end - start);
			total_size += Common::AlignUp(end - start, 64);
		});
	});
	if (copies.empty()) {
		return nullptr;
	}
	uint64_t needed = m_device_address_tick;
	for (const auto& copy: copies) {
		const auto begin = buffer_address + copy.srcOffset;
		const auto end   = begin + copy.size;
		for (const auto& [address, bytes]: m_pending_writes) {
			if (address < end && begin < address + bytes) {
				return nullptr;
			}
		}
		needed = std::max(needed, m_write_ticks.Latest(begin, end));
	}
	if (needed >= m_scheduler.CurrentTick()) {
		return nullptr; // the writes are still being recorded
	}
	size_t free_slot = AsyncReadbackSlots;
	for (size_t i = 0; i < AsyncReadbackSlots; i++) {
		if (!m_async_readbacks[i].request) {
			free_slot = i;
			break;
		}
	}
	if (free_slot == AsyncReadbackSlots) {
		return nullptr;
	}
	auto& device = m_graphics.device;
	if (!m_async_readback_pool) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		pool_info.queueFamilyIndex = m_graphics.readback_family;
		if (device.createCommandPool(&pool_info, nullptr, &m_async_readback_pool) != vk::Result::eSuccess) {
			EXIT("BufferCache: could not create the asynchronous readback pool\n");
		}
	}
	auto& slot = m_async_readbacks[free_slot];
	if (!slot.command) {
		vk::CommandBufferAllocateInfo allocate {};
		allocate.commandPool        = m_async_readback_pool;
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		const vk::FenceCreateInfo fence_info {};
		if (device.allocateCommandBuffers(&allocate, &slot.command) != vk::Result::eSuccess ||
		    device.createFence(&fence_info, nullptr, &slot.fence) != vk::Result::eSuccess) {
			EXIT("BufferCache: could not create the asynchronous readback resources\n");
		}
	}
	if (!slot.staging || slot.staging->Size() < total_size) {
		slot.staging = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                        vk::BufferUsageFlagBits::eTransferDst,
		                                        std::max<uint64_t>(Common::AlignUp(total_size, 64 * 1024), 256 * 1024));
	}
	// The copy waits for the needed submission on the GPU, but must not be submitted before it
	// (see DownloadOnReadbackQueue()).
	while (m_scheduler.SubmittedTick() < needed) {
		Common::SpinPause();
	}
	const auto command = slot.command;
	(void)command.reset({});
	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	(void)command.begin(&begin_info);
	command.copyBuffer(buffer.Handle(), slot.staging->Handle(), static_cast<uint32_t>(copies.size()), copies.data());
	vk::BufferMemoryBarrier after {};
	after.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.buffer              = slot.staging->Handle();
	after.offset              = 0;
	after.size                = total_size;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1,
	                        &after, 0, nullptr);
	(void)command.end();
	const auto                      semaphore  = m_scheduler.GetMasterSemaphore().Handle();
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
	if (m_graphics.readback_queue.submit(1, &submit, slot.fence) != vk::Result::eSuccess) {
		EXIT("BufferCache: readback queue submission failed\n");
	}
	auto request            = std::make_shared<AsyncReadback>();
	request->vaddr          = vaddr;
	request->size           = size;
	request->needed         = needed;
	request->buffer_address = buffer_address;
	request->source         = buffer.Handle();
	request->copies         = std::move(copies);
	request->slot           = free_slot;
	request->device         = device;
	request->fence          = slot.fence;
	request->staging        = slot.staging.get();
	slot.request            = request;
	return request;
}

// Any thread: waits for the copy on the GPU. False when the request was finished meanwhile.
bool BufferCache::WaitAsyncReadback(AsyncReadback& request) {
	// (The fence is reset only when the request is finished.)
	for (;;) {
		if (request.finished.load(std::memory_order_acquire)) {
			return false;
		}
		const auto status = request.device.getFenceStatus(request.fence);
		if (status == vk::Result::eSuccess) {
			return true;
		}
		if (status != vk::Result::eNotReady) {
			EXIT("BufferCache: readback queue wait failed: %s\n", vk::to_string(status).c_str());
		}
		Common::SpinPause();
	}
}

// Execution thread: the copy's bytes go to guest memory (after the asynchronous publications
// queued before it, as on the synchronous path) and the request is finished.
void BufferCache::CompleteAsyncReadback(AsyncReadback& request) {
	if (!WaitAsyncReadback(request)) {
		return;
	}
	m_scheduler.WaitPriorityOperations(request.needed);
	request.staging->Invalidate(0, request.staging->Size());
	const auto* mapped = request.staging->Mapped().data();
	for (const auto& copy: request.copies) {
		Libs::LibKernel::Memory::WriteBacking(request.buffer_address + copy.srcOffset, mapped + copy.dstOffset,
		                                      copy.size);
	}
	FinishAsyncReadback(request);
}

// Execution thread, once the bytes were written back: the copied ranges leave the GPU-modified
// set and the range becomes CPU-readable. When recorded work may have written it since the copy's
// submission (a store to the range, or any store through device addresses, which names no range),
// the range is read back here as the synchronous path reads it: the guest waits for this, so it
// always goes on with current bytes.
void BufferCache::FinishAsyncReadback(AsyncReadback& request) {
	if (request.finished.load(std::memory_order_acquire)) {
		return;
	}
	bool rewritten = m_write_ticks.Latest(request.vaddr, request.vaddr + request.size) > request.needed ||
	                 m_device_address_tick > request.needed || m_device_address_pending;
	// A write of the operation being recorded supersedes the copy (chenxiao07's KYTY_READBACK_DETACH):
	// the request is dropped, the range stays GPU-modified and the guest faults again once the write
	// is recorded, for its bytes, instead of a synchronous readback of bytes about to change.
	bool superseded = false;
	for (const auto& [address, bytes]: m_pending_writes) {
		superseded = superseded || (address < request.vaddr + request.size && request.vaddr < address + bytes);
	}
	(void)request.device.resetFences(1, &request.fence);
	request.finished.store(true, std::memory_order_release);
	m_async_readbacks[request.slot].request.reset();
	if (superseded || !IsRegionRegistered(request.vaddr, request.size)) {
		return;
	}
	if (!rewritten) {
		for (const auto& copy: request.copies) {
			m_gpu_modified_ranges.Subtract(request.buffer_address + copy.srcOffset, copy.size);
		}
		m_memory_tracker.UnmarkRegionAsGpuModified(request.vaddr, request.size);
		return;
	}
	++m_async_readback_rewritten;
	auto& buffer = m_slot_buffers[FindBuffer(request.vaddr, request.size)];
	if (DownloadBufferMemory<false>(buffer, request.vaddr, request.size)) {
		m_memory_tracker.UnmarkRegionAsGpuModified(request.vaddr, request.size);
	}
}


// Execution thread: the copies in flight of bytes of the range (or of the buffer `source`) are
// written back and finished, as an access of them that does not wait for them needs.
void BufferCache::DrainAsyncReadbacks(uint64_t vaddr, uint64_t size, vk::Buffer source) {
	for (auto& slot: m_async_readbacks) {
		const auto request = slot.request;
		if (!request || request->finished.load(std::memory_order_acquire)) {
			continue;
		}
		const bool overlaps = source ? request->source == source
		                             : vaddr < request->vaddr + request->size && request->vaddr < vaddr + size;
		if (!overlaps) {
			continue;
		}
		CompleteAsyncReadback(*request);
	}
}

// KYTY_READBACK_WINDOW=0: a guest access of GPU-written memory reads back its page only.
static bool ReadbackWindowEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_READBACK_WINDOW");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// What a guest access of GPU-written memory reads back. The guest reads (and fills) such memory in
// long runs, faulting page after page, and each page was a round trip to the execution thread and
// the GPU: Demon's Souls takes thousands a second. The readback takes the GPU-written bytes of the
// 512 KiB around the access with it (as chenxiao07's KytyPS5 fork does), within the access's
// buffer and guest mappings, when the readback queue can copy all of them without waiting for
// work still being recorded; otherwise only the access's page.
std::pair<uint64_t, uint64_t> BufferCache::ReadbackWindow(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	constexpr uint64_t WindowSize = 512 * 1024;
	const std::pair<uint64_t, uint64_t> exact {vaddr, size};
	if (!ReadbackWindowEnabled() || !m_graphics.readback_queue || m_device_address_pending) {
		return exact;
	}
	const auto buffer_begin = buffer.CpuAddress();
	const auto buffer_end   = buffer_begin + buffer.Size();
	const auto begin        = std::max(Common::AlignDown(vaddr, WindowSize), buffer_begin);
	const auto end          = std::min(Common::AlignDown(vaddr, WindowSize) + WindowSize, buffer_end);
	if (begin > vaddr || end < vaddr + size || !m_scheduler.Context().IsMapped(begin, end - begin)) {
		return exact;
	}
	for (const auto& [address, bytes]: m_pending_writes) {
		if (address < end && begin < address + bytes) {
			return exact;
		}
	}
	const auto current = m_scheduler.CurrentTick();
	bool       ready   = true;
	m_memory_tracker.ForEachDownloadRange<false>(begin, end - begin, [&](uint64_t address, uint64_t bytes) noexcept {
		m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t stop) {
			ready = ready && m_write_ticks.Latest(start, stop) < current;
		});
	});
	return ready ? std::pair {begin, end - begin} : exact;
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	auto& gpu = m_scheduler.Context().GetGpu();
	// The synchronous path, once the copies in flight are written back: the GPU's bytes, then a
	// write makes the range CPU-modified.
	// (Reads of the GPU threads themselves, for the work they record, keep to their range.)
	const bool guest       = !GuestGpu::IsGpuThread();
	const auto synchronous = [this, vaddr, size, is_write, guest] {
		DrainAsyncReadbacks(vaddr, size);
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
		// Bytes a finished copy snapshot holds need no readback.
		if (const auto page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE),
		    end = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
		    buffer.IsInBounds(page, end - page) && TryReadCopyFeedback(buffer, page, end - page)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(page, end - page);
			if (is_write) {
				m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			}
			return;
		}
		const auto [begin, bytes] = guest ? ReadbackWindow(buffer, vaddr, size) : std::pair {vaddr, size};
		if (begin != vaddr || bytes != size) {
			DrainAsyncReadbacks(begin, bytes);
		}
		if (DownloadBufferMemory<false>(buffer, begin, bytes)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(begin, bytes);
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
	};
	// KYTY_GUEST_READBACK_ASYNC=1, and KYTY_GUEST_WRITE_READBACK_ASYNC=1 for writes too.
	if ((!is_write || AsyncWriteReadbacksEnabled()) && !GuestGpu::IsGpuThread() && AsyncReadbacksEnabled()) {
		std::shared_ptr<AsyncReadback> request;
		gpu.SendUrgentCommandSync([&] {
			request = BeginAsyncReadback(vaddr, size);
			if (!request) {
				synchronous();
			}
		});
		// The copy completes on the GPU while the execution thread goes on; its bytes are
		// written back by an urgent command, as the synchronous path writes them. A write then
		// makes the range CPU-modified (synchronous() finds nothing left to read back).
		if (request) {
			const bool copied = WaitAsyncReadback(*request);
			if (copied || is_write) {
				gpu.SendUrgentCommandSync([&] {
					if (copied) {
						CompleteAsyncReadback(*request);
					}
					if (is_write) {
						synchronous();
					}
				});
			}
		}
		return;
	}
	// The guest reads (or writes) memory the GPU wrote only after a label the execution thread
	// writes once the work before it executed: the operations queued since cannot be what the
	// access needs, so the readback runs before them instead of after the whole backlog (a
	// frame of draws, 100-200 ms while an area streams in).
	m_scheduler.Context().GetGpu().SendUrgentCommandSync(synchronous);
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
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

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

// chenxiao07's KYTY_ASYNC_REPROTECT (476ae208b): the page protections a read upload adds (system calls, hundreds a
// frame) are applied by the host copy thread, before its copies of the uploaded ranges; 0 applies them here.
static bool AsyncReprotectEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_ASYNC_REPROTECT");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

static uint32_t AsyncUploadMode();

static void ReapplyProtectionCall(void* page_manager, uint64_t vaddr, uint64_t size) {
	static_cast<PageManager*>(page_manager)->ReapplyProtection(vaddr, size);
}

void BufferCache::ApplyDeferredProtection(bool queue) {
	auto* copies = queue ? m_scheduler.HostCopies() : nullptr;
	for (const auto& [vaddr, size]: m_deferred_protection) {
		if (copies != nullptr) {
			copies->Call(&ReapplyProtectionCall, m_page_manager, vaddr, size);
		} else {
			m_page_manager->ReapplyProtection(vaddr, size);
		}
	}
	m_deferred_protection.clear();
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	// A written range the GPU-modified set covers whole has nothing to upload and its pages are GPU-owned already
	// (chenxiao07's KYTY_BUFFER_RESIDENCY): ranges join the set only after their pages are marked GPU-owned, and a
	// readback subtracts them before the pages stop being so (CPU writes and unmaps take that path). This skips the
	// tracker's walk under its region locks; a hole or a partial cover takes it.
	if (is_written && t_download_depth == 0 && BufferResidencyEnabled() &&
	    m_gpu_modified_ranges.Contains(vaddr, size)) {
		return false;
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	// A read upload's new write protections wait until its copies are known: queued with them when they are all
	// queued (a guest write before the protection lands in the copy, one after it faults), applied here otherwise.
	const bool defer = !is_written && AsyncReprotectEnabled() && AsyncUploadMode() >= 1 &&
	                   m_scheduler.HostCopies() != nullptr && m_staging_buffer.IsCoherent() &&
	                   GuestGpu::IsGpuThread();
	if (defer) {
		PageManager::SetDeferredProtectionSink(&m_deferred_protection);
	}
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept {
		    if (defer) {
			    PageManager::SetDeferredProtectionSink(nullptr);
		    }
		    source = UploadCopies(buffer, copies, total_size);
	    });
	if (defer) {
		PageManager::SetDeferredProtectionSink(nullptr);
		ApplyDeferredProtection(false); // (none were queued: nothing to upload)
	}
	if (source && m_upload_batching && source == m_staging_buffer.Handle()) {
		m_upload_batch_buffers.emplace_back(buffer.Handle(), static_cast<uint32_t>(copies.size()));
		m_upload_batch_copies.insert(m_upload_batch_copies.end(), copies.begin(), copies.end());
	} else if (source) {
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
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
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

// chenxiao07's KYTY_ASYNC_UPLOAD (a678f274b): 1 (default) the buffer uploads' reads of guest memory, 2 the image
// stagings' too, go to the host copy thread instead of the execution thread; 0 off.
static uint32_t AsyncUploadMode() {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_ASYNC_UPLOAD");
		return value != nullptr && value[0] != '\0' ? static_cast<uint32_t>(value[0] - '0') : 2u;
	}();
	return mode;
}

// As StreamGuestData(): the copy reads the backing store (the guest range's bytes, never protected) before the
// submission that uses the staging memory (HostCopyQueue::WaitFor), and unmaps drain the queue first.
bool BufferCache::QueueGuestRead(uint8_t* destination, uint64_t vaddr, uint64_t size, uint32_t mode) {
	if (AsyncUploadMode() < mode || !m_staging_buffer.IsCoherent() || !GuestGpu::IsGpuThread()) {
		return false;
	}
	auto* copies = m_scheduler.HostCopies();
	if (copies == nullptr) {
		return false;
	}
	const void* source = Libs::LibKernel::Memory::FindBackingPointer(vaddr, size);
	if (source == nullptr) {
		return false;
	}
	copies->Copy(destination, source, size);
	return true;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}

	// Within a batch of uploads (see FlushUploadBatch()) the staging buffer must not wait for
	// the current command buffer: that would submit it without the batched copies. A copy that
	// does not fit then goes through a buffer of its own.
	// Deferred protections go first, on the copy thread when every copy goes there.
	if (!m_deferred_protection.empty()) {
		bool all_queued = AsyncUploadMode() >= 1;
		for (const auto& copy: copies) {
			all_queued = all_queued &&
			             Libs::LibKernel::Memory::FindBackingPointer(buffer.CpuAddress() + copy.dstOffset, copy.size) != nullptr;
		}
		ApplyDeferredProtection(all_queued);
	}
	const auto source = UploadMemory(m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (QueueGuestRead(mapped + copy.srcOffset, address, copy.size, 1)) {
				copy.srcOffset += base_offset;
				continue;
			}
			if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset, copy.size)) {
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

vk::DeviceAddress BufferCache::TableUpload(const void* data, uint64_t size) {
	if (!m_table_upload) {
		m_table_upload = std::make_unique<StreamBuffer>(m_graphics, m_scheduler, MemoryUsage::Stream, 32 * MiB,
		                                                AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress);
	}
	return m_table_upload->BufferDeviceAddress() + m_table_upload->Copy(data, size, 16);
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
	// it is in the stream buffer and no work wrote cached buffers. A copy made in the window
	// answers before the CPU-dirty checks: it was made from CPU-dirty pages that the guest has not
	// written since, and GPU writes advance the generation (or show in the GPU-dirty check).
	const bool    small_read = !is_written && size <= CACHING_PAGESIZE;
	StreamBuffer& ring       = m_host_upload ? *m_host_upload : m_stream_buffer;
	StreamMemo*   stream     = nullptr;
	uint64_t    window     = 0;
	if (small_read && !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		window = m_scheduler.Context().GetRenderExecutor().DrawWindow();
		stream = &(*m_stream_memo)[((vaddr ^ (size << 40u)) * 0x9e3779b97f4a7c15ull >> 32u) %
		                           StreamMemoSlots];
		if (stream->vaddr == vaddr && stream->size == size && stream->window == window &&
		    stream->lap == ring.Lap() &&
		    stream->write_generation == m_gpu_write_generation) {
			return {&ring, stream->offset};
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
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = ring.Map(size, alignment, false);
		if (mapped != nullptr) {
			StreamGuestData(mapped, vaddr, size);
			ring.Commit();
			*stream = {.vaddr            = vaddr,
			           .size             = size,
			           .window           = window,
			           .lap              = ring.Lap(),
			           .write_generation = m_gpu_write_generation,
			           .offset           = offset};
			return {&ring, offset};
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
		if (QueueGuestRead(mapped, vaddr, size, 2)) {
			return;
		}
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
		m_device_address_pending = false;
		m_device_address_tick    = tick;
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
		return false;
	}
	// Writes of the operation being recorded have no tick yet.
	for (const auto& [address, bytes]: m_pending_writes) {
		if (address < vaddr + size && vaddr < address + bytes) {
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
		return false;
	}
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
				return false;
			}
		}
		needed = std::max(needed, m_write_ticks.Latest(begin, end));
	}
	const bool device_address = needed != 0 && needed == m_device_address_tick;
	if (needed >= m_scheduler.CurrentTick()) {
		NoteReadbackWait(readback_begin, readback_end, device_address, true);
		return false;
	}
	NoteReadbackWait(readback_begin, readback_end, device_address, false);
	// The copy waits for the needed submission on the GPU, but must not be submitted before it:
	// the driver may then block inside vkQueueSubmit, holding locks a submission of the
	// recording thread (or a present) needs.
	while (m_scheduler.SubmittedTick() < needed) {
		Common::SpinPause();
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
	if (m_graphics.readback_queue.submit(1, &submit, m_readback.fence) != vk::Result::eSuccess) {
		EXIT("BufferCache: readback queue submission failed\n");
	}
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
	(void)device.resetFences(1, &m_readback.fence);
	// Earlier asynchronous publications of these bytes run first, as in the queued path.
	m_scheduler.WaitPriorityOperations(needed);
	staging.Invalidate(0, total_size);
	const auto* mapped = staging.Mapped().data();
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
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
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
		// A copy in flight on the readback queue may still read it (DeleteBuffer() does the same).
		DrainAsyncReadbacks(0, 0, buffer.Handle());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::ProcessUseBuffer() {
	m_fault_manager.ProcessUseBuffer([this](std::span<const uint64_t> pages) {
		for (const auto page: pages) {
			const auto* owner = m_page_table.Find(page >> PageTable::kPageBits);
			if (owner != nullptr && *owner) {
				TouchBuffer(m_slot_buffers[*owner]);
			}
		}
	});
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
	if (m_upload_batch_buffers.empty()) {
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
	for (const auto& [destination, count]: m_upload_batch_buffers) {
		native.copyBuffer(staging, destination, count, m_upload_batch_copies.data() + first);
		first += count;
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlagBits::eByRegion,
	                       1, &after, 0, nullptr, 0, nullptr);
	m_upload_batch_buffers.clear();
	m_upload_batch_copies.clear();
}

// Each buffer's upload would record its own pair of barriers: record them once, around the
// copies of every buffer (hundreds per frame in GTA V).
struct BufferCache::UploadBatch {
	BufferCache& cache;
	explicit UploadBatch(BufferCache& owner): cache(owner) {
		cache.m_upload_batching   = true;
		cache.m_upload_batch_tick = cache.m_scheduler.CurrentTick();
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

void BufferCache::GrowRegionTables(uint64_t region) {
	if (region < m_region_buffer_count.size()) {
		return;
	}
	const auto words = region / 64 + 1;
	m_region_buffer_count.resize(words * 64, 0);
	m_synced_region_epochs.resize(words * 64, 0);
	m_buffer_region_bits.resize(words, 0);
	m_forced_region_bits.resize(words, 0);
}

void BufferCache::NoteBufferRegions(const Buffer& buffer, bool insert) {
	const auto first = buffer.CpuAddress() / TRACKER_REGION_SIZE;
	const auto last  = (buffer.CpuAddress() + buffer.Size() - 1) / TRACKER_REGION_SIZE;
	GrowRegionTables(last);
	for (auto index = first; index <= last; index++) {
		const auto bit = uint64_t {1} << (index % 64);
		if (insert) {
			// A new buffer may cover CPU-dirty pages that no earlier synchronization uploaded.
			m_region_buffer_count[index]++;
			m_buffer_region_bits[index / 64] |= bit;
			m_forced_region_bits[index / 64] |= bit;
		} else {
			EXIT_IF(m_region_buffer_count[index] == 0);
			if (--m_region_buffer_count[index] == 0) {
				m_buffer_region_bits[index / 64] &= ~bit;
			}
		}
	}
}

void BufferCache::ForceSynchronization(uint64_t vaddr, uint64_t size) {
	if (size == 0 || m_buffer_region_bits.empty()) {
		return;
	}
	const auto first = vaddr / TRACKER_REGION_SIZE;
	const auto last  = std::min((vaddr + size - 1) / TRACKER_REGION_SIZE, m_region_buffer_count.size() - 1);
	for (auto index = first; index <= last; index++) {
		m_forced_region_bits[index / 64] |= uint64_t {1} << (index % 64);
	}
}

void BufferCache::SynchronizeBuffersInRanges(const RangeSet& ranges, bool all) {
	const UploadBatch batch {*this};
	// Read each region's epoch before scanning it: pages dirtied during or after the scan advance
	// it again, and the next call visits the region again. Regions without buffers have nothing
	// to upload; a buffer registered later forces its regions.
	m_sync_regions.clear();
	for (size_t word = 0; word < m_buffer_region_bits.size(); word++) {
		const auto forced = m_forced_region_bits[word];
		m_forced_region_bits[word] = 0;
		for (auto bits = m_buffer_region_bits[word]; bits != 0; bits &= bits - 1) {
			const auto index = word * 64 + static_cast<uint64_t>(std::countr_zero(bits));
			const auto epoch = m_memory_tracker.RegionCpuDirtyEpoch(index);
			// Synchronizing a buffer tracks its pages, so an untracked region (epoch 0) has no
			// pending write under a buffer unless the region was forced.
			if (!all && (forced & (uint64_t {1} << (index % 64))) == 0 &&
			    (epoch == 0 || m_synced_region_epochs[index] == epoch)) {
				continue;
			}
			m_sync_regions.emplace_back(index, epoch);
		}
	}
	for (const auto& [index, epoch]: m_sync_regions) {
		ranges.ForEachInRange(index * TRACKER_REGION_SIZE, TRACKER_REGION_SIZE,
		                      [this](uint64_t start, uint64_t end) {
			                      SynchronizeBuffersInRange(start, end - start);
		                      });
		m_synced_region_epochs[index] = epoch;
	}
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	// Only tracking regions with pending CPU writes can have anything to upload.
	m_memory_tracker.ForEachCpuDirtySpan(vaddr, size, [&](uint64_t span, uint64_t span_size) {
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
				(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
			}
		}
	});
}

} // namespace Libs::Graphics
