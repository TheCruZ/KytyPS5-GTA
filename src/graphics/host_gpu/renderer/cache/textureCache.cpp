#include "common/hackNotice.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "kernel/memory.h"

#include <fmt/format.h>
#include <algorithm>
#include <filesystem>
#include <array>
#include <bit>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <span>
#include <tuple>
#include <utility>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr uint64_t NumFramesBeforeRemoval = 32;
// Overlapping aliases (transient render targets sharing memory) stay registered until unused for
// this many presented frames as well; see AliasAgeByFrames.
constexpr uint64_t AliasFramesBeforeRemoval = 4;

// An overlapped alias was freed once unused for NumFramesBeforeRemoval scheduler ticks. A title
// submits hundreds of times a frame, so that is usually within the frame that used it: the
// aliases of a transient image pool (Wolverine: ~4 a frame) were destroyed and recreated from
// guest memory every frame. They now also have to be unused for AliasFramesBeforeRemoval
// presented frames, and live aliases follow a single-owner rule (CommitGpuWrite).
// KYTY_IMAGE_ALIAS_AGE=0 (or =ticks) restores the tick rule alone.
[[nodiscard]] bool AliasAgeByFrames() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_IMAGE_ALIAS_AGE");
		return value == nullptr ||
		       (std::strcmp(value, "0") != 0 && std::strcmp(value, "ticks") != 0);
	}();
	return enabled;
}

// The collector writes a GPU-written image back to guest memory before it frees it (under
// memory pressure). Whether the image is CPU-dirty is checked when the copy is recorded, but the
// bytes land only once the GPU is done with that tick, and the image is untracked by then: a
// guest that reused the memory in between (a retired render target becoming a CPU-written vertex
// table, as in Jetsku's 919e712a) had it overwritten with stale pixels. The guest bytes are hashed
// when the copy is recorded and again before it is published; if they changed, the write-back
// is dropped. KYTY_EVICT_WRITEBACK_GUARD=0 publishes unconditionally again.
[[nodiscard]] bool EvictWriteBackGuard() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_EVICT_WRITEBACK_GUARD");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// Hash of the guest backing bytes of a range, read in chunks; false if a chunk is unreadable.
[[nodiscard]] bool HashGuestBacking(uint64_t address, uint64_t size, uint64_t& hash) {
	constexpr uint64_t   Chunk = 64 * 1024;
	std::vector<uint8_t> bytes(static_cast<size_t>(std::min(size, Chunk)));
	uint64_t             seed = 0;
	for (uint64_t done = 0; done < size;) {
		const auto part = std::min(size - done, Chunk);
		if (!LibKernel::Memory::TryReadBacking(address + done, bytes.data(), part)) {
			return false;
		}
		seed = XXH3_64bits_withSeed(bytes.data(), static_cast<size_t>(part), seed);
		done += part;
	}
	hash = seed;
	return true;
}
// Largest GPU-written tiled image (in texels) that is written back to guest memory on submit.
constexpr uint64_t SmallTiledReadbackTexels = 64 * 64;

[[nodiscard]] bool DecodeColorClear(const TextureCache::ImageDesc& desc, uint8_t code,
                                  vk::ClearColorValue& clear) {
	const auto& metadata = desc.info.metadata;
	const auto  format   = desc.info.IsDepth()
	                           ? TextureGetSurfaceFormatInfo(desc.info.guest_format).vk_format
	                           : desc.view_info.format;
	const bool  cmask    = metadata.kind == ImageMetadataKind::Cmask;
	// PS5 DCC fast clears to the CB_COLOR_CLEAR_WORD value fill the keys with 0x10, not the
	// GFX10 0x20.
	if (cmask ? code == 0 : (code == 0x20 || code == 0x10)) {
		// Register clears belong to the color buffer; the texture pipe cannot decode them.
		if (desc.type != TextureCache::BindingType::RenderTarget || !metadata.clear_register_valid) {
			return false;
		}
		if (DecodePackedColorClear(format, metadata.clear_word, clear)) {
			return true;
		}
		// KYTY_CLEAR_REGISTER_WIDE: 64-bit, 16-bit-channel and B10G11R11 targets (both words).
		if (!DecodePackedColorClear64(format, static_cast<uint32_t>(metadata.clear_word),
		                                 static_cast<uint32_t>(metadata.clear_word >> 32u), clear)) {
			return false;
		}
		static std::mutex                   logged_mutex;
		static std::unordered_set<uint32_t> logged;
		if (std::lock_guard lock(logged_mutex); logged.insert(static_cast<uint32_t>(format)).second) {
			LOGF("Register clear of format %u decoded with both clear words (was dropped)\n",
			     static_cast<uint32_t>(format));
		}
		return true;
	}
	if (cmask) {
		return false;
	}
	switch (code) {
		case 0x00:
		case 0x40:
		case 0x80:
		case 0xc0: break;
		default: return false;
	}
	clear = {};
	if (code == 0x00) {
		return true;
	}
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eR5G6B5UnormPack16:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR4G4B4A4UnormPack16:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eB10G11R11UfloatPack32: break;
		default: return false;
	}
	const float          rgb   = (code & 0x80u) != 0 ? 1.0f : 0.0f;
	const float          alpha = (code & 0x40u) != 0 ? 1.0f : 0.0f;
	std::array<float, 4> channels {rgb, rgb, rgb, alpha};
	if (!metadata.dcc_alpha_msb) {
		std::swap(channels[0], channels[3]);
	}
	// DCC clear decoding clamps missing lanes before applying the format swizzle.
	const auto components = vk::componentCount(format);
	if (components == 1) {
		channels[0] = channels[3];
	} else if (components == 2) {
		channels[1] = channels[3];
	}
	switch (format) {
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR5G6B5UnormPack16: std::swap(channels[0], channels[2]); break;
		case vk::Format::eR4G4B4A4UnormPack16:
			std::reverse(channels.begin(), channels.end());
			break;
		default: break;
	}
	clear.float32 = channels;
	return true;
}

} // namespace

// KYTY_TEXTURE_CACHE_MB / KYTY_BUFFER_CACHE_MB: an upper bound for a cache's collection thresholds,
// whatever the card's size (default 6 GiB each; 0 = no bound). The thresholds otherwise follow the
// device budget, and on a 32 GB card the caches kept everything up to ~25 GB: textures and buffers
// of scenes the game had left stayed until the card was full (a new game's prologue, 2026-10-07).
static uint64_t CacheCapBytes(const char* name) {
	const char* value = std::getenv(name);
	const auto  mib   = value != nullptr ? std::strtoull(value, nullptr, 10) : 6144ull;
	return mib == 0 ? UINT64_MAX : mib << 20u;
}

TextureCache::TextureCache(GraphicContext& graphics, CommandScheduler& scheduler,
                           PageManager& page_manager, BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager),
      m_blit_helper(graphics, scheduler),
      m_tiler(graphics, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)),
      m_buffer_cache(buffer_cache),
      m_readback_linear_images(Config::ReadbackLinearImagesEnabled()),
      m_readback_tiled_images(Config::ReadbackTiledImagesEnabled()) {
	if (m_graphics.CanReportMemoryUsage()) {
		constexpr int64_t GiB = 1024ll * 1024 * 1024;
		const auto        budget =
		    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
		const auto threshold = std::min<int64_t>(budget, 8 * GiB);
		m_pressure_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 6 * threshold / 10, budget - GiB), GiB + GiB / 2));
		m_critical_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 2 * threshold / 10, budget - GiB / 2), 3 * GiB));
		m_trigger_gc_memory = static_cast<uint64_t>(std::max<int64_t>((budget - threshold) / 2, 0));
	}
	if (const auto cap = CacheCapBytes("KYTY_TEXTURE_CACHE_MB"); cap != UINT64_MAX) {
		m_pressure_gc_memory = std::min(m_pressure_gc_memory, cap);
		m_critical_gc_memory = std::min(m_critical_gc_memory, cap + cap / 4);
		m_trigger_gc_memory  = std::min(m_trigger_gc_memory, cap / 4 * 3);
		LOGF("TextureCache: thresholds capped at %" PRIu64 " MB (trigger %" PRIu64 ", critical %" PRIu64
		     " MB)\n",
		     m_pressure_gc_memory >> 20u, m_trigger_gc_memory >> 20u, m_critical_gc_memory >> 20u);
	}
}

TextureCache::~TextureCache() {
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		if (image.registered) {
			UnregisterImage(id);
		}
	});
}

bool TextureCache::SameBacking(const ImageInfo& cached, const ImageInfo& requested,
                               bool exact_format) {
	if (cached.data.address != requested.data.address) {
		return false;
	}
	if (cached.data.size != requested.data.size) {
		return false;
	}
	if (cached.extent != requested.extent) {
		return false;
	}
	if (cached.resources.levels < requested.resources.levels ||
	    cached.resources.layers < requested.resources.layers) {
		return false;
	}
	if (cached.samples != requested.samples) {
		return false;
	}
	if (cached.bytes_per_block != requested.bytes_per_block) {
		return false;
	}
	if (cached.tile_mode != requested.tile_mode) {
		return false;
	}
	if (cached.GetColorTransform() != requested.GetColorTransform()) {
		return false;
	}
	if (!ImageViewOps::FormatsCompatible(cached.pixel_format, requested.pixel_format) ||
	    cached.type != requested.type) {
		return false;
	}
	if (exact_format && cached.pixel_format != requested.pixel_format) {
		return false;
	}
	return true;
}

TextureCache::BindingType TextureCache::UploadBinding(const Image& image) {
	if (image.info.IsDepth()) {
		if (image.info.tile_mode == Prospero::TileMode::kDepth ||
		    image.info.tile_mode == Prospero::TileMode::kLinear) {
			return BindingType::DepthTarget;
		}
		return BindingType::Texture;
	}
	if (image.usage.render_target) {
		return BindingType::RenderTarget;
	}
	if (image.usage.video_out) {
		return BindingType::VideoOut;
	}
	return image.usage.storage ? BindingType::Storage : BindingType::Texture;
}

ImageId TextureCache::InsertImage(const ImageInfo& info) {
	const auto id = m_slot_images.insert(m_graphics, m_scheduler, info);
	m_slot_images[id].self_id = id;
	Profiler::Add(Profiler::Counter::ImagesCreated);
	RegisterImage(id);
	return id;
}

TextureCache::RegisteredPageCounts::~RegisteredPageCounts() {
	for (auto& leaf: m_leaves) {
		delete leaf.load(std::memory_order_relaxed);
	}
}

void TextureCache::RegisteredPageCounts::Add(size_t page, int32_t delta) {
	auto& slot = m_leaves[page >> kLeafBits];
	auto* leaf = slot.load(std::memory_order_relaxed);
	if (leaf == nullptr) {
		leaf = new Leaf {};
		slot.store(leaf, std::memory_order_release);
	}
	(*leaf)[page & (kLeafSize - 1)].fetch_add(static_cast<uint32_t>(delta),
	                                          std::memory_order_release);
}

bool TextureCache::RegisteredPageCounts::Any(size_t first, size_t last_exclusive) const noexcept {
	for (auto page = first; page < last_exclusive; page++) {
		const auto* leaf = m_leaves[page >> kLeafBits].load(std::memory_order_acquire);
		if (leaf != nullptr &&
		    (*leaf)[page & (kLeafSize - 1)].load(std::memory_order_acquire) != 0) {
			return true;
		}
	}
	return false;
}

void TextureCache::RegisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.registered || image.info.data.Empty()) {
		EXIT("TextureCache: invalid image registration\n");
	}
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: image registration is outside the guest address space\n");
	}
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		m_image_page_table[page].push_back(id);
		m_registered_pages.Add(page, 1);
	});
	image.registered = true;
	NoteBindlessStateChange(image);
	image.lru_tick   = LruClock();
	image.lru_id     = m_lru_cache.Insert(id, image.lru_tick);
	const auto epoch = ++m_image_set_epoch;
	m_image_set_changes[epoch % ImageSetChangeHistory] = {image.info.data.address,
	                                                      image.info.data.size};
	ForEachPage(image.info.data.address, image.info.data.size,
	            [this, epoch](uint64_t page) { m_image_page_epochs[page] = epoch; });
	m_total_used_memory += image.AccountedSize();
}

void TextureCache::UnregisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	if (image.bindless_pinned) {
		NoteBindlessStateChange(image);
		image.bindless_pinned = false;
		if (on_bindless_unregister) {
			on_bindless_unregister(id);
		}
	}
	UntrackImage(id);
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: registered image is outside the guest address space\n");
	}
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr || !owners->Erase(id)) {
			EXIT("TextureCache: image missing from page owner index\n");
		}
		m_registered_pages.Add(page, -1);
	});
	m_lru_cache.Free(image.lru_id);
	const auto accounted = image.AccountedSize();
	if (accounted > m_total_used_memory) {
		EXIT("TextureCache: image accounting underflow\n");
	}
	m_total_used_memory -= accounted;
	image.registered = false;
	const auto epoch = ++m_image_set_epoch;
	m_image_set_changes[epoch % ImageSetChangeHistory] = {image.info.data.address,
	                                                      image.info.data.size};
	ForEachPage(image.info.data.address, image.info.data.size,
	            [this, epoch](uint64_t page) { m_image_page_epochs[page] = epoch; });
}

bool TextureCache::HasImagesInRegion(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	return FindImagesInRegion(address, size, false).size() != 0;
}

uint64_t TextureCache::ImageEpochInRegion(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	return ImageEpochInRegionUnlocked(address, size);
}

uint64_t TextureCache::ImageEpochInRegionUnlocked(uint64_t address, uint64_t size) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
		return UINT64_MAX;
	}
	uint64_t epoch = 0;
	ForEachPage(address, size, [&](uint64_t page) {
		if (const auto* page_epoch = m_image_page_epochs.Find(page)) {
			epoch = std::max(epoch, *page_epoch);
		}
	});
	return epoch;
}

size_t TextureCache::ExactLookupSlot(const ImageInfo& info) noexcept {
	uint64_t hash = info.data.address * 0x9e3779b97f4a7c15ull;
	hash ^= (info.data.size + static_cast<uint64_t>(info.pixel_format)) * 0xc2b2ae3d27d4eb4full;
	hash ^= (static_cast<uint64_t>(info.extent.width) << 32u | info.extent.height) *
	        0x165667b19e3779f9ull;
	return static_cast<size_t>(hash >> 32u) % ExactLookupWays;
}

void TextureCache::DeleteImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	ReleaseDepthFeedbackCopy(id);
	if (image->depth_id) {
		std::erase(m_slot_images[image->depth_id].stencil_ids, id);
	} else {
		// Detach first: deleting each association also unlinks it from its depth owner.
		for (const auto association: std::exchange(image.stencil_ids, {})) {
			FreeImage(association);
		}
	}
	if (image.IsGpuModified()) {
		EXIT("TextureCache: deleting a GPU-modified image without resolving its contents\n");
	}
	m_download_images.erase(id);
	if (image.info.metadata.kind == ImageMetadataKind::Htile) {
		// A newer image at the same HTile address may already own the state: dropping it would make
		// that image clear its depth again in the middle of the frame.
		const auto meta = m_surface_metas.find(image.info.metadata.range.address);
		if (meta != m_surface_metas.end() && (!meta->second.owner || meta->second.owner == id)) {
			m_surface_metas.erase(meta);
		}
	}
	UnregisterImage(id);
	// Also between guest command buffers (a guest unmap runs there): command buffers submitted
	// earlier may still be sampling the image on the GPU. Destroying it at once let them read
	// freed memory (device loss at the title, 2026-10-04).
	m_scheduler.DeferRelease([this, id] { ReleaseImage(id); });
}

void TextureCache::ReleaseImage(ImageId id) {
	const auto* image = m_slot_images.try_get(id);
	if (image != nullptr && on_image_release && on_image_release(*image)) {
		// Work recorded with the repointed slot runs until the current tick is done.
		m_scheduler.DeferRelease([this, id] { ReleaseImage(id); });
		return;
	}
	if (image != nullptr && !m_scheduler.IsFree(image->tick_accessed_last)) {
		// Used after it was dropped, in a tick the GPU has not finished: keep it until then.
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 16) {
			LOGF("TextureCache: image %ux%u guest=0x%016" PRIx64 " used in tick %" PRIu64
			     " after its release was queued; destruction waits for that tick\n",
			     image->info.extent.width, image->info.extent.height, image->info.data.address,
			     image->tick_accessed_last);
		}
		m_scheduler.DeferRelease([this, id] { ReleaseImage(id); });
		return;
	}
	m_slot_images.erase(id);
	Profiler::Add(Profiler::Counter::ImagesDestroyed);
}

void TextureCache::ReleaseDepthFeedbackCopy(ImageId depth_id) {
	const auto found = m_depth_feedback_copies.find(depth_id);
	if (found == m_depth_feedback_copies.end()) {
		return;
	}
	const auto copy_id = found->second;
	m_depth_feedback_copies.erase(found);
	// The copy is never registered: queued work may still sample it.
	if (m_scheduler.Active()) {
		m_scheduler.DeferHostOperation([this, copy_id] { m_slot_images.erase(copy_id); });
	} else {
		m_slot_images.erase(copy_id);
	}
}

ImageId TextureCache::CopyDepthForFeedback(ImageId depth_id, const ImageSubresourceRange& range,
                                           vk::ImageAspectFlags aspects) {
	if (const auto found = m_depth_feedback_copies.find(depth_id);
	    found != m_depth_feedback_copies.end()) {
		const auto& depth = m_slot_images[depth_id].backing;
		const auto& copy  = m_slot_images[found->second].backing;
		if (copy.format != depth.format || copy.extent != depth.extent ||
		    copy.mip_levels != depth.mip_levels || copy.layers != depth.layers ||
		    copy.samples != depth.samples) {
			ReleaseDepthFeedbackCopy(depth_id);
		}
	}
	auto [entry, inserted] = m_depth_feedback_copies.try_emplace(depth_id);
	if (inserted) {
		// Same layout as the depth image, without guest memory: never registered or tracked.
		auto info     = m_slot_images[depth_id].info;
		info.data     = {};
		info.stencil  = {};
		info.metadata = {};
		entry->second = m_slot_images.insert(m_graphics, m_scheduler, info);
	}
	// References after the insertion: it may move the images.
	auto& depth = m_slot_images[depth_id];
	auto& copy  = m_slot_images[entry->second];
	EXIT_IF(range.base_level + range.level_count > depth.backing.mip_levels ||
	        range.base_layer + range.layer_count > depth.backing.layers);
	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	depth.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, range,
	              command);
	copy.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, range,
	             command);
	std::vector<vk::ImageCopy> copies;
	for (const auto aspect: {vk::ImageAspectFlagBits::eDepth, vk::ImageAspectFlagBits::eStencil}) {
		if (!(aspects & aspect)) {
			continue;
		}
		for (uint32_t level = range.base_level; level < range.base_level + range.level_count;
		     level++) {
			vk::ImageCopy region {};
			region.srcSubresource = {aspect, level, range.base_layer, range.layer_count};
			region.dstSubresource = region.srcSubresource;
			region.extent = vk::Extent3D {std::max(depth.backing.extent.width >> level, 1u),
			                              std::max(depth.backing.extent.height >> level, 1u), 1u};
			copies.push_back(region);
		}
	}
	if (!copies.empty()) {
		command.copyImage(depth.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  copy.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                  static_cast<uint32_t>(copies.size()), copies.data());
	}
	return entry->second;
}

void TextureCache::FreeImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.IsGpuModified()) {
		image.ClearGpuModified();
	}
	DeleteImage(id);
}

void TextureCache::PinImage(ImageId id) {
	if (auto* image = m_slot_images.try_get(id); image != nullptr) {
		image->table_pins++;
	}
}

void TextureCache::UnpinImage(ImageId id) {
	if (auto* image = m_slot_images.try_get(id); image != nullptr && image->table_pins != 0) {
		image->table_pins--;
	}
}

void TextureCache::TouchImage(Image& image) {
	// One load of the frame counter for both uses (LruClock() adds the garbage-collection ticks).
	const auto frames = m_graphics.presented_frames.load(std::memory_order_relaxed);
	if (image.frame_accessed_last != frames) {
		image.frame_accessed_last = frames;
	}
	// Draws touch the same images many times per clock value: only the first touch moves the entry.
	if (image.registered) {
		if (const auto clock = frames + m_gc_tick / 512; image.lru_tick < clock) {
			image.lru_tick = clock;
			m_lru_cache.Touch(image.lru_id, clock);
		}
	}
}

void TextureCache::MarkAsMaybeDirty(ImageId id, Image& image) {
	image.MarkMaybeCpuDirty();
	UntrackImage(id);
}

void TextureCache::TrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	const auto image_end   = image.info.data.End();
	if (image_begin == image.track_addr && image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked()) {
		image.track_addr     = image_begin;
		image.track_addr_end = image_end;
		m_page_manager.UpdatePageWatchers<true>(image_begin, image.info.data.size);
		return;
	}
	if (image_begin < image.track_addr) {
		TrackImageHead(id);
	}
	if (image.track_addr_end < image_end) {
		TrackImageTail(id);
	}
}

void TextureCache::TrackImageHead(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	if (image_begin == image.track_addr) {
		return;
	}
	if (!image.IsTracked() || image_begin > image.track_addr) {
		EXIT("TextureCache: invalid image head tracking range\n");
	}
	const auto size  = image.track_addr - image_begin;
	image.track_addr = image_begin;
	m_page_manager.UpdatePageWatchers<true>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_end = image.info.data.End();
	if (image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked() || image.track_addr_end > image_end) {
		EXIT("TextureCache: invalid image tail tracking range\n");
	}
	const auto address   = image.track_addr_end;
	const auto size      = image_end - address;
	image.track_addr_end = image_end;
	m_page_manager.UpdatePageWatchers<true>(address, size);
}

void TextureCache::UntrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.IsTracked()) {
		return;
	}
	const auto address   = image.track_addr;
	const auto size      = image.track_addr_end - image.track_addr;
	image.track_addr     = 0;
	image.track_addr_end = 0;
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::UntrackImageHead(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto begin = image.info.data.address;
	if (!image.IsTracked() || begin < image.track_addr) {
		return;
	}
	const auto address = Common::AlignDown(begin + TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE);
	const auto size    = address - begin;
	image.track_addr   = address;
	image.HeapChanged();
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(begin, size);
	}
}

void TextureCache::UntrackImageTail(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto end   = image.info.data.End();
	if (!image.IsTracked() || image.track_addr_end < end) {
		return;
	}
	const auto address   = Common::AlignDown(end, TRACKER_PAGE_SIZE);
	const auto size      = end - address;
	image.track_addr_end = address;
	image.HeapChanged();
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::TrackImageDownload(ImageId id, Image& image) {
	// Small tiled targets are written back too: games read GPU results such as reductions and
	// simulation state from them on the CPU (PPSA04263 keeps the origin of its ocean wave grid
	// in a 1x1 R32G32 target, and its water physics read it to sample the wave heights).
	const auto& info   = image.info;
	const auto  texels = static_cast<uint64_t>(info.extent.width) * info.extent.height *
	                     info.extent.depth * info.resources.layers;
	if (info.IsTiled() ? m_readback_tiled_images && texels <= SmallTiledReadbackTexels
	                   : m_readback_linear_images) {
		if (!image.IsGpuModified()) {
			EXIT("TextureCache: cannot enroll a non-GPU-owned image for download\n");
		}
		m_download_images.insert(id);
	}
}

TextureCache::ImageIds TextureCache::FindImagesInRegion(uint64_t address, uint64_t size,
                                                        bool page_overlap) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
		return {};
	}

	ImageIds result;
	ForEachPage(address, size, [&](uint64_t page) {
		const auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr) {
			return;
		}
		owners->ForEach([&](ImageId id) {
			const auto& image = m_slot_images[id];
			if (image.picked || !image.Overlaps(address, size, page_overlap)) {
				return;
			}
			image.picked = true;
			result.push_back(id);
		});
	});
	for (const auto id: result) {
		m_slot_images[id].picked = false;
	}
	return result;
}

void TextureCache::ValidateImageViewDesc(const ImageDesc& desc) const {
	if (desc.view_info.format == vk::Format::eUndefined || desc.view_info.level_count == 0 ||
	    desc.view_info.layer_count == 0 ||
	    desc.view_info.base_level >= desc.info.resources.levels ||
	    desc.view_info.level_count > desc.info.resources.levels - desc.view_info.base_level ||
	    (!desc.info.IsVolume() &&
	     (desc.view_info.base_layer >= desc.info.resources.layers ||
	      desc.view_info.layer_count > desc.info.resources.layers - desc.view_info.base_layer))) {
		EXIT("TextureCache: invalid image view description\n");
	}
	if (desc.type == BindingType::DepthTarget && !IsSupportedDepthTargetFormat(desc.info)) {
		EXIT("TextureCache: unsupported depth image description\n");
	}
	if (desc.type == BindingType::VideoOut && !IsSupportedVideoOutFormat(desc.info)) {
		EXIT("TextureCache: unsupported video-out image description\n");
	}
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression == VideoOutCompression::Unsupported) {
		EXIT("TextureCache: unsupported compressed video-out description\n");
	}
}

void TextureCache::PrepareImageCopy(Image& image) {
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
}

void TextureCache::RefreshCopySource(ImageId id) {
	auto& image = m_slot_images[id];
	RefreshImage(id);
	if (image.IsDefinitelyCpuDirty()) {
		EXIT("TextureCache: image copy source remained CPU-dirty after refresh\n");
	}
}

bool TextureCache::CopyD16(Image& destination, Image& source) {
	// Converts guest D16 between a four-byte host depth aspect and two-byte color or D16 depth.
	const bool source_wide = DepthAspectTransferBytes(source.backing.format) == sizeof(uint32_t);
	if (source_wide == (DepthAspectTransferBytes(destination.backing.format) == sizeof(uint32_t))) {
		return false;
	}
	auto& depth  = source_wide ? source : destination;
	auto& narrow = source_wide ? destination : source;
	if (depth.info.bytes_per_block != sizeof(uint16_t) ||
	    narrow.info.bytes_per_block != sizeof(uint16_t)) {
		return false;
	}
	EXIT_IF(source.backing.samples != 1 || destination.backing.samples != 1 ||
	        source.info.resources.levels != 1 || destination.info.resources.levels != 1 ||
	        source.info.extent != destination.info.extent ||
	        source.info.resources.layers != destination.info.resources.layers);

	const auto     layers = depth.info.resources.layers;
	const uint64_t depth_slice =
	    static_cast<uint64_t>(depth.info.pitch) * depth.info.extent.height * sizeof(uint32_t);
	const uint64_t narrow_slice =
	    static_cast<uint64_t>(narrow.info.pitch) * narrow.info.extent.height * sizeof(uint16_t);
	EXIT_IF(layers == 0 || depth_slice > UINT64_MAX / layers || narrow_slice > UINT64_MAX / layers);
	const std::array depth_copies {
	    vk::BufferImageCopy {0,
	                         depth.info.pitch,
	                         depth.info.extent.height,
	                         {vk::ImageAspectFlagBits::eDepth, 0, 0, layers},
	                         {},
	                         depth.info.extent}};
	const std::array narrow_copies {
	    vk::BufferImageCopy {0,
	                         narrow.info.pitch,
	                         narrow.info.extent.height,
	                         {narrow.info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
	                                                : vk::ImageAspectFlagBits::eColor,
	                          0, 0, layers},
	                         {},
	                         narrow.info.extent}};

	auto depth_buffer  = m_tiler.GetScratchBuffer(depth_slice * layers);
	auto narrow_buffer = m_tiler.GetScratchBuffer(narrow_slice * layers, depth_buffer.buffer);
	const TileManager::D16Layout promote_layout {
	    .width               = depth.info.extent.width,
	    .height              = depth.info.extent.height,
	    .layers              = layers,
	    .source_row_stride   = static_cast<uint64_t>(narrow.info.pitch) * sizeof(uint16_t),
	    .target_row_stride   = static_cast<uint64_t>(depth.info.pitch) * sizeof(uint32_t),
	    .source_slice_stride = narrow_slice,
	    .target_slice_stride = depth_slice,
	};
	const bool d32 = DepthAspectTransferFormat(depth.backing.format) == vk::Format::eD32Sfloat;
	if (source_wide) {
		source.Download(depth_copies, depth_buffer.buffer, depth_buffer.offset, depth_buffer.size);
		m_tiler.ConvertD16(depth_buffer, narrow_buffer, TileManager::D16Direction::Demote, d32,
		                   {.width               = promote_layout.width,
		                    .height              = promote_layout.height,
		                    .layers              = promote_layout.layers,
		                    .source_row_stride   = promote_layout.target_row_stride,
		                    .target_row_stride   = promote_layout.source_row_stride,
		                    .source_slice_stride = promote_layout.target_slice_stride,
		                    .target_slice_stride = promote_layout.source_slice_stride});
		destination.Upload(narrow_copies, narrow_buffer.buffer, narrow_buffer.offset,
		                   narrow_buffer.size);
	} else {
		source.Download(narrow_copies, narrow_buffer.buffer, narrow_buffer.offset,
		                narrow_buffer.size);
		m_tiler.ConvertD16(narrow_buffer, depth_buffer, TileManager::D16Direction::Promote, d32,
		                   promote_layout);
		destination.Upload(depth_copies, depth_buffer.buffer, depth_buffer.offset,
		                   depth_buffer.size);
	}
	return true;
}

void TextureCache::CopyImage(ImageId destination_id, ImageId source_id) {
	RefreshCopySource(source_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: cannot issue an unequal-sample image copy\n");
	}
	PrepareImageCopy(destination);
	if (source.IsBufferModified()) {
		if (source.info.data == destination.info.data) {
			destination.MarkBufferModified();
		}
		return;
	}
	const bool source_depth = source.info.IsDepth();
	const bool dest_depth   = destination.info.IsDepth();
	const bool direct_copy =
	    source.info.GetColorTransform() == destination.info.GetColorTransform() &&
	    (source.backing.image_type == destination.backing.image_type ||
	     (source.backing.image_type != vk::ImageType::e1D &&
	      destination.backing.image_type != vk::ImageType::e1D)) &&
	    (source.backing.format == destination.backing.format ||
	     (!source_depth && !dest_depth &&
	      vk::blockSize(source.backing.format) == vk::blockSize(destination.backing.format)));
	if (direct_copy) {
		destination.CopyImage(source);
	} else if (!CopyD16(destination, source)) {
		if (source.backing.samples != 1 || destination.backing.samples != 1) {
			EXIT("TextureCache: cross-format multisample image copy is unsupported\n");
		}
		auto& copy_buffer = m_buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
		destination.CopyImageWithBuffer(source, copy_buffer, m_tiler);
	}
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
	}
	destination.ClearBufferModified();
}

void TextureCache::CopyImageMip(ImageId destination_id, ImageId source_id, uint32_t mip,
                                uint32_t layer) {
	RefreshCopySource(source_id);
	if (m_slot_images[destination_id].IsBufferModified()) {
		RefreshImage(destination_id);
	}
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.IsBufferModified() || source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: invalid mip-copy ownership or sample count\n");
	}
	destination.CopyMip(source, mip, layer);
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
	}
}

bool TextureCache::DepthOverlapNeedsRecreate(const ImageInfo& requested, BindingType binding,
                                             const Image& cached) {
	// Mirrors ResolveDepthOverlap, which leaves other tile modes to the generic overlap path.
	if ((!cached.info.IsDepth() && !requested.IsDepth()) ||
	    cached.info.tile_mode != requested.tile_mode) {
		return false;
	}
	const bool stencil_match = requested.HasStencil() == cached.info.HasStencil();
	const bool bpp_match     = requested.bytes_per_block == cached.info.bytes_per_block;
	// PPSA04264
	const bool raw_d16_texture =
	    binding == BindingType::Texture && cached.info.IsDepth() &&
	    cached.info.guest_format == Prospero::BufferFormat::k16UNorm &&
	    requested.guest_format == Prospero::BufferFormat::k16UInt &&
	    requested.pixel_format == vk::Format::eR16Uint && cached.backing.samples == 1 &&
	    requested.samples == 1 && requested.data == cached.info.data &&
	    requested.extent == cached.info.extent && requested.resources == cached.info.resources &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    !cached.info.HasMetadata();
	bool recreate = cached.info.resources < requested.resources ||
	                requested.IsVolume() != cached.info.IsVolume();
	switch (binding) {
		case BindingType::Texture:
			recreate |= requested.IsDepth() && !cached.info.IsDepth();
			recreate |= raw_d16_texture;
			break;
		case BindingType::Storage: recreate |= cached.info.IsDepth(); break;
		case BindingType::RenderTarget: recreate |= cached.info.IsDepth(); break;
		case BindingType::DepthTarget:
			recreate |= !cached.info.IsDepth();
			recreate |= cached.info.IsDepth() && !(stencil_match && bpp_match);
			break;
		case BindingType::VideoOut: recreate |= cached.info.IsDepth(); break;
	}
	return recreate;
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
                                          ImageId cached_id) {
	auto& cached = m_slot_images[cached_id];
	if ((!cached.info.IsDepth() && !requested.IsDepth()) ||
	    cached.info.tile_mode != requested.tile_mode) {
		return {};
	}
	// PPSA04264, PPSA04288
	// A partial view retains the entire matching array layout. HTile belongs to
	// the depth source, independently of the data slices copied into color storage.
	const bool retain_cached_layout =
	    requested.samples == 1 && cached.info.samples == 1 && cached.backing.samples == 1 &&
	    requested.bytes_per_block == cached.info.bytes_per_block &&
	    requested.data.address == cached.info.data.address &&
	    requested.data.size < cached.info.data.size && requested.extent == cached.info.extent &&
	    requested.resources.levels == 1 && cached.info.resources.levels == 1 &&
	    requested.resources.layers != 0 && cached.info.resources.layers != 0 &&
	    requested.resources.layers < cached.info.resources.layers &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    requested.mip_layout[0].offset == 0 &&
	    cached.info.mip_layout[0].offset == 0 &&
	    requested.mip_layout[0].size == requested.data.size &&
	    cached.info.mip_layout[0].size == cached.info.data.size &&
	    requested.data.size % requested.resources.layers == 0 &&
	    cached.info.data.size % cached.info.resources.layers == 0 &&
	    requested.data.size / requested.resources.layers ==
	        cached.info.data.size / cached.info.resources.layers &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    (cached.info.metadata.kind == ImageMetadataKind::None ||
	     cached.info.metadata.kind == ImageMetadataKind::Htile);
	if (!DepthOverlapNeedsRecreate(requested, binding, cached)) {
		return cached_id;
	}
	RefreshImage(cached_id);
	auto info = requested;
	if (retain_cached_layout) {
		info.data       = cached.info.data;
		info.resources  = cached.info.resources;
		info.mip_layout          = cached.info.mip_layout;
		info.tiling              = cached.info.tiling;
		info.tiled_slice_stride  = cached.info.tiled_slice_stride;
		info.linear_slice_stride = cached.info.linear_slice_stride;
		info.first_tail_level    = cached.info.first_tail_level;
	}
	info.htile_initial_clear  = false;
	const auto replacement_id = InsertImage(info);
	auto&      replacement    = m_slot_images[replacement_id];
	replacement.usage         = cached.usage;
	if (cached.binding.is_bound || cached.binding.is_target) {
		cached.binding.needs_rebind = true;
	}
	if (cached.backing.samples == replacement.backing.samples) {
		const bool copy_supported =
		    cached.backing.samples == 1 || cached.backing.format == replacement.backing.format ||
		    (!cached.info.IsDepth() && !replacement.info.IsDepth() &&
		     ImageViewOps::FormatsCompatible(cached.backing.format, replacement.backing.format));
		if (copy_supported) {
			CopyImage(replacement_id, cached_id);
		} else {
			LOGF_COLOR(Log::Color::BrightYellow,
			           "TextureCache: unsupported cross-format multisample depth copy\n");
		}
	} else if (cached.backing.samples == 1 && replacement.backing.samples > 1 &&
	           replacement.info.IsDepth()) {
		RefreshCopySource(cached_id);
		if (cached.IsBufferModified() || cached.IsDefinitelyCpuDirty()) {
			EXIT("TextureCache: multisample depth conversion source is not native-current\n");
		}
		PrepareImageCopy(replacement);
		m_blit_helper.ReinterpretColorAsMsDepth(cached, replacement);
		CommitGpuWrite(replacement);
	} else {
		LOGF_COLOR(Log::Color::BrightYellow,
		           "TextureCache: unsupported unequal-sample depth overlap copy (%u -> %u)\n",
		           cached.backing.samples, replacement.backing.samples);
	}
	FreeImage(cached_id);
	return replacement_id;
}

namespace BcAlias {

bool Matches(const ImageInfo& requested, const ImageInfo& cached) {
	return requested.guest_format != cached.guest_format &&
	       requested.IsBlock() != cached.IsBlock() &&
	       requested.tile_mode == cached.tile_mode &&
	       requested.bytes_per_block == cached.bytes_per_block &&
	       requested.samples == 1 && cached.samples == 1 &&
	       (requested.data != cached.data || requested.resources != cached.resources ||
	        requested.BlockExtent() != cached.BlockExtent()) &&
	       !TextureCache::SameBacking(cached, requested, false);
}

bool Publish(TextureCache& cache, Image& image) {
	if (!cache.CanDownload(image)) {
		return false;
	}
	// ObtainBuffer preserves padding and records ownership before the GPU-only transfer.
	auto [buffer, offset] =
	    cache.m_buffer_cache.ObtainBuffer(image.info.data.address, image.info.data.size, true);
	cache.DownloadImage(image, *buffer, offset, image.info.data.size, image.info.resources.levels);
	image.ClearGpuModified();
	image.MarkBufferModified();
	return true;
}

void Resolve(TextureCache& cache, ImageId cached_id, ImageId merged_id) {
	auto& cached = cache.m_slot_images[cached_id];
	auto* merged = merged_id ? &cache.m_slot_images[merged_id] : nullptr;
	// PS5 raw block images can span several packed BC mips. Transfer their complete
	// guest bytes, including tail padding, before retiring the incompatible backing.
	if (cached.SafeToDownload()) {
		if (merged && merged->SafeToDownload() && !Publish(cache, *merged)) {
			return;
		}
		if (!Publish(cache, cached)) {
			return;
		}
		if (merged) {
			merged->MarkBufferModified();
		}
	}
	cached.binding.needs_rebind |= cached.binding.is_bound || cached.binding.is_target;
	cache.FreeImage(cached_id);
}

} // namespace BcAlias

TextureCache::OverlapResult TextureCache::ResolveOverlap(const ImageInfo& requested,
                                                         BindingType binding, ImageId cached_id,
                                                         ImageId merged_id, bool speculative) {
	auto owner = m_slot_images.try_get(cached_id);
	if (owner == nullptr || !owner->registered) {
		return {merged_id};
	}
	// PS5: image heaps keep BC images and raw block views of their layers resident together, so a
	// raw read of a layer subset with the same levels views the BC image instead of evicting it.
	const bool raw_view = binding == BindingType::Texture && !requested.IsBlock() &&
	                      requested.type == cached.info.type &&
	                      requested.data.address == cached.info.data.address &&
	                      requested.BlockExtent() == cached.info.BlockExtent() &&
	                      requested.resources.levels == cached.info.resources.levels &&
	                      requested.resources.layers < cached.info.resources.layers;
	if (!raw_view && BcAlias::Matches(requested, cached.info) && !cached.IsCpuDirty()) {
		BcAlias::Resolve(*this, cached_id, merged_id);
		return {merged_id};
	}
	auto&      cached       = *owner;
	const auto current_tick  = m_scheduler.CurrentTick();
	const auto current_frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
	const bool safe_to_delete =
	    cached.table_pins == 0 &&
	    current_tick - std::min(current_tick, cached.tick_accessed_last) > NumFramesBeforeRemoval &&
	    (!AliasAgeByFrames() || current_frame - std::min(current_frame, cached.frame_accessed_last) >
	                                AliasFramesBeforeRemoval);

	const uint32_t requested_block = requested.bytes_per_block * requested.samples;
	const uint32_t cached_block    = cached.info.bytes_per_block * cached.info.samples;
	// KYTY_LOG_IMAGE_OVERLAPS=1: each overlap of differently laid-out images, once per pair.
	if (static const bool log_overlaps = std::getenv("KYTY_LOG_IMAGE_OVERLAPS") != nullptr;
	    log_overlaps && (requested.tile_mode != cached.info.tile_mode ||
	                     requested.pixel_format != cached.info.pixel_format ||
	                     requested.extent != cached.info.extent ||
	                     requested.data.address != cached.info.data.address)) {
		static std::unordered_set<uint64_t> seen;
		const uint64_t key = requested.data.address * 31u ^ cached.info.data.address * 17u ^
		                     static_cast<uint64_t>(requested.tile_mode) << 56u ^
		                     static_cast<uint64_t>(cached.info.tile_mode) << 48u ^
		                     static_cast<uint64_t>(requested.pixel_format) << 32u ^
		                     static_cast<uint64_t>(cached.info.pixel_format) << 40u;
		if (seen.size() < 400 && seen.insert(key).second) {
			std::fprintf(
			    stderr,
			    "[img-overlap] req 0x%llx+0x%llx %ux%ux%u fmt=%d tile=%u lv=%u ly=%u bind=%u | "
			    "cached 0x%llx+0x%llx %ux%ux%u fmt=%d tile=%u lv=%u ly=%u target=%d bound=%d "
			    "gpu_written=%d age_ticks=%llu safe=%d\n",
			    static_cast<unsigned long long>(requested.data.address),
			    static_cast<unsigned long long>(requested.data.size), requested.extent.width,
			    requested.extent.height, requested.extent.depth,
			    static_cast<int>(requested.pixel_format), static_cast<uint32_t>(requested.tile_mode),
			    requested.resources.levels, requested.resources.layers,
			    static_cast<uint32_t>(binding),
			    static_cast<unsigned long long>(cached.info.data.address),
			    static_cast<unsigned long long>(cached.info.data.size), cached.info.extent.width,
			    cached.info.extent.height, cached.info.extent.depth,
			    static_cast<int>(cached.info.pixel_format),
			    static_cast<uint32_t>(cached.info.tile_mode), cached.info.resources.levels,
			    cached.info.resources.layers, cached.binding.is_target ? 1 : 0,
			    cached.binding.is_bound ? 1 : 0, cached.IsGpuModified() ? 1 : 0,
			    static_cast<unsigned long long>(current_tick -
			                                    std::min(current_tick, cached.tick_accessed_last)),
			    safe_to_delete ? 1 : 0);
		}
	}
	if (requested.data.address == cached.info.data.address &&
	    requested.BlockExtent() == cached.info.BlockExtent() && requested_block == cached_block) {
		if (speculative && (cached.info.IsDepth() || requested.IsDepth()) &&
		    DepthOverlapNeedsRecreate(requested, binding, cached)) {
			return {.rejected = true};
		}
		if (const auto depth_id = ResolveDepthOverlap(requested, binding, cached_id)) {
			return {depth_id};
		}
		// Equal pitch does not imply equal mip placement: a changed extent can move
		// a level into or out of the mip tail. These are separate guest layouts.
		if (requested.tile_mode != cached.info.tile_mode ||
		    (requested.resources == cached.info.resources &&
		     !std::equal(requested.mip_layout.begin(), requested.mip_layout.end(),
		                 cached.info.mip_layout.begin(),
		                 [](const ImageMipInfo& left, const ImageMipInfo& right) {
			                 return left.offset == right.offset && left.size == right.size &&
			                        left.pitch == right.pitch && left.height == right.height &&
			                        left.tail_x == right.tail_x && left.tail_y == right.tail_y &&
			                        left.surface_z == right.surface_z;
		                 }))) {
			if (safe_to_delete) {
				FreeImage(cached_id);
			}
			return {merged_id};
		}
		if (requested.GetColorTransform() != cached.info.GetColorTransform()) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.IsBlock() && !cached.info.IsBlock()) {
			return {ExpandImage(requested, cached_id)};
		}
		// Volume depth is not an array-layer count. A larger depth can retain the
		// same block-slice layout while requiring a larger native image.
		if ((requested.IsVolume() || cached.info.IsVolume()) &&
		    (requested.data.size == cached.info.data.size ||
		     (requested.type == cached.info.type && requested.resources == cached.info.resources &&
		      requested.extent.width == cached.info.extent.width &&
		      requested.extent.height == cached.info.extent.height &&
		      requested.extent.depth > cached.info.extent.depth))) {
			return {ExpandImage(requested, cached_id)};
		}
		// PPSA08394
		// A view cannot change the native image type or grow its extent.
		if (requested.data.size == cached.info.data.size &&
		    requested.resources == cached.info.resources &&
		    ImageViewOps::FormatsCompatible(cached.info.pixel_format, requested.pixel_format) &&
		    (requested.type != cached.info.type
		         ? requested.extent == cached.info.extent
		         : requested.extent.width > cached.info.extent.width &&
		               requested.extent.height >= cached.info.extent.height &&
		               requested.extent.depth >= cached.info.extent.depth)) {
			return {ExpandImage(requested, cached_id)};
		}
		// PS5 mip tails can expose more levels without increasing the guest allocation.
		if (requested.pixel_format == cached.info.pixel_format &&
		    requested.type == cached.info.type && requested.resources > cached.info.resources &&
		    (requested.data.size > cached.info.data.size ||
		     (requested.data.size == cached.info.data.size &&
		      requested.extent == cached.info.extent &&
		      cached.info.resources.levels > 1 &&
		      requested.resources.layers == cached.info.resources.layers))) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.type != cached.info.type) {
			return {merged_id};
		}
		if (requested.pixel_format != cached.info.pixel_format ||
		    requested.data.size <= cached.info.data.size) {
			const auto result_id = merged_id ? merged_id : cached_id;
			return {ImageViewOps::FormatsCompatible(m_slot_images[result_id].info.pixel_format,
			                                        requested.pixel_format)
			            ? result_id
			            : ImageId {}};
		}
		EXIT("TextureCache: unresolvable equal-address image overlap, address=0x%016" PRIx64
		     " requested=%ux%u "
		     "cached=%ux%u requested_size=0x%016" PRIx64 " cached_size=0x%016" PRIx64
		     " type=%u/%u tile=%u/%u\n",
		     requested.data.address, requested.resources.levels, requested.resources.layers,
		     cached.info.resources.levels, cached.info.resources.layers, requested.data.size,
		     cached.info.data.size, static_cast<uint32_t>(requested.type),
		     static_cast<uint32_t>(cached.info.type), static_cast<uint32_t>(requested.tile_mode),
		     static_cast<uint32_t>(cached.info.tile_mode));
	}

	const int32_t requested_mip = requested.MipOf(cached.info);
	if (requested_mip >= 0) {
		const int32_t layer = requested.SliceOf(cached.info, requested_mip);
		return {cached_id, requested_mip, layer};
	}

	const int32_t mip = cached.info.MipOf(requested);
	if (mip >= 0) {
		const int32_t layer = cached.info.SliceOf(requested, mip);
		if (!merged_id) {
			return {ExpandImage(requested, cached_id)};
		}
		cached.binding.needs_rebind |= cached.binding.is_bound || cached.binding.is_target;
		if (cached.binding.is_target && !m_slot_images[merged_id].binding.is_target) {
			NoteBindlessStateChange(m_slot_images[merged_id]);
		}
		m_slot_images[merged_id].binding.is_target |= cached.binding.is_target;
		CopyImageMip(merged_id, cached_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
		FreeImage(cached_id);
		return {merged_id};
	}
	if (requested.data.address >= cached.info.data.address && safe_to_delete) {
		FreeImage(cached_id);
	}
	return {merged_id};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId source_id) {
	RefreshCopySource(source_id);
	const auto expanded_id = InsertImage(info);
	auto&      source      = m_slot_images[source_id];
	if (source.binding.is_bound || source.binding.is_target) {
		source.binding.needs_rebind = true;
	}
	InitializeImage(expanded_id);
	const int32_t mip = source.info.MipOf(info);
	const int32_t layer = source.info.SliceOf(info, mip);
	if (layer >= 0) {
		CopyImageMip(expanded_id, source_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
	} else {
		CopyImage(expanded_id, source_id);
	}
	FreeImage(source_id);
	return expanded_id;
}

bool TextureCache::CanDownload(const Image& image) const {
	const auto& info = image.info;
	if (info.samples != 1 || image.backing.samples != 1 || info.resources.layers == 0 ||
	    info.data.size % info.resources.layers != 0) {
		return false;
	}
	if (UploadBinding(image) == BindingType::DepthTarget) {
		return IsSupportedDepthPlaneReadback(info) &&
		       Prospero::NumBytesPerElement(info.guest_format) == info.bytes_per_block;
	}
	return info.metadata.compression == VideoOutCompression::Uncompressed &&
	       (!info.IsTiled() || !Prospero::IsFmaskTextureFormat(info.guest_format)) &&
	       info.LinearSize() != 0;
}

void TextureCache::UploadImage(Image& image, Buffer& source, uint64_t source_offset) {
	auto& destination = image.depth_id ? m_slot_images[image.depth_id] : image;
	const auto& info        = image.info;
	const auto  binding     = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
	if (binding != BindingType::DepthTarget) {
		if ((binding == BindingType::RenderTarget || binding == BindingType::VideoOut) &&
		    (info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
		     info.samples != 1 || image.backing.samples != 1)) {
			EXIT("TextureCache: invalid color-attachment upload\n");
		}
		if (binding == BindingType::VideoOut &&
		    info.metadata.compression != VideoOutCompression::Uncompressed) {
			EXIT("TextureCache: invalid video-out upload\n");
		}
		const auto linear = m_tiler.Detile(source.Handle(), source_offset, info);
		const auto copies = info.BufferCopies(linear.offset);
		destination.Upload(copies, linear.buffer, linear.offset, linear.size);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureUploadBytes, linear.size);
		return;
	}

	if (info.samples != 1 || destination.backing.samples != 1 ||
	    info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
	    Prospero::NumBytesPerElement(info.guest_format) != info.bytes_per_block) {
		EXIT("TextureCache: invalid depth upload\n");
	}
	const auto layers          = info.resources.layers;
	const auto full_slice_size = info.linear_slice_stride;
	auto       linear          = m_tiler.Detile(source.Handle(), source_offset, info);
	auto       copies          = info.BufferCopies(linear.offset);
	if (image.depth_id) {
		for (auto& copy: copies) {
			copy.imageSubresource.baseArrayLayer += image.stencil_subresources.base_layer;
		}
	}
	const auto transfer_bytes = image.depth_id ? 1u : DepthAspectTransferBytes(info.pixel_format);
	if (transfer_bytes != info.bytes_per_block) {
		const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
		EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
		                     transfer_bytes != sizeof(uint32_t) || texels_per_slice > UINT32_MAX ||
		                     texels_per_slice > UINT64_MAX / transfer_bytes);
		const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
		EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
		auto promoted = m_tiler.GetScratchBuffer(transfer_slice * layers, linear.buffer);
		m_tiler.ConvertD16(
		    linear, promoted, TileManager::D16Direction::Promote,
		    info.pixel_format == vk::Format::eD32SfloatS8Uint,
		    {.width               = info.extent.width,
		     .height              = info.extent.height,
		     .layers              = layers,
		     .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
		     .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
		     .source_slice_stride = full_slice_size,
		     .target_slice_stride = transfer_slice});
		linear = promoted;
		copies.resize(1);
		copies[0].bufferOffset      = linear.offset;
		copies[0].bufferRowLength   = info.pitch;
		copies[0].bufferImageHeight = info.extent.height;
		copies[0].imageSubresource  = {vk::ImageAspectFlagBits::eDepth, 0, 0, layers};
		copies[0].imageExtent       = info.extent;
	}
	destination.Upload(copies, linear.buffer, linear.offset, linear.size);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureUploadBytes, linear.size);
}

void TextureCache::InitializeImage(ImageId id) {
	auto& image = m_slot_images[id];
	TrackImage(id);
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (image.IsCpuDirty()) {
			image.RefreshComplete();
		}
		return;
	}
	if (image.info.samples > 1) {
		return;
	}
	const bool upload = image.IsBufferModified() || image.IsCpuDirty();
	if (upload) {
		const auto [source, source_offset] =
		    m_buffer_cache.ObtainBufferForImage(image.info.data.address, image.info.data.size);
		if (source == nullptr) {
			EXIT("TextureCache: failed to obtain image upload source\n");
		}
		UploadImage(image, *source, source_offset);
		image.ClearBufferModified();
	}
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
}

void TextureCache::MaterializeColorClear(ImageId id, const ImageDesc& desc,
                                       uint32_t metadata_base_layer) {
	if (desc.info.metadata.kind != ImageMetadataKind::Dcc &&
	    desc.info.metadata.kind != ImageMetadataKind::Cmask) {
		return;
	}
	const auto range        = desc.info.metadata.range;
	const auto current_tick = m_scheduler.CurrentTick();
	const auto layers       = desc.info.TransferLayers();
	const auto& view           = desc.view_info;
	const bool  volume_texture = desc.info.IsVolume() && view.type == vk::ImageViewType::e3D;
	const auto  first          = volume_texture ? 0u : metadata_base_layer;
	const auto  image_first    = volume_texture ? 0u : view.base_layer;
	const auto  count          = volume_texture ? desc.info.extent.depth : view.layer_count;
	{
		std::scoped_lock lock {m_lock};
		auto& image         = m_slot_images[id];
		image.info.metadata = desc.info.metadata;
		// Native color metadata must not retain a reused HTile clear flag.
		m_surface_metas.erase(range.address);
		if (range.size == 0 || desc.info.resources.levels != 1) {
			return;
		}
		// A fast-clear stamps the whole DCC metadata range with one uniform code; an ordinary
		// draw also touches these bytes (per-block compression state) but leaves a mixed
		// pattern, so it never passes the all-same-code check below -- yet it re-marks the range
		// GPU-dirty just the same, so gating purely on that flag re-triggers the synchronous
		// readback on every draw to an already-bound render target. Gate on the scheduler tick
		// instead: within one still-unsubmitted recording no GPU write can land in guest memory
		// after the first check, so a later FindImage for the same slices of the same image
		// this tick is redundant. Other slices of the image (another layer or face) keep their
		// own check, a CPU store to the metadata (a guest-side clear) is caught through the
		// tracker below, and a real re-clear in a later recording still gets caught, since the
		// tick will have advanced.
		const bool already_resolved_this_tick =
		    image.dcc_clear_checked_tick == current_tick &&
		    image.dcc_clear_checked_first == first && image.dcc_clear_checked_count == count;
		image.dcc_clear_checked_tick  = current_tick;
		image.dcc_clear_checked_first = first;
		image.dcc_clear_checked_count = count;
		if (already_resolved_this_tick &&
		    !m_buffer_cache.IsRegionCpuModified(range.address, range.size)) {
			return;
		}
	}
	// These one-mip surfaces use complete 4 KiB color metadata blocks.
	constexpr uint64_t MetadataBlockSize = 0x1000;
	if (!range.Valid() || range.address % MetadataBlockSize != 0 || layers == 0 ||
	    range.size % layers != 0 || (range.size / layers) % MetadataBlockSize != 0) {
		EXIT("TextureCache: color metadata slices must contain aligned 4 KiB blocks\n");
	}
	if (first >= layers || count > layers - first) {
		EXIT("TextureCache: color view exceeds its native metadata slices\n");
	}
	const auto       slice_size = range.size / layers;
	const GuestRange viewed {range.address + slice_size * first, slice_size * count};
	// A metadata range last written by a recorded uniform compute fill is known without a GPU
	// drain or a conditional clear.
	uint32_t   known_value = 0;
	const bool known_fill  = m_buffer_cache.KnownFill(range.address, range.size, known_value);
	if (!known_fill && m_buffer_cache.IsRegionGpuModified(viewed.address, viewed.size)) {
		std::array<vk::ClearAttachment, TileManager::ColorClearSlots> clears {};
		uint32_t keys = 0;
		for (uint32_t key = 0; key < clears.size(); key++) {
			clears[key].aspectMask = vk::ImageAspectFlagBits::eColor;
			if (DecodeColorClear(desc, static_cast<uint8_t>(key << 4u),
			                     clears[key].clearValue.color)) {
				keys |= 1u << key;
			}
		}
		// No selected key to clear, or none left since the last check.
		if (m_buffer_cache.IsColorMetadataChecked(viewed.address, viewed.size, keys)) {
			return;
		}
		{
			std::scoped_lock lock {m_lock};
			auto&            image = m_slot_images[id];
			// Compressed images cannot upload, so they must already hold native contents.
			const bool native_current = (image.usage.render_target || image.IsGpuModified()) &&
			                            !image.IsBufferModified() && !image.IsCpuDirty();
			// 2D-array attachment views select 2D layers or, of a volume, depth slices.
			if (m_graphics.conditional_rendering_enabled &&
			    image.backing.image_type != vk::ImageType::e1D &&
			    count <= m_graphics.GetPhysicalDeviceProperties().limits.maxFramebufferLayers &&
			    (image.backing.usage & vk::ImageUsageFlagBits::eColorAttachment) &&
			    (m_graphics.GetFormatProperties(view.format).optimalTilingFeatures &
			     vk::FormatFeatureFlagBits::eColorAttachment) &&
			    (desc.info.metadata.compression == VideoOutCompression::Uncompressed ||
			     native_current)) {
				// The GPU selects each slice's clear, so the CPU never waits for the metadata.
				Common::DebugCounters::Add(Common::DebugCounters::Counter::ColorMetadataChecks);
				InitializeImage(id);
				const bool expand = desc.type != BindingType::VideoOut;
				const auto [buffer, offset] =
				    m_buffer_cache.ObtainBuffer(viewed.address, viewed.size, expand);
				const auto predicates = m_tiler.CheckColorClear(
				    {buffer->Handle(), offset, viewed.size}, count, keys, expand);
				if (expand) {
					m_buffer_cache.MarkColorMetadataChecked(viewed.address, viewed.size, keys);
				}
				const auto command = m_scheduler.Current().Recorder();
				const auto area    = BeginColorRendering(image, view.format, view.base_level,
				                                         image_first, count, nullptr, command);
				vk::ClearRect rect {.rect = area, .layerCount = 1};
				for (uint32_t slice = 0; slice < count; slice++) {
					rect.baseArrayLayer = slice;
					for (uint32_t key = 0; key < clears.size(); key++) {
						if ((keys & (1u << key)) != 0) {
							const vk::ConditionalRenderingBeginInfoEXT predicate {
							    .buffer = predicates,
							    .offset = (slice * clears.size() + key) * sizeof(uint32_t)};
							command.beginConditionalRenderingEXT(&predicate);
							command.clearAttachment(clears[key], rect);
							command.endConditionalRenderingEXT();
						}
					}
				}
				command.endRendering();
				CommitGpuWrite(image);
				return;
			}
		}
		// Finish native metadata writes before reading backing bytes. This can submit the
		// scheduler, so discovery runs before final draw uploads and never holds the texture
		// lock across it.
		Common::DebugCounters::ScopedTimer timer(
		    Common::DebugCounters::Counter::ColorMetadataReadbacks,
		    Common::DebugCounters::Counter::ColorMetadataReadbackNs);
		m_buffer_cache.ReadMemory(viewed.address, viewed.size, false);
	}
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto address = range.address + slice_size * (first + slice);
		uint8_t code = 0;
		if (known_fill) {
			code = static_cast<uint8_t>(known_value);
		} else if (!LibKernel::Memory::TryReadBacking(address, &code, sizeof(code))) {
			EXIT("TextureCache: failed to read color metadata backing\n");
		}
		vk::ClearValue clear {};
		if (!DecodeColorClear(desc, code, clear.color)) {
			continue;
		}
		if (known_fill) {
			if (known_value != 0x01010101u * code) {
				continue;
			}
		} else {
			std::vector<uint8_t> bytes(slice_size);
			if (!LibKernel::Memory::TryReadBacking(address, bytes.data(), bytes.size())) {
				EXIT("TextureCache: failed to read color metadata slice\n");
			}
			if (!std::all_of(bytes.begin(), bytes.end(),
			                 [code](uint8_t byte) { return byte == code; })) {
				continue;
			}
		}
		{
			std::scoped_lock lock {m_lock};
			const auto& image = m_slot_images[id];
			const bool depth = image.info.IsDepth();
			if (depth) {
				const auto value = clear.color.float32[0];
				clear.depthStencil = {value, 0u};
			}
			ClearImage(m_scheduler.Current(), id, depth ? image.backing.format : view.format,
			           {depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
			            view.base_level, view.level_count,
			            image_first + slice, 1}, clear);
		}
		// Publish the conversion's expanded keys without treating them as guest writes
		// to overlapping image data, and without reading GPU-owned metadata back first.
		if (desc.type != BindingType::VideoOut) {
			m_buffer_cache.FillInternalMemory(address, slice_size, UINT32_MAX);
		}
	}
}

void TextureCache::RefreshImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.depth_id &&
	    (m_slot_images[image.depth_id].info.metadata.stencil_compressed ||
	     m_slot_images[image.depth_id].info.samples != 1)) {
		return;
	}
	TrackImage(id);
	if (image.IsMaybeCpuDirty()) {
		(void)image.ResolveMaybeCpuHash(image.HashGuestEdges());
	}
	bool cpu_dirty = image.IsBufferModified() || image.IsDefinitelyCpuDirty();
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (cpu_dirty) {
			EXIT("TextureCache: compressed guest image refresh is unsupported\n");
		}
		return;
	}
	if (!cpu_dirty) {
		return;
	}
	InitializeImage(id);
}

ImageId TextureCache::AssociateStencil(ImageId depth_id, GuestRange stencil) {
	if (!stencil.Valid()) {
		EXIT("TextureCache: invalid stencil association range\n");
	}
	auto& depth = m_slot_images[depth_id];
	if (!depth.info.IsDepth() || !depth.info.HasStencil()) {
		EXIT("TextureCache: stencil association requires a depth/stencil image\n");
	}

	ImageId association {};
	for (const auto id: FindImagesInRegion(stencil.address, stencil.size, false)) {
		const auto& owner = m_slot_images[id];
		if (owner.info.data == stencil && owner.info.extent == depth.info.extent) {
			association = id;
		}
	}
	if (!association) {
		ImageInfo info {};
		info.data   = stencil;
		info.extent = depth.info.extent;
		association = InsertImage(info);
	}
	auto& record = m_slot_images[association];
	TouchImage(record);
	if (record.depth_id != depth_id || record.stencil_subresources != depth.stencil_subresources) {
		ImageInfo plane {};
		plane.data         = stencil;
		plane.pixel_format = vk::Format::eS8Uint;
		plane.guest_format = Prospero::BufferFormat::k8UInt;
		plane.type         = depth.info.type;
		plane.extent       = depth.info.extent;
		plane.resources    = {depth.stencil_subresources.level_count,
		                      depth.stencil_subresources.layer_count};
		plane.tile_mode    = depth.info.tile_mode;
		plane.pitch =
		    plane.IsTiled() ? TileGetDepthPitch(plane.extent.width, 1, 0) : depth.info.pitch;
		plane.bytes_per_block = 1;
		plane.UpdateSize();
		if (record.depth_id != depth_id) {
			if (record.depth_id) {
				std::erase(m_slot_images[record.depth_id].stencil_ids, association);
			}
			depth.stencil_ids.push_back(association);
		}
		record.info                 = plane;
		record.depth_id             = depth_id;
		record.stencil_subresources = depth.stencil_subresources;
	}

	return association;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format) {
	return FindImageImpl(desc, exact_format, false);
}

ImageId TextureCache::FindImageSpeculative(ImageDesc& desc, bool exact_format) {
	return FindImageImpl(desc, exact_format, true);
}

ImageId TextureCache::FindImageImpl(ImageDesc& desc, bool exact_format, bool speculative) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
	// The image description of a remembered lookup was validated when it was remembered.
	ValidateImageViewDesc(desc);
	EXIT_IF(desc.info.data.Empty());
	const auto metadata_base_layer = desc.view_info.base_layer;

	ImageId result {};
	{
		std::scoped_lock lock {m_lock};
		// Draws look up the same targets and textures over and over: remember lookups that found
		// an image with the same backing (no overlap resolution, no insertion) until the image set
		// changes.
		auto& exact = (*m_exact_lookups)[ExactLookupSlot(desc.info)];
		if (exact.valid && exact.exact_format == exact_format &&
		    desc.type != BindingType::VideoOut &&
		    std::memcmp(&exact.info, &desc.info, sizeof(ImageInfo)) == 0 &&
		    (exact.epoch == m_image_set_epoch ||
		     ImageEpochInRegionUnlocked(desc.info.data.address, desc.info.data.size) <=
		         exact.epoch)) {
			// No image of the range changed until now: the next lookup compares with now instead of
			// scanning the pages of the range again.
			exact.epoch              = m_image_set_epoch;
			auto& image              = m_slot_images[exact.id];
			image.tick_accessed_last = m_scheduler.CurrentTick();
			TouchImage(image);
			result = exact.id;
		}
	}
	if (result) {
		MaterializeColorClear(result, desc, metadata_base_layer);
		return result;
	}
	ImageOps::Validate(desc.info);
	{
		std::scoped_lock lock {m_lock};
		const auto       candidates =
		    FindImagesInRegion(desc.info.data.address, desc.info.data.size, false);
		bool block_alias = false;

		for (const auto id: candidates) {
			const auto& image = m_slot_images[id];
			// Stencil descriptors name the associated plane; its depth owner supplies
			// the native image and the canonical guest tiling.
			if ((desc.type == BindingType::Texture || desc.type == BindingType::Storage) &&
			    image.depth_id && image.info.data == desc.info.data &&
			    m_slot_images[image.depth_id].info.samples == desc.info.samples &&
			    image.info.extent == desc.info.extent && image.info.type == desc.info.type &&
			    image.info.resources.levels >= desc.info.resources.levels &&
			    image.info.resources.layers >= desc.info.resources.layers &&
			    desc.info.bytes_per_block == 1 &&
			    desc.info.guest_format == Prospero::BufferFormat::k8UInt) {
				result = id;
				break;
			}
			if (SameBacking(image.info, desc.info, exact_format)) {
				result = id;
			} else if (image.SafeToDownload() &&
			           BcAlias::Matches(desc.info, image.info)) {
				block_alias = true;
			}
		}
		const bool same_backing = static_cast<bool>(result);

		int32_t view_mip   = -1;
		int32_t view_layer = -1;
		if (!result) {
			for (const auto candidate: candidates) {
				view_mip                = -1;
				view_layer              = -1;
				const auto& merged_info = result ? m_slot_images[result].info : desc.info;
				const auto  overlap =
				    ResolveOverlap(merged_info, desc.type, candidate, result, speculative);
				if (overlap.rejected) {
					return {};
				}
				if (overlap.image) {
					result     = overlap.image;
					view_mip   = overlap.mip;
					view_layer = overlap.layer;
				}
			}
			// PS5: image heaps hold a descriptor per dynamic-resolution extent of a render target,
			// and only one extent at an address holds data, so heap lookups create no second one.
			if (!result && desc.heap &&
			    std::any_of(candidates.begin(), candidates.end(), [&](ImageId id) {
				    const auto* image = m_slot_images.try_get(id);
				    return image != nullptr && image->registered &&
					       image->heap_refs.load(std::memory_order_relaxed) != 0u &&
					       image->info.data.address == desc.info.data.address &&
					       image->info.type == desc.info.type &&
					       image->info.extent != desc.info.extent;
			    })) {
				return {};
			}
		} else if (block_alias) {
			for (const auto candidate: candidates) {
				const auto& owner = m_slot_images[candidate];
				if (candidate != result && owner.registered && !owner.IsCpuDirty() &&
				    BcAlias::Matches(m_slot_images[result].info, owner.info)) {
					BcAlias::Resolve(*this, candidate, result);
				}
			}
		}

		if (result && !m_slot_images[result].depth_id) {
			auto& resolved = m_slot_images[result];
			if (exact_format && resolved.info.pixel_format != desc.info.pixel_format) {
				result = {};
			} else if (resolved.info.resources < desc.info.resources) {
				FreeImage(result);
				result = {};
			}
		}
		if (result && same_backing && desc.type != BindingType::VideoOut) {
			auto& exact = (*m_exact_lookups)[ExactLookupSlot(desc.info)];
			std::memcpy(&exact.info, &desc.info, sizeof(ImageInfo));
			exact.id           = result;
			exact.epoch        = m_image_set_epoch;
			exact.exact_format = exact_format;
			exact.valid        = true;
		}
		if (!result) {
			result         = InsertImage(desc.info);
			auto& inserted = m_slot_images[result];
			if (m_buffer_cache.HasGpuDirtyBytes(inserted.info.data.address,
			                                    inserted.info.data.size)) {
				inserted.MarkBufferModified();
			}
		}
		auto& image = m_slot_images[result];
		if (view_mip >= 0) {
			desc.view_info.base_level = static_cast<uint32_t>(view_mip);
		}
		if (view_layer >= 0) {
			desc.view_info.base_layer = static_cast<uint32_t>(view_layer);
		}
		image.tick_accessed_last = m_scheduler.CurrentTick();
		TouchImage(image);
	}
	MaterializeColorClear(result, desc, metadata_base_layer);
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression != VideoOutCompression::Uncompressed) {
		std::scoped_lock lock {m_lock};
		const auto& image = m_slot_images[result];
		const bool guest_dirty = image.IsBufferModified() || image.IsCpuDirty();
		const bool native_current =
		    (image.usage.render_target || image.IsGpuModified()) && !guest_dirty;
		if (!native_current) {
			EXIT("TextureCache: compressed video-out read requires clean native GPU "
			     "contents\n");
		}
	}
	return result;
}

void TextureCache::UpdateImage(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	RefreshImage(id);
}

ImageId TextureCache::FindImageFromRange(uint64_t address, uint64_t size, bool ensure_valid) {
	if (!GuestRange {address, size}.Valid()) {
		return {};
	}
	std::scoped_lock lock {m_lock};
	ImageIds         matches;
	// Only an image that starts at the address qualifies, and it covers that byte: one page to
	// query instead of every page of the range (a multi-megabyte texel buffer on every draw).
	for (const auto id: FindImagesInRegion(address, 1, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr || owner->info.data.address != address) {
			continue;
		}
		if (ensure_valid && owner->depth_id) {
			owner = &m_slot_images[owner->depth_id];
		}
		if (ensure_valid && !owner->SafeToDownload()) {
			continue;
		}
		matches.push_back(id);
	}
	ImageId selected {};
	if (matches.size() == 1) {
		selected = matches.front();
	} else {
		for (const auto id: matches) {
			const auto& image = m_slot_images[id];
			if (image.info.data.size == size) {
				selected = id;
				break;
			}
		}
	}
	if (selected && ensure_valid && m_slot_images[selected].depth_id) {
		selected = m_slot_images[selected].depth_id;
	}
	return selected;
}

void TextureCache::BeginTextureDump(uint64_t frame) {
	std::scoped_lock lock {m_lock};
	if (m_dump_frame == UINT64_MAX) {
		m_dump_frame         = frame + 1;
		m_dump_wait_bindless = true;
	}
}

void TextureCache::AddTextureDumpImages(std::span<const ImageId> ids) {
	std::scoped_lock lock {m_lock};
	for (const auto id: ids) {
		if (std::find(m_dump_ids.begin(), m_dump_ids.end(), id) == m_dump_ids.end()) {
			m_dump_ids.push_back(id);
		}
	}
	m_dump_wait_bindless = false;
}

void TextureCache::DumpTextureNote(ImageId id) {
	if (m_dump_done || m_dump_frame == UINT64_MAX) {
		return;
	}
	const auto frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
	if (frame == m_dump_frame) {
		if (std::find(m_dump_ids.begin(), m_dump_ids.end(), id) == m_dump_ids.end()) {
			m_dump_ids.push_back(id);
		}
		return;
	}
	if (frame < m_dump_frame || m_dump_wait_bindless) {
		return;
	}
	if (m_dumps.empty() && !m_dump_ids.empty()) {
		// Record the copies; they land with the GPU's progress, no wait here.
		uint64_t total = 0;
		for (const auto dump_id: m_dump_ids) {
			auto* image = m_slot_images.try_get(dump_id);
			if (image == nullptr || m_dumps.size() >= 1500 || total >= (1ull << 30u)) {
				continue;
			}
			// Material textures only: block-compressed, 512 or more on a side.
			if (std::max(image->info.extent.width, image->info.extent.height) < 512 ||
			    !image->info.IsBlock()) {
				continue;
			}
			const auto& info = image->info;
			const auto  w = info.extent.width, h = info.extent.height;
			const auto  format = info.pixel_format;
			const bool  depth  = format == vk::Format::eD16Unorm || format == vk::Format::eD32Sfloat ||
			                   format == vk::Format::eD24UnormS8Uint ||
			                   format == vk::Format::eD32SfloatS8Uint ||
			                   format == vk::Format::eD16UnormS8Uint || format == vk::Format::eS8Uint ||
			                   format == vk::Format::eX8D24UnormPack32;
			if (w < 16 || h < 16 || info.extent.depth > 1 || info.bytes_per_block == 0 || depth) {
				continue;
			}
			const uint64_t bytes = info.IsBlock()
			                           ? uint64_t {(w + 3u) / 4u} * ((h + 3u) / 4u) * info.bytes_per_block
			                           : uint64_t {w} * h * info.bytes_per_block;
			if (bytes > (96ull << 20u)) {
				continue;
			}
			TextureDump dump {dump_id,
			                  std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
			                                           vk::BufferUsageFlagBits::eTransferDst, bytes),
			                  bytes};
			vk::BufferImageCopy region {};
			region.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
			region.imageExtent      = vk::Extent3D {w, h, 1};
			image->Download(std::span {&region, 1}, dump.buffer->Handle(), 0, bytes);
			m_dumps.push_back(std::move(dump));
			total += bytes;
		}
		m_dump_tick = m_scheduler.CurrentTick();
		std::fprintf(stderr, "[texdump] %zu textures sampled in frame %llu, %zu copies recorded\n",
		             m_dump_ids.size(), static_cast<unsigned long long>(m_dump_frame),
		             m_dumps.size());
		return;
	}
	if (m_dumps.empty() || !m_scheduler.IsFree(m_dump_tick)) {
		return;
	}
	static uint32_t dump_count = 0;
	const auto      dir = std::filesystem::path("_ShaderDump") / fmt::format("texdump_{}", dump_count++);
	std::error_code error;
	std::filesystem::create_directories(dir, error);
	FILE* index = std::fopen((dir / "index.txt").string().c_str(), "w");
	for (size_t i = 0; i < m_dumps.size(); i++) {
		auto&       dump  = m_dumps[i];
		auto*       image = m_slot_images.try_get(dump.id);
		dump.buffer->Invalidate(0, dump.bytes);
		const auto name = fmt::format("tex_{:04}.bin", i);
		if (FILE* file = std::fopen((dir / name).string().c_str(), "wb"); file != nullptr) {
			std::fwrite(dump.buffer->Mapped().data(), 1, dump.bytes, file);
			std::fclose(file);
		}
		if (index != nullptr && image != nullptr) {
			const auto& info = image->info;
			std::fprintf(index, "%s guest=0x%llx size=0x%llx w=%u h=%u mips=%u layers=%u fmt=%d tile=%u bpb=%u block=%d\n",
			             name.c_str(), static_cast<unsigned long long>(info.data.address),
			             static_cast<unsigned long long>(info.data.size), info.extent.width,
			             info.extent.height, info.resources.levels, info.resources.layers,
			             static_cast<int>(info.pixel_format), static_cast<uint32_t>(info.tile_mode),
			             info.bytes_per_block, info.IsBlock() ? 1 : 0);
		}
	}
	if (index != nullptr) {
		std::fclose(index);
	}
	std::fprintf(stderr, "[texdump] wrote %zu textures to %s\n", m_dumps.size(),
	             dir.string().c_str());
	m_dumps.clear();
	// Ready for the next trigger.
	m_dump_ids.clear();
	m_dump_frame = UINT64_MAX;
}

vk::ImageView TextureCache::FindTexture(ImageId id, const ImageDesc& desc) {
	std::scoped_lock lock {m_lock};
	DumpTextureNote(id);
	auto&            image = m_slot_images[id];
	TouchImage(image);
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: texture requires rediscovery before final acquisition\n");
	}
	if (desc.type == BindingType::Storage) {
		image.MarkGpuModified();
	}
	RefreshImage(id);
	if (image.info.HasStencil() &&
	    desc.info.data.address >= image.info.stencil.address &&
	    desc.info.data.End() <= image.info.stencil.End()) {
		for (const auto stencil_id:
		     FindImagesInRegion(image.info.stencil.address, image.info.stencil.size, false)) {
			const auto& stencil = m_slot_images[stencil_id];
			if (stencil.depth_id == id && stencil.info.data == image.info.stencil) {
				RefreshImage(stencil_id);
				break;
			}
		}
	}
	switch (desc.type) {
		case BindingType::Texture: break;
		case BindingType::Storage:
			CommitGpuWrite(image);
			TrackImageDownload(id, image);
			break;
		default: EXIT("TextureCache: invalid texture binding\n");
	}
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindRenderTarget(ImageId id, const ImageDesc& desc) {
	if (desc.type != BindingType::RenderTarget) {
		EXIT("TextureCache: invalid color-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: color target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	image.MarkGpuModified();
	image.usage.render_target = true;
	RefreshImage(id);
	CommitGpuWrite(image);
	TrackImageDownload(id, image);
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindDepthTarget(ImageId id, const ImageDesc& desc, ImageId* stencil) {
	if (desc.type != BindingType::DepthTarget) {
		EXIT("TextureCache: invalid depth-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: depth target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	image.MarkGpuModified();
	if (desc.info.HasStencil()) {
		const auto layers = image.info.resources.layers;
		EXIT_IF(layers == 0 || image.info.data.size % layers != 0 ||
		        desc.info.data.address < image.info.data.address);
		const auto slice_size = image.info.data.size / layers;
		const auto offset     = desc.info.data.address - image.info.data.address;
		EXIT_IF(slice_size == 0 || offset % slice_size != 0 || offset / slice_size >= layers ||
		        desc.info.resources.layers > layers - offset / slice_size);
		image.stencil_subresources = {0, desc.info.resources.levels,
		                             static_cast<uint32_t>(offset / slice_size),
		                             desc.info.resources.layers};
	}
	image.info.stencil = desc.info.stencil;
	image.info.metadata = desc.info.metadata;
	if (desc.info.HasMetadata()) {
		const auto [metadata, inserted] = m_surface_metas.try_emplace(desc.info.metadata.range.address);
		if (inserted && image.info.htile_initial_clear) {
			metadata->second.clear_mask.set();
		}
		metadata->second.owner = id;
	}
	RefreshImage(id);
	CommitGpuWrite(image);
	if (desc.info.HasStencil()) {
		const auto association = AssociateStencil(id, desc.info.stencil);
		RefreshImage(association);
		if (stencil != nullptr) {
			*stencil = association;
		}
	}
	return image.FindView(desc.view_info);
}

void TextureCache::MarkGpuWritten(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id) {
		EXIT("TextureCache: cannot mark an unavailable image GPU-written\n");
	}
	TrackImage(id);
	CommitGpuWrite(image);
	if (image.info.HasStencil()) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
}

void TextureCache::CommitGpuWrite(Image& image) {
	if (!image.depth_id && image.backing.image == nullptr) {
		EXIT("TextureCache: GPU writes require a native image or stencil association\n");
	}
	image.ClearBufferModified();
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
	// Aliases aged by frames stay alive side by side. A single owner among them: this write
	// supersedes the bytes of every other live image overlapping it, so none of those may later
	// be written back (collector, linear readback, buffer sync) over it. Their native contents
	// stay as they are; the tick rule freed them without a write-back. Scanned only when the
	// owner changes, not on every bind.
	if (AliasAgeByFrames() && !image.alias_owner && !image.depth_id && image.registered &&
	    !image.info.data.Empty()) {
		for (const auto id: FindImagesInRegion(image.info.data.address, image.info.data.size, false)) {
			auto* other = m_slot_images.try_get(id);
			if (other == nullptr || other == &image || other->depth_id || !other->registered) {
				continue;
			}
			other->alias_owner = false;
			if (other->IsGpuModified()) {
				// The new write replaces every byte of an image it covers: nothing of it is left to
				// write back, and dropping it is exact. An image that only partly overlaps
				// (a transient target inside a bigger pool, or a bigger one around it) also loses
				// the pixels outside the new image, which is what the notice reports.
				const auto& old_range = other->info.data;
				const auto& new_range = image.info.data;
				const bool  covered   = new_range.address <= old_range.address &&
				                      old_range.address + old_range.size <=
				                          new_range.address + new_range.size;
				if (!covered) {
					Common::HackNotice::Report(
					    "overlapping image no longer written back: this write supersedes part of it",
					    "TextureCache::CommitGpuWrite", 0, 0,
					    "alias single-owner rule (KYTY_IMAGE_ALIAS_AGE): its GPU-written pixels "
					    "outside the new image are dropped");
				}
				other->ClearGpuModified();
			}
		}
		image.alias_owner = true;
	}
	image.MarkGpuModified();
}

bool TextureCache::ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
                                        uint32_t packed_clear) {
	if (command.IsInvalid() || !GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid image clear\n");
	}
	std::scoped_lock      lock {m_lock};
	ImageId               selected {};
	ImageId               stencil_id {};
	vk::ImageAspectFlags  aspect {};
	ImageSubresourceRange subresources;
	for (const auto id: FindImagesInRegion(address, size, false)) {
		const auto&          owner = m_slot_images[id];
		vk::ImageAspectFlags candidate {};
		ImageId              candidate_id = id;
		ImageSubresourceRange candidate_subresources {
		    0, owner.info.resources.levels, 0, owner.info.TransferLayers()};
		if (owner.depth_id && owner.info.data.address == address &&
		    owner.info.data.size == size) {
			candidate              = vk::ImageAspectFlagBits::eStencil;
			candidate_id           = owner.depth_id;
			candidate_subresources = owner.stencil_subresources;
			const auto& depth      = m_slot_images[candidate_id];
			if (depth.backing.image == nullptr || !depth.info.HasStencil()) {
				continue;
			}
		} else if (!owner.depth_id && owner.info.data.address == address &&
		           owner.info.data.size == size) {
			candidate = owner.info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
			                                 : vk::ImageAspectFlagBits::eColor;
		}
		if (!candidate) {
			continue;
		}
		if (selected && (selected != candidate_id || subresources != candidate_subresources)) {
			return false;
		}
		selected     = candidate_id;
		stencil_id   = candidate == vk::ImageAspectFlagBits::eStencil ? id : ImageId {};
		aspect       = candidate;
		subresources = candidate_subresources;
	}
	if (!selected) {
		return false;
	}
	auto&          image = m_slot_images[selected];
	vk::ClearValue clear {};
	if (aspect == vk::ImageAspectFlagBits::eColor) {
		if (!DecodeColorDwordFill(image.info.pixel_format, packed_clear, clear.color)) {
			return false;
		}
	} else {
		uint8_t stencil_clear = 0;
		if ((aspect == vk::ImageAspectFlagBits::eDepth &&
		     !DecodePackedDepthClear(image.info.pixel_format, packed_clear, clear.depthStencil.depth)) ||
		    (aspect == vk::ImageAspectFlagBits::eStencil &&
		     !DecodePackedStencilClear(packed_clear, stencil_clear))) {
			return false;
		}
		clear.depthStencil.stencil = stencil_clear;
	}
	ClearImage(command, selected, image.backing.format,
	           {aspect, subresources.base_level, subresources.level_count,
	            subresources.base_layer, subresources.layer_count}, clear);
	if (stencil_id) {
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
	return true;
}

void TextureCache::ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
                              const vk::ImageSubresourceRange& range, const vk::ClearValue& clear) {
	auto& image = m_slot_images[id];
	const auto aspects = image.info.IsDepth() ? ImageViewOps::DepthAspectMask(image.backing.format)
	                                          : vk::ImageAspectFlagBits::eColor;
	EXIT_IF(range.baseMipLevel >= image.info.resources.levels);
	const auto layers = image.info.IsVolume()
	                        ? std::max(image.info.extent.depth >> range.baseMipLevel, 1u)
	                        : image.backing.layers;
	EXIT_IF(command.IsInvalid() || image.depth_id || !range.aspectMask || range.levelCount == 0 ||
	        range.levelCount > image.info.resources.levels - range.baseMipLevel ||
	        range.layerCount == 0 || range.baseArrayLayer >= layers ||
	        range.layerCount > layers - range.baseArrayLayer ||
	        (range.aspectMask & aspects) != range.aspectMask);
	const bool full_subresources = range.baseMipLevel == 0 &&
	                               range.levelCount == image.info.resources.levels &&
	                               range.baseArrayLayer == 0 && range.layerCount == layers;
	const bool full_image = range.aspectMask == aspects && full_subresources;
	TrackImage(id);
	if (!full_image && (image.IsBufferModified() || image.IsCpuDirty())) {
		InitializeImage(id);
		if (image.info.samples == 1 && (image.IsBufferModified() || image.IsCpuDirty())) {
			EXIT("TextureCache: image clear retained guest ownership\n");
		}
	}
	if (image.info.HasStencil() && (range.aspectMask & vk::ImageAspectFlagBits::eStencil)) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		if (!full_subresources) {
			RefreshImage(stencil_id);
		} else {
			TrackImage(stencil_id);
			CommitGpuWrite(m_slot_images[stencil_id]);
		}
	}
	command.EndRendering();
	// Transfer clears use the backing format; aliased clears must encode through their view.
	if (format != image.backing.format || (image.info.IsVolume() && !full_image)) {
		EXIT_NOT_IMPLEMENTED(range.aspectMask != vk::ImageAspectFlagBits::eColor ||
		                     range.levelCount != 1);
		BeginColorRendering(image, format, range.baseMipLevel, range.baseArrayLayer,
		                    range.layerCount, &clear, command.Recorder());
		command.Recorder().endRendering();
		CommitGpuWrite(image);
		return;
	}
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
	              command.Recorder());
	auto native_range = range;
	if (image.info.IsVolume()) {
		native_range.baseArrayLayer = 0;
		native_range.layerCount     = 1;
	}
	if (range.aspectMask == vk::ImageAspectFlagBits::eColor) {
		command.Recorder().clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                                 &clear.color, 1, &native_range);
	} else {
		command.Recorder().clearDepthStencilImage(image.backing.image,
		                                        vk::ImageLayout::eTransferDstOptimal,
		                                        &clear.depthStencil, 1, &native_range);
	}
	CommitGpuWrite(image);
}

vk::Rect2D TextureCache::BeginColorRendering(Image& image, vk::Format format, uint32_t level,
                                             uint32_t layer, uint32_t layers,
                                             const vk::ClearValue* clear,
                                             const CommandRecorder& command) {
	ImageViewInfo view {};
	view.format      = format;
	view.type        = layers == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
	view.base_level  = level;
	view.base_layer  = layer;
	view.layer_count = layers;
	view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
	image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
	              vk::AccessFlagBits2::eColorAttachmentRead |
	                  vk::AccessFlagBits2::eColorAttachmentWrite,
	              {}, command);
	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView   = image.FindView(view);
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
	attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	if (clear != nullptr) {
		attachment.loadOp     = vk::AttachmentLoadOp::eClear;
		attachment.clearValue = *clear;
	}
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {std::max(image.info.extent.width >> level, 1u),
	                                  std::max(image.info.extent.height >> level, 1u)};
	rendering.layerCount           = layers;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &attachment;
	command.beginRendering(&rendering);
	return rendering.renderArea;
}

void TextureCache::InvalidateMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid memory-invalidation range\n");
	}
	// Images register their pages before they are tracked, so CPU writes to pages without one
	// have nothing to invalidate. Most write faults land there and need not wait for the lock.
	ImagePageTable::PageRange pages {};
	if (ImagePageTable::TryGetPageRange(address, size, pages) &&
	    !m_registered_pages.Any(pages.first, pages.last_exclusive)) {
		return;
	}
	std::scoped_lock lock {m_lock};
	InvalidateCpuAliases(address, size);
}

void TextureCache::DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset) {
	const auto&    info             = image.info;
	const auto     layers           = info.resources.layers;
	const auto     full_slice_size  = info.linear_slice_stride;
	const auto     transfer_bytes   = DepthAspectTransferBytes(info.pixel_format);
	const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
	EXIT_NOT_IMPLEMENTED(transfer_bytes == 0 || texels_per_slice > UINT32_MAX ||
	                     texels_per_slice > UINT64_MAX / transfer_bytes ||
	                     texels_per_slice > UINT64_MAX / info.bytes_per_block);
	const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
	const uint64_t guest_slice    = texels_per_slice * info.bytes_per_block;
	EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
	const uint64_t transfer_size = transfer_slice * layers;
	EXIT_NOT_IMPLEMENTED(guest_slice > full_slice_size);
	if (transfer_bytes == info.bytes_per_block) {
		if (!info.IsTiled()) {
			const auto copies = info.BufferCopies(destination_offset);
			image.Download(copies, destination.Handle(), destination_offset, info.data.size);
		} else {
			const auto copies = info.BufferCopies();
			m_tiler.TileImage(image, copies, destination.Handle(), destination_offset,
			                  info.data.size);
		}
		return;
	}
	EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
	                     transfer_bytes != sizeof(uint32_t));
	const std::array copies {vk::BufferImageCopy {0,
	                                              info.pitch,
	                                              info.extent.height,
	                                              {vk::ImageAspectFlagBits::eDepth, 0, 0, layers},
	                                              {},
	                                              info.extent}};
	auto host_linear = m_tiler.GetScratchBuffer(transfer_size);
	image.Download(copies, host_linear.buffer, 0, host_linear.size);
	const bool tiled        = info.IsTiled();
	auto guest_linear = tiled ? m_tiler.GetScratchBuffer(info.LinearSize(), host_linear.buffer)
	                          : TileManager::Result {destination.Handle(), destination_offset,
	                                                 destination.Size() - destination_offset};
	m_tiler.ConvertD16(host_linear, guest_linear, TileManager::D16Direction::Demote,
	                   DepthAspectTransferFormat(info.pixel_format) == vk::Format::eD32Sfloat,
	                   {.width               = info.extent.width,
	                    .height              = info.extent.height,
	                    .layers              = layers,
	                    .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
	                    .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
	                    .source_slice_stride = transfer_slice,
	                    .target_slice_stride = full_slice_size});
	if (tiled) {
		m_tiler.Tile(guest_linear.buffer, guest_linear.offset, guest_linear.size,
		             destination.Handle(), destination_offset, info.data.size, info);
	}
}

void TextureCache::DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
                                 uint64_t destination_size, uint32_t levels) {
	if (!CanDownload(image)) {
		EXIT("TextureCache: invalid image download\n");
	}
	if (UploadBinding(image) == BindingType::DepthTarget) {
		EXIT_IF(destination_size != image.info.data.size || levels != image.info.resources.levels);
		DownloadDepth(image, destination, destination_offset);
		return;
	}

	const auto& info      = image.info;
	const auto  transform = info.GetColorTransform();
	if (info.IsTiled()) {
		const auto copies = info.BufferCopies(0, levels);
		m_tiler.TileImage(image, copies, destination.Handle(), destination_offset, destination_size,
		                  levels);
	} else if (transform != ColorTransform::None) {
		auto       linear = m_tiler.GetScratchBuffer(destination_size);
		const auto copies = info.BufferCopies(0, levels);
		image.Download(copies, linear.buffer, 0, linear.size);
		m_tiler.TransformColor(linear, {destination.Handle(), destination_offset, destination_size},
		                       transform, false);
	} else {
		const auto copies = info.BufferCopies(destination_offset, levels);
		image.Download(copies, destination.Handle(), destination_offset, destination_size);
	}
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	const auto selected = m_texture_cache.FindImageFromRange(vaddr, size);
	if (!selected) {
		return false;
	}

	std::scoped_lock lock {m_texture_cache.m_lock};
	auto& image = m_texture_cache.m_slot_images[selected];
	// The GPU thread owns image retirement; CPU invalidation can dirty this image after lookup.
	if (!image.SafeToDownload()) {
		return false;
	}
	if (!buffer.IsInBounds(image.info.data.address, 1)) {
		return false;
	}
	const auto buf_offset = buffer.Offset(image.info.data.address);
	const auto available  = buffer.Size() - buf_offset;
	uint32_t   levels     = 0;
	uint64_t   copy_size  = 0;
	if (buffer.IsInBounds(image.info.data.address, image.info.data.size)) {
		levels    = image.info.resources.levels;
		copy_size = image.info.data.size;
	} else if (image.info.IsVolume() || image.info.resources.layers > 1) {
		// Array and volume slices are strided across the full mip chain.
		return false;
	} else {
		for (; levels < image.info.resources.levels; ++levels) {
			const auto& mip = image.info.mip_layout[levels];
			if (mip.size == 0 || mip.offset > available || mip.size > available - mip.offset) {
				break;
			}
			copy_size = std::max(copy_size, mip.offset + mip.size);
		}
	}
	if (copy_size == 0) {
		return false;
	}
	if (!m_texture_cache.CanDownload(image) ||
	    (TextureCache::UploadBinding(image) == TextureCache::BindingType::DepthTarget &&
	     copy_size != image.info.data.size)) {
		return false;
	}
	ForgetColorMetadataChecks(image.info.data.address, copy_size);
	m_texture_cache.DownloadImage(image, buffer, buf_offset, copy_size, levels);
	return true;
}

bool TextureCache::DownloadImageMemory(ImageId id, bool evicting,
                                       std::shared_ptr<std::atomic<bool>>* landed) {
	auto& image = m_slot_images[id];
	if (image.depth_id) {
		return false;
	}
	if (!image.SafeToDownload() || !CanDownload(image)) {
		return false;
	}
	const auto range = image.info.data;
	auto&      ring  = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	// An image larger than the download ring gets a staging buffer of its own for this one
	// transfer: the ring cannot map it, and its regions cannot be cut into ring-sized pieces
	// the way a buffer download can.
	std::shared_ptr<Buffer> dedicated;
	Buffer*                 download = &ring;
	uint8_t*                mapped   = nullptr;
	uint64_t                offset   = 0;
	if (range.size > ring.Size()) {
		dedicated = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     AllFlags, range.size);
		download  = dedicated.get();
		mapped    = dedicated->Mapped().data();
		if (mapped == nullptr) {
			return false;
		}
	} else {
		std::tie(mapped, offset) =
		    ring.Map(range.size, std::max<uint64_t>(image.info.bytes_per_block, 4));
		if (mapped == nullptr) {
			EXIT("TextureCache: failed to map reusable download buffer\n");
		}
		ring.Commit();
	}
	if (!LibKernel::Memory::TryReadBacking(range.address, mapped, range.size)) {
		return false;
	}
	const bool guard      = evicting && EvictWriteBackGuard();
	uint64_t   guest_hash = 0;
	if (guard && !HashGuestBacking(range.address, range.size, guest_hash)) {
		return false;
	}
	download->Flush(offset, range.size);

	DownloadImage(image, *download, offset, range.size, image.info.resources.levels);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureDownloadBytes, range.size);
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download->Handle();
	barrier.offset              = offset;
	barrier.size                = range.size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                               vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                                               1, &barrier, 0, nullptr);
	std::shared_ptr<std::atomic<bool>> landed_flag;
	if (landed != nullptr) {
		landed_flag = std::make_shared<std::atomic<bool>>(false);
		*landed     = landed_flag;
	}
	m_scheduler.DeferPriorityOperation([download, dedicated, range, mapped, offset, guard,
	                                    guest_hash, landed_flag] {
		struct Landed {
			const std::shared_ptr<std::atomic<bool>>& flag;
			~Landed() {
				if (flag != nullptr) flag->store(true, std::memory_order_release);
			}
		} landed_scope {landed_flag};
		download->Invalidate(offset, range.size);
		uint64_t current = 0;
		if (guard && (!HashGuestBacking(range.address, range.size, current) ||
		              current != guest_hash)) {
			// The guest wrote (or unmapped) these bytes after the image was evicted: they are
			// newer than its pixels.
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
				LOGF("TextureCache: evicted image write-back dropped, guest=0x%016" PRIx64
				     " size=0x%" PRIx64 ": the guest bytes changed before it landed\n",
				     range.address, range.size);
			}
			return;
		}
		LibKernel::Memory::WriteBacking(range.address, mapped, range.size);
	}, range.address, range.size);
	return true;
}

void TextureCache::InvalidateMemoryFromGPU(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto& image = m_slot_images[id];
		if (image.info.data.address != address) {
			continue;
		}
		if (image.IsGpuModified()) {
			image.ClearGpuModified();
		}
		image.MarkBufferModified();
	}
}

bool TextureCache::IsRegionGpuModified(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, false)) {
		const auto& image = m_slot_images[id];
		// An old render target whose memory the CPU has reused keeps its earlier GPU-modified
		// flag (PPSA17168: S_LOAD_DWORD reads shader data over such memory).
		if (!image.depth_id && image.IsGpuModified() && !image.IsDefinitelyCpuDirty()) {
			return true;
		}
	}
	return false;
}

void TextureCache::InvalidateCpuAliases(uint64_t address, uint64_t size) {
	const auto page_begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto& owner = m_slot_images[id];
		if (owner.Overlaps(address, size)) {
			owner.MarkCpuDirty();
			UntrackImage(id);
			continue;
		}
		const auto image_begin = owner.info.data.address;
		const auto image_end   = owner.info.data.End();
		if (page_end < image_end) {
			UntrackImageHead(id);
		} else if (image_begin < page_begin) {
			UntrackImageTail(id);
		} else {
			MarkAsMaybeDirty(id, owner);
		}
	}
}

bool TextureCache::IsMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	return found != m_surface_metas.end();
}

bool TextureCache::IsMetaCleared(uint64_t address, uint32_t slice) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= found->second.clear_mask.size()) {
		return false;
	}
	return found->second.clear_mask.test(slice);
}

bool TextureCache::ClearMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end()) {
		return false;
	}
	found->second.clear_mask.set();
	return true;
}

bool TextureCache::TouchMeta(uint64_t address, uint32_t slice, bool is_clear) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= found->second.clear_mask.size()) {
		return false;
	}
	found->second.clear_mask.set(slice, is_clear);
	return true;
}

bool TextureCache::IsRegionRegistered(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	return !FindImagesInRegion(address, size, false).empty();
}

void TextureCache::UnmapMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid unmap range\n");
	}
	std::scoped_lock lock {m_lock};
	for (auto metadata = m_surface_metas.begin(); metadata != m_surface_metas.end();) {
		const auto base = metadata->first;
		if (base >= address && base < address + size) {
			metadata = m_surface_metas.erase(metadata);
		} else {
			++metadata;
		}
	}
	auto images = FindImagesInRegion(address, size, false);
	for (const auto id: images) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		FreeImage(id);
	}
}

uint64_t TextureCache::LruClock() const noexcept {
	return m_graphics.presented_frames.load(std::memory_order_relaxed) + m_gc_tick / 512;
}

namespace {
// Why the collector does not free: over each TexStats window, candidates it visited and what
// happened to them.
struct GcSkipStats {
	uint64_t visited      = 0;
	uint64_t freed        = 0;
	uint64_t freed_mb     = 0;
	uint64_t gone         = 0; // no longer registered
	uint64_t depth        = 0; // part of a depth/stencil association
	uint64_t gpu_tiled    = 0; // GPU-written tiled: needs a write-back it cannot do
	uint64_t gpu_unpress  = 0; // GPU-written, not under pressure
	uint64_t gpu_download = 0; // GPU-written, write-back failed
	uint64_t young        = 0; // runs where the LRU had nothing old enough
	uint64_t written_back = 0; // GPU-written tiled images written back, then freed
	uint64_t bindless     = 0; // bindless images freed: unsampled in the last usage probe
	uint64_t small_target = 0; // render targets of at most 1024x1024 kept (glyphs, cached layers)
	uint64_t pending      = 0; // evicted, write-back not landed yet
	uint64_t revived      = 0; // used again before the write-back landed: stays
};
GcSkipStats g_gc_skip;
} // namespace

void TextureCache::ReleaseDroppedByGuest(ImageId id) {
	std::scoped_lock lock {m_lock};
	if (auto* image = m_slot_images.try_get(id); image != nullptr) {
		image->frame_guest_dropped = m_graphics.presented_frames.load(std::memory_order_relaxed);
		m_guest_dropped.push_back(id);
	}
}

void TextureCache::RunGarbageCollector() {
	std::scoped_lock lock {m_lock};
	m_gc_tick++;
	const uint64_t clock = LruClock();
	// Evictions whose write-back has landed: free the image now, unless it was used since (then
	// its pixels are newer than the guest bytes and it stays, as if never a candidate).
	std::erase_if(m_pending_evictions, [&](const PendingEviction& pending) {
		if (!pending.landed->load(std::memory_order_acquire)) {
			return false;
		}
		m_evicting.erase(pending.id);
		auto* image = m_slot_images.try_get(pending.id);
		if (image == nullptr || !image->registered || image->info.data.address != pending.address) {
			return true;
		}
		if (image->frame_accessed_last != pending.frame) {
			g_gc_skip.revived++;
			return true;
		}
		const auto freed = image->AccountedSize();
		g_gc_skip.freed++;
		g_gc_skip.freed_mb += freed >> 20u;
		FreeImage(pending.id);
		return true;
	});
	// Memory pressure (GraphicContext::MemoryShortfall: device use past 90 % of the video memory
	// target): the thresholds sit that far below what the cache holds, so the collector frees its
	// oldest images until there is room again.
	if (m_trigger_gc_memory != m_applied_trigger_gc_memory ||
	    m_pressure_gc_memory != m_applied_pressure_gc_memory ||
	    m_critical_gc_memory != m_applied_critical_gc_memory) {
		m_base_trigger_gc_memory  = m_trigger_gc_memory;
		m_base_pressure_gc_memory = m_pressure_gc_memory;
		m_base_critical_gc_memory = m_critical_gc_memory;
	}
	if (const auto shortfall = m_graphics.MemoryShortfall(); shortfall != 0) {
		const auto collectable = CollectableMemory();
		const auto cap =
		    std::max<uint64_t>(collectable > shortfall ? collectable - shortfall : 0, 512ull << 20u);
		m_trigger_gc_memory  = std::min(m_base_trigger_gc_memory, cap);
		m_pressure_gc_memory = std::min(m_base_pressure_gc_memory, cap);
		m_critical_gc_memory = std::min(m_base_critical_gc_memory, cap + cap / 8);
	} else {
		m_trigger_gc_memory  = m_base_trigger_gc_memory;
		m_pressure_gc_memory = m_base_pressure_gc_memory;
		m_critical_gc_memory = m_base_critical_gc_memory;
	}
	m_applied_trigger_gc_memory  = m_trigger_gc_memory;
	m_applied_pressure_gc_memory = m_pressure_gc_memory;
	m_applied_critical_gc_memory = m_critical_gc_memory;
	if (!m_guest_dropped.empty()) {
		// Engine method: the guest's own streaming dropped these textures from its heaps, as an
		// engine unloads a texture; free them once no draw has used them for a few frames, not
		// when the cache reaches its memory threshold. Images the GPU wrote keep their data for
		// the regular collector (it writes them back); one a heap entry or a render target uses
		// again stays.
		const auto frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
		std::erase_if(m_guest_dropped, [&](ImageId id) {
			auto* image = m_slot_images.try_get(id);
			if (image == nullptr || !image->registered || image->bindless_pinned ||
			    image->binding.is_target || image->IsGpuModified()) {
				return true;
			}
			// A grace of 300 frames (10-20 s): the guest drops entries and takes the same texture
			// back shortly after (the jungle slash stalled 13 s with a grace of 3 frames, each
			// texture freed and created again). The drop time counts as a use.
			if (std::max(image->frame_accessed_last, image->frame_guest_dropped) + 300 > frame) {
				return false; // used or dropped lately: look again later
			}
			m_guest_dropped_freed++;
			m_guest_dropped_freed_bytes += image->AccountedSize();
			FreeImage(id);
			return true;
		});
		if (m_gc_tick % 500 == 0) {
			LOGF("TexGuestDrop: frame=%" PRIu64 " freed=%" PRIu64 " (%" PRIu64 "MB) waiting=%zu\n",
			     frame, m_guest_dropped_freed, m_guest_dropped_freed_bytes >> 20u,
			     m_guest_dropped.size());
		}
	}
	if (m_graphics.CanReportMemoryUsage()) {
		// Pressure is judged by this cache's own image bytes, which RegisterImage and
		// UnregisterImage already track. Device-wide usage counts the images this cache has
		// spilled to host memory (and every buffer), so it reads "critical" no matter how
		// small the cache is -- and never falls, so the collector cannot tell that it is the
		// one holding too much. Spider-Man's world then grew to 7.6 GB of images on a 6 GB
		// card, all of it spilled, and every shader access crossed PCIe (3 fps).
		// KYTY_GC_DEVICE_BYTES=1 restores the old policy, as in the buffer cache.
		static const bool device_bytes = std::getenv("KYTY_GC_DEVICE_BYTES") != nullptr;
		if (device_bytes) {
			m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
		}
	}
	if (m_pinned_frame != clock) {
		// Heap textures follow the guest (settled when it writes an entry, freed when it drops
		// them) and the collector cannot free them; counted against the bound they kept the cache
		// over it, and it freed every other image again and again (33,916 in 4 minutes of a new
		// game, the frame rate a fifth lower). The bounds apply to the images it can free.
		m_pinned_frame = clock;
		m_pinned_bytes = 0;
		m_slot_images.ForEach([&](ImageId, const Image& image) {
			if (image.registered && image.bindless_pinned) {
				m_pinned_bytes += image.AccountedSize();
			}
		});
	}
	if (CollectableMemory() < m_trigger_gc_memory) {
		return;
	}
	// The collector runs once per guest submission -- this title submits hundreds of times a
	// frame -- so a budget counted per run is really that many evictions per frame, and every
	// eviction is an image recreated, synchronized and submitted. Budget the work per frame
	// instead: 7 600 submissions a frame came from a per-run budget of 256.
	if (m_gc_budget_frame != clock) {
		m_gc_budget_frame       = clock;
		m_gc_freed_bytes_frame  = 0;
		m_gc_freed_images_frame = 0;
		m_gc_written_back_bytes_frame = 0;
	}
	const auto collect = [&](bool allow_aggressive) {
		bool pressured  = CollectableMemory() >= m_pressure_gc_memory;
		bool aggressive = allow_aggressive && CollectableMemory() >= m_critical_gc_memory;
		// Ages in frames. This collector runs once per guest submission, and a title can
		// submit a hundred times a frame: an age counted in its ticks evicted textures inside
		// the frame that used them, to recreate them the next (11 000 images a second at the
		// title menu). An image used in the frame being recorded is never a candidate, at any
		// pressure; over budget the collector reaches back only as far as the previous frame.
		const uint64_t age = std::min<uint64_t>(aggressive ? 1 : pressured ? 4 : 16, clock);
		// Under pressure the work is budgeted per frame, in bytes actually freed, and in
		// proportion to the excess, so the cache converges on its threshold instead of either
		// churning (a per-run budget: 7 600 submissions a frame) or growing without bound (a
		// small fixed count, spent on GPU-written render targets that are skipped: 13.5 GB).
		// Only frees count against it. A cap on images per frame bounds the synchronization
		// that recreating them costs.
		constexpr uint64_t MiB             = 1024 * 1024;
		constexpr size_t   MaxFreesInFrame = 1024;
		uint64_t           byte_budget     = 0;
		size_t             deletions       = 10;
		if (pressured || aggressive) {
			const auto threshold = aggressive ? m_critical_gc_memory : m_pressure_gc_memory;
			const auto excess    = CollectableMemory() - threshold;
			byte_budget = aggressive ? std::max<uint64_t>(64 * MiB, excess / 4)
			                         : std::max<uint64_t>(16 * MiB, excess / 8);
			if (m_gc_freed_bytes_frame >= byte_budget ||
			    m_gc_freed_images_frame >= MaxFreesInFrame) {
				return;
			}
			deletions = MaxFreesInFrame - m_gc_freed_images_frame;
		}
		// Skipped candidates (GPU-written tiled images, depth associations) do not use up the
		// budget, so look further down the LRU than the number of frees wanted.
		const size_t         visit_limit = pressured || aggressive ? deletions * 4 : deletions;
		std::vector<ImageId> candidates;
		candidates.reserve(std::min<size_t>(visit_limit, 4096));
		// Deleting depth recursively deletes its stencil association, so finish LRU traversal
		// first.
		m_lru_cache.ForEachItemBelow(clock - age, [&](ImageId id) {
			candidates.push_back(id);
			return candidates.size() >= visit_limit || candidates.size() >= 4096;
		});
		if (candidates.empty() && (pressured || aggressive)) {
			g_gc_skip.young++;
		}
		for (const auto id: candidates) {
			if (deletions == 0) {
				break;
			}
			if (!(pressured || aggressive)) {
				--deletions;
			}
			g_gc_skip.visited++;
			auto owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered) {
				g_gc_skip.gone++;
				continue;
			}
			if (owner->table_pins != 0) {
				// Bound every time without a touch: still in use.
				m_lru_cache.Touch(owner->lru_id, m_gc_tick);
				continue;
			}
			if (owner->depth_id) {
				g_gc_skip.depth++;
				continue;
			}
			if (m_evicting.contains(id)) {
				g_gc_skip.pending++;
				continue;
			}
			// Small render targets are the layers the game draws once and samples from then on
			// (glyph atlases, cached UI): a few MB each, and a lost one is lost text (letters went
			// missing when they were evicted under "critical", 2026-10-09). They are never evicted;
			// the guest dropping them (ReleaseDroppedByGuest) is what frees them.
			if (owner->usage.render_target && owner->info.extent.width <= 1024 &&
			    owner->info.extent.height <= 1024) {
				g_gc_skip.small_target++;
				continue;
			}
			if (owner->bindless_pinned) {
				// Pinned for bindless draws; under pressure one no draw sampled during the last
				// usage probe may go (ISSUES #15: on 4-8 GB cards these grew to 2-5 GB and never
				// shrank).
				if (!(pressured || aggressive) || !m_bindless_evictable.contains(id)) {
					continue;
				}
				m_bindless_evictable.erase(id);
				g_gc_skip.bindless++;
			}
			if (owner->IsGpuModified()) {
				const bool safe = owner->SafeToDownload();
				// A GPU-written tiled image was never evicted: render targets and textures the
				// GPU decompresses are almost all of a 4K title's images, so under pressure the
				// collector found nothing else it could free (Spider-Man: 3 500 of every 4 000
				// candidates) and the cache grew to 13.5 GB until the driver crashed in
				// present. Under pressure such an image is now written back, detiled, like any
				// other, within a per-frame write-back budget; candidates are never ones used in
				// the frame being recorded. KYTY_GC_KEEP_TILED=1 restores the old rule.
				static const bool keep_tiled = std::getenv("KYTY_GC_KEEP_TILED") != nullptr;
				constexpr uint64_t WriteBackPerFrame = 256 * MiB;
				if (safe && owner->info.IsTiled() &&
				    (keep_tiled || !(pressured || aggressive) ||
				     (m_gc_written_back_bytes_frame != 0 &&
				      m_gc_written_back_bytes_frame + owner->info.data.size > WriteBackPerFrame))) {
					g_gc_skip.gpu_tiled++;
					continue;
				}
				if (safe && !pressured) {
					g_gc_skip.gpu_unpress++;
					continue;
				}
				if (!safe) {
					// GPU-written, and CPU- or copy-written as well: its contents exist only in
					// this image and cannot be written back as they are. Freeing it lost them:
					// under a 6 GiB bound the Leap Attack prompt read "L p Att k" (the glyph atlas,
					// filled by both). It stays until it is safe or the guest drops it.
					g_gc_skip.gpu_download++;
					continue;
				}
				std::shared_ptr<std::atomic<bool>> landed;
				if (safe && !DownloadImageMemory(id, true, &landed)) {
					g_gc_skip.gpu_download++;
					continue;
				}
				if (safe && owner->info.IsTiled()) {
					m_gc_written_back_bytes_frame += owner->info.data.size;
					g_gc_skip.written_back++;
				}
				// Freed once the write-back has landed (the top of the next run); the bytes count
				// against this frame's budget now, since they are going.
				m_pending_evictions.push_back(
				    {id, owner->info.data.address, owner->frame_accessed_last, std::move(landed)});
				m_evicting.insert(id);
				if (pressured || aggressive) {
					--deletions;
					m_gc_freed_images_frame++;
					m_gc_freed_bytes_frame += owner->AccountedSize();
					if (m_gc_freed_bytes_frame >= byte_budget) {
						break;
					}
				}
				continue;
			}
			const auto freed = owner->AccountedSize();
			g_gc_skip.freed++;
			g_gc_skip.freed_mb += freed >> 20u;
			FreeImage(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureEvictions);
			if (pressured || aggressive) {
				--deletions;
				m_gc_freed_images_frame++;
				m_gc_freed_bytes_frame += freed;
				if (m_gc_freed_bytes_frame >= byte_budget) {
					break;
				}
			}
			if (CollectableMemory() < m_critical_gc_memory && aggressive) {
				deletions >>= 2;
				aggressive = false;
			}
			if (CollectableMemory() < m_pressure_gc_memory && pressured) {
				deletions >>= 1;
				pressured = false;
			}
		}
	};
	collect(false);
	if (CollectableMemory() >= m_critical_gc_memory) {
		collect(true);
	}
	if (m_gc_tick % 500 == 0) {
		uint64_t images = 0;
		uint64_t bytes  = 0;
		m_slot_images.ForEach([&](ImageId, const Image& image) {
			if (image.registered) {
				images++;
				bytes += image.AccountedSize();
			}
		});
		LOGF("TexStats: runs=%" PRIu64 " frame=%" PRIu64 " images=%" PRIu64 " image_bytes=%" PRIu64
		     "MB device_used=%" PRIu64 "MB pressure=%" PRIu64 "MB critical=%" PRIu64 "MB\n",
		     m_gc_tick, m_graphics.presented_frames.load(std::memory_order_relaxed), images,
		     bytes >> 20u, m_total_used_memory >> 20u, m_pressure_gc_memory >> 20u,
		     m_critical_gc_memory >> 20u);
		LOGF("TexGc: visited=%" PRIu64 " freed=%" PRIu64 " (%" PRIu64 "MB) written_back=%" PRIu64
		     " gone=%" PRIu64 " depth=%" PRIu64 " gpu_tiled=%" PRIu64 " gpu_unpressured=%" PRIu64
		     " gpu_download_failed=%" PRIu64 " nothing_old_enough=%" PRIu64 " bindless=%" PRIu64
		     " small_target=%" PRIu64 " pending=%" PRIu64 " revived=%" PRIu64 "\n",
		     g_gc_skip.visited, g_gc_skip.freed, g_gc_skip.freed_mb, g_gc_skip.written_back,
		     g_gc_skip.gone, g_gc_skip.depth, g_gc_skip.gpu_tiled, g_gc_skip.gpu_unpress,
		     g_gc_skip.gpu_download, g_gc_skip.young, g_gc_skip.bindless, g_gc_skip.small_target,
		     g_gc_skip.pending, g_gc_skip.revived);
		g_gc_skip = {};
	}
}

void TextureCache::SetBindlessEvictable(std::span<const ImageId> ids) {
	std::scoped_lock lock {m_lock};
	m_bindless_evictable.clear();
	m_bindless_evictable.insert(ids.begin(), ids.end());
}

void TextureCache::ProcessDownloadImages() {
	if (!m_readback_linear_images && !m_readback_tiled_images) {
		return;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: m_download_images) {
		if (m_slot_images[id].IsGpuModified()) {
			(void)DownloadImageMemory(id);
		}
	}
	m_download_images.clear();
}

} // namespace Libs::Graphics
