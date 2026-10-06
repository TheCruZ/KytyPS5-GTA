#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace Libs::Graphics {

struct ShaderStageRuntime;

// The state of a cached image that its lookups and acquisitions depend on. While an image keeps
// it (and the image set around its range is unchanged), looking it up again and acquiring its
// views finds the same image and views and changes nothing.
struct ImageLookupState {
	ImageId  id;
	ImageId  depth_id;
	uint64_t track_addr      = 0;
	uint64_t track_addr_end  = 0;
	bool     cpu_dirty       = false;
	bool     buffer_modified = false;
	bool     gpu_modified    = false;
	bool     storage         = false;
	bool     render_target   = false;

	bool operator==(const ImageLookupState&) const = default;

	[[nodiscard]] static ImageLookupState Of(ImageId id, const Image& image) {
		return {.id              = id,
		        .depth_id        = image.depth_id,
		        .track_addr      = image.track_addr,
		        .track_addr_end  = image.track_addr_end,
		        .cpu_dirty       = image.IsCpuDirty(),
		        .buffer_modified = image.IsBufferModified(),
		        .gpu_modified    = image.IsGpuModified(),
		        .storage         = image.usage.storage,
		        .render_target   = image.usage.render_target};
	}
};

// The image state a bindless table element's view was last acquired in: while the image keeps
// it, the view still holds and acquiring it again would only refresh the image's use.
struct TableViewMemo {
	ImageLookupState state;
	bool             valid = false;
};

struct TextureBinding {
	ImageId                    image_id;
	vk::ImageView              image_view = nullptr;
	TextureCache::ImageDesc    desc;
	vk::ImageLayout            layout = vk::ImageLayout::eUndefined;
	std::vector<vk::ImageView> mip_views;
};

struct PreparedBindings {
	struct BufferSource {
		uint64_t address = 0;
		uint64_t size    = 0;
		BufferId id;
	};

	// The draw owns the immutable compiled-program/runtime-snapshot association through commit.
	const ShaderStageRuntime* runtime = nullptr;
	// Keep the resolved guest range through cache preparation; only the host buffer ID may
	// become stale and need resolving again when bindings are rebound.
	std::vector<BufferSource>             buffer_sources;
	std::vector<vk::DescriptorBufferInfo> buffers;
	// The uniform views of the buffers that have a uniform_index, by that index.
	std::vector<vk::DescriptorBufferInfo> uniform_buffers;
	std::vector<TextureBinding>           images;
	// The distinct bindless table elements, with the T# and the root image each was resolved
	// for, and, for every slot of every table range in binding order, the element it binds.
	// GTA V's ray tracing tables have ~16k slots that name far fewer distinct textures. The
	// bindings live in the executor's table resolutions (see
	// RenderExecutor::TableElementResolution), which keep them until ResetBindings().
	std::vector<TextureBinding*>                        table_images;
	std::vector<TableViewMemo*>                         table_views;
	std::vector<ShaderRecompiler::IR::DescriptorValue> table_sources;
	std::vector<uint32_t>                               table_roots;
	std::vector<uint32_t>                               table_slots;
	// The program's table binding state (RenderExecutor::TableBindingState). When
	// `tables_reused`, the table vectors above stay empty: the tables bind what the state holds,
	// and `table_visits` lists the state's elements this binding looks up again.
	void*                                               table_state   = nullptr;
	bool                                                tables_reused = false;
	std::vector<uint32_t>                               table_visits;
	std::vector<vk::Sampler>              samplers;
	vk::DescriptorBufferInfo              gds {nullptr, 0, VK_WHOLE_SIZE};
	vk::DescriptorBufferInfo              flattened_srt;
	vk::DescriptorBufferInfo              shader_data_buffer;
	vk::DescriptorBufferInfo              shared_memory;
	std::vector<uint32_t>                 shader_data;
	// Bindless: flattened-SRT words to patch (offset, region, entries) and the heaps whose
	// resolved images must be readable for this draw (stable: the table keeps heaps in a deque).
	std::vector<std::array<uint32_t, 3>>  bindless_patches;
	std::vector<BindlessTable::Heap*>     bindless_heaps;
};

[[nodiscard]] vk::DescriptorType
NativeDescriptorType(ShaderRecompiler::IR::DescriptorBindingKind kind);
[[nodiscard]] vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture,
                                                    uint32_t              element = 0);

template <typename T>
[[nodiscard]] T DecodeNativeDescriptor(const ShaderRecompiler::IR::DescriptorValue& value) {
	static_assert(std::is_trivially_copyable_v<T>);
	static_assert(sizeof(T) % sizeof(uint32_t) == 0);
	T result {};
	EXIT_IF(value.dword_count < sizeof(result) / sizeof(uint32_t));
	std::memcpy(&result, value.dwords.data(), sizeof(result));
	return result;
}

[[nodiscard]] bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor,
                                                   bool r128 = false);
void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size);

// Diagnostics (KYTY_WATCH_SHADER): logs again, a few frames later, the user-data buffers a
// watched draw read, to tell data written after the draw from data never written.
void LogWatchedRereads(BufferCache& cache, uint64_t presented_frames);

// Diagnostics: KYTY_WATCH_SHADER=<hash> with KYTY_WATCH_SHADER_FRAME=<n>: whether a use of the
// shader falls in the n-th presented frame after its first use (each caller counts up to 64).
bool InWatchedShaderFrame(uint64_t hash, uint64_t presented_frame, std::atomic<uint32_t>& logged);
// The presented frame of the watched shader's first use, or UINT64_MAX.
uint64_t WatchedShaderFirstFrame();

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
