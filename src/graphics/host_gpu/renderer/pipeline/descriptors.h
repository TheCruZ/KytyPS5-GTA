#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace Libs::Graphics {

struct ShaderStageRuntime;

struct TextureDescDerivation;

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
	bool     texture         = false;
	bool     storage         = false;
	bool     render_target   = false;
	bool     depth_target    = false;

	bool operator==(const ImageLookupState&) const = default;

	[[nodiscard]] static ImageLookupState Of(ImageId id, const Image& image) {
		return {.id              = id,
		        .depth_id        = image.depth_id,
		        .track_addr      = image.track_addr,
		        .track_addr_end  = image.track_addr_end,
		        .cpu_dirty       = image.IsCpuDirty(),
		        .buffer_modified = image.IsBufferModified(),
		        .gpu_modified    = image.IsGpuModified(),
		        .texture         = image.usage.texture,
		        .storage         = image.usage.storage,
		        .render_target   = image.usage.render_target,
		        .depth_target    = image.usage.depth_target};
	}
};

struct TextureBinding {
	ImageId                    image_id;
	vk::ImageView              image_view = nullptr;
	TextureCache::ImageDesc    desc;
	vk::ImageLayout            layout = vk::ImageLayout::eUndefined;
	std::vector<vk::ImageView> mip_views;
	// The description whose last lookup this binding reuses (image_view is set) or records once
	// its view is acquired; see RenderExecutor::ResolveTexture().
	TextureDescDerivation* lookup = nullptr;
	bool                   reused = false;
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
	std::vector<TextureBinding>           images;
	// Bindless table descriptors in binding and table-range order, with the T# and the root
	// image each element was resolved for.
	std::vector<TextureBinding>                         table_images;
	std::vector<ShaderRecompiler::IR::DescriptorValue> table_sources;
	std::vector<uint32_t>                               table_roots;
	std::vector<vk::Sampler>              samplers;
	vk::DescriptorBufferInfo              gds {nullptr, 0, VK_WHOLE_SIZE};
	vk::DescriptorBufferInfo              flattened_srt;
	vk::DescriptorBufferInfo              shader_data_buffer;
	std::vector<uint32_t>                 shader_data;
	// Threads of the compute dispatch, bounding buffers whose NUM_RECORDS the shader computes.
	uint64_t                              dispatch_threads = 0;
	// What the previous draw of this stage bound: a stage that binds the same program with the
	// same T#s and S#s again (about half of them) skips the description lookups.
	const void*                                        last_program          = nullptr;
	uint64_t                                           last_descs_generation = 0;
	std::vector<ShaderRecompiler::IR::DescriptorValue> last_images;
	std::vector<ShaderRecompiler::IR::DescriptorValue> last_samplers;
	// The description each T# of `last_images` has (null when its lookups are not reused).
	std::vector<TextureDescDerivation*>                image_descs;
};

[[nodiscard]] vk::DescriptorType
NativeDescriptorType(ShaderRecompiler::IR::DescriptorBindingKind kind);
[[nodiscard]] uint32_t
NativeDescriptorCount(const ShaderRecompiler::IR::DescriptorBinding& binding);
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

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
