#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/execHelpers.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <span>
#include <vector>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

} // namespace

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	const auto image_class = ShaderRecompiler::IR::ImageBindingResourceClass(kind);
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return vk::DescriptorType::eSampledImage;
	}
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return vk::DescriptorType::eStorageImage;
	}
	switch (kind) {
		case BindingKind::Samplers: return vk::DescriptorType::eSampler;
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::FlattenedSrt:
		case BindingKind::ShaderData:
		case BindingKind::SharedMemory: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

uint32_t NativeDescriptorCount(const ShaderRecompiler::IR::DescriptorBinding& binding) {
	return binding.resources.empty() && binding.tables.empty() ? 1u : binding.ElementCount();
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(!texture.image_id || view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1DArray: return Prospero::ImageType::kColor1D;
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource& resource, uint32_t& buffer_offset) {
	buffer_offset = 0;

	const auto& [address, size, id] = source;
	if (address == 0 || size == 0) {
		return {context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, 16};
	}
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	auto [buffer, offset] = context.GetBufferCache().ObtainBuffer(
	    address, size, resource.written,
	    resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled()), id);
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	const vk::DescriptorBufferInfo result {buffer->Handle(), aligned_offset, size + adjustment};
	if (resource.written) {
		context.GetTextureCache().InvalidateMemoryFromGPU(address, size);
	}
	return result;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	const bool         common          = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                                     (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                                     descriptor.fields[3] == field3_expected;
	if (r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	const bool full = common && descriptor.fields[4] == field4_expected &&
	                  descriptor.fields[5] == field5_expected;
	if (!full ||
	    (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type()))) {
		return false;
	}
	const auto metadata_control = descriptor.fields[6] & 0x00ffffffu;
	if (metadata_control == 0) {
		return true;
	}
	constexpr uint32_t htile_control = 0x00280000u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && GuestRange {metadata_addr, 1}.Valid() &&
	       (metadata_addr & 0x7fffu) == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth;
}

static void ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	const bool encoding_ok = IsSupportedDepthTextureEncoding(descriptor, resource.r128);
	const bool view_ok =
	    IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW());
	if (resource_ok && encoding_ok && view_ok) {
		return;
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	EXIT("unsupported sampled depth image: resource=%d encoding=%d view=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	     "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	     "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, encoding_ok, view_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written, resource.atomic,
	     resource.depth_compare, static_cast<uint32_t>(descriptor.Format()),
	     descriptor.DstSelXYZW(), static_cast<int>(image.info.pixel_format),
	     static_cast<int>(view_format), image.info.resources.layers,
	     static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	     descriptor_pitch, image.info.pitch, descriptor.Base40(), size, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	// Storage cube coordinates address individual faces, including partial cube views.
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic);
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d) && supported_tile &&
	       descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() == 0 && supported_swizzle && descriptor.BCSwizzle() == 0 &&
	       !descriptor.MsaaDepth();
}

static bool IsSupportedStorageTextureEncoding(const ShaderRecompiler::IR::ImageResource& resource,
                                              const ShaderTextureResource& descriptor) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	constexpr uint32_t field5_expected      = 0x00700000u;
	constexpr uint32_t field5_max_mip_mask  = 0x000000f0u;
	const uint32_t     expected_field3 = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     expected_field4 =
	    descriptor.Depth() | (static_cast<uint32_t>(descriptor.BaseArray5()) << 16u);
	const bool common = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                    (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                    descriptor.fields[3] == expected_field3;
	if (resource.r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	return common && descriptor.fields[4] == expected_field4 &&
	       (descriptor.fields[5] & ~field5_max_mip_mask) == field5_expected;
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsSupportedStorageTextureEncoding(resource, descriptor);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage = format == Prospero::BufferFormat::k32SInt && uint_resource &&
	                              resource.written && !resource.read && !resource.atomic;
	const auto numeric_class = Prospero::SampledTextureNumericClass(format);
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	const bool format_ok =
	    raw_sint_storage || raw_float_atomic ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                       : Prospero::BufferFormat::k32UInt)));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	EXIT("unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.atomic, resource.depth_compare,
	     descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(), descriptor.MinLod(),
	     descriptor.BaseArray5(), descriptor.BCSwizzle(), descriptor.MsaaDepth(),
	     Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static TextureCache::ImageDesc NullTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                                               TextureCache::BindingType                  binding) {
	TextureCache::ImageDesc desc {};
	switch (resource.numeric_class) {
		case Prospero::TextureNumericClass::Float:
			desc.info.guest_format = Prospero::BufferFormat::k32Float;
			break;
		case Prospero::TextureNumericClass::Uint:
			desc.info.guest_format = resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
			                                         : Prospero::BufferFormat::k32UInt;
			break;
		case Prospero::TextureNumericClass::Sint:
			desc.info.guest_format = Prospero::BufferFormat::k32SInt;
			break;
		default: EXIT("null image has unsupported numeric class\n");
	}
	desc.info.pixel_format    = VulkanFormat(desc.info.guest_format);
	desc.info.type            = Prospero::ImageType::kColor2D;
	desc.info.extent          = {1, 1, 1};
	desc.info.resources       = {1, 1};
	desc.info.bytes_per_block = Prospero::NumBytesPerElement(desc.info.guest_format);
	desc.info.samples         = 1;
	desc.info.mip_layout[0]   = {0, 0, 1, 1};
	desc.view_info.format     = resource.atomic64 ? vk::Format::eR64Uint : desc.info.pixel_format;
	desc.view_info.type       = vk::ImageViewType::e2D;
	desc.view_info.aspect     = vk::ImageAspectFlagBits::eColor;
	desc.view_info.usage      = binding == TextureCache::BindingType::Storage
	                                ? vk::ImageUsageFlagBits::eStorage
	                                : vk::ImageUsageFlagBits::eSampled;
	desc.type                 = binding;
	return desc;
}

static void PopulateTextureMipLayout(ImageInfo& info) {
	if (info.IsVolume() && info.tile_mode != Prospero::TileMode::kLinear) {
		TileSurfaceLayout            surface {};
		const TileSurfaceDescription description {
		    info.guest_format,  info.tile_mode,    TileSurfaceDimension::Dim3D, info.extent.width,
		    info.extent.height, info.extent.depth, info.resources.levels,       1};
		if (!TileGetTiledTextureLayout(description, surface)) {
			EXIT("unsupported normalized volume texture layout\n");
		}
		for (uint32_t level = 0; level < info.resources.levels; level++) {
			const auto& mip        = surface.mips[level];
			info.mip_layout[level] = {
			    mip.offset,
			    mip.size,
			    mip.padded_width,
			    mip.padded_height,
			};
		}
		return;
	}

	TileSizeOffset levels[16] {};
	TilePaddedSize padded[16] {};
	TileGetTextureSize(info.guest_format, info.extent.width, info.extent.height,
	                   info.resources.levels, info.tile_mode, nullptr, levels, padded);
	const auto texel_shift = info.IsBlock() ? 2u : 0u;
	for (uint32_t level = 0; level < info.resources.levels; level++) {
		const auto offset =
		    levels[level].src_size != 0 ? levels[level].src_offset : levels[level].offset;
		auto size = static_cast<uint64_t>(levels[level].src_size != 0 ? levels[level].src_size
		                                                              : levels[level].size);
		if (info.IsVolume()) {
			size *= std::max(info.extent.depth >> level, 1u);
		} else {
			size *= info.resources.layers;
		}
		info.mip_layout[level] = {
		    offset,
		    size,
		    padded[level].width >> texel_shift,
		    padded[level].height >> texel_shift,
		};
	}
}

static ImageViewInfo TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                                     const ShaderTextureResource& descriptor, vk::Format format,
                                     const SurfaceFormatInfo& surface_format, bool storage,
                                     uint32_t view_levels, uint32_t image_layers) {
	ImageViewInfo view {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	if (descriptor.MinLod() > descriptor.LastLevel() * 256u) {
		EXIT("texture minimum LOD exceeds last mip level: min_lod=%u last_level=%u\n",
		     descriptor.MinLod(), descriptor.LastLevel());
	}
	const auto base_lod = view.base_level * 256u;
	if (descriptor.MinLod() > base_lod) {
		view.min_lod = descriptor.MinLod() - base_lod;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping =
	    storage || surface_format.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			view.type       = vk::ImageViewType::e1D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			view.type       = vk::ImageViewType::e1DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		default: EXIT("unsupported texture view dimension\n");
	}
	return view;
}

static bool ResolveTextureMipView(const TileSurfaceDescription& description, bool metadata,
                                   uint32_t view_levels, uint32_t& levels, uint32_t& base_level) {
	TileSurfaceLayout physical {};
	TileSurfaceLayout view {};
	auto              view_description = description;
	view_description.levels            = levels;
	if (!TileGetTiledTextureLayout(description, physical) ||
	    !TileGetTiledTextureLayout(view_description, view)) {
		return false;
	}
	if (physical.first_tail_level == view.first_tail_level &&
	    physical.block_slice_size == view.block_slice_size &&
	    physical.total_size == view.total_size &&
	    std::equal(std::begin(physical.mips), std::begin(physical.mips) + description.levels,
	               std::begin(view.mips))) {
		return true;
	}
	if (metadata || ((description.layers > 1 || description.depth > 1) &&
	                 physical.block_slice_size != view.block_slice_size)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= description.levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (physical.mips[target] != view.mips[source] ||
			    (target >= physical.first_tail_level) != (source >= view.first_tail_level) ||
			    std::max(description.width >> target, 1u) != std::max(description.width >> source, 1u) ||
			    std::max(description.height >> target, 1u) != std::max(description.height >> source, 1u) ||
			    std::max(description.depth >> target, 1u) != std::max(description.depth >> source, 1u)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			levels     = description.levels;
			base_level = base;
			return true;
		}
	}
	return false;
}

// What ResolveTexture derives from a T# and the shader's view of it before any texture-cache
// lookup: a pure function of both (EXITs on unsupported descriptors).
static TextureDescDerivation DeriveTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                                               const ShaderTextureResource& descriptor) {
	const bool storage = resource.written;
	const auto address         = descriptor.Base40();
	const auto width           = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height          = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level      = descriptor.BaseLevel();
	const auto last_level      = descriptor.LastLevel();
	const auto type            = TextureType(descriptor);
	const bool multisampled    = IsMultisampledTexture(type);
	const auto max_mip         = resource.r128 ? last_level : descriptor.MaxMip();
	const auto physical_levels = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	// IMAGE_STORE addresses BASE_LEVEL; only IMAGE_STORE_MIP selects other view mips.
	const bool single_storage_mip =
	    storage && resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic;
	const auto view_levels = multisampled || single_storage_mip
	                             ? 1u
	                             : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && base_level > last_level) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		EXIT("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto format         = descriptor.Format();
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || resource.numeric_class == Prospero::SampledTextureNumericClass(format);
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		EXIT("sampled image numeric class mismatch: numeric=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	const auto    image_layers = layered ? depth : 1u;
	auto          view_base    = static_cast<uint32_t>(base_level);
	if (levels > physical_levels) {
		const TileSurfaceDescription physical {
		    format, tile, volume ? TileSurfaceDimension::Dim3D : TileSurfaceDimension::Dim2D,
		    width, height, volume ? depth : 1u, physical_levels, image_layers};
		if (!ResolveTextureMipView(physical, !resource.r128 && descriptor.MetaCompress(),
		                           view_levels, levels, view_base)) {
			EXIT("unsupported texture mip view changes physical layout: base=%u last=%u max=%u "
			     "extent=%ux%ux%u tile=%u\n",
			     base_level, last_level, max_mip, width, height, depth,
			     static_cast<uint32_t>(tile));
		}
	}
	uint32_t      pitch = 0;
	TileSizeAlign size {};
	if (multisampled) {
		const auto bytes = Prospero::NumBytesPerElement(format);
		pitch            = depth_tile ? TileGetDepthPitch(width, bytes, last_level)
		                              : TileGetRenderTargetPitch(width, bytes, last_level);
		if (pitch == 0 || !TileGetRenderTargetSize(width, height, pitch, bytes, size, last_level) ||
		    size.size > UINT32_MAX / image_layers) {
			EXIT("unsupported multisample texture layout\n");
		}
		size.size *= image_layers;
	} else {
		pitch = TileGetTexturePitch(format, width, tile);
		TileGetTextureTotalSize(format, width, height, volume ? depth : image_layers,
		                        physical_levels, tile, volume, size);
	}
	EXIT_NOT_IMPLEMENTED(size.size == 0 || size.align == 0 ||
	                     (address & (static_cast<uint64_t>(size.align) - 1u)) != 0);
	if (storage) {
		ValidateStorageTexture(resource, descriptor, size.size);
	}

	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	const auto storage_view_format = resource.atomic64 ? vk::Format::eR64Uint
	                                 : storage && (resource.atomic ||
	                                               format == Prospero::BufferFormat::k32SInt)
	                                     ? vk::Format::eR32Uint
	                                     : SrgbStorageViewFormat(pixel_format);
	const auto view_format         = storage && storage_view_format != vk::Format::eUndefined
	                                     ? storage_view_format
	                                     : pixel_format;
	const auto block_bytes         = Prospero::BlockCompressedBytesPerBlock(format);
	TextureCache::ImageDesc desc {};
	desc.info.data         = {address, size.size};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.pitch        = pitch;
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth &&
	    !desc.info.IsDepth()) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.kind          = ImageMetadataKind::Dcc;
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
	}
	if (samples > 1) {
		desc.info.mip_layout[0] = {0, size.size, pitch, height};
	} else {
		PopulateTextureMipLayout(desc.info);
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, surface_format, storage,
	                                 view_levels, desc.info.resources.layers);
	desc.view_info.base_level = view_base;
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;
	return {std::move(desc), shader_conversion, pixel_format, view_format, size.size};
}

size_t RenderExecutor::TextureDescKeyHash::operator()(const TextureDescKey& key) const noexcept {
	// Every draw hashes the key of each texture: independent multiplies of 64-bit lanes instead
	// of a dependent chain per dword.
	std::array<uint64_t, 7> lanes {};
	static_assert(sizeof(key.dwords) + sizeof(key.view) <= sizeof(lanes));
	std::memcpy(lanes.data(), key.dwords.data(), sizeof(key.dwords));
	std::memcpy(reinterpret_cast<uint8_t*>(lanes.data()) + sizeof(key.dwords), key.view.data(),
	            sizeof(key.view));
	constexpr std::array<uint64_t, 7> Multipliers {
	    0x9e3779b97f4a7c15ull, 0xc2b2ae3d27d4eb4full, 0x165667b19e3779f9ull, 0xd6e8feb86659fd93ull,
	    0xff51afd7ed558ccdull, 0xc4ceb9fe1a85ec53ull, 0x94d049bb133111ebull};
	uint64_t hash = 0;
	for (size_t i = 0; i < lanes.size(); i++) {
		const auto mixed = (lanes[i] ^ (lanes[i] >> 31u)) * Multipliers[i];
		hash ^= (mixed << (i * 9u % 64u)) | (mixed >> ((64u - i * 9u % 64u) % 64u));
	}
	hash ^= hash >> 32u;
	hash *= 0x9e3779b97f4a7c15ull;
	return static_cast<size_t>(hash ^ (hash >> 29u));
}

RenderExecutor::TextureDescKey
RenderExecutor::MakeTextureDescKey(const ShaderRecompiler::IR::ImageResource&   resource,
                                   const ShaderRecompiler::IR::DescriptorValue& value) {
	TextureDescKey key {};
	key.dwords  = value.dwords;
	key.view[0] = static_cast<uint32_t>(resource.resource_class) |
	              (static_cast<uint32_t>(resource.numeric_class) << 8u) |
	              (static_cast<uint32_t>(resource.dimension) << 16u) |
	              (static_cast<uint32_t>(resource.mip_mode) << 24u);
	key.view[1] = resource.mip_count;
	key.view[2] = static_cast<uint32_t>(resource.conversion_format);
	key.view[3] = resource.shader_swizzle;
	key.view[4] =
	    static_cast<uint32_t>(resource.read) | (static_cast<uint32_t>(resource.written) << 1u) |
	    (static_cast<uint32_t>(resource.atomic) << 2u) |
	    (static_cast<uint32_t>(resource.depth_compare) << 3u) |
	    (static_cast<uint32_t>(resource.cube) << 4u) | (static_cast<uint32_t>(resource.r128) << 5u);
	return key;
}

bool RenderExecutor::ReuseTextureLookup(TextureDescDerivation& derived, TextureBinding& binding) {
	auto&       texture_cache = m_context.GetTextureCache();
	auto&       found         = derived.found;
	auto*       image         = texture_cache.m_slot_images.try_get(found.id);
	const auto& range         = found.desc.info.data;
	if (image == nullptr || !image->registered || image->binding.needs_rebind ||
	    ImageLookupState::Of(found.id, *image) != found.state) {
		found.valid = false;
		return false;
	}
	if (const auto set_epoch = texture_cache.ImageSetEpoch(); found.image_set_epoch != set_epoch) {
		if (texture_cache.ImageEpochInRegion(range.address, range.size) > found.image_set_epoch) {
			found.valid = false;
			return false;
		}
		// No image of the range changed until now: later draws compare with now instead of
		// scanning the pages of the range again.
		found.image_set_epoch = set_epoch;
	}
	// As the lookup and the acquisition would: the image stays in use.
	image->tick_accessed_last = m_context.GetCommandScheduler().CurrentTick();
	texture_cache.TouchImage(*image);
	binding.image_id   = found.id;
	binding.image_view = found.view;
	binding.desc       = found.desc;
	binding.layout     = vk::ImageLayout::eUndefined;
	binding.mip_views.clear();
	binding.lookup = &derived;
	binding.reused = true;
	return true;
}

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value,
                                              bool table_candidate, GuestRange* examined) {
	if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	auto       descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage    = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	auto& texture_cache = m_context.GetTextureCache();
	if (descriptor.IsNull()) {
		auto       desc = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                                    : TextureCache::BindingType::Texture);
		const auto id   = texture_cache.FindImage(desc);
		return {id, nullptr, std::move(desc)};
	}

	// The description is computed once per T# and view of it; images are looked up every time.
	const auto key    = MakeTextureDescKey(resource, value);
	auto       cached = m_texture_descs.find(key);
	if (cached == m_texture_descs.end()) {
		// Derived before insertion: a derivation that exits (softly, for table candidates) leaves
		// nothing behind.
		auto derived = DeriveTextureDesc(resource, descriptor);
		if (m_texture_descs.size() >= MaxTextureDescs) {
			m_clear_texture_descs = true;
		}
		cached = m_texture_descs.emplace(key, std::move(derived)).first;
	}
	auto& derived = cached->second;
	// Sampled textures without metadata or stencil: the image and view the last lookup of this
	// description found, when nothing they depend on changed (see TextureDescDerivation::Found).
	// Lookups with metadata materialize fast clears and run every time.
	const bool reusable = !table_candidate && !storage && examined == nullptr &&
	                      resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic &&
	                      derived.desc.info.metadata.kind == ImageMetadataKind::None &&
	                      !derived.desc.info.HasStencil();
	if (reusable && derived.found.valid) {
		TextureBinding binding;
		if (ReuseTextureLookup(derived, binding)) {
			return binding;
		}
	}
	auto        desc              = derived.desc;
	const bool  shader_conversion = derived.shader_conversion;
	const auto  pixel_format      = derived.pixel_format;
	const auto  view_format       = derived.view_format;
	const auto  size_bytes        = derived.size;

	if (examined != nullptr) {
		*examined = desc.info.data;
	}
	if (table_candidate &&
	    ((!texture_cache.IsRegionGpuModified(desc.info.data.address, desc.info.data.size) &&
	      !Libs::LibKernel::Memory::IsBackingReadable(desc.info.data.address,
	                                                  desc.info.data.size)) ||
	     (desc.info.metadata.kind != ImageMetadataKind::None &&
	      !Libs::LibKernel::Memory::IsBackingReadable(desc.info.metadata.range.address,
	                                                  desc.info.metadata.range.size)))) {
		EXIT("texture 0x%016" PRIx64 " size 0x%" PRIx64 " has no guest backing\n",
		     desc.info.data.address, desc.info.data.size);
	}
	auto id = table_candidate ? texture_cache.FindImageSpeculative(desc, shader_conversion)
	                          : texture_cache.FindImage(desc, shader_conversion);
	if (!id) {
		// Only speculative lookups decline: the table binds this element as a null image.
		return {id, nullptr, std::move(desc)};
	}
	auto*      image               = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, pixel_format, size_bytes);
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else {
		(void)SelectSampledColorView(image->info.pixel_format, pixel_format,
		                             descriptor.DstSelXYZW());
	}
	TextureBinding binding {id, nullptr, std::move(desc)};
	// Recorded once the view is acquired (AcquireImageViews()). Stencil associations and images
	// with stencil refresh another image as well: they are looked up every time.
	if (reusable && !stencil_association && !image->info.HasStencil()) {
		binding.lookup = &derived;
	}
	return binding;
}

// A bindless table holds every texture the game has loaded; bind a T# the shader's view cannot
// read (other class, dimension or layout) as a null image, which the shader never selects for it.
TextureBinding RenderExecutor::ResolveTableTexture(
    const ShaderRecompiler::IR::ImageResource&   root,
    const ShaderRecompiler::IR::DescriptorValue& value, GuestRange* examined) {
	if (ShaderRecompiler::IR::ImageTableSlotCompatible(root, value)) {
		try {
			Common::SoftExitScope soft_exit(true);
			auto texture = ResolveTexture(root, value, true, examined);
			if (texture.image_id) {
				return texture;
			}
		} catch (const Common::SoftExitError& error) {
			static std::atomic_uint reported = 0;
			if (reported.fetch_add(1) < 8u) {
				LOGF("bindless table texture bound as null: %s\n", error.what());
			}
		}
	}
	return ResolveTexture(root, ShaderRecompiler::IR::DescriptorValue {.dword_count = 8u});
}

size_t RenderExecutor::TableDescriptorHash::operator()(
    const std::array<uint32_t, 8>& dwords) const noexcept {
	uint64_t hash = 0xcbf29ce484222325ull;
	for (const auto dword: dwords) {
		hash = (hash ^ dword) * 0x100000001b3ull;
		hash ^= hash >> 29u;
	}
	return static_cast<size_t>(hash);
}

RenderExecutor::TableResolution&
RenderExecutor::FindTableResolution(const ShaderRecompiler::IR::ImageResource& root) {
	constexpr size_t MaxTables   = 8;
	constexpr size_t MaxElements = size_t {1} << 16u;
	const auto       tick        = ++m_table_resolution_tick;
	// A shader samples several roots from the same tables. Where a root reads its table and where
	// the shader uses it do not affect how a T# resolves, so such roots share their elements.
	auto view                 = root;
	view.source               = 0;
	view.first_use_pc         = 0;
	view.table                = ShaderRecompiler::IR::ImageResource::NoImageTable;
	view.table_capacity       = 0;
	view.table_mapping_offset = 0;
	view.table_entry_mask     = 0;
	auto found = std::ranges::find_if(m_table_resolutions,
	                                  [&](const TableResolution& entry) { return entry.root == view; });
	if (found == m_table_resolutions.end()) {
		if (m_table_resolutions.size() < MaxTables) {
			found = m_table_resolutions.emplace(m_table_resolutions.end());
		} else {
			found = std::ranges::min_element(m_table_resolutions, {}, &TableResolution::last_use);
		}
		found->root = std::move(view);
		m_retired_table_elements.push_back(std::move(found->elements));
		found->elements.clear();
		found->slots.clear();
		m_table_generation++;
	} else if (found->elements.size() > MaxElements) {
		m_retired_table_elements.push_back(std::move(found->elements));
		found->elements.clear();
		found->slots.clear();
		m_table_generation++;
	}
	found->last_use = tick;
	return *found;
}

// What ResolveTexture checks of the backing for a table candidate (a GPU-modified range needs no
// backing).
bool RenderExecutor::TableElementBackingReadable(const TableElementResolution& element) {
	const auto& data = element.range;
	const bool  data_readable =
	    m_context.GetTextureCache().IsRegionGpuModified(data.address, data.size) ||
	    Libs::LibKernel::Memory::IsBackingReadable(data.address, data.size);
	return data_readable &&
	       (element.metadata_range.Empty() ||
	        Libs::LibKernel::Memory::IsBackingReadable(element.metadata_range.address,
	                                                   element.metadata_range.size));
}

RenderExecutor::TableElementResolution&
RenderExecutor::ResolveTableElement(TableResolution&                             table,
                                    const ShaderRecompiler::IR::DescriptorValue& value,
                                    uint64_t stamp, TableElementResolution* known) {
	auto&      texture_cache = m_context.GetTextureCache();
	const auto backing_epoch = Libs::LibKernel::Memory::BackingEpoch();
	bool       inserted      = false;
	if (known == nullptr) {
		const auto [entry, emplaced] = table.elements.try_emplace(value.dwords);
		known                        = &entry->second;
		inserted                     = emplaced;
	}
	auto& element = *known;
	if (!inserted && element.prepared_stamp == stamp) {
		// Resolved for an earlier slot of the same bindings.
		return element;
	}
	const auto set_epoch = texture_cache.ImageSetEpoch();
	if (!inserted && !element.volatile_dcc) {
		if (element.permanent ||
		    (element.checked_epoch == set_epoch && element.backing_epoch == backing_epoch)) {
			return element;
		}
		if (element.backing_epoch != backing_epoch && element.backing_checkable &&
		    TableElementBackingReadable(element) == element.backing_readable) {
			element.backing_epoch = backing_epoch;
		}
		const auto image_epoch =
		    element.range.Valid()
		        ? texture_cache.ImageEpochInRegion(element.range.address, element.range.size)
		        : 0;
		if (image_epoch <= element.image_epoch && element.backing_epoch == backing_epoch) {
			element.checked_epoch = set_epoch;
			return element;
		}
	}
	// Record the epoch before resolving: an image the lookup itself inserts makes the next use
	// resolve again, after which the element is stable.
	const auto prepared_stamp = element.prepared_stamp;
	const auto prepared_index = element.prepared_index;
	element                   = {};
	element.prepared_stamp    = prepared_stamp;
	element.prepared_index    = prepared_index;
	element.image_epoch       = set_epoch;
	element.checked_epoch = set_epoch;
	element.backing_epoch = backing_epoch;
	GuestRange examined {};
	element.texture  = ResolveTableTexture(table.root, value, &examined);
	const auto& data = element.texture.desc.info.data;
	if (!data.Empty()) {
		element.range        = data;
		element.volatile_dcc = element.texture.desc.info.metadata.kind == ImageMetadataKind::Dcc;
		if (element.texture.desc.info.metadata.kind != ImageMetadataKind::None) {
			element.metadata_range = element.texture.desc.info.metadata.range;
		}
		element.backing_checkable = data.Valid();
		element.backing_readable  = element.backing_checkable && TableElementBackingReadable(element);
	} else if (!ShaderRecompiler::IR::ImageTableSlotCompatible(table.root, value)) {
		element.permanent = true;
	} else {
		// Declined by the texture cache lookup, or rejected before it from the T# and the
		// backing alone (no range examined: only a backing change can alter the outcome).
		element.range = examined;
	}
	return element;
}

vk::Sampler RenderExecutor::NativeSampler(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                          uint32_t                                        index,
                                          const ShaderRecompiler::IR::DescriptorValue&    value) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler    = program.info.samplers[index];
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	}
	if (sampler.force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	// Samplers live as long as the sampler cache: remember recent ones without its lock.
	uint64_t hash = sampler.integer_border ? 0x9e3779b97f4a7c15ull : 0;
	for (const auto field: descriptor.fields) {
		hash = (hash ^ field) * 0x100000001b3ull;
	}
	auto& entry = m_sampler_memo[(hash ^ (hash >> 32u)) % m_sampler_memo.size()];
	if (entry.sampler != nullptr && entry.integer_border == sampler.integer_border &&
	    std::memcmp(entry.fields.data(), descriptor.fields, sizeof(entry.fields)) == 0) {
		return entry.sampler;
	}
	const auto native = m_context.GetSamplerCache().GetSampler(descriptor, sampler.integer_border);
	std::memcpy(entry.fields.data(), descriptor.fields, sizeof(entry.fields));
	entry.integer_border = sampler.integer_border;
	entry.sampler        = native;
	return native;
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	auto& command_buffer = context.GetCommandScheduler().Current();
	EXIT_IF(command_buffer.IsInvalid());
	auto&      buffer = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = buffer.Copy(data.data(), data.size_bytes(), 256);
	return {buffer.Handle(), offset, data.size_bytes()};
}

// Whether binding `texture` (a table element nothing else in the operation binds) leaves its
// image as it is: CommitBindings() would record no barrier for it.
static bool TableImageReady(const Image& image, const TextureBinding& texture) {
	if (image.binding.is_bound || image.binding.is_target || image.binding.force_general ||
	    texture.desc.type == TextureCache::BindingType::Storage) {
		return false;
	}
	// As the transit of a sampled element (see CommitBindings()) and Image::GetBarriers().
	const auto layout = image.info.data.Empty() ? vk::ImageLayout::eGeneral
	                    : image.info.IsDepth()  ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
	                                            : vk::ImageLayout::eShaderReadOnlyOptimal;
	if (texture.layout != layout) {
		return false;
	}
	const auto ready = [layout](const VulkanImageState& state) {
		return state.layout == layout && state.access_mask == vk::AccessFlagBits2::eShaderRead;
	};
	const auto& subresources = image.backing.subresource_states;
	const auto& view         = texture.desc.view_info;
	ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
	                             view.layer_count};
	if (image.info.IsVolume()) {
		range.base_layer  = 0;
		range.layer_count = 1;
	}
	const auto& resources = image.info.resources;
	const bool  partial   = range.base_level != 0 || range.level_count != resources.levels ||
	                     range.base_layer != 0 || range.layer_count != resources.layers;
	if (!partial && subresources.empty()) {
		return ready(image.backing.state);
	}
	if (subresources.empty()) {
		// Every subresource has the image's state.
		return ready(image.backing.state);
	}
	if (subresources.size() != size_t {resources.levels} * resources.layers) {
		return false;
	}
	for (uint32_t level = range.base_level; level < range.base_level + range.level_count; level++) {
		for (uint32_t layer = range.base_layer; layer < range.base_layer + range.layer_count;
		     layer++) {
			const auto index = size_t {level} * resources.layers + layer;
			if (index >= subresources.size() || !ready(subresources[index])) {
				return false;
			}
		}
	}
	return true;
}

void RenderExecutor::MarkTableElement(TableBindingState& state, uint32_t element) {
	if (state.marks[element] != state.mark_stamp) {
		state.marks[element] = state.mark_stamp;
		m_table_marked.push_back(element);
	}
}

void RenderExecutor::MarkTablePages(TableBindingState& state, uint64_t address, uint64_t size) {
	if (size == 0) {
		return;
	}
	constexpr uint32_t PageBits = TextureCache::ImagePageBits;
	const uint64_t     first    = address >> PageBits;
	const uint64_t     last =
	    (size > UINT64_MAX - address ? UINT64_MAX : address + size - 1) >> PageBits;
	const uint64_t from = first > state.max_span ? first - state.max_span : 0;
	auto span = std::ranges::lower_bound(state.spans, from, {}, &TableBindingState::PageSpan::first);
	for (; span != state.spans.end() && span->first <= last; ++span) {
		if (span->last >= first) {
			MarkTableElement(state, span->element);
		}
	}
}

void RenderExecutor::MarkTableChanges(TableBindingState& state) {
	const auto backing_epoch = Libs::LibKernel::Memory::BackingEpoch();
	if (backing_epoch != state.backing_epoch) {
		for (const auto element: state.unchecked_backing) {
			MarkTableElement(state, element);
		}
		m_table_changes.clear();
		if (Libs::LibKernel::Memory::BackingChangesSince(state.backing_epoch, m_table_changes)) {
			for (const auto& [address, size]: m_table_changes) {
				MarkTablePages(state, address, size);
			}
		} else {
			for (uint32_t element = 0; element < state.elements.size(); element++) {
				MarkTableElement(state, element);
			}
		}
		state.backing_epoch = backing_epoch;
	}
	auto&      texture_cache = m_context.GetTextureCache();
	const auto set_epoch     = texture_cache.ImageSetEpoch();
	if (set_epoch != state.set_epoch) {
		m_table_changes.clear();
		const bool known = texture_cache.ForEachImageSetChangeSince(
		    state.set_epoch,
		    [&](uint64_t address, uint64_t size) { m_table_changes.emplace_back(address, size); });
		if (known) {
			for (const auto& [address, size]: m_table_changes) {
				MarkTablePages(state, address, size);
			}
		} else {
			for (uint32_t element = 0; element < state.elements.size(); element++) {
				MarkTableElement(state, element);
			}
		}
		state.set_epoch = set_epoch;
	}
}

void RenderExecutor::RebuildTableSpans(TableBindingState& state) {
	constexpr uint32_t PageBits = TextureCache::ImagePageBits;
	state.spans.clear();
	state.max_span = 0;
	for (uint32_t element = 0; element < state.elements.size(); element++) {
		auto&       entry      = state.elements[element];
		const auto& resolution = *entry.resolution;
		entry.span_range       = resolution.range;
		entry.span_metadata    = resolution.metadata_range;
		for (const auto& range: {resolution.range, resolution.metadata_range}) {
			if (range.size == 0) {
				continue;
			}
			const uint64_t first = range.address >> PageBits;
			const uint64_t last  = (range.size > UINT64_MAX - range.address
			                            ? UINT64_MAX
			                            : range.address + range.size - 1) >>
			                      PageBits;
			state.spans.push_back({first, last, element});
			state.max_span = std::max(state.max_span, last - first);
		}
	}
	std::ranges::sort(state.spans, {}, &TableBindingState::PageSpan::first);
}

void RenderExecutor::UnpinTableBindings(TableBindingState& state) {
	auto& texture_cache = m_context.GetTextureCache();
	for (auto& element: state.elements) {
		if (element.pinned) {
			texture_cache.UnpinImage(element.pinned);
			element.pinned = {};
		}
	}
	state.valid = false;
}

void RenderExecutor::FinishTableBindings(TableBindingState& state,
                                         const PreparedBindings& prepared) {
	if (state.generation != m_table_generation ||
	    state.elements.size() != m_table_image_infos.size() ||
	    state.values.size() != prepared.table_slots.size()) {
		UnpinTableBindings(state);
		return;
	}
	auto& texture_cache = m_context.GetTextureCache();
	state.infos         = m_table_image_infos;
	state.slots         = prepared.table_slots;
	state.slot_infos.resize(state.slots.size());
	for (size_t slot = 0; slot < state.slots.size(); slot++) {
		state.slot_infos[slot] = state.infos[state.slots[slot]];
	}
	// The slots of each element (a counting sort by element).
	state.element_slot_begin.assign(state.elements.size() + 1, 0);
	for (const auto element: state.slots) {
		state.element_slot_begin[element + 1]++;
	}
	for (size_t element = 0; element < state.elements.size(); element++) {
		state.element_slot_begin[element + 1] += state.element_slot_begin[element];
	}
	state.element_slots.resize(state.slots.size());
	std::vector<uint32_t> next(state.element_slot_begin.begin(), state.element_slot_begin.end() - 1);
	for (size_t slot = 0; slot < state.slots.size(); slot++) {
		state.element_slots[next[state.slots[slot]]++] = static_cast<uint32_t>(slot);
	}
	state.always.clear();
	state.unchecked_backing.clear();
	for (uint32_t index = 0; index < state.elements.size(); index++) {
		auto&       element    = state.elements[index];
		const auto& resolution = *element.resolution;
		if (resolution.volatile_dcc) {
			state.always.push_back(index);
		} else if (!resolution.permanent && !resolution.backing_checkable) {
			state.unchecked_backing.push_back(index);
		}
		element.pinned = resolution.texture.image_id;
		texture_cache.PinImage(element.pinned);
	}
	RebuildTableSpans(state);
	state.marks.assign(state.elements.size(), 0);
	state.mark_stamp = 0;
	state.valid      = true;
}

bool RenderExecutor::ReuseTableBindings(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                        const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
                                        TableBindingState& state, uint64_t stamp,
                                        PreparedBindings& prepared) {
	if (!state.valid || state.generation != m_table_generation) {
		return false;
	}
	// The same tables with the same T# in every slot.
	const ShaderRecompiler::IR::DescriptorValue null_value {.dword_count = 8u};
	size_t                                      range_index = 0;
	size_t                                      slot_index  = 0;
	for (const auto& binding: program.bindings.descriptors) {
		for (const auto& range: binding.tables) {
			if (range_index == state.ranges.size()) {
				return false;
			}
			const auto& cached = state.ranges[range_index++];
			const auto& root   = program.info.images.at(range.root);
			if (cached.table != range.table || cached.capacity != range.capacity ||
			    !(cached.root == root) || &FindTableResolution(root) != cached.resolution ||
			    state.generation != m_table_generation) {
				return false;
			}
			const auto& table = snapshot.image_tables.at(range.table);
			for (uint32_t slot = 0; slot < range.capacity; slot++) {
				const auto& value =
				    slot != 0u && slot < table.slots.size() ? table.slots[slot] : null_value;
				if (!(state.values[slot_index++] == value)) {
					return false;
				}
			}
		}
	}
	if (range_index != state.ranges.size()) {
		return false;
	}
	state.mark_stamp++;
	m_table_marked.clear();
	for (const auto element: state.always) {
		MarkTableElement(state, element);
	}
	MarkTableChanges(state);
	for (const auto element: m_table_marked) {
		auto& entry = state.elements[element];
		(void)ResolveTableElement(*entry.table, entry.source, stamp, entry.resolution);
	}
	prepared.table_visits = m_table_marked;
	return true;
}

void RenderExecutor::RebindReusedTables(PreparedBindings& prepared) {
	auto&       state         = *static_cast<TableBindingState*>(prepared.table_state);
	const auto& program       = *prepared.runtime->program;
	auto&       texture_cache = m_context.GetTextureCache();
	const auto  tick          = m_context.GetCommandScheduler().CurrentTick();
	const auto  stale         = [&](const TextureBinding& binding) {
        const auto* image = texture_cache.m_slot_images.try_get(binding.image_id);
        return image == nullptr || (!image->registered && !image->info.data.Empty()) ||
               image->binding.needs_rebind;
	};
	// The elements PrepareBindings() marked, and those whose image is no longer bound as the
	// state records: removed, replaced, dirty, with another view or in another layout.
	const auto unchanged = [&](uint32_t index) {
		const auto& element = state.elements[index];
		const auto& texture = element.resolution->texture;
		const auto& view    = element.resolution->view;
		const auto* image   = texture_cache.m_slot_images.try_get(texture.image_id);
		// Null images are acquired again every time (their view stays the same).
		return image != nullptr && !image->binding.needs_rebind &&
		       texture.image_id == element.pinned && texture.image_view != nullptr &&
		       texture.image_view == state.infos[index].imageView &&
		       texture.layout == state.infos[index].imageLayout &&
		       (image->info.data.Empty() ||
		        (image->registered && view.valid &&
		         ImageLookupState::Of(texture.image_id, *image) == view.state)) &&
		       TableImageReady(*image, texture);
	};
	// The check only reads the elements and their images (thousands, scattered in memory):
	// helper threads take part, each flagging the elements of its chunks.
	const auto element_count = static_cast<uint32_t>(state.elements.size());
	m_table_flags.assign(element_count, 0);
	GetExecHelpers().ParallelFor(element_count, 256, [&](uint32_t begin, uint32_t end) {
		for (uint32_t index = begin; index < end; index++) {
			if (state.marks[index] != state.mark_stamp && !unchanged(index)) {
				m_table_flags[index] = 1;
			}
		}
	});
	for (uint32_t index = 0; index < element_count; index++) {
		if (m_table_flags[index] != 0) {
			MarkTableElement(state, index);
		}
	}
	size_t visited = 0;
	for (;;) {
		for (; visited < m_table_marked.size(); visited++) {
			auto& element = state.elements[m_table_marked[visited]];
			auto& texture = element.resolution->texture;
			auto& view    = element.resolution->view;
			// As RebindImages() does for every element of resolved tables.
			if (stale(texture)) {
				if (auto* old_image = texture_cache.m_slot_images.try_get(texture.image_id)) {
					old_image->binding = {};
				}
				texture = ResolveTableTexture(program.info.images.at(element.root), element.source);
				view.valid = false;
			}
			BindImage(texture.image_id, false);
			if (auto* image = texture_cache.m_slot_images.try_get(texture.image_id);
			    view.valid && image != nullptr && texture.image_view &&
			    ImageLookupState::Of(texture.image_id, *image) == view.state) {
				image->tick_accessed_last = tick;
				texture_cache.TouchImage(*image);
			} else {
				texture.image_view  = texture_cache.FindTexture(texture.image_id, texture.desc);
				auto& acquired         = texture_cache.GetImage(texture.image_id);
				acquired.usage.texture = true;
				view.state             = ImageLookupState::Of(texture.image_id, acquired);
				view.valid = !view.state.cpu_dirty && !view.state.buffer_modified &&
				             !acquired.info.data.Empty();
			}
			if (texture.image_id != element.pinned) {
				texture_cache.UnpinImage(element.pinned);
				element.pinned = texture.image_id;
				texture_cache.PinImage(element.pinned);
			}
		}
		// The lookups above may have registered images over other elements' ranges.
		MarkTableChanges(state);
		if (visited == m_table_marked.size()) {
			break;
		}
		for (auto index = visited; index < m_table_marked.size(); index++) {
			auto& element = state.elements[m_table_marked[index]];
			(void)ResolveTableElement(*element.table, element.source, m_table_prepare_stamp,
			                          element.resolution);
		}
	}
	// Elements whose range changed are found by their new range from now on.
	for (const auto index: m_table_marked) {
		const auto& element    = state.elements[index];
		const auto& resolution = *element.resolution;
		if (resolution.range.address != element.span_range.address ||
		    resolution.range.size != element.span_range.size ||
		    resolution.metadata_range.address != element.span_metadata.address ||
		    resolution.metadata_range.size != element.span_metadata.size) {
			RebuildTableSpans(state);
			break;
		}
	}
	prepared.table_visits = m_table_marked;
}

void RenderExecutor::FlushDeferredDispatchBarrier() {
	m_dispatch_chained = false;
	if (!m_deferred_barrier) {
		return;
	}
	m_deferred_barrier = false;
	ShaderAccessBarrier(m_context.GetCommandScheduler().Current().Handle(),
	                    vk::PipelineStageFlagBits::eComputeShader);
}

void RenderExecutor::BeginGuestOperation(uint32_t queue, bool dispatch_direct, bool guest_sync) {
	m_guest_queue = queue;
	if (m_deferred_barrier && dispatch_direct && !guest_sync && queue == m_deferred_barrier_queue) {
		m_dispatch_chained = true;
		return;
	}
	FlushDeferredDispatchBarrier();
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.info.data.Empty()) {
		return;
	}
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	m_bound_images.push_back(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image             = m_context.GetTextureCache().GetImage(id);
	image.binding.is_target = true;
	m_bound_images.push_back(id);
}

void RenderExecutor::ResetBindings() {
	m_retired_table_elements.clear();
	for (const auto id: m_bound_images) {
		if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr) {
			image->binding = {};
		}
	}
	m_bound_images.clear();
	if (m_clear_texture_descs) {
		m_clear_texture_descs = false;
		m_texture_descs.clear();
		m_texture_descs_generation++;
	}
}

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(!runtime);
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.shader_data_buffer = {};
	prepared.shared_memory = {};
	prepared.images.resize(program.info.images.size());
	prepared.shader_data.clear();
	const bool same_program = prepared.last_program == &program &&
	                          prepared.last_descs_generation == m_texture_descs_generation;
	const bool same_images  = same_program && prepared.last_images == snapshot.images;
	if (!same_images) {
		prepared.image_descs.assign(program.info.images.size(), nullptr);
	}
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		// Sampled textures whose last lookup still holds are written in place.
		const auto& resource = program.info.images[i];
		if (!resource.written &&
		    resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			auto* derived = prepared.image_descs[i];
			if (!same_images) {
				const auto cached =
				    m_texture_descs.find(MakeTextureDescKey(resource, snapshot.images[i]));
				derived = cached != m_texture_descs.end() ? &cached->second : nullptr;
			}
			if (derived != nullptr && derived->found.valid &&
			    ReuseTextureLookup(*derived, prepared.images[i])) {
				prepared.image_descs[i] = derived;
				BindImage(prepared.images[i].image_id, false);
				continue;
			}
		}
		auto binding = ResolveTexture(program.info.images[i], snapshot.images[i]);
		// The description whose lookups later draws reuse (see ResolveTexture()).
		prepared.image_descs[i] = binding.lookup;
		BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
		binding.mip_views.swap(prepared.images[i].mip_views);
		binding.mip_views.clear();
		prepared.images[i] = std::move(binding);
	}
	if (!same_images) {
		prepared.last_images = snapshot.images;
	}
	prepared.table_images.clear();
	prepared.table_views.clear();
	prepared.table_sources.clear();
	prepared.table_roots.clear();
	prepared.table_slots.clear();
	prepared.table_state   = nullptr;
	prepared.tables_reused = false;
	prepared.table_visits.clear();
	const auto stamp = ++m_table_prepare_stamp;
	TableBindingState* table_state = nullptr;
	if (std::ranges::any_of(program.bindings.descriptors,
	                        [](const auto& binding) { return !binding.tables.empty(); })) {
		table_state           = &m_table_states[&program];
		prepared.table_state  = table_state;
		prepared.tables_reused =
		    ReuseTableBindings(program, snapshot, *table_state, stamp, prepared);
		if (!prepared.tables_reused) {
			// Resolved slot by slot below; the state records what this binding finds.
			UnpinTableBindings(*table_state);
			table_state->valid         = false;
			table_state->generation    = m_table_generation;
			table_state->set_epoch     = m_context.GetTextureCache().ImageSetEpoch();
			table_state->backing_epoch = Libs::LibKernel::Memory::BackingEpoch();
			table_state->ranges.clear();
			table_state->values.clear();
			table_state->elements.clear();
		}
	}
	for (const auto& binding: program.bindings.descriptors) {
		if (prepared.tables_reused) {
			break;
		}
		for (const auto& range: binding.tables) {
			const auto& root  = program.info.images.at(range.root);
			const auto& table = snapshot.image_tables.at(range.table);
			const ShaderRecompiler::IR::DescriptorValue null_value {.dword_count = 8u};
			auto& resolution = FindTableResolution(root);
			table_state->ranges.push_back({root, &resolution, range.table, range.capacity});
			// Bind each distinct element once, however many slots name it.
			const auto add = [&](TableElementResolution&                      element,
			                     const ShaderRecompiler::IR::DescriptorValue& value) {
				if (element.prepared_stamp != stamp) {
					element.prepared_stamp = stamp;
					element.prepared_index = static_cast<uint32_t>(prepared.table_images.size());
					BindImage(element.texture.image_id, false);
					prepared.table_images.push_back(&element.texture);
					prepared.table_views.push_back(&element.view);
					prepared.table_sources.push_back(value);
					prepared.table_roots.push_back(range.root);
					table_state->elements.push_back(
					    {.resolution = &element, .table = &resolution, .source = value,
					     .root = range.root});
				}
				prepared.table_slots.push_back(element.prepared_index);
			};
			if (resolution.slots.size() < range.capacity) {
				resolution.slots.resize(range.capacity);
			}
			for (uint32_t slot = 0; slot < range.capacity; slot++) {
				const auto& value =
				    slot != 0u && slot < table.slots.size() ? table.slots[slot] : null_value;
				table_state->values.push_back(value);
				auto& memo = resolution.slots[slot];
				if (memo.element != nullptr && memo.dwords == value.dwords) {
					add(ResolveTableElement(resolution, value, stamp, memo.element), value);
				} else {
					auto& element = ResolveTableElement(resolution, value, stamp);
					memo          = {value.dwords, &element};
					add(element, value);
				}
			}
		}
	}
	// Samplers live as long as the sampler cache: the same S#s of the same program give the same
	// samplers.
	if (!same_program || prepared.samplers.size() != program.info.samplers.size() ||
	    prepared.last_samplers != snapshot.samplers) {
		prepared.samplers.clear();
		prepared.samplers.reserve(program.info.samplers.size());
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			prepared.samplers.push_back(NativeSampler(
			    program, i, snapshot.samplers[program.info.samplers[i].snapshot_index]));
		}
		prepared.last_samplers = snapshot.samplers;
	}
	prepared.last_program          = &program;
	prepared.last_descs_generation = m_texture_descs_generation;
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.bindings.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
}

void RenderExecutor::FindBuffers(std::span<PreparedBindings* const> stages, bool find_buffers) {
	KYTY_PROFILER_FUNCTION();
	auto& cache = m_context.GetBufferCache();
	for (auto* stage: stages) {
		auto& prepared = *stage;
		EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
		const auto& program  = *prepared.runtime->program;
		const auto& snapshot = *prepared.runtime->resources;
		prepared.buffer_sources.clear();
		const auto& layout = program.bindings;
		if (layout.memory_offset_count == 0) {
			continue;
		}
		const auto& resources = layout.descriptors.front().resources;
		prepared.buffer_sources.reserve(resources.size());
		for (const auto resource: resources) {
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[resource]);
			const auto address = descriptor.Base48();
			auto size = descriptor.GetSize();
			if (address == 0 || size == 0) {
				prepared.buffer_sources.push_back({});
				continue;
			}
			if (descriptor.NumRecords() == UINT32_MAX) {
				if (program.info.buffers[resource].written) {
					// Sentinel write ranges end before tables captured by any bound stage.
					for (const auto* reader: stages) {
						for (const auto [read_address, read_size]: reader->runtime->resources->specialization_reads) {
							if (read_size == 0) continue;
							if (read_address > address) {
								size = std::min(size, read_address - address);
							} else if (address - read_address < read_size) {
								EXIT("scalar resource reads overlap a shader buffer write\n");
							}
						}
					}
				} else {
					const auto& graphics = m_context.GetGraphics();
					const auto limit = uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} -
					                   (graphics.StorageMinAlignment() - 1);
					size = std::min({size, uint64_t {256} * 1024 * 1024, limit});
				}
			}
			size = Libs::LibKernel::Memory::ClampRangeSize(address, size);
			// Without device-address reads, ObtainBuffer() finds the buffer when it needs one
			// (draws stream most small buffers from CPU-dirty pages without it).
			prepared.buffer_sources.push_back(
			    {address, size, find_buffers ? cache.FindBuffer(address, size) : NULL_BUFFER_ID});
		}
	}
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program   = *prepared.runtime->program;
	const auto& snapshot  = *prepared.runtime->resources;
	const auto& layout    = program.bindings;
	EXIT_IF(prepared.buffer_sources.size() != layout.memory_offset_count);

	prepared.buffers.clear();
	prepared.buffers.reserve(layout.memory_offset_count);
	EXIT_IF(prepared.shader_data.size() != layout.ShaderDataDwords());
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
	          prepared.shader_data.end(), 0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	for (uint32_t i = 0; i < layout.memory_offset_count; i++) {
		const auto resource = layout.descriptors.front().resources[i];
		uint32_t buffer_offset = 0;
		prepared.buffers.push_back(NativeStorageBuffer(m_context, prepared.buffer_sources[i],
		                                               program.info.buffers[resource],
		                                               buffer_offset));
		pack_memory_offset(i, buffer_offset);
	}
	if (ShaderRecompiler::IR::FindBinding(
	        layout, ShaderRecompiler::IR::DescriptorBindingKind::FlattenedSrt) != nullptr) {
		prepared.flattened_srt = NativeUpload(m_context, snapshot.flattened_srt);
	}
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::ShaderData) != nullptr) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       images   = prepared.images;
	EXIT_IF(images.size() != program.info.images.size());
	auto& texture_cache = m_context.GetTextureCache();
	const auto stale = [&](const TextureBinding& binding) {
		const auto* image = texture_cache.m_slot_images.try_get(binding.image_id);
		return image == nullptr || (!image->registered && !image->info.data.Empty()) ||
		       image->binding.needs_rebind;
	};
	// Table elements may share images with each other and with the direct images: decide which
	// are stale before any rebind clears an image's binding state.
	std::vector<bool> stale_tables(prepared.table_images.size());
	for (size_t k = 0; k < prepared.table_images.size(); k++) {
		stale_tables[k] = stale(*prepared.table_images[k]);
	}
	// Acquire each view before a later overlapping descriptor can replace its image.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto old_image = texture_cache.m_slot_images.try_get(images[i].image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			images[i] = ResolveTexture(program.info.images[i], snapshot.images[i]);
			BindImage(images[i].image_id,
			          images[i].desc.type == TextureCache::BindingType::Storage);
		}
		auto& binding = images[i];
		binding.mip_views.clear();
		const auto& resource = program.info.images[i];
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			EXIT_IF(resource.mip_count == 0u ||
			        resource.mip_count != binding.desc.view_info.level_count);
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += mip;
				desc.view_info.level_count = 1;
				// The shader selects the mip after applying the guest minimum LOD.
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else if (!binding.reused) {
			binding.image_view = texture_cache.FindTexture(binding.image_id, binding.desc);
		}
		auto&      image   = texture_cache.GetImage(binding.image_id);
		const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
		image.usage.storage |= storage;
		image.usage.texture |= !storage;
		if (binding.lookup != nullptr && !binding.reused) {
			auto& found           = binding.lookup->found;
			found.id              = binding.image_id;
			found.desc            = binding.desc;
			found.view            = binding.image_view;
			found.image_set_epoch = texture_cache.ImageSetEpoch();
			found.state           = ImageLookupState::Of(binding.image_id, image);
			found.valid           = true;
		}
		// The binding no longer refers to the description, which may be cleared.
		binding.lookup = nullptr;
		binding.reused = false;
	}
	const auto tick = m_context.GetCommandScheduler().CurrentTick();
	for (size_t k = 0; k < prepared.table_images.size(); k++) {
		auto& texture = *prepared.table_images[k];
		auto& view    = *prepared.table_views[k];
		// A rediscovery above may have replaced (expanded, merged or recreated) the image this
		// element resolved to.
		if (stale_tables[k] || stale(texture)) {
			if (auto* old_image = texture_cache.m_slot_images.try_get(texture.image_id)) {
				old_image->binding = {};
			}
			texture = ResolveTableTexture(program.info.images.at(prepared.table_roots[k]),
			                              prepared.table_sources[k]);
			BindImage(texture.image_id, false);
			view.valid = false;
		}
		// Most elements are textures nothing touched since their last acquisition: keep the
		// view and refresh the image's use, as the acquisition would. Guest-dirty images are
		// refreshed by the acquisition.
		if (auto* image = texture_cache.m_slot_images.try_get(texture.image_id);
		    view.valid && image != nullptr && texture.image_view &&
		    ImageLookupState::Of(texture.image_id, *image) == view.state) {
			image->tick_accessed_last = tick;
			texture_cache.TouchImage(*image);
			continue;
		}
		texture.image_view = texture_cache.FindTexture(texture.image_id, texture.desc);
		auto& image        = texture_cache.GetImage(texture.image_id);
		image.usage.texture = true;
		view.state = ImageLookupState::Of(texture.image_id, image);
		view.valid = !view.state.cpu_dirty && !view.state.buffer_modified &&
		             !image.info.data.Empty();
	}
	if (prepared.tables_reused) {
		RebindReusedTables(prepared);
	}
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors) {
	bool                                     dma_write = false;
	std::array<const ShaderStageRuntime*, 4> dma_stages {};
	uint32_t                                 dma_stage_count = 0;
	// Operations with device-address reads need their buffers before the BDA synchronization.
	FindBuffers(stages, std::ranges::any_of(stages, [](const auto* stage) {
		            return stage->runtime->program->info.uses_dma;
	            }));
	for (auto* stage: stages) {
		if (stage->runtime->program->info.uses_dma) {
			m_context.CacheDmaBases(*stage->runtime);
			dma_write |= stage->runtime->program->has_address_writes;
			EXIT_IF(dma_stage_count == dma_stages.size());
			dma_stages[dma_stage_count++] = stage->runtime;
		}
	}
	if (dma_stage_count != 0) {
		// Only operations whose device-address reach is unknown upload all cached memory.
		const bool reach_synchronized =
		    m_context.SynchronizeDmaFootprint(std::span {dma_stages.data(), dma_stage_count});
		m_context.PrepareBda(dma_write, !reach_synchronized);
	}
	for (auto* stage: stages) {
		RebindImages(*stage);
	}
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		const auto old_image = cache.m_slot_images.try_get(target.image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			// A draw that reuses the targets of the draw before updates them in the render
			// target memo, which no longer describes them.
			InvalidateRenderTargetMemo();
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	for (auto* stage: stages) {
		RebindBuffers(*stage);
	}
}

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	KYTY_PROFILER_FUNCTION();
	auto   vk_buffer        = buffer.Handle();
	size_t descriptor_count = 0;
	size_t write_count      = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	constexpr auto                 GraphicsStages =
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eMeshEXT |
	    vk::ShaderStageFlagBits::eTessellationControl |
	    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment;
	vk::ShaderStageFlags push_stages = pipeline_bind_point == vk::PipelineBindPoint::eGraphics
	                                       ? vk::ShaderStageFlagBits::eFragment
	                                       : vk::ShaderStageFlags {};
	for (const auto* prepared: prepared_bindings) {
		EXIT_IF(prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime);
		const auto& program = *prepared->runtime->program;
		write_count += program.bindings.descriptors.size();
		for (const auto& binding: program.bindings.descriptors) {
			descriptor_count += NativeDescriptorCount(binding);
		}
		const auto shader_stage = NativeShaderStage(program.stage);
		push_stages |= shader_stage;
		EXIT_IF((pipeline_bind_point == vk::PipelineBindPoint::eGraphics &&
		         (shader_stage & GraphicsStages) == vk::ShaderStageFlags {}) ||
		        (pipeline_bind_point == vk::PipelineBindPoint::eCompute &&
		         shader_stage != vk::ShaderStageFlagBits::eCompute));
	}
	// The image, attachment and buffer ranges the draw writes, collected once for the scalar read
	// checks below (most bound images are only sampled, and draws rarely write buffers).
	bool written_ranges_collected = false;
	for (const auto* reader: prepared_bindings) {
		const auto& reads = reader->runtime->resources->specialization_reads;
		if (reads.empty()) continue;
		if (!written_ranges_collected) {
			written_ranges_collected = true;
			for (const auto* writer: prepared_bindings) {
				if (writer->runtime->program->has_address_writes) {
					EXIT("scalar resource reads cannot be proven disjoint from shader address writes\n");
				}
			}
			m_written_image_ranges.clear();
			for (const auto id: m_bound_images) {
				const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
				if (image == nullptr ||
				    (!image->binding.shader_write && !image->binding.is_target)) continue;
				for (const auto written: {image->info.data, image->info.stencil,
				                          image->info.metadata.range}) {
					if (written.size != 0) {
						m_written_image_ranges.emplace_back(written.address, written.size);
					}
				}
			}
			m_written_buffer_ranges.clear();
			for (const auto* writer: prepared_bindings) {
				const auto& program = *writer->runtime->program;
				for (uint32_t i = 0; i < writer->buffer_sources.size(); ++i) {
					const auto  resource = program.bindings.descriptors.front().resources[i];
					const auto& written  = writer->buffer_sources[i];
					if (program.info.buffers[resource].written && written.size != 0) {
						m_written_buffer_ranges.push_back(
						    {written.address, written.size, writer, resource});
					}
				}
			}
		}
		for (size_t read = 0; read < reads.size(); ++read) {
			const auto [address, size] = reads[read];
			for (const auto& [written_address, written_size]: m_written_image_ranges) {
				if (ImageRangeOverlaps(address, size, written_address, written_size)) {
					EXIT("scalar resource reads overlap an image or attachment write\n");
				}
			}
			for (const auto& written: m_written_buffer_ranges) {
				// A flattened scalar load issued before the shader's own writes to the buffer may
				// observe its dispatch-time snapshot on hardware.
				if (ImageRangeOverlaps(address, size, written.address, written.size) &&
				    (written.writer != reader ||
				     ShaderRecompiler::IR::SpecializationReadFollowsBufferWrite(
				         *reader->runtime->program, *reader->runtime->resources, read,
				         written.resource))) {
					EXIT("scalar resource reads overlap a shader buffer write\n");
				}
			}
		}
	}
	m_descriptor_buffers.clear();
	m_descriptor_images.clear();
	m_descriptor_writes.clear();
	m_descriptor_buffers.reserve(descriptor_count);
	m_descriptor_images.reserve(descriptor_count);
	m_descriptor_writes.reserve(write_count);

	for (auto* prepared: prepared_bindings) {
		const auto& program       = *prepared->runtime->program;
		auto&       descriptors   = *prepared;
		const auto  shader_stage  = NativeShaderStage(program.stage);
		const auto  shader_stages = ShaderPipelineStages(shader_stage);
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier = MakeGdsDependency(descriptors.gds.buffer);
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		const auto transit = [&](TextureBinding& binding) {
			auto& image = m_context.GetTextureCache().GetImage(binding.image_id);
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.info.data.Empty()) {
				image.Transit(vk::ImageLayout::eGeneral,
				              storage ? vk::AccessFlagBits2::eShaderRead |
				                            vk::AccessFlagBits2::eShaderWrite
				                      : vk::AccessFlagBits2::eShaderRead,
				              range, vk_buffer);
			} else if (image.binding.is_target) {
				const auto layout = image.binding.attachment_layout;
				EXIT_IF(layout == vk::ImageLayout::eUndefined);
				if (image.info.IsDepth()) {
					const auto host_view =
					    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
					EXIT_IF(storage || host_view == image.views.end());
					const auto aspect = host_view->info.aspect;
					if (aspect & ~DepthReadableAspects(layout)) {
						EXIT("sampling a writable depth/stencil attachment aspect\n");
					}
				}
				image.Transit(layout,
				              image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead |
				                  (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                              : vk::AccessFlags2 {}),
				              {}, vk_buffer);
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | storage_access, {}, vk_buffer);
			} else if (storage) {
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
				              range, vk_buffer);
			} else {
				image.Transit(image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                   : vk::ImageLayout::eShaderReadOnlyOptimal,
				              vk::AccessFlagBits2::eShaderRead, range, vk_buffer);
			}
			binding.layout = image.backing.state.layout;
		};
		for (auto& binding: descriptors.images) {
			transit(binding);
		}
		for (auto* binding: descriptors.table_images) {
			transit(*binding);
		}
		auto* table_state = static_cast<TableBindingState*>(descriptors.table_state);
		if (descriptors.tables_reused) {
			for (const auto index: descriptors.table_visits) {
				transit(table_state->elements[index].resolution->texture);
			}
		}

		m_image_occurrences.assign(descriptors.images.size(), 0);
		size_t table_element       = 0;
		bool   table_infos_updated = false;
		m_table_image_infos.clear();
		for (const auto& binding: program.bindings.descriptors) {
			vk::WriteDescriptorSet write {};
			write.dstBinding     = ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind);
			write.descriptorType = NativeDescriptorType(binding.kind);
			write.descriptorCount   = NativeDescriptorCount(binding);
			const auto buffer_start = m_descriptor_buffers.size();
			const auto image_start  = m_descriptor_images.size();
			if (ShaderRecompiler::IR::ImageBindingResourceClass(binding.kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				for (const auto resource: binding.resources) {
					m_descriptor_images.push_back(MakeImageInfo(
					    descriptors.images.at(resource), m_image_occurrences.at(resource)++));
				}
				if (descriptors.tables_reused) {
					if (!binding.tables.empty() && !table_infos_updated) {
						table_infos_updated = true;
						for (const auto index: descriptors.table_visits) {
							table_state->infos[index] =
							    MakeImageInfo(table_state->elements[index].resolution->texture);
						}
						// Only the slots of the visited elements change.
						for (const auto index: descriptors.table_visits) {
							for (auto k = table_state->element_slot_begin[index];
							     k < table_state->element_slot_begin[index + 1]; k++) {
								table_state->slot_infos[table_state->element_slots[k]] =
								    table_state->infos[index];
							}
						}
					}
					for (const auto& range: binding.tables) {
						EXIT_IF(table_element + range.capacity > table_state->slot_infos.size());
						const auto first = table_state->slot_infos.begin() +
						                   static_cast<std::ptrdiff_t>(table_element);
						m_descriptor_images.insert(m_descriptor_images.end(), first,
						                           first + range.capacity);
						table_element += range.capacity;
					}
				} else {
					if (!binding.tables.empty() && m_table_image_infos.empty()) {
						m_table_image_infos.reserve(descriptors.table_images.size());
						for (const auto* texture: descriptors.table_images) {
							m_table_image_infos.push_back(MakeImageInfo(*texture));
						}
					}
					for (const auto& range: binding.tables) {
						EXIT_IF(table_element + range.capacity > descriptors.table_slots.size());
						for (uint32_t slot = 0; slot < range.capacity; slot++) {
							m_descriptor_images.push_back(
							    m_table_image_infos[descriptors.table_slots[table_element++]]);
						}
					}
				}
			} else {
				switch (binding.kind) {
					case BindingKind::Buffers:
						EXIT_IF(descriptors.buffers.size() != binding.resources.size());
						for (const auto& view: descriptors.buffers) {
							EXIT_IF(view.buffer == nullptr);
							m_descriptor_buffers.push_back(view);
						}
						break;
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer: {
						auto&       cache      = m_context.GetBufferCache();
						const auto* bda_buffer = binding.kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                             : cache.GetFaultBuffer();
						m_descriptor_buffers.emplace_back(bda_buffer->Handle(), 0,
						                                  bda_buffer->Size());
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (binding.kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (binding.kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						} else if (binding.kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						}
						EXIT_IF(view->buffer == nullptr);
						m_descriptor_buffers.push_back(*view);
						break;
					}
					case BindingKind::Samplers:
						for (const auto resource: binding.resources) {
							const auto sampler = descriptors.samplers.at(resource);
							EXIT_IF(sampler == nullptr);
							m_descriptor_images.emplace_back(sampler, nullptr,
							                                 vk::ImageLayout::eUndefined);
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (m_descriptor_buffers.size() != buffer_start) {
				write.pBufferInfo = m_descriptor_buffers.data() + buffer_start;
			}
			if (m_descriptor_images.size() != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		if (table_state != nullptr && !descriptors.tables_reused) {
			FinishTableBindings(*table_state, descriptors);
		}
		for (uint32_t i = 0; i < descriptors.images.size(); i++) {
			const auto expected =
			    descriptors.images[i].mip_views.empty()
			        ? 1u
			        : static_cast<uint32_t>(descriptors.images[i].mip_views.size());
			EXIT_IF(m_image_occurrences[i] != expected);
		}

		const auto shader_data_dwords = program.bindings.ShaderDataDwords();
		EXIT_IF(prepared->shader_data.size() != shader_data_dwords);
		if (program.bindings.UsesPushData()) {
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
	}

	if (has_push_data) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, push_stages, 0, sizeof(push_data),
		                        push_data.dwords.data());
	}

	if (!m_descriptor_writes.empty()) {
		EXIT_IF(pipeline.descriptor_set_layout == nullptr);
		if (pipeline.uses_push_descriptors) {
			vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, 0,
			                               static_cast<uint32_t>(m_descriptor_writes.size()),
			                               m_descriptor_writes.data());
		} else {
			const auto set = m_context.GetDescriptorHeap().Commit(pipeline.descriptor_set_layout);
			for (auto& write: m_descriptor_writes) {
				write.dstSet = set;
			}
			m_context.GetGraphics().device.updateDescriptorSets(
			    static_cast<uint32_t>(m_descriptor_writes.size()), m_descriptor_writes.data(), 0,
			    nullptr);
			vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 0, 1, &set,
			                             0, nullptr);
		}
	}
}

} // namespace Libs::Graphics
