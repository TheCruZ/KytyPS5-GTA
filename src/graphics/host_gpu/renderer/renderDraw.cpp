#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/hackNotice.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/spscQueue.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/barrierSink.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/dispatchRun.h"
#include "graphics/host_gpu/renderer/indirectDispatch.h"
#include "graphics/host_gpu/renderer/gpuProfiler.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <unordered_set>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.fetch_embedded) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = vs_input_info.stage.module->info;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
	    static_cast<int>(!depth.desc.info.data.Empty()) +
	        static_cast<int>(depth.desc.info.HasStencil()),
	    ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.module->info.info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
					const auto& r  = vs_input_info.resources[ai];
					const auto& rd = vs_input_info.resources_dst[ai];
					if (rd.buffer_index != bi) {
						continue;
					}
					const auto offset = static_cast<uint32_t>(r.Base48() - b.addr);
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
			const auto& r  = vs_input_info.resources[ai];
			const auto& rd = vs_input_info.resources_dst[ai];
			if (rd.buffer_index != bi) {
				continue;
			}
			LOGF("DrawInputState[%u]: attr[%d] offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, static_cast<uint32_t>(r.Base48() - b.addr), rd.register_start,
			     rd.registers_num, rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

static void UpdateGraphicsDynamicState(const CommandBuffer& buffer, const CommandRecorder& vk_buffer,
                                       DynamicState& dynamic_state,
                                       const ShaderVertexInputInfo& vs_input_info,
                                       const RenderDepthInfo& depth, const RenderState& rendering) {
	KYTY_PROFILER_FUNCTION();

	const auto& ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto& outputs = vs_input_info.stage.module->info.info.outputs;
	const bool  indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
	}
	dynamic_state.SetViewports(std::span {viewports}.first(viewport_count));
	dynamic_state.SetScissors(std::span {scissors}.first(viewport_count));

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	dynamic_state.SetLineWidth(line_width);
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	dynamic_state.SetBlendConstants(blend_constants);
	dynamic_state.SetDepthTestEnabled(depth.depth_test_enable);
	dynamic_state.SetDepthWriteEnabled(depth.depth_write_enable);
	dynamic_state.SetDepthCompareOp(depth.depth_compare_op);
#if !defined(__APPLE__)
	dynamic_state.SetDepthBoundsTestEnabled(depth.depth_bounds_test_enable);
	if (depth.depth_bounds_test_enable) {
		dynamic_state.SetDepthBounds(ctx.GetDepthBoundsMin(), ctx.GetDepthBoundsMax());
	}
#endif

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	dynamic_state.SetDepthBiasEnabled(depth_bias_enable);
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		dynamic_state.SetDepthBias(constant_factor, poly_offset.clamp, slope_factor);
	}

	// Rect lists are drawn without culling.
	vk::CullModeFlags cull_mode {};
	if (!Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		if (mode.cull_front) {
			cull_mode |= vk::CullModeFlagBits::eFront;
		}
		if (mode.cull_back) {
			cull_mode |= vk::CullModeFlagBits::eBack;
		}
	}
	dynamic_state.SetCullMode(cull_mode);
	dynamic_state.SetFrontFace(mode.face ? vk::FrontFace::eClockwise
	                                     : vk::FrontFace::eCounterClockwise);

	dynamic_state.SetStencilTestEnabled(depth.stencil_test_enable);
	if (depth.stencil_test_enable) {
		const auto& front = depth.stencil_front;
		const auto& back  = depth.stencil_back;
		dynamic_state.SetStencilOps({front.failOp, front.passOp, front.depthFailOp, front.compareOp},
		                            {back.failOp, back.passOp, back.depthFailOp, back.compareOp});
		dynamic_state.SetStencilCompareMasks(front.compareMask, back.compareMask);
		dynamic_state.SetStencilWriteMasks(front.writeMask, back.writeMask);
		dynamic_state.SetStencilReferences(front.reference, back.reference);
	}

	if (PipelineDynamicRasterStateEnabled()) {
		// Formerly pipeline-key fields (PipelineCache::TryGetGraphicsPipeline), from the same
		// registers and with the same rule: rect lists are drawn without culling.
		const bool        rect_list = Prospero::IsRectList(buffer.GetUserConfig().GetPrimType());
		vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
		if (!rect_list && mode.cull_back) {
			cull_mode |= vk::CullModeFlagBits::eBack;
		}
		if (!rect_list && mode.cull_front) {
			cull_mode |= vk::CullModeFlagBits::eFront;
		}
		vk_buffer.setCullMode(cull_mode);
		vk_buffer.setFrontFace(mode.face ? vk::FrontFace::eClockwise
		                                 : vk::FrontFace::eCounterClockwise);
	}

#if defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
#else
	vk::Bool32 enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
		enable[slot] = rendering.color_attachments[slot].image_view != nullptr;
	}
	dynamic_state.SetColorWriteEnables(std::span {enable}.first(rendering.num_color_attachments));
#endif
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

struct RenderExecutor::DrawRenderStorage {
	DrawRenderState state {};
	// Only tessellation draws write the input infos of the second and third vertex stages.
	bool tessellation_stages_written = false;
};
static_assert(std::is_trivially_destructible_v<DrawRenderState>);

template <typename T>
static void ValueInitialize(T& object) {
	::new (static_cast<void*>(std::addressof(object))) T {};
}

// A draw state is 36 KB, 30 KB of it the input infos of the three vertex stages, and
// value-initializing one on the stack took ~0.6 us per draw. Reuse one state and restore its
// value-initialized contents, skipping the tessellation stages when no draw wrote them.
// A resolution ahead of execution (see ResolveDrawAhead): the draw state with its shader
// programs, the storage their stages point to, and whether the execution thread still uses it.
struct RenderExecutor::ResolvedDraw {
	DrawRenderStorage storage;
	ProgramResolution programs;
	// A dispatch resolved ahead: its module and stage input (pointing into `programs`).
	ShaderComputeInputInfo compute;
	const ShaderModule*    compute_module = nullptr;
	std::atomic_bool       busy {false};
};

// More resolutions than operations can wait between the resolve and execution threads, so the
// resolve thread only ever waits for room in that queue.
struct RenderExecutor::ResolvedDrawRing {
	static constexpr size_t Size = 320;

	std::array<ResolvedDraw, Size> draws;
	size_t                         next = 0;
};

void RenderExecutor::ReleaseResolvedDraw(ResolvedDraw* resolved) noexcept {
	resolved->busy.store(false, std::memory_order_release);
}

// The render targets the last draw acquired. Nearly every draw renders to the targets of the
// draw before it, with the same target registers, and acquiring them again repeats the same
// lookups (image discovery, color metadata, attachment views, HTile state) with the same results.
// A draw reuses them while nothing those lookups depend on can have changed:
//  - only draws executed since the targets were acquired: any other operation and any
//    submission invalidate them, so no guest-visible effect let the guest write in between;
//  - the target registers and the slice offset are the same;
//  - no image was registered or unregistered, and each target image (and stencil association)
//    still has the tracking, dirty and ownership state the acquisition left it in;
//  - no work wrote cached buffers since, and no page of the color/HTile metadata became CPU
//    dirty. The guest writes metadata through the GPU; a CPU write between two draws would need
//    the GPU to wait for it (WAIT_REG_MEM), and the first operation after a wait invalidates the
//    targets too.
// Draws with depth or stencil clears are not reused. Image layout transitions, attachment
// layouts and binding state are still set by every draw.
struct RenderExecutor::RenderTargetMemo {
	struct MetadataEpoch {
		GuestRange range;
		uint64_t   epoch = 0;
	};

	// Registers ResolveRenderColorTarget() and ResolveRenderDepthTarget() read.
	uint64_t                                                   context_id   = 0;
	uint32_t                                                   mrt_mask     = 0;
	uint32_t                                                   slice_offset = 0;
	uint32_t                                                   target_mask  = 0;
	std::array<HW::RenderTarget, RENDER_COLOR_ATTACHMENTS_MAX> targets {};
	HW::DepthRenderTarget                                      depth {};
	HW::RenderControl                                          render_control {};
	std::array<float, 3>                                       depth_values {};
	uint8_t                                                    stencil_clear = 0;
	bool                                                       depth_wanted  = false;

	// A draw captured its registers and is acquiring its targets.
	bool pending = false;
	// The outputs below are the targets of the last draw.
	bool                                                           valid                = false;
	uint64_t                                                       tick                 = 0;
	uint64_t                                                       image_set_epoch      = 0;
	uint64_t                                                       gpu_write_generation = 0;
	std::array<MetadataEpoch, RENDER_COLOR_ATTACHMENTS_MAX + 1>    metadata {};
	uint32_t                                                       metadata_count = 0;
	std::array<ImageLookupState, RENDER_COLOR_ATTACHMENTS_MAX + 2> images {};
	uint32_t                                                       image_count = 0;
	std::array<RenderColorInfo, RENDER_COLOR_ATTACHMENTS_MAX>      colors {};
	uint32_t                                                       color_count = 0;
	RenderDepthInfo                                                depth_info;
	std::array<vk::ImageView, RENDER_COLOR_ATTACHMENTS_MAX>        color_views {};
	vk::ImageView                                                  depth_view = nullptr;
};

template <typename T>
static bool SameBytes(const T& a, const T& b) {
	static_assert(std::is_trivially_copyable_v<T>);
	return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <typename T>
static void CopyBytes(T& destination, const T& source) {
	static_assert(std::is_trivially_copyable_v<T>);
	std::memcpy(&destination, &source, sizeof(T));
}

void RenderExecutor::InvalidateRenderTargetMemo() noexcept {
	m_draw_window++;
	if (m_render_target_memo != nullptr) {
		m_render_target_memo->valid   = false;
		m_render_target_memo->pending = false;
	}
	m_render_targets_reused = false;
}

void RenderExecutor::CaptureRenderTargetKey(const HW::Context& hw, uint32_t mrt_mask,
                                            uint32_t slice_offset) {
	if (m_render_target_memo == nullptr) {
		m_render_target_memo = {new RenderTargetMemo {},
		                        [](RenderTargetMemo* memo) { delete memo; }};
	}
	auto& memo   = *m_render_target_memo;
	memo.valid   = false;
	memo.pending = !graphics_debug_dump_enabled();
	if (!memo.pending) {
		return;
	}
	memo.context_id   = m_context_id;
	memo.mrt_mask     = mrt_mask;
	memo.slice_offset = slice_offset;
	memo.target_mask  = hw.GetRenderTargetMask();
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((mrt_mask & (1u << slot)) != 0) {
			CopyBytes(memo.targets[slot], hw.GetRenderTarget(slot));
		}
	}
	CopyBytes(memo.depth, hw.GetDepthRenderTarget());
	CopyBytes(memo.render_control, hw.GetRenderControl());
	memo.depth_wanted  = DepthStencilTargetWanted(hw);
	memo.depth_values  = {hw.GetDepthClearValue(), hw.GetDepthBoundsMin(), hw.GetDepthBoundsMax()};
	memo.stencil_clear = hw.GetStencilClearValue();
}

bool RenderExecutor::RenderTargetMemoCurrent() {
	const auto& memo  = *m_render_target_memo;
	auto&       cache = m_context.GetTextureCache();
	if (memo.tick != m_context.GetCommandScheduler().CurrentTick() ||
	    memo.image_set_epoch != cache.ImageSetEpoch() ||
	    memo.gpu_write_generation != m_context.GetBufferCache().GpuWriteGeneration()) {
		return false;
	}
	for (uint32_t i = 0; i < memo.image_count; i++) {
		const auto& state = memo.images[i];
		const auto* image = cache.m_slot_images.try_get(state.id);
		if (image == nullptr || !image->registered || image->binding.needs_rebind ||
		    ImageLookupState::Of(state.id, *image) != state) {
			return false;
		}
	}
	auto& buffers = m_context.GetBufferCache();
	for (uint32_t i = 0; i < memo.metadata_count; i++) {
		const auto& metadata = memo.metadata[i];
		if (buffers.CpuDirtyEpoch(metadata.range.address, metadata.range.size) != metadata.epoch) {
			return false;
		}
	}
	return true;
}

bool RenderExecutor::TryReuseRenderTargets(const HW::Context& hw, uint32_t mrt_mask,
                                           uint32_t slice_offset, DrawRenderState& state) {
	if (m_render_target_memo == nullptr || !m_render_target_memo->valid) {
		return false;
	}
	auto& memo = *m_render_target_memo;
	if (memo.mrt_mask != mrt_mask || memo.slice_offset != slice_offset) {
		return false;
	}
	// Operations with the same context snapshot have the same registers.
	if (m_context_id == 0 || memo.context_id != m_context_id) {
		// The depth bounds only set dynamic state: draws that differ in them (a deferred light
		// each) reuse the targets and take the bounds below.
		// Neither do the depth and stencil test controls, as long as the draw still wants a
		// depth/stencil target: they are refreshed below (layouts follow per draw).
		const float depth_clear = hw.GetDepthClearValue();
		if (memo.target_mask != hw.GetRenderTargetMask() ||
		    !SameBytes(memo.depth, hw.GetDepthRenderTarget()) ||
		    !SameBytes(memo.render_control, hw.GetRenderControl()) ||
		    memo.depth_wanted != DepthStencilTargetWanted(hw) ||
		    !SameBytes(memo.depth_values[0], depth_clear) ||
		    memo.stencil_clear != hw.GetStencilClearValue()) {
			return false;
		}
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			if ((mrt_mask & (1u << slot)) != 0 &&
			    !SameBytes(memo.targets[slot], hw.GetRenderTarget(slot))) {
				return false;
			}
		}
		memo.context_id = m_context_id;
	}
	if (!RenderTargetMemoCurrent()) {
		return false;
	}
	// The draw uses the memo's targets in place: only a draw that acquires its targets itself
	// stores new ones.
	if (memo.depth_info.image_id) {
		ApplyDepthStencilTestState(hw, memo.depth_info);
	}
	state.color_count = memo.color_count;
	state.colors      = memo.colors.data();
	state.depth       = &memo.depth_info;
	for (uint32_t i = 0; i < memo.color_count; i++) {
		BindRenderTarget(memo.colors[i].image_id);
	}
	if (memo.depth_info.image_id) {
		BindRenderTarget(memo.depth_info.image_id);
	}
	return true;
}

void RenderExecutor::StoreRenderTargetMemo(const RenderColorInfo* colors, uint32_t color_count,
                                           const RenderDepthInfo& depth, const vk::ImageView* views,
                                           vk::ImageView depth_view, ImageId stencil) {
	auto& memo   = *m_render_target_memo;
	memo.pending = false;
	// Clears consume their state; the next draw acquires its targets itself.
	if (depth.depth_clear_enable || depth.load_clear_aspects || depth.stencil_clear_enable) {
		return;
	}
	auto& cache             = m_context.GetTextureCache();
	auto& buffers           = m_context.GetBufferCache();
	memo.metadata_count     = 0;
	memo.image_count        = 0;
	const auto add_metadata = [&](const ImageInfo& info) {
		if (info.metadata.kind == ImageMetadataKind::None || info.metadata.range.Empty()) {
			return true;
		}
		const auto range                     = info.metadata.range;
		memo.metadata[memo.metadata_count++] = {range,
		                                        buffers.CpuDirtyEpoch(range.address, range.size)};
		return true;
	};
	const auto add_image = [&](ImageId id) {
		memo.images[memo.image_count++] = ImageLookupState::Of(id, cache.m_slot_images[id]);
	};
	for (uint32_t i = 0; i < color_count; i++) {
		if (!add_metadata(colors[i].desc.info)) {
			return;
		}
		add_image(colors[i].image_id);
		memo.colors[i]      = colors[i];
		memo.color_views[i] = views[i];
	}
	if (depth.image_id) {
		if (!add_metadata(depth.desc.info)) {
			return;
		}
		add_image(depth.image_id);
		if (stencil) {
			add_image(stencil);
		}
	}
	memo.color_count          = color_count;
	memo.depth_info           = depth;
	memo.depth_view           = depth_view;
	memo.tick                 = m_context.GetCommandScheduler().CurrentTick();
	memo.image_set_epoch      = cache.ImageSetEpoch();
	memo.gpu_write_generation = buffers.GpuWriteGeneration();
	memo.valid                = true;
}

// Keep it in sync with the members of DrawRenderState.
static void ResetDrawRenderState(DrawRenderState& state, bool tessellation_stages) {
	ValueInitialize(state.depth_info);
	// Only the first color_count targets are read, and resolving a target resets it first; the
	// first one is also logged for draws without targets.
	ValueInitialize(state.color_info[0]);
	state.color_count              = 0;
	state.mrt_mask                 = 0;
	state.topology                 = vk::PrimitiveTopology::ePointList;
	state.primitive_restart_enable = false;
	state.ps_active                = false;
	ValueInitialize(state.vertex_info[0]);
	if (tessellation_stages) {
		ValueInitialize(state.vertex_info[1]);
		ValueInitialize(state.vertex_info[2]);
	}
	ValueInitialize(state.ps_input_info);
	state.indirect_offsets = {};
	state.colors           = state.color_info;
	state.depth            = &state.depth_info;
}

DrawRenderState& RenderExecutor::AcquireDrawRenderState(bool tessellation, bool& shaders_resolved,
                                                        const ShaderIndirectOffsets& indirect_offsets) {
	// The operations before this draw executed: the resolution holds when the guest memory it
	// read still has the same bytes.
	auto* resolved = std::exchange(m_resolved_draw, nullptr);
	// Resolutions ahead of execution read the shaders with the offsets of GPU-read indirect
	// records; draws that patch the SGPRs instead resolve themselves.
	shaders_resolved = resolved != nullptr && resolved->compute_module == nullptr &&
	                   resolved->storage.state.indirect_offsets == indirect_offsets &&
	                   resolved->programs.reads.StillValid();
	if (shaders_resolved) {
		return resolved->storage.state;
	}
	if (m_draw_state == nullptr) {
		m_draw_state = {new DrawRenderStorage {},
		                [](DrawRenderStorage* storage) { delete storage; }};
	} else {
		ResetDrawRenderState(m_draw_state->state, m_draw_state->tessellation_stages_written);
	}
	m_draw_state->tessellation_stages_written = tessellation;
	return m_draw_state->state;
}

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

// KYTY_MESH_ARGS_RUNS: the mesh-tasks command of the DRAW_INDIRECT being recorded, from the
// current batch, or converted now together with the run's later DRAW_INDIRECTs (see
// MeshArgsRunState). False: no run here; the caller converts this draw's counts alone.
static bool ConvertMeshArgsRun(RenderContext& context, MeshIndirectArgs& converter,
                               CommandBuffer& buffer, uint64_t args, vk::Buffer& command_buffer,
                               vk::DeviceSize& command_offset) {
	auto& run = MeshArgsRun();
	if (run.packet == nullptr) {
		return false;
	}
	if (const auto* entry = run.Take(run.packet); entry != nullptr) {
		if (entry->args != args) {
			// The command processor collected another address (should not happen): drop the batch.
			run.converted.clear();
			run.next = 0;
			return false;
		}
		command_buffer = entry->buffer;
		command_offset = entry->offset;
		run.hits++;
		return true;
	}
	if (run.upcoming.size() < 2 || run.upcoming.front().packet != run.packet ||
	    run.upcoming.front().args != args) {
		return false;
	}
	// Synchronize and register every argument range first: uploads record copies and barriers of
	// their own, which must come before the conversions read the counts.
	std::vector<std::pair<Buffer*, uint64_t>> sources;
	sources.reserve(run.upcoming.size());
	for (const auto& entry: run.upcoming) {
		const auto source = context.GetBufferCache().ObtainBuffer(entry.args, 16u, false, false, {},
		                                                          true);
		if (source.first == nullptr || !source.first->HasDeviceAddress()) {
			break;
		}
		sources.push_back(source);
	}
	if (sources.size() < 2) {
		return false;
	}
	context.GetCommandScheduler().EndRendering();
	const auto recorder = buffer.Recorder();
	vk::MemoryBarrier counts_ready {};
	counts_ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
	                             vk::AccessFlagBits::eTransferWrite |
	                             vk::AccessFlagBits::eIndirectCommandRead;
	counts_ready.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                         vk::PipelineStageFlagBits::eComputeShader, {}, 1, &counts_ready, 0,
	                         nullptr, 0, nullptr);
	run.converted.clear();
	run.next = 0;
	for (size_t i = 0; i < sources.size(); i++) {
		const auto converted = converter.Convert(
		    recorder, sources[i].first->BufferDeviceAddress() + sources[i].second, false, false);
		run.converted.push_back(
		    {run.upcoming[i].packet, run.upcoming[i].args, converted.buffer, converted.offset});
	}
	vk::MemoryBarrier command_ready {};
	command_ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	command_ready.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                         vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &command_ready, 0,
	                         nullptr, 0, nullptr);
	run.batches++;
	run.batched += run.converted.size();
	run.upcoming.clear();
	const auto* entry = run.Take(run.packet);
	EXIT_IF(entry == nullptr);
	command_buffer = entry->buffer;
	command_offset = entry->offset;
	if (static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr; log) {
		static auto last = std::chrono::steady_clock::now();
		const auto  now  = std::chrono::steady_clock::now();
		if (now - last >= std::chrono::seconds(5)) {
			const double s = std::chrono::duration<double>(now - last).count();
			last           = now;
			std::fprintf(stderr,
			             "[mesh-args-runs] batches %.0f/s converted in batches %.0f/s, used %.0f/s, "
			             "single %.0f/s\n",
			             static_cast<double>(run.batches) / s, static_cast<double>(run.batched) / s,
			             static_cast<double>(run.hits) / s, static_cast<double>(run.single) / s);
			run.batches = run.batched = run.hits = run.single = 0;
		}
	}
	return true;
}

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects = {};
	auto&       cache = m_context.GetTextureCache();
	// The targets of the draw before, when this draw reused them and they are still current (the
	// texture bindings of this draw may have rediscovered them).
	const RenderTargetMemo* memo = nullptr;
	if (m_render_targets_reused && m_render_target_memo->valid &&
	    m_render_target_memo->color_count == color_count &&
	    m_render_target_memo->depth_info.image_id == depth.image_id && RenderTargetMemoCurrent()) {
		memo = m_render_target_memo.get();
		for (uint32_t i = 0; i < color_count && memo != nullptr; i++) {
			if (memo->colors[i].image_id != colors[i].image_id) {
				memo = nullptr;
			}
		}
	}
	if (m_render_targets_reused && memo == nullptr) {
		// The draw acquires the memo's targets in place (see TryReuseRenderTargets()) and may
		// update them: the memo no longer describes the draw before.
		m_render_target_memo->valid = false;
	}
	const bool store =
	    memo == nullptr && m_render_target_memo != nullptr && m_render_target_memo->pending;
	std::array<vk::ImageView, RENDER_COLOR_ATTACHMENTS_MAX> color_views {};
	vk::ImageView                                           depth_view = nullptr;
	ImageId                                                 stencil_id {};
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto& owner = cache.m_slot_images[target.image_id];
		if ((!owner.registered && !owner.info.data.Empty()) || owner.binding.needs_rebind) {
			EXIT("color target changed after render-state discovery\n");
		}
		const auto image_view = memo != nullptr
		                            ? memo->color_views[i]
		                            : cache.FindRenderTarget(target.image_id, target.desc);
		color_views[i]        = image_view;
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Recorder());
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	if (depth.image_id) {
		const auto& owner = cache.m_slot_images[depth.image_id];
		if (!owner.registered || owner.binding.needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
		// A reused acquisition had no depth clear and no HTile clear to consume.
		const auto image_view =
		    memo != nullptr ? memo->depth_view
		                    : cache.FindDepthTarget(depth.image_id, depth.desc, &stencil_id);
		depth_view = image_view;
		if (memo == nullptr) {
			const auto& metadata = depth.desc.info.metadata;
			const bool meta_clear =
			    metadata.kind == ImageMetadataKind::Htile &&
			    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
			depth.load_clear_aspects = {};
			if (depth.depth_clear_enable || meta_clear)
				depth.load_clear_aspects |= vk::ImageAspectFlagBits::eDepth;
			// Apply pending clears to compressed stencil as well as depth.
			if (depth.stencil_clear_enable || (meta_clear && metadata.stencil_compressed))
				depth.load_clear_aspects |= vk::ImageAspectFlagBits::eStencil;
			if (meta_clear &&
			    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
				EXIT("failed to consume HTile clear state\n");
			}
		}
		EXIT_IF(image_view == nullptr ||
		        cache.GetImage(depth.image_id).backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		const bool feedback_loops = m_context.GetGraphics().attachment_feedback_loop_enabled;
		vk::ImageAspectFlags sampled_aspects;
		for (auto* stage: stages) {
			for (auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			// Without VK_EXT_attachment_feedback_loop_* (the AMD Windows driver), the draw samples
			// the depth target it writes in the GENERAL layout: not defined by Vulkan, but what
			// the hardware does anyway, and better than ending the emulator.
			static std::atomic_bool warned {false};
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				LOGF("Warning: depth attachment feedback loop without host support; "
				     "using the GENERAL layout\n");
			}
		}
		auto& image = cache.GetImage(depth.image_id);
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Recorder());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eStencil);
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	if (store) {
		StoreRenderTargetMemo(colors, color_count, depth, color_views.data(), depth_view,
		                      stencil_id);
	}
	return state;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

bool RenderExecutor::ConsumeMetadataColorOperation(CommandBuffer& buffer,
                                                   uint32_t render_target_slice_offset) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	if (mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear)) {
		// FCE can precede any ordinary draw. Discover the target to materialize its native
		// clear metadata before the expanded image is sampled.
		RenderColorInfo target {};
		ResolveRenderColorTarget(buffer, target, render_target_slice_offset, 0, true);
		return true;
	}
	// Metadata/decompression shaders must not export colors into the expanded image.
	return mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

// vertex_offset is the first vertex of non-indexed draws.
struct DrawEmitInfo {
	int32_t          vertex_offset  = 0; // the first vertex of non-indexed draws
	uint32_t         first_instance = 0;
	// Nonzero address: a draw whose records, counts and offsets the GPU reads.
	DrawIndirectArgs indirect;
	// Nonzero: a draw whose counts the GPU reads from this guest vk::DrawIndirectCommand.
	uint64_t gpu_args_address = 0;
	// Nonzero: a vertex-pipeline draw whose instance count the GPU reads from this guest dword.
	uint64_t gpu_instance_count_address = 0;
	// With gpu_args_address: START_INST_LOC's user SGPR + 1 (0 = none), see DrawAutoArgs.
	uint32_t start_instance_sgpr = 0;
};

struct DrawIndexBufferSource {
	uint64_t      address   = 0;
	const void*   host_data = nullptr;
	uint64_t      size      = 0;
	vk::IndexType type      = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? resource.Base48() - buffer.addr +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

// Trivial: draws fill only the entries they use (no 3.5 KiB of zeroing per draw).
struct VertexBufferRange {
	uint64_t base_address;
	uint64_t requested_end;
	uint64_t acquired_end;
	Buffer*  buffer;
	uint64_t buffer_offset;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	// Entries [0, count) are set.
	std::array<vk::Buffer, MaxBuffers>     buffers;
	std::array<vk::DeviceSize, MaxBuffers> offsets;
	std::array<vk::DeviceSize, MaxBuffers> sizes;
	uint32_t                               count = 0;
};

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes;
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges;
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		// A buffer only constant-only embedded fetches refer to is never read: bind the null
		// buffer instead of acquiring whatever its unconsumed descriptor names.
		bool fetched = false;
		for (int a = 0; a < vs_input_info.resources_num && !fetched; a++) {
			fetched = vs_input_info.resources_dst[a].buffer_index == i &&
			          VertexAttributeFetched(vs_input_info, a);
		}
		const auto size = fetched ? VertexBufferDescriptorSize(i, vs_input_info) : 0;
		sizes[i]        = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size, 0, nullptr, 0};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges;
	uint32_t                                                      merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end, 0, nullptr, 0};
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
		std::tie(range.buffer, range.buffer_offset) =
		    cache.ObtainBuffer(range.base_address, size, false);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			prepared.sizes[i]   = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->buffer->Handle();
		prepared.offsets[i] = range->buffer_offset + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}

	return prepared;
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count, draw.first_instance);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kPolygon:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}

	const auto element_size = source.guest_element_size;
	const auto index_mask   = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index  = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask) {
		// Use native restart; the 8-bit path widens its marker to 0xffff.
		return true;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

RenderExecutor::ResolvedDraw* RenderExecutor::ResolveDrawAhead(
    const HW::Context& context, const HW::UserConfig& user_config, const HW::Shader& shaders,
    const ShaderIndirectOffsets& indirect_offsets) {
	if (!DrawHasValidVertexShader(shaders)) {
		return nullptr;
	}
	if (m_resolved_ring == nullptr) {
		m_resolved_ring = {new ResolvedDrawRing {}, [](ResolvedDrawRing* ring) { delete ring; }};
	}
	auto& ring = *m_resolved_ring;
	auto& draw = ring.draws[ring.next];
	// The execution thread releases resolutions in order.
	for (Common::SpinWait wait; draw.busy.load(std::memory_order_acquire);) {
		if (!wait.Spin()) {
			std::this_thread::yield();
		}
	}
	ring.next           = (ring.next + 1) % ResolvedDrawRing::Size;
	draw.compute_module = nullptr;
	auto& state         = draw.storage.state;
	ResetDrawRenderState(state, draw.storage.tessellation_stages_written);
	draw.storage.tessellation_stages_written =
	    user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	state.indirect_offsets = indirect_offsets;
	draw.programs.Reset();
	if (!m_context.GetPipelineCache().GetGraphicsProgramsAhead(context, user_config, shaders, state,
	                                                          draw.programs) ||
	    draw.programs.reads.Failed() || state.vertex_info[0].stage.module == nullptr) {
		return nullptr;
	}
	// Published to the execution thread with the operation that carries it.
	draw.busy.store(true, std::memory_order_relaxed);
	return &draw;
}

bool DrawPrepHasVertexShader(const HW::Shader& sh_ctx) {
	return DrawHasValidVertexShader(sh_ctx);
}

bool DrawPrepPixelState(const HW::Context& ctx, const HW::Shader& sh_ctx,
                        std::array<Prospero::ColorComponentMapping, 8>& target_export_mapping) {
	static_assert(RENDER_COLOR_ATTACHMENTS_MAX == 8);
	const auto& shader_regs       = ctx.GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(ctx);
	const bool  ps_active         = sh_ctx.GetPs().ps_regs.data_addr != 0 &&
	                       (color_output_mask != 0 ||
	                        PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	target_export_mapping = {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			target_export_mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                 rt.info.channel_order)
			        .export_mapping;
		}
	}
	return ps_active;
}

// KYTY_LOG_DRAWS=<first_frame>:<count>: from that frame on, log count draws with their shader
// hashes, counts and whether their programs exist (-1: not resolved, the draw is empty). A census
// of what a scene submits.
static void LogDrawCensusLine(CommandBuffer& buffer, const char* kind, uint32_t count,
                              uint32_t instances, int vs_program, int ps_program, int ps_active) {
	static const auto setting = [] {
		struct Setting {
			uint64_t first_frame = 0;
			uint32_t count       = 0;
		} result;
		if (const char* value = std::getenv("KYTY_LOG_DRAWS"); value != nullptr) {
			char* end          = nullptr;
			result.first_frame = std::strtoull(value, &end, 10);
			if (end != nullptr && *end == ':') {
				result.count = static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 10));
			}
		}
		return result;
	}();
	static std::atomic<uint32_t> used {0};
	// KYTY_LOG_DRAWS_WITH_COUNT=<n>: log the first 64 draws of exactly n vertices/indices.
	static const uint32_t with_count = [] {
		const char* value = std::getenv("KYTY_LOG_DRAWS_WITH_COUNT");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	if (with_count != 0 && count == with_count) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 64) {
			const auto& sh = buffer.GetShaders();
			const auto  h  = [](uint64_t addr) { return addr != 0 ? ShaderDeclaredHash(addr) : 0ull; };
			std::fprintf(stderr,
			             "[draw-count] %s es=%016" PRIx64 " gs=%016" PRIx64 " ps=%016" PRIx64
			             " count=%u instances=%u vs_program=%d ps_program=%d\n",
			             kind, h(sh.GetVs().es_regs.data_addr), h(sh.GetVs().gs_regs.data_addr),
			             h(sh.GetPs().ps_regs.data_addr), count, instances, vs_program, ps_program);
			std::fflush(stderr);
		}
	}
	// KYTY_LOG_GS_SLOTS=<es hash>: each draw of that ES/VS shader with its pixel shader and GS
	// user-data slots 24..30 as the context holds them at the draw (first 200 draws).
	if (static const uint64_t watch = [] {
		    const char* value = std::getenv("KYTY_LOG_GS_SLOTS");
		    return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	    }();
	    watch != 0) {
		const auto& sh = buffer.GetShaders();
		const bool  es_match =
		    sh.GetVs().es_regs.data_addr != 0 && ShaderDeclaredHash(sh.GetVs().es_regs.data_addr) == watch;
		// Or the pixel shader's hash; with KYTY_TRACE_GS_SLOT_SEQ the lines come from its budget, with
		// the last write to the traced slot this thread saw.
		const bool ps_match =
		    !es_match && sh.GetPs().ps_regs.data_addr != 0 && ShaderDeclaredHash(sh.GetPs().ps_regs.data_addr) == watch;
		static std::atomic<uint32_t> logged {0};
		if ((es_match || ps_match) && std::getenv("KYTY_TRACE_GS_SLOT_SEQ") != nullptr) {
			// The command-buffer dwords these draws' slot came from, read again a few frames later:
			// does the game write them after the submission was parsed?
			struct Pending {
				const uint32_t* where;
				uint32_t        value;
				uint64_t        frame;
			};
			static std::mutex           pending_mutex;
			static std::vector<Pending> pending;
			const uint64_t frame_now = buffer.GetContext().GetGpu().GetFrameNum();
			{
				std::lock_guard lock(pending_mutex);
				for (auto it = pending.begin(); it != pending.end();) {
					if (frame_now >= it->frame + 3) {
						const uint32_t now = *it->where;
						if (TakeGsSlotSeqLine()) {
							std::fprintf(stderr, "[gs-seq] later at=%p parsed=%08x now=%08x (%llu frames later)%s\n",
							             static_cast<const void*>(it->where), it->value, now,
							             static_cast<unsigned long long>(frame_now - it->frame),
							             now != it->value ? " CHANGED" : "");
						}
						it = pending.erase(it);
					} else {
						++it;
					}
				}
				const auto w = GetGsSlotLastWriter();
				if (w.source == 1 && w.where != nullptr && pending.size() < 256) {
					pending.push_back({w.where, w.value, frame_now});
				}
			}
			if (TakeGsSlotSeqLine()) {
				const auto  writer = GetGsSlotLastWriter();
				const auto& users  = sh.GetVs().gs_user_sgpr.value;
				std::fprintf(stderr,
				             "[gs-seq] draw frame=%llu count=%u user_sgpr=%u s36=%08x last=#%llu src=%u value=%08x"
				             " at=%p now=%08x set=%u+%u\n",
				             static_cast<unsigned long long>(buffer.GetContext().GetGpu().GetFrameNum()), count,
				             static_cast<uint32_t>(sh.GetVs().gs_regs.rsrc2.user_sgpr), users[28],
				             static_cast<unsigned long long>(writer.seq), writer.source, writer.value,
				             static_cast<const void*>(writer.where),
				             writer.where != nullptr ? *writer.where : 0u, writer.start, writer.count);
			}
		} else if (es_match || ps_match) {
			if (logged.fetch_add(1) < 200u) {
				const auto& users = sh.GetVs().gs_user_sgpr.value;
				std::fprintf(stderr,
				             "[gs-slots] frame=%llu ps=%016llx count=%u instances=%u user_sgpr=%u"
				             " s32..s38: %08x %08x %08x %08x %08x %08x %08x\n",
				             static_cast<unsigned long long>(
				                 buffer.GetContext().GetGpu().GetFrameNum()),
				             static_cast<unsigned long long>(
				                 sh.GetPs().ps_regs.data_addr != 0
				                     ? ShaderDeclaredHash(sh.GetPs().ps_regs.data_addr)
				                     : 0ull),
				             count, instances, static_cast<uint32_t>(sh.GetVs().gs_regs.rsrc2.user_sgpr),
				             users[24], users[25], users[26], users[27], users[28], users[29], users[30]);
			}
		}
	}
	// KYTY_TRACE_GS_SLOT_SEQ with KYTY_TRACE_GS_SLOT=<slot> and KYTY_TRACE_GS_SLOT_VALUE=<hex>: every
	// draw whose GS shader reads that slot and sees that value, with the write that left it.
	if (static const struct BadSlot {
		    int32_t  slot  = -1;
		    uint32_t value = 0;
	    } bad = [] {
		    BadSlot     b;
		    const char* seq   = std::getenv("KYTY_TRACE_GS_SLOT_SEQ");
		    const char* slot  = std::getenv("KYTY_TRACE_GS_SLOT");
		    const char* value = std::getenv("KYTY_TRACE_GS_SLOT_VALUE");
		    if (seq != nullptr && slot != nullptr && value != nullptr) {
			    b.slot  = static_cast<int32_t>(std::strtol(slot, nullptr, 10));
			    b.value = static_cast<uint32_t>(std::strtoul(value, nullptr, 16));
		    }
		    return b;
	    }();
	    bad.slot >= 0 && bad.slot < 32) {
		// The dword the bad value came from, read again 1, 4, 16 and 40 ms later (checked at every
		// draw): does the game write it after the submission was parsed?
		struct Timed {
			const uint32_t*                       where;
			uint32_t                              parsed;
			std::chrono::steady_clock::time_point at;
			uint32_t                              seen[4];
			int                                   next;
		};
		static std::vector<Timed> timed;
		if (!timed.empty()) {
			const auto now = std::chrono::steady_clock::now();
			static constexpr int ages_ms[4] = {1, 4, 16, 40};
			for (auto it = timed.begin(); it != timed.end();) {
				while (it->next < 4 && now - it->at >= std::chrono::milliseconds(ages_ms[it->next])) {
					it->seen[it->next++] = *it->where;
				}
				if (it->next == 4) {
					std::fprintf(stderr, "[gs-seq] timed at=%p parsed=%08x +1ms=%08x +4ms=%08x +16ms=%08x +40ms=%08x\n",
					             static_cast<const void*>(it->where), it->parsed, it->seen[0], it->seen[1],
					             it->seen[2], it->seen[3]);
					it = timed.erase(it);
				} else {
					++it;
				}
			}
		}
		const auto& sh = buffer.GetShaders();
		if (sh.GetVs().gs_regs.rsrc2.user_sgpr > static_cast<uint32_t>(bad.slot) &&
		    sh.GetVs().gs_user_sgpr.value[bad.slot] == bad.value && TakeGsSlotSeqLine()) {
			const auto writer = GetGsSlotLastWriter();
			if (writer.where != nullptr && timed.size() < 64) {
				timed.push_back({writer.where, writer.value, std::chrono::steady_clock::now(), {}, 0});
			}
			const auto h      = [](uint64_t addr) { return addr != 0 ? ShaderDeclaredHash(addr) : 0ull; };
			// The command-buffer dwords after the write (the packets up to this draw).
			std::string next;
			if (writer.where != nullptr) {
				const auto addr = reinterpret_cast<uint64_t>(writer.where);
				next += fmt::format(" gpu-written={} gpu-value={:08x} |",
				                    Libs::LibKernel::Memory::IsGpuBufferWritten(addr, 4) ? 1 : 0,
				                    Libs::LibKernel::Memory::PeekGpuDword(addr));
			}
			if (writer.source == 1 && writer.where != nullptr) {
				for (int i = 1; i <= 40; i++) {
					next += fmt::format(" {:08x}", writer.where[i]);
				}
				// And the 48 dwords before the SET_SH_REG packet's header.
				const auto* header = writer.where - (bad.slot - static_cast<int32_t>(writer.start)) - 2;
				next += " | before:";
				for (int i = -48; i <= 0; i++) {
					next += fmt::format(" {:08x}", header[i]);
				}
			}
			std::fprintf(stderr,
			             "[gs-seq] bad-draw frame=%llu es=%016llx ps=%016llx count=%u instances=%u last=#%llu src=%u"
			             " value=%08x at=%p next:%s\n",
			             static_cast<unsigned long long>(buffer.GetContext().GetGpu().GetFrameNum()),
			             static_cast<unsigned long long>(h(sh.GetVs().es_regs.data_addr)),
			             static_cast<unsigned long long>(h(sh.GetPs().ps_regs.data_addr)), count, instances,
			             static_cast<unsigned long long>(writer.seq), writer.source, writer.value,
			             static_cast<const void*>(writer.where), next.c_str());
		}
	}
	// KYTY_LOG_DRAW_PAIRS=1: each new (es, gs, ps) combination once, with its first draw's shape.
	if (static const bool log_pairs = std::getenv("KYTY_LOG_DRAW_PAIRS") != nullptr; log_pairs) {
		static std::mutex                   pairs_mutex;
		static std::unordered_set<uint64_t> pairs;
		const auto& sh = buffer.GetShaders();
		const auto  h  = [](uint64_t addr) { return addr != 0 ? ShaderDeclaredHash(addr) : 0ull; };
		const uint64_t es = h(sh.GetVs().es_regs.data_addr);
		const uint64_t gs = h(sh.GetVs().gs_regs.data_addr);
		const uint64_t ps = h(sh.GetPs().ps_regs.data_addr);
		bool fresh = false;
		{
			std::lock_guard lock(pairs_mutex);
			fresh = pairs.insert(es ^ (gs * 31u) ^ (ps * 0x9e3779b97f4a7c15ull)).second;
		}
		if (fresh) {
			const auto& regs = buffer.GetRegisters();
			std::fprintf(stderr,
			             "[draw-pair] %s es=%016" PRIx64 " gs=%016" PRIx64 " ps=%016" PRIx64
			             " count=%u instances=%u vs_program=%d ps_program=%d ps_active=%d rt_mask=0x%x"
			             " cb_shader_mask=0x%x\n",
			             kind, es, gs, ps, count, instances, vs_program, ps_program, ps_active,
			             regs.GetRenderTargetMask(), regs.GetShaderRegisters().m_cbShaderMask);
			for (uint32_t i = 0; i < 8; i++) {
				const auto& rt = regs.GetRenderTarget(i);
				const auto& bc = regs.GetBlendControl(i);
				if (rt.base.addr == 0 || ((regs.GetRenderTargetMask() >> (i * 4u)) & 0xfu) == 0) {
					continue;
				}
				std::fprintf(stderr,
				             "[draw-pair]   rt%u addr=0x%010" PRIx64 " fmt=%u nfmt=%u order=%u %ux%u bypass=%d"
				             " blend=%d src=%u dst=%u comb=%u sep=%d asrc=%u adst=%u acomb=%u\n",
				             i, rt.base.addr, static_cast<uint32_t>(rt.info.format),
				             static_cast<uint32_t>(rt.info.channel_type),
				             static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
				             rt.attrib2.height + 1, rt.info.blend_bypass ? 1 : 0, bc.enable ? 1 : 0,
				             bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
				             bc.separate_alpha_blend ? 1 : 0, bc.alpha_srcblend, bc.alpha_destblend,
				             bc.alpha_comb_fcn);
			}
			{
				const auto& vs = sh.GetVs();
				std::string users;
				for (uint32_t i = 0; i < HW::UserSgprInfo::SGPRS_MAX; i++) {
					users += fmt::format(" {:08x}", vs.gs_user_sgpr.value[i]);
				}
				std::fprintf(stderr, "[draw-pair]   gs user_sgpr=%u written=%u:%s\n",
				             static_cast<uint32_t>(vs.gs_regs.rsrc2.user_sgpr), vs.gs_user_sgpr.count,
				             users.c_str());
			}
			const auto& dc = regs.GetDepthControl();
			std::fprintf(stderr, "[draw-pair]   depth z=%d zw=%d zfunc=%u\n", dc.z_enable ? 1 : 0,
			             dc.z_write_enable ? 1 : 0, static_cast<uint32_t>(dc.zfunc));
			std::fflush(stderr);
		}
	}
	if (setting.count == 0) {
		return;
	}
	const auto frame = static_cast<uint64_t>(buffer.GetContext().GetGpu().GetFrameNum());
	// KYTY_LOG_DRAWS_TRIGGER=<file>: also wait until that file exists (checked once per frame), so
	// the census can start at a screen reached by hand.
	static const char*           trigger    = std::getenv("KYTY_LOG_DRAWS_TRIGGER");
	// Polled every 1024 draws: the GPU frame number does not advance in every mode.
	static std::atomic<uint32_t> checked    = 0;
	static std::atomic_bool      triggered  = trigger == nullptr;
	if (!triggered && (checked.fetch_add(1, std::memory_order_relaxed) & 1023u) == 0u) {
		if (FILE* file = std::fopen(trigger, "rb"); file != nullptr) {
			std::fclose(file);
			triggered = true;
			LOGF("Draw census: triggered at frame %" PRIu64 "\n", frame);
		}
	}
	if (!triggered || frame < setting.first_frame ||
	    used.fetch_add(1, std::memory_order_relaxed) >= setting.count) {
		return;
	}
	const auto& sh_ctx = buffer.GetShaders();
	const auto  hash   = [](uint64_t addr) { return addr != 0 ? ShaderDeclaredHash(addr) : 0ull; };
	const auto& regs   = buffer.GetRegisters();
	const auto& dc     = regs.GetDepthControl();
	const auto& drt    = regs.GetDepthRenderTarget();
	std::fprintf(stderr,
	             "Draw census: frame=%" PRIu64 " %s es=%016" PRIx64 " gs=%016" PRIx64
	             " ps=%016" PRIx64 " count=%u instances=%u vs_program=%d ps_program=%d ps_active=%d"
	             " ls=%016" PRIx64 " hs=%016" PRIx64 " rt_mask=0x%x z=%d zw=%d zfunc=%u"
	             " zbase=0x%" PRIx64 " zfmt=%u\n",
	             frame, kind, hash(sh_ctx.GetVs().es_regs.data_addr),
	             hash(sh_ctx.GetVs().gs_regs.data_addr), hash(sh_ctx.GetPs().ps_regs.data_addr),
	             count, instances, vs_program, ps_program, ps_active,
	             hash(sh_ctx.GetVs().ls_regs.data_addr), hash(sh_ctx.GetVs().hs_regs.data_addr),
	             regs.GetRenderTargetMask(), dc.z_enable ? 1 : 0, dc.z_write_enable ? 1 : 0,
	             static_cast<unsigned>(dc.zfunc), drt.z_read_base_addr,
	             static_cast<unsigned>(drt.z_info.format));
	std::fflush(stderr);
}

static void LogDrawCensus(CommandBuffer& buffer, const DrawCallInfo& draw,
                          const DrawRenderState& state) {
	LogDrawCensusLine(buffer, draw.Name(), draw.index_count, draw.instance_count,
	                  state.programs.vertex[0] ? 1 : 0, state.programs.pixel ? 1 : 0,
	                  state.ps_active ? 1 : 0);
}

// Diagnostics: the draw parameters of the watched shader's draws in the watched frame (see
// InWatchedShaderFrame); each line precedes that draw's WatchShader bindings.
void RenderExecutor::LogWatchedDraw(const DrawCallInfo& draw, const DrawRenderState& state,
                                    uint32_t index_offset, int32_t base_vertex,
                                    uint32_t first_vertex, int32_t host_vertex_offset,
                                    uint32_t host_first_instance, bool indirect) {
	static std::atomic<uint32_t> logged {0};
	const auto*                  program = state.vertex_info[0].stage.program;
	if (program == nullptr ||
	    !InWatchedShaderFrame(program->shader_hash,
	                          m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed),
	                          logged)) {
		return;
	}
	LOGF("WatchShader draw: %s count=%u instances=%u first_instance=%u index_offset=%u "
	     "base_vertex=%d first_vertex=%u host_vertex_offset=%d host_first_instance=%u "
	     "indirect=%d\n",
	     draw.Name(), draw.index_count, draw.instance_count, draw.first_instance, index_offset,
	     base_vertex, first_vertex, host_vertex_offset, host_first_instance, indirect ? 1 : 0);
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state) {
	const auto& shader_regs       = buffer.GetRegisters().GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(buffer.GetRegisters());
	state.ps_active = buffer.GetShaders().GetPs().ps_regs.data_addr != 0 &&
	                  (color_output_mask != 0 ||
	                   PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	RefreshShaders(buffer, draw, color_output_mask, state);
	LogDrawCensus(buffer, draw, state);
	if (!state.programs.vertex[0] || (state.ps_active && !state.programs.pixel)) {
		return false;
	}
	uint32_t mrt_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
			    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
				mrt_mask |= 1u << output.location;
			}
		}
		if (draw.IsIndexed()) {
			LogDrawPhase(draw.Name(), "ResolveRenderDepthTarget");
		}
		ResolveRenderDepthTarget(buffer, state.depth_info);
		// TODO(PPSA09243): Vulkan passes take one sample count. Where PS5 attachments mix counts,
		// keep those matching the raster count instead of resolving multisampled shadows.
		const auto colors       = std::span {state.color_info, state.color_count};
		const auto pass_samples = state.depth_info.image_id ? state.depth_info.desc.info.samples
		                                                    : state.color_info[0].desc.info.samples;
		if (std::ranges::any_of(colors, [&](const RenderColorInfo& color) {
			    return color.desc.info.samples != pass_samples;
		    })) {
			const auto raster =
			    render_sample_count(buffer.GetRegisters().GetAaConfig().msaa_num_samples);
			const auto dropped = std::ranges::remove_if(colors, [&](const RenderColorInfo& color) {
				return color.desc.info.samples != raster;
			});
			state.color_count -= static_cast<uint32_t>(dropped.size());
			if (state.depth_info.image_id && state.depth_info.desc.info.samples != raster) {
				state.depth_info = {};
			}
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(
				    "Warning: dropping attachments whose sample count differs from the raster.\n");
			}
		}
	}

	if (state.color_count == 0 && !state.depth->image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), state.colors[0], *state.depth, buffer, draw.index_count, 0);
		return false;
	}

	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type = source.type;
	if (source.host_data != nullptr) {
		auto& cache     = buffer.GetContext().GetBufferCache();
		auto& stream    = cache.GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = cache.StreamUpload(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	return prepared;
}

static void CommitVertexBuffers(const CommandRecorder&       vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(const CommandRecorder&     vk_buffer,
                              const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
	                             const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.colors[0], *state.depth, buffer, state.ps_input_info,
		                   draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.colors[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

struct PreparedIndirectArgs {
	vk::Buffer     args         = nullptr;
	vk::DeviceSize args_offset  = 0;
	vk::Buffer     count        = nullptr;
	vk::DeviceSize count_offset = 0;
};

// Draws whose arguments stay in guest memory for the GPU (see DrawIndirectArgs).
static PreparedIndirectArgs PrepareIndirectArgs(CommandBuffer&            buffer,
                                                const DrawIndirectArgs& source, bool indexed) {
	auto&      cache     = buffer.GetContext().GetBufferCache();
	const auto args_size = indexed ? sizeof(vk::DrawIndexedIndirectCommand)
	                               : sizeof(vk::DrawIndirectCommand);
	PreparedIndirectArgs prepared;
	const auto [args, args_offset] = cache.ObtainBuffer(
	    source.address, uint64_t {source.count - 1} * source.stride + args_size, false);
	EXIT_IF(args == nullptr || (args_offset & 3u) != 0);
	prepared.args        = args->Handle();
	prepared.args_offset = args_offset;
	if (source.count_addr != 0) {
		const auto [count, count_offset] =
		    cache.ObtainBuffer(source.count_addr, sizeof(uint32_t), false);
		EXIT_IF(count == nullptr || (count_offset & 3u) != 0);
		prepared.count        = count->Handle();
		prepared.count_offset = count_offset;
	}
	return prepared;
}

static void EmitIndirectDraws(const CommandRecorder& vk_buffer, const DrawCallInfo& draw,
                              const DrawIndirectArgs& source, const PreparedIndirectArgs& args) {
	if (args.count != nullptr) {
		if (draw.IsIndexed()) {
			vk_buffer.drawIndexedIndirectCount(args.args, args.args_offset, args.count,
			                                   args.count_offset, source.count, source.stride);
		} else {
			vk_buffer.drawIndirectCount(args.args, args.args_offset, args.count, args.count_offset,
			                            source.count, source.stride);
		}
	} else if (draw.IsIndexed()) {
		vk_buffer.drawIndexedIndirect(args.args, args.args_offset, source.count, source.stride);
	} else {
		vk_buffer.drawIndirect(args.args, args.args_offset, source.count, source.stride);
	}
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, const CommandRecorder& vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kPolygon:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.IsIndexed()) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count,
				               static_cast<uint32_t>(emit.vertex_offset), emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count,
					               i + static_cast<uint32_t>(emit.vertex_offset),
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

bool RenderExecutor::CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
                                       const DrawEmitInfo& emit, DrawRenderState& state) {
	if (!buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
		return false;
	}
	auto& ucfg = buffer.GetUserConfig();
	const auto& hw = buffer.GetRegisters();
	auto& depth = state.depth_info;
	auto& color = state.color_info[0];
	EXIT_NOT_IMPLEMENTED(!m_context.GetGraphics().maintenance8_enabled ||
	                     !Prospero::IsRectList(ucfg.GetPrimType()) ||
	                     draw.IsIndexed() || draw.index_count != 3 || draw.instance_count != 1 ||
	                     emit.vertex_offset != 0 || emit.first_instance != 0 ||
	                     state.color_count != 1 || color.target_slot != 0 || !depth.image_id ||
	                     color.desc.info.samples != 1 ||
	                     depth.desc.info.extent.width != color.Extent().width ||
	                     depth.desc.info.extent.height != color.Extent().height ||
	                     depth.desc.view_info.layer_count != 1 ||
	                     color.desc.view_info.layer_count != 1 ||
	                     hw.GetBlendControl(0).enable || state.ps_input_info.ps_pixel_kill_enable ||
	                     state.ps_input_info.ps_depth_export_enable ||
	                     state.ps_input_info.ps_sample_mask_export_enable);
	vk::ImageAspectFlags feedback;
	const auto rendering = AcquireRenderTargets(buffer, &color, 1, depth, feedback);
	if (depth.load_clear_aspects) {
		m_context.GetCommandScheduler().BeginRendering(rendering);
	}
	auto& cache = m_context.GetTextureCache();
	auto& source = cache.GetImage(depth.image_id);
	auto& destination = cache.GetImage(color.image_id);
	EXIT_NOT_IMPLEMENTED((source.backing.format != vk::Format::eD32Sfloat &&
	                      source.backing.format != vk::Format::eD32SfloatS8Uint) ||
	                     destination.backing.format != vk::Format::eR32Sfloat);
	const auto rect = calc_final_scissor(hw.GetScreenViewport(), hw.GetScanModeControl(),
	                                    {rendering.width, rendering.height}, 0);
	if (rect.right > rect.left && rect.bottom > rect.top) {
		const vk::ImageCopy region {
		    {vk::ImageAspectFlagBits::eDepth, depth.desc.view_info.base_level,
		     depth.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {vk::ImageAspectFlagBits::eColor, color.desc.view_info.base_level,
		     color.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {static_cast<uint32_t>(rect.right - rect.left),
		     static_cast<uint32_t>(rect.bottom - rect.top), 1}};
		m_context.GetCommandScheduler().EndRendering();
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			LOGF("[depth-copy] first depth-to-color copy: %ux%u\n",
			     static_cast<uint32_t>(rect.right - rect.left),
			     static_cast<uint32_t>(rect.bottom - rect.top));
		}
		const auto command = buffer.Recorder();
		source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
		               {}, command);
		destination.Transit(vk::ImageLayout::eTransferDstOptimal,
		                    vk::AccessFlagBits2::eTransferWrite, {}, command);
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                  1, &region);
	}
	return true;
}

// Whether a draw may be skipped while the driver compiles its pipeline: the picture is all it
// changes. Its shaders write no memory, its samples reach no guest occlusion counter and it
// renders to no linear image (the CPU reads those back, e.g. GTA V's vehicle damage).
static bool DrawOnlyChangesPicture(RenderContext& context, const DrawRenderState& state,
                                   std::span<const ShaderVertexInputInfo> vertex_stages) {
	const auto writes_memory = [](const ShaderStageRuntime& stage) {
		if (stage.module == nullptr) {
			return true;
		}
		const auto& program = stage.module->info;
		return program.has_address_writes ||
		       std::ranges::any_of(program.info.buffers,
		                           [](const auto& buffer) { return buffer.written; }) ||
		       std::ranges::any_of(program.info.images,
		                           [](const auto& image) { return image.written; });
	};
	if (context.GetOcclusionQueries().Counting()) {
		return false;
	}
	for (const auto& stage: vertex_stages) {
		if (writes_memory(stage.stage)) {
			return false;
		}
	}
	if (state.ps_active && writes_memory(state.ps_input_info.stage)) {
		return false;
	}
	for (uint32_t i = 0; i < state.color_count; i++) {
		if (!state.colors[i].desc.info.IsTiled()) {
			return false;
		}
	}
	return !static_cast<bool>(state.depth->image_id) || state.depth->desc.info.IsTiled();
}

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         const DrawEmitInfo&          emit,
                                         const DrawIndexBufferSource& index_source) {
	auto&      ucfg                     = buffer.GetUserConfig();
	const bool primitive_restart_enable = state.primitive_restart_enable;
	if (CopyDepthToColor(buffer, draw, emit, state)) {
		return;
	}
	const auto vertex_stages = state.VertexStages();
	const bool mesh_active = state.vertex_info[0].stage.module->info.stage == ShaderType::Mesh;
	uint32_t   mesh_groups = 0;
	// Host mesh shaders size their launch from the counts (CanDrawIndirectOnGpu).
	EXIT_IF(mesh_active && emit.indirect.address != 0);
	if (mesh_active) {
		const auto& mesh = state.vertex_info[0].mesh;
		EXIT_NOT_IMPLEMENTED(mesh.fast_launch && (draw.IsIndexed() || primitive_restart_enable));
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed());
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(), primitive_restart_enable);
		}
		// Counts read on the GPU (DRAW_INDIRECT): one group per vertex, as fast launch runs.
		EXIT_NOT_IMPLEMENTED(emit.gpu_args_address != 0 && !mesh.fast_launch);
	}
	if (mesh_active && emit.gpu_args_address == 0) {
		const auto& mesh       = state.vertex_info[0].mesh;
		const auto  primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		if (draw.instance_count > limits.maxMeshWorkGroupCount[1]) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
	}

	vk::Buffer     mesh_command_buffer = nullptr;
	vk::DeviceSize mesh_command_offset = 0;
	if (emit.gpu_args_address != 0) {
		// The mesh shader reads the start vertex there, and the counts are copied from there:
		// synchronize and register the arguments like the guest indices below.
		const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
		    emit.gpu_args_address, 16u, false, false, {}, true);
		if (mesh_active && args_buffer != nullptr && args_buffer->HasDeviceAddress()) {
			Common::HackNotice::Report(
			    "GPU-driven mesh draw with counts above the host limits draws nothing",
			    "RenderExecutor::DrawAuto", 0, 0,
			    "armed: mesh_indirect_args.comp zeroes the groups on the GPU; occurrences are not counted");
			// The counts become the mesh-tasks command on the GPU, validated against the device's
			// limits (MeshIndirectArgs), before the draw's bindings are committed.
			if (m_mesh_args == nullptr) {
				m_mesh_args = std::make_unique<MeshIndirectArgs>(m_context.GetGraphics(),
				                                                 m_context.GetCommandScheduler());
			}
			if (!ConvertMeshArgsRun(m_context, *m_mesh_args, buffer, emit.gpu_args_address,
			                        mesh_command_buffer, mesh_command_offset)) {
				m_context.GetCommandScheduler().EndRendering();
				const auto converted = m_mesh_args->Convert(
				    buffer.Recorder(), args_buffer->BufferDeviceAddress() + args_offset);
				mesh_command_buffer = converted.buffer;
				mesh_command_offset = converted.offset;
				MeshArgsRun().single++;
			}
		}
	}
	if (emit.gpu_instance_count_address != 0) {
		(void)m_context.GetBufferCache().ObtainBuffer(emit.gpu_instance_count_address, 4u, false,
		                                              false, {}, true);
	}
	if (mesh_active && draw.IsIndexed()) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed.
		(void)m_context.GetBufferCache().FindBuffer(
		    index_source.address, static_cast<uint64_t>(draw.index_count) *
		                              index_source.guest_element_size);
	}
	LogDrawPhase(draw.Name(), "PrepareBindings");
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
		descriptor_stages[stage_count++] = &bindings.vertex[i];
	}
	if (state.ps_active) {
		if (!bindings.pixel) bindings.pixel.emplace();
		PrepareBindings(state.ps_input_info.stage, *bindings.pixel);
		descriptor_stages[stage_count++] = &*bindings.pixel;
	}
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	PrepareGraphicsBindings(stages, std::span {state.colors, state.color_count});
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	if (!mesh_active) {
		LogDrawPhase(draw.Name(), "PrepareVertexBuffers");
		vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
		index_binding   = PrepareIndexBuffer(buffer, index_source);
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "CreatePipeline");
	}
	// A draw that writes no memory may be skipped while its new pipeline compiles in the
	// background; its render targets are not touched. One that writes buffers or storage images
	// waits for the pipeline, as others may read what it writes.
	const auto writes_memory = [](const ShaderStageRuntime& runtime) {
		return HasShaderBufferWrites(runtime) ||
		       std::ranges::any_of(runtime.program->info.images, [](const auto& image) {
			       return image.written &&
			              image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
		       });
	};
	bool may_defer = !(state.ps_active && writes_memory(state.ps_input_info.stage));
	for (const auto& stage: vertex_stages) {
		may_defer = may_defer && !writes_memory(stage.stage);
	}
	// Glyphs and cached layers: drawn once into small targets (SmallColorTargetDraw).
	may_defer = may_defer && !SmallColorTargetDraw(buffer.GetRegisters());
	auto* deferred_pipeline = m_context.GetPipelineCache().TryGetGraphicsPipeline(
	    std::span {state.color_info, state.color_count}, state.depth_info, vertex_stages, buffer,
	    state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable,
	    state.programs, may_defer);
	if (deferred_pipeline == nullptr) {
		return;
	}
	auto& pipeline = *deferred_pipeline;
	vk::ImageAspectFlags feedback_aspects;
	const auto rendering = AcquireRenderTargets(buffer, state.colors, state.color_count,
	                                            *state.depth, feedback_aspects, stages);

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory. Recorded through the recording thread when it runs.
	const auto vk_buffer = buffer.Recorder();
	SetDrawDebugPhase(buffer, submit_id, draw, draw.IsIndexed() ? 0x100u : 0x200u);
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x300u);
	}
	CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	if (!mesh_active) {
		CommitIndexBuffer(vk_buffer, index_binding);
	}
	using ShaderRecompiler::IR::PushData;
	const auto draw_stages =
	    vk::ShaderStageFlagBits::eFragment |
	    (mesh_active ? vk::ShaderStageFlagBits::eMeshEXT : vk::ShaderStageFlagBits::eVertex);
	const auto draw_dwords =
	    PushData::GraphicsDrawDwords(mesh_active, state.ps_input_info.object_primitive_id);
	if (draw_dwords != 0) {
		std::array<uint32_t, PushData::MeshDrawDwordCount> draw_data {};
		if (mesh_active) {
			// GPU arguments: dword 3 = 0xFFFFFFFF tells the fast-launch entry to read the start
			// vertex at dwords 4-5's address + 8 (the shader's other draw dwords are unused then).
			const bool gpu_args = emit.gpu_args_address != 0;
			draw_data = {
			    gpu_args ? 0u : draw.index_count,
			    gpu_args ? 0u : static_cast<uint32_t>(emit.vertex_offset),
			    gpu_args ? emit.start_instance_sgpr : emit.first_instance,
			    gpu_args ? 0xFFFFFFFFu : index_source.guest_element_size,
			    static_cast<uint32_t>(gpu_args ? emit.gpu_args_address : index_source.address),
			    static_cast<uint32_t>((gpu_args ? emit.gpu_args_address : index_source.address) >>
			                          32u),
			    0u};
		} else {
			draw_data[0] = buffer.GetUserConfig().GetObjectId();
		}
		vk_buffer.pushConstants(pipeline.pipeline_layout, draw_stages, 0,
		                        draw_dwords * sizeof(uint32_t), draw_data.data());
	}
	if (const auto planes = vertex_stages.back().user_clip_planes; planes != 0) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, draw_stages,
		                        PushData::UserClipPlaneDword * sizeof(uint32_t),
		                        std::bit_width(planes) * 4u * sizeof(uint32_t),
		                        buffer.GetRegisters().GetUserClipPlanes().data());
	}

	auto& dynamic_state = m_context.GetCommandScheduler().GetDynamicState();
	UpdateGraphicsDynamicState(buffer, vk_buffer, dynamic_state, vertex_stages.back(),
	                           state.depth_info, rendering);
	dynamic_state.SetAttachmentFeedbackLoopAspects(feedback_aspects);
	vk_buffer.commitDynamicState(dynamic_state, m_context.GetGraphics());

	// GPU arguments: the group counts (vertex count, instance count, 1) go into a ring entry for
	// drawMeshTasksIndirectEXT, outside the render pass; the arguments are never read on the CPU.
	vk::Buffer     mesh_indirect_buffer = nullptr;
	vk::DeviceSize mesh_indirect_offset = 0;
	constexpr uint32_t MeshIndirectEntries = 4096;
	if (mesh_active && emit.gpu_args_address != 0 && mesh_command_buffer) {
		mesh_indirect_buffer = mesh_command_buffer;
		mesh_indirect_offset = mesh_command_offset;
	} else if (mesh_active && emit.gpu_args_address != 0) {
		auto&              cache = m_context.GetBufferCache();
		auto&              args  = cache.GetBuffer(cache.FindBuffer(emit.gpu_args_address, 16u));
		if (m_mesh_indirect == nullptr) {
			m_mesh_indirect = std::make_unique<Buffer>(
			    m_context.GetGraphics(), m_context.GetCommandScheduler(), MemoryUsage::DeviceLocal,
			    0, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer,
			    MeshIndirectEntries * 16u);
		}
		mesh_indirect_buffer = m_mesh_indirect->Handle();
		mesh_indirect_offset = (m_mesh_indirect_next++ % MeshIndirectEntries) * 16u;
		m_context.GetCommandScheduler().EndRendering();
		// The arguments' writer (a dispatch or a copy) and the entry's last indirect read come
		// before the copy.
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
		                       vk::AccessFlagBits::eTransferWrite |
		                       vk::AccessFlagBits::eIndirectCommandRead;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                          vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr,
		                          0, nullptr);
		const vk::BufferCopy copy {args.Offset(emit.gpu_args_address), mesh_indirect_offset, 8u};
		vk_buffer.copyBuffer(args.Handle(), mesh_indirect_buffer, 1, &copy);
		vk_buffer.fillBuffer(mesh_indirect_buffer, mesh_indirect_offset + 8u, 4u, 1u);
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &after, 0,
		                          nullptr, 0, nullptr);
	}

	// Vertex pipeline, counts on the GPU: the arguments' writer (a dispatch or a copy) comes
	// before the indirect read; the arguments are read where the cache holds them.
	PreparedIndirectArgs indirect_args;
	vk::Buffer     vertex_indirect_buffer = nullptr;
	vk::DeviceSize vertex_indirect_offset = 0;
	if (!mesh_active && emit.gpu_instance_count_address != 0) {
		// Only the instance count is on the GPU: a ring entry gets the known counts and offsets,
		// and the GPU copies the instance count into it.
		auto& cache  = m_context.GetBufferCache();
		auto& source = cache.GetBuffer(cache.FindBuffer(emit.gpu_instance_count_address, 4u));
		if (m_mesh_indirect == nullptr) {
			m_mesh_indirect = std::make_unique<Buffer>(
			    m_context.GetGraphics(), m_context.GetCommandScheduler(), MemoryUsage::DeviceLocal,
			    0, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer,
			    MeshIndirectEntries * 16u);
		}
		vertex_indirect_buffer = m_mesh_indirect->Handle();
		vertex_indirect_offset = (m_mesh_indirect_next++ % MeshIndirectEntries) * 16u;
		m_context.GetCommandScheduler().EndRendering();
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
		                       vk::AccessFlagBits::eTransferWrite |
		                       vk::AccessFlagBits::eIndirectCommandRead;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                          vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr,
		                          0, nullptr);
		const uint32_t command[4] {draw.index_count, 0u, static_cast<uint32_t>(emit.vertex_offset), emit.first_instance};
		vk_buffer.updateBuffer(vertex_indirect_buffer, vertex_indirect_offset, sizeof(command),
		                       command);
		const vk::BufferCopy copy {source.Offset(emit.gpu_instance_count_address),
		                           vertex_indirect_offset + sizeof(uint32_t), sizeof(uint32_t)};
		vk_buffer.copyBuffer(source.Handle(), vertex_indirect_buffer, 1, &copy);
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &after, 0,
		                          nullptr, 0, nullptr);
	} else if (!mesh_active && emit.gpu_args_address != 0) {
		auto& cache = m_context.GetBufferCache();
		auto& args  = cache.GetBuffer(cache.FindBuffer(emit.gpu_args_address, 16u));
		vertex_indirect_buffer = args.Handle();
		vertex_indirect_offset = args.Offset(emit.gpu_args_address);
		m_context.GetCommandScheduler().EndRendering();
		vk::MemoryBarrier ready {};
		ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
		ready.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &ready, 0,
		                          nullptr, 0, nullptr);
	} else if (!mesh_active && emit.indirect.address != 0) {
		// The draws read their arguments (and count) in guest memory: GPU work writes them.
		indirect_args = PrepareIndirectArgs(buffer, emit.indirect, draw.IsIndexed());
		m_context.GetCommandScheduler().EndRendering();
		vk::MemoryBarrier ready {};
		ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
		ready.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &ready, 0,
		                          nullptr, 0, nullptr);
	}

	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		// As for dispatches, buffer writes wait for earlier accesses to the memory.
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteHazardBarrier(vk_buffer, shader_write_stages);
	}
	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	}
	m_context.GetCommandScheduler().BeginRendering(rendering);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
	}
	GpuTiming::Before(m_context, buffer);
	int64_t gpu_zone = -1;
	if (Profiler::g_counters) {
		// The draw's program, as its GPU zone's name and its command zone's text.
		char      name[48];
		const int size = std::snprintf(
		    name, sizeof(name), "VS %016" PRIx64 " PS %016" PRIx64,
		    vertex_stages.back().stage.program->shader_hash,
		    state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0);
		gpu_zone = GpuProfiler::Begin(m_context, buffer, name, static_cast<size_t>(size), 0x4e79a7);
		Profiler::CommandZone::Annotate("%s", name);
	}
	// KYTY_BARRIER_SINK: a pending guest barrier may pass this draw when it writes only its own
	// attachments and samples none of them (barrierSink.h (b)). Attachments in GENERAL or a
	// feedback-loop layout are (or may be) sampled; read-only depth layouts are not written.
	bool draw_safe = false;
	if (BarrierSinkEnabled()) {
		const auto writes = [&](const ShaderStageRuntime& runtime) {
			return writes_memory(runtime) || runtime.program->has_address_writes;
		};
		draw_safe = !feedback_aspects && !(state.ps_active && writes(state.ps_input_info.stage));
		for (const auto& stage: vertex_stages) {
			draw_safe = draw_safe && !writes(stage.stage);
		}
		auto& cache = m_context.GetTextureCache();
		for (uint32_t i = 0; draw_safe && i < state.color_count; i++) {
			draw_safe = cache.GetImage(state.color_info[i].image_id).binding.attachment_layout ==
			            vk::ImageLayout::eColorAttachmentOptimal;
		}
		if (draw_safe && state.depth_info.image_id) {
			const auto layout = cache.GetImage(state.depth_info.image_id).binding.attachment_layout;
			draw_safe = layout == vk::ImageLayout::eDepthStencilAttachmentOptimal ||
			            layout == vk::ImageLayout::eDepthAttachmentOptimal ||
			            layout == vk::ImageLayout::eStencilAttachmentOptimal ||
			            layout == vk::ImageLayout::eDepthStencilReadOnlyOptimal ||
			            layout == vk::ImageLayout::eDepthReadOnlyOptimal ||
			            layout == vk::ImageLayout::eStencilReadOnlyOptimal ||
			            layout == vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal ||
			            layout == vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal;
		}
		for (const auto* stage: stages) {
			if (!draw_safe) break;
			draw_safe = stage->gds.buffer == nullptr && stage->shared_memory.buffer == nullptr;
			for (const auto& binding: stage->images) {
				bool target = state.depth_info.image_id && binding.image_id == state.depth_info.image_id;
				for (uint32_t i = 0; !target && i < state.color_count; i++) {
					target = binding.image_id == state.color_info[i].image_id;
				}
				draw_safe = draw_safe && !target;
			}
		}
	}
	SetNextDrawSafe(draw_safe);
	if (mesh_active) {
		if (mesh_indirect_buffer) {
			vk_buffer.drawMeshTasksIndirectEXT(mesh_indirect_buffer, mesh_indirect_offset, 1, 16u);
		} else {
			// A draw with more groups than the host allows in one dispatch (RADV: 65,535 in x) is
			// split; draw dword 6 gives each part its first group.
			constexpr auto base_offset = (ShaderRecompiler::IR::PushData::MeshDrawDwordCount - 1u) *
			                             sizeof(uint32_t);
			const auto& limits   = m_context.GetGraphics().mesh_shader_properties;
			const auto  per_draw = std::min<uint64_t>(
			    limits.maxMeshWorkGroupCount[0],
			    std::max<uint64_t>(1u, limits.maxMeshWorkGroupTotalCount /
			                               std::max(draw.instance_count, 1u)));
			uint32_t groups_done = 0;
			do {
				const auto groups =
				    static_cast<uint32_t>(std::min<uint64_t>(mesh_groups - groups_done, per_draw));
				if (groups_done != 0) {
					// The same stages as the draw dwords' push above (the layout's range).
					vk_buffer.pushConstants(pipeline.pipeline_layout,
					                        vk::ShaderStageFlagBits::eMeshEXT |
					                            vk::ShaderStageFlagBits::eFragment,
					                        base_offset, sizeof(groups_done), &groups_done);
				}
				vk_buffer.drawMeshTasksEXT(groups, draw.instance_count, 1);
				groups_done += groups;
			} while (groups_done < mesh_groups);
		}
	} else if (vertex_indirect_buffer) {
		vk_buffer.drawIndirect(vertex_indirect_buffer, vertex_indirect_offset, 1, 16u);
	} else if (emit.indirect.address != 0) {
		EmitIndirectDraws(vk_buffer, draw, emit.indirect, indirect_args);
	} else {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	}
	SetNextDrawSafe(false);
	Profiler::Add(Profiler::Counter::Draws);
	GpuProfiler::End(buffer, gpu_zone);
	GpuTiming::After(m_context, buffer,
	                 state.ps_active ? state.ps_input_info.stage.program->shader_hash
	                                 : vertex_stages.back().stage.program->shader_hash,
	                 GpuTiming::Draw);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Draws);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawInstances, draw.instance_count);

	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x600u);
	}
	// KYTY_DRAW_WRITE_BARRIER=0 (experiment): no barrier after a draw that writes buffers; only
	// the guest's own sync orders its writes, as on the console.
	static const bool draw_write_barrier = [] {
		const char* value = std::getenv("KYTY_DRAW_WRITE_BARRIER");
		return value == nullptr || value[0] != '0';
	}();
	if (shader_write_stages && draw_write_barrier) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	LogDrawPhase(draw.Name(), "DrawComplete");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x700u);
	}
}

bool RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	Common::LockGuard lock(m_context.GetMutex());
	if (args.index_count == 0 || (args.instance_count == 0 && args.indirect.address == 0)) {
		LogDrawCensusLine(buffer, "DrawIndex-empty", args.index_count, args.instance_count, -1, -1,
		                  -1);
		return true;
	}

	if (ConsumeMetadataColorOperation(buffer, args.render_target_slice_offset) ||
	    DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		InvalidateRenderTargetMemo();
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	// A pure check of the context registers: once per register snapshot.
	if (m_context_id == 0 || m_context_id != m_checked_context_id ||
	    graphics_debug_dump_enabled()) {
		hw_check(buffer);
		m_checked_context_id = m_context_id;
	}

	bool  shaders_resolved = false;
	auto& state = AcquireDrawRenderState(ucfg.GetPrimType() == Prospero::PrimitiveType::kPatch,
	                                     shaders_resolved, args.indirect.offsets);
	state.indirect_offsets = args.indirect.offsets;
	if (!GetDrawTopology(ucfg, state.topology)) {
		return true;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_source.type               = vk::IndexType::eUint16;
			index_source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			index_source.type               = vk::IndexType::eUint32;
			index_source.guest_element_size = 4;
			break;
		case Prospero::IndexType::kIndex8:
			index_source.guest_element_size = 1;
			break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	if (args.indirect.address != 0) {
		// The draws read the indices their arguments select.
		EXIT_IF(index_source.guest_element_size == 1);
		index_source.size = args.indirect.index_bytes;
	}
	const bool primitive_restart = ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] =
			    state.primitive_restart_enable && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count,
	                        args.instance_count, args.first_instance};
	if (!PrepareDrawRenderState(buffer, shaders_resolved, draw, args.render_target_slice_offset,
	                            state)) {
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     args.index_addr);

	DrawEmitInfo emit {};
	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	if (args.indirect.address != 0) {
		// The records carry their own offsets.
		emit.indirect = args.indirect;
	} else {
		const auto [vertex_offset, instance_offset] =
		    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
		             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
		// Native fetches consume the patched SGPRs; rewritten fetches need Vulkan offsets.
		const bool native_indirect = indirect && !state.vertex_info[0].fetch_embedded;
		emit.vertex_offset  = native_indirect ? 0 : vertex_offset + args.base_vertex;
		emit.first_instance = native_indirect ? 0 : instance_offset;
	}
	LogWatchedDraw(draw, state, ucfg.GetIndexOffset(), args.base_vertex, 0, emit.vertex_offset,
	               emit.first_instance, indirect);

	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart);
	ResetBindings();
	return true;
}

bool RenderExecutor::CanDrawIndirectOnGpu(const CommandBuffer& buffer, bool indexed,
                                          uint32_t index_type, uint64_t index_bytes) const {
	if (!m_context.GetGraphics().indirect_draws_enabled) {
		return false;
	}
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	// Guest geometry shaders (host mesh shaders) size their launch from the counts.
	if ((ctx.GetShaderStages() & 0x20u) != 0) {
		return false;
	}
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kTriList: break;
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
			// Restart with a custom index scans the indices on the CPU.
			if ((ucfg.GetPrimitiveResetControl() & 0x1u) != 0) {
				return false;
			}
			break;
		default: return false;
	}
	// Color and depth operations that replace the draw run only when the draw is not empty.
	const auto mode = ctx.GetColorControl().mode;
	if (mode != static_cast<uint8_t>(CbColorMode::Disable) &&
	    mode != static_cast<uint8_t>(CbColorMode::Normal)) {
		return false;
	}
	const auto& depth_override = ctx.GetDepthRenderOverride();
	if (mode == static_cast<uint8_t>(CbColorMode::Disable) &&
	    (depth_override.force_z_dirty || depth_override.force_stencil_dirty)) {
		return false;
	}
	if (indexed) {
		// 8-bit indices are widened on the CPU; the index range must be known.
		const auto type = static_cast<Prospero::IndexType>(index_type);
		if ((type != Prospero::IndexType::kIndex16 && type != Prospero::IndexType::kIndex32) ||
		    index_bytes == 0) {
			return false;
		}
	}
	return true;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);
	// KYTY_LOG_GPU_ARGS_DRAWS=1: each new (es, gs, ps) of a DrawIndexAuto with GPU-written
	// arguments, once, with the arguments' address.
	static const bool log_gpu_args = std::getenv("KYTY_LOG_GPU_ARGS_DRAWS") != nullptr;
	if (log_gpu_args && (args.gpu_args_address != 0 || args.gpu_instance_count_address != 0)) {
		static std::mutex                   seen_mutex;
		static std::unordered_set<uint64_t> seen;
		const auto h  = [](uint64_t addr) { return addr != 0 ? ShaderDeclaredHash(addr) : 0ull; };
		const auto es = h(sh_ctx.GetVs().es_regs.data_addr);
		const auto gs = h(sh_ctx.GetVs().gs_regs.data_addr);
		const auto ps = h(sh_ctx.GetPs().ps_regs.data_addr);
		const auto key = es ^ (gs * 0x9e3779b97f4a7c15ull) ^ (ps * 0xc2b2ae3d27d4eb4full);
		std::lock_guard lock(seen_mutex);
		if (seen.insert(key).second) {
			std::fprintf(stderr,
			             "[gpu-args-draw] submit=%llu es=%016llx gs=%016llx ps=%016llx args=0x%llx "
			             "inst=0x%llx prim=%u\n",
			             static_cast<unsigned long long>(submit_id),
			             static_cast<unsigned long long>(es), static_cast<unsigned long long>(gs),
			             static_cast<unsigned long long>(ps),
			             static_cast<unsigned long long>(args.gpu_args_address),
			             static_cast<unsigned long long>(args.gpu_instance_count_address),
			             static_cast<uint32_t>(ucfg.GetPrimType()));
			std::fflush(stderr);
		}
	}

	Common::LockGuard lock(m_context.GetMutex());
	if (args.gpu_args_address == 0 && args.indirect.address == 0 &&
	    (args.vertex_count == 0 ||
	     (args.instance_count == 0 && args.gpu_instance_count_address == 0))) {
		LogDrawCensusLine(buffer, "DrawIndexAuto-empty", args.vertex_count, args.instance_count, -1,
		                  -1, -1);
		return true;
	}

	if (ConsumeMetadataColorOperation(buffer, args.render_target_slice_offset) ||
	    DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		InvalidateRenderTargetMemo();
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	// A pure check of the context registers: once per register snapshot.
	if (m_context_id == 0 || m_context_id != m_checked_context_id ||
	    graphics_debug_dump_enabled()) {
		hw_check(buffer);
		m_checked_context_id = m_context_id;
	}

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto,
	                         args.vertex_count, args.instance_count, args.first_instance};

	bool  shaders_resolved = false;
	auto& state = AcquireDrawRenderState(ucfg.GetPrimType() == Prospero::PrimitiveType::kPatch,
	                                     shaders_resolved, args.indirect.offsets);
	state.indirect_offsets = args.indirect.offsets;
	if (!GetDrawTopology(ucfg, state.topology)) {
		ResetBindings();
		return true;
	}
	if (!PrepareDrawRenderState(buffer, shaders_resolved, draw, args.render_target_slice_offset,
	                            state)) {
		ResetBindings();
		return true;
	}

	// Counts read on the GPU work for fast-launch mesh draws and vertex-pipeline draws with one
	// draw call; others read the guest arguments here (waiting for the GPU when it wrote them).
	DrawAutoArgs drawn = args;
	// KYTY_GPU_ARGS_MESH / KYTY_GPU_ARGS_VERTEX=0: read that kind's GPU-written counts here.
	// Mesh draws read them here by default since 2026-10-10: with the counts read on the GPU,
	// ~1 boot in 4 lost the device (shader read fault) in the title scene's first GPU-driven
	// draws; read here, 0 in 22 (the counts read were all sane). Read here, they need the GPU
	// drained first (the page protection of a just-marked page is not in force yet), ~25 drains a
	// frame; on the GPU they are now validated against the device's limits (MeshIndirectArgs), so
	// they are read there by default again. KYTY_GPU_ARGS_MESH=0 reads them here.
	static const bool gpu_args_mesh = [] {
		const char* value = std::getenv("KYTY_GPU_ARGS_MESH");
		return value == nullptr || value[0] != '0';
	}();
	static const bool gpu_args_vertex = [] {
		const char* value = std::getenv("KYTY_GPU_ARGS_VERTEX");
		return value == nullptr || value[0] != '0';
	}();
	if (args.gpu_args_address != 0) {
		const bool mesh   = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
		const bool gpu_ok = mesh ? gpu_args_mesh && state.vertex_info[0].mesh.fast_launch
		                         : gpu_args_vertex &&
		                               ucfg.GetPrimType() != Prospero::PrimitiveType::kQuadListLegacy;
		if (!gpu_ok) {
			uint32_t words[4] {};
			m_context.GetBufferCache().WaitSharedWrites(args.gpu_args_address, sizeof(words));
			static const bool args_debug = std::getenv("KYTY_ARGS_DEBUG") != nullptr;
			const auto        before =
			    args_debug ? m_context.GetBufferCache().DescribeWriteState(args.gpu_args_address)
			                      : std::string {};
			std::memcpy(words, reinterpret_cast<const void*>(args.gpu_args_address), sizeof(words));
			if (args_debug && (words[0] > (1u << 22u) || words[1] > 65535u)) {
				std::fprintf(stderr, "[gpu-args]   before the read: %s\n", before.c_str());
			}
			drawn.vertex_count     = words[0];
			drawn.instance_count   = words[1];
			drawn.first_vertex     = words[2];
			drawn.first_instance   = words[3];
			drawn.gpu_args_address = 0;
			if (words[0] > (1u << 22u) || words[1] > 65535u) {
				static std::atomic<uint32_t> odd {0};
				if (odd.fetch_add(1) < 32) {
					std::fprintf(stderr, "[gpu-args] odd counts at 0x%llx: %u %u %u %u mesh=%d\n",
					             static_cast<unsigned long long>(args.gpu_args_address), words[0],
					             words[1], words[2], words[3], mesh ? 1 : 0);
					std::fprintf(stderr, "[gpu-args]   %s\n",
					             m_context.GetBufferCache()
					                 .DescribeWriteState(args.gpu_args_address)
					                 .c_str());
					std::fflush(stderr);
				}
			}
			if (drawn.vertex_count == 0 || drawn.instance_count == 0) {
				ResetBindings();
				return true;
			}
		}
	}
	if (args.gpu_instance_count_address != 0) {
		const bool mesh   = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
		const bool gpu_ok = !mesh && ucfg.GetPrimType() != Prospero::PrimitiveType::kQuadListLegacy;
		if (!gpu_ok) {
			m_context.GetBufferCache().WaitSharedWrites(args.gpu_instance_count_address,
			                                            sizeof(uint32_t));
			std::memcpy(&drawn.instance_count,
			            reinterpret_cast<const void*>(args.gpu_instance_count_address),
			            sizeof(uint32_t));
			drawn.gpu_instance_count_address = 0;
			if (drawn.instance_count == 0) {
				ResetBindings();
				return true;
			}
		}
	}
	const DrawCallInfo drawn_info {CommandBufferDebugOp::DrawIndexAuto, drawn.vertex_count,
	                               drawn.instance_count, drawn.first_instance};

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.module->info.param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);

	DrawEmitInfo emit {};
	const bool indirect = drawn.offset_source == DrawOffsetSource::IndirectArgs;
	if (drawn.indirect.address != 0) {
		emit.indirect = drawn.indirect;
	} else {
		const auto [vertex_offset, instance_offset] =
		    indirect ? std::pair<int32_t, uint32_t> {0, drawn.first_instance}
		             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
		const bool native_indirect = indirect && !state.vertex_info[0].fetch_embedded;
		emit.vertex_offset = native_indirect ? 0 : vertex_offset + static_cast<int32_t>(drawn.first_vertex);
		emit.first_instance = native_indirect ? 0 : instance_offset;
	}
	emit.gpu_args_address = drawn.gpu_args_address;
	emit.gpu_instance_count_address = drawn.gpu_instance_count_address;
	emit.start_instance_sgpr = drawn.gpu_args_address != 0 ? drawn.start_instance_sgpr : 0u;
	LogWatchedDraw(drawn_info, state, ucfg.GetIndexOffset(), 0, drawn.first_vertex,
	               emit.vertex_offset, emit.first_instance, indirect);

	DrawIndexBufferSource index_source {};
	ExecutePreparedDraw(submit_id, buffer, drawn_info, state, topology, emit, index_source, false);
	ResetBindings();
	return true;
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer, uint32_t render_target_slice_offset) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, render_target_slice_offset, 1, true, true);
	if (!src.image_id || !dst.image_id) {
		return false;
	}
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	// A target can live inside a larger image: GTA V resolves each face of its reflection cube by
	// pointing CB_COLOR1 at that face's slice. FindImage() then moved the view to the matching
	// level and layer of that image, which is where the resolve has to read and write.
	destination.Resolve(source,
	                    {src.desc.view_info.base_level, 1, src.desc.view_info.base_layer, 1},
	                    {dst.desc.view_info.base_level, 1, dst.desc.view_info.base_layer, 1});
	return true;
}

} // namespace Libs::Graphics
