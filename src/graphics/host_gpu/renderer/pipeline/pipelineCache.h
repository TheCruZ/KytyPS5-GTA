#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/drawPrep.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// On by default (KYTY_PIPELINE_CACHE=frank or KYTY_PROGRAM_RESOLUTION=0 turn it off): the resolve
// thread resolves the shader programs of draws and dispatches ahead of execution
// (ProgramResolution), instead of preparing only their SRT walks (DrawPrep).
[[nodiscard]] bool ProgramResolutionEnabled();
// A thread between the resolve and execution threads checks the guest reads of each resolution a
// few operations before execution (GuestReadLog::PreValidate); KYTY_VALIDATE_STAGE=0 turns it off.
[[nodiscard]] bool ValidateStageEnabled();

// The backing bytes of the guest pages one evaluation reads, found once per page (the definitions
// are in pipelineCache.cpp). Only pages with no GPU-written byte are cached: a page that is not
// clean stays on the exact path. Valid for one resolution: within it nothing marks a cached page
// GPU-written, and the execution thread checks the pages again before it uses the resolution.
struct GuestPageReadCache {
	static constexpr size_t Capacity = 16;
	struct Entry {
		uint64_t       page;
		const uint8_t* backing;
	};
	std::array<Entry, Capacity> entries;
	size_t                      count = 0;
	size_t                      next  = 0;

	void Reset() {
		count = 0;
		next  = 0;
	}
	// The page's backing bytes when it is GPU-clean, else null (SrtRuntime::map_clean_page).
	const uint8_t* Map(uint64_t page);
	// True when the page is GPU-clean and the bytes were copied from its backing.
	bool Read(uint64_t address, std::span<uint32_t> values);
	bool ReadBytes(uint64_t address, void* data, uint64_t size);
};

// Guest memory that a resolution ahead of execution read. The execution thread repeats the reads
// before it uses the resolution: guest memory may have changed in between (operations that
// execute before the draw write it), and ahead of execution GPU-owned bytes cannot be read. The
// reads of the stages' SRT walks are recorded by their stage results; this log adds those of the
// shader state (vertex fetch tables) and the verdict of the resolution.
class GuestReadLog {
public:
	static constexpr uint32_t MaxSets = 5;

	void Clear() {
		m_own.Clear();
		m_set_count   = 0;
		m_failed      = false;
		m_gpu_written = false;
		m_pre_state.store(PreNone, std::memory_order_relaxed);
		m_pages.Reset();
	}
	// The pages this resolution read, shared by its stages' walks and its vertex fetch reads.
	[[nodiscard]] GuestPageReadCache& PageCache() { return m_pages; }
	// Reads the bytes of the backing store no GPU work has written, and records them. False (and
	// the log fails) when the range cannot be read that way.
	bool Read(uint64_t address, void* data, uint64_t size);
	// A read set kept elsewhere (a stage result's) that StillValid() checks as well.
	void Attach(const DrawPrep::ReadSet* set) {
		if (m_set_count < m_sets.size()) {
			m_sets[m_set_count++] = set;
		} else {
			m_failed = true;
		}
	}
	void               Fail() { m_failed = true; }
	[[nodiscard]] bool Failed() const { return m_failed; }
	// Whether some read failed because the GPU wrote its bytes (the backing store gets them only
	// once read back).
	[[nodiscard]] bool MayHaveReadGpuWrites() const { return m_gpu_written; }
	// Execution thread: whether every read still gives the recorded bytes.
	[[nodiscard]] bool StillValid() const;
	// Validation stage (KYTY_VALIDATE_STAGE): the check StillValid() makes, run a few operations
	// before the execution thread needs it. StillValid() then answers from it as long as no bytes
	// became GPU-written since (GpuDirtySeq()), and checks again itself otherwise.
	void PreValidate();

private:
	DrawPrep::ReadSet                                m_own;
	GuestPageReadCache                               m_pages;
	std::array<const DrawPrep::ReadSet*, MaxSets>    m_sets {};
	uint32_t                                         m_set_count   = 0;
	bool                                             m_failed      = false;
	bool                                             m_gpu_written = false;
	static constexpr uint8_t PreNone = 0, PreValid = 1, PreInvalid = 2;
	std::atomic<uint8_t>                             m_pre_state {PreNone};
	std::atomic<uint64_t>                            m_pre_seq {0};
};

// Storage of the shader programs a draw resolved ahead of its execution: the stage results the
// stages' resource snapshots and specializations live in, and the guest memory they read.
struct ProgramResolution {
	static constexpr uint32_t MaxStages = 4;

	std::array<DrawPrep::StageResult, MaxStages> results;
	GuestReadLog                                 reads;
	uint32_t                                     stages = 0;

	void Reset() {
		reads.Clear();
		stages = 0;
	}
};

struct GraphicContext;
struct ShaderParams;
struct RenderColorInfo;
struct RenderDepthInfo;
class CommandBuffer;

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

namespace DrawPrep {
struct PacketResult;
} // namespace DrawPrep

// KYTY_SRT_VERIFY (ResourceSnapshot.h): a verifying shader reported a value the host used that
// differed from the GPU's. Its refreshes read GPU-written bytes exactly from now on.
void SrtVerifyReportMiss(uint64_t shader_hash);

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       alpha_to_coverage_enable = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       cull_front                                         = false;
	bool                       cull_back                                          = false;
	bool                       face                                               = false;
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 117);

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;

	bool operator==(const PipelineRenderingState&) const = default;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride                           = 0;
		bool     instance                         = false;
		bool     operator==(const Binding&) const = default;
	};
	struct Attribute {
		uint32_t offset                             = 0;
		uint8_t  binding                            = 0;
		bool     operator==(const Attribute&) const = default;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;

	// Only the used entries count (the others are zero in every key), and consecutive draws mostly
	// repeat a handful of them: the defaulted comparison walked all sixty-four.
	bool operator==(const PipelineVertexInputState& other) const {
		if (binding_count != other.binding_count || attribute_count != other.attribute_count) {
			return false;
		}
		for (uint32_t i = 0; i < binding_count; i++) {
			if (!(bindings[i] == other.bindings[i])) {
				return false;
			}
		}
		for (uint32_t i = 0; i < attribute_count; i++) {
			if (!(attributes[i] == other.attributes[i])) {
				return false;
			}
		}
		return true;
	}
};

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;
	uint32_t         spirv_words = 0;
	uint64_t         hash        = 0;

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

// Everything vkCreateGraphicsPipelines reads for one graphics pipeline, owned in one place so the
// driver compile can run on a worker thread (the structures point into each other: it stays
// where it was allocated).
struct GraphicsPipelineBuild {
	GraphicsPipelineBuild() = default;
	~GraphicsPipelineBuild();
	KYTY_CLASS_NO_COPY(GraphicsPipelineBuild);

	vk::Device                                                            device;
	vk::ShaderModule                                                      tess_control = nullptr;
	vk::ShaderModule                                                      tess_eval    = nullptr;
	std::array<vk::PipelineShaderStageCreateInfo, 4>                      stages {};
	std::array<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo, 4>  subgroup_sizes {};
	std::array<vk::VertexInputAttributeDescription, ShaderVertexInputInfo::RES_MAX> input_attr {};
	std::array<vk::VertexInputBindingDescription, ShaderVertexInputInfo::RES_MAX>   input_desc {};
	vk::PipelineVertexInputStateCreateInfo                                vertex_input_info {};
	vk::PipelineInputAssemblyStateCreateInfo                              input_assembly {};
	vk::PipelineViewportDepthClipControlCreateInfoEXT                     depth_clip_control {};
	vk::PipelineViewportStateCreateInfo                                   viewport_state {};
	vk::PipelineRasterizationDepthClipStateCreateInfoEXT                  clip_ext {};
	vk::PipelineRasterizationStateCreateInfo                              rasterizer {};
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT            provoking_vertex {};
	vk::PipelineMultisampleStateCreateInfo                                multisampling {};
	std::array<vk::PipelineColorBlendAttachmentState, RENDER_COLOR_ATTACHMENTS_MAX> color_blend {};
	std::array<vk::Bool32, RENDER_COLOR_ATTACHMENTS_MAX>                  color_write_enable {};
	vk::PipelineColorWriteCreateInfoEXT                                   color_write {};
	vk::PipelineColorBlendStateCreateInfo                                 color_blending {};
	vk::PipelineDepthStencilStateCreateInfo                               depth_stencil {};
	std::vector<vk::DynamicState>                                         dynamic_states;
	vk::PipelineDynamicStateCreateInfo                                    dynamic_state {};
	vk::PipelineTessellationStateCreateInfo                               tessellation {};
	PipelineRenderingState                                                rendering;
	vk::PipelineRenderingCreateInfo                                       rendering_info {};
	vk::GraphicsPipelineCreateInfo                                        pipeline_info {};
	uint64_t                                                              vs_hash = 0;
	uint64_t                                                              ps_hash = 0;
};

// A graphics pipeline a worker thread compiles (see PipelineCache::TryGetGraphicsPipeline).
struct PendingGraphicsPipeline;
class PipelineCompiler;

// The owning renderer serializes access, including saves while the GPU is running.
// Whether every color target the draw writes is at most KYTY_SMALL_TARGET_SYNC pixels wide and
// high (default 1024; 0 = never): such draws are never skipped while their shaders translate or
// their pipeline compiles (pipelineCache.cpp).
bool SmallColorTargetDraw(const HW::Context& context);

class PipelineCache {
public:
	explicit PipelineCache(GraphicContext& graphics);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	void Save();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		bool                    uses_push_descriptors = false;
		// Samples bindless images: descriptor set 1 is the bindless table.
		bool                    uses_bindless         = false;
		// Set while a worker compiles `pipeline`; draws skip it until the compile finishes.
		std::shared_ptr<PendingGraphicsPipeline> pending;
	};

	struct GraphicsPrograms {
		std::array<ShaderProgram, 3> vertex;
		ShaderProgram pixel;

		[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] ? 3u : 1u; }
	};

	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                    ShaderPixelInputInfo& pixel_info, ProgramResolution* ahead = nullptr);
	// Draw prep (KYTY_DRAW_PREP, drawPrep.h), scanner thread: the SRT refreshes GetGraphicsPrograms
	// would run for these registers, into `out`. False when nothing was prepared.
	bool DrawPrepGraphics(const HW::VertexShaderInfo& vertex_regs,
	                      const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                      const HW::Context& context, const HW::UserConfig& user_config,
	                      std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                      bool pixel_active, DrawPrep::PacketResult& out);
	// The same for a dispatch: the SRT refresh GetComputeProgram would run.
	bool DrawPrepCompute(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
	                     ShaderComputeInputInfo& input_info, DrawPrep::PacketResult& out);
	// With `ahead` (KYTY_PIPELINE_CACHE=ours, resolve thread): only a program and permutation that
	// exist, with the SRT walk read through the resolution's log (see GetGraphicsPrograms).
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info,
	                                ProgramResolution*           ahead = nullptr);
	// Whether the shader at this address is on the skip list (--skip-shaders): its dispatches can
	// be dropped before anything is prepared for them.
	[[nodiscard]] static bool IsSkipListed(uint64_t shader_addr);
	// Whether the compute shader at this address read guest memory to build its descriptors in its
	// last resolution (specialization reads). True until a resolution says otherwise.
	[[nodiscard]] static bool ReadsDescriptorMemory(uint64_t shader_addr);
	// What a resolution of the shader found (tests set it for shaders they never resolve).
	static void SetReadsDescriptorMemory(uint64_t shader_addr, bool reads);
	// KYTY_SRT_DEFER=1 (GPU thread, around one dispatch): SRT reads of GPU-written bytes start
	// their download and fail instead of waiting; EndSrtDefer returns the tick to wait for before
	// the dispatch runs again (0: it ran). SrtDeferPending: the current refresh was deferred.
	[[nodiscard]] static bool     SrtDeferEnabled();
	static void                   BeginSrtDefer(bool allowed);
	[[nodiscard]] static uint64_t EndSrtDefer();
	[[nodiscard]] static bool     SrtDeferPending();

	Pipeline& GetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                              const RenderDepthInfo&                 depth,
	                              std::span<const ShaderVertexInputInfo> vertex_info,
	                              CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs);
	// As GetGraphicsPipeline; with may_defer, a pipeline the driver does not compile within a
	// short wait is compiled on a worker thread, and nullptr says to skip this draw.
	Pipeline* TryGetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                                 const RenderDepthInfo&                 depth,
	                                 std::span<const ShaderVertexInputInfo> vertex_info,
	                                 CommandBuffer&                         command,
	                                 const ShaderPixelInputInfo*            ps_input_info,
	                                 vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                                 const GraphicsPrograms& programs, bool may_defer);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info,
	                             const ShaderProgram&          compute_program);

private:
	struct ProgramCache;

	// GetGraphicsPrograms' stage inputs from the registers (shared with DrawPrepGraphics).
	bool DescribeGraphicsStages(const HW::VertexShaderInfo& vertex_regs,
	                            const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                            const HW::Context& context, const HW::UserConfig& user_config,
	                            std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                            bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                            ShaderPixelInputInfo& pixel_info,
	                            std::array<ShaderParams, 3>& vertex_params, ShaderParams& pixel_params);

	struct GraphicsPipelineKey {
		PipelineRenderingState   rendering;
		std::array<uint64_t, 3>  vertex_shader_ids {};
		uint64_t                 ps_shader_id = 0;
		PipelineVertexInputState vertex_input;
		PipelineStaticParameters static_params;

		bool operator==(const GraphicsPipelineKey& other) const {
			// The parts that differ most often between consecutive draws first.
			return vertex_shader_ids[0] == other.vertex_shader_ids[0] &&
			       ps_shader_id == other.ps_shader_id && static_params == other.static_params &&
			       vertex_input == other.vertex_input && rendering == other.rendering &&
			       vertex_shader_ids == other.vertex_shader_ids;
		}
	};

	struct PipelineKeyHash {
		static void Mix(std::size_t& hash, std::size_t value) {
			hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
			        (hash >> 2u);
		}
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	GraphicContext&               m_graphics;
	std::unique_ptr<ProgramCache> m_program_cache;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	// Pipelines created since the driver cache was last written; a background thread writes
	// the cache periodically so a crash does not lose the pipelines compiled until then.
	std::atomic_uint64_t          m_unsaved_pipelines {0};
	Common::Mutex                 m_write_mutex;
	std::jthread                  m_save_thread;
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_graphics_pipelines;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_pipelines;
	// The pipeline the last lookup found; pipelines are never removed while the cache lives.
	GraphicsPipelineKey m_last_graphics_key {};
	Pipeline*           m_last_graphics_pipeline = nullptr;
	std::unique_ptr<PipelineCompiler>                       m_compiler;
	uint64_t                                                m_deferred_draws = 0;

	void InitializeDriverCache();
	// Takes a finished background compile into the pipeline; false while it still runs.
	bool FinishPending(Pipeline& pipeline);
	// Writes the driver cache data to m_driver_cache_path; the cache must stay alive meanwhile.
	void WriteDriverCache();
};

// KYTY_DYNAMIC_RASTER_STATE (default on): cull mode and front face are dynamic state of every
// renderer graphics pipeline, recorded per draw by UpdateGraphicsDynamicState, and leave the
// pipeline key; key fields that cannot change the pipeline are zeroed. =0 bakes them into each pipeline as before.
[[nodiscard]] bool PipelineDynamicRasterStateEnabled();

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
// Creates the pipeline's layouts and fills the create info; CreateGraphicsPipeline compiles it.
std::unique_ptr<GraphicsPipelineBuild>
PrepareGraphicsPipeline(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                        const PipelineRenderingState&          rendering,
                        const PipelineVertexInputState&        vertex_input,
                        std::span<const ShaderVertexInputInfo> vertex_info,
                        const ShaderPixelInputInfo*            ps_input_info,
                        const PipelineCache::GraphicsPrograms& programs,
                        const PipelineStaticParameters&        static_params);
// False for an embedded-fetch attribute whose components the shader only replaces by constants:
// it gets no native vertex attribute, and its buffer is not bound for it.
[[nodiscard]] bool VertexAttributeFetched(const ShaderVertexInputInfo& info, int attribute);
// Any thread.
vk::Result CreateGraphicsPipeline(const GraphicsPipelineBuild& build, vk::PipelineCache driver_cache,
                                  vk::Pipeline* pipeline);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
