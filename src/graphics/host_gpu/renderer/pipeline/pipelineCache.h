#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// Guest memory that a resolution ahead of execution read. The execution thread repeats the reads
// before it uses the resolution: guest memory may have changed in between (operations that
// execute before the draw write it), and ahead of execution GPU-owned bytes cannot be read.
class GuestReadLog {
public:
	void Clear() {
		m_entries.clear();
		m_bytes.clear();
		m_failed      = false;
		m_gpu_written = false;
	}
	// Reads the backing store without faulting. Strict reads are checked the way specialization
	// memory is read (GPU-owned bytes fail), the others as plain loads.
	bool               Read(uint64_t address, void* data, uint64_t size, bool strict);
	// A plain load the SRT walker made.
	void               RecordLoad(uint64_t address, uint32_t value);
	void               Fail() { m_failed = true; }
	[[nodiscard]] bool Failed() const { return m_failed; }
	// Whether some read may have taken stale bytes of memory the GPU wrote (the backing store
	// gets them only once read back).
	[[nodiscard]] bool MayHaveReadGpuWrites() const { return m_gpu_written; }
	// Execution thread: whether every read still gives the recorded bytes.
	[[nodiscard]] bool StillValid() const;

private:
	static constexpr uint64_t StrictCheckBytes = 64;
	struct Entry {
		uint64_t address = 0;
		uint32_t size    = 0;
		uint32_t offset  = 0;
		bool     strict  = false;
	};
	void                 Append(uint64_t address, const void* data, uint32_t size, bool strict);
	std::vector<Entry>   m_entries;
	std::vector<uint8_t> m_bytes;
	bool                 m_failed      = false;
	bool                 m_gpu_written = false;
};

// Storage of the shader programs a draw resolved ahead of its execution: the resource snapshots
// and specializations its stages point to, and the guest memory they read.
struct ProgramResolution {
	static constexpr uint32_t MaxStages = 4;

	std::array<ShaderRecompiler::IR::ResourceSnapshot, MaxStages>       resources;
	std::array<ShaderRecompiler::IR::ResourceSpecialization, MaxStages> specializations;
	GuestReadLog                                                        reads;
	uint32_t                                                            stages = 0;

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
class UserConfig;
class Shader;
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
	// 0 without tessellation, 3 for a rectangle list, else the input control points.
	uint32_t                   patch_control_points     = 0;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       alpha_to_coverage_enable = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
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
};

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride   = 0;
		uint8_t  instance = 0;
	};
	struct Attribute {
		uint32_t offset  = 0;
		uint8_t  binding = 0;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;
};

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;
	uint32_t         spirv_words = 0;
	uint64_t         hash        = 0;

	bool operator==(const GraphicsPipelineKey& other) const;
};

#pragma pack(pop)

// Keys are compared and hashed as bytes.
static_assert(std::has_unique_object_representations_v<GraphicsPipelineKey>);

// A compiled stage with what pipeline creation needs from it besides the key.
struct ShaderModule {
	uint64_t                                     id     = 0;
	vk::ShaderModule                             module = nullptr;
	ShaderRecompiler::IR::CompiledShaderInfo     info;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	std::vector<uint32_t>                        function_code;
	// VS or LS: the host format of each vertex input attribute.
	std::vector<vk::Format>                      vertex_formats;
	// Rectangle-list PS: the parameters it reads.
	std::vector<RectListParameter>               rect_parameters;
};

// A draw's topology, shader stages (from PipelineCache::GetGraphicsPrograms) and targets.
struct DrawRenderState {
	RenderDepthInfo                      depth_info;
	RenderColorInfo                      color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t                             color_count                              = 0;
	// Color targets the pixel program writes.
	uint32_t                             mrt_mask                                 = 0;
	vk::PrimitiveTopology                topology = vk::PrimitiveTopology::ePointList;
	bool                                 primitive_restart_enable                 = false;
	bool                                 ps_active                                = false;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo                 ps_input_info;
	// Input: the user-data slots a GPU-read indirect draw patches.
	ShaderIndirectOffsets                indirect_offsets;
	// The targets of the draw: color_info and depth_info, or those of the render target memo
	// when the draw reuses the targets of the draw before.
	RenderColorInfo*                     colors = color_info;
	RenderDepthInfo*                     depth  = &depth_info;

	[[nodiscard]] std::span<const ShaderVertexInputInfo> VertexStages() const {
		return {vertex_info.data(), vertex_info[1].stage ? 3u : 1u};
	}
};

// Writes the shader store on its own thread, running jobs in the order they were posted.
class ShaderStoreWriter {
public:
	// Keeps the first `kept` bytes of the store at `path`, or starts it over with `header`. The
	// thread runs only if that succeeded.
	void Open(const std::filesystem::path& path, std::span<const uint8_t> header, uint64_t kept);
	explicit operator bool() const { return m_thread.joinable(); }
	// Compresses `record` and appends it on the store thread, in call order.
	void Append(std::vector<uint8_t>&& record);

private:
	friend struct PipelineCacheTestAccess;

	void Post(Common::UniqueFunction<void>&& job);

	Common::File                             m_file;
	std::mutex                               m_mutex;
	std::condition_variable_any              m_ready;
	std::queue<Common::UniqueFunction<void>> m_jobs;
	// Declared last, so it drains the queue and joins before the rest is destroyed.
	std::jthread m_thread;
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
	// Without a shader store path nothing is kept across runs.
	PipelineCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap,
	              const std::filesystem::path& shader_store = {});
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

	// Reads the shaders and registers bound in `command` into the state's shader stages.
	void GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state);
	// The resolve thread: resolves the programs of a draw ahead of its execution, with programs
	// and modules that exist, into `ahead`, reading guest memory through its log. False when the
	// execution thread resolves the draw itself; programs it misses are queued to the workers.
	bool GetGraphicsProgramsAhead(const HW::Context& context, const HW::UserConfig& user_config,
	                              const HW::Shader& shaders, DrawRenderState& state,
	                              ProgramResolution& ahead);
	// Null skips the dispatch.
	const ShaderModule* GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                      ShaderComputeInputInfo&      input_info);
	// The resolve thread: as GetGraphicsProgramsAhead, for a dispatch; null when the execution
	// thread resolves it itself.
	const ShaderModule* GetComputeProgramAhead(const HW::ComputeShaderInfo& regs,
	                                           ShaderComputeInputInfo&      input_info,
	                                           ProgramResolution&           ahead);

		[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] ? 3u : 1u; }
	};

	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                    ShaderPixelInputInfo& pixel_info);
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
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info);
	// Whether the shader at this address is on the skip list (--skip-shaders): its dispatches can
	// be dropped before anything is prepared for them.
	[[nodiscard]] static bool IsSkipListed(uint64_t shader_addr);
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
	friend struct PipelineCacheTestAccess;

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

		[[nodiscard]] std::span<const uint32_t> FunctionCode() const;
		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime);
		template <bool HasCalls>
		ShaderModule* FindModule(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                         uint32_t                                push_data_cursor);
	};

	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	// Variants of one source share a bucket; ProgramKey equality compares the state.
	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const { return key.hash; }
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	using ModuleMap = std::unordered_map<uint64_t, const ShaderModule*>;

	GraphicContext&                        m_graphics;
	ShaderRecompiler::IR::ImageHeapSource* m_image_heap;
	std::unordered_map<ProgramKey, Program, ProgramKeyHash> m_program_cache;
	// Modules compiled per stage.
	std::array<uint32_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> m_compiled {};
	ProgramKey                             m_lookup_key;
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
	std::unique_ptr<PipelineCompiler>                       m_compiler;
	uint64_t                                                m_deferred_draws = 0;

	// Graphics programs compiled on worker threads: the resolve thread queues the programs and
	// modules its draws miss, so the shaders of a new area compile in parallel instead of one
	// after another on the execution thread. Workers only add programs; the execution thread
	// still resolves every draw itself and compiles what they did not add.
	struct CompileJob;
	static constexpr uint32_t CompileWorkers = 6;
	// Under the program lock: programs a worker is compiling, with their number of jobs.
	std::unordered_map<ProgramKey, uint32_t, ProgramKeyHash> m_in_flight;
	std::mutex                                               m_job_mutex;
	std::condition_variable                                  m_job_available;
	std::deque<std::unique_ptr<CompileJob>>                  m_jobs;
	bool                                                     m_stopping = false;
	std::vector<std::jthread>                                m_workers;

	// Records everything first created; closed without a shader store.
	ShaderStoreWriter m_store;

	bool ResolveGraphicsPrograms(const HW::Context& context, const HW::UserConfig& user_config,
	                             const HW::Shader& shaders, DrawRenderState& state,
	                             ProgramResolution* ahead);
	template <typename InputInfo>
	const ShaderModule* GetProgram(const ShaderParams& params, InputInfo& input_info,
	                               uint32_t& push_data_cursor);
	template <typename InputInfo>
	const ShaderModule* GetAhead(const ShaderParams& params, InputInfo& input_info,
	                             uint32_t& push_data_cursor, ProgramResolution& ahead);
	ShaderModule& CompileModule(Program& program, const ProgramKey& key,
	                            ShaderRecompiler::CompileOptions  options,
	                            ShaderRecompiler::TranslateResult translated,
	                            uint32_t                          push_data_cursor);
	// Compiles a module without touching the cache; `spirv` receives its code.
	ShaderModule BuildModule(const ProgramKey& key, const ShaderRecompiler::CompileOptions& options,
	                         ShaderRecompiler::TranslateResult                   translated,
	                         const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                         std::span<const uint32_t> function_code, uint32_t push_data_cursor,
	                         std::vector<uint32_t>& spirv) const;
	// Under the program lock: counts a module added to `program` and records it in the store.
	void AddedModule(const ProgramKey& key, const Program& program, const ShaderModule& module,
	                 const std::vector<uint32_t>& spirv);
	void LockPrograms();
	void UnlockPrograms();
	// Under the program lock (resolve thread): queues the program of m_ahead_key, or one of its
	// modules, to the compile workers.
	template <typename InputInfo>
	void QueueCompile(const ShaderParams& params, const InputInfo& input_info,
	                  uint32_t                                            push_data_cursor,
	                  const ShaderRecompiler::IR::ResourceSpecialization* specialization);
	// Execution thread, under the program lock: waits without the lock until no worker compiles
	// the program of `key`. False when none did.
	bool WaitForCompile(const ProgramKey& key);
	template <typename InputInfo>
	void RunCompileJob(CompileJob& job, InputInfo& input_info);
	void CompileWorker();
	void StopWorkers();

	// Pipelines new to the session, created on background threads instead of stalling the
	// execution thread: graphics pipelines of draws that may be skipped until they are there,
	// and compute pipelines of modules the compile workers compiled ahead of their dispatch.
	// Finished pipelines wait in m_async_ready until a lookup takes them; the pending keys are
	// under m_async_mutex.
	struct AsyncPipelineJob {
		bool                               compute = false;
		GraphicsPipelineKey                key {};
		std::array<const ShaderModule*, 4> stages {};
		const ShaderModule*                module = nullptr;
	};
	std::unordered_set<GraphicsPipelineKey, GraphicsPipelineKeyHash> m_async_pending_graphics;
	std::unordered_set<uint64_t>                                     m_async_pending_compute;
	// Keys a background thread failed to create: their draws create them and report the error.
	std::unordered_set<GraphicsPipelineKey, GraphicsPipelineKeyHash> m_async_failed_graphics;
	std::unordered_map<GraphicsPipelineKey, Pipeline, GraphicsPipelineKeyHash> m_async_ready;
	std::unordered_map<uint64_t, Pipeline>                                     m_async_ready_compute;
	std::mutex                                                       m_async_mutex;
	std::condition_variable                                          m_async_available;
	std::deque<AsyncPipelineJob>                                     m_async_jobs;
	bool                                                             m_async_stopping = false;
	std::vector<std::jthread>                                        m_async_threads;

	void QueueAsyncPipeline(const AsyncPipelineJob& job);
	// Under m_async_mutex: removes a queued job no thread has started; false when none matches.
	template <typename Matches>
	bool TakeQueuedAsyncPipeline(Matches&& matches);
	void AsyncPipelineWorker();
	void StopAsyncPipelines();
	// A compile worker added a compute module: its pipeline is created in the background.
	void CreateComputePipelineAhead(const ShaderModule& module);
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
                            const GraphicsPipelineKey&                 key,
                            const std::array<const ShaderModule*, 4>& stages,
                            vk::PipelineCache                          driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderModule& module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
