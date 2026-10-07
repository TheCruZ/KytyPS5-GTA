#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
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
struct PipelineStoreState;
struct StoredPipeline;
struct RenderColorInfo;
struct RenderDepthInfo;
class CommandBuffer;

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
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
static_assert(sizeof(PipelineStaticParameters) == 116);

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

	// Entries past the counts keep their default values: only the used ones are compared (draws
	// compare the key of the last pipeline every time).
	bool operator==(const PipelineVertexInputState& other) const {
		return binding_count == other.binding_count && attribute_count == other.attribute_count &&
		       std::equal(bindings.begin(), bindings.begin() + binding_count,
		                  other.bindings.begin()) &&
		       std::equal(attributes.begin(), attributes.begin() + attribute_count,
		                  other.attributes.begin());
	}
};

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

// The owning renderer serializes access, including saves while the GPU is running.
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
	// With `ahead`, only a program and permutation that exist, reading guest memory through the
	// resolution's log (see GetGraphicsPrograms).
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info,
	                                ProgramResolution*           ahead = nullptr);

	// With `allow_async`, a pipeline that does not exist yet is created on a background thread
	// and the lookup returns null until it is there (the draw can be skipped meanwhile);
	// otherwise the lookup creates the pipeline, or waits for a background thread creating it.
	Pipeline* GetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                              const RenderDepthInfo&                 depth,
	                              std::span<const ShaderVertexInputInfo> vertex_info,
	                              CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs, bool allow_async);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info,
	                             const ShaderProgram&          compute_program);

private:
	struct ProgramCache;

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
	// Draws resolve their programs on the resolve thread and, when a resolution is not usable,
	// on the execution thread, which also resolves dispatches.
	std::atomic_flag              m_program_lock;
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

	// The pipelines of earlier sessions (see PipelineStore): background threads create them once
	// the program store compiled their programs, and the execution thread takes them from here
	// when a lookup misses.
	std::unique_ptr<PipelineStoreState>        m_stored;
	std::mutex                                 m_precreated_mutex;
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_precreated_graphics;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_precreated_compute;
	std::vector<std::jthread>                               m_precreate_threads;

	// Pipelines new to every store, created on background threads instead of stalling the
	// execution thread: graphics pipelines of draws that may be skipped until they are there,
	// and compute pipelines of programs the compile workers translated ahead of their dispatch.
	// Finished pipelines join the precreated ones; the pending keys are under
	// m_precreated_mutex.
	struct AsyncPipelineJob;
	std::unordered_set<GraphicsPipelineKey, GraphicsPipelineKeyHash> m_async_pending_graphics;
	std::unordered_set<uint64_t>                                     m_async_pending_compute;
	// Keys a background thread failed to create: their draws create them and report the error.
	std::unordered_set<GraphicsPipelineKey, GraphicsPipelineKeyHash> m_async_failed_graphics;
	std::mutex                                                       m_async_mutex;
	std::condition_variable                                          m_async_available;
	std::deque<std::unique_ptr<AsyncPipelineJob>>                    m_async_jobs;
	bool                                                             m_async_stopping = false;
	std::vector<std::jthread>                                        m_async_threads;

	void QueueAsyncPipeline(std::unique_ptr<AsyncPipelineJob> job);
	// Removes a queued job no thread has started; false when none matches.
	bool TakeQueuedAsyncPipeline(const std::function<bool(const AsyncPipelineJob&)>& matches);
	void AsyncPipelineWorker();
	void StopAsyncPipelines();
	// Called by a compile worker when it added a compute permutation.
	void CreateComputePipelineAhead(const ShaderComputeInputInfo& input_info,
	                                const ShaderProgram&          program);

	void OpenStores(const std::string& title_id);
	void PrecreatePipelines(const std::stop_token& stop);
	bool PrecreatePipeline(const StoredPipeline& stored);
	void RecordGraphicsPipeline(const GraphicsPipelineKey&             key,
	                            std::span<const ShaderVertexInputInfo> vertex_info,
	                            const ShaderPixelInputInfo*            ps_input_info);
	void RecordComputePipeline(const ShaderComputeInputInfo& input_info, uint64_t program_id);

	void InitializeDriverCache();
	// Writes the driver cache data to m_driver_cache_path; the cache must stay alive meanwhile.
	void WriteDriverCache();
};

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
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
