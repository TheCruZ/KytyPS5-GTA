#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/indirectDispatch.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <source_location>
#include <unordered_set>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawRenderState;
class RenderContext;
class CommandScheduler;
class CommandRecorder;
struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	DispatchIndirect = 9,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// The draw records of a DRAW_INDEX_INDIRECT/DRAW_INDIRECT(_MULTI) packet left in guest memory for
// the host GPU to read (GPU work writes them: culling compute shaders, so reading them on the CPU
// would wait for the GPU to finish everything recorded before the draw). Nonzero address: the GPU
// reads count Vulkan indirect draw records, stride bytes apart.
struct DrawIndirectArgs {
	uint64_t address = 0;
	// The draw count, or the most the GPU may draw when count_addr is set.
	uint32_t count   = 0;
	uint32_t stride  = 0;
	// Nonzero: the GPU reads the draw count from this guest dword (at most `count` draws).
	uint64_t count_addr = 0;
	// Indexed draws: the bytes of the index buffer the draws may read from the index base.
	uint64_t index_bytes = 0;
};

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// With indirect records, index_count is INDEX_BUFFER_SIZE.
	DrawIndirectArgs indirect;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// DRAW_INDIRECT read on the GPU: the guest address of its vk::DrawIndirectCommand (vertex
	// count, instance count, start vertex, start instance). The counts above are then unknown on
	// the CPU; only fast-launch mesh draws come this way.
	uint64_t         gpu_args_address           = 0;
	// Nonzero (vertex count known): the instance count is the dword the GPU reads there (the
	// NUM_INSTANCES a DRAW_INDIRECT read on the GPU left).
	uint64_t         gpu_instance_count_address = 0;
	// With gpu_args_address: the user SGPR (index + 1, 0 = none) START_INST_LOC names, which
	// the shader loads from the arguments' start instance (the CP writes it on hardware).
	uint32_t         start_instance_sgpr        = 0;
	DrawIndirectArgs indirect;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	void EndRendering(std::source_location site = std::source_location::current()) const;
	[[nodiscard]] bool IsRendering() const noexcept { return m_rendering; }

	// The raw command buffer. With the recording thread (CommandScheduler::EnableRecordingThread)
	// this first waits until the thread has run everything queued; hot paths use Recorder().
	[[nodiscard]] vk::CommandBuffer Handle() const;
	// A pipeline outside the draws (the present overlay) replaced the dynamic state they set.
	void InvalidateDynamicState() const;
	// Records directly, or through the recording thread.
	[[nodiscard]] CommandRecorder Recorder() const;
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] const HW::Context&    GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] const HW::UserConfig& GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] const HW::Shader&     GetShaders() const noexcept { return *m_shaders; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(const HW::Context& registers, const HW::UserConfig& user_config,
	          const HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	CommandScheduler&   m_scheduler;
	// Begun and not submitted yet; m_buffer is its handle when not recorded on a thread.
	bool                m_open            = false;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	const HW::Context*    m_registers   = nullptr;
	const HW::UserConfig* m_user_config = nullptr;
	const HW::Shader*     m_shaders     = nullptr;

	friend class CommandScheduler;
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	// A draw window is the run of draws executed since the last other operation (a label,
	// interrupt, flip, host command or a WAIT_REG_MEM): the guest cannot have overwritten what
	// those draws read, as it learns the GPU consumed it only through such an operation.
	[[nodiscard]] uint64_t DrawWindow() const noexcept { return m_draw_window; }
	void                   EndDrawWindow() noexcept { m_draw_window++; }
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode, bool async_compute);
	// Whether an indirect draw with the bound registers can leave its arguments to the GPU
	// (see DrawIndirectArgs); `index_type` is the guest index type of an indexed draw.
	[[nodiscard]] bool CanDrawIndirectOnGpu(const CommandBuffer& buffer, bool indexed,
	                                        uint32_t index_type, uint64_t index_bytes) const;
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode, bool async_compute);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	// Bindless heaps: register the draw's heaps, patch their regions, and once per frame resolve
	// the keys shaders flagged as pending.
	void PrepareBindlessHeaps(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void ResolveBindlessRequests();
	// Engine method (engine-method.md, E2), at the start of a guest submission: the heap entries
	// the guest wrote since the last one are compared and their keys settled again.
	void SyncBindlessHeaps();
	void SyncWrittenBindlessEntries();
	// E3: settle entries no draw has asked for yet, within the per-submission budget.
	void SettleBindlessHeaps();
	bool ResolveBindlessKey(BindlessTable::Heap& heap, uint32_t key);
	// The key's T# changed: release what it was settled to and settle it again.
	bool ResettleBindlessKey(BindlessTable::Heap& heap, uint32_t key);
	// Release only: the key is pending again and its image unpinned when nothing else uses it.
	void ReleaseBindlessKey(BindlessTable::Heap& heap, uint32_t key);
	// Settles again, within the budget, the keys of a window of each heap whose T# changed since
	// they were settled; returns how many it resolved.
	uint32_t RevalidateBindlessKeys(uint32_t budget);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);

private:
	// False, with nothing drawn, when the draw's indirect records need CPU counts.
	bool DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	// Bindless table elements resolved for one root image, by T#. Most of a table is unchanged
	// from dispatch to dispatch, and resolving an element repeats the same texture-cache lookup
	// while no image in its range was registered or unregistered and the guest backing did not
	// change. Elements with DCC metadata are resolved again on every use: their lookup
	// materializes fast clears.
	struct TableElementResolution {
		TextureBinding texture;
		GuestRange     range; // Guest memory whose images decide the lookup, if any.
		uint64_t       image_epoch   = 0;
		uint64_t       backing_epoch = 0;
		// The global image-set epoch at which the element was last known valid: while it holds,
		// no image anywhere changed and the range needs no check.
		uint64_t       checked_epoch = 0;
		bool           permanent     = false; // Null for the root's view, whatever the memory.
		bool           volatile_dcc  = false;
	};
	struct TableDescriptorHash {
		size_t operator()(const std::array<uint32_t, 8>& dwords) const noexcept;
	};
	struct TableResolution {
		ShaderRecompiler::IR::ImageResource root;
		uint64_t                            last_use = 0;
		std::unordered_map<std::array<uint32_t, 8>, TableElementResolution, TableDescriptorHash>
		    elements;
	};

	[[nodiscard]] const TextureBinding&
	ResolveTableElement(TableResolution& table, const ShaderRecompiler::IR::DescriptorValue& value);
	[[nodiscard]] TableResolution& FindTableResolution(const ShaderRecompiler::IR::ImageResource& root);

	// table_candidate rejects (EXIT) a texture whose guest memory cannot be read, for tables
	// that keep T#s of freed textures, and declines depth/color conversions. examined receives
	// the guest range before the lookup depends on the texture cache: a failure without it
	// depends only on the T# and the guest backing.
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value,
	                                            bool        table_candidate = false,
	                                            GuestRange* examined        = nullptr);
	[[nodiscard]] TextureBinding FindResolvedTexture(
	    const ShaderRecompiler::IR::ImageResource& resource, const ShaderTextureResource& descriptor,
	    TextureCache::ImageDesc desc, bool shader_conversion, vk::Format pixel_format,
	    vk::Format view_format, uint32_t size, bool table_candidate = false,
	    GuestRange* examined = nullptr);
	[[nodiscard]] TextureBinding
	ResolveTableTexture(const ShaderRecompiler::IR::ImageResource&   root,
	                    const ShaderRecompiler::IR::DescriptorValue& value,
	                    GuestRange*                                  examined = nullptr);
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState& state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer,
	                                                       uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeFill(const ShaderComputeInputInfo& input,
	                                        CommandBuffer& command, uint32_t group_x,
	                                        uint32_t group_y, uint32_t group_z, uint32_t mode);

	RenderContext&                        m_context;
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	// Bindless heaps already surveyed (base ^ table offset << 48).
	std::unordered_set<uint64_t>          m_bindless_surveyed;

	void PrepareBindlessSamplers(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void LogWatchedDraw(const DrawCallInfo& draw, const DrawRenderState& state,
	                    uint32_t index_offset, int32_t base_vertex, uint32_t first_vertex,
	                    int32_t host_vertex_offset, uint32_t host_first_instance, bool indirect);
	uint64_t                              m_bindless_frame = UINT64_MAX;
	std::vector<uint32_t>                 m_bindless_requests;
	std::vector<std::array<uint32_t, 8>>  m_bindless_window;
	std::vector<std::pair<uint64_t, uint64_t>> m_bindless_written;
	std::vector<ImageId>                       m_bindless_changed;
	uint64_t                                   m_bindless_eager_frame = UINT64_MAX;
	uint32_t                                   m_bindless_eager_left  = 0;
	bool                                       m_bindless_eager_device_full = false;
	uint64_t                                   m_bindless_eager_bytes = 0;
	// Bindless usage probe (UpdateBindlessUsageProbe): the frame it was armed in (0 = idle), the
	// first frame the next may start, and the frame the pending feedback snapshot was recorded.
	uint64_t                              m_bindless_probe_frame      = 0;
	uint64_t                              m_bindless_next_probe       = 0;
	uint64_t                              m_bindless_snapshot_frame   = 0;
	std::vector<ImageId>                  m_bindless_unused;
	void UpdateBindlessUsageProbe(uint64_t frame, uint64_t snapshot_frame);
	std::vector<uint32_t>                 m_bindless_srt;
	std::vector<uint32_t>                 m_verify_srt; // KYTY_SRT_VERIFY: flat SRT with pairs
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	// Created at the first thread-dimension indirect dispatch.
	std::unique_ptr<IndirectDispatchGroups> m_indirect_groups;
	// Created at the first indirect fast-launch mesh draw with counts on the GPU.
	std::unique_ptr<MeshIndirectArgs>       m_mesh_args;
	// Ring of VkDrawMeshTasksIndirectCommandEXT for indirect fast-launch mesh draws (16 bytes an
	// entry); an entry is rewritten only after the ring has gone round, behind its last read.
	std::unique_ptr<Buffer>                 m_mesh_indirect;
	uint32_t                                m_mesh_indirect_next = 0;
	std::vector<TableResolution>          m_table_resolutions;
	uint64_t                              m_table_resolution_tick = 0;
	uint64_t                              m_draw_window           = 0;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
