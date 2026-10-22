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
#include <deque>
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
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
class RenderContext;
class CommandScheduler;
class CommandRecorder;
struct RenderExecutorTestAccess;
struct IndirectThreadPass;

// An image description derived from a T# and a shader's view of it (see ResolveTexture).
struct TextureDescDerivation {
	TextureCache::ImageDesc desc;
	bool                    shader_conversion = false;
	vk::Format              pixel_format      = vk::Format::eUndefined;
	vk::Format              view_format       = vk::Format::eUndefined;
	uint64_t                size              = 0;
	// The image the last lookup of a sampled texture with this description found, the
	// description it was found with and the view acquired for it. Reused while no image on the
	// pages of the range was registered or unregistered since and the image keeps the state the
	// acquisition left it in.
	struct Found {
		ImageId                 id;
		TextureCache::ImageDesc desc;
		vk::ImageView           view            = nullptr;
		uint64_t                image_set_epoch = 0;
		ImageLookupState        state;
		bool                    valid = false;
	};
	Found found;
};

// An image description derived from a T# and a shader's view of it (see ResolveTexture).
struct TextureDescDerivation {
	TextureCache::ImageDesc desc;
	bool                    shader_conversion = false;
	vk::Format              pixel_format      = vk::Format::eUndefined;
	vk::Format              view_format       = vk::Format::eUndefined;
	uint64_t                size              = 0;
	// The image the last lookup of a sampled texture with this description found, the
	// description it was found with and the view acquired for it. Reused while no image on the
	// pages of the range was registered or unregistered since and the image keeps the state the
	// acquisition left it in.
	struct Found {
		ImageId                 id;
		TextureCache::ImageDesc desc;
		vk::ImageView           view            = nullptr;
		uint64_t                image_set_epoch = 0;
		ImageLookupState        state;
		bool                    valid = false;
	};
	Found found;
};

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

	CommandScheduler&   m_scheduler;
	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	CommandScheduler&   m_scheduler;
	// Begun and not submitted yet; m_buffer is its handle when not recorded on a thread.
	bool                m_open            = false;
	vk::CommandBuffer   m_buffer          = nullptr;
	CommandStream*      m_stream          = nullptr;
	bool                m_open            = false;
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
	void                   InvalidateRenderTargetMemo() noexcept;
	// Also ends the reuse of the render targets of the draw before.
	void                   EndDrawWindow() noexcept;
	// Shader programs and resources of a draw resolved ahead of its execution
	// (KYTY_PIPELINE_CACHE=ours).
	struct ResolvedDraw;
	// Resolve thread: resolves the shader programs of a draw with the given registers ahead of
	// its execution; null when the draw resolves them itself (tessellation, programs that are not
	// compiled yet, guest memory that cannot be read ahead). Waits while every resolution is in
	// use: the execution thread releases them in order.
	[[nodiscard]] ResolvedDraw* ResolveDrawAhead(const HW::Context&    context,
	                                             const HW::UserConfig& user_config,
	                                             const HW::Shader&     shaders);
	// Resolve thread: the compute program of a DISPATCH_DIRECT with the given registers and
	// initiator; null when the dispatch resolves it itself. Shares the ring of draw resolutions.
	[[nodiscard]] ResolvedDraw* ResolveDispatchAhead(const HW::Context& context,
	                                                 const HW::Shader& shaders,
	                                                 const uint32_t (&groups)[3], uint32_t mode,
	                                                 bool async_compute);
	// Resolve thread: the compute program of a DISPATCH_INDIRECT with the given registers and
	// initiator, which does not depend on the arguments the dispatch reads when it executes.
	[[nodiscard]] ResolvedDraw* ResolveIndirectDispatchAhead(const HW::Context& context,
	                                                         const HW::Shader&  shaders,
	                                                         uint32_t           mode, uint64_t args_addr,
	                                                         bool               async_compute);
	// Execution thread: the program a dispatch resolved ahead, when the guest memory the
	// resolution read still holds the same bytes.
	bool TakeResolvedDispatch(ShaderComputeInputInfo& input_info, ShaderProgram& program);
	// Execution thread: the next draw uses `resolved` when the guest memory it read still holds
	// the same bytes.
	void UseResolvedDraw(ResolvedDraw* resolved) noexcept { m_resolved_draw = resolved; }
	// Execution thread: the operation that carried `resolved` executed.
	static void ReleaseResolvedDraw(ResolvedDraw* resolved) noexcept;
	// Validation thread: checks the guest reads of `resolved` ahead of the execution thread.
	static void PreValidateResolvedDraw(ResolvedDraw* resolved);
	// The context snapshot the operation about to execute reads (0: the live registers).
	void UseContextId(uint64_t id) noexcept { m_context_id = id; }
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode, bool async_compute);
	// Whether an indirect draw with the bound registers can leave its arguments to the GPU
	// (see DrawIndirectArgs); `index_type` is the guest index type of an indexed draw.
	[[nodiscard]] bool CanDrawIndirectOnGpu(const CommandBuffer& buffer, bool indexed,
	                                        uint32_t index_type, uint64_t index_bytes) const;
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode, bool async_compute);
	// DISPATCH_INDIRECT with USE_THREAD_DIMENSIONS: converts the thread counts, which GPU work
	// writes, on the GPU instead of reading them back.
	void DispatchIndirectThreads(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                             uint32_t mode, bool async_compute);

	// Shader programs and resources of a draw resolved ahead of its execution.
	struct ResolvedDraw;
	// Resolve thread: resolves the shader programs of a draw with the given registers ahead of
	// its execution; null when the draw resolves them itself (tessellation, programs that are not
	// compiled yet, guest memory that cannot be read ahead). Waits while every resolution is in
	// use: the execution thread releases them in order.
	[[nodiscard]] ResolvedDraw* ResolveDrawAhead(const HW::Context&    context,
	                                             const HW::UserConfig& user_config,
	                                             const HW::Shader&     shaders,
	                                             const ShaderIndirectOffsets& indirect_offsets = {});
	// Resolve thread: the compute program of a DISPATCH_DIRECT with the given registers and
	// initiator; null when the dispatch resolves it itself. Shares the ring of draw resolutions.
	[[nodiscard]] ResolvedDraw* ResolveDispatchAhead(const HW::Context& context,
	                                                 const HW::Shader& shaders,
	                                                 const uint32_t (&groups)[3], uint32_t mode,
	                                                 bool async_compute);
	// Resolve thread: the compute program of a DISPATCH_INDIRECT with the given registers and
	// initiator, which does not depend on the arguments the dispatch reads when it executes.
	[[nodiscard]] ResolvedDraw* ResolveIndirectDispatchAhead(const HW::Context& context,
	                                                         const HW::Shader&  shaders,
	                                                         uint32_t           mode, uint64_t args_addr,
	                                                         bool               async_compute);
	// Execution thread: the program a dispatch resolved ahead, when the guest memory the
	// resolution read still holds the same bytes.
	bool TakeResolvedDispatch(ShaderComputeInputInfo& input_info, const ShaderModule*& module);
	// Execution thread: the next draw uses `resolved` when the guest memory it read still holds
	// the same bytes.
	void UseResolvedDraw(ResolvedDraw* resolved) noexcept { m_resolved_draw = resolved; }
	// Execution thread: the operation that carried `resolved` executed.
	static void ReleaseResolvedDraw(ResolvedDraw* resolved) noexcept;
	// Execution thread: the context register snapshot of the operation about to execute (0 for
	// live registers); operations with the same id see the same context registers.
	void UseContextId(uint64_t context_id) noexcept { m_context_id = context_id; }
	// Execution thread: an operation of guest queue `queue` is about to execute. The barrier
	// after a direct dispatch is deferred to the next operation, unless that is a direct dispatch
	// of the same queue the guest did not order after it (see DispatchDirect()).
	void BeginGuestOperation(uint32_t queue, bool dispatch_direct, bool guest_sync);
	// Records the barrier a direct dispatch deferred, if any.
	void FlushDeferredDispatchBarrier();
	// Execution thread: an operation other than a draw is about to execute (see
	// RenderTargetMemo).
	void InvalidateRenderTargetMemo() noexcept;
	// Advances whenever an operation other than a draw executes (see
	// InvalidateRenderTargetMemo()): within one value only draws executed, so no guest-visible
	// effect let the guest write the memory they read.
	[[nodiscard]] uint64_t DrawWindow() const noexcept { return m_draw_window; }

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void PrepareBindingDiagnostics(const ShaderStageRuntime& runtime);
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
	void                           FindBuffers(std::span<PreparedBindings* const> stages,
	                                           bool                               find_buffers = true);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	// table_candidate rejects (EXIT) a texture whose guest memory cannot be read, for tables
	// that keep T#s of freed textures, and declines depth/color conversions. examined receives
	// the guest range before the lookup depends on the texture cache: a failure without it
	// depends only on the T# and the guest backing.
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value,
	                                            bool        table_candidate = false,
	                                            GuestRange* examined        = nullptr);

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
		// What every preparation of a table reads comes first, on one cache line: a table names
		// thousands of elements.
		// The PrepareBindings() call that last bound the element, and its index there.
		uint64_t       prepared_stamp = 0;
		uint32_t       prepared_index = 0;
		bool           permanent     = false; // Null for the root's view, whatever the memory.
		bool           volatile_dcc  = false;
		// A resolved element depends on the guest backing only through whether its data and
		// metadata are readable: when the backing epoch moves (any map or unmap anywhere,
		// constantly while a game streams), the element stays valid while that answer holds.
		bool           backing_checkable = false;
		bool           backing_readable  = false;
		uint64_t       image_epoch   = 0;
		uint64_t       backing_epoch = 0;
		// The global image-set epoch at which the element was last known valid: while it holds,
		// no image anywhere changed and the range needs no check.
		uint64_t       checked_epoch = 0;
		GuestRange     range; // Guest memory whose images decide the lookup, if any.
		GuestRange     metadata_range;
		TableViewMemo  view;
		TextureBinding texture;
	};
	struct TableDescriptorHash {
		size_t operator()(const std::array<uint32_t, 8>& dwords) const noexcept;
	};
	using TableElements =
	    std::unordered_map<std::array<uint32_t, 8>, TableElementResolution, TableDescriptorHash>;
	// The element the last preparation of a root found for one slot: a slot whose T# did not
	// change skips the lookup by T#.
	struct TableSlot {
		std::array<uint32_t, 8> dwords {};
		TableElementResolution* element = nullptr;
	};
	struct TableResolution {
		ShaderRecompiler::IR::ImageResource root;
		uint64_t                            last_use = 0;
		TableElements                       elements;
		std::vector<TableSlot>              slots;
	};

	// `stamp` identifies the PrepareBindings() call: an element it already resolved is not
	// resolved again for another slot. `known` is the element of `value`, when the caller has it.
	[[nodiscard]] TableElementResolution&
	ResolveTableElement(TableResolution& table, const ShaderRecompiler::IR::DescriptorValue& value,
	                    uint64_t stamp, TableElementResolution* known = nullptr);
	[[nodiscard]] TableResolution& FindTableResolution(const ShaderRecompiler::IR::ImageResource& root);
	[[nodiscard]] bool TableElementBackingReadable(const TableElementResolution& element);

	// The bindless tables a program bound last. GTA V's ray tracing dispatches bind ~16k slots
	// naming ~8k textures each, and almost all of them stay the same from frame to frame: while
	// the T#s of every slot are unchanged, a binding of the program revisits only the elements
	// that an image registration or a guest backing change may have affected, and the elements
	// whose image is no longer bound as before (see ReuseTableBindings()). The images of the
	// elements stay pinned in the texture cache meanwhile: they are not touched every binding.
	struct TableBindingState {
		struct Range {
			ShaderRecompiler::IR::ImageResource root;
			TableResolution*                    resolution = nullptr;
			uint32_t                            table      = 0;
			uint32_t                            capacity   = 0;
		};
		struct Element {
			TableElementResolution*               resolution = nullptr;
			TableResolution*                      table      = nullptr;
			ShaderRecompiler::IR::DescriptorValue source;
			uint32_t                              root = 0;
			// Image the element is pinned to.
			ImageId                               pinned;
			// Visited by every binding (DCC metadata), or whenever the guest backing changes
			// (the element's resolution does not record whether its backing is readable).
			bool always            = false;
			bool unchecked_backing = false;
			// The ranges `spans` holds for the element.
			GuestRange span_range;
			GuestRange span_metadata;
		};
		// The guest pages (in texture-cache page granularity) of an element's data or metadata.
		struct PageSpan {
			uint64_t first   = 0;
			uint64_t last    = 0;
			uint32_t element = 0;
		};
		std::vector<Range>                   ranges;
		// The T# of every slot of every range, in binding order.
		std::vector<ShaderRecompiler::IR::DescriptorValue> values;
		std::vector<Element>                 elements;
		// The element of every slot.
		std::vector<uint32_t>                slots;
		// The descriptor of every element, and of every slot.
		std::vector<vk::DescriptorImageInfo> infos;
		std::vector<vk::DescriptorImageInfo> slot_infos;
		// The slots of each element: element_slots[element_slot_begin[e] .. [e + 1]).
		std::vector<uint32_t>                element_slot_begin;
		std::vector<uint32_t>                element_slots;
		// The element of each element resolution.
		std::unordered_map<const TableElementResolution*, uint32_t> element_index;
		// Element spans by first page; `max_span` is the longest (last - first).
		std::vector<PageSpan>                spans;
		uint64_t                             max_span = 0;
		// Elements visited by every binding, and whenever the guest backing changes.
		std::vector<uint32_t>                always;
		std::vector<uint32_t>                unchecked_backing;
		// Elements marked for a visit by this binding.
		std::vector<uint64_t>                marks;
		uint64_t                             mark_stamp = 0;
		uint64_t                             generation = 0;
		uint64_t                             set_epoch  = 0;
		uint64_t                             backing_epoch = 0;
		bool                                 valid = false;
	};
	// Prepares the tables of `prepared` from the program's state when it still applies; false
	// when the tables must be resolved slot by slot.
	[[nodiscard]] bool ReuseTableBindings(const ShaderRecompiler::IR::CompiledShaderInfo& program,
	                                      const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
	                                      TableBindingState& state, uint64_t stamp,
	                                      PreparedBindings& prepared);
	// Marks the elements a change of the image set or of the guest backing since the state's
	// epochs may affect, and moves the epochs to now.
	void MarkTableChanges(TableBindingState& state);
	void MarkTableElement(TableBindingState& state, uint32_t element);
	void MarkTablePages(TableBindingState& state, uint64_t address, uint64_t size);
	void RebuildTableSpans(TableBindingState& state);
	// Adds the spans of an element that joined the state.
	static void AddTableSpans(TableBindingState& state, uint32_t element);
	// The slots of each element (element_slot_begin / element_slots) from `slots`.
	static void IndexTableSlots(TableBindingState& state);
	// ReuseTableBindings() of tables whose T#s changed in a few slots: binds those slots to the
	// elements of their new T#s (new elements join the state). False when the state should be
	// resolved slot by slot again.
	[[nodiscard]] bool PatchTableSlots(const ShaderRecompiler::IR::CompiledShaderInfo& program,
	                     const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
	                     TableBindingState& state, uint64_t stamp);
	// After CommitBindings() bound the tables: records the descriptors and pins the images.
	void FinishTableBindings(TableBindingState& state, const PreparedBindings& prepared);
	void UnpinTableBindings(TableBindingState& state);
	// RebindImages() of reused tables: visits the elements PrepareBindings() marked and those
	// whose image is no longer bound as the state records.
	void RebindReusedTables(PreparedBindings& prepared);


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
	    GuestRange* examined = nullptr, TextureDescDerivation* lookup = nullptr);
	// ResolveTexture's image descriptions by T# and shader view (see DeriveTextureDesc).
	struct TextureDescKey {
		std::array<uint32_t, 8> dwords {};
		std::array<uint32_t, 5> view {};
		bool                    operator==(const TextureDescKey&) const = default;
	};
	struct TextureDescKeyHash {
		size_t operator()(const TextureDescKey& key) const noexcept;
	};
	[[nodiscard]] static TextureDescKey
	MakeTextureDescKey(const ShaderRecompiler::IR::ImageResource&   resource,
	                   const ShaderRecompiler::IR::DescriptorValue& value);
	// Writes the image and view the last lookup of a description found into `binding`, when
	// they still hold (see TextureDescDerivation::Found).
	[[nodiscard]] bool ReuseTextureLookup(TextureDescDerivation& derived, TextureBinding& binding);
	static constexpr size_t MaxTextureDescs = size_t {1} << 16u;
	std::unordered_map<TextureDescKey, TextureDescDerivation, TextureDescKeyHash> m_texture_descs;
	// Bindings point into m_texture_descs until their views are acquired: a full map is cleared
	// once the bindings of the operation are reset.
	bool m_clear_texture_descs = false;
	// Advances when m_texture_descs is cleared (bindings keep pointers into it).
	uint64_t m_texture_descs_generation = 1;
	[[nodiscard]] TextureBinding
	ResolveTableTexture(const ShaderRecompiler::IR::ImageResource&   root,
	                    const ShaderRecompiler::IR::DescriptorValue& value,
	                    GuestRange*                                  examined = nullptr);
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	// The render targets the last draw acquired, and the descriptions derived from the target
	// registers; see RenderTargetMemo.
	struct DrawRenderStorage;
	struct ResolvedDrawRing;
	// Reused by every draw.
	[[nodiscard]] DrawRenderState& AcquireDrawRenderState(bool& shaders_resolved);
	struct RenderTargetMemo;
	struct ColorTargetMemo;
	struct DepthTargetMemo;
	void               CaptureRenderTargetKey(const HW::Context& registers, uint32_t mrt_mask,
	                                          uint32_t slice_offset);
	[[nodiscard]] bool RenderTargetMemoCurrent();
	[[nodiscard]] bool TryReuseRenderTargets(const HW::Context& registers, uint32_t mrt_mask,
	                                         uint32_t slice_offset, DrawRenderState& state);
	void StoreRenderTargetMemo(const RenderColorInfo* colors, uint32_t color_count,
	                           const RenderDepthInfo& depth, const vk::ImageView* views,
	                           vk::ImageView depth_view, std::span<const ImageId> stencil);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer, bool shaders_resolved,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState&    state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, const DrawEmitInfo& emit,
	                         const DrawIndexBufferSource& index_source);
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
	// Prepared bindings and slot memos point into the elements: a deque never relocates its
	// entries (std::unordered_map has no noexcept move, so a growing vector copied the elements
	// and left those pointers dangling).
	std::deque<TableResolution>           m_table_resolutions;
	// Elements a table resolution dropped while prepared bindings may still point into them;
	// freed by ResetBindings().
	std::deque<TableElements>             m_retired_table_elements;
	uint64_t                              m_table_resolution_tick = 0;
	uint64_t                              m_table_prepare_stamp   = 0;
	std::vector<vk::DescriptorImageInfo>  m_table_image_infos;
	// Advances whenever a table resolution drops its elements: table binding states pointing
	// into them no longer apply.
	uint64_t                              m_table_generation = 0;
	std::unordered_map<const void*, TableBindingState> m_table_states;
	std::vector<std::pair<uint64_t, uint64_t>>         m_table_changes;
	std::vector<uint32_t>                              m_table_marked;
	// ReuseTableBindings(): the slots whose T# changed (slot, range, slot in its range).
	struct ChangedTableSlot {
		uint32_t slot  = 0;
		uint32_t range = 0;
		uint32_t index = 0;
	};
	std::vector<ChangedTableSlot>                      m_table_changed_slots;
	// RebindReusedTables(): the elements whose binding changed, one byte per element.
	std::vector<uint8_t>                               m_table_flags;
	uint64_t                              m_draw_window           = 0;
	uint64_t                              m_context_id            = 0;
	// Resolutions ahead of execution; see ResolveDrawAhead().
	std::unique_ptr<ResolvedDrawRing, void (*)(ResolvedDrawRing*)> m_resolved_ring {nullptr,
	                                                                                nullptr};
	ResolvedDraw*                                                  m_resolved_draw = nullptr;
	std::unique_ptr<DrawRenderStorage, void (*)(DrawRenderStorage*)> m_draw_state {nullptr, nullptr};
	// The last color-target descriptions per slot; see ResolveRenderColorTarget().
	std::unique_ptr<ColorTargetMemo, void (*)(ColorTargetMemo*)> m_color_target_memo {nullptr,
	                                                                                  nullptr};
	// The last depth-target descriptions; see ResolveRenderDepthTarget().
	std::unique_ptr<DepthTargetMemo, void (*)(DepthTargetMemo*)> m_depth_target_memo {nullptr,
	                                                                                  nullptr};
	// The render targets of the last draw; see RenderTargetMemo.
	std::unique_ptr<RenderTargetMemo, void (*)(RenderTargetMemo*)> m_render_target_memo {nullptr,
	                                                                                     nullptr};
	// Whether the executing draw reused the render targets of the draw before.
	bool m_render_targets_reused = false;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
