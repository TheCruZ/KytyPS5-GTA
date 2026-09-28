#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H

#include "common/assert.h"
#include "graphics/guest_gpu/command_processor/gpuOperation.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <cstdint>
#include <source_location>
#include <memory>
#include <span>
#include <vector>

namespace Libs::Graphics {

class GuestGpu;

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func);

enum class Pm4ProcessResult { Complete, Blocked };

enum class ContextStateOperation : uint32_t {
	Clear     = 0,
	Push      = 1,
	Pop       = 2,
	PushClear = 3,
};

class Pm4Execution {
public:
	[[nodiscard]] bool MadeProgress() const noexcept { return m_made_progress; }
	// Draw prep (KYTY_DRAW_PREP): the submission number of a graphics command stream (0: not
	// counted), whose draw and dispatch packets are numbered as the scanner numbers them.
	void SetPrepSeq(uint64_t seq, uint32_t queue) noexcept {
		m_prep_seq   = seq;
		m_prep_queue = queue;
	}

private:
	friend class CommandProcessor;

	struct BufferCursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset_dw = 0;
	};

	std::vector<BufferCursor> m_buffer_stack;
	std::span<const uint32_t> m_next_buffer;
	bool                      m_chain         = false;
	bool                      m_suspended     = false;
	bool                      m_made_progress = false;
	uint64_t                  m_prep_seq      = 0;
	uint32_t                  m_prep_queue    = 0;
	uint32_t                  m_prep_ordinal  = 0;
	const uint32_t*           m_prep_packet   = nullptr; // the last numbered packet
	// KYTY_SRT_DEFER: the dispatch packet last suspended for SRT bytes to land, and how often.
	const uint32_t*           m_deferred_packet = nullptr;
	uint32_t                  m_defer_count     = 0;
	uint64_t                  m_deferred_tick   = 0;
};

class CommandProcessor {
public:
	struct FlipInfo {
		int     handle    = 0;
		int     index     = 0;
		int     flip_mode = 0;
		int64_t flip_arg  = 0;
	};

	CommandProcessor(RenderContext& renderer, int interrupt_event_id)
	    : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id) {}
	~CommandProcessor() = default;

	KYTY_CLASS_NO_COPY(CommandProcessor);

	void Reset();
	void ApplyContextStateOperation(ContextStateOperation operation);

	// Operations of a pipelined GPU go to its execution thread; without one they execute
	// when they are emitted.
	void AttachPipeline(GuestGpu* gpu) { m_pipeline = gpu; }

	void BufferInit();
	// `reason` names why the submission is flushed (a debugging aid of the scheduler).
	void BufferFlush(const char* reason = nullptr);
	void BufferFlushIfBatchReady();
	void BufferWait();
	// See CommandScheduler::CompleteReleaseMemWrite -- only safe for a RELEASE_MEM that already
	// wrote its guest-visible value synchronously and scheduled no interrupt callback.
	void CompleteReleaseMemWrite();
	// See CommandScheduler::CompleteReleaseMemInterrupt -- for a RELEASE_MEM that DOES request a
	// guest interrupt/event; uses a smaller batch bound as a hedge against delaying it.
	void CompleteReleaseMemInterrupt();
	// See CommandScheduler::CompleteDraw -- called after every DrawIndex/DrawAuto.
	void CompleteDraw();
	// Register writes go through these; queued operations keep snapshots of the old values.
	HW::Context& GetCtx() {
		m_ctx_dirty = true;
		return m_ctx;
	}
	HW::UserConfig& GetUcfg() {
		m_ucfg_dirty = true;
		return m_ucfg;
	}
	HW::Shader& GetShCtx() {
		m_sh_ctx_dirty = true;
		return m_sh_ctx;
	}

	void SetIndexType(uint32_t index_type_and_size);
	void SetIndexBaseAddress(uint64_t index_base_addr);
	void SetIndexBufferSize(uint32_t index_buffer_size);
	void SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr);
	void SetDispatchIndirectArgsBaseAddress(uint64_t dispatch_indirect_args_base_addr);
	[[nodiscard]] uint64_t GetDrawIndirectArgsBaseAddress() const {
		return m_draw_indirect_args_base_addr;
	}
	[[nodiscard]] uint64_t GetDispatchIndirectArgsBaseAddress() const {
		return m_dispatch_indirect_args_base_addr;
	}
	void SetNumInstances(uint32_t num_instances);
	void DrawIndex(DrawIndexArgs args);
	void DrawIndexOffset(uint32_t index_offset, uint32_t index_count);
	void DrawIndexAuto(DrawAutoArgs args);
	void DrawIndirect(uint32_t data_offset, IndirectDrawRegisters registers,
	                  uint32_t draw_initiator, bool indexed);
	void DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
	                       const volatile uint32_t* count_addr, uint32_t stride_in_bytes,
	                       IndirectDrawRegisters registers, uint32_t draw_initiator, bool indexed);
	void WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint32_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint64_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void Flip();
	void Flip(void* dst_gpu_addr, uint32_t value);
	void FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action, void* dst_gpu_addr,
	                       uint32_t value);
	void PrepareCpuFlip(uint64_t request_id);
	void SynchronizeGpu();
	void EmitGlobalBarrier(std::source_location site = std::source_location::current());
	void TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id);
	void DeferEopInterrupt(uint32_t interrupt_context_id);
	void DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y, uint32_t thread_group_z,
	                    uint32_t mode);
	void DispatchIndirect(uint64_t args_addr, uint32_t mode);
	void WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index);
	void TriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address = 0);
	void WriteLodStats(void* dst, uint32_t size);
	void RunGarbageCollector();
	// Runs a host command in operation order.
	void EmitHostCommand(Common::UniqueFunction<void>&& command);
	// Reads guest memory the CP decides on (WAIT_REG_MEM, COND_EXEC, predication, branches) as a
	// serial GPU would: after the earlier operations that write it.
	void ReadDecisionMemory(const volatile void* address, void* value, size_t size);
	template <typename T>
	[[nodiscard]] T ReadDecisionValue(const volatile void* address) {
		T value {};
		ReadDecisionMemory(address, &value, sizeof(T));
		return value;
	}

	// Executes an operation this command processor produced.
	void Execute(GpuOperation& operation);

	void SetUserDataMarker(HW::UserSgprType type) { m_user_data_marker = type; }
	[[nodiscard]] HW::UserSgprType GetUserDataMarker() const { return m_user_data_marker; }

	void ResetDeCe();
	void SetCeComplete(bool complete) { m_ce_complete = complete; }
	void WaitCe();
	void WaitDeDiff(uint32_t diff);
	void WaitForRewind(bool valid);
	void IncrementDe();
	void IncrementCe();

	void WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num);
	void DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num);

	template <typename T>
	void WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll, uint32_t wait_op);
	// A label as the command processor sees it: end-of-pipe writes count once recorded, even
	// before the CPU can see them (GuestGpu::LabelsAtCompletion).
	template <typename T>
	[[nodiscard]] T ReadLabel(const volatile T* addr) const;
	void WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num, uint32_t write_control);
	void WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes);
	void DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
	             uint64_t dst_address_or_offset, uint8_t src_sel, uint8_t src_cache_policy,
	             uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
	             uint8_t wait_for_previous, uint8_t write_confirm, uint8_t block_engine);
	void SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
	                    const volatile void* address, uint32_t count_in_dwords);
	[[nodiscard]] bool ShouldSkipPredicatedPackets() const { return m_predicate_skip; }

	Pm4ProcessResult Process(Pm4Execution& execution, std::span<const uint32_t> commands);
	void             ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain);

	void SetFlip(const FlipInfo& flip) { m_flip = flip; }

	[[nodiscard]] uint64_t GetSubmitId() const { return m_submit_id; }
	void                   SetSubmitId(uint64_t submit_id) { m_submit_id = submit_id; }
	[[nodiscard]] bool     IsAsyncComputeQueue() const { return m_interrupt_event_id >= 0x20; }
	[[nodiscard]] RenderContext& GetRenderContext() const { return m_renderer; }

private:
	template <typename T>
	void WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest, uint32_t eop_event_type,
	                      uint32_t cache_action, uint32_t event_index, uint32_t event_write_source,
	                      void* dst_gpu_addr, T value, uint32_t interrupt_selector,
	                      uint32_t interrupt_context_id);
	void ProcessPm4(Pm4Execution& execution);
	void SuspendPm4();
	// KYTY_SRT_DEFER: around a dispatch's RenderExecutor call. EndDispatchDefer suspends the
	// packet when its SRT refresh was deferred (PipelineCache::SrtDeferRead).
	// BeginDispatchDefer returns true when the packet is still waiting: it is suspended again and
	// the dispatch is not attempted.
	[[nodiscard]] bool BeginDispatchDefer();
	void               EndDispatchDefer();
	CommandScheduler&   GetScheduler() const { return m_renderer.GetCommandScheduler(); }
	CommandBuffer&      CurrentBuffer() { return GetScheduler().Current(); }

	// Producer side: operations carry the register state and the CP state they need.
	[[nodiscard]] GpuOperation MakeOperation(GpuOperationKind kind);
	[[nodiscard]] GpuRegisterState SnapshotState(uint64_t sequence);
	// Both return the sequence of a pipelined operation, 0 once it executed.
	uint64_t                   Emit(GpuOperation&& operation);
	uint64_t                   EmitCallback(GpuCallback&& callback);
	void NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size,
	                    const void* value = nullptr);

	// Execution side, in operation order. Only these touch the renderer.
	void BindState(const GpuRegisterState& state);
	void NoteRecordedWork();
	[[nodiscard]] uint32_t ResolveNumInstances(const GpuOperation& operation) const;
	// The packet being processed and the end of its command buffer, for execution-side scans.
	static void CaptureStreamPosition(const uint32_t*& packet, const uint32_t*& end);
	void ExecuteDrawIndex(const GpuOperation& operation, DrawIndexArgs args);
	void ExecuteDrawAuto(const GpuOperation& operation, DrawAutoArgs args);
	void ExecuteDrawIndirect(const GpuOperation& operation);
	bool PatchIndirectDrawOffsets(const GpuOperation& operation, uint32_t vertex_offset,
	                              uint32_t instance_offset, uint32_t first_index,
	                              uint32_t draw_index);
	void ExecuteDispatchDirect(const GpuOperation& operation, uint32_t thread_group_x,
	                           uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode);
	void ExecuteDispatchIndirect(const GpuOperation& operation);
	template <typename T>
	void ExecuteWriteAtEndOfPipe(uint64_t submit_id, uint32_t eop_event_type,
	                             uint32_t cache_action, uint32_t event_index,
	                             uint32_t event_write_source, void* dst_gpu_addr, T value,
	                             uint32_t interrupt_selector, uint32_t interrupt_context_id);
	void ExecuteGlobalBarrier(std::source_location site = std::source_location::current());
	void ExecuteTriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address);
	void ExecuteDmaData(bool dst_gds, uint64_t dst_address_or_offset, uint8_t src_sel,
	                    bool src_gds, uint64_t src_address_or_offset_or_immediate,
	                    uint32_t num_bytes);

	// Copies of the register state for operations a pipelined GPU has not executed yet. A slot
	// is overwritten only after the last operation that uses it executed. Slots start on cache
	// lines of their own and the command processor keeps the last uses apart, so it writes no
	// line the execution thread reads.
	template <typename T>
	struct SnapshotRing {
		static constexpr size_t Size = 512;
		struct alignas(64) Slot {
			T value;
		};
		std::unique_ptr<Slot[]>     slots;
		std::unique_ptr<uint64_t[]> last_use;
		size_t                      current = Size;
		size_t                      next    = 0;
	};
	template <typename T>
	const T* Snapshot(SnapshotRing<T>& ring, const T& live, bool& dirty, uint64_t sequence);

	RenderContext& m_renderer;
	GuestGpu*      m_pipeline = nullptr;
	const int      m_interrupt_event_id;

	// Command processor state, written by the command processor thread.
	alignas(64) HW::Context m_ctx;
	HW::Context      m_saved_ctx;
	bool             m_context_state_pushed = false;
	HW::UserConfig   m_ucfg;
	HW::Shader       m_sh_ctx;
	HW::UserSgprType m_user_data_marker                 = HW::UserSgprType::Unknown;
	uint32_t         m_index_type_and_size              = 0;
	uint32_t         m_index_buffer_size                = 0;
	uint64_t         m_index_base_addr                  = 0;
	uint64_t         m_draw_indirect_args_base_addr     = 0;
	uint64_t         m_dispatch_indirect_args_base_addr = 0;
	// Persistent draw state: indirect draws update it for subsequent draws when they execute.
	GpuNumInstances m_num_instances;

	uint32_t m_de_count    = 0;
	uint32_t m_ce_count    = 0;
	bool     m_ce_complete = false;

	uint32_t m_const_ram[0x3000] = {0};

	bool                         m_ctx_dirty    = true;
	bool                         m_ucfg_dirty   = true;
	bool                         m_sh_ctx_dirty = true;
	SnapshotRing<HW::Context>    m_ctx_snapshots;
	SnapshotRing<HW::UserConfig> m_ucfg_snapshots;
	SnapshotRing<HW::Shader>     m_sh_ctx_snapshots;

	FlipInfo m_flip;
	uint64_t m_submit_id      = 0;
	bool     m_predicate_skip = false;

	// State of the execution side, on cache lines of its own.
	alignas(64) uint64_t m_synthetic_occlusion_counter = 0;
	// Draws and dispatches recorded into the command buffer of m_batch_tick.
	uint64_t m_batch_tick = 0;
	uint32_t m_batch_work = 0;
	// The instance count of the last executed indirect draw, and the NUM_INSTANCES sequence it
	// followed.
	uint32_t m_indirect_instances         = 0;
	// Address of the arguments of a DRAW_INDIRECT whose instance count the GPU reads (0: the CPU
	// read it into m_indirect_instances).
	uint64_t m_indirect_args_address      = 0;
	uint64_t m_indirect_instances_sequence = 0;
	bool     m_has_indirect_instances      = false;
	// The shader registers of a pipelined indirect draw with its offsets written to user SGPRs.
	std::unique_ptr<HW::Shader> m_indirect_shaders;
	// Keeps later allocations off the last line of the execution-side state.
	alignas(64) uint8_t m_end_padding = 0;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
