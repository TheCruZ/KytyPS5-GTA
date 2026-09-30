#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GPU_OPERATION_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GPU_OPERATION_H

#include "common/inlineFunction.h"
#include "graphics/host_gpu/renderer/drawPrep.h"
#include "graphics/host_gpu/renderer/render.h"

#include <cstdint>

namespace Libs::Graphics {

class CommandProcessor;

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

// The register state a draw or a dispatch reads when it executes.
struct GpuRegisterState {
	const HW::Context*    context     = nullptr;
	const HW::UserConfig* user_config = nullptr;
	const HW::Shader*     shaders     = nullptr;
	// Unique per snapshot of the context registers (0: live registers, not a snapshot):
	// operations with the same id see the same context registers.
	uint64_t context_id = 0;
};

// NUM_INSTANCES as the command processor last set it. Indirect draws also set the persistent
// instance count, from arguments read when they execute; `sequence` counts NUM_INSTANCES
// packets so an execution can tell which of the two was written last.
struct GpuNumInstances {
	uint32_t value    = 1;
	uint64_t sequence = 0;
};

// The SH registers (user SGPRs) a draw-indirect packet writes its offsets to, or Pm4::SH_NOP.
struct IndirectDrawRegisters {
	uint32_t vertex_offset;
	uint32_t instance_offset;
	uint32_t index_offset;
	// DRAW_INDIRECT_MULTI: the SH register that receives the draw's index within the packet
	// (starts at zero for each packet); Pm4::SH_NOP for none.
	uint32_t draw_index;
};

struct GpuDrawIndirectArgs {
	uint64_t                 args_base         = 0;
	uint32_t                 data_offset       = 0;
	uint32_t                 max_count         = 1;
	const volatile uint32_t* count_addr        = nullptr;
	uint32_t                 stride            = 0;
	uint32_t                 draw_initiator    = 0;
	uint32_t                 index_type        = 0;
	uint32_t                 index_buffer_size = 0;
	uint64_t                 index_base        = 0;
	bool                     indexed           = false;
	bool                     multi             = false;
	IndirectDrawRegisters    registers {};
	// Where the packet sits in its command buffer, for the look-ahead of mesh argument runs.
	const uint32_t* run_packet = nullptr;
	const uint32_t* run_end    = nullptr;
};

struct GpuDispatchArgs {
	uint32_t thread_group_x = 0;
	uint32_t thread_group_y = 0;
	uint32_t thread_group_z = 0;
	uint32_t mode           = 0;
	uint64_t args_addr      = 0; // Indirect dispatches.
	// Where the packet sits in its command buffer and the base of the indirect arguments in force,
	// for the look-ahead of dispatch runs.
	const uint32_t* run_packet    = nullptr;
	const uint32_t* run_end       = nullptr;
	uint64_t        indirect_base = 0;
};

// Callbacks of queued operations live in the operation itself; the largest ones capture about
// ten words.
using GpuCallback = Common::InlineFunction<96>;

enum class GpuOperationKind : uint8_t {
	DrawIndex,
	DrawAuto,
	DrawIndirect,
	DispatchDirect,
	DispatchIndirect,
	// Any other guest-visible effect or renderer work, in command order.
	Callback,
};

// One unit of emulated GPU work produced by a command processor. Operations execute in the
// order the command processors produce them; every guest-visible effect of a command buffer
// (draws, dispatches, labels, interrupts, flips, DMA) is an operation.
struct GpuOperation {
	GpuOperationKind  kind      = GpuOperationKind::Callback;
	CommandProcessor* processor = nullptr;
	uint64_t          submit_id = 0;
	// Position in the stream of a pipelined GPU; see GuestGpu::EmitOperation().
	uint64_t          sequence  = 0;
	// The GPU queue (DrawPrep): 0 graphics, 1.. the compute queues.
	uint32_t          queue     = 0;
	GpuRegisterState  state;
	GpuNumInstances   num_instances;
	// Draws and dispatches: the descriptor walks the resolve thread prepared ahead of execution
	// (DrawPrep::Take), under this key.
	DrawPrep::Key prep_key {};
	bool          prepared = false;
	// The command processor processed a WAIT_REG_MEM since the previous operation: the guest may
	// have written memory the operations before it read.
	bool after_wait = false;
	union {
		DrawIndexArgs       draw_index;
		DrawAutoArgs        draw_auto;
		GpuDrawIndirectArgs draw_indirect;
		GpuDispatchArgs     dispatch;
	};
	GpuCallback callback;

	GpuOperation(): draw_index {} {}
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GPU_OPERATION_H
