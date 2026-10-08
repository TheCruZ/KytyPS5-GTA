#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "gpu_tiler_shaders/indirect_thread_dispatch_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {
static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

// Stores through V#s selected from a descriptor table write guest memory by device address.
// Cache every target of the tables so the BDA page table covers them. Returns whether any table
// names a target.
// Whether a dispatch may write memory that no bound range names: address stores and stores through
// V#s selected from descriptor tables. Other buffers reached through memory offsets are written
// within the ranges their V#s bind, recorded as GPU writes like any written binding.
static bool MayWriteThroughDeviceAddresses(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	return program.has_address_writes ||
	       (program.bindings.memory_offset_count != 0 &&
	        std::ranges::any_of(program.info.buffers, [](const auto& buffer) {
		        return buffer.indirect_write_table;
	        }));
}

static bool PrepareIndirectWriteTargets(RenderContext& context, const PreparedBindings& bindings) {
	const auto& program = *bindings.runtime->program;
	const auto& layout  = program.bindings;
	if (layout.memory_offset_count == 0) {
		return false;
	}
	// Buffer sources follow the bound resources, which leave out buffers the shader never
	// accesses, not the shader's buffer list.
	const auto& resources = layout.descriptors.front().resources;
	bool        prepared  = false;
	for (uint32_t i = 0; i < bindings.buffer_sources.size() && i < resources.size(); ++i) {
		const auto& resource = program.info.buffers[resources[i]];
		const auto& table    = bindings.buffer_sources[i];
		if (!resource.indirect_write_table || table.address == 0 ||
		    table.size <= resource.indirect_table_offset) {
			continue;
		}
		prepared |= context.GetIndirectWriteTables().Prepare(
		    table.address + resource.indirect_table_offset,
		    table.size - resource.indirect_table_offset);
	}
	return prepared;
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer&          buffer) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				if (cache.ClearMeta(descriptor.Base48())) {
					return true;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	if (invocations != descriptor.NumRecords() || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

bool RenderExecutor::TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
                                                CommandBuffer& command, uint32_t group_x,
                                                uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	if (!cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear)) {
		// A uniform fill of memory no image covers is performed like a DMA fill: on the CPU while
		// the GPU does not own the range. DCC metadata cleared this way is read by fast-clear
		// discovery without draining the GPU first.
		// Larger fills stay on the GPU: a CPU fill would also have to be uploaded again.
		constexpr uint64_t MaxBufferFill = uint64_t {1} << 20u;
		if (((descriptor.Base48() | size) & 3u) != 0 || size > MaxBufferFill ||
		    cache.HasImagesInRegion(descriptor.Base48(), size)) {
			return false;
		}
		m_context.GetBufferCache().FillBuffer(descriptor.Base48(), size, packed_clear, false);
		return true;
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute image clear shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
	}
	return true;
}

static void BindSharedMemory(RenderContext& context, ShaderComputeInputInfo& input,
                             PreparedBindings& bindings, uint64_t indirect_args = 0) {
	if (ShaderRecompiler::IR::FindBinding(input.stage.program->bindings,
	        ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory) == nullptr) {
		return;
	}
	auto& cache = context.GetBufferCache();
	if (indirect_args != 0) {
		cache.ReadMemory(indirect_args, sizeof(vk::DispatchIndirectCommand));
		std::memcpy(input.workgroup_counts, reinterpret_cast<const void*>(indirect_args),
		            sizeof(input.workgroup_counts));
	}
	// LDS has no contents to preserve between dispatches. The existing shader hazard
	// barriers also order other users of this GPU-only utility buffer.
	auto& storage = cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto limit = std::min<uint64_t>(storage.Size(),
	    context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	uint64_t size = sizeof(uint32_t);
	if (std::ranges::find(input.workgroup_counts, 0u) == std::end(input.workgroup_counts)) {
		size = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(size == 0 || size > limit);
		for (const auto count: input.workgroup_counts) {
			EXIT_IF(size > limit / count);
			size *= count;
		}
	}
	bindings.shared_memory = {storage.Handle(), 0, size};
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode) {
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	Common::LockGuard lock(m_context.GetMutex());
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo input_info {};
	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.workgroup_counts[0] = thread_group_x;
	input_info.workgroup_counts[1] = thread_group_y;
	input_info.workgroup_counts[2] = thread_group_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size = std::max(group_sizes[axis], 1u);
			const auto threads = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	ShaderProgram compute_program;
	if (!TakeResolvedDispatch(input_info, compute_program)) {
		compute_program = m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info);
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;
	if (resources.specialization_reads.empty() &&
	    (TryConsumeComputeMetaClear(input_info, buffer) ||
	     TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y,
	                                 thread_group_z, mode))) {
		ResetBindings();
		return;
	}
	ShaderBufferResource known_fill_descriptor;
	uint32_t             known_fill_value = 0;
	uint64_t             known_fill_size  = 0;
	const bool           known_fill =
	    ResolveComputeBufferFill(input_info, thread_group_x, thread_group_y, thread_group_z, mode,
	                             known_fill_descriptor, known_fill_value, known_fill_size);

	if (use_thread_dimensions) {
		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = input_info.workgroup_counts[0];
		thread_group_y       = input_info.workgroup_counts[1];
		thread_group_z       = input_info.workgroup_counts[2];

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	buffer.EndRendering();
	auto& pipeline =
	    m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		std::copy(std::begin(input_info.dispatch_threads_num), std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	PreparedBindings* descriptor_stage = &bindings;
	FindBuffers(std::span {&descriptor_stage, 1u});
	// New target buffers must exist before PrepareBda uploads the CPU writes of cached buffers.
	const bool indirect_writes = PrepareIndirectWriteTargets(m_context, bindings);
	if (program.info.uses_dma) {
		m_context.CacheDmaBases(input_info.stage);
		const ShaderStageRuntime* const dma_stage = &input_info.stage;
		const bool reach_synchronized = m_context.SynchronizeDmaFootprint(std::span {&dma_stage, 1});
		m_context.PrepareBda(MayWriteThroughDeviceAddresses(program), !reach_synchronized);
		if (program.bindings.memory_offset_count != 0) {
			// Its results are often read back soon: submit it at once (see CommitWriteTicks()).
			m_context.GetBufferCache().RequestFlushForReadbacks();
		}
	}
	RebindImages(bindings);
	BindSharedMemory(m_context, input_info, bindings);
	RebindBuffers(bindings);

	auto              vk_buffer        = buffer.Handle();
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage) || indirect_writes;
	has_storage_writes =
	    std::any_of(program.info.images.begin(), program.info.images.end(),
	                [](const auto& image) {
		                return image.written &&
		                       image.resource_class ==
		                           ShaderRecompiler::IR::ImageResourceClass::Storage;
	                }) ||
	    has_storage_writes;
	// On the guest's GPU, dispatches of one queue overlap unless the guest orders them (a wait,
	// a partial flush, an ACQUIRE_MEM; the operations other than dispatches record their own
	// barriers). Without such an order since the previous direct dispatch of its queue, nothing
	// was recorded after that one and its barrier is still deferred: the barrier before the
	// earlier dispatch already ordered everything before it, so neither is needed and the two
	// overlap. GTA V's ray tracing structure builds run hundreds of such dispatches per frame.
	const bool chained = std::exchange(m_dispatch_chained, false) && m_deferred_barrier;
	if (has_storage_writes && !chained) {
		// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
		// while allowing the queue to execute asynchronously.
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " hash=0x%016" PRIx64 " tick=%" PRIu64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     thread_group_x, thread_group_y, thread_group_z, mode,
		     input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
			    resources.samplers[program.info.samplers[i].snapshot_index]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);

	// The removed host fence also ordered read-only dispatches before later writers. Recorded
	// before the next operation, unless that is a dispatch the guest did not order after this one.
	m_deferred_barrier       = true;
	m_deferred_barrier_queue = m_guest_queue;
	if (known_fill) {
		// Recorded after binding, which forgets older fills over the written ranges.
		m_context.GetBufferCache().NoteKnownFill(known_fill_descriptor.Base48(), known_fill_size,
		                                         known_fill_value);
	}
	ResetBindings();
}

struct IndirectThreadPass {
	explicit IndirectThreadPass(GraphicContext& graphics): graphics(graphics) {
		const vk::DescriptorSetLayoutBinding bindings[] {
		    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
		    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
		    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
		};
		vk::DescriptorSetLayoutCreateInfo layout_info {};
		layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		layout_info.bindingCount = static_cast<uint32_t>(std::size(bindings));
		layout_info.pBindings    = bindings;
		RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
		                                                               &descriptor_layout),
		                     "create indirect-thread descriptor layout");
		vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
		vk::PipelineLayoutCreateInfo pipeline_layout_info {};
		pipeline_layout_info.setLayoutCount         = 1;
		pipeline_layout_info.pSetLayouts            = &descriptor_layout;
		pipeline_layout_info.pushConstantRangeCount = 1;
		pipeline_layout_info.pPushConstantRanges    = &push;
		RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
		                                                          &pipeline_layout),
		                     "create indirect-thread pipeline layout");
		const auto module = CompileSPV(INDIRECT_THREAD_DISPATCH_SPV, graphics.device);
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage  = vk::ShaderStageFlagBits::eCompute;
		stage.module = module;
		stage.pName  = "main";
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage  = stage;
		pipeline_info.layout = pipeline_layout;
		const auto result =
		    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline);
		graphics.device.destroyShaderModule(module, nullptr);
		RequireVulkanSuccess(result, "create indirect-thread pipeline");
	}
	~IndirectThreadPass() {
		graphics.device.destroyPipeline(pipeline, nullptr);
		graphics.device.destroyPipelineLayout(pipeline_layout, nullptr);
		graphics.device.destroyDescriptorSetLayout(descriptor_layout, nullptr);
	}
	KYTY_CLASS_NO_COPY(IndirectThreadPass);

	struct Push {
		uint32_t local_size[3];
		uint32_t args_index;
		uint32_t groups_index;
		uint32_t limit_index;
	};

	GraphicContext&         graphics;
	vk::DescriptorSetLayout descriptor_layout = nullptr;
	vk::PipelineLayout      pipeline_layout   = nullptr;
	vk::Pipeline            pipeline          = nullptr;
};

void RenderExecutor::DispatchIndirectThreads(uint64_t submit_id, CommandBuffer& buffer,
                                             uint64_t args_addr, uint32_t mode) {
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0 ||
	        (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) == 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    1, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	Common::LockGuard lock(m_context.GetMutex());
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	ShaderComputeInputInfo input_info {};
	ShaderProgram          compute_program;
	if (!TakeResolvedDispatch(input_info, compute_program)) {
		input_info.dispatch_thread_dimensions = true;
		input_info.dispatch_indirect_threads  = true;
		compute_program = m_context.GetPipelineCache().GetComputeProgram(
		    cs_regs, buffer.GetRegisters().GetShaderRegisters(), input_info);
	}
	if (!compute_program) {
		return;
	}
	const auto& program = *input_info.stage.program;
	EXIT_IF(program.bindings.UsesPushData());
	// Metadata clears are consumed as in the direct path; it does not depend on the size.
	if (TryConsumeComputeMetaClear(input_info, buffer)) {
		ResetBindings();
		return;
	}
	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	PreparedBindings* descriptor_stage = &bindings;
	FindBuffers(std::span {&descriptor_stage, 1u});
	if (program.info.uses_dma) {
		m_context.CacheDmaBases(input_info.stage);
		const ShaderStageRuntime* const dma_stage = &input_info.stage;
		const bool reach_synchronized = m_context.SynchronizeDmaFootprint(std::span {&dma_stage, 1});
		m_context.PrepareBda(MayWriteThroughDeviceAddresses(program), !reach_synchronized);
		if (program.bindings.memory_offset_count != 0) {
			// Its results are often read back soon: submit it at once (see CommitWriteTicks()).
			m_context.GetBufferCache().RequestFlushForReadbacks();
		}
	}
	RebindImages(bindings);
	(void)PrepareIndirectWriteTargets(m_context, bindings);
	constexpr uint64_t ArgsSize = 3 * sizeof(uint32_t);
	if (ShaderRecompiler::IR::FindBinding(program.bindings,
	        ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory) != nullptr) {
		// LDS kept in a device buffer is sized by the workgroup count: read the thread counts.
		m_context.GetBufferCache().ReadMemory(args_addr, ArgsSize);
		const auto* threads = reinterpret_cast<const uint32_t*>(args_addr);
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size = std::max(group_sizes[axis], 1u);
			input_info.workgroup_counts[axis] = threads[axis] / size + (threads[axis] % size != 0u);
		}
		BindSharedMemory(m_context, input_info, bindings);
	}
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	const auto [args_buffer, args_offset] =
	    m_context.GetBufferCache().ObtainBuffer(args_addr, ArgsSize, false);
	EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
	RebindBuffers(bindings);
	const auto& shader_data = bindings.shader_data_buffer;
	const bool  thread_limit =
	    program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart;
	EXIT_IF(thread_limit && shader_data.buffer == nullptr);

	auto&      stream  = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const std::array<uint32_t, 8> scratch_init {};
	const auto scratch_offset = stream.Copy(scratch_init.data(), sizeof(scratch_init), 256);

	if (!m_indirect_thread_pass) {
		m_indirect_thread_pass = {new IndirectThreadPass(m_context.GetGraphics()),
		                          [](IndirectThreadPass* pass) { delete pass; }};
	}
	auto&      pass      = *m_indirect_thread_pass;
	const auto vk_buffer = buffer.Handle();
	const auto alignment = m_context.GetGraphics().StorageMinAlignment();
	const auto args_base = Common::AlignDown(args_offset, alignment);
	const vk::DescriptorBufferInfo infos[] {
	    {args_buffer->Handle(), args_base, args_offset - args_base + ArgsSize},
	    {stream.Handle(), scratch_offset, sizeof(scratch_init)},
	    thread_limit
	        ? vk::DescriptorBufferInfo {shader_data.buffer, shader_data.offset, shader_data.range}
	        : vk::DescriptorBufferInfo {stream.Handle(), scratch_offset, sizeof(scratch_init)},
	};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	IndirectThreadPass::Push push {};
	push.local_size[0] = std::max(cs_regs.cs_regs.num_thread_x, 1u);
	push.local_size[1] = std::max(cs_regs.cs_regs.num_thread_y, 1u);
	push.local_size[2] = std::max(cs_regs.cs_regs.num_thread_z, 1u);
	push.args_index    = static_cast<uint32_t>((args_offset - args_base) / sizeof(uint32_t));
	push.groups_index  = 0;
	push.limit_index   = thread_limit ? program.bindings.dispatch_thread_dword : 4u;

	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite |
	                       vk::AccessFlagBits::eHostWrite;
	before.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                          vk::PipelineStageFlagBits::eComputeShader, {}, 1, &before, 0, nullptr,
	                          0, nullptr);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pass.pipeline);
	vk_buffer.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, pass.pipeline_layout, 0,
	                               writes);
	vk_buffer.pushConstants(pass.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                        sizeof(push), &push);
	vk_buffer.dispatch(1, 1, 1);
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	after.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead |
	                      vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eUniformRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                          vk::PipelineStageFlagBits::eDrawIndirect |
	                              vk::PipelineStageFlagBits::eComputeShader,
	                          {}, 1, &after, 0, nullptr, 0, nullptr);

	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	const bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage) ||
	    std::any_of(program.info.images.begin(), program.info.images.end(), [](const auto& image) {
		    return image.written && image.resource_class ==
		                                ShaderRecompiler::IR::ImageResourceClass::Storage;
	    });
	if (has_storage_writes) {
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	vk_buffer.dispatchIndirect(stream.Handle(), scratch_offset);
	ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	ResetBindings();
}

void RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer,
                                      uint64_t args_addr, uint32_t mode) {
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0 ||
	        (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	Common::LockGuard lock(m_context.GetMutex());
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	ShaderComputeInputInfo input_info {};
	ShaderProgram          compute_program;
	if (!TakeResolvedDispatch(input_info, compute_program)) {
		compute_program = m_context.GetPipelineCache().GetComputeProgram(
		    cs_regs, buffer.GetRegisters().GetShaderRegisters(), input_info);
	}
	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	PreparedBindings* descriptor_stage = &bindings;
	FindBuffers(std::span {&descriptor_stage, 1u});
	const auto& program = *input_info.stage.program;
	// New target buffers must exist before PrepareBda uploads the CPU writes of cached buffers.
	const bool indirect_writes = PrepareIndirectWriteTargets(m_context, bindings);
	if (program.info.uses_dma) {
		m_context.CacheDmaBases(input_info.stage);
		const ShaderStageRuntime* const dma_stage = &input_info.stage;
		const bool reach_synchronized = m_context.SynchronizeDmaFootprint(std::span {&dma_stage, 1});
		m_context.PrepareBda(MayWriteThroughDeviceAddresses(program), !reach_synchronized);
		if (program.bindings.memory_offset_count != 0) {
			// Its results are often read back soon: submit it at once (see CommitWriteTicks()).
			m_context.GetBufferCache().RequestFlushForReadbacks();
		}
	}
	BindSharedMemory(m_context, input_info, bindings, args_addr);
	RebindImages(bindings);
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
	    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
	RebindBuffers(bindings);
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	const auto vk_buffer = buffer.Handle();
	const bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage) || indirect_writes ||
	    std::any_of(program.info.images.begin(), program.info.images.end(), [](const auto& image) {
		    return image.written && image.resource_class ==
		                                ShaderRecompiler::IR::ImageResourceClass::Storage;
	    });
	if (has_storage_writes) {
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
	                              vk::PipelineStageFlagBits::eComputeShader |
	                              vk::PipelineStageFlagBits::eTransfer,
	                          vk::PipelineStageFlagBits::eDrawIndirect, {},
	                          1, &barrier, 0, nullptr, 0, nullptr);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	vk_buffer.dispatchIndirect(args_buffer->Handle(), args_offset);
	ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	ResetBindings();
}

} // namespace Libs::Graphics
