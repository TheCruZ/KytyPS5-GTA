#include "common/assert.h"
#include <fmt/format.h>
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/hackNotice.h"
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
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/dispatchRun.h"
#include "graphics/host_gpu/renderer/gpuProfiler.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/timeline.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
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
#include <cinttypes>
#include <deque>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

// KYTY_LOG_SKIPPED_DISPATCHES=<n>: log the first n dispatches of compute shaders that gave up,
// with their user data, to see which guest buffers they would have written.
static bool EarlyDispatchSkip() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_EARLY_DISPATCH_SKIP");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

static void LogSkippedDispatch(const HW::ComputeShaderInfo& cs, const char* groups) {
	static const uint32_t budget = [] {
		const char* value = std::getenv("KYTY_LOG_SKIPPED_DISPATCHES");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	static std::atomic<uint32_t> used {0};
	if (budget == 0 || used.fetch_add(1, std::memory_order_relaxed) >= budget) {
		return;
	}
	std::string user_data;
	for (uint32_t i = 0; i < cs.cs_user_sgpr.count && i < HW::UserSgprInfo::SGPRS_MAX; i++) {
		char word[12];
		std::snprintf(word, sizeof(word), " %08x", cs.cs_user_sgpr.value[i]);
		user_data += word;
	}
	LOGF("Skipped dispatch: hash=0x%016" PRIx64 " groups=%s user_data=%s\n",
	     ShaderDeclaredHash(cs.cs_regs.data_addr), groups, user_data.c_str());
}
namespace {

// A host GPU zone for a dispatch (gpuProfiler.h), named by its shader.
int64_t BeginGpuZone(RenderContext& context, CommandBuffer& buffer, uint64_t hash) {
	if (!Profiler::g_counters) {
		return -1;
	}
	char      name[32];
	const int size = std::snprintf(name, sizeof(name), "CS %016" PRIx64, hash);
	Profiler::CommandZone::Annotate("CS %016" PRIx64, hash);
	return GpuProfiler::Begin(context, buffer, name, static_cast<size_t>(size), 0xf28e2b);
}

// The rollback below drains the GPU twice per dispatch and almost never fires once a shader's
// pages are resident (one retry in a whole run, against ~2500 protected dispatches a frame). A
// shader is trusted after trust_runs clean attempts and dispatched without it; any fault that
// unprotected work reports re-arms every shader (and reports a [HACK]: that work may have read a
// missing page as zero; trusted dispatches that fault nothing report nothing). KYTY_DISPATCH_RECOVERY=always keeps it on every
// dispatch; KYTY_DISPATCH_RECOVERY_TRUST=<n> sets the clean runs needed.
struct DispatchRecoveryPolicy {
	std::unordered_map<uint64_t, uint32_t> clean_runs;
	uint64_t                               seen_faults          = 0;
	uint32_t                               trust_runs           = 4;
	bool                                   always               = false;
	uint64_t                               protected_dispatches = 0;
	uint64_t                               trusted_dispatches   = 0;
	uint64_t                               retries              = 0;
	uint64_t                               rearms               = 0;

	void Count(bool trusted) {
		(trusted ? trusted_dispatches : protected_dispatches)++;
		if (((protected_dispatches + trusted_dispatches) & 0x3fff) == 0) {
			LOGF("DispatchRecovery: protected=%" PRIu64 " trusted=%" PRIu64 " retries=%" PRIu64
			     " rearms=%" PRIu64 " shaders=%zu\n",
			     protected_dispatches, trusted_dispatches, retries, rearms, clean_runs.size());
		}
	}
};

// Dispatches are recorded on the GPU thread only.
DispatchRecoveryPolicy& RecoveryPolicy() {
	static DispatchRecoveryPolicy policy = [] {
		DispatchRecoveryPolicy p;
		if (const char* value = std::getenv("KYTY_DISPATCH_RECOVERY")) {
			p.always = std::strcmp(value, "always") == 0;
		}
		if (const char* value = std::getenv("KYTY_DISPATCH_RECOVERY_TRUST")) {
			p.trust_runs = static_cast<uint32_t>(std::strtoul(value, nullptr, 0));
		}
		return p;
	}();
	return policy;
}

// A missing BDA page returns zero during the first attempt. Preserve every
// externally writable buffer, including atomic destinations, so that attempt
// can be rolled back before any dependent dispatch observes its results.
static bool SingleRebind() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SINGLE_REBIND");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

class DispatchBufferRecovery {
public:
	DispatchBufferRecovery(RenderContext& context, const PreparedBindings& bindings,
	                       uint64_t shader_address)
	    : m_context(context), m_shader_address(shader_address) {
		const auto& program = *bindings.runtime->program;
		m_enabled           = program.info.uses_dma &&
		                      std::none_of(program.info.images.begin(), program.info.images.end(),
		                                   [](const auto& image) { return image.written; });
		if (!m_enabled) return;
		auto& cache     = context.GetBufferCache();
		auto& scheduler = context.GetCommandScheduler();
		auto& policy    = RecoveryPolicy();
		if (!policy.always) {
			if (const auto faults = cache.UnattributedFaults(); faults != policy.seen_faults) {
				policy.seen_faults = faults;
				if (!policy.clean_runs.empty()) {
					// Work that runs without the rollback (the trusted dispatches, draws) took a
					// device-address fault: a page it needed was not resident and read as zero there.
					Common::HackNotice::Report(
					    "device-address page fault outside a rollback-protected dispatch: its missing "
					    "pages read as zero",
					    "DispatchBufferRecovery", 0, 0,
					    "the shaders are protected again (KYTY_DISPATCH_RECOVERY_TRUST clean runs)");
				}
				policy.clean_runs.clear();
				policy.rearms++;
			}
			if (policy.clean_runs[shader_address] >= policy.trust_runs) {
				m_enabled = false;
				policy.Count(true);
				return;
			}
		}
		policy.Count(false);
		// Reports from earlier work cannot be attributed to this transaction.
		cache.ProcessFaultBuffer();
		(KYTY_SUBMIT_REASON(), scheduler.Finish());
		RangeSet ranges;
		for (const auto& resource: program.info.buffers) {
			// The sources are indexed by descriptor slot (FindBuffers), not by resource: the
			// buffers the shader never reads have no slot, and every one after them shifts down.
			if (!resource.written || resource.descriptor_index == UINT32_MAX) continue;
			EXIT_IF(resource.descriptor_index >= bindings.buffer_sources.size());
			const auto& source = bindings.buffer_sources[resource.descriptor_index];
			if (source.size == 0) continue;
			const auto start = source.address & ~uint64_t {3};
			const auto end   = (source.address + source.size + 3) & ~uint64_t {3};
			ranges.Add(start, end - start);
		}
		ranges.ForEach([&](uint64_t start, uint64_t end) {
			const auto id = cache.FindBuffer(start, end - start);
			cache.SynchronizeBuffersInRange(start, end - start);
			const auto& source = cache.GetBuffer(id);
			Save(source, source.Offset(start), start, end - start, false);
		});
		if (program.bindings.descriptor_counts[static_cast<size_t>(
		        ShaderRecompiler::IR::DescriptorBindingKind::Gds)] != 0) {
			const auto& source = *cache.GetGdsBuffer();
			Save(source, 0, 0, source.Size(), true);
		}
	}

	bool Retry() {
		if (!m_enabled) return false;
		auto&      cache  = m_context.GetBufferCache();
		const auto report = cache.CollectFaults();
		auto&      policy = RecoveryPolicy();
		if (report.page_count == 0) {
			if (report.trap.claimed != 0) {
				const auto hash =
				    (uint64_t {report.trap.shader_hash_high} << 32) | report.trap.shader_hash_low;
				EXIT("GPU shader trap after page residency: hash=0x%016" PRIx64
				     " pc=0x%08x code=0x%02x\n",
				     hash, report.trap.pc, report.trap.code);
			}
			if (m_attempts == 0) {
				policy.clean_runs[m_shader_address]++;
			}
			return false;
		}
		policy.clean_runs[m_shader_address] = 0;
		policy.retries++;
		if (++m_attempts > 128) {
			EXIT("GPU shader paging did not converge: shader=0x%016" PRIx64 " pages=%" PRIu64 "\n",
			     m_shader_address, report.page_count);
		}
		static uint32_t reports = 0;
		if (reports++ < 32) {
			::printf("GPU dispatch retry: shader=0x%016" PRIx64 " attempt=%u missing_pages=%" PRIu64
			         "\n",
			         m_shader_address, m_attempts, report.page_count);
		}
		for (const auto& saved: m_saved) {
			if (saved.gds) {
				cache.GetGdsBuffer()->CopyFrom(m_context.GetCommandScheduler().Current(),
				                               *saved.buffer, 0, 0, saved.buffer->Size());
			} else {
				// Resolving a fault can merge owners; never retain a native destination
				// handle or BufferId across CollectFaults().
				auto [destination, offset] =
				    cache.ObtainBuffer(saved.address, saved.buffer->Size(), true);
				destination->CopyFrom(m_context.GetCommandScheduler().Current(), *saved.buffer, 0,
				                      offset, saved.buffer->Size());
			}
		}
		m_context.PrepareBda();
		return true;
	}

private:
	void Save(const Buffer& source, uint64_t offset, uint64_t address, uint64_t size, bool gds) {
		auto saved =
		    std::make_unique<Buffer>(m_context.GetGraphics(), m_context.GetCommandScheduler(),
		                             MemoryUsage::DeviceLocal, 0, AllFlags, size);
		saved->CopyFrom(m_context.GetCommandScheduler().Current(), source, offset, 0, size);
		m_saved.push_back({address, gds, std::move(saved)});
	}
	struct Saved {
		uint64_t                address;
		bool                    gds;
		std::unique_ptr<Buffer> buffer;
	};
	RenderContext&     m_context;
	uint64_t           m_shader_address;
	bool               m_enabled  = false;
	uint32_t           m_attempts = 0;
	std::vector<Saved> m_saved;
};

} // namespace

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
	       (program.bindings.descriptor_counts[static_cast<size_t>(ShaderRecompiler::IR::DescriptorBindingKind::Buffers)] != 0 &&
	        std::ranges::any_of(program.info.buffers, [](const auto& buffer) {
		        return buffer.indirect_write_table;
	        }));
}

static bool PrepareIndirectWriteTargets(RenderContext& context, const PreparedBindings& bindings) {
	const auto& program = bindings.runtime->module->info;
	bool        prepared = false;
	for (const auto& resource: program.info.buffers) {
		// Buffer sources follow the bound resources, which leave out buffers the shader never
		// accesses, not the shader's buffer list.
		if (!resource.indirect_write_table || resource.descriptor_index == UINT32_MAX ||
		    resource.descriptor_index >= bindings.buffer_sources.size()) {
			continue;
		}
		const auto& table = bindings.buffer_sources[resource.descriptor_index];
		if (table.address == 0 || table.size <= resource.indirect_table_offset) {
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
	const auto& program   = input.stage.module->info;
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
	// Offset-or-swizzle bounds do not describe the element range proved by the shader.
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.OutOfBounds() == 3 || descriptor.Base48() == 0) {
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
	const auto elements = descriptor.NumRecords();
	const bool covered = fill.bounded_elements != 0
	                         ? fill.bounded_elements == elements && invocations >= elements
	                         : invocations == elements;
	if (!covered || invocations > uint64_t {UINT32_MAX} + 1 || size == 0 || size > UINT32_MAX ||
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

bool RenderExecutor::TryConsumeComputeFill(const ShaderComputeInputInfo& input,
                                          CommandBuffer& command, uint32_t group_x,
                                          uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = input.stage.module->info;
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
			    groups[axis] * threads > UINT32_MAX)
				return false;
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
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1])
			return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue                  clear {};
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
	const bool image_clear =
	    cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear);
	if (!image_clear) {
		// CPU fills invalidate overlapping cached images. Preserve the native dispatch for
		// partial or ambiguous image backings, including pooled metadata aliases.
		if ((descriptor.Base48() & 3u) != 0 ||
		    !m_context.IsMapped(descriptor.Base48(), size) ||
		    cache.IsRegionRegistered(descriptor.Base48(), size)) {
			return false;
		}
		// Metadata need not have an image backing yet. Keep proven fills in guest memory
		// when CPU-owned, so color-clear discovery can read them without a GPU download.
		m_context.GetBufferCache().FillBuffer(descriptor.Base48(), size, packed_clear, false);
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute %s shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     image_clear ? "image clear" : "buffer fill", input.stage.module->info.shader_hash,
		     descriptor.Base48(), size, packed_clear);
	}
	return true;
}

static void BindSharedMemory(RenderContext& context, ShaderComputeInputInfo& input,
                             PreparedBindings& bindings, uint64_t indirect_args = 0) {
	if (input.stage.module->info.bindings.descriptor_counts[
	        static_cast<size_t>(ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory)] == 0) {
		return;
	}
	auto& cache = context.GetBufferCache();
	// LDS has no contents to preserve between dispatches. The existing shader hazard
	// barriers also order other users of this GPU-only utility buffer.
	auto& storage = cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto limit = std::min<uint64_t>(storage.Size(),
	    context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	if (indirect_args != 0) {
		// The group counts are GPU data: reading them here made the GPU thread submit and wait
		// for the GPU (CommandScheduler::Wait, forced) on every such dispatch. The shader indexes
		// the buffer by its workgroup id; bind the whole range instead (KYTY_LDS_INDIRECT_READ=1:
		// read the counts as before). A range larger than groups x LDS changes nothing the shader
		// reads or writes; only a dispatch asking for more than the buffer holds would differ, and
		// Vulkan's robust access bounds it.
		static const bool read_counts = [] {
			const char* value = std::getenv("KYTY_LDS_INDIRECT_READ");
			return value != nullptr && value[0] == '1';
		}();
		if (!read_counts) {
			bindings.shared_memory = {storage.Handle(), 0, limit & ~uint64_t {3}};
			return;
		}
		cache.ReadMemory(indirect_args, sizeof(vk::DispatchIndirectCommand));
		std::memcpy(input.workgroup_counts, reinterpret_cast<const void*>(indirect_args),
		            sizeof(input.workgroup_counts));
	}
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

// KYTY_DISPATCH_RUNS: the run's last dispatch orders the whole run for what follows, also when it
// records nothing itself (skipped, empty).
class DispatchRunPost {
public:
	DispatchRunPost(RenderContext& context, CommandBuffer& buffer)
	    : m_context(context), m_buffer(buffer),
	      m_enabled(DispatchRunsEnabled() && DispatchSyncEnabled()) {}
	~DispatchRunPost() {
		if (!m_enabled) return;
		auto& run = DispatchRun();
		if (run.more || !run.pending_post) return;
		run.pending_post = false;
		Common::LockGuard lock(m_context.GetMutex());
		if (!m_buffer.IsInvalid()) {
			ShaderAccessBarrier(m_buffer.Recorder(), vk::PipelineStageFlagBits::eComputeShader);
		}
	}
	KYTY_CLASS_NO_COPY(DispatchRunPost);

private:
	RenderContext& m_context;
	CommandBuffer& m_buffer;
	bool           m_enabled;
};

// Diagnostics: KYTY_BVH_CHECK=<hash>:<counter buffer>:<leaf buffer> (bad108e74fb72e9f:5:0). The
// bottom-up BVH refit counts arrivals in a counter per node (bits 0-2; parent index in bits 3-29,
// slot in the parent above) and re-arms each counter when its node completes, so between refits
// every counter of a blob holds the value it started with. At the shader's first dispatch of each
// presented frame the GPU copies of the counters of the blobs refit in the last frames are compared
// with their reference ([bvh-check] on stderr):
// - only count bits differ: a refit left nodes incomplete (printed at once, with the last refits of
//   the blob and the workgroup and thread counts they ran with);
// - a valid armed tree with other links: the blob was instanced again (new reference, silently);
// - anything else: the memory holds other data now; printed only if the shader refits it again.
namespace {
struct BvhRefitNote {
	uint64_t frame     = 0;
	uint64_t seq       = 0;
	bool     continues = false;
	bool     batched   = false;
	uint32_t slot      = UINT32_MAX;
	uint32_t leaves    = 0;
	std::array<uint32_t, 6> counts {}; // groups, threads (filled by the first check after it)
	bool     counts_known = false;
};
struct BvhRecord {
	uint64_t                                    frame = 0, seq = 0, run = 0;
	const uint32_t*                             packet = nullptr;
	bool                                        continues = false, batched = false;
	uint32_t                                    slot = UINT32_MAX;
	std::array<std::pair<uint64_t, uint64_t>, 8> ranges {};
};
struct BvhBlob {
	uint64_t                 size = 0;
	std::vector<uint32_t>    ref;
	std::vector<uint32_t>    garbage; // the last snapshot, when it was neither the tree nor a tree
	std::deque<BvhRefitNote> notes;
	uint64_t                 last_refit = 0;
	bool                     reported   = false;
};
class BvhCheck {
public:
	static BvhCheck& Get() {
		static BvhCheck check;
		return check;
	}
	bool     enabled     = false;
	uint64_t hash        = 0;
	uint32_t counter_idx = 5;
	uint32_t leaf_idx    = 0;

	void OnDispatch(RenderContext& context, const PreparedBindings& bindings, bool batched,
	                vk::Buffer ring, uint64_t ring_bytes, vk::Buffer indirect_buffer,
	                uint64_t indirect_offset) {
		if (counter_idx >= bindings.buffer_sources.size()) return;
		const auto frame = context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
		std::lock_guard lock(m_mutex);
		if (frame != m_last_frame) {
			m_last_frame = frame;
			Check(context, frame, ring, ring_bytes);
		}
		const auto& source = bindings.buffer_sources[counter_idx];
		if (source.address == 0 || source.size == 0) return;
		auto& blob = m_blobs[source.address];
		const auto size = std::min<uint64_t>(source.size & ~uint64_t {3}, 0x2000u);
		if (blob.size != size) {
			blob = {};
			blob.size = size;
		}
		if (!blob.garbage.empty()) {
			if (m_reports < 60) {
				m_reports++;
				Print("refit reads foreign counters", frame, source.address, blob, blob.garbage);
			}
			blob.garbage.clear();
			blob.ref.clear(); // re-captured by the next check
		}
		BvhRefitNote note;
		note.frame     = frame;
		note.seq       = ++m_seq;
		note.continues = DispatchRunsEnabled() && DispatchRun().continues;
		note.batched   = batched;
		note.slot      = indirect_buffer == ring ? static_cast<uint32_t>(indirect_offset / 32u) : UINT32_MAX;
		note.leaves    = leaf_idx < bindings.buffer_sources.size()
		                     ? static_cast<uint32_t>(bindings.buffer_sources[leaf_idx].size / 4u)
		                     : 0u;
		blob.notes.push_back(note);
		if (blob.notes.size() > 6) blob.notes.pop_front();
		blob.last_refit = frame;
		if (!note.continues) m_run++;
		BvhRecord record;
		record.frame     = frame;
		record.seq       = note.seq;
		record.run       = m_run;
		record.packet    = DispatchRunsEnabled() ? DispatchRun().packet : nullptr;
		record.continues = note.continues;
		record.batched   = batched;
		record.slot      = note.slot;
		for (size_t i = 0; i < record.ranges.size() && i < bindings.buffer_sources.size(); i++) {
			record.ranges[i] = {bindings.buffer_sources[i].address, bindings.buffer_sources[i].size};
		}
		m_records.push_back(record);
		if (m_records.size() > 8192) m_records.pop_front();
	}

private:
	void Print(const char* what, uint64_t frame, uint64_t address, const BvhBlob& blob,
	           const std::vector<uint32_t>& now) {
		std::string changed;
		uint32_t    count = 0;
		for (size_t k = 0; k < now.size() && k < blob.ref.size(); k++) {
			if (now[k] != blob.ref[k] && count++ < 10) {
				changed += fmt::format(" [{}] {:08x}->{:08x}", k, blob.ref[k], now[k]);
			}
		}
		std::fprintf(stderr, "[bvh-check] %s: frame=%llu blob=0x%llx bytes=0x%llx words changed=%u:%s\n", what,
		             static_cast<unsigned long long>(frame), static_cast<unsigned long long>(address),
		             static_cast<unsigned long long>(blob.size), count, changed.c_str());
		PrintRecords(frame, address, blob.size, 0);
		for (const auto& note: blob.notes) {
			const std::string counts =
			    note.counts_known ? fmt::format(" groups={}x{}x{} threads={}x{}x{}", note.counts[0], note.counts[1],
			                                    note.counts[2], note.counts[3], note.counts[4], note.counts[5])
			                      : std::string(note.slot == UINT32_MAX ? " (direct groups)" : " (counts not captured)");
			std::fprintf(stderr, "[bvh-check]   refit frame=%llu seq=%llu continues=%d batched=%d leaves=%u%s\n",
			             static_cast<unsigned long long>(note.frame), static_cast<unsigned long long>(note.seq),
			             note.continues ? 1 : 0, note.batched ? 1 : 0, note.leaves, counts.c_str());
		}
	}

	// Every recent dispatch whose bindings overlap [address, address + size) (any binding).
	void PrintRecords(uint64_t frame, uint64_t address, uint64_t size, uint64_t focus) {
		uint32_t listed = 0;
		for (auto it = m_records.rbegin(); it != m_records.rend() && listed < 12; ++it) {
			if (it->frame + 4 < frame) break;
			int hit = -1;
			for (size_t b = 0; b < it->ranges.size(); b++) {
				const auto [a, n] = it->ranges[b];
				if (n != 0 && a < address + size && address < a + n) {
					hit = static_cast<int>(b);
					break;
				}
			}
			if (hit < 0) continue;
			listed++;
			std::string bufs;
			for (size_t b = 0; b < it->ranges.size(); b++) {
				if (it->ranges[b].second != 0) {
					bufs += fmt::format(" b{}=0x{:x}+0x{:x}", b, it->ranges[b].first, it->ranges[b].second);
				}
			}
			std::fprintf(stderr,
			             "[bvh-check]   %sdispatch frame=%llu seq=%llu run=%llu packet=%p continues=%d batched=%d slot=%u "
			             "hit=b%d%s\n",
			             it->seq == focus ? "* " : "", static_cast<unsigned long long>(it->frame), static_cast<unsigned long long>(it->seq),
			             static_cast<unsigned long long>(it->run), static_cast<const void*>(it->packet),
			             it->continues ? 1 : 0, it->batched ? 1 : 0, it->slot, hit, bufs.c_str());
		}
	}

	static bool ValidArmed(const std::vector<uint32_t>& words) {
		for (const auto w: words) {
			if (w == 0) continue;
			const auto count  = w & 7u;
			const auto parent = (w >> 3u) & 0x7ffffffu;
			if (count == 0 || count > 4 || parent >= words.size()) return false;
		}
		return true;
	}

	void Check(RenderContext& context, uint64_t frame, vk::Buffer ring, uint64_t ring_bytes) {
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
		std::vector<uint64_t>                      keys;
		for (const auto& [address, blob]: m_blobs) {
			if (blob.last_refit + 3 >= frame) {
				ranges.emplace_back(address, blob.size);
				keys.push_back(address);
			}
		}
		if (ranges.empty()) return;
		context.GetBufferCache().SnapshotGpu(
		    ranges, ring, 0, ring_bytes,
		    [this, frame, keys = std::move(keys)](std::vector<std::vector<uint32_t>>& words,
		                                          std::vector<uint32_t>&              ring_words) {
			    std::lock_guard lock(m_mutex);
			    m_checks++;
			    for (size_t i = 0; i < keys.size() && i < words.size(); i++) {
				    auto found = m_blobs.find(keys[i]);
				    if (found == m_blobs.end() || words[i].empty()) continue;
				    auto& blob = found->second;
				    if (words[i].size() * 4u != blob.size) continue;
				    // Counts of the refits since the last check: their ring slots are current.
				    for (auto& note: blob.notes) {
					    if (!note.counts_known && note.slot != UINT32_MAX && note.frame + 2 >= frame &&
					        (note.slot + 1u) * 8u <= ring_words.size()) {
						    const auto* e = &ring_words[note.slot * 8u];
						    note.counts       = {e[0], e[1], e[2], e[4], e[5], e[6]};
						    note.counts_known = true;
						    // Four threads a leaf: thread counts that disagree with the leaf list the
						    // dispatch bound ran this refit on another instance's arguments.
						    if (note.leaves != 0 && note.counts[3] != note.leaves * 4u) {
							    m_args_mismatch++;
							    if (m_mismatch_reports < 40) {
								    m_mismatch_reports++;
								    std::fprintf(stderr,
								                 "[bvh-check] refit arguments disagree with its leaf list: frame=%llu "
								                 "seq=%llu blob=0x%llx leaves=%u threads=%u groups=%u\n",
								                 static_cast<unsigned long long>(note.frame),
								                 static_cast<unsigned long long>(note.seq),
								                 static_cast<unsigned long long>(keys[i]), note.leaves, note.counts[3],
								                 note.counts[0]);
								    PrintRecords(frame, keys[i], blob.size, note.seq);
							    }
						    }
					    }
				    }
				    const auto& now = words[i];
				    if (blob.ref.empty()) {
					    if (ValidArmed(now)) blob.ref = now;
					    continue;
				    }
				    uint32_t count_only = 0, structural = 0, decremented = 0;
				    for (size_t k = 0; k < now.size(); k++) {
					    if (now[k] == blob.ref[k]) continue;
					    ((now[k] & ~7u) == (blob.ref[k] & ~7u) ? count_only : structural)++;
					    // Lowered by a few arrivals, borrowing into the parent index or not.
					    if (blob.ref[k] - now[k] <= 8u) decremented++;
				    }
				    if (structural != 0 && decremented == count_only + structural) {
					    // Refit arrivals beyond a node's count: the parent links are now wrong.
					    m_underflow++;
					    if (m_reports < 60) {
						    m_reports++;
						    Print("counters decremented past zero", frame, keys[i], blob, now);
					    }
					    blob.garbage = now;
					    continue;
				    }
				    if (count_only == 0 && structural == 0) {
					    m_matches++;
					    continue;
				    }
				    if (structural != 0) {
					    const bool valid = ValidArmed(now);
					    if (m_reports < 60) {
						    m_reports++;
						    Print(valid ? "counters now a different valid tree" : "counters now foreign", frame,
						          keys[i], blob, now);
					    }
					    if (valid) {
						    m_reinstanced++;
						    blob.ref = now;
					    } else {
						    m_foreign++;
						    blob.garbage = now;
					    }
					    continue;
				    }
				    m_partial++;
				    if (!blob.reported && m_reports < 60) {
					    blob.reported = true;
					    m_reports++;
					    Print("counters left incomplete", frame, keys[i], blob, now);
				    }
			    }
			    const auto t = std::chrono::steady_clock::now();
			    if (t - m_last_report > std::chrono::seconds(10)) {
				    m_last_report = t;
				    std::fprintf(stderr,
				                 "[bvh-check] blobs=%zu checks=%llu match=%llu incomplete=%llu reinstanced=%llu "
				                 "foreign=%llu underflow=%llu args-mismatch=%llu\n",
				                 m_blobs.size(), static_cast<unsigned long long>(m_checks),
				                 static_cast<unsigned long long>(m_matches), static_cast<unsigned long long>(m_partial),
				                 static_cast<unsigned long long>(m_reinstanced),
				                 static_cast<unsigned long long>(m_foreign),
				                 static_cast<unsigned long long>(m_underflow),
				                 static_cast<unsigned long long>(m_args_mismatch));
			    }
		    });
	}

	BvhCheck() {
		if (const char* value = std::getenv("KYTY_BVH_CHECK"); value != nullptr) {
			char* end = nullptr;
			hash      = std::strtoull(value, &end, 16);
			if (end != nullptr && *end == ':') counter_idx = std::strtoul(end + 1, &end, 10);
			if (end != nullptr && *end == ':') leaf_idx = std::strtoul(end + 1, &end, 10);
			enabled = hash != 0;
		}
	}
	std::recursive_mutex                  m_mutex;
	std::unordered_map<uint64_t, BvhBlob> m_blobs;
	uint64_t                              m_last_frame = UINT64_MAX;
	uint64_t                              m_seq        = 0;
	uint64_t m_checks = 0, m_matches = 0, m_partial = 0, m_reinstanced = 0, m_foreign = 0, m_underflow = 0;
	uint64_t                              m_run = 0;
	uint64_t                              m_args_mismatch = 0;
	uint32_t                              m_mismatch_reports = 0;
	std::deque<BvhRecord>                 m_records;
	uint32_t                              m_reports = 0;
	std::chrono::steady_clock::time_point m_last_report = std::chrono::steady_clock::now();
};
} // namespace

// Dispatches whose ordering the guest stream leaves out but which still need it (they keep their
// barriers inside runs). Marvel's Wolverine: 06544f382c6b0850 and 3378b4a64461d6ca (cs 0x11d04f8000
// and 0x11d04f3600, 14-26 groups each); overlapped inside a run they hung the GPU about 20 s into
// gameplay in about 1 boot in 3-4 (0 in 10 with their barriers kept, at the same frame rate).
// KYTY_DISPATCH_RUN_EXCLUDE=<hex shader hashes or cs addresses> replaces the list (empty: none);
// KYTY_DISPATCH_RUN_ONLY=<...> keeps runs for the listed shaders only (diagnostic).
static thread_local bool t_run_excluded = false;

static void NoteRunShader(uint64_t cs_addr, uint64_t shader_hash) {
	struct Lists {
		std::vector<uint64_t> exclude, only;
	};
	static const Lists lists = [] {
		Lists l;
		const auto parse = [](const char* value, std::vector<uint64_t>& out) {
			if (value == nullptr) return;
			for (const char* p = value; *p != '\0';) {
				char* end = nullptr;
				const auto v = std::strtoull(p, &end, 16);
				if (end == p) {
					p++;
					continue;
				}
				out.push_back(v);
				p = end;
			}
		};
		const char* exclude = std::getenv("KYTY_DISPATCH_RUN_EXCLUDE");
		parse(exclude != nullptr ? exclude : "06544f382c6b0850,3378b4a64461d6ca", l.exclude);
		parse(std::getenv("KYTY_DISPATCH_RUN_ONLY"), l.only);
		return l;
	}();
	const auto listed = [cs_addr, shader_hash](const std::vector<uint64_t>& list) {
		return std::find(list.begin(), list.end(), cs_addr) != list.end() ||
		       std::find(list.begin(), list.end(), shader_hash) != list.end();
	};
	t_run_excluded = listed(lists.exclude) || (!lists.only.empty() && !listed(lists.only));
	if (t_run_excluded && DispatchRunsEnabled()) {
		static std::atomic<uint64_t> excluded {0};
		if (Common::HackNotice::PowerOfTwoTick(excluded)) {
			Common::HackNotice::Report(
			    "dispatch of a listed shader keeps its barriers inside a dispatch run",
			    "NoteRunShader", shader_hash, cs_addr,
			    "KYTY_DISPATCH_RUN_EXCLUDE (default list: two shaders whose missing barriers hung the GPU)");
		}
	}
}

// Whether the barrier before this dispatch can be left out (an earlier dispatch of its run was
// recorded last), and the one after it (another dispatch of the run follows).
static bool RunSkipsBarrierBefore() {
	return DispatchRunsEnabled() && DispatchRun().continues && !t_run_excluded;
}
static bool RunSkipsBarrierAfter() {
	if (!DispatchRunsEnabled() || !DispatchRun().more || t_run_excluded) return false;
	DispatchRun().pending_post = true;
	DispatchRun().skipped_barriers++;
	return true;
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode, bool async_compute) {
	EXIT_IF(buffer.IsInvalid());
	DispatchRunPost run_post(m_context, buffer);
	m_context.GetCommandScheduler().PopPendingOperations();
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

	// The skip list's ray-tracing dispatches are most of this title's dispatches in the jungle
	// (~2,600 skip-list lookups a frame); drop them before preparing their inputs and program.
	// KYTY_EARLY_DISPATCH_SKIP=0 drops them in ProgramCache::Get as before.
	if (EarlyDispatchSkip() && PipelineCache::IsSkipListed(sh_ctx.GetCs().cs_regs.data_addr)) {
		Profiler::Add(Profiler::Counter::SkippedSkipList);
		ResetBindings();
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

	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	const bool use_thread_dimensions      = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
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
	{
		KYTY_PROFILER_BLOCK("Dispatch::GetComputeProgram");
		compute_program =
		    m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info);
	}
	if (!compute_program && PipelineCache::SrtDeferPending()) {
		// Suspended until the SRT bytes it reads land (KYTY_SRT_DEFER); not skipped.
		ResetBindings();
		return;
	}
	if (!compute_program) {
		char groups[48];
		std::snprintf(groups, sizeof(groups), "%ux%ux%u", thread_group_x, thread_group_y,
		              thread_group_z);
		LogSkippedDispatch(cs_regs, groups);
		Common::HackNotice::Report("dispatch skipped: its compute program is not available",
		                           "RenderExecutor::DispatchDirect",
		                           ShaderDeclaredHash(cs_regs.cs_regs.data_addr), 0);
		ResetBindings();
		return;
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0] = thread_group_x;
		input_info.dispatch_threads_num[1] = thread_group_y;
		input_info.dispatch_threads_num[2] = thread_group_z;
	}

	const auto& program   = *input_info.stage.program;
	NoteRunShader(cs_regs.cs_regs.data_addr, program.shader_hash);
	const auto& resources = *input_info.stage.resources;
	// KYTY_LOG_CS_IMAGES=1: each compute shader's written images (extent, format), once per shape.
	if (static const bool log_images = std::getenv("KYTY_LOG_CS_IMAGES") != nullptr; log_images) {
		static std::unordered_set<uint64_t> seen;
		for (uint32_t i = 0; i < program.info.images.size() && i < resources.images.size(); i++) {
			if (!program.info.images[i].written) continue;
			const auto r = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			const uint64_t key = program.shader_hash ^ (uint64_t {r.Width5()} << 1u) ^
			                     (uint64_t {r.Height5()} << 17u) ^ (uint64_t {static_cast<uint32_t>(r.Format())} << 33u) ^ (uint64_t {i} << 45u);
			if (seen.insert(key).second) {
				std::fprintf(stderr, "[cs-image] hash=%016llx image[%u] %ux%u fmt=%u groups=%ux%ux%u\n",
				             static_cast<unsigned long long>(program.shader_hash), i,
				             static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
				             static_cast<uint32_t>(r.Format()), thread_group_x, thread_group_y, thread_group_z);
			}
		}
	}
	{
		KYTY_PROFILER_BLOCK("Dispatch::TryConsumeClears");
		if (resources.specialization_reads.empty() &&
		    (TryConsumeComputeMetaClear(input_info, buffer) ||
		     TryConsumeComputeFill(input_info, buffer, thread_group_x, thread_group_y,
		                           thread_group_z, mode))) {
			ResetBindings();
			return;
		}
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
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info);
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
		KYTY_PROFILER_BLOCK("Dispatch::PrepareBda");
		m_context.CacheDmaBases(input_info.stage);
		m_context.PrepareBda();
	}
	RebindImages(bindings);
	BindSharedMemory(m_context, input_info, bindings);
	// The dispatch loop below binds the buffers again before recording; this first pass only
	// repeated every buffer's sync and the flattened SRT upload (KYTY_SINGLE_REBIND=0: as before).
	if (!SingleRebind()) RebindBuffers(bindings);

	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " hash=0x%016" PRIx64 " tick=%" PRIu64 " completed_tick=%" PRIu64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     m_context.GetCommandScheduler().GetMasterSemaphore().KnownGpuTick(),
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

	DispatchBufferRecovery recovery(m_context, bindings, cs_regs.cs_regs.data_addr);
	do {
		RebindImages(bindings);
		RebindBuffers(bindings);
		const auto        vk_buffer        = buffer.Recorder();
		PreparedBindings* descriptor_stage = &bindings;
		CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
		               std::span {&descriptor_stage, 1u});
		KYTY_PROFILER_BLOCK("Dispatch::Record");
		bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
		    HasShaderBufferWrites(input_info.stage);
		has_storage_writes =
		    std::any_of(program.info.images.begin(), program.info.images.end(),
		                [](const auto& image) {
			                return image.written &&
			                       image.resource_class ==
			                           ShaderRecompiler::IR::ImageResourceClass::Storage;
		                }) ||
		    has_storage_writes;
		if (has_storage_writes) {
			// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
			// while allowing the queue to execute asynchronously.
			if (DispatchSyncEnabled() && !RunSkipsBarrierBefore()) ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
		if (t_run_excluded && DispatchRunsEnabled() && DispatchRun().continues &&
		    DispatchSyncEnabled()) {
			// The dispatch before it left out its barrier after (KYTY_DISPATCH_RUN_EXCLUDE).
			ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
		GpuTiming::Before(m_context, buffer);
		const auto gpu_zone = BeginGpuZone(m_context, buffer, program.shader_hash);
		{
			// Name each compute shader once by its code address, so a device-fault checkpoint
			// (which prints the address) can be matched to a hash for KYTY_SKIP_SHADER_HASHES.
			static std::unordered_set<uint64_t> named;
			if (named.insert(cs_regs.cs_regs.data_addr).second) {
				std::fprintf(stderr, "[cs] addr=0x%016" PRIx64 " hash=0x%016" PRIx64 "\n",
				             static_cast<uint64_t>(cs_regs.cs_regs.data_addr), program.shader_hash);
			}
		}
		constexpr uint32_t DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE = 1u << 3u;
		const uint64_t     group_threads = uint64_t {std::max(input_info.threads_num[0], 1u)} *
		                                   std::max(input_info.threads_num[1], 1u) *
		                                   std::max(input_info.threads_num[2], 1u);
		if ((mode & DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE) != 0 && program.info.uses_gds &&
		    group_threads <= input_info.wave_size && !input_info.lds_storage) {
			// A single guest wave per group can run in launch order without a device-wide
			// spinlock. Keep the guest workgroup IDs when splitting the dispatch.
			for (uint32_t z = 0; z < thread_group_z; ++z) {
				for (uint32_t y = 0; y < thread_group_y; ++y) {
					for (uint32_t x = 0; x < thread_group_x; ++x) {
						if ((x | y | z) != 0u) {
							ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
						}
						vk_buffer.dispatchBase(x, y, z, 1, 1, 1);
					}
				}
			}
		} else {
			vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
		}
		Profiler::Add(Profiler::Counter::Dispatches);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
		GpuProfiler::End(buffer, gpu_zone);
		GpuTiming::After(m_context, buffer, program.shader_hash, GpuTiming::Dispatch);
		Timeline::Mark("dispatch", program.shader_hash,
		               (static_cast<uint64_t>(thread_group_x) << 32u) |
		                   (thread_group_y << 16u) | thread_group_z);

		// The removed host fence also ordered read-only dispatches before later writers.
		if (DispatchSyncEnabled() && !RunSkipsBarrierAfter()) {
			ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
			if (DispatchRunsEnabled()) DispatchRun().pending_post = false;
		}
	} while (recovery.Retry());
	ResetBindings();
}

void RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
                                      uint32_t mode, bool async_compute) {
	static const bool merged_barriers = [] {
		const char* value = std::getenv("KYTY_INDIRECT_MERGED_BARRIERS");
		return value != nullptr && value[0] == '1';
	}();
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0);
	DispatchRunPost run_post(m_context, buffer);
	// The arguments are thread counts: IndirectDispatchGroups converts them on the GPU, and the
	// shader bounds its threads by the counts it reads from the same memory.
	const bool use_thread_dimensions =
	    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	m_context.GetCommandScheduler().PopPendingOperations();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	Common::LockGuard lock(m_context.GetMutex());
	vk::DispatchIndirectCommand args {};
	if (!m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(args)) &&
	    LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args)) &&
	    (args.x == 0 || args.y == 0 || args.z == 0)) {
		return;
	}
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	if (EarlyDispatchSkip() && PipelineCache::IsSkipListed(cs_regs.cs_regs.data_addr)) {
		Profiler::Add(Profiler::Counter::SkippedSkipList);
		ResetBindings();
		return;
	}
	ShaderComputeInputInfo input_info {};
	input_info.dispatch_thread_dimensions   = use_thread_dimensions;
	input_info.dispatch_dimensions_indirect = use_thread_dimensions;
	input_info.async_compute = async_compute;
	input_info.indirect_args_address = args_addr;
	const auto compute_program = m_context.GetPipelineCache().GetComputeProgram(
	    cs_regs, buffer.GetRegisters().GetShaderRegisters(), input_info);
	if (!compute_program && PipelineCache::SrtDeferPending()) {
		// Suspended until the SRT bytes it reads land (KYTY_SRT_DEFER); not skipped.
		ResetBindings();
		return;
	}
	if (!compute_program) {
		LogSkippedDispatch(cs_regs, "indirect");
		Common::HackNotice::Report("dispatch skipped: its compute program is not available",
		                           "RenderExecutor::DispatchIndirect",
		                           ShaderDeclaredHash(cs_regs.cs_regs.data_addr), 0);
		ResetBindings();
		return;
	}
	{
		static std::unordered_set<uint64_t> named;
		if (named.insert(cs_regs.cs_regs.data_addr).second) {
			std::fprintf(stderr, "[cs-indirect] addr=0x%016" PRIx64 " hash=0x%016" PRIx64 "\n",
			             static_cast<uint64_t>(cs_regs.cs_regs.data_addr),
			             input_info.stage.program->shader_hash);
		}
	}
	if (use_thread_dimensions && m_indirect_groups == nullptr) {
		m_indirect_groups = std::make_unique<IndirectDispatchGroups>(
		    m_context.GetGraphics(), m_context.GetCommandScheduler());
	}
	buffer.EndRendering();
	// KYTY_DISPATCH_RUNS: the thread counts of the run's indirect dispatches are converted together
	// at its first one, between one barrier before and one after, instead of a conversion and its
	// barriers per dispatch.
	std::optional<DispatchRunConverted> run_converted;
	if (use_thread_dimensions && DispatchRunsEnabled() && DispatchSyncEnabled()) {
		auto& run = DispatchRun();
		if (const auto found = run.converted.find(run.packet);
		    run.packet != nullptr && found != run.converted.end()) {
			run_converted = found->second;
		} else if (run.upcoming.size() > 1 && run.upcoming.front().packet == run.packet) {
			run.converted.clear();
			const std::array<uint32_t, 3> current {std::max(cs_regs.cs_regs.num_thread_x, 1u),
			                                       std::max(cs_regs.cs_regs.num_thread_y, 1u),
			                                       std::max(cs_regs.cs_regs.num_thread_z, 1u)};
			vk::MemoryBarrier ready {};
			ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
			                      vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderRead;
			ready.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
			                      vk::AccessFlagBits::eIndirectCommandRead;
			buffer.Recorder().pipelineBarrier(
			    vk::PipelineStageFlagBits::eAllCommands,
			    vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eDrawIndirect,
			    {}, 1, &ready, 0, nullptr, 0, nullptr);
			for (const auto& entry: run.upcoming) {
				auto local = entry.local;
				for (uint32_t i = 0; i < 3; i++) {
					if (local[i] == 0) local[i] = current[i];
				}
				const auto [entry_buffer, entry_offset] = m_context.GetBufferCache().ObtainBuffer(
				    entry.args, sizeof(vk::DispatchIndirectCommand), false, false, {}, true);
				if (entry_buffer == nullptr || (entry_offset & 3u) != 0 ||
				    !entry_buffer->HasDeviceAddress()) {
					break;
				}
				const auto converted = m_indirect_groups->Convert(
				    buffer.Recorder(), entry_buffer->BufferDeviceAddress() + entry_offset, local,
				    false, false);
				run.converted[entry.packet] = {converted.groups_buffer, converted.groups_offset,
				                               converted.threads};
			}
			vk::MemoryBarrier groups_ready {};
			groups_ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
			groups_ready.dstAccessMask =
			    vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eShaderRead;
			buffer.Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
			                                  vk::PipelineStageFlagBits::eDrawIndirect |
			                                      vk::PipelineStageFlagBits::eComputeShader,
			                                  {}, 1, &groups_ready, 0, nullptr, 0, nullptr);
			run.batched += run.converted.size();
			run.upcoming.clear();
			if (const auto again = run.converted.find(run.packet); again != run.converted.end()) {
				run_converted = again->second;
			}
		}
	}
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	PreparedBindings* descriptor_stage = &bindings;
	FindBuffers(std::span {&descriptor_stage, 1u});
	const auto& program = *input_info.stage.program;
	NoteRunShader(cs_regs.cs_regs.data_addr, program.shader_hash);
	if (static const bool log_images = std::getenv("KYTY_LOG_CS_IMAGES") != nullptr; log_images) {
		static std::unordered_set<uint64_t> seen;
		const auto& resources = *input_info.stage.resources;
		for (uint32_t i = 0; i < program.info.images.size() && i < resources.images.size(); i++) {
			if (!program.info.images[i].written) continue;
			const auto r = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			const uint64_t key = program.shader_hash ^ (uint64_t {r.Width5()} << 1u) ^
			                     (uint64_t {r.Height5()} << 17u) ^ (uint64_t {static_cast<uint32_t>(r.Format())} << 33u) ^ (uint64_t {i} << 45u);
			if (seen.insert(key).second) {
				std::fprintf(stderr, "[cs-image] hash=%016llx image[%u] %ux%u fmt=%u indirect\n",
				             static_cast<unsigned long long>(program.shader_hash), i,
				             static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
				             static_cast<uint32_t>(r.Format()));
			}
		}
	}
	if (program.info.uses_dma) {
		m_context.CacheDmaBases(input_info.stage);
		m_context.PrepareBda();
	}
	BindSharedMemory(m_context, input_info, bindings, args_addr);
	if (program.info.uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	RebindImages(bindings);
	// Acquiring arguments can merge cache buffers; the loop below finalizes the shader bindings
	// afterward (KYTY_SINGLE_REBIND=0: also bind them here first, as before).
	if (!SingleRebind()) RebindBuffers(bindings);
	DispatchBufferRecovery recovery(m_context, bindings, cs_regs.cs_regs.data_addr);
	do {
		RebindImages(bindings);
		vk::Buffer     indirect_buffer = nullptr;
		vk::DeviceSize indirect_offset = 0;
		if (run_converted) {
			indirect_buffer  = run_converted->groups_buffer;
			indirect_offset  = run_converted->groups_offset;
			const auto dword = program.bindings.dispatch_thread_dword;
			if (dword != ShaderRecompiler::IR::PushData::NoStart) {
				bindings.shader_data[dword]      = static_cast<uint32_t>(run_converted->threads);
				bindings.shader_data[dword + 1u] = static_cast<uint32_t>(run_converted->threads >> 32u);
			}
		} else {
		const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
		    args_addr, sizeof(vk::DispatchIndirectCommand), false, false, {},
		    use_thread_dimensions);
		EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
		indirect_buffer = args_buffer->Handle();
		indirect_offset = args_offset;
		if (use_thread_dimensions) {
			EXIT_IF(!args_buffer->HasDeviceAddress());
			const std::array<uint32_t, 3> local_size {
			    std::max(cs_regs.cs_regs.num_thread_x, 1u),
			    std::max(cs_regs.cs_regs.num_thread_y, 1u),
			    std::max(cs_regs.cs_regs.num_thread_z, 1u)};
			// KYTY_INDIRECT_MERGED_BARRIERS=1: one barrier orders the thread counts' writers and the
			// guest dispatch's write hazards before the conversion, and the conversion's own barrier
			// orders the groups before the indirect read, instead of four barriers around the pair
			// (the async ray-tracing chain: ~1,300 tiny indirect dispatches a frame on the title).
			if (merged_barriers) {
				vk::MemoryBarrier ready {};
				ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
				                      vk::AccessFlagBits::eTransferWrite |
				                      vk::AccessFlagBits::eShaderRead;
				ready.dstAccessMask = vk::AccessFlagBits::eShaderRead |
				                      vk::AccessFlagBits::eShaderWrite |
				                      vk::AccessFlagBits::eIndirectCommandRead;
				buffer.Recorder().pipelineBarrier(
				    vk::PipelineStageFlagBits::eAllCommands,
				    vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eDrawIndirect,
				    {}, 1, &ready, 0, nullptr, 0, nullptr);
			}
			const auto converted = m_indirect_groups->Convert(
			    buffer.Recorder(), args_buffer->BufferDeviceAddress() + args_offset, local_size,
			    !merged_barriers);
			indirect_buffer = converted.groups_buffer;
			indirect_offset = converted.groups_offset;
			// The shader reads the thread counts through this device address (upstream's
			// dispatch-thread words hold the address instead of the counts).
			const auto dword = program.bindings.dispatch_thread_dword;
			if (dword != ShaderRecompiler::IR::PushData::NoStart) {
				bindings.shader_data[dword]      = static_cast<uint32_t>(converted.threads);
				bindings.shader_data[dword + 1u] = static_cast<uint32_t>(converted.threads >> 32u);
			}
		}
		}
		RebindBuffers(bindings);
		PreparedBindings* descriptor_stage = &bindings;
		CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
		               std::span {&descriptor_stage, 1u});
		const auto vk_buffer = buffer.Recorder();
		const bool has_storage_writes =
		    bindings.shared_memory.buffer != nullptr || HasShaderBufferWrites(input_info.stage) ||
		    std::any_of(
		        program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			        return image.written && image.resource_class ==
			                                    ShaderRecompiler::IR::ImageResourceClass::Storage;
		        });
		const bool merged = (merged_barriers && use_thread_dimensions) || run_converted.has_value();
		if (has_storage_writes && !merged) {
			if (DispatchSyncEnabled() && !RunSkipsBarrierBefore()) ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
		if (!merged) {
			vk::MemoryBarrier barrier {};
			barrier.srcAccessMask =
			    vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eAllGraphics | vk::PipelineStageFlagBits::eComputeShader |
			        vk::PipelineStageFlagBits::eTransfer,
			    vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier, 0, nullptr, 0, nullptr);
		}
		if (auto& check = BvhCheck::Get(); check.enabled && program.shader_hash == check.hash) {
			check.OnDispatch(m_context, bindings, run_converted.has_value(),
			                 m_indirect_groups ? m_indirect_groups->RingHandle() : vk::Buffer {},
			                 IndirectDispatchGroups::RingBytes(), indirect_buffer, indirect_offset);
		}
		if (t_run_excluded && DispatchRunsEnabled() && DispatchRun().continues &&
		    DispatchSyncEnabled()) {
			// The dispatch before it left out its barrier after (KYTY_DISPATCH_RUN_EXCLUDE).
			ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
		GpuTiming::Before(m_context, buffer);
		const auto gpu_zone = BeginGpuZone(m_context, buffer, program.shader_hash);
		vk_buffer.dispatchIndirect(indirect_buffer, indirect_offset);
		Profiler::Add(Profiler::Counter::Dispatches);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
		GpuProfiler::End(buffer, gpu_zone);
		GpuTiming::After(m_context, buffer, program.shader_hash, GpuTiming::Dispatch);
		Timeline::Mark("dispatch-indirect", program.shader_hash, 0);
		if (DispatchSyncEnabled() && !RunSkipsBarrierAfter()) {
			ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
			if (DispatchRunsEnabled()) DispatchRun().pending_post = false;
		}
	} while (recovery.Retry());
	ResetBindings();
}

} // namespace Libs::Graphics
