// Table dispatches (chenxiao07's KytyPS5 fork, src/local/table-xpr.inc TableDispatch; KYTY_TABLE_DISPATCH=1).
//
// A compute program translated in table mode (ShaderRecompiler::IR::EnterTableMode) reads its SRT slots and its
// buffers from a block the renderer writes for each dispatch: the slots its plan marks, evaluated from guest memory
// as it holds them now, and each buffer's device address and size, resolved from its V#. The dispatch needs no
// resource materialization (the SRT walk), no buffer descriptors and no flattened SRT upload: the block's device
// address goes in the push constants with the user data. Its images and samplers, words the plan evaluates too, are
// bound as the normal path binds them, and so is the page table of flat and global memory reads.
//
// A program is learnt from the normal path's lookup of one of its dispatches (TableDispatchLearn): its table program
// is compiled for that dispatch's specialization; later dispatches whose descriptors select another specialization
// get one compiled for it. Compilations run on a background thread, the dispatches taking the normal path until they
// are done. Programs table mode cannot serve (indirect descriptors and images, GDS, ...) take it for good.
//
// The resolve thread evaluates the block and selects the program ahead of the dispatch (TableDispatchAhead) instead
// of the normal resolution, reading guest memory through the resolution's log; the execution thread then only gives
// the buffers their device addresses.
//
// Included by renderCompute.cpp.

namespace {

bool TableDispatchEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_TABLE_DISPATCH");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

uint64_t TableDispatchKey(const HW::ComputeShaderInfo& cs, bool async_compute) {
	return XXH3_64bits_withSeed(&cs.cs_regs, sizeof(cs.cs_regs), async_compute ? 1u : 0u);
}

// What table dispatches serve: a table program whose shader data fits the push constants.
bool TableServes(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	return program.table_mode && program.bindings.UsesPushData() && !program.info.uses_gds &&
	       program.bindings.dispatch_thread_dword == ShaderRecompiler::IR::PushData::NoStart;
}

} // namespace

struct RenderExecutor::TableDispatchCache {
	struct Entry {
		const ShaderRecompiler::IR::ResourcePlan*                 plan = nullptr; // the normal program's
		std::vector<const PipelineCache::TableProgram*>           variants;
		std::vector<ShaderRecompiler::IR::ResourceSpecialization> refused_specializations;
		bool                                                      refused = false;
		// A compilation in flight (execution thread only) and the specialization it is for.
		std::shared_ptr<PipelineCache::TableTicket>  pending;
		ShaderRecompiler::IR::ResourceSpecialization pending_specialization;
	};
	// The execution thread changes entries under the lock; the resolve thread reads them under it.
	std::unordered_map<uint64_t, Entry> entries;
	std::atomic_flag                    lock;
	// Compilations in flight, at most MaxPending.
	static constexpr uint32_t MaxPending = 32;
	uint32_t                  pending    = 0;
	// The dispatch's resources (its bindings refer to them until the next operation) and scratch.
	ShaderRecompiler::IR::ResourceSnapshot       snapshot;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	std::vector<uint32_t>                        block;
	ShaderComputeInputInfo                       input;

	// Execution thread: takes the entry's finished compilation in.
	void Poll(Entry& entry) {
		if (entry.pending == nullptr || !entry.pending->done.load(std::memory_order_acquire)) return;
		const auto* table = entry.pending->programs[0];
		const TableLockGuard guard(lock);
		if (table != nullptr && TableServes(table->program) &&
		    (entry.variants.empty() || table->program.table_plan == entry.variants.front()->program.table_plan)) {
			entry.variants.push_back(table);
		} else if (entry.variants.empty()) {
			entry.refused = true;
		} else {
			entry.refused_specializations.push_back(entry.pending_specialization);
		}
		entry.pending.reset();
		--pending;
	}
	// Execution thread: queues the compilation of the entry's table program for a specialization.
	void Queue(Entry& entry, PipelineCache& pipelines, const HW::ComputeShaderInfo& cs, const HW::ShaderRegisters& sh,
	           bool async_compute, const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
		if (entry.pending != nullptr || pending >= MaxPending) return;
		entry.pending                = pipelines.CompileTableComputeAsync(cs, sh, async_compute, specialization);
		entry.pending_specialization = specialization;
		++pending;
	}
};

RenderExecutor::TableDispatchCache& RenderExecutor::TableDispatchState() {
	std::call_once(m_table_dispatch_once, [this] {
		m_table_dispatch = {new TableDispatchCache {}, [](TableDispatchCache* cache) { delete cache; }};
	});
	return *m_table_dispatch;
}

void RenderExecutor::TableDispatchLearn(const HW::ComputeShaderInfo& cs, const HW::ShaderRegisters& sh,
                                        const ShaderComputeInputInfo& input_info, const ShaderProgram& compute_program,
                                        bool async_compute) {
	if (!TableDispatchEnabled() || input_info.dispatch_thread_dimensions || input_info.stage.program == nullptr) return;
	auto&       cache   = TableDispatchState();
	const auto  key     = TableDispatchKey(cs, async_compute);
	auto        found   = cache.entries.find(key);
	const auto& program = *input_info.stage.program;
	if (found != cache.entries.end() && (found->second.refused || !found->second.variants.empty())) return;
	if (found == cache.entries.end()) {
		const TableLockGuard lock(cache.lock);
		found = cache.entries.emplace(key, TableDispatchCache::Entry {}).first;
	}
	auto& entry = found->second;
	cache.Poll(entry);
	if (entry.refused || !entry.variants.empty() || entry.pending != nullptr) return;
	const auto refuse = [&] {
		const TableLockGuard lock(cache.lock);
		entry.refused = true;
	};
	if (std::ranges::find(DemonsSoulsCopy::Shaders, program.shader_hash) != DemonsSoulsCopy::Shaders.end() ||
	    program.info.uses_gds || program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		refuse();
		return;
	}
	auto&      pipelines                 = m_context.GetPipelineCache();
	const auto [plan, specialization] = pipelines.ProgramPlanOf(compute_program);
	if (plan == nullptr || specialization == nullptr) {
		refuse();
		return;
	}
	{
		const TableLockGuard lock(cache.lock);
		entry.plan = plan;
	}
	cache.Queue(entry, pipelines, cs, sh, async_compute, *specialization);
}

// The table program of a dispatch and its resources, into `snapshot`; null when the normal path takes it. Ahead of
// execution (`reads`) guest memory is read through the resolution's log and no program is compiled.
const PipelineCache::TableProgram* RenderExecutor::TableDispatchSelect(
    const HW::ComputeShaderInfo& cs, const HW::ShaderRegisters* sh, bool async_compute,
    ShaderRecompiler::IR::ResourceSnapshot& snapshot, ShaderRecompiler::IR::ResourceSpecialization& specialization,
    GuestReadLog* reads) {
	auto&                                     cache     = TableDispatchState();
	const auto                                key       = TableDispatchKey(cs, async_compute);
	const ShaderRecompiler::IR::ResourcePlan* plan      = nullptr;
	const PipelineCache::TableProgram*        reference = nullptr;
	{
		std::optional<TableLockGuard> lock;
		if (reads != nullptr) lock.emplace(cache.lock);
		const auto found = cache.entries.find(key);
		if (found == cache.entries.end() || found->second.refused || found->second.variants.empty()) return nullptr;
		plan      = found->second.plan;
		reference = found->second.variants.front();
	}
	const auto user_data = std::span<const uint32_t>(cs.cs_user_sgpr.value).first(cs.cs_regs.user_sgpr);
	auto&      pipelines = m_context.GetPipelineCache();
	if (!TableEvaluate(reference->program, user_data, snapshot, reads) ||
	    !pipelines.SpecializeDirectResources(*plan, snapshot, specialization))
		return nullptr;
	if (reads != nullptr) {
		const TableLockGuard lock(cache.lock);
		for (const auto* variant: cache.entries.find(key)->second.variants) {
			if (variant->specialization == specialization) return variant;
		}
		return nullptr;
	}
	auto& entry = cache.entries.find(key)->second;
	cache.Poll(entry);
	for (const auto* variant: entry.variants) {
		if (variant->specialization == specialization) return variant;
	}
	if (std::ranges::find(entry.refused_specializations, specialization) == entry.refused_specializations.end())
		cache.Queue(entry, pipelines, cs, *sh, async_compute, specialization);
	return nullptr;
}

const PipelineCache::TableProgram* RenderExecutor::TableDispatchAhead(const HW::ComputeShaderInfo& cs,
                                                                      bool async_compute, ProgramResolution& ahead) {
	if (!TableDispatchEnabled()) return nullptr;
	return TableDispatchSelect(cs, nullptr, async_compute, ahead.resources[0], ahead.specializations[0],
	                           &ahead.reads);
}

bool RenderExecutor::TableDispatch(CommandBuffer& buffer, const HW::ComputeShaderInfo& cs, uint32_t x, uint32_t y,
                                   uint32_t z, bool async_compute) {
	if (!TableDispatchEnabled()) return false;
	auto&                                   cache    = TableDispatchState();
	const PipelineCache::TableProgram*      table    = nullptr;
	ShaderRecompiler::IR::ResourceSnapshot* resolved = nullptr;
	if (!TakeResolvedTable(table, resolved)) {
		table = TableDispatchSelect(cs, &buffer.GetRegisters().GetShaderRegisters(), async_compute, cache.snapshot,
		                            cache.specialization, nullptr);
		if (table == nullptr) return false;
		resolved = &cache.snapshot;
	}
	auto&       snapshot  = *resolved;
	const auto& slots     = snapshot.flattened_srt;
	const auto& plan      = table->program.table_plan;
	auto&       pipelines = m_context.GetPipelineCache();
	const auto& program = table->program;
	auto&       input   = cache.input;
	input               = table->input_info;
	input.workgroup_counts[0] = x;
	input.workgroup_counts[1] = y;
	input.workgroup_counts[2] = z;
	input.stage               = {.program = &program, .resources = &snapshot};
	// As DispatchDirect(): the metadata clears of the images it writes, and the clears it is.
	if (!program.info.images.empty()) {
		UpdateComputeMetaClears(input, buffer);
		if (TryConsumeComputeImageClear(input, buffer, x, y, z, 0x41u)) {
			ResetBindings();
			return true;
		}
	}
	buffer.EndRendering();
	auto& pipeline = pipelines.GetComputePipeline(input, table->handle);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input.stage, bindings);
	PreparedBindings* descriptor_stage = &bindings;
	FindBuffers(std::span {&descriptor_stage, 1u});
	if (program.info.uses_dma) {
		m_context.CacheDmaBases(input.stage);
		const ShaderStageRuntime* const dma_stage = &input.stage;
		const bool reach_synchronized = m_context.SynchronizeDmaFootprint(std::span {&dma_stage, 1});
		m_context.PrepareBda(program.has_address_writes, !reach_synchronized);
	}
	// Each buffer's device address and size. A registration (a new buffer joining older ones) may replace a buffer
	// an earlier lookup gave: the addresses are resolved again until none moved.
	auto& cache_buffers = m_context.GetBufferCache();
	auto& block         = cache.block;
	bool  writes        = false;
	for (int attempt = 0;; ++attempt) {
		if (attempt == 3) {
			ResetBindings();
			return false;
		}
		const auto epoch = cache_buffers.BufferSetEpoch();
		block.assign(plan.BlockDwords(), 0);
		std::copy(slots.begin(), slots.end(), block.begin());
		for (size_t i = 0; i < plan.buffers.size(); ++i) {
			const auto  v_sharp  = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
			const auto& resource = program.info.buffers[i];
			if (!resource.live || v_sharp.Type() != 0) continue;
			const auto address = v_sharp.Base48();
			auto       size    = v_sharp.GetSize();
			if (address == 0 || size == 0) continue;
			if ((address & 3u) != 0 || v_sharp.NumRecords() == UINT32_MAX) {
				ResetBindings();
				return false;
			}
			size = Libs::LibKernel::Memory::ClampRangeSize(address, size);
			if (size == 0) continue;
			auto [owner, owner_offset] = cache_buffers.ObtainBuffer(
			    address, size, resource.written,
			    resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled()));
			if (resource.written) {
				m_context.GetTextureCache().InvalidateMemoryFromGPU(address, size);
				writes = true;
			}
			const auto device = owner->BufferDeviceAddress() + owner_offset;
			const auto first  = plan.BufferDword(i);
			block[first]      = static_cast<uint32_t>(device);
			block[first + 1]  = static_cast<uint32_t>(device >> 32u);
			block[first + 2]  = static_cast<uint32_t>(std::min<uint64_t>(size / 4u, UINT32_MAX));
			block[first + 4]  = v_sharp.Stride();
		}
		if (cache_buffers.BufferSetEpoch() == epoch) break;
	}
	RebindImages(bindings);
	BindSharedMemory(m_context, input, bindings);
	RebindBuffers(bindings);
	const auto block_address = cache_buffers.TableUpload(block.data(), block.size() * sizeof(uint32_t));
	bindings.shader_data[program.bindings.TableBlockDword()]      = static_cast<uint32_t>(block_address);
	bindings.shader_data[program.bindings.TableBlockDword() + 1u] = static_cast<uint32_t>(block_address >> 32u);
	auto vk_buffer = buffer.Handle();
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline, std::span {&descriptor_stage, 1u});
	writes = writes || bindings.shared_memory.buffer != nullptr ||
	         std::ranges::any_of(program.info.images, [](const auto& image) {
		         return image.written && image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
	         });
	// As DispatchDirect(): a dispatch the guest did not order after the previous one of its queue overlaps it.
	const bool chained = std::exchange(m_dispatch_chained, false) && m_deferred_barrier;
	if (writes && !chained) {
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	vk_buffer.dispatch(x, y, z);
	m_deferred_barrier       = true;
	m_deferred_barrier_queue = m_guest_queue;
	ResetBindings();
	return true;
}
