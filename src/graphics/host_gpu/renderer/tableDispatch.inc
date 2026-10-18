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
// compile one for it (a few per frame), or take the normal path. Programs table mode cannot serve (indirect
// descriptors and images, GDS, ...) take the normal path for good.
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

// A dword of guest memory as a dispatch would read it now; false where the normal path must read it (its pages hold
// bytes the GPU wrote, or nothing is mapped there).
bool TableReadWord(uint64_t address, uint32_t& word) {
	if ((address & 3u) != 0 || !GuestRange {address, 4}.Valid()) return false;
	if (Libs::LibKernel::Memory::MayFaultOnGpuRead(address, 4))
		return Libs::LibKernel::Memory::TryReadAroundGpuWrites(address, &word, 4);
	return Libs::LibKernel::Memory::TryReadBacking(address, &word, 4);
}

} // namespace

struct RenderExecutor::TableDispatchCache {
	struct Entry {
		const ShaderRecompiler::IR::ResourcePlan*                 plan = nullptr; // the normal program's
		std::vector<const PipelineCache::TableProgram*>           variants;
		std::vector<ShaderRecompiler::IR::ResourceSpecialization> refused_specializations;
		bool                                                      refused = false;
	};
	std::unordered_map<uint64_t, Entry> entries;
	// Table programs compiled on the execution thread, per frame.
	static constexpr uint32_t CompileBudget = 4;
	uint64_t                  budget_frame  = UINT64_MAX;
	uint32_t                  compiles      = 0;
	// The dispatch's resources (its bindings refer to them until the next operation) and scratch.
	ShaderRecompiler::IR::ResourceSnapshot       snapshot;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	std::vector<uint32_t>                        block;
	ShaderComputeInputInfo                       input;

	bool TakeBudget(uint64_t frame) {
		if (frame != budget_frame) {
			budget_frame = frame;
			compiles     = 0;
		}
		return compiles++ < CompileBudget;
	}
};

RenderExecutor::TableDispatchCache& RenderExecutor::TableDispatchState() {
	if (!m_table_dispatch) {
		m_table_dispatch = {new TableDispatchCache {}, [](TableDispatchCache* cache) { delete cache; }};
	}
	return *m_table_dispatch;
}

void RenderExecutor::TableDispatchLearn(const HW::ComputeShaderInfo& cs, const HW::ShaderRegisters& sh,
                                        const ShaderComputeInputInfo& input_info, const ShaderProgram& compute_program,
                                        bool async_compute) {
	if (!TableDispatchEnabled() || input_info.dispatch_thread_dimensions || input_info.stage.program == nullptr) return;
	auto&       cache   = TableDispatchState();
	auto&       entry   = cache.entries[TableDispatchKey(cs, async_compute)];
	const auto& program = *input_info.stage.program;
	if (entry.refused || !entry.variants.empty()) return;
	if (std::ranges::find(DemonsSoulsCopy::Shaders, program.shader_hash) != DemonsSoulsCopy::Shaders.end() ||
	    program.info.uses_gds || program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		entry.refused = true;
		return;
	}
	if (!cache.TakeBudget(m_context.FrameNumber())) return;
	auto&      pipelines                 = m_context.GetPipelineCache();
	const auto [plan, specialization] = pipelines.ProgramPlanOf(compute_program);
	if (plan == nullptr || specialization == nullptr) {
		entry.refused = true;
		return;
	}
	ShaderComputeInputInfo table_input {};
	table_input.async_compute = async_compute;
	std::string refusal;
	const auto* table = pipelines.CompileTableCompute(cs, sh, table_input, *specialization, refusal);
	if (table == nullptr || !TableServes(table->program)) {
		entry.refused = true;
		return;
	}
	entry.plan = plan;
	entry.variants.push_back(table);
}

bool RenderExecutor::TableDispatch(CommandBuffer& buffer, const HW::ComputeShaderInfo& cs, uint32_t x, uint32_t y,
                                   uint32_t z, bool async_compute) {
	if (!TableDispatchEnabled()) return false;
	auto&      cache = TableDispatchState();
	const auto found = cache.entries.find(TableDispatchKey(cs, async_compute));
	if (found == cache.entries.end() || found->second.refused || found->second.variants.empty()) return false;
	auto&       entry     = found->second;
	const auto& reference = entry.variants.front()->program;
	const auto& plan      = reference.table_plan;
	const auto  user_data = std::span<const uint32_t>(cs.cs_user_sgpr.value).first(cs.cs_regs.user_sgpr);
	// The slots the renderer evaluates, in order (a slot's address depends on lower slots and user data only).
	using Kind     = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto& snapshot = cache.snapshot;
	auto& slots    = snapshot.flattened_srt;
	slots.assign(plan.slots.size(), 0);
	const auto value_of = [&](const ShaderRecompiler::IR::TablePlan::Operand& operand, uint32_t& value) {
		switch (operand.kind) {
			case Kind::Immediate: value = operand.value; return true;
			case Kind::UserData:
				if (operand.value < reference.user_data_base ||
				    operand.value - reference.user_data_base >= user_data.size())
					return false;
				value = user_data[operand.value - reference.user_data_base];
				return true;
			case Kind::Slot:
				if (operand.value >= slots.size()) return false;
				value = slots[operand.value];
				return true;
			default: return false;
		}
	};
	for (size_t index = 0; index < plan.slots.size(); ++index) {
		if (plan.cpu[index] == 0) continue;
		const auto& slot = plan.slots[index];
		uint32_t    low = 0, high = 0, offset = 0, records = 0;
		if (!value_of(slot.low, low) || !value_of(slot.high, high)) return false;
		uint64_t address = 0;
		if (slot.bounded) {
			// Read through a V#: past its range (NUM_RECORDS bytes, or times the stride) the normal path's evaluation
			// fails too.
			if (!value_of(slot.records, records)) return false;
			const uint64_t stride = (high >> 16u) & 0x3fffu;
			const uint64_t range  = stride == 0 ? records : stride * records;
			if (static_cast<uint64_t>(slot.immediate) + 4u > range) return false;
			address = (uint64_t {low} | uint64_t {high & 0xffffu} << 32u) + static_cast<uint64_t>(slot.immediate);
		} else {
			if (!value_of(slot.offset, offset)) return false;
			address = ((uint64_t {low} | uint64_t {high} << 32u) & 0xffff'ffff'ffffull) + (offset & ~3u) +
			          static_cast<int64_t>(slot.immediate);
		}
		if (!TableReadWord(address, slots[index])) return false;
	}
	// The descriptors, and the specialization they select.
	const auto evaluate = [&](const ShaderRecompiler::IR::TablePlan::Operand* words, uint32_t count,
	                          ShaderRecompiler::IR::DescriptorValue& value) {
		value             = {};
		value.dword_count = count;
		for (uint32_t word = 0; word < count; ++word)
			if (!value_of(words[word], value.dwords[word])) return false;
		return true;
	};
	snapshot.buffers.resize(plan.buffers.size());
	for (size_t i = 0; i < plan.buffers.size(); ++i)
		if (!evaluate(plan.buffers[i].data(), 4, snapshot.buffers[i])) return false;
	snapshot.images.resize(plan.images.size());
	for (size_t i = 0; i < plan.images.size(); ++i)
		if (!evaluate(plan.images[i].words.data(), plan.images[i].count, snapshot.images[i])) return false;
	snapshot.samplers.resize(plan.samplers.size());
	for (size_t i = 0; i < plan.samplers.size(); ++i)
		if (!evaluate(plan.samplers[i].words.data(), plan.samplers[i].count, snapshot.samplers[i])) return false;
	snapshot.user_data.assign(user_data.begin(), user_data.end());
	snapshot.specialization_reads.clear();
	snapshot.specialization_read_slots.clear();
	snapshot.image_tables.clear();
	snapshot.uniform_fill = {};
	auto& pipelines      = m_context.GetPipelineCache();
	auto& specialization = cache.specialization;
	if (!pipelines.SpecializeDirectResources(*entry.plan, snapshot, specialization)) return false;
	const PipelineCache::TableProgram* table = nullptr;
	for (const auto* variant: entry.variants) {
		if (variant->specialization == specialization) {
			table = variant;
			break;
		}
	}
	if (table == nullptr) {
		if (std::ranges::find(entry.refused_specializations, specialization) != entry.refused_specializations.end() ||
		    !cache.TakeBudget(m_context.FrameNumber()))
			return false;
		ShaderComputeInputInfo table_input {};
		table_input.async_compute = async_compute;
		std::string refusal;
		table = pipelines.CompileTableCompute(cs, buffer.GetRegisters().GetShaderRegisters(), table_input,
		                                      specialization, refusal);
		if (table == nullptr || !TableServes(table->program) || !(table->program.table_plan == plan)) {
			entry.refused_specializations.push_back(specialization);
			return false;
		}
		entry.variants.push_back(table);
	}
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
