// --- Table dispatches --------------------------------------------------------------------------
//
// Ported from chenxiao07's KytyPS5 fork (src/local/table-xpr.inc, TableDispatch). Included by renderDraw.cpp after
// tableDraw.inc, whose pairs, image sets and blocks they share.
//
// KYTY_TABLE_DISPATCH=1: a compute shader the normal path dispatched (its resources evaluated in full), whose program
// allows table mode (ShaderRecompiler::IR::EnterTableMode), is dispatched as table draws are drawn: its slots in a
// block, its read-only V#s resolved once, its written ones obtained as written at every dispatch (the buffer cache
// tracks the GPU's writes), its images and samplers in an image set (TableSet) bound with its native pipeline; the
// barriers as the normal path's DispatchDirect. Shaders a special path of the normal path takes (Demon's Souls' copy),
// thread-dimension dispatches, uniform fills and programs that use LDS storage stay on the normal path.

namespace {

bool TableDispatchEnabled() {
	static const bool enabled = [] {
		// (Demon's Souls by default, as chenxiao07's launcher sets it.)
		const char* value = std::getenv("KYTY_TABLE_DISPATCH");
		return value != nullptr ? value[0] == '1' : Loader::IsDemonsSoulsTitle();
	}();
	return enabled;
}

uint64_t TableDispatchKey(const HW::ComputeShaderInfo& cs, bool async_compute, bool thread_dimensions) {
	return XXH3_64bits_withSeed(&cs.cs_regs, sizeof(cs.cs_regs), (async_compute ? 1u : 0u) | (thread_dimensions ? 2u : 0u));
}

// chenxiao07's table dispatches also take indirect dispatches (vkCmdDispatchIndirect, the arguments read by the GPU,
// as the normal path's DispatchIndirect) and dispatches of thread counts (their groups, the counts in the program's
// shader data): KYTY_TABLE_DISPATCH_INDIRECT and KYTY_TABLE_DISPATCH_THREADS (default 1; 0 leaves them to the normal
// path as before).
bool TableDispatchIndirectEnabled() {
	static const bool enabled = NativeXprSwitch("KYTY_TABLE_DISPATCH_INDIRECT");
	return enabled;
}

bool TableDispatchThreadsEnabled() {
	static const bool enabled = NativeXprSwitch("KYTY_TABLE_DISPATCH_THREADS");
	return enabled;
}

// chenxiao07's table dispatches write storage images (the async culling chain's, the downsampling passes'): the set
// binds them in General as CommitBindings does, a use proven GPU-owned and CPU-clean (TableTransit).
// KYTY_TABLE_DISPATCH_STORAGE=0 leaves such programs to the normal path as before (default 1).
bool TableDispatchStorageEnabled() {
	static const bool enabled = NativeXprSwitch("KYTY_TABLE_DISPATCH_STORAGE");
	return enabled;
}

// What table dispatches serve: a table program whose shader data fits the push constants and whose set holds only
// images, samplers and the page table (TableResolveSet). Null, or why not.
const char* TableDispatchRefusal(const PipelineCache::TableProgram* table, bool thread_dimensions) {
	if (table == nullptr) return "no table mode program";
	const auto& program = table->program;
	using Kind          = ShaderRecompiler::IR::DescriptorBindingKind;
	if (!program.table_mode) return "no table mode program";
	if (!program.bindings.UsesPushData()) return "shader data that does not fit the push constants";
	if (program.info.uses_gds) return "GDS";
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart && !thread_dimensions)
		return "thread counts";
	if (table->input_info.lds_storage || program.bindings.descriptor_counts[static_cast<size_t>(Kind::SharedMemory)] != 0)
		return "LDS storage";
	for (const auto& image: program.info.images) {
		if (image.atomic) return "an image atomic";
		if ((image.written || image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage) &&
		    !TableDispatchStorageEnabled())
			return "a storage image (KYTY_TABLE_DISPATCH_STORAGE=0)";
	}
	return nullptr;
}

} // namespace

void RenderExecutor::TableDispatchLearn(const HW::ComputeShaderInfo& cs, const HW::ShaderRegisters& sh,
                                        const ShaderComputeInputInfo& input_info, const ShaderProgram& compute_program,
                                        bool async_compute, bool special) {
	const bool thread_dimensions = input_info.dispatch_thread_dimensions;
	if (!TableDispatchEnabled() || (thread_dimensions && !TableDispatchThreadsEnabled()) ||
	    input_info.dispatch_indirect_threads || input_info.stage.program == nullptr)
		return;
	auto& table           = OwnTable();
	auto [found, fresh]   = table.dispatches.try_emplace(TableDispatchKey(cs, async_compute, thread_dimensions));
	auto& pair            = found->second;
	if (fresh) {
		pair.vs                = cs.cs_regs.data_addr;
		pair.thread_dimensions = thread_dimensions;
	}
	pair.used_frame = m_context.FrameNumber();
	if (pair.refused || pair.programs[0] != nullptr || pair.pending != nullptr || pair.vs != cs.cs_regs.data_addr)
		return;
	const auto& program = *input_info.stage.program;
	auto&       pipelines              = m_context.GetPipelineCache();
	const auto [plan, specialization] = pipelines.ProgramPlanOf(compute_program);
	// (A uniform fill the normal path makes a clear or a known fill of: ResolveComputeBufferFill,
	// TryConsumeComputeImageClear.)
	const bool fill = input_info.stage.resources != nullptr &&
	                  input_info.stage.resources->uniform_fill.kind != ShaderRecompiler::IR::UniformFillKind::None;
	if (special || fill || plan == nullptr || specialization == nullptr || program.info.uses_gds ||
	    (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart && !thread_dimensions)) {
		pair.refused = true;
		return;
	}
	pair.plans       = {plan, nullptr};
	pair.code_hashes = {ShaderHashAt(pair.vs), 0};
	pair.generation  = ShaderMapGeneration();
	pair.pending     = pipelines.CompileTableComputeAsync(cs, sh, async_compute, *specialization, thread_dimensions);
}

bool RenderExecutor::TableDispatch(CommandBuffer& buffer, const HW::ComputeShaderInfo& cs, uint32_t x, uint32_t y,
                                   uint32_t z, bool async_compute, uint64_t indirect_args, bool thread_dimensions) {
	if (!TableDispatchEnabled() || (indirect_args != 0 && (!TableDispatchIndirectEnabled() || thread_dimensions)) ||
	    (thread_dimensions && !TableDispatchThreadsEnabled()))
		return false;
	auto&      table = OwnTable();
	const auto frame = m_context.FrameNumber();
	// The dispatches' pairs no dispatch used for PairKeepFrames leave, as the draws' (TableTry).
	if (frame - table.dispatches_swept >= TableXpr::PairSweepFrames) {
		table.dispatches_swept = frame;
		if (table.dispatches.size() > TableXpr::PairsKept) {
			const auto before = table.dispatches.size();
			std::erase_if(table.dispatches, [&](auto& item) {
				auto& held = item.second;
				if (held.used_frame + TableXpr::PairKeepFrames >= frame || held.pending != nullptr) return false;
				for (auto& [words, set]: held.sets) NativeXprRetireSet(set->pool, set->set);
				return true;
			});
			if (table.dispatches.size() != before) {
				table.gathered       = nullptr;
				table.last_set       = nullptr;
				table.found_dispatch = nullptr;
			}
		}
	}
	// (The pair found last again without its lookup: runs of dispatches keep their program.)
	const auto key   = TableDispatchKey(cs, async_compute, thread_dimensions);
	auto*      known = table.found_dispatch != nullptr && table.found_dispatch_key == key ? table.found_dispatch : nullptr;
	if (known == nullptr) {
		const auto found = table.dispatches.find(key);
		if (found == table.dispatches.end()) return false;
		known                    = &found->second;
		table.found_dispatch_key = key;
		table.found_dispatch     = known;
	}
	auto& pair      = *known;
	pair.used_frame = frame;
	if (pair.refused || pair.vs != cs.cs_regs.data_addr) return false;
	if (const auto generation = ShaderMapGeneration(); generation != pair.generation) {
		if (pair.programs[0] != nullptr && ShaderHashAt(pair.vs) != pair.code_hashes[0]) {
			// Learnt again from the normal path's next dispatch.
			TableReset(pair);
			pair.generation = generation;
			return false;
		}
		pair.generation = generation;
	}
	NativeXprCollect(); // (the retired sets)
	auto& pipelines = m_context.GetPipelineCache();
	if (pair.programs[0] == nullptr) {
		// The table program, once compiled.
		if (pair.pending == nullptr || !pair.pending->done.load(std::memory_order_acquire)) return false;
		const auto* program = pair.pending->programs[0];
		if (const char* refusal = TableDispatchRefusal(program, pair.thread_dimensions); refusal != nullptr) {
			static std::atomic<uint32_t> reports {0};
			if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
				std::printf("Table: CS 0x%llx refused: %s\n", static_cast<unsigned long long>(pair.vs),
				            program == nullptr && !pair.pending->refusal.empty() ? pair.pending->refusal.c_str() : refusal);
			pair.refused = true;
			pair.pending.reset();
			return false;
		}
		pair.pending.reset();
		pair.tables       = {program, nullptr};
		pair.programs     = {&program->program, nullptr};
		pair.compute_info = program->input_info;
		pair.compute_info.stage = {.program = &program->program, .resources = &g_table_no_resources};
		TableRuns(pair);
	}
	if (pair.pipeline == nullptr) pair.pipeline = &pipelines.GetNativeComputePipeline(pair.compute_info, pair.tables[0]->handle);
	const std::array<std::span<const uint32_t>, 2> user_data {
	    std::span<const uint32_t>(cs.cs_user_sgpr.value,
	                              std::min<uint32_t>(cs.cs_regs.user_sgpr, HW::UserSgprInfo::SGPRS_MAX)),
	    std::span<const uint32_t> {}};
	if (!TableEvaluateDraw(pair, user_data)) return false;
	TableGather(pair, user_data);
	// A buffer it writes over surface metadata: the normal path's UpdateComputeMetaClears records the clear.
	auto& textures = m_context.GetTextureCache();
	for (size_t i = 0; i < pair.programs[0]->info.buffers.size(); ++i) {
		const auto& resource = pair.programs[0]->info.buffers[i];
		if (!resource.written || i >= table.entries[0].size()) continue;
		ShaderBufferResource descriptor;
		std::copy_n(table.entries[0][i].first.begin(), 4, descriptor.fields);
		if (textures.IsMeta(descriptor.Base48())) return false;
	}
	// The program for the V#s' buffer specialization (the pair's, or a buffer variant's: KYTY_TABLE_BUFFER_VARIANTS).
	const auto selected = TableSelectBuffers(buffer, pair, true, async_compute);
	if (selected == TableNoBuffers) return false;
	if (selected != 0) {
		auto& variant = pair.buffer_variants[selected - 1];
		if (variant.pipeline == nullptr)
			variant.pipeline = &pipelines.GetNativeComputePipeline(variant.compute_info, variant.tables[0]->handle);
	}
	const auto& selected_pipeline = selected != 0 ? *pair.buffer_variants[selected - 1].pipeline : *pair.pipeline;
	auto*       set               = TableSet(buffer, pair, selected_pipeline, nullptr);
	if (set == nullptr) return false;
	std::array<vk::DeviceAddress, 2> blocks {};
	bool                             writes       = false;
	auto&                            buffer_cache = m_context.GetBufferCache();
	TablePrepareBda(pair, user_data); // (as TableDraw's: before the blocks)
	const auto registration = buffer_cache.BufferSetEpoch();
	if (!TableBlocks(pair, user_data, blocks, writes, selected)) return false;
	// An indirect dispatch's arguments, read by the GPU (as DispatchIndirect's: their owner made current).
	std::pair<Buffer*, uint64_t> arguments {};
	if (indirect_args != 0) {
		arguments = buffer_cache.ObtainBuffer(indirect_args, sizeof(vk::DispatchIndirectCommand), false);
		if (arguments.first == nullptr || (arguments.second & 3u) != 0) return false;
	}
	if (buffer_cache.BufferSetEpoch() != registration || buffer.IsInvalid()) return false;
	// Emission, as DispatchDirect's.
	buffer.EndRendering();
	auto vk_buffer = buffer.Handle();
	// (As a draw's: the set's images as its last dispatch's transit left them, while no image changed since.)
	if (set->transit_epoch != textures.ImagesEpoch() || set->transit_state != nullptr) {
		const TransitGroup transit_group;
		if (!TableTransit(*set, vk_buffer, nullptr)) return false;
		set->transit_epoch = textures.ImagesEpoch();
		set->transit_state = nullptr;
	}
	const auto& pipeline = selected_pipeline;
	const auto& program  = *pair.programs[0];
	const auto& bindings = program.bindings;
	vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.pipeline_layout, 0, 1, &set->set, 0, nullptr);
	ShaderRecompiler::IR::PushData push_data;
	const auto                     start     = bindings.push_data_start_dword;
	const auto&                    registers = program.info.user_data_registers;
	for (size_t i = 0; i < registers.size(); ++i) {
		const auto index            = registers[i] - program.user_data_base;
		push_data.dwords[start + i] = index < user_data[0].size() ? user_data[0][index] : 0u;
	}
	push_data.dwords[start + bindings.TableBlockDword()]      = static_cast<uint32_t>(blocks[0]);
	push_data.dwords[start + bindings.TableBlockDword() + 1u] = static_cast<uint32_t>(blocks[0] >> 32u);
	// A dispatch of thread counts: the counts where its program reads them, its groups the counts' cover (as
	// DispatchDirect's).
	const auto& r      = cs.cs_regs;
	const auto  groups = [](uint32_t threads, uint32_t size) { return (threads + std::max(size, 1u) - 1u) / std::max(size, 1u); };
	if (thread_dimensions) {
		if (bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
			push_data.dwords[start + bindings.dispatch_thread_dword]      = x;
			push_data.dwords[start + bindings.dispatch_thread_dword + 1u] = y;
			push_data.dwords[start + bindings.dispatch_thread_dword + 2u] = z;
		}
		x = groups(x, r.num_thread_x);
		y = groups(y, r.num_thread_y);
		z = groups(z, r.num_thread_z);
	}
	vk_buffer.pushConstants(pipeline.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                        (start + bindings.ShaderDataDwords()) * sizeof(uint32_t), push_data.dwords.data());
	// As DispatchDirect(): storage image writes order as buffer writes; a dispatch the guest did not order after the
	// previous one of its queue overlaps it.
	writes = writes || std::ranges::any_of(program.info.images, [](const ShaderRecompiler::IR::ImageResource& image) {
		         return image.written && image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
	         });
	const bool chained = std::exchange(m_dispatch_chained, false) && m_deferred_barrier;
	if (writes && !chained) ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	if (indirect_args != 0 && !chained) {
		// (As DispatchIndirect's: the arguments earlier work wrote, read as indirect commands.)
		vk::MemoryBarrier barrier {};
		barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics | vk::PipelineStageFlagBits::eComputeShader |
		                              vk::PipelineStageFlagBits::eTransfer,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier, 0, nullptr, 0, nullptr);
	}
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	if (indirect_args != 0) vk_buffer.dispatchIndirect(arguments.first->Handle(), arguments.second);
	else vk_buffer.dispatch(x, y, z);
	m_deferred_barrier       = true;
	m_deferred_barrier_queue = m_guest_queue;
	return true;
}
