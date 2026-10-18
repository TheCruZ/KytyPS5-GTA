// Table draws (chenxiao07's KytyPS5 fork, src/local/table-xpr.inc TableDraw; KYTY_TABLE_XPR=1).
//
// The draw path of table dispatches (tableDispatch.inc) for a vertex and a pixel program: both translated in table
// mode, they read their SRT slots and buffers from a block per stage, which the draw writes from the slots its plans
// mark, evaluated from guest memory, and from its buffers' V#s. The draw skips the resolution of its programs (the
// SRT walk) and the buffer descriptors and flattened SRT uploads; its images, samplers, targets, vertex and index
// buffers go as the normal path takes them.
//
// A pair (the stages' code and static input) is learnt after a normal draw of it (TableDrawLearn): both programs are
// compiled in table mode for that draw's specializations. A draw whose descriptors select others compiles a variant
// (a few per frame) or takes the normal path. Native XPR records leave the pairs table draws serve to them.
//
// Included by renderDraw.cpp.

namespace {

bool TableXprEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_TABLE_XPR");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool TableXprReadWord(uint64_t address, uint32_t& word) {
	if ((address & 3u) != 0 || !GuestRange {address, 4}.Valid()) return false;
	if (Libs::LibKernel::Memory::MayFaultOnGpuRead(address, 4))
		return Libs::LibKernel::Memory::TryReadAroundGpuWrites(address, &word, 4);
	return Libs::LibKernel::Memory::TryReadBacking(address, &word, 4);
}

} // namespace

struct RenderExecutor::TableDrawCache {
	struct Variant {
		const PipelineCache::TableProgram* vertex = nullptr;
		const PipelineCache::TableProgram* pixel  = nullptr;
	};
	struct Pair {
		const ShaderRecompiler::IR::ResourcePlan* vertex_plan = nullptr;
		const ShaderRecompiler::IR::ResourcePlan* pixel_plan  = nullptr;
		std::vector<Variant>                      variants;
		std::vector<std::pair<ShaderRecompiler::IR::ResourceSpecialization,
		                      ShaderRecompiler::IR::ResourceSpecialization>>
		     refused_specializations;
		bool refused = false;
	};
	std::unordered_map<uint64_t, Pair> pairs;
	// Shader address pairs (NativeXprPairOfShaders) some table pair serves: native XPR records leave them.
	std::unordered_set<uint64_t> served;
	static constexpr uint32_t    CompileBudget = 4;
	uint64_t                     budget_frame  = UINT64_MAX;
	uint32_t                     compiles      = 0;
	// The draw's resources per stage (its bindings refer to them until the next operation) and scratch.
	std::array<ShaderRecompiler::IR::ResourceSnapshot, 2>       snapshots;
	std::array<ShaderRecompiler::IR::ResourceSpecialization, 2> specializations;
	std::vector<uint32_t>                                       key;
	std::vector<uint32_t>                                       block;

	bool TakeBudget(uint64_t frame) {
		if (frame != budget_frame) {
			budget_frame = frame;
			compiles     = 0;
		}
		return compiles++ < CompileBudget;
	}
};

RenderExecutor::TableDrawCache& RenderExecutor::TableDrawState() {
	if (!m_table_draw) {
		m_table_draw = {new TableDrawCache {}, [](TableDrawCache* cache) { delete cache; }};
	}
	return *m_table_draw;
}

bool RenderExecutor::TableDrawServes(const HW::Shader& shaders) {
	return TableXprEnabled() && m_table_draw && m_table_draw->served.contains(NativeXprPairOfShaders(shaders));
}

// The pair's key: both stages' code and static input, and whether the pixel stage runs.
static uint64_t TableDrawKey(std::vector<uint32_t>& key, const ShaderParams& vertex_params,
                             const ShaderVertexInputInfo& vertex_info, const ShaderParams* pixel_params,
                             const ShaderPixelInputInfo& pixel_info) {
	key.clear();
	BuildStageStaticKey(vertex_info, key);
	const auto vertex_words = key.size();
	if (pixel_params != nullptr) BuildStageStaticKey(pixel_info, key);
	const std::array<uint64_t, 6> head {vertex_params.hash, vertex_params.user_data_count,
	                                    pixel_params != nullptr ? pixel_params->hash : 0,
	                                    pixel_params != nullptr ? pixel_params->user_data_count : 0, vertex_words,
	                                    vertex_params.Base() ^ (pixel_params != nullptr ? pixel_params->Base() << 1u : 0)};
	return XXH3_64bits_withSeed(key.data(), key.size() * sizeof(uint32_t),
	                            XXH3_64bits(head.data(), sizeof(head)));
}

static std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
TableDrawExportMapping(const HW::Context& ctx, uint32_t color_output_mask) {
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> mapping {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type, rt.info.channel_order).export_mapping;
		}
	}
	return mapping;
}

void RenderExecutor::TableDrawLearn(CommandBuffer& buffer, const DrawRenderState& state) {
	if (!TableXprEnabled() || state.programs.VertexStageCount() != 1 || !state.programs.vertex[0] ||
	    state.vertex_info[0].logical_stage == ShaderType::Mesh || state.vertex_info[0].stage.program == nullptr ||
	    state.vertex_info[0].stage.program->table_mode)
		return;
	auto&      cache = TableDrawState();
	const auto& ctx  = buffer.GetRegisters();
	const auto& sh   = buffer.GetShaders();
	ShaderVertexInputInfo vertex_info {};
	ShaderPixelInputInfo  pixel_info {};
	ShaderParams          vertex_params, pixel_params;
	auto&                 pipelines = m_context.GetPipelineCache();
	if (!pipelines.PrepareGraphicsStages(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx, buffer.GetUserConfig(),
	                                     TableDrawExportMapping(ctx, DrawColorOutputMask(ctx)), state.ps_active,
	                                     vertex_info, pixel_info, vertex_params, pixel_params))
		return;
	auto& pair = cache.pairs[TableDrawKey(cache.key, vertex_params, vertex_info, state.ps_active ? &pixel_params : nullptr,
	                                      pixel_info)];
	if (pair.refused || !pair.variants.empty() || !cache.TakeBudget(m_context.FrameNumber())) return;
	const auto [vertex_plan, vertex_specialization] = pipelines.ProgramPlanOf(state.programs.vertex[0]);
	const auto [pixel_plan, pixel_specialization] =
	    state.ps_active ? pipelines.ProgramPlanOf(state.programs.pixel)
	                    : std::pair<const ShaderRecompiler::IR::ResourcePlan*,
	                                const ShaderRecompiler::IR::ResourceSpecialization*> {nullptr, nullptr};
	if (vertex_plan == nullptr || (state.ps_active && pixel_plan == nullptr)) {
		pair.refused = true;
		return;
	}
	TableDrawCache::Variant variant;
	std::string             refusal;
	uint32_t                cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(false, false);
	if (state.ps_active) {
		variant.pixel = pipelines.CompileTableGraphics(pixel_params, nullptr, &pixel_info, *pixel_specialization, cursor,
		                                               refusal);
		if (variant.pixel == nullptr || !variant.pixel->program.bindings.UsesPushData()) {
			pair.refused = true;
			return;
		}
		variant.pixel->program.bindings.AdvancePushData(cursor);
	}
	variant.vertex = pipelines.CompileTableGraphics(vertex_params, &vertex_info, nullptr, *vertex_specialization, cursor,
	                                                refusal);
	if (variant.vertex == nullptr || !variant.vertex->program.bindings.UsesPushData()) {
		pair.refused = true;
		return;
	}
	pair.vertex_plan = vertex_plan;
	pair.pixel_plan  = pixel_plan;
	pair.variants.push_back(variant);
	cache.served.insert(NativeXprPairOfShaders(sh));
}

// Evaluates a table program's slots and descriptors for a use, into `snapshot`.
static bool TableEvaluateStage(const ShaderRecompiler::IR::CompiledShaderInfo& reference,
                               std::span<const uint32_t>                       user_data,
                               ShaderRecompiler::IR::ResourceSnapshot&          snapshot) {
	using Kind        = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	const auto& plan  = reference.table_plan;
	auto&       slots = snapshot.flattened_srt;
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
		if (!TableXprReadWord(address, slots[index])) return false;
	}
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
	return true;
}

bool RenderExecutor::TableDrawPrograms(CommandBuffer& buffer, uint32_t color_output_mask, DrawRenderState& state) {
	m_table_draw_active = false;
	if (!TableXprEnabled() || !m_table_draw) return false;
	auto&       cache = *m_table_draw;
	const auto& ctx   = buffer.GetRegisters();
	const auto& sh    = buffer.GetShaders();
	if (!cache.served.contains(NativeXprPairOfShaders(sh))) return false;
	ShaderVertexInputInfo vertex_info {};
	ShaderPixelInputInfo  pixel_info {};
	ShaderParams          vertex_params, pixel_params;
	auto&                 pipelines = m_context.GetPipelineCache();
	if (!pipelines.PrepareGraphicsStages(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx, buffer.GetUserConfig(),
	                                     TableDrawExportMapping(ctx, color_output_mask), state.ps_active, vertex_info,
	                                     pixel_info, vertex_params, pixel_params))
		return false;
	const auto found = cache.pairs.find(
	    TableDrawKey(cache.key, vertex_params, vertex_info, state.ps_active ? &pixel_params : nullptr, pixel_info));
	if (found == cache.pairs.end() || found->second.refused || found->second.variants.empty()) return false;
	auto&      pair      = found->second;
	const auto reference = pair.variants.front();
	auto&       snapshots = cache.snapshots;
	auto&       specs     = cache.specializations;
	if (!TableEvaluateStage(reference.vertex->program,
	                        std::span(vertex_params.user_data).first(vertex_params.user_data_count), snapshots[0]) ||
	    !pipelines.SpecializeDirectResources(*pair.vertex_plan, snapshots[0], specs[0]))
		return false;
	if (state.ps_active &&
	    (!TableEvaluateStage(reference.pixel->program, std::span(pixel_params.user_data).first(pixel_params.user_data_count),
	                         snapshots[1]) ||
	     !pipelines.SpecializeDirectResources(*pair.pixel_plan, snapshots[1], specs[1])))
		return false;
	const TableDrawCache::Variant* variant = nullptr;
	for (const auto& candidate: pair.variants) {
		if (candidate.vertex->specialization == specs[0] &&
		    (!state.ps_active || candidate.pixel->specialization == specs[1])) {
			variant = &candidate;
			break;
		}
	}
	if (variant == nullptr) {
		const auto refused = std::pair {specs[0], state.ps_active ? specs[1] : ShaderRecompiler::IR::ResourceSpecialization {}};
		if (std::ranges::find(pair.refused_specializations, refused) != pair.refused_specializations.end() ||
		    !cache.TakeBudget(m_context.FrameNumber()))
			return false;
		TableDrawCache::Variant compiled;
		std::string             refusal;
		uint32_t                cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(false, false);
		if (state.ps_active) {
			compiled.pixel = pipelines.CompileTableGraphics(pixel_params, nullptr, &pixel_info, specs[1], cursor, refusal);
			if (compiled.pixel != nullptr) compiled.pixel->program.bindings.AdvancePushData(cursor);
		}
		compiled.vertex = pipelines.CompileTableGraphics(vertex_params, &vertex_info, nullptr, specs[0], cursor, refusal);
		if (compiled.vertex == nullptr || (state.ps_active && compiled.pixel == nullptr) ||
		    !compiled.vertex->program.bindings.UsesPushData() ||
		    !(compiled.vertex->program.table_plan == reference.vertex->program.table_plan) ||
		    (state.ps_active && (!compiled.pixel->program.bindings.UsesPushData() ||
		                         !(compiled.pixel->program.table_plan == reference.pixel->program.table_plan)))) {
			pair.refused_specializations.push_back(refused);
			return false;
		}
		pair.variants.push_back(compiled);
		variant = &pair.variants.back();
	}
	state.programs           = {};
	state.programs.vertex[0] = variant->vertex->handle;
	state.vertex_info        = {};
	state.vertex_info[0]       = vertex_info;
	state.vertex_info[0].stage = {.program = &variant->vertex->program, .resources = &snapshots[0]};
	state.ps_input_info        = {};
	if (state.ps_active) {
		state.programs.pixel      = variant->pixel->handle;
		state.ps_input_info       = pixel_info;
		state.ps_input_info.stage = {.program = &variant->pixel->program, .resources = &snapshots[1]};
	}
	m_table_draw_active = true;
	return true;
}

// After the stages' bindings (PrepareGraphicsBindings()): each table stage's block, its buffers resolved to device
// addresses, and its address in the stage's shader data. False when a buffer cannot be given (the draw is skipped).
bool RenderExecutor::TableDrawBlocks(std::span<PreparedBindings* const> stages) {
	auto& cache         = TableDrawState();
	auto& cache_buffers = m_context.GetBufferCache();
	for (int attempt = 0;; ++attempt) {
		if (attempt == 3) return false;
		const auto epoch = cache_buffers.BufferSetEpoch();
		for (auto* stage: stages) {
			const auto& program = *stage->runtime->program;
			if (!program.table_mode) continue;
			const auto& plan     = program.table_plan;
			const auto& snapshot = *stage->runtime->resources;
			auto&       block    = cache.block;
			block.assign(plan.BlockDwords(), 0);
			std::copy(snapshot.flattened_srt.begin(), snapshot.flattened_srt.end(), block.begin());
			for (size_t i = 0; i < plan.buffers.size(); ++i) {
				const auto  v_sharp  = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
				const auto& resource = program.info.buffers[i];
				if (!resource.live || v_sharp.Type() != 0) continue;
				const auto address = v_sharp.Base48();
				auto       size    = v_sharp.GetSize();
				if (address == 0 || size == 0) continue;
				if ((address & 3u) != 0 || v_sharp.NumRecords() == UINT32_MAX) return false;
				size = Libs::LibKernel::Memory::ClampRangeSize(address, size);
				if (size == 0) continue;
				auto [owner, owner_offset] = cache_buffers.ObtainBuffer(
				    address, size, resource.written,
				    resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled()));
				if (resource.written) m_context.GetTextureCache().InvalidateMemoryFromGPU(address, size);
				const auto device = owner->BufferDeviceAddress() + owner_offset;
				const auto first  = plan.BufferDword(i);
				block[first]      = static_cast<uint32_t>(device);
				block[first + 1]  = static_cast<uint32_t>(device >> 32u);
				block[first + 2]  = static_cast<uint32_t>(std::min<uint64_t>(size / 4u, UINT32_MAX));
				block[first + 4]  = v_sharp.Stride();
			}
			const auto block_address = cache_buffers.TableUpload(block.data(), block.size() * sizeof(uint32_t));
			stage->shader_data[program.bindings.TableBlockDword()]      = static_cast<uint32_t>(block_address);
			stage->shader_data[program.bindings.TableBlockDword() + 1u] = static_cast<uint32_t>(block_address >> 32u);
		}
		if (cache_buffers.BufferSetEpoch() == epoch) return true;
	}
}
