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
// The resolve thread evaluates the blocks and selects the programs ahead of the draw (TableDrawAhead) instead of the
// normal resolution, reading guest memory through the resolution's log; the execution thread then only gives the
// buffers their device addresses.
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
	// The execution thread changes pairs and served under the lock; the resolve thread reads them under it.
	std::unordered_map<uint64_t, Pair> pairs;
	// Shader address pairs (NativeXprPairOfShaders) some table pair serves: native XPR records leave them.
	std::unordered_set<uint64_t> served;
	std::atomic_flag             lock;
	static constexpr uint32_t    CompileBudget = 4;
	uint64_t                     budget_frame  = UINT64_MAX;
	uint32_t                     compiles      = 0;
	// The draw's resources per stage (its bindings refer to them until the next operation) and scratch.
	std::array<ShaderRecompiler::IR::ResourceSnapshot, 2>       snapshots;
	std::array<ShaderRecompiler::IR::ResourceSpecialization, 2> specializations;
	std::vector<uint32_t>                                       key;
	std::vector<uint32_t>                                       ahead_key; // the resolve thread's
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
	std::call_once(m_table_draw_once, [this] {
		m_table_draw = {new TableDrawCache {}, [](TableDrawCache* cache) { delete cache; }};
	});
	return *m_table_draw;
}

bool RenderExecutor::TableDrawServes(const HW::Shader& shaders) {
	return TableXprEnabled() && TableDrawState().served.contains(NativeXprPairOfShaders(shaders));
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
	const auto key =
	    TableDrawKey(cache.key, vertex_params, vertex_info, state.ps_active ? &pixel_params : nullptr, pixel_info);
	auto found = cache.pairs.find(key);
	if (found != cache.pairs.end() && (found->second.refused || !found->second.variants.empty())) return;
	if (found == cache.pairs.end()) {
		const TableLockGuard lock(cache.lock);
		found = cache.pairs.emplace(key, TableDrawCache::Pair {}).first;
	}
	auto&      pair   = found->second;
	const auto refuse = [&] {
		const TableLockGuard lock(cache.lock);
		pair.refused = true;
	};
	if (!cache.TakeBudget(m_context.FrameNumber())) return;
	const auto [vertex_plan, vertex_specialization] = pipelines.ProgramPlanOf(state.programs.vertex[0]);
	const auto [pixel_plan, pixel_specialization] =
	    state.ps_active ? pipelines.ProgramPlanOf(state.programs.pixel)
	                    : std::pair<const ShaderRecompiler::IR::ResourcePlan*,
	                                const ShaderRecompiler::IR::ResourceSpecialization*> {nullptr, nullptr};
	if (vertex_plan == nullptr || (state.ps_active && pixel_plan == nullptr)) {
		refuse();
		return;
	}
	TableDrawCache::Variant variant;
	std::string             refusal;
	uint32_t                cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(false, false);
	if (state.ps_active) {
		variant.pixel = pipelines.CompileTableGraphics(pixel_params, nullptr, &pixel_info, *pixel_specialization, cursor,
		                                               refusal);
		if (variant.pixel == nullptr || !variant.pixel->program.bindings.UsesPushData()) {
			refuse();
			return;
		}
		variant.pixel->program.bindings.AdvancePushData(cursor);
	}
	variant.vertex = pipelines.CompileTableGraphics(vertex_params, &vertex_info, nullptr, *vertex_specialization, cursor,
	                                                refusal);
	if (variant.vertex == nullptr || !variant.vertex->program.bindings.UsesPushData()) {
		refuse();
		return;
	}
	const TableLockGuard lock(cache.lock);
	pair.vertex_plan = vertex_plan;
	pair.pixel_plan  = pixel_plan;
	pair.variants.push_back(variant);
	cache.served.insert(NativeXprPairOfShaders(sh));
}

// The draw's programs from its table pair, and their resources into `snapshots`; false when the normal path takes it.
// Ahead of execution (`ahead`) guest memory is read through the resolution's log and no program is compiled.
bool RenderExecutor::TableDrawSelect(const HW::Context& ctx, const HW::UserConfig& user_config, const HW::Shader& sh,
                                     uint32_t color_output_mask, DrawRenderState& state,
                                     ShaderRecompiler::IR::ResourceSnapshot*       snapshots,
                                     ShaderRecompiler::IR::ResourceSpecialization* specs, ProgramResolution* ahead) {
	auto&                         cache = TableDrawState();
	std::optional<TableLockGuard> lock;
	if (ahead != nullptr) lock.emplace(cache.lock);
	if (!cache.served.contains(NativeXprPairOfShaders(sh))) return false;
	lock.reset();
	ShaderVertexInputInfo vertex_info {};
	ShaderPixelInputInfo  pixel_info {};
	ShaderParams          vertex_params, pixel_params;
	auto&                 pipelines = m_context.GetPipelineCache();
	if (!pipelines.PrepareGraphicsStages(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx, user_config,
	                                     TableDrawExportMapping(ctx, color_output_mask), state.ps_active, vertex_info,
	                                     pixel_info, vertex_params, pixel_params, ahead))
		return false;
	const auto key = TableDrawKey(ahead != nullptr ? cache.ahead_key : cache.key, vertex_params, vertex_info,
	                              state.ps_active ? &pixel_params : nullptr, pixel_info);
	const ShaderRecompiler::IR::ResourcePlan* plans[2] {};
	TableDrawCache::Variant                   reference;
	if (ahead != nullptr) lock.emplace(cache.lock);
	{
		const auto found = cache.pairs.find(key);
		if (found == cache.pairs.end() || found->second.refused || found->second.variants.empty()) return false;
		plans[0]  = found->second.vertex_plan;
		plans[1]  = found->second.pixel_plan;
		reference = found->second.variants.front();
	}
	lock.reset();
	GuestReadLog* reads = ahead != nullptr ? &ahead->reads : nullptr;
	if (!TableEvaluate(reference.vertex->program,
	                   std::span(vertex_params.user_data).first(vertex_params.user_data_count), snapshots[0], reads) ||
	    !pipelines.SpecializeDirectResources(*plans[0], snapshots[0], specs[0]))
		return false;
	if (state.ps_active &&
	    (!TableEvaluate(reference.pixel->program, std::span(pixel_params.user_data).first(pixel_params.user_data_count),
	                    snapshots[1], reads) ||
	     !pipelines.SpecializeDirectResources(*plans[1], snapshots[1], specs[1])))
		return false;
	const auto matches = [&](const TableDrawCache::Variant& candidate) {
		return candidate.vertex->specialization == specs[0] &&
		       (!state.ps_active || candidate.pixel->specialization == specs[1]);
	};
	TableDrawCache::Variant variant;
	if (ahead != nullptr) lock.emplace(cache.lock);
	auto& pair = cache.pairs.find(key)->second;
	if (const auto it = std::ranges::find_if(pair.variants, matches); it != pair.variants.end()) variant = *it;
	lock.reset();
	if (variant.vertex == nullptr) {
		const auto refused =
		    std::pair {specs[0], state.ps_active ? specs[1] : ShaderRecompiler::IR::ResourceSpecialization {}};
		if (ahead != nullptr ||
		    std::ranges::find(pair.refused_specializations, refused) != pair.refused_specializations.end() ||
		    !cache.TakeBudget(m_context.FrameNumber()))
			return false;
		std::string refusal;
		uint32_t    cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(false, false);
		if (state.ps_active) {
			variant.pixel = pipelines.CompileTableGraphics(pixel_params, nullptr, &pixel_info, specs[1], cursor, refusal);
			if (variant.pixel != nullptr) variant.pixel->program.bindings.AdvancePushData(cursor);
		}
		variant.vertex = pipelines.CompileTableGraphics(vertex_params, &vertex_info, nullptr, specs[0], cursor, refusal);
		const TableLockGuard guard(cache.lock);
		if (variant.vertex == nullptr || (state.ps_active && variant.pixel == nullptr) ||
		    !variant.vertex->program.bindings.UsesPushData() ||
		    !(variant.vertex->program.table_plan == reference.vertex->program.table_plan) ||
		    (state.ps_active && (!variant.pixel->program.bindings.UsesPushData() ||
		                         !(variant.pixel->program.table_plan == reference.pixel->program.table_plan)))) {
			pair.refused_specializations.push_back(refused);
			return false;
		}
		pair.variants.push_back(variant);
	}
	state.programs             = {};
	state.programs.vertex[0]   = variant.vertex->handle;
	state.vertex_info          = {};
	state.vertex_info[0]       = vertex_info;
	state.vertex_info[0].stage = {.program = &variant.vertex->program, .resources = &snapshots[0]};
	state.ps_input_info        = {};
	if (state.ps_active) {
		state.programs.pixel      = variant.pixel->handle;
		state.ps_input_info       = pixel_info;
		state.ps_input_info.stage = {.program = &variant.pixel->program, .resources = &snapshots[1]};
	}
	state.table = true;
	return true;
}

bool RenderExecutor::TableDrawAhead(const HW::Context& context, const HW::UserConfig& user_config,
                                    const HW::Shader& shaders, uint32_t color_output_mask, DrawRenderState& state,
                                    ProgramResolution& ahead) {
	return TableXprEnabled() && user_config.GetPrimType() != Prospero::PrimitiveType::kPatch &&
	       TableDrawSelect(context, user_config, shaders, color_output_mask, state, ahead.resources.data(),
	                       ahead.specializations.data(), &ahead);
}

bool RenderExecutor::TableDrawPrograms(CommandBuffer& buffer, uint32_t color_output_mask, DrawRenderState& state) {
	m_table_draw_active = false;
	if (!TableXprEnabled()) return false;
	auto& cache         = TableDrawState();
	m_table_draw_active = TableDrawSelect(buffer.GetRegisters(), buffer.GetUserConfig(), buffer.GetShaders(),
	                                      color_output_mask, state, cache.snapshots.data(),
	                                      cache.specializations.data(), nullptr);
	return m_table_draw_active;
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
