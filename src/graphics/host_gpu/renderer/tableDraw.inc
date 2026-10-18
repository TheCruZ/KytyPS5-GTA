// --- Table draws -----------------------------------------------------------------------------
//
// Ported from chenxiao07's KytyPS5 fork (src/local/table-xpr.inc, docs/GPU-DRIVEN-XPR.md). Included by
// renderDraw.cpp after nativeXpr.inc, whose draw states and emission the table draws share.
//
// KYTY_TABLE_XPR=1: an indexed draw of an object (a single indexed indirect draw whose arguments the host GPU reads,
// or a direct one) whose shader pair has table mode programs (ShaderRecompiler::IR::EnterTableMode) is drawn like a
// native XPR record, without one. For each draw the renderer evaluates the programs' table plans (IR::TablePlan): the
// slots their shaders read and the T#/S# words, from guest memory as it is when the draw executes (as the normal
// path's SRT evaluation). The T#/S# words pick the pair's image set, a descriptor set with the images and samplers
// as the normal path resolves them, proven current once a frame. Each V# is resolved to its buffer's device address
// (cached while its buffer and range stay the same). The slots and buffer addresses go into a block in the table
// upload ring, whose device address is pushed with the user data: no SRT walk, no flattened SRT, no buffer
// descriptors, no texture lookups. The emission is a native record's: the draw state's targets and dynamic state,
// the pipeline and the set only when they change since the last table draw in the command buffer, and nothing at all
// in place (the same state and set, no image changed since).
//
// The normal path, asked to store (TableStore), compiles the pair's table mode programs on a worker for its draw's
// specializations, then each draw state's pipeline (native bindings), and captures the draw state
// (NativeXprCaptureState). Anything a table draw cannot prove goes back to the normal path.
//
// With native XPR on, table draws take only the pairs native records will not serve (KYTY_TABLE_XPR_COMPLEMENT=0:
// table draws first, records for what they refuse). Off by default (KYTY_TABLE_XPR=0).

namespace {

// KYTY_TABLE_XPR: 0 off (default), 1 table draws, 2 also continued runs.
uint32_t TableXprMode() {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_TABLE_XPR");
		return value != nullptr && (value[0] == '1' || value[0] == '2') ? static_cast<uint32_t>(value[0] - '0') : 0u;
	}();
	return mode;
}

// chenxiao07's KYTY_TABLE_STORE_BUDGET (default 0: no limit): table stores asked for per frame. Each store request
// draws the normal way (and compiles programs or a pipeline); hundreds of pairs warming up at once made every one of
// them a normal draw in the same frame.
uint32_t TableStoreBudget() {
	static const uint32_t budget = NativeXprNumberSwitch("KYTY_TABLE_STORE_BUDGET", 0u);
	return budget;
}

// chenxiao07's KYTY_TABLE_MRU (default 7): table lookups ask their last answer first. Bit 1: TableTry's pair cache
// has 4096 entries (else 256; a frame's draws use ~600 pairs); 2: a pair tries the variant it found last first; 4:
// TableSet asks the pair's set before the last before the words' hash (a pass's draws alternate two materials' sets).
uint32_t TableMru() {
	static const uint32_t mru = NativeXprNumberSwitch("KYTY_TABLE_MRU", 7u);
	return mru;
}

// chenxiao07's KYTY_TABLE_SET_PROOFS (default 1; 0 full checks, 2 verify mode): see TableValidateSet.
uint32_t TableSetProofs() {
	static const uint32_t mode = NativeXprNumberSwitch("KYTY_TABLE_SET_PROOFS", 1u);
	return mode;
}

// chenxiao07's KYTY_TABLE_PACK (default 1; 0 evaluates by the plans' operands): a pair's stages are evaluated from
// their packed flat form (TablePack).
bool TablePackEnabled() {
	static const bool enabled = NativeXprSwitch("KYTY_TABLE_PACK");
	return enabled;
}

// The stages' static input key a pair's programs were compiled for: both stages' code and static input, and whether
// the pixel stage runs.
uint64_t TableDrawKey(std::vector<uint32_t>& key, const ShaderParams& vertex_params,
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
	return XXH3_64bits_withSeed(key.data(), key.size() * sizeof(uint32_t), XXH3_64bits(head.data(), sizeof(head)));
}

std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
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

// A run of dwords of guest memory as the draw would read it now; false where the normal path must read it (nothing
// mapped there, or bytes the GPU wrote that only a readback gives).
bool TableReadRun(uint64_t address, uint32_t count, uint32_t* out) {
	const uint64_t bytes = uint64_t {count} * 4u;
	if ((address & 3u) != 0 || !GuestRange {address, bytes}.Valid()) return false;
	if (Libs::LibKernel::Memory::MayFaultOnGpuRead(address, bytes))
		return Libs::LibKernel::Memory::TryReadAroundGpuWrites(address, out, bytes);
	return Libs::LibKernel::Memory::TryReadBacking(address, out, bytes);
}

// An operand of a stage's plan: an immediate, a slot of its block, a user data register, or a word of the stage's
// code address (a V# of constants the code embeds).
uint32_t TableOperandValue(const ShaderRecompiler::IR::TablePlan::Operand& operand,
                           const ShaderRecompiler::IR::CompiledShaderInfo& program, const uint32_t* slots,
                           std::span<const uint32_t> user_data, uint64_t code) {
	using Kind = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	switch (operand.kind) {
		case Kind::Immediate: return operand.value;
		case Kind::Slot: return slots[operand.value];
		case Kind::ShaderBaseLow: return static_cast<uint32_t>(code + operand.value);
		case Kind::ShaderBaseHigh: return static_cast<uint32_t>((code + operand.value) >> 32u);
		case Kind::UserData: {
			const auto index = operand.value - program.user_data_base;
			return index < user_data.size() ? user_data[index] : 0u;
		}
	}
	return 0;
}

// Whether a V# selects the buffer specialization a table program was compiled for (as SpecializeDirectResources
// derives it): another one takes the normal path, which chooses the program for it.
bool TableBufferSpecialized(const ShaderRecompiler::IR::BufferResource&                 resource,
                            const ShaderRecompiler::IR::ResourceSpecialization::Buffer& specialized,
                            const std::array<uint32_t, 4>&                              words4) {
	ShaderBufferResource descriptor;
	std::copy_n(words4.begin(), 4, descriptor.fields);
	if (descriptor.Type() != 0) descriptor = {};
	auto       packed_stride = descriptor.PackedStride();
	const auto stride        = packed_stride & 0x3fffu;
	const bool swizzle       = stride != 0u && ((packed_stride >> 14u) & 1u) != 0u;
	if (stride == 0u) {
		packed_stride &= ~((1u << 14u) | (3u << 16u));
	} else if (!swizzle) {
		packed_stride &= ~(3u << 16u);
	}
	return packed_stride == specialized.packed_stride &&
	       (resource.formatted ? descriptor.Format() : Prospero::BufferFormat::kInvalid) == specialized.descriptor_format &&
	       (resource.formatted ? descriptor.DstSelXYZW() : DstSel(4, 5, 6, 7)) == specialized.descriptor_swizzle &&
	       (descriptor.OutOfBounds() == 0u && stride == 0u) == specialized.zero_stride_oob &&
	       (resource.scalar && descriptor.GetSize() == 0u) == specialized.scalar_empty;
}

const ShaderRecompiler::IR::ResourceSnapshot g_table_no_resources {};

} // namespace

// A draw state's pipeline.
struct RenderExecutor::TableVariant {
	uint64_t                       state          = 0;
	const PipelineCache::Pipeline* pipeline       = nullptr; // null while a worker creates it
	bool                           fetch_embedded = false;   // the vertex program fetches (a direct draw's offsets)
	bool                           refused        = false;   // other static input than the pair's programs
};

// The descriptor set of one choice of images and samplers (the T#/S# words of both stages).
struct RenderExecutor::TableImageSet {
	struct Image {
		ImageId                 id;
		uint64_t                serial = 0;
		vk::ImageLayout         layout = vk::ImageLayout::eUndefined;
		ImageSubresourceRange   range;
		TextureCache::ImageDesc desc;
		vk::AccessFlags2        access  = vk::AccessFlagBits2::eShaderRead;
		bool                    storage = false; // the shader writes it: GPU-modified at every use (TableTransit)
		bool                    whole   = false; // bound both as a texture and as storage: the whole image's state
		bool                    plain   = false; // a plain color texture (TextureCache::IsSampledImageCurrent)
		// The image's registrations when its use was last proven (KYTY_TABLE_SET_PROOFS): a plain texture's proof holds
		// while it is the same registration and no registration since overlaps its view.
		uint32_t                registration = 0;
	};
	std::vector<uint32_t> words; // without the LOD statistics fields (NativeXprLodStatsMask)
	vk::DescriptorSet     set  = nullptr;
	vk::DescriptorPool    pool = nullptr;
	std::vector<Image>    images;
	// Proven current in that frame under that ImageSetEpoch (made or refused then).
	uint64_t validated_frame = 0;
	uint64_t set_epoch       = 0;
	uint64_t used_frame      = 0;
	// TextureCache::ImagesEpoch after its images were last transited (TableTransit) under transit_state (its capture):
	// they stay in place while it does not move.
	uint64_t                  transit_epoch   = UINT64_MAX;
	const NativeXprDrawState* transit_state   = nullptr;
	uint64_t                  transit_capture = 0;
	// The words select another specialization, or an image no set can hold: the draws take the normal path
	// (TableResolveSet tries again after RetryFrames).
	bool                      refused     = false;
	static constexpr uint64_t RetryFrames = 300;
};

// A stage's flat form as TableRuns makes it (TablePack packs it into the pair): every operand an index into one of
// three arrays (the stage's user data, its block's slots, the stage's constants).
struct RenderExecutor::TableFlatStage {
	struct Operand {
		uint32_t index = 0;
		uint32_t base  = 2; // 0: user data, 1: slots, 2: constants
	};
	struct Run {
		std::array<Operand, 3> address; // low, high, offset
		int32_t                immediate = 0;
		uint32_t               first = 0, count = 0;
		Operand                records; // a bounded run's (TablePlan::Slot::bounded)
		bool                   bounded = false;
	};
	std::vector<Run>      runs;         // the runs in level order
	std::vector<uint32_t> level_ends;   // the run after each level
	std::vector<Operand>  image_words;  // every image's words, then every sampler's
	std::vector<uint32_t> image_counts;
	std::vector<Operand>  buffer_words; // every buffer's four V# words (TableGather)
	std::vector<uint32_t> constants;
	uint32_t              user_words = 0; // the user data it reads: below this
};

struct RenderExecutor::TablePair {
	// Evaluated slots at consecutive addresses (the same base, immediates 4 apart): one read, at the address of the
	// first. Ordered by level: a run whose address is made of slots is read after the runs of those slots.
	struct Run {
		ShaderRecompiler::IR::TablePlan::Slot read;
		uint32_t                              first = 0, count = 0, level = 0;
	};
	uint64_t vs = 0, ps = 0;
	bool     refused   = false;
	bool     has_pixel = false;
	// The table programs (vertex, pixel; null: inactive) and the normal programs' resource plans (an image set's
	// specialization is checked against them).
	std::array<const PipelineCache::TableProgram*, 2>          tables {};
	std::array<const ShaderRecompiler::IR::CompiledShaderInfo*, 2> programs {};
	std::array<const ShaderRecompiler::IR::ResourcePlan*, 2>   plans {};
	uint64_t                                                   static_key = 0; // TableDrawKey
	// The code hashes at vs and ps the programs were compiled from, proven under `generation` (ShaderMapGeneration):
	// code replaced at an address resets the pair.
	std::array<uint64_t, 2>          code_hashes {};
	uint64_t                         generation = UINT64_MAX;
	std::array<std::vector<Run>, 2>  runs;
	// TableEvaluateDraw's form of a stage's plan (TableRuns, TablePack): every operand an index into one of three arrays
	// (the stage's user data, its block's slots, its constants), so an evaluation branches on no operand kind. Both
	// stages' arrays in one block, each operand 16 bits (its array in the top two, its index in the low 14), each run
	// 16 bytes: a pair a draw switches to is cold, and its arrays were a dozen allocations, as many cold lines (in the
	// fork, same process: Stonefang standing render 8.94 -> 8.75 ms, Latria 13.47 -> 13.31).
	struct PackedRun {
		std::array<uint16_t, 3>   address {}; // low, high, offset
		uint16_t                  records   = NotBounded;
		int32_t                   immediate = 0;
		uint16_t                  first = 0, count = 0;
		static constexpr uint16_t NotBounded = 0xffff;
	};
	struct PackedStage {
		std::span<const PackedRun> runs;
		std::span<const uint16_t>  level_ends, image_words, buffer_words;
		std::span<const uint8_t>   image_counts;
		std::span<const uint32_t>  constants;
		uint32_t                   user_words = 0;
		bool                       valid      = false; // (else an index or a count past its packed width)
	};
	std::array<PackedStage, 2>   packed;
	std::unique_ptr<std::byte[]> packed_block;
	std::vector<TableVariant>        variants;
	uint64_t                         store_frame = UINT64_MAX, store_state = 0; // the last store request
	std::unordered_map<uint64_t, std::unique_ptr<TableImageSet>, NativeXprKeyHash> sets; // by T#/S# words
	TableImageSet*                   last_set = nullptr;
	// The set before the last (KYTY_TABLE_MRU bit 4), cleared with last_set.
	TableImageSet*                   previous_set = nullptr;
	uint32_t                         variant_hint = 0; // the variant TableTry found last (KYTY_TABLE_MRU bit 2)
	// The table programs' compilation (on a worker): until it is done the pair's draws take the normal path.
	std::shared_ptr<PipelineCache::TableTicket> pending;
	// A dispatch's (tableDispatch.inc): its program's input and native pipeline.
	ShaderComputeInputInfo         compute_info;
	const PipelineCache::Pipeline* pipeline = nullptr;
	uint64_t                                    used_frame = 0;
};

struct RenderExecutor::TableXpr {
	std::unordered_map<uint64_t, TablePair, NativeXprKeyHash> pairs; // by NativeXprPairHash
	// Table dispatches' pairs, by the compute stage's registers (TableDispatchKey), and the one found last.
	std::unordered_map<uint64_t, TablePair, NativeXprKeyHash> dispatches;
	uint64_t                                                 found_dispatch_key = 0;
	TablePair*                                               found_dispatch     = nullptr;
	uint64_t                                                 dispatches_swept   = 0;
	// The pairs found last, direct-mapped by their hash (KYTY_TABLE_MRU bit 1: else its first 256 only), and the one
	// TableTry found last by its shaders (consecutive draws mostly keep them). Pairs do not move until they leave.
	static constexpr size_t                                 PairCacheEntries = 4096;
	std::array<std::pair<uint64_t, TablePair*>, PairCacheEntries> pair_cache {};
	std::array<uint64_t, 2>                                  found_key {};
	TablePair*                                               found_pair = nullptr;
	// Pairs no draw used for PairKeepFrames leave (every PairSweepFrames, past PairsKept): they are keyed by the
	// shaders' addresses, and a new area loads its shaders elsewhere.
	static constexpr uint64_t PairSweepFrames = 600, PairKeepFrames = 3600;
	static constexpr size_t   PairsKept       = 2048;
	uint64_t                  pairs_swept     = 0;
	uint64_t                  collected       = 0; // the frame unused image sets were last dropped
	// The next prepared draw of the pair under this state stores (TableStore).
	TablePair* store       = nullptr;
	uint64_t   store_state = 0;
	// The frame's store requests so far (KYTY_TABLE_STORE_BUDGET).
	uint64_t store_frame    = UINT64_MAX;
	uint32_t store_requests = 0;
	// The draw's T#/S# words (the set's key) and its runs' addresses.
	std::vector<uint32_t> words;
	std::vector<uint32_t> static_key; // (scratch)
	// The last table draw: its pair, its variant's index, its draw state and set, and TextureCache::ImagesEpoch after
	// it.
	TablePair*                last_pair    = nullptr;
	size_t                    last_variant = 0;
	NativeXprDrawState*       last_state   = nullptr;
	const TableImageSet*      last_set     = nullptr;
	uint64_t                  images_epoch = 0;
	// The game buffer the last arguments came from, and the indices' by their address (each mesh has its own): used
	// again while no buffer was registered and the guest wrote none of its region (ObtainBuffer then returns the same
	// buffer).
	struct HeldBuffer {
		uint64_t begin = 0, end = 0, cpu_epoch = 0, registration = 0;
		Buffer*  buffer = nullptr;
	};
	HeldBuffer                   args;
	std::array<HeldBuffer, 8192> indices;
	// A read-only V#'s buffer as the shaders read it (direct-mapped by the V#'s words, one cache line each): its
	// device address and size in dwords, while no buffer was registered or unregistered (`registration`) and the
	// guest wrote none of its range (`cpu_epoch`, the range's CPU-dirty epoch, else the range found clean again). One
	// whose proof also needs the image starts and the mappings (formatted: an image may start at it; clamped: a
	// mapping may extend it) holds while `special` is the table's special generation (0: no such need).
	struct alignas(64) ResolvedBuffer {
		std::array<uint32_t, 4> words {};
		uint32_t                dwords = 0, special = 0;
		vk::DeviceAddress       device = 0;
		uint64_t                address = 0, size = 0, cpu_epoch = 0, registration = 0;
	};
	static constexpr size_t           ResolvedEntries = 65536;
	std::unique_ptr<ResolvedBuffer[]> resolved = std::make_unique<ResolvedBuffer[]>(ResolvedEntries);
	uint32_t                          special_generation = 1; // moves with the image set and the mappings
	uint64_t                          special_images = UINT64_MAX, special_mappings = UINT64_MAX;
	// Each stage's V#s and their entries (TableGather), for `gathered` (the pair of the draw being made).
	std::array<std::vector<std::pair<std::array<uint32_t, 4>, ResolvedBuffer*>>, 2> entries;
	const TablePair*                                                                gathered = nullptr;
	// The draw's blocks (per stage): its slots (IR::TablePlan::BlockDwords), then its buffers.
	std::array<std::vector<uint32_t>, 2> blocks;
	// Scratch of TableResolveSet and TablePrepareBda.
	std::array<ShaderRecompiler::IR::ResourceSnapshot, 2> snapshots;
	std::array<ShaderStageRuntime, 2>                     runtimes;
	std::array<PreparedBindings, 2>                       prepared;
	ShaderRecompiler::IR::ResourceSpecialization          specialization;
};

RenderExecutor::TableXpr& RenderExecutor::OwnTable() {
	if (!m_table_xpr) m_table_xpr = {new TableXpr {}, [](TableXpr* table) { delete table; }};
	return *m_table_xpr;
}

bool RenderExecutor::TableXprComplements() {
	static const bool complements = [] {
		const char* value = std::getenv("KYTY_TABLE_XPR_COMPLEMENT");
		return value == nullptr || value[0] != '0';
	}();
	return complements;
}

// The pair's programs and what was made for them leave (its shaders changed); its next draws store them again.
void RenderExecutor::TableReset(TablePair& pair) {
	auto& table = OwnTable();
	for (auto& [words, set]: pair.sets) {
		if (table.last_set == set.get()) table.last_set = nullptr;
		NativeXprRetireSet(set->pool, set->set);
	}
	if (table.store == &pair) table.store = nullptr;
	if (table.last_pair == &pair) table.last_pair = nullptr;
	if (table.gathered == &pair) table.gathered = nullptr;
	pair = {.vs = pair.vs, .ps = pair.ps, .generation = pair.generation};
}

// The pair's runs (TablePair::Run): the slots its stages evaluate, in reads of consecutive slots, by level.
void RenderExecutor::TableRuns(TablePair& pair) {
	using Kind = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	std::array<TableFlatStage, 2> flat_forms; // (packed into the pair: TablePack)
	for (uint32_t st = 0; st < 2; ++st) {
		pair.runs[st].clear();
		if (pair.programs[st] == nullptr) continue;
		const auto&           plan = pair.programs[st]->table_plan;
		std::vector<uint32_t> levels(plan.slots.size(), 0);
		const auto            same = [](const ShaderRecompiler::IR::TablePlan::Operand& a,
                             const ShaderRecompiler::IR::TablePlan::Operand& b) {
			return a.kind == b.kind && a.value == b.value;
		};
		for (uint32_t first = 0; first < plan.slots.size();) {
			if (plan.cpu[first] == 0) {
				++first;
				continue;
			}
			const auto& slot  = plan.slots[first];
			uint32_t    count = 1;
			while (first + count < plan.slots.size() && plan.cpu[first + count] != 0 &&
			       same(plan.slots[first + count].low, slot.low) && same(plan.slots[first + count].high, slot.high) &&
			       same(plan.slots[first + count].offset, slot.offset) &&
			       plan.slots[first + count].bounded == slot.bounded && same(plan.slots[first + count].records, slot.records) &&
			       plan.slots[first + count].immediate == slot.immediate + static_cast<int32_t>(4 * count))
				++count;
			uint32_t level = 0;
			for (const auto* operand: {&slot.low, &slot.high, &slot.offset, &slot.records})
				if (operand->kind == Kind::Slot) level = std::max(level, levels[operand->value] + 1);
			std::fill_n(levels.begin() + first, count, level);
			pair.runs[st].push_back({slot, first, count, level});
			first += count;
		}
		std::ranges::stable_sort(pair.runs[st], {}, &TablePair::Run::level);
		// The flat form: a user data register by its index (one of 64 and up is the constant 0, which a stage's user
		// data never reach), a slot by its index, an immediate or a code address word from the constants (the first 0).
		const auto& program = *pair.programs[st];
		auto&       flat    = flat_forms[st];
		flat.constants.push_back(0);
		const auto flat_of = [&](const ShaderRecompiler::IR::TablePlan::Operand& operand) -> TableFlatStage::Operand {
			if (operand.kind == Kind::Slot) return {operand.value, 1};
			if (operand.kind == Kind::UserData) {
				const auto index = operand.value - program.user_data_base;
				if (index >= 64) return {0, 2};
				flat.user_words = std::max(flat.user_words, index + 1);
				return {index, 0};
			}
			flat.constants.push_back(TableOperandValue(operand, program, nullptr, {}, st == 0 ? pair.vs : pair.ps));
			return {static_cast<uint32_t>(flat.constants.size() - 1), 2};
		};
		const auto& runs = pair.runs[st];
		for (size_t i = 0; i < runs.size(); ++i) {
			const auto& run = runs[i];
			flat.runs.push_back({{flat_of(run.read.low), flat_of(run.read.high), flat_of(run.read.offset)},
			                     run.read.immediate, run.first, run.count, flat_of(run.read.records), run.read.bounded});
			if (i + 1 == runs.size() || runs[i + 1].level != run.level) flat.level_ends.push_back(static_cast<uint32_t>(i + 1));
		}
		for (const auto& descriptor: plan.images) {
			flat.image_counts.push_back(descriptor.count);
			for (uint32_t w = 0; w < descriptor.count; ++w) flat.image_words.push_back(flat_of(descriptor.words[w]));
		}
		for (const auto& descriptor: plan.samplers)
			for (uint32_t w = 0; w < descriptor.count; ++w) flat.image_words.push_back(flat_of(descriptor.words[w]));
		for (const auto& words: plan.buffers)
			for (const auto& word: words) flat.buffer_words.push_back(flat_of(word));
	}
	TablePack(pair, flat_forms);
}

// The pair's flat forms packed into one block (TablePair::packed): a stage whose operand indices, counts or levels do
// not fit their packed widths is left invalid (evaluated by its plan's operands).
void RenderExecutor::TablePack(TablePair& pair, const std::array<TableFlatStage, 2>& flat_forms) {
	using Pair        = TablePair;
	pair.packed       = {};
	pair.packed_block = nullptr;
	const auto operand = [](const TableFlatStage::Operand& from, bool& fits) -> uint16_t {
		if (from.base > 2 || from.index >= 0x4000u) fits = false;
		return static_cast<uint16_t>(from.base << 14u | (from.index & 0x3fffu));
	};
	const auto align = [](size_t at, size_t to) { return (at + to - 1) / to * to; };
	// (Layout per stage: runs, constants, level ends, image words, buffer words, image counts.)
	std::array<std::array<size_t, 6>, 2> offsets {};
	size_t                               bytes = 0;
	std::array<bool, 2>                  fits {};
	for (uint32_t st = 0; st < 2; ++st) {
		if (pair.programs[st] == nullptr) continue;
		const auto& flat = flat_forms[st];
		fits[st]         = flat.runs.size() < 0x10000 && flat.image_words.size() < 0x10000;
		for (const auto end: flat.level_ends) fits[st] = fits[st] && end < 0x10000;
		for (const auto count: flat.image_counts) fits[st] = fits[st] && count < 0x100;
		for (const auto& run: flat.runs) fits[st] = fits[st] && run.first < 0x10000 && run.count < 0x10000;
		if (!fits[st]) continue;
		bytes          = align(bytes, 64);
		offsets[st][0] = bytes;
		bytes += flat.runs.size() * sizeof(Pair::PackedRun);
		offsets[st][1] = bytes;
		bytes += flat.constants.size() * sizeof(uint32_t);
		offsets[st][2] = bytes;
		bytes += flat.level_ends.size() * sizeof(uint16_t);
		offsets[st][3] = bytes;
		bytes += flat.image_words.size() * sizeof(uint16_t);
		offsets[st][4] = bytes;
		bytes += flat.buffer_words.size() * sizeof(uint16_t);
		offsets[st][5] = bytes;
		bytes += flat.image_counts.size();
	}
	if (bytes == 0) return;
	pair.packed_block = std::make_unique<std::byte[]>(bytes + 64);
	auto* base        = reinterpret_cast<std::byte*>(align(reinterpret_cast<uintptr_t>(pair.packed_block.get()), 64));
	for (uint32_t st = 0; st < 2; ++st) {
		if (!fits[st]) continue;
		const auto& flat         = flat_forms[st];
		auto&       packed       = pair.packed[st];
		bool        operand_fits = true;
		auto*       runs         = reinterpret_cast<Pair::PackedRun*>(base + offsets[st][0]);
		for (size_t i = 0; i < flat.runs.size(); ++i) {
			const auto& run = flat.runs[i];
			runs[i]         = {.address   = {operand(run.address[0], operand_fits), operand(run.address[1], operand_fits),
			                               operand(run.address[2], operand_fits)},
			                   .records   = run.bounded ? operand(run.records, operand_fits) : Pair::PackedRun::NotBounded,
			                   .immediate = run.immediate,
			                   .first     = static_cast<uint16_t>(run.first),
			                   .count     = static_cast<uint16_t>(run.count)};
			if (run.bounded && runs[i].records == Pair::PackedRun::NotBounded) operand_fits = false;
		}
		auto* constants = reinterpret_cast<uint32_t*>(base + offsets[st][1]);
		std::copy(flat.constants.begin(), flat.constants.end(), constants);
		auto* level_ends = reinterpret_cast<uint16_t*>(base + offsets[st][2]);
		for (size_t i = 0; i < flat.level_ends.size(); ++i) level_ends[i] = static_cast<uint16_t>(flat.level_ends[i]);
		auto* image_words = reinterpret_cast<uint16_t*>(base + offsets[st][3]);
		for (size_t i = 0; i < flat.image_words.size(); ++i) image_words[i] = operand(flat.image_words[i], operand_fits);
		auto* buffer_words = reinterpret_cast<uint16_t*>(base + offsets[st][4]);
		for (size_t i = 0; i < flat.buffer_words.size(); ++i) buffer_words[i] = operand(flat.buffer_words[i], operand_fits);
		auto* image_counts = reinterpret_cast<uint8_t*>(base + offsets[st][5]);
		for (size_t i = 0; i < flat.image_counts.size(); ++i) image_counts[i] = static_cast<uint8_t>(flat.image_counts[i]);
		packed = {.runs         = {runs, flat.runs.size()},
		          .level_ends   = {level_ends, flat.level_ends.size()},
		          .image_words  = {image_words, flat.image_words.size()},
		          .buffer_words = {buffer_words, flat.buffer_words.size()},
		          .image_counts = {image_counts, flat.image_counts.size()},
		          .constants    = {constants, flat.constants.size()},
		          .user_words   = flat.user_words,
		          .valid        = operand_fits};
	}
}

// The slots of the pair's stages, read into their blocks where the plans address them from the user data and lower
// slots, and the T#/S# words they make (TableXpr::words, the set's key). False: something the normal path must read.
bool RenderExecutor::TableEvaluateDraw(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	auto&      table  = OwnTable();
	const auto stages = pair.programs[1] != nullptr ? 2u : 1u;
	auto&      words  = table.words;
	words.clear();
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		const auto  code    = st == 0 ? pair.vs : pair.ps;
		// (Slots no stage reads keep what an earlier draw left: the block carries them unread.)
		auto& slots = table.blocks[st];
		slots.resize(plan.BlockDwords());
		const auto& form = pair.packed[st];
		if (TablePackEnabled() && form.valid && form.runs.size() == pair.runs[st].size() &&
		    user_data[st].size() >= form.user_words && user_data[st].size() <= 64) {
			// The flat form (TablePair::packed): every operand one load from one of three arrays.
			const std::array<const uint32_t*, 3> bases {user_data[st].data(), slots.data(), form.constants.data()};
			const auto get = [&](uint16_t operand) { return bases[operand >> 14u][operand & 0x3fffu]; };
			for (const auto& run: form.runs) {
				uint64_t address = 0;
				if (run.records != TablePair::PackedRun::NotBounded) {
					const auto high    = get(run.address[1]);
					const auto records = get(run.records);
					const auto stride  = (high >> 16u) & 0x3fffu;
					const auto range   = stride == 0 ? uint64_t {records} : uint64_t {stride} * records;
					if (run.immediate < 0 || static_cast<uint64_t>(run.immediate) + run.count * 4ull > range) return false;
					address = (uint64_t {get(run.address[0])} | uint64_t {high & 0xffffu} << 32u) +
					          static_cast<uint64_t>(run.immediate);
				} else {
					address = ((uint64_t {get(run.address[0])} | uint64_t {get(run.address[1])} << 32u) & 0x0000fffffffffffcull) +
					          (get(run.address[2]) & ~3u) + static_cast<uint64_t>(int64_t {run.immediate});
				}
				if (!TableReadRun(address, run.count, slots.data() + run.first)) return false;
			}
			size_t at = 0;
			for (const auto count: form.image_counts) {
				for (uint32_t w = 0; w < count; ++w) words.push_back(get(form.image_words[at + w]) & ~NativeXprLodStatsMask[w]);
				at += count;
			}
			for (; at < form.image_words.size(); ++at) words.push_back(get(form.image_words[at]));
			continue;
		}
		const auto value = [&](const ShaderRecompiler::IR::TablePlan::Operand& operand) {
			return TableOperandValue(operand, program, slots.data(), user_data[st], code);
		};
		for (const auto& run: pair.runs[st]) {
			const auto& read = run.read;
			uint64_t    address = 0;
			if (read.bounded) {
				// Read through a V#: past its range (NUM_RECORDS bytes, or times the stride) the normal path's
				// evaluation fails too.
				const auto high    = value(read.high);
				const auto records = value(read.records);
				const auto stride  = (high >> 16u) & 0x3fffu;
				const auto range   = stride == 0 ? uint64_t {records} : uint64_t {stride} * records;
				if (read.immediate < 0 || static_cast<uint64_t>(read.immediate) + run.count * 4ull > range) return false;
				address = (uint64_t {value(read.low)} | uint64_t {high & 0xffffu} << 32u) +
				          static_cast<uint64_t>(read.immediate);
			} else {
				address = ((uint64_t {value(read.low)} | uint64_t {value(read.high)} << 32u) & 0x0000fffffffffffcull) +
				          (value(read.offset) & ~3u) + static_cast<uint64_t>(int64_t {read.immediate});
			}
			if (!TableReadRun(address, run.count, slots.data() + run.first)) return false;
		}
		for (const auto& descriptor: plan.images)
			for (uint32_t w = 0; w < descriptor.count; ++w) words.push_back(value(descriptor.words[w]) & ~NativeXprLodStatsMask[w]);
		for (const auto& descriptor: plan.samplers)
			for (uint32_t w = 0; w < descriptor.count; ++w) words.push_back(value(descriptor.words[w]));
	}
	return true;
}

// The V#s' words and their resolved entries (TableXpr::entries), as soon as the slots they are read from are: the
// entries' lines (a 4 MiB direct-mapped table) load while the image set is found.
void RenderExecutor::TableGather(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	auto&      table  = OwnTable();
	const auto stages = pair.programs[1] != nullptr ? 2u : 1u;
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		const auto  code    = st == 0 ? pair.vs : pair.ps;
		auto&       entries = table.entries[st];
		entries.resize(plan.buffers.size());
		const auto& form = pair.packed[st];
		const bool  flat = TablePackEnabled() && form.valid && form.buffer_words.size() == 4 * plan.buffers.size() &&
		                  user_data[st].size() >= form.user_words && user_data[st].size() <= 64;
		const std::array<const uint32_t*, 3> bases {user_data[st].data(), table.blocks[st].data(), form.constants.data()};
		for (size_t i = 0; i < plan.buffers.size(); ++i) {
			auto& [words4, entry] = entries[i];
			for (uint32_t w = 0; w < 4; ++w) {
				if (flat) {
					const auto operand = form.buffer_words[4 * i + w];
					words4[w]          = bases[operand >> 14u][operand & 0x3fffu];
				} else {
					words4[w] = TableOperandValue(plan.buffers[i][w], program, table.blocks[st].data(), user_data[st], code);
				}
			}
			entry = &table.resolved[XXH3_64bits(words4.data(), sizeof(words4)) & (TableXpr::ResolvedEntries - 1)];
			_mm_prefetch(reinterpret_cast<const char*>(entry), _MM_HINT_T0);
		}
	}
	table.gathered = &pair;
}

// The image set of the T#/S# words TableEvaluateDraw made: its resources as the normal path evaluates them, which
// must select the pair's table programs' specializations, and its images and samplers as the normal path resolves
// and acquires them (PrepareBindings, RebindImages), in a descriptor set with the page table and the fault buffer.
// Refused (published as such) for another specialization, or when an image cannot be held in a set (a target of the
// draw, an image without data, a linear image the shader writes).
RenderExecutor::TableImageSet* RenderExecutor::TableResolveSet(CommandBuffer& buffer, TablePair& pair,
                                                               const PipelineCache::Pipeline& pipeline,
                                                               const NativeXprDrawState*      draw_state) {
	(void)buffer;
	auto&      table    = OwnTable();
	auto&      textures = m_context.GetTextureCache();
	auto&      buffers  = m_context.GetBufferCache();
	const auto stages   = pair.programs[1] != nullptr ? 2u : 1u;
	const auto epoch    = textures.ImageSetEpoch(); // (before the lookups: a registration they make moves it)
	auto       set      = std::make_unique<TableImageSet>();
	set->words          = table.words;
	set->validated_frame = m_context.FrameNumber();
	set->set_epoch       = epoch;
	const auto publish   = [&] {
		auto& slot = pair.sets[XXH3_64bits(set->words.data(), set->words.size() * sizeof(uint32_t))];
		if (slot) {
			if (table.last_set == slot.get()) table.last_set = nullptr;
			if (pair.last_set == slot.get()) pair.last_set = nullptr;
			if (pair.previous_set == slot.get()) pair.previous_set = nullptr;
			NativeXprRetireSet(slot->pool, slot->set);
		}
		slot = std::move(set);
		return slot.get();
	};
	const auto refuse = [&](const char* reason) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 32)
			std::printf("Table: VS 0x%llx PS 0x%llx image set refused: %s\n", static_cast<unsigned long long>(pair.vs),
			            static_cast<unsigned long long>(pair.ps), reason);
		set->refused = true;
		return publish();
	};
	// Each stage's resources from the evaluation: its V#s (TableGather), T#s and S#s (the set's words).
	size_t at = 0;
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& plan     = pair.programs[st]->table_plan;
		auto&       snapshot = table.snapshots[st];
		snapshot.buffers.resize(plan.buffers.size());
		for (size_t i = 0; i < plan.buffers.size(); ++i) {
			snapshot.buffers[i]             = {};
			snapshot.buffers[i].dword_count = 4;
			std::copy_n(table.entries[st][i].first.begin(), 4, snapshot.buffers[i].dwords.begin());
		}
		const auto take = [&](uint32_t count, ShaderRecompiler::IR::DescriptorValue& value) {
			value             = {};
			value.dword_count = count;
			std::copy_n(table.words.begin() + static_cast<std::ptrdiff_t>(at), count, value.dwords.begin());
			at += count;
		};
		snapshot.images.resize(plan.images.size());
		for (size_t i = 0; i < plan.images.size(); ++i) take(plan.images[i].count, snapshot.images[i]);
		snapshot.samplers.resize(plan.samplers.size());
		for (size_t i = 0; i < plan.samplers.size(); ++i) take(plan.samplers[i].count, snapshot.samplers[i]);
		snapshot.flattened_srt.clear();
		// (PrepareBindings copies the user data registers into shader data the set does not use.)
		uint32_t user_words = 0;
		for (const auto reg: pair.programs[st]->info.user_data_registers)
			user_words = std::max<uint32_t>(user_words, reg - pair.programs[st]->user_data_base + 1);
		snapshot.user_data.assign(user_words, 0u);
		snapshot.specialization_reads.clear();
		snapshot.specialization_read_slots.clear();
		snapshot.image_tables.clear();
		snapshot.uniform_fill = {};
		// (Its images' and samplers' specialization must be the program's: its buffers' are checked at every use.)
		auto& specialization = table.specialization;
		if (!m_context.GetPipelineCache().SpecializeDirectResources(*pair.plans[st], snapshot, specialization))
			return refuse("resources that do not specialize");
		if (specialization.images != pair.tables[st]->specialization.images ||
		    specialization.samplers != pair.tables[st]->specialization.samplers)
			return refuse(st == 0 ? "another specialization (VS)" : "another specialization (PS)");
		table.runtimes[st] = {.program = pair.programs[st], .resources = &snapshot};
	}
	// The normal path's lookups and acquisitions, and each image's layout and access as its CommitBindings gives them:
	// one bound both as a texture and as storage (force_general) wholly in General, storage in General, else
	// read-only.
	struct Unbind {
		RenderExecutor& executor;
		~Unbind() { executor.ResetBindings(); }
	} unbind {*this};
	for (uint32_t st = 0; st < stages; ++st) PrepareBindings(table.runtimes[st], table.prepared[st]);
	for (uint32_t st = 0; st < stages; ++st) RebindImages(table.prepared[st]);
	std::array<std::vector<TableImageSet::Image>, 2> deps;
	for (uint32_t st = 0; st < stages; ++st)
		for (auto& binding: table.prepared[st].images) {
			if (!binding.image_id) continue; // (a null image: bound as the normal path binds it)
			auto&      image   = textures.GetImage(binding.image_id);
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (draw_state != nullptr &&
			    std::ranges::find(draw_state->targets, binding.image_id, &NativeXprDrawState::Target::id) !=
			        draw_state->targets.end())
				return refuse("a target the shader reads");
			if (image.info.data.Empty()) return refuse("an image without data");
			if ((storage || (image.binding.force_general && image.binding.shader_write)) && !image.info.IsTiled())
				return refuse("a linear image the shader writes");
			const auto& view = binding.desc.view_info;
			auto&       dep  = deps[st].emplace_back();
			dep              = {.id           = binding.image_id,
			                    .serial       = image.serial,
			                    .range        = {view.base_level, view.level_count, view.base_layer, view.layer_count},
			                    .desc         = binding.desc,
			                    .registration = image.registrations};
			if (image.binding.force_general && !image.info.IsDepth()) {
				dep.layout  = vk::ImageLayout::eGeneral;
				dep.access  = vk::AccessFlagBits2::eShaderRead |
				              (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlags2 {});
				dep.storage = image.binding.shader_write;
				dep.whole   = true;
			} else if (storage) {
				dep.layout  = vk::ImageLayout::eGeneral;
				dep.access  = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
				dep.storage = true;
			} else {
				dep.layout = image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                  : vk::ImageLayout::eShaderReadOnlyOptimal;
				// (What TextureCache::IsSampledImageCurrent can prove, of the image as well as of the view: a color view of
				// a depth image or of one with metadata, or of a larger image, is proven as the other images are.)
				dep.plain = textures.IsSampledImageCurrent(binding.image_id, binding.desc);
			}
			binding.layout = dep.layout;
		}
	// The set: per stage its bindings (images, samplers, page table, fault buffer), as CommitBindings writes them.
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	struct Pending {
		vk::WriteDescriptorSet write;
		size_t                 start = 0;
		bool                   image = false;
	};
	std::vector<Pending>                  pending;
	std::array<vk::DescriptorBufferInfo, 4> buffer_infos {};
	size_t                                buffer_count = 0;
	std::vector<vk::DescriptorImageInfo>  image_infos;
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program  = *pair.programs[st];
		const auto& prepared = table.prepared[st];
		std::array<size_t, ShaderRecompiler::IR::ImageBindingCount> image_starts {};
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto kind_index = std::countr_zero(mask);
			const auto kind       = static_cast<Kind>(kind_index);
			Pending    item;
			item.write.dstBinding      = ShaderRecompiler::IR::NativeBinding(program.stage, kind);
			item.write.descriptorType  = NativeDescriptorType(kind);
			item.write.descriptorCount = program.bindings.descriptor_counts[kind_index];
			if (ShaderRecompiler::IR::ImageBindingResourceClass(kind) != ShaderRecompiler::IR::ImageResourceClass::None) {
				item.image = true;
				item.start = image_infos.size();
				image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] = item.start;
				image_infos.resize(image_infos.size() + item.write.descriptorCount);
			} else if (kind == Kind::Samplers) {
				item.image = true;
				item.start = image_infos.size();
				if (prepared.samplers.size() != item.write.descriptorCount) return refuse("sampler count");
				for (const auto sampler: prepared.samplers) {
					if (sampler == nullptr) return refuse("a null sampler");
					image_infos.emplace_back(sampler, nullptr, vk::ImageLayout::eUndefined);
				}
			} else if ((kind == Kind::BdaPagetable || kind == Kind::FaultBuffer) && buffer_count < buffer_infos.size()) {
				const auto* owner = kind == Kind::BdaPagetable ? buffers.GetBdaPageTableBuffer() : buffers.GetFaultBuffer();
				item.start        = buffer_count;
				buffer_infos[buffer_count++] = {owner->Handle(), 0, owner->Size()};
			} else {
				return refuse("a binding table mode does not have");
			}
			pending.push_back(item);
		}
		if (prepared.images.size() != program.info.images.size()) return refuse("image count");
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			const auto& resource = program.info.images[i];
			const auto& texture  = prepared.images[i];
			if (texture.image_id && resource.mip_count != (texture.mip_views.empty() ? 1u : texture.mip_views.size()))
				return refuse("image mips");
			const auto kind  = *ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			const auto start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] + resource.descriptor_index;
			for (uint32_t mip = 0; mip < resource.mip_count; ++mip) {
				if (start + mip >= image_infos.size()) return refuse("image slot");
				image_infos[start + mip] = MakeImageInfo(texture, mip);
			}
		}
		set->images.insert(set->images.end(), deps[st].begin(), deps[st].end());
	}
	set->set = NativeXprAllocateSet(pipeline.descriptor_set_layout, set->pool);
	if (!set->set) return refuse("no descriptor set");
	std::vector<vk::WriteDescriptorSet> writes;
	writes.reserve(pending.size());
	for (auto& item: pending) {
		item.write.dstSet = set->set;
		if (item.image) item.write.pImageInfo = image_infos.data() + item.start;
		else item.write.pBufferInfo = buffer_infos.data() + item.start;
		writes.push_back(item.write);
	}
	if (!writes.empty())
		m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0,
		                                                    nullptr);
	return publish();
}

// A set's images are the same objects, registered and current (as NativeXprValidateResources proves a record's):
// once a frame, and again when images were registered or unregistered since. A plain texture whose proof held is
// checked again only when its own registration changed or one since overlaps its view (KYTY_TABLE_SET_PROOFS: 0
// checks all of them in full, 2 does and reports the held proofs that did not pass).
bool RenderExecutor::TableValidateSet(TableImageSet& set) {
	const auto mode     = TableSetProofs();
	auto&      textures = m_context.GetTextureCache();
	const auto frame    = m_context.FrameNumber();
	const auto epoch    = textures.ImageSetEpoch();
	if (set.validated_frame == frame && set.set_epoch == epoch) return true;
	// Registrations since the last proof: a view whose range none of them overlaps is what a lookup still finds.
	const bool moved = set.set_epoch != epoch;
	std::vector<std::pair<uint64_t, uint64_t>> changes;
	const bool known = !moved || textures.ForEachImageSetChangeSince(set.set_epoch, [&](uint64_t address, uint64_t size) {
		changes.emplace_back(address, size);
	});
	const auto overlapped = [&](const TableImageSet::Image& dep) {
		return !known || std::ranges::any_of(changes, [&](const auto& change) {
			       return ImageRangeOverlaps(change.first, change.second, dep.desc.info.data.address,
			                                 dep.desc.info.data.size);
		       });
	};
	for (auto& dep: set.images) {
		auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr || image->serial != dep.serial || !image->registered || image->binding.needs_rebind ||
		    image->depth_id)
			return false;
		// The guest rewrote it (a texture streaming in), or a GPU buffer write its bytes: upload it as the normal
		// path's FindTexture does.
		if (TextureCache::RefreshPending(*image, dep.desc)) {
			if (dep.storage || image->depth_id) return false;
			(void)textures.FindTexture(dep.id, dep.desc);
			image = textures.m_slot_images.try_get(dep.id);
			if (image == nullptr || image->serial != dep.serial || TextureCache::RefreshPending(*image, dep.desc)) return false;
		}
		if (moved) {
			// A plain texture still in the registration its proof saw, with nothing registered over its view since, is
			// what the proof found (mode 1); else the full check: the texture cache's, or nothing registered over the
			// view (mode 0, and mode 2, which also counts the held proofs the full check fails).
			const bool held = mode != 0 && dep.plain && image->registrations == dep.registration && !overlapped(dep);
			if (!held || mode == 2) {
				const bool current = dep.plain ? textures.IsSampledImageCurrent(dep.id, dep.desc) : !overlapped(dep);
				if (!current) {
					if (held) {
						static std::atomic<uint32_t> reports {0};
						if (reports.fetch_add(1, std::memory_order_relaxed) < 16)
							std::printf("Table: proven image use failed: addr=0x%llx size=0x%llx serial=%llu "
							            "(image depth %d metadata %d stencil %d)\n",
							            static_cast<unsigned long long>(dep.desc.info.data.address),
							            static_cast<unsigned long long>(dep.desc.info.data.size),
							            static_cast<unsigned long long>(dep.serial), image->info.IsDepth() ? 1 : 0,
							            image->info.HasMetadata() ? 1 : 0, image->info.HasStencil() ? 1 : 0);
					}
					return false;
				}
			}
			dep.registration = image->registrations;
		}
		(void)textures.GetImage(dep.id); // LRU, as FindTexture does
	}
	set.validated_frame = frame;
	set.set_epoch       = epoch;
	return true;
}

// The image set of the words TableEvaluateDraw made: the last one when they are the same, else the pair's set for
// them, proven current. Null: the normal path takes the draw.
RenderExecutor::TableImageSet* RenderExecutor::TableSet(CommandBuffer& buffer, TablePair& pair,
                                                        const PipelineCache::Pipeline& pipeline,
                                                        const NativeXprDrawState*      draw_state) {
	auto&       table = OwnTable();
	const auto& words = table.words;
	const auto  frame = m_context.FrameNumber();
	auto*       set   = pair.last_set;
	if (set == nullptr || set->words != words) {
		auto* const last     = set;
		auto* const previous = pair.previous_set;
		// (The set before the last next, without the words' hash or the map's cold bucket and node.)
		if ((TableMru() & 4u) != 0 && previous != nullptr && previous != last && !previous->refused &&
		    previous->words == words) {
			set = previous;
		} else {
			const auto found = pair.sets.find(XXH3_64bits(words.data(), words.size() * sizeof(uint32_t)));
			set = found != pair.sets.end() && found->second->words == words ? found->second.get() : nullptr;
			if (set == nullptr || (set->refused && frame >= set->validated_frame + TableImageSet::RetryFrames))
				set = TableResolveSet(buffer, pair, pipeline, draw_state);
		}
		// (TableResolveSet may have cleared both: a set it replaced.)
		if (last != nullptr && pair.last_set == last) pair.previous_set = last;
		pair.last_set = set;
	}
	set->used_frame = frame;
	if (set->refused) return nullptr;
	if (!TableValidateSet(*set)) {
		// Resolved again at the next draw that uses it.
		if (table.last_set == set) table.last_set = nullptr;
		pair.last_set     = nullptr;
		pair.previous_set = nullptr;
		NativeXprRetireSet(set->pool, set->set);
		pair.sets.erase(XXH3_64bits(set->words.data(), set->words.size() * sizeof(uint32_t)));
		return nullptr;
	}
	// Unused image sets leave once a second.
	if (frame >= table.collected + 60) {
		table.collected = frame;
		for (auto* pairs: {&table.pairs, &table.dispatches})
			for (auto& [key, held]: *pairs)
				std::erase_if(held.sets, [&](auto& item) {
					if (item.second->used_frame + 60 >= frame) return false;
					if (held.last_set == item.second.get()) held.last_set = nullptr;
					if (held.previous_set == item.second.get()) held.previous_set = nullptr;
					if (table.last_set == item.second.get()) table.last_set = nullptr;
					NativeXprRetireSet(item.second->pool, item.second->set);
					return true;
				});
	}
	return set;
}

// The page table ranges the pair's flat and global loads read made current, as the normal path's
// PrepareGraphicsBindings does at every use (from the user data and the blocks' slots).
void RenderExecutor::TablePrepareBda(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	auto&                                    table = OwnTable();
	std::array<const ShaderStageRuntime*, 2> stages {};
	uint32_t                                 count = 0;
	for (uint32_t st = 0; st < 2; ++st) {
		const auto* program = pair.programs[st];
		if (program == nullptr || !program->info.uses_dma) continue;
		auto& snapshot = table.snapshots[st];
		snapshot.user_data.assign(user_data[st].begin(), user_data[st].end());
		snapshot.flattened_srt.assign(table.blocks[st].begin(),
		                              table.blocks[st].begin() + static_cast<std::ptrdiff_t>(program->table_plan.slots.size()));
		table.runtimes[st] = {.program = program, .resources = &snapshot};
		m_context.CacheDmaBases(table.runtimes[st]);
		stages[count++] = &table.runtimes[st];
	}
	if (count == 0) return;
	const bool reach_synchronized = m_context.SynchronizeDmaFootprint(std::span {stages.data(), count});
	const bool writes = (pair.programs[0] != nullptr && pair.programs[0]->has_address_writes) ||
	                    (pair.programs[1] != nullptr && pair.programs[1]->has_address_writes);
	m_context.PrepareBda(writes, !reach_synchronized);
}

// The pair's blocks (TableEvaluateDraw read their slots): each stage's buffers' device addresses and sizes in dwords
// after its slots, in the table upload ring. A read-only V# is resolved once (TableXpr::resolved); a written one is
// obtained as written at every use, as the normal path's (the buffer cache tracks the GPU's writes). False: a buffer
// the shaders cannot take.
bool RenderExecutor::TableBlocks(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data,
                                 std::array<vk::DeviceAddress, 2>& blocks, bool& writes) {
	auto&      table        = OwnTable();
	auto&      buffer_cache = m_context.GetBufferCache();
	auto&      textures     = m_context.GetTextureCache();
	const auto stages       = pair.programs[1] != nullptr ? 2u : 1u;
	if (table.special_images != textures.ImageSetEpoch() || table.special_mappings != m_context.MappedRangesVersion()) {
		table.special_images   = textures.ImageSetEpoch();
		table.special_mappings = m_context.MappedRangesVersion();
		++table.special_generation;
	}
	// As FindBuffers: a V# that is no buffer, or of nothing, is a null buffer (size 0).
	const auto range_of = [](const std::array<uint32_t, 4>& words4) -> std::tuple<uint64_t, uint64_t, uint64_t> {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		const auto requested = descriptor.GetSize();
		const auto address   = descriptor.Base48();
		if (descriptor.Type() != 0 || address == 0 || requested == 0) return {address, 0, 0};
		return {address, Libs::LibKernel::Memory::ClampRangeSize(address, requested), requested};
	};
	const auto write_buffer = [&](std::vector<uint32_t>& block, const ShaderRecompiler::IR::TablePlan& plan, size_t i,
	                              vk::DeviceAddress device, uint64_t dwords, const std::array<uint32_t, 4>& words4) {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		const auto first = plan.BufferDword(i);
		block[first]     = static_cast<uint32_t>(device);
		block[first + 1] = static_cast<uint32_t>(device >> 32u);
		block[first + 2] = static_cast<uint32_t>(std::min<uint64_t>(dwords, UINT32_MAX));
		block[first + 3] = 0;
		block[first + 4] = descriptor.Type() == 0 ? descriptor.Stride() : 0u;
	};
	if (table.gathered != &pair) TableGather(pair, user_data);
	table.gathered = nullptr; // (taken: a later draw gathers its own)
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		const auto& spec    = pair.tables[st]->specialization;
		auto&       block   = table.blocks[st];
		auto&       entries = table.entries[st];
		// The buffers it writes first: a range it also reads through another V# is GPU-written by then, as the
		// normal path's lookups leave it.
		for (size_t n = 0; n < 2 * plan.buffers.size(); ++n) {
			const auto  i        = n < plan.buffers.size() ? n : n - plan.buffers.size();
			const auto& words4   = entries[i].first;
			const auto& resource = program.info.buffers[i];
			const bool  written  = resource.written || resource.atomic;
			if (written != (n < plan.buffers.size())) continue;
			if (i >= spec.buffers.size() || !TableBufferSpecialized(resource, spec.buffers[i], words4)) return false;
			const bool texel = resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled());
			if (!resource.live) {
				write_buffer(block, plan, i, 0, 0, words4);
				continue;
			}
			if (written) {
				const auto [address, size, requested] = range_of(words4);
				vk::DeviceAddress device              = 0;
				if (size != 0) {
					if ((address & 3u) != 0 || words4[2] == UINT32_MAX) return false;
					const auto [owner, offset] = buffer_cache.ObtainBuffer(address, size, true, texel);
					if (owner == nullptr) return false;
					textures.InvalidateMemoryFromGPU(address, size);
					device = owner->BufferDeviceAddress() + offset;
					writes = writes || resource.written;
				}
				write_buffer(block, plan, i, device, size / 4u, words4);
				continue;
			}
			auto& held    = *entries[i].second;
			bool  current = held.words == words4 && (held.special == 0 || held.special == table.special_generation) &&
			               held.registration == buffer_cache.BufferSetEpoch();
			if (current && held.size != 0) {
				// Another page of its tracker regions was written: the range itself may still be clean.
				if (const auto epoch = buffer_cache.CpuDirtyEpoch(held.address, held.size);
				    epoch == 0 || epoch != held.cpu_epoch) {
					current = !buffer_cache.IsRegionCpuModified(held.address, held.size);
					if (current) held.cpu_epoch = epoch;
				}
			}
			if (current) {
				write_buffer(block, plan, i, held.device, held.dwords, words4);
				continue;
			}
			held                                  = {};
			const auto [address, size, requested] = range_of(words4);
			TableXpr::ResolvedBuffer fresh {.words = words4};
			if (texel || size < requested) fresh.special = table.special_generation;
			if (size != 0) {
				// The shader reads whole dwords from the device address; a buffer an image starts at is made from the
				// image by ObtainBuffer only (SynchronizeBufferFromImage).
				if ((address & 3u) != 0 || words4[2] == UINT32_MAX || (texel && textures.HasImageStartingAt(address)))
					return false;
				const auto cpu_epoch       = buffer_cache.CpuDirtyEpoch(address, size); // (before the uploads)
				const auto [owner, offset] = buffer_cache.ObtainBuffer(address, size, false, texel);
				if (owner == nullptr) return false;
				const auto device = owner->BufferDeviceAddress() + offset;
				// A small range the guest wrote is read from a copy in the stream ring (ObtainBuffer's): this use only.
				if (!owner->IsInBounds(address, size) || buffer_cache.IsRegionCpuModified(address, size)) {
					write_buffer(block, plan, i, device, size / 4u, words4);
					continue;
				}
				fresh.device       = device;
				fresh.dwords       = static_cast<uint32_t>(std::min<uint64_t>(size / 4u, UINT32_MAX));
				fresh.address      = address;
				fresh.size         = size;
				fresh.cpu_epoch    = cpu_epoch;
			}
			fresh.registration = buffer_cache.BufferSetEpoch(); // (after ObtainBuffer's own registrations)
			held               = fresh;
			write_buffer(block, plan, i, held.device, held.dwords, words4);
		}
		if (plan.BlockDwords() != 0) // (a stage with neither slots nor buffers reads no block)
			blocks[st] = buffer_cache.TableUpload(block.data(), block.size() * sizeof(uint32_t));
	}
	return true;
}

// A set's images in their layouts (other work may have moved them since the set was last bound), unless the draw's
// targets are not those the set was made for or an image needs what the normal path's acquisition does first (an
// upload, a pending fast clear, a storage image the GPU does not own).
bool RenderExecutor::TableTransit(const TableImageSet& set, CommandRecorder vk_buffer,
                                  const NativeXprDrawState* draw_state) {
	auto& textures = m_context.GetTextureCache();
	for (const auto& dep: set.images) {
		auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr || image->serial != dep.serial) return false;
		if (draw_state != nullptr &&
		    std::ranges::find(draw_state->targets, dep.id, &NativeXprDrawState::Target::id) != draw_state->targets.end())
			return false;
		if ((dep.storage && !image->SafeToDownload()) || TextureCache::RefreshPending(*image, dep.desc)) return false;
		if (dep.desc.info.metadata.kind == ImageMetadataKind::Dcc) {
			for (uint32_t layer = dep.range.base_layer; layer < dep.range.base_layer + dep.range.layer_count; ++layer)
				if (textures.IsMetaCleared(dep.desc.info.metadata.range.address, layer)) return false;
		}
	}
	// (In the draw's transit group, as CommitBindings' transitions.)
	for (const auto& dep: set.images)
		textures.m_slot_images[dep.id].Transit(dep.layout, dep.access,
		                                       dep.whole ? std::nullopt : std::optional {dep.range}, vk_buffer);
	return true;
}

RenderExecutor::TableResult RenderExecutor::TableDraw(CommandBuffer& buffer, TablePair& pair, TableVariant& variant,
                                                      NativeXprDrawState& draw_state, uint64_t args_addr,
                                                      uint64_t index_base, uint64_t index_bytes,
                                                      vk::IndexType index_type, const DrawIndexArgs* direct) {
	auto&       table  = OwnTable();
	auto&       cache  = *m_native_xpr;
	const auto& sh     = buffer.GetShaders();
	const auto  stages = pair.programs[1] != nullptr ? 2u : 1u;
	const auto  frame  = m_context.FrameNumber();
	const std::array<std::span<const uint32_t>, 2> user_data {
	    std::span<const uint32_t>(sh.GetVs().gs_user_sgpr.value,
	                              std::min<uint32_t>(sh.GetVs().gs_regs.rsrc2.user_sgpr, HW::UserSgprInfo::SGPRS_MAX)),
	    std::span<const uint32_t>(sh.GetPs().ps_user_sgpr.value,
	                              std::min<uint32_t>(sh.GetPs().ps_regs.rsrc2.user_sgpr, HW::UserSgprInfo::SGPRS_MAX))};
	if (!TableEvaluateDraw(pair, user_data)) return TableResult::Native;
	TableGather(pair, user_data);
	auto* set = TableSet(buffer, pair, *variant.pipeline, &draw_state);
	if (set == nullptr) return TableResult::Native;

	// Emission, as NativeXprEmit's: indices and arguments are read by the GPU; lookups may restart the scheduler, so
	// the command buffer handle is taken after them.
	auto&      buffer_cache = m_context.GetBufferCache();
	const auto obtain = [&](TableXpr::HeldBuffer& held, uint64_t address, uint64_t size,
	                        auto&& prepare) -> std::pair<Buffer*, uint64_t> {
		if (held.buffer != nullptr && address >= held.begin && address + size <= held.end &&
		    held.registration == buffer_cache.BufferSetEpoch() && held.cpu_epoch != 0 &&
		    buffer_cache.CpuDirtyEpoch(held.begin, held.end - held.begin) == held.cpu_epoch)
			return {held.buffer, held.buffer->Offset(address)};
		held = {};
		prepare();
		const auto result = buffer_cache.ObtainBuffer(address, size, false);
		if (result.first != nullptr && result.first->IsInBounds(address, size)) {
			// The buffer within the tracker region of the address (one region's epoch proves it) when none of it is
			// CPU-dirty, else the range. (The epoch before the dirty check: a write after it moves the epoch.)
			const auto region = address & ~(TRACKER_REGION_SIZE - 1);
			auto       begin  = std::max(result.first->CpuAddress(), region);
			auto       end    = std::min(result.first->CpuAddress() + result.first->Size(), region + TRACKER_REGION_SIZE);
			if (address + size <= end) {
				const auto epoch = buffer_cache.CpuDirtyEpoch(begin, end - begin);
				if (buffer_cache.IsRegionCpuModified(begin, end - begin)) {
					begin = address;
					end   = address + size;
				}
				if (epoch != 0 && !buffer_cache.IsRegionCpuModified(begin, end - begin))
					held = {begin, end, epoch, buffer_cache.BufferSetEpoch(), result.first};
			}
		}
		return result;
	};
	std::array<vk::DeviceAddress, 2> blocks {};
	bool                             writes = false;
	TablePrepareBda(pair, user_data);
	// The blocks take each buffer's device address as it resolves: a later lookup that registers a buffer (a merge
	// replaces an earlier one) leaves an earlier address stale, where the normal path binds after all its lookups.
	// Such a draw takes the normal path; the next one finds the buffers registered.
	const auto registration = buffer_cache.BufferSetEpoch();
	if (!TableBlocks(pair, user_data, blocks, writes)) return TableResult::Native;
	constexpr uint64_t           stride = sizeof(vk::DrawIndexedIndirectCommand);
	std::pair<Buffer*, uint64_t> arguments {};
	if (direct == nullptr) {
		arguments = obtain(table.args, args_addr, stride, [] {});
		if (arguments.first == nullptr) return TableResult::Native;
	}
	auto&      dynamic    = m_context.GetCommandScheduler().GetDynamicState();
	auto&      emit       = cache.emit;
	const bool same_index = emit.generation == dynamic.Generation() && emit.index_base == index_base &&
	                        emit.index_bytes == index_bytes && emit.index_type == index_type;
	std::pair<Buffer*, uint64_t> indices {};
	if (!same_index) {
		indices = obtain(table.indices[(index_base * 0x9e3779b97f4a7c15ull) >> 51u], index_base, index_bytes, [&] {
			if (index_bytes <= BufferCache::CACHING_PAGESIZE && buffer_cache.IsRegionCpuModified(index_base, index_bytes) &&
			    !buffer_cache.IsRegionGpuModified(index_base, index_bytes))
				buffer_cache.EnsureBufferContents(index_base, index_bytes);
		});
		if (indices.first == nullptr) return TableResult::Native;
	}
	if (buffer_cache.BufferSetEpoch() != registration || buffer.IsInvalid()) return TableResult::Native;
	auto vk_buffer = buffer.Handle();
	// The targets and the set's images as the last table draw in this command buffer left them (the same state and
	// set), no image's state, registration, metadata or bytes changed since: in place, nothing to check or transit.
	auto&      textures     = m_context.GetTextureCache();
	const bool same_command = emit.generation == dynamic.Generation();
	const auto images_epoch = textures.ImagesEpoch();
	const bool in_place     = same_command && emit.state == &draw_state && table.last_set == set &&
	                      table.images_epoch == images_epoch;
	if (!in_place) {
		const TransitGroup transit_group; // (the transitions of a guest draw)
		// The targets apart, as this draw state's last emission left them, and the set's images, as its last transit
		// under this state did (the passes alternate materials: another set or state in between).
		if (draw_state.emitted_epoch != images_epoch && !NativeXprEmitTargets(draw_state, vk_buffer))
			return TableResult::Native;
		if ((set->transit_epoch != images_epoch || set->transit_state != &draw_state ||
		     set->transit_capture != draw_state.captures) &&
		    !TableTransit(*set, vk_buffer, &draw_state))
			return TableResult::Native;
		// (Neither moves what the other placed: a target is no image the set holds.)
		draw_state.emitted_epoch = set->transit_epoch = textures.ImagesEpoch();
		set->transit_state       = &draw_state;
		set->transit_capture     = draw_state.captures;
	}
	if (draw_state.used_frame != frame) {
		// (Once a frame the targets' uses, as their acquisition touches them: the collector keeps them.)
		for (const auto& target: draw_state.targets) (void)textures.GetImage(target.id);
		draw_state.used_frame = frame;
	}
	m_context.GetCommandScheduler().BeginRendering(draw_state.rendering);
	if (!same_command || emit.state != &draw_state) {
		dynamic.Assign(draw_state.dynamic);
		dynamic.Commit(m_context.GetGraphics(), vk_buffer);
	}
	const auto& pipeline = *variant.pipeline;
	if (!same_command || emit.pipeline != &pipeline) vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	if (!same_command || emit.pipeline != &pipeline || table.last_set != set)
		vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.pipeline_layout, 0, 1, &set->set, 0,
		                             nullptr);
	// Each stage's shader data where its push data starts: its user data registers, then its block's address.
	ShaderRecompiler::IR::PushData push_data;
	uint32_t                       push_dwords = 0;
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program  = *pair.programs[st];
		const auto& bindings = program.bindings;
		const auto  start    = bindings.push_data_start_dword;
		push_dwords          = std::max(push_dwords, start + bindings.ShaderDataDwords());
		const auto& registers = program.info.user_data_registers;
		for (size_t i = 0; i < registers.size(); ++i) {
			const auto index           = registers[i] - program.user_data_base;
			push_data.dwords[start + i] = index < user_data[st].size() ? user_data[st][index] : 0u;
		}
		push_data.dwords[start + bindings.TableBlockDword()]      = static_cast<uint32_t>(blocks[st]);
		push_data.dwords[start + bindings.TableBlockDword() + 1u] = static_cast<uint32_t>(blocks[st] >> 32u);
	}
	if (push_dwords != 0)
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
		                        push_dwords * sizeof(uint32_t), push_data.dwords.data());
	if (!same_index) vk_buffer.bindIndexBuffer(indices.first->Handle(), indices.second, index_type);
	if (direct != nullptr) {
		// The offsets as DrawIndex() resolves them (ResolveDrawOffsets()): the index offset register, else (a vertex
		// program that fetches) the user data its fetch adds.
		const auto& vertex          = *pair.programs[0];
		auto        vertex_offset   = static_cast<int32_t>(buffer.GetUserConfig().GetIndexOffset());
		uint32_t    instance_offset = 0;
		if (variant.fetch_embedded) {
			const auto base = static_cast<int32_t>(vertex.user_data_base);
			if (vertex_offset == 0 && vertex.info.vertex_offset_sgpr >= base) {
				const auto index = static_cast<uint32_t>(vertex.info.vertex_offset_sgpr - base);
				if (index < user_data[0].size()) vertex_offset = static_cast<int32_t>(user_data[0][index]);
			}
			if (vertex.info.instance_offset_sgpr >= base) {
				const auto index = static_cast<uint32_t>(vertex.info.instance_offset_sgpr - base);
				if (index < user_data[0].size()) instance_offset = user_data[0][index];
			}
		}
		vk_buffer.drawIndexed(direct->index_count, direct->instance_count, 0, vertex_offset + direct->base_vertex,
		                      instance_offset);
	} else {
		vk_buffer.drawIndexedIndirect(arguments.first->Handle(), arguments.second, 1, static_cast<uint32_t>(stride));
	}
	// As the normal path's: shaders that write buffers end the rendering, and a barrier orders their writes.
	if (writes) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader);
	}
	emit.generation  = dynamic.Generation();
	emit.state       = &draw_state;
	emit.pipeline    = &pipeline;
	emit.index_base  = index_base;
	emit.index_bytes = index_bytes;
	emit.index_type  = index_type;
	table.last_pair    = &pair;
	table.last_variant = static_cast<size_t>(&variant - pair.variants.data());
	table.last_state   = &draw_state;
	table.last_set     = set;
	table.images_epoch = textures.ImagesEpoch();
	return TableResult::Drawn;
}

RenderExecutor::TableResult RenderExecutor::TableTry(CommandBuffer& buffer, uint64_t args_addr, uint64_t index_base,
                                                     uint64_t index_bytes, vk::IndexType index_type,
                                                     const DrawIndexArgs* direct) {
	if (TableXprComplements() && !NativeXprWontServe(buffer)) return TableResult::Native;
	const auto& sh   = buffer.GetShaders();
	const auto& vs   = sh.GetVs();
	const auto& ucfg = buffer.GetUserConfig();
	const auto& ctx  = buffer.GetRegisters();
	// What a table variant draws, as a native XPR record: an indexed triangle list of a vertex and a pixel stage (no
	// tessellation; TableStore refuses mesh shading), with no draw data pushed (object id) and no metadata color
	// operation.
	if (vs.es_regs.data_addr == 0 || vs.ls_regs.data_addr != 0 || vs.hs_regs.data_addr != 0 ||
	    ucfg.GetPrimType() != Prospero::PrimitiveType::kTriList || ctx.HasObjectPrimitiveIdOutput() ||
	    ctx.GetColorControl().mode > 1u || (ucfg.GetPrimitiveResetControl() & 0x1u) != 0)
		return TableResult::Native;
	auto&      table = OwnTable();
	auto&      cache = NativeXprState();
	const auto frame = m_context.FrameNumber();
	// The pairs no draw used for PairKeepFrames leave, their image sets retired, with every pointer to one.
	if (frame - table.pairs_swept >= TableXpr::PairSweepFrames) {
		table.pairs_swept = frame;
		if (table.pairs.size() > TableXpr::PairsKept) {
			const auto before = table.pairs.size();
			std::erase_if(table.pairs, [&](auto& item) {
				auto& held = item.second;
				// (A compilation it waits for keeps it.)
				if (held.used_frame + TableXpr::PairKeepFrames >= frame || held.pending != nullptr) return false;
				for (auto& [words, set]: held.sets) NativeXprRetireSet(set->pool, set->set);
				return true;
			});
			if (table.pairs.size() != before) {
				table.store      = nullptr;
				table.gathered   = nullptr;
				table.last_pair  = nullptr;
				table.last_set   = nullptr;
				table.pair_cache = {};
				table.found_pair = nullptr;
			}
		}
	}
	const auto vs_address = vs.es_regs.data_addr, ps_address = sh.GetPs().ps_regs.data_addr;
	const std::array<uint64_t, 2> pair_key {vs_address, ps_address};
	auto* known = table.found_pair != nullptr && table.found_key == pair_key ? table.found_pair : nullptr;
	if (known == nullptr) {
		const auto hash    = NativeXprPairHash(vs_address, ps_address);
		const auto entries = (TableMru() & 1u) != 0 ? TableXpr::PairCacheEntries : 256;
		auto&      cached  = table.pair_cache[hash & (entries - 1)];
		known              = cached.first == hash ? cached.second : nullptr;
		if (known == nullptr) {
			auto [found, fresh] = table.pairs.try_emplace(hash);
			if (fresh) {
				found->second.vs = vs_address;
				found->second.ps = ps_address;
			}
			known  = &found->second;
			cached = {hash, known};
		}
		table.found_key  = pair_key;
		table.found_pair = known;
	}
	auto& pair = *known;
	pair.used_frame = frame;
	if (pair.vs != vs_address || pair.ps != ps_address) return TableResult::Native;
	if (const auto generation = ShaderMapGeneration(); generation != pair.generation) {
		if (pair.programs[0] != nullptr &&
		    (ShaderHashAt(vs_address) != pair.code_hashes[0] ||
		     (pair.programs[1] != nullptr && ShaderHashAt(ps_address) != pair.code_hashes[1])))
			TableReset(pair);
		pair.generation = generation;
	}
	if (pair.refused) return TableResult::Native;
	// The pair's programs, once compiled.
	if (pair.pending != nullptr && pair.pending->done.load(std::memory_order_acquire)) {
		const auto& ticket = *pair.pending;
		const auto  serves = [](const PipelineCache::TableProgram* program) {
			return program != nullptr && program->program.table_mode && program->program.bindings.UsesPushData() &&
			       !program->program.has_address_writes && !program->program.info.uses_gds;
		};
		if (serves(ticket.programs[0]) && (!pair.has_pixel || serves(ticket.programs[1]))) {
			pair.tables   = {ticket.programs[0], pair.has_pixel ? ticket.programs[1] : nullptr};
			pair.programs = {&pair.tables[0]->program, pair.has_pixel ? &pair.tables[1]->program : nullptr};
			TableRuns(pair);
		} else {
			static std::atomic<uint32_t> reports {0};
			if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
				std::printf("Table: VS 0x%llx PS 0x%llx refused: %s\n", static_cast<unsigned long long>(pair.vs),
				            static_cast<unsigned long long>(pair.ps),
				            ticket.refusal.empty() ? "no table mode program" : ticket.refusal.c_str());
			pair.refused = true;
		}
		pair.pending.reset();
		if (pair.refused) return TableResult::Native;
	}
	cache.state_key      = NativeXprStateKeyOf(buffer);
	const auto state_key = cache.state_key;
	// A store at most once a frame per pair and draw state: while its programs or its pipeline are on their way, the
	// pair's draws take the normal path (Deferred).
	// Within the frame's budget (KYTY_TABLE_STORE_BUDGET): past it the others take native records or the normal path
	// meanwhile, and ask again in a later frame.
	const auto request_store = [&] {
		if (pair.pending != nullptr || (pair.store_frame == frame && pair.store_state == state_key))
			return TableResult::Deferred;
		if (table.store_frame != frame) {
			table.store_frame    = frame;
			table.store_requests = 0;
		}
		if (const auto budget = TableStoreBudget(); budget != 0 && table.store_requests >= budget)
			return TableResult::Native;
		++table.store_requests;
		pair.store_frame  = frame;
		pair.store_state  = state_key;
		table.store       = &pair;
		table.store_state = state_key;
		return TableResult::Store;
	};
	// (The variant found last first: a pair's draws mostly keep their state.)
	const bool hint    = (TableMru() & 2u) != 0 && pair.variant_hint < pair.variants.size() &&
	                  pair.variants[pair.variant_hint].state == state_key;
	const auto variant = hint ? pair.variants.begin() + pair.variant_hint
	                          : std::ranges::find(pair.variants, state_key, &TableVariant::state);
	if (variant != pair.variants.end()) pair.variant_hint = static_cast<uint32_t>(variant - pair.variants.begin());
	if (variant != pair.variants.end() && variant->refused) return TableResult::Native;
	if (pair.programs[0] == nullptr || variant == pair.variants.end() || variant->pipeline == nullptr)
		return request_store();
	const auto state = cache.states.find(state_key);
	if (state == cache.states.end()) return request_store();
	auto& draw_state = *state->second;
	if (!NativeXprTargetsCurrent(draw_state)) return request_store();
	return TableDraw(buffer, pair, *variant, draw_state, args_addr, index_base, index_bytes, index_type, direct);
}

// KYTY_TABLE_XPR=2: a clean draw (no register but user data and index state changed since the draw before:
// GpuOperation::clean) right after a table draw whose emission is the last one in the command buffer has that draw's
// pair, variant and draw state, with its pipeline and dynamic state still bound: no pair, state key or variant lookup.
bool RenderExecutor::TableContinue(CommandBuffer& buffer, uint64_t args_addr, uint64_t index_base,
                                   uint64_t index_bytes, vk::IndexType index_type, const DrawIndexArgs* direct) {
	if (!m_table_xpr || !m_native_xpr) return false;
	auto& table = *m_table_xpr;
	auto* pair  = table.last_pair;
	if (pair == nullptr || table.last_state == nullptr || table.last_variant >= pair->variants.size() ||
	    pair->generation != ShaderMapGeneration())
		return false;
	const auto& sh = buffer.GetShaders();
	if (sh.GetVs().es_regs.data_addr != pair->vs || sh.GetPs().ps_regs.data_addr != pair->ps) return false;
	auto&       variant = pair->variants[table.last_variant];
	const auto& emit    = m_native_xpr->emit;
	if (emit.generation != m_context.GetCommandScheduler().GetDynamicState().Generation() ||
	    emit.state != table.last_state || emit.pipeline != variant.pipeline || variant.pipeline == nullptr)
		return false;
	if (!NativeXprTargetsCurrent(*table.last_state)) return false;
	return TableDraw(buffer, *pair, variant, *table.last_state, args_addr, index_base, index_bytes, index_type,
	                 direct) == TableResult::Drawn;
}

bool RenderExecutor::TableXprTry(CommandBuffer& buffer, uint64_t args_addr, uint64_t index_base,
                                 uint64_t index_bytes, vk::IndexType index_type, const DrawIndexArgs* direct,
                                 bool clean) {
	m_table_store   = false;
	m_table_claimed = false;
	if (TableXprMode() == 0 || buffer.IsInvalid()) return false;
	if (direct != nullptr) {
		// A direct indexed draw of one object, as NativeXprTryDirect takes them.
		if (direct->gpu_indirect != nullptr || direct->offset_source != DrawOffsetSource::DrawState ||
		    direct->index_addr == nullptr || direct->index_count == 0 || direct->instance_count == 0 ||
		    direct->render_target_slice_offset != 0 || direct->first_instance != 0 || direct->index_type_and_size > 1)
			return false;
		index_base  = reinterpret_cast<uint64_t>(direct->index_addr);
		index_bytes = uint64_t {direct->index_count} * (direct->index_type_and_size == 0 ? 2u : 4u);
		index_type  = direct->index_type_and_size == 0 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
	}
	if (index_base == 0 || index_bytes == 0) return false;
	m_context.GetCommandScheduler().PopPendingOperations();
	Common::LockGuard lock(m_context.GetMutex());
	NativeXprCollect();
	const auto result =
	    TableXprMode() >= 2 && clean && TableContinue(buffer, args_addr, index_base, index_bytes, index_type, direct)
	        ? TableResult::Drawn
	        : TableTry(buffer, args_addr, index_base, index_bytes, index_type, direct);
	// (The resolve thread skips the resolution ahead of draws of pairs whose last try drew: NativeXprLikely.)
	m_native_xpr_drawn[NativeXprPairOfShaders(buffer.GetShaders()) & (m_native_xpr_drawn.size() - 1)].store(
	    result == TableResult::Drawn ? 1 : 0, std::memory_order_relaxed);
	switch (result) {
		case TableResult::Drawn: return true;
		case TableResult::Store: m_table_store = m_table_claimed = true; break;
		case TableResult::Deferred: m_table_claimed = true; break;
		case TableResult::Native: break;
	}
	return false;
}

void RenderExecutor::TableStore(CommandBuffer& buffer, const DrawRenderState& state, vk::PrimitiveTopology topology,
                                bool primitive_restart_enable, const RenderState& rendering,
                                const DynamicState& dynamic_state, bool storable) {
	auto& table = OwnTable();
	auto* pair  = std::exchange(table.store, nullptr);
	if (pair == nullptr) return;
	const auto& sh = buffer.GetShaders();
	if (sh.GetVs().es_regs.data_addr != pair->vs || sh.GetPs().ps_regs.data_addr != pair->ps) return;
	const auto refuse = [&](const char* reason) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
			std::printf("Table: VS 0x%llx PS 0x%llx refused: %s\n", static_cast<unsigned long long>(pair->vs),
			            static_cast<unsigned long long>(pair->ps), reason);
		pair->refused = true;
	};
	if (!storable) return refuse("a draw table programs cannot make (vertex buffers, draw data, host indices)");
	if (state.programs.VertexStageCount() != 1 || !state.programs.vertex[0] || !state.vertex_info[0].stage ||
	    state.vertex_info[0].logical_stage == ShaderType::Mesh || (state.ps_active && !state.programs.pixel))
		return refuse("stages");
	// The stages' static input, which the pair's programs were compiled for.
	auto&                 pipelines = m_context.GetPipelineCache();
	const auto&           ctx       = buffer.GetRegisters();
	ShaderVertexInputInfo vertex_info {};
	ShaderPixelInputInfo  pixel_info {};
	ShaderParams          vertex_params, pixel_params;
	if (!pipelines.PrepareGraphicsStages(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx, buffer.GetUserConfig(),
	                                     TableDrawExportMapping(ctx, DrawColorOutputMask(ctx)), state.ps_active,
	                                     vertex_info, pixel_info, vertex_params, pixel_params))
		return refuse("stages");
	const auto static_key = TableDrawKey(table.static_key, vertex_params, vertex_info,
	                                     state.ps_active ? &pixel_params : nullptr, pixel_info);
	if (pair->programs[0] == nullptr) {
		if (pair->pending != nullptr) return;
		const auto [vertex_plan, vertex_specialization] = pipelines.ProgramPlanOf(state.programs.vertex[0]);
		const auto [pixel_plan, pixel_specialization] =
		    state.ps_active ? pipelines.ProgramPlanOf(state.programs.pixel)
		                    : std::pair<const ShaderRecompiler::IR::ResourcePlan*,
		                                const ShaderRecompiler::IR::ResourceSpecialization*> {nullptr, nullptr};
		if (vertex_plan == nullptr || (state.ps_active && pixel_plan == nullptr)) return refuse("no resource plan");
		pair->plans       = {vertex_plan, pixel_plan};
		pair->static_key  = static_key;
		pair->has_pixel   = state.ps_active;
		pair->code_hashes = {ShaderHashAt(pair->vs), state.ps_active ? ShaderHashAt(pair->ps) : 0};
		// (Compiled by a worker: a later draw's store takes the programs and makes its state's pipeline.)
		pair->pending = pipelines.CompileTableGraphicsAsync(vertex_params, vertex_info,
		                                                    state.ps_active ? &pixel_params : nullptr,
		                                                    state.ps_active ? &pixel_info : nullptr, *vertex_specialization,
		                                                    state.ps_active ? pixel_specialization : nullptr);
		return;
	}
	auto variant = std::ranges::find(pair->variants, table.store_state, &TableVariant::state);
	if (variant == pair->variants.end()) variant = pair->variants.insert(variant, {.state = table.store_state});
	if (static_key != pair->static_key || state.ps_active != pair->has_pixel) {
		// Another specialization (the pixel program's export of another draw state's targets): this state's draws
		// take the normal path.
		variant->refused = true;
		return;
	}
	// The pipeline: the normal draw's stage input with the table programs, native bindings.
	ShaderVertexInputInfo table_vertex = state.vertex_info[0];
	table_vertex.stage                 = {.program = pair->programs[0], .resources = &g_table_no_resources};
	ShaderPixelInputInfo table_pixel   = state.ps_input_info;
	table_pixel.stage                  = {.program = pair->programs[1], .resources = &g_table_no_resources};
	PipelineCache::GraphicsPrograms programs {};
	programs.vertex[0] = pair->tables[0]->handle;
	if (state.ps_active) programs.pixel = pair->tables[1]->handle;
	const auto* pipeline = pipelines.GetGraphicsPipeline(
	    std::span {state.colors, state.color_count}, *state.depth, std::span {&table_vertex, 1}, buffer,
	    state.ps_active ? &table_pixel : nullptr, topology, primitive_restart_enable, programs,
	    NativeXprAsyncPipelines(), true);
	if (pipeline != nullptr && (!pipeline->native_bindings || pipeline->uses_push_descriptors)) {
		variant->refused = true;
		return;
	}
	variant->pipeline       = pipeline; // (null while a worker creates it: a later draw's store takes it)
	variant->fetch_embedded = state.vertex_info[0].fetch_embedded;
	NativeXprCaptureState(state, rendering, table.store_state, dynamic_state);
}
