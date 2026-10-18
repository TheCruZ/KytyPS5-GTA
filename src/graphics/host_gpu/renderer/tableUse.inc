// Table dispatches and draws (tableDispatch.inc, tableDraw.inc): the evaluation of a table program's use, on the
// execution thread or ahead of it on the resolve thread.
//
// Included by renderCompute.cpp and renderDraw.cpp.

namespace {

// The table caches: the execution thread changes them, the resolve thread reads them.
class TableLockGuard {
public:
	explicit TableLockGuard(std::atomic_flag& flag): m_flag(flag) {
		while (m_flag.test_and_set(std::memory_order_acquire)) {
			while (m_flag.test(std::memory_order_relaxed)) {
				Common::SpinPause();
			}
		}
	}
	~TableLockGuard() { m_flag.clear(std::memory_order_release); }
	TableLockGuard(const TableLockGuard&)            = delete;
	TableLockGuard& operator=(const TableLockGuard&) = delete;

private:
	std::atomic_flag& m_flag;
};

// A dword of guest memory as the use would read it now; false where the normal path must read it (its pages hold
// bytes the GPU wrote, or nothing is mapped there). Ahead of execution (`reads`), through the resolution's log, which
// the execution thread checks; memory the GPU may write fails there.
bool TableReadWord(uint64_t address, uint32_t& word, GuestReadLog* reads) {
	if ((address & 3u) != 0 || !GuestRange {address, 4}.Valid()) return false;
	if (Libs::LibKernel::Memory::MayFaultOnGpuRead(address, 4)) {
		return reads == nullptr && Libs::LibKernel::Memory::TryReadAroundGpuWrites(address, &word, 4);
	}
	if (reads != nullptr) return reads->Read(address, &word, 4, false);
	return Libs::LibKernel::Memory::TryReadBacking(address, &word, 4);
}

// The slots a table program's plan marks for the renderer, in order (a slot's address depends on lower slots and user
// data only), and the descriptors they give, into `snapshot`.
bool TableEvaluate(const ShaderRecompiler::IR::CompiledShaderInfo& reference, std::span<const uint32_t> user_data,
                   ShaderRecompiler::IR::ResourceSnapshot& snapshot, GuestReadLog* reads) {
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
		if (!TableReadWord(address, slots[index], reads)) return false;
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

} // namespace
