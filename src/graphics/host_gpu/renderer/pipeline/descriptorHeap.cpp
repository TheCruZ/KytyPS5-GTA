#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <algorithm>
#include <mutex>
#include <string>

namespace Libs::Graphics {
namespace {

constexpr uint32_t   DescriptorHeapCount = 1024;
constexpr std::array DescriptorPoolSizes = {
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, DescriptorHeap::MaxSampledImages},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 1024},
};

// Sets of a layout that does not fit a default pool that its own pool holds.
constexpr uint32_t LargeSetsPerPool = 4;

std::mutex                                                                       g_layout_mutex;
std::unordered_map<vk::DescriptorSetLayout, std::vector<vk::DescriptorPoolSize>> g_layout_sizes;

// The default pool sizes, raised to hold LargeSetsPerPool sets of the layout.
std::vector<vk::DescriptorPoolSize> LargePoolSizes(vk::DescriptorSetLayout layout) {
	std::vector<vk::DescriptorPoolSize> sizes(DescriptorPoolSizes.begin(),
	                                          DescriptorPoolSizes.end());
	std::lock_guard lock(g_layout_mutex);
	const auto      found = g_layout_sizes.find(layout);
	if (found == g_layout_sizes.end()) {
		return sizes;
	}
	for (const auto& needed: found->second) {
		const auto count = needed.descriptorCount * LargeSetsPerPool;
		auto       size  = std::find_if(sizes.begin(), sizes.end(),
		                                [&](const auto& size) { return size.type == needed.type; });
		if (size == sizes.end()) {
			sizes.push_back({needed.type, count});
		} else {
			size->descriptorCount = std::max(size->descriptorCount, count);
		}
	}
	return sizes;
}

} // namespace

void DescriptorHeap::RegisterLayout(vk::DescriptorSetLayout                         layout,
                                    std::span<const vk::DescriptorSetLayoutBinding> bindings) {
	std::vector<vk::DescriptorPoolSize> sizes;
	for (const auto& binding: bindings) {
		auto size = std::find_if(sizes.begin(), sizes.end(), [&](const auto& size) {
			return size.type == binding.descriptorType;
		});
		if (size == sizes.end()) {
			sizes.push_back({binding.descriptorType, binding.descriptorCount});
		} else {
			size->descriptorCount += binding.descriptorCount;
		}
	}
	std::lock_guard lock(g_layout_mutex);
	g_layout_sizes[layout] = std::move(sizes);
}

DescriptorHeap::DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore)
    : m_graphics(graphics), m_master_semaphore(master_semaphore) {
	CreateDescriptorPool(DescriptorPoolSizes);
}

DescriptorHeap::~DescriptorHeap() {
	m_graphics.device.destroyDescriptorPool(m_current_pool, nullptr);
	for (const auto& [pool, tick]: m_pending_pools) {
		m_master_semaphore.Wait(tick);
		m_graphics.device.destroyDescriptorPool(pool, nullptr);
	}
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout layout) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(layout == nullptr);

	auto& batch = m_sets[layout];
	if (batch.size != 0) {
		return batch.sets[--batch.size];
	}
	if (Allocate(layout, batch)) {
		return batch.sets[--batch.size];
	}

	RetireCurrentPool();
	if (!m_master_semaphore.IsFree(m_pending_pools.front().second)) {
		m_master_semaphore.Refresh();
	}
	if (const auto& [pool, tick] = m_pending_pools.front(); m_master_semaphore.IsFree(tick)) {
		m_current_pool = pool;
		m_pending_pools.pop_front();
		EXIT_IF(m_graphics.device.resetDescriptorPool(m_current_pool, {}) != vk::Result::eSuccess);
	} else {
		CreateDescriptorPool(DescriptorPoolSizes);
	}

	m_sets.clear();
	auto& fresh_batch = m_sets[layout];
	if (Allocate(layout, fresh_batch)) {
		return fresh_batch.sets[--fresh_batch.size];
	}

	// The set does not fit an empty default pool (drivers such as AMD's enforce the pool sizes):
	// give it a pool that also holds a few of its sets.
	const auto sizes = LargePoolSizes(layout);
	RetireCurrentPool();
	CreateDescriptorPool(sizes);
	m_sets.clear();
	auto& large_batch = m_sets[layout];
	if (!Allocate(layout, large_batch)) {
		std::string counts;
		for (const auto& size: sizes) {
			counts += " " + vk::to_string(size.type) + "=" + std::to_string(size.descriptorCount);
		}
		EXIT("descriptor set does not fit a pool sized for it:%s\n", counts.c_str());
	}
	return large_batch.sets[--large_batch.size];
}

void DescriptorHeap::RetireCurrentPool() {
	m_pending_pools.emplace_back(m_current_pool, m_master_semaphore.CurrentTick());
}

bool DescriptorHeap::Allocate(vk::DescriptorSetLayout layout, Batch& batch) {
	std::array<vk::DescriptorSetLayout, DescriptorSetBatch> layouts;
	layouts.fill(layout);

	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool = m_current_pool;
	allocate.pSetLayouts    = layouts.data();

	for (;;) {
		allocate.descriptorSetCount = batch.allocation;
		const auto result = m_graphics.device.allocateDescriptorSets(&allocate, batch.sets.data());
		if (result == vk::Result::eSuccess) {
			batch.size = batch.allocation;
			return true;
		}
		EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory &&
		        result != vk::Result::eErrorFragmentedPool);
		if (batch.allocation == 1) {
			return false;
		}
		batch.allocation /= 2;
	}
}

void DescriptorHeap::CreateDescriptorPool(std::span<const vk::DescriptorPoolSize> sizes) {
	vk::DescriptorPoolCreateInfo create {};
	create.maxSets       = DescriptorHeapCount;
	create.poolSizeCount = static_cast<uint32_t>(sizes.size());
	create.pPoolSizes    = sizes.data();
	EXIT_IF(m_graphics.device.createDescriptorPool(&create, nullptr, &m_current_pool) !=
	        vk::Result::eSuccess);
}

} // namespace Libs::Graphics
