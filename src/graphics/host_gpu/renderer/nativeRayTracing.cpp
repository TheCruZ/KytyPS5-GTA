#include "graphics/host_gpu/renderer/nativeRayTracing.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "gpu_tiler_shaders/hle_blas_refit_spv.h"
#include "gpu_tiler_shaders/hle_tlas_spv.h"
#include "gpu_tiler_shaders/rt_blas_walk_spv.h"
#include "gpu_tiler_shaders/rt_tlas_walk_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/shader/recompiler/frontend/decode/RayTracingPatch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

namespace {

constexpr uint32_t QueueCapacity    = 1u << 16u;
constexpr uint32_t InstanceCapacity = 1u << 16u;
constexpr uint32_t TableEntries     = 1u << 17u;
constexpr uint32_t SlotCapacity     = 1u << 15u;
constexpr uint32_t MissingCapacity  = 256;
constexpr uint32_t MaxJobs          = 256;
constexpr uint32_t ChangedCapacity  = 1024;
constexpr uint32_t PoolCapacity     = 1u << 23u;
constexpr uint64_t BuildScratchSize = uint64_t {64} << 20u;
constexpr uint32_t MaxRetries       = 8;
// Preparations a changed slot stays rebuilt before each use (about 30 frames).
constexpr uint32_t DynamicPrepares  = 120;

constexpr uint32_t StateSize      = 96;
constexpr uint32_t MissingStride  = 48;
constexpr uint32_t JobStride      = 56;
constexpr uint32_t SlotStride     = 64;
constexpr uint32_t TableStride    = 16;
constexpr uint32_t ContextStride  = 64;
constexpr uint32_t InstanceStride = 64;

constexpr uint32_t ModeCount  = 0;
constexpr uint32_t ModeBuild  = 1;
constexpr uint32_t ModeVerify = 2;

// RtState (rt_common.inc).
struct RtState {
	uint64_t header_va;
	uint64_t instance_base;
	uint32_t node_limit;
	uint32_t instance_count;
	uint32_t box_count;
	uint32_t overflow;
	uint32_t missing_count;
	uint32_t dispatch[3];
	uint32_t unknown_blas;
	uint32_t inactive;
	uint32_t max_depth;
	uint32_t verify_count;
	uint32_t verify[3];
	uint32_t changed_count;
	uint32_t pool_cursor;
	uint32_t pool_overflow;
	uint32_t pad[2];
};
static_assert(sizeof(RtState) == StateSize);
constexpr uint32_t StateDispatch = offsetof(RtState, dispatch);
constexpr uint32_t StateVerify   = offsetof(RtState, verify);

// The readback area: the state, the missing BLASes, the build jobs with their results and the
// slots whose BLAS changed.
constexpr uint64_t ReadbackMissing = StateSize;
constexpr uint64_t ReadbackJobs    = ReadbackMissing + uint64_t {MissingCapacity} * MissingStride;
constexpr uint64_t ReadbackChanged = ReadbackJobs + uint64_t {MaxJobs} * JobStride;
constexpr uint64_t ReadbackSize    = ReadbackChanged + uint64_t {ChangedCapacity} * 4;

struct TlasPush {
	uint64_t user_ptr_va;
	uint64_t state;
	uint64_t queue;
	uint64_t instance_nodes;
	uint64_t instances;
	uint64_t table;
	uint64_t slots;
	uint64_t missing;
	uint64_t context;
	uint64_t tlas_address;
	uint64_t verify_list;
	uint32_t header_offset;
	uint32_t root;
	uint32_t queue_capacity;
	uint32_t instance_capacity;
	uint32_t table_mask;
	uint32_t missing_capacity;
	uint32_t prepare_id;
	uint32_t verify;
	uint32_t flip_parity;
	uint32_t pad;
};

struct HleTlasPush {
	uint64_t input_va;
	uint64_t header_va;
	uint64_t scratch;
	uint32_t count;
	uint32_t stride;
};

// The instance and node bounds of hle_tlas.comp (32 bytes each).
constexpr uint64_t HleTlasScratchSize = uint64_t {4096 + 1365} * 32;

struct HleRefitPush {
	uint64_t jobs;
	uint64_t pool;
	uint64_t state;
	uint32_t mode;
	uint32_t job_count;
	uint32_t pool_capacity;
	uint32_t pad;
};

constexpr uint32_t HleRefitGeometryStride = 96;
constexpr uint32_t HleRefitBlasStride     = 16;
constexpr uint64_t HleRefitJobsSize       = uint64_t {4} << 20u;
constexpr uint32_t HleRefitPoolCapacity   = 1u << 22u;
constexpr uint32_t HleRefitStateSize      = 16;

struct BlasPush {
	uint64_t jobs;
	uint64_t pool;
	uint64_t slots;
	uint64_t state;
	uint64_t changed;
	uint32_t mode;
	uint32_t pool_capacity;
	uint32_t job_count;
	uint32_t changed_capacity;
};

struct Job {
	uint64_t blas_va;
	uint32_t root;
	uint32_t slot;
	uint64_t vertices0;
	uint64_t vertices1;
	uint64_t entries;
	uint32_t capacity0;
	uint32_t capacity1;
	uint32_t result;
	uint32_t pad;
};
static_assert(sizeof(Job) == JobStride);

struct SlotInfo {
	uint64_t as_address;
	uint64_t entry_table;
	uint32_t capacity0;
	uint32_t capacity1;
	uint32_t count0;
	uint32_t count1;
	uint32_t signature;
	uint32_t state;
	uint64_t blas_va;
	uint32_t root;
	uint32_t stamp;
	uint32_t leaf_count;
	uint32_t pad;
};
static_assert(sizeof(SlotInfo) == SlotStride);

struct MissingEntry {
	uint64_t blas_va;
	uint32_t root;
	uint32_t table_index;
	uint32_t leaf_count;
	uint32_t box_count;
	uint32_t opaque_count;
	uint32_t alpha_count;
	uint32_t node_limit;
	uint32_t flags;
	uint32_t pad[2];
};
static_assert(sizeof(MissingEntry) == MissingStride);

constexpr vk::BufferUsageFlags DataFlags =
    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
    vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc |
    vk::BufferUsageFlagBits::eIndirectBuffer;
constexpr vk::BufferUsageFlags AsFlags =
    DataFlags | vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;

constexpr vk::PipelineStageFlags2 AsBuildStage =
    vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR;

void MemoryBarrier(const CommandRecorder& command, vk::PipelineStageFlags2 src_stages,
                   vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stages,
                   vk::AccessFlags2 dst_access) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = src_stages;
	barrier.srcAccessMask = src_access;
	barrier.dstStageMask  = dst_stages;
	barrier.dstAccessMask = dst_access;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

// vkCmdUpdateBuffer of a small array (at most 64 KiB, dword multiples).
void UpdateBuffer(const CommandRecorder& command, vk::Buffer buffer, uint64_t offset,
                  const void* data, uint32_t size) {
	EXIT_IF(size == 0 || size > 65536 || (size & 3u) != 0 || (offset & 3u) != 0);
	const auto* copy = command.CopyArray(static_cast<const uint8_t*>(data), size);
	command.Custom([buffer, offset, copy, size](vk::CommandBuffer c) {
		c.updateBuffer(buffer, offset, size, copy);
	});
}

// Non-indexed triangles (3 floats per vertex): geometry 0 opaque, geometry 1 alpha tested (the
// guest alpha test runs once per candidate).
vk::AccelerationStructureGeometryKHR TriangleGeometry(uint64_t address, uint32_t capacity,
                                                      bool opaque) {
	vk::AccelerationStructureGeometryKHR geometry {};
	geometry.geometryType = vk::GeometryTypeKHR::eTriangles;
	geometry.flags        = opaque ? vk::GeometryFlagBitsKHR::eOpaque
	                               : vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation;
	auto& triangles                    = geometry.geometry.triangles;
	triangles.sType                    = vk::StructureType::eAccelerationStructureGeometryTrianglesDataKHR;
	triangles.vertexFormat             = vk::Format::eR32G32B32Sfloat;
	triangles.vertexData.deviceAddress = address;
	triangles.vertexStride             = 12;
	triangles.maxVertex                = capacity * 3 - 1;
	triangles.indexType                = vk::IndexType::eNoneKHR;
	return geometry;
}

} // namespace

namespace {
// Set when the device is created, before any shader compiles.
std::atomic<bool> g_device_support {false};
} // namespace

void NativeRayTracing::SetDeviceSupport(bool supported) noexcept {
	g_device_support.store(supported, std::memory_order_relaxed);
}

bool NativeRayTracing::PatchShaders() noexcept {
	return g_device_support.load(std::memory_order_relaxed);
}

const NativeRayTracing::TlasSource* NativeRayTracing::FindTlasSource(uint64_t shader_hash) noexcept {
	// Only the shaders the recompiler patches trace through the host structures.
	static thread_local TlasSource source;
	const auto* patch = ShaderRecompiler::Decoder::FindRayTracingPatch(shader_hash);
	if (patch == nullptr || !PatchShaders()) {
		return nullptr;
	}
	source = {patch->user_sgpr, patch->header_offset, patch->root};
	return &source;
}

NativeRayTracing::Allocation NativeRayTracing::Arena::Allocate(uint64_t size, uint64_t alignment) {
	size = (size + 255u) & ~uint64_t {255};
	if (m_chunks.empty() ||
	    ((m_used + alignment - 1) & ~(alignment - 1)) + size > m_chunks.back()->Size()) {
		m_chunks.push_back(std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal,
		                                            0, AsFlags, std::max(ChunkSize, size)));
		m_used = 0;
	}
	auto&      chunk  = *m_chunks.back();
	const auto offset = (m_used + alignment - 1) & ~(alignment - 1);
	m_used            = offset + size;
	m_total += size;
	return {&chunk, offset, chunk.BufferDeviceAddress() + offset};
}

NativeRayTracing::NativeRayTracing(GraphicContext& graphics, CommandScheduler& scheduler,
                                   BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_buffer_cache(buffer_cache),
      m_arena(graphics, scheduler) {
	m_enabled = graphics.native_ray_tracing_enabled;
	if (!m_enabled) {
		return;
	}
	CreatePipelines();
	CreateBuffers();
	CreateTlas(InstanceCapacity);
	LOGF("NativeRayTracing: enabled\n");
}

NativeRayTracing::~NativeRayTracing() {
	if (!m_enabled) {
		return;
	}
	auto device = m_graphics.device;
	for (auto& slot: m_slot_list) {
		if (slot->handle) {
			device.destroyAccelerationStructureKHR(slot->handle);
		}
	}
	if (m_tlas) {
		device.destroyAccelerationStructureKHR(m_tlas);
	}
	device.destroyPipeline(m_tlas_pipeline);
	device.destroyPipeline(m_blas_pipeline);
	device.destroyPipelineLayout(m_tlas_layout);
	device.destroyPipelineLayout(m_blas_layout);
	device.destroyPipeline(m_hle_tlas_pipeline);
	device.destroyPipelineLayout(m_hle_tlas_layout);
	device.destroyPipeline(m_hle_refit_pipeline);
	device.destroyPipelineLayout(m_hle_refit_layout);
	device.destroyDescriptorSetLayout(m_descriptor_layout);
}

void NativeRayTracing::CreatePipelines() {
	auto device = m_graphics.device;
	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1,
	                                              vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	RequireVulkanSuccess(device.createDescriptorSetLayout(&layout_info, nullptr, &m_descriptor_layout),
	                     "create native ray tracing descriptor layout");
	const auto make = [&](std::span<const uint32_t> code, uint32_t push_size,
	                      vk::PipelineLayout& layout, vk::Pipeline& pipeline, const char* name) {
		const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, push_size};
		vk::PipelineLayoutCreateInfo info {};
		info.setLayoutCount         = 1;
		info.pSetLayouts            = &m_descriptor_layout;
		info.pushConstantRangeCount = 1;
		info.pPushConstantRanges    = &push;
		RequireVulkanSuccess(device.createPipelineLayout(&info, nullptr, &layout),
		                     "create native ray tracing pipeline layout");
		const auto module = CompileSPV(code, device);
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage  = vk::ShaderStageFlagBits::eCompute;
		stage.module = module;
		stage.pName  = "main";
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage  = stage;
		pipeline_info.layout = layout;
		const auto result = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline);
		device.destroyShaderModule(module, nullptr);
		RequireVulkanSuccess(result, "create native ray tracing pipeline");
		SetVulkanObjectNameF(device, pipeline, "%s", name);
	};
	make(RT_TLAS_WALK_SPV, sizeof(TlasPush), m_tlas_layout, m_tlas_pipeline, "RT TLAS walk");
	make(RT_BLAS_WALK_SPV, sizeof(BlasPush), m_blas_layout, m_blas_pipeline, "RT BLAS walk");
	make(HLE_TLAS_SPV, sizeof(HleTlasPush), m_hle_tlas_layout, m_hle_tlas_pipeline, "HLE TLAS build");
	make(HLE_BLAS_REFIT_SPV, sizeof(HleRefitPush), m_hle_refit_layout, m_hle_refit_pipeline,
	     "HLE BLAS refit");
}

void NativeRayTracing::CreateBuffers() {
	const auto make = [&](uint64_t size, vk::BufferUsageFlags flags, const char* name) {
		auto buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
		                                       flags, size);
		SetVulkanObjectNameF(m_graphics.device, buffer->Handle(), "%s", name);
		return buffer;
	};
	m_state          = make(StateSize, DataFlags, "RT state");
	m_queue          = make(uint64_t {QueueCapacity} * 4, DataFlags, "RT queue");
	m_instance_nodes = make(uint64_t {InstanceCapacity} * 4, DataFlags, "RT instance nodes");
	m_instances      = make(uint64_t {InstanceCapacity} * InstanceStride, AsFlags, "RT instances");
	m_table          = make(uint64_t {TableEntries} * TableStride, DataFlags, "RT BLAS table");
	m_slots          = make(uint64_t {SlotCapacity} * SlotStride, DataFlags, "RT slots");
	m_missing        = make(uint64_t {MissingCapacity} * MissingStride, DataFlags, "RT missing");
	m_context        = make(uint64_t {MaxContexts} * ContextStride, DataFlags, "RT context");
	m_pool           = make(uint64_t {PoolCapacity} * 4, DataFlags, "RT walk pool");
	m_verify_list    = make(uint64_t {SlotCapacity} * 4, DataFlags, "RT verify list");
	m_changed        = make(uint64_t {ChangedCapacity} * 4, DataFlags, "RT changed");
	m_jobs           = make(uint64_t {MaxJobs} * JobStride, DataFlags, "RT jobs");
	m_build_scratch  = make(BuildScratchSize, AsFlags, "RT build scratch");
	m_hle_tlas_scratch = make(HleTlasScratchSize, DataFlags, "HLE TLAS bounds");
	m_hle_jobs       = make(HleRefitJobsSize, DataFlags, "HLE refit jobs");
	m_hle_pool       = make(uint64_t {HleRefitPoolCapacity} * 4, DataFlags, "HLE refit pool");
	m_hle_state      = make(HleRefitStateSize, DataFlags, "HLE refit state");
	for (auto& readback: m_readbacks) {
		readback.buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                           AllFlags, ReadbackSize);
	}
	m_context_address = m_context->BufferDeviceAddress();
}

void NativeRayTracing::CreateTlas(uint32_t capacity) {
	auto device = m_graphics.device;
	vk::AccelerationStructureGeometryKHR geometry {};
	geometry.geometryType                       = vk::GeometryTypeKHR::eInstances;
	geometry.geometry.instances.sType           = vk::StructureType::eAccelerationStructureGeometryInstancesDataKHR;
	geometry.geometry.instances.arrayOfPointers = VK_FALSE;
	vk::AccelerationStructureBuildGeometryInfoKHR info {};
	info.type          = vk::AccelerationStructureTypeKHR::eTopLevel;
	info.flags         = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;
	info.mode          = vk::BuildAccelerationStructureModeKHR::eBuild;
	info.geometryCount = 1;
	info.pGeometries   = &geometry;
	vk::AccelerationStructureBuildSizesInfoKHR sizes {};
	device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, &info,
	                                             &capacity, &sizes);
	m_tlas_storage = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
	                                          AsFlags, sizes.accelerationStructureSize);
	vk::AccelerationStructureCreateInfoKHR create {};
	create.buffer = m_tlas_storage->Handle();
	create.size   = sizes.accelerationStructureSize;
	create.type   = vk::AccelerationStructureTypeKHR::eTopLevel;
	RequireVulkanSuccess(device.createAccelerationStructureKHR(&create, nullptr, &m_tlas),
	                     "create native TLAS");
	vk::AccelerationStructureDeviceAddressInfoKHR address_info {};
	address_info.accelerationStructure = m_tlas;
	m_tlas_address      = device.getAccelerationStructureAddressKHR(address_info);
	m_tlas_scratch_size = sizes.buildScratchSize;
	m_tlas_capacity     = capacity;
	EXIT_IF(m_tlas_scratch_size > BuildScratchSize);
}

void NativeRayTracing::AllocateSlot(Slot& slot, uint32_t opaque, uint32_t alpha) {
	auto device    = m_graphics.device;
	slot.capacity0 = opaque + opaque / 8 + 16;
	slot.capacity1 = alpha + alpha / 8 + 16;
	const std::array geometries {TriangleGeometry(0, slot.capacity0, true),
	                             TriangleGeometry(0, slot.capacity1, false)};
	const std::array counts {slot.capacity0, slot.capacity1};
	vk::AccelerationStructureBuildGeometryInfoKHR info {};
	info.type          = vk::AccelerationStructureTypeKHR::eBottomLevel;
	info.flags         = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
	info.mode          = vk::BuildAccelerationStructureModeKHR::eBuild;
	info.geometryCount = static_cast<uint32_t>(geometries.size());
	info.pGeometries   = geometries.data();
	vk::AccelerationStructureBuildSizesInfoKHR sizes {};
	device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, &info,
	                                             counts.data(), &sizes);
	// The arena never frees: a grown slot leaves its old memory (and the GPU may still use it).
	slot.storage       = m_arena.Allocate(sizes.accelerationStructureSize, 256);
	slot.vertices0     = m_arena.Allocate(uint64_t {slot.capacity0} * 36, 256);
	slot.vertices1     = m_arena.Allocate(uint64_t {slot.capacity1} * 36, 256);
	slot.entries       = m_arena.Allocate(uint64_t {slot.capacity0 + slot.capacity1} * 16, 256);
	slot.build_scratch = sizes.buildScratchSize;
	vk::AccelerationStructureCreateInfoKHR create {};
	create.buffer = slot.storage.buffer->Handle();
	create.offset = slot.storage.offset;
	create.size   = sizes.accelerationStructureSize;
	create.type   = vk::AccelerationStructureTypeKHR::eBottomLevel;
	RequireVulkanSuccess(device.createAccelerationStructureKHR(&create, nullptr, &slot.handle),
	                     "create native BLAS");
	vk::AccelerationStructureDeviceAddressInfoKHR address_info {};
	address_info.accelerationStructure = slot.handle;
	slot.as_address = device.getAccelerationStructureAddressKHR(address_info);
}

void NativeRayTracing::CreateSlot(const PendingSlot& pending) {
	if (m_slot_list.size() >= SlotCapacity) {
		return;
	}
	auto slot         = std::make_unique<Slot>();
	slot->blas_va     = pending.blas_va;
	slot->root        = pending.root;
	slot->table_index = pending.table_index;
	AllocateSlot(*slot, pending.opaque, pending.alpha);
	slot->needs_build = true;
	const auto index  = static_cast<uint32_t>(m_slot_list.size());
	m_build_queue.push_back(index);
	m_slot_list.push_back(std::move(slot));
}

void NativeRayTracing::RecordBuilds(const CommandRecorder& command) {
	auto&                 readback = m_readbacks[m_readback_index];
	std::vector<Job>      jobs;
	std::vector<uint32_t> slots;
	uint64_t              scratch   = 0;
	constexpr uint64_t    Alignment = 256;
	{
		Common::LockGuard lock(m_mutex);
		while (!m_build_queue.empty() && jobs.size() < MaxJobs) {
			const auto index = m_build_queue.back();
			auto&      slot  = *m_slot_list[index];
			const auto size  = (slot.build_scratch + Alignment - 1) & ~(Alignment - 1);
			if (scratch + size > BuildScratchSize) {
				break;
			}
			m_build_queue.pop_back();
			if (!slot.needs_build) {
				continue;
			}
			slot.needs_build = false;
			scratch += size;
			jobs.push_back({slot.blas_va, slot.root, index, slot.vertices0.address,
			                slot.vertices1.address, slot.entries.address, slot.capacity0,
			                slot.capacity1, 0, 0});
			slots.push_back(index);
		}
	}
	if (jobs.empty()) {
		return;
	}
	// The slot infos (state 0 until built) and table entries (slot + 1) of the slots.
	for (const auto index: slots) {
		const auto&    slot = *m_slot_list[index];
		const SlotInfo info {.as_address  = slot.as_address,
		                     .entry_table = slot.entries.address,
		                     .capacity0   = slot.capacity0,
		                     .capacity1   = slot.capacity1,
		                     .blas_va     = slot.blas_va,
		                     .root        = slot.root};
		UpdateBuffer(command, m_slots->Handle(), uint64_t {index} * SlotStride, &info, sizeof(info));
		const uint32_t encoded = index + 1;
		UpdateBuffer(command, m_table->Handle(), uint64_t {slot.table_index} * TableStride + 8,
		             &encoded, sizeof(encoded));
	}
	UpdateBuffer(command, m_jobs->Handle(), 0, jobs.data(),
	             static_cast<uint32_t>(jobs.size() * JobStride));
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	              vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	const BlasPush push {m_jobs->BufferDeviceAddress(),    m_pool->BufferDeviceAddress(),
	                     m_slots->BufferDeviceAddress(),   m_state->BufferDeviceAddress(),
	                     m_changed->BufferDeviceAddress(), ModeBuild,
	                     PoolCapacity,                     static_cast<uint32_t>(jobs.size()),
	                     ChangedCapacity};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_blas_pipeline);
	command.pushConstants(m_blas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch(static_cast<uint32_t>(jobs.size()), 1, 1);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eShaderWrite, AsBuildStage | vk::PipelineStageFlagBits2::eTransfer,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead);

	// One build per slot, all in one command (distinct scratch ranges).
	struct BuildRecord {
		vk::AccelerationStructureKHR handle;
		uint64_t                     vertices0;
		uint64_t                     vertices1;
		uint64_t                     scratch;
		uint32_t                     capacity0;
		uint32_t                     capacity1;
	};
	std::vector<BuildRecord> records;
	uint64_t                 scratch_offset = 0;
	const auto               scratch_base   = m_build_scratch->BufferDeviceAddress();
	for (const auto index: slots) {
		const auto& slot = *m_slot_list[index];
		records.push_back({slot.handle, slot.vertices0.address, slot.vertices1.address,
		                   scratch_base + scratch_offset, slot.capacity0, slot.capacity1});
		scratch_offset += (slot.build_scratch + Alignment - 1) & ~(Alignment - 1);
	}
	const auto* copy  = command.CopyArray(records.data(), records.size());
	const auto  count = static_cast<uint32_t>(records.size());
	command.Custom([copy, count](vk::CommandBuffer c) {
		std::vector<vk::AccelerationStructureGeometryKHR>              geometries(count * 2);
		std::vector<vk::AccelerationStructureBuildGeometryInfoKHR>     infos(count);
		std::vector<vk::AccelerationStructureBuildRangeInfoKHR>        ranges(count * 2);
		std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> range_pointers(count);
		for (uint32_t i = 0; i < count; i++) {
			geometries[i * 2]     = TriangleGeometry(copy[i].vertices0, copy[i].capacity0, true);
			geometries[i * 2 + 1] = TriangleGeometry(copy[i].vertices1, copy[i].capacity1, false);
			auto& info                     = infos[i];
			info.type                      = vk::AccelerationStructureTypeKHR::eBottomLevel;
			info.flags                     = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
			info.mode                      = vk::BuildAccelerationStructureModeKHR::eBuild;
			info.dstAccelerationStructure  = copy[i].handle;
			info.geometryCount             = 2;
			info.pGeometries               = &geometries[i * 2];
			info.scratchData.deviceAddress = copy[i].scratch;
			ranges[i * 2].primitiveCount     = copy[i].capacity0;
			ranges[i * 2 + 1].primitiveCount = copy[i].capacity1;
			range_pointers[i]                = &ranges[i * 2];
		}
		c.buildAccelerationStructuresKHR(count, infos.data(), range_pointers.data());
	});
	MemoryBarrier(command, AsBuildStage, vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
	              AsBuildStage | vk::PipelineStageFlagBits2::eComputeShader |
	                  vk::PipelineStageFlagBits2::eTransfer,
	              vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead |
	                  vk::AccessFlagBits2::eAccelerationStructureWriteKHR |
	                  vk::AccessFlagBits2::eTransferWrite);
	// Built: the TLAS walk may reference them.
	constexpr uint32_t built = 1;
	for (const auto index: slots) {
		UpdateBuffer(command, m_slots->Handle(), uint64_t {index} * SlotStride + offsetof(SlotInfo, state),
		             &built, sizeof(built));
		m_slot_list[index]->built       = true;
		m_slot_list[index]->built_frame = m_frame;
	}
	readback.build_count = static_cast<uint32_t>(slots.size());
	readback.build_slots = std::move(slots);
}

void NativeRayTracing::Prepare(const CommandRecorder& command, uint64_t user_pointer,
                               const TlasSource& source, uint64_t frame) {
	if (!m_enabled) {
		return;
	}
	auto& readback = m_readbacks[m_readback_index];
	if (readback.tick != 0) {
		// The ring is full: the oldest readback must complete first (rare).
		m_scheduler.Wait(readback.tick);
		m_scheduler.PopPendingOperations();
	}
	readback.build_count = 0;
	readback.build_slots.clear();
	const auto source_key = source.user_sgpr << 16u | source.header_offset;
	readback.source_key   = source_key;
	uint32_t              tlas_count = 0;
	std::vector<uint32_t> retry;
	{
		Common::LockGuard lock(m_mutex);
		for (const auto& pending: m_new_slots) {
			CreateSlot(pending);
		}
		m_new_slots.clear();
		for (const auto& [index, opaque, alpha]: m_grow) {
			auto& slot = *m_slot_list[index];
			// The old structure may still be in use by recorded work: it is leaked.
			AllocateSlot(slot, opaque, alpha);
			slot.needs_build = true;
			m_build_queue.push_back(index);
		}
		m_grow.clear();
		retry.swap(m_retry);
		// Slots the guest changed lately are rebuilt before each use: the verification finds a
		// change only after the TLAS of that preparation was built.
		std::erase_if(m_dynamic, [&](uint32_t index) {
			auto& slot = *m_slot_list[index];
			if (slot.dynamic_until <= m_prepare_id) {
				slot.dynamic_until = 0;
				return true;
			}
			if (!slot.needs_build && slot.built_frame != frame) {
				slot.needs_build = true;
				m_build_queue.push_back(index);
			}
			return false;
		});
		const auto last = m_last_instance_counts[source_key];
		tlas_count      = std::min(m_tlas_capacity, std::max(1024u, (last + last / 4 + 255u) &
		                                                           ~255u));
	}
	m_prepare_id++;
	m_frame = frame;

	// Earlier guest work wrote the BVHs; earlier ray queries read the TLAS this rebuilds.
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
	              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer |
	                  AsBuildStage,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
	                  vk::AccessFlagBits2::eTransferWrite |
	                  vk::AccessFlagBits2::eAccelerationStructureWriteKHR);
	if (!m_cleared) {
		// The buffers start clear (empty table, invalid contexts).
		m_cleared = true;
		for (const auto* buffer: {m_table.get(), m_slots.get(), m_context.get()}) {
			command.fillBuffer(buffer->Handle(), 0, buffer->Size(), 0);
		}
		// Patched shaders find the context through the BDA page table.
		UpdateBuffer(command, m_buffer_cache.GetBdaPageTableBuffer()->Handle(),
		             BufferCache::NATIVE_RT_CONTEXT_PAGE * sizeof(uint64_t), &m_context_address,
		             sizeof(m_context_address));
	}
	for (const auto index: retry) {
		constexpr uint64_t empty = 0;
		UpdateBuffer(command, m_table->Handle(), uint64_t {index} * TableStride, &empty, sizeof(empty));
	}
	command.fillBuffer(m_state->Handle(), 0, StateSize, 0);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
	                  vk::AccessFlagBits2::eTransferWrite);

	auto* page_table = m_buffer_cache.GetBdaPageTableBuffer();
	const vk::DescriptorBufferInfo page_info {page_table->Handle(), 0, page_table->Size()};
	vk::WriteDescriptorSet write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &page_info;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_blas_layout, 0, 1, &write);

	RecordBuilds(command);

	command.fillBuffer(m_instances->Handle(), 0, uint64_t {tlas_count} * InstanceStride, 0);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	              vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_tlas_layout, 0, 1, &write);
	const TlasPush tlas_push {
	    .user_ptr_va       = user_pointer,
	    .state             = m_state->BufferDeviceAddress(),
	    .queue             = m_queue->BufferDeviceAddress(),
	    .instance_nodes    = m_instance_nodes->BufferDeviceAddress(),
	    .instances         = m_instances->BufferDeviceAddress(),
	    .table             = m_table->BufferDeviceAddress(),
	    .slots             = m_slots->BufferDeviceAddress(),
	    .missing           = m_missing->BufferDeviceAddress(),
	    .context           = m_context_address,
	    .tlas_address      = m_tlas_address,
	    .verify_list       = m_verify_list->BufferDeviceAddress(),
	    .header_offset     = source.header_offset,
	    .root              = source.root,
	    .queue_capacity    = QueueCapacity,
	    .instance_capacity = tlas_count,
	    .table_mask        = TableEntries - 1,
	    .missing_capacity  = MissingCapacity,
	    .prepare_id        = m_prepare_id,
	    .verify            = 1u,
	    .flip_parity       = 0u,
	    .pad               = 0,
	};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_tlas_pipeline);
	command.pushConstants(m_tlas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(tlas_push),
	                      &tlas_push);
	command.dispatch(1, 1, 1);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
	              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eDrawIndirect |
	                  AsBuildStage,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
	                  vk::AccessFlagBits2::eIndirectCommandRead);

	// Counts the leaves of the BLASes the walk found missing, and checks the built ones it uses.
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_blas_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_blas_layout, 0, 1, &write);
	BlasPush blas_push {m_missing->BufferDeviceAddress(), m_pool->BufferDeviceAddress(),
	                    m_slots->BufferDeviceAddress(),   m_state->BufferDeviceAddress(),
	                    m_changed->BufferDeviceAddress(), ModeCount,
	                    PoolCapacity,                     0,
	                    ChangedCapacity};
	command.pushConstants(m_blas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(blas_push),
	                      &blas_push);
	command.dispatchIndirect(m_state->Handle(), StateDispatch);
	blas_push.jobs = m_verify_list->BufferDeviceAddress();
	blas_push.mode = ModeVerify;
	command.pushConstants(m_blas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(blas_push),
	                      &blas_push);
	command.dispatchIndirect(m_state->Handle(), StateVerify);

	// The TLAS over the walk's instances (inactive ones have a null BLAS).
	{
		const auto handle    = m_tlas;
		const auto instances = m_instances->BufferDeviceAddress();
		const auto scratch   = m_build_scratch->BufferDeviceAddress();
		command.Custom([handle, instances, scratch, tlas_count](vk::CommandBuffer c) {
			vk::AccelerationStructureGeometryKHR geometry {};
			geometry.geometryType = vk::GeometryTypeKHR::eInstances;
			geometry.geometry.instances.sType =
			    vk::StructureType::eAccelerationStructureGeometryInstancesDataKHR;
			geometry.geometry.instances.arrayOfPointers    = VK_FALSE;
			geometry.geometry.instances.data.deviceAddress = instances;
			vk::AccelerationStructureBuildGeometryInfoKHR info {};
			info.type                      = vk::AccelerationStructureTypeKHR::eTopLevel;
			info.flags                     = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;
			info.mode                      = vk::BuildAccelerationStructureModeKHR::eBuild;
			info.dstAccelerationStructure  = handle;
			info.geometryCount             = 1;
			info.pGeometries               = &geometry;
			info.scratchData.deviceAddress = scratch;
			vk::AccelerationStructureBuildRangeInfoKHR range {};
			range.primitiveCount      = tlas_count;
			const auto* range_pointer = &range;
			c.buildAccelerationStructuresKHR(1, &info, &range_pointer);
		});
	}
	MemoryBarrier(command, AsBuildStage | vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eAccelerationStructureWriteKHR | vk::AccessFlagBits2::eShaderWrite,
	              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	              vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead |
	                  vk::AccessFlagBits2::eTransferRead);

	// Reads back the state, the counted missing BLASes, the build results and the changed slots.
	const std::array<vk::BufferCopy, 4> regions {{
	    {0, 0, StateSize},
	    {0, ReadbackMissing, uint64_t {MissingCapacity} * MissingStride},
	    {0, ReadbackJobs, uint64_t {MaxJobs} * JobStride},
	    {0, ReadbackChanged, uint64_t {ChangedCapacity} * 4},
	}};
	command.copyBuffer(m_state->Handle(), readback.buffer->Handle(), 1, &regions[0]);
	command.copyBuffer(m_missing->Handle(), readback.buffer->Handle(), 1, &regions[1]);
	command.copyBuffer(m_jobs->Handle(), readback.buffer->Handle(), 1, &regions[2]);
	command.copyBuffer(m_changed->Handle(), readback.buffer->Handle(), 1, &regions[3]);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	              vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	readback.tick = m_scheduler.CurrentTick();
	m_scheduler.DeferHostOperation([this, &readback] { ProcessReadback(readback); });
	m_readback_index = (m_readback_index + 1) % m_readbacks.size();
}

void NativeRayTracing::ProcessReadback(Readback& readback) {
	readback.buffer->Invalidate(0, ReadbackSize);
	const auto* bytes = readback.buffer->Mapped().data();
	RtState     state {};
	std::memcpy(&state, bytes, sizeof(state));
	const auto               missing_count = std::min(state.missing_count, MissingCapacity);
	std::vector<PendingSlot> pending;
	std::vector<uint32_t>    retry;
	Common::LockGuard        lock(m_mutex);
	for (uint32_t i = 0; i < missing_count; i++) {
		MissingEntry entry {};
		std::memcpy(&entry, bytes + ReadbackMissing + uint64_t {i} * MissingStride, sizeof(entry));
		if (entry.flags != 0 || entry.leaf_count == 0) {
			// Not countable now (memory not written yet, a too large BLAS): forget the entry so
			// the walk reports it again, a few times.
			const auto key = entry.blas_va ^ (uint64_t {entry.root} << 48u);
			if (++m_retries[key] <= MaxRetries) {
				m_retry.push_back(entry.table_index);
			}
			continue;
		}
		m_new_slots.push_back({entry.blas_va, entry.root, entry.table_index, entry.leaf_count,
		                       entry.opaque_count, entry.alpha_count});
	}
	for (uint32_t i = 0; i < readback.build_count && i < readback.build_slots.size(); i++) {
		Job job {};
		std::memcpy(&job, bytes + ReadbackJobs + uint64_t {i} * JobStride, sizeof(job));
		if ((job.result & 1u) != 0) {
			// More triangles than the slot holds (the guest rebuilt the BLAS bigger): the job
			// came back with the counts.
			m_grow.push_back({readback.build_slots[i], job.capacity0, job.capacity1});
		}
	}
	const auto changed = std::min(state.changed_count, ChangedCapacity);
	for (uint32_t i = 0; i < changed; i++) {
		uint32_t index = 0;
		std::memcpy(&index, bytes + ReadbackChanged + uint64_t {i} * 4, sizeof(index));
		if (index >= m_slot_list.size()) {
			continue;
		}
		auto& slot = *m_slot_list[index];
		if (slot.dynamic_until == 0) {
			m_dynamic.push_back(index);
		}
		slot.dynamic_until = m_prepare_id + DynamicPrepares;
		if (!slot.needs_build) {
			slot.needs_build = true;
			m_build_queue.push_back(index);
		}
	}
	m_last_instance_counts[readback.source_key] = state.instance_count;
	readback.tick = 0;
}

uint32_t NativeRayTracing::HleTlasHeader(uint32_t count, std::array<uint32_t, 32>& header) noexcept {
	count          = std::min(count, HleTlasMaxInstances);
	uint32_t nodes = 0;
	for (uint32_t n = std::max((count + 3u) / 4u, 1u);; n = (n + 3u) / 4u) {
		nodes += n;
		if (n == 1u) {
			break;
		}
	}
	const uint32_t instance_offset = 256u + nodes * 128u;
	header                         = {};
	header[0]                      = 0x5f525350u; // "PSR_BVHL"
	header[1]                      = 0x4c485642u;
	header[2]                      = 0x5d1cba8cu;
	header[4]                      = instance_offset + count * 128u;
	header[6]                      = 0x00020024u;
	header[12]                     = instance_offset;
	header[22]                     = count + 4u;
	header[28]                     = count;
	return header[4];
}

void NativeRayTracing::RecordHleTlas(const CommandRecorder& command, uint64_t input_va,
                                     uint32_t count, uint32_t stride, uint64_t header_va) {
	if (!m_enabled || header_va == 0 || (count != 0 && input_va == 0)) {
		return;
	}
	// The game's CPU wrote the list; earlier GPU work wrote the BLASes the build reads and read
	// the TLAS it replaces.
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
	              vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	auto* page_table = m_buffer_cache.GetBdaPageTableBuffer();
	const vk::DescriptorBufferInfo page_info {page_table->Handle(), 0, page_table->Size()};
	vk::WriteDescriptorSet write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &page_info;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_hle_tlas_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_hle_tlas_layout, 0, 1, &write);
	const HleTlasPush push {input_va, header_va, m_hle_tlas_scratch->BufferDeviceAddress(), count,
	                        stride};
	command.pushConstants(m_hle_tlas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch(1, 1, 1);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
	              vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
}

void NativeRayTracing::RecordHleRefit(const CommandRecorder& command, std::span<const uint64_t> blases,
                                      std::span<const HleRefitGeometry> geometries) {
	if (!m_enabled || blases.empty()) {
		return;
	}
	const uint64_t geometry_bytes = uint64_t {HleRefitGeometryStride} * geometries.size();
	const uint64_t blas_offset    = (geometry_bytes + 255u) & ~uint64_t {255};
	const uint64_t blas_bytes     = uint64_t {HleRefitBlasStride} * blases.size();
	if (blas_offset + blas_bytes > HleRefitJobsSize) {
		return;
	}
	static_assert(sizeof(HleRefitGeometry) == HleRefitGeometryStride);
	// Earlier work read the jobs of the last refit and wrote what this one reads.
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
	              vk::PipelineStageFlagBits2::eTransfer | vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderRead |
	                  vk::AccessFlagBits2::eShaderWrite);
	const auto upload = [&](uint64_t offset, const uint8_t* data, uint64_t size) {
		for (uint64_t done = 0; done < size;) {
			const auto chunk = static_cast<uint32_t>(std::min<uint64_t>(size - done, 65536));
			UpdateBuffer(command, m_hle_jobs->Handle(), offset + done, data + done, chunk);
			done += chunk;
		}
	};
	if (geometry_bytes != 0) {
		upload(0, reinterpret_cast<const uint8_t*>(geometries.data()), geometry_bytes);
	}
	std::vector<uint64_t> blas_jobs(blases.size() * 2);
	for (size_t i = 0; i < blases.size(); i++) {
		blas_jobs[i * 2] = blases[i];
	}
	upload(blas_offset, reinterpret_cast<const uint8_t*>(blas_jobs.data()), blas_bytes);
	command.fillBuffer(m_hle_state->Handle(), 0, HleRefitStateSize, 0);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	              vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	auto* page_table = m_buffer_cache.GetBdaPageTableBuffer();
	const vk::DescriptorBufferInfo page_info {page_table->Handle(), 0, page_table->Size()};
	vk::WriteDescriptorSet write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &page_info;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_hle_refit_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_hle_refit_layout, 0, 1, &write);
	const auto jobs = m_hle_jobs->BufferDeviceAddress();
	HleRefitPush push {jobs,
	                   m_hle_pool->BufferDeviceAddress(),
	                   m_hle_state->BufferDeviceAddress(),
	                   0,
	                   static_cast<uint32_t>(geometries.size()),
	                   HleRefitPoolCapacity,
	                   0};
	if (!geometries.empty()) {
		command.pushConstants(m_hle_refit_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch(static_cast<uint32_t>(geometries.size()), 1, 1);
		// The boxes enclose the new vertices.
		MemoryBarrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
		              vk::PipelineStageFlagBits2::eComputeShader,
		              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	}
	push.jobs      = jobs + blas_offset;
	push.mode      = 1;
	push.job_count = static_cast<uint32_t>(blases.size());
	command.pushConstants(m_hle_refit_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch(static_cast<uint32_t>(blases.size()), 1, 1);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
	              vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
}

} // namespace Libs::Graphics
