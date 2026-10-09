#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_NATIVERAYTRACING_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_NATIVERAYTRACING_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

class BufferCache;
class CommandRecorder;
class CommandScheduler;
struct GraphicContext;

// Host acceleration structures for guest ray tracing (VK_KHR_acceleration_structure).
//
// GTA V builds AMD BVHs with its own compute shaders and traverses them in software with
// IMAGE_BVH_INTERSECT_RAY, one node at a time. Before a dispatch that traces through such a
// BVH, this class mirrors it on the GPU, without the CPU reading what the GPU wrote:
// - every leaf of the guest top-level BVH (an instance) becomes a host TLAS instance with the
//   inverse of the instance's world-to-object matrix and its mask;
// - every guest bottom-level BVH becomes a host BLAS of the triangles of its leaf lists: opaque
//   ones (geometry 0) and alpha tested ones (geometry 1), with a table from the primitive to the
//   guest list entry (its leaf pointer and primitive word).
// A shader then lets the hardware find opaque hits and runs the guest's own leaf code (triangle
// and alpha test) only for the alpha tested triangles a ray hits. BLASes the host has not seen are reported by the top-level walk, counted on the
// GPU and read back asynchronously; the host allocates them and builds them the next time.
class NativeRayTracing {
public:
	NativeRayTracing(GraphicContext& graphics, CommandScheduler& scheduler,
	                 BufferCache& buffer_cache);
	~NativeRayTracing();
	KYTY_CLASS_NO_COPY(NativeRayTracing);

	// Whether the device supports host ray tracing and it is enabled.
	[[nodiscard]] bool Enabled() const noexcept { return m_enabled; }
	// Whether shader compilation applies the traversal patches (set once at startup).
	[[nodiscard]] static bool PatchShaders() noexcept;
	// The device was created with the ray tracing extensions (before any shader compiles).
	static void SetDeviceSupport(bool supported) noexcept;

	// The guest TLAS of a known tracing shader: its header's address is the 64-bit value at
	// user data pointer (user SGPR pair `user_sgpr`) + `header_offset`; `root` its root node.
	struct TlasSource {
		uint32_t user_sgpr     = 0;
		uint32_t header_offset = 0;
		uint32_t root          = 0;
	};
	// The source of a shader that traces through a guest TLAS, or null.
	[[nodiscard]] static const TlasSource* FindTlasSource(uint64_t shader_hash) noexcept;

	// Records, before the dispatch, the walk of the guest TLAS whose header the 64-bit value at
	// `user_pointer + source.header_offset` points to, the builds of pending BLASes and the TLAS
	// build. Outside a render pass.
	void Prepare(const CommandRecorder& command, uint64_t user_pointer, const TlasSource& source,
	             uint64_t frame);

	// The device address of the context table the shaders look their TLAS up in (see
	// ContextEntry in rt_tlas_walk.comp), 0 when disabled.
	[[nodiscard]] uint64_t ContextAddress() const noexcept { return m_context_address; }

	static constexpr uint32_t MaxContexts = 4;

private:
	struct Allocation {
		Buffer*           buffer  = nullptr;
		uint64_t          offset  = 0;
		vk::DeviceAddress address = 0;
	};

	// Sub-allocates device memory for acceleration structures and their inputs.
	class Arena {
	public:
		Arena(GraphicContext& graphics, CommandScheduler& scheduler): m_graphics(graphics),
		                                                              m_scheduler(scheduler) {}
		Allocation Allocate(uint64_t size, uint64_t alignment);
		[[nodiscard]] uint64_t Used() const noexcept { return m_total; }

	private:
		static constexpr uint64_t ChunkSize = uint64_t {128} << 20u;
		GraphicContext&                      m_graphics;
		CommandScheduler&                    m_scheduler;
		std::vector<std::unique_ptr<Buffer>> m_chunks;
		uint64_t                             m_used  = 0;
		uint64_t                             m_total = 0;
	};

	struct Slot {
		uint64_t                     blas_va     = 0;
		uint32_t                     root        = 0;
		uint32_t                     table_index = 0;
		uint32_t                     capacity0   = 0;
		uint32_t                     capacity1   = 0;
		Allocation                   vertices0;
		Allocation                   vertices1;
		Allocation                   entries;
		Allocation                   storage;
		vk::AccelerationStructureKHR handle      = nullptr;
		vk::DeviceAddress            as_address  = 0;
		uint64_t                     build_scratch = 0;
		bool                         built       = false;
		bool                         needs_build = false;
		// Until this preparation the slot is rebuilt every time (the guest changed it lately).
		uint32_t                     dynamic_until = 0;
		uint64_t                     built_frame   = UINT64_MAX;
	};

	struct PendingSlot {
		uint64_t blas_va     = 0;
		uint32_t root        = 0;
		uint32_t table_index = 0;
		uint32_t leaf_count  = 0;
		uint32_t opaque      = 0;
		uint32_t alpha       = 0;
	};

	struct Readback {
		std::unique_ptr<Buffer> buffer;
		uint64_t                tick  = 0;
		uint32_t                build_count = 0;
		std::vector<uint32_t>   build_slots;
	};

	void CreatePipelines();
	void CreateBuffers();
	void CreateTlas(uint32_t capacity);
	void CreateSlot(const PendingSlot& pending);
	void AllocateSlot(Slot& slot, uint32_t opaque, uint32_t alpha);
	void RecordBuilds(const CommandRecorder& command);
	void ProcessReadback(Readback& readback);

	GraphicContext&   m_graphics;
	CommandScheduler& m_scheduler;
	BufferCache&      m_buffer_cache;
	bool              m_enabled = false;
	bool              m_cleared = false;

	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_tlas_layout       = nullptr;
	vk::PipelineLayout      m_blas_layout       = nullptr;
	vk::Pipeline            m_tlas_pipeline     = nullptr;
	vk::Pipeline            m_blas_pipeline     = nullptr;

	std::unique_ptr<Buffer> m_state;
	std::unique_ptr<Buffer> m_queue;
	std::unique_ptr<Buffer> m_instance_nodes;
	std::unique_ptr<Buffer> m_instances;
	std::unique_ptr<Buffer> m_table;
	std::unique_ptr<Buffer> m_slots;
	std::unique_ptr<Buffer> m_missing;
	std::unique_ptr<Buffer> m_context;
	std::unique_ptr<Buffer> m_pool;
	std::unique_ptr<Buffer> m_verify_list;
	std::unique_ptr<Buffer> m_changed;
	std::unique_ptr<Buffer> m_jobs;
	std::unique_ptr<Buffer> m_build_scratch;
	std::unique_ptr<Buffer> m_tlas_storage;
	vk::AccelerationStructureKHR m_tlas         = nullptr;
	vk::DeviceAddress            m_tlas_address = 0;
	uint64_t                     m_tlas_scratch_size = 0;
	uint32_t                     m_tlas_capacity     = 0;
	uint64_t                     m_context_address   = 0;
	Arena                        m_arena;

	std::array<Readback, 32> m_readbacks;
	uint32_t                m_readback_index = 0;

	// Host state the readbacks (deferred host operations) update.
	Common::Mutex             m_mutex;
	std::vector<PendingSlot>  m_new_slots;
	std::vector<std::unique_ptr<Slot>> m_slot_list;
	std::vector<uint32_t>     m_build_queue;
	std::vector<uint32_t>     m_dynamic;
	// Slots whose builds overflowed: slot, opaque and alpha tested triangles.
	std::vector<std::array<uint32_t, 3>> m_grow;
	// Table entries of BLASes that could not be counted, cleared so the walk reports them again.
	std::vector<uint32_t>     m_retry;
	std::unordered_map<uint64_t, uint32_t> m_retries;
	uint32_t                  m_last_instance_count = 0;
	uint32_t                  m_prepare_id          = 0;
	uint64_t                  m_frame               = 0;

};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_NATIVERAYTRACING_H_
