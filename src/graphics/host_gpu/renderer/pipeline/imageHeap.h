#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandBuffer;
class PageManager;
class RenderContext;
class RenderExecutor;
struct ImageHeapTestAccess;

// Guest image tables that shaders index by GPU-computed offsets, kept resident in one bindless
// sampled-image array. A table is keyed by its base and follows its descriptor's size. Each view
// of a table owns device slot words, one per table dword, that select an array element. CPU
// writes to a table fault in its snapshot, which each draw refreshes once; a view takes the
// changes when a draw first acquires it. Changes to the images behind elements bump
// Image::heap_epoch. A smaller table leaves at a suspend point if the frame did not acquire it.
class ImageHeap final: public ShaderRecompiler::IR::ImageHeapSource {
public:
	ImageHeap(RenderContext& context, PageManager& page_manager);
	~ImageHeap();
	KYTY_CLASS_NO_COPY(ImageHeap);

	// Starts a draw or dispatch, before its programs acquire views.
	void BeginDraw() {
		++m_draw;
		Bury();
	}
	bool Acquire(const ShaderRecompiler::IR::ImageHeapRequest& request,
	             ShaderRecompiler::IR::ImageHeapView&          result) override;
	// The candidates of a root that selects descriptors through records. Roots with one profile
	// share a view (and so its elements and kinds); each root owns the slots of its last list.
	bool AcquireList(const ShaderRecompiler::IR::ImageHeapList& list,
	                 ShaderRecompiler::IR::ImageHeapView&       result) override;
	// Image discovery: resolves new elements and rediscovers changed images.
	void Rebind(RenderExecutor& executor);
	// Recording: uploads slots, moves resident images to their read layout and binds the array.
	void Commit(CommandBuffer& buffer, vk::PipelineBindPoint bind_point, vk::PipelineLayout layout);
	// CPU write fault, on any thread.
	void InvalidateMemory(uint64_t address, uint64_t size);
	void UnmapMemory(uint64_t address, uint64_t size);
	// At a suspend point.
	void RunGarbageCollector();

private:
	using DescriptorValue = ShaderRecompiler::IR::DescriptorValue;

	struct DescriptorHash {
		size_t operator()(const DescriptorValue& value) const noexcept;
	};
	struct ListKey {
		const void* owner    = nullptr;
		uint32_t    resource = 0;
		bool        operator==(const ListKey&) const = default;
	};
	struct ListKeyHash {
		size_t operator()(const ListKey& key) const noexcept {
			return std::hash<const void*> {}(key.owner) ^ (size_t {key.resource} * 0x9e3779b97f4a7c15ull);
		}
	};

	struct Element {
		DescriptorValue value;
		TextureBinding  binding;
		uint32_t        kind   = 0;
		uint32_t        refs   = 0; // Slots of the owning view that select it.
		bool            stale  = false;
		uint64_t        serial = 0; // Image::serial of binding.image_id.
	};

	struct Memory;

	struct View {
		Memory&                             memory;
		uint64_t                            step  = 0;
		uint64_t                            first = 0;
		uint64_t                            count = 0;
		uint64_t                            limit = 0; // Probes a key domain allows, if fewer.
		ShaderRecompiler::IR::ImageResource profile;
		std::vector<uint32_t>               slots; // Plus a trailing null slot.
		std::vector<uint32_t>               kind_keys;
		std::vector<DescriptorValue>        kinds; // One representative per kind_keys entry.
		std::unordered_map<DescriptorValue, uint32_t, DescriptorHash> elements;
		std::vector<uint32_t>   pending;  // Not yet resolved.
		std::vector<uint32_t>   released; // No slot selects them after the next upload.
		std::vector<uint32_t>   away;     // Not yet moved to the read layout.
		std::vector<uint32_t>   dcc;      // Color metadata clears apply on each discovery.
		std::vector<uint32_t>   held;     // Another extent holds their address.
		std::unique_ptr<Buffer> buffer;
		size_t                  dirty_begin   = 0; // Slots to upload.
		size_t                  dirty_end     = 0;
		size_t                  changed_begin = 0; // Table bytes changed since it was probed.
		size_t                  changed_end   = 0;
		uint64_t                resolve_epoch = 0;
		uint64_t                layout_epoch  = 0;
		uint64_t                held_serial   = 0; // Image::serials when held was last retried.
	};

	struct Memory {
		uint64_t                           base = 0;
		uint64_t                           size = 0;
		std::vector<uint32_t>              words;
		std::vector<uint8_t>               dirty_pages; // Unwatched, so unread since a write.
		std::vector<uint64_t>              dead;        // Per word, a buried descriptor's hash.
		uint64_t                           draw = 0;    // Last draw that refreshed it.
		std::vector<std::unique_ptr<View>> views;
	};

	// A root's record-table candidates: slots like a view's, which select elements of its view.
	struct List {
		View*                   view = nullptr;
		std::vector<uint32_t>   slots; // Null slot, one per candidate, trailing null slot.
		std::unique_ptr<Buffer> buffer;
		size_t                  dirty_begin = 0; // Slots to upload.
		size_t                  dirty_end   = 0;
	};

	[[nodiscard]] bool Refresh(Memory& memory);
	void               MarkDirty(Memory& memory, size_t begin, size_t end);
	template <bool track>
	void     Watch(const Memory& memory, size_t begin, size_t end);
	void     Unwatch(const Memory& memory);
	void     Drop(Memory& memory);
	void     UpdateProbes(View& view, uint64_t begin, uint64_t end);
	void     UpdateProbe(View& view, uint64_t probe);
	void     Assign(View& view, size_t slot, uint32_t entry);
	void     Assign(List& list, size_t slot, uint32_t entry);
	void     Upload(CommandBuffer& buffer, const std::vector<uint32_t>& slots, Buffer& device,
	                size_t& dirty_begin, size_t& dirty_end);
	void     Bury();
	uint32_t Intern(View& view, uint32_t kind, const DescriptorValue& value);
	void     Release(View& view, uint32_t entry);
	void     Unreference(Element& element);
	Image*   ImageOf(const Element& element) const;
	uint32_t Allocate();
	void     Retire(View& view);
	void     Resize(Memory& memory, uint64_t size);
	void     WriteDescriptors();

	RenderContext&                       m_context;
	PageManager&                         m_page_manager;
	std::mutex                           m_mutex; // Fault-visible Memory state and m_memories.
	std::vector<std::unique_ptr<Memory>> m_memories;
	// Views of record-table candidates, one per profile (Memory holds no table), and each root's
	// last list.
	std::vector<std::unique_ptr<Memory>>          m_list_memories;
	std::unordered_map<ListKey, List, ListKeyHash> m_lists;
	std::vector<View*>                   m_active; // Acquired since the last commit.
	std::vector<List*>                   m_active_lists;
	uint64_t                             m_draw         = 1;
	uint64_t                             m_suspend_draw = 0; // m_draw at the last suspend point.
	std::vector<Element>                 m_elements;
	std::vector<uint32_t>                m_free;
	std::unordered_map<uint64_t, uint64_t>    m_sightings; // Smaller tables: base, first draw.
	std::deque<std::pair<uint64_t, uint32_t>> m_retired; // Tick, element.
	std::vector<std::pair<size_t, size_t>>    m_runs;
	std::vector<uint32_t>                     m_scratch;
	std::vector<uint32_t>                     m_writes; // Elements whose descriptors changed.
	std::vector<GuestRange>                   m_unmapped; // Since the last Bury.
	vk::DescriptorSetLayout                   m_layout = nullptr;
	vk::DescriptorPool                        m_pool   = nullptr;
	vk::DescriptorSet                         m_set    = nullptr;

	friend struct ImageHeapTestAccess;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAP_H_
