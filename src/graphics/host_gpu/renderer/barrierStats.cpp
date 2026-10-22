#include "graphics/host_gpu/renderer/barrierStats.h"

#include "common/assert.h"
#include "common/sampler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::Graphics {

namespace {

thread_local const std::source_location* t_site = nullptr;
// Recording is per thread: the command buffer last recorded on this thread, and whether work
// (draw, dispatch, copy, clear) was recorded into it since its last barrier.
thread_local VkCommandBuffer t_last_command = VK_NULL_HANDLE;
thread_local bool            t_work         = false;

enum WorkKind : int {
	WorkDraw, WorkDrawIndexed, WorkDrawIndirect, WorkDrawIndexedIndirect, WorkDrawIndirectCount,
	WorkDrawIndexedIndirectCount, WorkDispatch, WorkDispatchIndirect, WorkCopyBuffer, WorkCopyImage,
	WorkCopyBufferToImage, WorkCopyImageToBuffer, WorkFillBuffer, WorkUpdateBuffer,
	WorkClearColor, WorkClearDepth, WorkBlit, WorkMesh, WorkMeshIndirect, WorkClearAttachments,
	WorkResolve,
	CountOnly, // below: counted, not work
	BeginRendering = CountOnly, BeginRenderingKhr, EndRendering, EndRenderingKhr, QueueSubmit,
	KindCount
};
constexpr const char* KindNames[KindCount] = {
    "draw", "drawIndexed", "drawIndirect", "drawIndexedIndirect", "drawIndirectCount",
    "drawIndexedIndirectCount", "dispatch", "dispatchIndirect", "copyBuffer", "copyImage",
    "copyBufferToImage", "copyImageToBuffer", "fillBuffer", "updateBuffer", "clearColor",
    "clearDepth", "blit", "mesh", "meshIndirect", "clearAttachments", "resolve",
    "beginRendering", "beginRenderingKHR", "endRendering", "endRenderingKHR", "queueSubmit"};
std::atomic<uint64_t> g_kind_counts[KindCount] {};

std::mutex                                         g_end_mutex;
std::map<std::pair<const char*, uint32_t>, uint64_t> g_end_sites;

void NoteWork(VkCommandBuffer command, int kind) {
	g_kind_counts[kind].fetch_add(1, std::memory_order_relaxed);
	if ((kind == EndRendering || kind == EndRenderingKhr) && t_site != nullptr) {
		std::scoped_lock lock(g_end_mutex);
		g_end_sites[{t_site->file_name(), t_site->line()}]++;
	}
	if (kind < CountOnly) {
		t_last_command = command;
		t_work         = true;
	}
}

template <int Id, typename F>
struct Hook;
template <int Id, typename R, typename... A>
struct Hook<Id, R(VKAPI_PTR*)(VkCommandBuffer, A...)> {
	using Fn                 = R(VKAPI_PTR*)(VkCommandBuffer, A...);
	static inline Fn original = nullptr;
	static R VKAPI_PTR Call(VkCommandBuffer command, A... args) {
		NoteWork(command, Id);
		return original(command, args...);
	}
	static void Install(Fn& slot) {
		if (slot != nullptr && slot != &Call) {
			original = slot;
			slot     = &Call;
		}
	}
};
template <int Id, typename R, typename... A>
struct Hook<Id, R(VKAPI_PTR*)(VkQueue, A...)> {
	using Fn                 = R(VKAPI_PTR*)(VkQueue, A...);
	static inline Fn original = nullptr;
	static R VKAPI_PTR Call(VkQueue queue, A... args) {
		g_kind_counts[Id].fetch_add(1, std::memory_order_relaxed);
		return original(queue, args...);
	}
	static void Install(Fn& slot) {
		if (slot != nullptr && slot != &Call) {
			original = slot;
			slot     = &Call;
		}
	}
};

PFN_vkCmdPipelineBarrier  g_barrier      = nullptr;
PFN_vkCmdPipelineBarrier2 g_barrier2     = nullptr;
PFN_vkCmdPipelineBarrier2 g_barrier2_khr = nullptr;

struct SiteKey {
	const char* file   = nullptr; // CommandRecorder caller (std::source_location)
	uint32_t    line   = 0;
	const void* pc[3]  = {};      // otherwise the host stack above the barrier call
	uint32_t    kind   = 0;       // 1 = vkCmdPipelineBarrier, 2 = vkCmdPipelineBarrier2
	uint64_t    src    = 0;
	uint64_t    dst    = 0;
	uint32_t    shapes = 0;       // bit 0 memory, 1 buffer, 2 image barriers present
	auto        Tie() const {
		       return std::tie(file, line, pc[0], pc[1], pc[2], kind, src, dst, shapes);
	}
	bool operator<(const SiteKey& other) const { return Tie() < other.Tie(); }
};

std::mutex                    g_mutex;
struct SiteCount {
	uint64_t total      = 0;
	uint64_t after_work = 0; // first barrier after a draw/dispatch/copy: a real pipeline drain
};
std::map<SiteKey, SiteCount>  g_counts;
uint64_t                      g_total        = 0;
int64_t                       g_window_start = 0;

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

std::string BaseName(const char* path) {
	std::string s = path != nullptr ? path : "?";
	if (const auto slash = s.find_last_of("\\/"); slash != std::string::npos) {
		s.erase(0, slash + 1);
	}
	return s;
}

// Prints and resets the window. Called with g_mutex held.
void Report(int64_t now) {
	const double seconds = (std::max)(1.0, static_cast<double>(now - g_window_start) / 1000.0);
	std::vector<std::pair<SiteKey, SiteCount>> rows(g_counts.begin(), g_counts.end());
	std::sort(rows.begin(), rows.end(),
	          [](const auto& a, const auto& b) { return a.second.total > b.second.total; });
	uint64_t drains = 0;
	for (const auto& row: rows) {
		drains += row.second.after_work;
	}
	std::string out = fmt::format("[barriers] {:.0f}/s over {:.1f} s, {} distinct, {:.0f}/s after work\n",
	                              static_cast<double>(g_total) / seconds, seconds, rows.size(),
	                              static_cast<double>(drains) / seconds);
	out += "[barriers] work/s:";
	for (int k = 0; k < KindCount; k++) {
		const auto n = g_kind_counts[k].exchange(0, std::memory_order_relaxed);
		if (n != 0) {
			out += fmt::format(" {}={:.0f}", KindNames[k], static_cast<double>(n) / seconds);
		}
	}
	out += "\n";
	{
		std::scoped_lock lock(g_end_mutex);
		std::vector<std::pair<std::pair<const char*, uint32_t>, uint64_t>> ends(g_end_sites.begin(),
		                                                                       g_end_sites.end());
		std::sort(ends.begin(), ends.end(),
		          [](const auto& a, const auto& b) { return a.second > b.second; });
		out += "[barriers] endRendering by site:";
		for (size_t i = 0; i < ends.size() && i < 12; i++) {
			out += fmt::format(" {}:{}={:.0f}", BaseName(ends[i].first.first), ends[i].first.second,
			                   static_cast<double>(ends[i].second) / seconds);
		}
		out += "\n";
		g_end_sites.clear();
	}
	const size_t shown = (std::min<size_t>)(rows.size(), 25);
	for (size_t i = 0; i < shown; i++) {
		const auto& [key, count] = rows[i];
		out += fmt::format("[barriers] {:7.0f}/s ({:3.0f}% after work) v{} src=0x{:x} dst=0x{:x}{}{}{}",
		                   static_cast<double>(count.total) / seconds,
		                   100.0 * static_cast<double>(count.after_work) /
		                       static_cast<double>((std::max)(count.total, uint64_t {1})),
		                   key.kind, key.src, key.dst, (key.shapes & 1) != 0 ? " mem" : "",
		                   (key.shapes & 2) != 0 ? " buf" : "", (key.shapes & 4) != 0 ? " img" : "");
		if (key.file != nullptr) {
			out += fmt::format(" at {}:{}\n", BaseName(key.file), key.line);
		} else {
			out += "\n";
			for (const auto* pc: key.pc) {
				if (pc != nullptr) {
					out += "[barriers]            " + Common::HostAddressName(pc) + "\n";
				}
			}
		}
	}
	std::fputs(out.c_str(), stderr);
	std::fflush(stderr);
	g_counts.clear();
	g_total        = 0;
	g_window_start = now;
}

#ifdef _WIN32
__declspec(noinline)
#endif
void Count(VkCommandBuffer command, SiteKey key) {
	const bool after_work = command == t_last_command && t_work;
	t_last_command        = command;
	t_work                = false;
	if (t_site != nullptr) {
		key.file = t_site->file_name();
		key.line = t_site->line();
	} else {
#ifdef _WIN32
		// Skip this function and the hook; the first frame left is the vk::CommandBuffer
		// wrapper's caller (the wrapper is inlined).
		void* frames[3] {};
		CaptureStackBackTrace(2, 3, frames, nullptr);
		for (int i = 0; i < 3; i++) {
			key.pc[i] = frames[i];
		}
#endif
	}
	const auto       now = NowMs();
	std::scoped_lock lock(g_mutex);
	if (g_window_start == 0) {
		g_window_start = now;
	}
	auto& count = g_counts[key];
	count.total++;
	count.after_work += after_work ? 1 : 0;
	g_total++;
	if (now - g_window_start >= 5000) {
		Report(now);
	}
}

VKAPI_ATTR void VKAPI_CALL HookBarrier(VkCommandBuffer command, VkPipelineStageFlags src,
                                       VkPipelineStageFlags dst, VkDependencyFlags flags,
                                       uint32_t memory_count, const VkMemoryBarrier* memory,
                                       uint32_t buffer_count, const VkBufferMemoryBarrier* buffers,
                                       uint32_t image_count, const VkImageMemoryBarrier* images) {
	SiteKey key;
	key.kind   = 1;
	key.src    = src;
	key.dst    = dst;
	key.shapes = (memory_count != 0 ? 1u : 0u) | (buffer_count != 0 ? 2u : 0u) |
	             (image_count != 0 ? 4u : 0u);
	Count(command, key);
	g_barrier(command, src, dst, flags, memory_count, memory, buffer_count, buffers, image_count,
	          images);
}

SiteKey Key2(const VkDependencyInfo* info) {
	SiteKey key;
	key.kind = 2;
	for (uint32_t i = 0; i < info->memoryBarrierCount; i++) {
		key.src |= info->pMemoryBarriers[i].srcStageMask;
		key.dst |= info->pMemoryBarriers[i].dstStageMask;
	}
	for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; i++) {
		key.src |= info->pBufferMemoryBarriers[i].srcStageMask;
		key.dst |= info->pBufferMemoryBarriers[i].dstStageMask;
	}
	for (uint32_t i = 0; i < info->imageMemoryBarrierCount; i++) {
		key.src |= info->pImageMemoryBarriers[i].srcStageMask;
		key.dst |= info->pImageMemoryBarriers[i].dstStageMask;
	}
	key.shapes = (info->memoryBarrierCount != 0 ? 1u : 0u) |
	             (info->bufferMemoryBarrierCount != 0 ? 2u : 0u) |
	             (info->imageMemoryBarrierCount != 0 ? 4u : 0u);
	return key;
}

VKAPI_ATTR void VKAPI_CALL HookBarrier2(VkCommandBuffer command, const VkDependencyInfo* info) {
	Count(command, Key2(info));
	g_barrier2(command, info);
}

VKAPI_ATTR void VKAPI_CALL HookBarrier2Khr(VkCommandBuffer command, const VkDependencyInfo* info) {
	Count(command, Key2(info));
	g_barrier2_khr(command, info);
}

} // namespace

void SetBarrierSite(const std::source_location* site) noexcept {
	t_site = site;
}

void InstallBarrierStats() {
	if (std::getenv("KYTY_BARRIER_STATS") == nullptr) {
		return;
	}
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (d.vkCmdPipelineBarrier != nullptr && d.vkCmdPipelineBarrier != HookBarrier) {
		g_barrier              = d.vkCmdPipelineBarrier;
		d.vkCmdPipelineBarrier = HookBarrier;
	}
	if (d.vkCmdPipelineBarrier2 != nullptr && d.vkCmdPipelineBarrier2 != HookBarrier2) {
		g_barrier2              = d.vkCmdPipelineBarrier2;
		d.vkCmdPipelineBarrier2 = HookBarrier2;
	}
	if (d.vkCmdPipelineBarrier2KHR != nullptr && d.vkCmdPipelineBarrier2KHR != HookBarrier2Khr) {
		g_barrier2_khr             = d.vkCmdPipelineBarrier2KHR;
		d.vkCmdPipelineBarrier2KHR = HookBarrier2Khr;
	}
	Hook<WorkDraw, decltype(d.vkCmdDraw)>::Install(d.vkCmdDraw);
	Hook<WorkDrawIndexed, decltype(d.vkCmdDrawIndexed)>::Install(d.vkCmdDrawIndexed);
	Hook<WorkDrawIndirect, decltype(d.vkCmdDrawIndirect)>::Install(d.vkCmdDrawIndirect);
	Hook<WorkDrawIndexedIndirect, decltype(d.vkCmdDrawIndexedIndirect)>::Install(
	    d.vkCmdDrawIndexedIndirect);
	Hook<WorkDrawIndirectCount, decltype(d.vkCmdDrawIndirectCount)>::Install(
	    d.vkCmdDrawIndirectCount);
	Hook<WorkDrawIndexedIndirectCount, decltype(d.vkCmdDrawIndexedIndirectCount)>::Install(
	    d.vkCmdDrawIndexedIndirectCount);
	Hook<WorkDispatch, decltype(d.vkCmdDispatch)>::Install(d.vkCmdDispatch);
	Hook<WorkDispatchIndirect, decltype(d.vkCmdDispatchIndirect)>::Install(d.vkCmdDispatchIndirect);
	Hook<WorkCopyBuffer, decltype(d.vkCmdCopyBuffer)>::Install(d.vkCmdCopyBuffer);
	Hook<WorkCopyImage, decltype(d.vkCmdCopyImage)>::Install(d.vkCmdCopyImage);
	Hook<WorkCopyBufferToImage, decltype(d.vkCmdCopyBufferToImage)>::Install(
	    d.vkCmdCopyBufferToImage);
	Hook<WorkCopyImageToBuffer, decltype(d.vkCmdCopyImageToBuffer)>::Install(
	    d.vkCmdCopyImageToBuffer);
	Hook<WorkFillBuffer, decltype(d.vkCmdFillBuffer)>::Install(d.vkCmdFillBuffer);
	Hook<WorkUpdateBuffer, decltype(d.vkCmdUpdateBuffer)>::Install(d.vkCmdUpdateBuffer);
	Hook<WorkClearColor, decltype(d.vkCmdClearColorImage)>::Install(d.vkCmdClearColorImage);
	Hook<WorkClearDepth, decltype(d.vkCmdClearDepthStencilImage)>::Install(
	    d.vkCmdClearDepthStencilImage);
	Hook<WorkBlit, decltype(d.vkCmdBlitImage)>::Install(d.vkCmdBlitImage);
	Hook<WorkMesh, decltype(d.vkCmdDrawMeshTasksEXT)>::Install(d.vkCmdDrawMeshTasksEXT);
	Hook<WorkMeshIndirect, decltype(d.vkCmdDrawMeshTasksIndirectEXT)>::Install(
	    d.vkCmdDrawMeshTasksIndirectEXT);
	Hook<WorkClearAttachments, decltype(d.vkCmdClearAttachments)>::Install(d.vkCmdClearAttachments);
	Hook<WorkResolve, decltype(d.vkCmdResolveImage)>::Install(d.vkCmdResolveImage);
	Hook<BeginRendering, decltype(d.vkCmdBeginRendering)>::Install(d.vkCmdBeginRendering);
	Hook<BeginRenderingKhr, decltype(d.vkCmdBeginRenderingKHR)>::Install(d.vkCmdBeginRenderingKHR);
	Hook<EndRendering, decltype(d.vkCmdEndRendering)>::Install(d.vkCmdEndRendering);
	Hook<EndRenderingKhr, decltype(d.vkCmdEndRenderingKHR)>::Install(d.vkCmdEndRenderingKHR);
	Hook<QueueSubmit, decltype(d.vkQueueSubmit)>::Install(d.vkQueueSubmit);
	std::fprintf(stderr, "[barriers] counting barriers by call site\n");
}

} // namespace Libs::Graphics
