#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/spscQueue.h"
#include "graphics/guest_gpu/execTrace.h"
#include "graphics/host_gpu/gpuProfiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <cinttypes>

#include "common/syncAcct.h"
#include <string>
#include <unordered_map>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	KYTY_SYNC_SITE(s_refresh, "vk: getSemaphoreCounterValue", Tax);
	Common::SyncAcct::Scope sync_scope(s_refresh, 0);
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	vk::Result result;
	{
		ExecTrace::GpuWaitScope trace_wait;
		if (ExecTrace::Enabled()) {
			const auto known = m_gpu_tick.load(std::memory_order_relaxed);
			ExecTrace::g_wait_tick_behind = std::max(ExecTrace::g_wait_tick_behind, tick - known);
		}
		GpuProfiler::TimedWait timed(GpuProfiler::CurrentWaitTag() != nullptr
		                                 ? GpuProfiler::CurrentWaitTag()
		                                 : "semaphore (untagged)");
		Common::WaitStats::gpu_waits[Common::WaitStats::t_stage].fetch_add(1);
		// Debug (KYTY_SYNC_ACCT): host GPU waits by tag; arg = tick, arg2 = callers, ret = the
		// GPU tick known when the wait began.
		uint16_t site = 0;
		if (Common::SyncAcct::Enabled()) {
			static thread_local std::unordered_map<const char*, uint16_t> sites;
			const char* tag = GpuProfiler::CurrentWaitTag() != nullptr ? GpuProfiler::CurrentWaitTag()
			                                                           : "untagged";
			auto& id = sites[tag];
			if (id == 0) {
				id = Common::SyncAcct::RegisterSite((std::string("gpu wait: ") + tag).c_str(),
				                                    Common::SyncAcct::Kind::Block);
			}
			site = id;
		}
		Common::SyncAcct::Scope sync_scope(site, tick, site != 0 ? Common::SyncAcct::CallerPair() : 0,
		                                   m_gpu_tick.load(std::memory_order_relaxed));
		result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	}
	if (result != vk::Result::eSuccess) {
		DumpGpuCheckpoints(m_graphics.queue);
		EXIT("MasterSemaphore: wait for tick %" PRIu64 " failed: %s\n", tick,
		     vk::to_string(result).c_str());
	}
	Refresh();
}

} // namespace Libs::Graphics
