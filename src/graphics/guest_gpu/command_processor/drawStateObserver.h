#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_DRAW_STATE_OBSERVER_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_DRAW_STATE_OBSERVER_H

// Ported from chenxiao07's KytyPS5 fork (src/local/draw-state-observer.h).
//
// Tracks whether the graphics state changed between consecutive indexed draws (indirect or direct), for the table
// draws' runs (RenderExecutor::TableContinue): a draw is "clean" when nothing but the per-draw user-data SGPRs, the
// index state and the indirect offset changed since the previous one, so the previous draw's draw state still applies.
//
// A register write only dirties the gap when it changes a value (shadow copies of the CX/SH/UC spaces). Graphics
// user-data SGPRs (PS 0x0c-0x2b, GS 0x8c-0xab, HS 0x10c-0x12b) and the compute space (0x200+) never dirty it;
// dispatches, synchronisation, index state and indirect bases never do either. One per command processor (each has
// its own registers).

#include "graphics/guest_gpu/pm4.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics {

class DrawStateObserver {
public:
	// Whether the latest indexed draw packet was clean.
	[[nodiscard]] bool LastClean() const noexcept { return m_last_clean; }

	// The registers changed in bulk (context clear/push/pop, a register reset): nothing the shadow holds is known any
	// more.
	void Invalidate() noexcept {
		// Each register's last value (low 32 bits) and the generation it was written in (high 32 bits): known while
		// that is `m_generation`, so a reset is one increment.
		m_generation += uint64_t {1} << 32u;
		if (m_generation == 0) { // (wrapped: an entry of the first generation would be taken as known again)
			m_cx.fill(0);
			m_sh.fill(0);
			m_uc.fill(0);
			m_generation = uint64_t {1} << 32u;
		}
		m_chain = false;
		m_dirty = true;
	}

	// Every packet the command processor is about to execute (`predicated_skip`: one it skips).
	void ObservePacket(uint32_t opcode, const uint32_t* packet, uint32_t packet_dw, bool predicated_skip) noexcept {
		if (predicated_skip) {
			m_dirty = true;
			return;
		}
		switch (opcode) {
			case Pm4::IT_DRAW_INDEX_INDIRECT:
			case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
			case Pm4::IT_DRAW_INDEX_2:
			case Pm4::IT_DRAW_INDEX_OFFSET_2:
				m_last_clean = m_chain && !m_dirty;
				m_chain      = true;
				m_dirty      = false;
				return;
			case Pm4::IT_SET_CONTEXT_REG:
			case Pm4::IT_SET_SH_REG:
			case Pm4::IT_SET_UCONFIG_REG:
			case Pm4::IT_SET_UCONFIG_REG_INDEX: {
				const uint32_t space = opcode == Pm4::IT_SET_CONTEXT_REG ? 0u : opcode == Pm4::IT_SET_SH_REG ? 1u : 2u;
				for (uint32_t i = 2; i < packet_dw; ++i) WriteRegister(space, packet[1] + (i - 2), packet[i]);
				return;
			}
			case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
			case Pm4::IT_SET_SH_REG_INDIRECT:
			case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
				if (packet_dw < 5) {
					m_dirty = true;
					return;
				}
				WriteIndirectBlock(opcode == Pm4::IT_SET_CONTEXT_REG_INDIRECT ? 0u
				                   : opcode == Pm4::IT_SET_SH_REG_INDIRECT    ? 1u
				                                                              : 2u,
				                   packet);
				return;
			// The context registers back to their defaults: a shadow value is no longer what a register holds.
			case Pm4::IT_CLEAR_STATE: Invalidate(); return;
			case Pm4::IT_INDEX_BASE:
			case Pm4::IT_INDEX_BUFFER_SIZE:
			case Pm4::IT_INDEX_TYPE:
			case Pm4::IT_DISPATCH_DIRECT:
			case Pm4::IT_DISPATCH_INDIRECT:
			case Pm4::IT_SET_BASE: return;
			case Pm4::IT_NOP:
				// NOP-encoded commands: R_CONTEXT_STATE clears, pushes or pops the graphics context and R_DISPATCH_RESET
				// resets every register (CommandProcessor::Reset); R_DRAW_RESET too.
				if (const auto r = KYTY_PM4_R(packet[0]);
				    r == Pm4::R_CONTEXT_STATE || r == Pm4::R_DISPATCH_RESET || r == Pm4::R_DRAW_RESET)
					Invalidate();
				return;
			case Pm4::IT_NUM_INSTANCES:
			case Pm4::IT_ACQUIRE_MEM:
			case Pm4::IT_RELEASE_MEM:
			case Pm4::IT_WAIT_REG_MEM:
			case Pm4::IT_WAIT_REG_MEM_64:
			case Pm4::IT_EVENT_WRITE:
			case Pm4::IT_EVENT_WRITE_EOP:
			case Pm4::IT_EVENT_WRITE_EOS:
			case Pm4::IT_WRITE_DATA:
			case Pm4::IT_PFP_SYNC_ME:
			case Pm4::IT_INDIRECT_BUFFER:
			case Pm4::IT_COPY_DATA:
			case Pm4::IT_DMA_DATA: return;
			default:
				// Any other draw ends the chain; anything unknown may change state.
				m_chain = false;
				m_dirty = true;
				return;
		}
	}

private:
	[[nodiscard]] static bool ShFree(uint32_t offset) noexcept {
		return (offset >= 0x0cu && offset <= 0x2bu) || (offset >= 0x8cu && offset <= 0xabu) ||
		       (offset >= 0x10cu && offset <= 0x12bu) || offset >= 0x200u;
	}

	// space: 0 CX, 1 SH, 2 UC.
	void WriteRegister(uint32_t space, uint32_t raw_offset, uint32_t value) noexcept {
		const auto offset = raw_offset & 0xffffu;
		const auto update = [&](auto& values) {
			if (offset >= values.size()) {
				m_dirty = true;
				return;
			}
			const uint64_t entry = m_generation | value;
			if (values[offset] != entry) {
				values[offset] = entry;
				m_dirty        = true;
			}
		};
		if (space == 1) {
			if (offset < Pm4::SH_NUM && ShFree(offset)) return;
			update(m_sh);
		} else if (space == 0) {
			update(m_cx);
		} else {
			update(m_uc);
		}
	}

	void WriteIndirectBlock(uint32_t space, const uint32_t* packet) noexcept {
		const auto* block =
		    reinterpret_cast<const uint32_t*>((uint64_t(packet[1]) & 0xfffffffcu) | (uint64_t(packet[2]) << 32u));
		const uint32_t pairs = packet[4] & 0x3fffu;
		if (block == nullptr) {
			m_dirty = true;
			return;
		}
		for (uint32_t i = 0; i < pairs; ++i) {
			if (block[2 * i] == 0xffffffffu) continue;
			WriteRegister(space, block[2 * i], block[2 * i + 1]);
		}
	}

	std::array<uint64_t, Pm4::CX_NUM> m_cx {};
	std::array<uint64_t, Pm4::SH_NUM> m_sh {};
	std::array<uint64_t, Pm4::UC_NUM> m_uc {};
	uint64_t                          m_generation = uint64_t {1} << 32u;
	bool                              m_chain      = false; // the previous graphics draw was an indexed one
	bool                              m_dirty      = true;  // graphics state changed since that draw
	bool                              m_last_clean = false;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_DRAW_STATE_OBSERVER_H
