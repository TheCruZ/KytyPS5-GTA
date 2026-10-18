#include "loader/demonsSoulsCopy.h"

#include "common/virtualMemory.h"
#include "graphics/host_gpu/hostMemory.h"
#include "kernel/memory.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <xxhash.h>

namespace Loader::DemonsSoulsCopy {
namespace {
uint64_t                site = 0;
std::array<uint8_t, 13> original {}, patch {};

bool IsDemonsSouls() {
	char title_id[16] {};
	if (!Loader::SystemContentParamSfoGetString("TITLE_ID", title_id, sizeof(title_id))) return false;
	const std::string_view title(title_id);
	return title == "PPSA01340" || title == "PPSA01341" || title == "PPSA01342";
}

bool CompareEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GUEST_MOVE_COMPARE");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// A large move page by page: a page whose bytes here may be stale (the GPU wrote it: reading it
// would fault and read it back) or differ is written, its GPU-written bytes dropped first (the
// move replaces them); a page that already holds the source's bytes is left alone.
void* CompareMove(uint64_t to, uint64_t from, size_t size) {
	constexpr uint64_t Page = 4096;
	// (The images over all of it as before: a GPU-written image's bytes here are stale, equal or not.)
	(void)Libs::LibKernel::Memory::InvalidateImages(to, size);
	uint64_t           run = 0, run_end = 0;
	const auto         write = [&] {
        if (run_end == run) return;
        (void)Libs::LibKernel::Memory::InvalidateMemoryForWrite(to + run, run_end - run);
        std::memmove(reinterpret_cast<void*>(to + run), reinterpret_cast<const void*>(from + run), run_end - run);
	};
	for (uint64_t at = 0; at < size;) {
		const auto next  = std::min<uint64_t>(size, ((to + at) / Page + 1) * Page - to);
		const auto bytes = next - at;
		const bool differs =
		    Libs::LibKernel::Memory::MayFaultOnGpuRead(to + at, bytes) ||
		    std::memcmp(reinterpret_cast<const void*>(to + at), reinterpret_cast<const void*>(from + at), bytes) != 0;
		if (differs) {
			if (run_end != at) {
				write();
				run = at;
			}
			run_end = next;
		}
		at = next;
	}
	write();
	return reinterpret_cast<void*>(to);
}

void* KYTY_SYSV_ABI CoherentMove(void* destination, const void* source, size_t size) {
	const auto to = reinterpret_cast<uint64_t>(destination), from = reinterpret_cast<uint64_t>(source);
	if (size == 0 || to == from) return destination;
	const bool overlap = to < from + size && from < to + size;
	if (!overlap && size >= 256 * 1024 && CompareEnabled()) return CompareMove(to, from, size);
	// A large move prepares its destination at once instead of a fault a page: a move within one
	// range reads bytes it then writes (their GPU data is read back first), else the destination's
	// GPU data is replaced whole and dropped.
	if (size >= 256 * 1024) {
		if (overlap) {
			Libs::LibKernel::Memory::InvalidateMemory(to, size);
		} else {
			(void)Libs::LibKernel::Memory::InvalidateMemoryForWrite(to, size);
		}
	}
	return std::memmove(destination, source, size);
}

// libc.prx's memmove/bcopy/memcpy body, shared by the three entries, which jump to it after the
// guest's argument and stack checks with one saved RBP. Hashed whole in each build audited (by
// chenxiao07); nothing else branches into the replaced bytes. Other builds fail closed.
struct Body {
	const char* build;
	uint64_t    offset, size, hash;
};
constexpr std::array<Body, 2> Bodies {{
    {"01.007.000", 0x3c36, 2048, 0xe654325b8be848a9ull},
    {"01.005.000", 0x3d06, 1705, 0xc4be692ee02e9f77ull},
}};
} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	// KYTY_GUEST_MOVE=0: the guest's own memmove.
	if (const char* value = std::getenv("KYTY_GUEST_MOVE"); value != nullptr && value[0] == '0') return;
	if (site != 0 || program == nullptr || program->file_name.filename() != "libc.prx" || !IsDemonsSouls()) return;
	const auto body = std::find_if(Bodies.begin(), Bodies.end(), [&](const Body& b) {
		const auto address = program->base_vaddr + b.offset;
		return program->mapped_size >= b.offset + b.size && program->base_vaddr <= UINT64_MAX - b.offset - b.size &&
		       Libs::Graphics::HostMemoryRangeIsReadable(address, b.size) &&
		       XXH3_64bits(reinterpret_cast<const void*>(address), b.size) == b.hash;
	});
	if (body == Bodies.end()) {
		std::printf("Demon's Souls copy: libc.prx of an unknown build; retaining its memmove\n");
		return;
	}
	const auto address = program->base_vaddr + body->offset;
	std::memcpy(original.data(), reinterpret_cast<const void*>(address), original.size());
	patch = Tailcall(reinterpret_cast<uint64_t>(&CoherentMove));
	// Installation occurs before module initializers or guest threads run; SetProgramMemoryProtection
	// keeps executable code writable for loader patches.
	std::memcpy(reinterpret_cast<void*>(address), patch.data(), patch.size());
	if (!Common::VirtualMemory::FlushInstructionCache(address, patch.size())) {
		std::memcpy(reinterpret_cast<void*>(address), original.data(), original.size());
		Common::VirtualMemory::FlushInstructionCache(address, original.size());
		return;
	}
	site = address;
	std::printf("Demon's Souls copy: installed the comparing memmove (libc.prx of %s)\n", body->build);
#endif
}

void Clear() {
	if (site == 0) return;
	if (std::memcmp(reinterpret_cast<const void*>(site), patch.data(), patch.size()) == 0) {
		std::memcpy(reinterpret_cast<void*>(site), original.data(), original.size());
		Common::VirtualMemory::FlushInstructionCache(site, original.size());
	}
	site = 0;
}
} // namespace Loader::DemonsSoulsCopy
