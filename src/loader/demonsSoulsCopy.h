#pragma once

#include "common/abi.h"

#include <array>
#include <cstdint>
#include <cstring>

namespace Loader {
struct Program;
// Demon's Souls' libc.prx memmove (and memcpy, bcopy, which share its body), replaced by one that
// leaves alone the destination pages of a large move that already hold the source's bytes: no
// write fault, no CPU-dirty page for the GPU caches to upload again. Ported from chenxiao07's
// KytyPS5 fork (loader/demonsSoulsCopy.cpp). KYTY_GUEST_MOVE_COMPARE=0 keeps every write.
namespace DemonsSoulsCopy {
void Install(Program* program);
void Clear();
// pop rbp; mov rax, function; jmp rax: the body is entered with one saved RBP.
inline std::array<uint8_t, 13> Tailcall(uint64_t function) {
	std::array<uint8_t, 13> code {0x5d, 0x48, 0xb8};
	std::memcpy(code.data() + 3, &function, sizeof(function));
	code[11] = 0xff;
	code[12] = 0xe0;
	return code;
}
} // namespace DemonsSoulsCopy
} // namespace Loader
