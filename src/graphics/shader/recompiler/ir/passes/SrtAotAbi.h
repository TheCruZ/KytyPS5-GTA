#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTAOTABI_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTAOTABI_H_

// The interface between the emulator and a library of resource plans compiled ahead of time (chenxiao07's SRT AOT,
// src/local/SrtAotAbi.h): the emulator exports each linear plan it builds as C++ source (KYTY_SRT_AOT_EXPORT=<dir>),
// the kyty_srt_aot target compiles a directory of them into a library (KYTY_SRT_AOT_SOURCES), and the emulator calls
// a plan's function from it when the library holds that exact source (LinearSrt.cpp). Standalone: no emulator types,
// no standard library layouts, no pointers baked into the code.

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(_WIN32)
#define KYTY_SRT_AOT_EXPORT __declspec(dllexport)
#else
#define KYTY_SRT_AOT_EXPORT __attribute__((visibility("default")))
#endif

namespace KytySrtAot {

constexpr uint32_t AbiVersion = 1;

// Reads the dword at `address` for compiled-plan node `node` through the emulator's reader (clean: the strict,
// specialization reader); a node read before in the same evaluation gives its value again.
using Reader = bool (*)(void* context, uint64_t address, uint32_t node, uint32_t clean, uint64_t* out);

struct Runtime {
	const uint32_t* user_data;
	uint64_t        user_count;
	uint64_t        shader_base;
	void*           context;
	Reader          read;
};

// Computes every node of a plan into values[]; false when the evaluation fails.
using Function = bool (*)(const Runtime* runtime, uint64_t* values);

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

inline bool AddSigned(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) return false;
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) return false;
		result = base - magnitude;
	} else {
		if (static_cast<uint64_t>(offset) > AddressMask - base) return false;
		result = base + static_cast<uint64_t>(offset);
	}
	return true;
}

// A scalar load at an address from user data or constants (low, high: the address words; immediate: byte offset).
inline bool ReadLeaf(const Runtime* r, uint64_t low, uint64_t high, int64_t immediate, uint32_t node, uint32_t clean,
                     uint64_t& out) {
	uint64_t address = 0;
	if (!AddSigned((((high << 32u) | static_cast<uint32_t>(low)) & AddressMask) & ~uint64_t {3}, immediate, address))
		return false;
	return r->read(r->context, address, node, clean, &out);
}

// LoadAddressU32 (buffer 0) or ReadConstBuffer (buffer 1): a buffer read past its V#'s range gives zero.
inline bool ReadRaw(const Runtime* r, uint32_t buffer, uint64_t low, uint64_t high, uint64_t records,
                    uint64_t offset, int64_t immediate, uint32_t node, uint32_t clean, uint64_t& out) {
	const auto base    = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	uint64_t   address = 0;
	if (buffer != 0u) {
		if (immediate < 0) return false;
		const auto bytes  = (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
		const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		const auto size   = stride == 0u ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                                 : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (bytes > size || size - bytes < 4u) {
			out = 0;
			return true;
		}
		address = (base & ~uint64_t {3}) + bytes;
	} else {
		const auto relative = (immediate & ~int64_t {3}) + static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSigned(base & ~uint64_t {3}, relative, address)) return false;
	}
	return r->read(r->context, address, node, clean, &out);
}

inline float F32(uint64_t bits) {
	const auto narrowed = static_cast<uint32_t>(bits);
	float      value    = 0;
	std::memcpy(&value, &narrowed, 4);
	return value;
}
inline uint64_t Bits(float value) {
	uint32_t bits = 0;
	std::memcpy(&bits, &value, 4);
	return bits;
}

inline bool BitFieldUExtract(uint64_t a, uint64_t b, uint64_t c, uint64_t& out) {
	const auto offset = static_cast<uint32_t>(b), width = static_cast<uint32_t>(c);
	if (offset > 32u || width > 32u - offset) return false;
	const auto mask = width == 32u ? 0xffffffffu : width == 0u ? 0u : (uint32_t {1} << width) - 1u;
	out             = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
	return true;
}
inline bool BitFieldSExtract(uint64_t a, uint64_t b, uint64_t c, uint64_t& out) {
	const auto offset = static_cast<uint32_t>(b), width = static_cast<uint32_t>(c);
	if (offset > 32u || width > 32u - offset) return false;
	if (width == 0u) {
		out = 0;
		return true;
	}
	const auto mask = width == 32u ? 0xffffffffu : (uint32_t {1} << width) - 1u;
	auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
	if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) bits |= ~mask;
	out = bits;
	return true;
}
inline bool BitFieldInsert(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t& out) {
	const auto offset = static_cast<uint32_t>(c), width = static_cast<uint32_t>(d);
	if (offset > 32u || width > 32u - offset) return false;
	if (width == 0u) {
		out = static_cast<uint32_t>(a);
		return true;
	}
	const auto mask = width == 32u ? 0xffffffffu : ((uint32_t {1} << width) - 1u) << offset;
	out             = (static_cast<uint32_t>(a) & ~mask) | ((static_cast<uint32_t>(b) << offset) & mask);
	return true;
}
inline bool ConvertU32F32(uint64_t a, uint64_t& out) {
	const auto value = F32(a);
	if (!std::isfinite(value) || value < 0.0f || static_cast<double>(value) > 4294967295.0) return false;
	out = static_cast<uint32_t>(value);
	return true;
}
inline bool FPRecipIFlag32(uint64_t a, uint64_t& out) {
	const auto exponent = (a >> 23u) & 0xffu;
	if ((a & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
	out = (254u - exponent) << 23u;
	return true;
}
// FPOrdLessThanEqual32 (greater 0) / FPOrdGreaterThanEqual32 (greater 1); flush: denormals compare as zero.
inline uint64_t FPOrdCompare(uint64_t a, uint64_t b, uint32_t greater, uint32_t flush) {
	const auto operand = [&](uint64_t bits) {
		if (flush != 0u && (bits & 0x7fffffffu) < 0x00800000u) bits &= 0x80000000u;
		return F32(bits);
	};
	return greater != 0u ? operand(a) >= operand(b) : operand(a) <= operand(b);
}

} // namespace KytySrtAot

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTAOTABI_H_
