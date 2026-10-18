#include "graphics/shader/recompiler/ir/passes/LinearSrt.h"

#include "common/abi.h"
#include "common/assert.h"
#include "graphics/shader/recompiler/ir/passes/SrtAotAbi.h"

#ifndef XBYAK_NO_EXCEPTION
#define XBYAK_NO_EXCEPTION
#endif
#include <xbyak/xbyak.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

using Kind = LinearSrtPlan::Kind;
using Node = LinearSrtPlan::Node;
using Leaf = LinearSrtPlan::Leaf;

constexpr uint32_t Invalid = LinearSrtPlan::Invalid;

bool Switch(const char* name, bool default_value) {
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') return default_value;
	return value[0] != '0';
}

// What the native code reads: plain data only (offsets are taken in the generated code).
struct LinearRuntime {
	const uint32_t*   user       = nullptr;
	uint64_t          user_count = 0;
	uint64_t          base       = 0;
	const SrtRuntime* runtime    = nullptr;
};

// As CompiledSrtWalker::ReadWord(): the plain reader (none: the memory itself), or for a clean read the strict
// reader (none: the read fails, as CleanRuntime() makes it).
bool ReadWord(const SrtRuntime& runtime, bool clean, uint64_t address, uint64_t& out) {
	uint32_t   word   = 0;
	const auto reader = clean ? runtime.read_specialization_memory : runtime.read_memory;
	bool       read   = true;
	if (reader != nullptr) {
		read = reader(runtime.userdata, address, {&word, 1});
	} else if (clean) {
		read = false;
	} else {
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	out = word;
	return read;
}

// The reads of one evaluation by compiled-plan node and read policy: as the interpreter's memos, a node read (or failed)
// once is not read again by the conditions' programs and the leaf that follows them.
struct ReadMemo {
	std::vector<uint64_t> values, states;
	uint64_t              generation = 0;
};
thread_local ReadMemo t_read_memo;

void BeginReads(size_t nodes) {
	auto& memo = t_read_memo;
	if (memo.states.size() < nodes * 2u) {
		memo.values.resize(nodes * 2u);
		memo.states.assign(nodes * 2u, 0);
		memo.generation = 0;
	}
	memo.generation += 2u;
}

// Only plans with conditions read a node from more than one program (each node is once in a leaf).
thread_local bool t_read_memo_active = false;

bool MemoRead(const SrtRuntime& runtime, bool clean, uint32_t node, uint64_t address, uint64_t& out) {
	if (!t_read_memo_active) return ReadWord(runtime, clean, address, out);
	auto&      memo  = t_read_memo;
	const auto index = static_cast<size_t>(node) * 2u + (clean ? 1u : 0u);
	if (node == Invalid || index >= memo.states.size()) return ReadWord(runtime, clean, address, out);
	if (memo.states[index] == memo.generation) {
		out = memo.values[index];
		return true;
	}
	if (memo.states[index] == memo.generation + 1u) return false;
	const bool read     = ReadWord(runtime, clean, address, out);
	memo.values[index] = out;
	memo.states[index] = read ? memo.generation : memo.generation + 1u;
	return read;
}

bool AotRead(void* context, uint64_t address, uint32_t node, uint32_t clean, uint64_t* out) {
	return MemoRead(*static_cast<const SrtRuntime*>(context), clean != 0u, node, address, *out);
}

// The helpers the native code calls (an explicit SysV convention on every host).
KYTY_SYSV_ABI bool ReadLeafNode(const LinearRuntime* r, const Node* n, const uint64_t* values, uint64_t* out) {
	const auto low = values[n->args[0]], high = values[n->args[1]];
	uint64_t   address = 0;
	if (!KytySrtAot::AddSigned((((high << 32u) | static_cast<uint32_t>(low)) & KytySrtAot::AddressMask) & ~uint64_t {3},
	                           static_cast<int64_t>(n->imm), address))
		return false;
	return MemoRead(*r->runtime, n->clean, n->plan_node, address, *out);
}

KYTY_SYSV_ABI bool ReadRawNode(const LinearRuntime* r, const Node* n, const uint64_t* values, uint64_t* out) {
	const bool buffer    = n->op == ValueOpcode::ReadConstBuffer;
	const auto low       = values[n->args[0]], high = values[n->args[1]];
	const auto offset    = values[n->args[4]];
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & KytySrtAot::AddressMask;
	const auto immediate = static_cast<int64_t>(n->imm);
	uint64_t   address   = 0;
	if (buffer) {
		if (immediate < 0) return false;
		const auto records = values[n->args[2]];
		const auto bytes   = (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
		const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		const auto size    = stride == 0u ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                                  : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (bytes > size || size - bytes < 4u) {
			*out = 0; // out-of-range scalar descriptor reads return zero
			return true;
		}
		address = (base & ~uint64_t {3}) + bytes;
	} else {
		const auto relative = (immediate & ~int64_t {3}) + static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!KytySrtAot::AddSigned(base & ~uint64_t {3}, relative, address)) return false;
	}
	return MemoRead(*r->runtime, n->clean, n->plan_node, address, *out);
}

KYTY_SYSV_ABI bool ReadUserNode(const LinearRuntime* r, uint64_t index, uint64_t* out) {
	if (index >= r->user_count) return false;
	*out = r->user[index];
	return true;
}

// The operations the native code does not inline, as CompiledSrtWalker::EvaluateInst() computes them.
KYTY_SYSV_ABI bool OpNode(const Node* n, const uint64_t* values, uint64_t* out) {
	const auto a = n->args[0] != Invalid ? values[n->args[0]] : 0;
	const auto b = n->args[1] != Invalid ? values[n->args[1]] : 0;
	const auto c = n->args[2] != Invalid ? values[n->args[2]] : 0;
	const auto d = n->args[3] != Invalid ? values[n->args[3]] : 0;
	switch (n->op) {
		case ValueOpcode::UMulHi:
			*out = (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
			return true;
		case ValueOpcode::UMax32: *out = std::max(static_cast<uint32_t>(a), static_cast<uint32_t>(b)); return true;
		case ValueOpcode::ConvertF32U32:
			*out = KytySrtAot::Bits(static_cast<float>(static_cast<uint32_t>(a)));
			return true;
		case ValueOpcode::ConvertU32F32: return KytySrtAot::ConvertU32F32(a, *out);
		case ValueOpcode::FPMul32: *out = KytySrtAot::Bits(KytySrtAot::F32(a) * KytySrtAot::F32(b)); return true;
		case ValueOpcode::FPTrunc32: *out = KytySrtAot::Bits(std::trunc(KytySrtAot::F32(a))); return true;
		case ValueOpcode::FPRecipIFlag32: return KytySrtAot::FPRecipIFlag32(a, *out);
		case ValueOpcode::FPIsNan32: *out = std::isnan(KytySrtAot::F32(a)) ? 1u : 0u; return true;
		case ValueOpcode::FPOrdLessThanEqual32: *out = KytySrtAot::FPOrdCompare(a, b, 0u, n->aux); return true;
		case ValueOpcode::FPOrdGreaterThanEqual32: *out = KytySrtAot::FPOrdCompare(a, b, 1u, n->aux); return true;
		case ValueOpcode::BitFieldUExtract: return KytySrtAot::BitFieldUExtract(a, b, c, *out);
		case ValueOpcode::BitFieldSExtract: return KytySrtAot::BitFieldSExtract(a, b, c, *out);
		case ValueOpcode::BitFieldInsert: return KytySrtAot::BitFieldInsert(a, b, c, d, *out);
		case ValueOpcode::SGreaterThanEqual32:
			*out = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >= std::bit_cast<int32_t>(static_cast<uint32_t>(b));
			return true;
		default: return false;
	}
}

uint32_t Arity(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::FPTrunc32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPIsNan32: return 1;
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::IEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::LogicalXor: return 2;
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract: return 3;
		case ValueOpcode::BitFieldInsert: return 4;
		default: return 0;
	}
}

// Builds a leaf from the compiled plan's nodes: each node once per read policy, operands first.
class Builder {
public:
	Builder(const CompiledSrtPlan& plan, Leaf& leaf, bool clean_evaluator)
	    : m_plan(plan), m_leaf(leaf), m_clean_evaluator(clean_evaluator) {
		for (auto& memo: m_memo) memo.assign(plan.nodes.size(), Invalid);
		for (auto& visiting: m_visiting) visiting.assign(plan.nodes.size(), 0);
	}

	// The leaf node of plan node `id` read with the strict reader or not; Invalid when the linear form cannot hold it.
	// Under `lazy` (an operand the interpreter evaluates only on some values) nothing that may fail or read may be
	// added: then the native evaluation fails exactly when the interpreter's does, and reads only what it reads. An eager read of
	// what the interpreter would not read could fail its resolution's read log.
	uint32_t Add(uint32_t id, bool clean, bool lazy, uint32_t depth = 0) {
		if (id >= m_plan.nodes.size() || depth > 256u || m_leaf.nodes.size() >= 8192u) return Invalid;
		const auto& n = m_plan.nodes[id];
		switch (n.kind) {
			case CompiledSrtPlan::NodeKind::Constant: return Push({.kind = Kind::Immediate, .imm = n.imm});
			case CompiledSrtPlan::NodeKind::Invalid: return Invalid;
			// (A user data index past the user data fails: not under lazy, see below.)
			case CompiledSrtPlan::NodeKind::UserData: return lazy ? Invalid : Push({.kind = Kind::User, .imm = n.imm});
			default: break;
		}
		// The interpreter evaluates a clean node in its clean evaluator, every read below it strict.
		if (n.clean && m_clean_evaluator && !clean) return Add(id, true, lazy, depth + 1u);
		auto& memo = m_memo[clean ? 1 : 0][id];
		if (memo != Invalid) return memo;
		auto& visiting = m_visiting[clean ? 1 : 0][id];
		if (visiting != 0u) return Invalid;
		visiting           = 1u;
		const auto result  = AddInst(id, n, clean, lazy, depth);
		visiting           = 0u;
		if (result != Invalid) {
			memo = result;
			m_journal.push_back({clean, id});
		}
		return result;
	}

	// Undoes the nodes and memos added since `mark` (a descriptor that cannot be built).
	struct Mark {
		size_t nodes = 0, journal = 0;
	};
	[[nodiscard]] Mark Save() const { return {m_leaf.nodes.size(), m_journal.size()}; }
	void Restore(const Mark& mark) {
		m_leaf.nodes.resize(mark.nodes);
		while (m_journal.size() > mark.journal) {
			const auto [clean, id] = m_journal.back();
			m_memo[clean ? 1 : 0][id] = Invalid;
			m_journal.pop_back();
		}
	}

private:
	uint32_t Push(Node node) {
		m_leaf.nodes.push_back(node);
		return static_cast<uint32_t>(m_leaf.nodes.size() - 1u);
	}

	uint32_t AddInst(uint32_t id, const CompiledSrtPlan::Node& n, bool clean, bool lazy, uint32_t depth) {
		const auto arg = [&](uint32_t index, bool arg_clean, bool arg_lazy) {
			const auto child = n.args[index];
			return child == CompiledSrtPlan::NoNode ? Invalid : Add(child, arg_clean, arg_lazy, depth + 1u);
		};
		Node node {.op = n.op, .aux = n.aux, .aux2 = n.aux2, .clean = clean, .plan_node = id};
		if (n.shape == CompiledSrtPlan::Shape::LeafAddressRead) {
			if (lazy) return Invalid;
			node.kind    = Kind::LeafRead;
			node.imm     = n.imm;
			node.slot    = id < m_plan.read_slots.size() ? m_plan.read_slots[id] : NoSrtSlot;
			node.args[0] = arg(0, clean, lazy);
			node.args[1] = arg(1, clean, lazy);
			if (node.args[0] == Invalid || node.args[1] == Invalid) return Invalid;
			return Push(node);
		}
		if (n.op == ValueOpcode::LoadAddressU32 || n.op == ValueOpcode::ReadConstBuffer) {
			const bool buffer = n.op == ValueOpcode::ReadConstBuffer;
			if (lazy || n.shape != CompiledSrtPlan::Shape::RawRead || (buffer && n.handle_args != 4u)) return Invalid;
			node.kind    = Kind::RawRead;
			node.imm     = n.imm;
			node.slot    = id < m_plan.read_slots.size() ? m_plan.read_slots[id] : NoSrtSlot;
			// In the interpreter's order: the address, the offset, then the V#'s other words.
			node.args[0] = arg(0, clean, lazy);
			node.args[1] = arg(1, clean, lazy);
			node.args[4] = arg(4, clean, lazy);
			if (node.args[0] == Invalid || node.args[1] == Invalid || node.args[4] == Invalid) return Invalid;
			if (buffer) {
				node.args[2] = arg(2, clean, lazy);
				node.args[3] = arg(3, clean, lazy);
				if (node.args[2] == Invalid || node.args[3] == Invalid) return Invalid;
			}
			return Push(node);
		}
		switch (n.op) {
			case ValueOpcode::GetShaderBase: return Push({.kind = Kind::Base, .clean = clean, .plan_node = id});
			case ValueOpcode::Phi:
				if (n.shape == CompiledSrtPlan::Shape::ControlPhi) return Invalid;
				return arg(0, clean, lazy);
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::ConditionRef: return arg(0, clean, lazy);
			case ValueOpcode::CompositeExtractU64:
				if (n.aux2 == 0u) return Invalid;
				node.kind    = Kind::Op;
				node.args[0] = arg(0, clean, lazy);
				return node.args[0] == Invalid ? Invalid : Push(node);
			case ValueOpcode::CompositeExtractU32x2:
				if (n.aux2 == 1u) return n.aux < 2u ? arg(n.aux, clean, lazy) : Invalid;
				if (n.aux2 != 2u) return Invalid;
				node.kind    = Kind::Op;
				node.args[0] = arg(0, clean, lazy);
				node.args[1] = arg(1, clean, lazy);
				return node.args[0] == Invalid || node.args[1] == Invalid ? Invalid : Push(node);
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32:
				// The predicate is evaluated in the clean evaluator when there is one, then one of the two values.
				node.kind    = Kind::Select;
				node.args[0] = arg(0, clean || m_clean_evaluator, lazy);
				node.args[1] = arg(1, clean, true);
				node.args[2] = arg(2, clean, true);
				return node.args[0] == Invalid || node.args[1] == Invalid || node.args[2] == Invalid ? Invalid
				                                                                                       : Push(node);
			case ValueOpcode::LogicalAnd:
			case ValueOpcode::LogicalOr:
			case ValueOpcode::ULessThanEqual32:
				// Short-circuits: the right operand is not always evaluated (and an unknown operand of an And or Or
				// does not fail it when the other decides it).
				node.kind    = Kind::Op;
				node.args[0] = arg(0, clean, lazy || n.op != ValueOpcode::ULessThanEqual32);
				node.args[1] = arg(1, clean, true);
				return node.args[0] == Invalid || node.args[1] == Invalid ? Invalid : Push(node);
			default: break;
		}
		const auto arity = Arity(n.op);
		if (arity == 0u) return Invalid;
		if (lazy && (n.op == ValueOpcode::ConvertU32F32 || n.op == ValueOpcode::FPRecipIFlag32 ||
		             n.op == ValueOpcode::BitFieldUExtract || n.op == ValueOpcode::BitFieldSExtract ||
		             n.op == ValueOpcode::BitFieldInsert))
			return Invalid; // operations that may fail
		node.kind = Kind::Op;
		for (uint32_t i = 0; i < arity; ++i) {
			node.args[i] = arg(i, clean, lazy);
			if (node.args[i] == Invalid) return Invalid;
		}
		return Push(node);
	}

	const CompiledSrtPlan&                   m_plan;
	Leaf&                                    m_leaf;
	bool                                     m_clean_evaluator;
	std::array<std::vector<uint32_t>, 2>     m_memo;
	std::array<std::vector<uint8_t>, 2>      m_visiting;
	std::vector<std::pair<bool, uint32_t>>   m_journal;
};

// Native code for a leaf: rdi = LinearRuntime, rsi = values; each node's value in values[i].
bool Generate(Leaf& leaf) {
	using namespace Xbyak::util;
	Xbyak::ClearError();
	auto code = std::make_shared<Xbyak::CodeGenerator>(4096, Xbyak::AutoGrow);
	if (Xbyak::GetError() != 0) {
		Xbyak::ClearError();
		return false;
	}
	auto&        c = *code;
	Xbyak::Label fail, done;
	c.push(r12);
	c.push(r13);
	c.sub(rsp, 8); // entry rsp is 8 mod 16: aligned for the helper calls
	c.mov(r12, rdi);
	c.mov(r13, rsi);
	const auto value = [&](uint32_t index) { return c.qword[r13 + index * 8u]; };
	const auto call  = [&](const void* function) {
		c.mov(rax, reinterpret_cast<uint64_t>(function));
		c.call(rax);
		c.test(al, al);
		c.jz(fail, Xbyak::CodeGenerator::T_NEAR);
	};
	for (uint32_t i = 0; i < leaf.nodes.size(); ++i) {
		const auto& n = leaf.nodes[i];
		switch (n.kind) {
			case Kind::Immediate:
				c.mov(rax, n.imm);
				c.mov(value(i), rax);
				continue;
			case Kind::User:
				c.mov(rdi, r12);
				c.mov(rsi, n.imm);
				c.lea(rdx, value(i));
				call(reinterpret_cast<const void*>(&ReadUserNode));
				continue;
			case Kind::Base:
				c.mov(rax, c.qword[r12 + offsetof(LinearRuntime, base)]);
				c.mov(value(i), rax);
				continue;
			case Kind::LeafRead:
			case Kind::RawRead:
				c.mov(rdi, r12);
				c.mov(rsi, reinterpret_cast<uint64_t>(&n));
				c.mov(rdx, r13);
				c.lea(rcx, value(i));
				call(n.kind == Kind::LeafRead ? reinterpret_cast<const void*>(&ReadLeafNode)
				                              : reinterpret_cast<const void*>(&ReadRawNode));
				continue;
			case Kind::Select:
				c.mov(rax, value(n.args[1]));
				c.mov(rcx, value(n.args[2]));
				c.cmp(value(n.args[0]), 0);
				c.cmovz(rax, rcx);
				c.mov(value(i), rax);
				continue;
			case Kind::Op: break;
		}
		// Operations: a, b in rax, rcx.
		const bool inline_op = [&] {
			switch (n.op) {
				case ValueOpcode::UMulHi:
				case ValueOpcode::UMax32:
				case ValueOpcode::ConvertF32U32:
				case ValueOpcode::ConvertU32F32:
				case ValueOpcode::FPMul32:
				case ValueOpcode::FPTrunc32:
				case ValueOpcode::FPRecipIFlag32:
				case ValueOpcode::FPIsNan32:
				case ValueOpcode::FPOrdLessThanEqual32:
				case ValueOpcode::FPOrdGreaterThanEqual32:
				case ValueOpcode::BitFieldUExtract:
				case ValueOpcode::BitFieldSExtract:
				case ValueOpcode::BitFieldInsert:
				case ValueOpcode::SGreaterThanEqual32: return false;
				default: return true;
			}
		}();
		if (!inline_op) {
			c.mov(rdi, reinterpret_cast<uint64_t>(&n));
			c.mov(rsi, r13);
			c.lea(rdx, value(i));
			call(reinterpret_cast<const void*>(&OpNode));
			continue;
		}
		c.mov(rax, value(n.args[0]));
		if (n.args[1] != Invalid) c.mov(rcx, value(n.args[1]));
		const auto compare = [&](auto&& set) {
			c.cmp(eax, ecx);
			set(al);
			c.movzx(eax, al);
		};
		switch (n.op) {
			case ValueOpcode::IAdd32: c.add(eax, ecx); break;
			case ValueOpcode::IAdd64: c.add(rax, rcx); break;
			case ValueOpcode::ISub32: c.sub(eax, ecx); break;
			case ValueOpcode::ISub64: c.sub(rax, rcx); break;
			case ValueOpcode::IMul32: c.imul(eax, ecx); break;
			case ValueOpcode::IMul64: c.imul(rax, rcx); break;
			case ValueOpcode::UMin32:
				c.cmp(eax, ecx);
				c.cmova(eax, ecx);
				c.mov(eax, eax);
				break;
			case ValueOpcode::BitwiseAnd32: c.and_(eax, ecx); break;
			case ValueOpcode::BitwiseAnd64: c.and_(rax, rcx); break;
			case ValueOpcode::BitwiseOr32: c.or_(eax, ecx); break;
			case ValueOpcode::BitwiseXor32: c.xor_(eax, ecx); break;
			case ValueOpcode::BitwiseNot32: c.not_(eax); break;
			// x86 masks the count as the interpreter does (& 31, & 63).
			case ValueOpcode::ShiftLeftLogical32: c.shl(eax, cl); break;
			case ValueOpcode::ShiftLeftLogical64: c.shl(rax, cl); break;
			case ValueOpcode::ShiftRightLogical32: c.shr(eax, cl); break;
			case ValueOpcode::ShiftRightLogical64: c.shr(rax, cl); break;
			case ValueOpcode::ShiftRightArithmetic32: c.sar(eax, cl); break;
			case ValueOpcode::ShiftRightArithmetic64: c.sar(rax, cl); break;
			case ValueOpcode::CompositeConstructU64:
				c.mov(eax, eax);
				c.mov(ecx, ecx);
				c.shl(rcx, 32);
				c.or_(rax, rcx);
				break;
			case ValueOpcode::CompositeExtractU64:
				if (n.aux != 0u) c.shr(rax, 32);
				c.mov(eax, eax);
				break;
			case ValueOpcode::CompositeExtractU32x2: // the 64-bit sum of the two words: its low word or its carry
				c.mov(eax, eax);
				c.mov(ecx, ecx);
				c.add(rax, rcx);
				if (n.aux != 0u) c.shr(rax, 32);
				c.mov(eax, eax);
				break;
			case ValueOpcode::IEqual32: compare([&](auto r) { c.sete(r); }); break;
			case ValueOpcode::INotEqual32: compare([&](auto r) { c.setne(r); }); break;
			case ValueOpcode::ULessThan32: compare([&](auto r) { c.setb(r); }); break;
			case ValueOpcode::ULessThanEqual32: compare([&](auto r) { c.setbe(r); }); break;
			case ValueOpcode::UGreaterThan32: compare([&](auto r) { c.seta(r); }); break;
			case ValueOpcode::LogicalNot:
				c.test(rax, rax);
				c.setz(al);
				c.movzx(eax, al);
				break;
			case ValueOpcode::LogicalAnd:
			case ValueOpcode::LogicalOr:
			case ValueOpcode::LogicalXor:
				c.test(rax, rax);
				c.setne(al);
				c.test(rcx, rcx);
				c.setne(cl);
				if (n.op == ValueOpcode::LogicalAnd) c.and_(al, cl);
				else if (n.op == ValueOpcode::LogicalOr) c.or_(al, cl);
				else c.xor_(al, cl);
				c.movzx(eax, al);
				break;
			default: return false;
		}
		c.mov(value(i), rax);
	}
	c.mov(eax, 1);
	c.jmp(done, Xbyak::CodeGenerator::T_NEAR);
	c.L(fail);
	c.xor_(eax, eax);
	c.L(done);
	c.add(rsp, 8);
	c.pop(r13);
	c.pop(r12);
	c.ret();
	if (Xbyak::GetError() != 0) {
		Xbyak::ClearError();
		return false;
	}
	c.readyRE();
	if (Xbyak::GetError() != 0) {
		Xbyak::ClearError();
		return false;
	}
	leaf.function = c.getCode<Leaf::Function>();
	leaf.code     = std::move(code);
	return true;
}

// ---- Ahead-of-time library (chenxiao07's native-resource-aot.h) ----

std::string EmitAot(const Leaf& leaf) {
	std::ostringstream out;
	const auto         v = [](uint32_t i) { return "v" + std::to_string(i); };
	for (uint32_t i = 0; i < leaf.nodes.size(); ++i) {
		const auto& n  = leaf.nodes[i];
		const auto  a  = n.args[0] != Invalid ? v(n.args[0]) : std::string("0");
		const auto  b  = n.args[1] != Invalid ? v(n.args[1]) : std::string("0");
		const auto  c3 = n.args[2] != Invalid ? v(n.args[2]) : std::string("0");
		const auto  d  = n.args[3] != Invalid ? v(n.args[3]) : std::string("0");
		const auto  a32 = "uint32_t(" + a + ")", b32 = "uint32_t(" + b + ")";
		out << "uint64_t " << v(i) << " = 0;\n";
		std::string expression;
		const auto  fallible = [&](const std::string& call) { out << "if (!" << call << ") return false;\n"; };
		switch (n.kind) {
			case Kind::Immediate: expression = std::to_string(n.imm) + "ull"; break;
			case Kind::User:
				out << "if (r->user_count <= " << n.imm << "ull) return false;\n";
				expression = "r->user_data[" + std::to_string(n.imm) + "]";
				break;
			case Kind::Base: expression = "r->shader_base"; break;
			case Kind::LeafRead:
				fallible("KytySrtAot::ReadLeaf(r, " + a + ", " + b + ", " + std::to_string(static_cast<int64_t>(n.imm)) +
				         "ll, " + std::to_string(n.plan_node) + "u, " + std::to_string(n.clean ? 1 : 0) + ", " + v(i) + ")");
				break;
			case Kind::RawRead:
				fallible("KytySrtAot::ReadRaw(r, " + std::to_string(n.op == ValueOpcode::ReadConstBuffer ? 1 : 0) + ", " + a +
				         ", " + b + ", " + c3 + ", " + v(n.args[4]) + ", " + std::to_string(static_cast<int64_t>(n.imm)) +
				         "ll, " + std::to_string(n.plan_node) + "u, " + std::to_string(n.clean ? 1 : 0) + ", " + v(i) + ")");
				break;
			case Kind::Select: expression = a + " != 0 ? " + b + " : " + c3; break;
			case Kind::Op:
				switch (n.op) {
					case ValueOpcode::IAdd32: expression = "uint32_t(" + a32 + " + " + b32 + ")"; break;
					case ValueOpcode::IAdd64: expression = a + " + " + b; break;
					case ValueOpcode::ISub32: expression = "uint32_t(" + a32 + " - " + b32 + ")"; break;
					case ValueOpcode::ISub64: expression = a + " - " + b; break;
					case ValueOpcode::IMul32: expression = "uint32_t(" + a32 + " * " + b32 + ")"; break;
					case ValueOpcode::IMul64: expression = a + " * " + b; break;
					case ValueOpcode::UMulHi: expression = "(uint64_t(" + a32 + ") * " + b32 + ") >> 32"; break;
					case ValueOpcode::UMin32: expression = a32 + " < " + b32 + " ? " + a32 + " : " + b32; break;
					case ValueOpcode::UMax32: expression = a32 + " > " + b32 + " ? " + a32 + " : " + b32; break;
					case ValueOpcode::ConvertF32U32: expression = "KytySrtAot::Bits(float(" + a32 + "))"; break;
					case ValueOpcode::ConvertU32F32: fallible("KytySrtAot::ConvertU32F32(" + a + ", " + v(i) + ")"); break;
					case ValueOpcode::FPMul32:
						expression = "KytySrtAot::Bits(KytySrtAot::F32(" + a + ") * KytySrtAot::F32(" + b + "))";
						break;
					case ValueOpcode::FPTrunc32: expression = "KytySrtAot::Bits(std::trunc(KytySrtAot::F32(" + a + ")))"; break;
					case ValueOpcode::FPRecipIFlag32: fallible("KytySrtAot::FPRecipIFlag32(" + a + ", " + v(i) + ")"); break;
					case ValueOpcode::FPIsNan32: expression = "std::isnan(KytySrtAot::F32(" + a + ")) ? 1u : 0u"; break;
					case ValueOpcode::FPOrdLessThanEqual32:
					case ValueOpcode::FPOrdGreaterThanEqual32:
						expression = "KytySrtAot::FPOrdCompare(" + a + ", " + b + ", " +
						             std::to_string(n.op == ValueOpcode::FPOrdGreaterThanEqual32 ? 1 : 0) + "u, " +
						             std::to_string(n.aux) + "u)";
						break;
					case ValueOpcode::BitwiseAnd32: expression = a32 + " & " + b32; break;
					case ValueOpcode::BitwiseAnd64: expression = a + " & " + b; break;
					case ValueOpcode::BitwiseOr32: expression = a32 + " | " + b32; break;
					case ValueOpcode::BitwiseXor32: expression = a32 + " ^ " + b32; break;
					case ValueOpcode::BitwiseNot32: expression = "uint32_t(~" + a32 + ")"; break;
					case ValueOpcode::ShiftLeftLogical32: expression = "uint32_t(" + a32 + " << (" + b + " & 31))"; break;
					case ValueOpcode::ShiftLeftLogical64: expression = a + " << (" + b + " & 63)"; break;
					case ValueOpcode::ShiftRightLogical32: expression = a32 + " >> (" + b + " & 31)"; break;
					case ValueOpcode::ShiftRightLogical64: expression = a + " >> (" + b + " & 63)"; break;
					case ValueOpcode::ShiftRightArithmetic32:
						expression = "uint32_t(int32_t(" + a32 + ") >> (" + b + " & 31))";
						break;
					case ValueOpcode::ShiftRightArithmetic64:
						expression = "uint64_t(int64_t(" + a + ") >> (" + b + " & 63))";
						break;
					case ValueOpcode::CompositeConstructU64: expression = "uint64_t(" + a32 + ") | (uint64_t(" + b32 + ") << 32)"; break;
					case ValueOpcode::CompositeExtractU64:
						expression = n.aux != 0u ? "uint32_t(" + a + " >> 32)" : a32;
						break;
					case ValueOpcode::CompositeExtractU32x2:
						expression = n.aux != 0u ? "(uint64_t(" + a32 + ") + " + b32 + ") >> 32"
						                         : "uint32_t(uint64_t(" + a32 + ") + " + b32 + ")";
						break;
					case ValueOpcode::IEqual32: expression = a32 + " == " + b32; break;
					case ValueOpcode::INotEqual32: expression = a32 + " != " + b32; break;
					case ValueOpcode::ULessThan32: expression = a32 + " < " + b32; break;
					case ValueOpcode::ULessThanEqual32: expression = a32 + " <= " + b32; break;
					case ValueOpcode::UGreaterThan32: expression = a32 + " > " + b32; break;
					case ValueOpcode::SGreaterThanEqual32: expression = "int32_t(" + a32 + ") >= int32_t(" + b32 + ")"; break;
					case ValueOpcode::LogicalAnd: expression = "(" + a + " != 0) && (" + b + " != 0)"; break;
					case ValueOpcode::LogicalOr: expression = "(" + a + " != 0) || (" + b + " != 0)"; break;
					case ValueOpcode::LogicalXor: expression = "(" + a + " != 0) != (" + b + " != 0)"; break;
					case ValueOpcode::LogicalNot: expression = a + " == 0"; break;
					case ValueOpcode::BitFieldUExtract:
						fallible("KytySrtAot::BitFieldUExtract(" + a + ", " + b + ", " + c3 + ", " + v(i) + ")");
						break;
					case ValueOpcode::BitFieldSExtract:
						fallible("KytySrtAot::BitFieldSExtract(" + a + ", " + b + ", " + c3 + ", " + v(i) + ")");
						break;
					case ValueOpcode::BitFieldInsert:
						fallible("KytySrtAot::BitFieldInsert(" + a + ", " + b + ", " + c3 + ", " + d + ", " + v(i) + ")");
						break;
					default: return {};
				}
				break;
		}
		if (!expression.empty()) out << v(i) << " = uint64_t(" << expression << ");\n";
		out << "values[" << i << "] = " << v(i) << ";\n";
	}
	out << "return true;\n";
	return out.str();
}

void* OpenAotLibrary() {
	static void* module = [] () -> void* {
		std::string path;
		if (const char* value = std::getenv("KYTY_SRT_AOT_LIBRARY"); value != nullptr && value[0] != '\0') {
			path = value;
		} else {
#if defined(_WIN32)
			char exe[MAX_PATH] {};
			if (GetModuleFileNameA(nullptr, exe, MAX_PATH) == 0) return nullptr;
			path = (std::filesystem::path(exe).parent_path() / "kyty_srt_aot.dll").string();
#else
			path = "kyty_srt_aot.so";
#endif
			std::error_code error;
			if (!std::filesystem::exists(path, error)) return nullptr;
		}
#if defined(_WIN32)
		auto* opened = reinterpret_cast<void*>(LoadLibraryA(path.c_str()));
#else
		auto* opened = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
		std::printf("SRT AOT library %s: %s\n", path.c_str(), opened != nullptr ? "loaded" : "failed to load");
		return opened;
	}();
	return module;
}

void* AotSymbol(void* module, const std::string& name) {
#if defined(_WIN32)
	return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module), name.c_str()));
#else
	return dlsym(module, name.c_str());
#endif
}

// A leaf's identity in the library: two FNV-1a hashes of its nodes and the ABI version (EmitAot() is a function of
// them: a change to what it emits bumps KytySrtAot::AbiVersion). Cheap enough for every plan built.
std::pair<uint64_t, uint64_t> AotKey(const Leaf& leaf) {
	uint64_t   a = 14695981039346656037ull ^ KytySrtAot::AbiVersion, b = 0x9e3779b97f4a7c15ull ^ leaf.nodes.size();
	const auto mix = [&](uint64_t word) {
		for (int i = 0; i < 8; ++i) {
			const auto byte = static_cast<uint8_t>(word >> (i * 8));
			a               = (a ^ byte) * 1099511628211ull;
			b               = (b ^ byte) * 0x100000001b3ull + 0x7f4a7c15u;
		}
	};
	for (const auto& n: leaf.nodes) {
		mix(static_cast<uint64_t>(n.kind) | static_cast<uint64_t>(n.op) << 8u | uint64_t {n.aux} << 24u |
		    uint64_t {n.aux2} << 32u | uint64_t {n.clean ? 1u : 0u} << 40u);
		for (const auto arg: n.args) mix(arg);
		mix(n.imm);
		mix(n.plan_node);
	}
	return {a, b};
}

// Exports the leaf's source (KYTY_SRT_AOT_EXPORT=<dir>) and takes its function from the library when that holds a leaf
// of the same nodes.
void PrepareAot(Leaf& leaf) {
	if (!Switch("KYTY_SRT_AOT", true)) return;
	const char* directory = std::getenv("KYTY_SRT_AOT_EXPORT");
	const bool  exporting = directory != nullptr && directory[0] != '\0';
	void*       module    = OpenAotLibrary();
	if (!exporting && module == nullptr) return;
	const auto [a, b] = AotKey(leaf);
	char symbol[64];
	std::snprintf(symbol, sizeof(symbol), "kyty_srt_aot_%016llx%016llx", static_cast<unsigned long long>(a),
	              static_cast<unsigned long long>(b));
	static std::mutex mutex;
	std::lock_guard   lock(mutex);
	if (exporting) {
		std::error_code error;
		std::filesystem::create_directories(directory, error);
		const auto path = std::filesystem::path(directory) / (std::string(symbol) + ".cpp");
		if (!error && !std::filesystem::exists(path, error)) {
			if (const auto body = EmitAot(leaf); !body.empty()) {
				std::ofstream file(path);
				file << "#include \"SrtAotAbi.h\"\n"
				     << "static_assert(KytySrtAot::AbiVersion == " << KytySrtAot::AbiVersion << ");\n"
				     << "extern \"C\" KYTY_SRT_AOT_EXPORT uint64_t " << symbol << "_nodes() { return "
				     << leaf.nodes.size() << "u; }\n"
				     << "extern \"C\" KYTY_SRT_AOT_EXPORT bool " << symbol
				     << "(const KytySrtAot::Runtime* r, uint64_t* values) {\n"
				     << body << "}\n";
			}
		}
	}
	if (module == nullptr) return;
	const auto nodes    = reinterpret_cast<uint64_t (*)()>(AotSymbol(module, std::string(symbol) + "_nodes"));
	const auto function = reinterpret_cast<KytySrtAot::Function>(AotSymbol(module, symbol));
	static uint32_t plans = 0, found = 0;
	++plans;
	if (nodes != nullptr && function != nullptr && nodes() == leaf.nodes.size()) {
		leaf.aot = reinterpret_cast<bool (*)(const void*, uint64_t*)>(function);
		++found;
	}
	if (plans % 1024u == 0u) std::printf("SRT AOT: %u of %u plans from the library\n", found, plans);
}

bool RunLeaf(const Leaf& leaf, const SrtRuntime& runtime, std::vector<uint64_t>& values) {
	values.resize(leaf.nodes.size());
	if (leaf.aot != nullptr) {
		const KytySrtAot::Runtime aot {.user_data   = runtime.user_data.data(),
		                               .user_count  = runtime.user_data.size(),
		                               .shader_base = runtime.shader_base,
		                               .context     = const_cast<SrtRuntime*>(&runtime),
		                               .read        = &AotRead};
		return leaf.aot(&aot, values.data());
	}
	const LinearRuntime linear {.user       = runtime.user_data.data(),
	                            .user_count = runtime.user_data.size(),
	                            .base       = runtime.shader_base,
	                            .runtime    = &runtime};
	return leaf.function(&linear, values.data());
}

// The sources whose descriptors a materialization evaluates directly (not through an indirect descriptor's table).
std::vector<uint8_t> DirectSources(const ResourcePlan& program) {
	std::vector<uint8_t> used(program.descriptor_sources.size(), 0);
	const auto direct = [&](uint32_t source) {
		if (source < used.size() && !program.descriptor_sources[source].indirect_descriptor.has_value()) used[source] = 1;
	};
	for (const auto& buffer: program.info.buffers) direct(buffer.source);
	for (const auto& image: program.info.images) direct(image.source);
	for (const auto& sampler: program.info.samplers) {
		if (sampler.source >= used.size()) continue;
		const auto& indirect = program.descriptor_sources[sampler.source].indirect_descriptor;
		if (!indirect.has_value()) used[sampler.source] = 1;
		else if (!indirect->sources.empty()) direct(indirect->sources[0]);
	}
	return used;
}

// A leaf for the given reached blocks (null: no control flow) and active sources.
std::shared_ptr<Leaf> BuildLeaf(const ResourcePlan& program, const CompiledSrtPlan& plan, bool clean_evaluator,
                                const std::vector<uint8_t>* reached, const std::vector<uint8_t>& active,
                                const std::vector<uint8_t>& used) {
	auto    leaf = std::make_shared<Leaf>();
	Builder builder(plan, *leaf, clean_evaluator);
	leaf->active_sources = active;
	leaf->flat_words.assign(plan.srt_reads.size(), Invalid);
	const auto add_read = [&](uint32_t slot) {
		if (slot >= plan.srt_reads.size()) return false;
		const auto& read = plan.srt_reads[slot];
		// A clean read needs the clean evaluator (and the strict reader, checked when it runs).
		if (read.clean && !clean_evaluator) return false;
		if (read.flat_offset >= leaf->flat_words.size()) return false;
		const auto node = builder.Add(read.node, read.clean, false);
		if (node == Invalid) return false;
		leaf->flat_words[read.flat_offset] = node;
		return true;
	};
	if (reached == nullptr) {
		for (uint32_t slot = 0; slot < plan.srt_reads.size(); ++slot)
			if (!add_read(slot)) return nullptr;
	} else {
		for (uint32_t block = 0; block < program.control_flow.size(); ++block) {
			if (!(*reached)[block]) continue;
			for (const auto slot: program.control_flow[block].srt_reads)
				if (!add_read(slot)) return nullptr;
		}
	}
	leaf->descriptor_words.assign(plan.descriptors.size() * 8u, Invalid);
	leaf->descriptor_built.assign(plan.descriptors.size(), 0);
	for (uint32_t source = 0; source < plan.descriptors.size(); ++source) {
		if (!used[source] || (!active.empty() && !active[source])) continue;
		const auto& descriptor = plan.descriptors[source];
		const auto  mark       = builder.Save();
		bool        built      = true;
		for (uint32_t word = 0; word < descriptor.dword_count && built; ++word) {
			const auto node = descriptor.dwords[word] == CompiledSrtPlan::NoNode
			                      ? Invalid
			                      : builder.Add(descriptor.dwords[word], false, false);
			built           = node != Invalid;
			leaf->descriptor_words[source * 8u + word] = node;
		}
		if (built) {
			leaf->descriptor_built[source] = 1;
		} else {
			leaf->complete = false;
			builder.Restore(mark);
			std::fill_n(leaf->descriptor_words.begin() + source * 8u, 8u, Invalid);
		}
	}
	if (!Generate(*leaf)) return nullptr;
	PrepareAot(*leaf);
	return leaf;
}

std::shared_ptr<LinearSrtPlan> BuildPlan(const ResourcePlan& program, const CompiledSrtPlan& plan, bool clean_evaluator) {
	if (!plan.srt_plan_complete || !program.srt_plan_complete || !plan.control_phis.empty()) return nullptr;
	auto       result      = std::make_shared<LinearSrtPlan>();
	result->clean_evaluator = clean_evaluator;
	result->plan_nodes      = plan.nodes.size();
	// The interpreter evaluates more after the native evaluation (fill values, indirect descriptors, descriptors the
	// linear form does not hold): its memos then need the native values.
	result->seed = program.uniform_fill.fill.words != 0u ||
	               std::ranges::any_of(program.descriptor_sources,
	                                   [](const auto& source) { return source.indirect_descriptor.has_value(); });
	const auto used        = DirectSources(program);
	if (program.control_flow.empty()) {
		auto leaf = BuildLeaf(program, plan, clean_evaluator, nullptr, {}, used);
		if (leaf == nullptr) return nullptr;
		result->leaves.push_back(std::move(leaf));
		result->seed = result->seed || !result->leaves.front()->complete;
		return result;
	}
	// KYTY_SRT_PREDICATES: the conditions natively, then the leaf of the blocks they reach.
	if (!Switch("KYTY_SRT_PREDICATES", true)) return nullptr;
	const auto blocks = static_cast<uint32_t>(program.control_flow.size());
	if (blocks > 256u) return nullptr;
	for (uint32_t block = 0; block < blocks; ++block) {
		const auto& b = program.control_flow[block];
		if (std::ranges::any_of(b.successors, [&](uint32_t s) { return s >= blocks; }) ||
		    std::ranges::any_of(b.sources, [&](uint32_t s) { return s >= program.descriptor_sources.size(); }))
			return nullptr;
		const auto condition = block < plan.conditions.size() ? plan.conditions[block] : CompiledSrtPlan::NoNode;
		if (condition != CompiledSrtPlan::NoNode) {
			if (b.successors.size() != 2u) return nullptr;
			result->conditionals.push_back(block);
		}
	}
	if (result->conditionals.size() > 3u) return nullptr;
	// One program per condition: as the interpreter, only conditions of reached blocks are evaluated, and one that
	// cannot be evaluated follows both edges.
	for (const auto block: result->conditionals) {
		auto    predicate = std::make_shared<Leaf>();
		Builder builder(plan, *predicate, clean_evaluator);
		const auto node = builder.Add(plan.conditions[block], clean_evaluator, false);
		if (node == Invalid || !Generate(*predicate)) return nullptr;
		predicate->flat_words.push_back(node);
		PrepareAot(*predicate);
		result->predicates.push_back(std::move(predicate));
	}
	uint32_t combinations = 1;
	for (size_t i = 0; i < result->conditionals.size(); ++i) combinations *= 3u;
	std::vector<std::vector<uint8_t>> leaf_reached;
	for (uint32_t combination = 0; combination < combinations; ++combination) {
		std::vector<uint8_t> choice(blocks, 2u);
		auto                 encoded = combination;
		for (const auto block: result->conditionals) {
			choice[block] = static_cast<uint8_t>(encoded % 3u);
			encoded /= 3u;
		}
		std::vector<uint8_t>  reached(blocks, 0u);
		std::vector<uint32_t> pending {0u};
		while (!pending.empty()) {
			const auto index = pending.back();
			pending.pop_back();
			if (reached[index]) continue;
			reached[index]    = 1u;
			const auto& block = program.control_flow[index];
			if (choice[index] < 2u) pending.push_back(block.successors[choice[index]]);
			else pending.insert(pending.end(), block.successors.begin(), block.successors.end());
		}
		const auto known = std::ranges::find(leaf_reached, reached);
		if (known != leaf_reached.end()) {
			result->leaf_of_choice.push_back(static_cast<uint32_t>(known - leaf_reached.begin()));
			continue;
		}
		std::vector<uint8_t> active(program.descriptor_sources.size(), 1u);
		for (const auto& block: program.control_flow)
			for (const auto source: block.sources) active[source] = 0u;
		for (uint32_t index = 0; index < blocks; ++index)
			if (reached[index])
				for (const auto source: program.control_flow[index].sources) active[source] = 1u;
		auto leaf = BuildLeaf(program, plan, clean_evaluator, &reached, active, used);
		if (leaf == nullptr) return nullptr;
		result->leaf_of_choice.push_back(static_cast<uint32_t>(result->leaves.size()));
		result->leaves.push_back(std::move(leaf));
		leaf_reached.push_back(std::move(reached));
	}
	for (const auto& leaf: result->leaves) result->seed = result->seed || !leaf->complete;
	return result;
}

} // namespace

static std::atomic<int> g_linear_override {-1};

void SetLinearSrtEnabledForTesting(bool enabled) {
	g_linear_override.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

bool LinearSrtEnabled() {
	static const bool enabled = Switch("KYTY_SRT_NATIVE", true);
	const auto        forced  = g_linear_override.load(std::memory_order_relaxed);
	return forced < 0 ? enabled : forced != 0;
}

bool LinearSrtVerifyEnabled() {
	static const bool enabled = Switch("KYTY_SRT_LINEAR_VERIFY", false);
	return enabled;
}

const LinearSrtPlan* LinearSrtOf(const ResourcePlan& program, bool clean_evaluator) {
	auto& slot = program.linear_srt[clean_evaluator ? 1 : 0];
	if (!slot.built) {
		slot.built = true;
		if (const auto* plan = CompileSrtPlan(program)) slot.plan = BuildPlan(program, *plan, clean_evaluator);
	}
	return slot.plan.get();
}

LinearSrtResult EvaluateLinearSrt(const ResourcePlan& program, const LinearSrtPlan& plan, const SrtRuntime& runtime,
                                  std::vector<uint32_t>& flat, std::vector<DescriptorValue>& descriptors,
                                  std::vector<uint8_t>& descriptor_valid, LinearSrtSeed seed, void* seed_context) {
	const auto seed_leaf = [&](const Leaf& leaf, const std::vector<uint64_t>& computed, size_t count) {
		if (!plan.seed) return;
		for (size_t i = 0; i < count; ++i) {
			const auto& n = leaf.nodes[i];
			if (n.plan_node != Invalid) seed(seed_context, n.clean, n.plan_node, computed[i]);
		}
	};
	thread_local std::vector<uint64_t> values;
	const Leaf*                        leaf = nullptr;
	// Clean reads need the strict reader (CompiledSrtWalker::RefreshFlatBuffer() fails without it).
	if (plan.clean_evaluator && runtime.read_specialization_memory == nullptr) return LinearSrtResult::NotApplicable;
	t_read_memo_active = !plan.predicates.empty();
	if (t_read_memo_active) BeginReads(plan.plan_nodes);
	if (plan.predicates.empty()) {
		leaf = plan.leaves.front().get();
	} else {
		// The reached blocks, as CompiledSrtWalker::RefreshFlatBuffer() finds them: a condition evaluated only in a
		// reached block; one that cannot be evaluated follows both edges.
		const auto blocks = program.control_flow.size();
		thread_local std::vector<uint8_t>  reached;
		thread_local std::vector<uint32_t> pending;
		thread_local std::vector<uint64_t> predicate_values;
		reached.assign(blocks, 0u);
		pending.assign(1u, 0u);
		uint32_t combination = 0, weight = 1;
		std::array<uint8_t, 3> choices {2u, 2u, 2u};
		while (!pending.empty()) {
			const auto index = pending.back();
			pending.pop_back();
			if (reached[index]) continue;
			reached[index]    = 1u;
			const auto& block = program.control_flow[index];
			const auto  k     = std::ranges::find(plan.conditionals, index);
			if (k == plan.conditionals.end()) {
				pending.insert(pending.end(), block.successors.begin(), block.successors.end());
				continue;
			}
			const auto position = static_cast<size_t>(k - plan.conditionals.begin());
			const auto& predicate = *plan.predicates[position];
			if (RunLeaf(predicate, runtime, predicate_values)) {
				seed_leaf(predicate, predicate_values, predicate.nodes.size());
				const bool taken  = predicate_values[predicate.flat_words[0]] != 0u;
				choices[position] = taken ? 0u : 1u;
				pending.push_back(block.successors[taken ? 0u : 1u]);
			} else {
				pending.insert(pending.end(), block.successors.begin(), block.successors.end());
			}
		}
		for (size_t position = 0; position < plan.conditionals.size(); ++position) {
			combination += choices[position] * weight;
			weight *= 3u;
		}
		leaf = plan.leaves[plan.leaf_of_choice[combination]].get();
	}
	const bool ran_ok = RunLeaf(*leaf, runtime, values);
	if (!ran_ok) return LinearSrtResult::Failed;
	seed_leaf(*leaf, values, leaf->nodes.size());
	flat.assign(leaf->flat_words.size(), 0u);
	for (size_t slot = 0; slot < flat.size(); ++slot)
		if (leaf->flat_words[slot] != Invalid) flat[slot] = static_cast<uint32_t>(values[leaf->flat_words[slot]]);
	const auto sources = leaf->descriptor_built.size();
	descriptors.resize(sources);
	descriptor_valid.assign(leaf->descriptor_built.begin(), leaf->descriptor_built.end());
	for (size_t source = 0; source < sources; ++source) {
		if (!descriptor_valid[source]) continue;
		auto& value       = descriptors[source];
		value             = {};
		value.dword_count = std::min<uint32_t>(program.descriptor_sources[source].dword_count, 8u);
		for (uint32_t word = 0; word < value.dword_count; ++word)
			value.dwords[word] = static_cast<uint32_t>(values[leaf->descriptor_words[source * 8u + word]]);
	}
	if (leaf->active_sources.empty()) program.active_sources.clear();
	else program.active_sources = leaf->active_sources;
	return LinearSrtResult::Evaluated;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
