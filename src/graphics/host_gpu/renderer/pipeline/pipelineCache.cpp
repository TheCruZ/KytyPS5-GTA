#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/programStore.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// The driver validates the data by pipelineCacheUUID and looks pipelines up by content, so
	// data from another emulator build is safe to reuse.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n", properties.vendorID,
	                   properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
}

// Execution thread: the ordinary loads of an SRT walk.
bool ReadGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	if (!Libs::LibKernel::Memory::TryReadAroundGpuWrites(address, values.data(),
	                                                     values.size_bytes())) {
		std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	}
	return true;
}

bool ReadAheadPlain(void* userdata, uint64_t address, std::span<uint32_t> values) {
	return !values.empty() && static_cast<GuestReadLog*>(userdata)->Read(
	                              address, values.data(), values.size_bytes(), false);
}

bool ReadAheadStrict(void* userdata, uint64_t address, std::span<uint32_t> values) {
	return !values.empty() && static_cast<GuestReadLog*>(userdata)->Read(address, values.data(),
	                                                                     values.size_bytes(), true);
}

bool ReadAheadShaderState(void* userdata, uint64_t address, void* data, uint64_t size) {
	return static_cast<GuestReadLog*>(userdata)->Read(address, data, size, false);
}

} // namespace

bool GuestReadLog::Read(uint64_t address, void* data, uint64_t size, bool strict) {
	if (size == 0 || size > UINT32_MAX ||
	    !Libs::LibKernel::Memory::TryReadBacking(address, data, size)) {
		m_failed = true;
		return false;
	}
	m_gpu_written = m_gpu_written || Libs::LibKernel::Memory::MayBeGpuWritten(address, size);
	const auto offset = static_cast<uint32_t>(m_bytes.size());
	m_bytes.insert(m_bytes.end(), static_cast<const uint8_t*>(data),
	               static_cast<const uint8_t*>(data) + size);
	// Contiguous reads of the same kind are checked as one (strict reads up to the size the
	// check reads at once).
	if (!m_entries.empty()) {
		auto& last = m_entries.back();
		if (last.strict == strict && last.address + last.size == address &&
		    last.offset + last.size == offset &&
		    (!strict || last.size + size <= StrictCheckBytes)) {
			last.size += static_cast<uint32_t>(size);
			return true;
		}
	}
	m_entries.push_back({address, static_cast<uint32_t>(size), offset, strict});
	return true;
}

// Most logged reads are a few dwords: compare those without a call.
static bool SameGuestBytes(const void* guest, const uint8_t* recorded, uint32_t size) {
	const auto load = [](const void* address, uint32_t offset, auto value) {
		std::memcpy(&value, static_cast<const uint8_t*>(address) + offset, sizeof(value));
		return value;
	};
	switch (size) {
		case 4: return load(guest, 0, uint32_t {}) == load(recorded, 0, uint32_t {});
		case 8: return load(guest, 0, uint64_t {}) == load(recorded, 0, uint64_t {});
		case 16:
			return ((load(guest, 0, uint64_t {}) ^ load(recorded, 0, uint64_t {})) |
			        (load(guest, 8, uint64_t {}) ^ load(recorded, 8, uint64_t {}))) == 0;
		case 32:
			return ((load(guest, 0, uint64_t {}) ^ load(recorded, 0, uint64_t {})) |
			        (load(guest, 8, uint64_t {}) ^ load(recorded, 8, uint64_t {})) |
			        (load(guest, 16, uint64_t {}) ^ load(recorded, 16, uint64_t {})) |
			        (load(guest, 24, uint64_t {}) ^ load(recorded, 24, uint64_t {}))) == 0;
		default: return std::memcmp(guest, recorded, size) == 0;
	}
}

bool GuestReadLog::StillValid() const {
	if (m_failed) {
		return false;
	}
	std::array<uint8_t, StrictCheckBytes> buffer {};
	// Most entries lie on pages the GPU does not own: decide that once per page.
	uint64_t checked_page  = UINT64_MAX;
	bool     checked_owned = false;
	for (const auto& entry: m_entries) {
		const auto* recorded = m_bytes.data() + entry.offset;
		if (!entry.strict) {
			const auto first_page = entry.address >> 12u;
			const auto last_page  = (entry.address + entry.size - 1) >> 12u;
			bool       gpu_owned  = false;
			if (first_page == last_page && first_page == checked_page) {
				gpu_owned = checked_owned;
			} else {
				gpu_owned = Libs::LibKernel::Memory::MayFaultOnGpuRead(entry.address, entry.size);
				if (first_page == last_page) {
					checked_page  = first_page;
					checked_owned = gpu_owned;
				}
			}
			// On GPU-owned pages a direct load faults and reads back all the GPU's queued work:
			// bytes the GPU did not write compare from the backing store.
			bool around = gpu_owned;
			for (uint32_t done = 0; around && done < entry.size; done += sizeof(buffer)) {
				const auto bytes = std::min<uint32_t>(entry.size - done, sizeof(buffer));
				around           = Libs::LibKernel::Memory::TryReadAroundGpuWrites(
                    entry.address + done, buffer.data(), bytes);
				if (around && std::memcmp(buffer.data(), recorded + done, bytes) != 0) {
					return false;
				}
			}
			if (around) {
				continue;
			}
		}
		if (entry.strict) {
			// As specialization memory is read on the execution thread: GPU-owned bytes fail.
			// Large reads (the key arrays of bindless tables) are checked from a scratch buffer.
			static thread_local std::vector<uint8_t> scratch;
			uint8_t* bytes = buffer.data();
			if (entry.size > buffer.size()) {
				scratch.resize(entry.size);
				bytes = scratch.data();
			}
			if (!Libs::LibKernel::Memory::TryReadGpuCleanBacking(entry.address, bytes,
			                                                     entry.size) ||
			    std::memcmp(bytes, recorded, entry.size) != 0) {
				return false;
			}
		} else if (!SameGuestBytes(reinterpret_cast<const void*>(entry.address), recorded,
		                           entry.size)) {
			// A plain load, as the execution thread's own resolution reads: GPU-owned pages
			// fault and read the GPU's bytes back.
			return false;
		}
	}
	return true;
}

namespace {

class ProgramLockGuard {
public:
	explicit ProgramLockGuard(std::atomic_flag& flag): m_flag(flag) {
		while (m_flag.test_and_set(std::memory_order_acquire)) {
			while (m_flag.test(std::memory_order_relaxed)) {
				Common::SpinPause();
			}
		}
	}
	~ProgramLockGuard() { m_flag.clear(std::memory_order_release); }
	ProgramLockGuard(const ProgramLockGuard&)            = delete;
	ProgramLockGuard& operator=(const ProgramLockGuard&) = delete;

private:
	std::atomic_flag& m_flag;
};

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {}

		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		// A deque: resolutions ahead of execution keep pointers to programs while new
		// permutations are added.
		std::deque<Permutation> permutations;
		// The permutation found last (see FindPermutation()) by the execution thread and by the
		// others (all under the program lock).
		std::array<size_t, 2> last_permutation {SIZE_MAX, SIZE_MAX};
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword) {
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, device);
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = next_shader_id.fetch_add(1) + 1, .module = module},
		};
	}

	template <typename InputInfo>
	static std::pair<ShaderRecompiler::CompileOptions, const char*>
	MakeCompileOptions(ShaderType stage, const ShaderParams& params, InputInfo& input_info) {
		const auto           user_data = std::span(params.user_data).first(params.user_data_count);
		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label      = nullptr;
		const char* stage_name = nullptr;
		switch (stage) {
			case ShaderType::Vertex:
				label      = "ShaderRecompiler VS";
				stage_name = "vs";
				break;
			case ShaderType::Mesh:
				label      = "ShaderRecompiler MS";
				stage_name = "ms";
				break;
			case ShaderType::Local:
				label      = "ShaderRecompiler LS";
				stage_name = "ls";
				break;
			case ShaderType::TessellationControl:
				label      = "ShaderRecompiler HS";
				stage_name = "hs";
				break;
			case ShaderType::TessellationEvaluation:
				label      = "ShaderRecompiler DS";
				stage_name = "ds";
				break;
			case ShaderType::Pixel:
				label      = "ShaderRecompiler PS";
				stage_name = "ps";
				break;
			case ShaderType::Compute:
				label      = "ShaderRecompiler CS";
				stage_name = "cs";
				break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code   = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size      = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		return {options, stage_name};
	}

	// The permutation of an entry that a specialization and push data cursor select. Consecutive
	// lookups of a thread mostly repeat it; a program can have thousands of permutations.
	static std::deque<Permutation>::iterator
	FindPermutation(SourceEntry& entry, const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                uint32_t push_data_cursor) {
		const auto matches = [&](const Permutation& candidate) {
			const auto& layout = candidate.program.bindings;
			return layout.push_data_start_dword ==
			           ShaderRecompiler::IR::PushData::StartFor(push_data_cursor,
			                                                    layout.ShaderDataDwords()) &&
			       candidate.specialization == specialization;
		};
		auto& last = entry.last_permutation[GuestGpu::IsGpuThread() ? 0u : 1u];
		if (last < entry.permutations.size() && matches(entry.permutations[last])) {
			return entry.permutations.begin() + static_cast<std::ptrdiff_t>(last);
		}
		const auto found = std::ranges::find_if(entry.permutations, matches);
		if (found != entry.permutations.end()) {
			last = static_cast<size_t>(found - entry.permutations.begin());
		}
		return found;
	}

	// Resolves a stage ahead of execution: only programs and permutations that exist, into the
	// storage of the resolution, reading guest memory through its log.
	template <typename InputInfo>
	ShaderProgram GetAhead(const ShaderParams& params, InputInfo& input_info,
	                       uint32_t& push_data_cursor, ProgramResolution& ahead) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		const auto entry = programs.find(lookup_key);
		if (entry == programs.end() && ahead.stages != ProgramResolution::MaxStages) {
			QueueCompile(stage, params, input_info, push_data_cursor, nullptr);
		}
		if (entry == programs.end() || ahead.stages == ProgramResolution::MaxStages) {
			ahead.reads.Fail();
			return {};
		}
		const auto                             stage_index    = ahead.stages++;
		auto&                                  resources      = ahead.resources[stage_index];
		auto&                                  specialization = ahead.specializations[stage_index];
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = std::span(params.user_data).first(params.user_data_count),
		    .shader_base                = params.Base(),
		    .read_memory                = ReadAheadPlain,
		    .userdata                   = &ahead.reads,
		    .read_specialization_memory = ReadAheadStrict,
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			// As Get(): descriptor tables indexed by the workgroup id probe one entry per group.
			runtime.workgroup_counts = input_info.workgroup_counts;
		}
		if (!ShaderRecompiler::IR::MaterializeResources(entry->second.resource_plan, runtime,
		                                                resources, specialization)) {
			ahead.reads.Fail();
			return {};
		}
		const auto permutation = FindPermutation(entry->second, specialization, push_data_cursor);
		if (permutation == entry->second.permutations.end()) {
			// A specialization made of stale bytes of GPU-written memory is usually one the
			// execution thread never asks for (in GTA V, V#s of garbage strides: thousands of
			// permutations of some compute shaders, each compiled, stored and given a pipeline).
			// Only the execution thread's own lookup compiles those.
			if (!ahead.reads.MayHaveReadGpuWrites()) {
				QueueCompile(stage, params, input_info, push_data_cursor, &specialization);
			}
			ahead.reads.Fail();
			return {};
		}
		input_info.stage = {.program = &permutation->program, .resources = &resources};
		permutation->program.bindings.AdvancePushData(push_data_cursor);
		return permutation->handle;
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto                                         entry = programs.find(lookup_key);
		ShaderRecompiler::IR::SrtRuntime             runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadGuestMemory,
		    .read_specialization_memory = ReadShaderGuestMemory,
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
		}
		if (entry != programs.end()) {
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    entry->second.resource_plan, runtime, entry->second.resources,
			    entry->second.specialization));
			if (const auto permutation =
			        FindPermutation(entry->second, entry->second.specialization, push_data_cursor);
			    permutation != entry->second.permutations.end()) {
				input_info.stage = {.program   = &permutation->program,
				                    .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		// A worker may be compiling this program: use it once it is there.
		if (WaitForCompile()) {
			return Get(params, input_info, push_data_cursor);
		}
		const auto [options, stage_name] = MakeCompileOptions(stage, params, input_info);
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		const auto push_data_start = push_data_cursor;
		auto translated = ShaderRecompiler::TranslateProgram(params.code, options);
		if (entry == programs.end()) {
			entry = programs.try_emplace(lookup_key,
			    ShaderRecompiler::IR::ExtractResourcePlan(translated.program)).first;
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    entry->second.resource_plan, runtime, entry->second.resources,
			    entry->second.specialization));
		}
		entry->second.permutations.push_back(CompilePermutation(
		    stage_name, options, std::move(translated), entry->second.specialization, push_data_cursor));
		const auto& permutation = entry->second.permutations.back();
		by_id[permutation.handle.id] = {&entry->first, &permutation};
		Store(lookup_key, params, input_info, push_data_start, permutation.specialization);
		input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	explicit ProgramCache(vk::Device device): device(device) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		StopWorkers();
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	vk::Device                                                  device;
	std::atomic_uint64_t                                        next_shader_id {0};

	// Graphics programs compiled on worker threads: the resolve thread queues the programs and
	// permutations its draws miss, so the shaders of a new area compile in parallel instead of
	// one after another on the execution thread. Workers only add programs; the execution
	// thread still resolves every draw itself and compiles what they did not add.
	struct CompileJob {
		ProgramKey                                                  key;
		ShaderType                                                  stage = ShaderType::Vertex;
		ShaderParams                                                params;
		std::unique_ptr<ShaderVertexInputInfo>                      vertex;
		std::unique_ptr<ShaderPixelInputInfo>                       pixel;
		std::unique_ptr<ShaderComputeInputInfo>                     compute;
		uint32_t                                                    push_data_cursor = 0;
		std::optional<ShaderRecompiler::IR::ResourceSpecialization> specialization;
		// A program of an earlier session: params.code points here, and it is not stored again.
		std::vector<uint32_t>                                       code;
		std::vector<uint32_t>                                       back_code;
		bool                                                        stored = false;
	};
	static constexpr uint32_t CompileWorkers = 6;

	std::atomic_flag* program_lock = nullptr;
	// Under the program lock: programs a worker is compiling, with their number of jobs.
	std::unordered_map<ProgramKey, uint32_t, ProgramKeyHash> in_flight;
	std::mutex                                     job_mutex;
	std::condition_variable                        job_available;
	std::deque<std::unique_ptr<CompileJob>>        jobs;
	bool                                           stopping = false;
	std::vector<std::jthread>                      workers;

	void LockPrograms() {
		while (program_lock->test_and_set(std::memory_order_acquire)) {
			while (program_lock->test(std::memory_order_relaxed)) {
				Common::SpinPause();
			}
		}
	}
	void UnlockPrograms() { program_lock->clear(std::memory_order_release); }

	// Under the program lock (resolve thread).
	template <typename InputInfo>
	void QueueCompile(ShaderType stage, const ShaderParams& params, const InputInfo& input_info,
	                  uint32_t                                            push_data_cursor,
	                  const ShaderRecompiler::IR::ResourceSpecialization* specialization) {
		if (in_flight.contains(lookup_key)) {
			return;
		}
		auto job              = std::make_unique<CompileJob>();
		job->key              = lookup_key;
		job->stage            = stage;
		job->params           = params;
		job->push_data_cursor = push_data_cursor;
		if (specialization != nullptr) {
			job->specialization = *specialization;
		}
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			job->vertex = std::make_unique<ShaderVertexInputInfo>(input_info);
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			job->pixel = std::make_unique<ShaderPixelInputInfo>(input_info);
		} else {
			job->compute = std::make_unique<ShaderComputeInputInfo>(input_info);
		}
		in_flight[lookup_key]++;
		{
			std::lock_guard lock(job_mutex);
			jobs.push_back(std::move(job));
		}
		job_available.notify_one();
	}

	// Execution thread, under the program lock: waits without the lock until no worker compiles
	// the program of lookup_key. False when none did.
	bool WaitForCompile() {
		if (!in_flight.contains(lookup_key)) {
			return false;
		}
		const auto key = lookup_key;
		do {
			UnlockPrograms();
			std::this_thread::yield();
			LockPrograms();
		} while (in_flight.contains(key));
		return true;
	}

	template <typename InputInfo>
	void RunCompileJob(CompileJob& job, InputInfo& input_info) {
		std::optional<ShaderRecompiler::IR::ResourceSpecialization> stored_specialization;
		std::optional<Permutation>                        permutation;
		std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
		const Permutation*                                          added = nullptr;
		try {
			Common::SoftExitScope soft_exit(true);
			const auto [options, stage_name] =
			    MakeCompileOptions(job.stage, job.params, input_info);
			auto translated = ShaderRecompiler::TranslateProgram(job.params.code, options);
			plan                = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			auto specialization = job.specialization;
			if (!specialization) {
				GuestReadLog                           reads;
				ShaderRecompiler::IR::ResourceSnapshot resources;
				specialization.emplace();
				const ShaderRecompiler::IR::SrtRuntime runtime {
				    .user_data =
				        std::span(job.params.user_data).first(job.params.user_data_count),
				    .shader_base                = job.params.Base(),
				    .read_memory                = ReadAheadPlain,
				    .userdata                   = &reads,
				    .read_specialization_memory = ReadAheadStrict,
				};
				if (!ShaderRecompiler::IR::MaterializeResources(*plan, runtime, resources,
				                                                *specialization)) {
					specialization.reset();
				}
			}
			if (specialization) {
				permutation =
				    CompilePermutation(stage_name, options, std::move(translated),
				                       std::move(*specialization), job.push_data_cursor);
			}
		} catch (const Common::SoftExitError&) {
			// The execution thread compiles the program itself and reports the error.
			permutation.reset();
		}
		LockPrograms();
		if (permutation) {
			auto entry = programs.find(job.key);
			if (entry == programs.end()) {
				entry = programs.try_emplace(job.key, std::move(*plan)).first;
			}
			auto&       permutations = entry->second.permutations;
			const auto& layout       = permutation->program.bindings;
			const bool  known = std::ranges::any_of(permutations, [&](const Permutation& other) {
				return other.specialization == permutation->specialization &&
				       other.program.bindings.push_data_start_dword == layout.push_data_start_dword;
			});
			if (known) {
				device.destroyShaderModule(permutation->handle.module, nullptr);
			} else {
				if (store != nullptr && !job.stored) {
					stored_specialization = permutation->specialization;
				}
				permutations.push_back(std::move(*permutation));
				by_id[permutations.back().handle.id] = {&entry->first, &permutations.back()};
				added                                = &permutations.back();
			}
		}
		if (const auto job_count = in_flight.find(job.key); --job_count->second == 0) {
			in_flight.erase(job_count);
		}
		UnlockPrograms();
		if (job.stored) {
			stored_pending.fetch_sub(1, std::memory_order_release);
		}
		if (stored_specialization) {
			Store(job.key, job.params, input_info, job.push_data_cursor, *stored_specialization);
		}
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			if (added != nullptr && !job.stored && compute_added) {
				compute_added(input_info, *added);
			}
		}
	}

	void CompileWorker(const std::stop_token& stop) {
		KYTY_PROFILER_THREAD("Thread_ShaderCompile");
		for (;;) {
			std::unique_ptr<CompileJob> job;
			{
				std::unique_lock lock(job_mutex);
				job_available.wait(lock, [&] { return stopping || !jobs.empty(); });
				if (stopping) {
					return;
				}
				job = std::move(jobs.front());
				jobs.pop_front();
			}
			if (job->vertex != nullptr) {
				RunCompileJob(*job, *job->vertex);
			} else if (job->pixel != nullptr) {
				RunCompileJob(*job, *job->pixel);
			} else {
				RunCompileJob(*job, *job->compute);
			}
		}
	}

	std::unique_ptr<ProgramStore> store;
	// Called by a compile worker, without the program lock, for each compute permutation it
	// added for a dispatch of this session.
	std::function<void(const ShaderComputeInputInfo&, const Permutation&)> compute_added;
	// Programs of the store that the workers have not finished compiling.
	std::atomic_uint32_t          stored_pending {0};
	// Under the program lock: every compiled permutation by its id, with its program's key.
	std::unordered_map<uint64_t, std::pair<const ProgramKey*, const Permutation*>> by_id;

	// Under the program lock: appends the identity of the permutation with this id (see
	// ProgramStore::PutIdentity). False for an id of no permutation.
	bool DescribeProgram(uint64_t id, std::vector<uint8_t>& identity) const {
		const auto found = by_id.find(id);
		if (found == by_id.end()) {
			return false;
		}
		const auto& [key, permutation] = found->second;
		ProgramStore::PutIdentity(identity, key->stage, key->hash, key->user_data_count,
		                          key->code_size, key->static_state,
		                          permutation->program.bindings.push_data_start_dword,
		                          permutation->specialization);
		return true;
	}

	// Under the program lock: the permutation a stored identity names, if it was compiled.
	const Permutation* FindProgram(const std::vector<uint8_t>& identity) const {
		StoreFile::Reader reader(identity);
		ProgramKey        key;
		uint32_t          stage           = 0;
		uint32_t          push_data_start = 0;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		reader.Get(stage);
		reader.Get(key.hash);
		reader.Get(key.user_data_count);
		reader.Get(key.code_size);
		reader.Get(push_data_start);
		reader.GetVector(key.static_state);
		reader.GetVector(specialization.buffers);
		reader.GetVector(specialization.images);
		if (!reader.Ok() || !reader.AtEnd()) {
			return nullptr;
		}
		key.stage        = static_cast<ShaderType>(stage);
		const auto entry = programs.find(key);
		if (entry == programs.end()) {
			return nullptr;
		}
		for (const auto& permutation: entry->second.permutations) {
			if (permutation.program.bindings.push_data_start_dword == push_data_start &&
			    permutation.specialization == specialization) {
				return &permutation;
			}
		}
		return nullptr;
	}

	template <typename InputInfo>
	void Store(const ProgramKey& key, const ShaderParams& params, const InputInfo& input_info,
	           uint32_t                                            push_data_cursor,
	           const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
		if (store == nullptr) {
			return;
		}
		StoredProgram program;
		program.stage           = key.stage;
		program.hash            = key.hash;
		program.user_data_count = key.user_data_count;
		program.code_size       = key.code_size;
		program.static_state    = key.static_state;
		program.code.assign(params.code.begin(), params.code.end());
		program.back_code.assign(params.back_code.begin(), params.back_code.end());
		program.user_data        = params.user_data;
		program.push_data_cursor = push_data_cursor;
		program.specialization   = specialization;
		auto info                = input_info;
		info.stage               = {};
		program.input_info.resize(sizeof(info));
		std::memcpy(program.input_info.data(), &info, sizeof(info));
		store->Append(program);
	}

	template <typename InputInfo>
	static std::unique_ptr<InputInfo> LoadInputInfo(const std::vector<uint8_t>& bytes) {
		if (bytes.size() != sizeof(InputInfo)) {
			return nullptr;
		}
		auto info = std::make_unique<InputInfo>();
		std::memcpy(info.get(), bytes.data(), sizeof(InputInfo));
		info->stage = {};
		return info;
	}

	// Queues the programs of earlier sessions to the workers (before any draw runs).
	void LoadStore(const std::filesystem::path& path) {
		store = std::make_unique<ProgramStore>();
		auto programs = store->Open(path);
		size_t queued = 0;
		for (auto& program: programs) {
			auto job   = std::make_unique<CompileJob>();
			job->key   = {.stage           = program.stage,
			              .hash            = program.hash,
			              .user_data_count = program.user_data_count,
			              .code_size       = program.code_size,
			              .static_state    = std::move(program.static_state)};
			job->stage = program.stage;
			switch (program.stage) {
				case ShaderType::Pixel:
					job->pixel = LoadInputInfo<ShaderPixelInputInfo>(program.input_info);
					break;
				case ShaderType::Compute:
					job->compute = LoadInputInfo<ShaderComputeInputInfo>(program.input_info);
					break;
				case ShaderType::Vertex:
				case ShaderType::Mesh:
				case ShaderType::Local:
				case ShaderType::TessellationControl:
				case ShaderType::TessellationEvaluation:
					job->vertex = LoadInputInfo<ShaderVertexInputInfo>(program.input_info);
					break;
				default: break;
			}
			if (job->vertex == nullptr && job->pixel == nullptr && job->compute == nullptr) {
				continue;
			}
			// The input info must give the stored key (and be in range for building it).
			if ((job->vertex != nullptr &&
			     (job->vertex->resources_num < 0 ||
			      job->vertex->resources_num > ShaderVertexInputInfo::RES_MAX ||
			      job->vertex->buffers_num < 0 ||
			      job->vertex->buffers_num > ShaderVertexInputInfo::RES_MAX)) ||
			    (job->pixel != nullptr &&
			     job->pixel->input_num > std::size(job->pixel->interpolator_settings))) {
				continue;
			}
			std::vector<uint32_t> static_state;
			if (job->vertex != nullptr) {
				BuildStageStaticKey(*job->vertex, static_state);
			} else if (job->pixel != nullptr) {
				BuildStageStaticKey(*job->pixel, static_state);
			} else {
				BuildStageStaticKey(*job->compute, static_state);
			}
			if (static_state != job->key.static_state) {
				continue;
			}
			job->code                   = std::move(program.code);
			job->back_code              = std::move(program.back_code);
			job->params.code            = job->code;
			job->params.back_code       = job->back_code;
			job->params.user_data       = program.user_data;
			job->params.user_data_count = program.user_data_count;
			job->params.hash            = program.hash;
			job->push_data_cursor       = program.push_data_cursor;
			job->specialization         = std::move(program.specialization);
			job->stored                 = true;
			LockPrograms();
			in_flight[job->key]++;
			UnlockPrograms();
			stored_pending.fetch_add(1, std::memory_order_relaxed);
			{
				std::lock_guard lock(job_mutex);
				jobs.push_back(std::move(job));
			}
			queued++;
		}
		job_available.notify_all();
		PipelineCacheLog("Shader program store: {} programs from {} queued for compilation", queued,
		                 Common::PathToString(path));
	}

	void StartWorkers(std::atomic_flag* lock) {
		program_lock = lock;
		for (uint32_t i = 0; i < CompileWorkers; i++) {
			workers.emplace_back([this](const std::stop_token& stop) { CompileWorker(stop); });
		}
	}

	void StopWorkers() {
		{
			std::lock_guard lock(job_mutex);
			stopping = true;
		}
		job_available.notify_all();
		workers.clear();
	}
};

struct PipelineStoreState {
	PipelineStore               store;
	std::vector<StoredPipeline> pipelines;
	std::atomic_size_t          next {0};
	std::atomic_uint32_t        running {0};
	std::atomic_size_t          created {0};
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
};

namespace {

std::string PipelineStateLayout() {
	return fmt::format("1:{}:{}:{}", sizeof(PipelineRenderingState),
	                   sizeof(PipelineVertexInputState), sizeof(PipelineStaticParameters));
}

// Stage resources are not needed to create a pipeline, but a stage without them is invalid.
const ShaderRecompiler::IR::ResourceSnapshot g_no_resources {};

} // namespace

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics.device)) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	// GTA V's ray tracing shader selects up to ~200 T#s by key from guest data that changes every
	// frame; each new layout is a pipeline compile of several seconds.
	ShaderRecompiler::IR::SetComputeIndirectFloors(256, 8);
	if (Config::AsyncPipelinesEnabled()) {
		constexpr uint32_t AsyncThreads = 3;
		for (uint32_t i = 0; i < AsyncThreads; i++) {
			m_async_threads.emplace_back([this] { AsyncPipelineWorker(); });
		}
		m_program_cache->compute_added = [this](const ShaderComputeInputInfo&    input_info,
		                                        const ProgramCache::Permutation& permutation) {
			auto info  = input_info;
			info.stage = {.program = &permutation.program, .resources = &g_no_resources};
			CreateComputePipelineAhead(info, permutation.handle);
		};
	}
	m_program_cache->StartWorkers(&m_program_lock);
	InitializeDriverCache();
	if (const auto title_id = PipelineCacheTitleId();
	    !title_id.empty() && KYTY_BUILD == KYTY_BUILD_RELEASE) {
		OpenStores(title_id);
	}
}

void PipelineCache::OpenStores(const std::string& title_id) {
	// One store per build: the records copy the layouts of the stage input structures, and
	// what a program key means may change between builds. A build with local changes uses the
	// time pipelineCache.cpp (which includes those structures) was compiled.
	const std::string_view hash = KYTY_GIT_HASH;
	auto build = std::string(std::string_view(KYTY_GIT_REVISION).substr(0, 12));
	if (hash.ends_with("-dirty")) {
		build += fmt::format("-{:08x}", static_cast<uint32_t>(XXH3_64bits(
		                                    __DATE__ __TIME__, sizeof(__DATE__ __TIME__))));
	}
	const auto directory = std::filesystem::path("_PipelineCache");
	const auto prefix    = title_id + ".";
	for (const std::string_view extension: {".programs", ".pipelines"}) {
		// Builds that share the directory (a debug and a release folder linked to one cache) keep
		// their own stores; only the oldest stores beyond a few recent builds are removed.
		constexpr size_t KeptStores = 4;
		const auto       name       = fmt::format("{}{}{}", prefix, build, extension);
		std::error_code  error;
		std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> stores;
		for (const auto& file: std::filesystem::directory_iterator(directory, error)) {
			const auto other = file.path().filename().string();
			if (other != name && other.starts_with(prefix) && other.ends_with(extension)) {
				stores.emplace_back(file.last_write_time(error), file.path());
			}
		}
		if (stores.size() >= KeptStores) {
			std::ranges::sort(stores, std::greater {}, [](const auto& store) { return store.first; });
			for (size_t index = KeptStores - 1; index < stores.size(); index++) {
				std::filesystem::remove(stores[index].second, error);
			}
		}
		// A new build starts from the stores of the newest build with the same layouts: it
		// recompiles their shaders and pipelines while the game loads, instead of stalling the
		// first time each one is drawn (with the driver cache cold for every changed shader).
		const auto path   = directory / name;
		const bool seeded = StoreFile::SeedFromNewest(
		    path, prefix, extension,
		    extension == ".programs" ? ProgramStore::Header()
		                             : PipelineStore::Header(PipelineStateLayout()));
		if (seeded) {
			PipelineCacheLog("Shader store: {} starts from the store of an earlier build",
			                 Common::PathToString(path));
		}
		if (extension == ".programs") {
			m_program_cache->LoadStore(path);
		} else {
			m_stored            = std::make_unique<PipelineStoreState>();
			m_stored->pipelines = m_stored->store.Open(path, PipelineStateLayout());
		}
	}
	if (!m_stored->pipelines.empty()) {
		constexpr uint32_t PrecreateThreads = 4;
		m_stored->running = PrecreateThreads;
		for (uint32_t i = 0; i < PrecreateThreads; i++) {
			m_precreate_threads.emplace_back(
			    [this](const std::stop_token& stop) { PrecreatePipelines(stop); });
		}
	}
}

void PipelineCache::PrecreatePipelines(const std::stop_token& stop) {
	KYTY_PROFILER_THREAD("Thread_PipelinePrecreate");
	// The programs of the pipelines come from the program store: wait until it is compiled.
	while (m_program_cache->stored_pending.load(std::memory_order_acquire) != 0) {
		if (stop.stop_requested()) {
			return;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}
	auto& state = *m_stored;
	for (;;) {
		const auto index = state.next.fetch_add(1, std::memory_order_relaxed);
		if (index >= state.pipelines.size() || stop.stop_requested()) {
			break;
		}
		if (PrecreatePipeline(state.pipelines[index])) {
			state.created.fetch_add(1, std::memory_order_relaxed);
		}
	}
	if (state.running.fetch_sub(1, std::memory_order_acq_rel) == 1 && !stop.stop_requested()) {
		const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		                    std::chrono::steady_clock::now() - state.start)
		                    .count();
		PipelineCacheLog("Pipeline store: created {} of {} stored pipelines in {} ms",
		                 state.created.load(), state.pipelines.size(), ms);
	}
}

bool PipelineCache::PrecreatePipeline(const StoredPipeline& stored) {
	std::array<const ProgramCache::Permutation*, ProgramResolution::MaxStages> programs {};
	if (stored.programs.size() > programs.size()) {
		return false;
	}
	{
		const ProgramLockGuard lock(m_program_lock);
		for (size_t i = 0; i < stored.programs.size(); i++) {
			programs[i] = m_program_cache->FindProgram(stored.programs[i]);
			if (programs[i] == nullptr) {
				return false;
			}
		}
	}
	auto pipeline = std::make_unique<Pipeline>();
	try {
		Common::SoftExitScope soft_exit(true);
		if (stored.compute) {
			ShaderComputeInputInfo input_info;
			if (stored.input_infos[0].size() != sizeof(input_info)) {
				return false;
			}
			std::memcpy(&input_info, stored.input_infos[0].data(), sizeof(input_info));
			input_info.stage = {.program = &programs[0]->program, .resources = &g_no_resources};
			const auto id    = programs[0]->handle.id;
			{
				std::lock_guard lock(m_precreated_mutex);
				if (m_precreated_compute.contains(id)) {
					return true;
				}
			}
			CreatePipelineInternal(m_graphics, *pipeline, input_info, programs[0]->handle.module,
			                       m_driver_cache);
			m_unsaved_pipelines.fetch_add(1, std::memory_order_relaxed);
			std::lock_guard lock(m_precreated_mutex);
			m_precreated_compute.try_emplace(id, std::move(pipeline));
			return true;
		}
		GraphicsPipelineKey                  key {};
		std::array<ShaderVertexInputInfo, 3> vertex_info;
		ShaderPixelInputInfo                 pixel_info;
		GraphicsPrograms                     handles;
		if (stored.state.size() !=
		    sizeof(key.rendering) + sizeof(key.vertex_input) + sizeof(key.static_params)) {
			return false;
		}
		const auto* state = stored.state.data();
		std::memcpy(&key.rendering, state, sizeof(key.rendering));
		std::memcpy(&key.vertex_input, state + sizeof(key.rendering), sizeof(key.vertex_input));
		std::memcpy(&key.static_params, state + sizeof(key.rendering) + sizeof(key.vertex_input),
		            sizeof(key.static_params));
		for (uint32_t i = 0; i < stored.vertex_stages; i++) {
			if (stored.input_infos[i].size() != sizeof(ShaderVertexInputInfo)) {
				return false;
			}
			std::memcpy(&vertex_info[i], stored.input_infos[i].data(), sizeof(vertex_info[i]));
			vertex_info[i].stage = {.program = &programs[i]->program, .resources = &g_no_resources};
			handles.vertex[i]    = programs[i]->handle;
			key.vertex_shader_ids[i] = programs[i]->handle.id;
		}
		if (stored.pixel) {
			const auto& bytes = stored.input_infos[stored.vertex_stages];
			if (bytes.size() != sizeof(pixel_info)) {
				return false;
			}
			std::memcpy(&pixel_info, bytes.data(), sizeof(pixel_info));
			const auto* program = programs[stored.vertex_stages];
			pixel_info.stage    = {.program = &program->program, .resources = &g_no_resources};
			handles.pixel       = program->handle;
			key.ps_shader_id    = program->handle.id;
		}
		{
			std::lock_guard lock(m_precreated_mutex);
			if (m_precreated_graphics.contains(key)) {
				return true;
			}
		}
		CreatePipelineInternal(m_graphics, *pipeline, key.rendering, key.vertex_input,
		                       std::span {vertex_info.data(), stored.vertex_stages},
		                       stored.pixel ? &pixel_info : nullptr, handles, key.static_params,
		                       m_driver_cache);
		m_unsaved_pipelines.fetch_add(1, std::memory_order_relaxed);
		std::lock_guard lock(m_precreated_mutex);
		m_precreated_graphics.try_emplace(key, std::move(pipeline));
		return true;
	} catch (const Common::SoftExitError&) {
		// The pipeline is created again when a draw needs it, and fails there if it still does.
		return false;
	}
}

struct PipelineCache::AsyncPipelineJob {
	bool compute = false;
	// Graphics.
	GraphicsPipelineKey                  key {};
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	uint32_t                             vertex_stages = 0;
	std::optional<ShaderPixelInputInfo>  pixel_info;
	GraphicsPrograms                     programs;
	// Compute.
	ShaderComputeInputInfo compute_info;
	ShaderProgram          compute_program;
};

void PipelineCache::QueueAsyncPipeline(std::unique_ptr<AsyncPipelineJob> job) {
	{
		std::lock_guard lock(m_async_mutex);
		// A dispatch waits for its pipeline, a draw is only skipped: compute pipelines first.
		if (job->compute) {
			m_async_jobs.push_front(std::move(job));
		} else {
			m_async_jobs.push_back(std::move(job));
		}
	}
	m_async_available.notify_one();
}

bool PipelineCache::TakeQueuedAsyncPipeline(
    const std::function<bool(const AsyncPipelineJob&)>& matches) {
	std::lock_guard lock(m_async_mutex);
	const auto      job = std::ranges::find_if(
        m_async_jobs, [&](const std::unique_ptr<AsyncPipelineJob>& queued) { return matches(*queued); });
	if (job == m_async_jobs.end()) {
		return false;
	}
	m_async_jobs.erase(job);
	return true;
}

void PipelineCache::StopAsyncPipelines() {
	{
		std::lock_guard lock(m_async_mutex);
		m_async_stopping = true;
		m_async_jobs.clear();
	}
	m_async_available.notify_all();
	m_async_threads.clear();
}

void PipelineCache::AsyncPipelineWorker() {
	KYTY_PROFILER_THREAD("Thread_PipelineAsync");
	for (;;) {
		std::unique_ptr<AsyncPipelineJob> job;
		{
			std::unique_lock lock(m_async_mutex);
			m_async_available.wait(lock, [&] { return m_async_stopping || !m_async_jobs.empty(); });
			if (m_async_stopping) {
				return;
			}
			job = std::move(m_async_jobs.front());
			m_async_jobs.pop_front();
		}
		auto pipeline = std::make_unique<Pipeline>();
		bool created  = false;
		try {
			Common::SoftExitScope soft_exit(true);
			if (job->compute) {
				CreatePipelineInternal(m_graphics, *pipeline, job->compute_info,
				                       job->compute_program.module, m_driver_cache);
			} else {
				CreatePipelineInternal(m_graphics, *pipeline, job->key.rendering,
				                       job->key.vertex_input,
				                       std::span {job->vertex_info.data(), job->vertex_stages},
				                       job->pixel_info ? &*job->pixel_info : nullptr, job->programs,
				                       job->key.static_params, m_driver_cache);
			}
			created = pipeline->pipeline != nullptr && pipeline->pipeline_layout != nullptr;
		} catch (const Common::SoftExitError&) {
			// The execution thread creates the pipeline itself and reports the error.
		}
		if (created) {
			m_unsaved_pipelines.fetch_add(1, std::memory_order_relaxed);
			if (job->compute) {
				RecordComputePipeline(job->compute_info, job->compute_program.id);
			} else {
				RecordGraphicsPipeline(job->key,
				                       std::span {job->vertex_info.data(), job->vertex_stages},
				                       job->pixel_info ? &*job->pixel_info : nullptr);
			}
		}
		std::lock_guard lock(m_precreated_mutex);
		bool            kept = false;
		if (job->compute) {
			if (created) {
				kept =
				    m_precreated_compute.try_emplace(job->compute_program.id, std::move(pipeline))
				        .second;
			}
			m_async_pending_compute.erase(job->compute_program.id);
		} else {
			if (created) {
				kept = m_precreated_graphics.try_emplace(job->key, std::move(pipeline)).second;
			} else {
				m_async_failed_graphics.insert(job->key);
			}
			m_async_pending_graphics.erase(job->key);
		}
		if (created && !kept) {
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	}
}

void PipelineCache::CreateComputePipelineAhead(const ShaderComputeInputInfo& input_info,
                                               const ShaderProgram&          program) {
	{
		std::lock_guard lock(m_precreated_mutex);
		if (m_precreated_compute.contains(program.id) ||
		    !m_async_pending_compute.insert(program.id).second) {
			return;
		}
	}
	auto job             = std::make_unique<AsyncPipelineJob>();
	job->compute         = true;
	job->compute_info    = input_info;
	job->compute_program = program;
	QueueAsyncPipeline(std::move(job));
}

void PipelineCache::RecordGraphicsPipeline(const GraphicsPipelineKey&             key,
                                           std::span<const ShaderVertexInputInfo> vertex_info,
                                           const ShaderPixelInputInfo*            ps_input_info) {
	if (m_stored == nullptr) {
		return;
	}
	StoredPipeline stored;
	stored.vertex_stages = static_cast<uint8_t>(vertex_info.size());
	stored.pixel         = ps_input_info != nullptr;
	stored.programs.resize(vertex_info.size() + (stored.pixel ? 1u : 0u));
	{
		const ProgramLockGuard lock(m_program_lock);
		for (size_t i = 0; i < vertex_info.size(); i++) {
			if (!m_program_cache->DescribeProgram(key.vertex_shader_ids[i], stored.programs[i])) {
				return;
			}
		}
		if (stored.pixel &&
		    !m_program_cache->DescribeProgram(key.ps_shader_id, stored.programs.back())) {
			return;
		}
	}
	const auto add_input = [&](const auto& input_info) {
		auto info  = input_info;
		info.stage = {};
		auto& out  = stored.input_infos.emplace_back(sizeof(info));
		std::memcpy(out.data(), &info, sizeof(info));
	};
	for (const auto& info: vertex_info) {
		add_input(info);
	}
	if (stored.pixel) {
		add_input(*ps_input_info);
	}
	StoreFile::Put(stored.state, key.rendering);
	StoreFile::Put(stored.state, key.vertex_input);
	StoreFile::Put(stored.state, key.static_params);
	m_stored->store.Append(stored);
}

void PipelineCache::RecordComputePipeline(const ShaderComputeInputInfo& input_info,
                                          uint64_t                      program_id) {
	if (m_stored == nullptr) {
		return;
	}
	StoredPipeline stored;
	stored.compute = true;
	stored.programs.resize(1);
	{
		const ProgramLockGuard lock(m_program_lock);
		if (!m_program_cache->DescribeProgram(program_id, stored.programs[0])) {
			return;
		}
	}
	auto info  = input_info;
	info.stage = {};
	auto& out  = stored.input_infos.emplace_back(sizeof(info));
	std::memcpy(out.data(), &info, sizeof(info));
	m_stored->store.Append(stored);
}

PipelineCache::~PipelineCache() {
	m_program_cache->StopWorkers();
	StopAsyncPipelines();
	m_precreate_threads.clear();
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	destroy(m_precreated_graphics);
	destroy(m_precreated_compute);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
	m_save_thread = std::jthread([this](const std::stop_token& stop) {
		constexpr auto Period = std::chrono::seconds(60);
		auto           next   = std::chrono::steady_clock::now() + Period;
		while (!stop.stop_requested()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
			if (std::chrono::steady_clock::now() < next) {
				continue;
			}
			next = std::chrono::steady_clock::now() + Period;
			if (m_unsaved_pipelines.exchange(0) != 0) {
				WriteDriverCache();
			}
		}
	});
}

void PipelineCache::Save() {
	if (m_save_thread.joinable()) {
		m_save_thread.request_stop();
		m_save_thread.join();
	}
	if (m_driver_cache == nullptr) {
		return;
	}
	WriteDriverCache();
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	m_driver_cache = nullptr;
}

void PipelineCache::WriteDriverCache() {
	// Pipeline creation may continue meanwhile: the driver synchronizes the cache object.
	Common::LockGuard lock(m_write_mutex);

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
	                 Common::PathToString(m_driver_cache_path));
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    ProgramResolution* ahead) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	// Ahead of execution, the vertex fetch tables are read through the resolution's log.
	const ShaderGuestReaderScope reader(ahead != nullptr ? ReadAheadShaderState : nullptr,
	                                    ahead != nullptr ? &ahead->reads : nullptr);
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	if (ahead != nullptr && ahead->reads.Failed()) {
		return {};
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms       result;
	const ProgramLockGuard lock(m_program_lock);
	if (ahead != nullptr) {
		if (pixel_active) {
			result.pixel =
			    m_program_cache->GetAhead(pixel_params, pixel_info, push_data_cursor, *ahead);
		}
		for (uint32_t i = 0; i < (tess_active ? 3u : 1u) && !ahead->reads.Failed(); i++) {
			result.vertex[i] = m_program_cache->GetAhead(vertex_params[i], vertex_info[i],
			                                             push_data_cursor, *ahead);
		}
		return ahead->reads.Failed() ? GraphicsPrograms {} : result;
	}
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
	}
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info,
                                               ProgramResolution*           ahead) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	// Indirect thread counts reach the shader through its shader data buffer.
	uint32_t          push_data_cursor = input_info.dispatch_indirect_threads
	                                         ? ShaderRecompiler::IR::PushData::NoStart
	                                         : 0u;
	const ProgramLockGuard lock(m_program_lock);
	if (ahead != nullptr) {
		return m_program_cache->GetAhead(params, input_info, push_data_cursor, *ahead);
	}
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline* PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs, bool allow_async) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	// Consecutive draws mostly use the pipeline of the draw before.
	if (m_last_graphics_pipeline != nullptr && key == m_last_graphics_key) {
		return m_last_graphics_pipeline;
	}
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		m_last_graphics_key      = key;
		m_last_graphics_pipeline = iter->second.get();
		return iter->second.get();
	}
	for (;;) {
		std::unique_lock lock(m_precreated_mutex);
		if (auto node = m_precreated_graphics.extract(key); !node.empty()) {
			lock.unlock();
			const auto result        = m_graphics_pipelines.insert(std::move(node));
			m_last_graphics_key      = key;
			m_last_graphics_pipeline = result.position->second.get();
			return result.position->second.get();
		}
		if (!m_async_pending_graphics.contains(key)) {
			if (!allow_async || m_async_threads.empty() || m_async_failed_graphics.contains(key)) {
				break;
			}
			// The driver compiles a new pipeline for up to half a second: the draws that need it
			// are skipped until a background thread created it.
			m_async_pending_graphics.insert(key);
			lock.unlock();
			auto job           = std::make_unique<AsyncPipelineJob>();
			job->key           = key;
			job->vertex_stages = static_cast<uint32_t>(vertex_info.size());
			for (uint32_t i = 0; i < job->vertex_stages; i++) {
				job->vertex_info[i]                 = vertex_info[i];
				job->vertex_info[i].stage.resources = &g_no_resources;
			}
			if (ps_active) {
				job->pixel_info.emplace(*ps_input_info);
				job->pixel_info->stage.resources = &g_no_resources;
			}
			job->programs = programs;
			QueueAsyncPipeline(std::move(job));
			return nullptr;
		}
		if (allow_async) {
			return nullptr;
		}
		// A draw that cannot be skipped creates the pipeline itself while it is still queued (the
		// queue may hold many pipelines of skipped draws), or waits for the thread creating it.
		lock.unlock();
		if (TakeQueuedAsyncPipeline([&](const AsyncPipelineJob& job) {
			    return !job.compute && job.key == key;
		    })) {
			std::lock_guard pending_lock(m_precreated_mutex);
			m_async_pending_graphics.erase(key);
			break;
		}
		std::this_thread::yield();
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
	                       ps_input_info, programs, static_params, m_driver_cache);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);
	m_unsaved_pipelines.fetch_add(1, std::memory_order_relaxed);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	RecordGraphicsPipeline(key, vertex_info, ps_input_info);
	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);

	return iter->second.get();
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}
	for (;;) {
		std::unique_lock lock(m_precreated_mutex);
		if (auto node = m_precreated_compute.extract(compute_program.id); !node.empty()) {
			lock.unlock();
			return *m_compute_pipelines.insert(std::move(node)).position->second;
		}
		if (!m_async_pending_compute.contains(compute_program.id)) {
			break;
		}
		// Dispatches always run: create the pipeline here while it is still queued, or wait for
		// the thread creating it.
		lock.unlock();
		if (TakeQueuedAsyncPipeline([&](const AsyncPipelineJob& job) {
			    return job.compute && job.compute_program.id == compute_program.id;
		    })) {
			std::lock_guard pending_lock(m_precreated_mutex);
			m_async_pending_compute.erase(compute_program.id);
			break;
		}
		std::this_thread::yield();
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);
	m_unsaved_pipelines.fetch_add(1, std::memory_order_relaxed);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	RecordComputePipeline(input_info, compute_program.id);
	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}
} // namespace Libs::Graphics
