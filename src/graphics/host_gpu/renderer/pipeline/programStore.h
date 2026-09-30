#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMSTORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMSTORE_H_

#include "common/file.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_set>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

// The compile inputs of every shader permutation a session compiled, appended to a file next to
// the Vulkan pipeline cache. The next session queues them to the compile workers when it starts,
// so the shaders of the areas played before are ready while the game loads instead of
// compiling (and stalling the execution thread) when an area first comes into view. Programs
// are recompiled from their inputs with the running build, so the file never holds code from
// another build; a build whose record layout differs starts a new file.
struct StoredProgram {
	ShaderType               stage           = ShaderType::Unknown;
	uint64_t                 hash            = 0;
	uint32_t                 user_data_count = 0;
	uint32_t                 code_size       = 0;
	std::vector<uint32_t>    static_state;
	std::vector<uint32_t>    code;
	std::vector<uint32_t>    back_code;
	std::array<uint32_t, 40> user_data {};
	// ShaderVertexInputInfo, ShaderPixelInputInfo or ShaderComputeInputInfo by stage.
	std::vector<uint8_t>     input_info;
	uint32_t                 push_data_cursor = 0;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
};

class ProgramStore {
public:
	static_assert(std::is_trivially_copyable_v<ShaderVertexInputInfo> &&
	              std::is_trivially_copyable_v<ShaderPixelInputInfo> &&
	              std::is_trivially_copyable_v<ShaderComputeInputInfo> &&
	              std::is_trivially_copyable_v<ShaderRecompiler::IR::ResourceSpecialization::Buffer> &&
	              std::is_trivially_copyable_v<ShaderRecompiler::IR::ResourceSpecialization::Image>);

	ProgramStore() = default;
	ProgramStore(const ProgramStore&)            = delete;
	ProgramStore& operator=(const ProgramStore&) = delete;
	~ProgramStore() {
		if (m_file != nullptr) {
			std::fclose(m_file);
		}
	}

	// Loads the distinct records of `path` and keeps the file open for appending.
	std::vector<StoredProgram> Open(const std::filesystem::path& path) {
		std::vector<StoredProgram> programs;
		const auto                 text       = Common::PathToString(path);
		const auto                 header     = Header();
		bool                       valid      = false;
		uint64_t                   valid_size = 0;
		uint64_t                   file_size  = 0;
		if (auto* file = std::fopen(text.c_str(), "rb"); file != nullptr) {
			std::vector<uint8_t> data;
			std::fseek(file, 0, SEEK_END);
			const auto size = std::ftell(file);
			std::fseek(file, 0, SEEK_SET);
			if (size > 0 && size < (int64_t {1} << 30u)) {
				data.resize(static_cast<size_t>(size));
				data.resize(std::fread(data.data(), 1, data.size(), file));
			}
			std::fclose(file);
			file_size = data.size();
			if (data.size() >= header.size() &&
			    std::memcmp(data.data(), header.data(), header.size()) == 0) {
				valid      = true;
				valid_size = header.size();
				std::unordered_set<uint64_t> seen;
				for (size_t offset = header.size();;) {
					StoredProgram program;
					uint64_t      record_hash = 0;
					const auto    next        = Parse(data, offset, program, record_hash);
					if (next == 0) {
						break;
					}
					offset     = next;
					valid_size = next;
					if (seen.insert(record_hash).second) {
						programs.push_back(std::move(program));
					}
				}
			}
		}
		if (!Common::File::CreateDirectories(path.parent_path())) {
			return {};
		}
		if (valid && valid_size != file_size) {
			// The torn record of a crash ends the file: drop it before appending.
			std::error_code error;
			std::filesystem::resize_file(path, valid_size, error);
			valid = !error;
		}
		m_file = std::fopen(text.c_str(), valid ? "ab" : "wb");
		if (m_file != nullptr && !valid) {
			programs.clear();
			std::fwrite(header.data(), 1, header.size(), m_file);
			std::fflush(m_file);
		}
		return programs;
	}

	void Append(const StoredProgram& program) {
		std::vector<uint8_t> record;
		Serialize(program, record);
		std::lock_guard lock(m_mutex);
		if (m_file != nullptr) {
			std::fwrite(record.data(), 1, record.size(), m_file);
			std::fflush(m_file);
		}
	}

private:
	static std::string Header() {
		// The layouts the records copy.
		return fmt::format("KytyPrograms1:{}:{}:{}:{}:{}\n", sizeof(ShaderVertexInputInfo),
		                   sizeof(ShaderPixelInputInfo), sizeof(ShaderComputeInputInfo),
		                   sizeof(ShaderRecompiler::IR::ResourceSpecialization::Buffer),
		                   sizeof(ShaderRecompiler::IR::ResourceSpecialization::Image));
	}

	template <typename T>
	static void Put(std::vector<uint8_t>& out, const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
		out.insert(out.end(), bytes, bytes + sizeof(T));
	}
	template <typename T>
	static void PutVector(std::vector<uint8_t>& out, const std::vector<T>& values) {
		static_assert(std::is_trivially_copyable_v<T>);
		Put(out, static_cast<uint32_t>(values.size()));
		const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
		out.insert(out.end(), bytes, bytes + values.size() * sizeof(T));
	}

	// A record is its payload size (u32), the payload and the XXH3 of the payload (u64).
	static void Serialize(const StoredProgram& program, std::vector<uint8_t>& out) {
		std::vector<uint8_t> payload;
		Put(payload, static_cast<uint32_t>(program.stage));
		Put(payload, program.hash);
		Put(payload, program.user_data_count);
		Put(payload, program.code_size);
		Put(payload, program.push_data_cursor);
		Put(payload, program.user_data);
		PutVector(payload, program.static_state);
		PutVector(payload, program.code);
		PutVector(payload, program.back_code);
		PutVector(payload, program.input_info);
		PutVector(payload, program.specialization.buffers);
		PutVector(payload, program.specialization.images);
		Put(out, static_cast<uint32_t>(payload.size()));
		out.insert(out.end(), payload.begin(), payload.end());
		Put(out, XXH3_64bits(payload.data(), payload.size()));
	}

	// The offset after the record, or 0 when no complete valid record starts at `offset`.
	static size_t Parse(const std::vector<uint8_t>& data, size_t offset, StoredProgram& program,
	                    uint64_t& record_hash) {
		uint32_t payload_size = 0;
		if (data.size() - offset < sizeof(payload_size)) {
			return 0;
		}
		std::memcpy(&payload_size, data.data() + offset, sizeof(payload_size));
		const auto begin = offset + sizeof(payload_size);
		if (data.size() - begin < uint64_t {payload_size} + sizeof(uint64_t)) {
			return 0;
		}
		uint64_t stored_hash = 0;
		std::memcpy(&stored_hash, data.data() + begin + payload_size, sizeof(stored_hash));
		record_hash = XXH3_64bits(data.data() + begin, payload_size);
		if (record_hash != stored_hash) {
			return 0;
		}
		size_t     cursor = begin;
		const auto end    = begin + payload_size;
		bool       ok     = true;
		const auto get    = [&](auto& value) {
			if (!ok || end - cursor < sizeof(value)) {
				ok = false;
				return;
			}
			std::memcpy(&value, data.data() + cursor, sizeof(value));
			cursor += sizeof(value);
		};
		const auto get_vector = [&](auto& values) {
			using T        = typename std::remove_reference_t<decltype(values)>::value_type;
			uint32_t count = 0;
			get(count);
			if (!ok || (end - cursor) / sizeof(T) < count) {
				ok = false;
				return;
			}
			values.resize(count);
			std::memcpy(values.data(), data.data() + cursor, count * sizeof(T));
			cursor += count * sizeof(T);
		};
		uint32_t stage = 0;
		get(stage);
		get(program.hash);
		get(program.user_data_count);
		get(program.code_size);
		get(program.push_data_cursor);
		get(program.user_data);
		get_vector(program.static_state);
		get_vector(program.code);
		get_vector(program.back_code);
		get_vector(program.input_info);
		get_vector(program.specialization.buffers);
		get_vector(program.specialization.images);
		program.stage = static_cast<ShaderType>(stage);
		if (!ok || cursor != end || program.code.empty() ||
		    program.code_size != program.code.size() || program.user_data_count > 40) {
			return 0;
		}
		return end + sizeof(uint64_t);
	}

	std::mutex m_mutex;
	std::FILE* m_file = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMSTORE_H_
