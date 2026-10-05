#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMSTORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMSTORE_H_

#include "common/file.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

// An append-only file of records after a one-line header: each record is its payload size (u32),
// the payload and the XXH3 of the payload (u64). A file whose header differs is started over, and
// the torn record of a crash ends the file.
class StoreFile {
public:
	StoreFile() = default;
	StoreFile(const StoreFile&)            = delete;
	StoreFile& operator=(const StoreFile&) = delete;
	~StoreFile() {
		if (m_file != nullptr) {
			std::fclose(m_file);
		}
	}

	// Loads the distinct payloads of `path` and keeps the file open for appending.
	std::vector<std::vector<uint8_t>> Open(const std::filesystem::path& path,
	                                       const std::string&           header) {
		std::vector<std::vector<uint8_t>> payloads;
		const auto                        text       = Common::PathToString(path);
		bool                              valid      = false;
		uint64_t                          valid_size = 0;
		uint64_t                          file_size  = 0;
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
					uint32_t payload_size = 0;
					if (data.size() - offset < sizeof(payload_size)) {
						break;
					}
					std::memcpy(&payload_size, data.data() + offset, sizeof(payload_size));
					const auto begin = offset + sizeof(payload_size);
					if (data.size() - begin < uint64_t {payload_size} + sizeof(uint64_t)) {
						break;
					}
					uint64_t stored_hash = 0;
					std::memcpy(&stored_hash, data.data() + begin + payload_size,
					            sizeof(stored_hash));
					const auto record_hash = XXH3_64bits(data.data() + begin, payload_size);
					if (record_hash != stored_hash) {
						break;
					}
					offset     = begin + payload_size + sizeof(uint64_t);
					valid_size = offset;
					if (seen.insert(record_hash).second) {
						payloads.emplace_back(data.begin() + static_cast<ptrdiff_t>(begin),
						                      data.begin() + static_cast<ptrdiff_t>(begin + payload_size));
					}
				}
			}
		}
		if (!Common::File::CreateDirectories(path.parent_path())) {
			return {};
		}
		if (valid && valid_size != file_size) {
			std::error_code error;
			std::filesystem::resize_file(path, valid_size, error);
			valid = !error;
		}
		m_file = std::fopen(text.c_str(), valid ? "ab" : "wb");
		if (m_file != nullptr && !valid) {
			payloads.clear();
			std::fwrite(header.data(), 1, header.size(), m_file);
			std::fflush(m_file);
		}
		return payloads;
	}

	void Append(const std::vector<uint8_t>& payload) {
		std::vector<uint8_t> record;
		record.reserve(payload.size() + sizeof(uint32_t) + sizeof(uint64_t));
		const auto size = static_cast<uint32_t>(payload.size());
		const auto hash = XXH3_64bits(payload.data(), payload.size());
		record.insert(record.end(), reinterpret_cast<const uint8_t*>(&size),
		              reinterpret_cast<const uint8_t*>(&size) + sizeof(size));
		record.insert(record.end(), payload.begin(), payload.end());
		record.insert(record.end(), reinterpret_cast<const uint8_t*>(&hash),
		              reinterpret_cast<const uint8_t*>(&hash) + sizeof(hash));
		std::lock_guard lock(m_mutex);
		if (m_file != nullptr) {
			std::fwrite(record.data(), 1, record.size(), m_file);
			std::fflush(m_file);
		}
	}

	// Whether the file at `path` starts with `header`.
	static bool HasHeader(const std::filesystem::path& path, const std::string& header) {
		std::string start(header.size(), '\0');
		auto*       file = std::fopen(Common::PathToString(path).c_str(), "rb");
		if (file == nullptr) {
			return false;
		}
		const auto read = std::fread(start.data(), 1, start.size(), file);
		std::fclose(file);
		return read == start.size() && start == header;
	}

	// A build without its own store starts from a copy of the newest store of another build
	// (`<prefix><build><extension>` in `path`'s directory) with the same header: the records hold
	// compile inputs, not results, so they mean the same to the new build. True when a copy was
	// made.
	static bool SeedFromNewest(const std::filesystem::path& path, std::string_view prefix,
	                           std::string_view extension, const std::string& header) {
		std::error_code error;
		if (std::filesystem::exists(path, error)) {
			return false;
		}
		std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> stores;
		for (const auto& file: std::filesystem::directory_iterator(path.parent_path(), error)) {
			const auto name = file.path().filename().string();
			if (name.starts_with(prefix) && name.ends_with(extension) &&
			    HasHeader(file.path(), header)) {
				stores.emplace_back(file.last_write_time(error), file.path());
			}
		}
		if (stores.empty()) {
			return false;
		}
		const auto newest = std::ranges::max_element(
		    stores, std::less {}, [](const auto& store) { return store.first; });
		return std::filesystem::copy_file(newest->second, path, error) && !error;
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

	// Reads a payload written with Put/PutVector; a read past the end fails the reader.
	class Reader {
	public:
		explicit Reader(const std::vector<uint8_t>& data): m_data(data) {}
		template <typename T>
		void Get(T& value) {
			static_assert(std::is_trivially_copyable_v<T>);
			if (!m_ok || m_data.size() - m_cursor < sizeof(T)) {
				m_ok = false;
				return;
			}
			std::memcpy(&value, m_data.data() + m_cursor, sizeof(T));
			m_cursor += sizeof(T);
		}
		template <typename T>
		void GetVector(std::vector<T>& values) {
			uint32_t count = 0;
			Get(count);
			if (!m_ok || (m_data.size() - m_cursor) / sizeof(T) < count) {
				m_ok = false;
				return;
			}
			values.resize(count);
			std::memcpy(values.data(), m_data.data() + m_cursor, count * sizeof(T));
			m_cursor += count * sizeof(T);
		}
		[[nodiscard]] bool Ok() const { return m_ok; }
		[[nodiscard]] bool AtEnd() const { return m_cursor == m_data.size(); }

	private:
		const std::vector<uint8_t>& m_data;
		size_t                      m_cursor = 0;
		bool                        m_ok     = true;
	};

private:
	std::mutex m_mutex;
	std::FILE* m_file = nullptr;
};

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

	// Loads the distinct records of `path` and keeps the file open for appending.
	std::vector<StoredProgram> Open(const std::filesystem::path& path) {
		std::vector<StoredProgram> programs;
		for (const auto& payload: m_file.Open(path, Header())) {
			StoredProgram program;
			if (Parse(payload, program)) {
				programs.push_back(std::move(program));
			}
		}
		return programs;
	}

	void Append(const StoredProgram& program) {
		std::vector<uint8_t> payload;
		Serialize(program, payload);
		m_file.Append(payload);
	}

	static std::string Header() {
		// The layouts the records copy.
		return fmt::format("KytyPrograms2:{}:{}:{}:{}:{}:{}\n", ShaderInputLayoutTag,
		                   sizeof(ShaderVertexInputInfo), sizeof(ShaderPixelInputInfo),
		                   sizeof(ShaderComputeInputInfo),
		                   sizeof(ShaderRecompiler::IR::ResourceSpecialization::Buffer),
		                   sizeof(ShaderRecompiler::IR::ResourceSpecialization::Image));
	}

	// A program's identity within a store: what picks one permutation of one program.
	static void PutIdentity(std::vector<uint8_t>& out, ShaderType stage, uint64_t hash,
	                        uint32_t user_data_count, uint32_t code_size,
	                        const std::vector<uint32_t>& static_state, uint32_t push_data_start,
	                        const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
		StoreFile::Put(out, static_cast<uint32_t>(stage));
		StoreFile::Put(out, hash);
		StoreFile::Put(out, user_data_count);
		StoreFile::Put(out, code_size);
		StoreFile::Put(out, push_data_start);
		StoreFile::PutVector(out, static_state);
		StoreFile::PutVector(out, specialization.buffers);
		StoreFile::PutVector(out, specialization.images);
	}

private:
	static void Serialize(const StoredProgram& program, std::vector<uint8_t>& payload) {
		StoreFile::Put(payload, static_cast<uint32_t>(program.stage));
		StoreFile::Put(payload, program.hash);
		StoreFile::Put(payload, program.user_data_count);
		StoreFile::Put(payload, program.code_size);
		StoreFile::Put(payload, program.push_data_cursor);
		StoreFile::Put(payload, program.user_data);
		StoreFile::PutVector(payload, program.static_state);
		StoreFile::PutVector(payload, program.code);
		StoreFile::PutVector(payload, program.back_code);
		StoreFile::PutVector(payload, program.input_info);
		StoreFile::PutVector(payload, program.specialization.buffers);
		StoreFile::PutVector(payload, program.specialization.images);
	}

	static bool Parse(const std::vector<uint8_t>& payload, StoredProgram& program) {
		StoreFile::Reader reader(payload);
		uint32_t          stage = 0;
		reader.Get(stage);
		reader.Get(program.hash);
		reader.Get(program.user_data_count);
		reader.Get(program.code_size);
		reader.Get(program.push_data_cursor);
		reader.Get(program.user_data);
		reader.GetVector(program.static_state);
		reader.GetVector(program.code);
		reader.GetVector(program.back_code);
		reader.GetVector(program.input_info);
		reader.GetVector(program.specialization.buffers);
		reader.GetVector(program.specialization.images);
		program.stage = static_cast<ShaderType>(stage);
		return reader.Ok() && reader.AtEnd() && !program.code.empty() &&
		       program.code_size == program.code.size() && program.user_data_count <= 40;
	}

	StoreFile m_file;
};

// The Vulkan pipelines a session created, by the identities of their programs (see
// ProgramStore::PutIdentity) and their fixed state. The next session creates them on background
// threads once the program store compiled their programs, so the driver compiles them while the
// game loads, also after a new build changed every shader (and the driver cache misses them all).
struct StoredPipeline {
	// Compute, or the graphics vertex stages (1 or 3) plus an optional pixel stage.
	bool                              compute = false;
	uint8_t                           vertex_stages = 0;
	bool                              pixel         = false;
	// Program identities by stage: vertex stages, then the pixel stage (or the compute stage).
	std::vector<std::vector<uint8_t>> programs;
	// The stage input structures by stage (their program pointers cleared), same order.
	std::vector<std::vector<uint8_t>> input_infos;
	// The fixed state of a graphics pipeline (rendering, vertex input and static parameters).
	std::vector<uint8_t>              state;
};

class PipelineStore {
public:
	// `state_layout` names the layout of StoredPipeline::state.
	std::vector<StoredPipeline> Open(const std::filesystem::path& path,
	                                 const std::string&           state_layout) {
		std::vector<StoredPipeline> pipelines;
		for (const auto& payload: m_file.Open(path, Header(state_layout))) {
			StoredPipeline pipeline;
			if (Parse(payload, pipeline)) {
				pipelines.push_back(std::move(pipeline));
			}
		}
		return pipelines;
	}

	void Append(const StoredPipeline& pipeline) {
		std::vector<uint8_t> payload;
		StoreFile::Put(payload, static_cast<uint8_t>(pipeline.compute));
		StoreFile::Put(payload, pipeline.vertex_stages);
		StoreFile::Put(payload, static_cast<uint8_t>(pipeline.pixel));
		StoreFile::Put(payload, static_cast<uint32_t>(pipeline.programs.size()));
		for (size_t i = 0; i < pipeline.programs.size(); i++) {
			StoreFile::PutVector(payload, pipeline.programs[i]);
			StoreFile::PutVector(payload, pipeline.input_infos[i]);
		}
		StoreFile::PutVector(payload, pipeline.state);
		m_file.Append(payload);
	}

	static std::string Header(const std::string& state_layout) {
		// The program identities and input structures follow the program store's layouts.
		return fmt::format("KytyPipelines1:{}:{}", state_layout, ProgramStore::Header());
	}

private:
	static bool Parse(const std::vector<uint8_t>& payload, StoredPipeline& pipeline) {
		StoreFile::Reader reader(payload);
		uint8_t           compute = 0;
		uint8_t           pixel   = 0;
		uint32_t          count   = 0;
		reader.Get(compute);
		reader.Get(pipeline.vertex_stages);
		reader.Get(pixel);
		reader.Get(count);
		pipeline.compute = compute != 0;
		pipeline.pixel   = pixel != 0;
		const uint32_t expected =
		    pipeline.compute ? 1u : pipeline.vertex_stages + (pipeline.pixel ? 1u : 0u);
		if (!reader.Ok() || count != expected ||
		    (!pipeline.compute && pipeline.vertex_stages != 1 && pipeline.vertex_stages != 3)) {
			return false;
		}
		pipeline.programs.resize(count);
		pipeline.input_infos.resize(count);
		for (uint32_t i = 0; i < count; i++) {
			reader.GetVector(pipeline.programs[i]);
			reader.GetVector(pipeline.input_infos[i]);
		}
		reader.GetVector(pipeline.state);
		return reader.Ok() && reader.AtEnd();
	}

	StoreFile m_file;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMSTORE_H_
