#include "common/emulatorConfig.h"

#include "common/assert.h"
#include "common/threads.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <string_view>

namespace Config {

static std::unique_ptr<ConfigOptions> g_config;

void Initialize() {
	EXIT_IF(g_config != nullptr);

	g_config = std::make_unique<ConfigOptions>();
}

void Shutdown() {
	g_config.reset();
}

void Load(const ConfigOptions& cfg) {
	EXIT_IF(g_config == nullptr);
	EXIT_IF(cfg.user_name.empty() || cfg.user_name.size() > MAX_USER_NAME_LENGTH);
	EXIT_IF(!IsConfiguredUserIdValid(cfg.user_id));
	EXIT_IF(cfg.controller_speaker_volume > 100 || cfg.controller_vibration_intensity > 100);

	*g_config = cfg;
}

uint32_t GetScreenWidth() {
	return g_config->screen_width;
}

uint32_t GetScreenHeight() {
	return g_config->screen_height;
}

const std::string& GetUserName() {
	return g_config->user_name;
}

int32_t GetUserId() {
	return g_config->user_id;
}

const std::string& GetAudioInputDevice() {
	return g_config->audio_input_device;
}

const std::optional<ControllerColor>& GetControllerColor() {
	return g_config->controller_color;
}

uint32_t GetControllerSpeakerVolume() {
	return g_config->controller_speaker_volume;
}

uint32_t GetControllerVibrationIntensity() {
	return g_config->controller_vibration_intensity;
}

PresentMode GetPresentMode() {
	return g_config->present_mode;
}

int32_t GetGpuIndex() {
	return g_config->gpu_index;
}

bool FullscreenEnabled() {
	return g_config->fullscreen_enabled;
}

bool HideCursorEnabled() {
	return g_config->hide_cursor_enabled;
}

bool VrEnabled() {
	return g_config->vr_enabled;
}

bool AmdCpuEnabled() {
	return g_config->amd_cpu_enabled;
}

uint32_t GetVblankFrequency() {
	return std::clamp(g_config->vblank_frequency, 30u, 360u);
}

uint32_t GetConsoleLanguage() {
	return g_config->console_language;
}

bool VulkanValidationEnabled() {
	return g_config->vulkan_validation_enabled;
}

bool ShaderValidationEnabled() {
	return g_config->shader_validation_enabled;
}

ShaderOptimizationType GetShaderOptimizationType() {
	return g_config->shader_optimization_type;
}

LogDirection GetShaderLogDirection() {
	return g_config->shader_log_direction;
}

std::filesystem::path GetShaderLogFolder() {
	return g_config->shader_log_folder;
}

bool CommandBufferDumpEnabled() {
	return g_config->command_buffer_dump_enabled;
}

std::filesystem::path GetCommandBufferDumpFolder() {
	return g_config->command_buffer_dump_folder;
}

bool GraphicsDebugDumpEnabled() {
	return g_config->graphics_debug_dump_enabled;
}

LogDirection GetPrintfDirection() {
	return g_config->printf_direction;
}

std::filesystem::path GetPrintfOutputFile() {
	return g_config->printf_output_file;
}

bool ProfilerEnabled() {
	return g_config->profiler_enabled;
}

bool SpirvDebugPrintfEnabled() {
	return g_config->spirv_debug_printf_enabled;
}

bool GpuAssistedValidationEnabled() {
	return g_config->gpu_assisted_validation_enabled && g_config->vulkan_validation_enabled;
}

bool RenderDocEnabled() {
	return g_config->renderdoc_enabled;
}

bool ReadbackLinearImagesEnabled() {
	return g_config->readback_linear_images;
}

bool SyncRawImageBuffersEnabled() {
	return g_config->sync_raw_image_buffers;
}

bool TessellationEnabled() {
	return g_config->tessellation_enabled;
}

bool TrophyEnabled() {
	return g_config->trophy_enabled;
}

bool PlayGoHackEnabled() {
	return g_config->playgo_hack_enabled;
}

bool SkipNoticeScreen() {
	return g_config->skip_notice_screen;
}

uint32_t ParseGpuPipelineStages(const char* value) {
	char*      end    = nullptr;
	const auto parsed = std::strtoul(value, &end, 0);
	if (end == value || *end != '\0') {
		EXIT("invalid GPU pipeline stages: %s\n", value);
	}
	return parsed == 1 ? GPU_PIPELINE_ALL : static_cast<uint32_t>(parsed) & GPU_PIPELINE_ALL;
}

void ConfigureGpuStageThread(GpuStageThread stage) {
	// Three physical cores of the first CCD of a Ryzen 9 5900X (SMT siblings are adjacent), away
	// from core 0: the stages share its L3.
	static const std::array<int, 3> cpus = [] {
		std::array<int, 3> result {4, 6, 8};
		const char*        value = std::getenv("KYTY_GPU_THREAD_CPUS");
		if (value == nullptr) {
			return result;
		}
		if (std::string_view(value) == "off") {
			return std::array<int, 3> {-1, -1, -1};
		}
		for (auto& cpu: result) {
			char* end = nullptr;
			cpu       = static_cast<int>(std::strtol(value, &end, 10));
			if (end == value) {
				EXIT("invalid KYTY_GPU_THREAD_CPUS: %s\n", std::getenv("KYTY_GPU_THREAD_CPUS"));
			}
			value = *end == ',' ? end + 1 : end;
		}
		return result;
	}();
	static const bool high_priority = [] {
		const char* value = std::getenv("KYTY_GPU_THREAD_PRIORITY");
		return value == nullptr || std::string_view(value) != "0";
	}();
	if (high_priority) {
		Common::Thread::RaiseCurrentPriority();
	}
	Common::Thread::PinCurrent(cpus[static_cast<uint32_t>(stage)]);
}

uint32_t GpuPipelineStages() {
	if (const char* value = std::getenv("KYTY_GPU_PIPELINE"); value != nullptr) {
		return ParseGpuPipelineStages(value);
	}
	// Tools and tests that do not load a configuration run the GPU inline.
	return g_config != nullptr ? g_config->gpu_pipeline_stages : 0;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled() {
	return g_config->red_zone_protection_enabled;
}
#endif

const Keymap& GetKeymap() {
	return g_config->keymap;
}

} // namespace Config
