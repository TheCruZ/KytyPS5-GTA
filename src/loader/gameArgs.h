#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>

// Engine console variables for the game's own command-line file: KYTY_GAME_CONVARS="name=value;name=value"
// (Demon's Souls gets doIndexBufferCulling=false unless the variable is set; KYTY_GAME_CONVARS= keeps the file)
// gives a read-only open of the PackageCmdLineArgs.txt the game ships a copy with a "+name=value" line per
// entry added (Demon's Souls passes every line of that file to its console at boot), so engine settings
// change without touching the game's files.
namespace Loader::GameArgs {

// The host file a read-only open of `real` should open instead (empty: `real` itself).
std::filesystem::path RedirectRead(const std::filesystem::path& real);
// A stat of `real`: the size of the copy a read would get (nothing: its own).
std::optional<uint64_t> RedirectedSize(const std::filesystem::path& real);

} // namespace Loader::GameArgs
