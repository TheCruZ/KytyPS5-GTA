#ifndef KYTY_COMMON_SAMPLER_H_
#define KYTY_COMMON_SAMPLER_H_

#include <string>

namespace Common {
// " function+0xoffset [file:line]" of one host code address (empty without symbols); assert.cpp.
// Declared here rather than in assert.h: assert.h is part of the shader cache version.
std::string HostAddressName(const void* address);
} // namespace Common

namespace Common::Sampler {

// KYTY_SAMPLE=<start seconds>:<seconds> (Windows): a sampling profiler for the threads that
// register themselves. From <start> seconds after the emulator started, for <seconds>, every
// millisecond each registered thread is suspended, its instruction pointer read, and resumed;
// then the hottest functions of each thread are printed ([sample]). Without KYTY_SAMPLE,
// registering does nothing.
void RegisterCurrentThread(const char* name);

} // namespace Common::Sampler

#endif /* KYTY_COMMON_SAMPLER_H_ */
