# Profile for profile-guided optimization (PGO)

`gtav.profdata` is the `llvm-profdata` profile that the Windows release build uses (`-DKYTY_PGO_USE`, see
`.github/workflows/build.yml`). It was recorded with LLVM 22.1.8 from GTA V in Performance and Fidelity modes
using an instrumented build (`-DKYTY_PGO_GENERATE=ON`), and merged with `llvm-profdata merge`.

- A profile written by a newer LLVM than the compiler reading it is rejected ("unsupported instrumentation
  profile format version"), so the workflow installs LLVM 22.1.8 before configuring. If this file is missing or the
  compiler cannot read it, the release is built without PGO (a warning in the job).
- A profile recorded on older code still helps: functions that changed since are optimized without it.
- Regenerate it before a release: build with `-DKYTY_NATIVE_V3=ON -DKYTY_THINLTO=ON -DKYTY_PGO_GENERATE=ON`, run the
  game for ~60 s in each mode with `KYTY_PGO_FILE=<name>.profraw`, then
  `llvm-profdata merge -o pgo/gtav.profdata perf.profraw fid.profraw`, using the same LLVM version as the workflow.
