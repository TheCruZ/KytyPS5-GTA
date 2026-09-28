# KytyPS5-GTA

A fork of [KytyPS5](https://github.com/KytyPS5/KytyPS5) focused on running
**Grand Theft Auto V (PS5, PPSA04263)**.

This fork is always the latest upstream `main` plus a small set of fixes on top, one commit per
fix. Fixes that are useful beyond GTA V are proposed upstream as pull requests.

For everything about the emulator itself (features, system requirements, building, usage,
licensing), read the **[original KytyPS5 README](https://github.com/KytyPS5/KytyPS5#readme)**.

> [!IMPORTANT]
> KytyPS5 is not affiliated with Sony Interactive Entertainment, PlayStation or Rockstar Games.
> The project does not distribute games or copyrighted system software. Use only game files that
> you have obtained legally.

## Status

- Boots to the main menu and plays the prologue in Performance, Performance RT and Fidelity modes.
- The prologue can be completed and the game continues into Los Santos (Franklin and Lamar).
- Performance on an RTX 3090 / Ryzen 9 5900X: about 20-22 fps walking outdoors in Los Santos
  (up from about 12-13 fps before the GPU-thread optimizations below); lower while driving fast.
- **Red zone protection** is enabled by default in this fork (disable it with `--no-redzone` or
  the "Windows SysV red zone crash protection" launcher option). Without it the game crashes in
  its streaming thread.

## Fixed bugs and crashes

Each item is one commit on top of upstream `main`.

### Crashes and unsupported shaders

- **Invalid depth upload on shadow cube maps**: depth views of the first slices of a layered
  depth image no longer shrink its stencil and HTile ranges.
- **Skinning compute shaders**: stores through buffer descriptors (V#) picked from a descriptor
  table on the GPU.
- **Fidelity mode device loss (ray-tracing BVH build)**: dispatches given in threads keep the
  extra threads of their last group inactive; the extra threads corrupted the BVH and the next
  pass never finished.
- **Unsupported `v_bfrev_b32` with an SDWA source** in the BVH build shaders.
- **Ray-tracing buffers whose record count the shader computes** are bound instead of rejected.
- **Single-DWORD and `FORMAT_X` loads through GPU-selected V#s** (Fidelity mode).
- **V# bases selected by a scalar branch** in shaders that write memory.
- **Samplers built with `s_brev_b32`** (constant-folded bit reverse).
- **Ray-hit shading shader (Fidelity/RT)**: GPU-indexed texture (T#) tables are bound as
  bindless sampled-image arrays.
- **"texture requires rediscovery before final acquisition"**: every image binding is settled
  before image views are acquired.
- **Crash when a gamepad is remapped** by SDL.
- **Missing letters in the boot legal notice and other text**: the compute shader that copies
  glyphs into the font atlas reads the CPU-rasterized bitmaps through DMA; memory without a cached
  buffer read as zero, so glyphs on those pages stayed blank. DMA base registers are now cached
  before the shader runs.
- **Crash at the car dealership (Franklin and Lamar)**: pixel shader `0xf6b18542ea0e938a` builds
  its sampler's border color word from lane data; the untrackable bits are now dropped (border
  color only) instead of failing the shader.
- **Streaming crash late in the prologue** ("[RAGE] HDD Streamer" in zlib `inflate_fast`): zlib
  keeps locals in the SysV red zone, which Windows overwrites when it delivers the exceptions the
  emulator's memory tracking raises. Red zone protection is now enabled by default.

### Shader instruction accuracy

- `V_CVT_PK_U16_U32` / `V_CVT_PK_I16_I32` saturate to 16 bits.
- `V_ALIGNBYTE_B32` uses only `S2[1:0]` as the byte offset (fixes sizes read from byte tables in
  the BVH shaders).
- `V_FRACT_F32` / `V_FRACT_F16` results stay below 1.0.
- Float inline constants read by 64-bit operands use their double-precision encoding.
- `DS_MIN_F32` / `DS_MAX_F32` behave as one-data min/max (bounds reduction).
- Texel offsets of `IMAGE_SAMPLE*_O` are applied.
- Formatted stores convert to SNORM, USCALED/SSCALED and the 11/10-bit float formats.

### Rendering

- Render targets are no longer replaced with stale data after dispatches that may write through
  descriptor tables (the table's 41 MiB view covered the render-target pool).
- Stencil and depth-bias dynamic state is set for every draw (Vulkan validation errors).
- Page faults on pages that cache tracking protects outside the GPU-mapped ranges are resolved
  instead of reported as guest crashes.

### Performance

- Bindless tables: resolved elements are cached between dispatches and only re-resolved when
  their memory changes.
- Buffer device address synchronization only visits CPU-dirty pages and is skipped when nothing
  changed.
- Render-target fast clears whose metadata a compute shader filled no longer read the metadata
  back from the GPU.
- Synchronous CPU readbacks publish their data on the waiting thread.
- The Vulkan pipeline cache is kept across emulator builds and written periodically, so a crash
  does not lose it.
- RELEASE_MEM label writes no longer submit a command buffer each (about 200 submits per frame
  down to about 20).
- The per-draw render state is reused instead of zeroing 36 KiB per draw.
- Guest memory reads and range clamps on the GPU thread resolve their mapping without the global
  memory lock (per-thread caches validated by a mapping generation).
- Buffer device address synchronization only visits tracking regions dirtied since the last one.
- Shader resource tables are evaluated through a compiled node graph per resource plan (about
  3.8x faster than interpreting the IR on every draw).
- Shader stage lookup keys are built without per-word vector work.

## Known bugs and crashes

- **Settings text**: menu text was once drawn with missing letters ("Re dimie" for "Rendimiento");
  probably the glyph-atlas bug fixed below, not seen since.
- **Streaming crash on an unmapped output buffer** (seen once): the streamer flushes a
  decompression destination that is not mapped. Under investigation.
- **Black particles and effects**: some particles, smoke and other effects that should be
  transparent render black.
- **Vehicle damage**: cars sometimes deform incorrectly in collisions and later snap back to
  their original shape.
- **Ray tracing is not implemented**: dispatches with BVH intersection instructions are skipped,
  so ray-traced effects are missing.
- **Performance**: 20-22 fps walking outdoors, lower while driving. The emulated GPU thread is
  still the bottleneck (about 3,000-4,000 draws per frame).
- **Occlusion queries** are treated as always visible, which costs GPU time.

## Reporting a problem

Open an issue with the crash message, the game mode (Performance, Performance RT or Fidelity) and
where in the game it happened. Crash reports from the emulator include the faulting thread,
registers and stack, which are usually enough to locate the problem.
