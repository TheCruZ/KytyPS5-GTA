# KytyPS5-GTA

Fork de KytyPS5 centrado en que **Grand Theft Auto V (PS5, PPSA04263)** funcione bien.
Algunos arreglos de aquí se proponen al emulador principal mediante pull requests.

Responde en español.

## Repositorios y ramas

| Remoto   | URL                                           | Uso                                  |
|----------|-----------------------------------------------|--------------------------------------|
| `origin`   | https://github.com/TheCruZ/KytyPS5-GTA        | este fork; rama `main`               |
| `upstream` | emulador principal (KytyPS5)                  | base; solo lectura                   |

| Rama             | Contenido                                                                 | Dónde             |
|------------------|---------------------------------------------------------------------------|-------------------|
| `main`           | main del fork: `upstream/main` + nuestros arreglos limpios. Sigue a `origin/main`. Es lo único que se sube. | sin worktree propio; build limpio para jugar en `_Build\clean` (ver abajo) |
| `debug`          | `main` + **un único commit final** `debug: local instrumentation and tooling (never push)` con toda la depuración local y este `CLAUDE.md`. **Nunca se sube.** | `C:\Users\mherr\NoSync\KytyPS5` (build de depuración) |

La carpeta de trabajo `C:\Users\mherr\NoSync\KytyPS5` tiene siempre sacada la rama `debug`, así que `git status` debe
estar limpio. El commit de depuración va siempre el último.

## Forma de la historia

- `origin/main` = `upstream/main` + nuestros commits encima. Nunca merges: siempre rebase sobre el principal.
- **1 commit = 1 arreglo.** Una corrección posterior de un arreglo nuestro se funde en su commit
  (`git commit --fixup=<sha>` + `git rebase -i --autosquash`), no se añade como commit aparte.
- Cada mensaje explica el problema en GTA V (hash del shader, síntoma) y la corrección, en inglés,
  con el estilo de los commits del principal (`shader: ...`, `graphics: ...`, `kernel: ...`).
- **Nada de depuración en los commits de `main`.** Contadores, perfiladores, prints de
  diagnóstico, variables `KYTY_*` de pruebas y experimentos opcionales que no mejoran van en el
  commit de depuración de la rama `debug`.

### Añadir un arreglo nuevo

1. En la rama `debug`, commitea **solo los ficheros del arreglo** (nunca `git add -A` si hay
   depuración nueva sin commitear). Si un fichero mezcla arreglo y depuración nueva, commitea el
   arreglo desde una copia limpia (`git show HEAD:<fichero>`, reaplicar solo el arreglo).
2. Llévalo a main y vuelve a poner la depuración encima:
   ```bash
   FIX=$(git rev-parse HEAD)                       # el arreglo, encima del commit debug
   git switch --detach main && git cherry-pick $FIX && git branch -f main HEAD
   git switch debug && git rebase main   # descarta el duplicado; debug vuelve a ir el último
   ```
3. Recompila el build limpio (`kyty_build_gtav_clean.cmd`) y el de depuración.

Para cambios nuevos de depuración: `git commit --fixup=<commit debug>` y
`git rebase -i --autosquash main` (se mantiene un único commit de depuración).

Si se reescribe la historia de `main` (squash de fixups, reordenar, rebase sobre el
principal), no uses `git rebase main` en `debug`: intentaría reaplicar los commits
antiguos. Reaplica solo el commit de depuración: `git rebase --onto main debug~1 debug`.

Para agentes en otros worktrees: parten de `main` y añaden la depuración con
`git cherry-pick debug` (o la dejan sin commitear con `git cherry-pick -n debug`).

### Depuración local (commit de la rama `debug`)

Incluye, entre otros: `KYTY_GPU_CHECKPOINTS`, `KYTY_SKIP_UNSUPPORTED_SHADERS` +
`KYTY_UNSUPPORTED_SHADER_DIR` (vuelca también pixel/vertex shaders que fallan), `KYTY_CACHE_STATS`,
`KYTY_VALIDATION_NONFATAL`, `KYTY_GPU_PROFILE` (gpuProfiler.*), `KYTY_PHASE_PROFILE` (ms/frame y
µs por fase del hilo GPU), contadores de espera de las colas del pipeline (`spscQueue.h`),
`KYTY_ASYNC_READBACK` y `KYTY_GPU_INDIRECT_THREADS` (experimentos más lentos, apagados),
`KYTY_NO_KNOWN_FILL`, `KYTY_TABLE_CACHE_STATS`, `KYTY_WATCH_WRITE`/`KYTY_WATCH_VALUE` (watchpoint
de hardware), `KYTY_TRACE_MAP=<fichero>` (traza de mapeos), `KYTY_FAULT_DUMP=addr:size` y volcados
de memoria en los informes de fallo del guest, prints de mando/ReadFile/APR/AMM y
`KYTY_TRANSLATE_BIN=<cs_hash.bin>` en `shader_cfg_tests` (traduce un shader volcado sin arrancar el
juego), `KYTY_SKIP_NOTICE_SCREENS=1` (`sceSystemServiceGetNoticeScreenSkipFlag` = true: GTA no
reproduce los logos ni el aviso legal; `kyty_perf_quick.sh` lo activa, `SKIPINTRO=0` lo quita;
fuerza la opción `--skip-notice-screen true` que el principal trae desde `cc44a43e`),
`KYTY_URGENT_UNMAP=0` (A/B: los unmaps vuelven a esperar a todo el atraso de la GPU), con
`KYTY_PHASE_PROFILE` los eventos `wrm q<cola> <cpu|journal|gpu-write> <dir>` (tiempo que el CP
espera en cada WAIT_REG_MEM; >15 ms imprime `WRM wait`) y las fases `gb: *`/`vb: *` (desglose de
`draw: graphics bindings` y de los búferes de vértices), y el nombre del hilo en `KYTY_WATCH_WRITE`.
Cuesta ~15-20 % de rendimiento compilado; para jugar se usa el build limpio.

**Ablación (medir el techo ANTES de optimizar)** (`src/common/ablation.h`, 07/08-10-2026): quitan trabajo aunque la
imagen salga mal, para saber cuánto cuesta de verdad. Todos se activan a los `KYTY_SKIP_AFTER_SECONDS=<n>` (en
partida: el arranque necesita esos shaders; desde el arranque GTA se queda en "Updating").
- `KYTY_SKIP_SHADERS=<hash>,...` / `KYTY_SKIP_QUEUES=<cola hex>,...` / `KYTY_ABL_SKIP_DRAWS=1`: el CP descarta esas
  dispatches/colas/draws (ni se resuelven ni se graban).
- `KYTY_NOGPU_SHADERS` / `KYTY_NOGPU_QUEUES_HEX` / `KYTY_NOGPU_DRAW_SHADERS`: mantienen la CPU y quitan la GPU (separa
  coste de CPU del hilo de ejecución y de GPU).
- `KYTY_ABL_SKIP_DRAW_SHADERS=<vs>:<ps>,...`, `KYTY_DRAW_HIST=1` (draws por par VS/PS), `KYTY_ABL_STALE_READS` (leer
  bytes viejos: DeviceLost en Fidelidad).
- `KYTY_EXEC_TRACE=<prefijo>`: traza por operación del hilo de ejecución (cola, hash, inicio/fin, esperas, dependencia
  WAIT_REG_MEM); análisis con `xtrace.py`/`chain.py` (en los `_Build` de los worktrees de exec-batch/fidelity-diff).
- `KYTY_WATCH_WRM=<cola>`: watchpoint de hardware sobre el contador que espera esa cola (>30 ms) para ver quién lo
  escribe (así se vio que los EOP los escribe el propio hilo de ejecución).
- `KYTY_GPU_PROFILE=1`: GPU por shader también con el grabador diferido.
- Experimentos de exec-batch (`src/common/expSwitch.h`): `KYTY_X_<NOMBRE>=0` apaga uno, `KYTY_X_ALL=0` todos; los que
  no entraron en main (ASYNC_UPLOAD, LONG_SPIN, STRICT_NOCOPY, UNLOCKED_MAT, IT_CPU, LANE_R, ASYNC_NARROW)
  son opt-in (`=1`).
Diarios con ejemplos: `_Tools/fidelity_diff_log.md` (ablación Fidelidad vs Rendimiento; partida de Rendimiento en la
misma escena: el modo va en `_DownloadData/PPSA04263/saves/1000/PROFILEB`), `_Tools/exec_batch_log.md`.

## Sincronizar con el principal y subir

```bash
git fetch upstream
git switch --detach main && git rebase upstream/main   # resolver conflictos si los hay
git branch -f main HEAD
#   compilar y pasar los tests en esta punta limpia (ver abajo)
git switch debug && git rebase main                  # la depuración vuelve encima
TOKEN=$(gh auth token -u TheCruZ)
git -c credential.helper= -c "credential.helper=!f() { echo username=TheCruZ; echo password=$TOKEN; }; f" \
    push --force-with-lease=main:<sha actual de origin/main> origin main
```

- Sube siempre con la cuenta **TheCruZ** (la cuenta activa de `gh` puede ser otra; no la cambies).
- Reescribir `origin/main` exige force push: pide confirmación al usuario antes.
- Basta con compilar y probar la punta; compilar cada commit por separado es demasiado lento.
- El commit del README es propio del fork y va el último de `main` (antes del de debug).

## Compilar y probar (Windows, clang-cl + Ninja)

```bat
C:\Users\mherr\NoSync\KytyPS5\_Tools\kyty_build_gtav_target.cmd kyty_emulator launcher ^
    resource_tracking_tests resource_materialization_tests shader_recompiler_compute_tests ^
    shader_cfg_tests memory_tracker_tests page_manager_tests gpu_pipeline_tests
```

Compila `C:\Users\mherr\NoSync\KytyPS5` (rama `debug`). Los ejecutables quedan en `_Build\windows` (tests en sus
subcarpetas); todos deben terminar con código 0. Si CMake intenta descargar FFmpeg,
`FFMPEG_PREBUILT_DIR` está fijado en `CMakeCache.txt` a `_Build/windows/externals/ffmpeg-9ac4cfd195f1`.

`kyty_build_gtav_clean.cmd [-Rev <commit>] [-Force]` compila el build limpio de `main` en
`_Build\clean` sin un segundo árbol de fuentes: cambia temporalmente esta carpeta a `main`, compila y
vuelve a `debug` (exige `git status` limpio; no hace nada si `_Build\clean` ya está en ese commit).
Restaura las fechas de los ficheros que difieren entre ramas, así que ninguno de los dos builds
recompila de más. Comparte `_PipelineCache` con el build de depuración mediante un junction.
`_SaveData` ya NO se comparte (30-09-2026). OJO: una partida de GTA son DOS carpetas de la
carpeta del exe: `_SaveData` (autoguardado en `sce_sdmemory@0\memory.dat`) y `_DownloadData`
(slots `PPSA04263\saves@0\SBGTA5xxxx`); el juego carga la más reciente, así que copiar solo
`_SaveData` no cambia de partida. La de casa de Franklin (rutas skyfall) está en `_Build\windows`
(sin slot SBGTA) y ahora también en `_Build\clean`; la de la misión de Franklin y Lamar del usuario
está en `_Build\clean\_SaveData_lamaryfranklin` + `_DownloadData_lamaryfranklin` y, para el test
de daño de vehículos, en `_Tools\saves\vehicle_damage` + `vehicle_damage_download`. Recompílalo tras cada commit nuevo en main; no lo lances mientras otro proceso use el árbol.

**Partida de lluvia (08-10-2026, modo Rendimiento)**: `_Tools\saves\rain_green_reflections_20261008` (`_SaveData` +
`_DownloadData`). Sirve para dos cosas:
- **Bug de reflejos**: llueve y Michael está dentro de su coche, con la cámara en tercera persona mostrando el césped
  de alrededor. A los pocos segundos de entrar aparece el agua del suelo y sus
  reflejos se ven VERDES (no deberían). Espera a que aparezca el agua antes de capturar.
- **Rendimiento (modo Rendimiento) y A/B**: empieza a ~50-55 fps y baja a ~40-45 tras 1-2 minutos dentro de la partida;
  mide pasado ese tiempo (o compara el mismo tramo temporal en A y B).
Instalarla en tu carpeta de build: `SAVE_DIR=/c/Users/mherr/NoSync/KytyPS5/_Tools/saves/rain_green_reflections_20261008
INSTALL_SAVE=1 bash _Tools/kyty_mountain.sh <exe> <etiqueta>` (`kyty_fidelity.sh` también acepta `SAVE_DIR`; la partida
carga en Rendimiento). Nunca la instales en `_Build\clean`/`_Build\windows`.

## Probar el juego

Herramientas en `C:\Users\mherr\NoSync\KytyPS5\_Tools`:

- `kyty_gtav_test.ps1`: prueba automática (arranca, pulsa teclas, capturas, `fps N`); resultados en
  `_Build\windows\gtav_runs\<fecha>`. Ruta rápida en Los Santos:
  `waitframe 1200; presslanding; menuindividual; waitingame; key I;
  wait 8; down C; fps 20; up C` (J=Cross, E=R1, I=Triangle, C=R2, W/A/S/D stick izquierdo).
  El aviso legal del arranque se ve entre los frames ~800 y ~1100. `menuindividual` mira qué pestaña
  del menú (ONLINE/GTA+/INDIVIDUAL) está resaltada y pulsa R1 hasta INDIVIDUAL antes de entrar
  (dos R1 a ciegas perdían pulsaciones); el script es consciente del escalado de Windows (125 %). `waitingame`
  pulsa J (pasa las cargas) solo hasta ver la barra de vida verde del HUD, en vez de 35 J fijas.
- `kyty_gtav_play.cmd`: sesión manual registrada (F9 marca + captura, FPS, volcado de shaders no
  soportados, `summary.txt`); usa el build limpio por defecto; `-Debug` usa el de depuración con
  watchpoint, traza de mapeos y volcados. Resultados en `_Build\windows\gtav_play\<fecha>`.
  `-Profiling` (o `kyty_gtav_profile.cmd`) usa el build de depuración con perfil de fases, tirones
  (`KYTY_HITCH_MS`, 50 por defecto: fases/eventos/caché de cada frame lento), GPU del host (`cb`),
  CacheStats, CPU por hilo cada 5 s (`threads.log`; los hilos llevan el nombre del juego) y muestreo
  de pilas (`_Tools\sampler`, cada 60 s y con F10); al cerrar `kyty_profile_report.py` escribe
  `profile_report.txt` (bloques lentos frente a rápidos, peores tirones, CPU por hilo, pilas).
- `kyty_gtav_play_renderdoc.cmd`: igual bajo RenderDoc (F1 captura). En Fidelity la captura usa
  ~30 GB de RAM; captura en modo Performance.
- `kyty_wpr_capture.ps1`: traza ETW de CPU con pilas (pide UAC al usuario); analizar con
  `xperf -i x.etl -symbols -o dump.txt` y los scripts etw_threads.py / etw_inclusive.py.
- Todas las ejecuciones del emulador (pruebas, agentes, partidas) respetan el candado
  `C:\Users\mherr\NoSync\KytyPS5\_Build\emulator_run.lock`: un solo emulador a la vez.
- Nunca navegues por los menús de ajustes del juego en pruebas automáticas: la partida guardada es
  la del usuario (una prueba cambió el modo gráfico a Fidelidad). Mide siempre en modo Rendimiento.
- Los vídeos de arranque son Bink del propio juego (no AvPlayer); el build de depuración los salta
  con `KYTY_SKIP_NOTICE_SCREENS=1` (el limpio ignora la variable, pero acepta la opción del principal
  `--skip-notice-screen true`, p. ej. `EXTRA="--skip-notice-screen true"` en los scripts).

Usa siempre estos dos builds (`C:\Users\mherr\NoSync\KytyPS5\_Build\windows` = depuración, `C:\Users\mherr\NoSync\KytyPS5\_Build\clean` = limpio).

## Rendimiento

El cuello de botella es la GPU emulada (hilos del pipeline: CP, resolución, ejecución, grabación
Vulkan; `KYTY_GPU_PIPELINE=0` lo ejecuta todo en un hilo, `=3` sin la etapa de resolución). La etapa
de resolución (`Thread_GpuResolve`, núcleo 10) elige programas y materializa el SRT de cada draw por
adelantado; ejecución valida las lecturas (`GuestReadLog`) y si no valen lo rehace.
Medición rápida: `_Tools\kyty_perf_quick.sh` (µs/draw de ejecución con `KYTY_PHASE_PROFILE`); ruta de
referencia fuera de la casa de Franklin: `ROUTE="down W; wait 8; down D; fps 30; up D; up W"`.

**Skyfall v2 (preferida)**: `bash _Tools/kyty_skyfall2.sh [exe] <etiqueta> [nosample]` (variables de
kyty_perf_quick.sh; `MEASURE=15` s de medida). Al aparecer anda 2,5 s (dentro de la casa el truco no
funciona), mete invencibilidad + caída del cielo y, en el aire, `seekminfps` da 8 toques de A (~45°,
una vuelta = ~5 s de A), mide fps contando frames en cada dirección, vuelve con D a la peor, ajusta
un toque y mide ahí: no depende de hacia dónde mire Franklin. test.log: "seekminfps: barrido [...]"
y "FPS mean=". Mirar hacia el centro da ~25-35 fps y hacia el mar ~60.

**Prueba skyfall (la mejor para medir en movimiento)**: `ROUTE="replay skyfall" bash
_Tools/kyty_perf_quick.sh [exe] <etiqueta> [nosample]` reproduce `_Tools\routes\skyfall.keys`
(66 s, grabada por el usuario con `_Tools\kyty_keyrec.py`: F8 empieza/termina, F9 marca captura;
el paso `replay <nombre>` de `kyty_gtav_test.ps1` la reproduce y da `FPS mean/min/max`). Sale de
casa de Franklin, mete invencibilidad + paracaídas + caída del cielo (Skyfall) y planea sobre la
ciudad hasta aterrizar. `skyfall_shots.keys` = la misma con capturas cada 10 s. La ruta no es
determinista: a veces acaba en la playa (~50 fps, pocos draws) y otras entre los edificios del
centro (la zona exigente). Compara varias pasadas y por bloques de 5 s según draws/frame
(`awk` sobre `==== phase profile` y `draw: tail calls/frame`), no solo la media. Para A/B dentro
del mismo binario usa un interruptor de depuración temporal (así se midió que dejar al juego un
frame por delante de la GPU daba +30-100 % en el centro; el principal lo trae desde `840b9f57`,
`GuestGpu::SuspendPoint`). Trucos que funcionan (teclas del emulador): invencibilidad
`→ J → ← → E → ← J I` (acaba en Triángulo: métela lejos de coches), paracaídas
`← → Q Z E C C ← ← → Q`, skyfall `Q Z E C ← → ← → Q Z E C ← → ← →`. El helicóptero (Buzzard) se
rechaza a menudo por falta de espacio y el Triángulo sube a coches cercanos: no lo uses.
El truco a veces no se activa y Franklin sigue a pie (pasada inválida): mira siempre las capturas
de `skyfall_shots` antes de dar cifras. Estado (30-09-2026, build de depuración con
`KYTY_PHASE_PROFILE`, modo Rendimiento): 36-42 fps de media en skyfall; 30-33 fps
con 3700-4000 draws/frame, ~20-26 con 4500.
Tirones de streaming (01-10-2026, resueltos en el commit de rendimiento): la cola gráfica espera
(WAIT_REG_MEM) a un contador (p. ej. `0x242d45900`) que un trabajo `[RAGE] Dependency` incrementa
tras vaciar un lote de `sceKernelBatchMap2` con un mutex del juego tomado; cada munmap del lote
esperaba al atraso entero del hilo de ejecución y `VirtualRanges` era O(n). Si vuelven, mira
`wrm q0 cpu ...` y `ev: unmap`/`game: SendUrgentCommandSync` en los tirones.

### Medir sincronizaciones y el camino crítico (KYTY_SYNC_ACCT)

`bash _Tools/sync_audit/run.sh [dir del exe] <etiqueta> fid|fidnort|perf` (build de depuración; instala la
partida salvo en `_Build\windows`/`_Build\clean`) deja en `<exe>/sync_runs/<fecha>/report.txt` la contabilidad
cerrada por hilo (pared = trabajo + esperas + residuo), el frame del hilo de ejecución por categorías, los huecos
de la GPU del host y el inventario de puntos de espera. Guía, métricas y tabla de referencia del 14-10-2026:
`_Tools/sync_audit/README.md` (diario: `_Tools/sync_wait_audit_log.md`). Cuesta ~6 % de fps.
- Medido y verificado con retardo inyectado (`XT=10 XTEXP=EXECDELAY`): **la etapa crítica es el hilo de
  ejecución** (ocupado 96-98 % del frame; 75 % trabajo de CPU en Fidelidad, 93 % en Rendimiento). Retrasar el
  juego (`GAMEDELAY`) no cambia los fps; la GPU parada es consecuencia de que ejecución no ha enviado aún.
- El reparto núcleo/usuario de `GetThreadTimes` no es fiable (45 % o 10 % según la pasada con el mismo
  trabajo): usa los sitios instrumentados o el muestreo de pilas.
- Toda afirmación de rendimiento debe apoyarse en esta contabilidad o en un A/B emparejado dentro de la misma
  pasada (`XT`/`xt.py`), no en suposiciones ni en pasadas sueltas.

Requisito del usuario: caminos calientes sin
bloqueos, espera activa con `_mm_pause`, hilos fijados a núcleos
(`KYTY_GPU_THREAD_CPUS`), ninguna transición al kernel entre draws. Diseño y plan en
`C:\Users\mherr\NoSync\KytyPS5\_Tools\gpu_pipeline_design.md`; contexto para agentes en
`perf_agent_context.md`. La coherencia por envío (propuesta C) está en la rama
`mcruzvenn/gpu-pipeline` (3 commits sobre el pipeline), sin integrar: no aporta con el pipeline
activo hasta M6.

## Equipo de agentes (Opus coordina, Sonnet/Haiku trabajan)

`_Tools\kyty_orchestrator.cmd ["tarea"]` abre una sesión con `claude --agent kyty-orchestrator --effort max` (el `effort` del frontmatter no se aplica a la sesión principal): Opus
solo planifica, revisa y delega; no tiene Edit/Write/Bash y solo puede lanzar estos subagentes
(`.claude/agents/`, en el commit de depuración):
- `kyty-scout` (Haiku, medium, sin CLAUDE.md): búsquedas de solo lectura, resúmenes de logs y diarios.
- `kyty-implementer` (Sonnet, high): código en su propio worktree, commits, compilación (-j 8) y tests.
- `kyty-runner` (Sonnet, medium): lanzamientos del juego y medidas (fps, `sync_audit`, A/B XT, capturas).
El script desactiva los forks (`CLAUDE_CODE_FORK_SUBAGENT=0`, heredarían Opus) y fija
`CLAUDE_CODE_SUBAGENT_MODEL=sonnet`; no definas `CLAUDE_CODE_EFFORT_LEVEL` (anula el effort de los subagentes).
Los subagentes gastan del mismo cupo, pero mucho menos que Opus; sus resúmenes vuelven a precio de Opus, por eso
se piden cortos (≤ 15 líneas). Comprueba modelo y effort de cada uno con `/tasks`.

## README del fork

`README.md` es propio del fork (commit `docs: describe the GTA V fork in the README`): enlaza el
README original y mantiene la lista de bugs/crashes arreglados y conocidos. Actualízala en ese
mismo commit (fixup) cuando se arregle o descubra algo. Nunca va en una PR al principal.

## Pull requests al principal

Rama nueva desde `upstream/main` con los commits elegidos (`git cherry-pick`), compilar y probar, y
abrir la PR contra el repositorio principal. No incluyas depuración ni commits que solo tengan
sentido para GTA V.
