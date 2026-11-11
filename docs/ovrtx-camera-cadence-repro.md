# OVRTX multiview capture reproduction

The pinned OVRTX build returns incorrect camera orientations when the active
set of multiview products changes. Frame counts can remain correct. Stable
normal batches, mixed-rate single-view products with stable native gating, and
persistent native renderers with one product each pass the standalone checks.
Serializing due products also avoids the ordinary changing-batch failure when
initialized through single-product submissions, but product re-entry is expensive
and startup transitions require separate checks.

`src/holosoma/tests/simulators/ovrtx_cadence_repro.py` imports only OVRTX, NumPy
and the Python standard library. It generates a USD scene with a red cube,
three camera groups and RGB/depth render variables. Alternate groups rotate
90 degrees about the optical axis, making camera substitutions observable.
There are no robot assets, physics, Holosoma, Isaac Lab, Kit, Warp kernels,
GPU attribute bindings or SDK image extraction. The minimal repro compares
CPU-mapped RGB against projected cube positions. The follow-up `ovrtx_instances_repro.py` and isolated layout/API experiments
strengthen this to distinct 0/90/180-degree camera rotations
and RGB/depth verification in every environment tile.

On far044's RTX 5090 with `ovrtx==0.5.1.385782`, run from an environment containing
that wheel and NumPy:

```bash
python src/holosoma/tests/simulators/ovrtx_cadence_repro.py \
  --mode periodic --instances 2 --output-dir /tmp/ovrtx-periodic
```

The three render products contain two camera views each. Native `OmniSensorAPI`
rates are 50, 10 and 25 Hz, aligned to the initial capture by `frameRate` and
`startTime`. Forty control cycles produce the correct 40/8/20 frame counts, but
some images have another group's orientation. The process exits with status 1
and writes the generated `scene.usda`, native log and per-frame measurements
to `result.json`.

Initial controls on the same host:

| Mode | Views per product | Result |
| --- | --- | --- |
| `auto` | 2 | Pass: unrestricted stable batch, 40/40/40 correct images |
| `periodic` | 1 | Pass: mixed native rates, 40/8/20 correct images |
| `periodic` | 2 | Fail: correct counts, incorrect camera orientations |
| `subset` | 2 | Fail: selecting due products with native rates is insufficient |
| `manual` | 2 | Fail: selecting due products without timing schemas is insufficient |
| `serial` | 2 | Pass: individual due-product submissions and immediate readback |

`--decimations 2,3` also reproduces the two-view native-rate failure while
returning the expected 20/14 frames. `--disable-spg` does not fix it.
`--shader-cache PATH` reuses an existing shader cache through OVRTX's settings
extension; it imports no SDK and changes no camera settings.

The later standalone fixtures give every environment a different moving-target
history. The layout/API fixture changes target x and z each cycle, making stale
depth distinguishable from current geometry. Both the layout and persistent
renderer fixtures check each due tile's RGB and depth against projected target
geometry, distinct camera orientations, expected deliveries, partial camera-pose
changes and held observations. Native mappings are read and copied immediately
before another native submission invalidates their borrowed output handles.
These tests verify copied CPU-buffer lifetimes; the SDK must separately verify
its GPU buffers, actual camera resets, movie coexistence and physics advancement.

The common environment was RTX 5090, driver `580.178.04`, OVRTX
`0.5.1.385782`, NumPy `2.5.1` and Python `3.12.14`. Experiments reused the
installed environment and warmed shader cache. Each benchmark cohort ran
sequentially under an exclusive GPU lock. Process VRAM is deduplicated NVIDIA
compute-process memory, not subtraction from changing device-wide allocations.
Startup includes native settings initialization, renderer construction, scene
opening and the first completed capture, with warm caches.

The first matched layout cohort used eight environments, two cameras at
decimations 1/5, 512×512 per environment, 100 cycles and 20 warmup cycles. Every
row uses the same generated cube scene, trajectory, camera settings and immediate
CPU readback; only the intended product layout and scheduling strategy change.
All four rows passed every environment's RGB/depth checks.

| Layout / submission | Native step ms/cycle | CPU read/copy ms/cycle | Capture total ms/cycle | Measured view images | Native product frames | Process VRAM MiB | Startup s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiled, stable normal batch | 4.158 | 28.766 | 32.924 | 1280 | 160 | 4409 | 2.779 |
| Tiled, serialized due products | 75.707 | 14.344 | 90.051 | 768 | 96 | 4713 | 2.807 |
| Flat, stable submission with native gating | 15.642 | 20.128 | 35.770 | 768 | 768 | 5264 | 4.187 |
| Flat, due-products subset | 292.202 | 20.321 | 312.524 | 768 | 768 | 5696 | 4.180 |

The measured window contains 80 cycles. Normal batching captures 80 views per
environment from each camera; scheduled strategies capture 80/16. A tiled
product frame contains eight views, while a flat product frame contains one.
Product frame counts therefore require the layout context. Flat native gating
is correct and about 4.8× faster than serialized submission in native step
time, but it remains about 3.8× slower than the efficient normal native batch.
Merely replacing multiview products with single-view products does not remove
the cost of product re-entry.

A smaller matched layout control used two environments, three cameras at
1/5/2, 128×128, 40 cycles and 10 warmup cycles:

| Layout / submission | Correctness | Capture total ms/cycle |
| --- | --- | ---: |
| Tiled, stable normal batch | Pass | 6.435 |
| Tiled, serialized due products | Pass | 152.103 |
| Flat, due-products subset | Pass | 227.320 |
| Flat, stable native gating | Pass | 7.220 |
| Packed, all views in one native-gated product | Fail: 104 idle RGB/depth tiles changed | 3.090 |

The packed product returned a framebuffer each cycle while the fixture's
expected due-camera counters still looked plausible. Every idle opportunity
changed a tile that should have held its previous image. Product frame counts
do not expose which views refreshed, so its low timing is not a correct
independent-camera performance result. The official multi-rate guide recommends
a separate RenderProduct for each independently scheduled sensor. It does not
state that the first camera necessarily determines all multiview clocks; the
packed failure is an empirical result.

The second matched cohort investigated persistent native renderer instances.
It used eight environments, two cameras at 1/5, 512×512, 60 cycles and 10
warmup cycles. Every row shares that cohort's scene, target histories, partial
pose/target changes, warmed caches and immediate readback. Its trajectory,
startup procedure and measured window differ from the layout cohort above;
compare strategies within each table rather than combining absolute timings.

| Strategy | Native step ms/cycle | Capture + CPU readback ms/cycle | Measured camera captures | Process VRAM MiB | CPU RSS MiB | Startup s | Correctness |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| Stable normal batch | 4.605 | 19.349 | 50/50 | 4409 | 3901 | 3.019 | Pass: 960 tile checks over all 60 cycles |
| Serialized selection after all-products prime | 93.730 | 103.840 | 50/10 | 4986 | 3902 | 3.102 | Fail: 11 tile checks; see startup caveat |
| Persistent renderer per product | 4.452 | 14.911 | 50/10 | 4686 | 4675 | 3.881 | Pass: 576 tile checks over all 60 cycles |

Full-run captures are 60/60 for normal batching and 60/12 for the scheduled
strategies. Persistent instances add approximately 0.86 seconds of startup,
277 MiB of process VRAM and 774 MiB of CPU RSS relative to normal batching in
this scene, while preserving approximately the normal native capture cost.
They skip idle captures without repeatedly switching the native engine's
product membership.

The serialized row in the persistent-instance cohort primes all products
together, then switches to single-product submissions. On cycle zero, camera
zero shows camera one's orientation in all eight tiles. This differs from the
layout cohort's single-product-first serial initialization. Three further
slow-camera RGB-only centroid errors occur at cycles 10, 45 and 50
(4.80–5.84 pixels); matching depth remains within 0.4 pixels. These failures
remain reported and were not attributed to partial reset. This row is a
startup/lifetime diagnostic, not a fully correct baseline. A serial comparison
must match the production startup sequence and verify its first delivered frames.

A later single-product-first serial control used the same 8-environment,
512×512, 1/5, 60-cycle/10-warmup workload. Its eight startup orientation failures
disappeared, but the same three RGB-only errors remained, with correct depth and
no orientation failures. The process retained exit status 1. Native step time
was 82.791 ms/cycle, capture plus CPU readback 91.840 ms/cycle, startup 3.414
seconds and final process VRAM 5194 MiB (sampled peak 5258 MiB). This supplemental
control ran later after unrelated resident GPU allocations were released, so it
isolates the startup behavior rather than replacing the matched cohort's timing
comparison. The remaining RGB errors are reported rather than treated as a
correctness pass.

The normal standalone control reads every frame for freshness checks, including
frames that are not due observations. The SDK normally consumes only requested
observations. Native step time is the primary standalone performance comparison;
the CPU-copy columns are not predictions of SDK GPU-buffer costs. SDK integration
has additional moving robots, bindings, output ownership, resets, movies and
physics checks, and needs its own matched measurements. Any newer SDK benchmark
numbers should be reported separately from these diagnostic cohorts.

Both flat native gating and persistent per-product renderers passed the required
two-environment, 40-cycle rate matrix with partial pose changes:

| Decimations | Per-environment captures over 40 cycles | Flat stable gating | Persistent renderers |
| --- | --- | --- | --- |
| 1/5 | 40/8 | Pass | Pass |
| 2/3 | 20/14 | Pass | Pass |
| 1/5/2 | 40/8/20 | Pass | Pass |
| Uniform 3/3 | 14/14 | Pass | Pass |

Persistent renderers also passed identical camera/product-path controls in
independent scenes and a control that created/opened both instances before
priming either. Those controls do not require distinct scopes or an artificial
one-instance-at-a-time startup ordering for the tested pinned scenes.

Additional API/build checks used the strengthened every-tile fixture:

| Runtime / control | Result |
| --- | --- |
| 0.5.1.385782, stable unrestricted multiview | Pass |
| 0.5.1.385782, multiview native rate gating | Fail: correct counts, RGB/depth orientation errors |
| 0.5.1.385782, CPU transform propagation | Fail: orientation errors remain |
| 0.5.1.385782, synchronous debug mode | Fail: orientation errors remain on the otherwise free GPU |
| 0.5.1.385782, motion BVH enabled | Unverified: first frame incomplete after the 90-second bound |
| 0.6.0.1791947691, stable unrestricted multiview | Unverified: first step incomplete after 263 seconds; terminated |

The isolated 0.6 native build was
`0.6.0+release.1791947691.c6d71501.gl`. It used renderer-owned USD-string mode,
avoiding its attached-stage requirement for OVStage 0.3.x; the shared pinned
OVStage `0.2.1.385922` installation stayed unchanged. Neither a timeout nor
cache writes establish correctness or a completed startup cost. No production
runtime upgrade was made. The motion-BVH warning about unsupported SRT GPU
transforms does not establish the cause of its timeout.

The experiments isolate the failure to the pinned native changing-multiview
batch path, without identifying its internal implementation cause. Keeping
enabled products selected is also the documented way to avoid discarding their
accumulated histories and reinitializing them on re-entry. Native gating works
for the tested flat products but requires control-clock and partial-reset phase
integration in the SDK. Flat extraction additionally needs correct per-environment
segmentation metadata/IDs and PPISP assembly.

Persistent per-camera native renderers preserve ordinary SDK multiview output
extraction and avoid those extra flat-product requirements. Each attached
OVStage must have its own native renderer owner; sharing a single attached stage
among engines is unsupported. An SDK adapter must reuse prepared host data
without reauthoring the shared host stage, update only due native owners, read
borrowed output handles before subsequent submissions, and retain the efficient
shared path for single-camera and uniform-rate setups. Scheduling continues
to use the existing `update_decimation`; the investigation needs no new user
configuration field or Kit/PhysX initialization.

The selected integration uses one persistent SDK/native renderer per product for
mixed decimations and perception/movie coexistence. Normal single-camera and
uniform-rate setups retain the SDK's shared batch and do not retain an extra USD
export. Children ingest one prepared immutable USD export and clone plan; they
have independent native stages, bindings, buffers and material writers. Only due
owners publish state and capture. SDK-owned borrowed outputs are extracted before
another step on that engine. The simulation registry owns native backends; cleanup
releases writers and camera bindings before those backends. This uses three private
prepared-state fields from the pinned SDK, which must be rechecked on SDK upgrades.
No user options, quality defaults, asset formats or unused runtime imports change.

Run the stronger persistent-instance control without Holosoma:

```bash
python src/holosoma/tests/simulators/ovrtx_instances_repro.py \
  --strategy instances --instances 8 --resolution 512 --frames 60 --warmup 10 \
  --shader-cache /path/to/existing/shader-cache --output-dir /tmp/ovrtx-instances
```

Use `--strategy normal` for the stable batching control and `--strategy serial
--serial-single-prime` for production-like serial startup. `--decimations 2,3`,
`1,5,2` and `3,3` cover the other rate patterns. `--identical-instance-paths
--defer-prime` checks SDK-like independent stages and initialization ordering.
The script writes every-tile measurements, held-buffer checks, captures, startup,
VRAM and CPU RSS to `result.json`. Failures produce exit status 1.

The integrated G1 workload requires a separate comparison; it includes physics,
USD import/cloning, GPU bindings and SDK extraction. The pre-integration prototype
passed all three matched controls at eight environments, 512×512, two cameras at
50/10 Hz, 60 cycles and 10 excluded warmup cycles:

| Policy | Capture ms/control cycle | Actual full-run captures | Startup s | Process VRAM MiB | Result |
| --- | ---: | --- | ---: | ---: | --- |
| Normal batching | 15.475 | 60/60 | 8.603 | 4853 | Pass |
| Serialized fallback | 114.179 | 60/12 | 7.971 | 5167 | Pass |
| Persistent renderers | 10.211 | 60/12 | 8.155 | 5294 | Pass |

Unlike the diagnostic standalone copies, integrated capture time includes only
SDK-requested due observation extraction. All controls use the same motion,
partial reset, caches and dependencies and check every environment's independently
changing RGB/depth geometry, held observations and exact physics step/time deltas.
The default DLSS reconstruction can leave color trails outside the current target
surface. Tests verify moving depth independently, then RGB on that surface, and
reject a real previous RGB frame paired with fresh depth. Movies additionally
check current geometry and decode saved MP4 frames. Rendering quality defaults
remain unchanged. Startup measurements are single warm-cache samples, not a
general startup benchmark. These prototype numbers are retained as a separate
cohort; the immutable production-image results are recorded in the PR.

### OVRTX API and alternate-build investigation

The standalone investigation uses OVRTX, NumPy and the Python standard library,
without Holosoma, Isaac Lab or Kit. The pinned runtime is **OVRTX 0.5.1.385782**
(native API version **0.5.1**) with **OVStage 0.2.1.385922**, on an RTX 5090 with
driver **580.178.04**.

The official [multi-rate guide][rates] schedules sensors with
`omni:sensor:frameRate`; it explicitly rejects substituting `tickRate`. All
enabled products should remain selected on every native step: omitting a product
discards its rendering history and can cause initialization and warmup when it
returns. `partialOutputs=false` suppresses intermediate delivery, rather than
skipping exposure accumulation. The guide recommends a separate product per
independently scheduled sensor; it does not establish that a multiview product
must use identical rates or that only its first camera controls scheduling.

The [stage ownership contract][ownership] permits only one renderer per attached
OVStage. Persistent renderer instances therefore need independent attached
stages or independent renderer-owned compatibility stages. Diagnostic outputs
were copied before the next native submission; held observations were checked
after later submissions.

The settings probes used three distinct camera orientations, decimations
**1/5/2**, two environments per camera, **128×128** tiles, 40 cycles and 10 warmup
cycles, with a partial reset. Every tile was checked for moving-target RGB/depth
freshness and orientation, rather than relying on capture counts.

| Control | Delivered product frames | Result |
| --- | ---: | --- |
| Pinned runtime, stable all-product batch | 120 | Pass |
| Pinned runtime, native rate gating | 68 | Fail: 64 orientation/RGB/depth errors |
| Native gating, `read_gpu_transforms=False` | 68 | Fail: 65 errors |
| Native gating, `sync_mode=True` | 68 | Fail: 60 errors |
| Native gating, `motion_bvh="enable"` | 0 | Unverified: first-frame timeout after 90 s |
| OVRTX 0.6, stable all-product batch | 0 | Unverified: first-step wait terminated after 263 s |

The alternate wheel was **OVRTX 0.6.0.1791947691**, with embedded native identity
`0.6.0+release.1791947691.c6d71501.gl`. It was installed in an isolated target;
the pinned environment remained unchanged. Version 0.6 attached mode requires
OVStage **0.3.x**; the corresponding published build is **0.3.0.1791986420**.
The probe used renderer-owned USD mode, preserving the installed OVStage.
Its [changelog][changes] includes multiple-renderer, camera/depth interference
and multi-rate Motion BVH fixes, but no explicit fix for this changing-batch
orientation defect. Cache writes during the timeout do not demonstrate completed
startup or correctness; no gated 0.6 run followed the incomplete control.

The movie validation separately verified RGB against the current depth geometry
and rejected an actual previous-frame RGB negative control. A global red centroid
also included temporal trails, so that centroid alone was insufficient evidence
of a stale capture. The pinned native schema defaults `omni:rtx:post:aa:op` to
`"dlss"`; [RTPT uses denoising][render-modes]. Production anti-aliasing defaults
remain unchanged. These checks establish current-surface freshness, without
claiming that temporal trails were eliminated.

[rates]: https://github.com/NVIDIA-Omniverse/ovrtx/blob/b9ae5f0913847502ff5ed29ab45a6e9f28bec8f1/docs/sensors/multi_rate.rst
[ownership]: https://github.com/NVIDIA-Omniverse/ovrtx/blob/b9ae5f0913847502ff5ed29ab45a6e9f28bec8f1/docs/core/ovstage_integration.rst
[changes]: https://github.com/NVIDIA-Omniverse/ovrtx/blob/b9ae5f0913847502ff5ed29ab45a6e9f28bec8f1/CHANGELOG.md
[render-modes]: https://github.com/NVIDIA-Omniverse/ovrtx/blob/v0.5.1/docs/sensors/cameras/render_modes.rst
