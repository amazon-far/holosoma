# OVRTX timing, image quality and SDK compatibility

The pinned OVRTX API advances a sensor clock on every submission. Holosoma
preserves Isaac Lab's fixed `1/60` snapshot increment. Adding a fast persistent
owner produced byte-identical slow-camera images in the matched standalone
fixture. The actual SDK scene also shows current geometry and materials with
both ordinary and independent owners; some old material color is already in
native output before SDK extraction. These are bounded image tests, rather than
a guarantee that every temporal rendering feature is equivalent.

## Exact timing contract

This investigation used OVRTX `0.5.1.385782`, OVStage `0.2.1.385922`,
`isaaclab-ov==5.1.5`, Isaac Lab source
`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`, Python `3.12.14`,
Warp `1.17.0`, Torch `2.11.0+cu130`, and NumPy `2.5.1`, on far044's RTX 5090
with driver `580.178.04`. Installed wrapper, header and schema hashes were
checked against the official pinned sources.

The [pinned native header][native-step] and [result types][result-times] define
`delta_time` in seconds: the renderer simulates `[t, t + delta_time]`, evaluates
sensor schedules over that window, then advances its clock to the endpoint.
Ordered scene writes describe endpoint state. Results expose the actual window
as `simulation_start_time` and `simulation_end_time`; individual frames expose
their capture start/end times. An instantaneous camera without a positive
native rate captures at the endpoint, as checked by the
[official camera test][camera-test].

The [pinned SDK renderer][sdk-renderer] defines `_RENDER_DELTA_TIME = 1/60`
and uses it in legacy, asynchronous and OVStage submissions. It neither derives
that increment from physics time nor resamples USD animation per camera capture.
This is a deliberate per-submission snapshot policy in the implementation;
the inspected sources do not explain its author's rationale for choosing 60.
Holosoma schedules these snapshots using `update_decimation`; it does not author
native `OmniSensorAPI` rates.

During 1.2 physics seconds at 50 control cycles per second:

| Owner / policy | Submissions | Native clock increment |
| --- | ---: | ---: |
| Slow snapshot at 10 Hz | 12 | 0.2 s |
| Fast or continuous snapshot at 50 Hz | 60 | 1.0 s |
| Explicit native 10 Hz sensor, stepped at 50 Hz with `0.02` | 60 | 1.2 s |

These increments were read from native results, rather than inferred from
observation counts. Uniform snapshots share one clock; persistent owners have
independent clocks. Replacing the snapshot increment with elapsed physics time
would change scheduling and capture windows. Two elapsed-time diagnostics
produced identical RGB/depth in the standalone fixture, but do not establish that
such a replacement is correct for every SDK feature.

| Feature | Established pinned behavior / limit |
| --- | --- |
| Native sensor scheduling | Rational `omni:sensor:frameRate` is Hz; `startTime` is seconds. Runtime rate changes are unsupported in the pinned schema. A step can return zero or multiple frames. |
| USD animation | Explicit `update_from_usd_time` and OVStage population sampling use seconds, converted with `timeCodesPerSecond`. Publication ordinals order committed state; they are not clocks. Step advancement alone is not a documented USD resampling guarantee. |
| Temporal reconstruction / auto exposure | Pinned schema defaults to DLSS and auto exposure off. The later [0.6 changelog][timing-changes] explicitly changes TAA/DLSS/auto-exposure advancement from wall time to simulation-step time. Exact 0.5 native internals are unavailable. |
| Native animation clock | The same 0.6 changelog makes animation permanently use accumulated step time. That later guarantee must not be imported into 0.5. |
| Exposure and shutter | Pinned brightness `exposure:time` is seconds; USD shutter open/close are frame-relative USD timecode units. They describe different controls. Post motion blur defaults off; its exposure fraction is frame-relative. |
| Motion vectors / interpolation | Endpoint writes and sensor windows are documented. Motion BVH is a separate creation-time option, off by default. The inspected pinned sources do not specify `TargetMotionSD` normalization or all skipped-frame behavior. |
| `useSimulationTime` | The token exists in 0.5 with default false, but its pinned description does not establish the later 0.6 behavior. It was not enabled in these controls. |

Camera reset also has distinct scopes. SDK `Camera.reset(env_ids)` clears
capture bookkeeping and drains pending results; its OVRTX implementation ignores
the environment subset and does not call native reset. Native
`Renderer.reset(time=T)` clears all product histories and rebases the clock.
The pinned RTX schema provides selected-tile history-reset requests, but their
effect has not been validated here. Existing reset semantics remain unchanged.

## Matched image controls

The standalone fixture uses two environment tiles at 512×512, distinct camera
orientations, moving geometry with changing depth, abrupt per-environment
material changes, 60 measured control cycles and ten initial static captures
per product. Camera zero remains the slow camera and first owner. Quality
settings are identical. The continuous reference also starts with ten captures;
its extra renders begin only in the measured phase.

Native mappings are copied immediately. Metrics, hashes and held-copy checks
run after the paced timeline, so image analysis does not slow the measured
captures. Every slow interval is 100 ms within 0.8 ms.

| Standalone control | Slow / other product frames | Slow RGB against single 10 Hz | Slow depth |
| --- | --- | --- | --- |
| Single snapshot, 10 Hz | 12 / — | Reference | Reference |
| Two products, both 10 Hz | 12 / 12 | Byte-identical, including warmup | Byte-identical |
| Persistent slow 10 Hz + fast 50 Hz | 12 / 60 | Byte-identical, including warmup | Byte-identical |
| Native 10 Hz timing schema on a 50 Hz clock | 12 / — | Surface mean absolute difference 0.082/255 | Same masks; maximum difference 4.77e-7 m |
| Continuous 50 Hz, consumed at 10 Hz | 60 / — | Surface mean absolute difference 1.437/255 | Byte-identical |

All controls passed every tile's current geometry, orientation, clock and
held-copy checks. Actual prior images at newly exposed surfaces provided stale
RGB negative controls: fresh signal averaged approximately 153/255, versus
0–0.011/255 for those previous images. No slow-rate control showed bright
pixels outside its one-pixel-expanded current depth surface. The continuous
reference had some edge trails; this does not establish a universal ordering
of quality between rates.

Sequential readback still creates fast-camera submission jitter: the mixed
fast owner's intervals ranged 8.06–42.64 ms, despite matched slow intervals.
The native-schema control's idle submissions also jittered. These are not claims
of equally spaced wall-time submissions for every fast frame; their physical
scene timestamps, capture counts and native clock windows are recorded.

Run the included OVRTX-only fixture:

```bash
for control in manual10 uniform10 mixed continuous50 native10; do
  python src/holosoma/tests/simulators/ovrtx_quality_repro.py \
    --control "$control" --resolution 512 \
    --shader-cache /path/to/existing/cache --output-dir "/tmp/quality-$control"
done
```

It imports OVRTX, NumPy and standard-library modules. `--clock-policy elapsed`
is an experimental diagnostic, not a Holosoma setting. Results include raw
warmup/test RGB/depth, native/frame windows, actual wall times, scene-write and
readback costs, color convergence, ghost metrics and actual prior-image checks.

The separate Holosoma fixture uses the existing panel, terrain and robot assets,
real `VisualMaterial.write_channels`, two cloned environments, and the same
single/uniform/mixed/continuous capture histories. An untouched SDK renderer
with only the 10 Hz product is the SDK control: it does not render that product
during intervening fast-camera captures. All cases passed every tile's current
geometry/material checks and held observations. Each run advanced exactly 440
physics steps / 2.2 seconds including warmup; capture did not advance physics.

The initial SDK-scene comparison measured panel RGB differences of 0.495/255
for mixed owners and 0.553/255 after the public-construction refactor, against
the SDK single-camera control. A single-camera adapter run itself differed
by 0.501/255. Panel depth differences were at most 1.19e-7 m and centroid errors
were below 0.45 pixels. Small last-warmup RGB differences prevent claiming
bit-identical initial reconstruction in this scene.

Three runs of each single-camera and mixed-owner control provide a limited
repeatability check. Within-control panel RGB differences were 0.504–0.556/255;
single-versus-mixed differences were 0.513–0.568/255. No measured quality
degradation accompanied the fast camera. These ranges overlap, but not every
convergence metric falls within repeat variation: environment zero's old-green
residue at later blue captures was consistently lower with mixed owners in this
small cohort. Slightly different warmup pixels, three repeats and submission
jitter prevent attributing that small difference to ownership alone.

Old green remains after a blue material change in the untouched SDK single
control and in persistent owners. Additional instrumented SDK-single and mixed
runs compared all 114 delivered RGB/depth observations, including warmup, against
CPU-owned native AOVs before SDK extraction: every tile matched exactly.
All recorded material buffers contained the exact requested color before their
capture. The native color at the blue switch already contained old green
(`[1.22,19.97,193.72]` in the single control's environment zero). Thus the tested
SDK extraction and Holosoma output handling do not create that color residue.
This instrumentation adds synchronization and is excluded from performance
comparisons.

A standalone replay of the recorded single-camera scene uses only OVRTX and
NumPy, with owned synchronous CPU material/transform writes. It reproduces the
old-green residue and its decay after the blue switch; recorded and replayed
depth match at those captures. RGB is not universally identical. The replay
preserves scene/write content and native submission history, but changes GPU
producer/buffer timing; one measured submission was 41.79 ms late. It therefore
establishes that this residue can occur without Holosoma or Isaac Lab, rather
than exact equivalence of every native scheduling or reconstruction detail.

The earlier material comparison rendered the SDK slow product during
intervening fast captures. It was not an ownership-only control. It cannot
establish that persistent owners caused ghosting or that ordinary low-rate
rendering universally explains it. These newer controls establish current
geometry/material delivery and shared native color residue for the tested
scene; exact native temporal-filter mechanisms remain uncertain.

These findings concern OVRTX. Isaac RTX has a separate completion-event and
product-layout integration; its earlier movie fix and performance measurements
do not establish equivalent temporal behavior.

## Maintenance cost and public alternatives

Child renderers now use an internal subclass of the SDK's public renderer
configuration, retaining all public rendering fields and adding only registry
identity. The ordinary SDK constructor creates the appropriate distinct native
backend. Holosoma no longer borrows and replaces a child's backend, reads
`_use_ovstage`, or calls `_read_gpu_transforms_enabled`. Material composition
uses the public `visual_material_writer` property, without calling or overriding
the SDK's private factory. These internal identities add no user options.

Three direct private SDK dependencies remain:

| Prepared field | Purpose | Upgrade risk / detecting check |
| --- | --- | --- |
| `_clone_plan` | Reuse source topology and environment clone positions | Changed/mutable plan semantics: multi-environment geometry and distinct-pose checks |
| `_exported_usd_string` | Ingest the already-prepared scene without mutating the shared stage again | Export/injection semantics or additional preparation state: startup, asset/cloning and movie tests |
| `_object_scales_by_path` | Preserve SDK scales when publishing pose-only SDP transforms | Changed scale representation: camera/body geometry tests; additional non-unit/deforming-geometry cases remain needed |

The adapter delegates thirteen concrete public SDK methods, including output
processing, pose/intrinsic updates, rendering, pending-result draining, reset
and cleanup. Children are manually composed under their parent rather than
registered as independent RenderContext consumers. This relies on initialization
order, fixed one-product ownership, early rate/movie declarations, stable render
data identity, SDK writer callable/close behavior and teardown of bindings
before native backends. A future constructor, nontrivial initialize hook,
prepared-state field or lifecycle requirement can require changes even if the
three field names survive. Holosoma-owned private names can also collide with
future superclass names.

No public reusable prepared-scene descriptor or backend-injection constructor
exists in the pinned SDK. Registering ordinary child configurations invokes
stage preparation again, whose partition authoring is not idempotent; late
registration also conflicts with finalized material consumers. Independent USD
stage views could avoid the three fields, but would add preparation/export work,
composition and resolver handling, cloning/lifecycle validation and memory.
A raw native implementation would additionally reimplement the SDK's SDP
bindings, clone handling, GPU tile extraction, materials, AOV/PPISP handling,
asynchronous buffering and teardown. It has a substantially larger maintenance
surface than the current adapter.

The current compromise is acceptable for the exact pinned stack and tested
camera paths, with SDK upgrades gated on real image/lifecycle tests. It is not
an SDK-independent abstraction. Existing checks cover four rate patterns,
all-tile RGB/depth, held buffers, SDK partial resets, actual material writes,
movies, physics advancement, OVStage with Newton/PhysX, and successful repeated
cleanup. Allocation/render failure recovery, stage rebuilds/late promotion,
PPISP, deforming geometry, all additional AOVs and large camera counts remain
coverage gaps. Happy-path tests cannot establish those untested behaviors.

[native-step]: https://github.com/NVIDIA-Omniverse/ovrtx/blob/022b00a40f3d05dd285230a39d70d80fd46f1539/include/ovrtx/ovrtx.h#L776-L825
[result-times]: https://github.com/NVIDIA-Omniverse/ovrtx/blob/022b00a40f3d05dd285230a39d70d80fd46f1539/python/ovrtx/_src/types.py#L874-L961
[camera-test]: https://github.com/NVIDIA-Omniverse/ovrtx/blob/022b00a40f3d05dd285230a39d70d80fd46f1539/tests/docs/python/test_camera_sensors.py#L170-L194
[sdk-renderer]: https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/source/isaaclab_ov/isaaclab_ov/renderers/ovrtx_renderer.py
[timing-changes]: https://github.com/NVIDIA-Omniverse/ovrtx/blob/b9ae5f0913847502ff5ed29ab45a6e9f28bec8f1/CHANGELOG.md#L66-L81
