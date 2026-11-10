# OVRTX multiview capture reproduction

`src/holosoma/tests/simulators/ovrtx_cadence_repro.py` imports only OVRTX, NumPy
and the Python standard library. It generates a small USD scene with a red cube,
three camera groups, and RGB/depth render variables. Alternate groups rotate
90 degrees about the optical axis, making camera substitutions observable.
There are no robot assets, physics, Holosoma, Isaac Lab, Kit, Warp kernels,
GPU attribute bindings, or SDK image extraction. Images are mapped to the CPU
through OVRTX's API and compared against their projected cube positions.

On far044's RTX 5090 with `ovrtx==0.5.1.385782`, run from an environment containing
that wheel and NumPy:

```bash
python src/holosoma/tests/simulators/ovrtx_cadence_repro.py \
  --mode periodic --instances 2 --output-dir /tmp/ovrtx-periodic
```

The three render products contain two camera views each. Their native
`OmniSensorAPI` rates are 50, 10 and 25 Hz, aligned to the initial capture by
`frameRate` and `startTime`. Forty control cycles produce the correct 40/8/20
frame counts, but some images have another group's orientation. The process
exits with status 1 and writes the generated `scene.usda`, native log and
per-frame measurements to `result.json`.

Controls on the same host:

| Mode | Views per product | Result |
| --- | --- | --- |
| `auto` | 2 | Pass: unrestricted stable batch, 40/40/40 correct images |
| `periodic` | 1 | Pass: mixed native rates, 40/8/20 correct images |
| `periodic` | 2 | Fail: correct frame counts, incorrect camera orientations |
| `subset` | 2 | Fail: selecting only due products with native rates is insufficient |
| `manual` | 2 | Fail: selecting due products without the timing schema is insufficient |
| `serial` | 2 | Pass: individual due-product submissions and immediate readback, 40/8/20 |

Use `--decimations 2,3` to check two camera groups; the two-view native-rate case
also fails while returning the expected 20/14 frames. Disabling SPG with
`--disable-spg` does not fix the two-view native-rate case. `--shader-cache PATH`
optionally reuses an existing shader cache through OVRTX's settings extension;
it does not import an SDK or change the cameras.

These comparisons isolate the failure to the pinned OVRTX package's changing
multiview batch path. They do not identify which native implementation component
is responsible. Holosoma therefore uses individual submissions for mixed rates
and movie coexistence, with immediate SDK readback because subsequent native
steps invalidate prior output handles. This skips idle captures but has
substantial re-entry cost; it should not be described as an OVRTX throughput
improvement. Uniform-rate camera batches retain the existing efficient path.
