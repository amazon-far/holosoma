# Holosoma Motion Retargeting

This repository provides tools for retargeting human motion data to humanoid robots. It supports multiple data formats (smplh, mocap, lafan) and task types including robot-only motion, object interaction, and climbing.

**Data Requirements**: The retargeting pipeline requires motion data in world joint positions. For custom data, you need to prepare world joint positions in shape `(T, J, 3)` where T is the number of frames and J is the number of joints, and modify `demo_joints` and `joints_mapping` defined in `config_types/data_type.py`.

## Installation

From the repository root, the uv setup creates a local virtual environment at `.venv/hsretargeting` and installs `src/holosoma_retargeting` in editable mode with its runtime dependencies, including the retargeting package's `numpy==2.3.5` compatibility pin.

```bash
# uv-based setup
bash scripts/setup_retargeting_via_uv.sh
source scripts/source_retargeting_uv_setup.sh

# optional: choose a Python version or force a clean reinstall
bash scripts/setup_retargeting_via_uv.sh --python 3.10
bash scripts/setup_retargeting_via_uv.sh --reinstall --dev
```

If you need the older conda-based setup, it is still available:

```bash
bash scripts/setup_retargeting.sh
source scripts/source_retargeting_setup.sh
```

After activation, run the commands below from `src/holosoma_retargeting/holosoma_retargeting`.

## Single Sequence Motion Retargeting

```bash
# Robot-only (OMOMO)
python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type robot_only --task-name sub3_largebox_003 --data_format smplh --retargeter.debug --retargeter.visualize

# Object interaction (OMOMO)
python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type object_interaction --task-name sub3_largebox_003 --data_format smplh --retargeter.debug --retargeter.visualize

# Climbing
python examples/robot_retarget.py --data_path demo_data/climb --task-type climbing --task-name mocap_climb_seq_0 --data_format mocap --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf --retargeter.debug --retargeter.visualize
```

**Note**: Add `--augmentation` to run sequences with augmentation. You must first run the original sequence before adding augmentation.

## Batch Processing for Motion Retargeting

```bash
# Robot-only (OMOMO)
python examples/parallel_robot_retarget.py --data-dir demo_data/OMOMO_new --task-type robot_only --data_format smplh --save_dir demo_results_parallel/g1/robot_only/omomo --task-config.object-name ground

# Object interaction (OMOMO)
python examples/parallel_robot_retarget.py --data-dir demo_data/OMOMO_new --task-type object_interaction --data_format smplh --save_dir demo_results_parallel/g1/object_interaction/omomo --task-config.object-name largebox

# Climbing
python examples/parallel_robot_retarget.py --data-dir demo_data/climb --task-type climbing --data_format mocap --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf --task-config.object-name multi_boxes --save_dir demo_results_parallel/g1/climbing/mocap_climb
```

**Note**: Add `--augmentation` to run original sequences and sequences with augmentation (for object interaction and climbing tasks).

## Xsens Tennis Retargeting

ActionNet-style Xsens HDF5 files can be retargeted as robot-only G1 motions with `--data_format xsens`.
The loader reads `xsens-segments/body_position_xyz_m` when available, otherwise `xsens-segments/position_cm`,
uses the Xsens stream timestamps to sample at 30 Hz, keeps `body_position_xyz_m` in its z-up convention, and converts
legacy `position_cm` streams from y-up to the retargeting z-up frame.

By default, the raw segment orientations and pelvis trajectory are first reconstructed on the in-memory
G1-proportioned Xsens avatar. The adapter uses G1-derived limb dimensions and dynamically aligns the lowest subject
and target outsole surfaces before the resulting 23 segment positions are passed to the existing G1 optimizer.
Because these targets are already G1-sized, the normal preprocessing height alignment is retained but no additional
uniform human-height scale is applied. This path does not require exporting or loading a USD model.

```bash
# Single Xsens tennis sequence
python examples/robot_retarget.py \
    --data_path demo_data/xsens_tennis \
    --task-type robot_only \
    --task-name 2026-06-14_tennis_S02_xsens_myo_data_01 \
    --data_format xsens \
    --task-config.ground-range -3 3 \
    --retargeter.foot-sticking-tolerance 0.02

# Batch all Xsens tennis HDF5 files in the directory
python examples/parallel_robot_retarget.py \
    --data-dir demo_data/xsens_tennis \
    --task-type robot_only \
    --data_format xsens \
    --task-config.ground-range -3 3 \
    --retargeter.foot-sticking-tolerance 0.02
```

When `--save-dir` is omitted, the output directory is inferred from the input dataset. The examples above save to
`demo_results/g1/robot_only/xsens_tennis` and `demo_results_parallel/g1/robot_only/xsens_tennis`, respectively.

### Tennis-racket orientation modes and attachment calibration

The physical G1 can use three right-hand orientation targets. `hand` is the backward-compatible default,
`racket` always follows `RightHandSword`, and `filtered` trials both equivalent racket targets before falling back
to the Xsens hand when the prop is detached or the target is infeasible. A 180-degree rotation about the racket's
local longitudinal `+X` axis is treated as equivalent. Filtered mode enforces joint limits, requires five feasible
frames to enter, uses 45-degree entry and 60-degree exit residuals, and keeps 5 degrees of wrist-limit margin.
The observed 45–60 degree hand/racket relative rotations are not pre-filtered.

```bash
# Always target the racket orientation.
python examples/robot_retarget.py ... \
    --retargeter.orientation.tennis-racket.mode racket

# Feasibility-filtered target with the default 45°/60° hysteresis.
python examples/robot_retarget.py ... \
    --retargeter.orientation.tennis-racket.mode filtered

# Inspect/edit the single right_rubber_hand_link → racket transform and save an override.
python examples/xsens_tennis/calibrate_racket_attachment.py \
    --save-path tennis_racket_attachment_override.json
```

Pass an override with `--retargeter.orientation.tennis-racket.attachment-path`. The default `embedded_tpose` source
applies the recording's embedded hand-to-sword T-pose correction to the global, palm-centered G1 grasp. Use
`attachment-source global` to disable the sequence-specific correction, or `attachment-source observed_window
--observed-window-s START END` to apply a mean correction over an explicit good-pose time window. The calibration
viewer reports whether the handle center lies inside the palm interior.

Raw results now store achieved racket position/orientation, per-frame tracking state, selected symmetry branch,
symmetry-aware residual, source-origin deviation, wrist-limit margin, and the effective versioned attachment.
The conversion tool resamples these arrays with linear interpolation, quaternion SLERP, and previous-hold discrete
states. The Viser player and analyzer use saved achieved poses when present and reconstruct legacy results through
the same shared attachment. Analysis summaries report symmetry-aware coverage at 30°, 45°, 60°, and 75°.

To inspect the achieved G1 racket directly against the nearest 0°/180°-equivalent Xsens target, generate a static
timeline, per-frame CSV, JSON summary, and animated target-versus-achieved orientation plot with:

```bash
python examples/xsens_tennis/analyze_g1_racket_target_error.py \
    --xsens-hdf5 <recording.hdf5> \
    --retargeted-npz <retargeting-result.npz> \
    --output-dir <analysis-directory>
```

For a retargeted excerpt, pass its first source frame with `--source-frame-start`. The animation traverses the whole
selected interval in a configurable duration and shows the global/local error histories, current tracking state,
wrist margin, symmetry branch, and target/achieved racket-frame axes.

### Analyze Xsens-to-G1 retargeting quality

`examples/xsens_tennis/analyze_xsens_g1_retargeting.py` compares the human Xsens recording, its G1-sized Xsens
targets, and the retargeted physical G1. One or more sequence names are required; there is intentionally no default
sequence. Names may include `.hdf5` or `.h5`, but must be basenames rather than paths:

```bash
python examples/xsens_tennis/analyze_xsens_g1_retargeting.py \
    --sequence-names <sequence-name>

python examples/xsens_tennis/analyze_xsens_g1_retargeting.py \
    --sequence-names <first-sequence> <second-sequence>
```

For each stem, the command resolves the recording from `demo_data/xsens_tennis/<stem>.hdf5` (or `.h5`), the exact
standard retarget from `demo_results/g1/robot_only/xsens_tennis/<stem>.npz`, and writes analysis artifacts under
`demo_results/g1/analysis/xsens_tennis/<stem>/`. A batch additionally writes `batch_summary.csv` and
`batch_summary.json` at the analysis root. Use `--data-dir`, `--retargeted-results-dir`, or `--output-root` to change
those locations. In particular, staged or experimental retargets must be selected explicitly with
`--retargeted-results-dir`; similarly named NPZ files are never selected heuristically. `--hdf5-path` and
`--qpos-npz` are single-sequence overrides.

The analysis exports per-frame and per-window CSVs, a machine-readable JSON summary, a Markdown report, and PNG/PDF
figures for stability margins, error distributions, support-polygon keyframes, and local racket trajectories. The
human Xsens motion is the error reference. Both world and full six-DoF root-relative CoM/racket errors are reported;
here the root is the Xsens pelvis or MuJoCo G1 pelvis. The G1-sized avatar CoM is an explicitly documented proxy that
attaches the physical G1 masses and calibrated inertial centroids to reduced Xsens segments. T-pose calibration is
cached in each sequence's output directory unless `--tpose-calibration-path` is supplied.

Use Viser for a synchronized diagnostic player or automated recordings. Viser modes accept exactly one sequence:

```bash
python examples/xsens_tennis/analyze_xsens_g1_retargeting.py \
    --sequence-names <sequence-name> \
    --viser-mode interactive

# Record the complete selected analysis interval. Open the printed Viser URL
# so its browser renderer can capture the video.
python examples/xsens_tennis/analyze_xsens_g1_retargeting.py \
    --sequence-names <sequence-name> \
    --viser-mode record \
    --record-path <output.mp4>

# Record each automatically selected diagnostic clip.
python examples/xsens_tennis/analyze_xsens_g1_retargeting.py \
    --sequence-names <sequence-name> \
    --viser-mode record-clips
```

The player offers overlay and side-by-side layouts, playback controls, automatic camera following, per-actor
visibility controls, actor-specific support polygons, CoM projections, racket trails, and live G1-versus-human error
values. Overlay mode is enabled by default because `--actor-spacing-m` defaults to `0.0`. It aligns the human and
G1-sized Xsens pelvis positions with the physical G1 pelvis in the XY plane at every displayed frame while preserving
each actor's original Z coordinate; it changes only the display transforms and leaves all world- and root-relative
metrics unchanged. Set a positive `--actor-spacing-m` to start in side-by-side mode. Enable camera following initially
with `--camera-follow`, or toggle `Automatically follow subjects` in the Camera folder. Full recordings can be
restricted with `--record-start-frame`, `--record-end-frame` (inclusive), and `--record-stride`; `--record-fps`
overrides the motion FPS. Recorded diagnostic clips use automatically selected, non-overlapping windows around worst
racket position/orientation errors, the worst stability-margin discrepancy, and representative labeled activities.

Use the legacy direct-human position targets for comparison or regression runs with:

```bash
python examples/robot_retarget.py \
    --data_path demo_data/xsens_tennis \
    --task-type robot_only \
    --task-name 2026-06-14_tennis_S02_xsens_myo_data_01 \
    --data_format xsens \
    --xsens-morphology.mode direct \
    --xsens-morphology.root-motion.mode preserve_world
```

The G1-proportioned mode defaults to collapsed compound-joint offsets and dynamic lowest-sole grounding. Override
these with `--xsens-morphology.preserve-joint-offsets` or `--xsens-morphology.grounding none`; use
`--xsens-morphology.g1-model-path <model.xml>` to measure proportions from a non-default G1 MuJoCo model.

### Configure floating-base translation

G1-proportioned Xsens transfer supports three root-motion modes through
`--xsens-morphology.root-motion.mode`:

- `preserve_world` retains the existing world-space trajectory.
- `scale_by_leg_length` (default) scales root XY displacement using the G1/human leg-length ratio.
- `scale_by_leg_length_contact_aware` applies the same scaling and removes horizontal drift while an outsole is
  detected near the ground and moving slowly.

The modes compose with the existing vertical `grounding` policy:

| Root-motion mode | `match_lowest_soles` | `none` |
|---|---|---|
| `preserve_world` | Match the source lowest-outsole world height. | Copy source root XYZ. |
| `scale_by_leg_length` | Scale root XY displacement and outsole height above ground. | Scale root XY displacement and root Z above ground. |
| `scale_by_leg_length_contact_aware` | Use the preceding scaled baseline plus horizontal contact correction. | Scale root XYZ plus horizontal contact correction. |

Horizontal displacement is anchored at the first source root position. Leg length is the mean neutral-pose
hip-to-lowest-outsole vertical distance across both sides. When no ground override is supplied, ground Z is estimated
robustly from the lowest source outsole samples. Root and segment orientations, timestamps, and frame count are never
changed.

```bash
# Geometric root scaling with outsole-relative vertical motion
python examples/robot_retarget.py \
    --data_path demo_data/xsens_tennis \
    --task-type robot_only \
    --task-name 2026-06-14_tennis_S02_xsens_myo_data_01 \
    --data_format xsens \
    --xsens-morphology.root-motion.mode scale_by_leg_length

# Contact-aware scaling with direct root-Z scaling about an explicit ground
python examples/robot_retarget.py \
    --data_path demo_data/xsens_tennis \
    --task-type robot_only \
    --task-name 2026-06-14_tennis_S02_xsens_myo_data_01 \
    --data_format xsens \
    --xsens-morphology.root-motion.mode scale_by_leg_length_contact_aware \
    --xsens-morphology.root-motion.ground-height-m 0.0 \
    --xsens-morphology.grounding none
```

Contact-aware thresholds can be tuned with `contact-height-tolerance-m`, `contact-speed-threshold-m-s`,
`contact-min-duration-s`, and `contact-max-gap-s` under the same `root-motion` prefix. Non-default root-motion modes
are intentionally rejected with `--xsens-morphology.mode direct` because direct mode does not build the two calibrated
morphologies needed to measure the leg-length ratio.

This default path also reconstructs the recording's T-pose with G1 proportions and uses those positions at scale
`1.0` to solve the physical G1 orientation-calibration pose. The recorded global Xsens segment orientations are
copied unchanged, then calibrated against the corresponding G1 link frames so torso, foot, and hand orientations
can be tracked during optimization. The fixed G1 head does not track independent Xsens head/neck rotation; head
position remains part of the interaction mesh. Direct mode retains the previous human-height-scaled calibration.
Supply an existing artifact with `--retargeter.orientation.calibration-path <calibration.npz>` to skip the in-memory
calibration, or use `--xsens-morphology.no-track-orientations` for the position-only optimizer.

For sparse debugging, `--motion-data-config.frame-indices 100 250 400` selects post-resampling frames and treats
them as a uniformly timed storyboard. Use the same indices in `viser_player.py` with `--xsens-target-fps 30` and
`--xsens-frame-indices 100 250 400`; increasing `--retargeter.iterations-per-frame` helps each widely separated
keyframe converge from the previous one.

The Xsens tennis files are local demo inputs; this code path works when `.hdf5`/`.h5` files are present in
`demo_data/xsens_tennis`, but the retargeting code itself does not require those large files to be tracked by Git.

### Export a calibrated OpenUSD kinematic model

Install the optional OpenUSD bindings, then export one independent subject model for one recording or every HDF5 in
a directory:

```bash
pip install -e '.[usd]'

python examples/xsens_tennis/export_xsens_usd.py \
    --hdf5-path demo_data/xsens_tennis/recording.hdf5

python examples/xsens_tennis/export_xsens_usd.py \
    --input-dir demo_data/xsens_tennis \
    --output-dir demo_data/xsens_tennis/usd_models
```

The output is `<recording>_xsens_model.usda`. It contains a floating pelvis, calibrated rigid segment transforms,
unrestricted spherical joints, local anatomical landmarks, and render-only avatar geometry. The tracked prop is
exposed canonically as `TennisRacket`; historical Xsens source identifiers are retained only as `xsens:*` metadata.
Motion remains in the HDF5 file.

The implementation is reusable by layer: `data_utils.xsens_hdf5` reads calibration data,
`xsens.kinematic_model` constructs a backend-independent tree, `kinematics` provides generic model/FK operations,
and `usd` reads, writes, validates, or replaces a kinematic subtree in an existing stage.

### Generate a source-independent Xsens model with G1 proportions

The G1 reduction reads fixed joint origins and link-local meshes from the packaged 29-DoF model. It does not load
an Xsens recording, accept a robot `qpos`, or solve against a particular G1 pose. Upper arm, forearm, hand, thigh,
shank, foot, and toe dimensions are measured independently before the canonical Xsens T-pose is assembled:

```bash
python examples/xsens_tennis/generate_g1_xsens_usd.py \
    --output-path demo_results/g1/models/g1_proportioned_xsens.usda
```

Its visuals reuse the calibrated Xsens avatar language (tapered spine shells, rear panels, palms, fingers, and
+X-facing thumbs) and scale those elements to G1-derived local envelopes. Pelvis, waist, and hip adapter visuals
cover static spans that would otherwise appear as gaps without changing any joint anchor or rigid limb length.

By default, translations between the axes of G1 compound joints are collapsed to produce idealized Xsens
spherical joints. The scalar shoulder and hip cluster extents are retained in straight adapter spans, so collapsing
the axes does not shrink the avatar. Generate the comparison variant with the full spatial offsets retained using:

```bash
python examples/xsens_tennis/generate_g1_xsens_usd.py \
    --output-path demo_results/g1/models/g1_proportioned_xsens_with_offsets.usda \
    --preserve-joint-offsets
```

The collapsed G1 wrist span is carried on the forearm side of the virtual Xsens wrist, which is co-located with
the hand-segment origin. Consequently, wrist rotation changes the hand orientation without making the fixed
inter-axis span orbit with the hand or detach it from the forearm visual.

Each command also writes a same-stem JSON report containing the raw G1 offsets, collapsed adapter offsets, applied
spatial offsets, root anchors, independently measured target lengths, generated lengths, and validation residuals.
The `g1_xsens` Viser mode preserves the recording's pelvis trajectory and global segment orientations, then
reconstructs connected body origins from the G1-Xsens model's authored joint anchors. This carries the recorded
joint motion onto the G1 proportions without runtime mesh scaling. The calibrated `xsens` mode continues to apply
the recording's global segment positions and orientations directly.

The G1-Xsens root receives a pose-dependent vertical correction derived from the calibrated subject and G1-Xsens
outsole meshes. At every sampled pose, the player aligns the lowest rendered G1 sole with the lowest rendered
subject sole after reconstructing the G1 body origins. This handles changing leg-length projections as the knees
bend while preserving the subject's measured airborne foot clearance without jump-height scaling or ground
clamping.

Use `viser_player.py --actor-modes robot` for the backward-compatible G1-only player. Actor modes compose freely;
for example, `--actor-modes xsens g1_xsens` renders both avatar proportions from the same HDF5 motion, while
`--actor-modes robot xsens` adds the physical G1. The `--actor-modes all` alias expands to `robot xsens g1_xsens`.
Any selection containing `xsens` or `g1_xsens` requires `--xsens-hdf5`; a selection containing `robot` also reads
`--qpos-npz`. When data sources are combined, the Xsens timestamps are the master clock.

The G1-proportioned actor uses `scale_by_leg_length` root motion by default. Select the legacy trajectory with
`--g1-xsens-root-motion.mode preserve_world`, or use
`--g1-xsens-root-motion.mode scale_by_leg_length_contact_aware` for the contact-corrected variant. All viewer
variants use lowest-sole grounding.

The active actors are placed on a centered lateral line by default, matching the T-pose comparison layout. Their
order is human-subject Xsens, G1-proportioned Xsens, then physical G1, with `2.0` metres of center-to-center spacing.
Thus, `--actor-modes all` uses Y offsets `-2`, `0`, and `+2` metres respectively. Change the distance with
`--actor-spacing-m 1.5`, or use `--actor-spacing-m 0` to overlay the selected actors around the recorded trajectory.

To inspect the proportions directly, render the human-subject Xsens avatar, the generated G1-proportioned Xsens
avatar, and the physical G1 side-by-side in canonical T- and N-poses:

```bash
python examples/xsens_tennis/compare_xsens_g1_poses.py \
    --hdf5-path demo_data/xsens_tennis/2026-06-14_tennis_S02_xsens_myo_data_02.hdf5
```

The script generates the human-subject avatar USD when `--calibrated-xsens-usd-path` is omitted, solves the physical
G1 T-pose from the same recording, constructs a hanging-arm N-pose from the same calibrated configuration,
ground-aligns all three models independently, and opens a frontal Viser view. Use the **Reference pose** selector in
the sidebar to switch all three actors together. Pass `--preserve-joint-offsets` to compare the offset-preserving G1
Xsens variant instead. The previous `compare_xsens_g1_tpose.py` command remains as a compatibility wrapper.

### Calibrate and visualize the retargeted G1 T-pose

The calibration follows the Xsens T-pose convention: arms and hands are horizontal, with both thumbs pointing
character-forward. Generate the one-frame G1 calibration result, then open it in Viser. The standard 29-DoF G1
model is used deliberately: its rubber-hand fingers are fixed and remain curled, but this is more faithful than
substituting a different end-effector model.

```bash
python examples/xsens_tennis/calibrate_tpose.py \
    --data-path demo_data/xsens_tennis \
    --task-name 2026-06-14_tennis_S02_xsens_myo_data_02 \
    --robot g1 \
    --variant Tpose \
    --save-path demo_results/g1/calibration/xsens_tennis/2026-06-14_tennis_S02_xsens_myo_data_02_tpose_calibration.npz

python viser_player.py \
    --robot-urdf models/g1/g1_29dof.urdf \
    --qpos-npz demo_results/g1/calibration/xsens_tennis/2026-06-14_tennis_S02_xsens_myo_data_02_tpose_calibration.npz \
    --no-assume-object-in-qpos
```

## Data Preparation

We provide `demo_data/` for fast testing. To test on more motion sequences, please follow the instructions below to download and prepare the data.

### OMOMO

Our pipeline uses the processed dataset by InterMimic. The data format differs from the original OMOMO dataset.

1. Download the processed OMOMO data from [this link](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view)
2. Extract the downloaded folder to `demo_data/OMOMO_new`

The data should contain `.pt` files.

### LAFAN

#### Download the Original LAFAN Data

1. Download [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip) by clicking "View Raw"
2. Put `lafan1.zip` in your designated data folder and uncompress it to `DATA_FOLDER_PATH/lafan`
3. The file structure should be `demo_data/lafan/*.bvh`

#### Convert the Original LAFAN Data Format for Motion Retargeting

We need some data processing files from the [LAFAN GitHub repo](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

```bash
cd holosoma_retargeting/data_utils/
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
mv ubisoft-laforge-animation-dataset/lafan1 .
python extract_global_positions.py --input_dir DATA_FOLDER_PATH/lafan --output_dir ../demo_data/lafan
```

This will convert the BVH files to `.npy` format with global joint positions.

**Note**: For LAFAN data, you need to relax the foot sticking constraint by setting `--retargeter.foot-sticking-tolerance` (default is stricter). You can adjust this tolerance number based on your data quality and retargeting results.

#### Single Sequence Retargeting on LAFAN

```bash
python examples/robot_retarget.py --data_path demo_data/lafan --task-type robot_only --task-name dance2_subject1 --data_format lafan --task-config.ground-range -10 10 --save_dir demo_results/g1/robot_only/lafan --retargeter.debug --retargeter.visualize --retargeter.foot-sticking-tolerance 0.02
```

#### Batch Processing for Motion Retargeting on LAFAN

```bash
python examples/parallel_robot_retarget.py --data-dir demo_data/lafan --task-type robot_only --data_format lafan --save_dir demo_results_parallel/g1/robot_only/lafan --task-config.object-name ground --task-config.ground-range -10 10 --retargeter.foot-sticking-tolerance 0.02
```

### AMASS SMPL-X

#### Download the Original AMASS Data

1. Follow the [AMASS](https://amass.is.tue.mpg.de/) instructions to download the original AMASS data
2. The AMASS data structure should be `/path/to/amass/dataset_name/subject_name/*.npz`

#### Download SMPL-X Models

1. Follow the [SMPL-X](https://smpl-x.is.tue.mpg.de/index.html) instructions to download SMPL-X models
2. For AMASS data, we tested on SMPL-X N (neutral) format
3. The SMPL-X models structure should be `/path/to/models/smplx/SMPLX_NEUTRAL.npz`

#### Convert the Original AMASS SMPL-X Data Format for Motion Retargeting

We provide `data_utils/prep_amass_smplx_for_rt.py` for converting AMASS SMPLX data to the format required for motion retargeting.

```bash
# Install dependencies
cd holosoma_retargeting/data_utils/
git clone https://github.com/nghorbani/human_body_prior.git
pip install tqdm dotmap PyYAML omegaconf loguru
cd human_body_prior/
python setup.py develop
cd ../

# Run data processing
python prep_amass_smplx_for_rt.py \
  --amass-root-folder /path/to/amass \
  --output-folder /path/to/output \
  --model-root-folder /path/to/models
```

This will convert the AMASS `.npz` files to `.npz` format with global joint positions and height information.

**Note**: You can optionally specify `--subdataset-folder` to process only a specific subdataset (e.g., `HumanEva`). If not specified, it will process all datasets recursively.

#### Single Sequence Retargeting on AMASS SMPL-X

```bash
python examples/robot_retarget.py --data_path demo_data/amass_smplx_processed --task-type robot_only --task-name HumanEva_S3_Jog_1_stageii --data_format smplx --task-config.ground-range -10 10 --save_dir demo_results/g1/robot_only/amass_smplx --retargeter.debug --retargeter.visualize
```

#### Batch Processing for Motion Retargeting on AMASS SMPL-X

```bash
python examples/parallel_robot_retarget.py --data-dir demo_data/amass_smplx_processed --task-type robot_only --data_format smplx --save_dir demo_results_parallel/g1/robot_only/amass_smplx --task-config.object-name ground --task-config.ground-range -10 10
```

## Check Visualizations of Saved Retargeting Results

```bash
# Visualize object-interaction results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --object_urdf models/largebox/largebox.urdf \
    --qpos_npz demo_results_parallel/g1/object_interaction/omomo/sub3_largebox_003_original.npz

# Visualize climbing results
python viser_player.py --robot_urdf models/g1/g1_29dof_spherehand.urdf \
    --object_urdf demo_data/climb/mocap_climb_seq_0/multi_boxes.urdf \
    --qpos_npz demo_results_parallel/g1/climbing/mocap_climb/mocap_climb_seq_0_original.npz

python viser_player.py --robot_urdf models/g1/g1_29dof_spherehand.urdf \
    --object_urdf demo_data/climb/mocap_climb_seq_0/multi_boxes_scaled_0.74_0.74_0.89.urdf \
    --qpos_npz demo_results_parallel/g1/climbing/mocap_climb/mocap_climb_seq_0_z_scale_1.2.npz

# Visualize robot only results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results_parallel/g1/robot_only/omomo/sub3_largebox_003_original.npz

# Visualize LAFAN robot only results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results/g1/robot_only/lafan/dance2_subject1.npz

# Record a LAFAN robot-only result from the connected Viser browser viewport
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results/g1/robot_only/lafan/dance2_subject1.npz \
    --record-video \
    --record-path videos/dance2_subject1.mp4 \
    --record-exit-after

# Visualize AMASS results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results/g1/robot_only/amass_smplx/HumanEva_S3_Jog_1_stageii.npz

# Visualize AMASS results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results_parallel/g1/robot_only/amass_smplx/HumanEva_S1_Box_1_stageii_original.npz
```

## Quantitative Evaluation

```bash
# Evaluate robot-object interaction
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/object_interaction/omomo --data_dir demo_data/OMOMO_new --data_type "robot_object"

# Evaluate climbing sequence
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/climbing/mocap_climb --data_dir demo_data/climb --data_type "robot_terrain" --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf

# Evaluate robot only (OMOMO)
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/robot_only/omomo --data_dir demo_data/OMOMO_new --data_type "robot_only"
```

## Prepare Data for Training RL Whole-Body Tracking Policy

To prepare data for training RL whole-body tracking policies, you need to follow a two-step process:

1. **First, run retargeting** to obtain `.npz` files containing the retargeted robot motion. Use the retargeting commands shown in the sections above (Single Sequence Motion Retargeting or Batch Processing for Motion Retargeting).

2. **Then, run the data conversion code** below to convert the retargeted `.npz` files into the format required for RL training. The conversion script takes the retargeted `.npz` files as input and outputs converted files with the specified frame rate and format.

**Note**: If you run this code on Mac, please use `mjpython` instead of `python`.

### Mac (using mjpython)

```bash
mjpython data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplh --object_name "ground" --once

mjpython data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz --data_format smplh --object_name "largebox" --has_dynamic_object --once
```

### Robot-Only Setting

```bash
python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplh --object_name "ground" --once

python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/lafan/dance2_subject1.npz --output_fps 50 --output_name converted_res/robot_only/dance2_subject1_mj_fps50.npz --data_format lafan --object_name "ground" --once
```

Converted files can also be inspected and recorded with the body-velocity Viser player:

```bash
python data_conversion/viser_body_vel_player.py \
    --npz_path converted_res/robot_only/dance2_subject1_mj_fps50.npz \
    --robot_urdf models/g1/g1_29dof.urdf \
    --record-video \
    --record-path videos/dance2_subject1_body_vel.mp4 \
    --record-exit-after
```

For qpos `.npz` samples, an entire directory can be recorded through one Viser browser session. The script starts one Viser server, waits for a browser client once, then records one MP4 per sample using the original sample stem as the video filename:

```bash
python data_conversion/batch_record_viser_player.py \
    --npz-dir /home/maxi/src/motion-diffusion-model/save/my_latent_tennis_g1_uncond_DiP/samples_600000/viser_npz
```

By default, videos are written next to the `.npz` files, for example `sample00_rep00_unconstrained.npz` becomes `sample00_rep00_unconstrained.mp4`. Existing MP4 files are skipped unless `--overwrite` is passed. Useful options include:

```bash
python data_conversion/batch_record_viser_player.py \
    --npz-dir /path/to/viser_npz \
    --output-dir videos \
    --record-width 1920 \
    --record-height 1080 \
    --record-start-delay 10 \
    --overwrite
```

Use `--dry-run` to preview the selected samples and output filenames without starting Viser.

### Robot-Object Setting

```bash
python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz --data_format smplh --object_name "largebox" --has_dynamic_object --once
```

### OmniRetarget Data

For OmniRetarget data downloaded from HuggingFace, please add `--use_omniretarget_data` for data conversion.

```bash
python data_conversion/convert_data_format_mj.py --input_file OmniRetarget/robot-object/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj_omnirt.npz --data_format smplh --object_name "largebox" --has_dynamic_object --use_omniretarget_data --once
```

## Custom Human Motion Data Format
Please see the instructions for custom human motion data formats: [ADD_MOTION_FORMAT_README.md](holosoma_retargeting/ADD_MOTION_FORMAT_README.md)

## Custom Robot Type
Please see the instructions for retargeting custom robot types: [ADD_ROBOT_TYPE_README.md](holosoma_retargeting/ADD_ROBOT_TYPE_README.md)
