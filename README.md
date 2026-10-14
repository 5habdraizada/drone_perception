# Drone perception without GNSS

Onboard perception for a drone that has to operate without GPS: it measures the
terrain under it with a downward stereo camera pair, learns what the scene
normally looks like, flags anything unusual, builds a map of all of it in the
world frame, and plans where to fly next.

Beyond what the cameras see directly, it keeps a **belief map of what can't be
seen**: objects hidden under tree canopy, and signs of voids underground
(ground depressions). The uncertainty in that
map is the **reward signal**: the planner, or a reinforcement-learning agent,
flies where it expects to learn the most about hidden things.

Everything runs on the CPU in fp32 (NumPy, OpenCV, ONNX Runtime), with no
model quantisation and no GPU required. The target computer is a Qualcomm
RB3 Gen 2.

**Status:** every module works and is tested against synthetic data with exact
ground truth. It has **not yet run on real images**; see
[Validation status](#13-validation-status).

---

## Contents

1. [What it does](#1-what-it-does)
2. [Architecture](#2-architecture)
3. [Repository layout](#3-repository-layout)
4. [Setup](#4-setup)
5. [Quick start](#5-quick-start)
6. [Testing on your own data](#6-testing-on-your-own-data)
7. [Calibrating the cameras](#7-calibrating-the-cameras)
8. [Modules](#8-modules)
9. [Interface with the control team](#9-interface-with-the-control-team)
10. [Conventions](#10-conventions)
11. [Simulation and tests](#11-simulation-and-tests)
12. [Performance](#12-performance)
13. [Validation status](#13-validation-status)
14. [Design decisions](#14-design-decisions)
15. [Limitations and next steps](#15-limitations-and-next-steps)

---

## 1. What it does

| Question | Answer from | Output |
|---|---|---|
| How high am I, and how am I tilted relative to the ground? | `terrain.py` | altitude, tilt, ground plane |
| What is below me and how tall is it? | `terrain.py` | height, slope and roughness per cell |
| What kinds of surface are there? | `scene.py` | regions with ids that stay stable over time |
| Is anything unusual? | `scene.py` | novelty z-score per cell |
| Where is everything in the world? | `mapping.py` | map of height, elevation, regions and anomalies |
| What might be hidden under trees, or underground? | `belief.py` | probability maps, confident finds |
| How much would flying there reveal? | `belief.py` | expected information gain (bits): the reward |
| Where should I fly next? | `decision.py` | one waypoint command per cycle |

**Ownership:**
- This code covers perception, mapping and mission planning.
- The **control team** provides the drone's pose (position and orientation) and flies the waypoints.
- A classmate's keypoint detector feeds the control team's pose estimation.

**Hardware assumed:**
- Two downward-facing cameras: camera 1 (left) has a long lens (fx ≈ 1719 px), camera 2 (right) a short lens (fx ≈ 1175 px). Both 1920 × 1080, about 10 cm apart.
- An RB3 Gen 2 (QCS6490, 8-core Kryo 670 CPU).

---

## 2. Architecture

```
left camera ─┐
             ├─► Rectifier ──► TerrainEstimator ──────────────────┐  ground plane, altitude, tilt,
right camera ┘  (calibration)   StereoBM + RANSAC plane           │  height / slope / roughness per cell
                     │                                            ▼
                     └──────► Backbone (left image) ──► SceneModel: regions + novelty + contrast
                              ResNet-18, ONNX, CPU                │
                                                                  ▼
control team: pose ─────────────────────────────────────► GridMap (world frame)
                                                                  │  height, elevation, regions,
                                                                  │  anomaly hits, coverage
                                                                  │
                           BeliefMap ◄────────────────────────────┤  frames + map
                           P(hidden object), P(void),             │
                           entropy, expected gain ───────────────►│
                                                                  ▼
battery, pose uncertainty ──────────────────────────────► MissionPlanner ──► waypoint (x, y, altitude)
                                                                              to the control team
```

`perception/pipeline.py` chains everything up to the map for one stereo pair.
The planner runs on top of the map.

**Rate:** the whole chain takes about 80–90 ms per frame on a laptop CPU, and
the target is 3–5 Hz on the RB3. The fast flight-control loop belongs to the
control team; perception doesn't need camera frame rate because the ground
changes slowly.

**Grid:** a rectified frame is 640 × 352 px. Terrain, backbone and scene all
work on the same 20 × 11 grid of 32 px cells (about 0.33 m on the ground from
6 m up), so every cell has both appearance and geometry.

---

## 3. Repository layout

```
perception/            the onboard library
  calibration.py       stereo calibration, rectification, distortion-model checks
  terrain.py           stereo depth -> ground plane, altitude, heights, slope, roughness
  backbone.py          appearance features from an ONNX model
  scene.py             regions and novelty, learned online
  mapping.py           world-frame grid map, pose helpers
  belief.py            hidden objects and signs of voids: beliefs, entropy, expected gain (the reward)
  decision.py          mission planner
  pipeline.py          one stereo pair through rectify -> terrain -> scene -> map
tools/                 command-line tools for real data
  calibrate_stereo.py  capture checkerboard pairs, calibrate the rig
  run_data.py          run the pipeline on your own recordings
  viewer.py, .html     browser dashboard showing every output live
  prepare_kagaru.py    convert the QUT Kagaru airborne stereo dataset
  fetch_pdok.py        download a real Dutch area (orthophoto + LiDAR) for the simulator
  make_sample_data.py  write a synthetic recording in run_data's format
  export_backbone.py   export pretrained models to ONNX (development machine)
  bench_backbone.py    compare models: speed, separability, feature visualisation
  viz.py               drawing helpers
demos/                 synthetic demos with ground truth
  terrain.py  scene.py  map.py  mission.py  real_area.py  search_benchmark.py
sim/                   simulator used by tests and demos
  synthetic_stereo.py  ray-traced stereo images of a heightfield world
  synthetic_board.py   checkerboard images through real lens models
  oracle_perception.py exact perception frames, for fast mission tests
  mission.py           closed-loop mission simulation
  search_env.py        RL environment: search for hidden targets, reward = information
  search_policies.py   random, lawnmower and information-greedy policies
  scenes.py            shared scenes and the simulated rig
  real_world.py        real places: orthophoto draped over a LiDAR surface model
tests/                 pytest suite, one file per module
config/intrinsics.json the cameras' per-camera calibration as provided
models/                exported backbones (.onnx + .json)
```

---

## 4. Setup

**Python 3.10+.** On the development machine:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt     # Windows
# .venv/bin/python -m pip install -r requirements-dev.txt       # Linux / macOS
python -m tools.export_backbone resnet18                         # downloads weights, writes models/
```

On the RB3, only the runtime packages are needed (numpy, opencv-python,
onnxruntime). Copy `perception/`, `tools/` and the `models/*.onnx` and
`models/*.json` you use.

```bash
pip install -r requirements.txt
```

`requirements-dev.txt` adds pytest, onnx, torch and timm. Those are only for
tests and exporting models; the drone never needs PyTorch.

---

## 5. Quick start

```bash
pytest                       # full suite, ~3 min
pytest -m "not slow"         # skip tests that render images, ~35 s
```

Demos run on synthetic scenes with exact ground truth, and save images to `out/`:

```bash
python -m demos.terrain      # terrain on one stereo pair vs ground truth
python -m demos.scene        # regions and novelty along a flight
python -m demos.map          # full chain into the world map, checked against truth
python -m demos.mission      # a complete autonomous mission (seconds; uses oracle perception)
python -m demos.mission --search     # the same, searching for an object hidden under trees
python -m demos.search_benchmark     # is the reward meaningful? policies compared on random scenes
```

To watch everything the pipeline produces, frame by frame:

```bash
python -m tools.make_sample_data --out out/sample_data --frames 24
python -m tools.viewer out/sample_data --model models/resnet18.onnx    # then open http://127.0.0.1:8000
```

---

## 6. Testing on your own data

`tools/run_data.py` runs the full pipeline on recorded stereo data and writes
per-frame measurements, images, and (if poses are given) the map and any
confirmed anomalies.

### 6.1 Try it on sample data first

```bash
python -m tools.make_sample_data --out out/sample_data              # rectified images + rig.json + poses.csv
python -m tools.run_data out/sample_data --model models/resnet18.onnx --out out/run_sample

python -m tools.make_sample_data --raw --out out/sample_raw         # raw distorted images + stereo.json
python -m tools.run_data out/sample_raw --calib out/sample_raw/stereo.json --out out/run_raw
```

Those folders show the exact formats described below.

### 6.2 Data layout

Put your recording in one folder. **Stereo images**, in any one of these layouts:

| Layout | Example | Notes |
|---|---|---|
| Two image folders | `left/000001.png`, `right/000001.png` | Matched by file name; by sorted order if names differ |
| Two videos | `left.mp4`, `right.mp4` (`.avi .mov .mkv` also) | Read frame by frame in lockstep |
| Capture-tool pairs | `pair_000_left.png`, `pair_000_right.png` | The layout `calibrate_stereo capture` writes |

Image formats: png, jpg, bmp, tif. `--left`/`--right` can point anywhere,
overriding detection.

**Camera geometry**, one of:
- **Raw images** (straight from the cameras): pass `--calib config/stereo.json`, produced by [calibration](#7-calibrating-the-cameras). Rectification happens inside the pipeline.
- **Already-rectified images** (rows aligned, same intrinsics): put a `rig.json` in the folder, or pass `--rig`:
  ```json
  {"fx": 577.0, "fy": 577.0, "cx": 320.0, "cy": 176.0, "baseline": 0.10}
  ```
  Pixel units of the rectified images; baseline in metres.

**Poses (optional, needed for the map):** `poses.csv` in the folder, or `--poses`:

```
frame,x,y,z,qw,qx,qy,qz
0,-8.0,0.0,6.0,0.0,0.99991,-0.01309,0.0
```

- `frame` is the 0-based index of the stereo pair (sorted order).
- The pose is the **left camera's** pose in the world: position in metres, plus a unit quaternion rotating camera coordinates into the world.
- Camera axes are OpenCV's (x right, y down, z forward); world z points up.
- With `--calib`, give the raw left camera's pose; the conversion into the rectified frame is automatic.
- Frames without a pose are processed but not mapped.

### 6.3 Running

```bash
python -m tools.run_data DATA --calib config/stereo.json --model models/resnet18.onnx --out out/run1
```

| Option | Meaning |
|---|---|
| `--model PATH` | Backbone for regions and novelty. Without it: terrain and the height map only. |
| `--fps N` | Frame rate, for timestamps (default 10) |
| `--start N`, `--step N`, `--max-frames N` | Choose which frames to process |
| `--save-every N` | Save a picture every N processed frames (default 10) |
| `--width N` | Rectified width with `--calib` (default 640; height follows the cameras' overlap) |
| `--height-range LO HI` | Colour range for height images, in metres |
| `--min-depth`, `--max-depth` | Expected depth range in metres (defaults 1.5 and 40). Raise both for high flight, e.g. 30 and 200 at 50–100 m |
| `--map-resolution` | Map cell size in metres (default 0.25; use about 1 m when flying high) |

### 6.4 Output

| File | Contents |
|---|---|
| `frames.csv` | One row per frame: altitude, tilt, depth coverage, ground-plane inlier fraction, number of regions, maximum novelty, whether it was mapped, and timing for each stage |
| `frames/frame_*.png` | Rectified image, height, regions, novelty |
| `map.png` | Map layers: height, regions, anomaly hits, times seen (needs poses) |
| `anomalies.json` | Confirmed anomalies: world position, area, peak novelty, hits, height |
| `summary.txt` | The summary printed at the end |

### 6.5 What to look for

- **`altitude_m`:** should match the drone's barometer or rangefinder. Jumps between frames point to a wrong ground plane (a large roof, or bad depth).
- **`depth_coverage`:** fraction of pixels with depth. About 80% is normal; the left strip of the image never has depth. Much lower suggests calibration errors, unsynchronised cameras, motion blur, or textureless ground.
- **`ground_inliers`:** fraction of depth pixels on the ground plane. It falls when structures fill the view, which is expected.
- **Height images:** flat ground should be uniformly near 0, and objects should have sharp, correct heights. A tilted or curved "ground" means calibration is off.
- **Regions:** colours should follow real surfaces and stay the same from frame to frame.
- **Novelty:** the first 5 frames are warm-up (no values). After that, z > 3 is unusual. Expect hits on genuinely odd things, and early on also on surfaces seen at new angles.
- **Timing columns:** run on the RB3 for the numbers that matter.

### 6.6 The viewer

```bash
python -m tools.viewer DATA --calib config/stereo.json --model models/resnet18.onnx
```

It takes the same data and options as `run_data`, then serves a dashboard at http://127.0.0.1:8000:

- **Image tiles:** both rectified cameras, disparity, height, slope, roughness, regions, novelty and contrast, each with its colour scale. Click a tile to enlarge it.
- **The current frame's numbers:** altitude, tilt, depth coverage, ground-plane fit, feature noise, and timing for each stage.
- **Regions table:** colour, size, height and share of each region.
- **Confirmed anomalies.**
- **Charts over time:** altitude, coverage, novelty (with the z = 3 line) and frame time.
- **The world map** when poses are given: height, regions, anomaly hits or coverage, with the flown path and anomalies marked.

**Controls:** Play/Pause (space), step (→), back through processed frames (←), the scrub bar, and Reset.

**Options:**
- `--play-fps` limits playback speed.
- `--keep` sets how many frames stay available for stepping back (default 300).
- `--host 0.0.0.0` lets you open it from another machine, e.g. the drone computer viewed from a laptop. It has no authentication, so only do that on a network you trust.
- No extra packages are needed; it uses Python's built-in web server.

---

## 7. Calibrating the cameras

Stereo depth needs both cameras' intrinsics **and** the exact position and
rotation between them. The second can only come from a joint stereo
calibration. Each pixel row in the left image must line up with the same row
in the right image to within a fraction of a pixel; a 1° mounting error moves
rows by about 10 px.

### 7.1 Procedure

1. **Print a checkerboard** with 10 × 7 squares (9 × 6 inner corners). Stick it to something rigid and flat. Measure one square in metres.
2. **Capture** on the drone computer:
   ```bash
   python -m tools.calibrate_stereo capture --left 0 --right 1 --out calib_captures [--preview]
   ```
   - `--left`/`--right` take a camera index, a device path, or a GStreamer pipeline ending in `appsink` (the usual way to open cameras on Qualcomm Linux).
   - Pairs are saved automatically when both cameras see the board and it has moved since the last save.
   - Move the board slowly: near and far, tilted, and **into every corner of the left camera's view**.
   - Then put the board in the **corners of the wide right camera's view**, where the left camera can't see it. Those single-camera frames are saved too, and they are the only way the wide lens's edges get measured.
3. **Calibrate:**
   ```bash
   python -m tools.calibrate_stereo calibrate --captures calib_captures --square 0.025 \
       --intrinsics config/intrinsics.json --out config/stereo.json
   ```
4. **Check the report and `config/stereo_rectified_check.png`:**

   | Line | Good |
   |---|---|
   | reprojection RMS | < 0.5 px |
   | baseline | matches a ruler measurement within a few mm |
   | row alignment | mean < 0.2 px |
   | warnings | none about distortion folding back |

   In the check image, matching features should sit on the same green line.

### 7.2 About the provided intrinsics

`config/intrinsics.json` holds the per-camera calibration as provided. The tool
checks every lens model before using it:

- **Left (long) lens: unusable.** Its distortion (k2 = 7.0, k3 = −26.7) is near zero in the centre, then folds back on itself 50% of the way to the corners. About 26% of the image can't be undistorted at all. Together with large tangential terms and an off-centre principal point, that's an overfit calibration, typically from a board that never reached the image edges.
- **Right (short) lens:** fine, apart from the extreme corners (3%), which stereo doesn't use.

With `--refine auto` (the default), the tool re-estimates any camera whose model
looks unreliable, starting from the provided matrix with the distortion reset.
On synthetic captures that reproduce exactly this situation, it recovers the
baseline to within 0.6 mm.

### 7.3 Mismatched lenses

Rectification maps both cameras onto one shared virtual camera. With these
lenses, the shared view is bounded by the narrow left lens: about 58° × 34°,
output at **640 × 352**, focal length ≈ 577 px.

Each raw image is first shrunk with area averaging to roughly that sampling, so
both rectified images are equally sharp and the downscale doesn't alias.

**Any mounting works.** The rectified x axis is built along the baseline, so the
cameras can be side by side, one behind the other, or rotated relative to each
other. If camera 2 sits on the left of camera 1, the outputs are swapped
automatically.

### 7.4 Correcting calibration drift from footage

Vibration and knocks shift the cameras' relative rotation over time.
`refine_extrinsics(calib, pairs)` re-estimates it from feature matches in ordinary
footage, keeping the intrinsics and the baseline length. On the Kagaru flight it
brought row alignment to 0.7 px at full resolution.

**It cannot recover vergence** (a small inward rotation of one camera). On a flat,
distant scene, vergence is almost indistinguishable from a change in depth, so it
biases every disparity by a constant. A checkerboard calibration fixes vergence;
self-calibration from flat ground does not. See [section 13](#13-validation-status).

---

## 8. Modules

### 8.1 `terrain.py`: stereo → terrain

**Steps:** StereoBM disparity → RANSAC ground plane → 3D points → height above
ground for every pixel → per-cell median height, slope and roughness.

- **Ground plane in disparity space.** A 3D plane is a linear function of pixel coordinates in disparity, `d = a·x + b·y + c`, and stereo noise is roughly constant in pixels. So one inlier threshold (0.5 px) works at any altitude.
- **Depth edges excluded.** Block matching smears a sharp depth step into a ramp. Pixels near steps are left out of the plane fit; without this, a roof edge tilts the ground estimate.
- **Plane prior.** Pass the previous frame's plane (`plane_prior`). Only candidate planes near it are accepted, and the refit is gently pulled towards it. That stops a roof filling the view from being taken for the ground, and holds the tilt steady when only a strip of ground is visible. A prior predicted from the control team's IMU attitude would be even better.
- **Slope and roughness** come from a PCA plane fit pooled over 3 × 3 cells. Disparity comes in 1/16 px steps, which tilts a fit over a single cell.
- **Outputs (`Terrain`):** disparity, 3D points, height, plane `(n, d)`, altitude, tilt, `cell_height`, `slope_deg`, `roughness`, `valid_fraction`, `inlier_fraction`. Invalid values are NaN.
- **Configuration:** `TerrainConfig.for_width(640)` scales pixel-sized settings from their 320 px defaults. Main knobs: `matcher`, `min_depth`, `inlier_px`, `prior_*`.

**Measured at 640 × 352, synthetic scenes:**

| Altitude | Altitude error | Tilt error | Height error (median / 90%) |
|---|---|---|---|
| 5 m | 1 cm | 0.04° | 1 / 3 cm |
| 10 m | 6 cm | 0.4° | 3 / 7 cm |

**Depth noise** grows with altitude²: σZ ≈ Z² × 0.2 px / (f × B). With this rig,
about ±9 cm per pixel at 5 m and ±34 cm at 10 m. Plane fits and cell medians
are far more precise than single pixels.

### 8.2 `calibration.py`: calibration and rectification

- **`find_board`:** sub-pixel checkerboard corners.
- **`calibrate_camera`:** calibrates one camera.
- **`stereo_calibrate`:** the relative pose of the two cameras, with intrinsics fixed or refined first.
- **`per_pair_errors`:** finds bad captures.
- **`check_model` / `model_is_usable` / `valid_radius`:** detect overfit distortion. A model is unusable if more than 10% of the image lies beyond the radius where the distortion polynomial folds back.
- **`Rectifier`:** builds the shared pinhole camera over the valid overlap. Provides `rectify()`, `rectify_points()`, `rig`, `R1` and `swapped`.
- **`StereoCalibration`:** saves to and loads from JSON.

### 8.3 `backbone.py`: appearance features

A pretrained network is cut at the layer whose output grid matches the terrain
cells: stride 16 on a 320 × 176 image gives 20 × 11.

- One cell of mirrored border is added around the image and cropped off the output. Otherwise the CNN's zero padding makes edge cells look different from everything else.
- Feature vectors are scaled to unit length, so comparisons are cosine similarities.
- `FeatureGrid.resample()` maps the features onto any other cell grid.

**Laptop CPU results** (`tools/bench_backbone.py`; the feature map shows how
well a model groups similar surfaces):

| Model | ms per frame | Size | Feature map on the synthetic scene |
|---|---|---|---|
| MobileNetV3-Small | 2.8 | 0.8 MB | noise |
| MobileNetV3-Large | 4.8 | 3.2 MB | faint structure |
| EfficientNet-Lite0 | 6.6 | 2.8 MB | faint structure |
| **ResNet-18 (default)** | **21** | 11 MB | clearest regions |
| DINOv2-S | 87 | 87 MB | clean regions; usually strongest on real photos |

The synthetic textures are noise, so **re-run the benchmark on real frames and on
the RB3** before committing:

```bash
python -m tools.bench_backbone --images FOLDER
```

**Backends:** `"cpu"` (default, fp32). `"gpu"` and `"npu"` go through ONNX
Runtime's QNN provider; the QCS6490's NPU needs an int8-quantised model.

### 8.4 `scene.py`: regions and novelty

Each cell's feature combines the backbone vector with tanh-squashed height,
slope and roughness (`geo_weight` = 1, so a step of 1 m or more is a different
surface on its own). Everything is learned online; there are no labels.

- **Feature noise:** each frame, the median distance of a cell from its 8 neighbours' average. Most cells are inside a uniform patch, so this measures normal within-surface variation. All thresholds are relative to it, so nothing is tuned to one backbone or terrain.
- **Regions (online DP-means):**
  - A new region starts when at least 3 similar cells are all more than 2 × noise away from every existing region. A single odd cell isn't enough, because noise produces scattered outliers while a genuinely new surface produces a group.
  - Regions whose centres come within 1 × noise of each other merge.
  - Region ids stay the same across frames.
  - Isolated single cells are relabelled to their neighbours' majority; anything 2 cells or larger survives.
- **Novelty (memory bank, as in PatchCore):**
  - The model keeps up to 2000 previously seen cell features. A cell's raw score is its mean distance to its 5 nearest entries, converted to a z-score against the running history of scores.
  - About 0 means typical; above 3 means unusual.
  - Each frame, 5% of the cells are added to the bank. Something seen repeatedly becomes familiar within about 12 frames, and things not seen for a long time are slowly forgotten.
  - There's no novelty for the first 5 frames (warm-up).
- **Contrast:** a cell's distance from its neighbours in the current frame. It needs no history, so it flags small odd things immediately (and is also high along boundaries).
- **Cells without depth** get a region from appearance alone, and NaN novelty.

**Output (`SceneFrame`):** `regions`, `novelty`, `contrast`, fused `features`,
`geometry_valid`, `noise`, and `region_info` (number of cells, weight and median
height per region).

### 8.5 `mapping.py`: the world map

`GridMap.update(frame, pose)` projects each cell (2 × 2 sub-points per cell)
along its camera ray to its measured height above the ground plane, then
transforms it into the world. The map uses 0.25 m cells and grows as the drone
flies past its edge.

| Layer | How it's built |
|---|---|
| `height` | Weighted average of height above ground. Nearer observations and more certain poses count more. |
| `elevation` | World z of the observed surface: a terrain map, for map-matching navigation later |
| `regions` | Region seen most often at each cell |
| `anomaly_hits` | Frames in which this cell, or a neighbour, scored novelty ≥ 3 |
| `novelty_peak` | Highest novelty ever seen there |
| `seen`, `last_seen` | How often and when each cell was observed |

**Anomalies:** a spot is confirmed after hits in **2 separate frames**, so
one-frame noise never qualifies. Later ordinary sightings don't count against
it: the scene model gets used to whatever it keeps seeing, and that shouldn't
erase what was unusual the first time.

**Queries:**
- `anomalies()`: blobs with position, area, peak novelty, hits, and height (90th percentile).
- `frontier()`: unseen cells bordering seen ones.
- `clearance(x, y, r)`: the tallest mapped surface within radius `r`.
- `ensure_area()`, `area_mask()`, `render()`.

**Pose helpers:** `camera_from_body`, `rectified_camera`, `pose_from_quaternion`,
`quaternion_from_rotation`, and the `PoseSource` protocol.

**On a synthetic 18 m flight:** heights within 0.8 cm of the truth (median),
8 cm at the 90th percentile. The striped test object was confirmed within
0.15 m of its true position, at 0.57 m tall (true value 0.5 m). Steep roof walls
gave three weaker false positives (see [limitations](#15-limitations-and-next-steps)).

### 8.6 `decision.py`: mission planner

`MissionPlanner.decide(grid, x, y, agl, battery, pose_sigma, t)` returns a
`Command`: mode, target x and y, altitude above ground, hold time, reason, and
the target anomaly if any. The first rule that applies wins:

1. **return → land:**
   - battery ≤ distance home × `battery_per_m` + `battery_reserve` (15%), or
   - pose uncertainty above `max_pose_sigma` (3 m), or
   - the search area is covered.

   It lands on reaching home.
2. **investigate:** a confirmed anomaly not yet visited. The best is chosen by peak novelty × hits ÷ distance. The drone descends to `investigate_agl` (4 m), holds for 5 s, then records it in `planner.report`.
3. **verify:** a spot that looked unusual only once. The drone flies back over it so the map can confirm or dismiss it.
4. **search** (with a belief map attached): the waypoint and altitude with the highest
   expected information gain minus a cost per metre (`search_energy`), over a lattice of
   candidates in `area`; altitudes from `search_altitudes` (default cruise and investigate),
   raised to what's safe there. Confident hidden finds are logged in `planner.hidden`.
5. **explore:** the most useful frontier inside `area`, weighing size against distance.
   Without a belief map, or when no search target is worth `min_search_gain` bits.

The current goal is kept until it's reached. Only a return, or a newly
confirmed anomaly while exploring, interrupts it.

**Altitude:** every command's altitude clears the tallest mapped surface within
1.5 m of the straight path, plus `min_clearance` (2 m). Unmapped cells are
assumed to be `unknown_height` (3 m) tall, because the cameras can't see ahead.

**Closed-loop mission tests (oracle perception):**
- Covers more than 95% of the area, then lands at home.
- Finds and investigates the object.
- Ignores one-frame false alarms.
- Comes home on low battery or when the position estimate drifts.
- Clears mapped obstacles.
- Flies at most 3× an ideal back-and-forth sweep, with no zig-zagging. A typical mission: 49 s, 95 m.

### 8.7 `belief.py`: hidden targets and the reward

The camera map records what is visible. `BeliefMap` keeps, on the same grid, the
probability of two things that are **not** directly visible:

| Hypothesis | Raised by | Lowered by |
|---|---|---|
| **Hidden object** at ground level (e.g. under trees) | a novelty hit, placed where the ray meets the ground, not on the canopy top | a clear view of the ground that finds nothing |
| **Buried void** (indirect) | ground depressions, clearly seen | nothing yet: no sensor sees underground |

**How views count:**
- Evidence comes from *improving on the best look so far*. Seeing the same spot again
  from the same height adds nothing; a closer look does. This keeps repeated frames from
  being counted as independent evidence.
- View quality falls with range. **Canopy** (a raised, rough surface) only lets part of
  the ground through its gaps, more when close: a high pass over trees barely clears them,
  a low pass does. **Roofs** (raised and smooth) are not searchable at all.
- Occlusion: the expected gain checks line of sight, so ground hidden behind a tree's edge
  is only worth something from a viewpoint that can actually see it.

**The reward signal:**
- `entropy(area)`: remaining uncertainty in bits, for objects, voids and in total.
- `expected_gain(x, y, agl)`: about how many bits a view from there would remove. The planner maximises this; an RL agent is
  rewarded with the entropy actually removed.
- `detections(kind)`: confident finds (`"object"` or `"subsurface"`).

**Usage:** call `belief.observe(frame, pose)` next to `grid.update(frame, pose)`.

**Measured (tests):** under trees, a low pass finds an object that high passes miss;
ground next to a tree's edge is only scored as seen after a pass from the side.

### 8.8 `pipeline.py`

- `Pipeline(rig, size, rectifier=None, backbone_path=None, ...)` or `Pipeline.from_calibration(path)`.
- `process(t, left, right, t_world_left=None, pose_cov=None)` returns a `FrameResult`: the rectified image, terrain, scene, whether the frame was mapped, and timing for each stage.
- The plane prior is carried between frames automatically.
- `pipe.map` is the `GridMap`.

---

## 9. Interface with the control team

**Pose (what perception needs):**

```python
class PoseSource(Protocol):
    def get_pose(self, t: float) -> tuple[np.ndarray, np.ndarray | None]:
        """4 × 4 world-from-body transform at time t, 6 × 6 covariance (x, y, z, roll, pitch, yaw) or None."""
```

- World frame: z up, origin at take-off.
- Also needed: the camera mounting (body-from-camera transform). `camera_from_body()` combines them, and `Pipeline` applies the rectification rotation itself.
- Uncertain poses automatically count less in the map, and too much uncertainty sends the drone home.
- **Agree on a shared clock early.** Timestamp mismatches are a classic integration bug.

**Commands (what perception provides):** `Command(mode, x, y, agl, hold, reason, target)`:

- World x and y in metres, plus target altitude **above ground**. Perception also measures altitude above ground every frame (`Terrain.altitude`).
- These are position targets: they assume the flight controller's position-hold or waypoint mode.

**Also available to the control team:**
- Terrain altitude and tilt every frame, as a cross-check for the barometer and IMU.
- The map's `elevation` layer, a terrain map for map-matching navigation.

---

## 10. Conventions

| Item | Convention |
|---|---|
| Camera frame | OpenCV: x right, y down, z forward (optical axis) |
| World frame | z up; map rows grow with world y; `render()` shows north (+y) up |
| Poses | 4 × 4 world-from-camera (or world-from-body) homogeneous transforms |
| Quaternions | (w, x, y, z), rotating camera coordinates into the world |
| Ground plane | (nx, ny, nz, d) in camera coordinates with n·X + d = 0 and d > 0, so d is the altitude |
| Stereo calibration | X_right = R · X_left + T, with T in metres |
| Units | metres, degrees in configs and outputs, pixels for image quantities |
| Missing values | NaN |

---

## 11. Simulation and tests

- **`sim/synthetic_stereo.py`** ray-traces a heightfield world (ground, boxes, pits, a striped object) into stereo images through any camera model, including your lens distortion. It also returns exact depth, height and the ground plane.
- **`sim/synthetic_board.py`** renders checkerboard captures through real lens models, to test calibration end to end.
- **`sim/real_world.py`** drapes a real orthophoto over a real LiDAR surface model, so the
  simulator can fly over real buildings, streets and trees with exact ground truth.
  The drone never sees the LiDAR: it is only the shape the photo is painted on and the answer key.
  Get an area with `python -m tools.fetch_pdok --centre X Y --size 300` (Dutch RD coordinates;
  PDOK open data: AHN4 elevation CC0, 8 cm orthophotos CC BY). Then:
  - score a flight with `python -m demos.real_area --area data/delft --altitude 45`;
  - record one for the viewer with `python -m tools.make_sample_data --area data/delft --altitude 45
    --baseline 0.3 --x-range -120 120 --y 40 --frames 31 --out out/delft`.

  Its limits: no real camera noise, blur, vibration or lighting; walls only appear as stretched
  roof edges, because the orthophoto only sees from above; shadows are baked into the photo.
- **`sim/oracle_perception.py`** produces the frames a perfect pipeline would output, so full missions run in seconds.
- **`sim/mission.py`** runs the closed loop: perception → map → planner → drone kinematics with a battery model.
  If the planner has a belief map, frames feed it too.
- **`sim/search_env.py`** is a reinforcement-learning environment (Gymnasium-style
  `reset()` / `step()`, no dependency): random scenes with roofs and trees hiding objects.
  16 actions (8 directions × cruise 20 m or low 8 m).
  Observation: an egocentric window of the belief layers plus position and time.
  **Reward** = bits of uncertainty removed + 20 per hidden object found − 10 per false
  find − 0.05 per metre flown − 100 for a crash.
- **`sim/search_policies.py`**: random, lawnmower (sweep at cruise) and information-greedy
  (best `expected_gain` minus energy, flying low only over mapped ground).

**Is the reward meaningful?** `python -m demos.search_benchmark`, 8 random scenes × 150 steps,
14 hidden objects in total:

| Policy | Mean reward | Uncertainty removed | Objects found | False finds | Crashes | Flown |
|---|---|---|---|---|---|---|
| random | 13 | 26% | 1/14 | 0 | 0 | 1489 m |
| lawnmower | 233 | 83% | 0/14 | 0 | 0 | 749 m |
| **information-greedy** | **253** | **84%** | **6/14** | 0 | 0 | 747 m |

- Following the reward finds the most hidden objects, with no learning. The lawnmower removes
  about as much uncertainty but never flies low, so it finds none of the objects under trees.
- Greedy still misses more than half. It's one step ahead only, and low passes over trees have
  small footprints. That gap is what an RL policy trained on this reward has to close.

There are 104 tests, one file per module, plus `test_pipeline.py` for the data
tools. Tests marked `slow` render images; skip them with `-m "not slow"`.

**Limit of these tests:** the simulator shares the code's assumptions (pinhole
cameras, uniform lighting, no motion during exposure). Bugs that live in those
assumptions can't show up here. Only real data can find them.

---

## 12. Performance

All timings are on a laptop CPU, single process. **The RB3 hasn't been measured yet.**

| Stage | Time |
|---|---|
| Rectify (1920×1080 → 640×352, both cameras) | ~4–6 ms |
| Terrain (StereoBM + plane + cell statistics) | ~50–65 ms |
| ResNet-18 backbone + scene model | ~25–40 ms |
| Map update | ~1 ms |
| Planner decision | < 5 ms |

The expectation is the same order on the RB3's Cortex-A78 cores, which fits the
3–5 Hz budget. Measure it with `run_data` (timing columns) and
`bench_backbone` on the board.

---

## 13. Validation status

**Verified on synthetic data:**
- Stereo geometry, ground plane, rectification and map placement (exact ground truth).
- Recovering the calibration of your mismatched, distorted pair, including the overfit left lens.
- The whole chain end to end.
- Planner behaviour in closed loop.

**First real-data test:** the [QUT Kagaru airborne stereo dataset](https://michaelwarren.info/docs/datasets/kagaru-airborne-stereo/),
run with `python -m tools.prepare_kagaru` and then `tools.viewer` or `run_data`.
It is a fixed-wing aircraft at about 85 m with downward stereo (0.78 m baseline), 100 frames.

- **Works:** rectification of cameras that are rotated relative to each other, demosaicing,
  dense disparity (85% coverage), regions and novelty, at about 120 ms per frame.
- **Found and fixed:** a bug where a stale ground-plane prior blocked the plane fit.
  There is now a regression test.
- **Found: stereo depth on that airframe is unreliable.**
  - The independent measurement (ground homography between frames, scaled by the INS distance flown)
    gives a steady ~85 m altitude.
  - Stereo reads ~40 m, oscillating at 5.2 Hz.
  - Raw SIFT disparities show the same thing, so it is not a block-matching artifact.
  - Cause: the mount flexes about ±0.2° with vibration, on top of a vergence bias that
    self-calibration from flat terrain can't remove.
- **Lesson for this rig:** mount the cameras rigidly, calibrate vergence with a checkerboard,
  and consider correcting the disparity offset online from the drone's own motion.

**Not yet verified:**
1. Real images from this project's own cameras.
2. **Camera synchronisation.** The two different sensors are probably not hardware-synchronised. Exposures 10 ms apart on a moving drone corrupt stereo depth. This is the biggest risk.
3. A real calibration.
4. Timings on the RB3.
5. Backbone choice and novelty thresholds on real terrain.
6. Integration with the control team's poses and flight controller.

**Validation plan:**

| Stage | Setup | Pass criteria |
|---|---|---|
| 0. Bench | Calibrate. Hold the rig still 2–3 m above boxes of known height. | Calibration RMS < 0.5 px; baseline matches a ruler; box heights within ~5 cm; RB3 timing within budget |
| 1. Handheld | Walk the rig over varied ground, with a few placed odd objects. Record, then `run_data`. | Coverage holds up while moving (tests sync); stable regions; placed objects flagged; acceptable false alarms |
| 2. Logging flight | Fly manually, recording frames and poses. | Altitude agrees with barometer or rangefinder; map is consistent across passes |
| 3. Advisory | Planner commands logged, not flown. | Commands make sense on review |
| 4. Closed loop | Planner flies, pilot ready to override. | Mission completes; safety returns trigger |

---

## 14. Design decisions

| Decision | Why |
|---|---|
| Classical geometry plus one feature network, no end-to-end learning | Runs on a CPU, needs no training data, and its behaviour can be explained |
| CPU fp32 | No quantisation toolchain. The GPU (fp16) or NPU (int8) remain available through the QNN backend. |
| StereoBM, not SGBM | SGBM pixel-locks: at this rig's few-pixel disparities it flattened a 5° slope to about 1° and skewed heights by 10–20 cm. BM keeps sub-pixel detail. SGBM remains an option (`matcher="sgbm"`) for denser coverage. |
| 640 × 352 rectified output | Halves depth error compared with 320 px. Its shape follows the lenses' 16:9 overlap. |
| Ground plane fitted in disparity space, depth edges excluded, prior-gated | One pixel threshold works at every altitude; robust to roofs and smeared edges |
| Thresholds relative to measured feature noise | Self-calibrating across backbones and terrain |
| Novelty from a memory bank, not distance to cluster centres | Distance to a centre in 259 dimensions is mostly noise; nearest-neighbour distance tracks what was actually seen |
| Anomalies confirmed by hits in separate frames, with no negative evidence | Rejects one-frame noise, and survives the scene model getting used to the object |
| Unmapped space assumed tall | The cameras only look down |
| Hidden targets as a probability map; reward = entropy removed | One currency (bits) for every sensor and target type; the same quantity drives the planner and trains RL |
| Evidence only from a better view than before | Repeated frames are not independent; counting them would make the drone hover |
| No gravimeter, real or simulated | There is no sensor on board, and simulated readings would only show assumed noise levels |

---

## 15. Limitations and next steps

**Known limitations:**
- **No real-data validation yet.** See [section 13](#13-validation-status).
- **Steep walls score as novel early on.** In the simulator they are smeared texture; on real data some of this will remain. Treat anomalies as a ranked list to check, not as certain detections.
- **On synthetic texture, road and ground merge** into one region. Real roads should separate better; to be checked.
- **No forward-looking sensor.** Obstacles ahead are only known once they've been mapped from above; unmapped space is assumed to be 3 m tall.
- **Exploration uses frontiers,** about 2.2× an ideal back-and-forth sweep including the return. A dedicated sweep pattern would be more efficient for known rectangular areas.
- **The mission simulator's kinematics are idealised** (no wind, no inertia).
- **The belief map is tested on oracle perception only.** Novelty hits under real canopy,
  and canopy/roof classification from real roughness, still need real data.
- **Underground: indirect cues only.** Ground depressions raise the void belief, but nothing
  on board senses underground, so voids without a surface sign stay invisible.
- **Edges blur in the map** by about one cell, so map-cell heights at object edges are averages.

**Next steps:**
1. **Onboard runtime:** live capture from both cameras → `Pipeline` → planner at 3–5 Hz, with the pose input and command output wired to the control team, plus recording in `run_data`'s format for replay.
2. **Map-based position fixes:** match the current terrain against the map's elevation layer when revisiting ground, and send the correction to the control team. This is the core GNSS-free navigation aid.
3. **Validation stages 0–2** above.
