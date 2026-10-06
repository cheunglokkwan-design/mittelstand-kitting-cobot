# PRD: Micro-Kitting Cobot for Mittelstand Workshops
**For: NVIDIA GTC Berlin Golden Ticket Developer Contest**
**Entry deadline: Sept 10, 2026**

> **Status note:** this revision reflects the system as actually built — dynamic
> closed-loop spatial grasping (Physical AI), not the original static-waypoint
> MVP. See "Changelog from original PRD" at the end for what moved and why.

---

## 1. Overview

A desktop robot that watches a workspace via a depth camera, **dynamically
detects where a part is sitting and how it's rotated**, classifies it by type
with a vision-language model, and picks it with an IK-driven closed-loop
grasp — sorting it into a labeled kitting bin. The system runs on-device on a
Jetson Orin Nano Super.

Unlike a fixed-waypoint pick (jog the arm once to a hardcoded position, replay
forever), the arm now computes a fresh 3D target and wrist orientation for
every part, every time, from live perception.

**Why this problem:** Germany's Mittelstand (small-and-mid-size manufacturing)
and Industry 4.0 sector faces a well-documented skilled-labor shortage —
manufacturing vacancy rates around 16.6%, with sustained shortages in
electrical trades, mechatronics, and industrial maintenance. Repetitive prep
work like parts kitting is exactly the kind of task that pulls skilled workers
away from higher-value work; automating it addresses a real, current
bottleneck.

## 2. Goals & Success Criteria

Primary goal: a working, filmable demo that correctly locates, classifies, and
sorts real small parts by type live — with the arm visibly adapting its pick
position and wrist angle to wherever the part actually is — plus a documented
open-source repo.

Maps to contest judging criteria:
| Criterion | How this PRD addresses it |
|---|---|
| Technical innovation | Zero-shot VLM classification + classical-CV 3D localization/orientation, fused into closed-loop IK grasping (no per-part training data, no hardcoded pick waypoint) |
| Effective use of NVIDIA/partner tech | On-device inference on Jetson Orin Nano |
| Impact/usefulness | Targets a specific, sourced German manufacturing labor-shortage stat |
| Documentation quality | Clean repo, README with setup + results, demo video, before/after time comparison |

**Definition of done:** A single Python entrypoint that takes a live
RealSense feed, detects a part's real-world position and orientation
wherever it's placed, confirms its category via VLM, computes a wrist-aligned
grasp via inverse kinematics, and drives the SO-101 to deposit it in the
correct kitting tray — demonstrated on video for all 3 part types, with parts
placed at visibly different positions/angles between takes (not the same
spot every time).

## 3. Non-Goals (explicitly out of scope given the timeline)

- Handling multiple parts simultaneously in one frame (the perception stack
  picks the single largest qualifying detection; see §12 for what happens
  when that assumption breaks)
- Sorting by precise dimension/thread size (type-level classification only)
- Very small parts below ~5mm (below reliable grasp/vision threshold for this rig)
- Industrial-speed throughput claims — this is a proof-of-concept, not a production line
- General 6-DOF grasp planning — the approach is top-down only (fixed overhead
  camera + fixed top-down grasp orientation), with only position and one
  wrist-roll axis driven dynamically. The SO-101's 5-DOF arm can't
  independently track full 3D orientation anyway.

## 4. Hardware

| Component | Role | Status |
|---|---|---|
| NVIDIA Jetson Orin Nano Super | On-device compute for VLM inference + arm control | Set up |
| Intel RealSense D435i | RGB-D perception of workspace | **Connected and streaming** (aligned depth-to-color) |
| LeRobot SO-101 follower arm | Physical pick-and-place | Connected; motor-level calibration confirmed under id `follower_arm` |
| LeRobot SO-101 leader arm | Not used by this build | Present on the rig (teleoperation hardware), unplugged/idle for this pipeline — see §12 |
| Bambu Lab P1S | Printing trays + camera mount | Available, shared with teammate |

**Note:** Hardware is shared with a teammate building a separate entry
(German waste sorter). Coordinate rig time — see shared-infra milestone below.

## 5. Part Categories (demo scope)

Reduced from the original 4 to 3 categories, still chosen for clear visual
and shape distinction:

| Category | Example demo items |
|---|---|
| Bolt | Hex-head bolts, various lengths |
| Nut | Hex nuts |
| Brass Connector | Brass barbed/tee hose connectors |

`classification/categories.py`'s `normalize_category` also accepts common
synonyms for Brass Connector ("brass terminal", "connector", "terminal") to
absorb VLM phrasing variance.

## 6. Physical Build Spec

- **Kitting trays:** 3 small printed trays/bins (one per category) plus a
  single catch-all `Container` tray used for mock/smoke-test runs where
  category sorting isn't being exercised, all positioned within the SO-101's
  reach and calibrated via `control/tray_calibration.py`.
- **Parts:** Real hardware-store parts (bolts, nuts, brass connectors) — well
  within the SO-101's gripper opening and payload.
- **Camera mount:** Fixed overhead mount, rotated ~180° relative to the mat's
  own "up" (cosmetic only — nothing in the pipeline assumes a particular
  frame orientation).
- **Workspace ROI:** `perception/camera.py`'s `RealSenseCamera` crops to a
  region of interest sized and positioned to the actual usable mat area for
  *this specific* camera mount and mat (an 880×220px band above the mat's
  printed logo text, which otherwise confuses classical-CV segmentation).
  This is mount/mat-specific — re-derive it (see `test_spatial_grasp.py` or
  a quick debug capture) if either ever moves.

## 7. Software Architecture

```
[RealSense D435i] --aligned RGB-D-->
        |                                        \
        |                                         \
 [perception/detector.py]                   [classification/vlm_client.py]
 classical-CV centroid (u,v)                 VLM category confirmation
 + in-plane angle_deg                         (Bolt / Nut / Brass Connector)
        |                                         |
 [perception/camera.py]                           |
 deproject_pixel_to_camera_point                  |
 -> (X_c, Y_c, Z_c), camera frame                 |
        |                                         |
 [control/grasp.py: camera_to_arm_base]            |
 configs/hand_eye.yaml (2D affine)                 |
 -> (X_b, Y_b, Z_b), arm base frame                |
        |                                         |
 [control/kinematics.py + control/grasp.py]         |
 placo IK (assets/so101 URDF), wrist rotated       |
 to angle_deg on top of calibrated pick_grasp pose  |
        |                                         |
        +-------------------- tray lookup <--------+
        |                     (configs/trays.yaml, by confirmed category)
        v
 [SO-101 pick_and_place]
 approach (hover) -> descend -> grasp -> lift -> move to tray -> release -> home
```

The pick *position and wrist angle* are now fully dynamic per part. The tray
*drop-off* position remains a static calibrated waypoint per category — only
the pick side needed to become closed-loop for this PRD's goals.

### 7.1 Perception module (`perception/`)
- `camera.py` — `RealSenseCamera` streams aligned RGB+depth via `pyrealsense2`
  (`rs.align(rs.stream.color)`), exposes:
  - `get_frames()` / `get_roi_frame()` — numpy arrays, for the trigger/polling loop.
  - `get_aligned_rgbd()` — returns the live `depth_frame` object + color-stream
    intrinsics directly, for callers that deproject an externally-found pixel.
  - `deproject_pixel_to_3d(u, v)` — single-pixel deprojection with a 3×3
    median-neighborhood fallback on invalid depth.
  - `deproject_pixel_to_camera_point(u, v, depth_frame, intrinsics, window=5)`
    — always median-samples a 5×5 window (IR stereo depth drops out easily on
    small/shiny/edge-adjacent pixels — exactly where a part's contour
    centroid tends to land).
  - Auto-exposure/white-balance are explicitly re-enabled and given a
    15-frame warm-up on `connect()`, so a prior manual sensor setting can't
    leave a color cast in the stream.
- `trigger.py` — `DepthDeltaTrigger`, unchanged: fires when enough ROI depth
  pixels read closer than a learned empty-mat baseline.
- `detector.py` **(new)** — `detect_workspace_object()`: classical-CV
  centroid + orientation, complementing the VLM (which classifies well but
  doesn't report rotation):
  - Otsu-thresholds the ROI, but tries **both** polarities
    (`THRESH_BINARY` and `THRESH_BINARY_INV`) and keeps whichever marks the
    *smaller* area as foreground — a part can come out either lighter or
    darker than the mat depending on lighting/specular highlights, and only
    the smaller-area guess is reliably "the part, not the mat."
  - Rejects (`None`) any contour occupying more than `MAX_CONTOUR_AREA_FRACTION`
    (35%) of the frame — catches a cable, printed mat logo, or the robot's
    own gripper getting merged into one giant blob instead of confidently
    reporting a bogus centroid from it.
  - Orientation (`angle_deg`) comes from `cv2.minAreaRect()`'s corner points
    (not its raw `angle` field, whose convention differs across OpenCV
    versions), normalized to `(-90, 90]` so a part and its 180°-rotated twin
    report identically.

### 7.2 Classification module (`classification/`)
- `categories.py` — `CATEGORIES = ["Bolt", "Nut", "Brass Connector"]`;
  `normalize_category()` matches case-insensitively plus a small synonym
  table for Brass Connector.
- `vlm_client.py` — `MoondreamClient` wraps local Ollama (`moondream`):
  - `classify()` — category only (used by the main loop, now purely for
    *confirming* what the classical-CV detector already localized).
  - `detect_part_location()` — category **and** a parsed pixel centroid from
    the VLM's own bounding-box/point response (handles both 0–1 normalized
    and raw-pixel conventions). Kept as a standalone VLM-only localization
    path — used by `test_spatial_grasp.py` for perception smoke-testing —
    but not what `main.py`'s primary loop uses for position now.

### 7.3 Kinematics + hand-eye calibration module (`control/`)
Replaces the original "static tray lookup for the pick position" with a real
IK pipeline, while keeping tray lookup for the *place* position.
- `kinematics.py` **(new)** — `solve_ik(x, y, z, grasp_angle=0.0) -> list[float]`
  (6 joint targets: 5 arm joints + gripper). Built on LeRobot's own
  placo-backed `RobotKinematics` (loaded from the real SO-ARM100 URDF —
  fetched into `assets/so101/` from the same source LeRobot's own docs
  point to) rather than a second IK library, so there's one
  hardware-validated solver, not two that could silently disagree.
- `grasp.py` —
  - `ROBOT_ID` — the single motor-calibration id (`"follower_arm"`) every
    script that connects to the physical arm must share; LeRobot looks up
    its one-time motor calibration (homing offsets + joint ranges) by this
    exact string.
  - `camera_to_arm_base(point_cam)` — prefers the interactively-calibrated
    2D affine in `configs/hand_eye.yaml` (see below); falls back to the
    identity-placeholder 3D rigid transform in `configs/camera_extrinsics.yaml`
    if hand-eye calibration hasn't been run yet.
  - `rotation_about_z(angle_deg)` — composes the detector's `angle_deg` onto
    the calibrated `pick_grasp` orientation to align the wrist.
  - `pick_and_place(robot, config, tray_category, target_point_cam, grasp_angle_deg=0.0)`
    — approach (hover `APPROACH_HEIGHT_M` above target) → descend → grasp →
    lift → move to the category's calibrated tray → release → home.
- `calibration_hand_eye.py` **(new)** — interactive 4-point calibration:
  for each point, the camera locates a physically-placed part
  (`perception/detector.py`) and you jog the (torque-disabled) arm tip to
  touch the same spot; `cv2.estimateAffine2D()` fits the 2D camera→arm-base
  mapping from the 4 pairs, saved to `configs/hand_eye.yaml`. Chosen over a
  full 3D rigid/checkerboard calibration because the mat is flat — a 2D fit
  is both easier to run and, in practice, absorbs small mat-tilt/lens
  residuals a pure rotation+translation wouldn't. Includes two safety
  checks earned the hard way (see §12): a minimum point-separation retry,
  and a post-fit sanity check that refuses to save (rather than silently
  writing) a numerically unstable affine.
- `tray_calibration.py` — unchanged in purpose (records `home`,
  `pick_approach`, `pick_grasp`, and per-tray joint-angle waypoints by hand
  into `configs/trays.yaml`), now accepts `--trays NAME [NAME ...]` to
  calibrate a subset (e.g. just `Container` for a quick mock run) instead of
  always all 3 categories.

### 7.4 Orchestration (`main.py`)
Loop, per part: depth-delta trigger → capture aligned RGB-D
(`get_aligned_rgbd`) → classical-CV centroid + orientation
(`detect_workspace_object`) → deproject to camera-frame 3D
(`deproject_pixel_to_camera_point`) → transform to arm-base frame
(`camera_to_arm_base`, logged) → VLM category confirmation (`classify`) →
tray lookup by confirmed category → `pick_and_place` (wrist rotated to the
detected angle) → log timestamp/category/latency to `logs/runs.csv` → repeat.

`--single-bin [TRAY_NAME]` skips category-based routing and sends every
detected part to one fixed tray (default `Container`) regardless of VLM
result — for validating the detect→deproject→grasp→drop chain with only one
tray calibrated, before committing to all 3.

## 8. Repo Structure (as built)

```
mittelstand-kitting-cobot/
├── README.md
├── PRD.md                        # this file
├── main.py                       # orchestration loop
├── test_spatial_grasp.py         # perception smoke test (no arm motion)
├── perception/
│   ├── camera.py                 # RealSense capture, ROI, deprojection
│   ├── detector.py               # classical-CV centroid + orientation
│   └── trigger.py                # depth-delta part-present trigger
├── classification/
│   ├── vlm_client.py             # VLM classify() + detect_part_location()
│   └── categories.py             # Bolt / Nut / Brass Connector + prompts
├── control/
│   ├── tray_calibration.py       # home/approach/grasp/tray waypoint recording
│   ├── calibration_hand_eye.py   # camera<->arm 2D affine calibration
│   ├── kinematics.py             # solve_ik() (placo-backed)
│   └── grasp.py                  # pick_and_place, camera_to_arm_base, IK execution
├── assets/
│   └── so101/                    # SO-ARM100 URDF + meshes, for placo IK
├── configs/
│   ├── trays.yaml                # calibrated home/approach/grasp/tray poses
│   ├── camera_extrinsics.yaml    # 3D rigid fallback (identity placeholder)
│   └── hand_eye.yaml             # 2D affine hand-eye calibration (generated)
└── logs/
    └── runs.csv                  # classification/timing log
```

## 9. Milestones

| Day | Task | Status |
|---|---|---|
| 1-2 | RealSense connected + streaming; camera mount printed and fixed; SO-101 calibration confirmed; basic LeRobot + VLM pipeline classifying *something* and triggering a grasp (shared infra with teammate) | **Done**, and extended: full dynamic IK pipeline (not just a basic trigger) |
| 3 | Trays + part categories defined and printed; tray coordinate calibration; part-classification prompt tuned | Categories finalized (3, not 4); tray waypoint calibration scripted; hand-eye calibration added beyond original scope |
| 4 | Full loop working end-to-end for at least 2 part types; grasp force tuned per part type | In progress — dynamic pick position/orientation working; grasp-force tuning per part still open |
| 5 | All 3 part types working reliably; test set finalized; manual-sort timing baseline captured | Open |
| 6 | Record demo video; write README/documentation, including labor-shortage framing and timing comparison | Open |
| 7 | Post with #NVIDIAGTC, tag chosen judge, submit via form if applicable | Open |
| 8 | Buffer day | — |

## 10. Testing / Validation Checklist

- [ ] D435i streaming reliably at consistent frame rate, auto-exposure/white-balance converged
- [ ] Workspace ROI actually covers the intended pick area for the current mount/mat (verify visually, not just by pixel math)
- [ ] Classical-CV detector isolates a single, correctly-shaped part contour — verified against both threshold polarities and with the gripper/cables intentionally in and out of frame
- [ ] Part-present trigger fires correctly without false positives on small items
- [ ] VLM correctly confirms each of the 3 part categories in isolation
- [ ] Hand-eye calibration's fitted affine has order-1 linear coefficients (sanity-checked automatically; see §12) and low per-point residuals
- [ ] Deprojected 3D targets land within a few mm of the true position across the sampled workspace (spot-check with `test_spatial_grasp.py`)
- [ ] Wrist rotation visibly tracks a part's placed angle across a range of orientations, not just 0°
- [ ] Grasp succeeds reliably (>80% success rate) for each part type across 3+ trials, at multiple positions/orientations — not just one fixed spot
- [ ] Full loop runs unattended for a sequence of 5+ parts, placed at varying positions, without manual intervention
- [ ] Classification logs + timing data captured for documentation

## 11. Documentation & Submission Deliverables

- Public GitHub repo (open-source, per contest entry requirement)
- README with: problem framing (Mittelstand labor shortage, with sourced
  stats), architecture diagram, setup instructions (including the
  motor-calibration and hand-eye-calibration one-time steps), results/limitations,
  rough before/after timing comparison
- 30-60s demo video showing all 3 part types sorted correctly, **with parts
  placed at visibly different positions and angles between takes** — the
  point of this revision is demonstrating that the pick adapts, not that it
  replays a memorized motion
- Social post with #NVIDIAGTC, tagging your chosen judge

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| RealSense integration takes longer than expected | Resolved — connected and streaming |
| Small parts harder to classify than larger items | Visually distinct categories (shape, not just size); tighter camera framing |
| Grasp reliability on small/slippery parts | Test and tune gripper force per part type; avoid parts below ~5mm |
| Shared hardware scheduling conflicts with teammate | Use the Day 1-2 shared-infra block jointly, then alternate time blocks |
| **Two IK solvers silently disagreeing** | Built `solve_ik()` on the same placo/`RobotKinematics` instance `pick_and_place` already uses, instead of adding a second library (e.g. ikpy) against a separately-derived model |
| **Scripts pointing at different motor-calibration ids** | Found via a real crash: `main.py`, `tray_calibration.py`, and `calibration_hand_eye.py` each used a different LeRobot calibration id, so none of them found the arm's existing calibration. Unified under one `ROBOT_ID` constant in `control/grasp.py` |
| **Classical-CV threshold polarity assumption breaking on real hardware** | Assumed parts are always darker than the mat; this specific mat/lighting made metal parts' specular highlights *brighter* than the mat, so `THRESH_BINARY_INV` picked the mat (plus its printed logo) as "foreground" instead. Fixed by trying both polarities and keeping the smaller-area result |
| **A stray cable, printed mat logo, or the robot's own gripper getting detected as "the part"** | Added a max-contour-area sanity cap (rejects >35% of frame) so a merged/false blob is discarded instead of confidently reported; the robot's gripper drooping into frame (torque is off during calibration) is now called out explicitly in the calibration script's prompt |
| **Hand-eye calibration numerically unstable from clustered/collinear points** | Two real runs produced affine coefficients in the 100s (vs. expected order-1) from insufficient point spread. Added a live minimum-separation check (retries a point that's too close to a previous one) and a post-fit sanity check that refuses to save an unstable result, printing the raw captured points for debugging instead of silently writing garbage |
| **Workspace ROI too small or mispositioned to actually calibrate against** | The original ROI was both miscentered (pointed at an empty storage bin) and, after fixing that, too small to physically spread 4 calibration points apart. Resized against an actual debug snapshot of the live frame, constrained by the mat's own printed logo eating the bottom half of the usable area |

---

## Changelog from original PRD

- **§1 Overview** — reframed around dynamic closed-loop grasping (Physical
  AI) instead of static-waypoint replay.
- **§3 Non-Goals** — added: no general 6-DOF grasp planning (top-down only,
  position + one wrist-roll axis dynamic).
- **§4 Hardware** — RealSense now connected; noted the leader arm present on
  the rig but unused by this pipeline.
- **§5 Part Categories** — reduced 4→3: `Screw, Nut, Washer, Connector` →
  `Bolt, Nut, Brass Connector`.
- **§6 Physical Build Spec** — added the mock-run `Container` tray and a note
  that the workspace ROI is mount/mat-specific, derived from an actual debug
  capture rather than assumed.
- **§7 Software Architecture** — substantially rewritten: added classical-CV
  detection (position *and* orientation), hand-eye calibration, and a
  placo-backed IK solver in place of a static per-category pick waypoint.
  VLM's role narrowed from "classify + implicitly define pick position" to
  "confirm category only" for the primary loop.
- **§8 Repo Structure** — added `perception/detector.py`,
  `control/kinematics.py`, `control/calibration_hand_eye.py`,
  `assets/so101/`, `test_spatial_grasp.py`, `configs/hand_eye.yaml`,
  `configs/camera_extrinsics.yaml`, and this `PRD.md`.
- **§9 Milestones** — added a Status column reflecting actual progress.
- **§10 Testing Checklist** — added calibration-fit, ROI-coverage, and
  wrist-orientation-specific checks discovered necessary during real
  hardware bring-up.
- **§12 Risks** — added five risks that only surfaced during actual
  integration (IK duplication, calibration-id mismatch, threshold-polarity
  assumption, false-positive detections, hand-eye numerical instability),
  none of which were anticipated in the original PRD.
