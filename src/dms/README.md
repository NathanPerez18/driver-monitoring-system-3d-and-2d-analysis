# Driver Monitoring System — 3D and 2D Analysis

Research prototype investigating camera-based driver attention and potential presentation-attack indicators using RGB pose landmarks and depth-camera analysis.

Developed by Group 2 for ENGR5510 — Foundations of Software Engineering, Ontario Tech University (2026).

## Current baseline features

- RGB pose landmark estimation using MediaPipe Pose Landmarker Lite.
- Head-orientation monitoring and sustained attention warnings.
- Relative head-roll tracking and CSV export.
- Depth-based RANSAC plane detection and missing-depth visualization.
- Experimental pose/plane bounding-box overlap visualization.
- Separate Normal (UNKNOWN-aware) and Strict (fail-closed) classification policies.

**Research warning:** This is an experimental prototype, not a validated driver safety or liveness system. A Strict-mode FAIL is a policy rejection, **not proof that a person is fake**. Head-motion and depth liveness criteria have not yet been validated. RGB/depth overlap is approximate; horizontal mirroring has been corrected for presentation, but full sensor registration/calibration is still outstanding.

## Project layout

```
src/
  orbbec_lib.py            # Camera wrapper (requires compatible native runtime)
  dms/
    __init__.py
    main.py                # Main application / keyboard controls
    landmarks.py           # MediaPipe pose
    attention.py           # Head orientation
    head_motion.py         # Relative head roll and CSV logging
    depth_detection.py     # Planes and missing-depth regions
    pose_plane_overlap.py  # Approximate pose/plane overlap
    classification.py      # Normal, UNKNOWN-aware policy
    classificationStrict.py # Strict fail-closed policy
models/
  pose_landmarker_lite.task # Download separately (not tracked)
requirements.txt
```

## Environment and installation

Tested development environment: Ubuntu 22.04, Python 3.10, Orbbec Astra Pro Plus RGB/depth camera and access to the vendor-compatible depth/IR runtime. Depending on the local Orbbec camera wrapper, additional system libraries, SDK installation or USB permissions may be necessary. Installing `requirements.txt` alone does **not** install those native dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the **MediaPipe Pose Landmarker Lite** model from the official model collection and place the file at `models/pose_landmarker_lite.task`:
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

Check `src/orbbec_lib.py` and your camera SDK setup for any further machine-specific installation requirements.

## Run

After connecting the RGB/depth camera and completing its vendor runtime setup:

```bash
cd src
python -m dms.main
```

Keyboard controls:

| Key | Function |
| --- | --- |
| P | Toggle skeleton overlay |
| H | Toggle head-orientation monitoring |
| M | Toggle relative head-motion measurement |
| S | Save recent head-motion measurements to CSV |
| F | Toggle RANSAC plane detection |
| N | Toggle missing-depth regions |
| O | Toggle approximate pose/plane overlap |
| C | Switch between Normal and Strict classification |
| D / I | Switch between depth and infrared views |
| + / - | Adjust maximum plane detection range |
| Q / Esc | Quit |

## Planned work

- Verify RGB-to-depth camera registration and overlap measurements.
- Validate liveness and spoofing indicators against controlled human/printed-photo/bobblehead trials.
- Increase head-motion sampling rate (currently approximately 2.7 measurements/s; target 10–15/s).
- Profile end-to-end latency and benchmark on Jetson devices.
- Add reproducible test datasets and evaluation methodology without publishing identifiable subjects without appropriate consent.

## Dependencies

Core Python packages are listed in `requirements.txt`; see the camera wrapper and system prerequisites for the native camera SDK. For a reproducible release, pin the dependency versions from the working virtual environment instead of assuming that the newest package versions are compatible.
