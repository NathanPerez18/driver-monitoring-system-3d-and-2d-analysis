#!/usr/bin/env python3
"""
File: main.py
Project: Engineering Edge-Based Computer Vision for Human Liveness
         Detection in Automotive Driver Monitoring Systems (DMS)

Course: ENGR5510 - Foundations of Software Engineering
Institution: Ontario Tech University
Group: 2

Authors:
    Nathan Perez
    Jacky Zhong
    Rivka Sagi

Date: November 13, 2026

Description:
    Main application entry point and central controller for the
    edge-based Driver Monitoring System (DMS).

    This script coordinates hardware, computer vision modules,
    classification logic, and the user interface. It uses an Orbbec
    Astra Pro Plus camera to acquire RGB images and depth/IR data.

    Integration stage 6: landmark, attention, head-motion, depth,
    normal/strict classification, and experimental pose-plane overlap.
    MediaPipe runs once per RGB frame; the resulting landmarks are
    reused by the attention and head-motion modules. Press C to
    switch between normal and strict driver decision policies.

Responsibilities:
    1. Hardware Initialization
       - Open the RGB camera through OpenCV.
       - Initialize the Orbbec depth/infrared stream.
       - Configure camera resolution and manage device resources.

    2. Frame Acquisition
       - Acquire RGB and depth/IR frames during each processing loop.
       - Generate monotonic timestamps for processing modules.
       - Handle occasional missing camera frames.

    3. Module Coordination
       - Initialize landmark, attention, and head-motion modules.
       - Reuse the same landmarks without repeated MediaPipe inference.
       - Maintain 30 seconds of relative head-angle measurements.
       - Analyze planar and invalid-depth regions using depth_detection.py.
       - Compare upper-body pose ROI with depth-plane bounding boxes
         using pose_plane_overlap.py (visual estimate only).
       - Forward module results to the selected decision policy.

    4. Classification Coordination
       - Combine attention, head-motion, and depth-analysis results.
       - Support a NORMAL policy that preserves UNKNOWN evidence.
       - Support an optional STRICT policy that rejects missing,
         inconclusive, and failed criteria without claiming proof of spoofing.
       - Pass pose and usable-depth availability to the strict policy.

    5. User Interface
       - Display side-by-side RGB and depth/IR camera feeds.
       - Display pose landmarks, head orientation, motion diagnostics.
       - Overlay RANSAC planes and missing-depth regions on depth view.
       - Display sustained non-forward head orientation warnings.
       - Export rolling head-motion measurements as CSV on request.
       - Display independent pose/depth/attention/liveness indicators.
       - Display provisional pose/plane bounding-box overlap.
       - Display the active classifier mode, decision and reason.
       - Handle skeleton, attention, classifier and depth/IR toggle keys.

    6. Performance Monitoring
       - Measure MediaPipe inference time, processing-loop time,
         and displayed RGB/depth frame rates.
       - Support future comparisons on NVIDIA Jetson hardware.

    7. Application Lifecycle
       - Initialize components, execute the processing loop,
         and cleanly close models, cameras, and windows.

Design Principles:
    - Single responsibility for each processing module.
    - One Python process to avoid inter-process communication overhead.
    - One pose inference per RGB frame.
    - Reusable module interfaces for independent testing.
    - Separate attention and human-liveness classifications.
    - Fail-closed policy rejection is NOT proof that a driver is fake.

Note:
    Experimental research prototype; not a validated automotive
    safety system.

Keyboard Controls:
    P = toggle skeleton overlay (pose detection continues)
    H = toggle head-orientation monitoring
    M = toggle relative head-motion measurement
    S = save last 30 seconds of head-motion measurements to CSV
    F = toggle RANSAC plane detection
    N = toggle missing-depth detection and overlay
    + / - = increase / decrease maximum plane distance by 50 mm
    D = show depth stream
    I = show infrared stream
    C = toggle NORMAL / STRICT classification policy
    O = toggle approximate RGB pose / depth-plane overlap overlay
    Q / Esc = quit

Run from src/ with:
    python -m dms.main
"""

# ============================================================
# IMPORTS
# ============================================================

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# orbbec_lib.py remains in the parent src/ directory.
# Running "python -m dms.main" from src/ makes it importable.
from orbbec_lib import Context, Stream, DEPTH, IR

from .landmarks import LandmarkDetector
from .attention import HeadOrientationTracker
from .head_motion import HeadMotionTracker
from .depth_detection import DepthDetector
from .classification import DriverClassifier as NormalDriverClassifier
from .classificationStrict import DriverClassifier as StrictDriverClassifier
from .pose_plane_overlap import (
    analyze_pose_plane_overlap, draw_pose_plane_overlap
)


# ============================================================
# SYSTEM CONFIGURATION
# ============================================================

RGB_DEVICE = "/dev/video0"
RGB_WIDTH = 640
RGB_HEIGHT = 480

WINDOW_TITLE = "DMS | P Pose | H Head | M Motion | F Planes | O Overlap | N Missing | C Mode | D/I View | Q Quit"

POSE_ENABLED_BY_DEFAULT = True
HEAD_ENABLED_BY_DEFAULT = True
MOTION_ENABLED_BY_DEFAULT = True
PLANES_ENABLED_BY_DEFAULT = True
INVALID_ENABLED_BY_DEFAULT = False
CLASSIFICATION_MODE_BY_DEFAULT = "NORMAL"  # NORMAL / STRICT; C toggles
OVERLAP_ENABLED_BY_DEFAULT = True  # O toggles; requires F and D
# WARNING: merely using the same pixel dimensions does NOT register RGB/depth.
# Keep False until calibrated RGB-to-depth registration is actually verified.
RGB_DEPTH_ALIGNED = False

# dms/main.py -> src/ -> project root
RECORDINGS_DIR = Path(__file__).resolve().parents[2] / "recordings" / "head_motion"

# OpenCV BGR warning colors, matching the original viewer.
WARNING_COLORS = {
    "DOWN": (0, 0, 180),
    "UP": (140, 130, 0),
    "LEFT": (180, 90, 0),
    "RIGHT": (0, 110, 205),
}


# ============================================================
# DISPLAY HELPERS
# ============================================================

def put_label(image, message, x, y, color=(255, 255, 255), scale=0.56):
    """Draw a label on a BGR image."""
    cv2.putText(
        image, str(message), (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )



def draw_attention_warning(image, direction):
    """Draw the sustained head-orientation warning from the old viewer."""
    if direction not in WARNING_COLORS:
        return

    height, width = image.shape[:2]
    top = int(height * 0.64)
    bottom = int(height * 0.96)
    overlay = image.copy()
    cv2.rectangle(
        overlay, (8, top), (width - 8, bottom),
        WARNING_COLORS[direction], -1
    )
    cv2.addWeighted(overlay, 0.88, image, 0.12, 0, dst=image)

    for message, baseline, scale in (
        (f"LOOKING {direction}", top + 34, 0.85),
        ("PLEASE PAY", top + 84, 1.35),
        ("ATTENTION TO ME", top + 132, 1.25),
    ):
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 3
        measured = cv2.getTextSize(message, font, scale, thickness)[0][0]
        if measured > width - 30:
            scale *= (width - 30) / measured
        measured = cv2.getTextSize(message, font, scale, thickness)[0][0]
        x = max(10, (width - measured) // 2)
        cv2.putText(
            image, message, (x + 2, baseline + 2), font,
            scale, (0, 0, 0), thickness + 3, cv2.LINE_AA
        )
        cv2.putText(
            image, message, (x, baseline), font, scale,
            (255, 255, 255), thickness, cv2.LINE_AA
        )

def colorize_depth(depth):
    """Preserve the depth color map from viewer_unified.py."""
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.clip(
        (depth.astype(np.float32) - 100) / 700 * 255,
        0, 255
    ).astype(np.uint8)
    image = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    image[~valid] = 0
    return image


def visualize_ir(ir_frame):
    """Convert the Orbbec IR frame into a three-channel display."""
    gray = cv2.normalize(
        ir_frame, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def blank_panel(message):
    """Show a placeholder if a camera frame is unavailable."""
    panel = np.zeros((RGB_HEIGHT, RGB_WIDTH, 3), dtype=np.uint8)
    put_label(panel, message, 15, 30, (0, 180, 255))
    return panel


# ============================================================
# HARDWARE INITIALIZATION
# ============================================================

def initialize_hardware():
    """Open the RGB camera; the depth stream starts in main()."""
    cap = cv2.VideoCapture(RGB_DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RGB_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RGB_HEIGHT)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open RGB camera: {RGB_DEVICE}")

    print(f"[Hardware] RGB camera opened: {RGB_DEVICE}")
    return cap


# ============================================================
# SOFTWARE MODULE INITIALIZATION
# ============================================================

def initialize_modules():
    """Initialize pose detection and independent temporal analysis modules."""
    detector = LandmarkDetector()
    detector.initialize()
    attention = HeadOrientationTracker()
    motion = HeadMotionTracker()
    depth_detector = DepthDetector()
    normal_classifier = NormalDriverClassifier()
    strict_classifier = StrictDriverClassifier()
    return (detector, attention, motion, depth_detector,
            normal_classifier, strict_classifier)


# ============================================================
# FRAME ACQUISITION
# ============================================================

def acquire_frames(cap, stream):
    """Read the camera frames; return None for missing frames."""
    depth_or_ir = stream.read()
    ok, bgr_frame = cap.read()

    if not ok or bgr_frame is None:
        bgr_frame = None
    else:
        bgr_frame = cv2.resize(
            bgr_frame, (RGB_WIDTH, RGB_HEIGHT)
        )

    return bgr_frame, depth_or_ir


# ============================================================
# MODULE PROCESSING
# ============================================================

def process_frames(rgb_frame, detector, attention_tracker, motion_tracker,
                   timestamp_ms, head_enabled, motion_enabled, now_seconds):
    """Run one pose inference; reuse landmarks for attention and head motion."""
    landmarks = None
    pose_ms = 0.0

    if rgb_frame is not None:
        landmarks, pose_ms = detector.detect(
            rgb_frame, timestamp_ms=timestamp_ms
        )

    attention_results = None
    if head_enabled:
        # Update even on frames with no pose so the warning timer resets
        # rather than holding a stale attention state.
        attention_results = attention_tracker.update(
            landmarks, now_seconds
        )

    motion_results = None
    if motion_enabled:
        # A missing pose is passed through so the motion tracker marks the
        # measurement unavailable rather than repeating an old angle.
        motion_results = motion_tracker.update(landmarks, now_seconds)

    return {
        "landmarks": landmarks,
        "pose_detected": landmarks is not None,
        "pose_ms": pose_ms,
        "attention": attention_results,
        "head_motion": motion_results,
        "breathing": None,
        "depth": None,
        "liveness": None,
        "classification": None,
        "pose_plane_overlap": None,
        "depth_available": False,
    }


# ============================================================
# DRIVER CLASSIFICATION
# ============================================================

def depth_frame_is_usable(sensor_mode, sensor_frame):
    """Usable measurements exist in a depth frame (NOT verified 3D liveness)."""
    return bool(
        sensor_mode == DEPTH
        and sensor_frame is not None
        and np.any(np.isfinite(sensor_frame) & (sensor_frame > 0))
    )


def classify_driver(results, normal_classifier, strict_classifier,
                    classification_mode, sensor_mode, sensor_frame):
    """Evaluate current evidence; keep NORMAL uncertainty and STRICT policy."""
    depth_available = depth_frame_is_usable(sensor_mode, sensor_frame)
    results["depth_available"] = depth_available
    arguments = {
        "attention_results": results.get("attention"),
        "head_motion_results": results.get("head_motion"),
        "depth_results": results.get("depth"),
    }
    if classification_mode == "STRICT":
        return strict_classifier.evaluate(
            **arguments,
            pose_detected=bool(results.get("pose_detected")),
            depth_available=depth_available,
        )
    if classification_mode == "NORMAL":
        return normal_classifier.evaluate(**arguments)
    raise ValueError(f"Unsupported classification mode: {classification_mode}")


def draw_classification_status(image, classification, classification_mode,
                               pose_detected, depth_available):
    """Independently show sensor availability, evidence and policy outcome.

    The overall UNKNOWN (normal) / FAIL (strict) can remain unchanged when
    pose disappears, so the availability indicators MUST remain explicit.
    """
    if classification is None:
        return

    height, width = image.shape[:2]
    top = height - 98
    overlay = image.copy()
    cv2.rectangle(overlay, (0, top), (width - 1, height - 1),
                  (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.86, image, 0.14, 0, dst=image)

    attention_status = classification.get("attention_status", "UNKNOWN")
    motion_status = (
        "PASS" if classification.get("isMovingAlive") is True else
        "FAIL" if classification.get("isMovingAlive") is False else "UNKNOWN"
    )
    depth_status = (
        "PASS" if classification.get("is3D") is True else
        "FAIL" if classification.get("is3D") is False else "UNKNOWN"
    )
    put_label(
        image,
        f"POSE: {'FOUND' if pose_detected else 'MISSING'}  "
        f"DEPTH: {'FOUND' if depth_available else 'MISSING'}  "
        f"ATTN: {attention_status}",
        10, top + 18,
        (0, 215, 255) if not pose_detected or not depth_available else (255, 255, 255),
        scale=0.48,
    )
    put_label(
        image,
        f"Motion liveness: {motion_status}   3D liveness: {depth_status}",
        10, top + 39, scale=0.47,
    )

    driver_status = classification.get("driver_status", "UNKNOWN")
    status_color = {
        "PASS": (0, 220, 0), "FAIL": (0, 75, 255),
        "UNKNOWN": (0, 220, 255),
    }.get(driver_status, (255, 255, 255))
    if classification_mode == "STRICT":
        summary = (f"STRICT (C) Driver: {driver_status}  "
                   f"Fake policy flag: {classification.get('isFakeDriver')}")
    else:
        summary = (f"NORMAL (C) Driver: {driver_status}  "
                   f"Spoof: {classification.get('spoofing_status', 'UNKNOWN')}")
    put_label(image, summary, 10, top + 61, status_color, scale=0.47)

    if not pose_detected:
        reason = "Pose not detected (no spoof proof)"
    elif not depth_available:
        reason = "Usable depth unavailable (no spoof proof)"
    elif classification_mode == "STRICT":
        reasons = classification.get("rejection_reasons", [])
        reason = reasons[0] if reasons else "All strict conditions passed"
    else:
        reasons = classification.get("reasons", [])
        reason = reasons[0] if reasons else "No further reason"

    message = "Reason: " + reason
    scale = 0.45
    available = max(50, width - 20)
    measured = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX,
                               scale, 1)[0][0]
    if measured > available:
        scale *= available / measured
    put_label(image, message, 10, top + 84, scale=scale)


# ============================================================
# USER INTERFACE
# ============================================================

def update_display(
    rgb_frame, sensor_frame, sensor_mode, results,
    detector, depth_detector, show_skeleton, head_enabled, motion_enabled,
    rgb_fps, sensor_fps, loop_ms, planes_enabled, invalid_enabled,
    classification_mode, overlap_enabled
):
    """Draw two panels and display the combined viewer."""
    if rgb_frame is None:
        left = blank_panel("RGB CAMERA: NO FRAME")
    else:
        # Do not modify the camera frame used for model inference.
        left = rgb_frame.copy()
        landmarks = results["landmarks"]
        if show_skeleton and landmarks is not None:
            detector.draw(left, landmarks)
        if overlap_enabled and results.get("pose_plane_overlap") is not None:
            draw_pose_plane_overlap(left, results["pose_plane_overlap"])

        put_label(left, "RGB + MediaPipe", 10, 24, (0, 255, 0))
        put_label(
            left,
            "POSE DETECTED" if results["pose_detected"] else "NO POSE",
            10, 50,
            (0, 255, 0) if results["pose_detected"] else (0, 0, 255)
        )
        put_label(left, f"Pose inference: {results['pose_ms']:.0f} ms", 10, 76)
        put_label(left, f"RGB FPS: {rgb_fps:.1f}", 10, 102)
        put_label(left, f"Loop: {loop_ms:.0f} ms", 10, 128)
        put_label(
            left,
            f"Skeleton: {'ON' if show_skeleton else 'OFF'} (P)",
            10, 154
        )

        attention = results["attention"]
        if head_enabled and attention is not None:
            warning = attention["warning"]
            put_label(
                left, f"Head: {attention['state']}", 10, 182,
                (0, 0, 255) if warning else (255, 255, 255)
            )

            measurements = attention["measurements"]
            for y, label, value in (
                (208, "Eye-ear delta", measurements["eye_ear_delta"]),
                (234, "Side asymmetry", measurements["side_asymmetry"]),
                (260, "Ear-nose delta", measurements["ear_nose_delta"]),
            ):
                formatted = f"{value:+.3f}" if value is not None else "--"
                put_label(left, f"{label}: {formatted}", 10, y)

            if warning:
                draw_attention_warning(left, attention["direction"])
        else:
            put_label(left, "Head monitor: OFF (H)", 10, 182)

    if sensor_frame is None:
        right = blank_panel("DEPTH/IR CAMERA: NO FRAME")
    else:
        depth_results = results.get("depth")
        if sensor_mode == DEPTH:
            # Analyze and draw overlays only for real depth frames, never IR.
            # draw() uses a fresh display and never modifies raw measurements.
            right = depth_detector.draw(
                sensor_frame, depth_results,
                show_planes=planes_enabled,
                show_invalid=invalid_enabled
            )
            right = cv2.flip(right,1)
            mode_name = "DEPTH"
        else:
            right = visualize_ir(sensor_frame)
            mode_name = "IR"

        right = cv2.resize(right, (RGB_WIDTH, RGB_HEIGHT))
        put_label(right, f"{mode_name} {sensor_fps:.1f} fps", 10, 24)

        if sensor_mode == DEPTH and depth_results is not None:
            valid_percent = depth_results["valid_depth_percent"]
            percent_text = f"{valid_percent:.1f}%" if valid_percent is not None else "--"
            put_label(right, f"Valid depth: {percent_text}", 10, 50)

    # Head-motion diagnostics appear on the sensor panel so that the
    # attention warning banner cannot obscure them on the RGB panel.
    motion = results.get("head_motion")
    if motion_enabled and motion is not None:
        put_label(right, "Head motion: ON (M)", 10, 90, (0, 255, 255))
        put_label(right, f"Motion: {motion['state']}", 10, 116, (0, 255, 255), 0.48)
        for y, caption, key, suffix in (
            (142, "Roll angle", "head_angle_degrees", " deg"),
            (168, "Smoothed", "smoothed_angle_degrees", " deg"),
            (194, "Angular velocity", "angular_velocity_deg_s", " deg/s"),
        ):
            value = motion[key]
            formatted = f"{value:+.2f}{suffix}" if value is not None else "--"
            put_label(right, f"{caption}: {formatted}", 10, y, scale=0.48)
        put_label(right, f"Samples (30s): {motion['sample_count']}", 10, 220, scale=0.48)
        put_label(right, "S: save motion CSV", 10, 246, scale=0.48)
    else:
        put_label(right, "Head motion: OFF (M)", 10, 90, scale=0.48)

    # The overlap feature only compares bounding boxes. Without verified
    # RGB/depth registration, it is NOT a reliable geometric match.
    if overlap_enabled:
        overlap = results.get("pose_plane_overlap") or {}
        percent = overlap.get("best_overlap_percent")
        if percent is None:
            overlap_message = f"Pose/plane: {overlap.get('status', 'UNKNOWN')}"
        else:
            alignment_note = ("REGISTERED" if overlap.get("rgb_depth_aligned")
                              else "ROUGH / UNALIGNED")
            overlap_message = f"Pose/plane box: {percent:.1f}% ({alignment_note})"
    else:
        overlap_message = "Pose/plane overlap: OFF (O)"
    put_label(right, overlap_message, 10, 271, (0, 220, 255), scale=0.45)

    # Separate depth diagnostics occupy the bottom of the sensor panel.
    if sensor_mode == DEPTH:
        depth_results = results.get("depth")
        if depth_results is not None:
            planes_text = (f"Planes: {depth_results['plane_count']}"
                           if planes_enabled else "Planes: OFF (F)")
            missing_text = (f"Missing regions: {depth_results['invalid_region_count']}"
                            if invalid_enabled else "Missing: OFF (N)")
            put_label(right, planes_text, 10, 292, scale=0.49)
            put_label(right, missing_text, 10, 318, scale=0.49)
            put_label(right, f"Depth analysis: {depth_results['analysis_ms']:.0f} ms",
                      10, 344, scale=0.49)
        else:
            put_label(right, "Depth analysis: no frame", 10, 292, scale=0.49)
        put_label(right, f"Plane range: {depth_detector.min_depth}-{depth_detector.max_depth} mm",
                  10, 370, scale=0.49)

    draw_classification_status(
        right, results.get("classification"), classification_mode,
        results.get("pose_detected", False), results.get("depth_available", False)
    )
    cv2.imshow(WINDOW_TITLE, np.hstack((left, right)))


# ============================================================
# PERFORMANCE MONITORING
# ============================================================

class FrameRateCounter:
    """Track an approximate per-stream FPS without external tools."""

    def __init__(self):
        self.last_update = time.perf_counter()
        self.frames = 0
        self.fps = 0.0

    def add_frame(self):
        self.frames += 1
        elapsed = time.perf_counter() - self.last_update
        if elapsed >= 1.0:
            self.fps = self.frames / elapsed
            self.frames = 0
            self.last_update = time.perf_counter()


def update_performance_metrics(start_time):
    """Measure application processing-loop time in milliseconds."""
    return (time.perf_counter() - start_time) * 1000.0


# ============================================================
# APPLICATION SHUTDOWN
# ============================================================

def shutdown(cap=None, detector=None):
    """Release model, RGB camera, and display resources."""
    if detector is not None:
        detector.close()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


# ============================================================
# MAIN APPLICATION LOOP
# ============================================================

def main():
    """Run acquisition, CV analysis, and switchable driver classification."""
    print("Starting Driver Monitoring System (landmarks + attention + head motion + depth)...")

    cap = None
    detector = None
    attention_tracker = None
    motion_tracker = None
    depth_detector = None
    normal_classifier = None
    strict_classifier = None
    stream = None
    show_skeleton = POSE_ENABLED_BY_DEFAULT
    head_enabled = HEAD_ENABLED_BY_DEFAULT
    motion_enabled = MOTION_ENABLED_BY_DEFAULT
    planes_enabled = PLANES_ENABLED_BY_DEFAULT
    invalid_enabled = INVALID_ENABLED_BY_DEFAULT
    classification_mode = CLASSIFICATION_MODE_BY_DEFAULT
    overlap_enabled = OVERLAP_ENABLED_BY_DEFAULT
    sensor_mode = DEPTH

    rgb_counter = FrameRateCounter()
    sensor_counter = FrameRateCounter()
    start = time.monotonic()
    previous_timestamp_ms = -1

    try:
        cap = initialize_hardware()
        (detector, attention_tracker, motion_tracker, depth_detector,
         normal_classifier, strict_classifier) = initialize_modules()

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        print("[Controls] P pose | H head | M motion | S save CSV | F planes | O overlap | N missing | +/- range | D depth | I infrared | C classifier | Q/Esc quit")
        print("[Overlap] Diagnostic bounding-box comparison ONLY; RGB/depth registration not verified.")
        print(f"[Classifier] Policy: {classification_mode} (NORMAL preserves UNKNOWN; STRICT rejects it)")

        # Context manages the Orbbec device; stream is stopped below.
        with Context() as ctx:
            stream = Stream(ctx, sensor_mode).start()

            try:
                while True:
                    loop_start = time.perf_counter()

                    rgb_frame, sensor_frame = acquire_frames(cap, stream)
                    if rgb_frame is not None:
                        rgb_counter.add_frame()
                    if sensor_frame is not None:
                        sensor_counter.add_frame()

                    # Use the same increasing timestamp behavior as the
                    # original viewer; landmarks.py also guards monotonicity.
                    timestamp_ms = max(
                        int((time.monotonic() - start) * 1000),
                        previous_timestamp_ms + 1
                    )
                    previous_timestamp_ms = timestamp_ms

                    results = process_frames(
                        rgb_frame, detector, attention_tracker, motion_tracker,
                        timestamp_ms, head_enabled, motion_enabled, time.monotonic()
                    )
                    # Analyze depth only in DEPTH mode. An IR frame is not a
                    # point cloud and must not be sent through RANSAC.
                    # Disabled features do not contribute processing overhead.
                    if sensor_mode == DEPTH and sensor_frame is not None:
                        results["depth"] = depth_detector.analyze(
                            sensor_frame,
                            detect_planes_enabled=planes_enabled,
                            detect_invalid_enabled=invalid_enabled
                        )
                    else:
                        results["depth"] = None
                    # Compare depth planes to the MediaPipe upper-body ROI.
                    # This is diagnostic only: never change the spoofing or
                    # liveness classification from unregistered bbox overlap.
                    if not overlap_enabled:
                        results["pose_plane_overlap"] = {"status": "OFF (O)"}
                    elif sensor_mode != DEPTH:
                        results["pose_plane_overlap"] = {"status": "DEPTH MODE REQUIRED (D)"}
                    elif not planes_enabled:
                        results["pose_plane_overlap"] = {"status": "PLANES OFF (F)"}
                    else:
                        results["pose_plane_overlap"] = analyze_pose_plane_overlap(
                            landmarks=results.get("landmarks"),
                            depth_results=results.get("depth"),
                            rgb_shape=(rgb_frame.shape if rgb_frame is not None
                                       else (RGB_HEIGHT, RGB_WIDTH, 3)),
                            depth_shape=(sensor_frame.shape if sensor_frame is not None
                                         else None),
                            rgb_depth_aligned=RGB_DEPTH_ALIGNED,
                        )
                    # The selected classifier sees the latest results only;
                    # previous-frame observations are never reused.
                    results["classification"] = classify_driver(
                        results, normal_classifier, strict_classifier,
                        classification_mode, sensor_mode, sensor_frame
                    )
                    loop_ms = update_performance_metrics(loop_start)

                    update_display(
                        rgb_frame, sensor_frame, sensor_mode,
                        results, detector, depth_detector, show_skeleton, head_enabled,
                        motion_enabled, rgb_counter.fps, sensor_counter.fps, loop_ms,
                        planes_enabled, invalid_enabled, classification_mode,
                        overlap_enabled
                    )

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    elif key == ord("c"):
                        classification_mode = (
                            "STRICT" if classification_mode == "NORMAL"
                            else "NORMAL"
                        )
                        print(f"[Classifier] Policy switched to: {classification_mode}")
                    elif key == ord("o"):
                        overlap_enabled = not overlap_enabled
                        print("[Overlap] Display:", overlap_enabled)
                    elif key == ord("p"):
                        show_skeleton = not show_skeleton
                        print("[View] Skeleton overlay:", show_skeleton)
                    elif key == ord("h"):
                        head_enabled = not head_enabled
                        attention_tracker.reset()
                        print("[View] Head monitoring:", head_enabled)
                    elif key == ord("m"):
                        motion_enabled = not motion_enabled
                        motion_tracker.reset()  # new recording on re-enable
                        print("[Motion] Measurement:", motion_enabled)
                    elif key == ord("s"):
                        if not motion_tracker.get_measurements():
                            print("[Motion] No measurements to save yet.")
                        else:
                            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
                            filename = "head_motion_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".csv"
                            target = RECORDINGS_DIR / filename
                            try:
                                saved_path = motion_tracker.save_csv(target)
                                print(f"[Motion] CSV saved: {saved_path}")
                            except OSError as exc:
                                print(f"[Motion] CSV save failed: {exc}")
                    elif key == ord("f"):
                        planes_enabled = not planes_enabled
                        print("[Depth] Plane detection:", planes_enabled)
                    elif key == ord("n"):
                        invalid_enabled = not invalid_enabled
                        print("[Depth] Invalid-depth detection/overlay:", invalid_enabled)
                    elif key in (ord("+"), ord("=")):
                        depth_detector.set_max_depth(depth_detector.max_depth + 50)
                        print("[Depth] Max plane distance:", depth_detector.max_depth, "mm")
                    elif key == ord("-"):
                        depth_detector.set_max_depth(depth_detector.max_depth - 50)
                        print("[Depth] Max plane distance:", depth_detector.max_depth, "mm")
                    elif key == ord("i") and sensor_mode != IR:
                        stream.stop()
                        stream = None
                        sensor_mode = IR
                        stream = Stream(ctx, sensor_mode).start()
                        sensor_counter = FrameRateCounter()
                        print("[Camera] Infrared view")
                    elif key == ord("d") and sensor_mode != DEPTH:
                        stream.stop()
                        stream = None
                        sensor_mode = DEPTH
                        stream = Stream(ctx, sensor_mode).start()
                        sensor_counter = FrameRateCounter()
                        print("[Camera] Depth view")
            finally:
                if stream is not None:
                    stream.stop()
                    stream = None
    finally:
        shutdown(cap, detector)
        print("Driver Monitoring System stopped.")


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
