#!/usr/bin/env python3
"""
File: head_motion.py
Project: Engineering Edge-Based Computer Vision for Human Liveness
         Detection in Automotive Driver Monitoring Systems (DMS)
Course: ENGR5510 - Foundations of Software Engineering
Institution: Ontario Tech University
Group: 2
Authors:
    Nathan Perez
    Jacky Zhong
    Rivka Sagi
Date: November 12, 2026

Description:
    Experimental head-motion measurement module. It measures the
    orientation of the ear line (landmarks 7 and 8) relative to the
    shoulder line (landmarks 11 and 12) in successive MediaPipe frames.

Responsibilities:
    - Accept already-detected MediaPipe pose landmarks and a timestamp.
    - Validate the landmarks needed for the measurement.
    - Compute an image-plane relative head-roll angle (degrees).
    - Maintain rolling timestamped raw and smoothed measurements.
    - Estimate angular velocity from consecutive valid observations.
    - Export the recording as CSV for waveform visualization/analysis.
    - Return diagnostic results to main.py without classifying liveness.

Design notes:
    - Does not initialize MediaPipe or open camera hardware.
    - Does not decide whether a driver is alive, fake, or attentive.
    - This 2D relative angle primarily reflects head roll; it is not a
      calibrated estimate of head yaw or pitch.
    - No bobblehead/damped-oscillation classifier is included yet.
    - Both lines use the same camera image coordinate system, where
      the vertical coordinate increases downward.
"""

import csv
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

from .landmarks import landmark_ok

# ============================================================
# CONFIGURATION
# ============================================================

EAR_LEFT = 7
EAR_RIGHT = 8
SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12

MIN_VISIBILITY = 0.55
MIN_EAR_SEPARATION = 0.025
MIN_SHOULDER_SEPARATION = 0.04

HISTORY_SECONDS = 30.0
SMOOTHING_ALPHA = 0.25  # 0 < alpha <= 1; smaller = smoother
MAX_SAMPLE_GAP_SECONDS = 1.0


def _wrap_degrees(angle):
    """Wrap an angle to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def _relative_line_angle(ear_left, ear_right, shoulder_left, shoulder_right):
    """Return ear-line orientation relative to shoulder-line orientation.

    Both lines use consistent left-to-right landmark order. The difference
    is wrapped modulo 180 degrees because a geometric line is undirected.
    The resulting angle is in [-90, 90) degrees.
    """
    ear_angle = math.degrees(math.atan2(
        ear_right.y - ear_left.y, ear_right.x - ear_left.x
    ))
    shoulder_angle = math.degrees(math.atan2(
        shoulder_right.y - shoulder_left.y,
        shoulder_right.x - shoulder_left.x
    ))

    relative = _wrap_degrees(ear_angle - shoulder_angle)
    relative = (relative + 90.0) % 180.0 - 90.0
    return float(relative), float(ear_angle), float(shoulder_angle)


class HeadMotionTracker:
    """Store and analyze a rolling history of relative head-roll angles."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear recording and temporal state."""
        self.samples = deque()
        self.last_timestamp = None
        self.last_raw_angle = None
        self.smoothed_angle = None
        self.angular_velocity = None
        self.last_ear_angle = None
        self.last_shoulder_angle = None
        self.state = "COLLECTING DATA"

    def update(self, landmarks, timestamp_seconds=None):
        """Process one set of MediaPipe landmarks.

        Args:
            landmarks: Sequence of pose landmarks, or None.
            timestamp_seconds: Monotonic time in seconds. Supply the same
                clock used by main.py for other time-based modules.
        Returns:
            A diagnostic dictionary from get_results().
        """
        now = time.monotonic() if timestamp_seconds is None else float(timestamp_seconds)
        if not math.isfinite(now):
            raise ValueError("timestamp_seconds must be finite")
        if self.last_timestamp is not None and now <= self.last_timestamp:
            self.state = "NON-INCREASING TIMESTAMP"
            return self.get_results()

        required = (EAR_LEFT, EAR_RIGHT, SHOULDER_LEFT, SHOULDER_RIGHT)
        if landmarks is None or len(landmarks) <= max(required):
            self._mark_missing(now, "LANDMARKS MISSING")
            return self.get_results()

        if not all(landmark_ok(landmarks[i], MIN_VISIBILITY) for i in required):
            self._mark_missing(now, "LANDMARKS UNCLEAR")
            return self.get_results()

        e_left, e_right = landmarks[EAR_LEFT], landmarks[EAR_RIGHT]
        s_left, s_right = landmarks[SHOULDER_LEFT], landmarks[SHOULDER_RIGHT]

        ear_separation = math.hypot(e_right.x - e_left.x, e_right.y - e_left.y)
        shoulder_separation = math.hypot(s_right.x - s_left.x, s_right.y - s_left.y)
        if ear_separation < MIN_EAR_SEPARATION or shoulder_separation < MIN_SHOULDER_SEPARATION:
            self._mark_missing(now, "LANDMARK SPACING UNCLEAR")
            return self.get_results()

        angle, ear_angle, shoulder_angle = _relative_line_angle(
            e_left, e_right, s_left, s_right
        )

        # Reset the smoothing baseline after a long gap so older motion does
        # not influence a newly visible head.
        gap = (now - self.last_timestamp) if self.last_timestamp is not None else None
        if self.smoothed_angle is None or gap is None or gap > MAX_SAMPLE_GAP_SECONDS:
            smoothed = angle
            velocity = None
        else:
            # Use the shortest angular change, not naive subtraction.
            change = _wrap_degrees(angle - self.last_raw_angle)
            change = (change + 90.0) % 180.0 - 90.0
            smoothed_change = SMOOTHING_ALPHA * (
                (angle - self.smoothed_angle + 90.0) % 180.0 - 90.0
            )
            smoothed = (self.smoothed_angle + smoothed_change + 90.0) % 180.0 - 90.0
            velocity = change / gap

        self.last_timestamp = now
        self.last_raw_angle = angle
        self.smoothed_angle = float(smoothed)
        self.angular_velocity = None if velocity is None else float(velocity)
        self.last_ear_angle = ear_angle
        self.last_shoulder_angle = shoulder_angle
        self.state = "MEASURING"

        self.samples.append({
            "timestamp_seconds": now,
            "head_angle_degrees": angle,
            "smoothed_angle_degrees": self.smoothed_angle,
            "angular_velocity_deg_s": self.angular_velocity,
        })
        self._trim_history(now)
        return self.get_results()

    def _mark_missing(self, now, reason):
        """Mark measurement unavailable without pretending it is zero."""
        self.last_timestamp = None
        self.last_raw_angle = None
        self.smoothed_angle = None
        self.angular_velocity = None
        self.last_ear_angle = None
        self.last_shoulder_angle = None
        self.state = reason
        self._trim_history(now)

    def _trim_history(self, now):
        while self.samples and now - self.samples[0]["timestamp_seconds"] > HISTORY_SECONDS:
            self.samples.popleft()

    def get_results(self):
        """Return current measurements; liveness remains unclassified."""
        return {
            "head_angle_degrees": self.last_raw_angle,
            "smoothed_angle_degrees": self.smoothed_angle,
            "angular_velocity_deg_s": self.angular_velocity,
            "ear_line_angle_degrees": self.last_ear_angle,
            "shoulder_line_angle_degrees": self.last_shoulder_angle,
            "sample_count": len(self.samples),
            "state": self.state,
            "damped_oscillation_detected": None,
            "isMovingAlive": None,
        }

    def get_measurements(self):
        """Return a copy of the current rolling 30-second recording."""
        return list(self.samples)

    def save_csv(self, filename):
        """Save the current recording; return the resulting file path."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = (
            "timestamp_seconds",
            "head_angle_degrees",
            "smoothed_angle_degrees",
            "angular_velocity_deg_s",
        )
        with path.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.samples)
        return path
