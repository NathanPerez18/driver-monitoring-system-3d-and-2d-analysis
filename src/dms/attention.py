#!/usr/bin/env python3
"""
File: attention.py
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
    Head-orientation analysis module for the edge-based Driver
    Monitoring System (DMS).

    Uses facial landmarks returned by MediaPipe Pose Landmarker Lite
    to estimate forward, downward, and sideways head orientations.
    Upward detection is temporarily disabled while its geometric
    threshold is calibrated.

Responsibilities:
    1. Validate required facial landmarks from landmarks.py.
    2. Calculate normalized eye-to-ear vertical separation for DOWN.
    3. Calculate facial asymmetry for LEFT and RIGHT.
    4. Retain ear-to-nose separation as an experimental diagnostic.
    5. Require sustained non-forward orientation before warning.
    6. Return direction, attention state, warnings, and diagnostics.

Design:
    - No camera acquisition or MediaPipe model initialization.
    - No direct UI or keyboard handling.
    - Keeps timing state within HeadOrientationTracker.
    - Checks DOWN before LEFT/RIGHT to reduce misclassification
      of downward head tilts as sideways turns.

Inputs:
    - MediaPipe pose landmarks (or None).
    - Monotonic timestamp in seconds.

Outputs:
    - Orientation: FORWARD, DOWN, LEFT, RIGHT, or UNKNOWN.
    - Attention and warning flags, state, and measurements.

Note:
    Head direction is only a proxy for attention; this is an
    experimental prototype, not a validated automotive DMS.
"""

import time

import numpy as np

from .landmarks import landmark_ok


# ============================================================
# HEAD ORIENTATION CONFIGURATION
# ============================================================

HEAD_DOWN_THRESHOLD = 0.0
HEAD_DOWN_SECONDS = 1.5

# Retained for later calibration; UP detection is disabled below.
HEAD_UP_THRESHOLD = -0.20
HEAD_UP_SECONDS = 1.5

SIDE_ASYMMETRY_THRESHOLD = 0.20
SIDE_TURN_SECONDS = 1.5
SWAP_LEFT_RIGHT = False

HEAD_MIN_VISIBILITY = 0.55
HEAD_MIN_EAR_SEPARATION = 0.025


# ============================================================
# HEAD ORIENTATION TRACKER
# ============================================================

class HeadOrientationTracker:
    """Estimate head direction from 2D MediaPipe pose landmarks."""

    def __init__(self):
        self.delta = None
        self.ear_nose_delta = None
        self.asymmetry = None

        self.state = "NO HEAD DATA"
        self.direction = "UNKNOWN"
        self.isPayingAttention = None
        self.warning = False

        self.pending_direction = None
        self.pending_since = None

    def reset(self):
        """Clear orientation measurements and warning timers."""
        self.__init__()

    @staticmethod
    def landmark_distance(landmarks, index_a, index_b):
        """Return 2D Euclidean distance between two pose landmarks."""
        a = landmarks[index_a]
        b = landmarks[index_b]
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def update(self, landmarks, now=None):
        """
        Analyze the current landmarks and update temporal warnings.

        Args:
            landmarks: MediaPipe pose landmarks or None.
            now: Monotonic timestamp in seconds (optional).

        Returns:
            Dictionary from get_results().
        """
        if now is None:
            now = time.monotonic()

        self.delta = None
        self.ear_nose_delta = None
        self.asymmetry = None
        self.warning = False
        self.direction = "UNKNOWN"
        self.isPayingAttention = None

        proposed = None

        if landmarks is None:
            self.state = "NO HEAD DATA"
        else:
            # 0 nose, 2/5 eyes, 3/6 outer eye corners, 7/8 ears.
            required_ids = (0, 2, 5, 7, 8, 6, 3)

            if (len(landmarks) <= max(required_ids)
                    or not all(
                        landmark_ok(landmarks[i], HEAD_MIN_VISIBILITY)
                        for i in required_ids
                    )):
                self.state = "LANDMARKS UNCLEAR"
            else:
                left_eye = landmarks[2]
                right_eye = landmarks[5]
                left_ear = landmarks[7]
                right_ear = landmarks[8]

                ear_width = float(np.hypot(
                    left_ear.x - right_ear.x,
                    left_ear.y - right_ear.y
                ))

                if ear_width < HEAD_MIN_EAR_SEPARATION:
                    self.state = "HEAD TURNED / UNCLEAR"
                else:
                    eye_y = (left_eye.y + right_eye.y) / 2.0
                    ear_y = (left_ear.y + right_ear.y) / 2.0

                    # Experimental vertical pitch proxy (DOWN).
                    self.delta = float((ear_y - eye_y) / ear_width)

                    # Retained as a diagnostic, but not used for UP.
                    self.ear_nose_delta = float(
                        (ear_y - landmarks[0].y) / ear_width
                    )

                    # Experimental horizontal yaw proxy (LEFT/RIGHT).
                    d86 = self.landmark_distance(landmarks, 8, 6)
                    d37 = self.landmark_distance(landmarks, 3, 7)
                    if d86 + d37 > 1e-6:
                        self.asymmetry = float(
                            (d86 - d37) / (d86 + d37)
                        )

                    # --------------------------------------------
                    # DIRECTION CLASSIFICATION
                    # --------------------------------------------
                    # Priority 1: DOWN. This previously came after
                    # LEFT/RIGHT, causing some downward tilts to be
                    # misclassified as sideways turns.
                    if self.delta < HEAD_DOWN_THRESHOLD:
                        proposed = "DOWN"

                    # Priority 2: LEFT / RIGHT.
                    elif (self.asymmetry is not None
                          and abs(self.asymmetry)
                          > SIDE_ASYMMETRY_THRESHOLD):
                        proposed = (
                            "LEFT" if self.asymmetry > 0 else "RIGHT"
                        )
                        if SWAP_LEFT_RIGHT:
                            proposed = (
                                "RIGHT" if proposed == "LEFT" else "LEFT"
                            )

                    # Priority 3: UP (TEMPORARILY DISABLED).
                    # Enable only after calibration; the previous
                    # threshold interfered with other orientations.
                    # elif (self.ear_nose_delta is not None
                    #       and self.ear_nose_delta <= HEAD_UP_THRESHOLD):
                    #     proposed = "UP"

                    # Otherwise treat valid landmarks as FORWARD.
                    else:
                        proposed = "FORWARD"

        # No reliable pose/orientation available.
        if proposed is None:
            self.pending_direction = None
            self.pending_since = None
            return self.get_results()

        # Forward orientation resets the warning timer.
        if proposed == "FORWARD":
            self.direction = "FORWARD"
            self.state = "HEAD FORWARD"
            self.isPayingAttention = True
            self.pending_direction = None
            self.pending_since = None
            return self.get_results()

        # Any non-forward direction is reported immediately,
        # but a warning requires a sustained orientation.
        self.direction = proposed
        self.isPayingAttention = False

        if proposed != self.pending_direction:
            self.pending_direction = proposed
            self.pending_since = now

        if proposed == "DOWN":
            delay = HEAD_DOWN_SECONDS
        elif proposed == "UP":
            delay = HEAD_UP_SECONDS
        else:
            delay = SIDE_TURN_SECONDS

        self.warning = (now - self.pending_since >= delay)
        self.state = (
            "LOOKING " + proposed if self.warning
            else "CHECKING " + proposed
        )
        return self.get_results()

    def get_results(self):
        """Return a structured snapshot of the current state."""
        return {
            "direction": self.direction,
            "isPayingAttention": self.isPayingAttention,
            "warning": self.warning,
            "state": self.state,
            "measurements": {
                "eye_ear_delta": self.delta,
                "ear_nose_delta": self.ear_nose_delta,
                "side_asymmetry": self.asymmetry,
            },
        }
