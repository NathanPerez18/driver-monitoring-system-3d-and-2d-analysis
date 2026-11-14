#!/usr/bin/env python3
"""Diagnostic apparent-pose-size check for DMS module test 4.

This uses *RGB image coordinates*, not physical dimensions or registered
camera depth. A small result is only a suspicion requiring corroboration.
Thresholds are provisional and must be calibrated with actual test subjects.
"""

from dataclasses import dataclass
import math

from .landmarks import landmark_ok


@dataclass(frozen=True)
class PoseSizeConfig:
    # Fraction of RGB image width for shoulder separation.
    min_shoulder_width: float = 0.10
    # Fraction of RGB image height for shoulder-midpoint to hip-midpoint.
    min_torso_height: float = 0.12
    min_visibility: float = 0.55


class PoseSizeAnalyzer:
    def __init__(self, config=None):
        self.config = config or PoseSizeConfig()

    def analyze(self, landmarks):
        """Return status NORMAL / TOO_SMALL / UNKNOWN and measured fractions.

        BOTH measurements must be available. Both small -> TOO_SMALL;
        both above thresholds -> NORMAL; disagreement -> UNKNOWN.
        These labels refer to apparent image size, NOT human liveness.
        """
        result = {
            "status": "UNKNOWN", "shoulder_width": None,
            "torso_height": None, "reason": "Insufficient landmarks",
        }
        if landmarks is None or len(landmarks) <= 24:
            return result

        indices = (11, 12, 23, 24)
        if not all(landmark_ok(landmarks[i], self.config.min_visibility) for i in indices):
            return result

        ls, rs, lh, rh = (landmarks[i] for i in indices)
        shoulder = math.hypot(ls.x - rs.x, ls.y - rs.y)
        shoulder_mid = ((ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0)
        hip_mid = ((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0)
        torso = math.hypot(shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
        if not (math.isfinite(shoulder) and math.isfinite(torso)):
            return result

        result["shoulder_width"] = shoulder
        result["torso_height"] = torso
        small_shoulder = shoulder < self.config.min_shoulder_width
        small_torso = torso < self.config.min_torso_height
        if small_shoulder and small_torso:
            result.update(status="TOO_SMALL", reason="Both apparent dimensions below provisional thresholds")
        elif not small_shoulder and not small_torso:
            result.update(status="NORMAL", reason="Both apparent dimensions meet provisional thresholds")
        else:
            result.update(status="UNKNOWN", reason="Shoulder and torso criteria disagree")
        return result
