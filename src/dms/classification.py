#!/usr/bin/env python3
"""
File: classification.py
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
    Central, hardware-independent decision module for the experimental DMS.
    Combines attention.py, head_motion.py and depth_detection.py outputs.

    IMPORTANT: Motion measurements, planar regions, and missing depth are
    observations, not proof of life or proof of spoofing. Until separate
    liveness criteria have been experimentally validated, the motion and
    depth classification inputs must remain None (UNKNOWN).

Responsibilities:
    1. Validate three-valued classification flags (True / False / None).
    2. Combine independently classified motion and depth evidence.
    3. Combine liveness with attention as a separate requirement.
    4. Keep suspected presentation attacks separate from liveness failure.
    5. Forward useful diagnostics without choosing empirical thresholds.
    6. Report attention warnings without creating additional timers.

Inputs:
    - attention_results: output from HeadOrientationTracker.update().
    - head_motion_results: output from HeadMotionTracker.update().
    - depth_results: output from DepthDetector.analyze().

Expected classification keys:
    attention_results["isPayingAttention"]: True / False / None
    head_motion_results["isMovingAlive"]: True / False / None
        (or future "motion_liveness_result")
    depth_results["is3D"]: True / False / None

Optional, *unvalidated* suspicion indicators (True / False / None):
    head_motion_results["damped_oscillation_detected"]
    head_motion_results["spoofing_suspected"]
    depth_results["flat_surface_suspected"]
    depth_results["spoofing_suspected"]

    Plane count alone is NOT an attack indicator: legitimate scenes contain
    flat surfaces. Unknown indicators never mean False/negative evidence.

Safety / interpretation:
    This is a research prototype, not a verified anti-spoofing system.
    UNKNOWN must not become FAIL simply because data is unavailable.
    Motion/depth liveness flags must not be set based on unvalidated tests.
    "isValidDriver" names the combined experimental criterion, NOT an
    identity verification or a judgment that driving is safe.
"""


# ============================================================
# CLASSIFICATION STATES
# ============================================================

TRUE = True
FALSE = False
UNKNOWN = None


# ============================================================
# THREE-VALUED LOGIC
# ============================================================

def logical_and(*conditions):
    """False if any False; True if all True; otherwise UNKNOWN."""
    if not conditions:
        return UNKNOWN
    if any(condition is False for condition in conditions):
        return FALSE
    if all(condition is True for condition in conditions):
        return TRUE
    return UNKNOWN


def logical_or(*conditions):
    """True if any True; False if all False; otherwise UNKNOWN."""
    if not conditions:
        return UNKNOWN
    if any(condition is True for condition in conditions):
        return TRUE
    if all(condition is False for condition in conditions):
        return FALSE
    return UNKNOWN


# ============================================================
# FLAG VALIDATION AND FORMATTING
# ============================================================

def validate_flag(value, name):
    """Accept only Python bool or None; reject numbers and strings."""
    if value is not None and type(value) is not bool:
        raise TypeError(
            f"{name} must be True, False, or None. Received: {value!r}"
        )
    return value


def classification_label(value):
    """Provide a human-readable three-state label."""
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "UNKNOWN"


def _results_dict(value, name):
    """Normalize a missing module result, but reject malformed results."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dictionary or None.")
    return value


def _optional_flag(results, key, source):
    """Return optional flag or UNKNOWN if its producer did not supply it."""
    return validate_flag(results.get(key), f"{source}.{key}")


# ============================================================
# DRIVER CLASSIFIER
# ============================================================

class DriverClassifier:
    """Combines processed results without accessing cameras or running CV."""

    def __init__(self):
        self.last_results = None
        self.classification_count = 0

    def evaluate(
        self,
        attention_results=None,
        head_motion_results=None,
        depth_results=None,
    ):
        """
        Evaluate a single snapshot from the three modules.

        Missing inputs and missing classification keys yield UNKNOWN.
        The caller must only provide definitive motion/depth classification
        flags after the corresponding algorithms have been validated.

        Spoofing suspicion is intentionally NOT included in isLive or
        isValidDriver. It is reported as a separate research signal.
        """
        attention_results = _results_dict(attention_results, "attention_results")
        head_motion_results = _results_dict(head_motion_results, "head_motion_results")
        depth_results = _results_dict(depth_results, "depth_results")

        # ----------------------------------------------------
        # EXTRACT CLASSIFICATION FLAGS
        # ----------------------------------------------------
        is_paying_attention = _optional_flag(
            attention_results, "isPayingAttention", "attention"
        )

        # Accept the current head_motion.py key, and allow an explicit,
        # more descriptive key when a validated classifier is added later.
        # Reject conflicting values rather than silently choosing one.
        legacy_motion = _optional_flag(
            head_motion_results, "isMovingAlive", "head_motion"
        )
        named_motion = _optional_flag(
            head_motion_results, "motion_liveness_result", "head_motion"
        )
        if (
            legacy_motion is not None
            and named_motion is not None
            and legacy_motion is not named_motion
        ):
            raise ValueError(
                "head_motion.isMovingAlive and motion_liveness_result conflict"
            )
        is_moving_alive = (
            named_motion if named_motion is not None else legacy_motion
        )
        is_3d = _optional_flag(depth_results, "is3D", "depth")

        # ----------------------------------------------------
        # ATTENTION WARNINGS
        # ----------------------------------------------------
        # The attention module already implements persistence/timing.
        attention_warning_flag = validate_flag(
            attention_results.get("warning", False), "attention.warning"
        )
        attention_warning = (
            attention_warning_flag is True and is_paying_attention is False
        )

        # ----------------------------------------------------
        # EXPERIMENTAL LIVENESS AND COMBINED CRITERIA
        # ----------------------------------------------------
        is_live = logical_and(is_moving_alive, is_3d)
        is_valid_driver = logical_and(is_live, is_paying_attention)

        # A genuine failed liveness *criterion* requires an explicit False.
        # It is not inferred from zero motion, no pose, or missing depth.
        liveness_warning = (is_live is False)
        overall_warning = attention_warning or liveness_warning

        # ----------------------------------------------------
        # SEPARATE, PRELIMINARY SPOOFING INDICATORS
        # ----------------------------------------------------
        damped_oscillation = _optional_flag(
            head_motion_results,
            "damped_oscillation_detected",
            "head_motion",
        )
        head_spoofing = _optional_flag(
            head_motion_results, "spoofing_suspected", "head_motion"
        )
        flat_surface_suspected = _optional_flag(
            depth_results, "flat_surface_suspected", "depth"
        )
        depth_spoofing = _optional_flag(
            depth_results, "spoofing_suspected", "depth"
        )

        spoofing_suspected = logical_or(
            damped_oscillation,
            head_spoofing,
            flat_surface_suspected,
            depth_spoofing,
        )
        # A suspected attack is informational until independently validated.
        suspicion_indicator = (spoofing_suspected is True)

        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------
        reasons = []
        if is_paying_attention is False:
            reasons.append(
                "Sustained non-forward head orientation"
                if attention_warning
                else "Non-forward head orientation detected"
            )
        elif is_paying_attention is None:
            reasons.append("Attention classification unavailable")

        if is_moving_alive is False:
            reasons.append("Head-motion liveness criterion failed")
        elif is_moving_alive is None:
            reasons.append("Head-motion liveness classification unavailable")

        if is_3d is False:
            reasons.append("Depth-based 3D criterion failed")
        elif is_3d is None:
            reasons.append("Depth-based 3D classification unavailable")

        if spoofing_suspected is True:
            reasons.append(
                "Potential presentation-attack indicator observed "
                "(not a verified spoofing classification)"
            )
        if not reasons:
            reasons.append("All classification conditions satisfied")

        # Pass through existing observation keys, without inventing values.
        head_diagnostics = {
            key: head_motion_results.get(key)
            for key in (
                "head_angle_degrees",
                "smoothed_angle_degrees",
                "angular_velocity_deg_s",
                "sample_count",
                "state",
            )
        }
        depth_diagnostics = {
            key: depth_results.get(key)
            for key in (
                "plane_count",
                "invalid_region_count",
                "valid_depth_percent",
                "analysis_ms",
            )
        }

        # ----------------------------------------------------
        # STRUCTURED RESULTS (KEEP EXISTING TOP-LEVEL KEYS)
        # ----------------------------------------------------
        self.last_results = {
            "isPayingAttention": is_paying_attention,
            "isMovingAlive": is_moving_alive,
            "is3D": is_3d,
            "isLive": is_live,
            "isValidDriver": is_valid_driver,
            "attention_warning": attention_warning,
            "liveness_warning": liveness_warning,
            "overall_warning": overall_warning,
            "attention_status": classification_label(is_paying_attention),
            "liveness_status": classification_label(is_live),
            "driver_status": classification_label(is_valid_driver),
            "reasons": reasons,
            # Additional distinct observations, not proof of non-liveness.
            "spoofing_suspected": spoofing_suspected,
            "spoofing_indicator": suspicion_indicator,
            "spoofing_status": (
                "SUSPECTED" if spoofing_suspected is True
                else "NOT INDICATED" if spoofing_suspected is False
                else "UNKNOWN"
            ),
            "attention": {
                "direction": attention_results.get("direction"),
                "state": attention_results.get("state"),
                "warning": attention_warning,
            },
            "head_motion": {
                **head_diagnostics,
                "motion_liveness_result": is_moving_alive,
                "damped_oscillation_detected": damped_oscillation,
                "spoofing_suspected": head_spoofing,
            },
            "depth": {
                **depth_diagnostics,
                "is3D": is_3d,
                "flat_surface_suspected": flat_surface_suspected,
                "spoofing_suspected": depth_spoofing,
            },
        }

        self.classification_count += 1
        return self.last_results

    def get_results(self):
        """Return the most recent classification results, or None."""
        return self.last_results

    def reset(self):
        """Reset the stored decision and classification counter."""
        self.last_results = None
        self.classification_count = 0
