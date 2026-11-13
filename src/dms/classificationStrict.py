#!/usr/bin/env python3
"""
File: classificationStrict.py
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
    OPTIONAL strict, fail-closed decision policy for the experimental DMS.
    It extends classification.py without changing that original module.

    Under this experimental policy, a driver is REJECTED whenever pose,
    attention, head-motion liveness, or depth evidence is missing, unknown,
    or explicitly fails. A positive spoofing-suspicion signal also rejects.

    The field "isFakeDriver" is a requested STRICT POLICY FLAG: True means
    "not verified / rejected by this rule", NOT that the person has been
    scientifically proven to be fake. Raw evidence remains available in
    the "evidence_*" fields, including UNKNOWN values.

    IMPORTANT: The current head_motion.py and depth_detection.py do not
    produce validated liveness results. They normally return None for
    isMovingAlive and is3D, so this strict policy will reject almost
    every frame until validated criteria are implemented. Do not use this
    flag as a ground-truth spoofing label or as an automotive safety system.

Input compatibility:
    evaluate(attention_results, head_motion_results, depth_results)
    Optional keyword inputs:
        pose_detected=True/False/None  (from main.py results["pose_detected"])
        depth_available=True/False/None (from actual depth frame acquisition)

Output compatibility:
    Preserves the existing top-level fields returned by classification.py
    while overriding isLive, isValidDriver, associated warnings/statuses,
    and adding strict-policy and original-evidence diagnostics.
"""

from .classification import (
    DriverClassifier as BaseDriverClassifier,
    classification_label,
    validate_flag,
)


class DriverClassifier(BaseDriverClassifier):
    """Fail-closed alternative to the normal experimental classifier."""

    def evaluate(
        self,
        attention_results=None,
        head_motion_results=None,
        depth_results=None,
        *,
        pose_detected=None,
        depth_available=None,
    ):
        """Classify missing, failing, or UNKNOWN evidence as strict rejection.

        The base classifier still computes the non-strict result first.
        This keeps real UNKNOWN evidence visible even when policy rejects.

        Args:
            attention_results: Dictionary from attention.py, or None.
            head_motion_results: Dictionary from head_motion.py, or None.
            depth_results: Dictionary from depth_detection.py, or None.
            pose_detected: Optional explicit pose-presence flag from main.py.
            depth_available: Optional explicit depth-frame-presence flag.

        Returns:
            Dictionary with strict policy decision and raw evidence state.
        """
        for value, name in (
            (attention_results, "attention_results"),
            (head_motion_results, "head_motion_results"),
            (depth_results, "depth_results"),
        ):
            if value is not None and not isinstance(value, dict):
                raise TypeError(f"{name} must be a dictionary or None.")

        attention = attention_results or {}
        depth = depth_results or {}

        pose_detected = validate_flag(pose_detected, "pose_detected")
        depth_available = validate_flag(depth_available, "depth_available")

        # Prefer explicit presence flags from main.py; otherwise infer only
        # when the producer supplies enough information to justify it.
        if pose_detected is None and "pose_detected" in attention:
            pose_detected = validate_flag(
                attention["pose_detected"], "attention.pose_detected"
            )
        if pose_detected is None:
            # An attention result of True/False implies a usable head pose.
            # UNKNOWN does not establish that a pose is present.
            attention_flag = validate_flag(
                attention.get("isPayingAttention"), "attention.isPayingAttention"
            )
            if attention_flag is not None:
                pose_detected = True
            elif attention.get("state") == "NO HEAD DATA":
                pose_detected = False

        if depth_available is None:
            for key in ("depth_available", "depth_frame_available"):
                if key in depth:
                    depth_available = validate_flag(
                        depth[key], f"depth.{key}"
                    )
                    break
        if depth_available is None:
            valid_percent = depth.get("valid_depth_percent")
            if valid_percent is not None:
                if isinstance(valid_percent, bool) or not isinstance(
                    valid_percent, (int, float)
                ):
                    raise TypeError(
                        "depth.valid_depth_percent must be a number or None"
                    )
                if not 0 <= valid_percent <= 100:
                    raise ValueError(
                        "depth.valid_depth_percent must be between 0 and 100"
                    )
                depth_available = valid_percent > 0
            elif validate_flag(depth.get("is3D"), "depth.is3D") is not None:
                # The depth criterion was explicitly assessed, so assume
                # a frame existed unless the caller explicitly says otherwise.
                depth_available = True

        raw = super().evaluate(
            attention_results=attention_results,
            head_motion_results=head_motion_results,
            depth_results=depth_results,
        )

        # Preserve the underlying research/evidence classification before
        # the policy turns UNKNOWN into rejection.
        evidence_is_live = raw["isLive"]
        evidence_is_valid_driver = raw["isValidDriver"]
        evidence_attention = raw["isPayingAttention"]
        evidence_motion = raw["isMovingAlive"]
        evidence_depth = raw["is3D"]
        evidence_reasons = list(raw["reasons"])

        rejection_reasons = []
        if pose_detected is not True:
            rejection_reasons.append(
                "No usable pose detected" if pose_detected is False
                else "Pose availability unknown"
            )
        if evidence_attention is not True:
            rejection_reasons.append(
                "Attention criterion failed" if evidence_attention is False
                else "Attention evidence inconclusive"
            )
        if evidence_motion is not True:
            rejection_reasons.append(
                "Head-motion liveness criterion failed" if evidence_motion is False
                else "Head-motion liveness evidence inconclusive"
            )
        if depth_available is not True:
            rejection_reasons.append(
                "Depth frame/data missing" if depth_available is False
                else "Depth availability unknown"
            )
        if evidence_depth is not True:
            rejection_reasons.append(
                "Depth-based 3D criterion failed" if evidence_depth is False
                else "Depth-based 3D evidence inconclusive"
            )
        if raw["spoofing_suspected"] is True:
            rejection_reasons.append("Potential presentation attack indicated")

        # Any failed OR inconclusive required condition yields REJECT.
        # A suspected attack may trigger a rejection but is not proof.
        strict_rejected = bool(rejection_reasons)
        strict_is_live = (
            pose_detected is True
            and evidence_motion is True
            and evidence_depth is True
            and depth_available is True
            and raw["spoofing_suspected"] is not True
        )
        strict_is_valid_driver = not strict_rejected

        raw.update({
            # Preserve original three-valued results for research/debugging.
            "evidence_reasons": evidence_reasons,
            "evidence_isLive": evidence_is_live,
            "evidence_isValidDriver": evidence_is_valid_driver,
            "evidence_liveness_status": classification_label(evidence_is_live),
            "evidence_driver_status": classification_label(
                evidence_is_valid_driver
            ),
            "pose_detected": pose_detected,
            "depth_available": depth_available,
            # Strict (binary) policy, not an empirical spoofing verdict.
            "isLive": strict_is_live,
            "isValidDriver": strict_is_valid_driver,
            "isFakeDriver": strict_rejected,
            "fake_driver_flag": strict_rejected,
            "strict_rejected": strict_rejected,
            "decision_policy": "STRICT_FAIL_CLOSED",
            "policy_outcome": (
                "REJECTED_AS_FAKE" if strict_rejected else "ACCEPTED"
            ),
            "liveness_status": classification_label(strict_is_live),
            "driver_status": classification_label(strict_is_valid_driver),
            "liveness_warning": not strict_is_live,
            "overall_warning": (
                raw["attention_warning"] or strict_rejected
            ),
            "rejection_reasons": rejection_reasons,
            "reasons": (
                rejection_reasons if rejection_reasons
                else ["All strict policy requirements satisfied"]
            ),
        })

        # The base class stores this same mutable results dictionary.
        self.last_results = raw
        return raw


# Optional explicit name for callers that prefer it.
StrictDriverClassifier = DriverClassifier
