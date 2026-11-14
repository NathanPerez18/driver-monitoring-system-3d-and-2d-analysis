#!/usr/bin/env python3
"""
Experimental motion-authenticity diagnostics for the DMS.

Consumes existing head_motion.py results, without another MediaPipe inference.
A periodic pattern is only MECHANICAL_LIKE evidence, not proof of a spoof.
Missing tracking -> INSUFFICIENT_DATA, never "fake".
All thresholds below are provisional and require controlled validation.
"""
from collections import deque
import math
import numpy as np

WINDOW_SECONDS = 30.0
MIN_SPAN_SECONDS = 12.0
MIN_SAMPLES = 25
MAX_GAP_SECONDS = 1.25
MIN_RANGE_DEGREES = 5.0
MIN_CYCLES = 3.0
MIN_PERIOD_SECONDS = 1.0
MAX_PERIOD_SECONDS = 5.0
MIN_PERIODIC_CORRELATION = 0.85
GRID_STEP_SECONDS = 0.10


class MotionAuthenticityAnalyzer:
    def __init__(self):
        self.samples = deque()

    def reset(self):
        self.samples.clear()

    def update(self, motion_results, now_seconds):
        """Return diagnostic state using the most recent head-angle samples.

        motion_results is the dictionary from HeadMotionTracker.update().
        Call on EVERY frame, including frames with no pose, to account for gaps.
        """
        now = float(now_seconds)
        if not math.isfinite(now):
            return self._result("INSUFFICIENT_DATA", "Invalid timestamp", 0)

        angle = None
        if isinstance(motion_results, dict):
            angle = motion_results.get("head_angle_degrees")
        if angle is not None:
            try:
                angle = float(angle)
                if not math.isfinite(angle):
                    angle = None
            except (TypeError, ValueError):
                angle = None

        if angle is not None:
            self.samples.append((now, angle))
        while self.samples and self.samples[0][0] < now - WINDOW_SECONDS:
            self.samples.popleft()

        n = len(self.samples)
        if n < MIN_SAMPLES:
            return self._result("INSUFFICIENT_DATA", "Too few valid landmarks", n)

        times, angles = np.asarray(self.samples, dtype=float).T
        span = times[-1] - times[0]
        gaps = np.diff(times)
        if span < MIN_SPAN_SECONDS or len(gaps) == 0:
            return self._result("INSUFFICIENT_DATA", "Observation window too short", n)
        largest_gap = float(np.max(gaps))
        if largest_gap > MAX_GAP_SECONDS:
            return self._result("INSUFFICIENT_DATA", "Tracking interruptions", n,
                                max_gap_s=round(largest_gap, 2))

        angle_range = float(np.ptp(angles))
        if angle_range < MIN_RANGE_DEGREES:
            return self._result("INSUFFICIENT_DATA", "Too little motion to characterize", n,
                                angle_range_deg=round(angle_range, 2),
                                max_gap_s=round(largest_gap, 2))

        # Uniform interpolation permits a basic normalized autocorrelation
        # despite slightly irregular camera sampling. Reject gaps above first.
        grid = np.arange(times[0], times[-1], GRID_STEP_SECONDS)
        signal = np.interp(grid, times, angles)
        signal = signal - float(np.mean(signal))
        std = float(np.std(signal))
        if std < 1.0:
            return self._result("INSUFFICIENT_DATA", "Insufficient motion variation", n)

        min_lag = round(MIN_PERIOD_SECONDS / GRID_STEP_SECONDS)
        max_lag = min(round(MAX_PERIOD_SECONDS / GRID_STEP_SECONDS), len(signal) // 3)
        scores = []
        for lag in range(min_lag, max_lag + 1):
            left, right = signal[:-lag], signal[lag:]
            if np.std(left) < 1e-6 or np.std(right) < 1e-6:
                continue
            corr = float(np.corrcoef(left, right)[0, 1])
            if math.isfinite(corr):
                scores.append((corr, lag))
        if not scores:
            return self._result("INSUFFICIENT_DATA", "Cannot estimate periodicity", n)

        # Choose strongest correlation; periodicity alone is not a liveness verdict.
        best_corr, best_lag = max(scores)
        period = best_lag * GRID_STEP_SECONDS
        cycles = span / period
        evidence = dict(angle_range_deg=round(angle_range, 2),
                        max_gap_s=round(largest_gap, 2),
                        periodicity=round(best_corr, 3),
                        period_s=round(period, 2),
                        estimated_cycles=round(cycles, 1))

        if best_corr >= MIN_PERIODIC_CORRELATION and cycles >= MIN_CYCLES:
            return self._result("MECHANICAL_LIKE",
                                "Strong repeated head-angle pattern (unverified)", n, **evidence)
        return self._result("HUMAN_LIKE",
                            "No strong periodic pattern (not proof of liveness)", n, **evidence)

    @staticmethod
    def _result(status, reason, sample_count, **kwargs):
        return {"status": status, "reason": reason,
                "sample_count": sample_count, **kwargs}
