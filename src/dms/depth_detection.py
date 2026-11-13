
#!/usr/bin/env python3
"""
File: depth_detection.py
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
    Depth-based planar surface and invalid-depth detection module
    for the edge-based Driver Monitoring System (DMS).

    This module processes depth measurements acquired from the
    Orbbec Astra Pro Plus camera to investigate geometric
    characteristics of objects within the camera's field of view.

    A Random Sample Consensus (RANSAC) algorithm is used to
    identify large planar surfaces in a three-dimensional
    point cloud. These surfaces may correspond to potential
    presentation attacks, including printed photographs.

    The module also identifies regions containing invalid
    depth measurements, which may occur when objects are too
    close to the camera, have difficult reflective properties,
    or cannot be measured reliably.

Responsibilities:
    1. Depth Frame Input
       - Accept depth measurements from main.py.
       - Validate the dimensions of incoming depth frames.
       - Identify valid and invalid depth measurements.

    2. Point Cloud Generation
       - Convert depth pixels into 3D coordinates.
       - Use approximate camera intrinsic parameters.
       - Filter measurements outside the detection range.

    3. Planar Surface Detection
       - Apply RANSAC plane fitting to 3D points.
       - Identify points belonging to planar surfaces.
       - Refine fitted planes using singular value
         decomposition (SVD).
       - Group planar points into connected regions.
       - Reject regions that fail size or occupancy criteria.

    4. Invalid-Depth Detection
       - Identify missing and invalid depth measurements.
       - Group invalid pixels into connected regions.
       - Filter small or irregular regions.
       - Return information about detected invalid regions.

    5. Depth Visualization
       - Convert depth measurements into a colour image.
       - Draw bounding boxes around detected planar regions.
       - Highlight invalid-depth pixels and regions.
       - Provide visualization functions for main.py.

    6. Performance Monitoring
       - Measure depth-analysis execution time.
       - Report the number of detected planar regions.
       - Report the percentage of valid depth measurements.

    7. Structured Results
       - Return planar surface detections.
       - Return invalid-depth detections.
       - Return diagnostic measurements.
       - Provide evidence for future liveness classification.

Design Principles:
    - No direct camera hardware access.
    - No MediaPipe model initialization.
    - No attention or breathing analysis.
    - No application-level keyboard handling.
    - No global classification flags.
    - Independent depth-processing interface.
    - Configurable detection thresholds.

Inputs:
    - Two-dimensional NumPy depth array.
    - Depth measurements in millimetres.

Outputs:
    - Detected planar regions.
    - Detected invalid-depth regions.
    - Valid depth percentage.
    - Depth-processing latency.
    - Experimental geometric indicators.

Note:
    This module does not directly establish human liveness.

    Flat surfaces and invalid-depth regions are potential
    spoofing indicators, not definitive evidence of an attack.

    Camera intrinsic parameters are approximate and should
    be replaced with calibrated values for precise geometry.

    This is an experimental research prototype.
"""


# ============================================================
# IMPORTS
# ============================================================

import time

import cv2
import numpy as np


# ============================================================
# DEPTH DETECTION CONFIGURATION
# ============================================================

# Depth detection range in millimetres.
MIN_DEPTH_MM = 300
MAX_DEPTH_MM = 700

# Maximum number of RANSAC planes.
MAX_PLANES = 1

# RANSAC fitting parameters.
RANSAC_ITERATIONS = 60
PLANE_TOLERANCE_MM = 15.0

# Minimum number of points required for plane fitting.
MIN_PLANE_PIXELS = 3000

# Minimum connected region size.
MIN_REGION_PIXELS = 3000

# Minimum ratio of detected pixels to bounding-box area.
MIN_OCCUPANCY = 0.45

# Number of randomly sampled points used during RANSAC.
SAMPLE_POINTS = 1500


# ============================================================
# INVALID-DEPTH CONFIGURATION
# ============================================================

MIN_INVALID_REGION_PIXELS = 1500
MIN_INVALID_OCCUPANCY = 0.70
MAX_INVALID_REGIONS = 5

# Invalid-depth overlay settings.
INVALID_OVERLAY_ALPHA = 0.65
INVALID_COLOR = (180, 0, 255)


# ============================================================
# CAMERA INTRINSIC PARAMETERS
# ============================================================

# Approximate focal lengths in pixels.
# Preserved from the original viewer_unified.py.

FX = 580.0
FY = 580.0


# ============================================================
# VISUALIZATION CONFIGURATION
# ============================================================

# OpenCV colours use BGR.
PLANE_COLORS = [
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255)
]

MORPH_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)


# ============================================================
# GENERAL DRAWING
# ============================================================

def put_label(
    image,
    message,
    x,
    y,
    color=(255, 255, 255),
    scale=0.56
):
    """
    Draw a text label on an OpenCV image.
    """

    cv2.putText(
        image,
        message,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA
    )


# ============================================================
# DEPTH VISUALIZATION
# ============================================================

def colorize_depth(depth):
    """
    Convert a depth frame into a colour visualization.

    Args:
        depth:
            Two-dimensional depth array in millimetres.

    Returns:
        OpenCV BGR image with a JET colour map.

    Notes:
        Invalid depth pixels are displayed as black.
        This visualization does not modify source depth data.
    """

    if depth is None:
        return None

    valid = (
        np.isfinite(depth)
        & (depth > 0)
    )

    normalized = np.clip(
        (depth.astype(np.float32) - 100) / 700 * 255,
        0,
        255
    ).astype(np.uint8)

    image = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    image[~valid] = 0

    return image


# ============================================================
# INVALID-DEPTH REGION DETECTION
# ============================================================

def detect_invalid_depth(depth):
    """
    Identify significant invalid-depth regions.

    Args:
        depth:
            Two-dimensional depth array.

    Returns:
        List of dictionaries containing:
            box:
                (x, y, width, height)

            area:
                Number of connected invalid pixels.

            occupancy:
                Invalid-pixel ratio within bounding box.

    Notes:
        Invalid pixels are non-finite or <= 0.

        Invalid measurements can have several causes and
        should not automatically be classified as spoofing.
    """

    if depth is None:
        return []

    invalid = (
        ~np.isfinite(depth)
        | (depth <= 0)
    )

    mask = invalid.astype(np.uint8) * 255

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8)
    )

    count, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            cleaned,
            connectivity=8
        )
    )

    regions = []

    for index in range(1, count):

        x, y, width, height, area = stats[index]

        if area < MIN_INVALID_REGION_PIXELS:
            continue

        if width * height == 0:
            continue

        occupancy = (
            area / float(width * height)
        )

        if occupancy < MIN_INVALID_OCCUPANCY:
            continue

        regions.append({
            "box": (
                int(x),
                int(y),
                int(width),
                int(height)
            ),

            "area": int(area),

            "occupancy": float(occupancy)
        })

    # Sort by region area, largest first.
    regions.sort(
        key=lambda region: region["area"],
        reverse=True
    )

    return regions[:MAX_INVALID_REGIONS]


# ============================================================
# INVALID-DEPTH VISUALIZATION
# ============================================================

def draw_invalid_depth(display, depth, regions=None):
    """
    Highlight invalid-depth measurements.

    Args:
        display:
            OpenCV BGR image to annotate.

        depth:
            Original depth array.

        regions:
            Optional previously detected invalid regions.

    Returns:
        Annotated display image.

    Notes:
        The supplied display image is modified in place.
        The original depth measurements are not modified.
    """

    if display is None or depth is None:
        return display

    if regions is None:
        regions = detect_invalid_depth(depth)

    invalid = (
        ~np.isfinite(depth)
        | (depth <= 0)
    )

    # Create a coloured overlay.
    tint = np.empty_like(display)
    tint[:] = INVALID_COLOR

    cv2.addWeighted(
        display,
        1.0 - INVALID_OVERLAY_ALPHA,
        tint,
        INVALID_OVERLAY_ALPHA,
        0,
        dst=tint
    )

    display[invalid] = tint[invalid]

    # Draw bounding boxes around invalid regions.
    for region in regions:

        x, y, width, height = region["box"]

        area = region["area"]
        occupancy = region["occupancy"]

        cv2.rectangle(
            display,
            (x, y),
            (x + width - 1, y + height - 1),
            (255, 255, 255),
            2
        )

        put_label(
            display,
            f"No depth: {area} px "
            f"{occupancy * 100:.0f}%",
            x,
            min(
                display.shape[0] - 8,
                y + height + 16
            ),
            scale=0.43
        )

    return display


# ============================================================
# POINT CLOUD GENERATION
# ============================================================

def make_point_cloud(
    depth,
    min_depth,
    max_depth
):
    """
    Convert depth pixels into 3D point-cloud coordinates.

    Args:
        depth:
            Two-dimensional depth array in millimetres.

        min_depth:
            Minimum included depth.

        max_depth:
            Maximum included depth.

    Returns:
        points:
            Nx3 array containing X, Y, Z coordinates.

        pixels:
            Nx2 array containing original row/column positions.

    Notes:
        The calculation uses approximate focal lengths and
        assumes the principal point is at the image centre.
    """

    height, width = depth.shape

    yy, xx = np.indices(
        (height, width)
    )

    # Select valid points within detection range.
    valid = (
        np.isfinite(depth)
        & (depth >= min_depth)
        & (depth <= max_depth)
    )

    z = depth[valid].astype(np.float32)

    u = xx[valid].astype(np.float32)
    v = yy[valid].astype(np.float32)

    # Convert pixels into approximate camera-space points.
    X = (
        (u - (width - 1) / 2)
        * z / FX
    )

    Y = (
        (v - (height - 1) / 2)
        * z / FY
    )

    points = np.column_stack((
        X,
        Y,
        z
    ))

    pixels = np.column_stack((
        v.astype(np.int32),
        u.astype(np.int32)
    ))

    return points, pixels


# ============================================================
# RANSAC PLANE FITTING
# ============================================================

def find_plane(points, rng):
    """
    Identify a dominant planar surface using RANSAC.

    Args:
        points:
            Nx3 array of point-cloud coordinates.

        rng:
            NumPy random number generator.

    Returns:
        Boolean inlier mask identifying points belonging
        to the fitted plane, or None if no plane is found.

    Algorithm:
        1. Randomly sample a subset of points.
        2. Fit candidate planes from three sampled points.
        3. Count points within the plane tolerance.
        4. Select the candidate with the most inliers.
        5. Refine its parameters using SVD.
        6. Identify inliers in the complete point cloud.
    """

    if len(points) < MIN_PLANE_PIXELS:
        return None

    sample_count = min(
        len(points),
        SAMPLE_POINTS
    )

    sample = points[
        rng.choice(
            len(points),
            sample_count,
            replace=False
        )
    ]

    best_normal = None
    best_offset = None
    best_count = 0

    # --------------------------------------------------------
    # RANDOM SAMPLE CONSENSUS
    # --------------------------------------------------------

    for _ in range(RANSAC_ITERATIONS):

        p0, p1, p2 = sample[
            rng.choice(
                sample_count,
                3,
                replace=False
            )
        ]

        # Plane normal from two vectors.
        normal = np.cross(
            p1 - p0,
            p2 - p0
        )

        norm = np.linalg.norm(normal)

        # Skip degenerate point combinations.
        if norm < 1e-6:
            continue

        normal /= norm

        offset = -np.dot(
            normal,
            p0
        )

        # Calculate point-to-plane distances.
        residuals = np.abs(
            sample @ normal + offset
        )

        count = np.count_nonzero(
            residuals < PLANE_TOLERANCE_MM
        )

        if count > best_count:

            best_normal = normal
            best_offset = offset
            best_count = count

    if best_normal is None:
        return None

    # --------------------------------------------------------
    # REFINE PLANE USING SVD
    # --------------------------------------------------------

    selected = sample[
        np.abs(
            sample @ best_normal + best_offset
        ) < PLANE_TOLERANCE_MM
    ]

    if len(selected) < 3:
        return None

    center = selected.mean(axis=0)

    _, _, vh = np.linalg.svd(
        selected - center,
        full_matrices=False
    )

    normal = vh[-1]

    normal /= np.linalg.norm(normal)

    offset = -np.dot(
        normal,
        center
    )

    # --------------------------------------------------------
    # IDENTIFY FULL POINT-CLOUD INLIERS
    # --------------------------------------------------------

    inliers = (
        np.abs(
            points @ normal + offset
        ) < PLANE_TOLERANCE_MM
    )

    if np.count_nonzero(inliers) < MIN_PLANE_PIXELS:
        return None

    return inliers


# ============================================================
# PLANAR SURFACE DETECTION
# ============================================================

def detect_planes(
    depth,
    rng,
    min_depth,
    max_depth
):
    """
    Identify planar regions in a depth frame.

    Args:
        depth:
            Two-dimensional depth array.

        rng:
            NumPy random number generator.

        min_depth:
            Minimum detection distance.

        max_depth:
            Maximum detection distance.

    Returns:
        List of planar region dictionaries containing:
            box:
                (x, y, width, height)

            occupancy:
                Ratio of planar pixels within bounding box.

            depth_mm:
                Median depth of the region.

            plane:
                Plane number.

    Notes:
        The fitting and region filtering are preserved
        from the original unified viewer.
    """

    height, width = depth.shape

    points, pixels = make_point_cloud(
        depth,
        min_depth,
        max_depth
    )

    detections = []

    for plane_id in range(MAX_PLANES):

        if len(points) < MIN_PLANE_PIXELS:
            break

        inliers = find_plane(
            points,
            rng
        )

        if inliers is None:
            break

        # ----------------------------------------------------
        # CONSTRUCT PLANE MASK
        # ----------------------------------------------------

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        plane_pixels = pixels[inliers]

        mask[
            plane_pixels[:, 0],
            plane_pixels[:, 1]
        ] = 255

        # Join nearby planar pixels.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            MORPH_KERNEL
        )

        # ----------------------------------------------------
        # CONNECTED COMPONENT ANALYSIS
        # ----------------------------------------------------

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        for label in range(1, count):

            x, y, w, h, area = stats[label]

            if (
                area < MIN_REGION_PIXELS
                or w * h == 0
            ):
                continue

            occupancy = area / float(w * h)

            if occupancy < MIN_OCCUPANCY:
                continue

            # Select original depth measurements
            # belonging to the detected region.
            valid_region = (
                (labels == label)
                & np.isfinite(depth)
                & (depth >= min_depth)
                & (depth <= max_depth)
            )

            region_depths = depth[valid_region]

            if region_depths.size:

                detections.append({
                    "box": (
                        int(x),
                        int(y),
                        int(w),
                        int(h)
                    ),

                    "occupancy": float(occupancy),

                    "depth_mm": float(
                        np.median(region_depths)
                    ),

                    "plane": plane_id + 1
                })

        # Remove detected plane points before searching
        # for another plane.
        points = points[~inliers]
        pixels = pixels[~inliers]

    return detections


# ============================================================
# PLANE VISUALIZATION
# ============================================================

def draw_planes(display, detections):
    """
    Draw bounding boxes around detected planar regions.

    Args:
        display:
            OpenCV BGR image.

        detections:
            List returned by detect_planes().

    Returns:
        Annotated display image.
    """

    if display is None:
        return None

    for detection in detections:

        x, y, w, h = detection["box"]

        color = PLANE_COLORS[
            (detection["plane"] - 1)
            % len(PLANE_COLORS)
        ]

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        put_label(
            display,
            f"P{detection['plane']} "
            f"{detection['depth_mm']:.0f}mm "
            f"{detection['occupancy'] * 100:.0f}%",
            x,
            max(18, y - 8),
            color,
            0.45
        )

    return display


# ============================================================
# DEPTH DETECTOR CLASS
# ============================================================

class DepthDetector:
    """
    Coordinates depth-based planar and invalid-region analysis.

    The class provides a single interface for main.py.

    It maintains:
        - Detection range settings.
        - Random number generator.
        - Latest processing results.
        - Processing latency measurements.

    The class does not acquire frames directly from the camera.
    """

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    def __init__(
        self,
        min_depth=MIN_DEPTH_MM,
        max_depth=MAX_DEPTH_MM
    ):

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.rng = np.random.default_rng()

        self.last_analysis_ms = 0.0

        self.last_results = None

    # --------------------------------------------------------
    # SET DETECTION RANGE
    # --------------------------------------------------------

    def set_max_depth(self, max_depth):
        """
        Update the maximum plane detection distance.

        Preserves the adjustable detection range from
        viewer_unified.py.
        """

        self.max_depth = max(
            self.min_depth + 50,
            int(max_depth)
        )

    # --------------------------------------------------------
    # ANALYZE DEPTH FRAME
    # --------------------------------------------------------

    def analyze(
        self,
        depth,
        detect_planes_enabled=True,
        detect_invalid_enabled=True
    ):
        """
        Process one depth frame.

        Args:
            depth:
                Two-dimensional depth array.

            detect_planes_enabled:
                Whether RANSAC planar detection is enabled.

            detect_invalid_enabled:
                Whether invalid-depth region detection is enabled.

        Returns:
            Dictionary containing depth-analysis results.

        Notes:
            Planar and invalid-depth analysis can be
            enabled or disabled independently.

            A missing frame returns an UNKNOWN result.
        """

        if depth is None:

            self.last_analysis_ms = 0.0

            self.last_results = {
                "planes": [],
                "invalid_regions": [],
                "plane_count": 0,
                "invalid_region_count": 0,
                "valid_depth_percent": None,
                "analysis_ms": 0.0,
                "is3D": None,
                "state": "NO DEPTH DATA"
            }

            return self.last_results

        # Validate expected depth shape.
        if depth.ndim != 2:

            raise ValueError(
                "DepthDetector requires a 2D depth array."
            )

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # VALID DEPTH PERCENTAGE
        # ----------------------------------------------------

        valid = (
            np.isfinite(depth)
            & (depth > 0)
        )

        valid_percent = float(
            100.0 * np.mean(valid)
        )

        # ----------------------------------------------------
        # PLANAR SURFACE DETECTION
        # ----------------------------------------------------

        planes = []

        if detect_planes_enabled:

            planes = detect_planes(
                depth,
                self.rng,
                self.min_depth,
                self.max_depth
            )

        # ----------------------------------------------------
        # INVALID-DEPTH DETECTION
        # ----------------------------------------------------

        invalid_regions = []

        if detect_invalid_enabled:

            invalid_regions = detect_invalid_depth(
                depth
            )

        # ----------------------------------------------------
        # PERFORMANCE MEASUREMENT
        # ----------------------------------------------------

        self.last_analysis_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        # ----------------------------------------------------
        # STRUCTURED RESULTS
        # ----------------------------------------------------

        self.last_results = {
            "planes": planes,

            "invalid_regions": invalid_regions,

            "plane_count": len(planes),

            "invalid_region_count": len(invalid_regions),

            "valid_depth_percent": valid_percent,

            "analysis_ms": self.last_analysis_ms,

            # Reserved for future validated classification.
            "is3D": None,

            "state": "ANALYSIS COMPLETE"
        }

        return self.last_results

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    def draw(
        self,
        depth,
        results=None,
        show_planes=True,
        show_invalid=False
    ):
        """
        Create a depth visualization with optional overlays.

        Args:
            depth:
                Original depth frame.

            results:
                Structured results from analyze().

            show_planes:
                Display planar regions.

            show_invalid:
                Display invalid-depth overlays.

        Returns:
            OpenCV BGR depth visualization.
        """

        if depth is None:
            return None

        if results is None:
            results = self.last_results

        display = colorize_depth(depth)

        if results is None:
            return display

        if show_invalid:

            draw_invalid_depth(
                display,
                depth,
                results["invalid_regions"]
            )

        if show_planes:

            draw_planes(
                display,
                results["planes"]
            )

        return display

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):
        """
        Reset stored results and latency measurements.
        """

        self.last_results = None
        self.last_analysis_ms = 0.0
