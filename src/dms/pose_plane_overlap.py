#!/usr/bin/env python3
"""
File: pose_plane_overlap.py
Project: Engineering Edge-Based Computer Vision for Human Liveness
         Detection in Automotive Driver Monitoring Systems (DMS)
Course: ENGR5510 - Foundations of Software Engineering
Institution: Ontario Tech University
Group: 2
Authors: Nathan Perez, Jacky Zhong, Rivka Sagi
Date: November 13, 2026

Description:
    Diagnostic-only comparison between the RGB MediaPipe upper-body pose
    bounding region and RANSAC planar-region bounding boxes in the depth
    image. It does NOT establish that a driver is genuine or spoofed.

    IMPORTANT: The Orbbec RGB and depth streams are NOT assumed spatially
    registered. With rgb_depth_aligned=False, rectangles are only scaled
    between image dimensions; lens offset, disparity and distortion are
    not corrected. Any apparent overlap is a PROVISIONAL VISUAL ESTIMATE.
    Even with alignment, bounding-box intersection can overestimate actual
    plane/pixel intersection. Overlap does not change liveness flags.

Inputs:
    landmarks: MediaPipe normalized 2D pose landmarks (33 items), or None
    depth_results: DepthDetector.analyze() result dictionary, or None
    rgb_shape: shape of RGB image, e.g. (480, 640, 3)
    depth_shape: shape of depth image, e.g. (480, 640)

Returns:
    Structured diagnostic: pose_box_rgb, projected plane boxes, maximum
    pose-area box coverage (0-100%), overlap candidate, and alignment flag.
"""

import math

import cv2

# MediaPipe pose indexes for the face and upper torso.
UPPER_BODY_LANDMARKS = tuple(range(11)) + (11, 12)
MIN_VISIBILITY = 0.45
# Width and height margins relative to the detected region's dimensions.
ROI_MARGIN_FRACTION = 0.15



def _valid_landmark(landmark):
    """Check visibility and finite normalized coordinates."""
    if landmark is None:
        return False
    try:
        x, y = float(landmark.x), float(landmark.y)
        visibility = float(getattr(landmark, "visibility", 1.0))
        presence = float(getattr(landmark, "presence", 1.0))
    except (ValueError, TypeError, AttributeError):
        return False
    return (math.isfinite(x) and math.isfinite(y)
            and visibility >= MIN_VISIBILITY and presence >= MIN_VISIBILITY
            and -0.1 <= x <= 1.1 and -0.1 <= y <= 1.1)


def _image_size(shape):
    if shape is None or len(shape) < 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    return (width, height) if width > 0 and height > 0 else None


def pose_box_from_landmarks(landmarks, image_shape):
    """Return (x, y, w, h) around visible face/shoulders, or None."""
    dims = _image_size(image_shape)
    if dims is None or landmarks is None:
        return None
    width, height = dims
    # Some model APIs expose a list containing one per-person landmark list.
    if (isinstance(landmarks, (list, tuple)) and landmarks
            and isinstance(landmarks[0], (list, tuple))):
        landmarks = landmarks[0]
    try:
        points = [
            (float(landmarks[i].x) * width, float(landmarks[i].y) * height)
            for i in UPPER_BODY_LANDMARKS
            if i < len(landmarks) and _valid_landmark(landmarks[i])
        ]
    except (TypeError, AttributeError, IndexError):
        return None
    if len(points) < 3:
        return None
    xs, ys = zip(*points)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # Margins avoid an unnaturally tight box around sparse landmarks.
    dx = max(8.0, (xmax - xmin) * ROI_MARGIN_FRACTION)
    dy = max(8.0, (ymax - ymin) * ROI_MARGIN_FRACTION)
    x1, y1 = max(0, int(math.floor(xmin - dx))), max(0, int(math.floor(ymin - dy)))
    x2, y2 = min(width, int(math.ceil(xmax + dx))), min(height, int(math.ceil(ymax + dy)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def _parse_plane_box(plane):
    """Support DepthDetector's (x,y,w,h) `box` convention safely."""
    if not isinstance(plane, dict):
        return None
    box = plane.get("box", plane.get("bbox", plane.get("rect")))
    mode = "xywh"
    if box is None and "xyxy" in plane:
        box, mode = plane["xyxy"], "xyxy"
    if box is None or len(box) != 4:
        return None
    try:
        x, y, c, d = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, c, d)):
        return None
    if mode == "xyxy":
        c, d = c - x, d - y
    return (x, y, c, d) if c > 0 and d > 0 else None


def _scale_box(box, src_size, target_size):
    if box is None or src_size is None or target_size is None:
        return None
    sx = target_size[0] / src_size[0]
    sy = target_size[1] / src_size[1]
    x, y, w, h = box
    x1 = max(0, min(target_size[0], int(round(x * sx))))
    y1 = max(0, min(target_size[1], int(round(y * sy))))
    x2 = max(0, min(target_size[0], int(round((x + w) * sx))))
    y2 = max(0, min(target_size[1], int(round((y + h) * sy))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def intersection_area(a, b):
    """Area in pixels where two (x,y,w,h) rectangles intersect."""
    if a is None or b is None:
        return 0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def analyze_pose_plane_overlap(landmarks, depth_results, rgb_shape, depth_shape,
                               rgb_depth_aligned=False):
    """Compare boxes. Does not return a liveness/spoofing verdict."""
    pose_box = pose_box_from_landmarks(landmarks, rgb_shape)
    output = {
        "status": "UNKNOWN",
        "pose_detected": landmarks is not None,
        "pose_box_rgb": pose_box,
        "plane_boxes_rgb_approx": [],
        "plane_count_considered": 0,
        "best_overlap_percent": None,
        "approx_overlap_detected": None,
        # Calibrated evidence is purposely unavailable unless verified.
        "registered_overlap_detected": None,
        "rgb_depth_aligned": bool(rgb_depth_aligned),
        "isSpoofing": None,
    }
    if landmarks is None or pose_box is None:
        output["status"] = "NO_POSE"
        return output
    if depth_results is None or _image_size(depth_shape) is None:
        output["status"] = "NO_DEPTH"
        return output
    planes = depth_results.get("planes")
    if planes is None:
        output["status"] = "NO_PLANE_BOXES"
        return output
    if not isinstance(planes, (list, tuple)):
        output["status"] = "UNSUPPORTED_PLANES"
        return output
    rgb_size, depth_size = _image_size(rgb_shape), _image_size(depth_shape)
    boxes = []
    for plane in planes:
        projected = _scale_box(_parse_plane_box(plane), depth_size, rgb_size)
        if projected is not None:
            x, y, w, h = projected
            projected = (rgb_size[0] - (x + w), y, w, h)
        if projected is not None:
            boxes.append(projected)
    output["plane_boxes_rgb_approx"] = boxes
    output["plane_count_considered"] = len(boxes)
    if len(planes) > 0 and not boxes:
        output["status"] = "UNSUPPORTED_PLANE_BOXES"
        return output
    if not boxes:
        output["status"] = "NO_PLANES"
        output["best_overlap_percent"] = 0.0
        output["approx_overlap_detected"] = False
        if rgb_depth_aligned:
            output["registered_overlap_detected"] = False
        return output
    pose_area = pose_box[2] * pose_box[3]
    percent = 100.0 * max(intersection_area(pose_box, box) for box in boxes) / pose_area
    output["best_overlap_percent"] = round(percent, 2)
    output["approx_overlap_detected"] = percent > 0.0
    if rgb_depth_aligned:
        output["registered_overlap_detected"] = percent > 0.0
        output["status"] = "BOX_OVERLAP_REGISTERED"
    else:
        output["status"] = "APPROX_UNREGISTERED"
    return output


def _dashed_rectangle(image, box, color, thickness=2, dash=8):
    x, y, w, h = box
    for xa in range(x, x+w, 2*dash):
        cv2.line(image, (xa, y), (min(xa+dash, x+w), y), color, thickness)
        cv2.line(image, (xa, y+h), (min(xa+dash, x+w), y+h), color, thickness)
    for ya in range(y, y+h, 2*dash):
        cv2.line(image, (x, ya), (x, min(ya+dash, y+h)), color, thickness)
        cv2.line(image, (x+w, ya), (x+w, min(ya+dash, y+h)), color, thickness)


def draw_pose_plane_overlap(image, result):
    """Draw pose ROI in cyan and projected plane boxes in dashed amber."""
    if not result:
        return image
    box = result.get("pose_box_rgb")
    if box:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
    for pbox in result.get("plane_boxes_rgb_approx", []):
        _dashed_rectangle(image, pbox, (0, 170, 255))
    return image
