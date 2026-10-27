"""Mars-only ellipse detector; the existing Hough detector remains unchanged."""

from __future__ import annotations

import math

import cv2
import numpy as np


def _ellipse_geometry(ellipse):
    (cx, cy), (axis_a, axis_b), angle = ellipse
    if float(axis_a) >= float(axis_b):
        major, minor, major_angle = float(axis_a) / 2.0, float(axis_b) / 2.0, float(angle)
    else:
        major, minor, major_angle = float(axis_b) / 2.0, float(axis_a) / 2.0, float(angle) + 90.0
    return float(cx), float(cy), major, minor, major_angle


def _ellipse_coverage(edges, ellipse, sectors=36):
    """Fraction of angular sectors containing edge pixels near the fitted rim."""
    cx, cy, major, minor, angle = _ellipse_geometry(ellipse)
    if major <= 0 or minor <= 0:
        return 0.0
    h, w = edges.shape[:2]
    x_radius = int(math.ceil(math.hypot(major * math.cos(math.radians(angle)), minor * math.sin(math.radians(angle))))) + 3
    y_radius = int(math.ceil(math.hypot(major * math.sin(math.radians(angle)), minor * math.cos(math.radians(angle))))) + 3
    x1, x2 = max(0, int(cx) - x_radius), min(w, int(cx) + x_radius + 1)
    y1, y2 = max(0, int(cy) - y_radius), min(h, int(cy) + y_radius + 1)
    ys, xs = np.nonzero(edges[y1:y2, x1:x2])
    if len(xs) == 0:
        return 0.0
    dx = xs.astype(float) + x1 - cx
    dy = ys.astype(float) + y1 - cy
    theta = math.radians(angle)
    local_x = (dx * math.cos(theta) + dy * math.sin(theta)) / major
    local_y = (-dx * math.sin(theta) + dy * math.cos(theta)) / minor
    normalized_r = np.sqrt(local_x * local_x + local_y * local_y)
    on_rim = (normalized_r >= 0.78) & (normalized_r <= 1.22)
    if not np.any(on_rim):
        return 0.0
    angles = np.mod(np.arctan2(local_y[on_rim], local_x[on_rim]), 2 * math.pi)
    occupied = np.unique(np.floor(angles * sectors / (2 * math.pi)).astype(int))
    return float(len(occupied) / sectors)


def _bbox(ellipse, width, height):
    cx, cy, major, minor, angle = _ellipse_geometry(ellipse)
    theta = math.radians(angle)
    rx = math.sqrt((major * math.cos(theta)) ** 2 + (minor * math.sin(theta)) ** 2)
    ry = math.sqrt((major * math.sin(theta)) ** 2 + (minor * math.cos(theta)) ** 2)
    return (max(0.0, cx - rx), max(0.0, cy - ry),
            min(float(width - 1), cx + rx), min(float(height - 1), cy + ry))


def _candidate_ellipses(image, edges, min_radius, max_radius, canny_ratio, use_hough, hough_param2):
    h, w = image.shape[:2]
    minimum_dimension = min(h, w)
    kernel_size = max(3, int(round(minimum_dimension * 0.008)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Morphological closing proposes connected crater-rim regions before fitting.
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    fitted = []
    min_perimeter = max(12.0, 2.0 * math.pi * min_radius * 0.65)
    min_area = max(8.0, math.pi * (min_radius * 0.25) ** 2)
    for contour in contours:
        if len(contour) < 5:
            continue
        if cv2.arcLength(contour, True) < min_perimeter or abs(cv2.contourArea(contour)) < min_area:
            continue
        try:
            # Fit an ellipse to each candidate rim contour.
            fitted.append(cv2.fitEllipse(contour))
        except cv2.error:
            continue

    if use_hough:
        # Hough candidates at three downscaled pyramid levels complement rim contours.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        median = float(np.median(gray))
        low = max(0, int(median * canny_ratio))
        param1 = max(40, int(median * 2.0))
        for scale in (0.75, 0.5, 0.35):
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            blur = cv2.GaussianBlur(small, (5, 5), 1.0)
            circles = cv2.HoughCircles(
                blur, cv2.HOUGH_GRADIENT, dp=1.2,
                minDist=max(8, int(min_radius * 1.5 * scale)),
                param1=param1, param2=int(hough_param2),
                minRadius=max(2, int(min_radius * scale)),
                maxRadius=max(3, int(max_radius * scale)),
            )
            if circles is None:
                continue
            for x, y, radius in circles[0]:
                cx, cy, r = float(x / scale), float(y / scale), float(radius / scale)
                roi_radius = int(math.ceil(r * 1.4))
                x1, x2 = max(0, int(cx) - roi_radius), min(w, int(cx) + roi_radius + 1)
                y1, y2 = max(0, int(cy) - roi_radius), min(h, int(cy) + roi_radius + 1)
                local_edges = edges[y1:y2, x1:x2]
                ey, ex = np.nonzero(local_edges)
                if len(ex):
                    dx, dy = ex + x1 - cx, ey + y1 - cy
                    radial = np.hypot(dx, dy)
                    rim = (radial >= r * 0.65) & (radial <= r * 1.35)
                    points = np.column_stack((ex[rim], ey[rim])).astype(np.int32).reshape(-1, 1, 2)
                    if len(points) >= 5:
                        try:
                            candidate_ellipse = cv2.fitEllipse(points)
                            fitted.append(((candidate_ellipse[0][0] + x1, candidate_ellipse[0][1] + y1), candidate_ellipse[1], candidate_ellipse[2]))
                            continue
                        except cv2.error:
                            pass
                fitted.append(((cx, cy), (2 * r, 2 * r), 0.0))
    return fitted


def _ellipse_iou(a, b):
    ax1, ay1, ax2, ay2 = a["bbox"]
    bx1, by1, bx2, by2 = b["bbox"]
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _merge_duplicates(candidates, threshold=0.5):
    kept = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        duplicate_index = None
        for index, existing in enumerate(kept):
            if _ellipse_iou(candidate, existing) <= threshold:
                continue
            smaller, larger = sorted((candidate, existing), key=lambda item: item["radius"])
            nested = (smaller["bbox"][0] >= larger["bbox"][0] and
                      smaller["bbox"][1] >= larger["bbox"][1] and
                      smaller["bbox"][2] <= larger["bbox"][2] and
                      smaller["bbox"][3] <= larger["bbox"][3])
            centers_near = math.hypot(candidate["x"] - existing["x"], candidate["y"] - existing["y"]) < larger["radius"] * 0.25
            if nested and smaller["radius"] / max(larger["radius"], 1e-6) <= 0.4:
                continue
            if nested and not centers_near:
                continue
            duplicate_index = index
            break
        if duplicate_index is None:
            kept.append(candidate)
        elif candidate["score"] > kept[duplicate_index]["score"]:
            kept[duplicate_index] = candidate
    return kept


def detect_craters_mars_v2(
    image_bgr,
    min_radius=10,
    max_radius=85,
    max_detections=45,
    blur_sigma_scale=0.8,
    arc_coverage=0.3,
    canny_ratio=0.5,
    use_hough=True,
    hough_param2=24,
    return_info=False,
):
    """Detect Mars craters as ellipses, returning crater dictionaries and optional fallback status.

    Method outline follows the candidate-region and ellipse-fitting approach described
    by Bandeira et al. (2011) and Kim & Muller: morphology proposes regions, rim edges
    are fit with ellipses, and angular edge coverage filters incomplete rims.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Use mild width-scaled Gaussian smoothing; v2 intentionally skips CLAHE.
    sigma = max(0.1, float(blur_sigma_scale) * (w / 640.0))
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    median = float(np.median(blurred))
    low = max(0, int(median * float(canny_ratio)))
    high = max(low + 1, int(median * min(2.5, float(canny_ratio) * 2.5)))
    edges = cv2.Canny(blurred, low, high)

    accepted, all_fits = [], []
    for ellipse in _candidate_ellipses(
        image_bgr, edges, min_radius, max_radius, canny_ratio, use_hough, hough_param2
    ):
        cx, cy, major, minor, angle = _ellipse_geometry(ellipse)
        axis_ratio = minor / major if major else 0.0
        if major < min_radius or major > max_radius or minor < max(2.0, min_radius * 0.3):
            continue
        # Coverage counts occupied sectors along the fitted rim (36 sectors total).
        coverage = _ellipse_coverage(edges, ellipse)
        record = {
            "x": cx, "y": cy,
            "radius": math.sqrt(major * minor),
            "axis_a": major * 2.0, "axis_b": minor * 2.0,
            "angle": angle,
            "bbox": _bbox(ellipse, w, h),
            "axis_ratio": axis_ratio,
            "rim_coverage": coverage,
            "score": coverage * axis_ratio,
        }
        all_fits.append(record)
        if axis_ratio >= 0.4 and coverage >= arc_coverage:
            accepted.append(record)

    fallback_used = False
    if not accepted:
        fallback_used = True
        accepted = sorted(all_fits, key=lambda item: item["score"], reverse=True)[:max_detections]
    # Merge overlapping fits by IoU while preserving clearly nested small craters.
    accepted = _merge_duplicates(accepted)
    accepted = sorted(accepted, key=lambda item: item["score"], reverse=True)[:max_detections]
    return (accepted, fallback_used) if return_info else accepted
