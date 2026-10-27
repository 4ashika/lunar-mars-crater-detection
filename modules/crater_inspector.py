"""Read-only diagnostics and crops for a crater already in the catalog."""

from __future__ import annotations

import math

import cv2
import numpy as np


def _ellipse_parameters(crater: dict) -> tuple[float, float, float, float, float]:
    """Return center, semi-axes, and angle; circles are represented as ellipses."""
    cx, cy = float(crater["x"]), float(crater["y"])
    if crater.get("axis_a") and crater.get("axis_b"):
        return cx, cy, float(crater["axis_a"]) / 2, float(crater["axis_b"]) / 2, float(crater.get("angle", 0.0))
    radius = max(1.0, float(crater["radius"]))
    return cx, cy, radius, radius, 0.0


def crater_crop(image: np.ndarray, crater: dict, margin_factor: float = 1.5) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop around the crater rim with room for context, clamped to image bounds."""
    height, width = image.shape[:2]
    cx, cy, axis_a, axis_b, angle = _ellipse_parameters(crater)
    theta = math.radians(angle)
    rx = math.hypot(axis_a * math.cos(theta), axis_b * math.sin(theta))
    ry = math.hypot(axis_a * math.sin(theta), axis_b * math.cos(theta))
    pad_x = max(4, int(math.ceil(rx * margin_factor)))
    pad_y = max(4, int(math.ceil(ry * margin_factor)))
    x1, y1 = max(0, int(cx) - pad_x), max(0, int(cy) - pad_y)
    x2, y2 = min(width, int(cx) + pad_x + 1), min(height, int(cy) + pad_y + 1)
    return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def crater_diagnostics(
    gray: np.ndarray,
    enhanced: np.ndarray,
    edges: np.ndarray,
    crater: dict,
    resolution_m_px: float,
    sun_elevation_deg: float,
) -> dict:
    """Recompute explainability measurements without changing crater scoring."""
    height, width = gray.shape[:2]
    cx, cy = float(crater["x"]), float(crater["y"])
    axis_x, axis_y, angle = _ellipse_parameters(crater)[2:]
    radius = max(1, int(round(float(crater["radius"]))))

    # Measure edge occupancy in 36 angular sectors near the detected rim.
    x_radius = int(math.ceil(math.hypot(axis_x * math.cos(math.radians(angle)), axis_y * math.sin(math.radians(angle))))) + 3
    y_radius = int(math.ceil(math.hypot(axis_x * math.sin(math.radians(angle)), axis_y * math.cos(math.radians(angle))))) + 3
    x1, x2 = max(0, int(cx) - x_radius), min(width, int(cx) + x_radius + 1)
    y1, y2 = max(0, int(cy) - y_radius), min(height, int(cy) + y_radius + 1)
    ys, xs = np.nonzero(edges[y1:y2, x1:x2])
    if len(xs):
        dx, dy = xs.astype(float) + x1 - cx, ys.astype(float) + y1 - cy
        theta = math.radians(angle)
        local_x = (dx * math.cos(theta) + dy * math.sin(theta)) / max(axis_x, 1e-6)
        local_y = (-dx * math.sin(theta) + dy * math.cos(theta)) / max(axis_y, 1e-6)
        normalized_radius = np.hypot(local_x, local_y)
        on_rim = (normalized_radius >= 0.78) & (normalized_radius <= 1.22)
        if np.any(on_rim):
            sectors = np.floor(np.mod(np.arctan2(local_y[on_rim], local_x[on_rim]), 2 * math.pi) * 36 / (2 * math.pi)).astype(int)
            rim_edge_coverage = float(len(np.unique(sectors)) / 36)
        else:
            rim_edge_coverage = 0.0
    else:
        rim_edge_coverage = 0.0

    # Match the existing degradation scorer's circular rim density and floor uniformity.
    mask_outer = np.zeros(edges.shape, dtype=np.uint8)
    cv2.circle(mask_outer, (int(round(cx)), int(round(cy))), radius + 3, 255, -1)
    mask_inner = np.zeros(edges.shape, dtype=np.uint8)
    cv2.circle(mask_inner, (int(round(cx)), int(round(cy))), max(1, radius - 3), 255, -1)
    rim_mask = cv2.subtract(mask_outer, mask_inner)
    rim_pixels = edges[rim_mask > 0]
    rim_edge_density = float(rim_pixels.mean() / 255.0) if len(rim_pixels) else 0.0

    floor_mask = np.zeros(enhanced.shape, dtype=np.uint8)
    cv2.circle(floor_mask, (int(round(cx)), int(round(cy))), max(1, int(radius * 0.6)), 255, -1)
    floor_pixels = enhanced[floor_mask > 0].astype(float)
    floor_std = float(floor_pixels.std()) if len(floor_pixels) else 0.0
    floor_smoothness = 1.0 - min(floor_std / 60.0, 1.0)

    row = min(height - 1, max(0, int(cy)))
    start_x, end_x = max(0, int(cx - radius)), min(width - 1, int(cx + radius))
    intensity = gray[row, start_x:end_x + 1].astype(float)
    if len(intensity):
        threshold = float(intensity.mean() * 0.70)
        shadow_fraction = float(np.count_nonzero(intensity < threshold) / len(intensity))
        shadow_length_px = shadow_fraction * (2.0 * radius)
    else:
        threshold, shadow_fraction, shadow_length_px = 0.0, 0.0, 0.0

    size_class = crater.get("category", "Unknown")
    degradation = crater.get("deg_label", "Unknown")
    return {
        "rim_edge_coverage": rim_edge_coverage,
        "detector_rim_coverage": (float(crater["rim_coverage"])
                                   if crater.get("rim_coverage") is not None else None),
        "rim_edge_density": rim_edge_density,
        "floor_smoothness": floor_smoothness,
        "floor_intensity_std": floor_std,
        "shadow_threshold": threshold,
        "shadow_fraction": shadow_fraction,
        "shadow_length_px": shadow_length_px,
        "shadow_length_m": shadow_length_px * float(resolution_m_px),
        "sun_elevation_deg": float(sun_elevation_deg),
        "depth_m": float(crater.get("depth_m", 0.0)),
        "depth_confidence": crater.get("depth_confidence", "unspecified"),
        "d_over_D": float(crater.get("d2d", 0.0)),
        "freshness": float(crater.get("freshness", 0.0)),
        "degradation": degradation,
        "size_class": size_class,
        "ellipse": crater.get("axis_a") is not None and crater.get("axis_b") is not None,
    }


def classification_sentence(diagnostics: dict) -> str:
    """Produce a short explanation tied to the displayed measurements."""
    freshness = diagnostics["freshness"]
    rim_density = diagnostics["rim_edge_density"]
    floor_smoothness = diagnostics["floor_smoothness"]
    if freshness > 0.48:
        reason = "strong rim edges and a relatively smooth floor support a fresh classification"
    elif freshness > 0.32:
        reason = "the rim and floor measurements support a slightly degraded classification"
    elif freshness > 0.18:
        reason = "weaker rim structure or a less uniform floor support a moderately degraded classification"
    else:
        reason = "weak rim structure or a textured floor supports a heavily degraded classification"
    return (
        f"Classified as {diagnostics['degradation']} and {diagnostics['size_class']}: {reason} "
        f"(rim edge density {rim_density:.2f}, floor smoothness {floor_smoothness:.2f})."
    )
