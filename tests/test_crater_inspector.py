import cv2
import numpy as np

from modules.crater_inspector import crater_crop, crater_diagnostics, classification_sentence


def test_inspector_reports_measurements_and_crop_bounds():
    gray = np.full((64, 64), 100, dtype=np.uint8)
    enhanced = gray.copy()
    edges = np.zeros_like(gray)
    cv2.circle(edges, (32, 32), 12, 255, 1)
    crater = {
        "x": 32, "y": 32, "radius": 12, "diameter_m": 24,
        "depth_m": 2.4, "d2d": 0.1, "freshness": 0.5,
        "deg_class": 1, "deg_label": "Fresh", "category": "Medium",
    }

    metrics = crater_diagnostics(gray, enhanced, edges, crater, 0.25, 30)
    crop, bounds = crater_crop(gray, crater)

    assert 0 <= metrics["rim_edge_coverage"] <= 1
    assert 0 <= metrics["floor_smoothness"] <= 1
    assert metrics["depth_m"] == 2.4
    assert crop.size > 0
    assert bounds == (14, 14, 51, 51)
    assert "Fresh" in classification_sentence(metrics)


def test_inspector_uses_fitted_ellipse_crop_geometry():
    image = np.zeros((80, 100), dtype=np.uint8)
    crater = {"x": 50, "y": 40, "radius": 10, "axis_a": 40, "axis_b": 12, "angle": 0}

    crop, (x1, y1, x2, y2) = crater_crop(image, crater)

    assert crop.shape == (y2 - y1, x2 - x1)
    assert crop.shape == (19, 61)
    assert x1 < 50 < x2
    assert y1 < 40 < y2
