"""Tune Mars v2 ellipse-detector defaults on the labeled validation split.

Running this tool writes the winning Mars v2 values into planets.py and prints
the baseline v1 Hough F1 beside the tuned v2 F1.
"""

from __future__ import annotations

import itertools
import os
import re
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import run_full_pipeline
from modules.evaluation import circle_boxes, match_boxes, yolo_boxes
from modules.mars_detector import detect_craters_mars_v2
from modules.tile_loader import discover_planet_images
from planets import get_planet


GRID = {
    "blur_sigma_scale": (1.0,),
    "arc_coverage": (0.55, 0.6),
    "canny_ratio": (0.6,),
    "min_radius": (12, 15),
    "hough_param2": (28, 30),
}
IOU_THRESHOLD = 0.5


def _aggregate_f1(tile_data, detector, params=None):
    tp = fp = fn = 0
    cfg = get_planet("Mars")
    for tile, image, gt in tile_data:
        if detector == "v1":
            result = run_full_pipeline(
                image, resolution_m_px=float(tile["res_m_px"]),
                sun_elev_deg=float(tile["sun_elev_deg"]),
                min_radius=int(cfg["default_min_radius"]),
                max_radius=int(cfg["default_max_radius"]),
                param2=int(cfg["default_param2"]),
                min_dist=int(cfg["default_min_dist"]),
                max_detections=int(cfg["default_max_detections"]),
                clahe_clip=float(cfg["default_clahe_clip"]),
                hough_param1=int(cfg.get("default_hough_param1", 130)),
            )
            predictions = circle_boxes(result["craters"])
        else:
            min_radius = int(params["min_radius"])
            detections = detect_craters_mars_v2(
                image,
                min_radius=min_radius,
                max_radius=int(cfg["default_max_radius"]),
                max_detections=int(cfg["default_max_detections"]),
                blur_sigma_scale=float(params["blur_sigma_scale"]),
                arc_coverage=float(params["arc_coverage"]),
                canny_ratio=float(params["canny_ratio"]),
                hough_param2=int(params["hough_param2"]),
            )
            predictions = circle_boxes(detections)
        m = match_boxes(predictions, gt, IOU_THRESHOLD)
        tp += m["tp"]
        fp += m["fp"]
        fn += m["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _write_defaults(best):
    path = os.path.join(ROOT, "planets.py")
    with open(path, encoding="utf-8") as f:
        source = f.read()
    mars_start = source.index('    "Mars": {')
    next_planet = source.find('\n    },', mars_start)
    mars_block = source[mars_start:next_planet + 7 if next_planet >= 0 else len(source)]
    keys = {
        "blur_sigma_scale": "mars_v2_blur_sigma_scale",
        "arc_coverage": "mars_v2_arc_coverage",
        "canny_ratio": "mars_v2_canny_ratio",
        "min_radius": "mars_v2_min_radius",
        "hough_param2": "mars_v2_param2",
    }
    for param, key in keys.items():
        value = best[param]
        pattern = rf'("{re.escape(key)}"\s*:\s*)([^,]+)'
        mars_block, replacements = re.subn(pattern, lambda m: m.group(1) + repr(value), mars_block, count=1)
        if replacements != 1:
            raise RuntimeError(f"Could not locate {key} in the Mars defaults block")
    source = source[:mars_start] + mars_block + source[mars_start + len(source[mars_start:next_planet + 7 if next_planet >= 0 else len(source)]):]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(source)


def main():
    cfg = get_planet("Mars")
    tiles = [tile for tile in discover_planet_images(cfg) if tile["split"] == "valid"]
    if not tiles:
        raise RuntimeError("No Mars validation tiles were found.")
    tile_data = []
    for tile in tiles:
        image = cv2.imread(tile["path"])
        if image is None:
            continue
        height, width = image.shape[:2]
        gt = yolo_boxes(tile["label_path"], width, height)
        tile_data.append((tile, image, gt))
    if not tile_data:
        raise RuntimeError("No readable Mars validation images were found.")

    v1_f1 = _aggregate_f1(tile_data, "v1")
    best, best_f1 = None, -1.0
    names = tuple(GRID)
    for values in itertools.product(*(GRID[name] for name in names)):
        params = dict(zip(names, values))
        f1 = _aggregate_f1(tile_data, "v2", params)
        if f1 > best_f1:
            best, best_f1 = params, f1
        print(f"v2 {params} -> F1={f1:.4f}")

    _write_defaults(best)
    print(f"\nValidation tiles: {len(tile_data)}")
    print(f"v1 Hough F1: {v1_f1:.4f}")
    print(f"v2 Ellipse best F1: {best_f1:.4f}")
    print(f"Best Mars defaults written to planets.py: {best}")


if __name__ == "__main__":
    main()
