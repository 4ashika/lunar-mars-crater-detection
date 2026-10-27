"""
modules/comparison.py
Cross-planet batch comparison helpers for the Moon vs Mars tab.
Runs run_full_pipeline() from app.py on each planet's tiles and aggregates stats.
Never edits any existing function.
"""
import cv2
import numpy as np
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import run_full_pipeline
from modules.tile_loader import discover_planet_images, get_tile_params


def run_planet_batch(planet_cfg: dict, sidebar_res: float, sidebar_sun: float,
                     min_radius: int, max_radius: int, param2: int,
                     min_dist: int, max_detections: int, clahe_clip: float,
                     progress_cb=None, detector_mode: str = "v1 Hough") -> dict:
    """
    Run the pipeline on all tiles of a planet.
    Returns aggregated result dict.
    """
    images = discover_planet_images(planet_cfg)
    all_craters = []
    total_area_km2 = 0.0
    tile_records = []

    for idx, tile in enumerate(images):
        img = cv2.imread(tile["path"])
        if img is None:
            continue
        res, sun = get_tile_params(tile, sidebar_res, sidebar_sun)
        h, w = img.shape[:2]
        result = run_full_pipeline(
            img_bgr=img,
            resolution_m_px=res,
            sun_elev_deg=sun,
            min_radius=min_radius,
            max_radius=max_radius,
            param2=param2,
            min_dist=min_dist,
            max_detections=max_detections,
            clahe_clip=clahe_clip,
            detector_mode=detector_mode if planet_cfg.get("display_name", "").startswith("Mars") else "v1 Hough",
            mars_detector_params={
                "blur_sigma_scale": planet_cfg.get("mars_v2_blur_sigma_scale", 0.8),
                "arc_coverage": planet_cfg.get("mars_v2_arc_coverage", 0.3),
                "canny_ratio": planet_cfg.get("mars_v2_canny_ratio", 0.5),
            },
            hough_param1=planet_cfg.get("default_hough_param1", 130),
        )
        craters = result["craters"]
        area_km2 = (w * h * res ** 2) / 1e6
        total_area_km2 += area_km2
        all_craters.extend(craters)
        tile_records.append(dict(
            tile=tile["name"], area_km2=area_km2,
            count=len(craters), res_m_px=res,
        ))
        if progress_cb:
            progress_cb(idx + 1, len(images), tile["name"])

    # Aggregate
    diameters  = [c["diameter_m"] for c in all_craters]
    depths     = [c["depth_m"]    for c in all_craters if c.get("depth_m") is not None]
    dd_ratios  = [c["d2d"]        for c in all_craters if c.get("d2d") is not None]
    deg_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for c in all_craters:
        deg_counts[c.get("deg_class", 1)] += 1

    density = len(all_craters) / total_area_km2 if total_area_km2 > 0 else 0.0

    return dict(
        all_craters=all_craters,
        tile_records=tile_records,
        diameters=diameters,
        depths=depths,
        dd_ratios=dd_ratios,
        deg_counts=deg_counts,
        total_area_km2=total_area_km2,
        crater_density=density,
        n_tiles=len(tile_records),
        n_craters=len(all_craters),
        mean_diam=float(np.mean(diameters)) if diameters else 0.0,
        mean_depth=float(np.mean(depths)) if depths else 0.0,
        mean_dd=float(np.mean(dd_ratios)) if dd_ratios else 0.0,
    )


def restrict_to_diameter_range(batch_result: dict, dmin: float, dmax: float) -> dict:
    """Return a copy of batch_result with craters/stats restricted to [dmin, dmax] m diameter."""
    craters = [c for c in batch_result["all_craters"]
               if dmin <= c["diameter_m"] <= dmax]
    diameters = [c["diameter_m"] for c in craters]
    depths    = [c["depth_m"]    for c in craters if c.get("depth_m") is not None]
    dd_ratios = [c["d2d"]        for c in craters if c.get("d2d") is not None]
    deg_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for c in craters:
        deg_counts[c.get("deg_class", 1)] += 1
    density = len(craters) / batch_result["total_area_km2"] if batch_result["total_area_km2"] > 0 else 0.0
    return dict(
        batch_result,
        all_craters=craters,
        diameters=diameters,
        depths=depths,
        dd_ratios=dd_ratios,
        deg_counts=deg_counts,
        crater_density=density,
        n_craters=len(craters),
        mean_diam=float(np.mean(diameters)) if diameters else 0.0,
        mean_depth=float(np.mean(depths)) if depths else 0.0,
        mean_dd=float(np.mean(dd_ratios)) if dd_ratios else 0.0,
    )
