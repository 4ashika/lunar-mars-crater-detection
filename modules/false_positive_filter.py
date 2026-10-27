"""
modules/false_positive_filter.py
Suppresses dune/streak false positives by checking rim edge coverage fraction.
This is a NEW module; it never edits detect_circles() in app.py.

A crater has a roughly complete circular rim. A dune or streak has a partial
edge arc. The fraction of rim pixels (in the edge image) that are lit is a
proxy for circularity.
"""
import math
import numpy as np
import cv2


def rim_edge_coverage(edges: np.ndarray, cx: float, cy: float, r: float,
                      rim_width: int = 3) -> float:
    """
    Return the fraction of rim arc pixels that contain an edge.

    Parameters
    ----------
    edges      : uint8 Canny edge map (same size as source image)
    cx, cy, r  : circle centre and radius (floats)
    rim_width  : pixel width of the annular rim band to sample

    Returns
    -------
    coverage : float in [0, 1]
        0 = no edges on rim, 1 = full rim is edge
    """
    h, w = edges.shape[:2]
    mask_outer = np.zeros(edges.shape, dtype=np.uint8)
    mask_inner = np.zeros(edges.shape, dtype=np.uint8)
    cv2.circle(mask_outer, (int(cx), int(cy)), int(r) + rim_width, 255, -1)
    cv2.circle(mask_inner, (int(cx), int(cy)), max(1, int(r) - rim_width), 255, -1)
    rim_mask = cv2.subtract(mask_outer, mask_inner)
    rim_pixels = edges[rim_mask > 0]
    if len(rim_pixels) == 0:
        return 0.0
    return float(np.count_nonzero(rim_pixels)) / float(len(rim_pixels))


def apply_dune_filter(craters: list, edges: np.ndarray,
                      min_coverage: float = 0.5) -> list:
    """
    Drop craters whose rim edge coverage is below min_coverage.

    Parameters
    ----------
    craters      : list of crater dicts from run_full_pipeline()
    edges        : uint8 edge map
    min_coverage : minimum fraction of rim that must have edges (default 0.5)

    Returns
    -------
    List of kept crater dicts (same schema, unmodified).
    The coverage value is stored in c["rim_coverage"] for debugging.
    """
    kept = []
    for c in craters:
        cov = rim_edge_coverage(edges, c["x"], c["y"], c["radius"])
        c = dict(c, rim_coverage=round(cov, 3))
        if cov >= min_coverage:
            kept.append(c)
    return kept
