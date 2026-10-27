"""
Lunar Crater Detection System
Classical Computer Vision Pipeline using OpenCV, NumPy, Matplotlib
Chandrayaan 2 OHRC Imagery Analysis
"""

import cv2
import numpy as np
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
import math
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import sys
import importlib

# ---------------------------------------------------------------------------
# New-module imports (never modify existing functions below this block)
# ---------------------------------------------------------------------------
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from planets import PLANETS, planet_names, get_planet
from modules.tile_loader import discover_planet_images, get_tile_params, save_tile_scale
from modules.mars_detector import detect_craters_mars_v2
# Streamlit can retain an older imported module after a hot-reload. Reload only
# when this newly added helper is missing so the app can recover without a
# manual process restart.
import modules.mission_planner as _mission_planner
if not hasattr(_mission_planner, "render_cost_overlay") or not hasattr(_mission_planner, "reachable_component"):
    _mission_planner = importlib.reload(_mission_planner)
from modules.mission_planner import (
    astar,
    build_cost_map,
    choose_landing_point,
    crater_goal_candidates,
    grid_to_image,
    reachable_component,
    render_cost_overlay,
    route_metrics,
)
from modules.false_positive_filter import apply_dune_filter
from modules.depth_sanity import flag_depth_confidence, filter_high_confidence

# ---------------------------------------------------------------------------
# Colors (BGR for OpenCV, Hex for CSS/Matplotlib)
# ---------------------------------------------------------------------------
COLOR_PURPLE = (220, 80, 160)       # BGR for #A050DC / #9333EA (User Reference Style)
COLOR_GT_CYAN = (248, 189, 56)      # BGR for #38BDF8 (Ground Truth overlay)
COLOR_WHITE   = (255, 255, 255)
COLOR_GREEN   = (80, 200, 70)       # Fresh
COLOR_AMBER   = (53, 184, 255)      # Moderate
COLOR_RED     = (60, 60, 240)       # Degraded / Hazard

HEX_PURPLE = "#A855F7"
HEX_CYAN   = "#38BDF8"
HEX_GREEN  = "#22C55E"
HEX_AMBER  = "#F59E0B"
HEX_RED    = "#EF4444"

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Lunar Crater Detection -using Chandrayaan 2 OHRC imagery-.v1i.yolov8",
)

# ---------------------------------------------------------------------------
# Dataset Discovery & Label Loading
# ---------------------------------------------------------------------------
@st.cache_data
def discover_images():
    """Discover all available lunar tiles across train and valid splits."""
    images = []
    for split in ("train", "valid"):
        img_dir = os.path.join(DATASET_DIR, split, "images")
        lbl_dir = os.path.join(DATASET_DIR, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for fp in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            fname = os.path.basename(fp)
            lbl = os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))
            num = fname.split("_")[0]
            images.append(dict(path=fp, label_path=lbl, name=f"Tile {num}", filename=fname, split=split))
    return images


def load_ground_truth(label_path, img_w, img_h, min_box_size=11):
    """
    Parse YOLO-format label file.
    Filters minute boxes (< min_box_size px) to avoid cluttered clustering.
    """
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = [float(v) for v in parts[:5]]
            pw, ph = w * img_w, h * img_h
            if max(pw, ph) >= min_box_size:
                boxes.append((cx * img_w, cy * img_h, pw, ph))
    return boxes


# ---------------------------------------------------------------------------
# Classical Computer Vision Pipeline
# ---------------------------------------------------------------------------
def preprocess_image(img_bgr, clahe_clip=2.0):
    """
    1. Grayscale conversion.
    2. Median blur (5x5) to eliminate salt-and-pepper regolith speckles.
    3. CLAHE local contrast equalization for balanced shadows and highlights.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    return gray, blurred, enhanced


def detect_edges(enhanced):
    """Canny edge detection highlighting circular crater rims."""
    med = int(np.median(enhanced))
    lower = max(0, int(med * 0.55))
    upper = min(255, int(med * 1.45))
    edges = cv2.Canny(enhanced, lower, upper)
    return edges


def segment_craters(enhanced):
    """Otsu thresholding and morphological opening/closing for shadow basins."""
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return binary, cleaned


def detect_circles(enhanced, edges, min_radius=10, max_radius=85, param2=24, min_dist=20, max_detections=45, param1=130):
    """
    Hough Circle Transform with Rim Edge Density scoring and Non-Maximum Suppression.
    Filters out minute noise and prevents clustered overlapping boxes.
    """
    h, w = enhanced.shape[:2]
    smoothed = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        smoothed,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min_dist),
        param1=int(param1),
        param2=int(param2),
        minRadius=int(min_radius),
        maxRadius=int(max_radius),
    )

    if circles is None:
        return []

    # Score candidates based on rim edge strength
    candidates = []
    for c in circles[0]:
        cx, cy, r = float(c[0]), float(c[1]), float(c[2])
        if cx - r < 0 or cx + r >= w or cy - r < 0 or cy + r >= h:
            continue

        mask_rim = np.zeros(edges.shape, dtype=np.uint8)
        cv2.circle(mask_rim, (int(cx), int(cy)), int(r + 2), 255, -1)
        mask_inner = np.zeros(edges.shape, dtype=np.uint8)
        cv2.circle(mask_inner, (int(cx), int(cy)), max(1, int(r - 2)), 255, -1)
        mask_rim = cv2.subtract(mask_rim, mask_inner)
        edge_pts = edges[mask_rim > 0]
        score = float(edge_pts.mean()) if len(edge_pts) > 0 else 0.0

        candidates.append(dict(x=cx, y=cy, radius=r, score=score))

    # Sort descending by edge prominence
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    # Non-Maximum Suppression to prevent redundant overlapping boxes
    kept = []
    for cand in candidates:
        cx, cy, cr = cand["x"], cand["y"], cand["radius"]
        is_duplicate = False
        for k in kept:
            dist = math.hypot(cx - k["x"], cy - k["y"])
            if dist < (cr + k["radius"]) * 0.65:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(cand)
        if len(kept) >= max_detections:
            break

    return kept


def classify_craters(raw_craters, resolution_m_px, small_thresh=10.0, large_thresh=30.0):
    """Convert pixel radius to physical meters and classify into small, medium, large."""
    classified = []
    for c in raw_craters:
        diam_m = c["radius"] * 2.0 * resolution_m_px
        if diam_m < small_thresh:
            cat = "Small"
        elif diam_m >= large_thresh:
            cat = "Large"
        else:
            cat = "Medium"
        classified.append({
            **c,
            "diameter_m": diam_m,
            "category": cat,
        })
    return classified


def estimate_shadow_depth(gray, craters, resolution_m_px, sun_elevation_deg):
    """Shadow-based crater depth estimation using solar geometry: depth = shadow_length * tan(sun_elev)."""
    tan_elev = math.tan(math.radians(max(5.0, min(85.0, sun_elevation_deg))))
    for c in craters:
        cx, cy, r = int(c["x"]), int(c["y"]), int(c["radius"])
        start_x = max(0, cx - r)
        end_x = min(gray.shape[1] - 1, cx + r)
        row = min(gray.shape[0] - 1, max(0, cy))

        region = gray[row, start_x:end_x + 1].astype(float)
        if len(region) == 0:
            c["depth_m"] = 0.0
            c["d2d"] = 0.0
            continue

        mean_val = region.mean()
        shadow_thresh = mean_val * 0.70
        shadow_pixels = int(np.sum(region < shadow_thresh))
        shadow_frac = shadow_pixels / len(region) if len(region) > 0 else 0.0

        shadow_len_px = shadow_frac * (2.0 * r)
        shadow_len_m = shadow_len_px * resolution_m_px
        depth_m = shadow_len_m * tan_elev

        c["depth_m"] = depth_m
        c["d2d"] = depth_m / c["diameter_m"] if c["diameter_m"] > 0 else 0.0
    return craters


def score_degradation(enhanced, edges, craters):
    """Morphological degradation analysis: rim sharpness + floor texture uniformity."""
    labels = ["Fresh", "Slightly Degraded", "Moderately Degraded", "Heavily Degraded"]
    for c in craters:
        cx, cy, r = int(c["x"]), int(c["y"]), int(c["radius"])

        # Rim sharpness
        mask_rim = np.zeros(edges.shape, dtype=np.uint8)
        cv2.circle(mask_rim, (cx, cy), r + 3, 255, -1)
        mask_inner = np.zeros(edges.shape, dtype=np.uint8)
        cv2.circle(mask_inner, (cx, cy), max(1, r - 3), 255, -1)
        mask_rim = cv2.subtract(mask_rim, mask_inner)
        rim_pixels = edges[mask_rim > 0]
        rim_sharpness = (rim_pixels.mean() / 255.0) if len(rim_pixels) > 0 else 0.0

        # Floor uniformity
        mask_floor = np.zeros(enhanced.shape, dtype=np.uint8)
        cv2.circle(mask_floor, (cx, cy), max(1, int(r * 0.6)), 255, -1)
        floor_pixels = enhanced[mask_floor > 0].astype(float)
        floor_std = floor_pixels.std() if len(floor_pixels) > 0 else 0.0
        floor_score = 1.0 - min(floor_std / 60.0, 1.0)

        freshness = rim_sharpness * 0.6 + floor_score * 0.4
        if freshness > 0.48:
            deg_class = 1
        elif freshness > 0.32:
            deg_class = 2
        elif freshness > 0.18:
            deg_class = 3
        else:
            deg_class = 4

        c["freshness"] = freshness
        c["deg_class"] = deg_class
        c["deg_label"] = labels[deg_class - 1]
    return craters


def run_full_pipeline(img_bgr, resolution_m_px=0.25, sun_elev_deg=30.0,
                      min_radius=10, max_radius=85, param2=24, min_dist=20,
                      max_detections=45, clahe_clip=2.0,
                      detector_mode="v1 Hough", mars_detector_params=None,
                      hough_param1=130):
    """Runs end-to-end processing and compiles summary metrics."""
    gray, blurred, enhanced = preprocess_image(img_bgr, clahe_clip)
    edges = detect_edges(enhanced)
    binary, cleaned = segment_craters(enhanced)
    fallback_used = False
    if detector_mode == "v2 Ellipse":
        mars_params = dict(mars_detector_params or {})
        mars_params.setdefault("hough_param2", param2)
        raw, fallback_used = detect_craters_mars_v2(
            img_bgr, min_radius=min_radius, max_radius=max_radius,
            max_detections=max_detections, return_info=True,
            **mars_params,
        )
    else:
        raw = detect_circles(enhanced, edges, min_radius, max_radius, param2, min_dist, max_detections, hough_param1)
    craters = classify_craters(raw, resolution_m_px)
    craters = estimate_shadow_depth(gray, craters, resolution_m_px, sun_elev_deg)
    craters = score_degradation(enhanced, edges, craters)

    diameters = [c["diameter_m"] for c in craters]
    depths = [c["depth_m"] for c in craters]

    stats = dict(
        total=len(craters),
        small=sum(1 for c in craters if c["category"] == "Small"),
        medium=sum(1 for c in craters if c["category"] == "Medium"),
        large=sum(1 for c in craters if c["category"] == "Large"),
        mean_diam=float(np.mean(diameters)) if diameters else 0.0,
        max_diam=float(max(diameters)) if diameters else 0.0,
        min_diam=float(min(diameters)) if diameters else 0.0,
        mean_depth=float(np.mean(depths)) if depths else 0.0,
        max_depth=float(max(depths)) if depths else 0.0,
        fresh=sum(1 for c in craters if c["deg_class"] == 1),
        slight=sum(1 for c in craters if c["deg_class"] == 2),
        moderate=sum(1 for c in craters if c["deg_class"] == 3),
        heavy=sum(1 for c in craters if c["deg_class"] == 4),
        diameters=diameters,
        depths=depths,
    )

    return dict(
        gray=gray,
        blurred=blurred,
        enhanced=enhanced,
        edges=edges,
        binary=binary,
        cleaned=cleaned,
        craters=craters,
        stats=stats,
        fallback_used=fallback_used,
    )


# ---------------------------------------------------------------------------
# Visual Annotations
# ---------------------------------------------------------------------------
def render_annotations(img_bgr, craters, gt_boxes=None,
                       style="Purple Bounding Boxes",
                       show_labels=False,
                       show_gt=False,
                       detection_color=COLOR_PURPLE,
                       outline_color=None,
                       scale_assumed=False):
    """
    Renders detections and optional cyan ground truth annotations.
    Avoids cluttered text overlays unless explicitly requested.
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # Draw detected craters
    for c in craters:
        cx, cy, r = int(c["x"]), int(c["y"]), int(c["radius"])
        if c.get("bbox"):
            x1, y1, x2, y2 = map(int, c["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
        else:
            x1, y1 = max(0, int(cx - r)), max(0, int(cy - r))
            x2, y2 = min(w - 1, int(cx + r)), min(h - 1, int(cy + r))

        if "Boxes" in style:
            if outline_color:
                cv2.rectangle(out, (x1, y1), (x2, y2), outline_color, 2, cv2.LINE_AA)
            cv2.rectangle(out, (x1, y1), (x2, y2), detection_color, 1 if outline_color else 2, cv2.LINE_AA)
        elif "Circles" in style:
            if outline_color:
                if c.get("bbox"):
                    cv2.ellipse(out, (cx, cy), (max(1, int(c["axis_a"] / 2)), max(1, int(c["axis_b"] / 2))), c.get("angle", 0), 0, 360, outline_color, 2, cv2.LINE_AA)
                else:
                    cv2.circle(out, (cx, cy), r, outline_color, 2, cv2.LINE_AA)
            if c.get("bbox"):
                cv2.ellipse(out, (cx, cy), (max(1, int(c["axis_a"] / 2)), max(1, int(c["axis_b"] / 2))), c.get("angle", 0), 0, 360, detection_color, 1 if outline_color else 2, cv2.LINE_AA)
            else:
                cv2.circle(out, (cx, cy), r, detection_color, 1 if outline_color else 2, cv2.LINE_AA)
        else:  # Combined
            if outline_color:
                cv2.rectangle(out, (x1, y1), (x2, y2), outline_color, 2, cv2.LINE_AA)
                if c.get("bbox"):
                    cv2.ellipse(out, (cx, cy), (max(1, int(c["axis_a"] / 2)), max(1, int(c["axis_b"] / 2))), c.get("angle", 0), 0, 360, outline_color, 2, cv2.LINE_AA)
                else:
                    cv2.circle(out, (cx, cy), r, outline_color, 2, cv2.LINE_AA)
            cv2.rectangle(out, (x1, y1), (x2, y2), detection_color, 1, cv2.LINE_AA)
            if c.get("bbox"):
                cv2.ellipse(out, (cx, cy), (max(1, int(c["axis_a"] / 2)), max(1, int(c["axis_b"] / 2))), c.get("angle", 0), 0, 360, detection_color, 1 if outline_color else 2, cv2.LINE_AA)
            else:
                cv2.circle(out, (cx, cy), r, detection_color, 1 if outline_color else 2, cv2.LINE_AA)

        if show_labels:
            label = f'{c["diameter_m"]:.1f}m' + (" (scale assumed)" if scale_assumed else "")
            tx = max(4, x1)
            ty = max(14, y1 - 4)
            cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, detection_color, 1, cv2.LINE_AA)

    # Optional Ground Truth boxes overlay
    if show_gt and gt_boxes:
        for (gcx, gcy, gw, gh) in gt_boxes:
            gx1 = max(0, int(gcx - gw / 2))
            gy1 = max(0, int(gcy - gh / 2))
            gx2 = min(w - 1, int(gcx + gw / 2))
            gy2 = min(h - 1, int(gcy + gh / 2))
            cv2.rectangle(out, (gx1, gy1), (gx2, gy2), COLOR_GT_CYAN, 2, cv2.LINE_AA)

    return out


def build_interactive_annotation_figure(
    img_bgr, craters, planet_cfg, style="Bounding Boxes", show_labels=True,
    gt_boxes=None, show_gt=False, scale_assumed=False,
):
    """Return a zoomable image overlay with numbered detections and hover details."""
    height, width = img_bgr.shape[:2]
    ok, encoded = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Could not encode the tile for the interactive annotation view")
    image_uri = "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")

    def to_hex(bgr):
        b, g, r = map(int, bgr)
        return f"#{r:02X}{g:02X}{b:02X}"

    detection_hex = planet_cfg["detection_color_hex"]
    label_font_color = "#111827" if detection_hex.upper() == "#FFD400" else "#FFFFFF"
    outline_bgr = planet_cfg.get("outline_color_bgr")
    outline_hex = to_hex(outline_bgr) if outline_bgr else "#111827"
    fig = go.Figure()
    fig.add_layout_image(dict(
        source=image_uri, x=0, y=0, sizex=width, sizey=height,
        xref="x", yref="y", sizing="stretch", layer="below",
    ))

    for crater_id, crater in enumerate(craters, start=1):
        cx, cy = float(crater["x"]), float(crater["y"])
        radius = max(1.0, float(crater.get("radius", 1.0)))
        has_ellipse = crater.get("bbox") and crater.get("axis_a") and crater.get("axis_b")
        if has_ellipse:
            rx, ry = max(1.0, float(crater["axis_a"]) / 2), max(1.0, float(crater["axis_b"]) / 2)
            angle = math.radians(float(crater.get("angle", 0.0)))
            points = []
            for theta in np.linspace(0, 2 * math.pi, 49):
                ex, ey = rx * math.cos(theta), ry * math.sin(theta)
                points.append((cx + ex * math.cos(angle) - ey * math.sin(angle),
                               cy + ex * math.sin(angle) + ey * math.cos(angle)))
        else:
            if crater.get("bbox"):
                x1, y1, x2, y2 = map(float, crater["bbox"])
            else:
                x1, y1, x2, y2 = cx - radius, cy - radius, cx + radius, cy + radius
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]

        visible = []
        if "Boxes" in style or "Combined" in style:
            if crater.get("bbox"):
                x1, y1, x2, y2 = map(float, crater["bbox"])
            else:
                x1, y1, x2, y2 = cx - radius, cy - radius, cx + radius, cy + radius
            visible.append(([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], "box"))
        if "Circles" in style or "Combined" in style:
            visible.append((points if has_ellipse else [
                (cx + radius * math.cos(theta), cy + radius * math.sin(theta))
                for theta in np.linspace(0, 2 * math.pi, 49)
            ], "rim"))
        if not visible:
            visible = [(points, "rim")]

        diameter = float(crater.get("diameter_m", 0.0))
        size_text = f"{diameter:.1f} m" + (" (scale assumed)" if scale_assumed else "")
        hover = (
            f"Crater #{crater_id}<br>Diameter: {size_text}<br>"
            f"Class: {crater.get('category', 'Unknown')}<br>"
            f"Degradation: {crater.get('deg_label', 'Unknown')}<br>"
            f"Freshness: {float(crater.get('freshness', 0.0)):.2f}<br>"
            f"Center: ({cx:.0f}, {cy:.0f}) px"
        )
        for vertices, kind in visible:
            xs, ys = zip(*vertices)
            # A slim dark under-stroke keeps the planet color legible over any terrain.
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", fill="toself",
                fillcolor="rgba(1,1,1,0.001)",
                line=dict(color=outline_hex, width=3),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", fill="toself",
                fillcolor="rgba(1,1,1,0.001)",
                line=dict(color=detection_hex, width=2),
                hoveron="fills", name=f"Crater #{crater_id}",
                hovertemplate=hover + "<extra></extra>", showlegend=False,
            ))

        # Plotly fill hover can be inconsistent across browser versions. This
        # transparent center target guarantees the full details are available
        # when the user points at the crater interior.
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers", name=f"Crater #{crater_id}",
            marker=dict(size=max(18, min(72, radius * 1.5)), color="rgba(255,255,255,0.01)", line_width=0),
            hovertemplate=hover + "<extra></extra>", showlegend=False,
        ))

        if show_labels:
            # IDs stay compact on crowded tiles; precise diameters are in the hover card.
            if crater.get("bbox"):
                label_x, label_y, _, _ = map(float, crater["bbox"])
            else:
                label_x, label_y = cx - radius, cy - radius
            fig.add_annotation(
                x=min(width - 1, max(0, label_x + 2)), y=min(height - 1, max(0, label_y + 2)),
                text=f"#{crater_id}", showarrow=False, xanchor="left", yanchor="top",
                bgcolor=detection_hex, bordercolor=outline_hex or "#111827", borderwidth=1,
                font=dict(size=11, color=label_font_color),
                opacity=0.96,
            )

    if show_gt and gt_boxes:
        for gcx, gcy, gw, gh in gt_boxes:
            gx = [gcx - gw / 2, gcx + gw / 2, gcx + gw / 2, gcx - gw / 2, gcx - gw / 2]
            gy = [gcy - gh / 2, gcy - gh / 2, gcy + gh / 2, gcy + gh / 2, gcy - gh / 2]
            fig.add_trace(go.Scatter(
                x=gx, y=gy, mode="lines", line=dict(color=HEX_CYAN, width=2),
                hoverinfo="skip", showlegend=False,
            ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), hovermode="closest",
        showlegend=False, plot_bgcolor="#111827", paper_bgcolor="#111827",
        xaxis=dict(range=[0, width], visible=False, constrain="domain"),
        yaxis=dict(range=[height, 0], visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def render_depth_map(img_bgr, craters):
    out = img_bgr.copy()
    if not craters:
        return out
    max_d = max((c.get("depth_m", 0.0) for c in craters), default=1.0) or 1.0
    for c in craters:
        cx, cy, r = int(c["x"]), int(c["y"]), int(c["radius"])
        ratio = min(c.get("depth_m", 0.0) / max_d, 1.0)
        b = int(240 - 180 * ratio)
        g = int(140 - 80 * ratio)
        rr = int(60 + 180 * ratio)
        cv2.circle(out, (cx, cy), r, (b, g, rr), 2, cv2.LINE_AA)
    return out


def render_landing_hazard_map(img_bgr, craters):
    out = img_bgr.copy()
    rows, cols = out.shape[:2]
    cell = 32
    grid_r = math.ceil(rows / cell)
    grid_c = math.ceil(cols / cell)
    hazard = np.zeros((grid_r, grid_c), dtype=np.float32)

    for c in craters:
        freshness = c.get("freshness", 0.5)
        h_val = 0.6 + (freshness * 0.6) + (0.4 if c["diameter_m"] > 25 else 0.0)
        gi = int(c["y"] // cell)
        gj = int(c["x"] // cell)
        spread = int(math.ceil(c["radius"] / cell)) + 1
        for di in range(-spread, spread + 1):
            for dj in range(-spread, spread + 1):
                ni, nj = gi + di, gj + dj
                if 0 <= ni < grid_r and 0 <= nj < grid_c:
                    dist = math.hypot(di, dj)
                    falloff = max(0.0, 1.0 - (dist / (spread + 1.0)))
                    hazard[ni, nj] += h_val * falloff

    max_h = hazard.max() if hazard.max() > 0 else 1.0
    overlay = out.copy()
    for i in range(grid_r):
        for j in range(grid_c):
            val = hazard[i, j] / max_h
            if val < 0.28:
                col = (70, 190, 60)
            elif val < 0.58:
                col = (40, 210, 240)
            else:
                col = (50, 50, 230)
            x1, y1 = j * cell, i * cell
            x2, y2 = min((j + 1) * cell, cols), min((i + 1) * cell, rows)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)

    cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)
    for c in craters:
        cv2.circle(out, (int(c["x"]), int(c["y"])), int(c["radius"]), COLOR_WHITE, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Scientific Plots
# ---------------------------------------------------------------------------
def plot_size_histogram(diameters, scale_assumed=False):
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#161b22")
    ax.set_facecolor("#0e1117")
    bins = [0, 4, 8, 12, 16, 20, 30, 45, 60, 90]
    _, _, patches = ax.hist(diameters, bins=bins, edgecolor="#30363d", linewidth=1.0)
    for p, b_left in zip(patches, bins[:-1]):
        if b_left < 10:
            p.set_facecolor("#22C55E")
        elif b_left < 30:
            p.set_facecolor("#F59E0B")
        else:
            p.set_facecolor("#EF4444")
    ax.set_xlabel("Crater Diameter (m)" + (" (scale assumed)" if scale_assumed else ""), color="#e6edf3", fontsize=9)
    ax.set_ylabel("Count", color="#e6edf3", fontsize=9)
    ax.set_title("Size-Frequency Distribution", color="#e6edf3", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    fig.tight_layout()
    return fig


def plot_category_pie(small, medium, large, scale_assumed=False):
    fig, ax = plt.subplots(figsize=(3.5, 3.2), facecolor="#161b22")
    ax.set_facecolor("#161b22")
    vals = [small, medium, large]
    scale_label = " (scale assumed)" if scale_assumed else ""
    labels = [f"Small (<10m){scale_label}", f"Medium (10-30m){scale_label}", f"Large (>30m){scale_label}"]
    colors = ["#22C55E", "#F59E0B", "#EF4444"]
    active = [(v, l, c) for v, l, c in zip(vals, labels, colors) if v > 0]
    if active:
        v_list, l_list, c_list = zip(*active)
        ax.pie(v_list, labels=l_list, colors=c_list, autopct="%1.0f%%", startangle=90,
               textprops={"color": "#e6edf3", "fontsize": 8},
               wedgeprops={"edgecolor": "#161b22", "linewidth": 1.5})
    else:
        ax.text(0.5, 0.5, "No Detections", color="#8b949e", ha="center", va="center")
    ax.set_title("Size Category Breakdown", color="#e6edf3", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_degradation_bars(fresh, slight, moderate, heavy):
    fig, ax = plt.subplots(figsize=(5, 3.0), facecolor="#161b22")
    ax.set_facecolor("#0e1117")
    labels = ["Fresh", "Slightly\nDegraded", "Moderately\nDegraded", "Heavily\nDegraded"]
    counts = [fresh, slight, moderate, heavy]
    colors = ["#22C55E", "#38BDF8", "#F59E0B", "#EF4444"]
    bars = ax.bar(labels, counts, color=colors, edgecolor="#30363d", width=0.55)
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", color="#e6edf3", fontsize=8)
    ax.set_ylabel("Count", color="#e6edf3", fontsize=9)
    ax.set_title("Degradation Classification", color="#e6edf3", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit Main App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Lunar Crater Detection | Chandrayaan-2 OHRC",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #e6edf3;
    }
    [data-testid="stSidebar"] {
        background-color: #111726;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
        color: #f1f5f9;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        font-weight: 500;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #f8fafc !important;
    }
    </style>
    """, unsafe_allow_html=True)

    all_images = discover_images()
    if not all_images:
        st.error(f"Dataset directory not found: {DATASET_DIR}")
        return

    # --- Sidebar Controls ---
    st.sidebar.markdown("### Lunar Crater Detection")
    st.sidebar.caption("Chandrayaan-2 OHRC Analysis | Classical CV Pipeline")
    st.sidebar.markdown("---")

    # ── NEW: Planet selector (default Moon, preserves all Moon behaviour) ──
    with st.sidebar.expander("Planet Selection", expanded=True):
        selected_planet_name = st.selectbox(
            "Planet",
            planet_names(),
            index=0,
            key="planet_selector",
            help="Moon uses Chandrayaan-2 OHRC tiles. Mars uses the Roboflow Mars-Craters dataset.",
        )
    planet_cfg = get_planet(selected_planet_name)
    detector_mode = "v1 Hough"
    if selected_planet_name == "Mars":
        detector_mode = st.sidebar.selectbox(
            "Detector: v1 Hough / v2 Ellipse",
            ["v1 Hough", "v2 Ellipse"],
            index=1,
            key="mars_detector_selector",
            help="Select the detector used for Mars tiles. Moon always uses the original Hough path.",
        )
    mars_detector_params = {
        "blur_sigma_scale": planet_cfg.get("mars_v2_blur_sigma_scale", 0.8),
        "arc_coverage": planet_cfg.get("mars_v2_arc_coverage", 0.3),
        "canny_ratio": planet_cfg.get("mars_v2_canny_ratio", 0.5),
    } if selected_planet_name == "Mars" else None

    # Discover tiles for chosen planet
    planet_images = discover_planet_images(planet_cfg)
    if not planet_images:
        st.error(f"No images found for {selected_planet_name} in {planet_cfg['dataset_dir']}")
        return

    # 1. Tile Selection
    st.sidebar.markdown("**Input Imagery**")
    default_idx = 0

    def _tile_option_label(idx):
        tile = planet_images[idx]
        source_name = tile["filename"].split(".rf", 1)[0]
        for suffix in ("_jpeg", "_jpg", "_png"):
            source_name = source_name.replace(suffix, "")
        if len(source_name) > 38:
            source_name = source_name[:35] + "..."
        if tile.get("recommended_rank"):
            return f"Recommended {tile['recommended_rank']} | {tile['name']} | {source_name}"
        return f"{tile['name']} ({tile['split']}) | {source_name}"

    tile_index = st.sidebar.selectbox(
        "Select Tile",
        list(range(len(planet_images))),
        index=default_idx,
        format_func=_tile_option_label,
        key="planet_tile_index_v2",
    )
    selected_tile = planet_images[tile_index]
    st.sidebar.caption("The first five are ranked for default detector label agreement and a reachable largest-crater route. They are demo picks, not accuracy guarantees.")
    selected_scale_note = " (scale assumed)" if selected_planet_name == "Mars" and not selected_tile.get("scale_calibrated", False) else ""

    if selected_planet_name == "Mars":
        with st.sidebar.expander("Calibrate from scale bar", expanded=False):
            st.caption("Enter the measured scale bar length and its distance in kilometers.")
            bar_px = st.number_input("Bar length in px", min_value=1.0, value=100.0, step=1.0, key="mars_bar_px")
            bar_km = st.number_input("Bar length in km", min_value=0.000001, value=1.0, step=0.1, format="%.6f", key="mars_bar_km")
            if st.button("Save tile calibration", key="save_mars_scale"):
                calibrated_m_px = bar_km * 1000.0 / bar_px
                save_tile_scale(planet_cfg["dataset_dir"], selected_tile["filename"], calibrated_m_px)
                selected_tile["res_m_px"] = calibrated_m_px
                selected_tile["scale_calibrated"] = True
                st.success(f"Saved {calibrated_m_px:.6g} m/px for this tile.")

    st.sidebar.markdown("---")

    # 2. Visualization Style (Following reference image 2)
    st.sidebar.markdown("**Annotation Display**")
    _det_color_hex = planet_cfg["detection_color_hex"]
    annot_style = st.sidebar.selectbox(
        "Annotation Style",
        ["Bounding Boxes", "Circles", "Combined (Boxes + Circles)"],
        index=0,
        help=f"Detections use {_det_color_hex}; ground truth uses cyan."
    )
    show_labels = st.sidebar.checkbox(
        "Show Crater IDs",
        value=True,
        help="Numbers identify detections. Hover over a detection to see its diameter and brief classification."
    )
    show_gt = st.sidebar.checkbox(
        "Overlay Ground Truth",
        value=False,
        help="Overlays manual annotations from dataset YOLO labels (cyan)."
    )
    if show_gt:
        min_gt_size = st.sidebar.slider(
            "Min Ground Truth Size (px)",
            min_value=4, max_value=30, value=11, step=1,
            help="Filters minute noise labels (<11px) to prevent clustering."
        )
    else:
        min_gt_size = 11

    st.sidebar.markdown("---")

    # 3. Detection Parameters (Live responsive sliders)
    st.sidebar.markdown("**Detection Parameters**")
    min_radius = st.sidebar.slider(
        "Min Crater Radius (px)",
        min_value=6, max_value=25,
        value=int(planet_cfg.get("mars_v2_min_radius", 10)) if selected_planet_name == "Mars" and detector_mode == "v2 Ellipse" else int(planet_cfg.get("default_min_radius", 10)),
        step=1,
        help="Minimum candidate radius. Smaller values can recover small labeled craters but may add false positives."
    )
    max_radius = st.sidebar.slider(
        "Max Crater Radius (px)",
        min_value=max(30, int(planet_cfg.get("default_min_radius", 10)) + 5),
        max_value=300,
        value=int(planet_cfg.get("default_max_radius", 85)),
        step=5,
        help="Upper radius bound for candidates. Mars defaults higher because its labeled tiles contain larger pixel-scale craters.",
    )
    max_craters = st.sidebar.slider(
        "Maximum Craters to Detect",
        min_value=15, max_value=120, value=int(planet_cfg.get("default_max_detections", 45)), step=5,
        help="Caps the number of candidates retained after scoring."
    )
    hough_param2 = st.sidebar.slider(
        "Detection Sensitivity (param2)",
        min_value=8, max_value=50,
        value=int(planet_cfg.get("mars_v2_param2", 30) if selected_planet_name == "Mars" and detector_mode == "v2 Ellipse" else planet_cfg.get("default_param2", 24)), step=1,
        help="Accumulator threshold for circular edges."
    )
    min_crater_dist = st.sidebar.slider(
        "Min Distance Between Craters (px)",
        min_value=5, max_value=50, value=int(planet_cfg.get("default_min_dist", 20)), step=1,
        help="Minimum spatial separation between centers."
    )
    clahe_clip = st.sidebar.slider(
        "CLAHE Contrast Clip Limit",
        min_value=1.0, max_value=4.0, value=2.0, step=0.5
    )

    st.sidebar.markdown("---")

    # 4. Planetary & Sensor Parameters
    st.sidebar.markdown("**Geophysical Parameters**")
    _def_res = planet_cfg["default_res_m_px"]
    _def_sun = int(planet_cfg["default_sun_elev"])
    resolution_m_px = st.sidebar.number_input(
        "Spatial Resolution (m/px)" + selected_scale_note,
        min_value=0.05, max_value=500.0, value=float(_def_res), step=0.05, format="%.2f",
        help="Per-tile value is used if tiles_config.json is present; this is the fallback."
    )
    sun_elev_deg = st.sidebar.slider(
        "Sun Elevation Angle (deg)",
        min_value=10, max_value=75, value=_def_sun, step=1,
        help="Used for shadow trigonometric depth calculation. Per-tile overrides apply."
    )

    # ── NEW: False-positive suppression expander (OFF by default) ──
    with st.sidebar.expander("False-Positive Suppression", expanded=False):
        suppress_dunes = st.checkbox(
            "Suppress dune/streak false positives",
            value=False,
            key="suppress_dunes",
            help="Drop circles whose rim edge coverage fraction is below the threshold.",
        )
        dune_threshold = st.slider(
            "Min rim edge coverage",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05,
            key="dune_threshold",
            help="Fraction of rim arc pixels that must contain an edge. 0.5 = 50% coverage.",
        ) if suppress_dunes else 0.5

    # ── NEW: Depth sanity expander (OFF by default) ──
    with st.sidebar.expander("Depth Sanity Filter", expanded=False):
        exclude_low_confidence = st.checkbox(
            "Exclude low-confidence depths from mean",
            value=False,
            key="exclude_low_conf",
            help="Craters with d/D < 0.05 or > 0.25 are flagged as low-confidence. "
                 "For Mars, depth and d/D are marked low-confidence by default because the images are shaded and lack reliable cast shadows. "
                 "This toggle excludes them from mean depth calculation only; "
                 "they still appear in the catalog.",
        )

    # --- Header & Processing ---
    planet_label = planet_cfg["display_name"]
    st.markdown(f"## Crater Detection — {planet_label} | {selected_tile['name']}")
    st.caption("Classical CV: Median Blur, CLAHE, Canny, Otsu, Hough Circle, shadow depth and degradation")

    img_bgr = cv2.imread(selected_tile["path"])
    if img_bgr is None:
        st.error(f"Failed to load image: {selected_tile['path']}")
        return

    h_img, w_img = img_bgr.shape[:2]

    # Use per-tile resolution/sun if available, else sidebar fallback
    tile_res, tile_sun = get_tile_params(selected_tile, resolution_m_px, sun_elev_deg)
    scale_assumed = selected_planet_name == "Mars" and not selected_tile.get("scale_calibrated", False)
    scale_note = " (scale assumed)" if scale_assumed else ""

    # Run Pipeline
    pipeline_res = run_full_pipeline(
        img_bgr=img_bgr,
        resolution_m_px=tile_res,
        sun_elev_deg=tile_sun,
        min_radius=min_radius,
        max_radius=max_radius,
        param2=hough_param2,
        min_dist=min_crater_dist,
        max_detections=max_craters,
        clahe_clip=clahe_clip,
        detector_mode=detector_mode,
        mars_detector_params=mars_detector_params,
        hough_param1=int(planet_cfg.get("default_hough_param1", 130)),
    )

    craters = pipeline_res["craters"]
    if pipeline_res.get("fallback_used"):
        st.info("v2 ellipse filters found no accepted craters; showing the highest-scoring ellipse candidates available.")

    # ── NEW: apply dune/streak false-positive filter (only when toggle ON) ──
    if suppress_dunes:
        craters = apply_dune_filter(craters, pipeline_res["edges"], min_coverage=dune_threshold)
        # Rebuild stats after filtering
        diameters = [c["diameter_m"] for c in craters]
        depths_raw = [c["depth_m"]    for c in craters]
        stats = dict(
            total=len(craters),
            small=sum(1 for c in craters if c["category"] == "Small"),
            medium=sum(1 for c in craters if c["category"] == "Medium"),
            large=sum(1 for c in craters if c["category"] == "Large"),
            mean_diam=float(np.mean(diameters)) if diameters else 0.0,
            max_diam=float(max(diameters)) if diameters else 0.0,
            min_diam=float(min(diameters)) if diameters else 0.0,
            mean_depth=float(np.mean(depths_raw)) if depths_raw else 0.0,
            max_depth=float(max(depths_raw)) if depths_raw else 0.0,
            fresh=sum(1 for c in craters if c["deg_class"] == 1),
            slight=sum(1 for c in craters if c["deg_class"] == 2),
            moderate=sum(1 for c in craters if c["deg_class"] == 3),
            heavy=sum(1 for c in craters if c["deg_class"] == 4),
            diameters=diameters,
            depths=depths_raw,
        )
    else:
        stats = pipeline_res["stats"]

    # ── NEW: depth sanity flags (added in-place, no change to depth values) ──
    craters = flag_depth_confidence(
        craters,
        force_low_confidence=bool(planet_cfg.get("depth_always_low_confidence", False)),
    )

    # For mean-depth metric: optionally exclude low-confidence craters
    _depth_craters = filter_high_confidence(craters) if exclude_low_confidence else craters
    _depth_vals = [c["depth_m"] for c in _depth_craters if c.get("depth_m") is not None]
    display_mean_depth = float(np.mean(_depth_vals)) if _depth_vals else 0.0

    gt_boxes = []
    if show_gt:
        gt_boxes = load_ground_truth(selected_tile["label_path"], w_img, h_img, min_box_size=min_gt_size)

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    with kpi1:
        st.metric("Total Craters", stats["total"])
    with kpi2:
        st.metric("Small (<10m)" + scale_note, stats["small"])
    with kpi3:
        st.metric("Medium (10-30m)" + scale_note, stats["medium"])
    with kpi4:
        st.metric("Large (>30m)" + scale_note, stats["large"])
    with kpi5:
        st.metric("Mean Diameter", f"{stats['mean_diam']:.1f} m{scale_note}")
    with kpi6:
        depth_label = "Mean Depth" + (" (filtered)" if exclude_low_confidence else "")
        st.metric(depth_label, f"{display_mean_depth:.2f} m{scale_note}")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Main Tabs  (new tabs ALWAYS appended after existing ones)
    tab_annot, tab_stages, tab_geophys, tab_stats, tab_catalog, tab_batch, tab_compare, tab_mission = st.tabs([
        "Annotated Detection",
        "Pipeline Stages",
        "Depth & Landing Hazard",
        "Size & Degradation Statistics",
        "Crater Catalog",
        "Batch Analysis (All Tiles)",
        "Moon vs Mars",
        "Mission Planner",
    ])

    # Tab 1: Detection View
    with tab_annot:
        st.caption(f"Detector: {detector_mode}")
        col_view, col_info = st.columns([3, 1])
        with col_view:
            annotation_figure = build_interactive_annotation_figure(
                img_bgr, craters, planet_cfg, style=annot_style,
                show_labels=show_labels, gt_boxes=gt_boxes, show_gt=show_gt,
                scale_assumed=scale_assumed,
            )
            st.plotly_chart(annotation_figure, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
            st.caption("Hover a numbered detection for its diameter, size class, degradation, and image coordinates. Use the plot controls to zoom or pan.")
            leg_a, leg_b, leg_c = st.columns(3)
            leg_a.markdown(f"<span style='color:{_det_color_hex}; font-weight:bold;'>Detected</span> ({annot_style})", unsafe_allow_html=True)
            if show_gt:
                leg_b.markdown(f"<span style='color:{HEX_CYAN}; font-weight:bold;'>Ground Truth YOLO ({len(gt_boxes)} boxes)</span>", unsafe_allow_html=True)
            else:
                leg_b.markdown("<span style='color:#8b949e;'>Ground truth overlay disabled</span>", unsafe_allow_html=True)
            leg_c.markdown(f"Active scale: **{tile_res:.6g} m/px{scale_note}** | Sun: **{tile_sun}°**")
            st.metric("Active spatial resolution", f"{tile_res:.6g} m/px{scale_note}")

        with col_info:
            st.markdown("#### Surface Quantities")
            st.write(f"Tile dimensions: **{w_img} × {h_img} px**")
            tile_area_km2 = (w_img * h_img * (tile_res ** 2)) / 1e6
            st.write(f"Physical coverage: **{tile_area_km2:.4f} km²{scale_note}**")
            crater_density = stats["total"] / tile_area_km2 if tile_area_km2 > 0 else 0
            st.write(f"Crater density: **{crater_density:.0f} craters/km²**")
            st.markdown("---")
            st.write(f"Fresh rims: **{stats['fresh']}**")
            st.write(f"Degraded rims: **{stats['moderate'] + stats['heavy']}**")
            st.write(f"Max diameter: **{stats['max_diam']:.1f} m{scale_note}**")
            st.write(f"Max depth: **{stats['max_depth']:.2f} m{scale_note}**")

    # Tab 2: Stages
    with tab_stages:
        stage_tabs = st.tabs(["1. Raw Image", "2. CLAHE Enhanced", "3. Canny Rims", "4. Otsu Shadows", "5. Circular Fits"])
        with stage_tabs[0]:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption("Raw Chandrayaan-2 High Resolution Camera (OHRC) tile.")
        with stage_tabs[1]:
            st.image(pipeline_res["enhanced"], use_container_width=True, clamp=True)
            st.caption("Grayscale conversion, 5×5 median blur filter, and CLAHE local contrast equalization.")
        with stage_tabs[2]:
            st.image(pipeline_res["edges"], use_container_width=True, clamp=True)
            st.caption("Canny edge detection highlighting the circular crater rims.")
        with stage_tabs[3]:
            st.image(pipeline_res["cleaned"], use_container_width=True, clamp=True)
            st.caption("Otsu binarization and morphological open/close isolating shadowed crater floor interiors.")
        with stage_tabs[4]:
            circle_view = img_bgr.copy()
            for c in craters:
                circle_center = (int(c["x"]), int(c["y"]))
                if planet_cfg["outline_color_bgr"]:
                    if c.get("bbox"):
                        cv2.ellipse(circle_view, circle_center, (max(1, int(c["axis_a"] / 2)), max(1, int(c["axis_b"] / 2))), c.get("angle", 0), 0, 360, planet_cfg["outline_color_bgr"], 2, cv2.LINE_AA)
                    else:
                        cv2.circle(circle_view, circle_center, int(c["radius"]), planet_cfg["outline_color_bgr"], 2, cv2.LINE_AA)
                if c.get("bbox"):
                    cv2.ellipse(circle_view, circle_center, (max(1, int(c["axis_a"] / 2)), max(1, int(c["axis_b"] / 2))), c.get("angle", 0), 0, 360, planet_cfg["detection_color_bgr"], 1 if planet_cfg["outline_color_bgr"] else 2, cv2.LINE_AA)
                else:
                    cv2.circle(circle_view, circle_center, int(c["radius"]), planet_cfg["detection_color_bgr"], 1 if planet_cfg["outline_color_bgr"] else 2, cv2.LINE_AA)
            st.image(cv2.cvtColor(circle_view, cv2.COLOR_BGR2RGB), use_container_width=True)
            fit_label = "Ellipse fits" if detector_mode == "v2 Ellipse" else "Hough Circle candidates"
            st.caption(f"{fit_label} ({len(craters)} detected).")

    # Tab 3: Depth & Hazard
    with tab_geophys:
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown("#### Shadow-Based Depth Estimation")
            if selected_planet_name == "Mars":
                st.caption("Depth and d/D are low-confidence by default for Mars because these shaded images do not provide reliable cast shadows.")
            depth_img = render_depth_map(img_bgr, craters)
            st.image(cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Depth color gradient: Blue (shallow) to Red (deep). Mean depth: {stats['mean_depth']:.2f} m{scale_note}.")
        with g_col2:
            st.markdown("#### Landing Site Hazard Map")
            hazard_img = render_landing_hazard_map(img_bgr, craters)
            st.image(cv2.cvtColor(hazard_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption("Landing safety: Green = Safe smooth terrain, Yellow = Moderate hazard, Red = Severe crater rim hazard.")

    # Tab 4: Statistics
    with tab_stats:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            if stats["diameters"]:
                fig_hist = plot_size_histogram(stats["diameters"], scale_assumed)
                st.pyplot(fig_hist)
                plt.close(fig_hist)
            else:
                st.info("No craters detected for histogram.")
        with c2:
            fig_pie = plot_category_pie(stats["small"], stats["medium"], stats["large"], scale_assumed)
            st.pyplot(fig_pie)
            plt.close(fig_pie)

        fig_deg = plot_degradation_bars(stats["fresh"], stats["slight"], stats["moderate"], stats["heavy"])
        st.pyplot(fig_deg)
        plt.close(fig_deg)

    # Tab 5: Catalog
    with tab_catalog:
        if craters:
            detail_data = []
            for i, c in enumerate(craters):
                detail_data.append({
                    "ID": i + 1,
                    "X (px)": int(c["x"]),
                    "Y (px)": int(c["y"]),
                    "Radius (px)": int(c["radius"]),
                    "Diameter (m)" + scale_note: round(c["diameter_m"], 1),
                    "Category": c["category"],
                    "Est Depth (m)" + scale_note: round(c.get("depth_m", 0.0), 2),
                    "d/D Ratio": round(c.get("d2d", 0.0), 3),
                    "Degradation": c.get("deg_label", "—"),
                    "Freshness Index": round(c.get("freshness", 0.0), 3),
                    # ── NEW columns ──
                    "Rim Coverage": round(c.get("rim_coverage", float("nan")), 3),
                    "Depth Confidence": "Low (Mars default)" if c.get("depth_confidence") == "low_planet_default" else c.get("depth_confidence", "—"),
                })
            df = pd.DataFrame(detail_data)
            if exclude_low_confidence:
                n_excl = len(df[df["Depth Confidence"] != "OK"])
                if n_excl:
                    st.info(f"{n_excl} low-confidence depth crater(s) excluded from mean depth KPI above. All are shown here.")
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Crater Catalog (CSV)",
                data=csv_data,
                file_name=f"crater_catalog_{selected_tile['name'].replace(' ', '_').lower()}.csv",
                mime="text/csv",
            )
        else:
            st.info("No craters detected with current parameter thresholds.")

    # Tab 6: Batch Analysis
    with tab_batch:
        st.markdown("#### Multi-Tile Batch Pipeline")
        st.caption(f"Run classical CV extraction across all {selected_planet_name} tiles using active parameters.")
        if st.button(f"Execute Batch Analysis - {selected_planet_name}"):
            batch_records = []
            all_diams = []
            b_prog = st.progress(0, text="Processing tiles...")
            for idx, item in enumerate(planet_images):
                b_img = cv2.imread(item["path"])
                if b_img is None:
                    continue
                b_res, b_sun = get_tile_params(item, resolution_m_px, sun_elev_deg)
                r_batch = run_full_pipeline(
                    img_bgr=b_img,
                    resolution_m_px=b_res,
                    sun_elev_deg=b_sun,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    param2=hough_param2,
                    min_dist=min_crater_dist,
                    max_detections=max_craters,
                    clahe_clip=clahe_clip,
                    detector_mode=detector_mode,
                    mars_detector_params=mars_detector_params,
                    hough_param1=int(planet_cfg.get("default_hough_param1", 130)),
                )
                b_stats = r_batch["stats"]
                all_diams.extend(b_stats["diameters"])
                batch_records.append({
                    "Tile": item["name"],
                    "Split": item["split"],
                    "Total Detected": b_stats["total"],
                    "Small (<10m)" + (" (scale assumed)" if item.get("planet", "").startswith("Mars") and not item.get("scale_calibrated") else ""): b_stats["small"],
                    "Medium (10-30m)" + (" (scale assumed)" if item.get("planet", "").startswith("Mars") and not item.get("scale_calibrated") else ""): b_stats["medium"],
                    "Large (>30m)" + (" (scale assumed)" if item.get("planet", "").startswith("Mars") and not item.get("scale_calibrated") else ""): b_stats["large"],
                    "Mean Diam (m)" + (" (scale assumed)" if item.get("planet", "").startswith("Mars") and not item.get("scale_calibrated") else ""): round(b_stats["mean_diam"], 1),
                    "Max Diam (m)" + (" (scale assumed)" if item.get("planet", "").startswith("Mars") and not item.get("scale_calibrated") else ""): round(b_stats["max_diam"], 1),
                    "Mean Depth (m)" + (" (scale assumed)" if item.get("planet", "").startswith("Mars") and not item.get("scale_calibrated") else ""): round(b_stats["mean_depth"], 2),
                    "Res (m/px)": round(b_res, 2),
                })
                b_prog.progress((idx + 1) / len(planet_images), text=f"Processing {idx + 1}/{len(planet_images)} ({item['name']})...")

            b_prog.empty()
            df_batch = pd.DataFrame(batch_records)
            st.success(f"Successfully processed {len(batch_records)} tiles. Total craters: {len(all_diams)}.")
            st.dataframe(df_batch, use_container_width=True, hide_index=True)

            if all_diams:
                fig_all = plot_size_histogram(all_diams)
                st.pyplot(fig_all)
                plt.close(fig_all)

    # ── NEW Tab 7: Moon vs Mars Comparison ──────────────────────────────────
    with tab_compare:
        st.markdown("#### Moon vs Mars - Cross-Planet Crater Statistics")
        st.info(
            "**Validity note:** This comparison is only meaningful when both planets "
            "are analysed at similar spatial resolution and within the same diameter range. "
            "Mars tiles here are mixed-source (HiRISE / Viking / MRO) at varying scales, "
            "so densities are indicative rather than absolute. "
            "Use the diameter-range toggle below to restrict both planets to the same window."
        )

        # Diameter restriction toggle
        restrict_diam = st.toggle(
            "Restrict to common diameter range (default 10–40 m)",
            value=False,
            key="compare_restrict_diam",
            help="When ON, only craters with diameter in [dmin, dmax] are included in all comparison charts.",
        )
        if restrict_diam:
            d_col1, d_col2 = st.columns(2)
            cmp_dmin = d_col1.number_input("Min diameter (m)", min_value=1.0, max_value=500.0, value=10.0, step=1.0)
            cmp_dmax = d_col2.number_input("Max diameter (m)", min_value=1.0, max_value=5000.0, value=40.0, step=1.0)
        else:
            cmp_dmin, cmp_dmax = 0.0, float("inf")

        if st.button("Run Moon vs Mars Batch Comparison", key="run_compare"):
            from modules.comparison import run_planet_batch, restrict_to_diameter_range

            moon_cfg  = get_planet("Moon")
            mars_cfg  = get_planet("Mars")

            prog_moon = st.progress(0, text="Processing Moon tiles...")
            def _cb_moon(i, n, name):
                prog_moon.progress(i / n, text=f"Moon: {i}/{n} — {name}")

            moon_batch = run_planet_batch(
                moon_cfg, sidebar_res=0.25, sidebar_sun=30.0,
                min_radius=min_radius, max_radius=int(moon_cfg.get("default_max_radius", 85)), param2=hough_param2,
                min_dist=min_crater_dist, max_detections=max_craters,
                clahe_clip=clahe_clip, progress_cb=_cb_moon,
            )
            prog_moon.empty()

            prog_mars = st.progress(0, text="Processing Mars tiles...")
            def _cb_mars(i, n, name):
                prog_mars.progress(i / n, text=f"Mars: {i}/{n} — {name}")

            mars_batch = run_planet_batch(
                mars_cfg, sidebar_res=mars_cfg["default_res_m_px"], sidebar_sun=mars_cfg["default_sun_elev"],
                min_radius=min_radius, max_radius=int(mars_cfg.get("default_max_radius", 85)), param2=hough_param2,
                min_dist=min_crater_dist, max_detections=max_craters,
                clahe_clip=clahe_clip, progress_cb=_cb_mars, detector_mode=detector_mode,
            )
            prog_mars.empty()

            if restrict_diam and cmp_dmin < cmp_dmax:
                moon_batch = restrict_to_diameter_range(moon_batch, cmp_dmin, cmp_dmax)
                mars_batch = restrict_to_diameter_range(mars_batch, cmp_dmin, cmp_dmax)

            # ── Summary metrics row ──
            st.markdown("---")
            mars_scale_assumed = any(not tile.get("scale_calibrated", False) for tile in discover_planet_images(mars_cfg))
            assumed_label = " (scale assumed)" if mars_scale_assumed else ""
            cmp_range_note = f" (diameter {cmp_dmin:.0f}–{cmp_dmax:.0f} m{assumed_label})" if restrict_diam else ""
            st.markdown(f"**Comparison results{cmp_range_note}**")
            c_m1, c_m2, c_m3, c_ma1, c_ma2, c_ma3 = st.columns(6)
            c_m1.metric("Moon craters",     moon_batch["n_craters"])
            c_m2.metric("Moon density",      f"{moon_batch['crater_density']:.0f}/km²")
            c_m3.metric("Moon mean d/D",     f"{moon_batch['mean_dd']:.3f}")
            c_ma1.metric("Mars craters",     mars_batch["n_craters"])
            c_ma2.metric("Mars density" + assumed_label, f"{mars_batch['crater_density']:.0f}/km²")
            c_ma3.metric("Mars mean d/D",    f"{mars_batch['mean_dd']:.3f}")

            # ── Side-by-side charts ──
            st.markdown("---")
            col_moon, col_mars = st.columns(2)

            # 1. Size-frequency histogram
            with col_moon:
                st.markdown("##### Moon - Size-Frequency")
                if moon_batch["diameters"]:
                    fig_mh, ax_mh = plt.subplots(figsize=(5, 3), facecolor="#161b22")
                    ax_mh.set_facecolor("#0e1117")
                    bins_m = moon_cfg["size_bins"]
                    ax_mh.hist(moon_batch["diameters"], bins=bins_m, color=HEX_CYAN, edgecolor="#30363d")
                    ax_mh.set_xlabel("Diameter (m)", color="#e6edf3", fontsize=8)
                    ax_mh.set_ylabel("Count", color="#e6edf3", fontsize=8)
                    ax_mh.tick_params(colors="#8b949e", labelsize=7)
                    for sp in ax_mh.spines.values(): sp.set_color("#30363d")
                    fig_mh.tight_layout()
                    st.pyplot(fig_mh); plt.close(fig_mh)
                else:
                    st.info("No Moon craters in this diameter range.")

            with col_mars:
                st.markdown("##### Mars - Size-Frequency")
                if mars_batch["diameters"]:
                    fig_mrh, ax_mrh = plt.subplots(figsize=(5, 3), facecolor="#161b22")
                    ax_mrh.set_facecolor("#0e1117")
                    bins_r = mars_cfg["size_bins"]
                    if restrict_diam:
                        import numpy as _np
                        bins_r = list(_np.linspace(cmp_dmin, min(cmp_dmax, max(mars_batch["diameters"])+1), 12))
                    ax_mrh.hist(mars_batch["diameters"], bins=bins_r, color=HEX_AMBER, edgecolor="#30363d")
                    ax_mrh.set_xlabel("Diameter (m)" + assumed_label, color="#e6edf3", fontsize=8)
                    ax_mrh.set_ylabel("Count", color="#e6edf3", fontsize=8)
                    ax_mrh.tick_params(colors="#8b949e", labelsize=7)
                    for sp in ax_mrh.spines.values(): sp.set_color("#30363d")
                    fig_mrh.tight_layout()
                    st.pyplot(fig_mrh); plt.close(fig_mrh)
                else:
                    st.info("No Mars craters in this diameter range.")

            # 2. Degradation distribution
            col_deg_m, col_deg_r = st.columns(2)
            deg_labels = ["Fresh", "Slightly\nDeg.", "Moderately\nDeg.", "Heavily\nDeg."]
            deg_colors = ["#22C55E", "#38BDF8", "#F59E0B", "#EF4444"]

            with col_deg_m:
                st.markdown("##### Moon - Degradation")
                fig_dm, ax_dm = plt.subplots(figsize=(4, 2.8), facecolor="#161b22")
                ax_dm.set_facecolor("#0e1117")
                m_deg = [moon_batch["deg_counts"].get(k, 0) for k in range(1, 5)]
                ax_dm.bar(deg_labels, m_deg, color=deg_colors, edgecolor="#30363d", width=0.5)
                ax_dm.tick_params(colors="#8b949e", labelsize=7)
                for sp in ax_dm.spines.values(): sp.set_color("#30363d")
                fig_dm.tight_layout(); st.pyplot(fig_dm); plt.close(fig_dm)

            with col_deg_r:
                st.markdown("##### Mars - Degradation")
                fig_dr, ax_dr = plt.subplots(figsize=(4, 2.8), facecolor="#161b22")
                ax_dr.set_facecolor("#0e1117")
                r_deg = [mars_batch["deg_counts"].get(k, 0) for k in range(1, 5)]
                ax_dr.bar(deg_labels, r_deg, color=deg_colors, edgecolor="#30363d", width=0.5)
                ax_dr.tick_params(colors="#8b949e", labelsize=7)
                for sp in ax_dr.spines.values(): sp.set_color("#30363d")
                fig_dr.tight_layout(); st.pyplot(fig_dr); plt.close(fig_dr)

            # 3. d/D distribution overlay
            st.markdown("##### d/D Ratio Distribution - Moon (cyan) vs Mars (amber)")
            if moon_batch["dd_ratios"] or mars_batch["dd_ratios"]:
                fig_dd, ax_dd = plt.subplots(figsize=(8, 3), facecolor="#161b22")
                ax_dd.set_facecolor("#0e1117")
                bins_dd = [i * 0.02 for i in range(21)]
                if moon_batch["dd_ratios"]:
                    ax_dd.hist(moon_batch["dd_ratios"], bins=bins_dd, alpha=0.65,
                               color=HEX_CYAN, label="Moon", edgecolor="#30363d")
                if mars_batch["dd_ratios"]:
                    ax_dd.hist(mars_batch["dd_ratios"], bins=bins_dd, alpha=0.65,
                               color=HEX_AMBER, label="Mars", edgecolor="#30363d")
                ax_dd.axvline(0.05, color="#ff6b6b", linestyle="--", linewidth=1, label="d/D bounds (0.05/0.25)")
                ax_dd.axvline(0.25, color="#ff6b6b", linestyle="--", linewidth=1)
                ax_dd.set_xlabel("d/D Ratio", color="#e6edf3", fontsize=9)
                ax_dd.set_ylabel("Count", color="#e6edf3", fontsize=9)
                ax_dd.legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3")
                ax_dd.tick_params(colors="#8b949e", labelsize=8)
                for sp in ax_dd.spines.values(): sp.set_color("#30363d")
                fig_dd.tight_layout()
                st.pyplot(fig_dd); plt.close(fig_dd)
            else:
                st.info("No crater data available for d/D comparison.")

            st.success(f"Comparison complete. Moon: {moon_batch['n_craters']} craters | Mars: {mars_batch['n_craters']} craters")

    with tab_mission:
        st.markdown("#### Mission Planner")
        st.caption("Plan a landing site and, optionally, a rover traverse using detected crater hazards.")
        if scale_assumed:
            st.warning("Tile scale is assumed. Distances, clearances, vehicle footprints, and the route are illustrative until this tile is calibrated.")
        st.info("Terrain slopes, rocks, cliffs, soil strength, and undetected craters are unknown here. The map only represents catalog detections and hazards you add.")

        manual_key = f"mission_manual_hazards:{selected_planet_name}:{selected_tile['filename']}"
        st.session_state.setdefault(manual_key, [])

        setup_left, setup_right = st.columns([1, 1])
        with setup_left:
            objective = st.selectbox(
                "Mission objective",
                ["Visit the largest crater rim", "Visit the nearest crater by size", "Reach image coordinates", "Land only"],
                key="mission_objective_v3",
                help="The landing point is the rover start. A crater-rim destination ends outside the crater keep-out zone.",
            )
            landing_mode = st.selectbox(
                "Landing site",
                ["Recommend the safest available site", "Choose a touchdown coordinate"],
                key="mission_landing_mode_v2",
            )
            if landing_mode == "Choose a touchdown coordinate":
                land_col1, land_col2 = st.columns(2)
                requested_land_x = land_col1.number_input("Touchdown x (px)", 0.0, float(max(0, w_img - 1)), float(w_img / 2), key="mission_land_x_v2")
                requested_land_y = land_col2.number_input("Touchdown y (px)", 0.0, float(max(0, h_img - 1)), float(h_img / 2), key="mission_land_y_v2")
        with setup_right:
            planner_policy = st.selectbox(
                "Route policy",
                ["Balanced", "Safest", "Shortest"],
                key="mission_policy_v2",
                help="All routes avoid blocked crater interiors. Safest penalizes rim and halo zones most; Shortest accepts more risk-zone travel.",
            )
            compare_policies = st.checkbox(
                "Compare all route policies",
                value=False,
                key="mission_compare_policies_v1",
                help="Runs two additional A* searches to compare Safest, Balanced, and Shortest routes.",
            )
            view_layer = st.selectbox(
                "Map view",
                ["Imagery + hazard layers", "Hazard layers only"],
                key="mission_map_view_v2",
            )

        vehicle_col1, vehicle_col2, vehicle_col3 = st.columns(3)
        with vehicle_col1:
            footprint_m = st.number_input(f"Lander footprint radius (m){scale_note}", min_value=0.25, max_value=100.0, value=2.0, step=0.25, key="mission_footprint_v2")
            landing_margin_m = st.number_input(f"Landing safety margin (m){scale_note}", min_value=0.0, max_value=100.0, value=1.0, step=0.25, key="mission_margin_v2")
        with vehicle_col2:
            rover_width_m = st.number_input(f"Rover width (m){scale_note}", min_value=0.25, max_value=20.0, value=1.5, step=0.25, key="mission_rover_width_v2")
            rover_length_m = st.number_input(f"Rover length (m){scale_note}", min_value=0.25, max_value=30.0, value=2.0, step=0.25, key="mission_rover_length_v2")
            rover_margin_m = st.number_input(f"Rover clearance margin (m){scale_note}", min_value=0.0, max_value=20.0, value=0.5, step=0.25, key="mission_rover_margin_v2")
        with vehicle_col3:
            turn_radius_m = st.number_input(f"Turn radius (m){scale_note}", min_value=0.0, max_value=50.0, value=3.0, step=0.5, key="mission_turn_radius_v2", help="Used as a cornering penalty approximation; this is not a full rover kinematic simulation.")
            rim_buffer_base_m = st.number_input(f"Rim safety buffer (m){scale_note}", min_value=0.0, max_value=20.0, value=1.0, step=0.25, key="mission_rim_buffer_v2", help="Additional buffer is 0.25 times crater radius.")
        halo_m = st.slider(f"Outer caution zone (m){scale_note}", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key="mission_halo_v2")

        with st.expander("Add a missed crater to this tile", expanded=False):
            st.caption("Use source-image pixels. Added hazards are saved for this tile in the current app session.")
            manual_col1, manual_col2, manual_col3 = st.columns(3)
            manual_x_px = manual_col1.number_input("Hazard center x (px)", min_value=0.0, max_value=float(max(0, w_img - 1)), value=float(w_img / 2), key=f"{manual_key}:x")
            manual_y_px = manual_col2.number_input("Hazard center y (px)", min_value=0.0, max_value=float(max(0, h_img - 1)), value=float(h_img / 2), key=f"{manual_key}:y")
            manual_radius_px = manual_col3.number_input("Hazard radius (px)", min_value=2.0, max_value=float(max(2, min(w_img, h_img) / 2)), value=float(max(2, min(w_img, h_img) / 12)), key=f"{manual_key}:radius")
            add_col, clear_col = st.columns(2)
            if add_col.button("Add hazard", key=f"{manual_key}:add"):
                st.session_state[manual_key].append({"x": manual_x_px, "y": manual_y_px, "radius": manual_radius_px, "category": "Manual hazard", "freshness": 1.0, "manual": True})
                st.rerun()
            if clear_col.button("Clear added hazards", key=f"{manual_key}:clear"):
                st.session_state[manual_key] = []
                st.rerun()

        mission_scale_note = " (scale assumed)" if scale_assumed else ""
        manual_hazards = st.session_state[manual_key]
        planner_craters = [*craters, *manual_hazards]
        rover_radius_m = math.hypot(rover_width_m, rover_length_m) / 2.0 + rover_margin_m
        mission_map = build_cost_map(
            planner_craters, w_img, h_img, tile_res,
            rim_buffer_m=rim_buffer_base_m,
            halo_m=halo_m,
            rover_radius_m=rover_radius_m,
        )
        required_clearance = max(footprint_m, rover_radius_m) + landing_margin_m
        landing = choose_landing_point(mission_map, required_clearance, 0.0)
        if landing is None:
            st.warning("No traversable cells are available for a landing point.")
        else:
            if landing_mode == "Choose a touchdown coordinate":
                sx_grid, sy_grid = mission_map["image_to_grid"]
                requested = (int(round(requested_land_y * sy_grid)), int(round(requested_land_x * sx_grid)))
                candidates = np.argwhere(mission_map["hazard_free"] > 0)
                if len(candidates):
                    chosen = candidates[np.argmin(np.sum((candidates - np.asarray(requested)) ** 2, axis=1))]
                    row, col = int(chosen[0]), int(chosen[1])
                    landing["grid"] = (row, col)
                    landing["image_px"] = (min(w_img - 1.0, (col + 0.5) / sx_grid), min(h_img - 1.0, (row + 0.5) / sy_grid))
                    landing["clearance_m"] = float(mission_map["landing_clearance"][row, col] * mission_map["cell_m"])
                    landing["satisfies_footprint"] = landing["clearance_m"] >= required_clearance
            if landing["fallback"] or not landing["satisfies_footprint"]:
                st.warning(
                    f"This touchdown candidate does not meet the vehicle clearance requirement ({required_clearance:.1f} m{mission_scale_note}). "
                    f"Its clearance from the crater keep-out zones is {landing['clearance_m']:.1f} m{mission_scale_note}."
                )
            else:
                st.success(f"Touchdown candidate meets the {required_clearance:.1f} m{mission_scale_note} vehicle clearance requirement.")

            st.markdown("##### Mission status")
            route_component = reachable_component(mission_map["costs"], landing["grid"])
            coverage_count = sum(c.get("rim_coverage") is not None for c in craters)
            status_cols = st.columns(6)
            status_cols[0].metric("Detected crater hazards", len(craters))
            status_cols[1].metric("Added hazards", len(manual_hazards))
            status_cols[2].metric("Touchdown clearance", f"{landing['clearance_m']:.1f} m{mission_scale_note}")
            scale_state = "assumed" if scale_assumed else ("calibrated" if selected_planet_name == "Mars" else "dataset default")
            status_cols[3].metric("Map scale", f"{tile_res:.3g} m/px ({scale_state})")
            status_cols[4].metric("Rim quality scores", f"{coverage_count}/{len(craters)}")
            status_cols[5].metric("Planning grid", f"{mission_map['cell_m']:.3g} m/cell{mission_scale_note}")
            if coverage_count:
                st.caption(f"Detector rim-coverage values are available for {coverage_count} of {len(craters)} detections. Other detections remain hazards, but their confidence is unreported.")
            else:
                st.caption("The detector does not provide per-crater confidence for this tile. All detections are treated as hazards.")

            map_overlay = render_cost_overlay(img_bgr, mission_map)
            if view_layer == "Hazard layers only":
                map_overlay = render_cost_overlay(np.full_like(img_bgr, 28), mission_map, opacity=0.88)
            landing_px = tuple(map(int, landing["image_px"]))
            lander_px = max(1, int(round(footprint_m / max(tile_res, 1e-6))))
            cv2.circle(map_overlay, landing_px, lander_px, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.circle(map_overlay, landing_px, lander_px, (70, 235, 120), 2, cv2.LINE_AA)
            cv2.drawMarker(map_overlay, landing_px, (255, 255, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            legend = st.columns(4)
            legend[0].markdown('<span style="display:inline-block;width:12px;height:12px;background:#e22d2d"></span> Blocked crater / rover footprint', unsafe_allow_html=True)
            legend[1].markdown('<span style="display:inline-block;width:12px;height:12px;background:#ff7d00"></span> High-cost rim zone', unsafe_allow_html=True)
            legend[2].markdown('<span style="display:inline-block;width:12px;height:12px;background:#ffdc00"></span> Medium-cost caution zone', unsafe_allow_html=True)
            legend[3].markdown('<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#46eb78"></span> Recommended touchdown', unsafe_allow_html=True)
            st.image(cv2.cvtColor(map_overlay, cv2.COLOR_BGR2RGB), caption="Current hazard map. It includes only detected craters and hazards added above.", use_container_width=True)

            start_grid = landing["grid"]
            detected_goal_craters = [
                item for item in mission_map["craters"]
                if not item["source"].get("manual", False)
            ]
            goal_crater = None
            goal_grid = None
            goal_candidates = []
            goal_label = "Landing site only"
            if objective == "Visit the largest crater rim":
                if detected_goal_craters:
                    ranked_targets = sorted(detected_goal_craters, key=lambda item: item["radius_m"], reverse=True)
                    for goal_crater in ranked_targets:
                        goal_candidates = crater_goal_candidates(
                            mission_map, goal_crater, start_grid, reachable=route_component
                        )
                        if goal_candidates:
                            break
                    if goal_candidates:
                        goal_label = f"Rim of largest reachable crater (radius {goal_crater['radius_m']:.1f} m{mission_scale_note})"
                        if goal_crater is not ranked_targets[0]:
                            st.info("The largest detected crater is cut off by overlapping blocked hazards; routing to the largest reachable crater instead.")
                    else:
                        goal_crater = None
                        st.info("No detected crater rim has an accessible destination cell from this touchdown point.")
                else:
                    st.info("No detected craters are available as a rim destination.")
            elif objective == "Visit the nearest crater by size":
                size_options = ["Small", "Medium", "Large"]
                size_counts = [sum(item["category"] == option for item in detected_goal_craters) for option in size_options]
                default_size_index = int(np.argmax(size_counts)) if any(size_counts) else 0
                size_class = st.selectbox(
                    "Crater size class", size_options, index=default_size_index,
                    key="mission_size_class_v3",
                )
                eligible = [item for item in detected_goal_craters if item["category"] == size_class]
                if eligible:
                    start_x_img, start_y_img = landing["image_px"]
                    start_x_m, start_y_m = start_x_img * tile_res, start_y_img * tile_res
                    ranked_targets = sorted(eligible, key=lambda item: math.hypot(item["x"] - start_x_m, item["y"] - start_y_m))
                    for goal_crater in ranked_targets:
                        goal_candidates = crater_goal_candidates(
                            mission_map, goal_crater, start_grid, reachable=route_component
                        )
                        if goal_candidates:
                            break
                    if goal_candidates:
                        goal_label = f"Rim of nearest reachable {size_class.lower()} crater"
                        if goal_crater is not ranked_targets[0]:
                            st.info(f"The nearest {size_class.lower()} crater is isolated by overlapping hazards; using the next reachable crater in that class.")
                    else:
                        goal_crater = None
                        st.info(f"No {size_class.lower()} crater has an accessible rim destination from this touchdown point.")
                else:
                    st.info(f"No {size_class.lower()} craters were detected on this tile.")
            elif objective == "Reach image coordinates":
                coord_col1, coord_col2 = st.columns(2)
                goal_x_px = coord_col1.number_input("Destination x (px)", 0.0, float(max(0, w_img - 1)), float(w_img * 0.75), key="mission_goal_x_v2")
                goal_y_px = coord_col2.number_input("Destination y (px)", 0.0, float(max(0, h_img - 1)), float(h_img * 0.5), key="mission_goal_y_v2")
                sx_grid, sy_grid = mission_map["image_to_grid"]
                requested = (int(round(goal_y_px * sy_grid)), int(round(goal_x_px * sx_grid)))
                free_cells = np.argwhere((mission_map["hazard_free"] > 0) & route_component)
                if not len(free_cells):
                    # A fully caution-covered tile may have no nominally safe
                    # cell; still let the user route to the nearest traversable
                    # cell while the hazard overlay makes the risk explicit.
                    free_cells = np.argwhere(route_component)
                if len(free_cells):
                    nearest = free_cells[np.argmin(np.sum((free_cells - np.asarray(requested)) ** 2, axis=1))]
                    goal_grid = (int(nearest[0]), int(nearest[1]))
                goal_candidates = [goal_grid] if goal_grid is not None else []
                goal_label = "Nearest reachable point to selected image coordinate"

            st.markdown("##### Start and destination")
            endpoint_cols = st.columns(2)
            endpoint_cols[0].metric("Rover start", f"x {landing['image_px'][0]:.0f}, y {landing['image_px'][1]:.0f} px")
            endpoint_cols[1].metric("Destination", goal_label if objective != "Land only" else "No rover drive")

            if objective == "Land only":
                st.caption("No rover route is requested. The touchdown marker is the mission endpoint.")
            elif goal_candidates:
                policy_factors = {"Safest": 2.0, "Balanced": 1.0, "Shortest": 0.15}
                policy_turns = {"Safest": 1.0, "Balanced": 0.45, "Shortest": 0.1}
                route_options = {}
                policies_to_run = policy_factors if compare_policies else {planner_policy: policy_factors[planner_policy]}
                for policy_name, factor in policies_to_run.items():
                    profile_costs = np.where(
                        np.isfinite(mission_map["costs"]),
                        1.0 + np.maximum(0.0, mission_map["costs"] - 1.0) * factor,
                        np.inf,
                    )
                    selected_route = astar(
                        profile_costs, start_grid, goal_candidates,
                        turn_penalty=(turn_radius_m / max(mission_map["cell_m"], 1e-6)) * policy_turns[policy_name],
                    )
                    if selected_route:
                        route_options[policy_name] = (selected_route, selected_route[-1])

                route = route_options.get(planner_policy, ([], None))[0]
                selected_goal = route_options.get(planner_policy, ([], None))[1]
                if not route:
                    st.warning("No safe route reaches that destination through the detected crater keep-outs. Try another reachable crater, adjust the touchdown point, or inspect whether detections block the corridor.")
                else:
                    measures = route_metrics(route, start_grid, selected_goal, mission_map)
                    route_img = render_cost_overlay(img_bgr, mission_map)
                    planet_color = planet_cfg["detection_color_bgr"]
                    mars_outline = planet_cfg["outline_color_bgr"]
                    for item in craters:
                        if item.get("bbox"):
                            x1, y1, x2, y2 = map(int, item["bbox"])
                        else:
                            cx, cy, radius = int(item["x"]), int(item["y"]), int(item["radius"])
                            x1, y1, x2, y2 = cx - radius, cy - radius, cx + radius, cy + radius
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
                        if mars_outline:
                            cv2.rectangle(route_img, (x1, y1), (x2, y2), mars_outline, 2, cv2.LINE_AA)
                        cv2.rectangle(route_img, (x1, y1), (x2, y2), planet_color, 1 if mars_outline else 2, cv2.LINE_AA)
                    for hazard in manual_hazards:
                        center = (int(hazard["x"]), int(hazard["y"]))
                        radius = int(round(hazard["radius"]))
                        for angle in range(0, 360, 30):
                            cv2.ellipse(route_img, center, (radius, radius), 0, angle, angle + 17, (0, 140, 255), 2, cv2.LINE_AA)

                    route_pixels = grid_to_image(route, mission_map)
                    start_px = tuple(map(int, landing["image_px"]))
                    end_px = grid_to_image([selected_goal], mission_map)[0]
                    line_length_px = math.hypot(end_px[0] - start_px[0], end_px[1] - start_px[1])
                    if line_length_px > 0:
                        ux = (end_px[0] - start_px[0]) / line_length_px
                        uy = (end_px[1] - start_px[1]) / line_length_px
                        position = 0.0
                        while position < line_length_px:
                            next_pos = min(position + 8.0, line_length_px)
                            a = (int(start_px[0] + ux * position), int(start_px[1] + uy * position))
                            b = (int(start_px[0] + ux * next_pos), int(start_px[1] + uy * next_pos))
                            cv2.line(route_img, a, b, (0, 0, 255), 2, cv2.LINE_AA)
                            position += 15.0
                    route_polyline = np.asarray(route_pixels, dtype=np.int32).reshape((-1, 1, 2))
                    if len(route_polyline) >= 2:
                        cv2.polylines(route_img, [route_polyline], False, (0, 0, 0), 5, cv2.LINE_AA)
                        cv2.polylines(route_img, [route_polyline], False, (255, 230, 40), 2, cv2.LINE_AA)
                    footprint_px = max(1, int(round(footprint_m / max(tile_res, 1e-6))))
                    cv2.circle(route_img, start_px, footprint_px, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.circle(route_img, start_px, footprint_px, (70, 235, 120), 2, cv2.LINE_AA)
                    cv2.circle(route_img, start_px, 8, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(route_img, start_px, 5, (35, 180, 75), -1, cv2.LINE_AA)
                    cv2.drawMarker(route_img, end_px, (255, 255, 255), cv2.MARKER_DIAMOND, 18, 4, cv2.LINE_AA)
                    cv2.drawMarker(route_img, end_px, (40, 125, 255), cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)

                    def draw_route_label(text, anchor, color):
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale, thickness = 0.42, 1
                        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
                        x = min(max(2, anchor[0] + 10), max(2, w_img - text_w - 10))
                        y = min(max(text_h + 6, anchor[1] - 10), max(text_h + 6, h_img - baseline - 4))
                        cv2.rectangle(route_img, (x - 4, y - text_h - 4), (x + text_w + 4, y + baseline + 3), (12, 16, 22), -1)
                        cv2.putText(route_img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

                    draw_route_label("TOUCHDOWN", start_px, (100, 255, 150))
                    draw_route_label("DESTINATION", end_px, (90, 175, 255))
                    st.image(cv2.cvtColor(route_img, cv2.COLOR_BGR2RGB), caption="Selected route over crater hazard layers. The dashed red line is the direct comparison.", use_container_width=True)
                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric("Planned path length", f"{measures['path_length_m']:.1f} m{mission_scale_note}")
                    mcol2.metric("Straight-line length", f"{measures['straight_length_m']:.1f} m{mission_scale_note}")
                    mcol3.metric("Detour", f"{measures['detour_percent']:.1f}%")
                    mcol4, mcol5, mcol6 = st.columns(3)
                    mcol4.metric("Known craters crossed, straight", measures["straight_craters_crossed"])
                    mcol5.metric("Known craters crossed, route", measures["route_craters_crossed"])
                    rim_clearance = measures["minimum_rim_clearance_m"]
                    mcol6.metric("Minimum rim clearance", f"{rim_clearance:.1f} m{mission_scale_note}" if np.isfinite(rim_clearance) else "n/a")
                    option_rows = []
                    for option_name, (option_route, option_goal) in route_options.items():
                        option_metrics = route_metrics(option_route, start_grid, option_goal, mission_map)
                        option_rows.append({
                            "Policy": option_name,
                            "Path length (m)": round(option_metrics["path_length_m"], 1),
                            "Detour (%)": round(option_metrics["detour_percent"], 1),
                            "Crater keep-outs crossed": option_metrics["route_craters_crossed"],
                            "Minimum clearance (m)": round(option_metrics["minimum_rim_clearance_m"], 1) if np.isfinite(option_metrics["minimum_rim_clearance_m"]) else "n/a",
                        })
                    if compare_policies and option_rows:
                        st.markdown("##### Route alternatives")
                        st.dataframe(pd.DataFrame(option_rows), use_container_width=True, hide_index=True)
                        st.caption("Route length and clearance use the tile scale shown above. Turn radius is an approximate penalty on sharp direction changes.")
            else:
                if objective != "Land only":
                    st.warning("No valid destination candidates were found for this objective.")


if __name__ == "__main__":
    main()
