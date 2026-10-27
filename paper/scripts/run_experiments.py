"""Run the repository's current pipeline and export paper evidence.

All numerical paper tables and figures are generated from these exports. This
script does not edit project source files; it writes only beneath paper/.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from planets import PLANETS
from modules.tile_loader import discover_planet_images
from modules.evaluation import yolo_boxes, circle_boxes, match_boxes
from modules.mission_planner import (
    astar, build_cost_map, choose_landing_point, crater_goal_candidates,
    grid_to_image, render_cost_overlay, route_metrics,
)
import app as project

OUT = ROOT / "paper"
EXPORTS = OUT / "exports"
FIGURES = OUT / "figures"
for directory in (EXPORTS, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 9})
COLORS = {"Moon": "#8c62b5", "Mars": "#d8ad00"}
RUNS: dict[tuple[str, str, str], dict] = {}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def detector_params(planet: str, mode: str) -> dict:
    cfg = PLANETS[planet]
    return dict(
        resolution_m_px=cfg["default_res_m_px"], sun_elev_deg=cfg["default_sun_elev"],
        min_radius=cfg["mars_v2_min_radius"] if planet == "Mars" and mode == "v2 Ellipse" else cfg["default_min_radius"],
        max_radius=cfg["default_max_radius"], param2=cfg["mars_v2_param2"] if planet == "Mars" and mode == "v2 Ellipse" else cfg["default_param2"],
        min_dist=cfg["default_min_dist"], max_detections=cfg["default_max_detections"],
        clahe_clip=cfg["default_clahe_clip"], detector_mode=mode,
        hough_param1=cfg["default_hough_param1"],
        mars_detector_params={
            "blur_sigma_scale": cfg["mars_v2_blur_sigma_scale"],
            "arc_coverage": cfg["mars_v2_arc_coverage"],
            "canny_ratio": cfg["mars_v2_canny_ratio"],
        } if planet == "Mars" and mode == "v2 Ellipse" else None,
    )


def process(entry: dict, planet: str, mode: str) -> dict:
    key = (planet, mode, entry["filename"])
    if key in RUNS:
        return RUNS[key]
    image = cv2.imread(entry["path"], cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {entry['path']}")
    scale = float(entry.get("res_m_px") or PLANETS[planet]["default_res_m_px"])
    params = detector_params(planet, mode)
    params["resolution_m_px"] = scale
    params["sun_elev_deg"] = float(entry.get("sun_elev_deg") or PLANETS[planet]["default_sun_elev"])
    result = project.run_full_pipeline(image, **params)
    result.update(image=image, entry=entry, planet=planet, mode=mode, scale=scale)
    RUNS[key] = result
    return result


def save_detection_exports(entries_by_planet: dict[str, list[dict]]) -> list[dict]:
    detections, tiles = [], []
    for planet, entries in entries_by_planet.items():
        modes = ["v1 Hough"] if planet == "Moon" else ["v2 Ellipse"]
        for entry in entries:
            for mode in modes:
                result = process(entry, planet, mode)
                craters, stats = result["craters"], result["stats"]
                tile = {
                    "planet": planet, "mode": mode, "split": entry["split"],
                    "filename": entry["filename"], "tile": entry["name"],
                    "width_px": result["image"].shape[1], "height_px": result["image"].shape[0],
                    "m_per_px": result["scale"], "scale_calibrated": entry.get("scale_calibrated", True),
                    "detections": len(craters), "small": stats["small"], "medium": stats["medium"],
                    "large": stats["large"], "fresh": stats["fresh"], "slight": stats["slight"],
                    "moderate": stats["moderate"], "heavy": stats["heavy"],
                    "mean_depth_m": float(np.mean([c["depth_m"] for c in craters])) if craters else "",
                    "mean_d2d": float(np.mean([c["d2d"] for c in craters])) if craters else "",
                    "low_confidence_depth_share": 1.0 if planet == "Mars" and craters else (0.0 if craters else ""),
                }
                tiles.append(tile)
                for idx, crater in enumerate(craters, 1):
                    x1, y1, x2, y2 = circle_boxes([crater])[0]
                    detections.append({
                        "planet": planet, "mode": mode, "split": entry["split"],
                        "filename": entry["filename"], "tile": entry["name"], "crater_id": idx,
                        "x_px": crater["x"], "y_px": crater["y"], "radius_px": crater["radius"],
                        "x1_px": x1, "y1_px": y1, "x2_px": x2, "y2_px": y2,
                        "diameter_m": crater["diameter_m"], "depth_m": crater["depth_m"],
                        "d_D": crater["d2d"], "freshness": crater["freshness"],
                        "degradation": crater["deg_label"], "size_class": crater["category"],
                        "scale_calibrated": entry.get("scale_calibrated", True),
                        "depth_low_confidence": planet == "Mars",
                    })
    write_csv(EXPORTS / "tile_summary.csv", tiles)
    write_csv(EXPORTS / "detections.csv", detections)
    return tiles


def validate(entries_by_planet: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for planet, entries in entries_by_planet.items():
        holdout = [e for e in entries if e["split"] in (("valid",) if planet == "Moon" else ("valid", "test"))]
        modes = ["v1 Hough"] if planet == "Moon" else ["v1 Hough", "v2 Ellipse"]
        for entry in holdout:
            image = cv2.imread(entry["path"], cv2.IMREAD_COLOR)
            h, w = image.shape[:2]
            truth = yolo_boxes(entry["label_path"], w, h)
            for mode in modes:
                result = process(entry, planet, mode)
                match = match_boxes(circle_boxes(result["craters"]), truth, 0.5)
                rows.append({"planet": planet, "mode": mode, "split": entry["split"],
                             "filename": entry["filename"], "tile": entry["name"],
                             "gt": len(truth), **match})
    write_csv(EXPORTS / "validation_tile_metrics.csv", rows)
    return rows


def draw_detections(image: np.ndarray, craters: list[dict], planet: str) -> np.ndarray:
    out = image.copy()
    color = (220, 80, 160) if planet == "Moon" else (0, 212, 255)
    for c in craters:
        if c.get("bbox"):
            x1, y1, x2, y2 = map(lambda v: int(round(v)), c["bbox"])
        else:
            x, y, r = int(round(c["x"])), int(round(c["y"])), int(round(c["radius"]))
            x1, y1, x2, y2 = x-r, y-r, x+r, y+r
        if planet == "Mars":
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2 if planet == "Moon" else 1, cv2.LINE_AA)
    return out


def save_system_diagram() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    labels = ["Orbital tile", "Preprocess\nmedian + CLAHE", "Edges and\nshadow mask", "Fit + score\nHough / ellipse", "Measure\nscale + morphology", "Hazard map\n+ A* planner"]
    xs = np.linspace(0.09, 0.91, len(labels))
    for x, label in zip(xs, labels):
        ax.text(x, .55, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=.45", fc="#f1f1f1", ec="#333333", lw=.8))
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate("", (right-.065,.55), (left+.065,.55), arrowprops=dict(arrowstyle="->", lw=1, color="#333333"))
    ax.text(.5,.1,"Moon: Hough circles     |     Mars: ellipse fitting     |     explainable intermediate outputs",ha="center",fontsize=7)
    fig.savefig(FIGURES / "system_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def choose_entry(entries: list[dict], planet: str, preferred: str | None = None) -> dict:
    if preferred:
        found = [e for e in entries if preferred.lower() in e["filename"].lower()]
        if found: return found[0]
    split_pref = [e for e in entries if e["split"] == "valid"]
    return split_pref[0] if split_pref else entries[0]


def save_pipeline_and_annotations(entries_by_planet: dict[str, list[dict]]) -> dict[str, dict]:
    examples = {}
    for planet in ("Moon", "Mars"):
        # Jezero is the best-known labelled Mars locality in this export and is
               # preferred for the paper's Mars dashboard/example image.
        entry = choose_entry(entries_by_planet[planet], planet, "17_jpg" if planet == "Moon" else "jezero")
        mode = "v1 Hough" if planet == "Moon" else "v2 Ellipse"
        result = process(entry, planet, mode)
        examples[planet] = result
    moon, mars = examples["Moon"], examples["Mars"]
    fig, axes = plt.subplots(1, 5, figsize=(7.1, 1.75))
    panels = [(moon["image"], "Input tile"), (moon["enhanced"], "CLAHE"),
              (moon["edges"], "Canny"), (moon["binary"], "Otsu"),
              (draw_detections(moon["image"], moon["craters"], "Moon"), "Fitted detections")]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im, cmap="gray")
        ax.set_title(title); ax.axis("off")
    fig.tight_layout(pad=.3); fig.savefig(FIGURES / "pipeline_stages.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    for ax, (planet, result) in zip(axes, (("Moon", moon), ("Mars", mars))):
        ax.imshow(cv2.cvtColor(draw_detections(result["image"], result["craters"], planet), cv2.COLOR_BGR2RGB))
        ax.set_title(f"{planet}: {len(result['craters'])} detections"); ax.axis("off")
    fig.tight_layout(pad=.3); fig.savefig(FIGURES / "detections_planets.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Show the labelled Jezero tile with dataset boxes and detector boxes in
    # distinct colors so the example can be inspected against its annotations.
    jezero = mars
    jezero_image = jezero["image"].copy()
    h, w = jezero_image.shape[:2]
    for x1, y1, x2, y2 in yolo_boxes(jezero["entry"]["label_path"], w, h):
        cv2.rectangle(jezero_image, (int(x1), int(y1)), (int(x2), int(y2)), (248, 189, 56), 1, cv2.LINE_AA)
    jezero_image = draw_detections(jezero_image, jezero["craters"], "Mars")
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.imshow(cv2.cvtColor(jezero_image, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Jezero Crater tile: detector (yellow) and Roboflow labels (cyan)")
    ax.axis("off")
    fig.tight_layout(pad=.3); fig.savefig(FIGURES / "jezero_labelled_example.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.0))
    for col, (planet, result) in enumerate((("Moon", moon), ("Mars", mars))):
        depth = project.render_depth_map(result["image"], result["craters"])
        hazard = project.render_landing_hazard_map(result["image"], result["craters"])
        axes[0,col].imshow(cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)); axes[0,col].set_title(f"{planet}: depth estimates")
        axes[1,col].imshow(cv2.cvtColor(hazard, cv2.COLOR_BGR2RGB)); axes[1,col].set_title(f"{planet}: crater hazard overlay")
        axes[0,col].axis("off"); axes[1,col].axis("off")
    fig.tight_layout(pad=.3); fig.savefig(FIGURES / "depth_hazard_maps.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return examples


def save_distribution(tiles: list[dict], examples: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    for planet in ("Moon", "Mars"):
        sizes = [c["diameter_m"] for (p, mode, _), result in RUNS.items() if p == planet
                 for c in result["craters"]]
        sizes = [x for x in sizes if x > 0]
        if sizes:
            bins = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 14) if min(sizes) < max(sizes) else [sizes[0]*.9,sizes[0]*1.1]
            axes[0].hist(sizes, bins=bins, histtype="step", linewidth=1.5, color=COLORS[planet], label=planet)
        counts = Counter(c["deg_label"] for (p, _, _), result in RUNS.items() if p == planet for c in result["craters"])
        classes = ["Fresh", "Slightly Degraded", "Moderately Degraded", "Heavily Degraded"]
        positions = np.arange(4) + (0 if planet == "Moon" else .36)
        axes[1].bar(positions, [counts[c] for c in classes], width=.34, color=COLORS[planet], label=planet)
    axes[0].set_xscale("log"); axes[0].set_xlabel("Diameter (m; Mars fallback scale is assumed)"); axes[0].set_ylabel("Detections")
    axes[0].set_title("Diameter frequency"); axes[0].legend(frameon=False)
    axes[1].set_xticks(np.arange(4)+.18, ["Fresh","Slight","Moderate","Heavy"]); axes[1].set_ylabel("Detections"); axes[1].set_title("Degradation labels"); axes[1].legend(frameon=False)
    fig.tight_layout(pad=.4); fig.savefig(FIGURES / "size_degradation.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def mission_example(planet: str, result: dict) -> dict:
    entry, craters, image = result["entry"], result["craters"], result["image"]
    scale = result["scale"]
    rover_radius = math.hypot(1.5, 2.0) / 2.0 + .5
    cmap = build_cost_map(craters, image.shape[1], image.shape[0], scale, rim_buffer_m=1.0, halo_m=5.0,
                          cell_m=max(1.0, scale), rover_radius_m=rover_radius)
    landing = choose_landing_point(cmap, footprint_radius_m=2.0, margin_m=1.0)
    row = {"planet": planet, "filename": entry["filename"], "scale_m_per_px": scale,
           "scale_assumed": not bool(entry.get("scale_calibrated", True)),
           "landing_clearance_m": landing["clearance_m"] if landing else "",
           "landing_satisfies_footprint": landing["satisfies_footprint"] if landing else False,
           "path_length_m": "", "straight_length_m": "", "detour_percent": "",
           "straight_craters_crossed": "", "route_craters_crossed": "", "minimum_rim_clearance_m": ""}
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    overlay = render_cost_overlay(image, cmap)
    route = []
    goal = None
    if landing and cmap["craters"]:
        largest = max(cmap["craters"], key=lambda c: c["radius_m"])
        candidates = crater_goal_candidates(cmap, largest, landing["grid"])
        profiled = np.where(np.isfinite(cmap["costs"]), 1 + np.maximum(0, cmap["costs"]-1), np.inf)
        route = astar(profiled, landing["grid"], candidates, turn_penalty=0.45 / cmap["cell_m"])
        if route:
            goal = route[-1]
            metrics = route_metrics(route, landing["grid"], goal, cmap)
            row.update({k: metrics[k] for k in ("path_length_m", "straight_length_m", "detour_percent", "straight_craters_crossed", "route_craters_crossed", "minimum_rim_clearance_m")})
            pixels = grid_to_image(route, cmap)
            cv2.polylines(overlay, [np.array(pixels, np.int32).reshape(-1,1,2)], False, (255,230,40), 3, cv2.LINE_AA)
            gp = pixels[-1]
            cv2.drawMarker(overlay, gp, (255,255,255), cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA)
            cv2.drawMarker(overlay, gp, (30,110,240), cv2.MARKER_DIAMOND, 9, 2, cv2.LINE_AA)
            start = tuple(map(int, landing["image_px"]))
            line_length = math.hypot(gp[0]-start[0], gp[1]-start[1])
            if line_length:
                ux, uy = (gp[0]-start[0])/line_length, (gp[1]-start[1])/line_length
                position = 0.0
                while position < line_length:
                    end_position = min(position+8.0, line_length)
                    a=(int(start[0]+ux*position),int(start[1]+uy*position))
                    b=(int(start[0]+ux*end_position),int(start[1]+uy*end_position))
                    cv2.line(overlay,a,b,(0,0,255),2,cv2.LINE_AA)
                    position += 15.0
            cv2.circle(overlay, start, max(1,int(round(2/scale))), (0,0,0), 4, cv2.LINE_AA)
            cv2.circle(overlay, start, max(1,int(round(2/scale))), (70,235,120), 2, cv2.LINE_AA)
            cv2.circle(overlay, start, 4, (255,255,255), -1, cv2.LINE_AA)
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); ax.axis("off")
    if landing is None or not route:
        ax.text(.5,.5,"No feasible route in this tile",transform=ax.transAxes,ha="center",color="red",bbox=dict(fc="white",alpha=.8))
    plt.tight_layout(pad=.2)
    return row, fig


def save_mission(examples: dict[str, dict]) -> None:
    rows=[]
    fig, axes = plt.subplots(1,2,figsize=(7.1,2.5))
    for ax, planet in zip(axes,("Moon","Mars")):
        row, panel = mission_example(planet,examples[planet]); rows.append(row)
        panel.canvas.draw()
        # Copy the actual plotted result from its figure canvas into the paired figure.
        arr=np.asarray(panel.canvas.buffer_rgba())
        ax.imshow(arr); ax.axis("off"); ax.set_title(f"{planet}: mission planner")
        plt.close(panel)
    fig.tight_layout(pad=.3); fig.savefig(FIGURES/"mission_planner.png",dpi=300,bbox_inches="tight");plt.close(fig)
    write_csv(EXPORTS/"mission_results.csv",rows)


def save_validation_plot(rows: list[dict]) -> None:
    grouped={}
    for planet, mode in (("Moon","v1 Hough"),("Mars","v1 Hough"),("Mars","v2 Ellipse")):
        values=[r for r in rows if r["planet"]==planet and r["mode"]==mode]
        tp=sum(r["tp"] for r in values); fp=sum(r["fp"] for r in values); fn=sum(r["fn"] for r in values)
        p=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
        grouped[f"{planet} {mode}"]=2*p*rec/(p+rec) if p+rec else 0
    fig,ax=plt.subplots(figsize=(4.2,2.4))
    names=list(grouped); vals=list(grouped.values())
    bars=ax.bar(names,vals,color=[COLORS["Moon"],"#9f7f00",COLORS["Mars"]])
    ax.set_ylim(0,1);ax.set_ylabel("F1 at box IoU 0.5");ax.set_title("Held-out labeled tiles");ax.tick_params(axis="x",rotation=15)
    for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+.02,f"{val:.2f}",ha="center",fontsize=7)
    fig.tight_layout(pad=.4);fig.savefig(FIGURES/"validation_f1.png",dpi=300,bbox_inches="tight");plt.close(fig)


def main() -> None:
    entries={planet:discover_planet_images(cfg) for planet,cfg in PLANETS.items()}
    save_detection_exports(entries)
    val=validate(entries)
    examples=save_pipeline_and_annotations(entries)
    save_distribution([],examples)
    save_mission(examples)
    save_system_diagram()
    save_validation_plot(val)
    metadata={"tile_counts":{p:len(es) for p,es in entries.items()},
              "validation_counts":{p:sum(e["split"]=="valid" if p=="Moon" else e["split"] in ("valid","test") for e in es) for p,es in entries.items()},
              "note":"All metrics derive from current repository functions and exported YOLO labels; Mars depth is marked low confidence. Mars tile scale falls back to the per-planet default where no per-tile m_per_px exists."}
    (EXPORTS/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(metadata,indent=2))
    print(f"Wrote actual run exports and figures under {OUT}")


if __name__=="__main__":
    main()
