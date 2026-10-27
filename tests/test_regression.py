"""
Regression test for the Lunar Crater Detection pipeline.

Runs the CURRENT pipeline on Tile 17 with the default sidebar values:
  min_radius=10, max_detections=45, param2=24, min_dist=20,
  clahe_clip=2.0, resolution_m_px=0.25, sun_elev_deg=30

On first run (--generate-baseline flag):
  - Runs the pipeline and saves a snapshot to tests/baseline/.
  - Writes baseline_meta.json  (crater count + key stats)
  - Writes baseline_catalog.csv (full per-crater catalog)

On subsequent runs (default, no flag):
  - Re-runs the pipeline with the same parameters.
  - Asserts crater count matches the saved baseline EXACTLY.
  - Asserts each catalog row matches within floating-point tolerance.

Usage
-----
  # Generate (or regenerate) the baseline:
  python tests/test_regression.py --generate-baseline

  # Run the regression check (also invoked by pytest):
  pytest tests/test_regression.py -v

RULES:
  (1) This file NEVER edits app.py functions -- it imports them directly.
  (2) To regenerate baseline after an INTENTIONAL change, pass --generate-baseline.
  (3) Never patch the baseline to make a failing test pass; revert the change instead.
"""

import sys
import os
import json
import glob
import argparse
import math

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the pipeline functions directly from app.py (no edits to app.py!)
from app import (
    run_full_pipeline,
    DATASET_DIR,
)

# ---------------------------------------------------------------------------
# Default parameter values (must match the sidebar defaults in app.py)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    min_radius=10,
    max_radius=85,
    param2=24,
    min_dist=20,
    max_detections=45,
    clahe_clip=2.0,
    resolution_m_px=0.25,
    sun_elev_deg=30.0,
)

BASELINE_DIR = os.path.join(os.path.dirname(__file__), "baseline")
BASELINE_META = os.path.join(BASELINE_DIR, "baseline_meta.json")
BASELINE_CSV  = os.path.join(BASELINE_DIR, "baseline_catalog.csv")

TILE_NUMBER = "17"

FLOAT_RTOL = 1e-4
FLOAT_ATOL = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_tile17() -> str:
    for split in ("train", "valid"):
        img_dir = os.path.join(DATASET_DIR, split, "images")
        if not os.path.isdir(img_dir):
            continue
        for fp in glob.glob(os.path.join(img_dir, "*.jpg")):
            fname = os.path.basename(fp)
            num = fname.split("_")[0]
            if num == TILE_NUMBER:
                return fp
    raise FileNotFoundError(
        f"Tile {TILE_NUMBER} not found under {DATASET_DIR}."
    )


def run_pipeline_on_tile17() -> dict:
    img_path = find_tile17()
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f"cv2.imread failed for path: {img_path}")
    result = run_full_pipeline(
        img_bgr=img_bgr,
        resolution_m_px=DEFAULT_PARAMS["resolution_m_px"],
        sun_elev_deg=DEFAULT_PARAMS["sun_elev_deg"],
        min_radius=DEFAULT_PARAMS["min_radius"],
        max_radius=DEFAULT_PARAMS["max_radius"],
        param2=DEFAULT_PARAMS["param2"],
        min_dist=DEFAULT_PARAMS["min_dist"],
        max_detections=DEFAULT_PARAMS["max_detections"],
        clahe_clip=DEFAULT_PARAMS["clahe_clip"],
    )
    return result


def craters_to_dataframe(craters: list) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(craters):
        rows.append({
            "ID": i + 1,
            "X (px)": int(c["x"]),
            "Y (px)": int(c["y"]),
            "Radius (px)": int(c["radius"]),
            "Diameter (m)": round(c["diameter_m"], 1),
            "Category": c["category"],
            "Est Depth (m)": round(c.get("depth_m", 0.0), 2),
            "d/D Ratio": round(c.get("d2d", 0.0), 3),
            "Degradation": c.get("deg_label", "---"),
            "Freshness Index": round(c.get("freshness", 0.0), 3),
        })
    return pd.DataFrame(rows)


def build_meta(stats: dict, params: dict) -> dict:
    return {
        "tile": f"Tile {TILE_NUMBER}",
        "params": params,
        "crater_count": stats["total"],
        "small": stats["small"],
        "medium": stats["medium"],
        "large": stats["large"],
        "mean_diam_m": round(stats["mean_diam"], 4),
        "max_diam_m": round(stats["max_diam"], 4),
        "min_diam_m": round(stats["min_diam"], 4),
        "mean_depth_m": round(stats["mean_depth"], 4),
        "max_depth_m": round(stats["max_depth"], 4),
        "fresh": stats["fresh"],
        "slight": stats["slight"],
        "moderate": stats["moderate"],
        "heavy": stats["heavy"],
    }


# ---------------------------------------------------------------------------
# Baseline generation
# ---------------------------------------------------------------------------

def generate_baseline():
    os.makedirs(BASELINE_DIR, exist_ok=True)
    print(f"[baseline] Running pipeline on Tile {TILE_NUMBER} with default params ...")
    result = run_pipeline_on_tile17()
    craters = result["craters"]
    stats   = result["stats"]
    meta = build_meta(stats, DEFAULT_PARAMS)

    with open(BASELINE_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[baseline] Saved meta  -> {BASELINE_META}")

    df = craters_to_dataframe(craters)
    df.to_csv(BASELINE_CSV, index=False)
    print(f"[baseline] Saved catalog ({len(df)} rows) -> {BASELINE_CSV}")

    print(f"\n  Crater count : {meta['crater_count']}")
    print(f"  Small / Med / Large : {meta['small']} / {meta['medium']} / {meta['large']}")
    print(f"  Mean diameter : {meta['mean_diam_m']} m")
    print(f"  Mean depth    : {meta['mean_depth_m']} m")
    print("\n[baseline] Done.")


# ---------------------------------------------------------------------------
# Regression assertions (called by pytest)
# ---------------------------------------------------------------------------

def _load_baseline():
    if not os.path.isfile(BASELINE_META) or not os.path.isfile(BASELINE_CSV):
        raise FileNotFoundError(
            "Baseline not found. Run: python tests/test_regression.py --generate-baseline"
        )
    with open(BASELINE_META) as f:
        meta = json.load(f)
    df_base = pd.read_csv(BASELINE_CSV)
    return meta, df_base


def test_tile17_crater_count():
    """Crater count must match baseline exactly."""
    meta, _ = _load_baseline()
    result = run_pipeline_on_tile17()
    actual = result["stats"]["total"]
    expected = meta["crater_count"]
    assert actual == expected, (
        f"Crater count CHANGED: expected {expected}, got {actual}. "
        "If intentional, regenerate baseline with --generate-baseline and document why. "
        "Do NOT patch the baseline to hide a regression."
    )


def test_tile17_catalog_rows():
    """Each catalog row must match baseline within tolerance."""
    meta, df_base = _load_baseline()
    result = run_pipeline_on_tile17()
    df_new = craters_to_dataframe(result["craters"])

    assert len(df_new) == len(df_base), (
        f"Row count mismatch: baseline={len(df_base)}, current={len(df_new)}."
    )

    int_cols   = ["ID", "X (px)", "Y (px)", "Radius (px)"]
    float_cols = ["Diameter (m)", "Est Depth (m)", "d/D Ratio", "Freshness Index"]
    str_cols   = ["Category", "Degradation"]

    for col in int_cols:
        mismatch = df_new[col].values != df_base[col].values
        if mismatch.any():
            raise AssertionError(
                f"Integer column '{col}' mismatch at IDs: {df_base['ID'].values[mismatch].tolist()}"
            )

    for col in float_cols:
        ok = np.isclose(
            df_new[col].values.astype(float),
            df_base[col].values.astype(float),
            rtol=FLOAT_RTOL, atol=FLOAT_ATOL,
        )
        if not ok.all():
            raise AssertionError(
                f"Float column '{col}' mismatch at IDs: {df_base['ID'].values[~ok].tolist()}"
            )

    for col in str_cols:
        mismatch = df_new[col].values != df_base[col].values
        if mismatch.any():
            raise AssertionError(
                f"String column '{col}' mismatch at IDs: {df_base['ID'].values[mismatch].tolist()}"
            )


def test_tile17_stats_summary():
    """Key summary stats must match baseline within tolerance."""
    meta, _ = _load_baseline()
    stats = run_pipeline_on_tile17()["stats"]

    assert stats["total"]  == meta["crater_count"], "total count mismatch"
    assert stats["small"]  == meta["small"],         "small count mismatch"
    assert stats["medium"] == meta["medium"],         "medium count mismatch"
    assert stats["large"]  == meta["large"],          "large count mismatch"
    assert stats["fresh"]  == meta["fresh"],          "fresh count mismatch"
    assert math.isclose(stats["mean_diam"],  meta["mean_diam_m"],  rel_tol=FLOAT_RTOL), "mean_diam mismatch"
    assert math.isclose(stats["mean_depth"], meta["mean_depth_m"], rel_tol=FLOAT_RTOL), "mean_depth mismatch"


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression test helper")
    parser.add_argument(
        "--generate-baseline", action="store_true",
        help="Generate (or overwrite) the baseline snapshot in tests/baseline/.",
    )
    args = parser.parse_args()

    if args.generate_baseline:
        generate_baseline()
    else:
        print("Running regression checks ...")
        test_tile17_crater_count()
        test_tile17_catalog_rows()
        test_tile17_stats_summary()
        print("All regression checks passed.")
