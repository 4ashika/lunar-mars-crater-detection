"""
modules/tile_loader.py
Tile discovery and per-tile config lookup for multi-planet support.
Wraps the existing per-planet dataset layout; never modifies app.py.
"""
import os
import glob
import json


# These are the strongest current demo tiles with both label agreement and a
# reachable largest-crater route at the default settings. The score is a
# browsing aid, not a claim that the detector is accurate (the absolute F1 is
# still low). Remaining tiles keep their deterministic filename order.
DEMO_TILE_ORDER = {
    "Moon": (
        "19_jpg.rf.e62ff00b8f0c28fd8ff53c069c771533.jpg",
        "20_jpg.rf.6a1691350358ddddd656ea0671bcb4e6.jpg",
        "30_jpg.rf.89e64da231c072c64732ce23da385601.jpg",
        "52_jpg.rf.5b2c8a95e9db0c805c9ab46012016e5a.jpg",
        "96_jpg.rf.ba42128a708dbed8a71198ceb4ebf116.jpg",
    ),
    "Mars": (
        "download_jpeg.rf.06974bf7e43c3e9c92a48a452ed9060c.jpg",
        "images_jpeg.rf.02e917ab94a7e6c35cb702576560ce7f.jpg",
        "Perspective_view_of_Acidalia_Planitia_and_Tempe_Terra_jpeg.rf.23a403703e0560b9cedf5db38225a769.jpg",
        "images--1-_jpeg.rf.78d605df1f63054173e9ca762cbec458.jpg",
        "BeerMartianCrater_jpeg.rf.d781b85b48a7193bb2fdb879798c7522.jpg",
    ),
}


def _load_tiles_config(dataset_dir: str) -> dict:
    """Load tiles_config.json from a dataset directory, return {} if missing."""
    path = os.path.join(dataset_dir, "tiles_config.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def discover_planet_images(planet_cfg: dict) -> list:
    """
    Discover all image tiles for the given planet config dict.
    Returns a list of dicts identical in shape to the Moon dicts in app.discover_images(),
    plus extra 'planet' and per-tile 'res_m_px' / 'sun_elev_deg' fields.
    """
    tiles_cfg = _load_tiles_config(planet_cfg["dataset_dir"])
    images = []
    for split in planet_cfg["splits"]:
        img_dir = os.path.join(planet_cfg["dataset_dir"], split, "images")
        lbl_dir = os.path.join(planet_cfg["dataset_dir"], split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for fp in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            fname = os.path.basename(fp)
            lbl = os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))
            num = fname.split("_")[0]
            tc = tiles_cfg.get(fname, {})
            is_mars = planet_cfg.get("display_name", "").startswith("Mars")
            calibrated_scale = tc.get("m_per_px") if is_mars else tc.get(
                "resolution_m_px", planet_cfg["default_res_m_px"]
            )
            # Uncalibrated Mars tiles use the documented fallback for calculations,
            # while retaining a flag so the UI can label those measurements honestly.
            fallback_scale = tc.get("resolution_m_px", planet_cfg["default_res_m_px"])
            images.append(dict(
                path=fp,
                label_path=lbl,
                name=f"Tile {num}",
                filename=fname,
                split=split,
                planet=planet_cfg["display_name"],
                # Per-tile overrides (fall back to planet defaults)
                res_m_px=calibrated_scale if calibrated_scale else fallback_scale,
                scale_calibrated=bool(calibrated_scale),
                sun_elev_deg=tc.get("sun_elev_deg", planet_cfg["default_sun_elev"]),
            ))
    planet_key = "Mars" if planet_cfg.get("display_name", "").startswith("Mars") else "Moon"
    priority = {filename: index + 1 for index, filename in enumerate(DEMO_TILE_ORDER.get(planet_key, ()))}
    images.sort(key=lambda entry: (
        0 if entry["filename"] in priority else 1,
        priority.get(entry["filename"], 0),
        entry["filename"].casefold(),
    ))
    for entry in images:
        entry["recommended_rank"] = priority.get(entry["filename"])
    return images


def get_tile_params(tile: dict, sidebar_res: float, sidebar_sun: float) -> tuple:
    """
    Return (resolution_m_px, sun_elev_deg) for a tile.
    Uses per-tile config values; falls back to sidebar values if not available.
    """
    res = tile.get("res_m_px", sidebar_res)
    sun = tile.get("sun_elev_deg", sidebar_sun)
    return res, sun


def save_tile_scale(dataset_dir: str, filename: str, m_per_px: float) -> None:
    """Persist a Mars tile calibration in tiles_config.json."""
    path = os.path.join(dataset_dir, "tiles_config.json")
    config = _load_tiles_config(dataset_dir)
    tile = config.setdefault(filename, {})
    tile["m_per_px"] = float(m_per_px)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
