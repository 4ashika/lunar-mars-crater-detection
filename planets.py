"""
planets.py  --  per-body configuration for the Lunar/Mars Crater Detection system.

Detector implementations live in app.py and modules; this file holds per-planet defaults.
Moon defaults are tuned against the labeled validation tiles.
"""
import os

_BASE = os.path.dirname(os.path.abspath(__file__))

PLANETS = {
    "Moon": {
        "display_name": "Moon (Chandrayaan-2 OHRC)",
        "dataset_dir": os.path.join(
            _BASE,
            "Lunar Crater Detection -using Chandrayaan 2 OHRC imagery-.v1i.yolov8",
        ),
        "splits": ("train", "valid"),
        "default_res_m_px": 0.25,
        "default_sun_elev": 30.0,
        "default_min_radius": 6,
        "default_max_radius": 85,
        "default_param2": 14,
        "default_min_dist": 10,
        "default_max_detections": 80,
        "default_hough_param1": 220,
        "default_clahe_clip": 2.0,
        "size_bins": [0, 4, 8, 12, 16, 20, 30, 45, 60, 90],
        "size_labels": ["<4", "4-8", "8-12", "12-16", "16-20", "20-30", "30-45", "45-60", "60+"],
        # Detection annotation colors - Moon stays purple (original COLOR_PURPLE)
        "detection_color_bgr": (220, 80, 160),
        "detection_color_hex": "#A855F7",
        "outline_color_bgr": None,
        "depth_always_low_confidence": False,
    },
    "Mars": {
        "display_name": "Mars (Roboflow Dataset)",
        "dataset_dir": os.path.join(
            _BASE,
            "Mars-Craters.v1-mars-craters-v1.yolov8",
        ),
        "splits": ("train", "valid", "test"),
        "default_res_m_px": 5.0,
        "default_sun_elev": 45.0,
        "default_min_radius": 10,
        # Keep the candidate cap at the validated value: raising it increased false
        # positives without recovering additional matches on the Mars validation tiles.
        "default_max_radius": 85,
        "default_param2": 34,
        "default_min_dist": 20,
        "default_max_detections": 45,
        "default_hough_param1": 130,
        "default_clahe_clip": 2.0,
        "mars_v2_blur_sigma_scale": 1.0,
        "mars_v2_arc_coverage": 0.6,
        "mars_v2_canny_ratio": 0.6,
        "mars_v2_min_radius": 12,
        "mars_v2_param2": 28,
        "size_bins": [0, 50, 100, 200, 400, 800, 1500, 3000, 5000],
        "size_labels": ["<50", "50-100", "100-200", "200-400", "400-800", "800-1500", "1500-3000", "3000+"],
        # Detection annotation colors - bright yellow #FFD400 with 1px black outline for contrast
        "detection_color_bgr": (0, 212, 255),
        "detection_color_hex": "#FFD400",
        "outline_color_bgr": (0, 0, 0),
        # Mars tiles are shaded/oblique - shadow-depth is unreliable by default
        "depth_always_low_confidence": True,
    },
}


def get_planet(name: str) -> dict:
    if name not in PLANETS:
        raise KeyError(f"Unknown planet '{name}'. Available: {list(PLANETS.keys())}")
    return PLANETS[name]


def planet_names() -> list:
    return list(PLANETS.keys())
