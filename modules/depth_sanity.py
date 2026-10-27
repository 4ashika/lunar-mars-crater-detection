"""
modules/depth_sanity.py
Adds a depth sanity flag to each crater dict without touching estimate_shadow_depth() in app.py.

Physically realistic d/D ratios for fresh craters: 0.10-0.20 (simple) / 0.05-0.25 (all).
Outside that range the shadow estimate is likely unreliable (very shallow fill,
oblique illumination, or interior in total shadow).
"""


DD_LOW  = 0.05
DD_HIGH = 0.25


def flag_depth_confidence(craters: list,
                           dd_low: float = DD_LOW,
                           dd_high: float = DD_HIGH,
                           force_low_confidence: bool = False) -> list:
    """
    Add 'depth_confidence' key to each crater dict.

    'depth_confidence' is:
      'OK'           -- d/D in [dd_low, dd_high]
      'low_dd_low'   -- d/D < dd_low  (very shallow / no shadow)
      'low_dd_high'  -- d/D > dd_high (unrealistically deep; possible fill/glare)

    Does NOT modify the depth_m or d2d values.
    """
    for c in craters:
        if force_low_confidence:
            c["depth_confidence"] = "low_planet_default"
            continue
        dd = c.get("d2d", 0.0)
        if dd < dd_low:
            c["depth_confidence"] = "low_dd_low"
        elif dd > dd_high:
            c["depth_confidence"] = "low_dd_high"
        else:
            c["depth_confidence"] = "OK"
    return craters


def filter_high_confidence(craters: list) -> list:
    """Return only craters with depth_confidence == 'OK'."""
    return [c for c in craters if c.get("depth_confidence") == "OK"]
