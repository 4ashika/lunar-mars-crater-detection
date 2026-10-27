import numpy as np

from modules.mission_planner import astar, build_cost_map


def test_astar_routes_around_blocked_cell():
    costs = np.ones((5, 5), dtype=float)
    costs[2, 2] = np.inf

    path = astar(costs, (0, 0), (4, 4))

    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    assert (2, 2) not in path


def test_astar_returns_empty_when_no_path_exists():
    costs = np.ones((3, 3), dtype=float)
    costs[:, 1] = np.inf

    assert astar(costs, (1, 0), (1, 2)) == []


def test_astar_can_choose_from_multiple_safe_destinations():
    costs = np.ones((5, 5), dtype=float)
    costs[2, 2] = np.inf

    path = astar(costs, (0, 0), [(4, 4), (0, 4)], turn_penalty=0.5)

    assert path[0] == (0, 0)
    assert path[-1] in {(4, 4), (0, 4)}
    assert (2, 2) not in path


def test_ellipse_crater_uses_major_axis_for_conservative_obstacle():
    crater = {"x": 20, "y": 20, "radius": 5, "axis_a": 20, "axis_b": 8}

    result = build_cost_map([crater], 40, 40, m_per_px=1.0, halo_m=0)

    # Major semi-axis is 10 m. Geometric-mean radius (previously 6.3 m) would
    # incorrectly leave this point traversable.
    assert result["blocked"][20, 29]
    assert not np.isfinite(result["costs"][0, 20])  # tile-edge safety buffer


def test_rover_footprint_expands_crater_keepout():
    crater = {"x": 32, "y": 32, "radius": 5}

    base = build_cost_map([crater], 64, 64, m_per_px=1.0, halo_m=0)
    rover = build_cost_map([crater], 64, 64, m_per_px=1.0, halo_m=0, rover_radius_m=4.0)

    assert not base["blocked"][32, 42]
    assert rover["blocked"][32, 41]
    assert rover["blocked"].sum() > base["blocked"].sum()
