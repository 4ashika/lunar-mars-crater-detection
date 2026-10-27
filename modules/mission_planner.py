"""Cost mapping, landing-site selection, and rover routing for crater tiles."""

from __future__ import annotations

import heapq
import math
from collections import deque

import cv2
import numpy as np


# Keep the Python heap-based search responsive on high-resolution Mars tiles.
# At coarser ground scales the grid still follows the source-pixel resolution;
# this cap only coarsens unusually large map footprints.
MAX_COST_GRID_CELLS = 100_000


def build_cost_map(
    craters: list[dict],
    image_width: int,
    image_height: int,
    m_per_px: float,
    rim_buffer_m: float = 1.0,
    halo_m: float = 5.0,
    cell_m: float | None = None,
    rover_radius_m: float = 0.0,
) -> dict:
    """Build a traversability grid, expanding crater hazards by rover radius."""
    scale = max(float(m_per_px), 1e-6)
    if cell_m is None:
        cell_m = max(1.0, scale)
        footprint_area_m2 = image_width * image_height * scale * scale
        if footprint_area_m2 / (cell_m * cell_m) > MAX_COST_GRID_CELLS:
            cell_m = max(cell_m, math.sqrt(footprint_area_m2 / MAX_COST_GRID_CELLS))
    else:
        cell_m = max(float(cell_m), 1e-6)
    grid_h = max(1, int(math.ceil(image_height * scale / cell_m)))
    grid_w = max(1, int(math.ceil(image_width * scale / cell_m)))
    sx, sy = grid_w / image_width, grid_h / image_height
    costs = np.ones((grid_h, grid_w), dtype=np.float32)
    blocked = np.zeros((grid_h, grid_w), dtype=np.uint8)
    crater_interiors = np.zeros_like(blocked)
    landing_obstacles = np.zeros_like(blocked)
    high_cost = np.zeros_like(blocked)
    medium_cost = np.zeros_like(blocked)
    crater_rows = []

    for crater in craters:
        cx_img = float(crater["x"])
        cy_img = float(crater["y"])
        # Ellipse detections store ``radius`` as the geometric mean of their
        # axes. That is useful for reporting, but unsafe for navigation: it can
        # leave part of an elongated crater traversable. Use the major semi-axis
        # as a conservative circular footprint whenever axis lengths are known.
        major_radius_px = max(
            float(crater.get("radius", 0.0)),
            0.5 * float(crater.get("axis_a", 0.0)),
            0.5 * float(crater.get("axis_b", 0.0)),
        )
        if crater.get("bbox") and not crater.get("axis_a") and not crater.get("axis_b"):
            x1, y1, x2, y2 = map(float, crater["bbox"])
            major_radius_px = max(major_radius_px, 0.5 * math.hypot(x2 - x1, y2 - y1))
        radius_m = max(0.5 * scale, major_radius_px * scale)
        cx, cy = int(round(cx_img * sx)), int(round(cy_img * sy))
        vehicle_radius = max(0.0, float(rover_radius_m))
        radius_cells = max(1, int(math.ceil((radius_m + vehicle_radius) / cell_m)))
        rim_m = max(0.0, float(rim_buffer_m)) + 0.25 * radius_m
        high_cells = max(radius_cells + 1, int(math.ceil((radius_m + vehicle_radius + rim_m) / cell_m)))
        halo_cells = max(high_cells, int(math.ceil((radius_m + vehicle_radius + rim_m + max(0.0, halo_m)) / cell_m)))

        sharpness = float(crater.get("freshness", 0.0))
        if not crater.get("freshness"):
            sharpness = max(0.0, min(1.0, (4 - int(crater.get("deg_class", 4))) / 3.0))
        high_value = 50.0 * (1.0 + sharpness)
        medium_value = 5.0 * (1.0 + 0.5 * sharpness)

        cv2.circle(crater_interiors, (cx, cy), max(1, int(math.ceil(radius_m / cell_m))), 1, thickness=-1)
        landing_radius_cells = max(1, int(math.ceil((radius_m + rim_m) / cell_m)))
        cv2.circle(landing_obstacles, (cx, cy), landing_radius_cells, 1, thickness=-1)
        cv2.circle(blocked, (cx, cy), radius_cells, 1, thickness=-1)
        cv2.circle(high_cost, (cx, cy), high_cells, 1, thickness=-1)
        cv2.circle(medium_cost, (cx, cy), halo_cells, 1, thickness=-1)
        crater_rows.append({
            "x": cx_img * scale,
            "y": cy_img * scale,
            "radius_m": radius_m,
            "rim_buffer_m": rim_m,
            "rover_radius_m": vehicle_radius,
            "category": crater.get("category", "Unknown"),
            "source": crater,
        })

        # Higher hazard classes win where crater buffers overlap.
        costs[medium_cost > 0] = np.maximum(costs[medium_cost > 0], medium_value)
        costs[high_cost > 0] = np.maximum(costs[high_cost > 0], high_value)

    # Keep the cost layers separate so the UI can explain the route visually.
    high_cost[blocked > 0] = 0
    medium_cost[(high_cost > 0) | (blocked > 0)] = 0
    costs[blocked > 0] = np.inf
    # Image edges are not valid rover terrain. Reserve a ~2 m border so A* does
    # not exploit the tile boundary as a zero-risk shortcut.
    edge_cells = max(1, int(math.ceil(2.0 / cell_m)))
    if grid_h > 2 * edge_cells and grid_w > 2 * edge_cells:
        costs[:edge_cells, :] = np.inf
        costs[-edge_cells:, :] = np.inf
        costs[:, :edge_cells] = np.inf
        costs[:, -edge_cells:] = np.inf
    hazard_free = ((medium_cost == 0) & (high_cost == 0) & (blocked == 0) & np.isfinite(costs)).astype(np.uint8)
    landing_clearance = cv2.distanceTransform(
        np.pad(1 - (landing_obstacles > 0).astype(np.uint8), 1, mode="constant", constant_values=0),
        cv2.DIST_L2, 5,
    )[1:-1, 1:-1]
    return {
        "costs": costs,
        "blocked": blocked.astype(bool),
        "crater_interiors": crater_interiors.astype(bool),
        "high_cost": high_cost.astype(bool),
        "medium_cost": medium_cost.astype(bool),
        "landing_clearance": landing_clearance,
        "hazard_free": hazard_free,
        "craters": crater_rows,
        "cell_m": cell_m,
        "image_to_grid": (sx, sy),
        "image_size": (image_width, image_height),
    }


def choose_landing_point(cost_map: dict, footprint_radius_m: float = 2.0, margin_m: float = 1.0) -> dict | None:
    """Find the maximum-clearance hazard-free grid cell, with best-available fallback."""
    costs = cost_map["costs"]
    cell_m = cost_map["cell_m"]
    safe = cost_map["hazard_free"].copy()
    fallback = False
    if not np.any(safe):
        safe = np.isfinite(costs).astype(np.uint8)
        fallback = True
    if not np.any(safe):
        return None

    # Treat the tile edge as an obstacle so the footprint stays inside the image.
    clearance_cells = cost_map.get("landing_clearance")
    if clearance_cells is None:
        padded = np.pad(safe, 1, mode="constant", constant_values=0)
        clearance_cells = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    row, col = np.unravel_index(int(np.argmax(clearance_cells)), clearance_cells.shape)
    clearance_m = float(clearance_cells[row, col] * cell_m)
    required_m = max(0.0, float(footprint_radius_m)) + max(0.0, float(margin_m))
    sx, sy = cost_map["image_to_grid"]
    image_width, image_height = cost_map["image_size"]
    image_x = min(image_width - 1.0, max(0.0, (col + 0.5) / sx))
    image_y = min(image_height - 1.0, max(0.0, (row + 0.5) / sy))
    return {
        "grid": (int(row), int(col)),
        "image_px": (image_x, image_y),
        "clearance_m": clearance_m,
        "required_m": required_m,
        "satisfies_footprint": clearance_m >= required_m,
        "fallback": fallback,
        "clearance_grid": clearance_cells,
    }


def astar(
    cost_grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int] | list[tuple[int, int]],
    turn_penalty: float = 0.0,
) -> list[tuple[int, int]]:
    """8-connected A*, with an optional penalty for sharp heading changes."""
    grid = np.asarray(cost_grid, dtype=float)
    rows, cols = grid.shape
    start = (int(start[0]), int(start[1]))
    goals = {(int(point[0]), int(point[1])) for point in (goal if isinstance(goal, list) else [goal])}
    goals = {point for point in goals if 0 <= point[0] < rows and 0 <= point[1] < cols and math.isfinite(grid[point])}
    if not goals:
        return []
    for point in (start,):
        r, c = point
        if not (0 <= r < rows and 0 <= c < cols) or not math.isfinite(grid[r, c]):
            return []
    minimum_cost = max(1e-6, float(np.min(grid[np.isfinite(grid)])))

    # Precompute distance to the nearest goal once. The destination can have
    # many perimeter candidates; recomputing a min over every candidate for
    # every heap pop made large Mars tiles needlessly slow.
    goal_mask = np.ones((rows, cols), dtype=np.uint8)
    for goal_row, goal_col in goals:
        goal_mask[goal_row, goal_col] = 0
    heuristic_grid = cv2.distanceTransform(goal_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * minimum_cost

    def heuristic(point):
        # Euclidean distance is admissible for this 8-connected grid.
        return float(heuristic_grid[point[0], point[1]])

    # Heading is part of the search state when smoothing is enabled. The
    # penalty approximates a rover's finite turning radius; it is not a
    # kinematic vehicle simulation.
    turn_penalty = max(0.0, float(turn_penalty))
    directions = ((0, 1, 1.0), (1, 1, math.sqrt(2)), (1, 0, 1.0),
                  (1, -1, math.sqrt(2)), (0, -1, 1.0), (-1, -1, math.sqrt(2)),
                  (-1, 0, 1.0), (-1, 1, math.sqrt(2)))
    initial = (start[0], start[1], 8)
    start_h = heuristic(start)
    # Prefer the more goal-directed node when f scores tie. This preserves A*
    # optimality while reducing broad exploration in large, weighted maps.
    frontier = [(start_h, start_h, 0.0, initial)]
    came_from = {}
    best_cost = {initial: 0.0}
    while frontier:
        _, _, path_cost, state = heapq.heappop(frontier)
        if path_cost > best_cost.get(state, math.inf):
            continue
        row, col, heading = state
        current = (row, col)
        if current in goals:
            path = [current]
            while state in came_from:
                state = came_from[state]
                path.append((state[0], state[1]))
            return path[::-1]
        for direction, (dr, dc, distance) in enumerate(directions):
            nr, nc = row + dr, col + dc
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols or not math.isfinite(grid[nr, nc]):
                continue
            # Prevent diagonal corner-cutting between two impassable cells.
            if dr and dc and (not math.isfinite(grid[current[0] + dr, current[1]]) or
                              not math.isfinite(grid[current[0], current[1] + dc])):
                continue
            step_cost = distance * 0.5 * (grid[current] + grid[nr, nc])
            if heading != 8 and turn_penalty:
                turn_steps = min((direction - heading) % 8, (heading - direction) % 8)
                step_cost += turn_penalty * (turn_steps / 4.0)
            new_cost = path_cost + step_cost
            nxt_state = (nr, nc, direction)
            if new_cost < best_cost.get(nxt_state, math.inf):
                best_cost[nxt_state] = new_cost
                came_from[nxt_state] = state
                next_h = heuristic((nr, nc))
                heapq.heappush(frontier, (new_cost + next_h, next_h, new_cost, nxt_state))
    return []


def reachable_component(cost_grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Return the traversable 8-connected component containing ``start``.

    This quick connectivity pass avoids running a full weighted A* search for
    a destination separated from the start by a wall of blocked crater cells.
    A* still performs the stricter diagonal corner-cut check on reachable goals.
    """
    grid = np.asarray(cost_grid, dtype=float)
    rows, cols = grid.shape
    row, col = int(start[0]), int(start[1])
    component = np.zeros((rows, cols), dtype=bool)
    if not (0 <= row < rows and 0 <= col < cols) or not math.isfinite(grid[row, col]):
        return component
    finite = np.isfinite(grid)
    component[row, col] = True
    frontier = deque([(row, col)])
    directions = ((0, 1), (1, 1), (1, 0), (1, -1),
                  (0, -1), (-1, -1), (-1, 0), (-1, 1))
    while frontier:
        current_row, current_col = frontier.popleft()
        for dr, dc in directions:
            next_row, next_col = current_row + dr, current_col + dc
            if (next_row < 0 or next_col < 0 or next_row >= rows or next_col >= cols
                    or component[next_row, next_col] or not finite[next_row, next_col]):
                continue
            # Match A*'s no-corner-cutting rule exactly so unreachable goals
            # are removed before the weighted heap search begins.
            if dr and dc and (not finite[current_row + dr, current_col]
                              or not finite[current_row, current_col + dc]):
                continue
            component[next_row, next_col] = True
            frontier.append((next_row, next_col))
    return component


def crater_goal_candidates(
    cost_map: dict,
    crater: dict,
    start: tuple[int, int],
    limit: int = 32,
    reachable: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Return accessible points just outside a crater's rim buffer, nearest first.

    Multiple slightly wider rings handle overlapping crater halos that block
    the first sampled ring, while keeping the endpoint close to the selected rim.
    """
    cell_m = cost_map["cell_m"]
    sx, sy = cost_map["image_to_grid"]
    center_x, center_y = crater["x"] / cell_m, crater["y"] / cell_m
    base_radius_cells = (crater["radius_m"] + crater["rim_buffer_m"] + crater.get("rover_radius_m", 0.0)) / cell_m + 1.5
    rows, cols = cost_map["costs"].shape
    points = set()
    # First try the nominal keep-out boundary, then step outward only as needed.
    for extra_cells in (0.0, 1.0, 2.0, 4.0, 6.0, 8.0):
        radius_cells = base_radius_cells + extra_cells
        for index in range(limit):
            angle = 2 * math.pi * index / limit
            row = int(round(center_y + radius_cells * math.sin(angle)))
            col = int(round(center_x + radius_cells * math.cos(angle)))
            if not (0 <= row < rows and 0 <= col < cols):
                continue
            if not np.isfinite(cost_map["costs"][row, col]):
                continue
            if reachable is not None and not reachable[row, col]:
                continue
            points.add((row, col))
    return sorted(points, key=lambda point: math.hypot(point[0] - start[0], point[1] - start[1]))


def route_metrics(path: list[tuple[int, int]], start: tuple[int, int], goal: tuple[int, int], cost_map: dict) -> dict:
    """Calculate route lengths, detour, crater crossings, and rim clearance."""
    cell_m = cost_map["cell_m"]
    straight_m = math.hypot(goal[0] - start[0], goal[1] - start[1]) * cell_m
    if len(path) < 2:
        path_m = 0.0
    else:
        path_m = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])) * cell_m
    detour = ((path_m / straight_m) - 1.0) * 100.0 if straight_m > 0 and path else 0.0
    sx, sy = cost_map["image_to_grid"]
    # Coordinates in the cost-map's physical metre space.
    points_m = [(c * cell_m, r * cell_m) for r, c in path]
    start_m = (start[1] * cell_m, start[0] * cell_m)
    goal_m = (goal[1] * cell_m, goal[0] * cell_m)
    straight_points = [start_m, goal_m]

    def segment_distance(point, a, b):
        vx, vy = b[0] - a[0], b[1] - a[1]
        denom = vx * vx + vy * vy
        t = 0.0 if denom == 0 else max(0.0, min(1.0, ((point[0] - a[0]) * vx + (point[1] - a[1]) * vy) / denom))
        return math.hypot(point[0] - (a[0] + t * vx), point[1] - (a[1] + t * vy))

    def crossed(route_points):
        found = 0
        for crater in cost_map["craters"]:
            intersects = any(
                segment_distance((crater["x"], crater["y"]), a, b) <= crater["radius_m"] + crater.get("rover_radius_m", 0.0)
                for a, b in zip(route_points, route_points[1:])
            ) if len(route_points) >= 2 else False
            found += int(intersects)
        return found

    clearances = []
    for x, y in points_m:
        if cost_map["craters"]:
            clearances.append(min(
                math.hypot(x - c["x"], y - c["y"]) - c["radius_m"] - c["rim_buffer_m"] - c.get("rover_radius_m", 0.0)
                for c in cost_map["craters"]
            ))
    return {
        "path_length_m": path_m,
        "straight_length_m": straight_m,
        "detour_percent": detour,
        "straight_craters_crossed": crossed(straight_points),
        "route_craters_crossed": crossed(points_m),
        "minimum_rim_clearance_m": min(clearances) if clearances else float("nan"),
    }


def grid_to_image(path: list[tuple[int, int]], cost_map: dict) -> list[tuple[int, int]]:
    """Convert row/column grid points to original image pixel coordinates."""
    sx, sy = cost_map["image_to_grid"]
    width, height = cost_map["image_size"]
    return [
        (min(width - 1, max(0, int(round((col + 0.5) / sx)))),
         min(height - 1, max(0, int(round((row + 0.5) / sy)))))
        for row, col in path
    ]


def render_cost_overlay(image_bgr: np.ndarray, cost_map: dict, opacity: float = 0.48) -> np.ndarray:
    """Overlay blocked interiors, high-cost rims, and medium-cost halos."""
    out = image_bgr.copy()
    opacity = min(1.0, max(0.0, float(opacity)))
    width, height = cost_map["image_size"]
    colors = ((cost_map["blocked"], (45, 45, 220)),
              (cost_map["high_cost"], (0, 125, 255)),
              (cost_map["medium_cost"], (0, 220, 255)))
    for mask, color in colors:
        layer = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        if np.any(layer):
            tint = np.empty_like(out)
            tint[:] = color
            out[layer] = cv2.addWeighted(out[layer], 1.0 - opacity, tint[layer], opacity, 0)
    return out
