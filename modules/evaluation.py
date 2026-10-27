"""Box matching helpers for offline detector tuning (not an app tab)."""

from __future__ import annotations

import os


def circle_boxes(craters: list[dict]) -> list[tuple[float, float, float, float]]:
    """Convert detected circles or fitted ellipses to axis-aligned boxes."""
    result = []
    for crater in craters:
        if crater.get("bbox"):
            result.append(tuple(map(float, crater["bbox"])))
        else:
            x, y, radius = float(crater["x"]), float(crater["y"]), float(crater["radius"])
            result.append((x - radius, y - radius, x + radius, y + radius))
    return result


def yolo_boxes(label_path: str, width: int, height: int) -> list[tuple[float, float, float, float]]:
    """Read normalized YOLO labels into pixel-space xyxy boxes."""
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, encoding="utf-8") as labels:
        for line in labels:
            fields = line.split()
            if len(fields) < 5:
                continue
            _, cx, cy, box_w, box_h = map(float, fields[:5])
            boxes.append(((cx - box_w / 2) * width, (cy - box_h / 2) * height,
                          (cx + box_w / 2) * width, (cy + box_h / 2) * height))
    return boxes


def box_iou(a: tuple, b: tuple) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(predictions: list[tuple], ground_truth: list[tuple], threshold: float = 0.5) -> dict:
    """Greedy one-to-one IoU matching, prioritizing the highest overlaps."""
    pairs = sorted(
        ((box_iou(pred, truth), pi, gi)
         for pi, pred in enumerate(predictions)
         for gi, truth in enumerate(ground_truth)),
        reverse=True,
    )
    used_predictions, used_ground_truth, matched_ious = set(), set(), []
    for iou, pi, gi in pairs:
        if iou < threshold:
            break
        if pi not in used_predictions and gi not in used_ground_truth:
            used_predictions.add(pi)
            used_ground_truth.add(gi)
            matched_ious.append(iou)
    tp = len(matched_ious)
    fp, fn = len(predictions) - tp, len(ground_truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "mean_iou": sum(matched_ious) / tp if tp else 0.0}


def aggregate_metrics(rows: list[dict]) -> list[dict]:
    """Pool per-tile counts by planet for offline reports."""
    planets = sorted({row.get("planet", "Overall") for row in rows})
    output = []
    for planet in planets:
        items = [row for row in rows if row.get("planet", "Overall") == planet]
        tp = sum(row["tp"] for row in items)
        fp = sum(row["fp"] for row in items)
        fn = sum(row["fn"] for row in items)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        output.append({
            "Planet": planet, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mean_iou": (sum(row["mean_iou"] * row["tp"] for row in items) / tp) if tp else 0.0,
        })
    return output
