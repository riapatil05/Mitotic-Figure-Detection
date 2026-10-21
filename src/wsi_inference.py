"""Core WSI coordinate and detection utilities."""

import numpy as np


def box_iou(box_a, box_b):
    """Compute IoU for xyxy boxes."""
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def tile_box_to_slide(box, offset_x, offset_y):
    """Map tile-local xyxy coordinates to global slide coordinates."""
    x1, y1, x2, y2 = map(float, box)
    return [
        x1 + offset_x,
        y1 + offset_y,
        x2 + offset_x,
        y2 + offset_y,
    ]


def global_nms(detections, iou_threshold=0.5):
    """
    Greedy class-aware NMS on global slide coordinates.

    Rows: [x1, y1, x2, y2, confidence, class_id]
    Returns rows sorted by descending confidence.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")

    if len(detections) == 0:
        return []

    detections = np.asarray(detections, dtype=float)
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError("Expected detections with shape (N, 6)")

    kept = []

    for class_id in np.unique(detections[:, 5]):
        class_dets = detections[detections[:, 5] == class_id]
        order = np.argsort(-class_dets[:, 4])

        while len(order):
            best_idx = order[0]
            best = class_dets[best_idx]
            kept.append(best)

            remaining = order[1:]
            order = np.array([
                idx for idx in remaining
                if box_iou(best[:4], class_dets[idx, :4]) <= iou_threshold
            ], dtype=int)

    return sorted(kept, key=lambda row: -row[4])
