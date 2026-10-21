"""Tile-based whole-slide inference for mitosis detection.

Coordinates in output CSV are level-0 slide pixels.
This module does not claim clinical validity.
"""

from pathlib import Path
import csv
import math

import cv2
import numpy as np
import openslide
from PIL import Image
from ultralytics import YOLO

from wsi_inference import global_nms, tile_box_to_slide


def tissue_mask_from_thumbnail(thumbnail, saturation_threshold=15,
                               brightness_threshold=245):
    """Simple tissue heuristic; may need tuning for different stains/scanners."""
    rgb = np.asarray(thumbnail.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    brightness = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    return (saturation > saturation_threshold) & (
        brightness < brightness_threshold
    )


def run_wsi_inference(
    slide_path,
    checkpoint_path,
    output_dir,
    confidence=0.35,
    tile_size=640,
    stride=512,
    prediction_iou=0.7,
    global_nms_iou=0.5,
    thumbnail_max_size=2048,
    device=0,
):
    """Run tiled detector inference and export slide-coordinate detections."""
    slide_path = Path(slide_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not slide_path.exists():
        raise FileNotFoundError(slide_path)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(checkpoint_path)
    if tile_size <= 0 or stride <= 0 or stride > tile_size:
        raise ValueError("Require tile_size > 0 and 0 < stride <= tile_size")

    slide = openslide.OpenSlide(str(slide_path))
    try:
        width, height = slide.dimensions
        mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)

        thumb = slide.get_thumbnail((thumbnail_max_size, thumbnail_max_size))
        mask = tissue_mask_from_thumbnail(thumb)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_path = output_dir / "tissue_mask.png"
        mask_image.save(mask_path)

        # Map level-0 tile centers into thumbnail coordinates for tissue filtering.
        thumb_w, thumb_h = thumb.size
        scale_x = thumb_w / width
        scale_y = thumb_h / height

        model = YOLO(str(checkpoint_path))
        raw_detections = []
        processed_tiles = 0
        skipped_tiles = 0

        for y in range(0, height, stride):
            for x in range(0, width, stride):
                tile_w = min(tile_size, width - x)
                tile_h = min(tile_size, height - y)

                if tile_w <= 0 or tile_h <= 0:
                    continue

                # Skip tiles whose center is background in the thumbnail mask.
                cx = min(thumb_w - 1, max(0, int((x + tile_w / 2) * scale_x)))
                cy = min(thumb_h - 1, max(0, int((y + tile_h / 2) * scale_y)))

                if not mask[cy, cx]:
                    skipped_tiles += 1
                    continue

                region = slide.read_region((x, y), 0, (tile_w, tile_h)).convert("RGB")

                # Pad edge tiles to the detector input tile size.
                tile = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
                tile.paste(region, (0, 0))

                result = model.predict(
                    source=np.asarray(tile),
                    imgsz=tile_size,
                    conf=confidence,
                    iou=prediction_iou,
                    max_det=300,
                    device=device,
                    verbose=False,
                )[0]

                processed_tiles += 1

                if result.boxes is None or len(result.boxes) == 0:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for box, score, cls in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = map(float, box)

                    # Ignore detections entirely inside padded edge regions.
                    x1 = min(max(x1, 0.0), float(tile_w))
                    x2 = min(max(x2, 0.0), float(tile_w))
                    y1 = min(max(y1, 0.0), float(tile_h))
                    y2 = min(max(y2, 0.0), float(tile_h))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    slide_box = tile_box_to_slide(
                        [x1, y1, x2, y2], x, y
                    )

                    raw_detections.append([
                        *slide_box, float(score), int(cls), int(x), int(y)
                    ])

        # Save raw detections before global duplicate suppression.
        raw_csv = output_dir / "detections_raw.csv"
        with raw_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "x1", "y1", "x2", "y2", "confidence",
                "class_id", "tile_x", "tile_y"
            ])
            writer.writerows(raw_detections)

        # Apply global NMS on [x1,y1,x2,y2,confidence,class_id].
        nms_input = [row[:6] for row in raw_detections]
        final_detections = global_nms(
            nms_input, iou_threshold=global_nms_iou
        )

        final_csv = output_dir / "detections.csv"
        with final_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "x1", "y1", "x2", "y2", "confidence", "class_id"
            ])
            writer.writerows(final_detections)

        # Create an overview image and overlay at thumbnail resolution.
        overlay = np.asarray(thumb.convert("RGB")).copy()
        for row in final_detections:
            x1, y1, x2, y2, score, cls = row
            tx1 = int(round(x1 * scale_x))
            ty1 = int(round(y1 * scale_y))
            tx2 = int(round(x2 * scale_x))
            ty2 = int(round(y2 * scale_y))

            cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)
            cv2.putText(
                overlay, f"{score:.2f}",
                (tx1, max(12, ty1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1
            )

        overlay_path = output_dir / "detection_overlay.png"
        Image.fromarray(overlay).save(overlay_path)

        summary = {
            "slide_path": str(slide_path),
            "slide_dimensions_level0": [width, height],
            "level_count": slide.level_count,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "confidence_threshold": confidence,
            "tile_size": tile_size,
            "stride": stride,
            "prediction_nms_iou": prediction_iou,
            "global_nms_iou": global_nms_iou,
            "tiles_processed": processed_tiles,
            "tiles_skipped": skipped_tiles,
            "raw_detections": len(raw_detections),
            "final_detections": len(final_detections),
            "density_per_mm2": None,
            "density_note": (
                "Not calculated: detector training images have no verified "
                "physical-scale metadata and detections are not ground-truth validated."
            ),
            "interpretation": (
                "Engineering demonstration on a slide without verified mitosis "
                "annotations; detections are unvalidated."
            ),
        }

        import json
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        return summary

    finally:
        slide.close()
