import numpy as np
from wsi_inference import box_iou, tile_box_to_slide, global_nms


def test_tile_box_to_slide():
    assert tile_box_to_slide([10, 20, 30, 40], 640, 1280) == [
        650.0, 1300.0, 670.0, 1320.0
    ]


def test_box_iou_identical_boxes():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_global_nms_suppresses_duplicate():
    detections = [
        [100, 100, 140, 140, 0.91, 0],
        [103, 102, 143, 142, 0.82, 0],
        [155, 105, 195, 145, 0.88, 0],
        [300, 300, 340, 340, 0.77, 0],
    ]
    result = global_nms(detections, iou_threshold=0.5)
    assert len(result) == 3
    assert np.isclose(result[0][4], 0.91)


def test_global_nms_keeps_different_classes():
    detections = [
        [100, 100, 140, 140, 0.91, 0],
        [100, 100, 140, 140, 0.85, 1],
    ]
    assert len(global_nms(detections, 0.5)) == 2
