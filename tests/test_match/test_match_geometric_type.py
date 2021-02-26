# flake8: noqa: F841
from contextlib import contextmanager

import pytest
import numpy as np
from pygeos import box, polygons, points

from playground_metrics.match_detections import MatchEngineIoU, MatchEngineConstantBox, MatchEngineEuclideanDistance, \
    MatchEnginePointInBox


@contextmanager
def not_raises(ExpectedException):
    try:
        yield
    except ExpectedException as err:
        raise AssertionError(
            "Did raise exception {0} when it should not!".format(
                repr(ExpectedException)
            )
        )
    except Exception as err:
        raise AssertionError(
            "An unexpected exception {0} raised.".format(repr(err))
        )


det_bbox = np.array([[box(0, 0, 1, 1), 0.9, 0]], dtype=np.dtype('O'))
det_poly = np.array([[polygons([[0, 0], [0, 1], [1, 1], [1, 0]]), 0.9, 0]], dtype=np.dtype('O'))
det_point = np.array([[points(0, 0), 0.9, 0]], dtype=np.dtype('O'))

gt_bbox = np.array([[box(0, 0, 1, 1), 0]], dtype=np.dtype('O'))
gt_poly = np.array([[polygons([[0, 0], [0, 1], [1, 1], [1, 0]]), 0]], dtype=np.dtype('O'))
gt_point = np.array([[points(0, 0), 0]], dtype=np.dtype('O'))


det_bbox_poly = np.array([[box(0, 0, 1, 1), 0.9, 0],
                          [polygons([[0, 0], [0, 1], [1, 1], [1, 0]]), 0.9, 0]], dtype=np.dtype('O'))
det_bbox_point = np.array([[box(0, 0, 1, 1), 0.9, 0],
                           [points(0, 0), 0.9, 0]], dtype=np.dtype('O'))
det_poly_point = np.array([[polygons([[0, 0], [0, 1], [1, 1], [1, 0]]), 0.9, 0],
                           [points(0, 0), 0.9, 0]], dtype=np.dtype('O'))

gt_bbox_poly = np.array([[box(0, 0, 1, 1), 0],
                         [polygons([[0, 0], [0, 1], [1, 1], [1, 0]]), 0]], dtype=np.dtype('O'))
gt_bbox_point = np.array([[box(0, 0, 1, 1), 0],
                          [points(0, 0), 0]], dtype=np.dtype('O'))
gt_poly_point = np.array([[polygons([[0, 0], [0, 1], [1, 1], [1, 0]]), 0],
                          [points(0, 0), 0]], dtype=np.dtype('O'))


det_bbox_poly_point = np.array([[box(0, 0, 1, 1), 0.9, 0],
                                [polygons([[0, 0], [0, 1], [1, 1], [1, 0]]), 0.9, 0],
                                [points(0, 0), 0.9, 0]], dtype=np.dtype('O'))

gt_bbox_poly_point = np.array([[box(0, 0, 1, 1), 0],
                               [polygons([[0, 0], [0, 1], [1, 1], [1, 0]]), 0],
                               [points(0, 0), 0]], dtype=np.dtype('O'))

det_types = (det_bbox, det_poly, det_point, det_bbox_poly, det_bbox_point, det_poly_point, det_bbox_poly_point)
gt_types = (gt_bbox, gt_poly, gt_point, gt_bbox_poly, gt_bbox_point, gt_poly_point, gt_bbox_poly_point)


_allowed = {
    'MatchEngineIoU': (det_bbox, det_poly, gt_bbox, gt_poly, det_bbox_poly, gt_bbox_poly),
    'MatchEngineConstantBox': (det_bbox, det_poly, det_point, gt_bbox, gt_poly, gt_point,
                               det_bbox_poly, det_bbox_point, det_poly_point,
                               gt_bbox_poly, gt_bbox_point, gt_poly_point,
                               det_bbox_poly_point, gt_bbox_poly_point),
    'MatchEngineEuclideanDistance': (det_bbox, det_poly, det_point, gt_bbox, gt_poly, gt_point,
                                     det_bbox_poly, det_bbox_point, det_poly_point,
                                     gt_bbox_poly, gt_bbox_point, gt_poly_point,
                                     det_bbox_poly_point, gt_bbox_poly_point),
    'MatchEnginePointInBox': (det_bbox, det_poly, det_point, gt_bbox, gt_poly,
                              det_bbox_poly, det_bbox_point, det_poly_point,
                              gt_bbox_poly,
                              det_bbox_poly_point)
}
_match_engine_args = {
    'MatchEngineIoU': (0.5, 'coco'),
    'MatchEngineConstantBox': (0.5, 'coco', 10),
    'MatchEngineEuclideanDistance': (10, 'coco'),
    'MatchEnginePointInBox': ('coco', )
}


def _get_context(det, gt, match_engine_class):
    if any([np.all(det == ndarray) for ndarray in _allowed[match_engine_class.__name__]]) and \
            any([np.all(gt == ndarray) for ndarray in _allowed[match_engine_class.__name__]]):
        return not_raises(TypeError)
    else:
        return pytest.raises(TypeError)


class TestMatchEngineTyping:
    @staticmethod
    def _make_type_test(match_engine_class):
        for det in det_types:
            for gt in gt_types:
                ctx = _get_context(det, gt, match_engine_class)
                with ctx:
                    match_engine = match_engine_class(*_match_engine_args[match_engine_class.__name__])
                    match_engine.match(det, gt)

    def test_iou(self):
        self._make_type_test(MatchEngineIoU)

    def test_point(self):
        self._make_type_test(MatchEngineEuclideanDistance)

    def test_point_in_box(self):
        self._make_type_test(MatchEnginePointInBox)

    def test_constant_box(self):
        self._make_type_test(MatchEngineConstantBox)
