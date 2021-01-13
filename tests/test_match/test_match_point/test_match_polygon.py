import numpy as np
import pytest
from playground_metrics.match_detections import MatchEnginePointInBox, MatchEngineConstantBox, \
    MatchEngineEuclideanDistance

from tests.resources.reference_functions import bbox_to_polygon, sort_detection_by_confidence, \
    naive_compute_point_in_box_distance_similarity_matrix, naive_compute_constant_box_similarity_matrix, \
    naive_compute_threshold_distance_similarity_matrix

from tests.test_match.test_match_point.test_match_bbox import detections, gt

detections = np.array([[bbox_to_polygon(detections[i, :]), detections[i, 1]] for i in range(detections.shape[0])])
gt = np.array([[bbox_to_polygon(gt[i, :])] for i in range(gt.shape[0])])


@pytest.fixture(params=[10 * i for i in range(1, 100, 4)])
def th(request):
    return request.param


class TestMatchEnginePolygonConstantBox:
    def test_similarity(self, th):
        matcher = MatchEngineConstantBox(0.5, 'coco', th)
        ref_iou = naive_compute_constant_box_similarity_matrix(sort_detection_by_confidence(detections), gt, th)
        iou = matcher.compute_similarity_matrix(detections, gt)
        print(iou)
        print(ref_iou)
        assert np.all(iou == ref_iou)


class TestMatchEnginePolygonPointInBox:
    def test_similarity(self):
        matcher = MatchEnginePointInBox('coco')
        ref_iou = naive_compute_point_in_box_distance_similarity_matrix(sort_detection_by_confidence(detections), gt)
        iou = matcher.compute_similarity_matrix(detections, gt)
        print(iou)
        print(ref_iou)
        assert np.all(iou[np.logical_not(np.isinf(iou))] == ref_iou[np.logical_not(np.isinf(iou))])

    def test_match_coco(self):
        matcher = MatchEnginePointInBox('coco')
        assert np.all(matcher.match(detections, gt) == np.array([[0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 1, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0]]))

    def test_match_xview(self):
        matcher = MatchEnginePointInBox('coco')
        assert np.all(matcher.match(detections, gt) == np.array([[0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 1, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0]]))


class TestMatchEnginePolygonEuclidean:
    def test_similarity(self, th):
        matcher = MatchEngineEuclideanDistance(th, 'coco')
        ref_iou = naive_compute_threshold_distance_similarity_matrix(sort_detection_by_confidence(detections), gt, th)
        iou = matcher.compute_similarity_matrix(detections, gt)
        print(iou)
        print(ref_iou)
        assert np.all(iou == ref_iou)

    def test_match_coco_at_100(self):
        matcher = MatchEngineEuclideanDistance(100, 'coco')
        assert np.all(matcher.match(detections, gt) == np.array([[0, 1, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 1, 0, 0],
                                                                 [1, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0]]))

    def test_match_coco_at_150(self):
        matcher = MatchEngineEuclideanDistance(150, 'coco')
        assert np.all(matcher.match(detections, gt) == np.array([[0, 1, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 1, 0, 0],
                                                                 [1, 0, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0]]))

    def test_match_coco_at_200(self):
        matcher = MatchEngineEuclideanDistance(200, 'coco')
        assert np.all(matcher.match(detections, gt) == np.array([[0, 1, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 1, 0, 0],
                                                                 [1, 0, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0]]))

    def test_match_xview_at_100(self):
        matcher = MatchEngineEuclideanDistance(100, 'xview')
        assert np.all(matcher.match(detections, gt) == np.array([[0, 1, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 1, 0, 0],
                                                                 [1, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0]]))

    def test_match_xview_at_150(self):
        matcher = MatchEngineEuclideanDistance(150, 'xview')
        assert np.all(matcher.match(detections, gt) == np.array([[0, 1, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 1, 0, 0],
                                                                 [1, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0]]))

    def test_match_xview_at_200(self):
        matcher = MatchEngineEuclideanDistance(200, 'xview')
        assert np.all(matcher.match(detections, gt) == np.array([[0, 1, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 1, 0, 0],
                                                                 [1, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0]]))
