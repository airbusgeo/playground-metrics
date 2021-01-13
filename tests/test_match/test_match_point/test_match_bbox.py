import numpy as np
import pytest
from playground_metrics.match_detections import MatchEngineEuclideanDistance, MatchEnginePointInBox, MatchEngineConstantBox
from playground_metrics.utils.geometry_utils import convert_to_bounding_box

from tests.resources.reference_functions import naive_compute_threshold_distance_similarity_matrix, \
    sort_detection_by_confidence, naive_compute_point_in_box_distance_similarity_matrix, \
    naive_compute_constant_box_similarity_matrix

detections = np.concatenate((10 * np.array([[14.5, 0, 26, 5],
                                            [34, 41, 36, 43],
                                            [10, 40.2, 21, 45],
                                            [19, 41, 24, 46.6],
                                            [0, 0, 9, 5],
                                            [11, 6, 20, 11],
                                            [23, 13, 29, 18],
                                            [1, 3, 51, 38]]),
                             np.array([[0.9],
                                       [0.75],
                                       [0.752],
                                       [0.753],
                                       [0.7],
                                       [0.5],
                                       [0.25],
                                       [0.1]])), axis=1)

gt = 10 * np.array([[5, 2, 15, 9],
                    [18, 8.5, 26, 15.5],
                    [6, 16, 36, 23],
                    [33, 39, 34.2, 41.2],
                    [9.5, 40.5, 22.5, 46],
                    [20.2, 42, 24, 46]])

detections = convert_to_bounding_box(detections)
gt = convert_to_bounding_box(gt)

gt_mean_area = np.array([det[0].area for det in gt]).mean()


@pytest.fixture(params=[10 * i for i in range(1, 100, 4)])
def th(request):
    return request.param


class TestMatchEngineBboxConstantBox:
    def test_similarity(self, th):
        matcher = MatchEngineConstantBox(0.5, 'coco', th)
        ref_iou = naive_compute_constant_box_similarity_matrix(sort_detection_by_confidence(detections), gt, th)
        iou = matcher.compute_similarity_matrix(detections, gt)
        print(iou)
        print(ref_iou)
        assert np.all(iou == ref_iou)


class TestMatchEngineBboxPointInBox:
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


class TestMatchEngineBboxEuclidean:
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
