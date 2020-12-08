import numpy as np
from playground_metrics.match_detections import MatchEngineIoU
from playground_metrics.utils.geometry_utils import convert_to_bounding_box

from tests.resources.reference_functions import naive_compute_IoU_matrix, sort_detection_by_confidence

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


class TestMatchEngineBbox:
    def test_rtree_iou_matrix(self):
        matcher = MatchEngineIoU(0.1, 'coco')
        ref_IoU = naive_compute_IoU_matrix(sort_detection_by_confidence(detections), gt)
        IoU = matcher.compute_similarity_matrix(detections, gt)
        print(IoU)
        print(ref_IoU)
        assert np.all(IoU == ref_IoU)

    def test_match_non_unitary_at_001(self):
        matcher = MatchEngineIoU(0.01, 'non-unitary')
        assert np.all(matcher.match(detections, gt) == np.array([[1, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 1, 1],
                                                                 [0, 0, 0, 0, 1, 1],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [1, 0, 0, 0, 0, 0],
                                                                 [1, 1, 0, 0, 0, 0],
                                                                 [0, 1, 1, 0, 0, 0],
                                                                 [1, 1, 1, 0, 0, 0]]))

    def test_match_non_unitary_at_005(self):
        matcher = MatchEngineIoU(0.05, 'non-unitary')
        assert np.all(matcher.match(detections, gt) == np.array([[0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 1, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [1, 0, 0, 0, 0, 0],
                                                                 [1, 1, 0, 0, 0, 0],
                                                                 [0, 1, 1, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0]]))

    def test_match_non_unitary_at_01(self):
        matcher = MatchEngineIoU(0.1, 'non-unitary')
        assert np.all(matcher.match(detections, gt) == np.array([[0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 1, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [1, 0, 0, 0, 0, 0],
                                                                 [1, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0]]))

    def test_match_coco_at_001(self):
        matcher = MatchEngineIoU(0.01, 'coco')
        assert np.all(matcher.match(detections, gt) == np.array([[1, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 1, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0]]))

    def test_match_coco_at_005(self):
        matcher = MatchEngineIoU(0.05, 'coco')
        assert np.all(matcher.match(detections, gt) == np.array([[0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [1, 0, 0, 0, 0, 0],
                                                                 [0, 1, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0]]))

    def test_match_coco_at_01(self):
        matcher = MatchEngineIoU(0.1, 'coco')
        assert np.all(matcher.match(detections, gt) == np.array([[0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [1, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0]]))

    def test_match_xview_at_001(self):
        matcher = MatchEngineIoU(0.01, 'xview')
        assert np.all(matcher.match(detections, gt) == np.array([[1, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 1, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0]]))

    def test_match_xview_at_005(self):
        matcher = MatchEngineIoU(0.05, 'xview')
        assert np.all(matcher.match(detections, gt) == np.array([[0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [1, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 1, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0]]))

    def test_match_xview_at_01(self):
        matcher = MatchEngineIoU(0.1, 'xview')
        assert np.all(matcher.match(detections, gt) == np.array([[0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 1],
                                                                 [0, 0, 0, 0, 1, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [1, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 1, 0, 0, 0]]))


class TestMatchEngineBboxIIoU:
    def test_rtree_iou_matrix(self):
        matcher = MatchEngineIoU(0.1, 'coco')
        ref_IoU = naive_compute_IoU_matrix(sort_detection_by_confidence(detections), gt) * \
            (gt_mean_area / np.array([det[0].area for det in gt]))
        IoU = matcher.compute_similarity_matrix(detections, gt, label_mean_area=gt_mean_area)
        print(IoU)
        print(ref_IoU)
        assert np.all(IoU == ref_IoU)

    def test_match_coco_at_001(self):
        matcher = MatchEngineIoU(0.01, 'coco')
        assert np.all(matcher.match(detections, gt, label_mean_area=gt_mean_area) == np.array([[1, 0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 1],
                                                                                               [0, 0, 0, 0, 1, 0],
                                                                                               [0, 0, 0, 1, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 0],
                                                                                               [0, 1, 0, 0, 0, 0],
                                                                                               [0, 0, 1, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 0]]))

    def test_match_coco_at_005(self):
        matcher = MatchEngineIoU(0.05, 'coco')
        assert np.all(matcher.match(detections, gt, label_mean_area=gt_mean_area) == np.array([[0, 0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 1],
                                                                                               [0, 0, 0, 0, 1, 0],
                                                                                               [0, 0, 0, 1, 0, 0],
                                                                                               [1, 0, 0, 0, 0, 0],
                                                                                               [0, 1, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 0]]))

    def test_match_coco_at_01(self):
        matcher = MatchEngineIoU(0.1, 'coco')
        assert np.all(matcher.match(detections, gt, label_mean_area=gt_mean_area) == np.array([[0, 0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 1],
                                                                                               [0, 0, 0, 0, 1, 0],
                                                                                               [0, 0, 0, 1, 0, 0],
                                                                                               [1, 0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 0],
                                                                                               [0, 1, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 0]]))

    def test_match_xview_at_001(self):
        matcher = MatchEngineIoU(0.01, 'xview')
        assert np.all(matcher.match(detections, gt, label_mean_area=gt_mean_area) == np.array([[1, 0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 1],
                                                                                               [0, 0, 0, 0, 1, 0],
                                                                                               [0, 0, 0, 1, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 0],
                                                                                               [0, 1, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 0]]))

    def test_match_xview_at_005(self):
        matcher = MatchEngineIoU(0.05, 'xview')
        assert np.all(matcher.match(detections, gt, label_mean_area=gt_mean_area) == np.array([[0, 0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 1],
                                                                                               [0, 0, 0, 0, 1, 0],
                                                                                               [0, 0, 0, 1, 0, 0],
                                                                                               [1, 0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 0],
                                                                                               [0, 1, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 0]]))

    def test_match_xview_at_01(self):
        matcher = MatchEngineIoU(0.1, 'xview')
        assert np.all(matcher.match(detections, gt, label_mean_area=gt_mean_area) == np.array([[0, 0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 1],
                                                                                               [0, 0, 0, 0, 1, 0],
                                                                                               [0, 0, 0, 1, 0, 0],
                                                                                               [1, 0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 0],
                                                                                               [0, 1, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0, 0]]))
