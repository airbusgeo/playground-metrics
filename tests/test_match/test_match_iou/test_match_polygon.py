import numpy as np
from playground_metrics.match_detections import MatchEngineIoU

from tests.resources.reference_functions import bbox_to_polygon, \
    naive_compute_IoU_matrix, sort_detection_by_confidence

from tests.test_match.test_match_iou.test_match_bbox import detections, gt, gt_mean_area

detections = np.array([[bbox_to_polygon(detections[i, :]), detections[i, 1]] for i in range(detections.shape[0])])
gt = np.array([[bbox_to_polygon(gt[i, :])] for i in range(gt.shape[0])])


class TestMatchEnginePolygon:
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
