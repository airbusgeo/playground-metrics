import numpy as np

import os.path

from pytest import raises

from playground_metrics.match_detections import MatchEngineIoU, MatchEngineBase
from playground_metrics.utils.geometry_utils import convert_to_bounding_box

from tests.test_match.test_match_iou.test_match_bbox import gt_mean_area
from tests.test_match.test_match_iou.test_match_bbox import detections as detections_bbox
from tests.test_match.test_match_iou.test_match_bbox import gt as gt_bbox
from tests.test_match.test_match_iou.test_match_polygon import detections as detections_poly
from tests.test_match.test_match_iou.test_match_polygon import gt as gt_poly
from tests.test_match.test_match_iou.test_match_polygon import detections as detections_corr_poly
from tests.test_match.test_match_iou.test_match_polygon import gt as gt_corr_poly


class TestMatch:

    def test_metaclass(self):

        with raises(TypeError):
            MatchEngineBase('coco')

    def test_rtree_iou_matrix(self):
        matcher_poly = MatchEngineIoU(0.1, 'coco')
        matcher_bbox = MatchEngineIoU(0.1, 'coco')
        IoU_poly = matcher_poly.compute_similarity_matrix(detections_corr_poly, gt_corr_poly)
        IoU_bbox = matcher_bbox.compute_similarity_matrix(detections_bbox, gt_bbox)
        print(IoU_poly)
        print(IoU_bbox)
        assert np.all(IoU_poly == IoU_bbox)

    def test_match_non_unitary_at_001(self):
        matcher_poly = MatchEngineIoU(0.01, 'non-unitary')
        matcher_bbox = MatchEngineIoU(0.01, 'non-unitary')
        assert np.all(matcher_bbox.match(detections_bbox, gt_bbox) == matcher_poly.match(detections_poly, gt_poly))

    def test_match_non_unitary_at_005(self):
        matcher_poly = MatchEngineIoU(0.05, 'non-unitary')
        matcher_bbox = MatchEngineIoU(0.05, 'non-unitary')
        assert np.all(matcher_bbox.match(detections_bbox, gt_bbox) == matcher_poly.match(detections_poly, gt_poly))

    def test_match_non_unitary_at_01(self):
        matcher_poly = MatchEngineIoU(0.1, 'non-unitary')
        matcher_bbox = MatchEngineIoU(0.1, 'non-unitary')
        assert np.all(matcher_bbox.match(detections_bbox, gt_bbox) == matcher_poly.match(detections_poly, gt_poly))

    def test_match_coco_at_001(self):
        matcher_poly = MatchEngineIoU(0.01, 'coco')
        matcher_bbox = MatchEngineIoU(0.01, 'coco')
        assert np.all(matcher_bbox.match(detections_bbox, gt_bbox) == matcher_poly.match(detections_poly, gt_poly))

    def test_match_coco_at_005(self):
        matcher_poly = MatchEngineIoU(0.05, 'coco')
        matcher_bbox = MatchEngineIoU(0.05, 'coco')
        assert np.all(matcher_bbox.match(detections_bbox, gt_bbox) == matcher_poly.match(detections_poly, gt_poly))

    def test_match_coco_at_01(self):
        matcher_poly = MatchEngineIoU(0.1, 'coco')
        matcher_bbox = MatchEngineIoU(0.1, 'coco')
        assert np.all(matcher_bbox.match(detections_bbox, gt_bbox) == matcher_poly.match(detections_poly, gt_poly))

    def test_match_xview_at_001(self):
        matcher_poly = MatchEngineIoU(0.01, 'xview')
        matcher_bbox = MatchEngineIoU(0.01, 'xview')
        assert np.all(matcher_bbox.match(detections_bbox, gt_bbox) == matcher_poly.match(detections_poly, gt_poly))

    def test_match_xview_at_005(self):
        matcher_poly = MatchEngineIoU(0.05, 'xview')
        matcher_bbox = MatchEngineIoU(0.05, 'xview')
        assert np.all(matcher_bbox.match(detections_bbox, gt_bbox) == matcher_poly.match(detections_poly, gt_poly))

    def test_match_xview_at_01(self):
        matcher_poly = MatchEngineIoU(0.1, 'xview')
        matcher_bbox = MatchEngineIoU(0.1, 'xview')
        assert np.all(matcher_bbox.match(detections_bbox, gt_bbox) == matcher_poly.match(detections_poly, gt_poly))


class TestMatchIIoU:
    def test_rtree_iou_matrix(self):
        matcher_poly = MatchEngineIoU(0.1, 'coco')
        matcher_bbox = MatchEngineIoU(0.1, 'coco')
        IoU_poly = matcher_poly.compute_similarity_matrix(detections_corr_poly, gt_corr_poly,
                                                          label_mean_area=gt_mean_area)
        IoU_bbox = matcher_bbox.compute_similarity_matrix(detections_bbox, gt_bbox,
                                                          label_mean_area=gt_mean_area)
        print(IoU_poly)
        print(IoU_bbox)
        assert np.all(IoU_poly == IoU_bbox)

    def test_match_coco_at_001(self):
        matcher_poly = MatchEngineIoU(0.01, 'coco')
        matcher_bbox = MatchEngineIoU(0.01, 'coco')
        assert \
            np.all(matcher_bbox.match(detections_bbox,
                                      gt_bbox,
                                      label_mean_area=gt_mean_area) == matcher_poly.match(detections_poly,
                                                                                          gt_poly,
                                                                                          label_mean_area=gt_mean_area))

    def test_match_coco_at_005(self):
        matcher_poly = MatchEngineIoU(0.05, 'coco')
        matcher_bbox = MatchEngineIoU(0.05, 'coco')
        assert \
            np.all(matcher_bbox.match(detections_bbox,
                                      gt_bbox,
                                      label_mean_area=gt_mean_area) == matcher_poly.match(detections_poly,
                                                                                          gt_poly,
                                                                                          label_mean_area=gt_mean_area))

    def test_match_coco_at_01(self):
        matcher_poly = MatchEngineIoU(0.1, 'coco')
        matcher_bbox = MatchEngineIoU(0.1, 'coco')
        assert \
            np.all(matcher_bbox.match(detections_bbox,
                                      gt_bbox,
                                      label_mean_area=gt_mean_area) == matcher_poly.match(detections_poly,
                                                                                          gt_poly,
                                                                                          label_mean_area=gt_mean_area))

    def test_match_xview_at_001(self):
        matcher_poly = MatchEngineIoU(0.01, 'xview')
        matcher_bbox = MatchEngineIoU(0.01, 'xview')
        assert \
            np.all(matcher_bbox.match(detections_bbox,
                                      gt_bbox,
                                      label_mean_area=gt_mean_area) == matcher_poly.match(detections_poly,
                                                                                          gt_poly,
                                                                                          label_mean_area=gt_mean_area))

    def test_match_xview_at_005(self):
        matcher_poly = MatchEngineIoU(0.05, 'xview')
        matcher_bbox = MatchEngineIoU(0.05, 'xview')
        assert \
            np.all(matcher_bbox.match(detections_bbox,
                                      gt_bbox,
                                      label_mean_area=gt_mean_area) == matcher_poly.match(detections_poly,
                                                                                          gt_poly,
                                                                                          label_mean_area=gt_mean_area))

    def test_match_xview_at_01(self):
        matcher_poly = MatchEngineIoU(0.1, 'xview')
        matcher_bbox = MatchEngineIoU(0.1, 'xview')
        assert \
            np.all(matcher_bbox.match(detections_bbox,
                                      gt_bbox,
                                      label_mean_area=gt_mean_area) == matcher_poly.match(detections_poly,
                                                                                          gt_poly,
                                                                                          label_mean_area=gt_mean_area))


class TestMatchScalability:
    data_detect = np.load(os.path.dirname(__file__) + '/../../resources/data/data_detect.npy')
    data_detect_conv = convert_to_bounding_box(data_detect, trim_invalid_geometry=True)
    data_detect_conv[:, -1] = 0

    def _make_scale_match(self, threshold, method):
        matcher = MatchEngineIoU(threshold, method)
        matcher.match(self.data_detect_conv, self.data_detect_conv[:, (True, False, True)])

    def test_match_xview_at_001(self):
        self._make_scale_match(0.01, 'xview')

    def test_match_coco_at_001(self):
        self._make_scale_match(0.01, 'coco')


if __name__ == '__main__':
    t = TestMatchScalability()
    t.test_match_coco_at_001()
