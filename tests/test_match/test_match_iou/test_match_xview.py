import numpy as np

from playground_metrics.match_detections import MatchEngineIoU
from tests.test_match.test_match_iou.test_match_bbox import detections, gt

from tests.resources.reference_functions import sort_detection_by_confidence

from tests.resources.xview_scoring.rectangle import Rectangle
from tests.resources.xview_scoring.matching import Matching

detections_rectangles = [Rectangle(coord[0].xmin, coord[0].ymin, coord[0].xmax, coord[0].ymax)
                         for coord in sort_detection_by_confidence(detections)]
gt_rectangles = [Rectangle(coord[0].xmin, coord[0].ymin, coord[0].xmax, coord[0].ymax) for coord in gt]


class TestXviewMatch:
    def test_match_xview_at_001(self):
        matcher = MatchEngineIoU(0.01, 'xview')
        xview_matcher = Matching(gt_rectangles, detections_rectangles)
        assert np.all(matcher.match(detections, gt).sum(1) == xview_matcher.greedy_match(0.01)[0])

    def test_match_xview_at_005(self):
        matcher = MatchEngineIoU(0.05, 'xview')
        xview_matcher = Matching(gt_rectangles, detections_rectangles)
        assert np.all(matcher.match(detections, gt).sum(1) == xview_matcher.greedy_match(0.05)[0])

    def test_match_xview_at_01(self):
        matcher = MatchEngineIoU(0.1, 'xview')
        xview_matcher = Matching(gt_rectangles, detections_rectangles)
        assert np.all(matcher.match(detections, gt).sum(1) == xview_matcher.greedy_match(0.1)[0])

    def test_match_xview_at_02(self):
        matcher = MatchEngineIoU(0.2, 'xview')
        xview_matcher = Matching(gt_rectangles, detections_rectangles)
        assert np.all(matcher.match(detections, gt).sum(1) == xview_matcher.greedy_match(0.2)[0])

    def test_match_xview_at_03(self):
        matcher = MatchEngineIoU(0.3, 'xview')
        xview_matcher = Matching(gt_rectangles, detections_rectangles)
        assert np.all(matcher.match(detections, gt).sum(1) == xview_matcher.greedy_match(0.3)[0])

    def test_match_xview_at_04(self):
        matcher = MatchEngineIoU(0.4, 'xview')
        xview_matcher = Matching(gt_rectangles, detections_rectangles)
        assert np.all(matcher.match(detections, gt).sum(1) == xview_matcher.greedy_match(0.4)[0])

    def test_match_xview_at_05(self):
        matcher = MatchEngineIoU(0.5, 'xview')
        xview_matcher = Matching(gt_rectangles, detections_rectangles)
        assert np.all(matcher.match(detections, gt).sum(1) == xview_matcher.greedy_match(0.5)[0])

    def test_match_xview_at_07(self):
        matcher = MatchEngineIoU(0.7, 'xview')
        xview_matcher = Matching(gt_rectangles, detections_rectangles)
        assert np.all(matcher.match(detections, gt).sum(1) == xview_matcher.greedy_match(0.7)[0])

    def test_match_xview_at_095(self):
        matcher = MatchEngineIoU(0.95, 'xview')
        xview_matcher = Matching(gt_rectangles, detections_rectangles)
        assert np.all(matcher.match(detections, gt).sum(1) == xview_matcher.greedy_match(0.95)[0])
