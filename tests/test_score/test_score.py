# flake8: noqa: F841
import pytest
from playground_metrics.map_metric import MeanAveragePrecisionMetric
from playground_metrics.match_detections import MatchEngineBase, MatchEngineIoU
import numpy as np

# This breaks th xview equality test because the last gt (class 2) is never counted as a false negative because there
# are no detections of this class. This makes a final mAP of 0.21 (AP = {0: 0.3333333333333333, 1: 0.3}), notice that
# the class 2 is non-existent in this setting, however, in most cases, the bug would only heighten AP of classes which
# are present in ground truth but seldom predicted, like AP(0) which is absent from image 1 and has its AP increased)
# when it should be 0.15 (AP = {0: 0.16666666666666666, 1: 0.3, 2: 0.0})

detections = {0: np.array([[15, 0, 26, 5, 0.9, 0],
                           [11, 6, 20, 11, 0.5, 0],
                           [0, 0, 9, 5, 0.7, 0],
                           [23, 13, 29, 18, 0.25, 1]]),
              1: np.array([[0, 0, 3, 2, 0.7, 1],
                           [11, 6, 20, 11, 0.5, 1],
                           [23, 13, 29, 18, 0.25, 1],
                           [15, 0, 26, 5, 0.9, 1]])}


gt = np.array([[5, 2, 15, 9, 1],
               [18, 10, 26, 15, 1],
               [18, 10, 26, 15, 1],
               [18, 10, 26, 15, 0],
               [18, 10, 26, 15, 2]])

fnames = [0, 1]
gt_coords = np.concatenate((gt[:, :-1], gt[:, :-1]))
gt_chips = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
gt_classes = np.array(gt[:, 4].tolist() * 2)
detections_xview = {key: detections[key][:, (0, 1, 2, 3, 5, 4)] for key in detections.keys()}


class TestScoreSynth:
    def test_empty_detection(self):
        detections = np.array([])
        ground_truths = np.array([[2, 6, 11, 16, 0], [20, 11, 45, 25, 0]])
        map_computer = MeanAveragePrecisionMetric(0.5, 'coco')
        map_computer.update(detections, ground_truths)
        assert map_computer.compute() == 0.0
        assert map_computer.number_true_detection_per_class[0] == 0
        assert map_computer.number_false_detection_per_class[0] == 0
        assert map_computer.number_found_ground_truth_per_class[0] == 0
        assert map_computer.number_missed_ground_truth_per_class[0] == 2

    def test_empty_ground_truth(self):
        detections = np.array([[2, 6, 11, 16, 0.9, 0], [20, 11, 45, 25, 0.8, 0]])
        ground_truths = np.array([])
        map_computer = MeanAveragePrecisionMetric(0.5, 'coco')
        map_computer.update(detections, ground_truths)
        assert map_computer.compute() == 0.0
        assert map_computer.number_true_detection_per_class[0] == 0
        assert map_computer.number_false_detection_per_class[0] == 2
        assert map_computer.number_found_ground_truth_per_class[0] == 0
        assert map_computer.number_missed_ground_truth_per_class[0] == 0

        assert 0 in map_computer.precision_per_class
        assert map_computer.precision_per_class[0] == 0
        assert 0 in map_computer.recall_per_class
        assert np.isnan(map_computer.recall_per_class[0])
        assert 0 in map_computer.average_precision_per_class

        assert 0 in map_computer.ground_truth_labels

    @pytest.mark.filterwarnings("always")
    def test_empty(self):
        detections = np.array([])
        ground_truths = np.array([])
        map_computer = MeanAveragePrecisionMetric(0.5, 'coco')
        map_computer.update(detections, ground_truths)
        assert np.isnan(map_computer.compute())
        with pytest.warns(RuntimeWarning, match='Mean of empty slice'):
            map_computer.compute()

    def test_empty_ground_truth_decreases(self):
        detections = np.array([[2, 6, 11, 16, 0.9, 0], [20, 11, 45, 25, 0.8, 0]])
        ground_truths = np.array([[2, 6, 11, 16, 0], [20, 11, 45, 25, 0]])
        map_computer = MeanAveragePrecisionMetric(0.5, 'coco')
        map_computer.update(detections, ground_truths)
        temp = map_computer.compute()
        map_computer.update(detections, np.array([]))
        assert map_computer.compute() < temp

    def test_empty_detection_decreases(self):
        detections = np.array([[2, 6, 11, 16, 0.9, 0], [20, 11, 45, 25, 0.8, 0]])
        ground_truths = np.array([[2, 6, 11, 16, 0], [20, 11, 45, 25, 0]])
        map_computer = MeanAveragePrecisionMetric(0.5, 'coco')
        map_computer.update(detections, ground_truths)
        temp = map_computer.compute()
        map_computer.update(np.array([]), ground_truths)
        assert map_computer.compute() < temp

    def test_empty_equals(self):
        detections = np.array([[2, 6, 11, 16, 0.9, 0], [20, 11, 45, 25, 0.8, 0]])
        ground_truths = np.array([[2, 6, 11, 16, 0], [20, 11, 45, 25, 0]])
        map_computer = MeanAveragePrecisionMetric(0.5, 'coco')
        map_computer.update(detections, ground_truths)
        temp = map_computer.compute()
        map_computer.update(np.array([]), np.array([]))
        assert map_computer.compute() == temp

    def test_asymmetrical_empty_detection(self):
        detections = np.array([[2, 6, 11, 16, 0.9, 0]])
        ground_truths = np.array([[2, 6, 11, 16, 0], [20, 11, 45, 25, 1]])
        map_computer = MeanAveragePrecisionMetric(0.5, 'coco')
        map_computer.update(detections, ground_truths)
        assert np.isclose(map_computer.compute(), 0.5)

    def test_asymmetrical_empty_ground_truth(self):
        detections = np.array([[2, 6, 11, 16, 0.9, 0], [20, 11, 45, 25, 0.8, 1]])
        ground_truths = np.array([[2, 6, 11, 16, 0]])
        map_computer = MeanAveragePrecisionMetric(0.5, 'coco')
        map_computer.update(detections, ground_truths)
        assert np.isclose(map_computer.compute(), 0.5)

    def test_accumulate_pg_xview(self):
        map1 = MeanAveragePrecisionMetric(0.01, 'xview')
        map1.update(detections[0], gt)
        map2 = MeanAveragePrecisionMetric(0.01, 'xview')
        map2.update(detections[0], gt)
        map2.update(detections[0], gt)
        assert np.isclose(map1.compute(), map2.compute())

    def test_accumulate_pg_coco(self):
        map1 = MeanAveragePrecisionMetric(0.01, 'coco')
        map1.update(detections[0], gt)
        map2 = MeanAveragePrecisionMetric(0.01, 'coco')
        map2.update(detections[0], gt)
        map2.update(detections[0], gt)
        assert np.isclose(map1.compute(), map2.compute())

    def test_trivial_solutions_non_unitary(self):
        map_computer = MeanAveragePrecisionMetric(0.0, 'non-unitary')
        map_computer.update([[0, 0, 26, 26, 1, 0]], [[5, 2, 15, 9, 0],
                                                     [18, 10, 26, 15, 0],
                                                     [18, 10, 26, 15, 0],
                                                     [18, 10, 26, 15, 0],
                                                     [18, 10, 26, 15, 0]])
        _ = map_computer.compute()
        assert np.allclose(map_computer.precision_per_class[0], 1.0)
        assert np.allclose(map_computer.recall_per_class[0], 1.0)
        assert map_computer.number_true_detection_per_class[0] == 1
        assert map_computer.number_false_detection_per_class[0] == 0
        assert map_computer.number_found_ground_truth_per_class[0] == 5
        assert map_computer. number_missed_ground_truth_per_class[0] == 0
        map_computer.reset()
        map_computer.update([[5, 2, 15, 9, 1, 0],
                             [18, 10, 26, 15, 1, 0],
                             [18, 10, 26, 15, 1, 0],
                             [18, 10, 26, 15, 1, 0],
                             [18, 10, 26, 15, 1, 0]], [[0, 0, 26, 26, 0]])
        _ = map_computer.compute()
        assert np.allclose(map_computer.precision_per_class[0], 1.0)
        assert np.allclose(map_computer.recall_per_class[0], 1.0)
        assert map_computer.number_true_detection_per_class[0] == 5
        assert map_computer.number_false_detection_per_class[0] == 0
        assert map_computer.number_found_ground_truth_per_class[0] == 1
        assert map_computer.number_missed_ground_truth_per_class[0] == 0


class TestUserMatchEngine:
    def test_match_engine_override(self):
        class MatchEngine(MatchEngineBase):
            def compute_similarity_matrix(self, detections, ground_truths, label_mean_area=None):
                pass

            def trim_similarity_matrix(self, similarity_matrix, detections, ground_truths, label_mean_area=None):
                pass

        map = MeanAveragePrecisionMetric(0.5, 'coco', match_engine=MatchEngine('coco'))

        assert isinstance(map.match_engine, MatchEngine)

    def test_threshold_property(self):
        class MatchEngine(MatchEngineBase):
            def compute_similarity_matrix(self, detections, ground_truths, label_mean_area=None):
                pass

            def trim_similarity_matrix(self, similarity_matrix, detections, ground_truths, label_mean_area=None):
                pass

        map_with_threshold = MeanAveragePrecisionMetric(0.5, 'coco', match_engine=MatchEngineIoU(0.5, 'coco'))
        map_without_threshold = MeanAveragePrecisionMetric(0.5, 'coco', match_engine=MatchEngine('coco'))

        assert map_with_threshold.threshold == 0.5
        assert map_without_threshold.threshold is None

    @pytest.mark.filterwarnings("always")
    def test_threshold_warning(self):
        import warnings
        warnings.filterwarnings("always", category=RuntimeWarning)

        class MatchEngine(MatchEngineBase):
            def compute_similarity_matrix(self, detections, ground_truths, label_mean_area=None):
                pass

            def trim_similarity_matrix(self, similarity_matrix, detections, ground_truths, label_mean_area=None):
                pass

        with pytest.warns(RuntimeWarning, match='Discrepancy between user provided threshold'):
            map_with_threshold = MeanAveragePrecisionMetric(0.1, 'coco', match_engine=MatchEngineIoU(0.5, 'coco'))

        with pytest.warns(RuntimeWarning, match='Discrepancy between user provided threshold'):
            map_without_threshold = MeanAveragePrecisionMetric(0.1, 'coco', match_engine=MatchEngine('coco'))

    @pytest.mark.filterwarnings("always")
    def test_match_algorithm_warning(self):
        with pytest.warns(RuntimeWarning, match='Discrepancy between user provided match_algorithm'):
            _ = MeanAveragePrecisionMetric(0.5, 'xview', match_engine=MatchEngineIoU(0.5, 'coco'))


class TestGroundTruthLabel:
    @staticmethod
    def _ground_truth_label_test(detections, ground_truths, ground_truth_labels_list):
        map_computer = MeanAveragePrecisionMetric(0.5, 'coco')
        map_computer.update(detections, ground_truths)
        assert sorted(list(map_computer.ground_truth_labels)) == sorted(ground_truth_labels_list)
        map_computer.compute()
        assert sorted(list(map_computer.precision_per_class.keys())) == sorted(ground_truth_labels_list)
        assert sorted(list(map_computer.recall_per_class.keys())) == sorted(ground_truth_labels_list)
        assert sorted(list(map_computer.average_precision_per_class.keys())) == sorted(ground_truth_labels_list)
        assert sorted(list(map_computer.number_false_detection_per_class.keys())) == sorted(ground_truth_labels_list)
        assert sorted(list(map_computer.number_found_ground_truth_per_class.keys())) == sorted(ground_truth_labels_list)
        assert sorted(list(map_computer.number_missed_ground_truth_per_class.keys())) == \
            sorted(ground_truth_labels_list)
        assert sorted(list(map_computer.number_true_detection_per_class.keys())) == sorted(ground_truth_labels_list)

    def _make_ground_truth_label_test(self, ground_truth_type):
        detection = np.array([[0, 0, 1, 1, 0.9, ground_truth_type('0')],
                              [1, 1, 2, 2, 0.8, ground_truth_type('1')]], dtype=np.dtype('O'))
        gt = np.array([[0, 0, 1, 1, ground_truth_type('0')],
                       [1, 1, 2, 2, ground_truth_type('1')]], dtype=np.dtype('O'))

        ground_truth_labels_list = [ground_truth_type('0'), ground_truth_type('1')]

        self._ground_truth_label_test(detection, gt, ground_truth_labels_list)
        self._ground_truth_label_test(detection, gt[1:, :], ground_truth_labels_list)
        self._ground_truth_label_test(detection[1:, :], gt, ground_truth_labels_list)
        self._ground_truth_label_test(detection, gt[:1, :], ground_truth_labels_list)
        self._ground_truth_label_test(detection[:1, :], gt, ground_truth_labels_list)

    def test_int(self):
        self._make_ground_truth_label_test(int)

    def test_float(self):
        self._make_ground_truth_label_test(float)

    def test_str(self):
        self._make_ground_truth_label_test(str)

    def test_tuple(self):
        self._make_ground_truth_label_test(tuple)
