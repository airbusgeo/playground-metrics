import os.path

import numpy as np
from pytest import raises, approx

from playground_metrics.metrics_helper import MeanFBetaAtThresholds


class TestMeanFBetaAtThresholds:

    def test_inputs(self):
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

        with raises(ValueError):
            MeanFBetaAtThresholds(beta=0, thresholds=thresholds)

        with raises(ValueError):
            MeanFBetaAtThresholds(beta="a", thresholds=thresholds)

        with raises(ValueError):
            MeanFBetaAtThresholds(beta=0.0, thresholds=thresholds)

        with raises(TypeError):
            MeanFBetaAtThresholds(beta=1, thresholds=0.1)

    def test_no_predictions_no_gt(self):
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        f2_computer = MeanFBetaAtThresholds(beta=2, thresholds=thresholds)
        assert np.isnan(f2_computer.compute())

    def test_no_predictions_single_class(self):
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        f2_true = 0.0
        f2_computer = MeanFBetaAtThresholds(beta=2, thresholds=thresholds)
        n_samples = 10

        for i in range(n_samples):
            if i % 2 == 0:
                f2_computer.update(detections=[], ground_truths=[[0, 0, 10, 10, 0],
                                                                 [12, 12, 20, 20, 0]])
            else:
                f2_computer.update(detections=[], ground_truths=[])
                f2_true += 1.0 / n_samples

        assert f2_computer._counter == n_samples
        assert f2_computer._internal_score_per_class == {0: len(thresholds) * n_samples * f2_true}
        assert f2_true == f2_computer.compute()

    def test_no_predictions_multiple_classes(self):
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        f2_true = 0.0
        f2_computer = MeanFBetaAtThresholds(beta=2, thresholds=thresholds)
        n_samples = 12
        n_classes = 3
        f2_per_class = {i: 0.0 for i in range(n_classes)}

        for i in range(n_samples):
            if i % 2 == 0:
                f2_computer.update(detections=[], ground_truths=[[0, 0, 10, 10, i % n_classes],
                                                                 [12, 12, 20, 20, i % n_classes]])
                # Other classes are not predicted and this is good
                f2_true += 1.0 / n_samples * (n_classes - 1.0) / n_classes
                other_classes = list(range(n_classes))
                other_classes.remove(i % n_classes)
                for j in other_classes:
                    f2_per_class[j] += 1.0 / n_samples
            else:
                f2_computer.update(detections=[], ground_truths=[])
                f2_true += 1.0 / n_samples
                for j in range(n_classes):
                    f2_per_class[j] += 1.0 / n_samples

        assert f2_computer._counter == n_samples
        assert len(f2_computer._internal_score_per_class) == n_classes
        assert f2_true == approx(f2_computer.compute())
        assert len(f2_computer.score_per_class) == n_classes
        for i in range(n_classes):
            assert f2_computer._internal_score_per_class[i] == (n_samples - 2) * len(thresholds)
            assert f2_computer.score_per_class[i] == approx(f2_per_class[i])

    def test_trivial_predictions_single_classes(self):
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        f2_true = 0.0
        f2_computer = MeanFBetaAtThresholds(beta=2, thresholds=thresholds)
        n_samples = 10

        for i in range(n_samples):
            if i % 2 == 0:
                # intersection 85% on 1 of 2 gts
                f2_computer.update(detections=[[0, 0, 10, 8.3, 0.99, 0]],
                                   ground_truths=[[0, 0, 10, 10, 0],
                                                  [120, 120, 122, 122, 0]])
                # f2 is defined as 5 * Pr * R / (4 * Pr + R)
                v = 5 * 1.0 * 0.5 / (4 * 1.0 + 0.5)
                f2_true += v * 4.0 / len(thresholds) / n_samples
            else:
                f2_computer.update(detections=[], ground_truths=[])
                f2_true += 1.0 / n_samples

        assert f2_computer._counter == n_samples
        assert f2_true == approx(f2_computer.compute())

    def test_itegration_on_ship_data(self):
        data = np.load(os.path.dirname(__file__) + "/../resources/data/ships/ships_data.npz", allow_pickle=True)
        # f2 for complete data:
        # f2_true_all is 0.6428503771289549
        # f2 for first 1000 entries:
        f2_true_1000 = 0.6395173984808165

        all_preds_array = data['preds_df']
        all_gt_targets_array = data['gt_df']

        from collections import defaultdict

        all_preds = defaultdict(list)
        for d in all_preds_array:
            pred = [[np.array(d[2]).reshape(-1, 2)], d[1], 0] if not np.isnan(d[2]).any() else []
            all_preds[d[5]].append(pred)

        all_gt_targets = defaultdict(list)
        for d in all_gt_targets_array:
            gt = [[np.array(d[2]).reshape(-1, 2)], 0] if not np.isnan(d[2]).any() else []
            all_gt_targets[d[3]].append(gt)

        assert set(all_preds.keys()) == set(all_gt_targets.keys())

        tile_ids = sorted(all_gt_targets.keys())
        # Define decision thresholds
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        f2_computer = MeanFBetaAtThresholds(beta=2, thresholds=thresholds, match_algorithm='coco')
        # Iterate over the tiles
        n = 1000
        for tile_id in tile_ids[:n]:
            # Ground truth targets
            gt_targets = all_gt_targets[tile_id]
            preds = all_preds[tile_id]

            f2_computer.update(detections=preds, ground_truths=gt_targets)

        # Log global F2 score
        global_f2 = f2_computer.compute()
        assert approx(global_f2) == f2_true_1000
