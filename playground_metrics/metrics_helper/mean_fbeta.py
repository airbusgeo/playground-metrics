import numpy as np

from playground_metrics.map_metric import MeanAveragePrecisionMetric


class MeanFBetaAtThresholds:
    r"""Class to compute mean F-beta score (averaged and per class) at list of thresholds for detection task.

    Notes:
        Special cases are resolved as following:

        * if no predictions neither ground truth, then f2 = 1 for all classes
        * if no predictions but there is a ground truth, f2 = 0 for all classes
        * if there are predictions, we compute f2 from precision and recall using
          :class:`~playground_metrics.map_metric.MeanAveragePrecisionMetric`

    Args:
        beta (int/float): F-beta scoring parameter
        thresholds (list/tuple of float): List/tuple of similarity thresholds for which we consider a valid
            match between detection and ground truth. For example, `[0.5, 0.7, 0.9]`.
        **kwargs: kwargs to configure internal :class:`~playground_metrics.map_metric.MeanAveragePrecisionMetric`

    Raises:
        ValueError: If ``thresholds`` is not a list/tuple or if ``beta`` is not a positive float

    Attributes:
        score (float) : Mean F-beta score computed by ``compute()`` from accumulated values

    """

    def __init__(self, beta, thresholds, **kwargs):
        """Initialize instance."""
        if not isinstance(thresholds, (tuple, list)):
            raise TypeError("Argument thresholds should be list or tuple, but given {}".format(type(thresholds)))
        if not (isinstance(beta, (int, float)) and beta > 0):
            raise ValueError("Argument beta should be positive float")

        self._map_computers = [
            MeanAveragePrecisionMetric(threshold=t, **kwargs) for t in thresholds
        ]
        self.beta = beta
        self.score = self.score_per_class = self._internal_score_per_class = self._counter = None
        self.reset()

    def reset(self):
        r"""Reset all intermediate and return values to their initial value.

        If ``reset()`` is not called in-between two ``compute()`` call, the values returned by ``compute()`` will take
        into account the entire prediction stack, not just the predictions in-between the two ``compute()`` calls.
        """
        for map_computer in self._map_computers:
            map_computer.reset()
        self.score = 0.0
        self.score_per_class = {}
        self._internal_score_per_class = {}
        self._counter = 0  # counter for seen tiles

    def update(self, detections, ground_truths):
        """Accumulate values necessary to compute mAP with detections and ground truths of a single image.

        Arguments are same as in :meth:`~playground_metrics.map_metric.MeanAveragePrecisionMetric.update`

        """
        detections = np.array(detections, copy=False)
        ground_truths = np.array(ground_truths, copy=False)

        self._counter += 1

        if detections.size == 0:
            if ground_truths.size == 0:
                for label in self._internal_score_per_class:
                    self._internal_score_per_class[label] += 1 * len(self._map_computers)
                # all done -> exit update
                return

        for map_computer in self._map_computers:
            map_computer.reset()
            map_computer.update(detections=detections, ground_truths=ground_truths)
            map_computer.compute()

            # At first we need to discover new classes if any appeared
            for label in map_computer.ground_truth_labels:
                if label not in self._internal_score_per_class:
                    # unseen class -> all previous tiles are correctly predicted for this class
                    self._internal_score_per_class[label] = (self._counter - 1) * len(self._map_computers)

            for label in self._internal_score_per_class:
                if label not in map_computer.ground_truth_labels:
                    f_beta = 1.0
                else:
                    precision = map_computer.precision_per_class[label]
                    recall = map_computer.recall_per_class[label]

                    if not (np.isnan(precision) or np.isnan(recall)):
                        f_beta = self._compute_fbeta(precision, recall)
                    else:
                        f_beta = 0.0
                self._internal_score_per_class[label] += f_beta

    def compute(self):
        """Compute the F-beta score averaged over all classes and thresholds according to the accumulated values.

        Returns:
            float : mean F-beta at thresholds

        """
        if not self._internal_score_per_class:
            return np.nan

        self.score = 0.0
        for label in self._internal_score_per_class:
            self.score_per_class[label] = self._internal_score_per_class[label]
            self.score_per_class[label] /= 1.0 * self._counter  # ~ /= len(tiles)
            self.score_per_class[label] /= 1.0 * len(self._map_computers)  # ~ /= len(thresholds)
            self.score += self.score_per_class[label]

        self.score /= 1.0 * len(self._internal_score_per_class)  # ~ /= n_classes
        return self.score

    def _compute_fbeta(self, precision, recall):
        return (1.0 + self.beta ** 2) * precision * recall / (self.beta ** 2 * precision + recall + 1e-20)
