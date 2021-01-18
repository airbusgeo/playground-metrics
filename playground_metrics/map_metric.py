"""Implement the public interface to compute mAP from a set of detections and ground truths.

For more in formation on how the metric is computed, see: :doc:`../content/map_metric`.

If one wants to integrate the module into a framework to use it as a validation metric, the
:class:`~MeanAveragePrecisionMetric` class described below should be wrappped accordingly to follow the framework
convention.
"""
import warnings
from collections import defaultdict

import numpy as np

from .utils import to_builtin, to_list
from .match_detections import MatchEngineIoU
from .utils.geometry_utils import get_type_and_convert


class MeanAveragePrecisionMetric:
    r"""Implement an API to compute mAP.

    It gives three methods:

        * :meth:`update(detections, ground_truths) <update>` which accumulates TP, FP and FN over examples
        * :meth:`compute` which computes mAP and AP per label from accumulated values
        * :meth:`reset` which resets accumulated values to their initial values to start mAP computation from scratch

    See Also:
        Information on how **mAP**, **AP**, **precision** and **recall** are computed may be found in
        :doc:`../content/map_metric`.

    Args:
        threshold (float): Optional, default to 0.5. Similarity threshold for which we consider a valid
            match between detection and ground truth.
        match_algorithm (str): Optional, default to 'coco'. 'xview' or 'coco' to choose the matching algorithm (c.f.
            :ref:`match`) or 'non-unitary' to use non-unitary matching.
        label_mean_area (dict) : Optional, default to ``None``. A dictionary containing the mean area for each label in
            the dataset, if given, it is used to match with *iIoU* instead of *IoU* (c.f. :ref:`iiou`).
        trim_invalid_geometry (bool): Optional, default to ``False``. If set to ``True`` conversion will ignore invalid
            geometries and leave them out of mAP computations. This means that the function will work on arrays where
            ``work_array.shape[0] <= input_array.shape[0]``.  If set to ``False``, an invalid geometry will raise an
            :exc:`~playground_metrics.utils.geometry_utils.InvalidGeometryError`.
        autocorrect_invalid_geometry (Bool): Optional, default to ``False``. Whether to attempt correcting a faulty
            geometry to form a valid one. If set to ``True`` and the autocorrect attempt is unsuccessful, it falls back
            to the behaviour defined in ``trim_invalid_geometry``.
        match_engine (:class:`~map_metric_api.match_detections.MatchEngineBase`): Optional, default to
            :class:`~playground_metrics.match_detections.MatchEngineIoU`. If provided matching will be done using the
            provided ``match_engine`` instead of the default one. Note that the ``threshold`` and ``match_algorithm``
            provided parameters will be overridden by those provided in the ``match_engine``.

    Warning:
        When using non-unitary matching, the AP per class and the mAP are ill-defined and must be taken with a grain
        of salt.

    Warns:
        UserWarning: If ``match_algorithm`` is 'non-unitary' to warn that mAP and AP per class values are
            ill-defined.
        RuntimeWarning: If a ``match_engine`` is provided and its ``threshold`` or ``match_algorithm``
            attribute differs from those provided as arguments to the constructor.

    Note:
        * Polygon auto-correction only corrects self-crossing exterior rings, in which case it creates one Polygon
          out of every simple ring which might be extracted from the original Polygon exterior.
        * Polygon auto-correction will systematically fail on Polygons with at least one inner ring.

    Attributes:
        mAP (float) : The mAP computed by :meth:`compute` from accumulated values
        average_precision_per_class (defaultdict) : The AP for each label as constructed by :meth:`compute` from
            accumulated values
        precision_per_class (defaultdict) : The precision for each label as constructed by :meth:`compute` from
            accumulated values
        recall_per_class (defaultdict) : The recall for each label as constructed by :meth:`compute` from accumulated
            values
        number_true_detection_per_class (defaultdict): The number of detection matched to a ground truth as
            constructed by :meth:`compute` from accumulated values
        number_false_detection_per_class (defaultdict): The number of detection not matched to a ground truth as
            constructed by :meth:`compute` from accumulated value
        number_found_ground_truth_per_class (defaultdict): The number of ground truth matched to a detection as
            constructed by :meth:`compute` from accumulated values
        number_missed_ground_truth_per_class (defaultdict): The number of ground truth not matched to a detection as
            constructed by :meth:`compute` from accumulated values
        match_engine (:class:`~map_metric_api.match_detections.MatchEngineBase`) : The match_engine object used to match
            detections and ground truths. If none where provided in the constructor call, it defaults to
            :class:`~playground_metrics.match_detections.MatchEngineIoU`.

    """

    def __init__(self, threshold=None, match_algorithm=None, label_mean_area=None, trim_invalid_geometry=False,
                 autocorrect_invalid_geometry=False, match_engine=None):

        if match_engine is not None and (threshold is not None or match_algorithm is not None):
            warnings.warn('In the future match_engine will be made incompatible with threshold and match_algorithm. '
                          'Providing both will raise a ValueError.', FutureWarning)

        # Set configurations values
        threshold = threshold if threshold is not None else 0.5
        match_algorithm = match_algorithm or 'coco'
        self.match_engine = match_engine or MatchEngineIoU(threshold, match_algorithm)
        if threshold != self.threshold:
            warnings.warn('Discrepancy between user provided threshold and '
                          'match_engine threshold ({} != {})'.format(threshold, self.threshold), RuntimeWarning)

        if match_algorithm != self.match_engine.match_algorithm:
            warnings.warn('Discrepancy between user provided match_algorithm and '
                          'match_engine match_algorithm ({} != {})'.format(match_algorithm,
                                                                           self.match_engine.match_algorithm),
                          RuntimeWarning)

        if match_algorithm == 'non-unitary':
            warnings.warn('When using non-unitary matching, the AP per class and the mAP are '
                          'ill-defined and must be taken with a grain of salt.', UserWarning)

        self.label_mean_area = label_mean_area
        self.trim_invalid_geometry = trim_invalid_geometry
        self.autocorrect_invalid_geometry = autocorrect_invalid_geometry
        # Set intermediate and return values
        self._init_values()

    @property
    def threshold(self):  # noqa: D205,D400
        """float: The IoU threshold by :attr:`self.match_engine <match_engine>` or ``None``
        if :attr:`self.match_engine <match_engine>` doesn't use any threshold.
        """
        try:
            return self.match_engine.threshold
        except AttributeError:
            return None

    @property
    def ground_truth_labels(self):
        """set: The set of unique label accumulated up to this point."""
        return self._ground_truth_labels

    def _init_values(self):
        # Set intermediate values
        self._detection_matched = defaultdict(self._empty_array)
        self._ground_truth_matched = defaultdict(self._empty_array)
        self._number_of_ground_truths = defaultdict(int)
        self._ground_truth_labels = set()
        self._confidence = defaultdict(self._empty_array)
        # Set return values
        self.mAP = 0.0  # pylint: disable=invalid-name
        self.average_precision_per_class = defaultdict(float)
        self.precision_per_class = defaultdict(float)
        self.recall_per_class = defaultdict(float)
        self.number_true_detection_per_class = defaultdict(int)
        self.number_false_detection_per_class = defaultdict(int)
        self.number_found_ground_truth_per_class = defaultdict(int)
        self.number_missed_ground_truth_per_class = defaultdict(int)

    def update(self, detections, ground_truths):
        r"""Accumulate values necessary to compute mAP with detections and ground truths of a single image.

        Args:
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[x_min, y_min, x_max, y_max, confidence, label]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[[[outer_ring], [inner_rings]], confidence, label]``
                * Points for a given class where each row is a detection stored as:
                  ``[x, y, confidence, label]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[x_min, y_min, x_max, y_max, label]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[[[outer_ring], [inner_rings]], label]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[x, y, label]``

        Raises:
            KeyError : If ``self.label_mean_area`` is not ``None`` but a label is missing

        The input detections and ground truths are allowed to be **points** in the documentation.
        This is to allow the use of a custom `MatchEngine` for points, however, the default
        :class:`~playground_metrics.match_detections.MatchEngineIoU` works on **intersection-over-union** which
        is incompatible with **points**. More information on input geometrical types can be found in
        :doc:`playground_metrics.match_detections`.

        Note:
            The labels provided in the input arrays can theoretically be any hashable type, however,
            only numeric types, strings and tuples are officially supported.

        """
        detections_type, detections = self._format_input(detections)
        ground_truths_type, ground_truths = self._format_input(ground_truths)

        if detections.size == ground_truths.size == 0:
            return

        if detections.size == 0:
            self._ground_truth_labels.update(to_list(ground_truths[:, 1]))
            for ground_truth_label in self._ground_truth_labels:
                self._number_of_ground_truths[ground_truth_label] += \
                    len(ground_truths[to_builtin(ground_truths[:, 1]) == ground_truth_label, :1])
                self._ground_truth_matched[ground_truth_label] = \
                    np.concatenate((self._ground_truth_matched[ground_truth_label],
                                    np.zeros((ground_truths[to_builtin(ground_truths[:, 1]) == ground_truth_label,
                                                            :].shape[0]))))
            return

        if ground_truths.size == 0:
            self._ground_truth_labels.update(to_list(detections[:, 2]))
            for ground_truth_label in self._ground_truth_labels:
                self._detection_matched[ground_truth_label] = np.concatenate(
                    (self._detection_matched[ground_truth_label],
                     np.zeros((detections[to_builtin(detections[:, 2]) == ground_truth_label, :2].shape[0]))))
                self._confidence[ground_truth_label] = \
                    np.concatenate((self._confidence[ground_truth_label],
                                    np.sort(detections[to_builtin(detections[:, 2]) == ground_truth_label, 1])[::-1]))
            return

        self._ground_truth_labels.update(to_list(ground_truths[:, 1]), to_list(detections[:, 2]))

        for ground_truth_label in self._ground_truth_labels:
            try:
                mean_area = self.label_mean_area[ground_truth_label]
            except KeyError:
                raise KeyError('label_mean_area is missing the label {}'.format(ground_truth_label))
            except TypeError:
                mean_area = None

            match_matrix = \
                self.match_engine.match(detections[to_builtin(detections[:, 2]) == ground_truth_label, :2],
                                        ground_truths[to_builtin(ground_truths[:, 1]) == ground_truth_label, :1],
                                        label_mean_area=mean_area)

            # Having this before checking if there were detections for this particular class breaks the xview score
            # equality test, however this is the way to go to ensure that False-Negative are correctly accounted for
            # in all cases. So xView scoring code is wrong again here, sorry xView.
            self._number_of_ground_truths[ground_truth_label] += \
                len(ground_truths[to_builtin(ground_truths[:, 1]) == ground_truth_label, :1])

            self._ground_truth_matched[ground_truth_label] = \
                np.concatenate((self._ground_truth_matched[ground_truth_label], np.clip(match_matrix.sum(0), 0, 1)))

            # If no detections for this label pass here
            if match_matrix.shape[0] == 0:
                continue

            self._detection_matched[ground_truth_label] = \
                np.concatenate((self._detection_matched[ground_truth_label], np.clip(match_matrix.sum(1), 0, 1)))
            self._confidence[ground_truth_label] = \
                np.concatenate((self._confidence[ground_truth_label],
                                np.sort(detections[to_builtin(detections[:, 2]) == ground_truth_label, 1])[::-1]))

    def compute(self):
        r"""Compute the **mAP** according to the accumulated values.

        Moreover it sets the value for the following attributes:

            * :attr:`self.precision_per_class <precision_per_class>`: A dict of precisions per label
            * :attr:`self.recall_per_class <recall_per_class>`: A dict of recall per label
            * :attr:`self.average_precision_per_class <average_precision_per_class>`: A dict of average precisions
              per label
            * :attr:`self.number_true_detection_per_class <number_true_detection_per_class>`: A dict of the number
              of detection matched to a ground truth
            * :attr:`self.number_false_detection_per_class <number_false_detection_per_class>`: A dict of the number
              of detection not matched to a ground truth
            * :attr:`self.number_found_ground_truth_per_class <number_found_ground_truth_per_class>`: A dict of the
              number of ground truth matched to a detection
            * :attr:`self.number_missed_ground_truth_per_class <number_missed_ground_truth_per_class>`: A dict of the
              number of ground truth not matched to a detection

        Returns:
            float : The Mean Average Precision metric

        """
        for ground_truth_label in self._ground_truth_labels:

            # Compute the Det positive, Det negative, Gt positive and gt negative counters
            self.number_true_detection_per_class[ground_truth_label] = \
                np.sum(self._detection_matched[ground_truth_label]).item()
            self.number_false_detection_per_class[ground_truth_label] = \
                np.sum(np.logical_not(self._detection_matched[ground_truth_label])).item()
            self.number_found_ground_truth_per_class[ground_truth_label] = \
                np.sum(self._ground_truth_matched[ground_truth_label]).item()
            self.number_missed_ground_truth_per_class[ground_truth_label] = \
                np.sum(np.logical_not(self._ground_truth_matched[ground_truth_label])).item()

            if self._number_of_ground_truths[ground_truth_label] != 0:
                # Prepare the cumulative sum along confidence-sorted detections to compute Precision(Recall)
                sorted_detection_indices = np.argsort(self._confidence[ground_truth_label])[::-1]
                tp_sum = np.cumsum(self._detection_matched[ground_truth_label][sorted_detection_indices])
                fp_sum = np.cumsum(
                    np.logical_not(self._detection_matched[ground_truth_label][sorted_detection_indices])
                )

                # Compute the Precision(Recall) function
                precision = tp_sum / (tp_sum + fp_sum + np.spacing(1))
                recall = tp_sum / self._number_of_ground_truths[ground_truth_label]

                # Compute the precision and recall
                # For precision tp is the number of detections matched to the ground truth (unique, non-unique matches)
                tp = np.sum(self._detection_matched[ground_truth_label])
                p = len(self._detection_matched[ground_truth_label])
                self.precision_per_class[ground_truth_label] = (tp / p).item()
                # For recall tp is the number of ground-truth targets matched to detections
                tp = np.sum(self._ground_truth_matched[ground_truth_label])
                a = self._number_of_ground_truths[ground_truth_label]
                self.recall_per_class[ground_truth_label] = (tp / a).item()

                # Average precision and mAP computation
                precision, recall = self._remove_jaggedness(precision, recall)
                self.average_precision_per_class[ground_truth_label] = \
                    self._integrate_precision_recall_curve(precision, recall).item()
            else:
                self.precision_per_class[ground_truth_label] = 0.0
                self.recall_per_class[ground_truth_label] = np.nan

        # self.average_precision_per_class is a defaultdict, when label is absent the value is 0 by default
        self.mAP = np.nanmean(np.array([self.average_precision_per_class[label]
                                        for label in self._ground_truth_labels]))
        return self.mAP

    def reset(self):
        r"""Reset all intermediate and return values to their initial value.

        If :meth:`reset` is not called in-between two :meth:`compute` call, the values returned by :meth:`compute`
        will take into account the entire prediction stack, not just the predictions in-between the two
        :meth:`compute` calls.

        """
        self._init_values()

    def _format_input(self, input_array):
        return get_type_and_convert(input_array, trim_invalid_geometry=self.trim_invalid_geometry,
                                    autocorrect_invalid_geometry=self.autocorrect_invalid_geometry)

    @staticmethod
    def _remove_jaggedness(precision, recall):
        recall = np.concatenate([[0], recall, [1]])
        precision = np.concatenate([[0], precision, [0]])
        for i in range(precision.shape[0] - 2, 0, -1):
            if precision[i] > precision[i - 1]:
                precision[i - 1] = precision[i]
        return precision, recall

    @staticmethod
    def _integrate_precision_recall_curve(precision, recall):
        # The indices where recall changes value
        i = np.where(recall[1:] != recall[:len(recall) - 1])[0] + 1

        # Integration with step interpolation
        average_precision = np.sum((recall[i] - recall[i - 1]) * precision[i])

        return average_precision

    # Default factories
    @staticmethod
    def _nan():
        return np.nan

    @staticmethod
    def _empty_array():
        return np.array([])
