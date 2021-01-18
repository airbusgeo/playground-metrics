"""Implement the public interface to match a set of detections and ground truths."""

from abc import ABC, abstractmethod

import rtree
import numpy as np

from .utils.geometry_utils import BoundingBox, Polygon, Point


def _make_rtree(detections, transform_fn):
    """Make a fast RTree index from an iterable of detections.

    Args:
        detections (Iterable): An iterable of |Geometry| object.
        transform_fn (callable): A function taking a |Geometry| object an returning a box coordinates as a
            ``(minx, miny, maxx, maxy)`` tuple.

    Returns:
        rtree.index.Index: A RTree index populated with detections bounding box.

    """
    def enumerate_detections(iterable):
        for i, element in enumerate(iterable):
            yield i, transform_fn(element), None

    rtree_index_prop = rtree.index.Property()
    rtree_index_prop.fill_factor = 0.5
    rtree_index_prop.dimension = 2
    return rtree.index.Index(enumerate_detections(detections), properties=rtree_index_prop)


class MatchEngineBase(ABC):
    """Match detection with their ground truth according a similarity matrix and a detection confidence score.

    Matching may be done using coco algorithm or xView algorithm (which yield different matches as described for an
    intersection-over-union similarity matrix in :ref:`match`) or with non-unitary matching.

    Subclasses must implement :meth:`compute_similarity_matrix` and :meth:`trim_similarity_matrix` to be functional.

    Args:
        match_algorithm (str) : Either 'coco', 'xview' or 'non-unitary' to choose the match algorithm

    Attributes:
        match_algorithm (str) : Either 'coco', 'xview' or 'non-unitary' and indicates the match algorithm used

    """

    def __init__(self, match_algorithm):
        if match_algorithm not in ['coco', 'xview', 'non-unitary']:
            raise ValueError("match_algorithm must be either coco, xview or non-unitary")

        self.match_algorithm = match_algorithm

        # Authorized geometric types fot this match engine
        self._detection_types = (BoundingBox, Polygon, Point)
        self._ground_truth_types = (BoundingBox, Polygon, Point)

    def __repr__(self):
        """Represent the :class:`~playground_metrics.match_detections.MatchEngineBase` as a string."""
        d_arg = []
        for arg in ['threshold', 'match_algorithm', 'bounding_box_size']:
            if hasattr(self, arg):
                d_arg.append('{}={}'.format(arg, self.__getattribute__(arg)))
        return '{}({})'.format(self.__class__.__name__, ', '.join(d_arg))

    def __str__(self):
        """Represent the :class:`~playground_metrics.match_detections.MatchEngineBase` as a string."""
        d_arg = []
        for arg in ['threshold', 'match_algorithm', 'bounding_box_size']:
            if hasattr(self, arg):
                d_arg.append('{}={}'.format(arg, self.__getattribute__(arg)))
        return '{}({})'.format(self.__class__.__name__.replace('MatchEngine', ''), ', '.join(d_arg))

    def _compute_similarity_matrix_and_trim(self, detections, ground_truths, label_mean_area=None):
        similarity_matrix = self.compute_similarity_matrix(detections, ground_truths, label_mean_area)
        return similarity_matrix, self.trim_similarity_matrix(similarity_matrix, detections, ground_truths,
                                                              label_mean_area)

    @abstractmethod
    def compute_similarity_matrix(self, detections, ground_truths, label_mean_area=None):
        r"""Compute a similarity matrix between detections and ground truths.

        Abstract method.

        This method must be overridden in subsequent subclasses to handle both bounding box and polygon input format.

        Args:
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[Point]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray : A similarity matrix of dimension (#detections, #ground truth)

        """
        raise NotImplementedError

    @abstractmethod
    def trim_similarity_matrix(self, similarity_matrix, detections, ground_truths,
                               label_mean_area=None):  # noqa: D205,D400
        r"""Compute an array containing the indices in columns of similarity passing the first trimming (typically for
        IoU this would be the result of a simple thresholding) but it might be any method fit to do a rough filtering of
        possible ground truth candidates to match with a given detection.

        Abstract method.

        Args:
            similarity_matrix: The similarity matrix between detections and ground truths : dimension (#detection, #gt)
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[Point]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray: An array of dimension (2, N) where each column is a tuple (detection, ground truth) describing
            a potential match. To be more precise, each match-tuple in the array corresponds to a position in the
            similarity matrix which will be used by the match algorithm to compute the final match.

        """
        raise NotImplementedError

    def match(self, detections, ground_truths, label_mean_area=None):  # noqa: D205,D400
        r"""Match detections :class:`~playground_metrics.utils.geometry_utils.geometry.Geometry` with ground truth
        :class:`~playground_metrics.utils.geometry_utils.geometry.Geometry` at a given similarity matrix and trim
        method using either Coco algorithm, xView algorithm or a naive *non-unitary* match.

        Args:
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[Point]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray : A binary matrix of all matches of dimension (#detections, #ground truth)

        """
        if detections.shape[0] == 0:
            return np.zeros((0, ground_truths.shape[0]))

        if ground_truths.shape[0] == 0:
            return np.zeros((detections.shape[0], 0))

        # Geometric static typing
        if not all((isinstance(geom, self._detection_types) for geom in detections[:, 0])):
            raise TypeError('Invalid geometric type provided in '
                            'detections, expected to be on of {}'
                            ''.format(' '.join(['{}'.format(geom_type.__name__)
                                                for geom_type in self._detection_types])))
        if not all((isinstance(geom, self._ground_truth_types) for geom in ground_truths[:, 0])):
            raise TypeError('Invalid geometric type provided in '
                            'detections, expected to be on of {}'
                            ''.format(' '.join(['{}'.format(geom_type.__name__)
                                                for geom_type in self._ground_truth_types])))

        # We sort detections by confidence before computing the similarity matrix
        detections = self._sort_detection_by_confidence(detections)

        # Compute similarity matrix and An array containing the indices in columns of similarity passing the first
        # trimming (Typically for IoU this would be the result of a simple thresholding).
        similarity_matrix, similarity_matches = self._compute_similarity_matrix_and_trim(detections, ground_truths,
                                                                                         label_mean_area)

        # We match the detection and the ground truth using the configured algorithm
        if self.match_algorithm == 'coco':
            return self._coco_match(similarity_matrix, similarity_matches)
        if self.match_algorithm == 'non-unitary':
            return self._non_unitary_match(similarity_matrix, similarity_matches)
        if self.match_algorithm == 'xview':
            return self._xview_match(similarity_matrix, similarity_matches)
        raise ValueError('Invalid match algorithm: must be either coco, xview or non-unitary')

    @staticmethod
    def _sort_detection_by_confidence(detections):
        # We sort the detection by decreasing confidence
        sort_indices = np.argsort(detections[:, 1])[::-1]
        return detections[sort_indices, :]

    @staticmethod
    def _coco_match(similarity_matrix, similarity_matches):  # noqa: D205,D400
        r"""Match detections bounding boxes with ground truth bounding boxes for a given similarity matrix and trim
        method using Coco algorithm.

        Args:
            similarity_matrix: The similarity matrix between detections and ground truths : dimension (#detection, #gt)

        Returns:
            ndarray: A binary matrix of all matches of dimension (#detections, #ground truth)

        """
        # We prepare the detection match matrix
        match_matrix = np.zeros_like(similarity_matrix)

        if similarity_matches.shape[1] == 0:  # No matches at all
            return match_matrix

        forward = {match[0, 0]: match[1, :]
                   for match in np.hsplit(similarity_matches, np.where(np.diff(similarity_matches[0, :]) != 0)[0] + 1)}
        similarity_matches_by_gt = similarity_matches[:, np.argsort(similarity_matches[1, :])]
        backward = {match[1, 0]: match[0, :]
                    for match in np.hsplit(similarity_matches_by_gt,
                                           np.where(np.diff(similarity_matches_by_gt[1, :]) != 0)[0] + 1)}

        for k in range(similarity_matrix.shape[0]):
            # For each detection we select its ground truth match
            detection_matches = forward.get(k, np.zeros((0, 0)))

            # If we don't have anything left to match -> skip
            if detection_matches.size == 0:
                continue

            # We select the biggest similarity_matrix over them
            m = np.argmax(similarity_matrix[k, detection_matches])
            n = detection_matches[m]

            # We delete the ground truth column index from future match testing
            for d in backward[n]:
                forward[d] = forward[d][forward[d] != n]

            # We set the match flag to 1
            match_matrix[k, n] = 1

        return match_matrix

    @staticmethod
    def _xview_match(similarity_matrix, similarity_matches):  # noqa: D205,D400
        r"""Match detections bounding boxes with ground truth bounding boxes for a0 given similarity matrix and trim
        method using xView algorithm.

        Args:
            similarity_matrix: The similarity matrix between detections and ground truths : dimension (#detection, #gt)

        Returns:
            ndarray: A binary matrix of all matches of dimension (#detections, #ground truth)

        """
        # We prepare the detection match matrix
        match_matrix = np.zeros_like(similarity_matrix)

        if similarity_matches.shape[1] == 0:  # No matches at all
            return match_matrix

        ground_truth_match_vector = [0] * similarity_matrix.shape[1]

        forward = {match[0, 0]: match[1, :]
                   for match in np.hsplit(similarity_matches, np.where(np.diff(similarity_matches[0, :]) != 0)[0] + 1)}

        for k in range(similarity_matrix.shape[0]):
            # For each detection we select its ground truth match
            detection_matches = forward.get(k, np.zeros((0, 0)))

            # If we don't have anything left to match -> skip
            if detection_matches.size == 0:
                continue

            # We select the biggest similarity_matrix over them
            m = np.argmax(similarity_matrix[k, detection_matches])
            n = detection_matches[m]

            if ground_truth_match_vector[n] == 0:
                # We match the detection and the ground truth
                ground_truth_match_vector[n] = 1
                match_matrix[k, n] = 1

        return match_matrix

    @staticmethod
    def _non_unitary_match(similarity_matrix, similarity_matches):  # noqa: D205,D400
        r"""Match detections bounding boxes with ground truth bounding boxes for a given similarity matrix for every
        positive example yielded by the  trim method.

        Args:
            similarity_matrix: The similarity matrix between detections and ground truths : dimension (#detection, #gt)

        Returns:
            ndarray: A binary matrix of all matches of dimension (#detections, #ground truth)

        """
        # We prepare the detection match matrix
        match_matrix = np.zeros_like(similarity_matrix)
        match_matrix[similarity_matches[0, :], similarity_matches[1, :]] = 1

        return match_matrix


class MatchEngineIoU(MatchEngineBase):
    """Match detection with their ground truth according the their IoU and the detection confidence score.

    Args:
        threshold (float): The IoU threshold at which one considers a potential match as valid
        match_algorithm (str) : Either 'coco', 'xview' or 'non-unitary' to choose the match algorithm

    """

    def __init__(self, threshold, match_algorithm):
        super(MatchEngineIoU, self).__init__(match_algorithm)

        self._detection_types = (BoundingBox, Polygon)
        self._ground_truth_types = (BoundingBox, Polygon)

        self.threshold = threshold

    def _rtree_compute_iou_matrix(self, detections, ground_truths, label_mean_area=None):  # noqa: D205,D400
        r"""Compute the iou scores between all pairs of :class:`~playground_metrics.utils.geometry_utils.geometry.Geometry`
        with an Rtree on detections to speed up computation.

        Args:
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset,
                if given, it is used to match with *iIoU* instead of *IoU* (c.f. :ref:`iiou`)

        Returns:
            ndarray : An IoU matrix (#detections, #ground truth)

        """
        # We prepare the IoU matrix (#detection, #gt)
        iou = np.zeros((detections.shape[0], len(ground_truths)))

        detections = self._sort_detection_by_confidence(detections)

        # We construct a Rtree on detections
        def get_bounds(geometry):
            return geometry[0].bounds

        rtree_index = _make_rtree(detections, get_bounds)

        for k, ground_truth in enumerate(ground_truths):
            overlapping_detections = rtree_index.intersection(ground_truth[0].bounds, objects=False)
            for m in overlapping_detections:
                if label_mean_area is not None:
                    iou[m, k] = (label_mean_area / ground_truth[0].area) * \
                        detections[m, 0].intersection_over_union(ground_truth[0])
                else:
                    iou[m, k] = detections[m, 0].intersection_over_union(ground_truth[0])

        return iou

    compute_similarity_matrix = _rtree_compute_iou_matrix

    def trim_similarity_matrix(self, similarity_matrix, detections, ground_truths, label_mean_area=None):
        r"""Compute an array containing the indices in columns of similarity passing the first trimming.

        Here this is the result of a simple thresholding over IoU.

        Args:
            similarity_matrix: The similarity matrix between detections and ground truths : dimension (#detection, #gt)
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset,
                if given, it is used to match with *iIoU* instead of *IoU* (c.f. :ref:`iiou`)

        Returns:
            ndarray: An array of dimension (2, N) where each column is a tuple (detection, ground truth) describing
            a potential match. To be more precise, each match-tuple in the array corresponds to a position in the
            similarity matrix which will be used by the match algorithm to compute the final match.

        """
        res = np.stack(np.nonzero(similarity_matrix >= self.threshold))
        return res[:, np.argsort(np.nonzero(similarity_matrix >= self.threshold)[0])]


class MatchEngineEuclideanDistance(MatchEngineBase):
    """Match detection with their ground truth according the their relative distance and the detection confidence score.

    Args:
        threshold (float): The distance threshold at which one considers a potential match as valid
        match_algorithm (str) : Either 'coco', 'xview' or 'non-unitary' to choose the match algorithm

    """

    def __init__(self, threshold, match_algorithm):
        super(MatchEngineEuclideanDistance, self).__init__(match_algorithm)
        self._threshold = 1 - threshold

    @property
    def threshold(self):
        """float: The distance threshold at which one considers a potential match as valid."""
        return 1 - self._threshold

    def compute_similarity_matrix(self, detections, ground_truths, label_mean_area=None):  # noqa: D205,D400
        r"""Compute a partial similarity matrix based on the euclidean distance between all pairs of points with an
        Rtree on detections to speed up computation.

        The difference with :class:`~playground_metrics.match_detections.MatchEnginePointInBox` lies in the
        similarity matrix rough trimming which depends on a threshold rather than on whether a detection (as a point)
        lies within a ground truth polygon (or bounding box).

        The computed matrix is :math:`\mathcal{S} = 1 - \mathcal{D}` with:

        .. math::

            \mathcal{D}_{ij} = \begin{cases} \left\lVert d_i - gt_i \right\rVert_2 &\mbox{if } d_i \in B(gt_i, t)\\
                \inf &\mbox{if }  d_i \notin B(gt_i, t) \end{cases}

        Where :math:`B(gt_i, t)` is a square box centered in :math:`gt_i` of size length :math:`t`.

        Args:
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[Point]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray : An similarity matrix (#detections, #ground truth)

        """
        # We prepare the distance matrix (#detection, #gt)
        distance_matrix = np.Inf * np.ones((detections.shape[0], len(ground_truths)))

        detections = self._sort_detection_by_confidence(detections)

        # We construct a Rtree on detections
        def get_bounds(geometry):
            centroid = geometry[0].centroid
            return centroid.x, centroid.y, centroid.x, centroid.y

        rtree_index = _make_rtree(detections, get_bounds)

        for k, ground_truth in enumerate(ground_truths):
            threshold_box = (ground_truth[0].centroid.x - self.threshold, ground_truth[0].centroid.y - self.threshold,
                             ground_truth[0].centroid.x + self.threshold, ground_truth[0].centroid.y + self.threshold)
            overlapping_detections = rtree_index.intersection(threshold_box, objects=False)

            for m in overlapping_detections:
                distance_matrix[m, k] = ground_truth[0].distance(detections[m, 0])
        return 1 - distance_matrix

    def trim_similarity_matrix(self, similarity_matrix, detections, ground_truths, label_mean_area=None):
        r"""Compute an array containing the indices in columns of similarity passing the first trimming.

        Here this is the result of a simple thresholding over the distance matrix.

        Args:
            similarity_matrix: The similarity matrix between detections and ground truths : dimension (#detection, #gt)
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[Point]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray: An array of dimension (2, N) where each column is a tuple (detection, ground truth) describing
            a potential match. To be more precise, each match-tuple in the array corresponds to a position in the
            similarity matrix which will be used by the match algorithm to compute the final match.

        """
        res = np.stack(np.nonzero(similarity_matrix >= self._threshold))
        return res[:, np.argsort(np.nonzero(similarity_matrix >= self._threshold)[0])]


class MatchEnginePointInBox(MatchEngineBase):  # noqa: D205,D400
    """Match detection with their ground truth according the their relative distance, whether a detection point is in a
    ground truth box and the detection confidence score.

    Args:
        match_algorithm (str) : Either 'coco', 'xview' or 'non-unitary' to choose the match algorithm

    """

    def __init__(self, match_algorithm):
        super(MatchEnginePointInBox, self).__init__(match_algorithm)

        self._ground_truth_types = (BoundingBox, Polygon)

    def compute_similarity_matrix(self, detections, ground_truths, label_mean_area=None):  # noqa: D205,D400
        r"""Compute a partial similarity matrix based on the euclidean distance between all pairs of points with an
        Rtree on detections to speed up computation.

        The difference with :class:`~playground_metrics.match_detections.MatchEngineEuclideanDistance` lies in the
        similarity matrix rough trimming which depends on whether a detection (as a point) lies within a ground truth
        polygon (or bounding box) rather than on a threshold.

        The computed matrix is :math:`\mathcal{S} = 1 - \mathcal{D}` with:

        .. math::

            \mathcal{D}_{ij} = \left\lVert d_i - gt_i \right\rVert_2

        Args:
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray : An similarity matrix (#detections, #ground truth)

        """
        # We prepare the distance matrix (#detection, #gt)
        distance_matrix = np.Inf * np.ones((detections.shape[0], len(ground_truths)))

        detections = self._sort_detection_by_confidence(detections)

        # We construct a Rtree on detections
        def get_bounds(geometry):
            centroid = geometry[0].centroid
            return centroid.x, centroid.y, centroid.x, centroid.y

        rtree_index = _make_rtree(detections, get_bounds)

        for k, ground_truth in enumerate(ground_truths):
            overlapping_detections = rtree_index.intersection(ground_truth[0].bounds, objects=False)

            for m in overlapping_detections:
                distance_matrix[m, k] = ground_truth[0].distance(detections[m, 0])
        return 1 - distance_matrix

    def trim_similarity_matrix(self, similarity_matrix, detections, ground_truths, label_mean_area=None):
        r"""Compute an array containing the indices in columns of similarity passing the first trimming.

        Here a detection/ground truth pair is kept if the detection
        :class:`~playground_metrics.utils.geometry_utils.geometry.Point` is within the ground truth
        :class:`~playground_metrics.utils.geometry_utils.geometry.BoundingBox` or
        :class:`~playground_metrics.utils.geometry_utils.geometry.Polygon`

        Args:
            similarity_matrix: The similarity matrix between detections and ground truths : dimension (#detection, #gt)
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray: An array of dimension (2, N) where each column is a tuple (detection, ground truth) describing
            a potential match. To be more precise, each match-tuple in the array corresponds to a position in the
            similarity matrix which will be used by the match algorithm to compute the final match.

        """
        potential = np.stack(np.nonzero(similarity_matrix != -np.Inf))[:, np.argsort(np.nonzero(similarity_matrix !=
                                                                                                -np.Inf)[0])]
        trim = []
        for i in range(potential.shape[1]):
            r, c = potential[:, i]
            if detections[r, 0].intersection(ground_truths[c, 0]).is_empty:
                trim.append(i)

        return np.delete(potential, trim, axis=1)


class MatchEngineConstantBox(MatchEngineBase):  # noqa: D205,D400
    """Match detection with their ground truth according the IoU computed on fixed-size
    bounding boxes around detection and ground truth points and the detection confidence score.

    Args:
        threshold (float): The IoU threshold at which one considers a potential match as valid
        match_algorithm (str) : Either 'coco', 'xview' or 'non-unitary' to choose the match algorithm
        bounding_box_size (float): The fixed-size bounding box size

    """

    def __init__(self, threshold, match_algorithm, bounding_box_size):
        super(MatchEngineConstantBox, self).__init__(match_algorithm)
        self.bounding_box_size = bounding_box_size
        self.threshold = threshold

    def compute_similarity_matrix(self, detections, ground_truths, label_mean_area=None):  # noqa: D205,D400
        r"""Compute a parial similarity matrix based on the intersection-over-union between all pairs of constant-sized
        bounding box around points with an Rtree on detections to speed up computation.

        Args:
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[Point]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray : An IoU matrix (#detections, #ground truth)

        """
        # We prepare the distance matrix (#detection, #gt)
        similarity_matrix = np.zeros((detections.shape[0], len(ground_truths)))

        detections = self._sort_detection_by_confidence(detections)

        # We construct a Rtree on detections
        def get_bounds(geometry):
            return BoundingBox(geometry[0].centroid.x - (self.bounding_box_size // 2),
                               geometry[0].centroid.y - (self.bounding_box_size // 2),
                               geometry[0].centroid.x + (self.bounding_box_size // 2),
                               geometry[0].centroid.y + (self.bounding_box_size // 2)).bounds

        rtree_index = _make_rtree(detections, get_bounds)

        for k, ground_truth in enumerate(ground_truths):
            ground_truth_box = BoundingBox(ground_truth[0].centroid.x - (self.bounding_box_size // 2),
                                           ground_truth[0].centroid.y - (self.bounding_box_size // 2),
                                           ground_truth[0].centroid.x + (self.bounding_box_size // 2),
                                           ground_truth[0].centroid.y + (self.bounding_box_size // 2))
            overlapping_detections = rtree_index.intersection(ground_truth_box.bounds, objects=False)

            for m in overlapping_detections:
                detection_box = BoundingBox(detections[m, 0].centroid.x - (self.bounding_box_size // 2),
                                            detections[m, 0].centroid.y - (self.bounding_box_size // 2),
                                            detections[m, 0].centroid.x + (self.bounding_box_size // 2),
                                            detections[m, 0].centroid.y + (self.bounding_box_size // 2))
                similarity_matrix[m, k] = ground_truth_box.intersection_over_union(detection_box)
        return similarity_matrix

    def trim_similarity_matrix(self, similarity_matrix, detections, ground_truths, label_mean_area=None):
        r"""Compute an array containing the indices in columns of similarity passing the first trimming.

        Here this is the result of a simple thresholding over the intersection-over-union matrix.

        Args:
            similarity_matrix: The similarity matrix between detections and ground truths : dimension (#detection, #gt)
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[Point]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray: An array of dimension (2, N) where each column is a tuple (detection, ground truth) describing
            a potential match. To be more precise, each match-tuple in the array corresponds to a position in the
            similarity matrix which will be used by the match algorithm to compute the final match.

        """
        res = np.stack(np.nonzero(similarity_matrix >= self.threshold))
        return res[:, np.argsort(np.nonzero(similarity_matrix >= self.threshold)[0])]
