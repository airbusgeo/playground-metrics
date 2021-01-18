import numpy as np

from ..match_detections import MatchEngineIoU


def add_confidence_from_max_iou(detections, ground_truths):
    r"""Compute confidence scores for detections based on the maximum IoU between detections and ground truths.

    Args:
        detections (ndarray, list) : A ndarray of detections stored as:

            * Bounding boxes for a given class where each row is a detection stored as:
              ``[BoundingBox]``
            * Polygons for a given class where each row is a detection stored as:
              ``[Polygon]``

        ground_truths (ndarray,list) : A ndarray of ground truth stored as:

            * Bounding boxes for a given class where each row is a ground truth stored as:
              ``[BoundingBox]``
            * Polygons for a given class where each row is a ground truth stored as:
              ``[Polygon]``

    Returns:
        ndarray: The detection with an additional column containing maximum IoU between detections and ground truths
        to use as confidence scores, as:

        * Bounding boxes for a given class where each row is a detection stored as:
          ``[BoundingBox, confidence]``
        * Polygons for a given class where each row is a detection stored as:
          ``[Polygon, confidence]``


    Examples:
        With a bounding box input:

        >>> from playground_metrics.utils.geometry_utils import BoundingBox
        >>> detections = np.array([[BoundingBox(0, 0, 9, 5)],
        ...                        [BoundingBox(23, 13, 29, 18)]])
        >>> ground_truths = np.array([[BoundingBox(5, 2, 15, 9)],
        ...                           [BoundingBox(18, 10, 26, 15)]])
        >>> add_confidence_from_max_IoU(detections, ground_truths)
        array([[BoundingBox(xmin=0, ymin=0, xmax=9, ymax=5), 0.11650485436893204],
               [BoundingBox(xmin=23, ymin=13, xmax=29, ymax=18), 0.09375]],
              dtype=object)

        The same with polygons:

        >>> from playground_metrics.utils.geometry_utils import Polygon
        >>> detections = np.array([[Polygon([[0.0, 0.0], [0.0, 5.0], [9.0, 5.0], [9.0, 0.0]])],
        ...                       [Polygon([[23.0, 13.0], [23.0, 18.0], [29.0, 18.0], [29.0, 13.0]])]])
        >>> ground_truths = np.array([[Polygon([[5, 2], [5, 9], [15, 9], [15, 2]])],
        ...                           [Polygon([[18, 10], [18, 15], [26, 15], [26, 10]])]])
        >>> add_confidence_from_max_IoU(detections, ground_truths)
        array([[Polygon(shell=[[0. 0.] [0. 5.] [9. 5.] [9. 0.]], holes=[]),
                0.11650485436893204],
               [Polygon(shell=[[23. 13.] [23. 18.] [29. 18.] [29. 13.]], holes=[]),
                0.09375]], dtype=object)

    """
    match_engine = MatchEngineIoU(0.5, 'coco')
    iou = match_engine.compute_similarity_matrix(np.insert(detections, 1, np.linspace(1, 0,
                                                                                      detections.shape[0]), axis=1),
                                                 ground_truths)
    return np.insert(detections, 1, np.max(iou, axis=1), axis=1)
