import numpy as np

from playground_metrics.utils.geometry_utils import Polygon, BoundingBox, Point


def bbox_to_polygon(bbox):
    if isinstance(bbox[0], BoundingBox):
        return Polygon(list(bbox[0]._internal.exterior.coords))
    elif isinstance(bbox, (list, tuple, np.ndarray)):
        return Polygon(list(BoundingBox(*bbox[:4])._internal.exterior.coords))


def bbox_to_point(bbox):
    if isinstance(bbox[0], BoundingBox):
        return bbox[0].centroid
    elif isinstance(bbox, (list, tuple, np.ndarray)):
        return BoundingBox(*bbox[:4]).centroid


def sort_detection_by_confidence(detections):
    # We sort the detection by decreasing confidence
    sort_indices = np.argsort(detections[:, 1])[::-1]
    return detections[sort_indices, :]


def convert_point_to_constant_box(input_array, box_size):
    input_array = np.copy(input_array)
    for i in range(input_array.shape[0]):
        x, y = input_array[i, 0].centroid.x, input_array[i, 0].centroid.y
        input_array[i, 0] = BoundingBox(x - (box_size / 2), y - (box_size / 2),
                                        x + (box_size / 2), y + (box_size / 2))

    return input_array


def naive_compute_IoU_matrix(sorted_detections, ground_truths):
    """Computes the iou scores between all pairs of geometries in a naive fashion.

    Args:
        sorted_detections (ndarray, list) : A ndarray of detections stored as:

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

    Returns:
        ndarray : An IoU matrix (#detections, #ground truth)
    """

    # We prepare the IoU matrix (#detection, #gt)
    IoU = np.zeros((sorted_detections.shape[0], len(ground_truths)))

    # Naive iterative IoU matrix construction (Note: we iterate over the sorted detections)
    for k, ground_truth in enumerate(ground_truths):
        for m, detection in enumerate(sorted_detections):
            IoU[m, k] = detection[0].intersection_over_union(ground_truth[0])
    return IoU


def naive_compute_distance_similarity_matrix(sorted_detections, ground_truths):
    """Computes a similarity based on euclidean distance between all pairs of geometries in a naive fashion.

    Args:
        sorted_detections (ndarray, list) : A ndarray of detections stored as:

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

    Returns:
        ndarray : An similarity matrix (#detections, #ground truth)
    """

    # We prepare the distance matrix (#detection, #gt)
    distance_matrix = np.zeros((sorted_detections.shape[0], len(ground_truths)))

    # Naive iterative distance matrix construction (Note: we iterate over the sorted detections)
    for k, ground_truth in enumerate(ground_truths):
        for m, detection in enumerate(sorted_detections):
            distance_matrix[m, k] = detection[0].distance(ground_truth[0])
    return 1 - distance_matrix


def naive_compute_threshold_distance_similarity_matrix(sorted_detections, ground_truths, threshold):
    """Computes a similarity based on euclidean distance between all pairs of geometries in a naive fashion.

    Args:
        sorted_detections (ndarray, list) : A ndarray of detections stored as:

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

    Returns:
        ndarray : An similarity matrix (#detections, #ground truth)
    """

    # We prepare the distance matrix (#detection, #gt)
    distance_matrix = np.zeros((sorted_detections.shape[0], len(ground_truths)))

    # Naive iterative distance matrix construction (Note: we iterate over the sorted detections)
    for k, ground_truth in enumerate(ground_truths):
        for m, detection in enumerate(sorted_detections):
            if np.absolute(detection[0].centroid.x - ground_truth[0].centroid.x) > threshold or \
                    np.absolute(detection[0].centroid.y - ground_truth[0].centroid.y) > threshold:
                distance_matrix[m, k] = np.inf
            else:
                distance_matrix[m, k] = detection[0].distance(ground_truth[0])
    return 1 - distance_matrix


def naive_compute_thresholded_distance_similarity_matrix(sorted_detections, ground_truths, threshold):
    """Computes a similarity based on euclidean distance between all pairs of geometries in a naive fashion.

    Args:
        sorted_detections (ndarray, list) : A ndarray of detections stored as:

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

    Returns:
        ndarray : An similarity matrix (#detections, #ground truth)
    """

    # We prepare the distance matrix (#detection, #gt)
    distance_matrix = np.zeros((sorted_detections.shape[0], len(ground_truths)))

    # Naive iterative distance matrix construction (Note: we iterate over the sorted detections)
    for k, ground_truth in enumerate(ground_truths):
        for m, detection in enumerate(sorted_detections):
            distance_matrix[m, k] = detection[0].distance(ground_truth[0])

    distance_matrix[distance_matrix > threshold] = np.inf

    return 1 - distance_matrix


def naive_compute_point_in_box_distance_similarity_matrix(sorted_detections, ground_truths):
    """Computes a similarity based on euclidean distance between all pairs of geometries in a naive fashion.

    Args:
        sorted_detections (ndarray, list) : A ndarray of detections stored as:

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

    Returns:
        ndarray : An similarity matrix (#detections, #ground truth)
    """

    # We prepare the distance matrix (#detection, #gt)
    distance_matrix = np.zeros((sorted_detections.shape[0], len(ground_truths)))

    # Naive iterative distance matrix construction (Note: we iterate over the sorted detections)
    for k, ground_truth in enumerate(ground_truths):
        for m, detection in enumerate(sorted_detections):
            distance_matrix[m, k] = detection[0].distance(ground_truth[0])

    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if sorted_detections[i, 0].centroid.intersection(ground_truths[j, 0]).is_empty:
                distance_matrix[i, j] = np.inf

    return 1 - distance_matrix


def naive_compute_constant_box_similarity_matrix(sorted_detections, ground_truths, box_size):
    """Computes a similarity based on euclidean distance between all pairs of geometries in a naive fashion.

    Args:
        sorted_detections (ndarray, list) : A ndarray of detections stored as:

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

    Returns:
        ndarray : An similarity matrix (#detections, #ground truth)
    """

    return naive_compute_IoU_matrix(convert_point_to_constant_box(sorted_detections, box_size),
                                    convert_point_to_constant_box(ground_truths, box_size))
