# Third-party libraries
import playground_metrics
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import sklearn.utils.multiclass


def _sort_detection_by_confidence(detections):
    # We sort the detection by decreasing confidence
    sort_indices = np.argsort(detections[:, 1])[::-1]
    # sort_indices = np.arange(detections.shape[0])
    return detections[sort_indices, :]


def compute_mAP(ground_truths, detections, iou_threshold, match_algorithm='coco'):
    """mAP computation

    Args:
        ground_truths {list}    -- List of ground truths
        detections {list}       -- List of detections
        iou_threshold {float}   -- Threshold to use for IOU
        match_algorithm {str}   -- 'xview', 'non-unitary' or 'coco' to choose the matching algorithm

    Returns:
        {float} -- Computed mean average precision

    """
    map_computer = playground_metrics.MeanAveragePrecisionMetric(iou_threshold, match_algorithm, trim_invalid_geometry=True)
    map_computer.update(detections, ground_truths)
    mean_average_precision = map_computer.compute()
    map_computer.reset()
    return mean_average_precision


def compute_metrics(ground_truths, detections, iou_threshold, match_algorithm='coco'):
    """Compute metrics and format them in a dataframe

    Args:
        ground_truths {list}    -- List of ground truths
        detections {list}       -- List of detections
        iou_threshold {float}   -- Threshold to use for IOU
        match_algorithm {str}   -- 'xview', 'non-unitary' or 'coco' to choose the matching algorithm

    Returns:
        {pd.DataFrame} -- Dataframe with all metrics for the selected ground truths and predictions

    """
    # Compute metrics
    map_computer = playground_metrics.MeanAveragePrecisionMetric(iou_threshold, match_algorithm, trim_invalid_geometry=True)
    map_computer.update(detections, ground_truths)
    map_computer.compute()

    # Re-map metrics
    metrics_dict = {
        tag.lower(): [
            iou_threshold, map_computer.precision_per_class[tag], map_computer.recall_per_class[tag],
            map_computer.number_true_detection_per_class[tag], map_computer.number_false_detection_per_class[tag],
            map_computer.number_missed_ground_truth_per_class[tag], 0
        ]
        for tag in map_computer.precision_per_class.keys()
    }

    # Reset computer
    map_computer.reset()

    # Create dataframe
    columns = ['Threshold', 'Precision', 'Recall', 'TP', 'FP', 'FN', 'TN']
    return pd.DataFrame.from_dict(metrics_dict, orient='index', columns=columns).sort_index()


def compute_confusion_matrix(ground_truths, detections, iou_threshold, match_algorithm='coco'):
    """Compute confusion matrix.

    Args:
        ground_truths {list}    -- List of ground truths
        detections {list}       -- List of detections
        iou_threshold {float}   -- Threshold to use for IOU
        match_algorithm {str}   -- 'xview', 'non-unitary' or 'coco' to choose the matching algorithm

    Returns:
        {np.ndarray} -- Dataframe with all metrics for the selected ground truths and predictions

    """
    # Init engine
    match_engine = playground_metrics.match_detections.MatchEngineIoU(iou_threshold, match_algorithm)

    # Convert to map metric format
    _, gt_array = playground_metrics.utils.geometry_utils.get_type_and_convert(ground_truths, trim_invalid_geometry=True)
    _, detec_array = playground_metrics.utils.geometry_utils.get_type_and_convert(detections, trim_invalid_geometry=True)

    # Sort detections by confidence
    # FIXME: detections are sorted twice in detection scoring library
    detec_array_2 = _sort_detection_by_confidence(detec_array)
    detec_array_3 = _sort_detection_by_confidence(detec_array_2)

    # Match ground truths with detections
    match_matrix = match_engine.match(detec_array[:, :2], gt_array[:, :1])

    # Compute the detected ground truth (without classif) and detection
    left_match = match_matrix.sum(axis=1)
    right_match = match_matrix.sum(axis=0)

    # Compute confusion matrix
    y_true, y_pred = [], []

    for i, j in zip(*np.where(match_matrix == 1)):
        y_true.append(gt_array[j][1])
        y_pred.append(detec_array_3[i][2])
    for i in np.where(left_match == 0)[0]:
        y_true.append('')
        y_pred.append(detec_array_3[i][2])
    for j in np.where(right_match == 0)[0]:
        y_true.append(gt_array[j][1])
        y_pred.append('')

    # Labels to use
    labels = sklearn.utils.multiclass.unique_labels(y_true, y_pred)

    # Compute confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels)

    # Put it in a dataframe
    confusion_matrix_df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

    # Plot heatmap
    return confusion_matrix, confusion_matrix_df, labels
