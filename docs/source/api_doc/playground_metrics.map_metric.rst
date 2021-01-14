playground\_metrics.map\_metric
===============================

Usage
-----

The :class:`~playground_metrics.playground_metrics.MeanAveragePrecisionMetric` class is responsible for `mAP` computation. To do so
it relies on a **MatchEngine** whose role is to associate detection to corresponding ground truth in order to assess
whether a particular detection counts as a **False Positive** or **True Positive**. For more information on the matching
algorithms provided, please see :ref:`match`.

By default, :class:`~playground_metrics.map_metric.MeanAveragePrecisionMetric` expects either `bounding boxes` or `polygons`
but this may be overridden by providing a `user-instantiated` :class:`~playground_metrics.match_detections.MatchEngineBase`
subclasses (If no :class:`~playground_metrics.match_detections.MatchEngineBase` is provided, a default
:class:`~playground_metrics.match_detections.MatchEngineIoU` is created upon instantiation).
More information on other possible :class:`~playground_metrics.match_detections.MatchEngineBase`, please see
:doc:`playground_metrics.match_detections`.

.. note::

    * The :meth:`~playground_metrics.map_metric.MeanAveragePrecisionMetric.update` method takes an entire image/tile worth
      of inputs.
    * The following behaviour are to be expected in case of empty input:

      * No detection and no ground truth keep the **mAP** constant
      * No detection with ground truths decreases the **mAP** (they are assumed to be **False Negative**)
      * Detections with no ground truth decreases the **mAP** (they are assumed to be **False Positive**)

API doc
-------

.. automodule:: playground_metrics.map_metric
    :member-order: bysource
    :members:
    :undoc-members:
    :show-inheritance:
