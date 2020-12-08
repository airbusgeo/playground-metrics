mAP metric API
==============

The *mAP* API is accessible from the :mod:`~playground_metrics.map_metric` and :mod:`~playground_metrics.match_detections` module.

For more information on the role of each module see :ref:`map_computation`.

For a basic usage example, please see

Modules
-------

The :mod:`~playground_metrics.match_detections` module exposes the matching API used by :mod:`~playground_metrics.map_metric`.


Main modules
............

.. toctree::

   playground_metrics.map_metric
   playground_metrics.match_detections

Helper modules
..............

Helper modules are a collection of *classes* and *functions* which extends the functionality of the main **mAP**
computer.

.. toctree::

   playground_metrics.metrics_helper


Utils
-----

The *Utils* section exposes *classes* and *functions* used by either of the mentioned modules in their computation
which may be useful to understand the scoring algorithm.

.. toctree::

    playground_metrics.utils.iou_utils
    playground_metrics.utils.geometry_utils


