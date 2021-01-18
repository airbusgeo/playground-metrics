Playground metrics documentation
=================================

mAP stands for *mean average precision* and is a common metric in detection tasks and challenges.

In an effort to standardize evaluation tasks to implement efficient benchmarkings of algorithms, the following module
implements a generic python API to compute **mAP**, **AP** per label as well as **precision** and **recall** per label.

Basic usage example:

.. code-block:: python

    >>> import numpy as np
    >>> import playground_metrics
    >>> detections = np.array([[1, 3, 12, 14, 0.8, 0], [23, 14, 33, 25, 0.9, 0]])
    >>> ground_truths = np.array([[2, 6, 11, 16, 0], [20, 11, 45, 25, 0]])
    >>> map_computer = playground_metrics.MeanAveragePrecisionMetric(0.5, 'coco')
    >>> map_computer.update(detections, ground_truths)
    >>> map_computer.compute()
    0.25
    >>> map_computer.reset()

This implementation was inspired by the
`Coco <https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools>`_ and the
`xView <https://github.com/DIUx-xView/baseline/tree/master/scoring>`_
implementations, both of which build upon the
`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit>`_
implementation.

Computation algorithm and API usage are documented below.

.. toctree::
   :maxdepth: 4
   :caption: Guides:

   getting_started
   guides

.. toctree::
   :maxdepth: 4
   :caption: API:

   api_doc/playground_metrics
