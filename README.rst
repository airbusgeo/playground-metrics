Playground mAP scoring python API
=================================

Jump to the `complete documentation <https://playground-metrics.readthedocs.io/en/latest/>`_ on ReadTheDocs.

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

Installation
------------

The library is pure-python and can be installed with ``pip`` from PyPI:

.. code-block:: shell

    pip install playground-metrics

However the ``rtree`` library used for fast intersection-over-union matrix
computation is not a pure-python one and depends on the ``libspatialindex``
C library which have to be separately installed.

For information on how to install libspatialindex or rtree for your particular
environment please refer to their respective documentation
(`libspatialindex <http://libspatialindex.github.io/>`_
and `rtree <http://toblerity.org/rtree/>`_).

For Ubuntu/Debian user, ``libspatialindex`` is in the ``libspatialindex-dev`` package.

To install from source, simply clone the repository and checkout the
correct tag if you wish to install a specific version rather than master.

The module may then be installed preferably with ``pip`` via:

.. code:: bash

   pip install .

Usage
-----

A basic usage example where data is read from a geojson per image:

.. code-block:: python

   import json
   import playground_metrics

   # Instanciate the Metric computer object
   map_computer = playground_metrics.MeanAveragePrecisionMetric(0.5, 'coco')

   def to_coordinates(feature):
       """Make valid coordinates out of a GeoJSON feature."""
       coordinates = feature['geometry']['coordinates']
       if len(coordinates) > 1:
           return [coordinates[0], coordinates[1:]]
       return coordinates

   # Load relevant data for the first image
   with open('detection_1.json', 'r') as f:
       data = json.load(f)
       detections = []
       for feature in data['features']:
           detections.append([to_coordinates(feature),
                              feature['properties']['confidence'],
                              feature['properties']['tags']])

   with open('gt_1.json', 'r') as f:
       data = json.load(f)
       ground_truths = []
       for feature in data['features']:
           ground_truths.append([to_coordinates(feature),
                                 feature['properties']['tags']])

   # Update
   map_computer.update(detections, ground_truths)

   # Load relevant data for the second image
   with open('detection_2.json', 'r') as f:
       data = json.load(f)
       detections = []
       for feature in data['features']:
           detections.append([to_coordinates(feature),
                              feature['properties']['confidence'],
                              feature['properties']['tags']])

   with open('gt_2.json', 'r') as f:
       data = json.load(f)
       ground_truths = []
       for feature in data['features']:
           ground_truths.append([to_coordinates(feature),
                                 feature['properties']['tags']])

   # Update
   map_computer.update(detections, ground_truths)

   # Compute metric from accumulated values
   metric = map_computer.compute()

   # Reset before restarting
   map_computer.reset()

   # And so on...

Computing a metric is thus done with a sequence of ``update()`` calls followed by
a ``compute()`` call to get the final metric value and an eventual
``reset()`` call whenever one wishes to start over.
