Quickstart
----------

Installation
............

The library is pure-python and can be installed with ``pip`` from PyPI:

.. code-block:: shell

    pip install playground-metrics

However the ``rtree`` library used for fast intersection-over-union matrix
computation is not a pure-python one and depends on the ``libspatialindex``
C library which have to be separately installed.

For information on how to install libspatialindex or rtree for your particular
environment please refer to their respective documentation
(`here <http://libspatialindex.github.io/>`_
and `here <http://toblerity.org/rtree/>`_).

For Ubuntu/Debian user, ``libspatialindex`` is in the ``libspatialindex-dev`` package.

.. hint::

    To run the tests, extra requirements are needed and may be installed
    through ``pip``:

    .. code:: bash

       pip install playground-metrics[tests]

    Similarly, to build the documentation, extra packages may be installed
    with:

    .. code:: bash

       pip install playground-metrics[docs]

To install from source, simply clone the repository and checkout the
correct tag if you wish to install a specific version rather than master.

The module may then be installed preferably with ``pip`` via:

.. code:: bash

   pip install .

Usage
.....

.. _quickstart_usage:

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

.. warning::
    The ``update()`` input format for polygon expects ``[[outer_ring], [inner_rings]]`` which
    is **not** the content of a GeoJSON feature's geometry coordinates, hence the conversion
    function in the snippet above.

This is because every metric provided in the package follow the same interface contract which
is equivalent to them being inherited from the following abstract base class prototype:

.. py:class:: Metric

   Bases: :class:`abc.ABC`

   .. py:method:: Metric.update(detections, ground_truths)

      Accumulates values necessary to compute the metric with detections and ground truths from a **single** image

      :param detections: A ndarray of detections stored as:

                         * Bounding boxes for a given class where each row is a detection stored as:
                          ``[x_min, y_min, x_max, y_max, confidence, label]``
                         * Polygons for a given class where each row is a detection stored as:
                           ``[[[outer_ring], [inner_rings]], confidence, label]``
                         * Points for a given class where each row is a detection stored as:
                           ``[x, y, confidence, label]``

      :type detections: ndarray, list
      :param ground_truths: A ndarray of ground truth stored as:

                         * Bounding boxes for a given class where each row is a ground truth stored as:
                           ``[x_min, y_min, x_max, y_max, label]``
                         * Polygons for a given class where each row is a ground truth stored as:
                           ``[[[outer_ring], [inner_rings]], label]``
                         * Points for a given class where each row is a ground truth stored as:
                           ``[x, y, label]``

      :type ground_truths: ndarray,list

      .. note::
          Note that the labels provided in the input arrays can theoretically be any hashable type, however,
          only numeric types, strigns and tuples are officially supported.

   .. py:method:: Metric.compute()

      Computes the metric according to the accumulated values

      :returns: The computed metric
      :rtype: float

   .. py:method:: Metric.reset()

      Resets all intermediate and return values to their initial value.

      If ``reset()`` is not called in-between two ``compute()`` call, the values returned by ``compute()`` will take
      into account the entire prediction stack, not just the predictions in-between the two ``compute()`` calls.
