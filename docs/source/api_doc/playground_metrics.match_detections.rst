map\_metric\_api.match\_detections
=========================================

Usage
-----

A ``MatchEngine`` is a class which computes associations between pairs of geometrical features (typically `detections`
and `ground truths`) based on a similarity matrix (In the most common case, it is a **intersection-over-union** matrix
which is computed as described in :ref:`iou`).

The actual matching is made from the similarity matrix described before and a first trim which is used to quickly
eliminate obviously non-pairable elements.

.. seealso::
    The full matching documentation section: :ref:`matching`.

The **abstract** class :class:`~playground_metrics.match_detections.MatchEngineBase` defines the template for every
``MatchEngine``  and implements the 3 matching algorithms used (c.f. :ref:`match`). Subclasses
implements the :meth:`~playground_metrics.match_detections.MatchEngineBase.compute_similarity_matrix` method which computes
a similarity matrix from ever pairs of elements possible and the
:meth:`~playground_metrics.match_detections.MatchEngineBase.trim_similarity_matrix` method which computes the indices of
the object pairs which pass the first trim.

4 ``MatchEngine`` are implemented in this module:

* :class:`~playground_metrics.match_detections.MatchEngineIoU` which is the default ``MatchEngine`` and the one used in the
  mAP computation guide in :ref:`map_computation`. The first trim then assesses a pairs' viability by
  checking if its  similarity is above a given threshold.
* :class:`~playground_metrics.match_detections.MatchEngineEuclideanDistance` which works on points (or features' centroid
  if given non-point features) and uses a similarity matrix made from the matrix of euclidean distances. The first trim
  then assesses a pairs' viability by checking if its  similarity  is above a given threshold.
* :class:`~playground_metrics.match_detections.MatchEnginePointInBox` which works on points (or features' centroid if
  given non-point features) for detections and `bounding boxes` or `polygons` for ground truths and uses a similarity
  matrix made from the matrix of euclidean distances. The main difference with
  :class:`~playground_metrics.match_detections.MatchEngineEuclideanDistance` is that the first trim checks whether a point
  is within a ground truth footprint to assess a pairs' viability rather than checking if its similarity is above a
  given threshold.
* :class:`~playground_metrics.match_detections.MatchEngineConstantBox` which works on points (or features' centroid if
  given non-point features) and uses a similarity matrix made by assigning a fixed shape `bounding box` to every points
  and computing **intersection-over-union** between those. The first trim then assesses a pairs' viability by checking
  if its similarity is above a given threshold.


API doc
-------

.. automodule:: playground_metrics.match_detections
    :members:
    :undoc-members:
    :show-inheritance:
