from enum import Enum
from os import cpu_count
from typing import Sequence

import dask.array
import numpy as np
from pygeos import intersection, union, area, STRtree, get_type_id, get_coordinates, box, centroid, bounds, distance, \
    prepare


class GeometryType(Enum):
    """An enumeration of all  geometry types available in GEOS with their respective GEOS numeric code."""

    NONE = -1
    POINT = 0
    LINESTRING = 1
    LINEARRING = 2
    POLYGON = 3
    MULTIPOINT = 4
    MULTILINESTRING = 5
    MULTIPOLYGON = 6
    GEOMETRYCOLLECTION = 7


def _sanitize(x):
    if len(x.shape) > 1:
        for i in range(1, len(x.shape)):
            x = x.squeeze(i)
    prepare(x)
    return x


class _IntersectionOverUnionRTree:
    @staticmethod
    def _area_op_threaded(operation, candidates, x, y, default):
        areas = default((len(x), len(y)))

        # Define the operation
        x_dask = dask.array.from_array(x[candidates[:, 0]], chunks=int(len(x[candidates[:, 0]]) // cpu_count()))
        y_dask = dask.array.from_array(y[candidates[:, 1]], chunks=int(len(y[candidates[:, 1]]) // cpu_count()))
        res = x_dask.map_blocks(operation, y_dask, dtype=object)
        res = res.map_blocks(area, dtype=float).squeeze()

        areas[candidates[:, 0], candidates[:, 1]] = res.compute(scheduler="threads",
                                                                optimize_graph=True,
                                                                num_workers=cpu_count())
        return areas

    @staticmethod
    def _area_op(operation, candidates, x, y, default):
        areas = default((len(x), len(y)))
        areas[candidates[:, 0], candidates[:, 1]] = area(operation(x[candidates[:, 0]], y[candidates[:, 1]])).squeeze()
        return areas

    def _call(self, x, y, force_thread=False):
        x, y = _sanitize(x), _sanitize(y)

        tree = STRtree(y)
        candidates = tree.query_bulk(x, predicate='intersects').transpose()

        area_op = self._area_op_threaded \
            if force_thread else self._area_op_threaded \
            if x.shape[0] * y.shape[0] > (cpu_count() * 100) else self._area_op

        return area_op(intersection, candidates, x, y, np.zeros) / area_op(union, candidates, x, y, np.ones)

    def __call__(self, x, y, force_thread=False):
        if x.size == 0 or y.size == 0:
            return np.zeros((len(x), len(y)))

        if x.size == 1 and y.size == 1:
            return self._call(x, y, force_thread=force_thread)

        return self._call(x, y, force_thread=force_thread)


_iou_rtree_computer = _IntersectionOverUnionRTree()


def intersection_over_union(x, y, force_thread=False):
    """Compute the intersection-over-union in between every possible geometry pairs from two arrays of geometries.

    Args:
        x (ArrayLike): An array of geometries.
        y (ArrayLike): An array of geometries.
        force_thread (bool): Force the use of the threaded implementation. Note that this can incur a significant
            computational cost for small input and fail on input too small to be reliably chunked.

    Returns:
        numpy.ndarray: An intersection-over-union matrix.

    """
    if not isinstance(x, (Sequence, np.ndarray)):
        x = [x]

    if not isinstance(y, (Sequence, np.ndarray)):
        y = [y]

    x, y = np.asarray(x), np.asarray(y)

    return _iou_rtree_computer(x, y, force_thread=force_thread)


class _EuclideanDistanceRTree:
    @staticmethod
    def _distance_op_threaded(candidates, x, y):
        point_x = as_points(x)
        point_y = as_points(y)

        areas = np.Inf * np.ones((len(x), len(y)))

        # Define the operation
        x_dask = dask.array.from_array(point_x[candidates[:, 0]], chunks=int(len(x[candidates[:, 0]]) // cpu_count()))
        y_dask = dask.array.from_array(point_y[candidates[:, 1]], chunks=int(len(y[candidates[:, 1]]) // cpu_count()))
        res = x_dask.map_blocks(distance, y_dask, dtype=float).squeeze()

        areas[candidates[:, 0], candidates[:, 1]] = res.compute(scheduler="threads",
                                                                optimize_graph=True,
                                                                num_workers=cpu_count())
        return areas

    @staticmethod
    def _distance_op(candidates, x, y):
        point_x = as_points(x)
        point_y = as_points(y)

        areas = np.Inf * np.ones((len(x), len(y)))

        areas[candidates[:, 0], candidates[:, 1]] = distance(point_x[candidates[:, 0]],
                                                             point_y[candidates[:, 1]]).squeeze()

        return areas

    def _call(self, x, y, force_thread=False):
        x, y = _sanitize(x), _sanitize(y)

        tree = STRtree(y)
        candidates = tree.query_bulk(x).transpose()

        distance_op = self._distance_op_threaded \
            if force_thread else self._distance_op_threaded \
            if x.shape[0] * y.shape[0] > (cpu_count() * 100) else self._distance_op

        return 1 - distance_op(candidates, x, y)

    def __call__(self, x, y, force_thread=False):
        if x.size == 0 or y.size == 0:
            return np.zeros((len(x), len(y)))

        if x.size == 1 and y.size == 1:
            return self._call(x, y, force_thread=force_thread)

        return self._call(x, y, force_thread=force_thread)


_distance_rtree_computer = _EuclideanDistanceRTree()


def euclidean_distance(x, y, force_thread=False):
    """Compute the euclidean distance in between every possible centroid pairs from two arrays of geometries.

    Args:
        x (ArrayLike): An array of geometries.
        y (ArrayLike): An array of geometries.
        force_thread (bool): Force the use of the threaded implementation. Note that this can incur a significant
            computational cost for small input and fail on input too small to be reliably chunked.

    Returns:
        numpy.ndarray: An intersection-over-union matrix.

    """
    if not isinstance(x, (Sequence, np.ndarray)):
        x = [x]

    if not isinstance(y, (Sequence, np.ndarray)):
        y = [y]

    x, y = np.asarray(x), np.asarray(y)

    return _distance_rtree_computer(x, y, force_thread=force_thread)


def point_to_box(x, width=64., height=64.):
    """Convert an array of points to an array of constant size boxes.

    Args:
        x (numpy.ndarray): An array of points.
        width (float): The output boxes' width.
        height (float): The output boxes' width.

    Returns:
        numpy.ndarray: An array of boxes.

    """
    x = enforce_point(x)
    coordinates = get_coordinates(x)
    return box(coordinates[:, 0] - (width / 2),
               coordinates[:, 1] - (height / 2),
               coordinates[:, 0] + (width / 2),
               coordinates[:, 1] + (height / 2))


def is_type(x, *geometry_types):
    """Return whether geometries in the provided array are of one of the provided types.

    Args:
        x (numpy.ndarray): A 1D ``[geometry]`` array.
        *geometry_types (GeometryType): A geometry type.

    Returns:
        numpy.ndarray: A boolean array of whether geometries are of any of the provided``geometry_types``.

    """
    types = get_type_id(x)
    return sum(types == geometry_type.value for geometry_type in geometry_types).astype(bool)


def as_points(x):
    """Convert an array of geometries to an array of centroid points.

    Args:
        x (numpy.ndarray): An array of geometries.

    Returns:
        numpy.ndarray: An array of points.

    """
    return centroid(x)


def as_boxes(x):
    """Convert an array of geometries to an array of bounding boxes polygons.

    Args:
        x (numpy.ndarray): An array of points.

    Returns:
        numpy.ndarray: An array of polygons.

    """
    coordinates = bounds(x)
    return box(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], coordinates[:, 3])


def enforce_type(x, convert_fn, *geometry_types, in_place=False):
    """Convert geometries which are not of one of the provided types with the provided conversion function.

    Args:
        x (numpy.ndarray): A 1D ``[geometry]`` array.
        convert_fn (callable): A geometry type conversion function.
        *geometry_types (GeometryType): A geometry type.
        in_place (bool): If ``True`` convert the geometries in place.

    Returns:
        numpy.ndarray: An array of geometries, some of which where converted.

    """
    x = np.array(x, copy=not in_place)
    offending_mask = ~is_type(x, *geometry_types)
    x[offending_mask] = convert_fn(x[offending_mask])
    return x


def enforce_point(x, in_place=False):
    """Convert non-point geometries to points.

    Args:
        x (numpy.ndarray): A 1D ``[geometry]`` array.
        in_place (bool): If ``True`` convert the geometries in place.

    Returns:
        numpy.ndarray: An array of points, some of which where converted.

    """
    return enforce_type(x, as_points, GeometryType.POINT, in_place=in_place)


def enforce_polygon(x, in_place=False):
    """Convert non-polygon geometries to polygons.

    Args:
        x (numpy.ndarray): A 1D ``[geometry]`` array.
        in_place (bool): If ``True`` convert the geometries in place.

    Returns:
        numpy.ndarray: An array of points, some of which where converted.

    """
    return enforce_type(x, as_boxes, GeometryType.POLYGON, in_place=in_place)
