import numpy as np
import shapely.geometry

from playground_metrics.utils.geometry_utils.base_geometry import BaseGeometry, AbstractBaseGeometry
from playground_metrics.utils.exception import ShapelySpecificTypeError, InvalidGeometryError


def geometry_factory(geom):
    r"""Convert a shapely.geometry object into either a BoundingBox, a Polygon or a Point object.

    Args:
        geom (:class:`shapely.geometry.BaseGeometry`): The shapely :class:`~shapely.geometry.BaseGeometry` to be
            converted into a :class:`~playground_metrics.utils.geometry_utils.geometry.Geometry` object.

    Returns:
        :class:`~map_metric_api.utils.geometry_utils.geometry.Geometry`: A
        :class:`~playground_metrics.utils.geometry_utils.geometry.Geometry` object.

    Raises:
        :exc:`~playground_metrics.utils.exception.ShapelySpecificTypeError`: If the provided shapely.geometry object has
            no reliable correspondence with any :class:`~playground_metrics.utils.geometry_utils.geometry.Geometry` type
            (e.g. if provided with a :class:`shapely.geometry.MultiPolygon`)

    """
    if not geom.is_empty and isinstance(geom, shapely.geometry.Polygon):  # Might as well be a genuine polygon or a box
        if shapely.geometry.box(*geom.bounds).equals(geom):  # It's a box
            return BoundingBox(*geom.bounds)
        else:  # It is a genuine polygon, we don't take any risk anyway
            return Polygon(list(iter(geom.exterior.coords)), *[list(iter(interior.coords))
                                                               for interior in geom.interiors])
    elif not geom.is_empty and isinstance(geom, shapely.geometry.Point):  # It's a simple point
        return Point(geom.x, geom.y)
    else:
        raise ShapelySpecificTypeError('Resulting shapely.geometry ({}) '
                                       'type may not reliably be mapped to a '
                                       'valid playground_metrics BaseGeometry'.format(geom))


class Geometry(BaseGeometry):
    r"""A concrete realisation of :class:`~playground_metrics.utils.geometry_utils.base_geometry.BaseGeometry`.

    It implements the backward mapping in :meth:`from_shapely` and acts a base Geometry type for undefined Geometries
    (such as an empty Geometry).
    """

    __slots__ = ()

    @classmethod
    def from_shapely(cls, geom):
        r"""Perform the inverse mapping from a shapely object to a Geometry type.

        To be more precise, :class:`~playground_metrics.utils.geometry_utils.base_geometry.BaseGeometry` subclasses
        implement a forward mapping from a Geometry type to a shapely object. However, because geometric operations are
        not guaranteed to keep the Geometric type (e.g. intersection between a Polygon and a Point is a Point, not a
        Polygon), and because all the actual operations are performed by shapely, it is necessary to be able to perform
        the backward mapping to transform an operation result back into valid Geometry type.

        It internally calls upon the mapping implementation in :func:`geometry_factory` and catches
        :exc:`~playground_metrics.utils.exception.ShapelySpecificTypeError` to instantiate
        a :class:`~playground_metrics.utils.geometry_utils.geometry.Geometry` in case of a undefined type.

        Args:
            geom (:class:`shapely.geometry.BaseGeometry`): The shapely :class:`~shapely.geometry.BaseGeometry` to be
                converted into a :class:`~playground_metrics.utils.geometry_utils.geometry.Geometry` object.

        Returns:
            :class:`~map_metric_api.utils.geometry_utils.base_geometry.BaseGeometry`: The backward mapping resulting
            Geometry type object.

        """
        try:
            return geometry_factory(geom)
        except ShapelySpecificTypeError:
            return Geometry(geom)


class Point(Geometry):
    r"""A point geometry.

    Args:
        x (float): The x coordinate
        y (float): The y coordinate

    Attributes:
        x (float): The x coordinate
        y (float): The y coordinate

    """

    __slots__ = ()

    def __init__(self, x, y):
        shapely_internal = shapely.geometry.Point(x, y)
        super(Point, self).__init__(shapely_internal, x=x, y=y)

    def distance(self, *others):  # noqa: D205,D400
        r"""Return the euclidean distance between the object centroid and the `other` object centroid or
        a tuple of distance if multiple input were given.

        Args:
            *others (:class:`~map_metric_api.utils.geometry_utils.base_geometry.AbstractBaseGeometry`): Other
                geometries to compute centroid distance with

        Returns:
            (float, tuple): The centroid distance or a tuple of centroid distance depending on the number of other
            geometries

        """
        if len(others) == 1:
            return self._internal.distance(others[0].centroid)
        else:
            return tuple(self._internal.distance(other.centroid) for other in others)


class Polygon(Geometry):
    r"""A polygon geometry.

    Args:
        outer (list, ndarray): An array of (x, y) couples containing the outer ring of the polygon (i.e. the shell)
        *inners (tuple): Optional. One or more array similar to `outer` containing the inner rings of polygon
            (i.e. the holes)

    Attributes:
        shell (ndarray): An array of (x, y) couples containing the outer ring of the polygon (i.e. the shell)
        holes (ndarray): One or more array similar to `outer` containing the inner rings of polygon (i.e. the holes)
            or an empty array.

    """

    __slots__ = ()

    def __init__(self, outer, *inners):
        outer = np.array(outer)
        inners = np.array(inners)
        shapely_internal = shapely.geometry.Polygon(outer, inners)
        super(Polygon, self).__init__(shapely_internal, shell=outer, holes=inners)


class FastBoundingBox(AbstractBaseGeometry):
    r"""A fast bounding box geometry.

    It differs from the canonical :class:`~playground_metrics.utils.geometry_utils.geometry.ShapelyBoundingBox` by
    delaying the shapely internal creation only to when it really is needed and tries to perform fast box-to-box
    operation whenever possible.

    Args:
        xmin (float): The left-most coordinate
        ymin (float): The upper-most coordinate
        xmax (float): The right-most coordinate
        ymax (float): The lower-most coordinate

    Raises:
        :exc:`~playground_metrics.utils.geometry_utils.InvalidGeometryError`: If ``xmax <= xmin`` or ``ymax <= ymin``.

    Attributes:
        xmin (float): The left-most coordinate
        ymin (float): The upper-most coordinate
        xmax (float): The right-most coordinate
        ymax (float): The lower-most coordinate
        area (float): The area of the object.

    """

    __slots__ = '__internal', 'xmin', 'xmax', 'ymin', 'ymax', 'area'

    def __init__(self, xmin, ymin, xmax, ymax):
        if (xmax <= xmin or ymax <= ymin) and (not xmin == xmax == ymin == ymax == np.inf):
            raise InvalidGeometryError('Invalid box coordinates (xmax ({xmax}) <= xmin ({xmin}) or '
                                       'ymax ({ymax}) <= ymin ({ymin}))'.format(xmin=xmin, ymin=ymin,
                                                                                xmax=xmax, ymax=ymax))
        # Delay _internal creation for as long as it is possible
        self.__internal = None
        super(FastBoundingBox, self).__init__(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

        # Fast coordinates access
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        # Fast area pre-computation
        self.area = (self.xmax - self.xmin) * (self.ymax - self.ymin)

    @property
    def width(self):
        """float: The box's width."""
        return self.xmax - self.xmin

    @property
    def height(self):
        """float: The box's height."""
        return self.ymax - self.ymin

    @property
    def _internal(self):
        # When we do need _internal, if it was not created, we have to now.
        if self.__internal is None:
            self.__internal = shapely.geometry.box(self.xmin, self.ymin, self.xmax, self.ymax)
        return self.__internal

    @property
    def is_empty(self):
        r"""bool: Whether the feature's `interior` and `boundary` (in point set terms) coincide with the empty set."""
        return (self.xmin == self.xmax) or (self.ymin == self.ymax)

    @property
    def bounds(self):
        r"""(float, float, float, float): A ``(minx, miny, maxx, maxy)`` tuple of coordinates that bound the object."""
        return self.xmin, self.ymin, self.xmax, self.ymax

    @property
    def centroid(self):  # noqa: D205,D400
        r""":class:`~playground_metrics.utils.geometry_utils.geometry.Point`:
        The objects centroid as a Geometry type object.
        """
        return Point(0.5 * (self.xmin + self.xmax), 0.5 * (self.ymin + self.ymax))

    def equals(self, other):  # noqa: D205,D400
        r"""Return ``True`` if the set-theoretic `boundary`, `interior`, and `exterior` of the object coincide with
        those of the other.

        Args:
            other (:class:`~map_metric_api.utils.geometry_utils.base_geometry.AbstractBaseGeometry`): Other
                geometry to test equality with

        Returns:
            bool: Whether the subsets of the 2D plane described by the 2 geometries are equal.

        """
        if hasattr(other, 'xmin') and hasattr(other, 'ymin') and hasattr(other, 'xmax') and hasattr(other, 'ymax'):
            # Try fast box-to-box equals
            return self.xmin == other.xmin and self.ymin == other.ymin and \
                self.xmax == other.xmax and self.ymax == other.ymax
        else:  # If not possible, fallback to shapely equals
            return self._internal.equals(other)

    def distance(self, *others):  # noqa: D205,D400
        r"""Return the euclidean distance between the object centroid and the `other` object centroid or
        a tuple of distance if multiple input were given.

        Args:
            *others (:class:`~map_metric_api.utils.geometry_utils.base_geometry.AbstractBaseGeometry`): Other
                geometries to compute centroid distance with

        Returns:
            (float, tuple): The centroid distance or a tuple of centroid distance depending on the number of other
            geometries

        """
        if len(others) == 1:
            return self.centroid.distance(others[0].centroid)
        else:
            return tuple(self.centroid.distance(other.centroid) for other in others)

    def _intersection_bounding_box(self, other):
        """Compute the intersection of self with bounding boxes.

        Args:
            self (FastBoundingBox, ShapelyBoundingBox): A bounding box type geometry
            other (FastBoundingBox, ShapelyBoundingBox): A bounding box type geometry

        Returns:
            (float, float, float, float) : The intersection coordinates

        """
        # Determine the (x, y)-coordinates of the intersection rectangle
        x_min = max(self.xmin, other.xmin)
        y_min = max(self.ymin, other.ymin)
        x_max = min(self.xmax, other.xmax)
        y_max = min(self.ymax, other.ymax)

        return x_min, y_min, max(x_max, x_min), max(y_max, y_min)

    def _intersection(self, other):
        if hasattr(other, 'xmin') and hasattr(other, 'ymin') and hasattr(other, 'xmax') and hasattr(other, 'ymax'):
            # Try fast box-to-box intersection
            try:
                return FastBoundingBox(*self._intersection_bounding_box(other))
            except InvalidGeometryError:
                return shapely.geometry.MultiPolygon()
        else:  # If not possible, fallback to shapely intersection
            return self._internal.intersection(other)

    def intersection(self, *others):  # noqa: D205,D400
        r"""Return a representation of the intersection of this object with the `other` geometric object or a tuple
        of intersection if multiple input were given.

        Args:
            *others (:class:`~map_metric_api.utils.geometry_utils.base_geometry.AbstractBaseGeometry`): Other
                geometries to compute intersection with

        Returns:
            (:class:`~map_metric_api.utils.geometry_utils.base_geometry.AbstractBaseGeometry`, tuple): The intersection
            or a tuple of intersection depending on the number of other geometries

        """
        if len(others) == 1:
            return self.from_shapely(self._intersection(others[0]))
        else:
            return tuple(self.from_shapely(self._intersection(other)) for other in others)

    def intersection_over_union(self, *others):  # noqa: D205,D400
        r"""Return a representation of the intersection-over-union of this object with the `other` geometric object or
        a tuple of intersection-over-union if multiple input were given (c.f. :ref:`iou`).

        Args:
            *others (:class:`~map_metric_api.utils.geometry_utils.base_geometry.AbstractBaseGeometry`): Other
                geometries to compute intersection over union with

        Returns:
            (float, tuple): The intersection over union or a tuple of intersection over union depending on the number
            of other geometries

        """
        if self.xmin == self.xmax == self.ymin == self.ymax == np.inf:
            if len(others) == 1:
                return 0.0
            else:
                return tuple(0.0 for _ in others)

        out = [None] * len(others)

        if len(others) == 1:
            inters = (self._intersection(others[0]), )
        else:
            inters = tuple(self._intersection(other) for other in others)

        for i, (other, inter) in enumerate(zip(others, inters)):

            if inter.is_empty or inter.area == 0:
                out[i] = 0.0
                continue

            # Compute the intersection over union by taking the intersection area and dividing it by the sum of
            # prediction + ground-truth areas - the intersection area
            iou = inter.area / float(self.area + other.area - inter.area)

            # Return the intersection over union value
            out[i] = iou

        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)

    @classmethod
    def from_shapely(cls, geom):
        r"""Perform the inverse mapping from a shapely object to a Geometry type.

        To be more precise, :class:`~playground_metrics.utils.geometry_utils.base_geometry.BaseGeometry` subclasses
        implement a forward mapping from a Geometry type to a shapely object. However, because geometric operations are
        not guaranteed to keep the Geometric type (e.g. intersection between a Polygon and a Point is a Point, not a
        Polygon), and because all the actual operations are performed by shapely, it is necessary to be able to perform
        the backward mapping to transform an operation result back into valid Geometry type.

        It internally calls upon the mapping implementation in :func:`geometry_factory` and catches
        :exc:`~playground_metrics.utils.exception.ShapelySpecificTypeError` to instantiate
        a :class:`~playground_metrics.utils.geometry_utils.geometry.Geometry` in case of a undefined type.

        Args:
            geom (:class:`shapely.geometry.BaseGeometry`): The shapely :class:`~shapely.geometry.BaseGeometry` to be
                converted into a :class:`~playground_metrics.utils.geometry_utils.geometry.Geometry` object.

        Returns:
            :class:`~map_metric_api.utils.geometry_utils.base_geometry.AbstractBaseGeometry`: The backward mapping
            resulting Geometry type object.

        """
        if isinstance(geom, FastBoundingBox):
            return geom
        try:
            return geometry_factory(geom)
        except ShapelySpecificTypeError:
            return Geometry(geom)


class ShapelyBoundingBox(Geometry):
    r"""A bounding box geometry.

    Args:
        xmin (float): The left-most coordinate
        ymin (float): The upper-most coordinate
        xmax (float): The right-most coordinate
        ymax (float): The lower-most coordinate

    Raises:
        :exc:`~playground_metrics.utils.geometry_utils.InvalidGeometryError`: If ``xmax <= xmin`` or ``ymax <= ymin``.

    Attributes:
        xmin (float): The left-most coordinate
        ymin (float): The upper-most coordinate
        xmax (float): The right-most coordinate
        ymax (float): The lower-most coordinate

    """

    def __init__(self, xmin, ymin, xmax, ymax):
        if (xmax <= xmin or ymax <= ymin) and (not xmin == xmax == ymin == ymax == np.inf):
            raise InvalidGeometryError('Invalid box coordinates (xmax ({xmax}) <= xmin ({xmin}) or '
                                       'ymax ({ymax}) <= ymin ({ymin}))'.format(xmin=xmin, ymin=ymin,
                                                                                xmax=xmax, ymax=ymax))
        shapely_internal = shapely.geometry.box(xmin, ymin, xmax, ymax)
        super(ShapelyBoundingBox, self).__init__(shapely_internal, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    @property
    def width(self):
        """float: The box's width."""
        return self.xmax - self.xmin

    @property
    def height(self):
        """float: The box's height."""
        return self.ymax - self.ymin

    def intersection_over_union(self, *others):  # noqa: D205,D400
        r"""Return a representation of the intersection-over-union of this object with the `other` geometric object or
        a tuple of intersection-over-union if multiple input were given (c.f. :ref:`iou`).

        Args:
            *others (:class:`~map_metric_api.utils.geometry_utils.base_geometry.AbstractBaseGeometry`): Other
                geometries to compute intersection over union with

        Returns:
            (float, tuple): The intersection over union or a tuple of intersection over union depending on the number
            of other geometries

        """
        if self.xmin == self.xmax == self.ymin == self.ymax == np.inf:
            if len(others) == 1:
                return 0.0
            else:
                return tuple(0.0 for o in others)
        return super(ShapelyBoundingBox, self).intersection_over_union(*others)


BoundingBox = FastBoundingBox
