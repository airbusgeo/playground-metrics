from abc import ABC, abstractmethod
from collections.abc import MutableMapping

import numpy as np
import shapely.geometry

from ..exception import InvalidGeometryError


class AbstractBaseGeometry(ABC):
    r"""Abstract base class for every Geometry types.

    It gives exepcted valid interfaces of every geometric types,
    a pythonic repr and a convenient access to build-time parameters saved through ``**kwargs``.

    Subclasses must implement every abstract interfaces to be considered a valid class.

    Args:
        **kwargs (dict): Assumed as the build time coordinates arguments, passed to private attribute instance of
            :class:`~playground_metrics.utils.geometry_utils.base_geometry.GeometryCoordinates` to allow easy repr and
            attribute access (c.f. example).

    """

    __slots__ = '_coordinates'

    def __init__(self, **kwargs):
        super(AbstractBaseGeometry, self).__init__()
        self._coordinates = GeometryCoordinates(**kwargs)

    def __repr__(self):
        """Represent the geometry and its coordinates."""
        return '{}{}({})'.format('Empty' if self.is_empty else '', self.__class__.__name__, self._coordinates)

    def __dir__(self):
        """Return dir of geometry and its coordinates."""
        return list(super(AbstractBaseGeometry, self).__dir__()) + list(self._coordinates.keys())

    def __getattr__(self, item):
        r"""Allow coordinate access from outside as well as shapely internal attributes access.

        Not that overriding __getattr__ instead of __getattribute__ allows actual attributes to take precedence over
        dynamic attributes mapping.
        """
        try:
            return self._coordinates.__getattr__(item)
        except AttributeError:
            try:
                return self._internal.__getattribute__(item)
            except AttributeError:
                raise AttributeError('\'{}\' object has no attribute \'{}\''.format(self.__class__.__name__, item))

    @property
    @abstractmethod
    def is_empty(self):
        r"""bool: Whether the feature's `interior` and `boundary` (in point set terms) coincide with the empty set."""
        return NotImplemented

    @property
    @abstractmethod
    def area(self):
        r"""float: The area of the object."""
        return NotImplemented

    @property
    @abstractmethod
    def bounds(self):
        r"""(float, float, float, float): A ``(minx, miny, maxx, maxy)`` tuple of coordinates that bound the object."""
        return NotImplemented

    @property
    @abstractmethod
    def centroid(self):  # noqa: D205,D400
        r""":class:`~playground_metrics.utils.geometry_utils.geometry.Point`:
        The objects centroid as a Geometry type object.
        """
        return NotImplemented

    def __eq__(self, other):  # noqa: D205,D400
        r"""Return ``True`` if the two objects are of the same geometric type, and the coordinates of the two objects
        match precisely.

        Args:
            other (:class:`~map_metric_api.utils.geometry_utils.base_geometry.AbstractBaseGeometry`): Other
                geometry to test equality with

        Returns:
            bool: Whether the 2 geometries are equal

        """
        if isinstance(other, AbstractBaseGeometry):
            return self.__class__ == other.__class__ and self._coordinates.__eq__(other._coordinates)
        else:
            return NotImplemented

    @abstractmethod
    def equals(self, other):  # noqa: D205,D400
        r"""Return ``True`` if the set-theoretic `boundary`, `interior`, and `exterior` of the object coincide with
        those of the other.

        Args:
            other (:class:`~map_metric_api.utils.geometry_utils.base_geometry.AbstractBaseGeometry`): Other
                geometry to test equality with

        Returns:
            bool: Whether the subsets of the 2D plane described by the 2 geometries are equal

        """
        return NotImplemented

    @abstractmethod
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
        return NotImplemented

    @abstractmethod
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
        return NotImplemented

    @abstractmethod
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
        return NotImplemented


class BaseGeometry(AbstractBaseGeometry):
    r"""Abstract base class for every Geometry types.

    It is merely a wrapper around :mod:`shapely.geometry` which
    implements a few differences in some binary operation, an intersection-over-union operator, a pythonic repr and
    a convenient access to build-time parameters saved through ``**kwargs``.

    Subclasses must implement the
    :meth:`~playground_metrics.utils.geometry_utils.base_geometry.BaseGeometry.from_shapely` class method to be
    instantiable.

    Args:
        shapely_internal (:class:`shapely.geometry.BaseGeometry`): A shapely
            :class:`~shapely.geometry.BaseGeometry` which will serve as the backend object for every geometric
            operations performed later on.
        **kwargs (dict): Assumed as the build time coordinates arguments, passed to private attribute instance of
            :class:`~playground_metrics.utils.geometry_utils.base_geometry.GeometryCoordinates` to allow easy repr and
            attribute access (c.f. example).

    Raises:
        ValueError: If the provided shapely geometry is not shapely.geometry geometry.
        :exc:`~playground_metrics.utils.geometry_utils.InvalidGeometryError`: If the provided geometry is deemed not
            valid by shapely.

    Examples:
        >>> class Point(BaseGeometry):
        ...    def __init__(self, x, y):
        ...        shapely_internal = shapely.geometry.Point(x, y)
        ...        super(Point, self).__init__(shapely_internal, x=x, y=y)
        ...
        ...    @classmethod   # To allow instantiation
        ...    def from_shapely(cls, geom):  # This is not a valid implementation
        ...        pass
        >>> point = Point(4, 6)
        >>> point
        Point(x=4, y=6)
        >>> point.x
        4
        >>> point.y
        6
        >>> print(point._internal)  # The underlying shapely object passed as shapely_internal to the constructor
        POINT (4 6)

    """

    __slots__ = '_internal',

    def __init__(self, shapely_internal, **kwargs):
        super(BaseGeometry, self).__init__()
        if not isinstance(shapely_internal, shapely.geometry.base.BaseGeometry):
            raise ValueError('BaseGeometry internal is not a valid '
                             'shapely.BaseGeometry instance '
                             '(found {} instead)'.format(shapely_internal.__class__.__name__))
        self._coordinates = GeometryCoordinates(**kwargs)
        if not shapely_internal.is_valid and not shapely_internal.bounds == (np.inf, np.inf, np.inf, np.inf):
            raise InvalidGeometryError('Invalid shapely.BaseGeometry created from '
                                       'provided parameters ({})'.format(self._coordinates))
        self._internal = shapely_internal

    @property
    def is_empty(self):
        r"""bool: Whether the feature's `interior` and `boundary` (in point set terms) coincide with the empty set."""
        return self._internal.is_empty

    @property
    def area(self):
        r"""float: The area of the object."""
        return self._internal.area

    @property
    def bounds(self):
        r"""(float, float, float, float): A ``(minx, miny, maxx, maxy)`` tuple of coordinates that bound the object."""
        return self._internal.bounds

    @property
    def centroid(self):  # noqa: D205,D400
        r""":class:`~playground_metrics.utils.geometry_utils.geometry.Point`:
        The objects centroid as a Geometry type object
        """
        return self.from_shapely(self._internal.centroid)

    def equals(self, other):  # noqa: D205,D400
        r"""Return ``True`` if the set-theoretic `boundary`, `interior`, and `exterior` of the object coincide with
        those of the other.

        Args:
            other (:class:`~map_metric_api.utils.geometry_utils.base_geometry.AbstractBaseGeometry`): Other
                geometry to test equality with

        Returns:
            bool: Whether the subsets of the 2D plane described by the 2 geometries are equal.

        """
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
            return self._internal.centroid.distance(others[0].centroid)
        else:
            return tuple(self._internal.centroid.distance(other.centroid) for other in others)

    def _intersection(self, other):
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
    @abstractmethod
    def from_shapely(cls, geom):
        r"""Perform the inverse mapping from a shapely object to a Geometry type.

        To be more precise, :class:`~playground_metrics.utils.geometry_utils.base_geometry.BaseGeometry` subclasses
        implement a forward mapping from a Geometry type to a shapely object. However, because geometric operations are
        not guaranteed to keep the Geometric type (e.g. intersection between a Polygon and a Point is a Point, not a
        Polygon), and because all the actual operations are performed by shapely, it is necessary to be able to perform
        the backward mapping to transform an operation result back into valid Geometry type.

        The method was made left to subclasses to implement to avoid the base class to be explicitly aware of its
        subclasses, which would be bad design.

        Args:
            geom (:class:`shapely.geometry.BaseGeometry`): The shapely :class:`~shapely.geometry.BaseGeometry` to be
                converted into a :class:`~playground_metrics.utils.geometry_utils.geometry.Geometry` object.

        Returns:
            :class:`~map_metric_api.utils.geometry_utils.base_geometry.BaseGeometry`: The backward mapping resulting
            Geometry type object.

        """
        raise NotImplementedError


class GeometryCoordinates(MutableMapping):
    r"""A container for coordinates with dict-like interface.

    It stores every keyword arguments passed to its constructor and makes them accessible as attributes as well.

    It is used internally by :class:`~playground_metrics.utils.geometry_utils.base_geometry.BaseGeometry` to delegate
    some coordinates specific function or representation.

    """

    __slots__ = '_internal',

    def __init__(self, **kwargs):
        super(GeometryCoordinates, self).__setattr__('_internal', kwargs)

    def __eq__(self, other):
        """Return True if two GeometryCoordinates instances have the same coordinates."""
        if isinstance(other, GeometryCoordinates):
            if list(self.keys()) == list(other.keys()):
                for k in self.keys():
                    if isinstance(self[k], np.ndarray) and isinstance(other[k], np.ndarray):
                        if not np.all(self[k] == other[k]):
                            return False
                    elif not self[k] == other[k]:
                        return False
                return True
            else:
                return False
        else:
            return NotImplemented

    def __repr__(self):
        """Return the coordinate as a comma-separated key=value format string."""
        return ' '.join(', '.join(('{}={}'.format(key, value)
                                   for key, value in self._internal.items())).replace('\n', '').split())

    def __dir__(self):
        """Return the coordinate dir."""
        return super(GeometryCoordinates, self).__dir__() + list(self._internal.keys())

    def __getattr__(self, key):
        """Get a coordinate value."""
        try:
            return self._internal[key]
        except KeyError:
            raise AttributeError('\'GeometryCoordinates\' object has no attribute \'{}\''.format(key))

    def __setattr__(self, key, value):
        """Add or set a coordinate."""
        self._internal[key] = value

    def __delattr__(self, key):
        """Delete a coordinate."""
        del self._internal[key]

    def __setitem__(self, key, value):
        """Add or set a coordinate."""
        self._internal[key] = value

    def __delitem__(self, key):
        """Delete a coordinate."""
        del self._internal[key]

    def __getitem__(self, key):
        """Get a coordinate value."""
        return self._internal[key]

    def __len__(self):
        """Return length of coordinates array."""
        return len(self._internal)

    def __iter__(self):
        """Iterate over coordinates."""
        return iter(self._internal)

    def keys(self):
        r"""Return a generator over coordinates name."""
        return self._internal.keys()
