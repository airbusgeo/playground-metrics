from collections.abc import Sized

import numpy as np
import shapely.ops
import shapely.geometry

from ..exception import InvalidGeometryError
from .geometry import Point, Polygon, BoundingBox, Geometry


# Helpers functions
def get_type_and_convert(input_array, trim_invalid_geometry=False, autocorrect_invalid_geometry=False):
    r"""Automatically find the geometry type from the input array shape and convert it to a geometry array.

    Args:
        input_array (ndarray, list):

            * A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[x_min, y_min, x_max, y_max, confidence, label]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[[[outer_ring], [inner_rings]], confidence, label]``
                * Points for a given class where each row is a detection stored as:
                  ``[x, y, confidence, label]``

            * A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[x_min, y_min, x_max, y_max, label]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[[[outer_ring], [inner_rings]], label]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[x, y, label]``

        trim_invalid_geometry (bool): Optional, default to ``False``. If set to ``True`` conversion will ignore invalid
            geometries and leave them out of ``output_array``. This means that the function will return an array where
            ``output_array.shape[0] <= input_array.shape[0]``.  If set to ``False``, an invalid geometry will raise an
            :exc:`~playground_metrics.utils.geometry_utils.InvalidGeometryError`.

        autocorrect_invalid_geometry (Bool): Optional, default to ``False``. Whether to attempt correcting a faulty
            geometry to form a valid one. If set to ``True`` and the autocorrect attempt is unsuccessful, it falls back
            to the behaviour defined in ``trim_invalid_geometry``.

    Note:
        * Polygon auto-correction only corrects self-crossing exterior rings, in which case it creates one Polygon
          out of every simple ring which might be extracted from the original Polygon exterior.
        * Polygon auto-correction will systematically fail on Polygons with at least one inner ring.

    Returns:
        (str, ndarray): A tuple of output containing:

        * The BaseGeometry type as a string which may be either ``"point"``, ``"polygon"`` or ``"bbox"``
        * A geometry ndarray where each row contains a geometry followed by optionally confidence and a label
          e.g.: ``[BaseGeometry, (confidence), label]``

    Raises:
        ValueError: If ``input_array`` have invalid dimensions.

    """
    input_array = np.array(input_array, dtype=np.dtype('O'))
    # If nothing pass here
    if input_array.size == 0:
        return 'undefined', input_array

    if len(input_array.shape) < 2:
        # If we have less than 2 dimensions, pull the plug its useless
        raise ValueError('Invalid array number of dimensions: '
                         'Expected a 2D array, found {}D.'.format(len(input_array.shape)))

    elif len(input_array.shape) == 2:
        # It might just be anything at this point. We need to check the second dimension value to decide
        if input_array.shape[1] < 2:
            # If it's less than 2, pull the plug its useless
            raise ValueError('Invalid array second dimension: '
                             'Expected at least 2, found {}.'.format(input_array.shape[1]))
        elif input_array.shape[1] == 2:
            # It's a Polygon ndarray
            type_ = 'polygon'
            # Convert rings to Polygon
            input_array = convert_to_polygon(input_array, trim_invalid_geometry=trim_invalid_geometry,
                                             autocorrect_invalid_geometry=autocorrect_invalid_geometry)
        elif input_array.shape[1] == 3:
            # It might be either a Polygon ndarray or a Point ndarray
            # We check the first element to decide. One might argue that it is not robust to a mixed-type input array,
            # that is true but the conversion functions implicitly assumes that the input array is of fixed-type for
            # performances issues.
            if isinstance(input_array[0, 0], Sized):
                # Reasonable guess is the first element is a list or a ndarray -> It's a Polygon ndarray
                type_ = 'polygon'
                # Convert rings to Polygon
                input_array = convert_to_polygon(input_array, trim_invalid_geometry=trim_invalid_geometry,
                                                 autocorrect_invalid_geometry=autocorrect_invalid_geometry)
            else:
                # Resonable guess is the first element is a number -> It's a Point ndarray
                type_ = 'point'
                input_array = convert_to_point(input_array, trim_invalid_geometry=trim_invalid_geometry)
        elif input_array.shape[1] == 4:
            # It's a Point ndarray
            type_ = 'point'
            input_array = convert_to_point(input_array, trim_invalid_geometry=trim_invalid_geometry)
        elif 7 > input_array.shape[1] > 4:
            # It's a BoundingBox ndarray
            type_ = 'bbox'
            input_array = convert_to_bounding_box(input_array, trim_invalid_geometry=trim_invalid_geometry)
        else:
            raise ValueError('Invalid array second dimension: '
                             'Expected less than 6, found {}.'.format(input_array.shape[1]))

    elif len(input_array.shape) == 3:
        # It's the weirdest Polygon and tuple class corner case ! This really is not a drill !
        # It's a Polygon ndarray
        type_ = 'polygon'
        # Convert rings to Polygon
        input_array = convert_to_polygon(input_array, trim_invalid_geometry=trim_invalid_geometry,
                                         autocorrect_invalid_geometry=autocorrect_invalid_geometry)

    elif len(input_array.shape) == 5:
        # It's the mildly weird Polygon corner case ! This is not a drill !
        # It's a Polygon ndarray
        type_ = 'polygon'
        # Convert rings to Polygon
        input_array = convert_to_polygon(input_array, trim_invalid_geometry=trim_invalid_geometry,
                                         autocorrect_invalid_geometry=autocorrect_invalid_geometry)

    else:
        # If we have neither 2 dimensions nor 5 (in a weird polygon corner case), pull the plug its useless
        raise ValueError('Invalid array number of dimensions: '
                         'Expected a 2D array, found {}D.'.format(len(input_array.shape)))

    return type_, input_array


def convert_to_polygon(input_array, trim_invalid_geometry=False, autocorrect_invalid_geometry=False):
    r"""Convert an input array to a Polygon array.

    Args:
        input_array (ndarray, list): A ndarray of Polygons optionally followed by a confidence value and/or a label
            where each row is: ``[[[outer_ring], [inner_rings]], (confidence), (label)]``
        trim_invalid_geometry (bool): Optional, default to ``False``. If set to ``True`` conversion will ignore invalid
            geometries and leave them out of ``output_array``. This means that the function will return an array where
            ``output_array.shape[0] <= input_array.shape[0]``.  If set to ``False``, an invalid geometry will raise an
            :exc:`~playground_metrics.utils.geometry_utils.InvalidGeometryError`.
        autocorrect_invalid_geometry (Bool): Optional, default to ``False``. Whether to attempt correcting a faulty
            geometry to form a valid one. If set to ``True`` and the autocorrect attempt is unsuccessful, it falls back
            to the behaviour defined in ``trim_invalid_geometry``.

    Note:
        * Polygon auto-correction only corrects self-crossing exterior rings, in which case it creates one Polygon
          out of every simple ring which might be extracted from the original Polygon exterior.
        * Polygon auto-correction will systematically fail on Polygons with at least one inner ring.

    Returns:
        ndarray: A Polygon ndarray where each row contains a geometry followed by optionally confidence and a label
        e.g.: ``[Polygon, (confidence), (label)]``

    Raises:
        ValueError: If ``input_array`` have invalid dimensions.

    """
    input_array = np.array(input_array, dtype=np.dtype('O'))
    if input_array.size == 0:
        return 'undefined', input_array

    if (len(input_array.shape) == 1 or len(input_array.shape) > 2) and \
            (not len(input_array.shape) == 5 and not len(input_array.shape) == 3):
        raise ValueError('Invalid array number of dimensions: '
                         'Expected a 2D array, found {}D.'.format(len(input_array.shape)))

    if len(input_array.shape) == 5 and not input_array.shape[4] == 2:
        raise ValueError('Invalid array fifth dimension: '
                         'Expected 2, found {}.'.format(len(input_array.shape)))

    elif len(input_array.shape) == 3 and not input_array.shape[2] == 1:
        raise ValueError('Invalid array third dimension: '
                         'Expected 1, found {}.'.format(len(input_array.shape)))

    object_array = np.ndarray((input_array.shape[0], input_array.shape[1]), dtype=np.dtype('O'))
    to_trim = []
    to_add = []
    for i in range(input_array.shape[0]):
        try:
            if len(input_array[i, 0]) > 1:
                list_ = [Polygon(input_array[i, 0][0], *input_array[i, 0][1:])]
                list_.extend(input_array[i, 1:])
            else:
                list_ = [Polygon(input_array[i, 0][0])]
                list_.extend(input_array[i, 1:])
            object_array[i] = np.array(list_, dtype=np.dtype('O'))
        except InvalidGeometryError as e:
            if autocorrect_invalid_geometry:
                if len(shapely.geometry.Polygon(input_array[i, 0][0], input_array[i, 0][1:]).interiors) > 0:
                    raise e
                ext = shapely.geometry.Polygon(input_array[i, 0][0], input_array[i, 0][1:]).exterior
                shapely_polygon_list = \
                    shapely.ops.polygonize(shapely.geometry.LineString(ext.coords[:] + ext.coords[0:1]).intersection(
                        shapely.geometry.LineString(ext.coords[:] + ext.coords[0:1])))
                polygon_list = []
                try:
                    for polygon in shapely_polygon_list:
                        list_ = [Polygon(polygon.exterior.coords)]
                        list_.extend(input_array[i, 1:])
                        polygon_list.append(list_)
                except InvalidGeometryError:
                    if trim_invalid_geometry:
                        to_trim.append(i)
                    else:
                        raise e
                else:
                    to_trim.append(i)
                    to_add.extend(polygon_list)
            else:
                if trim_invalid_geometry:
                    to_trim.append(i)
                else:
                    raise e
    return np.concatenate((np.delete(object_array, to_trim, axis=0),
                           np.array(to_add).reshape(-1, object_array.shape[1])),
                          axis=0)


def convert_to_bounding_box(input_array, trim_invalid_geometry=False, autocorrect_invalid_geometry=False):
    r"""Convert an input array to a BoundingBox array.

    Args:
        input_array (ndarray, list): A ndarray of BoundingBox optionally followed by a confidence value and/or a label
            where each row is: ``[xmin, ymin, xmax, ymax, (confidence), (label)]``
        trim_invalid_geometry (bool): Optional, default to ``False``. If set to ``True`` conversion will ignore invalid
            geometries and leave them out of ``output_array``. This means that the function will return an array where
            ``output_array.shape[0] <= input_array.shape[0]``.  If set to ``False``, an invalid geometry will raise an
            :exc:`~playground_metrics.utils.geometry_utils.InvalidGeometryError`.
        autocorrect_invalid_geometry (Bool): Optional, default to ``False``. Doesn't do anything, introduced to unify
            convert functions interfaces.

    Returns:
        ndarray: A BoundingBox ndarray where each row contains a geometry followed by optionally confidence and a label
        e.g.: ``[BoundingBox, (confidence), (label)]``

    Raises:
        ValueError: If ``input_array`` have invalid dimensions.

    """
    input_array = np.array(input_array, dtype=np.dtype('O'))
    if input_array.size == 0:
        return 'undefined', input_array
    if len(input_array.shape) == 1 or len(input_array.shape) > 2:
        raise ValueError('Invalid array number of dimensions: '
                         'Expected a 2D array, found {}D.'.format(len(input_array.shape)))
    object_array = np.ndarray((input_array.shape[0], input_array.shape[1] - 3), dtype=np.dtype('O'))
    to_trim = []
    for i in range(input_array.shape[0]):
        try:
            list_ = [BoundingBox(*input_array[i, :4])]
            list_.extend(input_array[i, 4:])
            object_array[i] = np.array(list_, dtype=np.dtype('O'))
        except InvalidGeometryError as e:
            if trim_invalid_geometry:
                to_trim.append(i)
            else:
                raise e

    return np.delete(object_array, to_trim, axis=0)


def convert_to_point(input_array, trim_invalid_geometry=False, autocorrect_invalid_geometry=False):
    r"""Convert an input array to a Point array.

    Args:
        input_array (ndarray, list): A ndarray of Point optionally followed by a confidence value and/or a label
            where each row is: ``[x, y, (confidence), (label)]``
        trim_invalid_geometry (bool): Optional, default to ``False``. If set to ``True`` conversion will ignore invalid
            geometries and leave them out of ``output_array``. This means that the function will return an array where
            ``output_array.shape[0] <= input_array.shape[0]``.  If set to ``False``, an invalid geometry will raise an
            :exc:`~playground_metrics.utils.geometry_utils.InvalidGeometryError`.
        autocorrect_invalid_geometry (Bool): Optional, default to ``False``. Doesn't do anything, introduced to unify
            convert functions interfaces.

    Returns:
        ndarray: A Point ndarray where each row contains a geometry followed by optionally confidence and a label
        e.g.: ``[Point, (confidence), (label)]``

    Raises:
        ValueError: If ``input_array`` have invalid dimensions.

    """
    input_array = np.array(input_array, dtype=np.dtype('O'))
    if input_array.size == 0:
        return 'undefined', input_array
    if len(input_array.shape) == 1 or len(input_array.shape) > 2:
        raise ValueError('Invalid array number of dimensions: '
                         'Expected a 2D array, found {}D.'.format(len(input_array.shape)))
    object_array = np.ndarray((input_array.shape[0], input_array.shape[1] - 1), dtype=np.dtype('O'))
    to_trim = []
    for i in range(input_array.shape[0]):
        try:
            list_ = [Point(*input_array[i, :2])]
            list_.extend(input_array[i, 2:])
            object_array[i] = np.array(list_, dtype=np.dtype('O'))
        except InvalidGeometryError as e:
            if trim_invalid_geometry:
                to_trim.append(i)
            else:
                raise e

    return np.delete(object_array, to_trim, axis=0)
