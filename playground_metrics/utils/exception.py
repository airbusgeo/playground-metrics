# Top-level exception
class MapMetricApiError(Exception):
    r"""Generic exception class for playground_metrics specific exceptions."""

    pass


# Match detections exceptions
class MatchEngineError(MapMetricApiError):
    r"""Generic exception class for MatchEngine specific exceptions."""

    pass


# Map metric exceptions
class MeanAveragePrecisionMetricError(MapMetricApiError):
    r"""Generic exception class for MeanAveragePrecisionMetric specific exceptions."""

    pass


# Geometry utils exceptions
class GeometryError(MapMetricApiError):
    r"""Generic exception class for geometry_utils specific exceptions."""

    pass


class ShapelySpecificTypeError(GeometryError):  # noqa: D205,D400
    r"""Raised whenever asked to handle a shapely.geometry type which have no equivalent type in geometry_utils
    (e.g. MultiPolygons)
    """

    pass


class InvalidGeometryError(GeometryError):
    r"""Raised when asked to create an invalid Geometry object."""

    pass


class InvalidGeometryOperationError(GeometryError):
    r"""Raised if an operation on Geometry returns an invalid Geometry."""

    pass
