import numpy as np
import pytest
from contextlib import contextmanager

from playground_metrics.utils.geometry_utils import Point, Polygon, BoundingBox, convert_to_bounding_box, \
    convert_to_polygon, convert_to_point, get_type_and_convert
from playground_metrics.utils.geometry_utils.geometry import geometry_factory
from playground_metrics.utils.exception import InvalidGeometryError, ShapelySpecificTypeError


@contextmanager
def not_raises(ExpectedException):
    try:
        yield
    except ExpectedException as err:  # noqa: F841
        raise AssertionError(
            "Did raise exception {0} when it should not!".format(
                repr(ExpectedException)
            )
        )
    except Exception as err:
        raise AssertionError(
            "An unexpected exception {0} raised.".format(repr(err))
        )


class TestAttributes:
    def test_polygon_full(self):
        poly = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert hasattr(poly, 'shell')
        assert hasattr(poly, 'holes')

    def test_polygon_hollow(self):
        poly = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]], [[0.5, 0.5], [0.5, 0.7], [0.7, 0.7], [0.7, 0.5]])
        assert hasattr(poly, 'shell')
        assert hasattr(poly, 'holes')

    def test_bbox(self):
        poly = BoundingBox(0, 0, 1, 1)
        assert hasattr(poly, 'xmin')
        assert hasattr(poly, 'xmax')
        assert hasattr(poly, 'ymin')
        assert hasattr(poly, 'ymax')

    def test_point(self):
        poly = Point(0, 1)
        assert hasattr(poly, 'x')
        assert hasattr(poly, 'y')


class TestOp:
    def make_single_geom_test(self, geom):
        assert geom == geom
        assert geom.centroid == Point(geom._internal.centroid.x, geom._internal.centroid.y)
        assert geom.intersection(geom.centroid) == geom.centroid
        for k in geom._coordinates.keys():
            with pytest.raises(AttributeError, match='object has no attribute'):
                geom._coordinates.__getattribute__(k)
            if isinstance(geom._coordinates[k], np.ndarray) and \
                    isinstance(geom._coordinates.__getattr__(k), np.ndarray):
                assert np.all(geom._coordinates[k] == geom._coordinates.__getattr__(k))
            else:
                assert geom._coordinates[k] == geom._coordinates.__getattr__(k)
            # Not true anymore with FastBoundingBox
            # with pytest.raises(AttributeError, match='object has no attribute'):
            #     geom.__getattribute__(k)
            if isinstance(geom._coordinates[k], np.ndarray) and \
                    isinstance(geom._coordinates.__getattr__(k), np.ndarray):
                assert np.all(geom._coordinates[k] == geom.__getattr__(k))
            else:
                assert geom._coordinates[k] == geom.__getattr__(k)
        assert geom.intersection(geom).equals(geom)
        assert geom.distance(geom) == 0.0
        if not isinstance(geom, Point):  # Intersection between points is always empty
            assert geom.intersection_over_union(geom) == 1.0
        assert geom.intersection(BoundingBox(5, 5, 10, 10)).is_empty
        assert geom.intersection_over_union(BoundingBox(5, 5, 10, 10)) == 0.0
        p = Point(np.inf, np.inf)
        assert geom.distance(p) == 1.7976931348623157e+308
        assert geom.intersection_over_union(p) == 0.0
        pb = BoundingBox(p.centroid.x - (1000 // 2),
                         p.centroid.y - (1000 // 2),
                         p.centroid.x + (1000 // 2),
                         p.centroid.y + (1000 // 2))
        assert geom.distance(pb) == 1.7976931348623157e+308
        assert geom.intersection_over_union(pb) == 0.0

    def make_list_geom_test(self, geom):
        res = geom.intersection(geom, geom)
        for r in res:
            assert r.equals(geom)
        assert geom.distance(geom, geom) == (0.0, 0.0)
        if not isinstance(geom, Point):  # Intersection between points is always empty
            assert geom.intersection_over_union(geom, geom) == (1.0, 1.0)
        res = geom.intersection(BoundingBox(5, 5, 10, 10), BoundingBox(5, 5, 10, 10))
        for r in res:
            assert r.is_empty
        assert geom.intersection_over_union(BoundingBox(5, 5, 10, 10), BoundingBox(5, 5, 10, 10)) == (0.0, 0.0)

    def make_commutativity_test(self, geom1, geom2):
        assert geom1.intersection(geom2).equals(geom2.intersection(geom1))
        assert geom1.distance(geom2) == geom2.distance(geom1)
        assert geom1.intersection_over_union(geom2) == geom2.intersection_over_union(geom1)

    def test_polygon_full(self):
        poly = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
        self.make_single_geom_test(poly)
        self.make_list_geom_test(poly)

    def test_polygon_hollow(self):
        poly = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]], [[0.5, 0.5], [0.5, 0.7], [0.7, 0.7], [0.7, 0.5]])
        self.make_single_geom_test(poly)
        self.make_list_geom_test(poly)

    def test_bbox(self):
        poly = BoundingBox(0, 0, 1, 1)
        self.make_single_geom_test(poly)
        self.make_list_geom_test(poly)

    def test_point(self):
        poly = Point(0, 1)
        self.make_single_geom_test(poly)
        self.make_list_geom_test(poly)

    def test_iou_unit_multitype(self):
        assert BoundingBox(0, 0, 1, 1).intersection_over_union(Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])) == 1.0
        assert BoundingBox(0, 0, 1, 1).intersection_over_union(Polygon([[0, 0], [0, 1],
                                                                        [1, 1], [1, 0]],
                                                                       [[0.5, 0.5], [0.5, 0.7],
                                                                        [0.7, 0.7], [0.7, 0.5]])) < 1.0

    def test_poly_bbox(self):
        self.make_commutativity_test(BoundingBox(0.5, 0.5, 1.5, 1.5), Polygon([[0, 0], [0, 1], [1, 1], [1, 0]]))

    def test_poly_point(self):
        self.make_commutativity_test(Point(0.5, 0.5), Polygon([[0, 0], [0, 1], [1, 1], [1, 0]]))

    def test_bbox_point(self):
        self.make_commutativity_test(Point(0.5, 0.5), BoundingBox(0, 0, 1, 1))


class TestConversion:
    def test_invalid_bbox(self):
        with pytest.raises(InvalidGeometryError, match='Invalid box coordinates'):
            res = convert_to_bounding_box([[0, 1, 1, 1], [0, 0, 1, 1]])
        with not_raises(InvalidGeometryError):
            res = convert_to_bounding_box([[0, 1, 1, 1], [0, 0, 1, 1]], trim_invalid_geometry=True)
        with pytest.raises(InvalidGeometryError, match='Invalid box coordinates'):
            res = convert_to_bounding_box([[0, 1, 1, 1], [0, 0, 1, 1]], autocorrect_invalid_geometry=True)
        with not_raises(InvalidGeometryError):
            res = convert_to_bounding_box([[0, 1, 1, 1], [0, 0, 1, 1]], trim_invalid_geometry=True,
                                          autocorrect_invalid_geometry=True)
        assert res == np.array([[BoundingBox(0, 0, 1, 1)]])

    def test_invalid_poly(self):
        with pytest.raises(InvalidGeometryError, match='Invalid shapely.BaseGeometry'):
            res = convert_to_polygon([[[[[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 1], [1, 1], [1, 0]]]],
                                      [[[[0, 0], [0, 0.5], [0.5, -1], [0.5, 1], [1, 1], [1, 0]]]]])
        with not_raises(InvalidGeometryError):
            res = convert_to_polygon([[[[[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 1], [1, 1], [1, 0]]]],
                                      [[[[0, 0], [0, 0.5], [0.5, -1], [0.5, 1], [1, 1], [1, 0]]]]],
                                     trim_invalid_geometry=True)
        assert res == np.array([[Polygon([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 1], [1, 1], [1, 0]])]])

    def test_bowtie_invalid_poly(self):
        with pytest.raises(InvalidGeometryError, match='Invalid shapely.BaseGeometry'):
            res = convert_to_polygon([[[[[0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5], [0, 0]]]]])
        with not_raises(InvalidGeometryError):
            res = convert_to_polygon([[[[[0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5], [0, 0]]]]],
                                     autocorrect_invalid_geometry=True)

        assert np.all([r.equals(gt)
                       for r_row, gt_row in zip(res, np.array([[Polygon([[0, 0], [0, 0.5], [0.25, 0.25]])],
                                                               [Polygon([[0.25, 0.25], [0.5, 0.5], [0.5, 0]])]]))
                       for r, gt in zip(r_row, gt_row)])

        with not_raises(InvalidGeometryError):
            res = convert_to_polygon([[[[[0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5], [0, 0]]]]],
                                     trim_invalid_geometry=True)

        assert res.size == 0

        with not_raises(InvalidGeometryError):
            res = convert_to_polygon([[[[[0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5], [0, 0]]]]],
                                     trim_invalid_geometry=True, autocorrect_invalid_geometry=True)
        assert np.all([r.equals(gt)
                       for r_row, gt_row in zip(res, np.array([[Polygon([[0, 0], [0, 0.5], [0.25, 0.25]])],
                                                               [Polygon([[0.25, 0.25], [0.5, 0.5], [0.5, 0]])]]))
                       for r, gt in zip(r_row, gt_row)])

    def test_candy_invalid_poly(self):
        with pytest.raises(InvalidGeometryError, match='Invalid shapely.BaseGeometry'):
            res = convert_to_polygon([[[[[0, 0], [0, 0.5], [0.5, 0], [1.0, 0.5], [1.0, 0.0], [0.5, 0.5], [0, 0]]]]])

        with not_raises(InvalidGeometryError):
            res = convert_to_polygon([[[[[0, 0], [0, 0.5], [0.5, 0], [1.0, 0.5], [1.0, 0.0], [0.5, 0.5], [0, 0]]]]],
                                     autocorrect_invalid_geometry=True)

        assert np.all([r.equals(gt)
                       for r_row, gt_row in zip(res, np.array([[Polygon([[0, 0], [0, 0.5], [0.25, 0.25]])],
                                                               [Polygon([[0.25, 0.25], [0.5, 0.5],
                                                                         [0.75, 0.25], [0.5, 0]])],
                                                               [Polygon([[0.75, 0.25], [1.0, 0.5], [1.0, 0]])]]))
                       for r, gt in zip(r_row, gt_row)])

        with not_raises(InvalidGeometryError):
            res = convert_to_polygon([[[[[0, 0], [0, 0.5], [0.5, 0], [1.0, 0.5], [1.0, 0.0], [0.5, 0.5], [0, 0]]]]],
                                     trim_invalid_geometry=True)

        assert res.size == 0

        with not_raises(InvalidGeometryError):
            res = convert_to_polygon([[[[[0, 0], [0, 0.5], [0.5, 0], [1.0, 0.5], [1.0, 0.0], [0.5, 0.5], [0, 0]]]]],
                                     trim_invalid_geometry=True, autocorrect_invalid_geometry=True)
        assert np.all([r.equals(gt)
                       for r_row, gt_row in zip(res, np.array([[Polygon([[0, 0], [0, 0.5], [0.25, 0.25]])],
                                                               [Polygon([[0.25, 0.25], [0.5, 0.5],
                                                                         [0.75, 0.25], [0.5, 0]])],
                                                               [Polygon([[0.75, 0.25], [1.0, 0.5], [1.0, 0]])]]))
                       for r, gt in zip(r_row, gt_row)])

    def test_invalid_hollow_poly(self):
        arr = np.zeros((1, 1), dtype=np.dtype('O'))
        arr[0, 0] = [[[0, 0], [0, 1.0], [1.0, 1.0], [1.0, 0.0]],
                     [[0.25, 0.5], [0.5, 0.25], [1.0, 0.5], [1.0, 0.75], [0.5, 0.75], [0.25, 0.5]]]

        with pytest.raises(InvalidGeometryError, match='Invalid shapely.BaseGeometry'):
            res = convert_to_polygon(arr)

        with pytest.raises(InvalidGeometryError, match='Invalid shapely.BaseGeometry'):
            res = convert_to_polygon(arr, autocorrect_invalid_geometry=True)

        with not_raises(InvalidGeometryError):
            res = convert_to_polygon(arr, trim_invalid_geometry=True)

        assert res.size == 0

        with pytest.raises(InvalidGeometryError):
            res = convert_to_polygon(arr, autocorrect_invalid_geometry=True, trim_invalid_geometry=True)

        assert res.size == 0

    def test_invalid_bbox_type(self):
        with pytest.raises(InvalidGeometryError, match='Invalid box coordinates'):
            _, res = get_type_and_convert([[0, 1, 1, 1, 0.9], [0, 0, 1, 1, 0.9]])
        with not_raises(InvalidGeometryError):
            _, res = get_type_and_convert([[0, 1, 1, 1, 0.9], [0, 0, 1, 1, 0.9]], trim_invalid_geometry=True)
        assert np.all(res == np.array([[BoundingBox(0, 0, 1, 1), 0.9]]))
