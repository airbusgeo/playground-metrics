from itertools import product
from os import cpu_count

import numpy as np
from numpy.random import randn
from pygeos import polygons, box, points, apply

from playground_metrics.utils.geometry import intersection_over_union, euclidean_distance


class _TestSimilarity:
    SIMILAR = []
    DISSIMILAR = []
    VALUE_GEOMETRIES = []
    VALUE = []

    SIMILAR_VALUE = 1.0
    DISSIMILAR_VALUE = 0

    def _similarity(self, geometry_1, geometry_2, force_thread=False):
        raise NotImplementedError

    def test_empty(self):
        assert np.array_equal(self._similarity(self.SIMILAR, []), np.zeros((len(self.SIMILAR), 0)))
        assert np.array_equal(self._similarity([], self.SIMILAR), np.zeros((0, len(self.SIMILAR))))
        assert np.array_equal(self._similarity([], []), np.zeros((0, 0)))

    def test_trivial(self):
        for geometry in self.SIMILAR:
            # Unary
            assert self._similarity(geometry, geometry) == 1
            # Array left
            assert np.array_equal(self._similarity([geometry, geometry], geometry), np.array([[1], [1]]))
            # Array right
            assert np.array_equal(self._similarity(geometry, [geometry, geometry]), np.array([[1, 1]]))
            # Array broadcast
            assert np.array_equal(self._similarity([geometry, geometry], [geometry, geometry]),
                                  np.array([[1, 1],
                                            [1, 1]]))

    def test_similar(self):
        for geometry_1, geometry_2 in product(self.SIMILAR, self.SIMILAR):
            assert self._similarity(geometry_1, geometry_2) == self.SIMILAR_VALUE

    def test_dissimilar(self):
        for geometry_1, geometry_2 in product(self.DISSIMILAR, self.DISSIMILAR):
            if geometry_1 is geometry_2:
                continue
            assert self._similarity(geometry_1, geometry_2) == self.DISSIMILAR_VALUE

    def test_value(self):
        for (geometry_1, geometry_2), value in zip(zip(*self.VALUE_GEOMETRIES), self.VALUE):
            assert np.absolute(self._similarity(geometry_1, geometry_2) - value) < 1e-9

    def test_threaded(self):
        for (geometry_1, geometry_2), value in zip(zip(*self.VALUE_GEOMETRIES), self.VALUE):
            geometry_1 = np.array([geometry_1], dtype=object)
            for _ in range(4 * cpu_count()):
                geometry_1 = np.concatenate([geometry_1,
                                             apply(geometry_1[-2:], lambda x: x + randn(2))])

            geometry_2 = np.array([geometry_2], dtype=object)
            for _ in range(4 * cpu_count()):
                geometry_2 = np.concatenate([geometry_1,
                                             apply(geometry_2[-2:], lambda x: x + randn(2))])

            assert np.allclose(self._similarity(geometry_1, geometry_2, force_thread=True),
                               self._similarity(geometry_1, geometry_2, force_thread=False))


class _TestPolygons(_TestSimilarity):
    SIMILAR = [box(0, 0, 1, 1),
               polygons([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]),
               polygons([[0.5, 0], [1, 0], [1, 1], [0, 1], [0, 0.5], [0, 0]])]
    DISSIMILAR = [box(10, 10, 11, 11),
                  polygons([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]),
                  polygons([[8, 8], [8, 13], [13, 13], [13, 8], [8, 8]],
                           holes=[[[9, 9], [9, 12], [12, 12], [12, 9], [9, 9]]]),
                  polygons([[4.5, 4], [5, 4], [5, 5], [4, 5], [4, 4.5], [4, 4]]),
                  points([0.5, 0.5])]
    VALUE_GEOMETRIES = [[box(0, 0, 2, 2), polygons([[8, 8], [8, 13], [13, 13], [13, 8], [8, 8]],
                                                   holes=[[[9, 9], [9, 12], [12, 12], [12, 9], [9, 9]]])],
                        [box(-1, -1, 1, 1), box(8, 8, 13, 13)]]
    VALUE = [0.14285714285714285, 0.64]


class _TestPoints(_TestSimilarity):
    SIMILAR = [points(0, 1), box(-1, 0, 1, 2)]
    DISSIMILAR = [points(10, 10),
                  box(0.5, 0.5, 3, 3)]
    VALUE_GEOMETRIES = [[points(10, 10), points(10, 10), points(10, 10)],
                        [box(9, 8, 11, 14), box(8, 9, 14, 11), box(8, 8, 14, 14)]]
    VALUE = [0, 0, 1 - np.sqrt(2)]

    DISSIMILAR_VALUE = -np.inf


class TestIntersectionOverUnionRTree(_TestPolygons):
    def _similarity(self, geometry_1, geometry_2, force_thread=False):
        return intersection_over_union(geometry_1, geometry_2, force_thread=force_thread)


class TestEuclideanDistanceRTree(_TestPoints):
    def _similarity(self, geometry_1, geometry_2, force_thread=False):
        return euclidean_distance(geometry_1, geometry_2, force_thread=force_thread)
