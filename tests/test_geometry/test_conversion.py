import pytest
import numpy as np

from playground_metrics.utils.geometry_utils import BoundingBox, Polygon, Point, \
    get_type_and_convert, convert_to_bounding_box, convert_to_polygon, convert_to_point

vector_input_list_score_class = [[[[[0, 0], [0, 1], [1, 1], [1, 0]]], 0.2, 0],
                                 [[[[0, 0], [0, 1], [1, 1], [1, 0]]], 0.2, 0]]
vector_input_list_class = [[[[[0, 0], [0, 1], [1, 1], [1, 0]]], 0],
                           [[[[0, 0], [0, 1], [1, 1], [1, 0]]], 0]]
vector_input_list = [[[[[0, 0], [0, 1], [1, 1], [1, 0]]]],
                     [[[[0, 0], [0, 1], [1, 1], [1, 0]]]]]

bbox_input_list_score_class = [[0, 0, 1, 1, 0.2, 0],
                               [0, 0, 1, 1, 0.2, 0]]
bbox_input_list_class = [[0, 0, 1, 1, 0],
                         [0, 0, 1, 1, 0]]
bbox_input_list = [[0, 0, 1, 1],
                   [0, 0, 1, 1]]

point_input_list_score_class = [[0, 1, 0.2, 0],
                                [0, 1, 0.2, 0]]
point_input_list_class = [[0, 1, 0],
                          [0, 1, 0]]
point_input_list = [[0, 1],
                    [0, 1]]

vector_input_ndarray_score_class = np.array(vector_input_list_score_class)
vector_input_ndarray_class = np.array(vector_input_list_class)
vector_input_ndarray = np.array(vector_input_list)

bbox_input_ndarray_score_class = np.array(bbox_input_list_score_class)
bbox_input_ndarray_class = np.array(bbox_input_list_class)
bbox_input_ndarray = np.array(bbox_input_list)

point_input_ndarray_score_class = np.array(point_input_list_score_class)
point_input_ndarray_class = np.array(point_input_list_class)
point_input_ndarray = np.array(point_input_list)

vector_input_list_score_class_str = [[[[[0, 0], [0, 1], [1, 1], [1, 0]]], 0.2, '0'],
                                     [[[[0, 0], [0, 1], [1, 1], [1, 0]]], 0.2, '0']]
vector_input_list_class_str = [[[[[0, 0], [0, 1], [1, 1], [1, 0]]], '0'],
                               [[[[0, 0], [0, 1], [1, 1], [1, 0]]], '0']]

bbox_input_list_score_class_str = [[0, 0, 1, 1, 0.2, '0'],
                                   [0, 0, 1, 1, 0.2, '0']]
bbox_input_list_class_str = [[0, 0, 1, 1, '0'],
                             [0, 0, 1, 1, '0']]

point_input_list_score_class_str = [[0, 1, 0.2, '0'],
                                    [0, 1, 0.2, '0']]
point_input_list_class_str = [[0, 1, '0'],
                              [0, 1, '0']]

vector_input_list_score_class_tuple = [[[[[0, 0], [0, 1], [1, 1], [1, 0]]], 0.2, (0, )],
                                       [[[[0, 0], [0, 1], [1, 1], [1, 0]]], 0.2, (0, )]]
vector_input_list_class_tuple = [[[[[0, 0], [0, 1], [1, 1], [1, 0]]], (0, )],
                                 [[[[0, 0], [0, 1], [1, 1], [1, 0]]], (0, )]]

bbox_input_list_score_class_tuple = [[0, 0, 1, 1, 0.2, (0, )],
                                     [0, 0, 1, 1, 0.2, (0, )]]
bbox_input_list_class_tuple = [[0, 0, 1, 1, (0, )],
                               [0, 0, 1, 1, (0, )]]

point_input_list_score_class_tuple = [[0, 1, 0.2, (0, )],
                                      [0, 1, 0.2, (0, )]]
point_input_list_class_tuple = [[0, 1, (0, )],
                                [0, 1, (0, )]]


class TestTypeDetection:
    class_conv = {'bbox': BoundingBox, 'polygon': Polygon, 'point': Point}
    class_start = {'bbox': 4, 'polygon': 1, 'point': 2}

    def test_number_dimension_mismatch(self):
        # Dimension number mismatch
        with pytest.raises(ValueError, match='Invalid array number of dimensions: '
                                             'Expected a 2D array, found 0D.'):
            type_, input_array = get_type_and_convert("abcdef")
        with pytest.raises(ValueError, match='Invalid array number of dimensions: '
                                             'Expected a 2D array, found 0D.'):
            type_, input_array = get_type_and_convert(1)
        with pytest.raises(ValueError, match='Invalid array number of dimensions: '
                                             'Expected a 2D array, found 1D.'):
            type_, input_array = get_type_and_convert(np.array([0, 0, 1, 1, 0.9]))
        with pytest.raises(ValueError, match='Invalid array number of dimensions: '
                                             'Expected a 2D array, found 6D.'):
            type_, input_array = get_type_and_convert(np.array([[[[[[0, 0, 1, 1, 0.9]]]]]]))

        # Dimension value mismatch
        with pytest.raises(ValueError, match='Invalid array fifth dimension: '
                                             'Expected 2, found 5'):
            type_, input_array = get_type_and_convert(np.array([[[[[0, 0, 1, 1, 0.9]]]]]))
        with pytest.raises(ValueError, match='Invalid array third dimension: '
                                             'Expected 1, found 3.'):
            type_, input_array = get_type_and_convert(np.array([[[0, 0, 1, 1, 0.9]]]))
        with pytest.raises(ValueError, match='Invalid array second dimension: '
                                             'Expected at least 2, found 1.'):
            type_, input_array = get_type_and_convert(np.array([[0]]))
        with pytest.raises(ValueError, match='Invalid array second dimension: '
                                             'Expected less than 6, found 8.'):
            type_, input_array = get_type_and_convert(np.array([[0, 0, 0, 0, 0.6, 0, 4, 0]]))

    def test_empty(self):
        type_, input_array = get_type_and_convert(np.array([]))
        assert type_ == 'undefined'
        assert input_array.size == 0

    def _make_type_test(self, data_in, true_type):
        type_, data_conv = get_type_and_convert(data_in)
        print(data_in, data_conv)
        print(type_, true_type)
        assert true_type == type_
        assert np.all(np.array([isinstance(data_conv[i, 0], self.class_conv[true_type])
                                for i in range(data_conv.shape[0])]))
        assert np.all(np.array(data_in, dtype=np.dtype('O'))[:, self.class_start[true_type]:] == data_conv[:, 1:])

    def test_vector_input_list_score_class(self):
        self._make_type_test(vector_input_list_score_class, 'polygon')

    def test_vector_input_list_class(self):
        self._make_type_test(vector_input_list_class, 'polygon')

    def test_bbox_input_list_score_class(self):
        self._make_type_test(bbox_input_list_score_class, 'bbox')

    def test_bbox_input_list_class(self):
        self._make_type_test(bbox_input_list_class, 'bbox')

    def test_point_input_list_score_class(self):
        self._make_type_test(point_input_list_score_class, 'point')

    def test_point_input_list_class(self):
        self._make_type_test(point_input_list_class, 'point')

    def test_vector_input_list_score_class_str(self):
        self._make_type_test(vector_input_list_score_class_str, 'polygon')

    def test_vector_input_list_class_str(self):
        self._make_type_test(vector_input_list_class_str, 'polygon')

    def test_bbox_input_list_score_class_str(self):
        self._make_type_test(bbox_input_list_score_class_str, 'bbox')

    def test_bbox_input_list_class_str(self):
        self._make_type_test(bbox_input_list_class_str, 'bbox')

    def test_point_input_list_score_class_str(self):
        self._make_type_test(point_input_list_score_class_str, 'point')

    def test_point_input_list_class_str(self):
        self._make_type_test(point_input_list_class_str, 'point')

    def test_vector_input_list_score_class_tuple(self):
        self._make_type_test(vector_input_list_score_class_tuple, 'polygon')

    def test_vector_input_list_class_tuple(self):
        self._make_type_test(vector_input_list_class_tuple, 'polygon')

    def test_bbox_input_list_score_class_tuple(self):
        self._make_type_test(bbox_input_list_score_class_tuple, 'bbox')

    def test_bbox_input_list_class_tuple(self):
        self._make_type_test(bbox_input_list_class_tuple, 'bbox')

    def test_point_input_list_score_class_tuple(self):
        self._make_type_test(point_input_list_score_class_tuple, 'point')

    def test_point_input_list_class_tuple(self):
        self._make_type_test(point_input_list_class, 'point')

    def test_vector_input_ndarray_score_class(self):
        self._make_type_test(vector_input_ndarray_score_class, 'polygon')

    def test_vector_input_ndarray_class(self):
        self._make_type_test(vector_input_ndarray_class, 'polygon')

    def test_bbox_input_ndarray_score_class(self):
        self._make_type_test(bbox_input_ndarray_score_class, 'bbox')

    def test_bbox_input_ndarray_class(self):
        self._make_type_test(bbox_input_ndarray_class, 'bbox')

    def test_point_input_ndarray_score_class(self):
        self._make_type_test(point_input_ndarray_score_class, 'point')

    def test_point_input_ndarray_class(self):
        self._make_type_test(point_input_ndarray_class, 'point')


class TestConversion:
    class_conv = {'bbox': BoundingBox, 'polygon': Polygon, 'point': Point}
    class_conv_fn = {'bbox': convert_to_bounding_box, 'polygon': convert_to_polygon, 'point': convert_to_point}
    class_start = {'bbox': 4, 'polygon': 1, 'point': 2}

    def _make_type_test(self, data_in, true_type):
        # Test sanity checks
        with pytest.raises(ValueError, match='Invalid array number of dimensions: '
                                             'Expected a 2D array, found 1D.'):
            type_, input_array = self.class_conv_fn[true_type](np.array([0, 0, 1, 1, 0.9]))
        with pytest.raises(ValueError, match='Invalid array number of dimensions: '
                                             'Expected a 2D array, found 4D.'):
            type_, input_array = self.class_conv_fn[true_type](np.array([[[[0, 0, 1, 1, 0.9]]]]))
        with pytest.raises(ValueError, match='Invalid array number of dimensions: '
                                             'Expected a 2D array, found 6D.'):
            type_, input_array = self.class_conv_fn[true_type](np.array([[[[[[0, 0, 1, 1, 0.9]]]]]]))

        # Test empty
        type_, input_array = self.class_conv_fn[true_type](np.array([]))
        assert type_ == 'undefined'
        assert input_array.size == 0

        # Test actual conversion
        data_conv = self.class_conv_fn[true_type](data_in)
        print(data_in, data_conv)
        assert np.all(np.array([isinstance(data_conv[i, 0], self.class_conv[true_type])
                                for i in range(data_conv.shape[0])]))
        data_in = np.array(data_in)
        if len(data_in.shape) > 2:
            object_array = np.ndarray((data_in.shape[0], 1), dtype=np.dtype('O'))
            for i in range(data_in.shape[0]):
                object_array[i] = data_in[i, :].tolist()
            data_in = object_array
        assert np.all(data_in[:, self.class_start[true_type]:] == data_conv[:, 1:])

    def test_vector_input_list_score_class(self):
        self._make_type_test(vector_input_list_score_class, 'polygon')

    def test_vector_input_list_class(self):
        self._make_type_test(vector_input_list_class, 'polygon')

    def test_vector_input_list(self):
        self._make_type_test(vector_input_list, 'polygon')

    def test_bbox_input_list_score_class(self):
        self._make_type_test(bbox_input_list_score_class, 'bbox')

    def test_bbox_input_list_class(self):
        self._make_type_test(bbox_input_list_class, 'bbox')

    def test_bbox_input_list(self):
        self._make_type_test(bbox_input_list, 'bbox')

    def test_point_input_list_score_class(self):
        self._make_type_test(point_input_list_score_class, 'point')

    def test_point_input_list_class(self):
        self._make_type_test(point_input_list_class, 'point')

    def test_point_input_list(self):
        self._make_type_test(point_input_list, 'point')

    def test_vector_input_ndarray_score_class(self):
        self._make_type_test(vector_input_ndarray_score_class, 'polygon')

    def test_vector_input_ndarray_class(self):
        self._make_type_test(vector_input_ndarray_class, 'polygon')

    def test_vector_input_ndarray(self):
        self._make_type_test(vector_input_ndarray, 'polygon')

    def test_bbox_input_ndarray_score_class(self):
        self._make_type_test(bbox_input_ndarray_score_class, 'bbox')

    def test_bbox_input_ndarray_class(self):
        self._make_type_test(bbox_input_ndarray_class, 'bbox')

    def test_bbox_input_ndarray(self):
        self._make_type_test(bbox_input_ndarray, 'bbox')

    def test_point_input_ndarray_score_class(self):
        self._make_type_test(point_input_ndarray_score_class, 'point')

    def test_point_input_ndarray_class(self):
        self._make_type_test(point_input_ndarray_class, 'point')

    def test_point_input_ndarray(self):
        self._make_type_test(point_input_ndarray, 'point')
