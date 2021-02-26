# flake8: noqa: E501
import pytest

import numpy as np
from pygeos import box

from playground_metrics.utils.deprecation_utils import deprecated
from playground_metrics.utils.iou_utils import add_confidence_from_max_iou


def test_deprecation_no_docstring():
    @deprecated('Reason')
    def some_deprecated_function():
        print('Some text')

    with pytest.warns(DeprecationWarning):
        some_deprecated_function()

    print(some_deprecated_function.__doc__)

    assert some_deprecated_function.__doc__ == """
            .. warning::
                The function ``some_deprecated_function`` is deprecated and may not work anymore or disappear in the future.

                Reason for deprecation: *Reason*

            """


def test_deprecation_docstring():
    @deprecated('Reason')
    def some_deprecated_function():
        """Some docstring"""
        print('Some text')

    with pytest.warns(DeprecationWarning):
        some_deprecated_function()

    print(some_deprecated_function.__doc__)

    assert some_deprecated_function.__doc__ == """
            .. warning::
                The function ``some_deprecated_function`` is deprecated and may not work anymore or disappear in the future.

                Reason for deprecation: *Reason*

            Some docstring"""


def test_confidence_from_max_iou_bbox():
    detections = np.array([[box(0, 0, 9, 5)],
                           [box(23, 13, 29, 18)]])
    ground_truths = np.array([[box(5, 2, 15, 9)],
                              [box(18, 10, 26, 15)]])
    res = add_confidence_from_max_iou(detections, ground_truths)

    assert np.all(res == np.array([[box(0, 0, 9, 5), 0.11650485436893204],
                                   [box(23, 13, 29, 18), 0.09375]],
                                  dtype=object))
