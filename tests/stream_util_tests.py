import numpy as np
import pytest

# Import the arrays_to_data_line function
from signal_tools.stream_utils import arrays_to_data_line


@pytest.mark.parametrize(
    "arrays,expected_data_line",
    [
        (
            [np.array([[1, 2, 3], [4, 5, 6]]), np.array([1.0, 2.0])],
            "1 4 2 5 3 6 | 1.0 2.0",
        ),
        (
            [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])],
            "1 3 2 4 | 5 7 6 8",
        ),
        (
            [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])],
            "1 2 3 | 4 5 6 | 7 8 9",
        ),
        (
            [np.array([[1, 2, 3], [4, 5, 6]]), np.array([1.0e-8, 2.0e-8])],
            "1 4 2 5 3 6 | 1e-08 2e-08",
        ),
    ],
)
def test_arrays_to_data_line(arrays, expected_data_line):
    assert arrays_to_data_line(arrays) == expected_data_line
