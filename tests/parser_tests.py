import io
import pytest
from signal_tools.parsers import SignalStreams
import numpy as np
from typing import Any, Tuple, List


@pytest.mark.parametrize("input_data,expected_lines", [
    (
        "Line1\n"
        "# Comment1\n"
        "Line2\n",
        ["Line1\n", "Line2\n", ""]
    ),
    (
        "# Comment1\n"
        "\n"
        "# Comment2\n"
        "Line1\n",
        ["\n", "Line1\n", ""]
    ),
    (
        "Line1\n"
        "\n"
        "Line2\n",
        ["Line1\n", "\n", "Line2\n", ""]
    ),
    (
        "# Comment1\n"
        "# Comment2\n"
        "# Comment3\n",
        [""]
    ),
])
def test_readline_skip_comments(input_data: str, expected_lines: list[str]):
    input_stream = io.StringIO(input_data)

    for expected_line in expected_lines:
        line = SignalStreams._readline_skip_comments(input_stream)
        assert line == expected_line


@pytest.mark.parametrize("input_data,expected_tensors", [
    (
        "Metadata:\n"
        "streams:\n"
        "  - shape: [2]\n"
        "    type: int\n"
        "Data:\n"
        "1, 2\n"
        "3, 4\n",
        [
            [np.array([1, 2], dtype=np.int32)],
            [np.array([3, 4], dtype=np.int32)],
        ]
    ),
    (
        "Metadata:\n"
        "streams:\n"
        "  - shape: [1]\n"
        "    type: float\n"
        "  - shape: [2]\n"
        "    type: int\n"
        "Data:\n"
        "1.0 | 1, 2\n"
        "2.0 | 3, 4\n",
        [
            [
                np.array([[1.0]], dtype=np.float32),
                np.array([1, 2], dtype=np.int32),
            ],
            [
                np.array([[2.0]], dtype=np.float32),
                np.array([3, 4], dtype=np.int32),
            ],
        ]
    ),
    (
        "# This is a comment\n"
        "Metadata:\n"
        "# Another comment\n"
        "streams:\n"
        "  - shape: [2]\n"
        "    type: int\n"
        "Data:\n"
        "# Data comment\n"
        "1, 2\n"
        "3, 4\n",
        [
            [np.array([1, 2], dtype=np.int32)],
            [np.array([3, 4], dtype=np.int32)],
        ]
    ),
])
def test_data_stream(input_data: str,
                     expected_tensors: list[list[np.ndarray]]):
    input_stream = io.StringIO(input_data)
    ds = SignalStreams(input_stream)

    for expected, parsed in zip(expected_tensors, ds):
        for e, p in zip(expected, parsed):
            assert np.array_equal(e, p)


@pytest.mark.parametrize("stream, value_str, expected_value", [
    ({"type": "int"}, "42", 42),
    ({"type": "int"}, "-5", -5),
    ({"type": "float"}, "3.14", 3.14),
    ({"type": "float"}, "-0.5", -0.5),
])
def test_convert_type(stream: dict[str, Any], value_str: str, expected_value: Any):
    converted_value = SignalStreams._convert_type(stream, value_str)
    assert converted_value == expected_value


@pytest.mark.parametrize("stream, value_str", [
    ({"type": "unsupported_type"}, "42"),
    ({"type": "unsupported_type"}, "3.14"),
])
def test_convert_type_unsupported_type(stream: dict[str, Any], value_str: str):
    with pytest.raises(ValueError, match="Unsupported data type:"):
        SignalStreams._convert_type(stream, value_str)


@pytest.mark.parametrize("streams, example_data", [
    (
        [{"type": "int", "shape": [2]}],
        ["42 -5", "100 200", "-1 0"]
    ),
    (
        [{"type": "float", "shape": [2]}],
        ["3.14 -0.5", "1.0 2.0", "1.23e-4 5.67E+8"]
    ),
    (
        [{"type": "int", "shape": (2,)}, {"type": "float", "shape": (2,)}],
        ["42 -5 | 3.14 -0.5", "1 2 | 3.0 4.0", "0 0 | 0.0 0.0"]
    ),
    (
        [{"type": "float", "shape": (2, 2)}],
        ["1.0 2.0 3.0 4.0", "1.23 -4.56 7.89 -0.12"]
    ),
    (
        [{"type": "int", "shape": (1,)}, {"type": "float", "shape": (2, 1)}],
        ["1 | 2.0 3.0", "0 | 0.0 0.0", "-1 | 1.23 4.56"]
    ),
    (
        [{"type": "int", "shape": (2,)}],
        ["42 -5", "100, 200", "-1, 0"]
    ),
    (
        [{"type": "float", "shape": (2,)}],
        ["3.14 -0.5", "1.0, 2.0", "1.23e-4, 5.67E+8"]
    ),
    (
        [{"type": "int", "shape": (2,)}, {"type": "float", "shape": (2,)}],
        ["42 -5 | 3.14 -0.5", "1, 2 | 3.0, 4.0", "0, 0 | 0.0, 0.0"]
    ),
    (
        [{"type": "float", "shape": (2, 2)}],
        ["1.0 2.0 3.0 4.0", "1.23, -4.56, 7.89, -0.12"]
    ),
    (
        [{"type": "int", "shape": (1,)}, {"type": "float", "shape": (2, 1)}],
        ["1 | 2.0 3.0", "0 | 0.0, 0.0", "-1 | 1.23, 4.56"]
    ),
])
def test_generate_data_regex(streams: list[dict[str, Any]], example_data: list[str]):
    regex = SignalStreams._generate_data_regex(streams)
    for data in example_data:
        print(regex)
        match = regex.fullmatch(data)
        assert match is not None

@pytest.mark.parametrize("streams, invalid_data", [
    (
        [{"type": "int", "shape": (2,)}],
        "42 -5 3"
    ),
    (
        [{"type": "float", "shape": (2,)}],
        "3.14 -0.5 1.0"
    ),
    (
        [{"type": "int", "shape": (2,)}, {"type": "float", "shape": (2,)}],
        "42 -5 3.14"
    ),
    (
        [{"type": "float", "shape": (2, 2)}],
        "1.0 2.0 3.0"
    ),
    (
        [{"type": "int", "shape": (1,)}, {"type": "float", "shape": (2, 1)}],
        "1 2.0"
    ),
])
def test_generate_data_regex_invalid(streams: list[dict[str, Any]], invalid_data: str):
    regex = SignalStreams._generate_data_regex(streams)
    match = regex.fullmatch(invalid_data)
    assert match is None


@pytest.mark.parametrize("input_str,expected", [
    (
        "Metadata:\n"
        "  streams:\n"
        "    - shape: [-1]\n"
        "      type: int\n"
        "    - shape: [4]\n"
        "      type: int\n"
        "Data:\n"
        "1 2 3 | 4 5 6 7\n"
        "1 | 4 5 6 7\n"
        "1 2 | 4 5 6 7\n"
        "1 2 3 4 | 4 5 6 7\n",
        [
            (np.array([1, 2, 3]), np.array([4, 5, 6, 7])),
            (np.array([1, ]), np.array([4, 5, 6, 7])),
            (np.array([1, 2]), np.array([4, 5, 6, 7])),
            (np.array([1, 2, 3, 4]), np.array([4, 5, 6, 7]))
        ]
    ),
    (
        "Metadata:\n"
        "  streams:\n"
        "    - shape: [-1]\n"
        "      type: float\n"
        "    - shape: [3]\n"
        "      type: float\n"
        "Data:\n"
        "1.1, 2.2, 3.3 | 4.4 5.5, 6.6\n"
        "1.1, 2.2 | 4.4 5.5, 6.6\n"
        "1.1 | 4.4 5.5, 6.6\n"
        "1.1, 2.2, 3.3, 4.4 | 4.4 5.5, 6.6\n",
        [
            (np.array([1.1, 2.2, 3.3]), np.array([4.4, 5.5, 6.6])),
            (np.array([1.1, 2.2, ]), np.array([4.4, 5.5, 6.6])),
            (np.array([1.1, ]), np.array([4.4, 5.5, 6.6])),
            (np.array([1.1, 2.2, 3.3, 4.4]), np.array([4.4, 5.5, 6.6]))
        ]
    ),
    (
        "Metadata:\n"
        "  streams:\n"
        "    - shape: [-1]\n"
        "      type: float\n"
        "Data:\n"
        "1e-3, 2.0e3\n"
        "1e-3, 2.0e3\n"
        "1e-3, 2.0e3\n",
        [
            (np.array([1e-3, 2.0e3]), np.array([3.0e+3])),
            (np.array([1e-3, 2.0e3]), np.array([3.0e+3])),
            (np.array([1e-3, 2.0e3]), np.array([3.0e+3])),
        ]
    ),
    (
        "Metadata:\n"
        "  streams:\n"
        "    - shape: [-1]\n"
        "      type: int\n"
        "Data:\n"
        "1 2 3 \n"
        "\n"
        "1 2 3 4 \n"
        "1 2\n",
        [
            (np.array([1, 2, 3])),
            (np.array([])),
            (np.array([1, 2, 3, 4])),
            (np.array([1, 2]))
        ]
    ),
    (
        "Metadata:\n"
        "  streams:\n"
        "    - shape: [-1]\n"
        "      type: float\n"
        "    - shape: [3]\n"
        "      type: float\n"
        "Data:\n"
        "1 | 4.4 5.5, 6.6\n"
        "| 4.4 5.5, 6.6\n"
        "0 | 4.4 5.5, 6.6\n",
        [
            (np.array([1.]), np.array([4.4, 5.5, 6.6])),
            (np.array([]), np.array([4.4, 5.5, 6.6])),
            (np.array([0.]), np.array([4.4, 5.5, 6.6]))
        ]
    )
])
def test_variable_length_stream(input_str: str, expected: List[Tuple[np.ndarray]]):
    text_stream = io.StringIO(input_str)
    data_stream = SignalStreams(text_stream)

    output = []
    for tensors in data_stream:
        output.append(tuple(tensors))

    for i, (expected_tensors, output_tensors) in enumerate(zip(expected, output)):
        for j, (expected_tensor, output_tensor) in enumerate(zip(expected_tensors, output_tensors)):
            assert np.array_equal(expected_tensor, output_tensor), f"Error in tensors {i}, {j}: expected {expected_tensor}, got {output_tensor}"

