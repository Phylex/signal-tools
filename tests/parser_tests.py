import io
import pytest
from signal_tools.parsers import DataStream
import numpy as np
from typing import Any


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
        line = DataStream._readline_skip_comments(input_stream)
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
    ds = DataStream(input_stream)

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
    converted_value = DataStream._convert_type(stream, value_str)
    assert converted_value == expected_value


@pytest.mark.parametrize("stream, value_str", [
    ({"type": "unsupported_type"}, "42"),
    ({"type": "unsupported_type"}, "3.14"),
])
def test_convert_type_unsupported_type(stream: dict[str, Any], value_str: str):
    with pytest.raises(ValueError, match="Unsupported data type:"):
        DataStream._convert_type(stream, value_str)


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
    regex = DataStream._generate_data_regex(streams)
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
    regex = DataStream._generate_data_regex(streams)
    match = regex.fullmatch(invalid_data)
    assert match is None
