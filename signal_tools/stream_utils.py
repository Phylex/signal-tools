from collections.abc import Callable
from typing import Generator, Iterable, Tuple, Any, Dict, List, Iterator, Sequence
import numpy as np


def arrays_to_data_line(arrays: list[np.ndarray]) -> str:
    """
    Take a sequence of numpy arrays and turn them into a line of our streaming format
    """
    tensor_strings = []

    for array in arrays:
        # Flatten the array in column-major order
        flat_array = array.flatten(order="F")

        # Convert the elements to strings
        elements = [str(element) for element in flat_array]

        # Join the elements into a single string
        tensor_string = " ".join(elements)
        tensor_strings.append(tensor_string)

    # Join the tensor strings with the pipe symbol '|'
    data_line = " | ".join(tensor_strings)
    return data_line


def validate_metadata(metadata: Dict[str, Any]) -> None:
    """
    Validate the metadata
    """
    # check that the required attributes are present
    if "shape" not in metadata or "type" not in metadata:
        raise ValueError("Each stream must have a 'shape' and 'type'")
    # check the type
    if metadata['type'] == "int":
        metadata['type'] = int
    elif metadata['type'] == "float":
        metadata['type'] = float
    else:
        raise ValueError("The indicated type must be either int or float")
    # check the shape
    if not isinstance(metadata["shape"], list | int):
        raise ValueError(
            "The shape attribute needs to be a list of "
            "intger or an integer")
    if isinstance(metadata["shape"], list):
        for elem in metadata["shape"]:
            if not isinstance(elem, int):
                raise ValueError(
                    "Dimensions of the tensor need to be integer")


def apply_operator_on_stream(metadata_operator: Callable, data_operator: Callable, stream: Tuple[dict, Iterable], *args, **kwargs) -> Tuple[dict, Iterable]:
    """
    Applies the two functions that are part of an operator to the stream
    """
    metadata = metadata_operator(stream[0])
    validate_metadata(metadata)
    datastream = map(lambda x: data_operator(x, *args, **kwargs), stream[1])
    return (metadata, datastream)


def collect_stream_into_string(streams: List[Tuple[Dict[str, Any], Iterable]]) -> Iterator[str]:
    """
    Generate the string written to the file from the stream
    This is the final transformation back into a text file
    """
    metadata = [s[0] for s in streams]
    data = [s[1] for s in streams]

    # Write metadata section
    metadata_str = ""
    metadata_str += "Metadata:\n"
    metadata_str += "streams:\n"
    for stream in metadata:
        if stream['type'] == int:
            type_str = 'int'
        else:
            type_str = 'float'
        print(type_str)
        metadata_str += f"  - name: {stream['name']}\n"
        metadata_str += f"    shape: {stream['shape']}\n"
        metadata_str += f"    type: {type_str}\n"

    # Write data section

    metadata_str += "Data:\n"
    yield metadata_str

    # define the function that produces one tuple from n iterators
    def gen_tuple_from_n_iterators(iterators) -> Generator:
        while True:
            try:
                t = [next(it) for it in iterators]
            except StopIteration:
                return
            yield t

    dlines = gen_tuple_from_n_iterators(data)
    for line_data in dlines:
        row_strs = []
        for i, entry in enumerate(line_data):
            tensor = np.array(
                entry, dtype=metadata[i]["type"]).flatten(order="F")
            tensor_str = ", ".join(str(x) for x in tensor)
            row_strs.append(tensor_str)
        yield " | ".join(row_strs) + "\n"
