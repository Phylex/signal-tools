import numpy as np
import re
import yaml
from typing import TextIO, Tuple, List, Dict, Any


class DataStream:
    def __init__(self, text_stream: TextIO):
        self.text_stream = text_stream
        self.metadata = DataStream._parse_metadata(self.text_stream)
        self.num_streams = len(self.metadata["streams"])
        self.data_pattern = self._generate_data_regex(self.metadata["streams"])

    @staticmethod
    def _readline_skip_comments(text_stream: TextIO) -> str:
        line = text_stream.readline()
        while re.match(r"^\s*#", line) is not None:
            if line == "":  # End of file
                break
            line = text_stream.readline()
        return line

    @staticmethod
    def _parse_metadata(text_stream: TextIO) -> Dict[str, Any]:
        metadata_str = ""
        line = DataStream._readline_skip_comments(text_stream)
        if line.startswith("Metadata:"):
            line = DataStream._readline_skip_comments(text_stream)
            while not line.startswith("Data:"):
                metadata_str += line
                line = DataStream._readline_skip_comments(text_stream)
        try:
            metadata = yaml.safe_load(metadata_str)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing metadata: {e}")

        if "streams" not in metadata or not isinstance(metadata["streams"], list):
            raise ValueError("Metadata must contain a list of streams")

        for stream in metadata["streams"]:
            if "shape" not in stream or "type" not in stream:
                raise ValueError("Each stream must have a 'shape' and 'type'")
        if not isinstance(stream["shape"], list | int):
            raise ValueError(
                "The shape attribute needs to be a list of "
                "intger or an integer")
        if isinstance(stream["shape"], list):
            for elem in stream["shape"]:
                if not isinstance(elem, int):
                    raise ValueError(
                        "Dimensions of the tensor need to be integer")

        return metadata

    @staticmethod
    def _convert_type(stream: Dict[str, Any], value: str) -> Any:
        if stream["type"] == "int":
            return int(value)
        elif stream["type"] == "float":
            return float(value)
        else:
            raise ValueError(f"Unsupported data type: {stream['type']}")

    @staticmethod
    def _generate_data_regex(streams: list[dict[str, Any]]) -> re.Pattern:
        multi_val_next_pattern = r"((\s*,?\s*)%s){%d}"
        int_pattern = r"[\+\-]?\d+"
        float_pattern = r"[\+\-]?\d+(\.\d+)?([eE][\+\-]?\d+)?"
        regex_parts = []
        for stream in streams:
            shape = tuple(stream["shape"])
            num_elements = np.prod(shape)
            dtype = stream["type"]

            if dtype == "int":
                value_pattern = int_pattern
            elif dtype == "float":
                value_pattern = float_pattern
            else:
                raise ValueError(f"Unsupported data type: {dtype}")

            if num_elements > 1:
                regex_parts.append(
                    value_pattern +
                    (multi_val_next_pattern % (value_pattern, num_elements - 1)))
            else:
                regex_parts.append(value_pattern)
        return re.compile(r"\s*\|\s*".join(regex_parts))

    def __iter__(self):
        return self

    def __next__(self) -> List[np.ndarray]:
        line = self._readline_skip_comments(self.text_stream)
        print(line)

        if not line:
            raise StopIteration

        match = self.data_pattern.match(line)
        if not match:
            raise ValueError(
                f"Data line doesn't match the expected format: {line}")

        tensor_strings = match.string.replace(
            ',', ' ').replace('\t', ' ').split('|')

        tensors = []
        for stream, tensor_str in zip(self.metadata["streams"], tensor_strings):
            num_elements = int(np.prod(stream["shape"]))
            data_values = tensor_str.split()
            tensor_values = [
                self._convert_type(stream, data_values[i])
                for i in range(num_elements)
            ]
            if len(stream["shape"]) > 1 and stream["shape"][1] > 1:
                tensor = np.array(tensor_values, dtype=stream["type"]).reshape(
                    stream["shape"], order="F")
                tensors.append(tensor)
            else:
                tensor = np.array(tensor_values, dtype=stream["type"])

        return tensors
