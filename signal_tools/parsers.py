from itertools import tee
import numpy as np
import re
import yaml
from functools import partial
from .stream_utils import validate_metadata
from typing import TextIO, List, Dict, Any, Iterable, Tuple


class SignalStreams:
    def __init__(self, text_stream: TextIO):
        self.text_stream = text_stream
        self.metadata = SignalStreams._parse_metadata(self.text_stream)
        self.num_streams = len(self.metadata["streams"])
        self.data_pattern = self._generate_data_regex(self.metadata["streams"])

    @staticmethod
    def _readline_skip_comments(text_stream: TextIO) -> str:
        """
        Reads a line from the text stream, skipping lines that start with a
        comment character ('#').

        :param text_stream: The input text stream.
        :type text_stream: TextIO
        :return: The next non-comment line from the text stream. Returns an
                 empty string if the end of the file is reached.
        :rtype: str
        """
        line = text_stream.readline()
        while re.match(r"^\s*#", line) is not None:
            if line == "":  # End of file
                break
            line = text_stream.readline()
        return line

    @staticmethod
    def _parse_metadata(text_stream: TextIO) -> Dict[str, Any]:
        """
        Parses the metadata section from the given text stream.

        This function reads the metadata section from the input text stream,
        and returns a dictionary containing the parsed metadata. The function
        checks for valid metadata structure and content, including the presence
        of a list of streams, as well as the 'shape' and 'type' attributes for
        each stream.

        :param text_stream: The input text stream containing the metadata
                            section.
        :type text_stream: TextIO
        :return: A dictionary containing the parsed metadata.
        :rtype: Dict[str, Any]
        :raises ValueError: If the metadata format is invalid, or if any of the
                            required attributes are missing or
                            have incorrect data types.
        """
        metadata_str = ""
        line = SignalStreams._readline_skip_comments(text_stream)
        if line.startswith("Metadata:"):
            line = SignalStreams._readline_skip_comments(text_stream)
            while not line.startswith("Data:"):
                metadata_str += line
                line = SignalStreams._readline_skip_comments(text_stream)
        try:
            metadata = yaml.safe_load(metadata_str)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing metadata: {e}")

        if "streams" not in metadata or not isinstance(metadata["streams"], list):
            raise ValueError("Metadata must contain a list of streams")

        for stream in metadata["streams"]:
            validate_metadata(stream)
        return metadata

    @staticmethod
    def _convert_type(stream: Dict[str, Any], value: str) -> Any:
        """
        Converts the given value string to the data type specified in the
        stream dictionary.

        :param stream: A dictionary containing the stream metadata, including
                       the 'type' key.
        :type stream: Dict[str, Any]
        :param value: The value string to be converted.
        :type value: str
        :return: The converted value with the appropriate data type.
        :rtype: Any
        :raises ValueError: If the specified data type in the stream dictionary
                            is not supported.
        """
        if stream["type"] == "int":
            return int(value)
        elif stream["type"] == "float":
            return float(value)
        else:
            raise ValueError(f"Unsupported data type: {stream['type']}")

    @staticmethod
    def _generate_data_regex(streams: list[dict[str, Any]]) -> re.Pattern:
        """
        Generates a regular expression pattern for parsing data lines based on
        the stream metadata.

        This function generates a regular expression pattern to match and parse
        data lines in the data section. The pattern is based on the information
        provided in the metadata section. The function handles variable-length
        streams (indicated by a shape of [-1]) and fixed-length streams.

        :param streams: A list of dictionaries containing the metadata for each
                        stream. Each dictionary should have a 'shape' and
                        'type' key.
        :type streams: list[dict[str, Any]]
        :return: A compiled regular expression pattern for matching and parsing
                 data lines.
        :rtype: re.Pattern
        """
        multi_val_next_pattern = r"((\s*,?\s*)%s){%d}"
        int_pattern = r"[\+\-]?\d+"
        float_pattern = r"[\+\-]?\d+(\.\d+)?([eE][\+\-]?\d+)?"
        regex_parts = []
        for stream in streams:
            if isinstance(stream["shape"], int):
                stream["shape"] = [stream["shape"]]
            shape = tuple(stream["shape"])
            num_elements = np.prod(shape) if shape[0] != -1 else 0
            dtype = stream["type"]

            if dtype == "int":
                value_pattern = int_pattern
            elif dtype == "float":
                value_pattern = float_pattern
            else:
                raise ValueError(f"Unsupported data type: {dtype}")
            if shape[0] == -1:
                regex_parts.append(
                        rf"({value_pattern}(\s*,?\s*{value_pattern})*)?")
            elif num_elements > 0:
                regex_parts.append(
                    value_pattern +
                    (multi_val_next_pattern % (value_pattern, num_elements - 1)))
            else:
                regex_parts.append(value_pattern)
        return re.compile(r"\s*\|\s*".join(regex_parts))

    def split_into_individual_streams(self) -> list[Tuple[dict, Iterable]]:
        """
        Split the single data stream into many different data streams
        """
        print(list(enumerate(self.metadata["streams"])))
        whole_data_streams = tee(iter(self), self.num_streams)
        split_data_streams = []

        def f(x, index):
            return x[index]
        indexed_f = [partial(f, index=i) for i in range(self.num_streams)]
        split_data_streams = [map(idx_f, stream)
                              for idx_f, stream in
                              zip(indexed_f, whole_data_streams)]
        return list(zip(self.metadata["streams"], split_data_streams))

    def __iter__(self):
        """
        This class implements the iter method all by itself so
        just return the class
        """
        return self

    def __next__(self) -> List[np.ndarray]:
        """
        Returns the next parsed tensors from the data stream.

        :return: A list of numpy arrays representing the parsed tensors.
        :rtype: List[np.ndarray]
        :raises StopIteration: If the end of the data stream is reached.
        :raises ValueError: If the data line doesn't match the expected format.
        """
        line = self._readline_skip_comments(self.text_stream)

        if not line:
            raise StopIteration

        match = self.data_pattern.match(line)
        if not match:
            raise ValueError(
                f"Data line doesn't match the expected format: {line}")

        tensor_strings = match.string.replace(
            ',', ' ').replace('\t', ' ').split('|')

        tensors = []
        for stream_metadata, tensor_str in zip(self.metadata["streams"], tensor_strings):
            data_values = tensor_str.split()
            tensor_values = [
                self._convert_type(stream_metadata, dv)
                for dv in data_values
            ]
            if len(stream_metadata["shape"]) > 1 and stream_metadata["shape"][1] > 1:
                tensor = np.array(tensor_values, dtype=stream_metadata["type"]).reshape(
                    stream_metadata["shape"], order="F")
                tensors.append(tensor)
            else:
                tensor = np.array(tensor_values, dtype=stream_metadata["type"])
                tensors.append(tensor)
        print(tensors)
        return tensors
