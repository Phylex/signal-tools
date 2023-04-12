import numpy as np


def arrays_to_data_line(arrays: list[np.ndarray]) -> str:
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
