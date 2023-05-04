.. _format:

======================
Stream Data Format
======================

Tone he stream data format is designed to store and transport multiple columns of tensor objects in a text-based format. The format consists of two sections: metadata and data.

Metadata Section
----------------

The metadata section is defined using YAML syntax and starts with a "Metadata:" line. It contains a list of dictionaries, each describing a stream. Each dictionary must have the following keys:

- ``shape``: A list of integers specifying the shape of the tensor, or a single integer for a 1D tensor. If the shape is -1, it indicates a variable-length 1D tensor.
- ``type``: A string specifying the data type of the tensor. Supported types are "int" and "float".

a ``name`` field is recommended but not required. The name should give an understandable and short description/name to the data

Example metadata section::

    Metadata:
    streams:
        - shape: [2, 2]
          type: int
        - shape: 3
          type: float
        - shape: -1
          type: int

Data Section
------------

The data section starts with a "Data:" line and contains the tensor elements. Each line in this section represents one element from every stream. The elements are separated by the pipe symbol '|'. Tensors within a stream are serialized using column-major indexing. Numbers within a tensor can be separated by whitespace or commas.

Example data section::

    Data:
    1, 2, 3, 4 | 1.1, 2.2, 3.3 | 5, 6
    5, 6, 7, 8 | 4.4, 5.5, 6.6 | 7, 8, 9

In the case of variable-length streams, an empty element is allowed. The length of the 1D tensor can vary between elements::

    Data:
    1, 2, 3, 4 | 1.1, 2.2, 3.3 | 5, 6, 7
    5, 6, 7, 8 | 4.4, 5.5, 6.6 |

Comment Lines
-------------

Lines starting with a '#' character are considered comments and are ignored by the parser.
