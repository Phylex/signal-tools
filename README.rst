Signal-Tools
============
Signal-Tools is a collection of tools and operators that act on and transform numerical data streams.
It is intended as tool to explore signal transformations and filters on the command line before comitting to writing a dedicated program.
It also tries to provide the filters and operators that are implemented in the command line tools as library functions so that the transition
from command line experimentation to dedicated application is as smooth as possible.

Quickstart
----------
to get started quickly, the tools can be installed via pypi with the command ``pip install signal-tools``. After installation the command ``signal-tools``
should be available. ``signal-tools --help`` should bring up a help message for the tool.

.. I should put some examples here to get people started

Design Goals
------------
The tools are designed so that they can be interconnected using the unix Pipe. Using this technique, mltiple invocations of the command can be
combined with very little effort to form more complex processing pipelines. This should allow for experimentation and exploration with minimal
coding required.

Due to the nature of unix Pipes, only a single stream can be processed per command invocation. If more complex pipelines are to be built using
these tools, it is recommended to split the pipeline into multiple stages that are executed sequentially. The data in between every command is
transmitted as human readable text, which enables the splitting of a single pipeline into multiple stages at arbitrary points, aiding exploration
of intermediate results and hopefully deepening the understanding of the transformation system.

Intermediate Data Format
------------------------
Each possible filter streams it's output to ``stdout``. The output can consist of multiple streams that can each be a stream of multidimensional tensors.
``#`` marks the line as a comment and all content including escape sequences or special characters (except for the new-line of course) are ignored.

The data format for interchanging data consists of two sections: Metadata and Data.

Metadata
~~~~~~~~

The Metadata section is in YAML-like syntax and describes the properties of each stream, such as the shape and data type. The Metadata section starts with the keyword 
``Metadata:``, followed by a list of dictionaries under the key ``streams``. Each dictionary represents a stream and must have the keys ``shape`` and ``type``.

The "shape" key is a tuple specifying the dimensions of the tensor for that stream. The "type" key specifies the data type of the tensor elements, which can be ``int32``, ``float32``, or ``float64``.

Example:

.. code-block:: none

   Metadata:
   streams:
     - shape: (2, 2)
       type: float32
     - shape: (3,)
       type: int32
     - shape: (2, 3)
       type: float64

Data
~~~~

The Data section contains the actual data of the tensors. Each line in the Data section represents a set of tensors, one from each stream.
The tensors are separated by the pipe symbol "|". The elements within each tensor are listed in column-major order and separated by a comma and/or whitespace.

Example:

.. code-block:: none

   Data:
   1.0, 2.0, 3.0, 4.0 | 1, 2, 3 | 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
   5.0, 6.0, 7.0, 8.0 | 4, 5, 6 | 7.0, 8.0, 9.0, 10.0, 11.0, 12.0

or also in the case of mixed commas and spaces 

.. code-block:: none

   Data:
   1, 2 3, 4 5, 6 | 1.0 2.0
   7, 8 9, 10 11, 12 | 3.0, 4.0
