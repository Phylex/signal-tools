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
Each possible filter produces a stream of data. A stream is allways characterized by consisting of a sequence of elements. The form of the element
depends on the specific command but will most likey be a vector of numbers. An element is delimited by a new-line ``\\n``. A ``#`` at the beginning of a line
marks the line as a comment and all content including escape sequences or special characters (except for the new-line of course) are ignored.
Elements of differing lengths in a single stream may be possible and depend on the filter used in the pipeline. If in doubt consult the documentation
of the tool and interrogate the output.

Tools may be able to provide a description of the data as comments at the beginning of the file, this functionality has mostly however not been implemented yet.
