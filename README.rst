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
