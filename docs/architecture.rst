Architecture
============

The Core Idea is that an arbitrary amount of "datastreams" should be transported from one process to the other, with each process performing a transformation on the data and then forwarding the data to the next process

The data is transported between the processes using a fairly efficient text based format. It is described in :ref:`format`.
As such the main component of the suite is the parser/generator for this file format.
Data is first read in from many different sources and converted into this textual representation. It is then propagated to the next program that may perform transformations on the data.

The data is split into different streams. It is assumed that there is one element per stream and that the elements are read in at the same rate.
This may be addressed in the future by introducing a sentinel value for an empty element.
Various different interpolation techniques may then be used to adapt different data rates to each other.

The :ref:`SignalStreams` class parses the incoming text and then generates one iterator per stream that the operators can be mapped over before a final function call collects all iterators
in a round-robin procedure with each round generating one line of output.

This makes it fairly easy to perform arbitraty mappings on the input data. An arbitrary function may be given to each map that can be chosen from the set of built in functions.

Development Goals
-----------------
* Make N:M mappings possible
* Allow for variable data rates
* Allow for user generated mapping functions
