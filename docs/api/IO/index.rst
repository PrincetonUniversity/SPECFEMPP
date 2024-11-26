
Input/Output modules
====================

There are three high-level functions that are essential for the SPECFEM++
codebase: :cpp:func:`specfem::IO::read_mesh`,
:cpp:func:`specfem::IO::read_sources`, and
:cpp:func:`specfem::IO::read_receivers`. These functions are
used to read the mesh, sources and receivers from disk. The underlying
implementations are not exposed to the user. Currently supported
formats for the reading of the mesh is binary, and for sources and receivers it
is yaml.

In addition to these basic read functions, there are also the two
reader and writer classes, :cpp:class:`specfem::reader::wavefield` and
:cpp:class:`specfem::writer::wavefield`, which support both HDF5 and ASCII I/O.
And, to write seismograms, we can use :cpp:class:`specfem::writer::seismogram`.
Seismogram I/O is only supported in ASCII format thus far.

The slightly-lower level functionality to read and write data to and from disk
are exposed through the following modules:

.. toctree::
    :maxdepth: 2

    Libraries/index
    fortran_io
    writer/index
    reader/index


Read Mesh, Sources and Receivers
--------------------------------


.. doxygenfunction:: specfem::IO::read_mesh

.. doxygenfunction:: specfem::IO::read_sources

.. doxygenfunction:: specfem::IO::read_receivers
