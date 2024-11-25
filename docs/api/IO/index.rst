
Input/Output modules
====================

There are three high-level functions that are essential for the SPECFEM++
codebase: :cpp:function:`specfem::IO::mesh::read_mesh`,
:cpp:function:`specfem::IO::sources::read_sources`, and
:cpp:function:`specfem::IO::receivers::read_receivers`. These functions are
used to read the mesh, sources and receivers from disk. The undelying
implementations are not expose to the user. Currently supporte
formats for the reading of the mesh is binary, and for sources and receivers it
is yaml.

The slightly-lower level functions to read and write data to and from disk are
exposed through the following abstractions.

.. toctree::
    :maxdepth: 2

    Libraries/index
    fortran_io
    writer/index
    reader/index
