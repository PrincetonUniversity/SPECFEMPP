
Input/Output modules
====================

There are three high-level functions that are essential for the SPECFEM++
codebase: :cpp:func:`specfem::io::read_2d_mesh`,
:cpp:func:`specfem::io::read_sources`, and
:cpp:func:`specfem::io::read_receivers`. These functions are
used to read the mesh, sources and receivers from disk. The underlying
implementations are not exposed to the user. Currently supported
formats for the reading of the mesh is binary, and for sources and receivers it
is yaml.

In addition to these basic read functions, there are also the two
reader and writer classes, :cpp:class:`specfem::io::wavefield_reader` and
:cpp:class:`specfem::io::wavefield_writer`, which support Numpy Binary and Zip (NPY and NPZ, respectively), HDF5, ADIOS and ASCII I/O.
And, to write seismograms, we can use :cpp:class:`specfem::io::seismogram_writer`.
Seismogram I/O is only supported in ASCII format thus far.

The slightly-lower level functionality to read and write data to and from disk
are exposed through the following modules:

.. toctree::
    :maxdepth: 2

    Libraries/index
    fortran_io
    writer/index
    reader/index


Input for the 2D Solver
+++++++++++++++++++++++

Read the 2D Mesh
----------------

.. doxygenfunction:: specfem::io::read_2d_mesh


Read 2D Sources
---------------

.. doxygenfunction:: specfem::io::read_2d_sources(const std::string &sources_file, const int nsteps, const type_real user_t0, const type_real dt, const specfem::simulation::type simulation_type)


Read 2D Receivers
-----------------

.. doxygenfunction:: specfem::io::read_2d_receivers(const std::string &stations_file, const type_real angle)



Input for the 3D Solver
+++++++++++++++++++++++


Read the 3D Mesh
----------------

.. doxygenfunction:: specfem::io::read_3d_mesh


Read 3D Sources
---------------

.. doxygenfunction:: specfem::io::read_3d_sources(const std::string &sources_file, const int nsteps, const type_real user_t0, const type_real dt, const specfem::simulation::type simulation_type)


Read 3D Receivers
-----------------

.. doxygenfunction:: specfem::io::read_3d_receivers(const std::string &stations_file)



Helper functions
''''''''''''''''

**Reading any values**

.. doxygenfunction:: specfem::io::mesh::impl::fortran::dim3::try_read_line

**Reading any array**

.. doxygenfunction:: specfem::io::mesh::impl::fortran::dim3::try_read_array

.. doxygenfunction:: specfem::io::mesh::impl::fortran::dim3::read_array

**Reading index arrays**

.. doxygenfunction:: specfem::io::mesh::impl::fortran::dim3::try_read_index_array

.. doxygenfunction:: specfem::io::mesh::impl::fortran::dim3::read_index_array
