.. _mesh_reader:

Mesh Reader
===========


The below functions are helper functions for the Fortran binary mesh database
reader. The functions are called in :cpp:func:`specfem::IO::read_2d_mesh`
in sequential order and faciliate the reading of the mesh database.

.. doxygenfunction:: specfem::IO::mesh::impl::fortran::dim2::read_mesh_database_header

.. doxygenfunction:: specfem::IO::mesh::impl::fortran::dim2::read_coorg_elements

.. doxygenfunction:: specfem::IO::mesh::impl::fortran::dim2::read_properties

.. doxygenfunction:: specfem::IO::mesh::impl::fortran::dim2::read_mesh_database_attenuation

.. doxygenfunction:: specfem::IO::mesh::impl::fortran::dim2::read_material_properties

.. doxygenfunction:: specfem::IO::mesh::impl::fortran::dim2::read_boundaries

.. doxygenfunction:: specfem::IO::mesh::impl::fortran::dim2::read_coupled_interfaces

.. doxygenfunction:: specfem::IO::mesh::impl::fortran::dim2::read_tangential_elements

.. doxygenfunction:: specfem::IO::mesh::impl::fortran::dim2::read_axial_elements

Finally, we add tags to the :cpp:struct:`specfem::mesh::mesh` using the
:cpp:struct:`specfem::mesh::tags` struct. The description of which can be found
here: :ref:`mesh_tags`.
