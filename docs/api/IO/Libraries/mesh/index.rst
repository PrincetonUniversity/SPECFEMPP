.. _mesh_reader:

Mesh Reader
===========

.. doxygenfunction:: specfem::IO::read_mesh


The above function reads the fortran binary mesh database and the following
functions are called in the order of appearance in the code:


.. doxygenfunction:: specfem::IO::mesh::fortran::read_mesh_database_header

.. doxygenfunction:: specfem::IO::mesh::fortran::read_coorg_elements

.. doxygenfunction:: specfem::IO::mesh::fortran::read_properties

.. doxygenfunction:: specfem::IO::mesh::fortran::read_mesh_database_attenuation

.. doxygenfunction:: specfem::IO::mesh::fortran::read_material_properties

.. doxygenfunction:: specfem::IO::mesh::fortran::read_boundaries

.. doxygenfunction:: specfem::IO::mesh::fortran::read_coupled_interfaces

.. doxygenfunction:: specfem::IO::mesh::fortran::read_tangential_elements

.. doxygenfunction:: specfem::IO::mesh::fortran::read_axial_elements

Finally, we add tags to the :cpp:struct:`specfem::mesh::mesh` using the
:cpp:struct:`specfem::mesh::tags` struct. The description of which can be found
here: :ref:`mesh_tags`.
