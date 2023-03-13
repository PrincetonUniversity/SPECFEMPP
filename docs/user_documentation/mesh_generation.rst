Mesh Generation
===============

In this version of the package meshfem has not been implemented. However, the package can read internal meshes generated via `SPECFEM2D mesh generator <https://specfem2d.readthedocs.io/en/latest/03_mesh_generation/>`_ . Please refer to the documentation there to generate meshes. Thus for now, we require a *Par_file* for generation of mesh and a *configuration file* for setting up and running the solver.

The recommended workflow for running the code would be to generate an internal mesh using ``xmeshfem2D``. Make sure the domain is entirely elastic, this version does not support acoustic domains. Then define the path to the generated database file using :ref:`database-file-parameter`.

Please have a look at the :ref:`cookbooks` for examples on generating a mesh.
