.. _mesh_dev_guide:

Mesh developer guide
=====================

``mesh`` struct defines IO routines and data structures to store data related to spectral element mesh. The mesh itself is generated using either the internal mesher or an external mesher. The current implementation only supports meshes generated using `MESHFEM2D <https://specfem2d.readthedocs.io/en/latest/03_mesh_generation/>`_. This guide describes the data structures and IO routines required to incorporate a new mesher.

.. note::
    A few changes need to made to the code before I can write this guide. These changes would be incorporated in the next release.
