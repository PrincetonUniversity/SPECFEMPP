Mesh Generation
===============

Mesh generation requires use of a meshing software. We provide an internal mesher to generate meshes for simple domain geometries i.e. layer cake model (rectangular domains with interfaces between different material systems). However, for more complex geometries, we recommend using external meshing software such as Gmsh (http://gmsh.info/) or CUBIT (https://cubit.sandia.gov/).

.. note::
    The internal mesher is a slightly modified version of the mesher provided with the original `SPECFEM2D <https://specfem2d.readthedocs.io/en/latest/03_mesh_generation/>`_ package.

We configure the mesher using a :ref:`Parameter_File` to define the meshing specifications and a :ref:`Topography_File` to define the topography of the domain. The meshing software is called using the following command:

.. code-block:: bash

    $ ./xmeshfem2d -p <Parameter_File>
