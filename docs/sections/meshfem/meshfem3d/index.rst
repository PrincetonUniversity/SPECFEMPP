.. _MESHFEM3D_Parameter_documentation:

MESHFEM3D Parameter Documentation
=================================

SPECFEM++ uses a modified version of the mesher provided by the original
`SPECFEM3D <https://specfem3d.readthedocs.io/en/latest/03_mesh_generation/>`_
code. The modifications isolate only the necessary parameters for the meshing
process and remove those needed by the solver from the original ``Par_File``.
We document only the parameters that are used by the meshing process.
However you should also refer to the `SPECFEM3D documentation
<https://specfem3d.readthedocs.io/en/latest/03_mesh_generation/>`_ for a more
in-depth description of the mesher.

To define the meshing parameters, you will need to create a ``Mesh_Par_File``.
This is a simple text file that contains the parameters you wish to use. The
parameters are defined in the following format:

.. code-block:: bash

    parameter_name = parameter_value

Since the mesh is generated in a 2 step process, the parameters are divided
into two sections. The first section is used to generate the mesh and the
second section is used to generate the mesh databases.

.. code-block:: bash

    xmeshfem3D -p Mesh_Par_File
    xgenerate_databases -p Par_File



Parameter Description
---------------------

.. toctree::
    :maxdepth: 1

    mesh_par_file
    generate_databases



.. _interfaces_file:

Interfaces File
---------------

Topography files are used to define the surface topography of the mesh. The
topography file is a simple text file that describes the topography of every
interface in the simulation domain.  For example the following topography file
describes a simple 1 layer model with a flat surface and a flat interface file.

.. code-block:: bash
    :caption: Example ``interfaces.txt``

    # number of interfaces
    1
    #
    # We describe each interface below, structured as a 2D-grid, with several parameters :
    # number of points along XI and ETA, minimal XI ETA coordinates
    # and spacing between points which must be constant.
    # Then the records contain the Z coordinates of the NXI x NETA points.
    #
    # interface number 1 (topography, top of the mesh)
    .true. 2 2 0.0 0.0 1000.0 1000.0
    path/to/interface1.txt
    #
    # for each layer, we give the number of spectral elements in the vertical direction
    #
    # layer number 1 (top layer)
    16

Here the simplest case of an interface file is show:

.. code-block:: bash
    :caption: Example ``interfaces1.txt``

    0
    0
    0
    0
