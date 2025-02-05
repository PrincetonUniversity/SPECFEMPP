.. _MESHFEM_Parameter_documentation:

MESHFEM Parameter Documentation
===============================

SPECFEM++ uses a modified version of the mesher provided by the original `SPECFEM2D <https://specfem2d.readthedocs.io/en/latest/03_mesh_generation/>`_ code. The modifications isolate only the nacessary parameters for the meshing process and remove those needed by the solver from the original ``Par_File``. Here I will document the parameters that are used by the meshing process. However you should also refer to the `SPECFEM2D documentation <https://specfem2d.readthedocs.io/en/latest/03_mesh_generation/>`_ for a more in-depth description of the mesher.

To define the meshing parameters, you will need to create a ``Par_File``. This is a simple text file that contains the parameters you wish to use. The parameters are defined in the following format:

.. code-block:: bash

    parameter_name = parameter_value

Parameter Description
---------------------

.. toctree::
    :maxdepth: 1

    header
    meshing_parameters
    receivers
    velocity_models
    internal_mesher
    external_mesher
    display_parameter
    receiver_documentation

.. _topography_file:

Topography File
---------------

Topography files are used to define the surface topography of the mesh. The topography file is a simple text file that describes the topography of every interface in the simulation domain.  For example the following topography file describes a simple 2 layer model with a flat surface and a flat interface between the two layers:

.. code-block:: bash

    #
    # number of interfaces
    #
    3
    #
    # for each interface below, we give the number of points and then x,z for each point
    #
    #
    # interface number 1 (bottom of the mesh)
    #
    2
    0 0
    6400 0
    #
    # interface number 2 (ocean bottom)
    #
    2
    0 2400
    6400 2400
    #
    # interface number 3 (topography, top of the mesh)
    #
    2
    0 4800
    6400 4800
    #
    # for each layer, we give the number of spectral elements in the vertical direction
    #
    #
    # layer number 1 (bottom layer)
    #
    54
    #
    # layer number 2 (top layer)
    #
    54
