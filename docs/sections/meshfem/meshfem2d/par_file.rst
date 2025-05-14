``xmeshfem2D``: ``Par_file`` Documentation
++++++++++++++++++++++++++++++++++++++++++

Simulation input parameters
===========================

``title``
~~~~~~~~~

Title of the simulation. This is used to identify the simulation in the output files and logs.

:Type: ``string``

.. code-block::
    :caption: Example

    title = My cool simulation

``NPROC``
~~~~~~~~~

Number of MPI processors to use for the simulation. This is used to determine the number of processes to spawn for the simulation.

:Type: ``integer``

.. code-block::
    :caption: Example

    NPROC = 1

.. note::
    SPECFEM++ currently only supports a single MPI processor. Please set this value to 1.


``OUTPUT_FILES``
~~~~~~~~~~~~~~~~

This is the location where the mesher will store all the output files.

:Type: ``Path`` as ``string``

.. code-block::
    :caption: Example

    OUTPUT_FILES = ./OUTPUT_FILES

Meshing Parameters
==================

``PARTITIONING_TYPE``
~~~~~~~~~~~~~~~~~~~~~

The type of mesh partitioning to use for MPI simulations.
    - SCOTCH: ``3``
    - METIS: ``2``
    - INTERNAL (Ascending): ``1``

:Type: ``int``

.. code-block::
    :caption: Example

    PARTITIONING_TYPE = 3

.. note::

    This parameter is only used for MPI simulations and is ignored for serial simulations.


``NGNOD``
~~~~~~~~~

Number of control nodes per element. Supported values: ``4`` and ``9``

:Type: ``int``

.. code-block::
    :caption: Example

    NGNOD = 9


``database_filename``
~~~~~~~~~~~~~~~~~~~~~

Name of the database file to store the generated mesh.

:Type: ``string``

.. code-block::
    :caption: Example

    database_filename = database.bin

.. note::
    The database filename needs to be same in meshing and solver parameter files.



Velocity Models
===============

Velocity models are defined in the ``meshfem2d`` module. The velocity model is
defined in the ``model`` section of the input file. The velocity model is
defined by a set of parameters that describe the material properties of the
model. An example of a velocity model is shown below:

.. code-block:: bash

    # number of material systems
    nbmodels = 2
    # acoustic elastic material system
    1 1 2500.d0 3400.d0 1963.d0 0 0 9999 9999 0 0 0 0 0 0
    2 1 1020.d0 1500.d0 0.d0 0 0 9999 9999 0 0 0 0 0 0

Note that ``nbmodels`` must always be followed by the number of material systems
in the model, one for each material system. See paramter descriptions below for
more details on the parameters.

**Metaparameters**

``nbmodels``
~~~~~~~~~~~~

Number of material systems in the model.

:Type: ``int``

.. code-block::
    :caption: Example

    nbmodels = 1


``TOPOGRAPHY_FILE``
~~~~~~~~~~~~~~~~~~~

Path to an external topography file.

:Type: ``string``

.. code-block::
    :caption: Example

    TOPOGRAPHY_FILE = topography.dat


``read_external_mesh``
~~~~~~~~~~~~~~~~~~~~~~

If ``True`` the mesh is read from an external file.

Type
    ``logical``

.. code-block::
    :caption: Example

    read_external_mesh = .true.


**Description of material system**

Each material system in the model is described by a string.

.. note::
    Only elastic, poroelastic, and acoustic material systems are supported.


Elastic material system
~~~~~~~~~~~~~~~~~~~~~~~

An elastic medium can be described by the following parameters:

- ``model_number``: integer number to refence the material system
- ``rho``: density
- ``Vp``: P-wave velocity
- ``QKappa``: Attenuation parameter (set to ``9999`` for no attenuation)

:Type: ``string``

:Format: ``model_number 1 rho Vp 0 0 QKappa 9999 0 0 0 0 0 0``

.. code-block::
    :caption: Example

    1 1 2700.d0 3000.d0 1732.051d0 0 0 9999 9999 0 0 0 0 0 0

Acoustic material system
~~~~~~~~~~~~~~~~~~~~~~~~

An acoustic medium can be described by the following parameters:

- ``model_number``: integer number to refence the material system
- ``rho``: density
- ``Vp``: P-wave velocity
- ``Vs``: S-wave velocity
- ``QKappa``: Attenuation parameter (set to ``9999`` for no attenuation)
- ``QMu``: Attenuation parameter (set to ``9999`` for no attenuation)

Type
    ``string``

:Format: ``model_number 1 rho Vp Vs 0 0 QKappa QMu 0 0 0 0 0 0``

.. code-block::
    :caption: Example

    1 1 2700.d0 3000.d0 1732.051d0 0 0 9999 9999 0 0 0 0 0 0


Poroelastic material system
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A poroelastic medium can be described by the following parameters:

- ``model_number``: integer number to refence the material system
- ``rhos``: solid density
- ``rhof``: fluid density
- ``phi``: porosity
- ``c``: Biot coefficient
- ``kxx``: permeability in x direction
- ``kxz``: permeability in z direction
- ``kzz``: permeability in z direction
- ``Ks``: bulk modulus of solid
- ``Kf``: bulk modulus of fluid
- ``Kfr``: bulk modulus of fluid in the frame
- ``etaf``: viscosity of fluid
- ``mufr``: shear modulus of fluid in the frame
- ``Qmu``: attenuation parameter (set to ``9999`` for no attenuation)

:Type: ``string``

:Format: ``model_number 3 rhos rhof phi c kxx kxz kzz Ks Kf Kfr etaf mufr Qmu``

.. code-block::
    :caption: Example

    1 3 2650.d0 880.d0 0.1d0 2.0 1d-9 0.d0 1d-9 12.2d9 1.985d9 9.6d9 0.d0 5.1d9 9999


Internal Meshing
================

``interfacesfile``
~~~~~~~~~~~~~~~~~~

The file describing the topography of the simulation domain. For more information, see the :ref:`topography_file` section.

Type
    ``path`` as string

.. code-block::
    :caption: Example

    interfacesfile = "topography.dat"


``xmin``
~~~~~~~~

The minimum x-coordinate of the simulation domain.

Type
    ``real``

.. code-block::
    :caption: Example

    xmin = 0.0


``xmax``
~~~~~~~~

The maximum x-coordinate of the simulation domain.

Type
    ``real``

.. code-block::
    :caption: Example

    xmax = 6400.0


``nx``
~~~~~~

The number of elements in the x-direction.

:Type: ``int``

.. code-block::
    :caption: Example

    nx = 64


``STACEY_ABSORBING_CONDITIONS``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``True``, Stacey absorbing boundary conditions are used.

Type
    ``logical``

.. code-block::
    :caption: Example

    STACEY_ABSORBING_CONDITIONS = .true.


``absorbbottom``
~~~~~~~~~~~~~~~~


If ``True``, the bottom boundary is an absorbing boundary.

Type
    ``logical``

.. code-block::
    :caption: Example

    absorbbottom = .true.



``absorbtop``
~~~~~~~~~~~~~

If ``True``, the top boundary is an absorbing boundary.

:Type: ``logical``

.. code-block::
    :caption: Example

    absorbtop = .true.


``absorbleft``
~~~~~~~~~~~~~~

If ``True``, the left boundary is an absorbing boundary.

:Type: ``logical``

.. code-block::
    :caption: Example

    absorbleft = .true.


``absorbright``
~~~~~~~~~~~~~~~

If ``True``, the right boundary is an absorbing boundary.

:Type: ``logical``

.. code-block::
    :caption: Example

    absorbright = .true.

``nbregions``
~~~~~~~~~~~~~

The number of regions in the simulation domain.

:Type: ``int``

.. code-block::
    :caption: Example

    nbregions = 1


Describing a region
~~~~~~~~~~~~~~~~~~~

The region is described using a string. And has following parameters:

- ``nxmin``: Integer value describing the x-coordinate of the spectral element at the bottom left corner of the region.
- ``nxmax``: Integer value describing the x-coordinate of the spectral element at the top right corner of the region.
- ``nzmin``: Integer value describing the z-coordinate of the spectral element at the bottom left corner of the region.
- ``nzmax``: Integer value describing the z-coordinate of the spectral element at the top right corner of the region.
- ``material_number``: Integer value describing the type of material in the region. This value references the material number in the ``velocity_model``.

:Type: ``string``

:Format: ``nxmin nxmax nzmin nzmax material_number``

.. code-block::
    :caption: Example

    0 63 0 63 1

.. note::

    The region description(s) must be preceded by the number of regions in the
    simulation domain. For example, if there are 2 regions, the file should look
    like this if you remove comments:

    .. code-block:: bash

        nbregions = 2
        0 63 0 30 1
        0 63 31 63 2


External Meshing
================

Parameters here describe the paths to different files generated when using an external mesher.

``mesh_file``
~~~~~~~~~~~~~

Path to the mesh file.

Type
    ``path`` as ``string``

.. code-block::
    :caption: Example

    mesh_file = ./DATA/Mesh_canyon/canyon_mesh_file

``nodes_coords_file``
~~~~~~~~~~~~~~~~~~~~~

Path to the file containing the coordinates of the nodes.

:Type: ``path`` as ``string``

.. code-block::
    :caption: Example

    nodes_coords_file = ./DATA/Mesh_canyon/canyon_nodes_coords


``materials_file``
~~~~~~~~~~~~~~~~~~

Path to the file containing the materials number for each element.

:Type: ``path`` as ``string``

.. code-block::
    :caption: Example

    materials_file = ./DATA/Mesh_canyon/canyon_materials_file

``free_surface_file``
~~~~~~~~~~~~~~~~~~~~~

Path to the file containing the element number describing the free surface.

:Type: ``path`` as ``string``

.. code-block::
    :caption: Example

    free_surface_file = ./DATA/Mesh_canyon/canyon_free_surface_file


``axial_elements_file``
~~~~~~~~~~~~~~~~~~~~~~~

Path to the file containing the element number for elements on the axis.

:Type: ``path``

.. code-block::
    :caption: Example

    axial_elements_file = ./DATA/Mesh_canyon/canyon_axial_elements_file

.. note::
    This parameter is not supported in the solver.


``absorbing_surface_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Path to the file containing the element number for elements on the absorbing surface.

:Type: ``path`` as ``string``

.. code-block::
    :caption: Example

    absorbing_surface_file = ./DATA/Mesh_canyon/canyon_absorbing_surface_file

.. note::
    This parameter is not supported in the solver.


``acoustic_forcing_surface_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Path to the file containing the element number for elements on the acoustic forcing surface.

:Type: ``path`` as ``string``

.. code-block::
    :caption: Example

    acoustic_forcing_surface_file = ./DATA/Mesh_canyon/canyon_acoustic_forcing_surface_file

.. note::
    This parameter is not supported in the solver.


``absorbing_cpml_file``
~~~~~~~~~~~~~~~~~~~~~~~

Path to the file containing the element number for elements on the absorbing PML surface.

:Type: ``path``

.. code-block::
    :caption: Example

    absorbing_cpml_file = ./DATA/Mesh_canyon/canyon_absorbing_cpml_file

.. note::
    This parameter is not supported in the solver.


``tangential_detection_curve_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Path to the file containing the element number for elements on the tangential curve.

:Type: ``path`` as ``string``

.. code-block::
    :caption: Example

    tangential_detection_curve_file = ./DATA/Mesh_canyon/canyon_tangential_detection_curve_file

.. note::
    This parameter is not supported in the solver.



Display parameters
==================

``output_grid_Gnuplot``
~~~~~~~~~~~~~~~~~~~~~~~

Output grid as a Gnuplot file

:Type: ``logical``

.. code-block::
    :caption: Example

    output_grid_Gnuplot = .true.

``output_grid_ASCII``
~~~~~~~~~~~~~~~~~~~~~

Output grid as an ASCII file

Type
    ``logical``

.. code-block::
    :caption: Example

    output_grid_ASCII = .true.



Receiver Parameters
====================

**Define meta parameters**

``use_existing_STATIONS``
~~~~~~~~~~~~~~~~~~~~~~~~~

If set to ``.true.``, the receivers will be places based on an existing STATIONS file.

:Type: ``logical``

.. code-block::
    :caption: Example

    use_existing_STATIONS = .false.


``nreceiversets``
~~~~~~~~~~~~~~~~~

Number of receiver sets.

:Type: ``int``

.. code-block::
    :caption: Example

    nreceiversets = 1


``anglerec``
~~~~~~~~~~~~

Angle to rotate components at receivers

:Type: ``real``

.. code-block::
    :caption: Example

    anglerec = 0.0d0


``rec_normal_to_surface``
~~~~~~~~~~~~~~~~~~~~~~~~~

If set to ``.true.``, the receiver base angle will be set normal to the surface. Requires external mesh and tangential curve file.

:Type: ``logical``

.. code-block::
    :caption: Example

    rec_normal_to_surface = .false.

.. note::
    This paramter is not supported yet in the solver.


**Define receiver sets**

Next we define each receiver sets using the following parameters:

``nrec``
~~~~~~~~

Number of receivers in this set. The receivers will be placed at equal distances.

Type
    ``int``

.. code-block::
    :caption: Example

    nrec = 10


``xdeb``
~~~~~~~~

X coordinate of the first receiver in this set.

:Type: ``real``

.. code-block::
    :caption: Example

    xdeb = 0.0d0


``zdeb``
~~~~~~~~

Y coordinate of the first receiver in this set.

:Type: ``real``

.. code-block::
    :caption: Example

    zdeb = 0.0d0

``xfin``
~~~~~~~~

X coordinate of the last receiver in this set.

:Type: ``real``

.. code-block::
    :caption: Example

    xfin = 6400.0d0


``zfin``
~~~~~~~~

Z coordinate of the last receiver in this set.

:Type: ``real``

.. code-block::
    :caption: Example

    zfin = 0.0d0


``record_at_surface_same_vertical``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If set to ``.true.``, the receivers will be placed at the surface of the medium. The vertical position of the receivers will be replaces with topography height.

:Type: ``logical``

.. code-block::
    :caption: Example

    record_at_surface_same_vertical = .false.


``stations_filename``
~~~~~~~~~~~~~~~~~~~~~

Name of the STATIONS file to use. if ``use_existing_STATIONS`` is set to ``.true.``, this defines a file to read receiver locations from. If ``use_existing_STATIONS`` is set to ``.false.``, this defines a file to write receiver locations to.

:Type: ``string``

.. code-block::
    :caption: Example

    stations_filename = stations.dat
