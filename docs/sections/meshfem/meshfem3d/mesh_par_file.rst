
``xmeshfem3D``: ``Mesh_Par_file`` Documentation
===============================================


``MESH_A_CHUNK_OF_THE_EARTH``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whether to mesh a chunk of the Earth.

:Type: ``logical``

.. code-block::
    :caption: Example

    title = My cool simulation

.. note::

    This parameter is not currently supported by the solver, so it should be
    set to ``false``.


``CHUNK_MESH_PAR_FILE``
~~~~~~~~~~~~~~~~~~~~~~~

The name of the parameter file for the chunk mesh. This file is used to
define the parameters for the chunk mesh.

:Type: ``string``

.. code-block::
    :caption: Example

    CHUNK_MESH_PAR_FILE = chunk_mesh_par_file


.. note::

    This parameter is not currently supported by the solver, so it's value
    is not used.

``LATITUDE_MIN``
~~~~~~~~~~~~~~~~

The minimum latitude of the mesh block. If ``SUPPRESS_UTM_PROJECTION`` is set to
``.true.``, this parameter is in meters. If ``SUPPRESS_UTM_PROJECTION`` is set to
``.false.``, this parameter is in degrees.

:Type: ``double``

.. code-block::
    :caption: Example

    LATITUDE_MIN = 0.0d0


``LATITUDE_MAX``
~~~~~~~~~~~~~~~~

The maximum latitude of the mesh block. If ``SUPPRESS_UTM_PROJECTION`` is set to
``.true.``, this parameter is in meters. If ``SUPPRESS_UTM_PROJECTION`` is set to
``.false.``, this parameter is in degrees.

:Type: ``double``

.. code-block::
    :caption: Example

    LATITUDE_MAX = 10.0d0


``LONGITUDE_MIN``
~~~~~~~~~~~~~~~~~

The minimum longitude of the mesh block. If ``SUPPRESS_UTM_PROJECTION`` is set to
``.true.``, this parameter is in meters. If ``SUPPRESS_UTM_PROJECTION`` is set to
``.false.``, this parameter is in degrees.

:Type: ``double``

.. code-block::
    :caption: Example

    LONGITUDE_MIN = 0.0d0


``LONGITUDE_MAX``
~~~~~~~~~~~~~~~~~

The maximum longitude of the mesh block. If ``SUPPRESS_UTM_PROJECTION`` is set to
``.true.``, this parameter is in meters. If ``SUPPRESS_UTM_PROJECTION`` is set to
``.false.``, this parameter is in degrees.

:Type: ``double``

.. code-block::
    :caption: Example

    LONGITUDE_MAX = 10.0d0


``DEPTH_BLOCK_KM``
~~~~~~~~~~~~~~~~~~

The depth of the mesh block in kilometers. This parameter is used to define the
depth of the mesh block in the vertical direction. The depth is defined as the
distance from the surface to the bottom of the mesh block.

:Type: ``double``

.. code-block::
    :caption: Example

    DEPTH_BLOCK_KM = 10.0d0

``UTM_PROJECTION_ZONE``
~~~~~~~~~~~~~~~~~~~~~~~

UTM projection zone in which your model resides, only valid when
`SUPPESS_UTM_PROJECTION` is `.false.`. Use a negative zone number for the Southern
hemisphere: the Northern hemisphere corresponds to zones `+1` to `+60`, the Southern
hemisphere to zones `-1` to `-60`.

We use the WGS84 (World Geodetic System 1984) reference ellipsoid for the UTM
projection. If you prefer to use the Clarke 1866 ellipsoid, edit file
`src/shared/utm_geo.f90`, uncomment that ellipsoid and recompile the code.

From `The Universal Transverse Mercator coordinate system
<http://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system>`_
: The Universal Transverse Mercator coordinate system was developed by the
United States Army Corps of Engineers in the 1940s. The system was based on an
ellipsoidal model of Earth. For areas within the contiguous United States the
Clarke Ellipsoid of 1866 was used. For the remaining areas of Earth, including
Hawaii, the International Ellipsoid was used. The WGS84 ellipsoid is now
generally used to model the Earth in the UTM coordinate system, which means that
current UTM northing at a given point can be `200+` meters different from the old
one. For different geographic regions, other datum systems (e.g.: ``ED50``, ``NAD83``)
can be used.

:Type: ``integer``

.. code-block::
    :caption: Example

    UTM_PROJECTION_ZONE = 11


``SUPPRESS_UTM_PROJECTION``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set to be `.false.`` when your model range is specified in geographical
coordinates, and needs to be `.true.`` when your model is specified in Cartesian
coordinates.

:Type: ``logical``

.. code-block::
    :caption: Example

    SUPPRESS_UTM_PROJECTION = .false.


``INTERFACES_FILE``
~~~~~~~~~~~~~~~~~~~

File which contains the description of the topography and of the interfaces
between the different layers of the model, if any. The number of spectral
elements in the vertical direction within each layer is also defined in this
file.

:Type: ``string``

.. code-block::
    :caption: Example

    INTERFACES_FILE = interfaces.text


``CAVITY_FILE``
~~~~~~~~~~~~~~~

File which contains the description of the cavity and of the interfaces.

:Type: ``string``

.. code-block::
    :caption: Example

    CAVITY_FILE = no_cavity.dat


``NGNOD``
~~~~~~~~~

Number of nodes for 2D and 3D shape functions for hexahedra.
Use 8 for 8-node mesh elements (bricks). 27-node elements are not supported by the internal mesher.

:Type: ``integer``

.. code-block::
    :caption: Example

    NGNOD = 8

``NEX_XI``
~~~~~~~~~~

Number of elements at the surface along the xi edge of the mesh at the surface.
Must be a multiple of NPROC_XI (and 8*NPROC_XI if mesh is not regular and contains doublings).

:Type: ``integer``

.. code-block::
    :caption: Example

    NEX_XI = 36

``NEX_ETA``
~~~~~~~~~~~

Number of elements at the surface along the eta edge of the mesh at the surface.
Must be a multiple of NPROC_ETA (and 8*NPROC_ETA if mesh is not regular and contains doublings).

:Type: ``integer``

.. code-block::
    :caption: Example

    NEX_ETA = 36

``NPROC_XI``
~~~~~~~~~~~~

Number of MPI processors along the xi direction.

:Type: ``integer``

.. code-block::
    :caption: Example

    NPROC_XI = 1

.. note::

    ``NPROC_XI > 1`` is not supported by the solver currently.

``NPROC_ETA``
~~~~~~~~~~~~~

Number of MPI processors along the eta direction.

:Type: ``integer``

.. code-block::
    :caption: Example

    NPROC_ETA = 1

.. note::

    ``NPROC_ETA > 1`` is not supported by the solver currently.


``USE_REGULAR_MESH``
~~~~~~~~~~~~~~~~~~~~

Whether to use a regular mesh. Set to ``.true.`` for regular meshes, ``.false.``
for irregular meshes.

:Type: ``logical``

.. code-block::
    :caption: Example

    USE_REGULAR_MESH = .true.


``NDOUBLINGS``
~~~~~~~~~~~~~~

Number of mesh doubling layers (for irregular meshes only).

:Type: ``integer``

.. code-block::
    :caption: Example

    NDOUBLINGS = 0


``NZ_DOUBLING_1``, ``NZ_DOUBLING_2``, ...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Position(s) of mesh doubling layers (for irregular meshes only).
Set these parameters if `NDOUBLINGS > 0`.

:Type: ``integer``

.. code-block::
    :caption: Example

    NZ_DOUBLING_1 = 40
    NZ_DOUBLING_2 = 48



``CREATE_ABAQUS_FILES``
~~~~~~~~~~~~~~~~~~~~~~~

Whether to create mesh files for ABAQUS visualization.

:Type: ``logical``

.. code-block::
    :caption: Example

    CREATE_ABAQUS_FILES = .false.



``CREATE_DX_FILES``
~~~~~~~~~~~~~~~~~~~

Whether to create mesh files for DX visualization.

:Type: ``logical``

.. code-block::
    :caption: Example

    CREATE_DX_FILES = .false.



``CREATE_VTK_FILES``
~~~~~~~~~~~~~~~~~~~~

Whether to create mesh files for VTK visualization.

:Type: ``logical``

.. code-block::
    :caption: Example

    CREATE_VTK_FILES = .true.


``SAVE_MESH_AS_CUBIT``
~~~~~~~~~~~~~~~~~~~~~~

Whether to store mesh files as Cubit-exported files into directory MESH/ (for single process run).

:Type: ``logical``

.. code-block::
    :caption: Example

    SAVE_MESH_AS_CUBIT = .false.


``LOCAL_PATH``
~~~~~~~~~~~~~~

Path to store the database files.

:Type: ``string``

.. code-block::
    :caption: Example

    LOCAL_PATH = /path/to/OUTPUT_FILES/DATABASES_MPI


``THICKNESS_OF_X_PML``
~~~~~~~~~~~~~~~~~~~~~~

Thickness of the CPML absorbing layer in the x direction (in model units).

:Type: ``double``

.. code-block::
    :caption: Example

    THICKNESS_OF_X_PML = 12.3d0

.. note::

    CPML is not yet supported by the solver.

``THICKNESS_OF_Y_PML``
~~~~~~~~~~~~~~~~~~~~~~

Thickness of the CPML absorbing layer in the y direction (in model units).

:Type: ``double``

.. code-block::
    :caption: Example

    THICKNESS_OF_Y_PML = 12.3d0

.. note::

    CPML is not yet supported by the solver.


``THICKNESS_OF_Z_PML``
~~~~~~~~~~~~~~~~~~~~~~

Thickness of the CPML absorbing layer in the z direction (in model units).

:Type: ``double``

.. code-block::
    :caption: Example

    THICKNESS_OF_Z_PML = 12.3d0

.. note::

    CPML is not yet supported by the solver.


``NMATERIALS``
~~~~~~~~~~~~~~

Number of materials in the model.

:Type: ``integer``

.. code-block::
    :caption: Example

    NMATERIALS = 1


``Material Properties Table``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines the properties for each material.

- ``Q_Kappa``: Attenuation quality factor for bulk modulus
- ``Q_mu``: Attenuation quality factor for shear modulus
- ``anisotropy_flag``: 0 = no anisotropy; 1,2,... = see implementation
- ``domain_id``: 1 = acoustic, 2 = elastic

:Type: ``string``

:Format: ``material_id  rho  vp  vs  Q_Kappa  Q_mu  anisotropy_flag  domain_id``

.. code-block::
    :caption: Example

    1   2300.0   2800.0   1500.0   2444.4    300.0 0 2


``NREGIONS``
~~~~~~~~~~~~

Number of regions in the model.

:Type: ``integer``

.. code-block::
    :caption: Example

    NREGIONS = 1


``Region Properties Table``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines the regions of the model.

:Type: ``string``

:Format: ``NEX_XI_BEGIN  NEX_XI_END  NEX_ETA_BEGIN  NEX_ETA_END  NZ_BEGIN  NZ_END  material_id``


.. code-block::
    :caption: Example

    1   36   1   36   1   16   1
