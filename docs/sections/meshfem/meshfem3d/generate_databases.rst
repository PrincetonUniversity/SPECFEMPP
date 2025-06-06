``xgenerate_databases``: `Par_file` Documentation
=================================================

Below documented are the parameters used by ``xgenerate_databases`` to generate
the databases.

We have not yet updated the ``read_parameters()`` function to read only the
required parameters, which is why we are documenting all of them here for now.
Otherwise, ``xgenerate_databases`` would throw an error.

Simulation input parameters
+++++++++++++++++++++++++++


``SIMULATION_TYPE``
~~~~~~~~~~~~~~~~~~~

Type of simulation to run.

- 1 = forward simulation
- 2 = adjoint simulation
- 3 = both simultaneously

:Type: ``integer``

.. code-block::
    :caption: Example

    SIMULATION_TYPE = 1

``NOISE_TOMOGRAPHY``
~~~~~~~~~~~~~~~~~~~~

Type of noise simulation.

- 0 = earthquake simulation
- 1/2/3 = three steps in noise simulation

:Type: ``integer``

.. code-block::
    :caption: Example

    NOISE_TOMOGRAPHY = 0

``SAVE_FORWARD``
~~~~~~~~~~~~~~~~

Save the forward wavefield for later use (e.g., in adjoint simulations).

:Type: ``logical``

.. code-block::
    :caption: Example

    SAVE_FORWARD = .false.

``INVERSE_FWI_FULL_PROBLEM``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve a full FWI inverse problem from a single calling program with no I/Os, storing everything in memory,
or run a classical forward or adjoint problem only and save the seismograms and/or sensitivity kernels to disk (with costlier I/Os).

:Type: ``logical``

.. code-block::
    :caption: Example

    INVERSE_FWI_FULL_PROBLEM = .false.

``UTM_PROJECTION_ZONE``
~~~~~~~~~~~~~~~~~~~~~~~

UTM projection zone number. Use a negative zone number for the Southern hemisphere.

:Type: ``integer``

.. code-block::
    :caption: Example

    UTM_PROJECTION_ZONE = 11

``SUPPRESS_UTM_PROJECTION``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppress UTM projection and use Cartesian coordinates.

:Type: ``logical``

.. code-block::
    :caption: Example

    SUPPRESS_UTM_PROJECTION = .true.

``NPROC``
~~~~~~~~~

Number of MPI processors to use.

:Type: ``integer``

.. code-block::
    :caption: Example

    NPROC = 1

``NSTEP``
~~~~~~~~~

Number of time steps in the simulation.

:Type: ``integer``

.. code-block::
    :caption: Example

    NSTEP = 5000

``DT``
~~~~~~

Time step size (in seconds).

:Type: ``real``

.. code-block::
    :caption: Example

    DT = 0.05

``LTS_MODE``
~~~~~~~~~~~~

Set to true to use local-time stepping (LTS).

:Type: ``logical``

.. code-block::
    :caption: Example

    LTS_MODE = .false.

``PARTITIONING_TYPE``
~~~~~~~~~~~~~~~~~~~~~

Partitioning algorithm for mesh decomposition.

- 1 = SCOTCH (default)
- 2 = METIS
- 3 = PATOH
- 4 = ROWS_PART

:Type: ``integer``

.. code-block::
    :caption: Example

    PARTITIONING_TYPE = 1


LDDRK time scheme
+++++++++++++++++


``USE_LDDRK``
~~~~~~~~~~~~~

Use the LDDRK time integration scheme.

:Type: ``logical``

.. code-block::
    :caption: Example

    USE_LDDRK = .false.

``INCREASE_CFL_FOR_LDDRK``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Increase the CFL number when using LDDRK.

:Type: ``logical``

.. code-block::
    :caption: Example

    INCREASE_CFL_FOR_LDDRK = .false.

``RATIO_BY_WHICH_TO_INCREASE_IT``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ratio by which to increase the CFL number if using LDDRK.

:Type: ``real``

.. code-block::
    :caption: Example

    RATIO_BY_WHICH_TO_INCREASE_IT = 1.4

Mesh
++++


``NGNOD``
~~~~~~~~~

Number of nodes for 2D and 3D shape functions for hexahedra.

:Type: ``integer``

.. code-block::
    :caption: Example

    NGNOD = 8

``MODEL``
~~~~~~~~~

Model type to use.

- ``default``: model parameters described by mesh properties
- ``1d_prem``, ``1d_socal``, ``1d_cascadia``: 1D models
- ``aniso``, ``external``, ``gll``, ``salton_trough``, ``tomo``, ``SEP``, ``coupled``: 3D models

:Type: ``string``

.. code-block::
    :caption: Example

    MODEL = default

``COUPLED_MODEL_DIRECTORY``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Path for external model files (if MODEL = coupled).

:Type: ``string``

.. code-block::
    :caption: Example

    COUPLED_MODEL_DIRECTORY = /path/to/coupled_model/

``TOMOGRAPHY_PATH``
~~~~~~~~~~~~~~~~~~~

Path for external tomographic model files.

:Type: ``string``

.. code-block::
    :caption: Example

    TOMOGRAPHY_PATH = /path/to/tomo_files/

``SEP_MODEL_DIRECTORY``
~~~~~~~~~~~~~~~~~~~~~~~

Path for SEP model files (oil-industry format).

:Type: ``string``

.. code-block::
    :caption: Example

    SEP_MODEL_DIRECTORY = /path/to/my_SEP_model/


----


``APPROXIMATE_OCEAN_LOAD``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Include approximate ocean load in the model.

:Type: ``logical``

.. code-block::
    :caption: Example

    APPROXIMATE_OCEAN_LOAD = .false.

``TOPOGRAPHY``
~~~~~~~~~~~~~~

Include topography in the model.

:Type: ``logical``

.. code-block::
    :caption: Example

    TOPOGRAPHY = .false.

``ATTENUATION``
~~~~~~~~~~~~~~~

Include attenuation in the simulation.

:Type: ``logical``

.. code-block::
    :caption: Example

    ATTENUATION = .false.

``ANISOTROPY``
~~~~~~~~~~~~~~

Include anisotropy in the simulation.

:Type: ``logical``

.. code-block::
    :caption: Example

    ANISOTROPY = .false.

``GRAVITY``
~~~~~~~~~~~

Include gravity in the simulation.

:Type: ``logical``

.. code-block::
    :caption: Example

    GRAVITY = .false.

``ATTENUATION_f0_REFERENCE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reference frequency in Hz at which the velocity values in the velocity model are given (used only if attenuation is enabled).

:Type: ``real``

.. code-block::
    :caption: Example

    ATTENUATION_f0_REFERENCE = 18.d0

``MIN_ATTENUATION_PERIOD``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimum attenuation period (in seconds) over which to mimic a constant Q factor.

:Type: ``real``

.. code-block::
    :caption: Example

    MIN_ATTENUATION_PERIOD = 999999998.d0

``MAX_ATTENUATION_PERIOD``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum attenuation period (in seconds) over which to mimic a constant Q factor.

:Type: ``real``

.. code-block::
    :caption: Example

    MAX_ATTENUATION_PERIOD = 999999999.d0

``COMPUTE_FREQ_BAND_AUTOMATIC``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute the attenuation period range automatically based on mesh resolution.

:Type: ``logical``

.. code-block::
    :caption: Example

    COMPUTE_FREQ_BAND_AUTOMATIC = .true.

``USE_OLSEN_ATTENUATION``
~~~~~~~~~~~~~~~~~~~~~~~~~

Use Olsen's constant for Q_mu = constant * V_s attenuation rule.

:Type: ``logical``

.. code-block::
    :caption: Example

    USE_OLSEN_ATTENUATION = .false.

``OLSEN_ATTENUATION_RATIO``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Olsen's attenuation ratio (used if USE_OLSEN_ATTENUATION is true).

:Type: ``real``

.. code-block::
    :caption: Example

    OLSEN_ATTENUATION_RATIO = 0.05

Absorbing boundary conditions
+++++++++++++++++++++++++++++


``PML_CONDITIONS``
~~~~~~~~~~~~~~~~~~

Enable C-PML absorbing boundary conditions.

:Type: ``logical``

.. code-block::
    :caption: Example

    PML_CONDITIONS = .false.

``PML_INSTEAD_OF_FREE_SURFACE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use C-PML at the top surface instead of a free surface.

:Type: ``logical``

.. code-block::
    :caption: Example

    PML_INSTEAD_OF_FREE_SURFACE = .false.

``f0_FOR_PML``
~~~~~~~~~~~~~~

Dominant frequency for C-PML boundary conditions.

:Type: ``real``

.. code-block::
    :caption: Example

    f0_FOR_PML = 0.05555

``STACEY_ABSORBING_CONDITIONS``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable Stacey absorbing boundary conditions.

:Type: ``logical``

.. code-block::
    :caption: Example

    STACEY_ABSORBING_CONDITIONS = .true.

``STACEY_INSTEAD_OF_FREE_SURFACE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Stacey absorbing conditions at the top surface instead of a free surface.

:Type: ``logical``

.. code-block::
    :caption: Example

    STACEY_INSTEAD_OF_FREE_SURFACE = .false.

``BOTTOM_FREE_SURFACE``
~~~~~~~~~~~~~~~~~~~~~~~

Make the bottom (zmin) a free surface instead of absorbing.

:Type: ``logical``

.. code-block::
    :caption: Example

    BOTTOM_FREE_SURFACE = .false.


undoing attenuation and/or PMLs for sensitivity kernel calculations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


``UNDO_ATTENUATION_AND_OR_PML``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Undo attenuation and/or PMLs for sensitivity kernel calculations or forward runs with SAVE_FORWARD.

:Type: ``logical``

.. code-block::
    :caption: Example

    UNDO_ATTENUATION_AND_OR_PML = .false.

``NT_DUMP_ATTENUATION``
~~~~~~~~~~~~~~~~~~~~~~~

Interval (in time steps) for dumping restart files when undoing attenuation/PMLs.

:Type: ``integer``

.. code-block::
    :caption: Example

    NT_DUMP_ATTENUATION = 200


Visualization
+++++++++++++


``CREATE_SHAKEMAP``
~~~~~~~~~~~~~~~~~~~

Create a shakemap of ground motion.

:Type: ``logical``

.. code-block::
    :caption: Example

    CREATE_SHAKEMAP = .false.

``MOVIE_SURFACE``
~~~~~~~~~~~~~~~~~

Create a movie of the top surface.

:Type: ``logical``

.. code-block::
    :caption: Example

    MOVIE_SURFACE = .false.

``MOVIE_TYPE``
~~~~~~~~~~~~~~

Type of movie to create.

- 1 = top surface
- 2 = all external faces

:Type: ``integer``

.. code-block::
    :caption: Example

    MOVIE_TYPE = 1

``MOVIE_VOLUME``
~~~~~~~~~~~~~~~~

Create a movie of the volume.

:Type: ``logical``

.. code-block::
    :caption: Example

    MOVIE_VOLUME = .false.

``SAVE_DISPLACEMENT``
~~~~~~~~~~~~~~~~~~~~~

Save displacement field for visualization.

:Type: ``logical``

.. code-block::
    :caption: Example

    SAVE_DISPLACEMENT = .false.

``MOVIE_VOLUME_STRESS``
~~~~~~~~~~~~~~~~~~~~~~~

Create a movie of the volume stress.

:Type: ``logical``

.. code-block::
    :caption: Example

    MOVIE_VOLUME_STRESS = .false.

``USE_HIGHRES_FOR_MOVIES``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use high resolution for movies.

:Type: ``logical``

.. code-block::
    :caption: Example

    USE_HIGHRES_FOR_MOVIES = .false.

``NTSTEP_BETWEEN_FRAMES``
~~~~~~~~~~~~~~~~~~~~~~~~~

Number of time steps between movie frames.

:Type: ``integer``

.. code-block::
    :caption: Example

    NTSTEP_BETWEEN_FRAMES = 200

``HDUR_MOVIE``
~~~~~~~~~~~~~~

Half duration for movie source time function.

:Type: ``real``

.. code-block::
    :caption: Example

    HDUR_MOVIE = 0.0

``SAVE_MESH_FILES``
~~~~~~~~~~~~~~~~~~~

Save AVS or OpenDX mesh files for mesh checking.

:Type: ``logical``

.. code-block::
    :caption: Example

    SAVE_MESH_FILES = .false.

``LOCAL_PATH``
~~~~~~~~~~~~~~

Path to store the local database file on each node.

:Type: ``string``

.. code-block::
    :caption: Example

    LOCAL_PATH = /path/to/OUTPUT_FILES/DATABASES_MPI

``NTSTEP_BETWEEN_OUTPUT_INFO``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interval at which to output time step info and max norm of displacement.

:Type: ``integer``

.. code-block::
    :caption: Example

    NTSTEP_BETWEEN_OUTPUT_INFO = 500

Sources
+++++++


``USE_SOURCES_RECEIVERS_Z``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use true Z coordinates for sources and receivers instead of depth.

:Type: ``logical``

.. code-block::
    :caption: Example

    USE_SOURCES_RECEIVERS_Z = .false.

``USE_FORCE_POINT_SOURCE``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a (tilted) force point source instead of a CMTSOLUTION moment-tensor source.

:Type: ``logical``

.. code-block::
    :caption: Example

    USE_FORCE_POINT_SOURCE = .false.

``SOURCE_FILENAME``
~~~~~~~~~~~~~~~~~~~

Path to the source file (CMTSOLUTION or FORCESOLUTION).

:Type: ``string``

.. code-block::
    :caption: Example

    SOURCE_FILENAME = /path/to/CMTSOLUTION

``USE_RICKER_TIME_FUNCTION``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a Ricker source time function.

:Type: ``logical``

.. code-block::
    :caption: Example

    USE_RICKER_TIME_FUNCTION = .false.

``USE_EXTERNAL_SOURCE_FILE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use an external source time function file.

:Type: ``logical``

.. code-block::
    :caption: Example

    USE_EXTERNAL_SOURCE_FILE = .false.

``PRINT_SOURCE_TIME_FUNCTION``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Print the source time function.

:Type: ``logical``

.. code-block::
    :caption: Example

    PRINT_SOURCE_TIME_FUNCTION = .false.

``USE_SOURCE_ENCODING``
~~~~~~~~~~~~~~~~~~~~~~~

Use source encoding (for acoustic simulations only).

:Type: ``logical``

.. code-block::
    :caption: Example

    USE_SOURCE_ENCODING = .false.

Seismograms
+++++++++++

``NTSTEP_BETWEEN_OUTPUT_SEISMOS``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interval (in time steps) for writing seismograms.

:Type: ``integer``

.. code-block::
    :caption: Example

    NTSTEP_BETWEEN_OUTPUT_SEISMOS = 10000

``NTSTEP_BETWEEN_OUTPUT_SAMPLE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Down-sampling factor for output seismograms.

:Type: ``integer``

.. code-block::
    :caption: Example

    NTSTEP_BETWEEN_OUTPUT_SAMPLE = 1

``SAVE_SEISMOGRAMS_DISPLACEMENT``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save displacement seismograms.

:Type: ``logical``

.. code-block::
    :caption: Example

    SAVE_SEISMOGRAMS_DISPLACEMENT = .true.

``SAVE_SEISMOGRAMS_VELOCITY``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save velocity seismograms.

:Type: ``logical``

.. code-block::
    :caption: Example

    SAVE_SEISMOGRAMS_VELOCITY = .false.

``SAVE_SEISMOGRAMS_ACCELERATION``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save acceleration seismograms.

:Type: ``logical``

.. code-block::
    :caption: Example

    SAVE_SEISMOGRAMS_ACCELERATION = .false.

``SAVE_SEISMOGRAMS_PRESSURE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save pressure seismograms (acoustic elements only).

:Type: ``logical``

.. code-block::
    :caption: Example

    SAVE_SEISMOGRAMS_PRESSURE = .false.

``SAVE_SEISMOGRAMS_STRAIN``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save strain seismograms.

:Type: ``logical``

.. code-block::
    :caption: Example

    SAVE_SEISMOGRAMS_STRAIN = .false.

``SAVE_SEISMOGRAMS_IN_ADJOINT_RUN``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save seismograms during adjoint runs.

:Type: ``logical``

.. code-block::
    :caption: Example

    SAVE_SEISMOGRAMS_IN_ADJOINT_RUN = .true.

``USE_BINARY_FOR_SEISMOGRAMS``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save seismograms in binary format.

:Type: ``logical``

.. code-block::
    :caption: Example

    USE_BINARY_FOR_SEISMOGRAMS = .false.

``SU_FORMAT``
~~~~~~~~~~~~~

Output seismograms in Seismic Unix format.

:Type: ``logical``

.. code-block::
    :caption: Example

    SU_FORMAT = .false.

``ASDF_FORMAT``
~~~~~~~~~~~~~~~

Output seismograms in ASDF format (requires asdf-library).

:Type: ``logical``

.. code-block::
    :caption: Example

    ASDF_FORMAT = .false.

``HDF5_FORMAT``
~~~~~~~~~~~~~~~

Output seismograms in HDF5 format (requires hdf5-library and WRITE_SEISMOGRAMS_BY_MAIN).

:Type: ``logical``

.. code-block::
    :caption: Example

    HDF5_FORMAT = .false.

``WRITE_SEISMOGRAMS_BY_MAIN``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Main process writes all seismograms (otherwise all processes write in parallel).

:Type: ``logical``

.. code-block::
    :caption: Example

    WRITE_SEISMOGRAMS_BY_MAIN = .false.

``SAVE_ALL_SEISMOS_IN_ONE_FILE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save all seismograms in one large file instead of one per seismogram.

:Type: ``logical``

.. code-block::
    :caption: Example

    SAVE_ALL_SEISMOS_IN_ONE_FILE = .false.

``USE_TRICK_FOR_BETTER_PRESSURE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a trick to increase accuracy of pressure seismograms in fluid elements.

:Type: ``logical``

.. code-block::
    :caption: Example

    USE_TRICK_FOR_BETTER_PRESSURE = .false.

Fault simulations
+++++++++++++++++


``HAS_FINITE_FAULT_SOURCE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulation includes a finite fault source.

:Type: ``logical``

.. code-block::
    :caption: Example

    HAS_FINITE_FAULT_SOURCE = .false.

``FAULT_PAR_FILE``
~~~~~~~~~~~~~~~~~~

File containing parameters of the fault.

:Type: ``string``

.. code-block::
    :caption: Example

    FAULT_PAR_FILE = dummy.txt

``FAULT_STATIONS``
~~~~~~~~~~~~~~~~~~

File containing stations in the fault plane.

:Type: ``string``

.. code-block::
    :caption: Example

    FAULT_STATIONS = dummy.txt

``STRESS_FRICTION_FILE``
~~~~~~~~~~~~~~~~~~~~~~~~

File for heterogeneous stresses and friction for linear slip weakening.

:Type: ``string``

.. code-block::
    :caption: Example

    STRESS_FRICTION_FILE = dummy.txt

``RSF_HETE_FILE``
~~~~~~~~~~~~~~~~~

File for heterogeneous stresses and friction for rate and state friction.

:Type: ``string``

.. code-block::
    :caption: Example

    RSF_HETE_FILE = dummy.txt


Energy calculation
++++++++++++++++++


``OUTPUT_ENERGY``
~~~~~~~~~~~~~~~~~

Output energy curves for monitoring (expensive, usually off).

:Type: ``logical``

.. code-block::
    :caption: Example

    OUTPUT_ENERGY = .false.

``NTSTEP_BETWEEN_OUTPUT_ENERGY``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interval (in time steps) for computing energy.

:Type: ``integer``

.. code-block::
    :caption: Example

    NTSTEP_BETWEEN_OUTPUT_ENERGY = 10


Adjoint kernel outputs
++++++++++++++++++++++

``NTSTEP_BETWEEN_READ_ADJSRC``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interval (in time steps) for reading adjoint traces (0 = read all at start).

:Type: ``integer``

.. code-block::
    :caption: Example

    NTSTEP_BETWEEN_READ_ADJSRC = 0

``READ_ADJSRC_ASDF``
~~~~~~~~~~~~~~~~~~~~

Read adjoint sources using ASDF format.

:Type: ``logical``

.. code-block::
    :caption: Example

    READ_ADJSRC_ASDF = .false.

``ANISOTROPIC_KL``
~~~~~~~~~~~~~~~~~~

Compute anisotropic kernels (21 Cij in geographical coordinates).

:Type: ``logical``

.. code-block::
    :caption: Example

    ANISOTROPIC_KL = .false.

``SAVE_TRANSVERSE_KL``
~~~~~~~~~~~~~~~~~~~~~~

Compute and save transverse isotropic kernels.

:Type: ``logical``

.. code-block::
    :caption: Example

    SAVE_TRANSVERSE_KL = .false.

``ANISOTROPIC_VELOCITY_KL``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute anisotropic kernels for velocity observable.

:Type: ``logical``

.. code-block::
    :caption: Example

    ANISOTROPIC_VELOCITY_KL = .false.

``APPROXIMATE_HESS_KL``
~~~~~~~~~~~~~~~~~~~~~~~

Output approximate Hessian for preconditioning.

:Type: ``logical``

.. code-block::
    :caption: Example

    APPROXIMATE_HESS_KL = .false.

``SAVE_MOHO_MESH``
~~~~~~~~~~~~~~~~~~

Save Moho mesh and compute Moho boundary kernels.

:Type: ``logical``

.. code-block::
    :caption: Example

    SAVE_MOHO_MESH = .false.


Coupling with an injection technique (DSM, AxiSEM, or FK)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++


``COUPLE_WITH_INJECTION_TECHNIQUE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable coupling with an injection technique (DSM, AxiSEM, or FK).

:Type: ``logical``

.. code-block::
    :caption: Example

    COUPLE_WITH_INJECTION_TECHNIQUE = .false.

``INJECTION_TECHNIQUE_TYPE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Type of injection technique.

- 1 = DSM
- 2 = AxiSEM
- 3 = FK

:Type: ``integer``

.. code-block::
    :caption: Example

    INJECTION_TECHNIQUE_TYPE = 3

``MESH_A_CHUNK_OF_THE_EARTH``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mesh is a chunk of the Earth (for coupling).

:Type: ``logical``

.. code-block::
    :caption: Example

    MESH_A_CHUNK_OF_THE_EARTH = .false.

``TRACTION_PATH``
~~~~~~~~~~~~~~~~~

Path to traction files for coupling.

:Type: ``string``

.. code-block::
    :caption: Example

    TRACTION_PATH = /path/to/AxiSEM_tractions/3/

``FKMODEL_FILE``
~~~~~~~~~~~~~~~~

File for FK model.

:Type: ``string``

.. code-block::
    :caption: Example

    FKMODEL_FILE = FKmodel

``RECIPROCITY_AND_KH_INTEGRAL``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable reciprocity and KH integral (not yet functional).

:Type: ``logical``

.. code-block::
    :caption: Example

    RECIPROCITY_AND_KH_INTEGRAL = .false.


Prescribed wavefield discontinuity on an interface
++++++++++++++++++++++++++++++++++++++++++++++++++


``IS_WAVEFIELD_DISCONTINUITY``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prescribe a wavefield discontinuity on an interface.

:Type: ``logical``

.. code-block::
    :caption: Example

    IS_WAVEFIELD_DISCONTINUITY = .false.

Run modes
+++++++++


``NUMBER_OF_SIMULTANEOUS_RUNS``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of simultaneous runs to perform in parallel.

:Type: ``integer``

.. code-block::
    :caption: Example

    NUMBER_OF_SIMULTANEOUS_RUNS = 1

``BROADCAST_SAME_MESH_AND_MODEL``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Broadcast mesh and model files to all simultaneous runs.

:Type: ``logical``

.. code-block::
    :caption: Example

    BROADCAST_SAME_MESH_AND_MODEL = .true.


----


``GPU_MODE``
~~~~~~~~~~~~

Enable GPU mode.

:Type: ``logical``

.. code-block::
    :caption: Example

    GPU_MODE = .false.

``ADIOS_ENABLED``
~~~~~~~~~~~~~~~~~

Enable ADIOS for I/O.

:Type: ``logical``

.. code-block::
    :caption: Example

    ADIOS_ENABLED = .false.

``ADIOS_FOR_DATABASES``
~~~~~~~~~~~~~~~~~~~~~~~

Use ADIOS for database files.

:Type: ``logical``

.. code-block::
    :caption: Example

    ADIOS_FOR_DATABASES = .false.

``ADIOS_FOR_MESH``
~~~~~~~~~~~~~~~~~~

Use ADIOS for mesh files.

:Type: ``logical``

.. code-block::
    :caption: Example

    ADIOS_FOR_MESH = .false.

``ADIOS_FOR_FORWARD_ARRAYS``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ADIOS for forward arrays.

:Type: ``logical``

.. code-block::
    :caption: Example

    ADIOS_FOR_FORWARD_ARRAYS = .false.

``ADIOS_FOR_KERNELS``
~~~~~~~~~~~~~~~~~~~~~

Use ADIOS for kernel files.

:Type: ``logical``

.. code-block::
    :caption: Example

    ADIOS_FOR_KERNELS = .false.

``ADIOS_FOR_UNDO_ATTENUATION``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ADIOS for undo attenuation files.

:Type: ``logical``

.. code-block::
    :caption: Example

    ADIOS_FOR_UNDO_ATTENUATION = .false.

``HDF5_ENABLED``
~~~~~~~~~~~~~~~~

Enable HDF5 for I/O.

:Type: ``logical``

.. code-block::
    :caption: Example

    HDF5_ENABLED = .false.

``HDF5_FOR_MOVIES``
~~~~~~~~~~~~~~~~~~~

Use HDF5 for movie files.

:Type: ``logical``

.. code-block::
    :caption: Example

    HDF5_FOR_MOVIES = .false.

``HDF5_IO_NODES``
~~~~~~~~~~~~~~~~~

Number of IO dedicated processes for HDF5 IO server.

:Type: ``integer``

.. code-block::
    :caption: Example

    HDF5_IO_NODES = 0
