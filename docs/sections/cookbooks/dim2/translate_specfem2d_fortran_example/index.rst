Migrating from SPECFEM2D Fortran to SPECFEM++
=============================================

This tutorial demonstrates how to migrate a SPECFEM2D Fortran example to the
SPECFEM++ format using the anisotropic zinc crystal example. The original
SPECFEM2D uses a single ``Par_file`` for both meshing and simulation, while
SPECFEM++ separates these concerns into multiple configuration files.


.. note::

    Not all physics are yet supported in SPECFEM++. This example focuses on
    currently working changes (the list may not be complete).

    **NOT SUPPORTED**:
      * General physics
          * Attenuation
          * Axisymmetric simulations
          * Noise tomography
      * Timeschemes
          * Classical 4-th order Runge-Kutta
          * LDDRK
      * Sources
          * Moving sources
          * Bielak conditions
          * Acoustic forcing
      * Seismogram Output formats
          * Seismic Unix
          * Binary
      * Boundary Conditions
          * PML
      * Architecture
          * MPI
          * Simultaneous runs


Original SPECFEM2D Structure
----------------------------

The original example located at
``/specfem2d/EXAMPLES/applications/anisotropy/anisotropic_zinc_crystal/``
in the SPECFEM2D repository contains:

.. code-block:: text

    anisotropic_zinc_crystal/
    ├── DATA/
    │   ├── Par_file              # Single configuration file
    │   ├── SOURCE               # Fortran-style source definition
    │   └── topoaniso.dat        # Topography file
    └── run_this_example.sh      # Execution script

You download :download:`Anisotropic Crystal ZIP </_static/anisotropic_zinc_crystal.zip>` to
access the original example.

The specfem fortran execution patter uses a single ``Par_file`` for both meshing
and simulation:

.. code-block:: bash

    ./bin/xmeshfem2D    # Uses Par_file for meshing
    ./bin/xspecfem2D    # Uses Par_file for simulation

SPECFEM++ Structure
-------------------

The SPECFEM++ version separates the configuration into multiple files for
clarity and modularity.

.. code-block:: text

    anisotropic-crystal/
    ├── Par_file                 # Updated:   Mesh parameters [and receiver parameters]
    ├── specfem_config.yaml      # New:       Solver configuration
    ├── source.yaml              # New:       Source parameters
    └── topoaniso.dat            # Unchanged: Topography file

Execution uses a two-step workflow:

.. code-block:: bash

    xmeshfem2D -p Par_file             # Meshing only [and receiver generation]
    specfem2d -p specfem_config.yaml   # Simulation with YAML config

The reason we want to separate the configuration files is to improve clarity and
modularity, as well as distinguishing between fortran code and new C++
implementation.

Migration Steps
---------------

These steps will follow how we approach updates to the original
``Par_file`` and translation to the new SPECFEM++ structure.

Step 1: Split the Par_file
~~~~~~~~~~~~~~~~~~~~~~~~~~

The original ``Par_file`` contains both meshing and simulation parameters. In
SPECFEM++, we separate these:

**Parameters to keep from the Par_file (meshing parameters):**

.. literalinclude:: Par_file
    :caption: Par_file
    :language: bash

**Additional Par_file parameters:**

Add these to the Par_file:

.. code-block:: bash

    OUTPUT_FILES = OUTPUT_FILES
    database_filename = ./OUTPUT_FILES/database.bin
    stations_filename = ./OUTPUT_FILES/STATIONS

**Add new specfem_config.yaml (simulation parameters):**

The solver parameters are converted from Fortran format to YAML:

.. literalinclude:: specfem_config.yaml
    :caption: `specfem_config.yaml` lines with comments are adopting parameters
              from the original `Par_file`
    :language: yaml

Here, it is easiest to copy the box, where we already added the necessary
parameters.

The `writer.display` is too different to list all the different changes, but
we believe the parameters are fairly self-explanatory. See the
:ref:`parameter_documentation` for additional details.

Step 2: Convert SOURCE to source.yaml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original ``SOURCE`` file uses Fortran-style parameters:

.. code-block:: fortran

    source_surf = .false.
    xs = 0.165
    zs = 0.165
    source_type = 1
    time_function_type = 1
    f0 = 170000.
    tshift = 0.0
    anglesource = 0.0
    factor = 1.d10
    vx = 0.0
    vz = 0.0

This converts to YAML format in ``source.yaml``:

.. code-block:: yaml

    number-of-sources: 1
    sources:
      - force:
          x : 0.165
          z : 0.165
          source_surf: false
          angle : 0.0
          vx : 0.0
          vz : 0.0
          Ricker:
            factor: 1e10
            tshift: 0.0
            f0: 170000.0

.. note::

    Instead of numbers we use names for the different source time functions.
    The Ricker wavelet is used here, but you can define other types as needed.
    Some of the supported time functions include:
    - Ricker
    - Gaussian
    - dGaussian (first derivative of Gaussian)
    - external
    Please refer to the parameters documentation for more details. See
    :ref:`source_description` for more information on the source parameters.

Running the example using SPECFEM++
-----------------------------------

.. code-block:: bash

    # Create output directories as specified in Par_file and specfem_config.yaml
    mkdir -p OUTPUT_FILES/seismograms OUTPUT_FILES/display

    # Separate configuration files
    xmeshfem2D -p Par_file
    specfem2d -p specfem_config.yaml

Summary
-------

This migration pattern applies to any SPECFEM2D example - identify meshing vs.
solver parameters, convert source definitions to YAML, and structure the
configuration files according to their specific roles in the simulation
workflow.


Benefits of the New Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Separation of Concerns:**
- Meshing parameters stay in ``Par_file``
- Solver parameters moved to ``specfem_config.yaml``

**Improved Readability:**
- YAML format is more human-readable
- Hierarchical structure reflects parameter relationships
- Clear separation between different simulation aspects



Parameter Mapping Reference
---------------------------

Here, we list a non-exhaustive mapping of parameters from the original
anisotropic zinc crystal example to the new SPECFEM++ format. This is useful for
understanding how to translate the parameters from the Fortran
``Par_file`` and ``SOURCE`` file to the new SPECFEM++ configuration files.

Par_file Parameters
~~~~~~~~~~~~~~~~~~~

Some parameters are newly added or modified in SPECFEM++:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - SPECFEM2D Par_file
     - SPECFEM++ Par_file
   * - ``title = Test of anisotropic zinc crystal``
     - ``title = Anisotropic Crystal``
   * - (implicit output directory)
     - ``OUTPUT_FILES = OUTPUT_FILES``
   * - (implicit stations file)
     - ``stations_filename = ./OUTPUT_FILES/STATIONS``

Solver Parameters
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - SPECFEM2D Par_file
     - SPECFEM++ specfem_config.yaml
   * - ``DT = 55.e-9``
     - ``dt: 55.e-9``
   * - ``NSTEP = 1500``
     - ``nstep: 1500``
   * - ``time_stepping_scheme = 1``
     - ``type: Newmark``
   * - ``seismotype = 1``
     - ``seismogram-type: [displacement]``
   * - ``save_ASCII_seismograms = .true.``
     - ``format: ascii``
   * - ``NTSTEP_BETWEEN_OUTPUT_IMAGES = 100``
     - ``time-interval: 100``
   * - ``output_color_image = .true.``
     - ``format: PNG``
   * - ``imagetype_JPEG = 2``
     - ``field: displacement``

Source Parameters
~~~~~~~~~~~~~~~~~

Only listing the parameters that are different from the original Fortran
``SOURCE`` file:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - SPECFEM2D SOURCE
     - SPECFEM++ source.yaml
   * - ``xs = 0.165``
     - ``x: 0.165``
   * - ``zs = 0.165``
     - ``z: 0.165``
   * - ``source_surf = .false.``
     - ``source_surf: false``
   * - ``anglesource = 0.0``
     - ``angle: 0.0``
   * - ``time_function_type = 1``
     - ``Ricker:`` (section)
