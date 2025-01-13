Wave propagration through solid-solid interface
===============================================

This `example
<https://github.com/PrincetonUniversity/SPECFEMPP/tree/main/examples/solid-solid-interface>`_
simulates a solid-solid example with horizontal interface from the computational
seismology class at Princeton University. Setting up the workspace


Setting up the workspace
------------------------

Let's start by creating a workspace from where we can run this example.

.. code-block:: bash

    mkdir -p ~/specfempp-examples/solid-solid-interface
    cd ~/specfempp-examples/solid-solid-interface

We also need to check that the SPECFEM++ build directory is added to the ``PATH``.

.. code:: bash

    which specfem2d

If the above command returns a path to the ``specfem2d`` executable, then the
build directory is added to the ``PATH``. If not, you need to add the build
directory to the ``PATH`` using the following command.

.. code:: bash

    export PATH=$PATH:<PATH TO SPECFEM++ BUILD DIRECTORY/bin>

.. note::

    Make sure to replace ``<PATH TO SPECFEM++ BUILD DIRECTORY/bin>`` with the
    actual path to the SPECFEM++ build directory on your system.

Now let's create the necessary directories to store the input files and output
artifacts.

.. code:: bash

    mkdir -p OUTPUT_FILES
    mkdir -p OUTPUT_FILES/seismograms

    touch specfem_config.yaml
    touch sources.yaml
    touch topography.dat
    touch Par_File


Meshing the domain
------------------

We first start by generating a mesh for our simulation domain using ``xmeshfem2D``. To do this, we first define our simulation domain and the meshing parmeters in a parameter file.

Parameter file
~~~~~~~~~~~~~

.. code-block:: bash
    :caption: Par_File

    #-----------------------------------------------------------
    #
    # Simulation input parameters
    #
    #-----------------------------------------------------------

    # title of job
    title                           = Flat solid/solid interface

    # parameters concerning partitioning
    NPROC                           = 1              # number of processes

    # Output folder to store mesh related files
    OUTPUT_FILES                   = OUTPUT_FILES

    #-----------------------------------------------------------
    #
    # Mesh
    #
    #-----------------------------------------------------------

    # Partitioning algorithm for decompose_mesh
    PARTITIONING_TYPE               = 3              # SCOTCH = 3, ascending order (very bad idea) = 1

    # number of control nodes per element (4 or 9)
    NGNOD                           = 9

    # location to store the mesh
    database_filename               = OUTPUT_FILES/database.bin

    #-----------------------------------------------------------
    #
    # Receivers
    #
    #-----------------------------------------------------------

    # use an existing STATION file found in ./DATA or create a new one from the receiver positions below in this Par_file
    use_existing_STATIONS           = .false.

    # number of receiver sets (i.e. number of receiver lines to create below)
    nreceiversets                   = 1

    # orientation
    anglerec                        = 0.d0           # angle to rotate components at receivers
    rec_normal_to_surface           = .false.        # base anglerec normal to surface (external mesh and curve file needed)

    # first receiver set (repeat these 6 lines and adjust nreceiversets accordingly)
    nrec                            = 1             # number of receivers
    xdeb                            = 150000.d0     # first receiver x in meters
    zdeb                            = 80000.d0      # first receiver z in meters
    xfin                            = 150000.d0     # last receiver x in meters (ignored if only one receiver)
    zfin                            = 3480000.d0    # last receiver z in meters (ignored if only one receiver)
    record_at_surface_same_vertical = .false.       # receivers inside the medium or at the surface

    # filename to store stations file
    stations_filename              = OUTPUT_FILES/STATIONS

    #-----------------------------------------------------------
    #
    # Velocity and density models
    #
    #-----------------------------------------------------------

    # number of model materials
    nbmodels                        = 2
    # available material types (see user manual for more information)
    #   acoustic:    model_number 1 rho Vp 0  0 0 QKappa Qmu 0 0 0 0 0 0
    #   elastic:     model_number 1 rho Vp Vs 0 0 QKappa Qmu 0 0 0 0 0 0
    #   anistoropic: model_number 2 rho c11 c13 c15 c33 c35 c55 c12 c23 c25 0 0 0
    #   poroelastic: model_number 3 rhos rhof phi c kxx kxz kzz Ks Kf Kfr etaf mufr Qmu
    #   tomo:        model_number -1 0 9999 9999 A 0 0 9999 9999 0 0 0 0 0
    #
    # The problem values are as follows:
    # top:    rho = 2.60 * 10^3 kg/m3, kappa= 5.2 * 10^10 Pa, mu =2.66 * 10^10 Pa
    # bottom: rho = 3.38 * 10^3 kg/m3, kappa= 1.3 * 10^11 Pa, mu = 6.8 * 10^10 Pa
    # After conversion to VP/VS we have following model values.
    1 1 3380.d0 8079.98d0 4485.35d0 0 0 9999 9999 0 0 0 0 0 0
    2 1 2600.d0 5859.90d0 3199.40d0 0 0 9999 9999 0 0 0 0 0 0

    # external tomography file
    TOMOGRAPHY_FILE                 = ./DATA/tomo_file.xyz

    # use an external mesh created by an external meshing tool or use the internal mesher
    read_external_mesh              = .false.

    #-----------------------------------------------------------
    #
    # PARAMETERS FOR EXTERNAL MESHING
    #
    #-----------------------------------------------------------

    # data concerning mesh, when generated using third-party app (more info in README)
    # (see also absorbing_conditions above)
    mesh_file                       = ./DATA/Mesh_canyon/canyon_mesh_file   # file containing the mesh
    nodes_coords_file               = ./DATA/Mesh_canyon/canyon_nodes_coords_file   # file containing the nodes coordinates
    materials_file                  = ./DATA/Mesh_canyon/canyon_materials_file   # file containing the material number for each element
    free_surface_file               = ./DATA/Mesh_canyon/canyon_free_surface_file   # file containing the free surface
    axial_elements_file             = ./DATA/axial_elements_file   # file containing the axial elements if AXISYM is true
    absorbing_surface_file          = ./DATA/Mesh_canyon/canyon_absorbing_surface_file   # file containing the absorbing surface
    acoustic_forcing_surface_file   = ./DATA/MSH/Surf_acforcing_Bottom_enforcing_mesh   # file containing the acoustic forcing surface
    absorbing_cpml_file             = ./DATA/absorbing_cpml_file   # file containing the CPML element numbers
    tangential_detection_curve_file = ./DATA/courbe_eros_nodes  # file containing the curve delimiting the velocity model

    #-----------------------------------------------------------
    #
    # PARAMETERS FOR INTERNAL MESHING
    #
    #-----------------------------------------------------------

    # file containing interfaces for internal mesh
    interfacesfile                  = topography.dat

    # geometry of the model (origin lower-left corner = 0,0) and mesh description
    xmin                            = 0.d0          # abscissa of left side of the model
    xmax                            = 200000.d0     # abscissa of right side of the model
    nx                              = 188           # number of elements along X

    STACEY_ABSORBING_CONDITIONS    = .true.

    # absorbing boundary parameters (see absorbing_conditions above)
    absorbbottom                    = .true.
    absorbright                     = .true.
    absorbtop                       = .false.
    absorbleft                      = .true.

    # define the different regions of the model in the (nx,nz) spectral-element mesh
    nbregions                       = 2              # then set below the different regions and model number for each region
    1 188 1   38 1
    1 188 39  76 2

    #-----------------------------------------------------------
    #
    # Display parameters
    #
    #-----------------------------------------------------------

    # meshing output
    output_grid_Gnuplot             = .false.        # generate a GNUPLOT file containing the grid, and a script to plot it
    output_grid_ASCII               = .false.        # dump the grid in an ASCII text file consisting of a set of X,Y,Z points or not


- We define the elastic velocity model layers in the `Velocity and
  density models` section of the parameter file. - Firstly, ``nbmodels`` defines
  the number of material systems in the simulation domain. - We then define the
  velocity model for each material system using the following format:
  ``model_number rho Vp Vs 0 0 QKappa Qmu 0 0 0 0 0 0``.

  As you can see, we added two material systems in the simulation domain. With
  the properties used from the model where the faster material is model number
  1 and the slower material is model number 2.

- We define stacey absorbing boundary conditions on all the edges of the domain
  using the ``STACEY_ABSORBING_BOUNDARY``, ``absorbbottom``, ``absorbright``,
  ``absorbtop`` and ``absorbleft`` parameters.

Defining the topography of the domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We define the topography of the domain using the following topography file

.. code-block:: bash
    :caption: topography_file.dat

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
    ## The original 2000 Geophysics paper used nz = 90 but NGLLZ = 6
    ## here I rescale it to nz = 108 and NGLLZ = 5 because nowadays we almost always use NGLLZ = 5
     54
    #
    # layer number 2 (top layer)
    #
     54

Running ``xmeshfem2D``
~~~~~~~~~~~~~~~~~~~~~~

To execute the mesher run

.. code:: bash

    xmeshfem2D -p Par_File

.. note::

    Make sure either your are in the build directory of SPECFEM2D kokkos or the build directory is added to your ``PATH``.

Note the path of the database file and a stations file generated after successfully running the mesher.

Defining the source
~~~~~~~~~~~~~~~~~~~

We define the source location and the source time function in the source file.

.. code-block:: yaml
    :caption: single_source.yaml

    number-of-sources: 1
    sources:
      - force:
          x : 1575.0
          z : 2900.0
          source_surf: false
          angle : 0.0
          vx : 0.0
          vz : 0.0
          Ricker:
            factor: 1e9
            tshift: 0.0
            f0: 10.0

Running the simulation
----------------------

To run the solver, we first need to define a configuration file ``specfem_config.yaml``.

.. code-block:: yaml
    :caption: specfem_config.yaml

    parameters:

      header:
        ## Header information is used for logging. It is good practice to give your simulations explicit names
        title: Heterogeneous acoustic-elastic medium with 1 acoustic-elastic interface (orientation horizontal)  # name for your simulation
        # A detailed description for your simulation
        description: |
          Material systems : Elastic domain (1), Acoustic domain (1)
          Interfaces : Acoustic-elastic interface (1) (orientation horizontal with acoustic domain on top)
          Sources : Force source (1)
          Boundary conditions : Neumann BCs on all edges
          Debugging comments: This tests checks coupling acoustic-elastic interface implementation.
                              The orientation of the interface is horizontal with acoustic domain on top.

      simulation-setup:
        ## quadrature setup
        quadrature:
          quadrature-type: GLL4

        ## Solver setup
        solver:
          time-marching:
            type-of-simulation: forward
            time-scheme:
              type: Newmark
              dt: 0.85e-3
              nstep: 600

        simulation-mode:
          forward:
            writer:
              seismogram:
                format: ascii
                directory: OUTPUT_FILES/seismograms

      receivers:
        stations-file: OUTPUT_FILES/STATIONS
        angle: 0.0
        seismogram-type:
          - displacement
        nstep_between_samples: 1

      ## Runtime setup
      run-setup:
        number-of-processors: 1
        number-of-runs: 1

      ## databases
      databases:
        mesh-database: OUTPUT_FILES/database.bin
        source-file: single_source.yaml

With the configuration file in place, we can run the solver using the following command

.. code:: bash

    specfem2d -p specfem_config.yaml
