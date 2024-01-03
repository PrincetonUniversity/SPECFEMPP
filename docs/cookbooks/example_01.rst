.. homogeneous_example::

Wave propagration through homogeneous media
===========================================

In this example we simulate wave propagation through a 2-dimensional homogeneous medium.

Generating a mesh
-----------------

To generate the mesh for the homogeneous media we need a parameter file, ``Par_File``, a topography file, `topography_file.dat`, and the mesher executible, ``xmeshfem2D``, which should have been compiled during the installation process.

.. note::
  Currently, we still use a mesher that was developed for the original `SPECFEM2D <https://specfem2d.readthedocs.io/en/latest/03_mesh_generation/>`_ code. More details on the meshing process can be found `here <https://specfem2d.readthedocs.io/en/latest/03_mesh_generation/>`_.

We first define the meshing parameters in a Parameter file.

Parameter File
~~~~~~~~~~~~~~~~

.. code-block:: bash
    :caption: PAR_FILE

    #-----------------------------------------------------------
    #
    # Simulation input parameters
    #
    #-----------------------------------------------------------

    # title of job
    title                           = Elastic Simulation with point source

    # parameters concerning partitioning
    NPROC                           = 1              # number of processes

    # Output folder to store mesh related files
    OUTPUT_FILES                   = <Location to store output artifacts>


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
    database_filename               = <Output file to store the mesh generated>

    #-----------------------------------------------------------
    #
    # Receivers
    #
    #-----------------------------------------------------------

    # use an existing STATION file found in ./DATA or create a new one from the receiver positions below in this Par_file
    use_existing_STATIONS           = .false.

    # number of receiver sets (i.e. number of receiver lines to create below)
    nreceiversets                   = 2

    # orientation
    anglerec                        = 0.d0           # angle to rotate components at receivers
    rec_normal_to_surface           = .false.        # base anglerec normal to surface (external mesh and curve file needed)

    # first receiver set (repeat these 6 lines and adjust nreceiversets accordingly)
    nrec                            = 3             # number of receivers
    xdeb                            = 2200.           # first receiver x in meters
    zdeb                            = 2200.          # first receiver z in meters
    xfin                            = 2800.          # last receiver x in meters (ignored if only one receiver)
    zfin                            = 2200.          # last receiver z in meters (ignored if only one receiver)
    record_at_surface_same_vertical = .true.         # receivers inside the medium or at the surface (z values are ignored if this is set to true, they are replaced with the topography height)

    # second receiver set
    nrec                            = 3             # number of receivers
    xdeb                            = 2500.          # first receiver x in meters
    zdeb                            = 2500.          # first receiver z in meters
    xfin                            = 2500.          # last receiver x in meters (ignored if only one receiver)
    zfin                            = 1900.             # last receiver z in meters (ignored if only one receiver)
    record_at_surface_same_vertical = .false.        # receivers inside the medium or at the surface (z values are ignored if this is set to true, they are replaced with the topography height)


    # filename to store stations file
    stations_filename              = <Location to stations file>

    #-----------------------------------------------------------
    #
    # Velocity and density models
    #
    #-----------------------------------------------------------

    # number of model materials
    nbmodels                        = 1
    # available material types (see user manual for more information)
    #   acoustic:              model_number 1 rho Vp 0  0 0 QKappa 9999 0 0 0 0 0 0 (for QKappa use 9999 to ignore it)
    #   elastic:               model_number 1 rho Vp Vs 0 0 QKappa Qmu  0 0 0 0 0 0 (for QKappa and Qmu use 9999 to ignore them)
    #   anisotropic:           model_number 2 rho c11 c13 c15 c33 c35 c55 c12 c23 c25   0 QKappa Qmu
    #   anisotropic in AXISYM: model_number 2 rho c11 c13 c15 c33 c35 c55 c12 c23 c25 c22 QKappa Qmu
    #   poroelastic:           model_number 3 rhos rhof phi c kxx kxz kzz Ks Kf Kfr etaf mufr Qmu
    #   tomo:                  model_number -1 0 0 A 0 0 0 0 0 0 0 0 0 0
    #
    # note: When viscoelasticity or viscoacousticity is turned on,
    #       the Vp and Vs values that are read here are the UNRELAXED ones i.e. the values at infinite frequency
    #       unless the READ_VELOCITIES_AT_f0 parameter above is set to true, in which case they are the values at frequency f0.
    #
    #       Please also note that Qmu is always equal to Qs, but Qkappa is in general not equal to Qp.
    #       To convert one to the other see doc/Qkappa_Qmu_versus_Qp_Qs_relationship_in_2D_plane_strain.pdf and
    #       utils/attenuation/conversion_from_Qkappa_Qmu_to_Qp_Qs_from_Dahlen_Tromp_959_960.f90.
    1 1 2700.d0 3000.d0 1732.051d0 0 0 9999 9999 0 0 0 0 0 0

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
    mesh_file                       = ./DATA/mesh_file          # file containing the mesh
    nodes_coords_file               = ./DATA/nodes_coords_file  # file containing the nodes coordinates
    materials_file                  = ./DATA/materials_file     # file containing the material number for each element
    free_surface_file               = ./DATA/free_surface_file  # file containing the free surface
    axial_elements_file             = ./DATA/axial_elements_file   # file containing the axial elements if AXISYM is true
    absorbing_surface_file          = ./DATA/absorbing_surface_file   # file containing the absorbing surface
    acoustic_forcing_surface_file   = ./DATA/MSH/Surf_acforcing_Bottom_enforcing_mesh   # file containing the acoustic forcing surface
    absorbing_cpml_file             = ./DATA/absorbing_cpml_file   # file containing the CPML element numbers
    tangential_detection_curve_file = ./DATA/courbe_eros_nodes  # file containing the curve delimiting the velocity model

    #-----------------------------------------------------------
    #
    # PARAMETERS FOR INTERNAL MESHING
    #
    #-----------------------------------------------------------

    # file containing interfaces for internal mesh
    interfacesfile                  = <Location to topography file>

    # geometry of the model (origin lower-left corner = 0,0) and mesh description
    xmin                            = 0.d0           # abscissa of left side of the model
    xmax                            = 4000.d0        # abscissa of right side of the model
    nx                              = 80             # number of elements along X

    # absorbing boundary parameters (see absorbing_conditions above)
    absorbbottom                    = .false.
    absorbright                     = .false.
    absorbtop                       = .false.
    absorbleft                      = .false.

    # define the different regions of the model in the (nx,nz) spectral-element mesh
    nbregions                       = 1              # then set below the different regions and model number for each region
    # format of each line: nxmin nxmax nzmin nzmax material_number
    1 80  1 60 1

    #-----------------------------------------------------------
    #
    # DISPLAY PARAMETERS
    #
    #-----------------------------------------------------------

    # meshing output
    output_grid_Gnuplot             = .false.        # generate a GNUPLOT file containing the grid, and a script to plot it
    output_grid_ASCII               = .false.        # dump the grid in an ASCII text file consisting of a set of X,Y,Z points or not


At this point, it is worthwhile to note few key parameters within the ``PAR_FILE`` as it pertains to SPECFEM++.

- This version of SPECFEM++ does not support simulations running across multiple nodes, i.e., we have not enabled MPI. Relevant parameter value:

.. code:: bash

        NPROC   = 1

- The path to the topography file is provided using the ``interfacesfile`` parameter. Relevant values:

.. code:: bash

    interfacesfile = topography_file.dat


Topography file
~~~~~~~~~~~~~~~~~

.. code-block:: bash
    :caption: topography_file.dat
    :linenos:

    #
    # number of interfaces
    #
     2
    #
    # for each interface below, we give the number of points and then x,z for each point
    #
    #
    # interface number 1 (bottom of the mesh)
    #
     2
     0 0
     5000 0
    # interface number 2 (topography, top of the mesh)
    #
     2
        0 3000
     5000 3000
    #
    # for each layer, we give the number of spectral elements in the vertical direction
    #
    #
    # layer number 1 (bottom layer)
    #
     60

Running ``xmeshfem2D``
~~~~~~~~~~~~~~~~~~~~~~

To execute the mesher run

.. code:: bash

    ./xmeshfem2D -p <PATH TO PAR_FILE>

.. note::

    Make sure either your are in the build directory of SPECFEM++ or the build directory is added to your ``PATH``.

Note the path of the database file and :ref:`stations_file` generated after successfully running the mesher.

Defining sources
----------------

Next we define the sources using a YAML file. For full description on parameters used to define sources refer :ref:`source_description`.

.. code:: yaml
    :linenos:
    :caption: single_source.yaml

    number-of-sources: 1
    sources:
      - force:
          x : 2500.0
          z : 2500.0
          source_surf: false
          angle : 0.0
          vx : 0.0
          vz : 0.0
          Ricker:
            factor: 1e10
            tshift: 0.0
            f0: 10.0

Configuring the solver
-----------------------

Now that we have generated a mesh and defined the sources, we need to set up the solver. To do this we define another YAML file ``specfem_config.yaml``. For full description on parameters used to define sources refer :ref:`parameter_documentation`.

.. code-block:: yaml
    :linenos:
    :caption: specfem_config.yaml

    parameters:

      header:
        ## Header information is used for logging. It is good practice to give your simulations explicit names
        title: Isotropic Elastic simulation # name for your simulation
        # A detailed description for your simulation
        description: |
          Material systems : Elastic domain (1)
          Interfaces : None
          Sources : Force source (1)
          Boundary conditions : Neumann BCs on all edges

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
              dt: 1.1e-3
              nstep: 1600

      receivers:
        stations-file: <PATH TO STATIONS FILE>
        angle: 0.0
        seismogram-type:
          - velocity
        nstep_between_samples: 1

      seismogram:
        seismogram-format: ascii
        output-folder: <PATH TO DIRECTORY FOR STORING OUTPUTS>

      ## Runtime setup
      run-setup:
        number-of-processors: 1
        number-of-runs: 1

      ## databases
      databases:
        mesh-database: <PATH TO MESHFEM DATABASE FILE>
        source-file: <PATH TO SOURCES YAML FILE>

At this point lets focus on a few sections in this file:

- Define the path to the meshfem generated database file using the ``mesh-database`` parameter and the path to source description file using ``source-file`` parameter. Relevant parameter values:

.. code:: yaml

    ## databases
    databases:
      mesh-database: <PATH TO MESHFEM DATABASE FILE>
      source-file: <PATH TO SOURCES YAML FILE>

- Define the path to :ref:`stations_file` and a directory to store output. If an output directory is not specified the seismogram outputs will be stored in the current working directory. Relevant parameter values:

.. code:: yaml

    seismogram:
      stations-file: <PATH TO STATIONS FILE>
      output-folder: <PATH TO DIRECTORY FOR STORING OUTPUTS>

- It is good practice to have distinct header section for you simulation. These sections will be printed to standard output during runtime helping the you to distinguish between runs using standard strings. Relevant paramter values

.. code:: yaml

    header:
      ## Header information is used for logging. It is good practice to give your simulations explicit names
      title: Isotropic Elastic simulation # name for your simulation
      # A detailed description for your simulation
      description: |
        Material systems : Elastic domain (1)
        Interfaces : None
        Sources : Force source (1)
        Boundary conditions : Neumann BCs on all edges

Running the solver
-------------------

Finally, to run the SPECFEM++ solver

.. code:: bash

    ./specfem2d -p <PATH TO specfem_config.yaml>

.. note::

    Make sure either your are in the build directory of SPECFEM++ or the build directory is added to your ``PATH``.

Visualizing seimograms
----------------------

Let us now plot the traces generated by the solver using ``obspy``. This version of the code only supports ASCII output format for seismograms. To plot the seismograms we need to read the ASCII files as ``numpy`` arrays and them convert them to ``obspy`` streams. The following code snippet shows how to do this.

.. code-block:: python

    import os
    import numpy as np
    import obspy

    def get_traces(directory):
        traces = []
        ## iterate over all seismograms
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            station_name = os.path.splitext(filename)[0]
            trace = np.loadtxt(f, delimiter=' ')
            starttime = trace[0,0]
            dt = trace[1,0] - trace[0,0]
            traces.append(obspy.Trace(trace[:,1], {'network': station_name, 'starttime': starttime, 'delta': dt}))

        stream = obspy.Stream(traces)

        return stream

    directory = ## PATH TO DIRECTORY WHERE SEISMOGRAMS ARE STORED
    stream = get_traces(directory)
    stream.plot(size=(800, 1000))

.. figure:: ../../examples/homogeneous-medium-flat-topography/traces.svg
   :alt: Traces
   :width: 800
   :align: center

   Traces.
