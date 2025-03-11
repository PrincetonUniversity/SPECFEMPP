.. _parameter_documentation:

SPECFEM++ Parameter Documentation
=================================


On this page, we first show an example of a parameter file for SPECFEM++ and then
provide detailed documentation for each parameter in the file in a collapsible
format since a lot of the parameters are optional and only required if parent
parameters are defined.

Example parameter file
----------------------


.. dropdown:: ``specfem_config.yaml``
    :open:

    .. code-block:: yaml

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
                time-scheme:
                  type: Newmark
                  dt: 1.1e-3
                  nstep: 1600

            simulation-mode:
              forward:
                writer:
                  seismogram:
                    format: "ascii"
                    directory: path/to/output/folder

          receivers:
            stations: path/to/stations_file
            angle: 0.0
            seismogram-type:
              - velocity
            nstep_between_samples: 1

          ## Runtime setup
          run-setup:
            number-of-processors: 1
            number-of-runs: 1

          ## databases
          databases:
            mesh-database: /path/to/mesh_database.bin

          ## sources
          sources: path/to/sources.yaml



Parameter definitions
---------------------


.. dropdown:: ``parameters``
    :open:

    .. dropdown:: ``header``

        Define header section for your simulation. This section is used for naming
        the run. but has no impact on the simulation itself.

        .. dropdown:: ``title``

            Brief name for this simulation

            :default value: None

            :possible values: [string]

            .. code-block:: yaml
                :caption: Example

                title: Isotropic Elastic simulation

        .. dropdown:: ``description``

            Detailed description for this run.

            :default value: None

            :possible values: [string]

            .. code-block:: yaml
                :caption: Example

                description: |
                  Material systems : Elastic domain (1)
                  Interfaces : None
                  Sources : Force source (1)
                  Boundary conditions : Neumann BCs on all edges


    .. dropdown:: ``simulation-setup``

        Section to define the simulation parameters

        .. dropdown:: ``quadrature`` [optional]

            Type of quadrature used for the simulation. There are 2 ways to
            define the 4th order GLL quadrature

            1. Using predefined quadrature type

                .. code-block:: yaml

                    quadrature:
                      quadrature-type: GLL4

            2. Using individual parameters

                .. code-block:: yaml

                    quadrature:
                      alpha: 0.0
                      beta: 0.0
                      ngllx: 5
                      ngllz: 5

            .. dropdown:: ``quadrature-type`` [optional]

                Predefined quadrature types.

                1. ``GLL4`` defines 4th order GLL quadrature with 5 GLL points.
                2. ``GLL7`` defines 7th order GLL quadrature with 8 GLL points.

                :default value: GLL4

                :possible values: [GLL4, GLL7]


            .. dropdown:: ``alpha``

                Alpha value of the Gauss-Jacobi quadrature. For GLL quadrature alpha
                = 0.0

                :default value: None

                :possible values: [float, double]

                .. code-block:: yaml
                    :caption: Example

                    quadrature:
                      alpha: 0.0


            .. dropdown:: ``beta``

                Beta value of the Gauss-Jacobi quadrature. For GLL quadrature beta =
                0.0, and for GLJ quadrature beta = 1.0

                :default value: None

                :possible values: [float, double]

                .. code-block:: yaml
                    :caption: Example

                    quadrature:
                      beta: 0.0


            .. dropdown:: ``ngllx``

                Number of GLL points in ``X`` dimension.

                :default value: None

                :possible values: [int]

                .. code-block:: yaml
                    :caption: Example

                    quadrature:
                      ngllx: 5


            .. dropdown:: ``ngllz``

                Number of GLL points in ``X`` dimension.

                :default value: None

                :possible values: [int]

                .. code-block:: yaml
                    :caption: Example

                    quadrature:
                        ngllz: 5


        .. dropdown:: ``solver``

            Section to define the type of solver to use for the simulation.

            .. code-block:: yaml
                :caption: Example solver section

                solver:
                    time-marching:
                        time-scheme:
                            type: Newmark
                            dt: 0.001
                            nstep: 1000
                            t0: 0.0


            .. dropdown:: time-marching

                Select either a time-marching or an explicit solver. Only
                time-marching solver is implemented currently.

                .. dropdown:: ``time-scheme``

                    Section to define the time scheme for the solver.

                    .. dropdown:: ``type``

                        Select time scheme for the solver

                        :default value: None

                        :possible values: [Newmark]

                        .. code-block:: yaml
                            :caption: Example

                            time-scheme:
                                type: Newmark


                    .. dropdown:: ``dt``

                        Value of time step in seconds

                        :default value: None

                        :possible values: [float, double]

                        .. code-block:: yaml
                            :caption: Example

                            time-scheme:
                                dt: 0.001


                    .. dropdown:: ``nstep``

                        Total number of time steps in the simulation.

                        :default value: None

                        :possible values: [int]

                        .. code-block:: yaml
                            :caption: Example

                            time-scheme:
                                nstep: 1000


                    .. dropdown:: ``t0`` [optional]

                        Start time of the simulation.

                        :default value: 0.0

                        :possible values: [float, double]

                        .. code-block:: yaml
                            :caption: Example

                            time-scheme:
                                t0: 0.0

        .. dropdown:: ``simulation-mode``

            Defines the type of simulation to run (e.g. forward, adjoint, combined,
            etc.)

            .. code-block:: yaml
                :caption: Example

                simulation-mode:
                    forward:
                        ...
                    # or
                    combined:
                        ...

            .. note::

                Exactly one of forward or combined simulation nodes should be
                defined.

            .. dropdown:: ``forward``

                Section to define the forward solver simulation parameters.

                .. code-block:: yaml
                    :caption: Example forward simulation node

                    forward:
                        writer:
                            seismogram:
                                format: ASCII
                                directory: /path/to/output/folder

                            wavefield:
                                format: HDF5
                                directory: /path/to/output/folder

                            display:
                                format: PNG
                                directory: /path/to/output/folder
                                field: displacement
                                simulation-field: forward
                                time-interval: 10

                .. note::

                    At least one writer node should be defined in the forward simulation node.


                .. dropdown:: ``writer``

                    Defines the outputs to be stored to disk during the forward
                    simulation.

                    .. dropdown:: ``seismogram``

                        Seismogram writer parameters.

                        .. code-block:: yaml

                            writer:
                                seismogram:
                                    format: ASCII
                                    directory: /path/to/output/folder

                        .. dropdown:: ``format`` [optional]

                            Output format of the seismogram.

                            :default value: ASCII

                            :possible values: [ASCII]


                        .. dropdown:: ``directory`` [optional]

                            Output folder for the seismogram.

                            :default value: Current working directory

                            :possible values: [string]


                    .. dropdown:: ``wavefield``

                        Forward wavefield writer parameters.

                        .. code-block:: yaml
                            :caption: Example

                            writer:
                              wavefield:
                                format: HDF5
                                directory: /path/to/output/folder


                        .. dropdown:: ``format`` [optional]

                            Output format of the wavefield.

                            :default value: ASCII

                            :possible values: [ASCII, HDF5]


                        .. dropdown:: ``directory`` [optional]

                            Output folder for the wavefield.

                            :default value: Current working directory

                            :possible values: [string]


                    .. dropdown:: ``display``

                        Plot the wavefield during the forward simulation.

                        .. code-block:: yaml
                            :caption: Example

                            writer:
                              display:
                                format: PNG
                                directory: /path/to/output/folder
                                field: displacement
                                simulation-field: forward
                                time-interval: 10

                        .. dropdown:: ``format`` [optional]

                            Output format for resulting plots.

                            :default value: PNG

                            :possible values: [PNG, JPG, on_screen]


                        .. dropdown:: ``directory`` [optional]

                            Output folder for the plots (not applicable for
                            on_screen).

                            :default value: Current working directory

                            :possible values: [string]


                        .. dropdown:: ``field``

                            Component of the wavefield to be plotted.

                            :default value: None

                            :possible values: [displacement, velocity, acceleration, pressure]


                        .. dropdown:: ``simulation-field``

                            Type of wavefield to be plotted.

                            :default value: None

                            :possible values: [forward]


                        .. dropdown:: ``time-interval``

                            Time step interval for plotting the wavefield.

                            :default value: None

                            :possible values: [int]


            .. dropdown:: ``combined`` [optional]

                Combined (forward + adjoint) simulation parameters.

                .. code-block:: yaml
                    :caption: Example combined simulation node

                    simulation-mode:
                      combined:
                        reader:
                          wavefield:
                            format: HDF5
                            directory: /path/to/input/folder

                        ## This example avoids writing seismograms
                        writer:
                          kernels:
                            format: HDF5
                            directory: /path/to/output/folder

                        display:
                          format: PNG
                          directory: /path/to/output/folder
                          field: displacement
                          simulation-field: adjoint
                          time-interval: 10

                .. note::

                    Exactly one of forward or combined simulation nodes should
                    be defined.


                .. dropdown:: ``reader`` [optional]

                    Defines the inputs to be read from disk during the combined
                    simulation.

                    .. dropdown:: ``wavefield``

                        Wavefield reader parameters.

                        :default value: None

                        :possible values: [YAML Node]


                        .. dropdown:: ``format`` [optional]

                            Format of the wavefield to be read.

                            :default value: ASCII

                            :possible values: [ASCII, HDF5]


                        .. dropdown:: ``directory`` [optional]

                            Folder containing the wavefield to be read.

                            :default value: Current working directory

                            :possible values: [string]


                .. dropdown:: ``writer`` [optional]

                    Defines the outputs to be stored to disk during the combined
                    simulation.

                    .. dropdown:: ``seismogram`` [optional]

                        Seismogram writer parameters.

                        .. dropdown:: ``format`` [optional]

                            Output format of the seismogram.

                            :default value: ASCII

                            :possible values: [ASCII]


                        .. dropdown:: ``directory`` [optional]

                            Output folder for the seismogram.

                            :default value: Current working directory

                            :possible values: [string]


                    .. dropdown:: ``kernels``

                        Kernel writer parameters.

                        .. dropdown:: ``format`` [optional]

                            Output format of the kernels.

                            :default value: ASCII

                            :possible values: [ASCII, HDF5]


                        .. dropdown:: ``directory`` [optional]

                            Output folder for the kernels.

                            :default value: Current working directory

                            :possible values: [string]


                    .. dropdown:: ``display`` [optional]

                        Plot the wavefield during the combined simulation.

                        .. dropdown:: ``format`` [optional]

                            Output format for resulting plots.

                            :default value: PNG

                            :possible values: [PNG, JPG, on_screen]


                        .. dropdown:: ``directory`` [optional]

                            Output folder for the plots (not applicable for
                            on_screen).

                            :default value: Current working directory

                            :possible values: [string]


                        .. dropdown:: ``field``

                            Component of the wavefield to be plotted.

                            :default value: None

                            :possible values: [displacement, velocity, acceleration, pressure]


                        .. dropdown:: ``simulation-field``

                            Type of wavefield to be plotted.

                            :default value: None

                            :possible values: [adjoint, backward]


                        .. dropdown:: ``time-interval``

                            Time step interval for plotting the wavefield.

                            :default value: None

                            :possible values: [int]


    .. dropdown:: ``receivers``

        Parameter file section that contains the receiver information required to
        calculate seismograms.

        .. code-block:: yaml
            :caption: Example receivers section

            receivers:
                stations: /path/to/stations_file
                angle: 0.0
                seismogram-type:
                    - velocity
                    - displacement
                nstep_between_samples: 1

        .. note::

            Please note that the ``stations_file`` is generated using SPECFEM2D mesh
            generator i.e. xmeshfem2d

        .. dropdown:: ``stations``

            Path to ``stations_file``.

            :default value: None

            :possible values: [string]

            .. code-block:: yaml
                :caption: Example

                stations: /path/to/stations_file


        .. dropdown:: ``angle``

            Angle to rotate components at receivers

            :default value: None

            :possible values: [float]

            .. code-block:: yaml
                :caption: Example

                angle: 0.0


        .. dropdown:: ``seismogram-type``

            Type of seismograms to be written.

            :default value: None

            :possible values: [YAML list]

            .. code-block:: yaml
                :caption: Example

                seismogram-type:
                    - velocity
                    - displacement

            .. rst-class:: center-table

                +-------------------+---------------------------------------+-------------------------------------+
                |  Seismogram       | SPECFEM Par_file ``seismotype`` value | ``receivers.seismogram-type`` value |
                +===================+=======================================+=====================================+
                | Displacement      |                   1                   |   ``displacement``                  |
                +-------------------+---------------------------------------+-------------------------------------+
                | Velocity          |                   2                   |    ``velocity``                     |
                +-------------------+---------------------------------------+-------------------------------------+
                | Acceleration      |                   3                   |     ``acceleration``                |
                +-------------------+---------------------------------------+-------------------------------------+
                | Pressure          |                   4                   |      ``pressure``                   |
                +-------------------+---------------------------------------+-------------------------------------+
                | Displacement Curl |                   5                   |     ✘ Unsupported                   |
                +-------------------+---------------------------------------+-------------------------------------+
                | Fluid Potential   |                   6                   |     ✘ Unsupported                   |
                +-------------------+---------------------------------------+-------------------------------------+



        .. dropdown:: ``nstep_between_samples``

            Number of time steps between sampling the wavefield at station locations
            for writing seismogram.

            :default value: None

            :possible values: [int]

            .. code-block:: yaml
                :caption: Example

                nstep_between_samples: 1



    .. dropdown:: ``run-setup``

        Define run-time configuration for your simulation.

        .. code-block:: yaml
            :caption: Example run-setup section

            run-setup:
                number-of-processors: 1
                number-of-runs: 1

        .. dropdown:: ``number-of-processors``

            Number of MPI processes used in the simulation. MPI version is not
            enabled in this version of the package. number-of-processors == 1

            :default value: 1

            :possible values: [int]

            .. code-block:: yaml
                :caption: Example

                number-of-processors: 1


        .. dropdown:: ``number-of-runs``

            Number of runs in this simulation. Only single run implemented in this
            version of the package. number-of-runs == 1

            :default value: 1

            :possible values: [int]

            .. code-block:: yaml
                :caption: Example

                number-of-runs: 1



    .. dropdown:: ``databases``

        The databases section defines the location of files to be read by the
        solver.

        .. code-block:: yaml
            :caption: Example of databases section

            databases:
                mesh-database: /path/to/mesh_database.bin


        .. _database-file-parameter:

        .. dropdown:: ``mesh-database``
            :open:

            Location of the fortran binary database file defining the mesh

            :default value: None

            :possible values: [string]

            .. code-block:: yaml
                :caption: Example

                mesh-database: /path/to/mesh_database.bin


    .. dropdown:: ``sources``

        Define sources

        :default value: None

        :possible values: [string, YAML Node]

        .. admonition:: Example sources section

            The sources is a path to a YAML file.

            .. code-block:: yaml

                sources: path/to/sources.yaml

            The sources section is a YAML node that contains the source information

            .. code-block:: yaml

                sources:
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

        .. note::

            The parameters below are only relevant if the sources section is
            defined as a YAML node.

        .. dropdown:: ``number-of-sources``

            Number of sources in the simulation

            :default value: None

            :possible values: [int]

            .. code-block:: yaml
                :caption: Example

                number-of-sources: 1


        .. dropdown:: ``sources``

            List of sources

            :default value: None

            :possible values: [YAML Node]

            .. code-block:: yaml
                :caption: Example

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
