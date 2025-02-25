

Parameter Documentation
=======================

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
