Simulation Setup
################

Simulation setup defines the run-time behaviour of the simulation. Below
are the parameter definition for the ``specfem_config.yaml``

.. admonition:: Example of the simulation setup with GLL4 and combined
                simulation mode, reading the wavefield, writing the kernels,
                and plotting the wavefields in PNG format.

    .. code-block:: yaml

        simulation-setup:
            quadrature:
                quadrature-type: GLL4

            solver:
                time-marching:
                    time-scheme:
                        type: Newmark
                        dt: 0.001
                        nstep: 1000
                        t0: 0.0

            simulation-mode:
                combined:
                    reader:
                        wavefield:
                            format: HDF5
                            directory: /path/to/input/folder

                    # This example avoids writing seismograms. That is, we avoid
                    # defining the seismogram node in the writer section.
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

    Exactly one of forward or combined simulation nodes should be defined.


Simulation Setup Parameter definitions
++++++++++++++++++++++++++++++++++++++


.. dropdown:: ``simulation-setup``
    :open:

    Simulation setup parameters

    :default value: None

    :possible values: [YAML Node]


Quadrature
----------

.. dropdown:: ``simulation-setup.quadrature`` [optional]
    :open:

    Type of quadrature used for the simulation.

    :default value:  4th order GLL quadrature with 5 GLL points

    :possible values: [YAML Node]

    .. admonition:: Example for defining 4th order GLL quadrature

        There are 2 ways to define the 4th order GLL quadrature

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

    .. dropdown:: ``simulation-setup.quadrature.quadrature-type`` [optional]

        Predefined quadrature types.

        1. ``GLL4`` defines 4th order GLL quadrature with 5 GLL points.
        2. ``GLL7`` defines 7th order GLL quadrature with 8 GLL points.

        :default value: GLL4

        :possible values: [GLL4, GLL7]


    .. dropdown:: ``simulation-setup.quadrature.alpha``

        Alpha value of the Gauss-Jacobi quadrature. For GLL quadrature alpha
        = 0.0

        :default value: None

        :possible values: [float, double]

        .. admonition:: Example for defining alpha value

            .. code-block:: yaml

                quadrature:
                    alpha: 0.0


    .. dropdown:: ``simulation-setup.quadrature.beta``

        Beta value of the Gauss-Jacobi quadrature. For GLL quadrature beta =
        0.0, and for GLJ quadrature beta = 1.0

        :default value: None

        :possible values: [float, double]

        .. admonition:: Example for defining beta value

            .. code-block:: yaml

                quadrature:
                    beta: 0.0


    .. dropdown:: ``simulation-setup.quadrature.ngllx``

        Number of GLL points in ``X`` dimension.

        :default value: None

        :possible values: [int]

        .. admonition:: Example for defining number of GLL points in X-dimension

            .. code-block:: yaml

                quadrature:
                    ngllx: 5


    .. dropdown:: ``simulation-setup.quadrature.ngllz``

        Number of GLL points in ``X`` dimension.

        :default value: None

        :possible values: [int]

        .. admonition:: Example for defining number of GLL points in Z-dimension

            .. code-block:: yaml

                quadrature:
                    ngllz: 5


Solver
------

.. dropdown:: ``simulation-setup.solver``
    :open:

    Section to define the type of solver to use for the simulation.

    :default value: None

    :possible values: [YAML Node]

    .. admonition:: Example for defining time-marching Newmark solver

        .. code-block:: yaml

            solver:
                time-marching:
                    time-scheme:
                        type: Newmark
                        dt: 0.001
                        nstep: 1000
                        t0: 0.0

    .. dropdown:: ``simulation-setup.solver.time-marching``

        Select either a time-marching or an explicit solver. Only
        time-marching solver is implemented currently.

        :default value: None

        :possible values: [YAML Node]

        .. admonition:: Example for defining time-marching solver

            .. code-block:: yaml

                solver:
                    time-marching:
                        time-scheme:
                            type: Newmark
                            dt: 0.001
                            nstep: 1000
                            t0: 0.0


        .. dropdown:: ``simulation-setup.solver.time-marching.time-scheme.type``

            Select time scheme for the solver

            :default value: None

            :possible values: [Newmark]

            .. admonition:: Example for defining Newmark time scheme

                .. code-block:: yaml

                    time-scheme:
                        type: Newmark



        .. dropdown:: ``simulation-setup.solver.time-marching.time-scheme.dt``

            Value of time step in seconds

            :default value: None

            :possible values: [float, double]

            .. admonition:: Example for defining time step

                .. code-block:: yaml

                    time-scheme:
                        dt: 0.001


        .. dropdown:: ``simulation-setup.solver.time-marching.time-scheme.nstep``

            Total number of time steps in the simulation.

            :default value: None

            :possible values: [int]

            .. admonition:: Example for defining number of time steps

                .. code-block:: yaml

                    time-scheme:
                        nstep: 1000


        .. dropdown:: ``simulation-setup.solver.time-marching.time-scheme.t0`` [optional]

            Start time of the simulation.

            :default value: 0.0

            :possible values: [float, double]

            .. admonition:: Example for defining start time

                .. code-block:: yaml

                    time-scheme:
                        t0: 0.0


Simulation Mode
---------------

.. dropdown:: ``simulation-setup.simulation-mode``
    :open:

    Defines the type of simulation to run (e.g. forward, adjoint, combined,
    etc.)

    :default value: None

    :possible values: [YAML Node]

    .. admonition:: Example for defining a forward simulation node

        .. code-block:: yaml

            simulation-mode:
                forward:
                    ...
                # or
                combined:
                    ...

    .. dropdown:: ``simulation-setup.simulation-mode.forward``

        Section to define the forward solver simulation parameters.

        :default value: None

        :possible values: [YAML Node]

        .. admonition:: Example for defining a forward simulation node

            .. code-block:: yaml

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



        .. dropdown:: ``simulation-setup.simulation-mode.forward.writer``
            :open:

            Defines the outputs to be stored to disk during the forward
            simulation.

            :default value: None

            :possible values: [YAML Node]

            .. admonition:: Example for defining a writer node

                .. code-block:: yaml

                    writer:
                        seismogram:
                            ...

                        wavefield:
                            ...

                        display:
                            ...

            .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.seismogram``

                Seismogram writer parameters.

                :default value: None

                :possible values: [YAML Node]

                .. admonition:: Example for defining a seismogram writer node

                    .. code-block:: yaml

                        writer:
                            seismogram:
                                format: ASCII
                                directory: /path/to/output/folder

                .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.seismogram.format`` [optional]

                    Output format of the seismogram.

                    :default value: ASCII

                    :possible values: [ASCII]


                .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.seismogram.directory`` [optional]

                    Output folder for the seismogram.

                    :default value: Current working directory

                    :possible values: [string]


            .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.wavefield``

                Forward wavefield writer parameters.

                :default value: None

                :possible values: [YAML Node]

                .. admonition:: Example for defining a wavefield writer node

                    .. code-block:: yaml

                        writer:
                            wavefield:
                                format: HDF5
                                directory: /path/to/output/folder


                .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.wavefield.format`` [optional]

                    Output format of the wavefield.

                    :default value: ASCII

                    :possible values: [ASCII, HDF5]


                .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.wavefield.directory`` [optional]

                    Output folder for the wavefield.

                    :default value: Current working directory

                    :possible values: [string]


            .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.display``

                Plot the wavefield during the forward simulation.

                :default value: None

                :possible values: [YAML Node]

                .. admonition:: Example for defining a display writer node

                    .. code-block:: yaml

                        writer:
                            display:
                                format: PNG
                                directory: /path/to/output/folder
                                field: displacement
                                simulation-field: forward
                                time-interval: 10

                .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.display.format`` [optional]

                    Output format for resulting plots.

                    :default value: PNG

                    :possible values: [PNG, JPG, on_screen]


                .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.display.directory`` [optional]

                    Output folder for the plots (not applicable for
                    on_screen).

                    :default value: Current working directory

                    :possible values: [string]


                .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.display.field``

                    Component of the wavefield to be plotted.

                    :default value: None

                    :possible values: [displacement, velocity, acceleration, pressure]


                .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.display.simulation-field``

                    Type of wavefield to be plotted.

                    :default value: None

                    :possible values: [forward]


                .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.display.time-interval``

                    Time step interval for plotting the wavefield.

                    :default value: None

                    :possible values: [int]

    .. dropdown:: ``simulation-setup.simulation-mode.combined`` [optional]

        Combined (forward + adjoint) simulation parameters.

        :default value: None

        :possible values: [YAML Node]

        .. admonition:: Example for defining a combined simulation node

            .. code-block:: yaml

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

            Exactly one of forward or combined simulation nodes should be defined.


        .. dropdown:: ``simulation-setup.simulation-mode.combined.reader`` [optional]
            :open:

            Defines the inputs to be read from disk during the combined
            simulation.

            :default value: None

            :possible values: [YAML Node]

            .. admonition:: Example for defining a reader node

                .. code-block:: yaml

                    reader:
                        wavefield:
                            format: HDF5
                            directory: /path/to/input/folder


            .. dropdown:: ``simulation-setup.simulation-mode.combined.reader.wavefield``

                Wavefield reader parameters.

                :default value: None

                :possible values: [YAML Node]


                .. dropdown:: ``simulation-setup.simulation-mode.combined.reader.wavefield.format`` [optional]

                    Format of the wavefield to be read.

                    :default value: ASCII

                    :possible values: [ASCII, HDF5]


                .. dropdown:: ``simulation-setup.simulation-mode.combined.reader.wavefield.directory`` [optional]

                    Folder containing the wavefield to be read.

                    :default value: Current working directory

                    :possible values: [string]


        .. dropdown:: ``simulation-setup.simulation-mode.combined.writer`` [optional]

            Defines the outputs to be stored to disk during the combined
            simulation.

            :default value: None

            :possible values: [YAML Node]

            .. admonition:: Example for defining a writer node

                .. code-block:: yaml

                    writer:
                        kernels:
                            format: HDF5
                            directory: /path/to/output/folder

                        seismogram:
                            format: ASCII
                            directory: /path/to/output/folder

                        display:
                            format: PNG
                            directory: /path/to/output/folder
                            field: displacement
                            simulation-field: adjoint
                            time-interval: 10


            .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.seismogram`` [optional]

                Seismogram writer parameters.

                :default value: None

                :possible values: [YAML Node]


                .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.seismogram.format`` [optional]

                    Output format of the seismogram.

                    :default value: ASCII

                    :possible values: [ASCII]


                .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.seismogram.directory`` [optional]

                    Output folder for the seismogram.

                    :default value: Current working directory

                    :possible values: [string]


            .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.kernels``

                Kernel writer parameters.

                :default value: None

                :possible values: [YAML Node]


                .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.kernels.format`` [optional]

                    Output format of the kernels.

                    :default value: ASCII

                    :possible values: [ASCII, HDF5]


                .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.kernels.directory`` [optional]

                    Output folder for the kernels.

                    :default value: Current working directory

                    :possible values: [string]


            .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.display`` [optional]

                Plot the wavefield during the combined simulation.

                :default value: None

                :possible values: [YAML Node]


                .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.display.format`` [optional]

                    Output format for resulting plots.

                    :default value: PNG

                    :possible values: [PNG, JPG, on_screen]


                .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.display.directory`` [optional]

                    Output folder for the plots (not applicable for
                    on_screen).

                    :default value: Current working directory

                    :possible values: [string]


                .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.display.field``

                    Component of the wavefield to be plotted.

                    :default value: None

                    :possible values: [displacement, velocity, acceleration, pressure]


                .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.display.simulation-field``

                    Type of wavefield to be plotted.

                    :default value: None

                    :possible values: [adjoint, backward]


                .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.display.time-interval``

                    Time step interval for plotting the wavefield.

                    :default value: None

                    :possible values: [int]
