Simulation Setup
################

Simulation setup defines the run-time behaviour of the simulation. Below
are the parameter definition for the ``specfem_config.yaml``

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


Parameter definitions
+++++++++++++++++++++


.. dropdown:: ``simulation-setup``
    :open:

    :default value: None

    :possible values: [YAML Node]

    :documentation: Simulation setup parameters


    .. dropdown:: ``simulation-setup.quadrature`` [optional]

        :default value:  4th order GLL quadrature with 5 GLL points

        :possible values: [YAML Node]

        :documentation: Type of quadrature used for the simulation.


        .. dropdown:: ``simulation-setup.quadrature.alpha``

            :default value: None

            :possible values: [float, double]

            :documentation: Alpha value of the Gauss-Jacobi quadrature. For GLL quadrature alpha = 0.0


        .. dropdown:: ``simulation-setup.quadrature.beta``

            :default value: None

            :possible values: [float, double]

            :documentation: Beta value of the Gauss-Jacobi quadrature. For GLL quadrature beta = 0.0, and for GLJ quadrature beta = 1.0


        .. dropdown:: ``simulation-setup.quadrature.ngllx``


            :default value: None

            :possible values: [int]

            :documentation: Number of GLL points in X-dimension


        .. dropdown:: ``simulation-setup.quadrature.ngllz``


            :default value: None

            :possible values: [int]

            :documentation: Number of GLL points in Z-dimension


        .. dropdown:: ``simulation-setup.quadrature.quadrature-type`` [optional]
            :open:

            :default value: GLL4

            :possible values: [GLL4, GLL7]

            :documentation: Predefined quadrature types.

            1. ``GLL4`` defines 4th order GLL quadrature with 5 GLL points.
            2. ``GLL7`` defines 7th order GLL quadrature with 8 GLL points.

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


    .. dropdown:: ``simulation-setup.solver``

        :default value: None

        :possible values: [YAML Node]

        :documentation: Type of solver to use for the simulation.

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

            :default value: None

            :possible values: [YAML Node]

            :documentation: Select either a time-marching or an explicit solver. Only time-marching solver is implemented currently.


            .. dropdown:: ``simulation-setup.solver.time-marching.time-scheme.type``

                :default value: None

                :possible values: [Newmark]

                :documentation: Select time scheme for the solver



            .. dropdown:: ``simulation-setup.solver.time-marching.time-scheme.dt``

                :default value: None

                :possible values: [float, double]

                :documentation: Value of time step in seconds


            .. dropdown:: ``simulation-setup.solver.time-marching.time-scheme.nstep``

                :default value: None

                :possible values: [int]

                :documentation: Total number of time steps in the simulation


            .. dropdown:: ``simulation-setup.solver.time-marching.time-scheme.t0`` [optional]

                :default value: 0.0

                :possible values: [float, double]

                :documentation: Start time of the simulation


    .. dropdown:: ``simulation-setup.simulation-mode``

        :default value: None

        :possible values: [YAML Node]

        :documentation: Defines the type of simulation to run (e.g. forward, adjoint, combined, etc.)

        .. dropdown:: ``simulation-setup.simulation-mode.forward``

            :default value: None

            :possible values: [YAML Node]

            :documentation: Forward simulation parameters

            .. admonition:: Example for defining a forward simulation node

                .. code-block:: yaml

                    simulation-mode:
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

                :default value: None

                :possible values: [YAML Node]

                :documentation: Defines the outputs to be stored to disk during the forward simulation

                .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.seismogram``

                    :default value: None

                    :possible values: [YAML Node]

                    :documentation: Seismogram writer parameters

                    .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.seismogram.format`` [optional]

                        :default value: ASCII

                        :possible values: [ASCII]

                        :documentation: Output format of the seismogram

                    .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.seismogram.directory`` [optional]

                        :default value: Current working directory

                        :possible values: [string]

                        :documentation: Output folder for the seismogram

                .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.wavefield``

                    :default value: None

                    :possible values: [YAML Node]

                    :documentation: Forward wavefield writer parameters

                    .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.wavefield.format`` [optional]

                        :default value: ASCII

                        :possible values: [ASCII, HDF5]

                        :documentation: Output format of the wavefield

                    .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.wavefield.directory`` [optional]

                        :default value: Current working directory

                        :possible values: [string]

                        :documentation: Output folder for the wavefield

                .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.display``

                    :default value: None

                    :possible values: [YAML Node]

                    :documentation: Plot the wavefield during the forward simulation

                    .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.display.format`` [optional]

                        :default value: PNG

                        :possible values: [PNG, JPG, on_screen]

                        :documentation: Output format for resulting plots

                    .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.display.directory`` [optional]

                        :default value: Current working directory

                        :possible values: [string]

                        :documentation: Output folder for the plots (not applicable for on_screen)


                    .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.display.field``

                        :default value: None

                        :possible values: [displacement, velocity, acceleration, pressure]

                        :documentation: Component of the wavefield to be plotted

                    .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.display.simulation-field``

                        :default value: None

                        :possible values: [forward]

                        :documentation: Type of wavefield to be plotted

                    .. dropdown:: ``simulation-setup.simulation-mode.forward.writer.display.time-interval``

                        :default value: None

                        :possible values: [int]

                        :documentation: Time step interval for plotting the wavefield


        .. dropdown:: ``simulation-setup.simulation-mode.combined`` [optional]

            :default value: None

            :possible values: [YAML Node]

            :documentation: Combined (forward + adjoint) simulation parameters

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

                :default value: None

                :possible values: [YAML Node]

                :documentation: Defines the inputs to be read from disk during the combined simulation

                .. dropdown:: ``simulation-setup.simulation-mode.combined.reader.wavefield``

                    :default value: None

                    :possible values: [YAML Node]

                    :documentation: Wavefield reader parameters

                    .. dropdown:: ``simulation-setup.simulation-mode.combined.reader.wavefield.format`` [optional]

                        :default value: ASCII

                        :possible values: [ASCII, HDF5]

                        :documentation: Format of the wavefield to be read

                    .. dropdown:: ``simulation-setup.simulation-mode.combined.reader.wavefield.directory`` [optional]

                        :default value: Current working directory

                        :possible values: [string]

                        :documentation: Folder containing the wavefield to be read


            .. dropdown:: ``simulation-setup.simulation-mode.combined.writer`` [optional]

                :default value: None

                :possible values: [YAML Node]

                :documentation: Defines the outputs to be stored to disk during the combined simulation

                .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.seismogram`` [optional]

                    :default value: None

                    :possible values: [YAML Node]

                    :documentation: Seismogram writer parameters

                    .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.seismogram.format`` [optional]

                        :default value: ASCII

                        :possible values: [ASCII]

                        :documentation: Output format of the seismogram

                    .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.seismogram.directory`` [optional]

                        :default value: Current working directory

                        :possible values: [string]

                        :documentation: Output folder for the seismogram

                .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.kernels``

                    :default value: None

                    :possible values: [YAML Node]

                    :documentation: Kernel writer parameters

                    .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.kernels.format`` [optional]

                        :default value: ASCII

                        :possible values: [ASCII, HDF5]

                        :documentation: Output format of the kernels

                    .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.kernels.directory`` [optional]

                        :default value: Current working directory

                        :possible values: [string]

                        :documentation: Output folder for the kernels

                .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.display`` [optional]

                    :default value: None

                    :possible values: [YAML Node]

                    :documentation: Plot the wavefield during the combined simulation

                    .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.display.format`` [optional]

                        :default value: PNG

                        :possible values: [PNG, JPG, on_screen]

                        :documentation: Output format for resulting plots

                    .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.display.directory`` [optional]

                        :default value: Current working directory

                        :possible values: [string]

                        :documentation: Output folder for the plots (not
                                            applicable for on_screen)

                    .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.display.field``

                        :default value: None

                        :possible values: [displacement, velocity, acceleration, pressure]

                        :documentation: Component of the wavefield to be plotted


                    .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.display.simulation-field``

                        :default value: None

                        :possible values: [adjoint, backward]

                        :documentation: Type of wavefield to be plotted


                    .. dropdown:: ``simulation-setup.simulation-mode.combined.writer.display.time-interval``

                        :default value: None

                        :possible values: [int]

                        :documentation: Time step interval for plotting the wavefield
