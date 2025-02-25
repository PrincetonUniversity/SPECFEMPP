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
