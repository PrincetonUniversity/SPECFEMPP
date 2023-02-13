.. _parameter:

Simulation setup
================

To setup a simulation use a yaml configuration (specfem_config.yaml) file as defined below

Specfem configuration file
--------------------------

.. code-block:: yaml

     run-config:

        header:
            ## Header information is used for logging.
            ## It is good practice to give your simulations explicit names
            title: Elastic simulation # name for your simulation
            description: None # A detailed description for your simulation

        simulation-setup:
            ## quadrature setup
            quadrature:
            alpha: 0.0
            beta: 0.0
            ngllx: 5
            ngllz: 5

            ## Solver setup
            solver:
                time-marching:
                    type-of-simulation: forward
                    time-scheme:
                        type: Newmark
                        dt: 1e-3
                        nstep: 1600

        ## Runtime setup
        run-setup:
            number-of-processors: 1
            number-of-runs: 1

Setup object used to instantiate a simulation
---------------------------------------------

.. doxygenfile:: parameter_parser.h
    :project: SPECFEM KOKKOS IMPLEMENTATION
