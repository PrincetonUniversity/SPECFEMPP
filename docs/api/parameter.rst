.. _parameter:

Simulation setup
================

To setup a simulation use a yaml configuration (specfem_config.yaml) file as defined below

Specfem configuration file
--------------------------

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
                dt: 1.1e-5
                nstep: 1600

        ## Runtime setup
        run-setup:
            number-of-processors: 1
            number-of-runs: 1

        ## databases
        databases:
            mesh-database: "../DATA/databases/database.bin"
            source-file: "../DATA/source.yaml"

Setup object used to instantiate a simulation
---------------------------------------------

.. doxygenfile:: parameter_parser.h
    :project: SPECFEM KOKKOS IMPLEMENTATION
