
.. _Chapter11:

Chapter 11: Putting it all together
===================================

Finally, we have all the components that we need to put together to create a complete simulation. Let us now create the ``main`` function that will drive the simulation.

.. code:: cpp

    struct simulation_params {
        const type_real t0 = 0.0;
        const type_real dt = 1e-3;
        const int nsteps = 1000;
        const type_real angle = 0.0;
        const auto simulation_type = specfem::simulation::type::forward;
        const auto seismo_type = specfem::seismogram::type::displacement;
        const auto nseismo_steps = 1000;
    };

    int main(int argc, char **argv){
        // Initialize MPI
        specfem::MPI::MPI *mpi = new specfem::MPI::MPI(&argc, &argv);

        // Initialize Kokkos
        Kokkos::initialize(argc, argv);

        // Read the mesh
        specfem::mesh::mesh mesh("OUTPUT_FILES/database.bin", mpi);

        // Read the sources
        const auto sources = specfem::IO::sources::read_sources(
                                        "OUTPUT_FILES/sources.yaml",
                                        simulation_params.nsteps,
                                        simulation_params.t0,
                                        simulation_params.dt,
                                        simulation_params.simulation_type);

        // Read the receivers
        const auto receivers = specfem::receivers::read_receivers(
                                        "OUTPUT_FILES/STATIONS",
                                        simulation_params.angle);

        // Generate the assembly
        const auto assembly = generate_assembly(mesh, sources, receivers, simulation_params);

        // Initialize the domain
        domain domain(assembly)

        // Initialize the time-scheme
        newmark newmark(simulation_params.dt, simulation_params.nsteps, assembly);

        // Initialize the solver
        time_marching solver(domain, newmark);

        // Run the simulation
        solver.run();

        // Finalize Kokkos
        Kokkos::finalize();

        // Finalize MPI
        delete mpi;

        return 0;
    }
