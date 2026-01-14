
.. _Chapter2:

Chapter 2: Reading the mesh, solver and receivers
=================================================

In this chapter, we will read the mesh, solver and receivers generated in :ref:`Chapter 1 <Chapter1>`.

Reading the mesh
----------------

The mesh generated in :ref:`Chapter1 <Chapter1>` is stored in the ``OUTPUT_FILES`` directory.

.. note::

    To read the mesh we will use SPECFEM++ :ref:`mesh reader <mesh>`.

.. code:: cpp

    #include "specfem/mesh.hpp"
    #include "specfem_mpi/specfem_mpi.hpp"

    int main(int argc, char **argv) {
        // Initialize MPI
        // The current SPECFEM++ implementation does not use MPI,
        // but we will initialize it for forward compatibility.
        specfem::MPI::MPI *mpi = new specfem::MPI::MPI(&argc, &argv);
        const std::string database_filename = "OUTPUT_FILES/database.bin";

        // We read and store the mesh within the mesh C++ struct.
        const specfem::mesh::mesh mesh(database_filename, mpi);

        return 0;
    }

Reading the sources and receivers
---------------------------------

Next we will read the sources and receivers generated in :ref:`Chapter1 <Chapter1>`.

.. note::

    Relevant modules for reading the :ref:`sources <sources_api>` and :ref:`receivers <receivers_api>`.

.. code:: cpp

    #include "source/source.hpp"
    #include "receivers/receivers.hpp"
    #include "specfem_setup.hpp"
    #include <iostream>

    int main(int argc, char **argv) {
        const std::string sources_filename = "OUTPUT_FILES/sources.yaml";
        const std::string receivers_filename = "OUTPUT_FILES/STATIONS";

        // The source time function requires the starting time (t0) and the time step (dt)
        const type_real t0 = 0.0; /// type_real is defined in specfem_setup.hpp
        const type_real dt = 1e-3; /// Time step
        const int nsteps = 1000; // Number of time steps
        const type_real angle = 0.0; /// Station angle
        const auto simulation_type = specfem::simulation::type::forward;

        // Read the sources and receivers
        const auto sources = specfem::io::sources::read_sources(sources_filename, nsteps, t0, dt, simulation_type);
        const auto receivers = specfem::receivers::read_receivers(receivers_filename, angle);

        // Output source information
        for (const auto &source : sources)
            std::cout << source->print() << std::endl;

        for (const auto &receiver : receivers)
            std::cout << receiver->print() << std::endl;

        return 0;
    }

Since we'd be using the simulation parameters often during this whole tutorial, we can define them in a struct and pass it around. Eventually, in :ref:`Chapter 12 <Chapter12>`, we will replace this struct and configure the simulation using a YAML file.

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

Generating the Quadrature
-------------------------

Lastly, we will need the integration quadrature to compute the evolution of the wavefield.

.. note::

    Refer :ref:`Quadrature <quadrature_api>` for more details.

.. code:: cpp

    #include "quadrature/quadrature.hpp"

    int main(int argc, char **argv) {

        // We use a lambda function to restrict the scope of gll
        const auto quadrature = []() {
            /// Gauss-Lobatto-Legendre quadrature with 5 GLL points
            const specfem::quadrature::gll::gll gll(0, 0, 5);

            return specfem::quadrature::quadratures(gll);
        };

        return 0;
    }
