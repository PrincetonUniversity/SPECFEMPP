
.. _Chapter3:

Chapter 3: Generating Assembly
==============================

Lets now generate the :ref:`assembly <assembly_index>`.

.. code:: cpp

    #include <iostream>
    #include <string>

    specfem::assembly::assembly
        generate_assembly(const mesh &mesh,
                          const std::vector<source> &sources,
                          const std::vector<receiver> &receivers,
                          const simulation_params &params) {

        const int nsteps = params.nsteps;
        const int nseismo_steps = params.nseismo_steps;
        const type_real t0 = params.t0;
        const type_real dt =  params.dt;
        const auto seismo_type = params.seismo_type;
        const auto simulation = params.simulation_type;

        specfem::assembly::assembly assembly(mesh, sources, receivers,
                seimo_type, t0, dt, nsteps, nseismo_steps, simulation);

        return assembly;
    }
