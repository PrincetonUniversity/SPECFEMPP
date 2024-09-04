
.. _Chapter10:

Chapter 10: Solver
==================

In this chapter, we will put together the ``domain`` class that we developed in :ref:`Chapter 9 <Chapter8>` and the time-scheme that we implemented in :ref:`Chapter 9 <Chapter9>` into a ``solver`` class. The ``time-marching`` solver class will iterate over the time-steps and compute the evolution of wavefield over the course of the simulation.

.. code:: cpp

    #include "solver/solver.hpp"
    #include "domain/domain.hpp"
    #include "time_scheme/newmark.hpp"

    class time_marching : public solver {
    public:

        void run() override;

    private:

        domain domain;
        newmark newmark;
    };

The ``run`` method will be responsible for iterating over the time-steps and updating the wavefield at each time-step.

.. code:: cpp

    void time_marching::run(){
        for (const auto [istep, dt] :
                newmark->iterate_forward()) {
            newmark.apply_predictor_phase_forward();
            domain.compute_source_interaction(istep);
            domain.compute_stiffness_interaction();
            domain.divide_by_mass_matrix();
            newmark.apply_corrector_phase_forward();
        }
    }
