
.. _Chapter9:

Chapter 9: Creating a Time Scheme
=================================

Time scheme is used to advance the wavefield in time. For the case of this tutorial, we will only show how to implement a newmark forward time scheme. To do this we need to create a ``newmark`` timescheme class :

.. code:: cpp

    #include "timescheme/timescheme.hpp"

    class newmark : public time_scheme {
    public:

        void apply_predictor_phase_forward() override;
        void apply_corrector_phase_forward() override;

        void apply_predictor_phase_backward() override {
            throw std::runtime_error("Not implemented");
        };

        void apply_corrector_phase_backward() override {
            throw std::runtime_error("Not implemented");
        };

    private:
        constexpr static auto medium_tag = elastic;
        type_real dt;
        type_real deltat;
        simulation_field<forward> field;
    };

The ``newmark`` timescheme is derived from a base ``time_scheme`` class. The ``time_scheme`` class provides default methods to step forward in time ``iterate_forward`` and backward in time ``iterate_backward``. These default methods either increment or decrement the current timestep. Within, 2 stage stepping methods like runge-kutta you can override these methods to implement the correct stepping.

The time-scheme is divided into 2 phases: 1. Predictor phase and 2. Corrector phase. For the case of newmark method with a explicit central differnce schment (:math:`\alpha = 0.5` & :math:`\beta = 0`) the predictor phase is given by:

.. math::

    \begin{align*}
    u^{n+1} &= u^n + \Delta t v^n + \frac{1}{2} \Delta t^2 a^n
    v^{n+1} &= v^n + \frac{1}{2} \Delta t a^n
    M u^{n+1} + C v^{n+1} + K u^{n+1} = F^{n+1}
    \end{align*}

Here, the 3rd equation is the discretized form of the equation of motion solved by the spectral element method. The corrector phase is given by:

.. math::

    \begin{align*}
    v^{n} &= v^n + \frac{1}{2} \Delta t a^{n+1}
    \end{align*}

Implemeting the Time Scheme
---------------------------

1. Predictor Phase:

.. code:: cpp

    void newmark::apply_predictor_phase_forward() {
        using LoadFieldType =
            point::field<dim2, elastic, false, true, true, false, using_simd>
        using AddFieldType =
            specfem::point::field<specfem::dimension::type::dim2, MediumType, true,
                                    true, false, false, using_simd>;
        using StoreFieldType =
            specfem::point::field<specfem::dimension::type::dim2, MediumType, false,
                                    false, true, false, using_simd>;

        RangePolicyType range(nglob);

        Kokkos::parallel_for("predictor", range, KOKKOS_LAMBDA(const int i) {
            const auto iterator = range.range_iterator(iglob);
            const auto index = iterator(0);

            // Load the field
            LoadFieldType load;
            load_on_device(index, field, load);

            AddFieldType add;
            StoreFieldType store;

            // Compute the predictor phase
            add.displacement = dt * load.velocity + 0.5 * dt * dt * load.acceleration;
            add.velocity = 0.5 * dt * load.acceleration;
            store.acceleration = 0; /// We zero out the acceleration.
                                    /// Updates to the acceleration
                                    /// happen through the domain class.

            add_on_device(index, add, field);
            store_on_device(index, store, field);
        });

        Kokkos::fence();
    }

2. Corrector Phase:

.. code:: cpp

    void newmark::apply_corrector_phase_forward() {
        using LoadFieldType =
            point::field<dim2, elastic, false, false, true, false, using_simd>
        using AddFieldType =
            point::field<dim2, elastic, false, true, false, false, using_simd>;

        RangePolicyType range(nglob);

        Kokkos::parallel_for("corrector", range, KOKKOS_LAMBDA(const int i) {
            const auto iterator = range.range_iterator(iglob);
            const auto index = iterator(0);

            // Load the field
            LoadFieldType load;
            load_on_device(index, field, load);

            AddFieldType add;
            add.velocity = 0.5 * dt * load.acceleration;

            add_on_device(index, add, field);
        });

        Kokkos::fence();
    }
