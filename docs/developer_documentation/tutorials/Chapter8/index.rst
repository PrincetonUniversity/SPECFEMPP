
.. _Chapter8:

Chapter 8: Implementing the ``Domain`` class
============================================

In this chapter, we will use the concepts described in Chapters :ref:`4 <Chapter4>`, :ref:`5 <Chapter5>`, :ref:`6 <Chapter6>`, and :ref:`7 <Chapter7>` to implement a ``Domain`` class that stores the computational kernels required to compute the evolution of the wavefield.

.. math::

    \int_{\Omega} \rho w \cdot \partial_t^{2} \mathbf{s} dV = - \int_{\Omega} \nabla w : \mathbf{T} dV

In particular, we will implement 2 kernels - 1. Compute the interaction of stiffness matrix with the wavefield i.e. right hand side of the above equation, and 2. Divide the computed stiffness term by the mass-matrix.

``Domain`` class definition
---------------------------

.. code:: cpp

    /// Lets define some compile-time constants &
    /// datatypes that will be used in the kernels
    class KernelDatatypes {
    public:
        constexpr static auto wavefield_type = forward;
        constexpr static auto dimension = dim2;
        constexpr static auto medium_tag = elastic;
        constexpr static auto property_tag = isotropic;
        constexpr static int ngll = 5;
        constexpr static bool using_simd = false;

        // For brevity, I will not define the datatypes here.

        // Range policy datatypes
        using RangePolicy = ...;
        using LoadFieldType = ...;
        using StoreFieldType = ...;

        constexpr static int num_dimensions = 2;
        constexpr static int components = 2;

        // Chunk element policy datatypes
        using ChunkPolicyType = ...;
        using ChunkElementFieldType = ...;
        using ChunkStressIntegrandType = ...;
        using ElementQuadratureType = ...;
        using PointAccelerationType = ...;
        using PointFieldDerivativesType = ...;
        using PointMassType = ...;
        using PointPropertyType = ...;
        using PointPartialDerivativesType = ...;
    };

    class domain {
    public:
        using DataTypes = KernelDatatypes;

        void compute_stiffness_interaction();

        void divide_by_mass_matrix();

        void compute_source_interaction(); /// Not implemented for
                                           /// brevity within this tutorial

    private:
        View<int *> elements; ///< Spectral element indices within
                              ///< elastic domain
        simulation_field<forward> field; ///< Reference to the
                                         ///< field on which
                                         ///< the kernels will act
        assembly assembly; ///< Assembly object that
                           ///< stores the mesh information
    };

.. note::

    Have a look at the :ref:`Chapter5` for more details on the data-types defined in ``KernelDatatypes``.

Implementing the kernels
------------------------

Lets start with with implementing the kernel to divide the stiffness term by the mass matrix using the :ref:`range policy <RangePolicy>`.

.. code:: cpp

    void domain::divide_by_mass_matrix() {
        const int nglob = field.get_nglob<DataTypes::medium_tag>();

        typename DataTypes::RangePolicy range(nglob);
        using policy_type = typename DataTypes::RangePolicy::policy_type;

        Kokkos::parallel_for("divide_by_mass_matrix",
            static_cast<policy_type &>(range),
            KOKKOS_LAMBDA(const int iglob){
                const auto iterator = range.range_iterator(iglob);
                const auto index = iterator(0);

                typename DataTypes::LoadFieldType
                    load_field;
                load_on_device(index.index, field, load_field);

                typename DataTypes::StoreFieldType
                    store_field(load_field.divide_mass_matrix());
                store_on_device(index.index, store_field, field);
            });

        Kokkos::fence();
    }

Next, lets implement ``compute_stiffness_interaction`` kernel.

.. code:: cpp

    KOKKOS_FUNCTION
    stress_integrand stiffness_component(
        const partial_derivatives &partial_derivatives,
        const property &property,
        const field_derivatives &du) {

            stress_integrand F;

            const type_real sigmaxx =
                properties.lambdaplus2mu * du(0, 0) + properties.lambda * du(1, 1);

            const type_real sigmazz =
                properties.lambdaplus2mu * du(1, 1) + properties.lambda * du(0, 0);

            const type_real sigmaxz =
                properties.mu * (du(0, 1) + du(1, 0));

            F(0, 0) =
                sigma_xx * partial_derivatives.xix +
                sigma_xz * partial_derivatives.xiz;
            F(0, 1) =
                sigma_xz * partial_derivatives.xix +
                sigma_zz * partial_derivatives.xiz;
            F(1, 0) =
                sigma_xx * partial_derivatives.gammax +
                sigma_xz * partial_derivatives.gammaz;
            F(1, 1) =
                sigma_xz * partial_derivatives.gammax +
                sigma_zz * partial_derivatives.gammaz;
        };

    void domain::compute_stiffness_interaction() {
        constexpr int ngll = DataTypes::ngll;
        ChunkPolicyType chunk_policy(elements, ngll, ngll);

        int scratch_size =
            DataTypes::ChunkElementFieldType::scratch_size() +
            DataTypes::ChunkStressIntegrandType::scratch_size() +
            DataTypes::ElementQuadratureType::scratch_size();

        Kokkos::parallel_for("compute_stiffness_interaction",
            chunk_policy.set_scratch_size(scratch_size),
            KOKKOS_LAMBDA(const member_type &member) {
                // Scratch views
                ChunkElementFieldType element_field(team);
                ElementQuadratureType element_quadrature(team);
                ChunkStressIntegrandType stress_integrand(team);

                load_on_device(team, quadrature, element_quadrature);

                for (int tile = 0; tile < ChunkPolicyType::tile_size * simd_size;
                        tile += ChunkPolicyType::chunk_size * simd_size) {
                    const int starting_element_index =
                        team.league_rank() * ChunkPolicyType::tile_size * simd_size +
                        tile;

                    if (starting_element_index >= nelements) {
                        break;
                    }

                    const auto iterator =
                        chunk_policy.league_iterator(starting_element_index);
                    load_on_device(team, iterator, field, element_field);

                    team.team_barrier();

                    algorithms::gradient(
                        team, iterator, partial_derivatives,
                        element_quadrature.hprime_gll, element_field.displacement,
                        [&](const index_type &index,
                            const field_derivatives &du) {

                            // load partial derivatives and properties into point types
                            // ...
                            // ...
                            // ...

                            const auto stress =
                                stiffness_component(partial_derivatives, properties, du);

                            for (int idim = 0; idim < num_dimensions; ++idim) {
                                for (int icomponent = 0; icomponent < components;
                                        ++icomponent) {
                                    stress_integrand.F(ielement, index.iz, index.ix, idim,
                                       icomponent) = stress(idim, icomponent);
                                }
                            }
                        });

                    team.team_barrier();

                    // Compute the divergence term
                    algorithms::divergence(
                        team, iterator, partial_derivatives, element_quadrature.hprime_gll,
                        [&](const index_type &index,
                            const ScalarViewType &result) {

                            PointAccelerationType acceleration(result);

                            for (int icomponent = 0; icomponent < components;
                                    icomponent++) {
                                acceleration.acceleration(icomponent) *= -1.0;
                            }

                            atomic_add(index, acceleration, field);
                        });
                }
            });

        Kokkos::fence();

    };
