
.. _Divergence:

Divergence operator
-------------------

Divergence operator computes the divergence of a given vector field :math:`F_{ik} = \sum T_{ij} \partial_j \xi_k` evaluated at quadrature points using the spectral element formulation given by:

.. math::

    \begin{align*}
    \int_{\Omega} \nabla w : T &= \int_{\Omega} \sum_{i=1}^{N_D} F_{ik} \frac{\partial w_i}{\partial \xi_k} \\
    &\approx \sum_{i=1}^{N_D} w_i^{ \alpha \beta \gamma } \left[ \omega_{\beta} \omega_{\gamma} \sum_{\alpha = 0}^{n_l} \omega_{\alpha} J^{ \alpha \beta \gamma } F_{i1}^{ \alpha \beta \gamma } l'_{\alpha} + \omega_{\alpha} \omega_{\gamma} \sum_{\beta = 0}^{n_l} \omega_{\beta} J^{ \alpha \beta \gamma } F_{i2}^{ \alpha \beta \gamma } l'_{\beta} + \omega_{\alpha} \omega_{\beta} \sum_{\gamma = 0}^{n_l} \omega_{\gamma} J^{ \alpha \beta \gamma } F_{i3}^{ \alpha \beta \gamma } l'_{\gamma} \right]
    \end{align*}

The following code snippet demonstrates how to compute the divergence of a vector field within the elastic domain using the operator.

.. code:: cpp

    #include "specfem/algorithms.hpp"

    // Let us assume that the vector field F is given to us
    // as struct F;

    // Let us also assume that the interface provides us a
    // function to load the vector field within a chunk of elements
    // into a chunk view using `load_on_device` function.

    using namespace specfem::algorithms;

    View<type_real ****> compute_divergence(
            const assembly &assembly,
            const F &vector_field,
            View<int *> elements){
        // Chunk element view type
        using ChunkVectorFieldViewType;

        // Quadrature point view type
        using QuadratureViewType;

        // Create an output view to store the divergence
        View<type_real ****> divergence("divergence", nelements, ngll, ngll, 2);

        // Scratch pad size
        int scratch_size = ChunkVectorFieldViewType::shmem_size() +
                     ChunkStressIntegrandType::shmem_size()

        Kokkos::parallel_for("divergence", chunk_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_CLASS_LAMBDA(const typename ChunkPolicyType::member_type &team) {
                // Scratch views
                ChunkVectorFieldViewType element_field(team);
                QuadratureViewType element_quadrature(team);

                specfem::assembly::load_on_device(team, quadrature, element_quadrature);
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
                    specfem::assembly::load_on_device(team, iterator, vector_field,
                                                    element_field);

                    team.team_barrier();
                    divergence(team, iterator, jacobian_matrix, wgll,
                        element_quadrature.hprime_wgll, element_field,
                        [&](const typename ChunkPolicyType::iterator_type::index_type
                                &iterator_index,
                            const typename PointAccelerationType::ViewType &result) {
                                // Store the divergence in the output view
                                const int ispec = iterator_index.index.ispec;
                                const int iz = iterator_index.index.iz;
                                const int ix = iterator_index.index.ix;

                                divergence(ispec, iz, ix, 0) = result(0);
                                divergence(ispec, iz, ix, 1) = result(1);
                            });
                }
            });

            Kokkos::fence();

            return divergence;
    }
