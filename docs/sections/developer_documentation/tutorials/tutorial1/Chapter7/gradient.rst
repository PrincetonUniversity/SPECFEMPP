
.. _Gradient:

Gradient Operator
=================

Gradient operator computes the gradient of a given field :math:`f` evaluated at quadrature points using the spectral element formulation given by:

.. math::

    \nabla f \approx \sum_{i=1}^{N_D} \hat{x}_i \left[ \sum_{ \alpha = 0}^{n_l} f^{ \alpha \beta^{'} \gamma^{'} } l_{\alpha}^{'} \partial_{i} \xi + \sum_{ \beta = 0}^{n_l} f^{ \alpha^{'} \beta \gamma^{'} } l_{\beta}^{'} \partial_{i} \eta + \sum_{ \gamma = 0}^{n_l} f^{ \alpha^{'} \beta^{'} \gamma } l_{\gamma}^{'} \partial_{i} \zeta \right]

The following code snippet demonstrates how to compute the gradient of displacement field within the elastic domain using the operator.

.. code:: cpp

    #include "specfem/algorithms.hpp"
    #include "polcies/chunk_element.hpp"

    using namespace specfem::algorithms;

    View<type_real *****> gradient(
            const assembly &assembly,
            const simulation_field &forward,
            View<int *> elements){
        // Chunk element view type
        using ChunkElementView = specfem::chunk_element::field<
            ParallelConfig::chunk_size, 5, dim2, elastic,
            ScratchSpace, Unmanaged, true, false, false, false, using_simd>;

        // Quadrature point view type
        using QuadratureViewType = specfem::quadrature::lagrange_derivative<
                5, dim2, ScratchSpace, Unmanaged>;

        // Create an output view to store the gradient
        View<type_real *****> compute_gradient("gradient", nelements, ngll, ngll, 2, 2);

        // Scratch pad size
        int scratch_size = ChunkElementFieldType::shmem_size() +
                     ChunkStressIntegrandType::shmem_size();

        Kokkos::parallel_for(
            "gradient", chunk_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_CLASS_LAMBDA(const typename ChunkPolicyType::member_type &team) {
                // Scratch views
                ChunkElementFieldType element_field(team);
                ElementQuadratureType lagrange_derivative(team);

                specfem::assembly::load_on_device(team, quadrature, lagrange_derivative);
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
                    specfem::assembly::load_on_device(team, iterator, field,
                                                    element_field);

                    team.team_barrier();

                    gradient(team, iterator, jacobian_matrix,
                        lagrange_derivative, element_field.displacement,
                        [&](const typename ChunkPolicyType::iterator_type::index_type
                                &iterator_index,
                            const typename PointFieldDerivativesType::ViewType &du) {
                                // Store the gradient in the output view
                                const int ispec = iterator_index.index.ispec;
                                const int iz = iterator_index.index.iz;
                                const int ix = iterator_index.index.ix;
                                gradient(ispec, iz, ix, 0, 0) = du(0, 0);
                                gradient(ispec, iz, ix, 0, 1) = du(0, 1);
                                gradient(ispec, iz, ix, 1, 0) = du(1, 0);
                                gradient(ispec, iz, ix, 1, 1) = du(1, 1);
                            });
                }
            });

            Kokkos::fence();

            return gradient;
    }

There are two new concepts introduced in the code snippet above:

1. `Scratch Pad <https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/HierarchicalParallelism.html#team-scratch-pad-memory>`_ : The scratch pad is a shared memory space that is used to store temporary data that is shared among threads in a team. The main advantage of using the scratch pad is that it provides access to shared memory that is faster (lower latency) than global memory. Within the context of the gradient operator, the scratch pad is used to store the displacement field within a chunk - this is necessary because the gradient operation access the displacement field in a non-contiguous manner.

2. Callback Functor : The gradient operator requires a callback functor that is called after the gradient is computed at each quadrature point. This enables the developer to use the gradient in a way that is specific to their application. In the code snippet above, the callback functor stores the gradient in an output view.
