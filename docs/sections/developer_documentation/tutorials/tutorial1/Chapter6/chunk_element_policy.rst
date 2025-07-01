
.. _ChunkElementPolicy:

Chunk Element Policy
--------------------

Chunk Element Policy is used to iterate over the elements of the mesh. Chunk element policy implements a heirarchical parallelism scheme based on ``Kokkos::TeamPolicy``. The following code snippet demonstrates how to use the chunk element policy to compute the mass matrix inside the acoustic medium. We first divide the mesh into chunks and assign each chunk to a team of threads (Kokkos team). Each team then computes the mass matrix for all the quadrature points within that chunk. A key thing to note here is that the quadrature points that are shared between elements are visited more than once, and in a parallel scheme could lead to race conditions. To avoid this, we use atomic operations to update the mass matrix.

.. code:: cpp

    #include "policies/chunk_element_policy.hpp"
    #include <Kokkos_Core.hpp>

    using namespace specfem::policy;
    using namespace specfem::parallel_config;
    using namespace specfem::point;
    using namespace specfem::assembly;
    using namespace Kokkos;
    using namespace specfem::datatype;

    // elements is view of indices of spectral elements
    // inside the acoustic medium
    KOKKOS_FUNCTION void compute_mass_matrix(
        const jacobian_matrix &derivatives,
        const properties &properties,
        const simulation_field &forward,
        const View<int *> &elements){

            const auto acoustic = forward.acoustic;
            using simd_type = simd<type_real, false>;
            constexpr simd_size = 1;
            using ParallelConfig =
                default_chunk_element_config<dim2, simd_type, DefaultExecutionSpace>;
            using ChunkElementPolicy = chunk_element<ParallelConfig>;
            using policy = typename ChunkElementPolicy::policy_type;

            ChunkElementPolicy chunk(elements, NGLL, NGLL);

            parallel_for("compute_mass_matrix", static_cast<policy &>(chunk),
                KOKKOS_LAMBDA(const member_type &member) {
                    // Iterate over all tiles that are assigned to this team.
                    // Within the policy, we divide the entire mesh into a set of tiles.
                    // Each tile is then divided into a set of chunks, where each chunk is
                    // assigned to a team of threads.
                    // The team then loops over the tiles in a sequential manner.

                    // nelements = ntiles * nchunks;
                    // nteams = nchunks;
                    for (int tile = 0; tile < ChunkPolicyType::tile_size * simd_size;
                            tile += ChunkPolicyType::chunk_size * simd_size) {
                        const int starting_element_index =
                            team.league_rank() * ChunkPolicyType::tile_size * simd_size +
                            tile;

                        if (starting_element_index >= nelements) {
                            break;
                        }
                        // Generate the iterator for chunk policy
                        const auto iterator = chunk_policy.league_iterator(starting_element_index);

                        // Compute the mass matrix for all quadrature points within the chunk
                        parallel_for(Kokkos::TeamThreadRange(member, iterator.chunk_size()),
                            [&](const int i) {
                                const auto iterator_index = iterator(i);
                                const auto index = iterator_index.index;
                                const int ix = iterator_index.index.ix;
                                const int iz = iterator_index.index.iz;

                                // ... load properties and derivatives into point types

                                PointMassType mass_matrix(jacobian_matrix.jacobian / properties.kappa);

                                // atomic add mass matrix to the global mass matrix
                                atomic_add(index, mass_matrix, forward);
                            });
                    }
                });

            Kokkos::fence();
        }
