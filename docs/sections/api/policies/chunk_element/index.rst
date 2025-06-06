
.. _policy_chunk_element_index:

Chunk Element Policy
--------------------

.. toctree::
    :maxdepth: 2

    policy
    parallel_configuration
    iterator

Example Usage
-------------

.. code-block:: cpp
    :linenos:

    #include "mesh/mesh.hpp"
    #include "compute/assembly.hpp"
    #include "policies/chunk_element.hpp"
    #include <Kokkos_core.hpp>

    int main() {

        // Create a mesh from mesh database
        const specfem::mesh::mesh mesh("mesh_database.bin");

        // Create a assembly object
        specfem::compute::assembly assembly(mesh);
        const auto &derivatives = assembly.partial_derivatives;

        // Lets generate a Kokkos::View for elements in the domain
        const int nspec = mesh.nspec;
        Kokkos::View<int*, Kokkos::DefaultExecutionSpace> elements("elements", nspec);

        Kokkos::parallel_for("generate_elements", nspec, KOKKOS_LAMBDA(const int ispec) {
            elements(ispec) = ispec;
        });

        // Create a chunk element policy
        using SIMD = specfem::datatype::simd<type_real, false>;
        using ParallelConfiguration = default_chunk_config<dim2, SIMD, DefaultExecutionSpace>;
        using PointPartialDerivativeType = specfem::point::partial_derivatives<dim2, false, false>
        using ChunkElementPolicy = specfem::policy::chunk_element<ParallelConfiguration>

        ChunkElementPolicy chunk_element(elements);
        const auto &policy = static_cast<typename ChunkElementPolicy::policy_type&>(chunk_element);

        Kokkos::parallel_for("compute_partial_derivatives", policy, KOKKOS_LAMBDA(const typename ChunkElementPolicy::member_type &team) {
            for (int tile = 0; tile < ParallelConfiguration::tile_size; tile+= ParallelConfiguration::chunk_size) {
                const int starting_element_index = team.league_rank() * ChunkPolicyType::tile_size * simd_size + tile;
                const auto iterator =
                    chunk_policy.league_iterator(starting_element_index);

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, iterator.chunk_size()),
                    [&](const int i) {
                        const auto iterator_index = iterator(i);
                        const auto index = iterator_index.index;

                        PointPartialDerivativeType point_derivatives(0, 0, 0, 0);

                        specfem::compute::store_on_device(index, point_derivatives, derivatives);
                    });
            }
        });

        // Wait for all kernels to finish
        Kokkos::fence();

        return 0;

    }
