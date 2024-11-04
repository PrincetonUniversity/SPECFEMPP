
.. _policy_edge_index:

Edge Policy
-----------

.. toctree::
    :maxdepth: 2

    policy
    iterator
    parallel_config

Example Usage
-------------

.. code-block:: cpp
    :linenos:

    #include "mesh/mesh.hpp"
    #include "compute/assembly.hpp"
    #include "policies/edge.hpp"
    #include <Kokkos_core.hpp>

    int main() {

        // Create a mesh from mesh database
        const specfem::mesh::mesh mesh("mesh_database.bin");

        // Create a assembly object
        specfem::compute::assembly assembly(mesh);

        // Lets get the acoustic elastic interface
        constexpr auto acoustic_tag = specfem::element::medium_tag::acoustic;
        constexpr auto elastic_tag = specfem::element::medium_tag::elastic;
        constexpr auto dimension_tag = specfem::dimension::type::dim2;

        const auto &interface = assembly.coupled_interface.get_interface_container<acoustic_tag, elastic_tag>();
        const auto &field = assembly.fields.forward;
        const auto [acoustic_index, elastic_index] = interface.get_index_mapping();
        const auto [acoustic_edge_type, elastic_edge_type] = interface.get_edge_type();
        const auto factor = interface.get_edge_factor();
        const auto normal = interface.get_edge_normal();

        // Create a edge policy
        using ParallelConfig = specfem::parallel_config::default_edge_config<dimension_tag, DefaultExecutionSpace>
        using EdgePolicyType = specfem::policy::edge<ParallelConfig>;

        EdgePolicyType edge(acoustic_index, elastic_index, acoustic_edge_type, elastic_edge_type);

        // We have to use a hack here. We cast the policy to the policy_type of the edge.
        // Since Kokkos does not support edge policy directly, we have to use this hack.
        const auto &policy = static_cast<typename EdgePolicyType::policy_type&>(edge);

        Kokkos::parallel_for("compute_edge", policy, KOKKOS_LAMBDA(const member_type &team) {
            const auto iterator = edge(team.league_rank());

            Kokkos::parallel_for(Kokkos::TeamThreadRange(iterator.edge_size()), [&](const int i) {
                const auto index = iterator(i);
                const auto acoustic_index = index.self_index;
                const auto elastic_index = index.coupled_index;

                DisplacemetPointField elastic_field;
                specfem::compute::load_on_device(coupled_index, field, elastic_field);

                AccelerationPointField acoustic_field;
                acoustic_field.acceleration(0) = factor * (elastic_field.displacement(0) * normal(0) + elastic_field.displacement(1) * normal(1));

                specfem::compute::store_on_device(acoustic_index, acoustic_field, field);

            });
        });

        return 0;

    }
