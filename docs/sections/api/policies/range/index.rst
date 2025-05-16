
.. _policy_range_index:

Range Policy
------------

.. toctree::
    :maxdepth: 2

    policy
    iterator
    parallel_configuration

Example Usage
-------------

.. code-block:: cpp
    :linenos:

    #include "mesh/mesh.hpp"
    #include "compute/assembly.hpp"
    #include "policies/range.hpp"
    #include <Kokkos_core.hpp>

    int main() {

        // Create a mesh from mesh database
        const specfem::mesh::mesh mesh("mesh_database.bin");

        // Create a assembly object
        specfem::compute::assembly assembly(mesh);

        // Lets get number of quadrature points within acoustic domain
        const auto &forward_field = assembly.fields.forward;
        const int nglob = forward.acoustic.nglob;

        // Create a range policy
        using SIMD = specfem::datatype::simd<type_real, false>;
        using ParallelConfiguration = specfem::parallel_config::default_range_config<SIMD, Kokkos::DefaultExecutionSpace>;
        using PointFieldType = specfem::point::field<dim2, acoustic, false, false, true, false>;
        using RangePolicy = specfem::policy::range<ParallelConfiguration>;

        RangePolicy range(nglob);
        // We have to use a hack here. We cast the policy to the policy_type of the range.
        // Since Kokkos does not support range policy directly, we have to use this hack.
        const auto &policy = static_cast<typename RangePolicy::policy_type&>(range);

        // Iterate over all points in the acoustic domain and set acceleration to 0.0
        Kokkos::parallel_for("assign_values", policy, KOKKOS_LAMBDA(const int iglob) {
            const auto iterator = range(i);
            const auto index = iterator(0);
            PointFieldType acceleration;
            acceleration.acceleration(0) = 0.0;

            specfem::compute::store_on_device(index.index, acceleration, forward_field);
        });

        // Wait for all kernels to finish
        Kokkos::fence();

        return 0;
    }
