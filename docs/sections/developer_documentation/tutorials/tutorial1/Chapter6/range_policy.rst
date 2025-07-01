
.. _RangePolicy:

Range Policy
------------

Range policy is used to iterator over the entire range of quadrature points in the mesh, where every quadrature point is visited exactly once. The following code snippet demonstrates how to use the range policy to divide the acceleration by mass matrix inside acoustic medium - an operation that is common in the time-stepping loop of the spectral element solver.

.. code:: cpp

    #include "policies/range_policy.hpp"
    #include <Kokkos_Core.hpp>

    using namespace specfem::policy;
    using namespace specfem::parallel_config;
    using namespace specfem::point;
    using namespace specfem::assembly;
    using namespace Kokkos;
    using namespace specfem::datatype;

    void divide_by_mass_matrix(const simulation_field &forward){

        // Parallel configuration for range policy
        using ParallelConfig =
            default_range_config<simd<type_real,false>, DefaultExecutionSpace>;
        using RangePolicy = range<ParallelConfig>;
        // Load acceleration and mass matrix fields
        using LoadFieldType =
            field<dim2, acoustic, false, false, true, true, false>;
        // Store the result of the division at acceleration field
        using StoreFieldType =
            field<dim2, acoustic, false, false, true, false, false>;

        const int nglob = forward.get_nglob<acoustic>();

        RangePolicy range(nglob);

        parallel_for("divide_by_mass_matrix",
            static_cast<typename RangePolicy::policy_type &>(range),
            KOKKOS_LAMBDA(const int iglob){
                // Generate indices for range policy
                const auto iterator = range.range_iterator(iglob);
                const auto index = iterator(0);

                // Load acceleration and mass matrix fields
                LoadFieldType load_field;
                load_on_device(index.index, forward, load_field);

                // Divide acceleration by mass matrix and store the result
                StoreFieldType store_field(load_field.divide_mass_matrix());
                store_on_device(index.index, store_field, forward);
            });

        Kokkos::fence();
    }
