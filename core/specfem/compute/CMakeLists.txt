
file(GLOB_RECURSE COMPUTE_SOURCE_FILES "*.cpp")

add_library(specfem_compute
    ${COMPUTE_SOURCE_FILES}
)
add_library(specfem::compute ALIAS specfem_compute)

set_target_properties(specfem_compute PROPERTIES UNITY_BUILD ON)

# Keeps TensorOperations headers out of the other sources in a unity batch.
set_source_files_properties(impl/stiffness_graph.cpp PROPERTIES
    SKIP_UNITY_BUILD_INCLUSION ON
)

target_link_libraries(specfem_compute
    Kokkos::kokkos
    specfem::assembly
    $<$<BOOL:${SPECFEM_ENABLE_TENSOROPS}>:specfem::tensorops>
)
