
file(GLOB_RECURSE COMPUTE_SOURCE_FILES "*.cpp")

add_library(specfem_compute
    ${COMPUTE_SOURCE_FILES}
)
add_library(specfem::compute ALIAS specfem_compute)

set_target_properties(specfem_compute PROPERTIES UNITY_BUILD OFF)

target_link_libraries(specfem_compute
    Kokkos::kokkos
    specfem::assembly
)
