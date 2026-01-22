file(GLOB_RECURSE COMPUTE_SOURCE_FILES "./*.cpp")

add_library(specfem_compute ${COMPUTE_SOURCE_FILES})
add_library(specfem::compute ALIAS specfem_compute)

set_target_properties(specfem_compute PROPERTIES
        UNITY_BUILD $<NOT:$<BOOL:${__APPLE__}>>
        UNITY_BUILD_BATCH_SIZE $<IF:$<BOOL:${__APPLE__}>,0,4>
)

target_link_libraries(
        specfem_compute
        PRIVATE
        assembly
        Kokkos::kokkos
)
