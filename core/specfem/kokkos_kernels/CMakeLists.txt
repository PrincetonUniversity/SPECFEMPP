file(GLOB_RECURSE KOKKOS_KERNELS_SOURCE_FILES "./*.cpp")

add_library(specfem_kokkos_kernels ${KOKKOS_KERNELS_SOURCE_FILES})
add_library(specfem::kokkos_kernels ALIAS specfem_kokkos_kernels)

set_target_properties(specfem_kokkos_kernels PROPERTIES
        UNITY_BUILD $<NOT:$<BOOL:${__APPLE__}>>
        UNITY_BUILD_BATCH_SIZE $<IF:$<BOOL:${__APPLE__}>,0,4>
)

target_link_libraries(
        specfem_kokkos_kernels
        PRIVATE
        assembly
        Kokkos::kokkos
)
