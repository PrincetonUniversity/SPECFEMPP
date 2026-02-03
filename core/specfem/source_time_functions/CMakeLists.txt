file(GLOB_RECURSE COMPUTE_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(
        source_time_functions
        ${COMPUTE_SOURCE_FILES}
)
add_library(specfem::source_time_functions ALIAS source_time_functions)

target_link_libraries(
        source_time_functions
        read_seismogram
        Kokkos::kokkos
        specfem::point
        specfem::utilities
        specfem::logger
)

# Disable unity build for source_time_function due to KOKKOS_INLINE_FUNCTION issues
set_target_properties(source_time_functions PROPERTIES UNITY_BUILD OFF)
