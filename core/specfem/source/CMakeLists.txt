file(GLOB_RECURSE COMPUTE_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(
        source_class
        ${COMPUTE_SOURCE_FILES}
)
add_library(specfem::source ALIAS source_class)

target_link_libraries(
        source_class
        Kokkos::kokkos
        specfem::utilities
        specfem::source_time_functions
        yaml-cpp
        enumerations
        specfem::algorithms
        ${BOOST_LIBS}
)
