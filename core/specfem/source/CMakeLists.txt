file(GLOB_RECURSE COMPUTE_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(
        source_class
        ${COMPUTE_SOURCE_FILES}
)

target_link_libraries(
        source_class
        Kokkos::kokkos
        specfem_mpi
        utilities
        source_time_function
        yaml-cpp
        enumerations
        algorithms
        ${BOOST_LIBS}
)
