file(GLOB_RECURSE ALGORITHMS_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(
        specfem_algorithms
        ${ALGORITHMS_SOURCE_FILES}
)
add_library(specfem::algorithms ALIAS specfem_algorithms)

target_link_libraries(
        specfem_algorithms
        Kokkos::kokkos
        jacobian
        point
        assembly
        ${BOOST_LIBS}
)
