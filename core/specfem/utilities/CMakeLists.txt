file(GLOB_RECURSE UTILITIES_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_utilities ${UTILITIES_SOURCE_FILES})
add_library(specfem::utilities ALIAS specfem_utilities)

target_link_libraries(
        specfem_utilities
        Kokkos::kokkos
        ${BOOST_LIBS}
)
