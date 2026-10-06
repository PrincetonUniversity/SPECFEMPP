file(GLOB_RECURSE INJECTION_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_injection ${INJECTION_SOURCE_FILES})
add_library(specfem::injection ALIAS specfem_injection)

target_link_libraries(
        specfem_injection
        Kokkos::kokkos
        specfem::constants
        specfem::element
        specfem::utilities
        ${BOOST_LIBS}
)
