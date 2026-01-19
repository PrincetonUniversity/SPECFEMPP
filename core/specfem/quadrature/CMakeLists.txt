file(GLOB_RECURSE QUADRATURE_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_quadrature ${QUADRATURE_SOURCE_FILES})
add_library(specfem::quadrature ALIAS specfem_quadrature)

target_link_libraries(
        specfem_quadrature
        Kokkos::kokkos
)
