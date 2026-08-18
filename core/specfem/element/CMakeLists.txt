file(GLOB_RECURSE ELEMENT_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_element ${ELEMENT_SOURCE_FILES})
add_library(specfem::element ALIAS specfem_element)

target_link_libraries(
        specfem_element
        Kokkos::kokkos
)
