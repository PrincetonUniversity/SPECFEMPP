file(GLOB_RECURSE ELEMENT_CONNECTIONS_SOURCE_FILES "./*.cpp")

add_library(specfem_element_connections ${ELEMENT_CONNECTIONS_SOURCE_FILES})
add_library(specfem::element_connections ALIAS specfem_element_connections)

target_link_libraries(
        specfem_element_connections
        Kokkos::kokkos
)
