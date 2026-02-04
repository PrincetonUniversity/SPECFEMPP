file(GLOB_RECURSE MESH_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_mesh ${MESH_SOURCE_FILES})
add_library(specfem::mesh ALIAS specfem_mesh)

target_link_libraries(
        specfem_mesh
        enumerations
        Kokkos::kokkos
        specfem::macros
        specfem::mesh_entity
        specfem::element
        specfem::element_connections
        # material_class
        yaml-cpp
        specfem::io
        ${BOOST_LIBS}
)
