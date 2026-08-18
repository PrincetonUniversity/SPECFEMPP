file(GLOB_RECURSE MESH_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_mesh ${MESH_SOURCE_FILES})
add_library(specfem::mesh ALIAS specfem_mesh)

target_link_libraries(
        specfem_mesh
        Kokkos::kokkos
        specfem::macros
        specfem::mesh_entity
        specfem::enums
        specfem::element
        specfem::element_connections
        # material_class
        yaml-cpp
        specfem::io
        ${BOOST_LIBS}
)

set_target_properties(specfem_mesh PROPERTIES
        UNITY_BUILD $<NOT:$<BOOL:${__APPLE__}>>
)
