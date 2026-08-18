file(GLOB_RECURSE mesh_entity_SOURCE_FILES "./*.cpp")

add_library(specfem_mesh_entity ${mesh_entity_SOURCE_FILES})
add_library(specfem::mesh_entity ALIAS specfem_mesh_entity)

target_link_libraries(
        specfem_mesh_entity
        Kokkos::kokkos
)
