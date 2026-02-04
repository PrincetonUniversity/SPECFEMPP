file(GLOB_RECURSE MESH_ENTITIES_SOURCE_FILES "./*.cpp")

add_library(specfem_mesh_entities ${MESH_ENTITIES_SOURCE_FILES})
add_library(specfem::mesh_entities ALIAS specfem_mesh_entities)

target_link_libraries(
        specfem_mesh_entities
        Kokkos::kokkos
)
