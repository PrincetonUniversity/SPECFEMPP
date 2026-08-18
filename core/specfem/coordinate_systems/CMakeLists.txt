file(GLOB_RECURSE COORDINATE_SYSTEMS_SOURCE_FILES "./*.cpp")

add_library(specfem_coordinate_systems ${COORDINATE_SYSTEMS_SOURCE_FILES})
add_library(specfem::coordinate_systems ALIAS specfem_coordinate_systems)

target_link_libraries(
    specfem_coordinate_systems
    specfem::element
    specfem::constants
    specfem::ellipticity
)
