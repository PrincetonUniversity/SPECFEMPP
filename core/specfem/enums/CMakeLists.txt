file(GLOB_RECURSE ENUMS_SOURCE_FILES "./*.cpp")

add_library(specfem_enums ${ENUMS_SOURCE_FILES})
add_library(specfem::enums ALIAS specfem_enums)

target_link_libraries(
        specfem_enums
        specfem::element
)
