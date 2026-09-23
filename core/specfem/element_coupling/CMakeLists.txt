file(GLOB_RECURSE ELEMENT_COUPLING_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_element_coupling ${ELEMENT_COUPLING_SOURCE_FILES})
add_library(specfem::element_coupling ALIAS specfem_element_coupling)
