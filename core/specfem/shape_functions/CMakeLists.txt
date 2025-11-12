file(GLOB_RECURSE SOURCE_FILES "./*.cpp" "**/*.cpp")

add_library(shape_functions ${SOURCE_FILES})
add_library(specfem::shape_functions ALIAS shape_functions)

target_link_libraries(
        shape_functions
)
