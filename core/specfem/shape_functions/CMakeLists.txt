file(GLOB_RECURSE SOURCE_FILES "./*.cpp" "**/*.cpp")

add_library(shape_functions ${SOURCE_FILES})

target_link_libraries(
        shape_functions
)
