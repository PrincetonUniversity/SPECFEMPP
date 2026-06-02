file(GLOB_RECURSE DATETIME_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(
        specfem_datetime
        ${DATETIME_SOURCE_FILES}
)
add_library(specfem::datetime ALIAS specfem_datetime)
