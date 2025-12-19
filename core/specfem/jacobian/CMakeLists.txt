file(GLOB_RECURSE JACOBIAN_SOURCE_FILES "./*.cpp")

add_library(jacobian ${JACOBIAN_SOURCE_FILES})
add_library(specfem::jacobian ALIAS jacobian)

target_link_libraries(
        jacobian
        Kokkos::kokkos
        point
        shape_functions
)
