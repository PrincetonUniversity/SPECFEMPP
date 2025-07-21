file(GLOB JACOBIAN_SOURCE_FILES "./*.cpp")

add_library(jacobian ${JACOBIAN_SOURCE_FILES})

target_link_libraries(
        jacobian
        Kokkos::kokkos
        point
        shape_functions
)
