file(GLOB POINT_SOURCE_FILES "./*.cpp")

add_library(point ${POINT_SOURCE_FILES})

target_link_libraries(
        point
        Kokkos::kokkos
)
