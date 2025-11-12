file(GLOB POINT_SOURCE_FILES "./*.cpp")

add_library(point ${POINT_SOURCE_FILES})
add_library(specfem::point ALIAS point)

target_link_libraries(
        point
        Kokkos::kokkos
        ${BOOST_LIBS}
)
