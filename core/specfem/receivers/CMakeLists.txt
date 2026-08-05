file(GLOB_RECURSE COMPUTE_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(
        receiver_class
        ${COMPUTE_SOURCE_FILES}
)

add_library(specfem::receivers ALIAS receiver_class)

target_link_libraries(
        receiver_class
        Kokkos::kokkos
        specfem::coordinate_systems
        yaml-cpp
        ${BOOST_LIBS}
)
