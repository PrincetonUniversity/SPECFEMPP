file(GLOB_RECURSE IO_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_io ${IO_SOURCE_FILES})
add_library(specfem::io ALIAS specfem_io)

target_link_libraries(
        specfem_io
        specfem::mesh
        source_class
        specfem::receivers
        specfem::assembly
        yaml-cpp::yaml-cpp
        specfem::logger
        specfem::utilities
        ${BOOST_LIBS}
        Kokkos::kokkos
        specfem::io_backends
)

set_target_properties(specfem_io PROPERTIES
    UNITY_BUILD $<NOT:$<BOOL:${__APPLE__}>>
)
