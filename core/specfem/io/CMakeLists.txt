file(GLOB_RECURSE IO_SOURCE_FILES "./*.cpp" "./**/*.cpp")

# Keep the reader independent of the mesh/assembly/source IO stack so external
# source time functions can use it without introducing a static-library cycle.
list(REMOVE_ITEM IO_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/seismogram/reader.cpp")
add_library(specfem_seismogram_reader seismogram/reader.cpp)
add_library(specfem::seismogram_reader ALIAS specfem_seismogram_reader)
target_include_directories(specfem_seismogram_reader
        PUBLIC
        ${CMAKE_SOURCE_DIR}/core
        ${CMAKE_BINARY_DIR}/include
)
target_link_libraries(specfem_seismogram_reader PUBLIC Kokkos::kokkos)

add_library(specfem_io ${IO_SOURCE_FILES})
add_library(specfem::io ALIAS specfem_io)

target_link_libraries(
        specfem_io
        specfem::seismogram_reader
        specfem::mesh
        source_class
        specfem::receivers
        specfem::assembly
        specfem::globe
        yaml-cpp::yaml-cpp
        specfem::logger
        specfem::utilities
        ${BOOST_LIBS}
        Kokkos::kokkos
        specfem::io_backends
)
