file(GLOB_RECURSE IO_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_io ${IO_SOURCE_FILES})
add_library(specfem::io ALIAS specfem_io)

target_link_libraries(
        specfem_io
        specfem::mesh
        source_class
        specfem::receivers
        yaml-cpp::yaml-cpp
        specfem::logger
        specfem::utilities
        ${BOOST_LIBS}
        $<$<BOOL:${SPECFEM_ENABLE_NPZ}>:zlib>
        $<$<BOOL:${SPECFEM_ENABLE_HDF5}>:hdf5>
        $<$<BOOL:${SPECFEM_ENABLE_ADIOS2}>:adios2>
)
