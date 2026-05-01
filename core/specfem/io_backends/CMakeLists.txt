file(GLOB_RECURSE IO_BACKENDS_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_io_backends ${IO_BACKENDS_SOURCE_FILES})
add_library(specfem::io_backends ALIAS specfem_io_backends)

target_link_libraries(
        specfem_io_backends
        PUBLIC
        Kokkos::kokkos
        $<$<BOOL:${SPECFEM_ENABLE_NPZ}>:zlib>
        $<$<BOOL:${SPECFEM_ENABLE_HDF5}>:hdf5>
        $<$<BOOL:${SPECFEM_ENABLE_ADIOS2}>:adios2>
)

target_compile_definitions(
        specfem_io_backends
        PUBLIC
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_NPZ}>>:-DNO_NPZ>
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_HDF5}>>:-DNO_HDF5>
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_ADIOS2}>>:-DNO_ADIOS2>
)

set_target_properties(specfem_io_backends PROPERTIES
    UNITY_BUILD $<NOT:$<BOOL:${__APPLE__}>>
)
