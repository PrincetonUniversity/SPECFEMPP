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

target_compile_definitions(
        specfem_io
        PUBLIC
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_NPZ}>>:-DNO_NPZ>
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_HDF5}>>:-DNO_HDF5>
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_ADIOS2}>>:-DNO_ADIOS2>
)

if(SPECFEM_ENABLE_ADIOS2)
    target_link_libraries(specfem_io PUBLIC adios2::adios2_cxx11_mpi)
endif()

if(SPECFEM_ENABLE_HDF5)
    target_link_libraries(specfem_io PUBLIC hdf5::hdf5)
endif()

if(SPECFEM_ENABLE_MPI)
    target_link_libraries(specfem_io PUBLIC MPI::MPI_CXX)
endif()
