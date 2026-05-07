# Static MPI wrapper library for SPECFEM++
# Provides context-managed MPI initialization and static access to rank/size
add_library(specfem_mpi STATIC mpi.cpp)
add_library(specfem::mpi ALIAS specfem_mpi)

target_include_directories(specfem_mpi
    PUBLIC
        ${CMAKE_SOURCE_DIR}/core
)

# Link MPI if available
if(TARGET MPI::MPI_CXX)
    target_link_libraries(specfem_mpi PUBLIC MPI::MPI_CXX)
endif()
