# Program library - contains context and execution logic
file(GLOB_RECURSE PROGRAM_SOURCES "*.cpp")
add_library(specfem_program STATIC ${PROGRAM_SOURCES})
add_library(specfem::program ALIAS specfem_program)

target_include_directories(specfem_program PUBLIC ${CMAKE_SOURCE_DIR}/core)

target_link_libraries(specfem_program
    PUBLIC
        yaml-cpp::yaml-cpp
        ${BOOST_LIBS}
        specfem_mpi
        specfem::logger
        specfem::mpi
        Kokkos::kokkos
        mesh
        quadrature
        assembly
        source_class
        parameter_reader
        specfem::receivers
        writer
        periodic_tasks
        reader
        kokkos_kernels
        solver
        # Add MPI dependency when available
        $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
)
