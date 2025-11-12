# Simulation library - contains context and execution logic
file(GLOB_RECURSE SIMULATION_SOURCES "*.cpp")
add_library(specfem_simulation STATIC ${SIMULATION_SOURCES})
add_library(specfem::simulation ALIAS specfem_simulation)

target_include_directories(specfem_simulation PUBLIC ${CMAKE_SOURCE_DIR}/core)

target_link_libraries(specfem_simulation
    PUBLIC
        yaml-cpp::yaml-cpp
        ${BOOST_LIBS}
        specfem_mpi
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
