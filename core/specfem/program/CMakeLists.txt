# Program library - contains context and execution logic
# abort.cpp is excluded here: it is compiled into specfem_program_abort (defined
# in the parent CMakeLists.txt) so that assembly can link it without a cycle.
file(GLOB_RECURSE PROGRAM_SOURCES "*.cpp")
list(REMOVE_ITEM PROGRAM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/abort.cpp")
add_library(specfem_program STATIC ${PROGRAM_SOURCES})
add_library(specfem::program ALIAS specfem_program)

target_include_directories(specfem_program PUBLIC ${CMAKE_SOURCE_DIR}/core)

target_link_libraries(specfem_program
    PUBLIC
        yaml-cpp::yaml-cpp
        ${BOOST_LIBS}
        specfem::logger
        specfem::mpi
        specfem::program_abort
        Kokkos::kokkos
        specfem::mesh
        specfem::quadrature
        assembly
        source_class
        specfem::runtime_configuration
        specfem::receivers
        specfem::io
        periodic_tasks
        specfem::solver
        specfem::attenuation
        specfem::version
        # Add MPI dependency when available
        $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
)
