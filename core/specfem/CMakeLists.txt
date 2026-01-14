# Add existing subdirectories
add_subdirectory(algorithms)
add_subdirectory(assembly)
add_subdirectory(jacobian)
add_subdirectory(kokkos_kernels)
add_subdirectory(logger)
add_subdirectory(macros)
add_subdirectory(mesh)
add_subdirectory(mpi)
add_subdirectory(periodic_tasks)
add_subdirectory(point)
add_subdirectory(program)
add_subdirectory(quadrature)
add_subdirectory(receivers)
add_subdirectory(runtime_configuration)
add_subdirectory(shape_functions)
add_subdirectory(solver)
add_subdirectory(source)
add_subdirectory(source_time_functions)
add_subdirectory(timescheme)
add_subdirectory(utilities)



# Create specfem::core interface library that ties everything together
add_library(specfem_core_interface INTERFACE)
add_library(specfem::core ALIAS specfem_core_interface)

target_link_libraries(specfem_core_interface
    INTERFACE
        specfem::algorithms
        specfem::assembly
        specfem::jacobian
        specfem::kokkos_kernels
        specfem::logger
        specfem::macros
        specfem::mesh
        specfem::mpi
        specfem::periodic_tasks
        specfem::point
        specfem::program
        specfem::quadrature
        specfem::receivers
        specfem::runtime_configuration
        specfem::shape_functions
        specfem::solver
        specfem::source
        specfem::source_time_functions
        specfem::timescheme
        specfem::utilities
)

target_include_directories(specfem_core_interface
    INTERFACE
        ${CMAKE_SOURCE_DIR}/core
)
