# Add existing subdirectories
add_subdirectory(assembly)
add_subdirectory(point)
add_subdirectory(receivers)
add_subdirectory(shape_functions)
add_subdirectory(source)
add_subdirectory(jacobian)
add_subdirectory(timescheme)
add_subdirectory(periodic_tasks)
add_subdirectory(simulation)

# Create specfem::core interface library that ties everything together
add_library(specfem_core_interface INTERFACE)
add_library(specfem::core ALIAS specfem_core_interface)

target_link_libraries(specfem_core_interface
    INTERFACE
        specfem::assembly
        specfem::point
        specfem::receivers
        specfem::shape_functions
        specfem::source
        specfem::jacobian
        specfem::timescheme
        specfem::periodic_tasks
        specfem::simulation
)

target_include_directories(specfem_core_interface
    INTERFACE
        ${CMAKE_SOURCE_DIR}/core
)
