# Configure setup.hpp from setup.hpp.in
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.hpp.in
               ${CMAKE_BINARY_DIR}/include/specfem/setup.hpp)

# Add the configure files to the clean target
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
    ${CMAKE_BINARY_DIR}/include/specfem/setup.hpp
)

# Add existing subdirectories
add_subdirectory(mpi)
add_subdirectory(logger)
add_subdirectory(macros)
add_subdirectory(assembly)
add_subdirectory(point)
add_subdirectory(receivers)
add_subdirectory(shape_function)
add_subdirectory(solver)
add_subdirectory(source)
add_subdirectory(source_time_functions)
add_subdirectory(jacobian)
add_subdirectory(timescheme)
add_subdirectory(periodic_tasks)
add_subdirectory(program)



# Create specfem::core interface library that ties everything together
add_library(specfem_core_interface INTERFACE)
add_library(specfem::core ALIAS specfem_core_interface)

target_link_libraries(specfem_core_interface
    INTERFACE
        specfem::assembly
        specfem::jacobian
        specfem::mpi
        specfem::periodic_tasks
        specfem::point
        specfem::program
        specfem::receivers
        specfem::shape_function
        specfem::source
        specfem::timescheme
)

target_include_directories(specfem_core_interface
    INTERFACE
        ${CMAKE_SOURCE_DIR}/core
)
