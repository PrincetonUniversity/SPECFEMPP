# Configure the setup headers
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.hpp.in
               ${CMAKE_BINARY_DIR}/include/specfem/setup.hpp)

# Add the configure files to the clean target
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
    ${CMAKE_BINARY_DIR}/include/specfem/setup.hpp
)

# Generate version.hpp at build time so the git hash stays in sync after commits
# without requiring a full CMake reconfigure.
# The script (version_header.cmake) was rendered by cmake/version.cmake with
# all paths and the semver baked in; only the git-dynamic fields are re-queried.
add_custom_target(specfem_version_header ALL
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake/version_header.cmake
    BYPRODUCTS ${CMAKE_BINARY_DIR}/include/specfem/version.hpp
    COMMENT "Updating version.hpp from git"
)

# Add the generated file to the clean target
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
    ${CMAKE_BINARY_DIR}/include/specfem/version.hpp
)

add_library(specfem_version
    INTERFACE ${CMAKE_BINARY_DIR}/include/specfem/version.hpp)
add_library(specfem::version ALIAS specfem_version)
target_include_directories(specfem_version INTERFACE ${CMAKE_BINARY_DIR}/include)
add_dependencies(specfem_version specfem_version_header)

# Add existing subdirectories
add_subdirectory(algorithms)
add_subdirectory(element_coupling)
add_subdirectory(assembly)
add_subdirectory(compute)
add_subdirectory(attenuation)
add_subdirectory(element)
add_subdirectory(element_connections)
add_subdirectory(enums)
add_subdirectory(jacobian)
add_subdirectory(io)
add_subdirectory(logger)
add_subdirectory(macros)
add_subdirectory(mesh)
add_subdirectory(mesh_entity)
add_subdirectory(mpi)
add_subdirectory(periodic_tasks)
add_subdirectory(point)
add_subdirectory(program)
add_subdirectory(quadrature)
add_subdirectory(receivers)
add_subdirectory(runtime_configuration)
add_subdirectory(shape_function)
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
        specfem::attenuation
        specfem::compute
        specfem::assembly
        specfem::jacobian
        specfem::element
        specfem::element_connections
        specfem::element_coupling
        specfem::enums
        specfem::io
        specfem::logger
        specfem::macros
        specfem::mesh
        specfem::mesh_entity
        specfem::mpi
        specfem::periodic_tasks
        specfem::point
        specfem::program
        specfem::quadrature
        specfem::receivers
        specfem::runtime_configuration
        specfem::shape_function
        specfem::solver
        specfem::source
        specfem::source_time_functions
        specfem::timescheme
        specfem::utilities
        specfem::version
)

target_include_directories(specfem_core_interface
    INTERFACE
        ${CMAKE_SOURCE_DIR}/core
)
