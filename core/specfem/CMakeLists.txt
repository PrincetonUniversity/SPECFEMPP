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

# Header-only constants library (constants.hpp needs setup.hpp + Kokkos + ellipticity)
add_library(specfem_constants INTERFACE)
add_library(specfem::constants ALIAS specfem_constants)
target_include_directories(specfem_constants
    INTERFACE
        ${CMAKE_SOURCE_DIR}/core
        ${CMAKE_BINARY_DIR}/include
)
target_link_libraries(specfem_constants
    INTERFACE
        Kokkos::kokkos
        specfem::ellipticity
)

# Add existing subdirectories
# mpi and logger must come first so specfem_program_abort can depend on them
add_subdirectory(mpi)
add_subdirectory(logger)

# Minimal abort library -- depends only on mpi + logger, so assembly can safely
# link it without creating a circular dependency with specfem::program.
add_library(specfem_program_abort STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/program/abort.cpp)
add_library(specfem::program_abort ALIAS specfem_program_abort)
target_include_directories(specfem_program_abort PUBLIC ${CMAKE_SOURCE_DIR}/core)
target_link_libraries(specfem_program_abort
    PUBLIC
        specfem::mpi
        specfem::logger
        $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
)

add_subdirectory(algorithms)
add_subdirectory(element_coupling)
add_subdirectory(ellipticity)
add_subdirectory(assembly)
add_subdirectory(compute)
add_subdirectory(datetime)
add_subdirectory(attenuation)
add_subdirectory(element)
add_subdirectory(element_connections)
add_subdirectory(enums)
add_subdirectory(jacobian)
add_subdirectory(linear_system)
add_subdirectory(io)
add_subdirectory(io_backends)
add_subdirectory(macros)
add_subdirectory(mesh)
add_subdirectory(mesh_entity)
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
add_subdirectory(coordinate_systems)

# C++ binding for the SPECFEM3D_GLOBE model oracle. Requires the globe Fortran
# tree, so it is only available when that is built. Not part of specfem::core.
if(SPECFEM_BUILD_MESHFEM3D_GLOBE)
    add_subdirectory(globe_model)
endif()



# Create specfem::core interface library that ties everything together
add_library(specfem_core_interface INTERFACE)
add_library(specfem::core ALIAS specfem_core_interface)

target_link_libraries(specfem_core_interface
    INTERFACE
        specfem::algorithms
        specfem::attenuation
        specfem::compute
        specfem::constants
        specfem::datetime
        specfem::assembly
        specfem::jacobian
        specfem::linear_system
        specfem::element
        specfem::element_connections
        specfem::element_coupling
        specfem::ellipticity
        specfem::enums
        specfem::io
        specfem::io_backends
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
        specfem::coordinate_systems
        specfem::version
)

target_include_directories(specfem_core_interface
    INTERFACE
        ${CMAKE_SOURCE_DIR}/core
)
