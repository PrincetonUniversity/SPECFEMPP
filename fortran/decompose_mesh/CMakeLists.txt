project(decompose_mesh LANGUAGES Fortran C CXX)

cmake_minimum_required(VERSION 3.10)

enable_language(Fortran)
enable_language(C)
enable_language(CXX)

set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# Partitioner options (placeholders — all OFF by default; implementation deferred)
option(WITH_SCOTCH "Build with SCOTCH support" OFF)
option(WITH_PATOH  "Build with PaToH support"  OFF)
# METIS is controlled by SPECFEM_ENABLE_METIS (set in top-level CMakeLists.txt)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/archive)

# Fortran .mod files for this project
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})

# Generated headers from meshfem3D setup (constants.h, config.h, etc.)
include_directories(${CMAKE_BINARY_DIR}/fortran/meshfem3d/setup)

# Compiled Fortran module files from meshfem3D
include_directories(${CMAKE_BINARY_DIR}/fortran/meshfem3d/modules)

# ======================================================================
# Library 1: decompose_mesh_part_module
# Provides: part_decompose_mesh module
# ======================================================================
add_library(decompose_mesh_part_module
    part_decompose_mesh.F90
)
add_library(decompose_mesh::part_module ALIAS decompose_mesh_part_module)
target_link_libraries(decompose_mesh_part_module
    PRIVATE
    meshfem3D::shared_module
    meshfem3D::constants
    meshfem3D::mesh_module
)

# ======================================================================
# Library 3: decompose_mesh_par_module
# Provides: decompose_mesh_par module
# ======================================================================
add_library(decompose_mesh_par_module
    decompose_mesh_par.F90
)
add_library(decompose_mesh::par_module ALIAS decompose_mesh_par_module)
target_link_libraries(decompose_mesh_par_module
    PRIVATE
    decompose_mesh::part_module
    meshfem3D::shared_module
    meshfem3D::mesh_module
)

# ======================================================================
# Library 4: decompose_mesh_fault_module
# Provides: fault_scotch module
# ======================================================================
add_library(decompose_mesh_fault_module
    fault_scotch.f90
)
add_library(decompose_mesh::fault_module ALIAS decompose_mesh_fault_module)
target_link_libraries(decompose_mesh_fault_module
    PRIVATE
    decompose_mesh::par_module
    meshfem3D::shared
)

# ======================================================================
# Library 5: decompose_mesh_c_wrappers (OBJECT)
# C/C++ wrappers for external partitioners. Without USE_METIS / USE_PATOH
# defined, the #ifdef guards produce empty compilation units.
# ======================================================================
add_library(decompose_mesh_c_wrappers OBJECT
    wrap_metis.c
    wrap_patoh.cpp
)
add_library(decompose_mesh::c_wrappers ALIAS decompose_mesh_c_wrappers)
target_link_libraries(decompose_mesh_c_wrappers
    PRIVATE
    meshfem3D::config
)

if(SPECFEM_ENABLE_METIS)
    target_compile_definitions(decompose_mesh_c_wrappers PRIVATE USE_METIS)
    target_link_libraries(decompose_mesh_c_wrappers PRIVATE METIS::metis)
endif()

# ======================================================================
# Library 6: decompose_mesh_impl
# Main decomposition logic, partitioner stubs, mesh I/O
# ======================================================================
set(DECOMPOSE_MESH_IMPL_SOURCES
    check_valence.F90
    decompose_mesh.F90
    lts_helper.F90
    lts_setup_elements.F90
    partition_scotch.F90
    partition_metis.F90
    partition_patoh.F90
    partition_rows.F90
    read_decompose_mesh_par.F90
    read_mesh_files.F90
    write_mesh_databases.F90
)

add_library(decompose_mesh_impl
    ${DECOMPOSE_MESH_IMPL_SOURCES}
)
add_library(decompose_mesh::impl ALIAS decompose_mesh_impl)
target_link_libraries(decompose_mesh_impl
    PRIVATE
    decompose_mesh::par_module
    decompose_mesh::fault_module
    meshfem3D::shared
)


# ======================================================================
# Executable: xdecompose_mesh (serial)
# ======================================================================
add_executable(xdecompose_mesh
    program_decompose_mesh.f90
)
# Force Fortran as the linker language: the main entry point is a Fortran
# `program`, but CMake otherwise picks CXX because of wrap_patoh.cpp pulled in
# via decompose_mesh::c_wrappers, leaving `main` undefined at link time.
set_target_properties(xdecompose_mesh PROPERTIES LINKER_LANGUAGE Fortran)
target_link_libraries(xdecompose_mesh
    PRIVATE
    decompose_mesh::impl
    decompose_mesh::c_wrappers
    decompose_mesh::par_module
    decompose_mesh::fault_module
    meshfem3D::shared
    meshfem3D::shared_module
    meshfem3D::param_reader
    specfem::version
)


if (APPLE)
    target_link_options(xdecompose_mesh PRIVATE "LINKER:-no_compact_unwind")
endif()

# ======================================================================
# Install
# ======================================================================
if (SPECFEM_INSTALL)
    install(TARGETS xdecompose_mesh
        RUNTIME DESTINATION .
        COMPONENT executables
    )
endif()
