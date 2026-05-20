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

if (SPECFEM_ENABLE_MPI)
    add_compile_definitions(WITH_MPI)
    message(STATUS "-- Building decompose_mesh with MPI support")
    if (NOT TARGET MPI::MPI_Fortran)
        find_package(MPI REQUIRED COMPONENTS Fortran)
    endif()
endif()

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
# Library 2: decompose_mesh_hdf5_module (conditional on SPECFEM_ENABLE_HDF5)
# Provides: part_decompose_mesh_hdf5 module
# ======================================================================
if (SPECFEM_ENABLE_HDF5)
    add_library(decompose_mesh_hdf5_module
        part_decompose_mesh_hdf5.F90
    )
    add_library(decompose_mesh::hdf5_module ALIAS decompose_mesh_hdf5_module)
    target_compile_definitions(decompose_mesh_hdf5_module PRIVATE USE_HDF5)
    target_link_libraries(decompose_mesh_hdf5_module
        PRIVATE
        decompose_mesh::part_module
        meshfem3D::shared_hdf5_module
    )
endif()

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
    write_mesh_databases_hdf5.F90
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

if (SPECFEM_ENABLE_HDF5)
    target_compile_definitions(decompose_mesh_impl PRIVATE USE_HDF5)
    target_link_libraries(decompose_mesh_impl
        PRIVATE
        decompose_mesh::hdf5_module
        meshfem3D::shared_hdf5_module
    )
endif()

# ======================================================================
# Library 7: decompose_mesh_mpi_modules (only when SPECFEM_ENABLE_MPI)
# Provides: module_qsort, module_database, module_mesh, module_partition
# ======================================================================
if (SPECFEM_ENABLE_MPI)
    add_library(decompose_mesh_mpi_modules
        module_qsort.f90
        module_database.f90
        module_mesh.f90
        module_partition.f90
    )
    add_library(decompose_mesh::mpi_modules ALIAS decompose_mesh_mpi_modules)
    target_link_libraries(decompose_mesh_mpi_modules
        PRIVATE
        decompose_mesh::par_module
        decompose_mesh::fault_module
        meshfem3D::shared_module
        MPI::MPI_Fortran
    )
endif()

# ======================================================================
# Executable: xdecompose_mesh (serial)
# ======================================================================
add_executable(xdecompose_mesh
    program_decompose_mesh.f90
)
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

if (SPECFEM_ENABLE_HDF5)
    target_link_libraries(xdecompose_mesh
        PRIVATE
        decompose_mesh::hdf5_module
        meshfem3D::shared_hdf5_module
    )
endif()

if (SPECFEM_ENABLE_MPI)
    target_link_libraries(xdecompose_mesh PRIVATE MPI::MPI_Fortran)
endif()

if (APPLE)
    target_link_options(xdecompose_mesh PRIVATE "LINKER:-no_compact_unwind")
endif()

# ======================================================================
# Executable: xdecompose_mesh_mpi (parallel — MPI only)
# ======================================================================
if (SPECFEM_ENABLE_MPI)
    add_executable(xdecompose_mesh_mpi
        program_decompose_mesh_mpi.f90
    )
    target_link_libraries(xdecompose_mesh_mpi
        PRIVATE
        decompose_mesh::mpi_modules
        decompose_mesh::par_module
        decompose_mesh::fault_module
        meshfem3D::shared
        meshfem3D::shared_module
        meshfem3D::param_reader
        MPI::MPI_Fortran
        specfem::version
    )
    if (APPLE)
        target_link_options(xdecompose_mesh_mpi PRIVATE "LINKER:-no_compact_unwind")
    endif()
endif()

# ======================================================================
# Install
# ======================================================================
if (SPECFEM_INSTALL)
    install(TARGETS xdecompose_mesh
        RUNTIME DESTINATION .
        COMPONENT executables
    )
    if (SPECFEM_ENABLE_MPI)
        install(TARGETS xdecompose_mesh_mpi
            RUNTIME DESTINATION .
            COMPONENT executables
        )
    endif()
endif()
