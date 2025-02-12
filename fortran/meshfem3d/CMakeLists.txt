# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(meshfem3D LANGUAGES Fortran C)

# Set the Fortran standard to 95
set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# languages
enable_language(Fortran)
enable_language(C)

option(ENABLE_DOUBLE_PRECISION "Enable double precision" OFF)
option(WITH_MPI "Build with MPI support" OFF)
option(WITH_SCOTCH "Build with SCOTCH support" OFF)
option(WITH_ADIOS "Build with ADIOS support" OFF)

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/fortran/meshfem3d/modules)
include_directories("${CMAKE_BINARY_DIR}/fortran/meshfem3d/modules")

# Setting values dependent on options
if (ENABLE_DOUBLE_PRECISION)
    message("-- Enabling double precision")
    # These are used by Meshfem
    set(CUSTOM_REAL "SIZE_DOUBLE")
    set(CUSTOM_MPI_TYPE "MPI_DOUBLE_PRECISION")

else(ENABLE_DOUBLE_PRECISION)
    message("-- Enabling single precision")
    # These are used by Meshfem
    set(CUSTOM_REAL "SIZE_REAL")
    set(CUSTOM_MPI_TYPE "MPI_REAL")
endif(ENABLE_DOUBLE_PRECISION)


if (WITH_ADIOS)
  message(STATUS "ADIOS enabled for Meshfem3D Compilation")
else(WITH_ADIOS)
  message(STATUS "ADIOS disabled for Meshfem3D Compilation")
endif(WITH_ADIOS)

add_subdirectory(setup)
add_subdirectory(shared)
add_subdirectory(meshfem3D)
add_subdirectory(generate_databases)
