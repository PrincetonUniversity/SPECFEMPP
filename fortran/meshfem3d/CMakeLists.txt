# Project name and language
project(meshfem3D LANGUAGES Fortran C)

# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# languages
enable_language(Fortran)
enable_language(C)

# Set the Fortran standard to 95
set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

unset(WITH_MPI)
unset(WITH_SCOTCH)

option(ENABLE_DOUBLE_PRECISION "Enable double precision" OFF)
option(WITH_MPI "Build with MPI support" OFF)
option(WITH_SCOTCH "Build with SCOTCH support" OFF)
option(WITH_ADIOS "Build with ADIOS support" OFF)
option(FORCE_VECTORIZATION "Force vectorization" OFF)
option(BEOWULF_CLUSTER "Build for Beowulf cluster" OFF)
option(SPECFEM_INSTALL "Install SPECFEM" ON)


if (SPECFEM_INSTALL)
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set_property(CACHE CMAKE_INSTALL_PREFIX PROPERTY VALUE "${CMAKE_SOURCE_DIR}/bin")
    message(STATUS "CMAKE_INSTALL_PREFIX not set. Defaulting to ${CMAKE_INSTALL_PREFIX}")
    else()
    message(STATUS "User-defined CMAKE_INSTALL_PREFIX set to ${CMAKE_INSTALL_PREFIX}")
    endif()
else(SPECFEM_INSTALL)
    message(STATUS "SPECFEM_INSTALL is OFF. No executable directory will be created.")
endif()


# For the runtime we use the same directory as the main project
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# For output files we use the current directory
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/archive)

# Custom variables
set(SETUP_DIR ${CMAKE_CURRENT_BINARY_DIR}/setup)

# include_directories
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})
include_directories(${SETUP_DIR})

# Setting values dependent on options
if (ENABLE_DOUBLE_PRECISION)
    message(STATUS "-- Enabling double precision")
    # These are used by Meshfem
    set(CUSTOM_REAL "SIZE_DOUBLE")
    set(CUSTOM_MPI_TYPE "MPI_DOUBLE_PRECISION")

else(ENABLE_DOUBLE_PRECISION)
    message(STATUS "-- Enabling single precision")
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
