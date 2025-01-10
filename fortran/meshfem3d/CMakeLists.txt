# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(MESHFEM3D LANGUAGES Fortran C)

# Set the Fortran standard to 95
set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/fortran/meshfem3d/modules)
include_directories("${CMAKE_BINARY_DIR}/fortran/meshfem3d/modules")

set(ADIOS "OFF" CACHE BOOL "Use ADIOS for I/O")

if (ADIOS)
  message(STATUS "ADIOS enabled for Meshfem3D Compilation")
else(ADIOS)
  message(STATUS "ADIOS disabled for Meshfem3D Compilation")
endif(ADIOS)

add_subdirectory(setup)
add_subdirectory(shared)
add_subdirectory(meshfem3D)
add_subdirectory(generate_databases)
