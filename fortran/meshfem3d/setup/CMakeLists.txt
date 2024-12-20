# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(MESHFEM3D_SETUP LANGUAGES Fortran C)

# Set the Fortran standard to 95
set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/fortran/meshfem3d/modules)
include_directories("${CMAKE_BINARY_DIR}/fortran/meshfem3d/modules")

add_library(meshfem3D_version INTERFACE version.fh)
add_library(meshfem3D::version ALIAS meshfem3D_version)
target_include_directories(meshfem3D_version INTERFACE ./)

add_library(meshfem3D_constants INTERFACE constants.h)
add_library(meshfem3D::constants ALIAS meshfem3D_constants)
target_include_directories(meshfem3D_constants INTERFACE ./)

add_library(meshfem3D_config INTERFACE config.h)
add_library(meshfem3D::config ALIAS meshfem3D_config)
target_include_directories(meshfem3D_config INTERFACE ./)
