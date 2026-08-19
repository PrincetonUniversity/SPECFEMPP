project(meshfem3D_globe LANGUAGES Fortran C CXX)

cmake_minimum_required(VERSION 3.10)

enable_language(Fortran)
enable_language(C)
enable_language(CXX)

set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

option(SPECFEM_ENABLE_DOUBLE_PRECISION "Enable double precision" OFF)
option(WITH_ADIOS "Build with ADIOS support" OFF)
option(BEOWULF_CLUSTER "Build for Beowulf cluster" OFF)
option(SPECFEM_INSTALL "Install SPECFEM" ON)

if(NOT SPECFEM_ENABLE_MPI)
  message(FATAL_ERROR
    "meshfem3D_globe requires SPECFEM_ENABLE_MPI=ON because the "
    "SPECFEM3D_GLOBE mesher sources use Fortran MPI.")
endif()

if(NOT TARGET MPI::MPI_Fortran)
  find_package(MPI REQUIRED COMPONENTS Fortran)
endif()

if(NOT TARGET MPI::MPI_Fortran)
  message(FATAL_ERROR
    "meshfem3D_globe requires MPI::MPI_Fortran, but Fortran MPI was not found.")
endif()

message(STATUS "-- Building meshfem3D_globe with Fortran MPI support")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/archive)

set(SETUP_DIR ${CMAKE_CURRENT_BINARY_DIR}/setup)

include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})
include_directories(${SETUP_DIR})

if(SPECFEM_ENABLE_DOUBLE_PRECISION)
  message(STATUS "-- Enabling meshfem3D_globe double precision")
  set(CUSTOM_REAL "SIZE_DOUBLE")
  set(CUSTOM_MPI_TYPE "MPI_DOUBLE_PRECISION")
else()
  message(STATUS "-- Enabling meshfem3D_globe single precision")
  set(CUSTOM_REAL "SIZE_REAL")
  set(CUSTOM_MPI_TYPE "MPI_REAL")
endif()

if(WITH_ADIOS)
  message(STATUS "ADIOS enabled for meshfem3D_globe compilation")
else()
  message(STATUS "ADIOS disabled for meshfem3D_globe compilation")
endif()

add_subdirectory(setup)
add_subdirectory(shared)
add_subdirectory(meshfem3D)
