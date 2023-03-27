cmake_minimum_required(VERSION 3.17.5)

project(specfem2d_kokkos VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
option(MPI_PARALLEL "MPI enabled" OFF)

# Install Kokkos as a dependency
## TODO: Add options for on utilizing in house builds
include(FetchContent)
FetchContent_Declare(
  kokkos
  URL https://github.com/kokkos/kokkos/archive/refs/tags/4.0.00.zip
)
FetchContent_MakeAvailable(kokkos)

FetchContent_Declare(
        yaml
	URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.7.0.zip
)
FetchContent_MakeAvailable(yaml)
include_directories(BEFORE SYSTEM ${yaml_SOURCE_DIR} ${yaml_BINARY_DIR}/include)

add_subdirectory(boost-cmake)

# Build specfem2d libraries
## TODO: Reorgnanize into separate directories
add_library(
        gll_utils
        src/gll_utils.cpp
)

target_link_libraries(
        gll_utils PUBLIC
        Kokkos::kokkos
)

add_library(
        gll_library
        src/gll_library.cpp
)

target_link_libraries(
        gll_library
        gll_utils
)

add_library(
        lagrange
        src/lagrange_poly.cpp
)

target_link_libraries(
        lagrange
        gll_library
        Kokkos::kokkos
)

add_library(
        quadrature
        src/quadrature.cpp
)

target_link_libraries(
        quadrature
        lagrange
        gll_library
        Kokkos::kokkos
)

add_library(
        fortranio
        src/fortran_IO.cpp
        include/fortran_IO.h
)

target_link_libraries(
        fortranio
        PUBLIC Boost::boost
)

add_library(
        specfem_mpi
        src/specfem_mpi.cpp
)

if (MPI_PARALLEL)
        target_compile_definitions(
                specfem_mpi
                PUBLIC -DMPI_PARALLEL
        )
        message("-- Compiling SPECFEM with MPI")
else()
        message("-- Compiling SPECFEM without MPI")
endif(MPI_PARALLEL)

add_library(
        material_class
        src/material.cpp
)

target_link_libraries(
        material_class
        Kokkos::kokkos
        specfem_mpi
)

add_library(
        material_reader
        src/read_material_properties.cpp
)

target_link_libraries(
        material_reader
        material_class
        specfem_mpi
        fortranio
)

add_library(
        boundaries
        src/boundaries.cpp
)

target_link_libraries(
        boundaries
        Kokkos::kokkos
        specfem_mpi
)

add_library(
        surfaces
        src/surfaces.cpp
)

target_link_libraries(
        surfaces
        Kokkos::kokkos
        specfem_mpi
)

add_library(
        elements
        src/elements.cpp
)

target_link_libraries(
        elements
        Kokkos::kokkos
        specfem_mpi
)

add_library(
        mpi_interfaces
        src/mpi_interfaces.cpp
)

target_link_libraries(
        mpi_interfaces
        Kokkos::kokkos
        specfem_mpi
)

add_library(
        database_reader
        src/read_mesh_database.cpp
)

target_link_libraries(
        database_reader
        specfem_mpi
        fortranio
        Kokkos::kokkos
        mpi_interfaces
)

add_library(
        mesh_properties
        src/mesh_properties.cpp
)

target_link_libraries(
        mesh_properties
        specfem_mpi
)

add_library(
        material_indic
        src/material_indic.cpp
)

target_link_libraries(
        material_indic
        specfem_mpi
        Kokkos::kokkos
)

add_library(
        mesh
        src/mesh.cpp
)

target_link_libraries(
        mesh
        Kokkos::kokkos
        specfem_mpi
        material_class
        database_reader
        material_reader
        mesh_properties
        material_indic
        boundaries
        elements
        surfaces
        yaml-cpp
)

add_library(
        shape_functions
        src/shape_functions.cpp
)

target_link_libraries(
        shape_functions
        Kokkos::kokkos
)

add_library(
        jacobian
        src/jacobian.cpp
)

target_link_libraries(
        jacobian
        shape_functions
        Kokkos::kokkos
)

add_library(
        utilities
        src/utils.cpp
)

target_link_libraries(
        utilities
        jacobian
        Kokkos::kokkos
        specfem_mpi
)

add_library(
        source_time_function
        src/source_time_function.cpp
)

target_link_libraries(
        source_time_function
        Kokkos::kokkos
        yaml-cpp
)

add_library(
        source_class
        src/source.cpp
)

target_link_libraries(
        source_class
        Kokkos::kokkos
        specfem_mpi
        utilities
        quadrature
        lagrange
        source_time_function
        yaml-cpp
)

add_library(
        receiver_class
        src/receiver.cpp
)

target_link_libraries(
        receiver_class
        specfem_mpi
        Kokkos::kokkos
        utilities
        quadrature
        lagrange
)

add_library(
        source_reader
        src/read_sources.cpp
)

target_link_libraries(
        source_reader
        source_class
        receiver_class
        yaml-cpp
        utilities
        Boost::system
)

add_library(
        compute
        src/compute.cpp
        src/compute_partial_derivatives.cpp
        src/compute_properties.cpp
        src/compute_sources.cpp
        src/compute_receivers.cpp
)

target_link_libraries(
        compute
        quadrature
        material_class
        source_class
        jacobian
        shape_functions
        receiver_class
        Kokkos::kokkos
)

add_library(
        domain
        src/domain.cpp
)

target_link_libraries(
        domain
        compute
        quadrature
        Kokkos::kokkos
)

add_library(
        timescheme
        src/timescheme.cpp
)

target_link_libraries(
        timescheme
        domain
)

add_library(
        writer
        src/writer.cpp
)

target_link_libraries(
        writer
        compute
        receiver_class
)

add_library(
        solver
        src/solver.cpp
)

target_link_libraries(
        solver
        domain
        timescheme
        writer
)

add_library(
        parameter_reader
        src/parameter_parser.cpp
)

target_link_libraries(
        parameter_reader
        quadrature
        timescheme
        receiver_class
        yaml-cpp
        Boost::filesystem
)

add_executable(
        specfem2d
        src/specfem2d.cpp
)

target_link_libraries(
        specfem2d
        material_class
        specfem_mpi
        database_reader
        Kokkos::kokkos
        yaml-cpp
        mesh
        quadrature
        compute
        source_class
        source_reader
        parameter_reader
        domain
        solver
        utilities
        receiver_class
        writer
        Boost::program_options
)

# Include tests
add_subdirectory(tests/unittests)

# Doxygen

# look for Doxygen package
find_package(Doxygen)

if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
    add_custom_target( docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)
