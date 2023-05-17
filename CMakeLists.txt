cmake_minimum_required(VERSION 3.17.5)

project(specfem2d_kokkos VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
option(MPI_PARALLEL "MPI enabled" OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        set(CMAKE_CXX_FLAGS "-fp-model=precise")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        message("-- Detected Intel classic compiler which will be deprecated soon.")
        message("-- It is recommended you use IntelLLVM compiler.")
        set(CMAKE_CXX_FLAGS "-diag-disable=10441 -fp-model=precise")
endif()

if (DEFINED Kokkos_ENABLE_CUDA)
        if (Kokkos_ENABLE_CUDA)
                set(Kokkos_ENABLE_CUDA_LAMBDA ON)
                set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON)
        endif()
endif()

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

configure_file(constants.hpp.in constants.hpp)

include_directories(include)
include_directories(${CMAKE_BINARY_DIR})

# Build specfem2d libraries
add_library(
        quadrature
        src/quadrature/quadrature.cpp
        src/quadrature/gll/gll_utils.cpp
        src/quadrature/gll/gll_library.cpp
        src/quadrature/gll/lagrange_poly.cpp
        src/quadrature/gll/gll.cpp
)

target_link_libraries(
        quadrature
        Kokkos::kokkos
)

add_library(
        fortranio
        src/fortranio/fortran_io.cpp
)

target_link_libraries(
        fortranio
        Boost::boost
        Kokkos::kokkos
)

add_library(
        specfem_mpi
        src/specfem_mpi/specfem_mpi.cpp
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
        operators
        src/mathematical_operators/mathematical_operators.cpp
)

target_link_libraries(
        operators
        Kokkos::kokkos
)

add_library(
        material_class
        src/material/elastic_material.cpp
        src/material/acoustic_material.cpp
)

target_link_libraries(
        material_class
        Kokkos::kokkos
        specfem_mpi
)

add_library(
        mesh
        src/mesh/IO/fortran/read_mesh_database.cpp
        src/mesh/IO/fortran/read_material_properties.cpp
        src/mesh/boundaries/forcing_boundaries.cpp
        src/mesh/boundaries/absorbing_boundaries.cpp
        src/mesh/elements/tangential_elements.cpp
        src/mesh/elements/axial_elements.cpp
        src/mesh/properties/properties.cpp
        src/mesh/mpi_interfaces/mpi_interfaces.cpp
        src/mesh/material_indic/material_indic.cpp
        src/mesh/surfaces/acoustic_free_surface.cpp
        src/mesh/mesh.cpp
)

target_link_libraries(
        mesh
        Kokkos::kokkos
        specfem_mpi
        material_class
        fortranio
        yaml-cpp
)

add_library(
        jacobian
        src/jacobian/shape_functions.cpp
        src/jacobian/jacobian.cpp
)

target_link_libraries(
        jacobian
        Kokkos::kokkos
)

add_library(
        utilities
        src/utilities/utilities.cpp
)

target_link_libraries(
        utilities
        jacobian
        Kokkos::kokkos
        specfem_mpi
)

add_library(
        source_time_function
        src/source_time_function/dirac.cpp
)

target_link_libraries(
        source_time_function
        Kokkos::kokkos
)

add_library(
        source_class
        src/source/source.cpp
        src/source/force_source.cpp
        src/source/moment_tensor_source.cpp
        src/source/read_sources.cpp
)

target_link_libraries(
        source_class
        Kokkos::kokkos
        specfem_mpi
        utilities
        quadrature
        source_time_function
        yaml-cpp
        Boost::boost
)

add_library(
        receiver_class
        src/receiver/receiver.cpp
        src/receiver/read_receiver.cpp
)

target_link_libraries(
        receiver_class
        specfem_mpi
        Kokkos::kokkos
        utilities
        quadrature
        yaml-cpp
        Boost::boost
)

add_library(
        compute
        src/compute/compute.cpp
        src/compute/compute_partial_derivatives.cpp
        src/compute/compute_properties.cpp
        src/compute/compute_sources.cpp
        src/compute/compute_receivers.cpp
)

target_link_libraries(
        compute
        quadrature
        material_class
        source_class
        jacobian
        receiver_class
        Kokkos::kokkos
)

# there seems to be a bug when compiling compute with -O3 mode using the icpx compiler.
# set_target_properties(compute PROPERTIES COMPILE_OPTIONS "$<$<CONFIG:RELEASE>:-fp-model=precise>")

add_library(
        domain
        src/domain/elastic_domain.cpp
)

target_link_libraries(
        domain
        compute
        quadrature
        operators
        Kokkos::kokkos
)

add_library(
        timescheme
        src/timescheme/timescheme.cpp
        src/timescheme/newmark.cpp
)

target_link_libraries(
        timescheme
        domain
)

add_library(
        writer
        src/writer/seismogram.cpp
)

target_link_libraries(
        writer
        compute
        receiver_class
)

add_library(
        solver
        src/solver/time_marching.cpp
)

target_link_libraries(
        solver
        domain
        timescheme
        writer
)

add_library(
        parameter_reader
        src/parameter_parser/run_setup.cpp
        src/parameter_parser/solver/solver.cpp
        src/parameter_parser/solver/time_marching.cpp
        src/parameter_parser/database_configuration.cpp
        src/parameter_parser/header.cpp
        src/parameter_parser/quadrature.cpp
        src/parameter_parser/receivers.cpp
        src/parameter_parser/seismogram.cpp
        src/parameter_parser/writer.cpp
        src/parameter_parser/setup.cpp
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
        Kokkos::kokkos
        mesh
        quadrature
        compute
        source_class
        parameter_reader
        domain
        solver
        receiver_class
        writer
        Boost::program_options
)

# Include tests
add_subdirectory(tests/unit-tests)
# add_subdirectory(tests/regression-tests)
add_subdirectory(examples)

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
