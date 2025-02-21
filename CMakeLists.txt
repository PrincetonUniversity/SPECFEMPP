cmake_minimum_required(VERSION 3.17.5)

project(SPECFEMPP VERSION 0.1.0 LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
option(HDF5_CXX_BUILD "Build HDF5 C++" ON)
option(VTK_CXX_BUILD "Build VTK C++" ON)
option(MPI_PARALLEL "MPI enabled" OFF)
option(BUILD_TESTS "Tests included" OFF)
option(BUILD_EXAMPLES "Examples included" ON)
option(ENABLE_SIMD "Enable SIMD" OFF)
option(ENABLE_PROFILING "Enable profiling" OFF)
option(SPECFEMPP_BINDING_PYTHON "Enable Python binding" OFF)
option(ENABLE_DOUBLE_PRECISION "Enable double precision" OFF)

# set(CMAKE_BUILD_TYPE Release)
set(CHUNK_SIZE 32)
set(NUM_CHUNKS 1)
set(NUM_THREADS 160)
set(NUM_VECTOR_LANES 1)

# Set binary output directories.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/archive)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# For external modules that need to be built suppress build output
set(FETCHCONTENT_QUIET TRUE)

if (SPECFEMPP_BINDING_PYTHON)
    message(STATUS "Adding -fPIC flag for Python binding.")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
            set(CMAKE_CXX_FLAGS "-fp-model=precise -fPIC")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            message(STATUS "Detected Intel classic compiler which will be deprecated soon.")
            message(STATUS "It is recommended you use IntelLLVM compiler.")
            set(CMAKE_CXX_FLAGS "-diag-disable=10441 -fp-model=precise -fPIC")
    else()
            set(CMAKE_CXX_FLAGS "-fPIC")
    endif()
else (SPECFEMPP_BINDING_PYTHON)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
            set(CMAKE_CXX_FLAGS "-fp-model=precise")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            message(STATUS "Detected Intel classic compiler which will be deprecated soon.")
            message(STATUS "It is recommended you use IntelLLVM compiler.")
            set(CMAKE_CXX_FLAGS "-diag-disable=10441 -fp-model=precise")
    endif()
endif (SPECFEMPP_BINDING_PYTHON)

# Check if MacOS
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(__APPLE__ TRUE)
    message(STATUS "macOS detected -- setting __APPLE__ TRUE")
else(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(__APPLE__ FALSE)
    message(STATUS "macOS not detected -- setting __APPLE__ FALSE")
endif()

if (DEFINED Kokkos_ENABLE_CUDA)
    if (Kokkos_ENABLE_CUDA)
        # message(STATUS "Setting CUDA variables")
        set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Using CUDA Lambda by default")
        set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "Using CUDA Relocatable device by default")
    endif()
endif()


# Install Kokkos as a dependency
## TODO: Add options for on utilizing in house builds
include(FetchContent)
FetchContent_Declare(
  kokkos
  URL https://github.com/kokkos/kokkos/archive/refs/tags/4.5.01.zip
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_MakeAvailable(kokkos)

FetchContent_Declare(
        yaml
        URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_MakeAvailable(yaml)

include_directories(BEFORE SYSTEM ${yaml_BINARY_DIR} ${yaml_SOURCE_DIR}/include)

find_package(VTK COMPONENTS
  CommonColor
  CommonCore
  FiltersSources
  InteractionStyle
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2
)


if (NOT VTK_FOUND)
  message(STATUS "VTK not found: ${VTK_NOT_FOUND_MESSAGE}")
  set(VTK_CXX_BUILD OFF)
else ()
  message(STATUS "    VTK:   ${VTK_LIBRARIES}")
endif()


# Try finding boost and if not found install.
find_package(Boost 1.85.0 COMPONENTS program_options filesystem system)


if (NOT ${Boost_FOUND})
    # Add boost lib sources
    set(BOOST_INCLUDE_LIBRARIES program_options filesystem system algorithm tokenizer preprocessor)
    set(BOOST_LIBS Boost::program_options Boost::filesystem Boost::system
                   Boost::algorithm Boost::tokenizer Boost::preprocessor)
    set(BOOST_ENABLE_CMAKE ON)
    set(BOOST_ENABLE_MPI OFF CACHE INTERNAL "Boost MPI Switch") # Assume outer variable
    set(BOOST_ENABLE_PYTHON OFF CACHE INTERNAL "Boost Python Switch") # Assume outer variable
    set(BOOST_BUILD_TESTING OFF CACHE BOOL INTERNAL "Boost Test Switch") # Disable testing for boost
    # The test flag is not really working... added it for completeness

    # Download and extract the boost library from GitHub
    set(BOOST_VERSION 1.87.0)
    message(STATUS "Downloading and extracting boost (${BOOST_VERSION}) library sources. This will take <1 min.")
    include(FetchContent)

    # Fetch boost from the Github release zip file to reduce download time
    FetchContent_Declare(
        Boost
        URL https://github.com/boostorg/boost/releases/download/boost-${BOOST_VERSION}/boost-${BOOST_VERSION}-cmake.tar.gz # downloading a zip release speeds up the download
        USES_TERMINAL_DOWNLOAD True
        GIT_PROGRESS TRUE
        DOWNLOAD_NO_EXTRACT FALSE
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    )
    FetchContent_MakeAvailable(Boost)

else()
    # Check which boost LIBRARY_DIRS to use
    set(BOOST_LIBS Boost::boost Boost::program_options Boost::filesystem Boost::system)
    message(STATUS "Boost libs/ and incs/:")
    message(STATUS "    LIB:   ${Boost_LIBRARY_DIRS}")
    message(STATUS "    INC:   ${Boost_INCLUDE_DIRS}")
    message(STATUS "    LIBSO: ${Boost_LIBRARIES}")
endif()

# Install HDF5 as a dependency if not found
find_package(HDF5 COMPONENTS CXX)

if (NOT ${HDF5_FOUND})
    message(STATUS "HDF5 not found. Building without HDF5.")
    set(HDF5_CXX_BUILD OFF)
else()
    message(STATUS "HDF5 libs/ and incs/:.")
    message(STATUS "    LIB:   ${HDF5_LIBRARIES}")
    message(STATUS "    INC:   ${HDF5_INCLUDE_DIRS}")
    message(STATUS "    LIBSO: ${HDF5_CXX_LIBRARIES}")
endif()


if (ENABLE_SIMD)
    message(STATUS "Enabling SIMD")
    add_definitions(-DENABLE_SIMD)
endif()

if (ENABLE_PROFILING)
    message(STATUS "Enabling profiling")
    add_definitions(-DENABLE_PROFILING)
endif()

if (ENABLE_DOUBLE_PRECISION)
    message(STATUS "Enabling double precision")
    set(TYPE_REAL "double")
else(ENABLE_DOUBLE_PRECISION)
    message(STATUS "Enabling single precision")
    set(TYPE_REAL "float")
endif(ENABLE_DOUBLE_PRECISION)

# Configure the setup headers
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/specfem_setup.hpp.in
               ${CMAKE_BINARY_DIR}/include/specfem_setup.hpp)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/constants.hpp.in
               ${CMAKE_BINARY_DIR}/include/constants.hpp)

# Add the include directories so that the generated files can be found
include_directories(include)
include_directories(${CMAKE_BINARY_DIR}/include)

# Set loglevel to STATUS if build type is debug
if (CMAKE_BUILD_TYPE MATCHES Release)
  set(SUBDIR_LOG_LEVEL NOTICE)
else()
  set(SUBDIR_LOG_LEVEL STATUS)
endif()

# ========== MESHFEM2D ==============
message(STATUS "Building meshfem2D")
set(_saved_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL ${SUBDIR_LOG_LEVEL})
add_subdirectory(fortran/meshfem2d)
set(CMAKE_MESSAGE_LOG_LEVEL ${_saved_CMAKE_MESSAGE_LOG_LEVEL})
# =================================

# ========== MESHFEM3D ==============
message(STATUS "Building xmeshfem3D & xgenerate_databases")
set(_saved_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL ${SUBDIR_LOG_LEVEL})
add_subdirectory(fortran/meshfem3d)
set(CMAKE_MESSAGE_LOG_LEVEL ${_saved_CMAKE_MESSAGE_LOG_LEVEL})
# =================================


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
        IO
        src/IO/fortranio/fortran_io.cpp
        src/IO/sources.cpp
        src/IO/receivers.cpp
        # Fortran 2D part
        src/IO/mesh/impl/fortran/dim2/mesh.cpp
        src/IO/mesh/impl/fortran/dim2/read_boundaries.cpp
        src/IO/mesh/impl/fortran/dim2/read_elements.cpp
        src/IO/mesh/impl/fortran/dim2/read_material_properties.cpp
        src/IO/mesh/impl/fortran/dim2/read_mesh_database.cpp
        src/IO/mesh/impl/fortran/dim2/read_interfaces.cpp
        src/IO/mesh/impl/fortran/dim2/read_parameters.cpp
        # Fortran 3D part
        src/IO/mesh/impl/fortran/dim3/mesh.cpp
        src/IO/mesh/impl/fortran/dim3/read_parameters.cpp
        src/IO/mesh/impl/fortran/dim3/read_coordinates.cpp
        src/IO/mesh/impl/fortran/dim3/read_partial_derivatives.cpp
        src/IO/mesh/impl/fortran/dim3/utilities.cpp
)

if (NOT HDF5_CXX_BUILD)
    target_compile_definitions(
        IO
        PUBLIC -DNO_HDF5
    )

    target_link_libraries(
        IO
        mesh
        source_class
        receiver_class
        yaml-cpp
        ${BOOST_LIBS}
        Kokkos::kokkos)
else()
    target_link_libraries(
        IO
        mesh
        source_class
        receiver_class
        Kokkos::kokkos
        yaml-cpp
        ${BOOST_LIBS}
        ${HDF5_LIBRARIES}
    )
endif()


add_library(
        point
        src/point/coordinates.cpp
        src/point/partial_derivatives.cpp
)

target_link_libraries(
        point
        Kokkos::kokkos
)

add_library(
        enumerations
        src/enumerations/medium.cpp
)

target_link_libraries(
        enumerations
        ${BOOST_LIBS}
)


add_library(
        edge
        src/edge/interface.cpp
)

target_link_libraries(
        edge
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
        message(STATUS "Compiling SPECFEM with MPI")
else()
        message(STATUS "Compiling SPECFEM without MPI")
endif(MPI_PARALLEL)

# add_library(
#         material_class
#         src/material/elastic_isotropic_material.cpp
#         src/material/acoustic_isotropic_material.cpp
# )

# target_link_libraries(
#         material_class
#         Kokkos::kokkos
#         specfem_mpi
# )

add_library(
        mesh
        # 2-D
        src/mesh/dim2/mesh.cpp
        src/mesh/dim2/boundaries/forcing_boundaries.cpp
        src/mesh/dim2/boundaries/absorbing_boundaries.cpp
        src/mesh/dim2/boundaries/acoustic_free_surface.cpp
        src/mesh/dim2/elements/tangential_elements.cpp
        src/mesh/dim2/elements/axial_elements.cpp
        # src/mesh/mpi_interfaces/mpi_interfaces.cpp
        src/mesh/dim2/materials/materials.cpp
        src/mesh/dim2/coupled_interfaces/interface_container.cpp
        src/mesh/dim2/coupled_interfaces/coupled_interfaces.cpp
        src/mesh/dim2/tags/tags.cpp
        # 3-D
        src/mesh/dim3/mesh.cpp
        src/mesh/dim3/boundaries/absorbing_boundary.cpp
        src/mesh/dim3/boundaries/free_surface.cpp
        src/mesh/dim3/element_types/element_types.cpp
        src/mesh/dim3/parameters/parameters.cpp
        src/mesh/dim3/parameters/parameters.cpp
        src/mesh/dim3/mapping/mapping.cpp
        src/mesh/dim3/materials/materials.cpp
        src/mesh/dim3/coordinates/coordinates.cpp
        src/mesh/dim3/partial_derivatives/partial_derivatives.cpp

)

target_link_libraries(
        mesh
        Kokkos::kokkos
        specfem_mpi
        # material_class
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
        point
)

add_library(
        read_seismogram
        src/IO/seismogram/reader.cpp
)

target_link_libraries(
        read_seismogram
        Kokkos::kokkos
        )

add_library(
        reader
        src/IO/wavefield/reader.cpp
        src/IO/property/reader.cpp
)

target_link_libraries(
        reader
        compute
        IO
        read_seismogram
)

add_library(
        algorithms
        src/algorithms/locate_point.cpp
)

target_link_libraries(
        algorithms
        Kokkos::kokkos
        jacobian
        point
)

add_library(
        source_time_function
        src/source_time_function/dirac.cpp
        src/source_time_function/dgaussian.cpp
        src/source_time_function/ricker.cpp
        src/source_time_function/external.cpp
)

target_link_libraries(
        source_time_function
        read_seismogram
        Kokkos::kokkos
        point
)

add_library(
        source_class
        src/source/source.cpp
        src/source/force_source.cpp
        src/source/moment_tensor_source.cpp
        src/source/adjoint_source.cpp
        src/source/external.cpp
)

target_link_libraries(
        source_class
        Kokkos::kokkos
        specfem_mpi
        # utilities
        quadrature
        source_time_function
        yaml-cpp
        point
        algorithms
        ${BOOST_LIBS}
)

add_library(
        receiver_class
        src/receiver/receiver.cpp
)

target_link_libraries(
        receiver_class
        specfem_mpi
        Kokkos::kokkos
        # utilities
        quadrature
        yaml-cpp
        ${BOOST_LIBS}
)

add_library(
        compute
        src/compute/compute_mesh.cpp
        src/compute/element_types/element_types.cpp
        src/compute/compute_partial_derivatives.cpp
        src/compute/compute_properties.cpp
        src/compute/compute_kernels.cpp
        src/compute/compute_sources.cpp
        src/compute/compute_receivers.cpp
        src/compute/coupled_interfaces.cpp
        src/compute/boundaries/impl/acoustic_free_surface.cpp
        src/compute/boundaries/impl/stacey.cpp
        src/compute/boundaries/boundaries.cpp
        src/compute/fields/fields.cpp
        src/compute/compute_boundary_values.cpp
        src/compute/assembly/assembly.cpp
        src/compute/assembly/compute_wavefield.cpp
)

target_link_libraries(
        compute
        enumerations
        quadrature
        mesh
        # material_class
        source_class
        jacobian
        point
        edge
        receiver_class
        Kokkos::kokkos
)

add_library(
        boundary_conditions
        src/boundary_conditions/boundary_conditions.cpp
)

target_link_libraries(
        boundary_conditions
        Kokkos::kokkos
        ${BOOST_LIBS}
)

add_library(
        medium
        src/medium/compute_mass_matrix.cpp
        src/medium/compute_stress_integrand.cpp
)

target_link_libraries(
        medium
        Kokkos::kokkos
        compute
        boundary_conditions
        enumerations
)

add_library(coupled_interface
        src/coupled_interface/coupled_interface.cpp
)

target_link_libraries(
        coupled_interface
        Kokkos::kokkos
        compute
)

add_library(
        kokkos_kernels
        src/kokkos_kernels/impl/compute_mass_matrix.cpp
        src/kokkos_kernels/impl/invert_mass_matrix.cpp
        src/kokkos_kernels/impl/divide_mass_matrix.cpp
        src/kokkos_kernels/impl/compute_seismogram.cpp
        src/kokkos_kernels/impl/compute_source_interaction.cpp
        src/kokkos_kernels/impl/compute_stiffness_interaction.cpp
        src/kokkos_kernels/impl/compute_material_derivatives.cpp
        src/kokkos_kernels/frechet_kernels.cpp
)

target_link_libraries(
        kokkos_kernels
        Kokkos::kokkos
        compute
        boundary_conditions
)

add_library(
        timescheme
        src/timescheme/timescheme.cpp
        src/timescheme/newmark.cpp
)

target_link_libraries(
        timescheme
        Kokkos::kokkos
        yaml-cpp
        compute
)

add_library(
        solver
        src/solver/time_marching.cpp
)

target_link_libraries(
        solver
        Kokkos::kokkos
        timescheme
        kokkos_kernels
        medium
)

add_library(
        writer
        src/IO/seismogram/writer.cpp
        src/IO/wavefield/writer.cpp
        src/IO/kernel/writer.cpp
        src/IO/property/writer.cpp
)

target_link_libraries(
        writer
        compute
        receiver_class
        IO
)

add_library(
        periodic_tasks
        src/periodic_tasks/plot_wavefield.cpp
)

if (NOT VTK_CXX_BUILD)
        target_compile_definitions(
                periodic_tasks
                PUBLIC -DNO_VTK
        )

        target_link_libraries(
                periodic_tasks
                compute
        )
else ()
        target_link_libraries(
                periodic_tasks
                compute
                ${VTK_LIBRARIES}
                )

        # Only define the __APPLE__ compile definition if it is defined
        if (__APPLE__)
        target_compile_definitions(periodic_tasks PRIVATE __APPLE__)
        endif(__APPLE__)
endif()

add_library(
        parameter_reader
        src/parameter_parser/run_setup.cpp
        # src/parameter_parser/solver/solver.cpp
        src/parameter_parser/time_scheme/time_scheme.cpp
        src/parameter_parser/database_configuration.cpp
        src/parameter_parser/header.cpp
        src/parameter_parser/quadrature.cpp
        src/parameter_parser/receivers.cpp
        src/parameter_parser/writer/seismogram.cpp
        src/parameter_parser/setup.cpp
        src/parameter_parser/writer/wavefield.cpp
        src/parameter_parser/writer/plot_wavefield.cpp
        src/parameter_parser/writer/kernel.cpp
        src/parameter_parser/writer/property.cpp
)

target_link_libraries(
        parameter_reader
        quadrature
        timescheme
        receiver_class
        yaml-cpp
        writer
        reader
        kokkos_kernels
        medium
        solver
        ${BOOST_LIBS}
)

add_library(
        execute
        src/execute.cpp
)

target_link_libraries(
        execute
        specfem_mpi
        Kokkos::kokkos
        mesh
        quadrature
        compute
        source_class
        parameter_reader
        receiver_class
        writer
        periodic_tasks
        reader
        medium
        coupled_interface
        kokkos_kernels
        solver
        ${BOOST_LIBS}
)

add_executable(
        specfem2d
        src/specfem2d.cpp
)

target_link_libraries(
        specfem2d
        execute
)

add_executable(
        specfem3d
        src/specfem3d.cpp
)

target_link_libraries(
        specfem3d
        specfem_mpi
        IO
        Kokkos::kokkos
        mesh
        quadrature
        compute
        source_class
        parameter_reader
        receiver_class
        writer
        periodic_tasks
        reader
        medium
        coupled_interface
        kokkos_kernels
        solver
        ${BOOST_LIBS}
        yaml-cpp
)


# Include tests
if (BUILD_TESTS)
        message(STATUS "Including tests.")
        add_subdirectory(tests/unit-tests)
endif()

if (BUILD_EXAMPLES)
        message(STATUS "Including examples.")
        add_subdirectory(examples)
endif()

# Doxygen

# look for Doxygen package
find_package(Doxygen)

if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message(STATUS "Doxygen build started")

    # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
    add_custom_target( docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message(STATUS "Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

if (SPECFEMPP_USE_SKBUILD AND EXISTS ${SKBUILD_SCRIPTS_DIR})
    install(TARGETS specfem2d DESTINATION ${SKBUILD_SCRIPTS_DIR})
    install(FILES ${CMAKE_BINARY_DIR}/bin/xmeshfem2D DESTINATION ${SKBUILD_SCRIPTS_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
endif ()

if (SPECFEMPP_BINDING_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    python_add_library(_core MODULE src/python/core.cpp WITH_SOABI)
    target_link_libraries(
        _core PRIVATE
        execute
        ${BOOST_LIBS}
        pybind11::headers
    )
    target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
    install(TARGETS _core DESTINATION specfempp_core)
endif (SPECFEMPP_BINDING_PYTHON)
