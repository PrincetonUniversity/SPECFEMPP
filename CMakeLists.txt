cmake_minimum_required(VERSION 3.17.5)

project(SPECFEMPP VERSION 0.1.0 LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 17)

# SPECFEM options
option(SPECFEM_ENABLE_ADIOS2 "Enable ADIOS2 I/O in SPECFEM++" OFF)
option(SPECFEM_ENABLE_HDF5 "Enable HDF5 I/O in SPECFEM++" OFF)
option(SPECFEM_ENABLE_HDF5_FORCE_INSTALL "Force install HDF5 dependency if SPECFEM_ENABLE_HDF5 is ON even if system HDF5 is found" OFF)
option(SPECFEM_ENABLE_NPZ "Enable NPZ I/O in SPECFEM++" OFF)
option(SPECFEM_ENABLE_ZLIB_FORCE_INSTALL "Force install ZLIB dependency even if system ZLIB is found" OFF)
option(SPECFEM_ENABLE_VTK "Enable VTK visualization in SPECFEM++" ON)
option(SPECFEM_ENABLE_MPI "MPI enabled" OFF)
option(SPECFEM_BUILD_TESTS "Tests included" OFF)
option(SPECFEM_ENABLE_SIMD "Enable SIMD" OFF)
option(SPECFEM_ENABLE_PROFILING "Enable profiling" OFF)
option(SPECFEM_BINDING_PYTHON "Enable Python binding" OFF)
option(SPECFEM_INSTALL "Creates clean executable directory. Location Controlled by CMAKE_INSTALL_PREFIX" ON)
option(SPECFEM_ENABLE_DOUBLE_PRECISION "Enable double precision" OFF)
option(SPECFEM_BUILD_BENCHMARKS "Benchmarks included" ON)
option(SPECFEM_ENABLE_UNITY_BUILD "Enable unity build to speed up compilation" ON)
option(SPECFEMPP_CONFIGURE_TESTS "Configure test files used to generate data for unit tests" OFF)

# set(CMAKE_BUILD_TYPE Release)
set(CHUNK_SIZE 32)
set(NUM_CHUNKS 1)
set(NUM_THREADS 160)
set(NUM_VECTOR_LANES 1)

# Set my default CMAKE_INSTALL_PREFIX
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

# Set binary output directories.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/archive)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# For the installation of dependencies
include(FetchContent)

# For external modules that need to be built suppress build output
set(FETCHCONTENT_QUIET TRUE)

# Enable unity build for faster compilation
if(SPECFEM_ENABLE_UNITY_BUILD)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "Unity build disabled for Debug build type.")
        set(CMAKE_UNITY_BUILD OFF)
    else (CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "Unity build enabled - this will speed up compilation")
        set(CMAKE_UNITY_BUILD ON)
        set(CMAKE_UNITY_BUILD_BATCH_SIZE 8)
    endif()
endif()

if (SPECFEM_BINDING_PYTHON)
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
else (SPECFEM_BINDING_PYTHON)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
            set(CMAKE_CXX_FLAGS "-fp-model=precise")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            message(STATUS "Detected Intel classic compiler which will be deprecated soon.")
            message(STATUS "It is recommended you use IntelLLVM compiler.")
            set(CMAKE_CXX_FLAGS "-diag-disable=10441 -fp-model=precise")
    endif()
endif (SPECFEM_BINDING_PYTHON)

# Check if MacOS
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(__APPLE__ TRUE)
    message(STATUS "macOS detected -- setting __APPLE__ TRUE")
else(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(__APPLE__ FALSE)
    message(STATUS "macOS not detected -- setting __APPLE__ FALSE")
endif()

# Check if IntelLLVM
if(${CMAKE_CXX_COMPILER_ID} MATCHES "IntelLLVM")
    set(__INTEL_LLVM__ TRUE)
    message(STATUS "IntelLLVM detected -- setting __INTEL_LLVM__ TRUE")
else(${CMAKE_CXX_COMPILER_ID} MATCHES "IntelLLVM")
    set(__INTEL_LLVM__ FALSE)
    message(STATUS "IntelLLVM not detected -- setting __INTEL_LLVM__ FALSE")
endif()

# Check if Windows
# https://cmake.org/cmake/help/latest/variable/MSVC.html
if(MSVC)
    set(__MSVC__ TRUE)
    message(STATUS "MSVC detected -- setting __MSVC__ TRUE")
else(MSVC)
    set(__MSVC__ FALSE)
    message(STATUS "MSVC not detected -- setting __MSVC__ FALSE")
endif()

# Install Kokkos as a dependency
include("${CMAKE_SOURCE_DIR}/cmake/kokkos.cmake")

# Install yaml-cpp as a dependency
include("${CMAKE_SOURCE_DIR}/cmake/yaml.cmake")

# Find or install boost library function
include("${CMAKE_SOURCE_DIR}/cmake/boost.cmake")

# Find HDF5
include("${CMAKE_SOURCE_DIR}/cmake/hdf5.cmake")

# Find or install NPZ library function
include("${CMAKE_SOURCE_DIR}/cmake/npz.cmake")

# Find or install ADIOS2 library function
include("${CMAKE_SOURCE_DIR}/cmake/adios2.cmake")

# Include VTK if available (MUST BE AFTER ADIOS2, VTK is smarter about finding
# the pugixml dependency than ADIOS2)
include("${CMAKE_SOURCE_DIR}/cmake/vtk.cmake")

if (SPECFEM_ENABLE_SIMD)
    message(STATUS "Enabling SIMD")
    add_compile_definitions(SPECFEM_ENABLE_SIMD)
endif()

if (SPECFEM_ENABLE_PROFILING)
    message(STATUS "Enabling profiling")
    add_compile_definitions(SPECFEM_ENABLE_PROFILING)
endif()

if (SPECFEM_ENABLE_DOUBLE_PRECISION)
    message(STATUS "Enabling double precision")
    set(TYPE_REAL "double")
else(SPECFEM_ENABLE_DOUBLE_PRECISION)
    message(STATUS "Enabling single precision")
    set(TYPE_REAL "float")
endif(SPECFEM_ENABLE_DOUBLE_PRECISION)

# Configure the setup headers
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/specfem_setup.hpp.in
               ${CMAKE_BINARY_DIR}/include/specfem_setup.hpp)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/constants.hpp.in
               ${CMAKE_BINARY_DIR}/include/constants.hpp)

# Add the configure files to the clean target
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
    ${CMAKE_BINARY_DIR}/include/specfem_setup.hpp
    ${CMAKE_BINARY_DIR}/include/constants.hpp
)

# Add the include directories so that the generated files can be found
include_directories(core)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)


message(STATUS "INCLUDE DIRECTORIES:")
message(STATUS "    core: ${CMAKE_CURRENT_SOURCE_DIR}/core")
message(STATUS "    include: ${CMAKE_SOURCE_DIR}/include")
message(STATUS "    binary: ${CMAKE_BINARY_DIR}/include")


# Set loglevel to STATUS if build type is debug
if (CMAKE_BUILD_TYPE MATCHES Release)
  set(SUBDIR_LOG_LEVEL NOTICE)
else()
  set(SUBDIR_LOG_LEVEL STATUS)
endif()

# ========== MESHFEM2D ==============
message(STATUS "Configuring meshfem2D")
set(_saved_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL ${SUBDIR_LOG_LEVEL})
add_subdirectory(fortran/meshfem2d)
set(CMAKE_MESSAGE_LOG_LEVEL ${_saved_CMAKE_MESSAGE_LOG_LEVEL})
# =================================

# ========== MESHFEM3D ==============
message(STATUS "Configuring xmeshfem3D & xgenerate_databases")
set(_saved_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL ${SUBDIR_LOG_LEVEL})
add_subdirectory(fortran/meshfem3d)
set(CMAKE_MESSAGE_LOG_LEVEL ${_saved_CMAKE_MESSAGE_LOG_LEVEL})
# =================================

# Add module subdirectories
add_subdirectory(core)

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
        io
        # IO libraries
        src/io/NPY/impl/npy_header.cpp
        # SPECFEM IO
        src/io/fortranio/fortran_io.cpp
        src/io/sources/dim2/read_sources.cpp
        src/io/sources/dim3/read_sources.cpp
        src/io/receivers/dim2/read_receivers.cpp
        src/io/receivers/dim3/read_receivers.cpp
        # Fortran 2D part
        src/io/mesh/impl/fortran/dim2/mesh.cpp
        src/io/mesh/impl/fortran/dim2/read_boundaries.cpp
        src/io/mesh/impl/fortran/dim2/read_elements.cpp
        src/io/mesh/impl/fortran/dim2/read_material_properties.cpp
        src/io/mesh/impl/fortran/dim2/read_mesh_database.cpp
        src/io/mesh/impl/fortran/dim2/read_interfaces.cpp
        src/io/mesh/impl/fortran/dim2/read_parameters.cpp
        src/io/mesh/impl/fortran/dim2/read_adjacency_graph.cpp
        # Fortran 3D part
        src/io/mesh/impl/fortran/dim3/generate_database/mesh.cpp
        src/io/mesh/impl/fortran/dim3/generate_database/read_parameters.cpp
        src/io/mesh/impl/fortran/dim3/generate_database/read_coordinates.cpp
        src/io/mesh/impl/fortran/dim3/generate_database/read_jacobian_matrix.cpp
        src/io/mesh/impl/fortran/dim3/generate_database/utilities.cpp
        # Fortran 3D meshfem part
        src/io/mesh/impl/fortran/dim3/meshfem3d/mesh.cpp
        src/io/mesh/impl/fortran/dim3/meshfem3d/read_boundaries.cpp
        src/io/mesh/impl/fortran/dim3/meshfem3d/read_control_nodes.cpp
        src/io/mesh/impl/fortran/dim3/meshfem3d/read_materials.cpp
        src/io/mesh/impl/fortran/dim3/meshfem3d/read_mpi_interfaces.cpp
        src/io/mesh/impl/fortran/dim3/meshfem3d/read_pml_boundaries.cpp
        src/io/mesh/impl/fortran/dim3/meshfem3d/read_adjacency_graph.cpp
)

target_link_libraries(
        io
        mesh
        source_class
        specfem::receivers
        yaml-cpp
        boost
        $<$<BOOL:${SPECFEM_ENABLE_NPZ}>:zlib>
        $<$<BOOL:${SPECFEM_ENABLE_HDF5}>:hdf5>
        $<$<BOOL:${SPECFEM_ENABLE_ADIOS2}>:adios2>
)

target_compile_definitions(
        io
        PUBLIC
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_NPZ}>>:-DNO_NPZ>
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_HDF5}>>:-DNO_HDF5>
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_ADIOS2}>>:-DNO_ADIOS2>
)

add_library(
        enumerations
        src/enumerations/medium.cpp
        src/enumerations/wavefield.cpp
        src/enumerations/dim2/connections.cpp
        src/enumerations/dim3/connections.cpp
        src/enumerations/dim3/mesh_entities.cpp
        src/enumerations/dim2/mesh_entities.cpp
        src/enumerations/coupled_interface.cpp
)

target_link_libraries(
        enumerations
        ${BOOST_LIBS}
        Kokkos::kokkos
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

target_compile_definitions(
        specfem_mpi
        PUBLIC
        $<$<BOOL:${SPECFEM_ENABLE_MPI}>:-DMPI_PARALLEL>
)

if (SPECFEM_ENABLE_MPI)
    message(STATUS "Compiling SPECFEM with MPI")
else()
    message(STATUS "Compiling SPECFEM without MPI")
endif()

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
        src/mesh/dim2/coupled_interfaces/interface_container.cpp
        src/mesh/dim2/coupled_interfaces/coupled_interfaces.cpp
        src/mesh/dim2/tags/tags.cpp
        src/mesh/dim2/adjacency_graph/assert_symmetry.cpp
        # 3-D
        src/mesh/dim3/generate_database/mesh.cpp
        src/mesh/dim3/generate_database/boundaries/absorbing_boundary.cpp
        src/mesh/dim3/generate_database/boundaries/acoustic_free_surface.cpp
        src/mesh/dim3/generate_database/coupled_interfaces/coupled_interfaces.cpp
        src/mesh/dim3/generate_database/element_types/element_types.cpp
        src/mesh/dim3/generate_database/parameters/parameters.cpp
        src/mesh/dim3/generate_database/parameters/parameters.cpp
        src/mesh/dim3/generate_database/mapping/mapping.cpp
        src/mesh/dim3/generate_database/materials/materials.cpp
        src/mesh/dim3/generate_database/coordinates/coordinates.cpp
        src/mesh/dim3/generate_database/jacobian_matrix/jacobian_matrix.cpp
        src/mesh/dim3/generate_database/control_nodes/control_nodes.cpp
        src/mesh/dim3/generate_database/tags/tags.cpp

        src/mesh/dim3/meshfem3d/adjacency_graph/assert_symmetry.cpp
        src/mesh/dim3/meshfem3d/mesh.cpp
)

target_link_libraries(
        mesh
        enumerations
        Kokkos::kokkos
        specfem_mpi
        # material_class
        yaml-cpp
        ${BOOST_LIBS}
)

add_library(
        read_seismogram
        src/io/seismogram/reader.cpp
)

target_link_libraries(
        read_seismogram
        Kokkos::kokkos
        )

add_library(utilities
        src/utilities/strings.cpp
)

target_link_libraries(
        utilities
        Kokkos::kokkos
        ${BOOST_LIBS}
)

add_library(
        reader
        src/io/property/reader.cpp
        src/io/wavefield/reader.cpp
)

target_link_libraries(
        reader
        assembly
        io
        read_seismogram
        utilities
)

add_library(
        algorithms
        src/algorithms/dim2/locate_point.cpp
        src/algorithms/dim2/locate_point_impl.cpp
        src/algorithms/dim3/locate_point.cpp
        src/algorithms/dim3/locate_point_impl.cpp
)

target_link_libraries(
        algorithms
        Kokkos::kokkos
        jacobian
        point
        ${BOOST_LIBS}
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
        utilities
)

# Disable unity build for source_time_function due to KOKKOS_INLINE_FUNCTION issues
set_target_properties(source_time_function PROPERTIES UNITY_BUILD OFF)

add_library(
        kokkos_kernels
        src/kokkos_kernels/impl/compute_mass_matrix.cpp
        src/kokkos_kernels/impl/invert_mass_matrix.cpp
        src/kokkos_kernels/impl/divide_mass_matrix.cpp
        src/kokkos_kernels/impl/compute_seismogram.cpp
        src/kokkos_kernels/impl/compute_source_interaction.cpp
        src/kokkos_kernels/impl/compute_stiffness_interaction.cpp
        src/kokkos_kernels/impl/compute_material_derivatives.cpp
        src/kokkos_kernels/impl/compute_coupling.cpp
        src/kokkos_kernels/frechet_kernels.cpp
)
set_target_properties(kokkos_kernels PROPERTIES
        UNITY_BUILD $<NOT:$<BOOL:${__APPLE__}>>
        UNITY_BUILD_BATCH_SIZE $<IF:$<BOOL:${__APPLE__}>,0,4>
)

if(__MSVC__)
target_compile_definitions(kokkos_kernels PRIVATE __MSVC__)
endif(__MSVC__)

target_link_libraries(
        kokkos_kernels
        Kokkos::kokkos
        assembly
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
)

add_library(
        writer
        src/io/seismogram/writer.cpp
        src/io/kernel/writer.cpp
        src/io/property/writer.cpp
        src/io/wavefield/writer.cpp
)

target_link_libraries(
        writer
        assembly
        specfem::receivers
        io
        utilities
)


add_library(
        parameter_reader
        src/parameter_parser/run_setup.cpp
        # src/parameter_parser/solver/solver.cpp
        src/parameter_parser/time_scheme/time_scheme.cpp
        src/parameter_parser/database_configuration.cpp
        src/parameter_parser/header.cpp
        src/parameter_parser/quadrature.cpp
        src/parameter_parser/elastic_wave.cpp
        src/parameter_parser/electromagnetic_wave.cpp
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
        specfem::receivers
        yaml-cpp
        periodic_tasks
        writer
        reader
        kokkos_kernels
        solver
        ${BOOST_LIBS}
)


# specfem2d executable for backward compatibility
add_executable(specfem2d src/specfem2d.cpp)
target_link_libraries(specfem2d
    PRIVATE
        specfem::core
        yaml-cpp::yaml-cpp
        ${BOOST_LIBS}
)

# specfem3d executable for backward compatibility
add_executable(specfem3d src/specfem3d.cpp)
target_link_libraries(specfem3d
    PRIVATE
        specfem::core
        yaml-cpp::yaml-cpp
        ${BOOST_LIBS}
)

# Include tests
if (SPECFEM_BUILD_TESTS)
        message(STATUS "Including tests.")
        add_subdirectory(tests/unit-tests)
endif()

if (SPECFEM_BUILD_BENCHMARKS)
    # Define option with default value

    if(DEFINED SPECFEM_BENCHMARKS_BUILD_DIR)
        message(STATUS "SPECFEM_BENCHMARKS_BUILD_DIR was defined as: ${SPECFEM_BENCHMARKS_BUILD_DIR}")
    else()
        set(SPECFEM_BENCHMARKS_BUILD_DIR ${CMAKE_SOURCE_DIR}/benchmarks/build)
        message(STATUS "SPECFEM_BENCHMARKS_BUILD_DIR was not defined, using default: ${SPECFEM_BENCHMARKS_BUILD_DIR}")
    endif()

    # Create the benchmarks build directory if it does not exist
    message(STATUS "Building benchmarks...")
    add_subdirectory(benchmarks/src)
    message(STATUS "Benchmarks built.")
endif()


# Include Doxygen
include("${CMAKE_SOURCE_DIR}/cmake/doxygen.cmake")

if (SPECFEM_USE_SKBUILD AND EXISTS ${SKBUILD_SCRIPTS_DIR})
    install(TARGETS specfem DESTINATION ${SKBUILD_SCRIPTS_DIR})
    install(FILES ${CMAKE_BINARY_DIR}/bin/xmeshfem2D DESTINATION ${SKBUILD_SCRIPTS_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
endif ()

if (SPECFEM_BINDING_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    python_add_library(_core MODULE src/python/core.cpp WITH_SOABI)
    target_link_libraries(
        _core PRIVATE
        specfem::core
        periodic_tasks
        ${BOOST_LIBS}
        pybind11::headers
    )
    target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
    install(TARGETS _core DESTINATION specfempp_core)
endif (SPECFEM_BINDING_PYTHON)


if (SPECFEM_INSTALL AND NOT SPECFEM_USE_SKBUILD)
    install(
        TARGETS specfem
        RUNTIME DESTINATION .
    )

    # Create a custom target that will run after your main build
    add_custom_target(organize_output ALL
        COMMAND ${CMAKE_COMMAND} --install . --prefix ${CMAKE_INSTALL_PREFIX}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Organizing specfem++ executables into: ${CMAKE_INSTALL_PREFIX}"
        # Make this run after your main targets
        DEPENDS specfem xmeshfem2D xmeshfem3D xadj_seismogram xgenerate_databases # list all your targets here
    )

    # Add the prefix directory to the clean target
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
        ${CMAKE_INSTALL_PREFIX}
    )

endif(SPECFEM_INSTALL AND NOT SPECFEM_USE_SKBUILD)
