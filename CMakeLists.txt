cmake_minimum_required(VERSION 3.17.5)

project(SPECFEMPP VERSION 0.1.0 LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
option(HDF5_CXX_BUILD "Build HDF5 C++" ON)
option(VTK_CXX_BUILD "Build VTK C++" ON)
option(MPI_PARALLEL "MPI enabled" OFF)
option(BUILD_TESTS "Tests included" OFF)
option(ENABLE_SIMD "Enable SIMD" OFF)
option(ENABLE_PROFILING "Enable profiling" OFF)
option(SPECFEMPP_BINDING_PYTHON "Enable Python binding" OFF)
option(SPECFEM_INSTALL "Creates clean executable directory. Location Controlled by CMAKE_INSTALL_PREFIX" ON)
option(ENABLE_DOUBLE_PRECISION "Enable double precision" OFF)
option(BUILD_BENCHMARKS "Benchmarks included" ON)
option(ENABLE_UNITY_BUILD "Enable unity build to speed up compilation" ON)

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

# For external modules that need to be built suppress build output
set(FETCHCONTENT_QUIET TRUE)

# Enable unity build for faster compilation
if(ENABLE_UNITY_BUILD)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "Unity build disabled for Debug build type.")
        set(CMAKE_UNITY_BUILD OFF)
    else (CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "Unity build enabled - this will speed up compilation")
        set(CMAKE_UNITY_BUILD ON)
        set(CMAKE_UNITY_BUILD_BATCH_SIZE 8)
    endif()
endif()

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
        set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "Using CUDA Constexpr by default")
    endif()
endif()

# For the installation of dependencies
include(FetchContent)

# Install Kokkos as a dependency
# Set Kokkos options before fetching
set(KOKKOS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        kokkos
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
        URL https://github.com/kokkos/kokkos/archive/refs/tags/4.6.01.zip
)

# Using this instead of FetchContent_MakeAvailable to be backwards compatible
# with older CMake versions
FetchContent_GetProperties(kokkos)
if(NOT kokkos_POPULATED)
  FetchContent_Populate(kokkos)
  add_subdirectory(${kokkos_SOURCE_DIR} ${kokkos_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Install yaml content
FetchContent_Declare(
        yaml
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 2f86d13775d119edbb69af52e5f566fd65c6953b
)

# Disable yaml-cpp installation
set(YAML_CPP_BUILD_INSTALL OFF CACHE BOOL "Disable yaml-cpp installation" FORCE)

FetchContent_MakeAvailable(yaml)

include_directories(BEFORE SYSTEM ${yaml_BINARY_DIR} ${yaml_SOURCE_DIR}/include)


if (VTK_CXX_BUILD)

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

    if (VTK_FOUND)
        message(STATUS "VTK libs/ and incs/:")
        message(STATUS "    LIB:   ${VTK_LIBRARY_DIRS}")
        message(STATUS "    INC:   ${VTK_INCLUDE_DIRS}")
        message(STATUS "    LIBSO: ${VTK_LIBRARIES}")
    else(NOT VTK_FOUND)
        message(STATUS "VTK not found. Building without VTK.")
        set(VTK_CXX_BUILD OFF)
    endif()
else ()
    set(VTK_FOUND OFF)
    message(STATUS "Building without VTK.")
endif()


# Try finding boost and if not found install.
find_package(Boost 1.85.0 COMPONENTS program_options filesystem system)


if (NOT ${Boost_FOUND})
    # Add boost lib sources
    set(BOOST_INCLUDE_LIBRARIES program_options filesystem system algorithm tokenizer preprocessor vmd)
    set(BOOST_LIBS Boost::program_options Boost::filesystem Boost::system
                   Boost::algorithm Boost::tokenizer Boost::preprocessor Boost::vmd)
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

    # Disable Boost installation
    set(BOOST_INSTALL OFF CACHE BOOL "Don't install Boost" FORCE)
    set(BOOST_INSTALL_LIBRARIES OFF CACHE BOOL "Don't install Boost libraries" FORCE)
    set(BOOST_SKIP_INSTALL_RULES ON CACHE BOOL "Skip Boost install rules" FORCE)

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

# Add the configure files to the clean target
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
    ${CMAKE_BINARY_DIR}/include/specfem_setup.hpp
    ${CMAKE_BINARY_DIR}/include/constants.hpp
)

# Add the include directories so that the generated files can be found
include_directories(include)
include_directories(core)
include_directories(${CMAKE_BINARY_DIR}/include)

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
add_subdirectory(core/specfem)

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
        src/io/fortranio/fortran_io.cpp
        src/io/sources.cpp
        src/io/receivers.cpp
        # Fortran 2D part
        src/io/mesh/impl/fortran/dim2/mesh.cpp
        src/io/mesh/impl/fortran/dim2/read_boundaries.cpp
        src/io/mesh/impl/fortran/dim2/read_elements.cpp
        src/io/mesh/impl/fortran/dim2/read_material_properties.cpp
        src/io/mesh/impl/fortran/dim2/read_mesh_database.cpp
        src/io/mesh/impl/fortran/dim2/read_interfaces.cpp
        src/io/mesh/impl/fortran/dim2/read_parameters.cpp
        # Fortran 3D part
        src/io/mesh/impl/fortran/dim3/mesh.cpp
        src/io/mesh/impl/fortran/dim3/read_parameters.cpp
        src/io/mesh/impl/fortran/dim3/read_coordinates.cpp
        src/io/mesh/impl/fortran/dim3/read_jacobian_matrix.cpp
        src/io/mesh/impl/fortran/dim3/utilities.cpp
)

if (NOT HDF5_CXX_BUILD)
    target_compile_definitions(
        io
        PUBLIC -DNO_HDF5
    )

    target_link_libraries(
        io
        mesh
        source_class
        specfem::receivers
        yaml-cpp
        ${BOOST_LIBS}
        Kokkos::kokkos)
else()
    target_link_libraries(
        io
        mesh
        source_class
        specfem::receivers
        Kokkos::kokkos
        yaml-cpp
        ${BOOST_LIBS}
        ${HDF5_LIBRARIES}
    )
endif()

add_library(
        enumerations
        src/enumerations/medium.cpp
        src/enumerations/wavefield.cpp
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
        src/mesh/dim2/coupled_interfaces/interface_container.cpp
        src/mesh/dim2/coupled_interfaces/coupled_interfaces.cpp
        src/mesh/dim2/tags/tags.cpp
        # 3-D
        src/mesh/dim3/mesh.cpp
        src/mesh/dim3/boundaries/absorbing_boundary.cpp
        src/mesh/dim3/boundaries/free_surface.cpp
        src/mesh/dim3/coupled_interfaces/coupled_interfaces.cpp
        src/mesh/dim3/element_types/element_types.cpp
        src/mesh/dim3/parameters/parameters.cpp
        src/mesh/dim3/parameters/parameters.cpp
        src/mesh/dim3/mapping/mapping.cpp
        src/mesh/dim3/materials/materials.cpp
        src/mesh/dim3/coordinates/coordinates.cpp
        src/mesh/dim3/jacobian_matrix/jacobian_matrix.cpp
        src/mesh/dim3/control_nodes/control_nodes.cpp
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
        jacobian
        src/jacobian/jacobian.cpp
)

target_link_libraries(
        jacobian
        Kokkos::kokkos
        point
        shape_functions
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
        src/utilities/utilities.cpp
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
        src/algorithms/locate_point.cpp
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
        source_class
        src/source/source.cpp
        src/source/force_source.cpp
        src/source/cosserat_force_source.cpp
        src/source/moment_tensor_source.cpp
        src/source/adjoint_source.cpp
        src/source/external.cpp
)

target_link_libraries(
        source_class
        Kokkos::kokkos
        specfem_mpi
        utilities
        quadrature
        source_time_function
        yaml-cpp
        point
        enumerations
        algorithms
        ${BOOST_LIBS}
)


add_library(coupled_interface
        src/coupled_interface/coupled_interface.cpp
)

target_link_libraries(
        coupled_interface
        Kokkos::kokkos
        assembly
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
set_target_properties(kokkos_kernels PROPERTIES UNITY_BUILD_BATCH_SIZE 4)

target_link_libraries(
        kokkos_kernels
        Kokkos::kokkos
        assembly
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
        periodic_tasks
        src/periodic_tasks/plot_wavefield.cpp
        src/periodic_tasks/check_signal.cpp
        src/periodic_tasks/wavefield_writer.cpp
        src/periodic_tasks/wavefield_reader.cpp
)

target_link_libraries(
        periodic_tasks
        reader
        writer
)

if (NOT VTK_CXX_BUILD)
        target_compile_definitions(
                periodic_tasks
                PUBLIC -DNO_VTK
        )

        target_link_libraries(
                periodic_tasks
                assembly
        )
else ()
        target_link_libraries(
                periodic_tasks
                assembly
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
        assembly
        source_class
        parameter_reader
        specfem::receivers
        writer
        periodic_tasks
        reader
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
        io
        Kokkos::kokkos
        mesh
        quadrature
        assembly
        source_class
        parameter_reader
        specfem::receivers
        writer
        periodic_tasks
        reader
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

if (BUILD_BENCHMARKS)
    # Define option with default value

    if(DEFINED BENCHMARKS_BUILD_DIR)
        message(STATUS "BENCHMARKS_BUILD_DIR was defined as: ${BENCHMARKS_BUILD_DIR}")
    else()
        set(BENCHMARKS_BUILD_DIR ${CMAKE_SOURCE_DIR}/benchmarks/build)
        message(STATUS "BENCHMARKS_BUILD_DIR was not defined, using default: ${BENCHMARKS_BUILD_DIR}")
    endif()

    # Create the benchmarks build directory if it does not exist
    message(STATUS "Building benchmarks...")
    add_subdirectory(benchmarks/src)
    message(STATUS "Benchmarks built.")
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
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
        ${DOXYGEN_OUT}
    )

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
        periodic_tasks
        ${BOOST_LIBS}
        pybind11::headers
    )
    target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
    install(TARGETS _core DESTINATION specfempp_core)
endif (SPECFEMPP_BINDING_PYTHON)


if (SPECFEM_INSTALL)
    install(
        TARGETS specfem2d specfem3d
        RUNTIME DESTINATION .
    )

    # Create a custom target that will run after your main build
    add_custom_target(organize_output ALL
        COMMAND ${CMAKE_COMMAND} --install . --prefix ${CMAKE_INSTALL_PREFIX}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Organizing specfem++ executables into: ${CMAKE_INSTALL_PREFIX}"
        # Make this run after your main targets
        DEPENDS specfem2d specfem3d xmeshfem2D xmeshfem3D xadj_seismogram xgenerate_databases # list all your targets here
    )

    # Add the prefix directory to the clean target
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
        ${CMAKE_INSTALL_PREFIX}
    )

endif(SPECFEM_INSTALL)
