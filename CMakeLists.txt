cmake_minimum_required(VERSION 3.17.5)

# C is needed so MPI-enabled distro VTK packages can resolve MPI::MPI_C
project(SPECFEMPP VERSION 0.6.0 LANGUAGES C CXX Fortran)

# Extract Git version metadata
include("${CMAKE_SOURCE_DIR}/cmake/version.cmake")

set(CMAKE_CXX_STANDARD 20)

# SPECFEM options
option(SPECFEM_ENABLE_ADIOS2 "Enable ADIOS2 I/O in SPECFEM++" OFF)
option(SPECFEM_ENABLE_ADIOS2_FORCE_INSTALL "Force install ADIOS2 dependency if SPECFEM_ENABLE_ADIOS2 is ON even if system ADIOS2 is found" OFF)
option(SPECFEM_ENABLE_HDF5 "Enable HDF5 I/O in SPECFEM++" OFF)
option(SPECFEM_ENABLE_HDF5_FORCE_INSTALL "Force install HDF5 dependency if SPECFEM_ENABLE_HDF5 is ON even if system HDF5 is found" OFF)
option(SPECFEM_ENABLE_NPZ "Enable NPZ I/O in SPECFEM++" OFF)
option(SPECFEM_ENABLE_ZLIB_FORCE_INSTALL "Force install ZLIB dependency even if system ZLIB is found" OFF)
option(SPECFEM_ENABLE_BOOST_FORCE_INSTALL "Force install Boost dependency even if system Boost is found" OFF)
option(SPECFEM_ENABLE_VTK "Enable VTK visualization in SPECFEM++" ON)
option(SPECFEM_ENABLE_MPI "MPI enabled" OFF)
option(SPECFEM_BUILD_TESTS "Tests included" OFF)
option(SPECFEM_ENABLE_SIMD "Enable SIMD" OFF)
option(SPECFEM_ENABLE_PROFILING "Enable profiling" OFF)
option(SPECFEM_ENABLE_COVERAGE "Instrument C++ for LLVM source-based coverage (Clang only)" OFF)
option(SPECFEM_BINDING_PYTHON "Enable Python binding" OFF)
option(SPECFEM_INSTALL "Creates clean executable directory. Location Controlled by CMAKE_INSTALL_PREFIX" ON)
option(SPECFEM_ENABLE_DOUBLE_PRECISION "Enable double precision" OFF)
option(SPECFEM_ENABLE_METIS "Enable METIS partitioning support in xdecompose_mesh" OFF)
option(SPECFEM_ENABLE_TRILINOS "Enable Trilinos implicit solvers (Tpetra/Belos/Ifpack2/MueLu/Amesos2)" OFF)
option(SPECFEM_BUILD_BENCHMARKS "Benchmarks included" ON)
option(SPECFEM_NIGHTLY_BENCHMARKS "Include nightly benchmarks" OFF)
option(SPECFEM_ENABLE_UNITY_BUILD "Enable unity build to speed up compilation" ON)
option(SPECFEMPP_CONFIGURE_TESTS "Configure test files used to generate data for unit tesats" OFF)
set(SPECFEMPP_UNITY_BUILD_BATCH_SIZE 8 CACHE STRING "Set the batch size for unity builds (only used if SPECFEM_ENABLE_UNITY_BUILD is ON)")

if (SPECFEM_ENABLE_MPI)
    message(STATUS "Compiling SPECFEM with MPI")
    find_package(MPI REQUIRED)
    add_compile_definitions(SPECFEM_ENABLE_MPI)
    # SPECFEM_ENABLE_MPI is defined for every target in the tree, so
    # <specfem/mpi.hpp> pulls in <mpi.h> (and the inline MPI calls in
    # specfem::mpi::Context) everywhere -- not just in targets that happen to
    # link specfem::mpi. Make MPI usage requirements global to match.
    link_libraries(MPI::MPI_CXX)
    message(STATUS "MPI enabled - found MPI::MPI_CXX")
else()
    message(STATUS "Compiling SPECFEM without MPI")
endif()

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
        set(CMAKE_UNITY_BUILD_BATCH_SIZE ${SPECFEMPP_UNITY_BUILD_BATCH_SIZE})
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

# check GNU compiler
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(__GNUC__ TRUE)
    message(STATUS "GNU compiler detected -- setting __GNUC__ TRUE")
else(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(__GNUC__ FALSE)
    message(STATUS "GNU compiler not detected -- setting __GNUC__ FALSE")
endif()

# Install Kokkos as a dependency
include("${CMAKE_SOURCE_DIR}/cmake/kokkos.cmake")

# Configure SPECFEM++ build variables based on Kokkos settings
include("${CMAKE_SOURCE_DIR}/cmake/variables.cmake")

# Load CMake configuration summary function
include("${CMAKE_SOURCE_DIR}/cmake/cmake_summary.cmake")

# Install yaml-cpp as a dependency
include("${CMAKE_SOURCE_DIR}/cmake/yaml.cmake")

# Find or install boost library function
include("${CMAKE_SOURCE_DIR}/cmake/boost.cmake")

# Install CLI11 header-only library
include("${CMAKE_SOURCE_DIR}/cmake/cli11.cmake")

# Find HDF5
include("${CMAKE_SOURCE_DIR}/cmake/hdf5.cmake")

# Find or install NPZ library function
include("${CMAKE_SOURCE_DIR}/cmake/npz.cmake")

# Find or install ADIOS2 library function
include("${CMAKE_SOURCE_DIR}/cmake/adios2.cmake")

# Include VTK if available (MUST BE AFTER ADIOS2, VTK is smarter about finding
# the pugixml dependency than ADIOS2)
include("${CMAKE_SOURCE_DIR}/cmake/vtk.cmake")

# Find or install METIS graph partitioner (used by xdecompose_mesh)
include("${CMAKE_SOURCE_DIR}/cmake/metis.cmake")

# Find Trilinos (implicit solvers). MUST be after cmake/kokkos.cmake so Kokkos is
# resolved first and Trilinos validates against the same Kokkos.
include("${CMAKE_SOURCE_DIR}/cmake/trilinos.cmake")

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
    set(MPI_TYPE_REAL "MPI_DOUBLE")
else(SPECFEM_ENABLE_DOUBLE_PRECISION)
    message(STATUS "Enabling single precision")
    set(TYPE_REAL "float")
    set(MPI_TYPE_REAL "MPI_FLOAT")
endif(SPECFEM_ENABLE_DOUBLE_PRECISION)

# Add the include directories so that the generated files can be found
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core)
include_directories(${CMAKE_BINARY_DIR}/include)

message(STATUS "INCLUDE DIRECTORIES:")
message(STATUS "    core: ${CMAKE_CURRENT_SOURCE_DIR}/core")
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
message(STATUS "Configuring xmeshfem3D")
set(_saved_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL ${SUBDIR_LOG_LEVEL})
add_subdirectory(fortran/meshfem3d)
set(CMAKE_MESSAGE_LOG_LEVEL ${_saved_CMAKE_MESSAGE_LOG_LEVEL})
# =================================

# ========== DECOMPOSE_MESH =========
message(STATUS "Configuring decompose_mesh")
set(_saved_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL ${SUBDIR_LOG_LEVEL})
add_subdirectory(fortran/decompose_mesh)
set(CMAKE_MESSAGE_LOG_LEVEL ${_saved_CMAKE_MESSAGE_LOG_LEVEL})
# ===================================

# LLVM source-based code coverage instrumentation.
# Appended here -- AFTER the Fortran subdirectories above -- on purpose: none of
# the Fortran tools or their C++ glue (e.g. fortran/decompose_mesh/wrap_patoh.cpp)
# may be instrumented, because those executables link with gfortran, which
# neither understands -fprofile-instr-generate nor provides the LLVM profile
# runtime (an instrumented object would leave __llvm_profile_runtime undefined).
# core/ and the unit tests below still receive full instrumentation.
# Driven by scripts/coverage.smk.
if(SPECFEM_ENABLE_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR
            "SPECFEM_ENABLE_COVERAGE requires Clang or AppleClang "
            "(CMAKE_CXX_COMPILER_ID is '${CMAKE_CXX_COMPILER_ID}').")
    endif()
    message(STATUS "Coverage instrumentation enabled (LLVM source-based)")
    # Keep -g for symbols but let CMAKE_BUILD_TYPE choose the optimization level:
    # the coverage preset uses RelWithDebInfo (-O2) so the heavy integration tests
    # (displacement_tests_2d/3d) run at a tolerable speed. Forcing -O0 here makes
    # them glacial. Coverage stays reliable under optimization (minor region-count
    # imprecision only).
    string(APPEND CMAKE_CXX_FLAGS
        " -fprofile-instr-generate -fcoverage-mapping -g")
    # Re-enable assert()/NDEBUG-guarded code so it is compiled and shows up in
    # coverage. RelWithDebInfo (and Release) define NDEBUG via
    # CMAKE_CXX_FLAGS_<CONFIG>, which CMake places after CMAKE_CXX_FLAGS; using
    # add_compile_options() puts -UNDEBUG after that, so it wins. Scoped to C++.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>)
    # Belt-and-suspenders: scope the profiling-runtime link flag to C++-linked
    # targets (CMake >= 3.18) so any Fortran-linked target added later still
    # links cleanly even if it pulls in an instrumented C++ object.
    add_link_options($<$<LINK_LANGUAGE:CXX>:-fprofile-instr-generate>)
    # Coverage needs per-source-file attribution, so force unity builds off.
    set(CMAKE_UNITY_BUILD OFF)
endif()

# Add module subdirectories
add_subdirectory(core)

# Include tests
if (SPECFEM_BUILD_TESTS)
        message(STATUS "Including tests.")
        add_subdirectory(tests)
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

if (SPECFEM_NIGHTLY_BENCHMARKS)
   message(STATUS "Including nightly benchmarks.")
   set(SPECFEM_NIGHTLY_BENCHMARKS_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nightly_benchmarks/build)
   message(STATUS "SPECFEM_NIGHTLY_BENCHMARKS_BUILD_DIR was set to: ${SPECFEM_NIGHTLY_BENCHMARKS_BUILD_DIR}")

   message(STATUS "Building nightly benchmarks...")
   add_subdirectory(nightly_benchmarks)
   message(STATUS "Nightly benchmarks built.")
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
        COMPONENT executables
    )

    # Create a custom target that will run after your main build
    add_custom_target(organize_output ALL
        COMMAND ${CMAKE_COMMAND} --install . --prefix ${CMAKE_INSTALL_PREFIX} --component executables
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Organizing specfem++ executables into: ${CMAKE_INSTALL_PREFIX}"
        # Make this run after your main targets
        DEPENDS specfem xmeshfem2D xmeshfem3D xadj_seismogram xdecompose_mesh # list all your targets here
    )

    # Add the prefix directory to the clean target
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
        ${CMAKE_INSTALL_PREFIX}
    )

endif(SPECFEM_INSTALL AND NOT SPECFEM_USE_SKBUILD)

# Print configuration summary
specfem_print_configuration_summary()
