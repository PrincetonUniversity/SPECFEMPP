project(meshfem2D Fortran C)

cmake_minimum_required(VERSION 3.10)

# Set the Fortran compiler
enable_language(Fortran)
enable_language(C)

# Options
option(SPECFEM_ENABLE_DOUBLE_PRECISION "Enable double precision" OFF)
option(WITH_MPI "Build with MPI support" OFF)
option(WITH_SCOTCH "Build with SCOTCH support" OFF)
option(FORCE_VECTORIZATION "Force vectorization" OFF)
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

# Print options to the screen

# Set output file directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/archive)

# Custom variables
set(SETUP_DIR ${CMAKE_CURRENT_BINARY_DIR}/setup)

# Set include directories
include_directories("${CMAKE_Fortran_MODULE_DIRECTORY}")
include_directories(${SETUP_DIR})

# Setting values dependent on options
if (SPECFEM_ENABLE_DOUBLE_PRECISION)
    message(STATUS "-- Enabling double precision")
    # These are used by Meshfem
    set(CUSTOM_REAL "SIZE_DOUBLE")
    set(CUSTOM_MPI_TYPE "MPI_DOUBLE_PRECISION")

else(SPECFEM_ENABLE_DOUBLE_PRECISION)
    message(STATUS "-- Enabling single precision")
    # These are used by Meshfem
    set(CUSTOM_REAL "SIZE_REAL")
    set(CUSTOM_MPI_TYPE "MPI_REAL")
endif(SPECFEM_ENABLE_DOUBLE_PRECISION)

# Configure files
add_subdirectory(setup)

# Set the C compiler
set(FCFLAGS_f90 -g -O2 -fbacktrace)

# Check if CFLAGS is defined
if (DEFINED ENV{CFLAGS})
    set(CFLAGS "$ENV{CFLAGS}")
endif()

message(STATUS "-- CFLAGS: ${CFLAGS}")

# Check if Clang
if (CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(CLANG TRUE)
    message(STATUS "-- Detected usage of clang on macOS")
endif()

# Fix for macos and clang include files... sometimes I hate mac.
# If we are on macOS the __APPLE__ variable is defined in the CMakeLists.txt file
# in the main directory.
if (__APPLE__ AND CLANG)
    message(STATUS "-- Fixing include files for macOS and Clang")
    set (FCFLAGS_f90 "${FCFLAGS_f90}")
    # In the future, we can use the following line to add the flags to the compiler
    # add_compile_options($<$<COMPILE_LANGUAGE:C>:-isysroot/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk>)
    # set (CFLAGS "${CFLAGS})
    # Right now we can use the following line to add the flags to the compiler
    if (DEFINED CFLAGS)
      set(CFLAGS "$ENV{CFLAGS} -isysroot/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
    else()
      set(CFLAGS "-isysroot/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
    endif()
endif()

message(STATUS "-- CFLAGS: ${CFLAGS}")

# Shared Fortran Module
add_library(meshfem2D_shared_module shared_par.f90)
add_library(meshfem2D::shared_module ALIAS meshfem2D_shared_module)
target_link_libraries(
    meshfem2D_shared_module
    PUBLIC
    meshfem2D::constants
    meshfem2D::precision
    meshfem2D::version
    meshfem2D::config
    meshfem2D::fh_config
)

# Utilities
add_library(meshfem2D_mesh_utilities
    command_line_arguments.f90
    file_management.f90
)
add_library(meshfem2D::mesh_utilities ALIAS meshfem2D_mesh_utilities)
target_link_libraries(meshfem2D_mesh_utilities PRIVATE meshfem2D::shared_module)


# Shared MPI Objects
add_library(meshfem2D_shared_mpi
    read_parameter_file.F90
    parallel.F90
    exit_mpi.F90
)
add_library(meshfem2D::shared_mpi ALIAS meshfem2D_shared_mpi)
target_link_libraries(meshfem2D_shared_mpi
  PRIVATE
  meshfem2D::mesh_utilities
  meshfem2D::shared_objects
  )


# Shared Objects
add_library(meshfem2D_shared_objects
    define_shape_functions.f90
    gll_library.f90
    lagrange_poly.f90
    read_interfaces_file.f90
    read_material_table.f90
    read_regions.f90
    read_source_file.f90
    read_value_parameters.f90
    set_color_palette.f90
    spline_routines.f90
    write_VTK_data.f90
)
add_library(meshfem2D::shared_objects ALIAS meshfem2D_shared_objects)
target_link_libraries(meshfem2D_shared_objects
  PRIVATE
  meshfem2D::shared_module
  meshfem2D::shared_mpi
  meshfem2D::constants
  )

# C Source Compilation
add_library(meshfem2D_cc_objects OBJECT
    force_ftz.c
    param_reader.c
)
add_library(meshfem2D::cc_objects ALIAS meshfem2D_cc_objects)



# MeshFEM Shared Module
add_library(meshfem2D_paramaters
  meshfem2D_par.f90)
add_library(meshfem2D::parameters ALIAS meshfem2D_paramaters)
target_link_libraries(meshfem2D_paramaters PRIVATE meshfem2D::shared_module)

# Compute Elements Load
add_library(meshfem2D_compute_elements_load
    compute_elements_load_par.f90)
add_library(meshfem2D::compute_elements_load ALIAS meshfem2D_compute_elements_load)
target_link_libraries(meshfem2D_compute_elements_load
    PRIVATE
    meshfem2D::shared_module
    meshfem2D::parameters
)

# MeshFEM Objects
add_library(meshfem2D_mesh
    determine_abs_surface.f90
    determine_acoustic_surface.f90
    get_node_number.f90
    repartition_coupling.f90
    rotate_mesh.f90
    adjacency_graph.f90
    save_databases.f90
    save_gnuplot_file.f90
    save_stations_file.f90
)
add_library(meshfem2D::mesh ALIAS meshfem2D_mesh)
target_link_libraries(meshfem2D_mesh
  PRIVATE
  meshfem2D::shared_module
  meshfem2D::parameters)

# MeshFEM Preprocessor Objects
add_library(meshfem2D_mesh_mpi
    decompose_mesh.F90
    metis_partitioning.F90
    part_unstruct.F90
    read_external_mesh_files.F90
    read_mesh_files.F90
    scotch_partitioning.F90
    meshfem2D.F90
)
add_library(meshfem2D::mesh_mpi ALIAS meshfem2D_mesh_mpi)
target_link_libraries(meshfem2D_mesh_mpi
  PRIVATE
  meshfem2D::shared_module
  meshfem2D::parameters
  meshfem2D::mesh
)

# Executable
add_executable(xmeshfem2D
    meshfem2D.F90
)
target_link_libraries(
    xmeshfem2D
    PRIVATE
    meshfem2D::compute_elements_load
    meshfem2D::mesh
    meshfem2D::mesh_mpi
    meshfem2D::parameters
    meshfem2D::shared_module
    meshfem2D::shared_mpi
    meshfem2D::mesh_utilities
    meshfem2D::cc_objects
)


# Additional Executable
add_executable(xadj_seismogram adj_seismogram.f90)

if (SPECFEM_INSTALL)
  install(TARGETS xadj_seismogram xmeshfem2D
    RUNTIME DESTINATION .
  )
endif()
