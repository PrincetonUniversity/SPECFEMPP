# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(MESHFEM3D LANGUAGES Fortran C)

# Set the Fortran standard to 95
set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(SETUP ${CMAKE_CURRENT_SOURCE_DIR}/setup)
set(SHARED ${CMAKE_CURRENT_SOURCE_DIR}/shared)
set(MESHFEM3D ${CMAKE_CURRENT_SOURCE_DIR}/meshfem3D)

set(MESHFEM3D_SHARED_MODULE
  ${SHARED}/serial.f90
  ${SHARED}/assemble_MPI_scalar.f90
  ${SHARED}/check_mesh_resolution.f90
  ${SHARED}/count_number_of_sources.f90
  ${SHARED}/create_name_database.f90
  ${SHARED}/define_derivation_matrices.f90
  ${SHARED}/define_mass_matrices.f90
  ${SHARED}/detect_surface.f90
  ${SHARED}/exit_mpi.f90
  ${SHARED}/get_attenuation_model.f90
  ${SHARED}/get_element_face.f90
  ${SHARED}/get_global.f90
  ${SHARED}/get_jacobian_boundaries.f90
  ${SHARED}/get_shape2D.f90
  ${SHARED}/get_shape3D.f90
  ${SHARED}/gll_library.f90
  ${SHARED}/hex_nodes.f90
  ${SHARED}/heap_sort.f90
  ${SHARED}/init_openmp.f90
  ${SHARED}/lagrange_poly.f90
  ${SHARED}/merge_sort.f90
  ${SHARED}/netlib_specfun_erf.f90
  ${SHARED}/prepare_assemble_MPI.f90
  ${SHARED}/read_parameter_file.f90
  ${SHARED}/read_topo_bathy_file.f90
  ${SHARED}/read_value_parameters.f90
  ${SHARED}/recompute_jacobian.f90
  ${SHARED}/safe_alloc_mod.f90
  ${SHARED}/save_header_file.f90
  ${SHARED}/search_kdtree.f90
  ${SHARED}/sort_array_coordinates.f90
  ${SHARED}/utm_geo.f90
  ${SHARED}/write_VTK_data.f90
)

add_library(meshfem3D_version INTERFACE ${SETUP}/version.fh)
add_library(meshfem3D::version ALIAS meshfem3D_version)
target_include_directories(meshfem3D_version INTERFACE ${SETUP})

add_library(meshfem3D_constants INTERFACE
  ${SETUP}/constants.h
)
add_library(meshfem3D::constants ALIAS meshfem3D_constants)
target_include_directories(meshfem3D_constants INTERFACE
  ${SETUP}
)

add_library(meshfem3D_config INTERFACE
  ${SETUP}/config.h
)
add_library(meshfem3D::config ALIAS meshfem3D_config)
target_include_directories(meshfem3D_config INTERFACE
  ${SETUP}
)

add_library(meshfem3D_shared_module
    ${SHARED}/shared_par.F90
)
add_library(meshfem3D::shared_module ALIAS meshfem3D_shared_module)
target_link_libraries(meshfem3D_shared_module PRIVATE
  meshfem3D::constants
)

add_library(meshfem3D_shared_hdf5_module
    ${SHARED}/hdf5_manager.F90
)
add_library(meshfem3D::shared_hdf5_module ALIAS meshfem3D_shared_hdf5_module)
target_link_libraries(meshfem3D_shared_hdf5_module PRIVATE
  meshfem3D::shared_module
)

add_library(meshfem3D_shared_adios_module
    ${SHARED}/adios_manager.F90
)
add_library(meshfem3D::shared_adios_module ALIAS meshfem3D_shared_adios_module)
target_link_libraries(meshfem3D_shared_adios_module PRIVATE
  meshfem3D::constants
  meshfem3D::shared_module
)

add_library(meshfem3D_adios_methods_stubs ${SHARED}/adios_method_stubs.c)
add_library(meshfem3D::adios_methods_stubs ALIAS meshfem3D_adios_methods_stubs)
target_link_libraries(meshfem3D_adios_methods_stubs PRIVATE
    meshfem3D::config
)

add_library(meshfem3D_param_reader ${SHARED}/param_reader.c)
add_library(meshfem3D::param_reader ALIAS meshfem3D_param_reader)
target_link_libraries(meshfem3D_param_reader PRIVATE
    meshfem3D::config
)

add_library(meshfem3D_shared
  ${MESHFEM3D_SHARED_MODULE}
)
add_library(meshfem3D::shared ALIAS meshfem3D_shared)
target_link_libraries(meshfem3D_shared PRIVATE
  meshfem3D::shared_module
  meshfem3D::constants
  meshfem3D::shared_hdf5_module
)



# target_link_libraries(meshfem3D_shared_module PRIVATE meshfem3D_constants)

# add_library(meshfem3D_shared PRIVATE ${MESHFEM3D_SHARED_MODULE})

# target_link_libraries(meshfem3D_shared PRIVATE meshfem3D_shared_module meshfem3D_constants)

# add_library(meshfem3D_shared_adios_module PRIVATE adios_manager.f90)

# target_link_libraries(meshfem3D_shared_adios_module PRIVATE meshfem3D_shared_module meshfem3D_constants)

# add_library(meshfem3D_shared_adios PRIVATE ${MESHFEM3D_SHARED_ADIOS_MODULE})

# target_link_libraries(meshfem3D_shared_adios PRIVATE meshfem3D_shared_adios_module

# cmake_minimum_required(VERSION 3.10)

# message("-- Building meshfem3D/shared along with this project")
# message("--     meshfem3D/shared is build without MPI support")
# message("--     meshfem3D/shared is build without SCOTCH support")

# unset(WITH_MPI)
# unset(WITH_SCOTCH)

# # Set the Fortran compiler
# enable_language(Fortran)
# enable_language(C)

# # Set the C compiler
# set(FCFLAGS_f90 -g -O2 -fbacktrace)

# # Check if CFLAGS is defined
# if (DEFINED ENV{CFLAGS})
# set(CFLAGS "$ENV{CFLAGS}")
# endif()

# message("-- CFLAGS: ${CFLAGS}")

# # Check if Clang
# if (CMAKE_C_COMPILER_ID MATCHES "Clang")
# set(CLANG TRUE)
# message("-- Detected usage of clang on macOS")
# endif()

# # Fix for macos and clang include files... sometimes I hate mac.
# # If we are on macOS the __APPLE__ variable is defined in the CMakeLists.txt file
# # in the main directory.
# if (__APPLE__ AND CLANG)
# message("-- Fixing include files for macOS and Clang")
# set (FCFLAGS_f90 "${FCFLAGS_f90}")
# # In the future, we can use the following line to add the flags to the compiler
# # add_compile_options($<$<COMPILE_LANGUAGE:C>:-isysroot/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk>)
# # set (CFLAGS "${CFLAGS})
# # Right now we can use the following line to add the flags to the compiler
# if (DEFINED CFLAGS)
# set(CFLAGS "$ENV{CFLAGS} -isysroot/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
# else()
# set(CFLAGS "-isysroot/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
# endif()
# endif()

# message("-- CFLAGS: ${CFLAGS}")

# set(SETUP "../setup")

# include_directories("${CMAKE_BINARY_DIR}/fortran/meshfem3d/shared")
# include_directories("${CMAKE_SOURCE_DIR}/fortran/meshfem3d/setup")

# add_library(constants shared_par.F90)

# # The module names are inconsistent with the file names...
# add_library(safe_alloc_mod safe_alloc_mod.f90)
# add_library(kdtree_search search_kdtree.f90)
# add_library(shared_input_parameters shared_par.F90 "${SETUP}/constants.h")
# add_library(shared_compute_parameters shared_par.F90 "${SETUP}/constants.h")
# add_library(shared_parameters shared_par.F90 "${SETUP}/constants.h")

# target_link_libraries(kdtree_search constants)
# target_link_libraries(safe_alloc_mod constants)
# target_link_libraries(shared_input_parameters constants)
# target_link_libraries(shared_compute_parameters constants)
# target_link_libraries(shared_parameters constants)

# # add_library(manager_hdf5 hdf5_manager.F90)
# # add_library(attenuation_model get_attenuation_model.f90)
# # target_link_libraries(manager_hdf5 constants shared_par)
# # target_link_libraries(attenuation_model constants shared_par)

# if (ADIOS)
# add_library(adios_manager adios_manager.f90 "${SETUP}/constants.h")
# endif(ADIOS)

# set(SOURCEFILES
# assemble_MPI_scalar
# check_mesh_resolution
# count_number_of_sources
# create_name_database
# define_mass_matrices
# detect_surface
# exit_mpi
# get_element_face
# get_global
# get_jacobian_boundaries
# get_shape2D
# get_shape3D
# gll_library
# define_derivation_matrices
# hex_nodes
# heap_sort
# init_openmp
# lagrange_poly
# merge_sort
# netlib_specfun_erf
# prepare_assemble_MPI
# read_parameter_file
# read_topo_bathy_file
# read_value_parameters
# recompute_jacobian
# safe_alloc_mod
# save_header_file
# search_kdtree
# sort_array_coordinates
# utm_geo
# get_attenuation_model
# )

# set(capitalf90 "F90")
# set(lowerc_f90 "f90")

# foreach(SOURCEFILE ${SOURCEFILES})
# # Check for F90
# if (EXISTS "${SOURCEFILE}.${capitalf90}")
# message("-- Adding source file: ${SOURCEFILE}.${capitalf90}")
# add_library(${SOURCEFILE} ${SOURCEFILE}.${capitalf90})
# target_link_libraries(${SOURCEFILE} shared_par constants)
# # else
# elseif(EXISTS "${SOURCEFILE}.${lowerc_f90}")
# message("-- Adding source file: ${SOURCEFILE}.${capitalf90}")
# add_library(${SOURCEFILE} ${SOURCEFILE}.${capitalf90})
# target_link_libraries(${SOURCEFILE} shared_par constants)
# endif()

# endforeach()

# set(CSOURCE_FILES
# write_c_binary
# param_reader
# force_ftz
# )

# foreach(CSOURCE_FILE ${CSOURCE_FILES})
# message("-- Adding source file: ${CSOURCE_FILE}.cc")
# add_library(${CSOURCE_FILE} ${CSOURCE_FILE}.c "${SETUP}/config.h")
# target_link_libraries(${CSOURCE_FILE} shared_par constants)
# endforeach()

# # write_c_binary.cc
# # param_reader.cc
# # force_ftz.cc
# # write_VTK_data.shared
# # hdf5_manager
# # adios_manager

# # Add later
# # write_VTK_data
# # hdf5_manager
# # adios_manager

set(MESHFEM3D_MESH_MODULE
    ${MESHFEM3D}/calc_gll_points.f90
    ${MESHFEM3D}/check_mesh_quality.f90
    ${MESHFEM3D}/chunk_earth_mesh_mod.f90
    ${MESHFEM3D}/compute_parameters.f90
    ${MESHFEM3D}/create_meshfem_mesh.f90
    ${MESHFEM3D}/create_CPML_regions.f90
    ${MESHFEM3D}/create_interfaces_mesh.f90
    ${MESHFEM3D}/create_visual_files.f90
    ${MESHFEM3D}/define_subregions.f90
    ${MESHFEM3D}/define_subregions_heuristic.f90
    ${MESHFEM3D}/define_superbrick.f90
    ${MESHFEM3D}/determine_cavity.f90
    ${MESHFEM3D}/earth_chunk.f90
    ${MESHFEM3D}/get_flags_boundaries.f90
    ${MESHFEM3D}/get_MPI_cutplanes_eta.f90
    ${MESHFEM3D}/get_MPI_cutplanes_xi.f90
    ${MESHFEM3D}/meshfem3D.f90
    ${MESHFEM3D}/get_wavefield_discontinuity.f90
    ${MESHFEM3D}/read_mesh_parameter_file.f90
    ${MESHFEM3D}/read_value_mesh_parameters.f90
    ${MESHFEM3D}/save_databases.f90
    ${MESHFEM3D}/store_boundaries.f90
    ${MESHFEM3D}/store_coords.f90
)

add_library(meshfem3D_mesh_module ${MESHFEM3D}/meshfem3D_par.f90)
add_library(meshfem3D::mesh_module ALIAS meshfem3D_mesh_module)
target_link_libraries(meshfem3D_mesh_module PRIVATE meshfem3D::constants meshfem3D::shared_module)

add_library(meshfem3D_mesh_hdf5_module
    ${MESHFEM3D}/save_databases_hdf5.F90
)
add_library(meshfem3D::mesh_hdf5_module ALIAS meshfem3D_mesh_hdf5_module)
target_link_libraries(meshfem3D_mesh_hdf5_module PRIVATE
  meshfem3D::shared_hdf5_module
  meshfem3D::shared_module
  meshfem3D::mesh_module
)

add_library(meshfem3D_mesh
  ${MESHFEM3D_MESH_MODULE}
)
add_library(meshfem3D::mesh ALIAS meshfem3D_mesh)
target_link_libraries(meshfem3D_mesh PRIVATE
  meshfem3D::mesh_module
  meshfem3D::shared_module
  meshfem3D::version
)

add_executable(xmeshfem3D ${MESHFEM3D}/meshfem3D.F90)
target_link_libraries(xmeshfem3D PRIVATE
    meshfem3D::mesh
    meshfem3D::mesh_module
    meshfem3D::shared
    meshfem3D::shared_module
    meshfem3D::mesh_hdf5_module
    meshfem3D::shared_hdf5_module
    meshfem3D::shared_adios_module
    meshfem3D::adios_methods_stubs
    meshfem3D::param_reader
    meshfem3D::version
)
