cmake_minimum_required(VERSION 3.10)

message("-- Building meshfem3D/shared along with this project")
message("--     meshfem3D/shared is build without MPI support")
message("--     meshfem3D/shared is build without SCOTCH support")

unset(WITH_MPI)
unset(WITH_SCOTCH)

# Set the Fortran compiler
enable_language(Fortran)
enable_language(C)

# Set the C compiler
set(FCFLAGS_f90 -g -O2 -fbacktrace)

# Check if CFLAGS is defined
if (DEFINED ENV{CFLAGS})
    set(CFLAGS "$ENV{CFLAGS}")
endif()

message("-- CFLAGS: ${CFLAGS}")

# Check if Clang
if (CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(CLANG TRUE)
    message("-- Detected usage of clang on macOS")
endif()

# Fix for macos and clang include files... sometimes I hate mac.
# If we are on macOS the __APPLE__ variable is defined in the CMakeLists.txt file
# in the main directory.
if (__APPLE__ AND CLANG)
    message("-- Fixing include files for macOS and Clang")
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

message("-- CFLAGS: ${CFLAGS}")

set(SETUP "../setup")

include_directories("${CMAKE_BINARY_DIR}/fortran/meshfem3d/shared")
include_directories("${CMAKE_SOURCE_DIR}/fortran/meshfem3d/setup")


add_library(constants shared_par.F90)

# The module names are inconsistent with the file names...
add_library(safe_alloc_mod safe_alloc_mod.f90)
add_library(kdtree_search search_kdtree.f90)
add_library(shared_input_parameters shared_par.F90 "${SETUP}/constants.h")
add_library(shared_compute_parameters shared_par.F90 "${SETUP}/constants.h")
add_library(shared_parameters shared_par.F90 "${SETUP}/constants.h")

target_link_libraries(kdtree_search constants)
target_link_libraries(safe_alloc_mod constants)
target_link_libraries(shared_input_parameters constants)
target_link_libraries(shared_compute_parameters constants)
target_link_libraries(shared_parameters constants)


# add_library(manager_hdf5 hdf5_manager.F90)
# add_library(attenuation_model get_attenuation_model.f90)
# target_link_libraries(manager_hdf5 constants shared_par)
# target_link_libraries(attenuation_model constants shared_par)

if (ADIOS)
  add_library(adios_manager adios_manager.f90 "${SETUP}/constants.h")
endif(ADIOS)


set(SOURCEFILES
	assemble_MPI_scalar
	check_mesh_resolution
	count_number_of_sources
	create_name_database
	define_mass_matrices
	detect_surface
	exit_mpi
	get_element_face
	get_global
	get_jacobian_boundaries
	get_shape2D
	get_shape3D
	gll_library
	define_derivation_matrices
	hex_nodes
	heap_sort
	init_openmp
	lagrange_poly
	merge_sort
	netlib_specfun_erf
	prepare_assemble_MPI
	read_parameter_file
	read_topo_bathy_file
	read_value_parameters
	recompute_jacobian
	safe_alloc_mod
	save_header_file
	search_kdtree
	sort_array_coordinates
	utm_geo
	get_attenuation_model
)

set(capitalf90 "F90")
set(lowerc_f90 "f90")

foreach(SOURCEFILE ${SOURCEFILES})
  # Check for F90
  if (EXISTS "${SOURCEFILE}.${capitalf90}")
    message("-- Adding source file: ${SOURCEFILE}.${capitalf90}")
    add_library(${SOURCEFILE} ${SOURCEFILE}.${capitalf90})
    target_link_libraries(${SOURCEFILE} shared_par constants)
  # else
  elseif(EXISTS "${SOURCEFILE}.${lowerc_f90}")
    message("-- Adding source file: ${SOURCEFILE}.${capitalf90}")
    add_library(${SOURCEFILE} ${SOURCEFILE}.${capitalf90})
    target_link_libraries(${SOURCEFILE} shared_par constants)
  endif()

  endforeach()


set(CSOURCE_FILES
  write_c_binary
  param_reader
  force_ftz
)

foreach(CSOURCE_FILE ${CSOURCE_FILES})
  message("-- Adding source file: ${CSOURCE_FILE}.cc")
  add_library(${CSOURCE_FILE} ${CSOURCE_FILE}.c "${SETUP}/config.h")
  target_link_libraries(${CSOURCE_FILE} shared_par constants)
endforeach()

	# write_c_binary.cc
	# param_reader.cc
	# force_ftz.cc
	# write_VTK_data.shared
	# hdf5_manager
	# adios_manager

  # Add later
  # write_VTK_data
  # hdf5_manager
  # adios_manager
