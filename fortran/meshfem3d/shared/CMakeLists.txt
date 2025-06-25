# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(MESHFEM3D_SHARED LANGUAGES Fortran C)

# Set the Fortran standard to 95
set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/../modules)
# include_directories("${CMAKE_Fortran_MODULE_DIRECTORY}")

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../setup)


set(MESHFEM3D_SHARED_MODULE
  serial.f90
  assemble_MPI_scalar.f90
  check_mesh_resolution.f90
  count_number_of_sources.f90
  create_name_database.f90
  define_derivation_matrices.f90
  define_mass_matrices.f90
  detect_surface.f90
  exit_mpi.f90
  get_attenuation_model.f90
  get_element_face.f90
  get_global.f90
  get_jacobian_boundaries.f90
  get_shape2D.f90
  get_shape3D.f90
  gll_library.f90
  hex_nodes.f90
  heap_sort.f90
  init_openmp.F90
  lagrange_poly.f90
  merge_sort.f90
  netlib_specfun_erf.f90
  prepare_assemble_MPI.f90
  # command_line_arguments.f90
  read_parameter_file.F90
  read_topo_bathy_file.f90
  read_value_parameters.f90
  recompute_jacobian.f90
  safe_alloc_mod.f90
  save_header_file.f90
  search_kdtree.f90
  sort_array_coordinates.f90
  utm_geo.f90
  write_VTK_data.f90
)

add_library(meshfem3D_shared_module shared_par.F90)
add_library(meshfem3D::shared_module ALIAS meshfem3D_shared_module)
target_link_libraries(meshfem3D_shared_module
  meshfem3D::constants
)

add_library(meshfem3D_shared_hdf5_module
    hdf5_manager.F90
)
add_library(meshfem3D::shared_hdf5_module ALIAS meshfem3D_shared_hdf5_module)
target_link_libraries(meshfem3D_shared_hdf5_module PRIVATE
  meshfem3D::shared_module
)

set(SHARED_ADIOS_MODULES adios_manager.F90)

# This is not tested yet since ADIOS is not supported in the current version of
# SPECFEMPP
# ------------------------------------------------------------------------------
if (ADIOS)
  add_library(meshfem3D_adios_helpers_addons adios_helpers_addons.c)
  add_library(meshfem3D::adios_helpers_addons ALIAS meshfem3D_adios_helpers_addons)
  target_link_libraries(meshfem3D_adios_helpers_addons PRIVATE
    meshfem3D::config
    meshfem3D::constants
  )

  list(APPEND SHARED_ADIOS_MODULES
    adios_helpers_definitions
    adios_helpers_readers
    adios_helpers_writers
    adios_helpers
  )
endif(ADIOS)
# ------------------------------------------------------------------------------

add_library(meshfem3D_shared_adios_module
    ${SHARED_ADIOS_MODULES}
)

# This seems ugly, but it's the only way I could get the adios_helpers_addons
# library to be linked in the right order.
# This also definitely needs a link to the ADIOS library!
# ------------------------------------------------------------------------------
if (ADIOS)
  target_link_libraries(meshfem3D_shared_adios_module PRIVATE
    meshfem3D::adios_helpers_addons
  )
endif(ADIOS)
# ------------------------------------------------------------------------------

add_library(meshfem3D::shared_adios_module ALIAS meshfem3D_shared_adios_module)
target_link_libraries(meshfem3D_shared_adios_module PRIVATE
  meshfem3D::constants
  meshfem3D::shared_module
)

add_library(meshfem3D_adios_methods_stubs adios_method_stubs.c)
add_library(meshfem3D::adios_methods_stubs ALIAS meshfem3D_adios_methods_stubs)
target_link_libraries(meshfem3D_adios_methods_stubs PRIVATE
    meshfem3D::config
)

add_library(meshfem3D_param_reader param_reader.c)
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
  meshfem3D::param_reader
)
