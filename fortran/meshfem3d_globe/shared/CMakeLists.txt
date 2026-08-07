cmake_minimum_required(VERSION 3.10)

project(MESHFEM3D_GLOBE_SHARED LANGUAGES Fortran C CXX)

set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../setup)

set(MESHFEM3D_GLOBE_SHARED_MODULE
  shared_par.f90
)

set(MESHFEM3D_GLOBE_SHARED_SOURCES
  assemble_MPI_scalar.f90
  assemble_MPI_vector.f90
  auto_ner.f90
  binary_c_io.c
  broadcast_computed_parameters.f90
  calendar.f90
  count_elements.f90
  count_number_of_sources.f90
  count_points.f90
  create_addressing.f90
  create_name_database.f90
  create_serial_name_database.f90
  define_all_layers.f90
  euler_angles.f90
  exit_mpi.f90
  fft.f90
  flush_system.f90
  get_all_eight_slices.f90
  get_global.F90
  get_model_parameters.F90
  get_timestep_and_layers.f90
  gll_library.f90
  heap_sort.f90
  hex_nodes.f90
  init_openmp.F90
  interpolate.f90
  intgrl.f90
  lagrange_poly.f90
  make_ellipticity.f90
  memory_eval.f90
  model_mars_1D.f90
  model_prem.f90
  model_Sohl.f90
  model_topo_bathy.f90
  model_vpremoon.f90
  parallel.f90
  param_reader.c
  read_compute_parameters.f90
  read_parameter_file.F90
  read_value_parameters.f90
  recompute_jacobian.f90
  reduce.f90
  rotate_tensor.f90
  rthetaphi_xyz.f90
  save_header_file.F90
  search_kdtree.f90
  SIEM_math_library.F90
  smooth_weights_vec.F90
  sort_array_coordinates.f90
  spl_A3d.c
  spline_routines.f90
  write_VTK_file.f90
  ylm.f90
)

add_library(meshfem3D_globe_shared_module ${MESHFEM3D_GLOBE_SHARED_MODULE})
add_library(meshfem3D_globe::shared_module ALIAS meshfem3D_globe_shared_module)
target_link_libraries(meshfem3D_globe_shared_module PRIVATE meshfem3D_globe::constants)

add_library(meshfem3D_globe_shared_hdf5_module hdf5_manager.F90)
add_library(meshfem3D_globe::shared_hdf5_module ALIAS meshfem3D_globe_shared_hdf5_module)
target_link_libraries(meshfem3D_globe_shared_hdf5_module PRIVATE
  meshfem3D_globe::shared_module
)

add_library(meshfem3D_globe_shared_adios_module adios_manager.F90)
add_library(meshfem3D_globe::shared_adios_module ALIAS meshfem3D_globe_shared_adios_module)
target_link_libraries(meshfem3D_globe_shared_adios_module PRIVATE
  meshfem3D_globe::constants
  meshfem3D_globe::shared_module
)

add_library(meshfem3D_globe_adios_methods_stubs adios_method_stubs.c)
add_library(meshfem3D_globe::adios_methods_stubs ALIAS meshfem3D_globe_adios_methods_stubs)
target_link_libraries(meshfem3D_globe_adios_methods_stubs PRIVATE meshfem3D_globe::config)

add_library(meshfem3D_globe_asdf_methods_stubs asdf_method_stubs.c)
add_library(meshfem3D_globe::asdf_methods_stubs ALIAS meshfem3D_globe_asdf_methods_stubs)
target_link_libraries(meshfem3D_globe_asdf_methods_stubs PRIVATE meshfem3D_globe::config)

add_library(meshfem3D_globe_param_reader param_reader.c)
add_library(meshfem3D_globe::param_reader ALIAS meshfem3D_globe_param_reader)
target_link_libraries(meshfem3D_globe_param_reader PRIVATE meshfem3D_globe::config)

add_library(meshfem3D_globe_shared ${MESHFEM3D_GLOBE_SHARED_SOURCES})
add_library(meshfem3D_globe::shared ALIAS meshfem3D_globe_shared)
target_link_libraries(meshfem3D_globe_shared PRIVATE
  meshfem3D_globe::shared_module
  meshfem3D_globe::shared_hdf5_module
  meshfem3D_globe::shared_adios_module
  meshfem3D_globe::param_reader
  meshfem3D_globe::constants
)

target_link_libraries(meshfem3D_globe_shared PRIVATE MPI::MPI_Fortran)
