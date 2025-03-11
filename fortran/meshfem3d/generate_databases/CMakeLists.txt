# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(MESHFEM3D_MESHFEM3D LANGUAGES Fortran C)

# Set the Fortran standard to 95
set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../modules)
# include_directories(CMAKE_Fortran_MODULE_DIRECTORY)

set(MESHFEM3D_GENERATE_DATABASES_MODULE
  calc_jacobian.f90
  create_mass_matrices.f90
  create_regions_mesh.f90
  fault_generate_databases.f90
  finalize_databases.f90
  # generate_databases_par.F90
  generate_databases.f90
  command_line_arguments.f90
  get_absorbing_boundary.F90
  get_coupling_surfaces.f90
  get_model.F90
  get_MPI.f90
  get_perm_color.f90
  lts_generate_databases.F90
  memory_eval.f90
  model_1d_cascadia.f90
  model_1d_prem.f90
  model_1d_socal.f90
  model_aniso.f90
  model_coupled.f90
  model_default.f90
  model_external_values.F90
  model_ipati.f90
  model_gll.f90
  model_salton_trough.f90
  model_tomography.f90
  parse_sep.c
  pml_set_local_dampingcoeff.f90
  read_parameters.f90
  read_partition_files.f90
  save_arrays_solver.F90
  save_parameters.f90
  setup_color_perm.f90
  setup_mesh_adjacency.f90
  setup_mesh.f90
  wavefield_discontinuity_generate_databases.f90
)

set(MESHFEM3D_GEN_DB_HDF5_MODULE
  read_partition_files_hdf5.F90
  save_arrays_solver_hdf5.F90
)

if (ADIOS)
  message(STATUS "ADIOS enabled")
  list(APPEND MESHFEM3D_GENERATE_DATABASES_MODULE
          model_gll_adios.F90
          model_ipati_adios.F90
          read_partition_files_adios.F90
          save_arrays_solver_adios.F90
          save_moho_adios.F90
  )
endif(ADIOS)

# Add MPI dependency
if (MPI_PARALLEL)
  list(APPEND MESHFEM3D_GENERATE_DATABASES_MODULE model_sep)
else(MPI_PARALLEL)
  list(APPEND MESHFEM3D_GENERATE_DATABASES_MODULE model_sep_nompi)
endif(MPI_PARALLEL)

# Create parameter library module
add_library(meshfem3D_generate_databases_module generate_databases_par.F90)
add_library(meshfem3D::generate_databases_module ALIAS meshfem3D_generate_databases_module)
target_link_libraries(meshfem3D_generate_databases_module PRIVATE meshfem3D::constants meshfem3D::shared_module meshfem3D::shared)

# Create HDF5 module
message(STATUS "Creating HDF5 module using: ${MESHFEM3D_GEN_DB_HDF5_MODULE}")
add_library(meshfem3D_generate_databases_hdf5_module
  ${MESHFEM3D_GEN_DB_HDF5_MODULE}
)
add_library(meshfem3D::generate_databases_hdf5_module ALIAS meshfem3D_generate_databases_hdf5_module)
target_link_libraries(meshfem3D_generate_databases_hdf5_module PRIVATE
  meshfem3D::shared_hdf5_module
  meshfem3D::shared_module
  meshfem3D::generate_databases_module
)

# Compile library
add_library(meshfem3D_generate_databases
  ${MESHFEM3D_GENERATE_DATABASES_MODULE}
)

add_library(meshfem3D::generate_databases ALIAS meshfem3D_generate_databases)
target_link_libraries(meshfem3D_generate_databases PRIVATE
  meshfem3D::generate_databases_module
  meshfem3D::generate_databases_hdf5_module
  meshfem3D::shared_module
  meshfem3D::constants
  meshfem3D::version
  meshfem3D::shared_hdf5_module
  meshfem3D::config
)

# Create executable
add_executable(xgenerate_databases generate_databases.f90)
target_link_libraries(xgenerate_databases PRIVATE
  meshfem3D::generate_databases_module
  meshfem3D::generate_databases_hdf5_module
  meshfem3D::shared
  meshfem3D::shared_module
  meshfem3D::mesh_hdf5_module
  meshfem3D::shared_hdf5_module
  meshfem3D::shared_adios_module
  meshfem3D::adios_methods_stubs
  meshfem3D::param_reader
  meshfem3D::version
  meshfem3D::generate_databases
)
