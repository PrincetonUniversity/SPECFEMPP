# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(MESHFEM3D_MESHFEM3D LANGUAGES Fortran C)

# Set the Fortran standard to 95
set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/fortran/meshfem3d/modules)
include_directories("${CMAKE_BINARY_DIR}/fortran/meshfem3d/modules")

set(MESHFEM3D_GENERATE_DATABASES_MODULE
calc_jacobian.f90
create_mass_matrices.f90
create_regions_mesh.f90
fault_generate_databases.f90
finalize_databases.f90
generate_databases_par.F90
generate_databases.f90
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
pml_set_local_dampingcoeff
read_parameters.f90
read_partition_files.F90
read_partition_files_hdf5.F90
save_arrays_solver.F90
save_arrays_solver_hdf5.F90
setup_color_perm.f90
setup_mesh_adjacency.f90
setup_mesh.f90
wavefield_discontinuity_generate_databases.f90
)

# Create parameter library module
add_library(meshfem3D_generate_databases generate_databases_par.f90)
add_library(meshfem3D::generate_databases ALIAS meshfem3D_generate_databases)
target_link_libraries(meshfem3D_generate_databases PRIVATE meshfem3D::constants meshfem3D::shared_module)
