# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(MESHFEM3D_MESHFEM3D LANGUAGES Fortran C)

# Set the Fortran standard to 95
set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../modules)
# include_directories("${CMAKE_Fortran_MODULE_DIRECTORY}")

set(MESHFEM3D_MESH_MODULE
    calc_gll_points.f90
    check_mesh_quality.f90
    chunk_earth_mesh_mod.f90
    compute_parameters.f90
    create_meshfem_mesh.f90
    create_CPML_regions.f90
    create_interfaces_mesh.f90
    create_visual_files.f90
    define_subregions.f90
    define_subregions_heuristic.f90
    define_superbrick.f90
    determine_cavity.f90
    earth_chunk.f90
    get_flags_boundaries.f90
    get_MPI_cutplanes_eta.f90
    get_MPI_cutplanes_xi.f90
    meshfem3D.F90
    get_wavefield_discontinuity.f90
    command_line_arguments.f90
    read_mesh_parameter_file.f90
    read_value_mesh_parameters.f90
    save_databases.F90
    store_boundaries.f90
    store_coords.f90
)

if (ADIOS)
  message(STATUS "ADIOS enabled")
  list(APPEND MESHFEM3D_MESH_MODULE
          save_databases_adios.F90
  )
endif(ADIOS)


add_library(meshfem3D_mesh_module meshfem3D_par.f90)
add_library(meshfem3D::mesh_module ALIAS meshfem3D_mesh_module)
target_link_libraries(meshfem3D_mesh_module PRIVATE meshfem3D::constants meshfem3D::shared_module)

add_library(meshfem3D_mesh_hdf5_module
    save_databases_hdf5.F90
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

add_executable(xmeshfem3D meshfem3D.F90)
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
if (ADIOS)
  target_link_libraries(meshfem3D_mesh_hdf5_module PRIVATE
    meshfem3D::shared_adios_module
  )
endif(ADIOS)
