cmake_minimum_required(VERSION 3.10)

project(MESHFEM3D_GLOBE_MESHFEM3D LANGUAGES Fortran C)

set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

set(MESHFEM3D_GLOBE_MESH_SOURCES
  add_missing_nodes.f90
  add_topography.f90
  add_topography_410_650.f90
  add_topography_cmb.f90
  add_topography_icb.f90
  assemble_MPI_central_cube_mesh.f90
  calc_jacobian.f90
  check_mesh_resolution.f90
  compute_coordinates_grid.f90
  compute_element_properties.f90
  compute_volumes_and_areas.F90
  create_central_cube.f90
  create_central_cube_buffers.f90
  create_chunk_buffers.f90
  create_doubling_elements.f90
  create_mass_matrices.f90
  create_meshes.f90
  create_MPI_interfaces.f90
  create_regions_mesh.F90
  create_regions_elements.f90
  create_regular_elements.f90
  define_superbrick.f90
  finalize_mesher.f90
  fix_non_blocking_flags.f90
  get_absorb.f90
  get_ellipticity.f90
  get_jacobian_boundaries.f90
  get_jacobian_discontinuities.f90
  get_model.F90
  get_MPI_1D_buffers.f90
  get_MPI_cutplanes_eta.f90
  get_MPI_cutplanes_xi.f90
  get_MPI_interfaces.f90
  get_perm_color.f90
  get_shape2D.f90
  get_shape3D.f90
  gravity_integrals.F90
  initialize_layers.f90
  initialize_mesher.f90
  lgndr.f90
  meshfem3D_models.F90
  model_1dberkeley.f90
  model_1dref.f90
  model_1066a.f90
  model_ak135.f90
  model_sea1d.f90
  model_aniso_inner_core.f90
  model_aniso_mantle.f90
  model_atten3D_QRFSI12.f90
  model_atten3D_berkeley.f90
  model_attenuation_gll.f90
  model_attenuation.f90
  model_berkeley.f90
  model_bkmns.f90
  model_case65TAY.f90
  model_ccrem.f90
  model_crust_1_0.f90
  model_crust_2_0.f90
  model_crust_berkeley.f90
  model_crustmaps.f90
  model_eucrust.f90
  model_epcrust.f90
  model_full_sh.f90
  model_gapp2.f90
  model_gll.f90
  model_heterogen_mantle.f90
  model_iasp91.f90
  model_jp1d.f90
  model_jp3d.f90
  model_ppm.f90
  model_s20rts.f90
  model_s40rts.f90
  model_s362ani.f90
  model_scattering.F90
  model_sea99_s.f90
  model_sglobe.f90
  model_sglobecrust.f90
  model_sh_mars.f90
  model_spiral.f90
  moho_stretching.f90
  save_arrays_solver.f90
  save_arrays_solver_hdf5.F90
  save_model_meshfiles.f90
  save_model_meshfiles_hdf5.F90
  setup_color_perm.f90
  setup_counters.f90
  setup_inner_outer.f90
  setup_model.f90
  setup_MPI_interfaces.f90
  SIEM_meshing.f90
  stretching_function.f90
  test_MPI_interfaces.f90
  write_AVS_DX_global_chunks_data.f90
  write_AVS_DX_global_data.f90
  write_AVS_DX_global_faces_data.f90
  write_AVS_DX_output.f90
  write_AVS_DX_surface_data.f90
)

add_library(meshfem3D_globe_mesh_module meshfem3D_par.f90)
add_library(meshfem3D_globe::mesh_module ALIAS meshfem3D_globe_mesh_module)
target_link_libraries(meshfem3D_globe_mesh_module PRIVATE
  meshfem3D_globe::constants
  meshfem3D_globe::shared_module
)

add_library(meshfem3D_globe_mesh ${MESHFEM3D_GLOBE_MESH_SOURCES})
add_library(meshfem3D_globe::mesh ALIAS meshfem3D_globe_mesh)
target_link_libraries(meshfem3D_globe_mesh PRIVATE
  meshfem3D_globe::mesh_module
  meshfem3D_globe::shared
  meshfem3D_globe::shared_module
  meshfem3D_globe::shared_hdf5_module
  meshfem3D_globe::shared_adios_module
  meshfem3D_globe::version
)

add_executable(xmeshfem3D_globe meshfem3D.f90)
target_link_libraries(xmeshfem3D_globe PRIVATE
  meshfem3D_globe::mesh
  meshfem3D_globe::mesh_module
  meshfem3D_globe::shared
  meshfem3D_globe::shared_module
  meshfem3D_globe::shared_hdf5_module
  meshfem3D_globe::shared_adios_module
  meshfem3D_globe::adios_methods_stubs
  meshfem3D_globe::asdf_methods_stubs
  meshfem3D_globe::param_reader
  meshfem3D_globe::version
  specfem::version
)

target_link_libraries(meshfem3D_globe_mesh PRIVATE MPI::MPI_Fortran)
target_link_libraries(xmeshfem3D_globe PRIVATE MPI::MPI_Fortran)

if(SPECFEM_INSTALL)
  install(TARGETS xmeshfem3D_globe
    RUNTIME DESTINATION .
    COMPONENT executables
  )
endif()
