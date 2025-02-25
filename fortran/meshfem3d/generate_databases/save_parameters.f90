subroutine save_parameters()

! stores a header file of parameters needed by the solver

  use shared_parameters, only: LOCAL_PATH, MAX_STRING_LEN

  use generate_databases_par, only: &
    IOUT, myrank, sizeprocs, &
    NDIM, NSPEC_AB, NGLOB_AB, &
    NGLLX, NGLLY, NGLLZ, nfaces_surface, num_neighbors_all, NGLLSQUARE, &
    nspec2D_xmin, nspec2D_xmax, nspec2D_ymin, nspec2D_ymax, &
    NSPEC2D_BOTTOM, NSPEC2D_TOP, &
    ACOUSTIC_SIMULATION, ELASTIC_SIMULATION, POROELASTIC_SIMULATION, &
    ANISOTROPY, &
    STACEY_ABSORBING_CONDITIONS, PML_CONDITIONS, APPROXIMATE_OCEAN_LOAD, &
    USE_MESH_COLORING_GPU, &
    num_interfaces_ext_mesh, max_nibool_interfaces_ext_mesh

  use create_regions_mesh_ext_par, only: NSPEC_PORO, &
    NGLOB_OCEAN, nspec_irregular, &
    num_abs_boundary_faces, num_free_surface_faces, &
    num_coupling_ac_el_faces, num_coupling_ac_po_faces, &
    num_coupling_el_po_faces, &
    nspec_inner_acoustic, nspec_outer_acoustic, &
    nspec_inner_elastic, nspec_outer_elastic, &
    nspec_inner_poroelastic, nspec_outer_poroelastic, &
    num_phase_ispec_acoustic, num_phase_ispec_elastic, &
    num_phase_ispec_poroelastic, &
    num_colors_inner_acoustic, num_colors_outer_acoustic, &
    num_colors_inner_elastic, num_colors_outer_elastic



  ! Error handling
  integer :: ier

  ! Test write
  integer :: itest
  ! name of the database file
  character(len=MAX_STRING_LEN) :: filename
  filename = LOCAL_PATH(1:len_trim(LOCAL_PATH)) // '/mesh_parameters.bin'

  if (myrank == 0) then
    open(unit=IOUT,file=trim(filename),status='unknown',action='write',form='unformatted',iostat=ier)
    if (ier /= 0) stop 'error opening database mesh_parameters.bin'

      ! Write test parameter
      itest = 9999
      WRITE(IOUT) itest

      ! Booleans
      WRITE(IOUT) ACOUSTIC_SIMULATION
      WRITE(IOUT) ELASTIC_SIMULATION
      WRITE(IOUT) POROELASTIC_SIMULATION
      WRITE(IOUT) ANISOTROPY
      WRITE(IOUT) STACEY_ABSORBING_CONDITIONS
      WRITE(IOUT) PML_CONDITIONS
      WRITE(IOUT) APPROXIMATE_OCEAN_LOAD
      WRITE(IOUT) USE_MESH_COLORING_GPU

      ! Write test parameter
      itest = 9998
      WRITE(IOUT) itest


      ! Integers
      WRITE(IOUT) NDIM
      WRITE(IOUT) NGLLX
      WRITE(IOUT) NGLLY
      WRITE(IOUT) NGLLZ
      WRITE(IOUT) NGLLSQUARE
      WRITE(IOUT) sizeprocs

      ! Write test parameter
      itest = 9997
      WRITE(IOUT) itest

      WRITE(IOUT) NSPEC_AB ! nspec
      WRITE(IOUT) NSPEC_PORO ! nspec_poro == nspec, if POROELASTIC_SIMULATION
      WRITE(IOUT) NGLOB_AB ! nglob
      WRITE(IOUT) NGLOB_OCEAN

      ! Write test parameter
      itest = 9996
      WRITE(IOUT) itest

      WRITE(IOUT) NSPEC2D_BOTTOM
      WRITE(IOUT) NSPEC2D_TOP
      WRITE(IOUT) nspec2D_xmin
      WRITE(IOUT) nspec2D_xmax
      WRITE(IOUT) nspec2D_ymin
      WRITE(IOUT) nspec2D_ymax
      WRITE(IOUT) nspec_irregular

      ! Write test parameter
      itest = 9995
      WRITE(IOUT) itest

      WRITE(IOUT) num_neighbors_all
      WRITE(IOUT) nfaces_surface
      WRITE(IOUT) num_abs_boundary_faces
      WRITE(IOUT) num_free_surface_faces
      WRITE(IOUT) num_coupling_ac_el_faces
      WRITE(IOUT) num_coupling_ac_po_faces
      WRITE(IOUT) num_coupling_el_po_faces
      WRITE(IOUT) num_coupling_po_el_faces
      WRITE(IOUT) num_interfaces_ext_mesh
      WRITE(IOUT) max_nibool_interfaces_ext_mesh

      ! Write test parameter
      itest = 9994
      WRITE(IOUT) itest

      WRITE(IOUT) nspec_inner_acoustic
      WRITE(IOUT) nspec_outer_acoustic
      WRITE(IOUT) nspec_inner_elastic
      WRITE(IOUT) nspec_outer_elastic
      WRITE(IOUT) nspec_inner_poroelastic
      WRITE(IOUT) nspec_outer_poroelastic

      ! Write test parameter
      itest = 9993
      WRITE(IOUT) itest

      WRITE(IOUT) num_phase_ispec_acoustic
      WRITE(IOUT) num_phase_ispec_elastic
      WRITE(IOUT) num_phase_ispec_poroelastic
      WRITE(IOUT) num_colors_inner_acoustic
      WRITE(IOUT) num_colors_outer_acoustic
      WRITE(IOUT) num_colors_inner_elastic
      WRITE(IOUT) num_colors_outer_elastic

      ! Write test parameter
      itest = 9992
      WRITE(IOUT) itest
  endif

end subroutine save_parameters
