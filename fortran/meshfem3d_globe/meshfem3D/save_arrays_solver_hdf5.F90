!=====================================================================
!
!                       S p e c f e m 3 D  G l o b e
!                       ----------------------------
!
!     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                        Princeton University, USA
!                and CNRS / University of Marseille, France
!                 (there are currently many more authors!)
! (c) Princeton University and CNRS / University of Marseille, April 2014
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
!=====================================================================


  subroutine save_arrays_solver_hdf5(idoubling,ibool,xstore,ystore,zstore, &
                                     NSPEC2D_TOP,NSPEC2D_BOTTOM)

  use constants

  use meshfem_par, only: &
    nspec

#ifdef USE_HDF5
  use shared_parameters, only: LOCAL_PATH, ATT_F_C_SOURCE, H5_COL
  use meshfem_models_par, only: &
    OCEANS,TRANSVERSE_ISOTROPY,ANISOTROPIC_3D_MANTLE, &
    ANISOTROPIC_INNER_CORE,ATTENUATION
  use meshfem_par, only: &
    nglob, iregion_code, &
    NCHUNKS,ABSORBING_CONDITIONS, &
    ROTATION,EXACT_MASS_MATRIX_FOR_ROTATION, &
    NPROCTOT, &
    ATT1,ATT2,ATT3
  use regions_mesh_par2, only: &
    xixstore,xiystore,xizstore,etaxstore,etaystore,etazstore, &
    gammaxstore,gammaystore,gammazstore, &
    rhostore,kappavstore,kappahstore,muvstore,muhstore,eta_anisostore, &
    c11store,c12store,c13store,c14store,c15store,c16store,c22store, &
    c23store,c24store,c25store,c26store,c33store,c34store,c35store, &
    c36store,c44store,c45store,c46store,c55store,c56store,c66store, &
    mu0store, &
    rmassx,rmassy,rmassz,rmass_ocean_load, &
    b_rmassx,b_rmassy, &
    ibelm_xmin,ibelm_xmax,ibelm_ymin,ibelm_ymax,ibelm_bottom,ibelm_top, &
    normal_xmin,normal_xmax,normal_ymin,normal_ymax,normal_bottom,normal_top, &
    jacobian2D_xmin,jacobian2D_xmax,jacobian2D_ymin,jacobian2D_ymax, &
    jacobian2D_bottom,jacobian2D_top, &
    rho_vp,rho_vs, &
    nspec2D_xmin,nspec2D_xmax,nspec2D_ymin,nspec2D_ymax, &
    ispec_is_tiso, &
    tau_s_store,tau_e_store,Qmu_store, &
    nglob_oceans, nglob_xy

  use manager_hdf5
#endif

  implicit none

  ! doubling mesh flag
  integer,dimension(nspec),intent(in) :: idoubling
  integer,dimension(NGLLX,NGLLY,NGLLZ,nspec),intent(in) :: ibool

  ! arrays with the mesh in double precision
  double precision,dimension(NGLLX,NGLLY,NGLLZ,nspec),intent(in) :: xstore,ystore,zstore

  ! boundary parameters locator
  integer,intent(in) :: NSPEC2D_TOP,NSPEC2D_BOTTOM

#ifdef USE_HDF5
  ! local parameters
  integer :: i,j,k,ispec,iglob,ier
  real(kind=CUSTOM_REAL),dimension(:),allocatable :: tmp_array

  ! MPI variables
  integer :: info, comm

  ! processor dependent group names
  character(len=64) :: gname_region

  ! offset arrays
  integer, dimension(0:NPROCTOT-1) :: offset_nnodes
  integer, dimension(0:NPROCTOT-1) :: offset_nnodes_xy
  integer, dimension(0:NPROCTOT-1) :: offset_nnodes_oceans
  integer, dimension(0:NPROCTOT-1) :: offset_nelems

  ! nspec2d_* arrays (for storing actual number of elements)
  integer, dimension(0:NPROCTOT-1) :: act_arr_nspec2D_xmin
  integer, dimension(0:NPROCTOT-1) :: act_arr_nspec2D_xmax
  integer, dimension(0:NPROCTOT-1) :: act_arr_nspec2D_ymin
  integer, dimension(0:NPROCTOT-1) :: act_arr_nspec2D_ymax
  integer, dimension(0:NPROCTOT-1) :: act_arr_nspec2D_bottom
  integer, dimension(0:NPROCTOT-1) :: act_arr_nspec2D_top

  ! sub array for nspec2d_* arrays (for storing the padded arrays)
  integer, dimension(0:NPROCTOT-1) :: arr_nspec2D_xmin
  integer, dimension(0:NPROCTOT-1) :: arr_nspec2D_xmax
  integer, dimension(0:NPROCTOT-1) :: arr_nspec2D_ymin
  integer, dimension(0:NPROCTOT-1) :: arr_nspec2D_ymax
  integer, dimension(0:NPROCTOT-1) :: arr_nspec2D_bottom
  integer, dimension(0:NPROCTOT-1) :: arr_nspec2D_top

  ! get MPI parameters
  call world_get_comm(comm)
  call world_get_info_null(info)

  ! initialize HDF5
  call h5_initialize() ! called in initialize_mesher()
  ! set MPI
  call h5_set_mpi_info(comm, info, myrank, NPROCTOT)

  ! share the offset info
  !
  ! offset_nnodes
  ! offset_nelems
  ! offset_n_elms_bounds
  call gather_all_all_singlei(nglob, offset_nnodes, NPROCTOT)
  call gather_all_all_singlei(nglob_xy, offset_nnodes_xy, NPROCTOT)
  call gather_all_all_singlei(nglob_oceans, offset_nnodes_oceans, NPROCTOT)
  call gather_all_all_singlei(nspec, offset_nelems, NPROCTOT)

 ! arr_nspec2D_* arrays
  call gather_all_all_singlei(nspec2D_xmin,       act_arr_nspec2D_xmin,   NPROCTOT)
  call gather_all_all_singlei(nspec2D_xmax,       act_arr_nspec2D_xmax,   NPROCTOT)
  call gather_all_all_singlei(nspec2D_ymin,       act_arr_nspec2D_ymin,   NPROCTOT)
  call gather_all_all_singlei(nspec2D_ymax,       act_arr_nspec2D_ymax,   NPROCTOT)
  call gather_all_all_singlei(NSPEC2D_BOTTOM,     act_arr_nspec2D_bottom, NPROCTOT)
  call gather_all_all_singlei(NSPEC2D_TOP,        act_arr_nspec2D_top,    NPROCTOT)
  call gather_all_all_singlei(size(ibelm_xmin),   arr_nspec2D_xmin,       NPROCTOT)
  call gather_all_all_singlei(size(ibelm_xmax),   arr_nspec2D_xmax,       NPROCTOT)
  call gather_all_all_singlei(size(ibelm_ymin),   arr_nspec2D_ymin,       NPROCTOT)
  call gather_all_all_singlei(size(ibelm_ymax),   arr_nspec2D_ymax,       NPROCTOT)
  call gather_all_all_singlei(size(ibelm_bottom), arr_nspec2D_bottom,     NPROCTOT)
  call gather_all_all_singlei(size(ibelm_top),    arr_nspec2D_top,        NPROCTOT)

  !
  ! prepare file, group and dataset by myrank == 0
  !
  if (myrank == 0) then
    ! create and open solver_data.h5
    name_database_hdf5 = LOCAL_PATH(1:len_trim(LOCAL_PATH)) // '/' // 'solver_data.h5'
    if (iregion_code == 1) then
      call h5_create_file(name_database_hdf5)
    else
      call h5_open_file(name_database_hdf5)
    endif

    ! create group for iregion_code
    write(gname_region, "('reg',i1)") iregion_code
    call h5_create_group(gname_region)

    ! open group for iregion_code
    call h5_open_group(gname_region)

    ! create dataset for offset arrays and store them by myrank == 0

    call h5_create_dataset_gen_in_group('offset_nnodes', (/NPROCTOT/), 1, 1) ! nnode = nglob
    call h5_create_dataset_gen_in_group('offset_nnodes_xy', (/NPROCTOT/), 1, 1) ! nnode = nglob_xy
    call h5_create_dataset_gen_in_group('offset_nnodes_oceans', (/NPROCTOT/), 1, 1) ! nnode = nglob_oceans
    call h5_create_dataset_gen_in_group('offset_nelems' , (/NPROCTOT/), 1, 1) ! nspec = nelem

    ! the other dataset need to be written by all ranks
    ! so here we just create the dataset
    call h5_create_dataset_gen_in_group('xstore', (/sum(offset_nnodes(:))/), 1, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('ystore', (/sum(offset_nnodes(:))/), 1, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('zstore', (/sum(offset_nnodes(:))/), 1, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('ibool', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, 1)
    call h5_create_dataset_gen_in_group('idoubling', (/sum(offset_nelems(:))/), 1, 1)
    call h5_create_dataset_gen_in_group('ispec_is_tiso', (/sum(offset_nelems(:))/), 1, 0)
    call h5_create_dataset_gen_in_group('xixstore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('xiystore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('xizstore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('etaxstore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('etaystore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('etazstore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('gammaxstore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('gammaystore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('gammazstore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('rhostore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('kappavstore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
    if (iregion_code /= IREGION_OUTER_CORE) then
      call h5_create_dataset_gen_in_group('muvstore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
    endif

    select case(iregion_code)
    case (IREGION_CRUST_MANTLE)
      ! crusl/mantle region mesh
      ! save anisotropy in the mantle only
      if (ANISOTROPIC_3D_MANTLE) then
        call h5_create_dataset_gen_in_group('c11store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c12store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c13store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c14store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c15store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c16store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c22store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c23store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c24store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c25store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c26store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c33store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c34store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c35store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c36store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c44store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c45store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c46store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c55store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c56store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c66store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
      else
        if (TRANSVERSE_ISOTROPY) then
          call h5_create_dataset_gen_in_group('kappahstore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
          call h5_create_dataset_gen_in_group('muhstore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
          call h5_create_dataset_gen_in_group('eta_anisostore', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        endif
      endif
      ! for azimutahl aniso kernels
      call h5_create_dataset_gen_in_group('mu0store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)

    case(IREGION_INNER_CORE)
      ! inner core region mesh
      if (ANISOTROPIC_INNER_CORE) then
        call h5_create_dataset_gen_in_group('c11store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c12store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c13store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c33store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('c44store', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
      endif
    end select

    ! Stacy
    if (ABSORBING_CONDITIONS) then
      if (iregion_code == IREGION_CRUST_MANTLE) then
        call h5_create_dataset_gen_in_group('rho_vp', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('rho_vs', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
      else if (iregion_code == IREGION_OUTER_CORE) then
        call h5_create_dataset_gen_in_group('rho_vp', (/NGLLX,NGLLY,NGLLZ,sum(offset_nelems(:))/), 4, CUSTOM_REAL)
      endif
    endif

    ! mass matrices
    if (((NCHUNKS /= 6 .and. ABSORBING_CONDITIONS) .and. iregion_code == IREGION_CRUST_MANTLE) .or. &
      ((ROTATION .and. EXACT_MASS_MATRIX_FOR_ROTATION) .and. iregion_code == IREGION_CRUST_MANTLE) .or. &
      ((ROTATION .and. EXACT_MASS_MATRIX_FOR_ROTATION) .and. iregion_code == IREGION_INNER_CORE)) then
      call h5_create_dataset_gen_in_group('rmassx', (/sum(offset_nnodes_xy(:))/), 1, CUSTOM_REAL)
      call h5_create_dataset_gen_in_group('rmassy', (/sum(offset_nnodes_xy(:))/), 1, CUSTOM_REAL)
    endif

    call h5_create_dataset_gen_in_group('rmassz', (/sum(offset_nnodes(:))/), 1, CUSTOM_REAL)

    ! maxx matrices for backward simulation when ROTATION is .true.
    if (ROTATION .and. EXACT_MASS_MATRIX_FOR_ROTATION) then
      if (iregion_code == IREGION_CRUST_MANTLE .or. iregion_code == IREGION_INNER_CORE) then
        call h5_create_dataset_gen_in_group('b_rmassx', (/sum(offset_nnodes_xy(:))/), 1, CUSTOM_REAL)
        call h5_create_dataset_gen_in_group('b_rmassy', (/sum(offset_nnodes_xy(:))/), 1, CUSTOM_REAL)
      endif
    endif

    ! ocean load mass matrix
    if (OCEANS .and. iregion_code == IREGION_CRUST_MANTLE) then
      call h5_create_dataset_gen_in_group('rmass_ocean_load', (/sum(offset_nnodes_oceans(:))/), 1, CUSTOM_REAL)
    endif

    ! close group for iregion_code
    call h5_close_group()
    ! close solver_data.h5
    call h5_close_file()

    ! create and open bounday.h5
    name_database_hdf5 = LOCAL_PATH(1:len_trim(LOCAL_PATH))//'/'//'boundary.h5'
    if (iregion_code == 1) then
      call h5_create_file(name_database_hdf5)
    else
      call h5_open_file(name_database_hdf5)
    endif

    ! create group for iregion_code
    write(gname_region, "('reg',i1)") iregion_code
    call h5_create_group(gname_region)
    ! open group for iregion_code
    call h5_open_group(gname_region)

    ! create node/element number dumps
    call h5_create_dataset_gen_in_group('nspec2D_xmin', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('nspec2D_xmax', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('nspec2D_ymin', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('nspec2D_ymax', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('nspec2D_bottom', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('nspec2D_top', (/NPROCTOT/), 1, 1)

    call h5_create_dataset_gen_in_group('sub_nspec2D_xmin', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('sub_nspec2D_xmax', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('sub_nspec2D_ymin', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('sub_nspec2D_ymax', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('sub_nspec2D_bottom', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('sub_nspec2D_top', (/NPROCTOT/), 1, 1)

    call h5_create_dataset_gen_in_group('ibelm_xmin', (/sum(arr_nspec2d_xmin(:))/), 1, 1)
    call h5_create_dataset_gen_in_group('ibelm_xmax', (/sum(arr_nspec2d_xmax(:))/), 1, 1)
    call h5_create_dataset_gen_in_group('ibelm_ymin', (/sum(arr_nspec2d_ymin(:))/), 1, 1)
    call h5_create_dataset_gen_in_group('ibelm_ymax', (/sum(arr_nspec2d_ymax(:))/), 1, 1)
    call h5_create_dataset_gen_in_group('ibelm_bottom', (/sum(arr_nspec2d_bottom(:))/), 1, 1)
    call h5_create_dataset_gen_in_group('ibelm_top', (/sum(arr_nspec2d_top(:))/), 1, 1)

    call h5_create_dataset_gen_in_group('normal_xmin', (/NDIM,NGLLY,NGLLZ,sum(arr_nspec2d_xmin(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('normal_xmax', (/NDIM,NGLLY,NGLLZ,sum(arr_nspec2d_xmax(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('normal_ymin', (/NDIM,NGLLX,NGLLZ,sum(arr_nspec2d_ymin(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('normal_ymax', (/NDIM,NGLLX,NGLLZ,sum(arr_nspec2d_ymax(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('normal_bottom', (/NDIM,NGLLX,NGLLY,sum(arr_nspec2d_bottom(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('normal_top', (/NDIM,NGLLX,NGLLY,sum(arr_nspec2d_top(:))/), 4, CUSTOM_REAL)

    call h5_create_dataset_gen_in_group('jacobian2D_xmin', (/NGLLY,NGLLZ,sum(arr_nspec2d_xmin(:))/), 3, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('jacobian2D_xmax', (/NGLLY,NGLLZ,sum(arr_nspec2d_xmin(:))/), 3, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('jacobian2D_ymin', (/NGLLX,NGLLZ,sum(arr_nspec2d_ymin(:))/), 3, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('jacobian2D_ymax', (/NGLLX,NGLLZ,sum(arr_nspec2d_ymax(:))/), 3, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('jacobian2D_bottom', (/NGLLX,NGLLY,sum(arr_nspec2d_bottom(:))/), 3, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('jacobian2D_top', (/NGLLX,NGLLY,sum(arr_nspec2d_top(:))/), 3, CUSTOM_REAL)

    ! close group for iregion_code
    call h5_close_group()
    ! close boundary.h5
    call h5_close_file()

    ! create and open attenuation.h5
    if (ATTENUATION) then
      name_database_hdf5 = LOCAL_PATH(1:len_trim(LOCAL_PATH))//'/'//'attenuation.h5'
      if (iregion_code == 1) then
        call h5_create_file(name_database_hdf5)
      else
        call h5_open_file(name_database_hdf5)
      endif

      ! create group for iregion_code
      write(gname_region, "('reg',i1)") iregion_code
      call h5_create_group(gname_region)
      ! open group for iregion_code
      call h5_open_group(gname_region)

      ! create dataset for attenuation
      call h5_create_dataset_gen_in_group('tau_s_store', (/N_SLS*NPROCTOT/), 1, 8)
      call h5_create_dataset_gen_in_group('tau_e_store', (/ATT1, ATT2, ATT3, N_SLS, sum(offset_nelems(:))/), 5, CUSTOM_REAL)
      call h5_create_dataset_gen_in_group('Qmu_store', (/ATT1, ATT2, ATT3, sum(offset_nelems(:))/), 4, CUSTOM_REAL)
      call h5_create_dataset_gen_in_group('att_f_c_source', (/NPROCTOT/), 1, 8)

      call h5_close_group()
      call h5_close_file()
    endif

  endif ! myrank == 0

  !
  ! write the data to the HDF5 file by all ranks
  ! (except the number of nodes/elements, which is written by myrank == 0)
  !
  ! create and open solver_data.h5
  name_database_hdf5 = LOCAL_PATH(1:len_trim(LOCAL_PATH))//'/'//'solver_data.h5'
  call h5_open_file_p_collect(name_database_hdf5)

  ! open group for iregion_code
  write(gname_region, "('reg',i1)") iregion_code
  call h5_open_group(gname_region)

  call h5_write_dataset_collect_hyperslab_in_group("offset_nnodes", (/offset_nnodes(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group("offset_nnodes_xy", (/offset_nnodes_xy(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group("offset_nnodes_oceans", (/offset_nnodes_oceans(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group("offset_nelems", (/offset_nelems(myrank)/), (/myrank/), H5_COL)

  allocate(tmp_array(nglob),stat=ier)
  if (ier /= 0 ) call exit_MPI(myrank,'Error allocating temporary array for mesh topology')

  !--- x coordinate
  tmp_array(:) = 0._CUSTOM_REAL
  do ispec = 1,nspec
    do k = 1,NGLLZ
      do j = 1,NGLLY
        do i = 1,NGLLX
          iglob = ibool(i,j,k,ispec)
          ! distinguish between single and double precision for reals
          tmp_array(iglob) = real(xstore(i,j,k,ispec), kind=CUSTOM_REAL)
        enddo
      enddo
    enddo
  enddo

  call h5_write_dataset_collect_hyperslab_in_group('xstore', tmp_array, (/sum(offset_nnodes(0:myrank-1))/), H5_COL)
  !--- y coordinate
  tmp_array(:) = 0._CUSTOM_REAL
  do ispec = 1,nspec
    do k = 1,NGLLZ
      do j = 1,NGLLY
        do i = 1,NGLLX
          iglob = ibool(i,j,k,ispec)
          ! distinguish between single and double precision for reals
          tmp_array(iglob) = real(ystore(i,j,k,ispec), kind=CUSTOM_REAL)
        enddo
      enddo
    enddo
  enddo
  call h5_write_dataset_collect_hyperslab_in_group('ystore', tmp_array, (/sum(offset_nnodes(0:myrank-1))/), H5_COL)
  !--- z coordinate
  tmp_array(:) = 0._CUSTOM_REAL
  do ispec = 1,nspec
    do k = 1,NGLLZ
      do j = 1,NGLLY
        do i = 1,NGLLX
          iglob = ibool(i,j,k,ispec)
          ! distinguish between single and double precision for reals
          tmp_array(iglob) = real(zstore(i,j,k,ispec), kind=CUSTOM_REAL)
        enddo
      enddo
    enddo
  enddo
  call h5_write_dataset_collect_hyperslab_in_group('zstore', tmp_array, (/sum(offset_nnodes(0:myrank-1))/), H5_COL)
  deallocate(tmp_array)

  call h5_write_dataset_collect_hyperslab_in_group('ibool', ibool, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('idoubling', idoubling, (/sum(offset_nelems(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('ispec_is_tiso', ispec_is_tiso, (/sum(offset_nelems(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('xixstore', xixstore, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('xiystore', xiystore, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('xizstore', xizstore, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('etaxstore', etaxstore, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('etaystore', etaystore, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('etazstore', etazstore, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('gammaxstore', gammaxstore, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('gammaystore', gammaystore, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('gammazstore', gammazstore, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('rhostore', rhostore, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('kappavstore', kappavstore, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
  if (iregion_code /= IREGION_OUTER_CORE) then
    call h5_write_dataset_collect_hyperslab_in_group('muvstore', muvstore, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
  endif

  select case(iregion_code)
  case (IREGION_CRUST_MANTLE)
    ! crusl/mantle region mesh
    ! save anisotropy in the mantle only
    if (ANISOTROPIC_3D_MANTLE) then
      call h5_write_dataset_collect_hyperslab_in_group('c11store', c11store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c12store', c12store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c13store', c13store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c14store', c14store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c15store', c15store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c16store', c16store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c22store', c22store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c23store', c23store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c24store', c24store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c25store', c25store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c26store', c26store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c33store', c33store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c34store', c34store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c35store', c35store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c36store', c36store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c44store', c44store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c45store', c45store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c46store', c46store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c55store', c55store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c56store', c56store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c66store', c66store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
    else
      if (TRANSVERSE_ISOTROPY) then
        call h5_write_dataset_collect_hyperslab_in_group('kappahstore', kappahstore, &
                                                         (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
        call h5_write_dataset_collect_hyperslab_in_group('muhstore', muhstore, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
        call h5_write_dataset_collect_hyperslab_in_group('eta_anisostore', eta_anisostore, &
                                                         (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      endif
    endif
    ! for azimutahl aniso kernels
    call h5_write_dataset_collect_hyperslab_in_group('mu0store', mu0store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
!!
  case(IREGION_INNER_CORE)
    ! inner core region mesh
    if (ANISOTROPIC_INNER_CORE) then
      call h5_write_dataset_collect_hyperslab_in_group('c11store', c11store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c12store', c12store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c13store', c13store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c33store', c33store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('c44store', c44store, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
    endif
  end select
!!
  ! Stacy
  if (ABSORBING_CONDITIONS) then
    if (iregion_code == IREGION_CRUST_MANTLE) then
      call h5_write_dataset_collect_hyperslab_in_group('rho_vp', rho_vp, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('rho_vs', rho_vs, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
    else if (iregion_code == IREGION_OUTER_CORE) then
      call h5_write_dataset_collect_hyperslab_in_group('rho_vp', rho_vp, (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
    endif
  endif
!!!
  ! mass matrices
  if (((NCHUNKS /= 6 .and. ABSORBING_CONDITIONS) .and. iregion_code == IREGION_CRUST_MANTLE) .or. &
    ((ROTATION .and. EXACT_MASS_MATRIX_FOR_ROTATION) .and. iregion_code == IREGION_CRUST_MANTLE) .or. &
    ((ROTATION .and. EXACT_MASS_MATRIX_FOR_ROTATION) .and. iregion_code == IREGION_INNER_CORE)) then
    call h5_write_dataset_collect_hyperslab_in_group('rmassx', rmassx, (/sum(offset_nnodes_xy(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('rmassy', rmassy, (/sum(offset_nnodes_xy(0:myrank-1))/), H5_COL)
  endif

  call h5_write_dataset_collect_hyperslab_in_group('rmassz', rmassz, (/sum(offset_nnodes(0:myrank-1))/), H5_COL)

  ! maxx matrices for backward simulation when ROTATION is .true.
  if (ROTATION .and. EXACT_MASS_MATRIX_FOR_ROTATION) then
    if (iregion_code == IREGION_CRUST_MANTLE .or. iregion_code == IREGION_INNER_CORE) then
      call h5_write_dataset_collect_hyperslab_in_group('b_rmassx', b_rmassx, (/sum(offset_nnodes_xy(0:myrank-1))/), H5_COL)
      call h5_write_dataset_collect_hyperslab_in_group('b_rmassy', b_rmassy, (/sum(offset_nnodes_xy(0:myrank-1))/), H5_COL)
    endif
  endif

  ! ocean load mass matrix
  if (OCEANS .and. iregion_code == IREGION_CRUST_MANTLE) then
    call h5_write_dataset_collect_hyperslab_in_group('rmass_ocean_load', rmass_ocean_load, &
                                                     (/sum(offset_nnodes_oceans(0:myrank-1))/), H5_COL)
  endif

  ! close group for iregion_code
  call h5_close_group()
  ! close solver_data.h5
  call h5_close_file_p()

  ! create and open bounday.h5
  name_database_hdf5 = LOCAL_PATH(1:len_trim(LOCAL_PATH))//'/'//'boundary.h5'
  call h5_open_file_p_collect(name_database_hdf5)

  ! create group for iregion_code
  write(gname_region, "('reg',i1)") iregion_code
  ! open group for iregion_code
  call h5_open_group(gname_region)

  call h5_write_dataset_collect_hyperslab_in_group('nspec2D_xmin', (/act_arr_nspec2d_xmin(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('nspec2D_xmax', (/act_arr_nspec2d_xmax(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('nspec2D_ymin', (/act_arr_nspec2d_ymin(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('nspec2D_ymax', (/act_arr_nspec2d_ymax(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('nspec2D_bottom', (/act_arr_nspec2d_bottom(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('nspec2D_top', (/act_arr_nspec2d_top(myrank)/), (/myrank/), H5_COL)

  call h5_write_dataset_collect_hyperslab_in_group('sub_nspec2D_xmin', (/arr_nspec2d_xmin(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('sub_nspec2D_xmax', (/arr_nspec2d_xmax(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('sub_nspec2D_ymin', (/arr_nspec2d_ymin(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('sub_nspec2D_ymax', (/arr_nspec2d_ymax(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('sub_nspec2D_bottom', (/arr_nspec2d_bottom(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('sub_nspec2D_top', (/arr_nspec2d_top(myrank)/), (/myrank/), H5_COL)

  if (arr_nspec2d_xmin(myrank) /= 0) then
    call h5_write_dataset_collect_hyperslab_in_group('ibelm_xmin', ibelm_xmin, &
                                                     (/sum(arr_nspec2d_xmin(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('normal_xmin', normal_xmin, &
                                                     (/0,0,0,sum(arr_nspec2d_xmin(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('jacobian2D_xmin', jacobian2D_xmin, &
                                                     (/0,0,sum(arr_nspec2d_xmin(0:myrank-1))/), H5_COL)
  endif
  if (arr_nspec2d_xmax(myrank) /= 0) then
    call h5_write_dataset_collect_hyperslab_in_group('ibelm_xmax', ibelm_xmax, &
                                                     (/sum(arr_nspec2d_xmax(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('normal_xmax', normal_xmax, &
                                                     (/0,0,0,sum(arr_nspec2d_xmax(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('jacobian2D_xmax', jacobian2D_xmax, &
                                                     (/0,0,sum(arr_nspec2d_xmax(0:myrank-1))/), H5_COL)
  endif
  if (arr_nspec2d_ymin(myrank) /= 0) then
    call h5_write_dataset_collect_hyperslab_in_group('ibelm_ymin', ibelm_ymin, &
                                                     (/sum(arr_nspec2d_ymin(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('normal_ymin', normal_ymin, &
                                                     (/0,0,0,sum(arr_nspec2d_ymin(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('jacobian2D_ymin', jacobian2D_ymin, &
                                                     (/0,0,sum(arr_nspec2d_ymin(0:myrank-1))/), H5_COL)
  endif
  if (arr_nspec2d_ymax(myrank) /= 0) then
    call h5_write_dataset_collect_hyperslab_in_group('ibelm_ymax', ibelm_ymax, &
                                                     (/sum(arr_nspec2d_ymax(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('normal_ymax', normal_ymax, &
                                                     (/0,0,0,sum(arr_nspec2d_ymax(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('jacobian2D_ymax', jacobian2D_ymax, &
                                                     (/0,0,sum(arr_nspec2d_ymax(0:myrank-1))/), H5_COL)
  endif
  if (arr_nspec2d_bottom(myrank) /= 0) then
    call h5_write_dataset_collect_hyperslab_in_group('ibelm_bottom', ibelm_bottom, &
                                                     (/sum(arr_nspec2d_bottom(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('normal_bottom', normal_bottom, &
                                                     (/0,0,0,sum(arr_nspec2d_bottom(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('jacobian2D_bottom', jacobian2D_bottom, &
                                                     (/0,0,sum(arr_nspec2d_bottom(0:myrank-1))/), H5_COL)
  endif
  if (arr_nspec2d_top(myrank) /= 0) then
    call h5_write_dataset_collect_hyperslab_in_group('ibelm_top', ibelm_top, &
                                                     (/sum(arr_nspec2d_top(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('normal_top', normal_top, &
                                                     (/0,0,0,sum(arr_nspec2d_top(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('jacobian2D_top', jacobian2D_top, &
                                                     (/0,0,sum(arr_nspec2d_top(0:myrank-1))/), H5_COL)
  endif

  ! close group for iregion_code
  call h5_close_group()
  ! close boundary.h5
  call h5_close_file_p()

  ! create and open attenuation.h5
  if (ATTENUATION) then
    name_database_hdf5 = LOCAL_PATH(1:len_trim(LOCAL_PATH))//'/'//'attenuation.h5'
    call h5_open_file_p_collect(name_database_hdf5)
    ! create group for iregion_code
    write(gname_region, "('reg',i1)") iregion_code
    ! open group for iregion_code
    call h5_open_group(gname_region)

    ! create dataset for attenuation
    call h5_write_dataset_collect_hyperslab_in_group('tau_s_store', tau_s_store, (/myrank*N_SLS/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('tau_e_store', tau_e_store, &
                                                     (/0,0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('Qmu_store', Qmu_store, &
                                                     (/0,0,0,sum(offset_nelems(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group('att_f_c_source', (/ATT_F_C_SOURCE/), (/myrank/), H5_COL)

    call h5_close_group()
    call h5_close_file_p()
  endif

#else
  ! no HDF5 compilation

  ! to avoid compiler warnings
  integer :: idummy

  idummy = size(idoubling,kind=4)
  idummy = size(ibool,kind=4)
  idummy = size(xstore,kind=4)
  idummy = size(ystore,kind=4)
  idummy = size(zstore,kind=4)
  idummy = max(NSPEC2D_TOP,NSPEC2D_BOTTOM)

  ! user output
  print *
  print *, "Error: HDF5 routine save_databases_hdf5() called without HDF5 Support."
  print *, "To enable HDF5 support, reconfigure with --with-hdf5 flag."
  print *
  stop 'Error HDF5 save_databases_hdf5(): called without compilation support'

#endif

  end subroutine save_arrays_solver_hdf5

!
!-----------------------------------------------------------------------
!

  subroutine save_arrays_boundary_hdf5()

! saves arrays for boundaries such as MOHO, 400 and 670 discontinuities

#ifdef USE_HDF5
  use shared_parameters, only: LOCAL_PATH, H5_COL

  use constants, only: myrank,SUPPRESS_CRUSTAL_MESH,CUSTOM_REAL, &
                        NDIM, NGLLX, NGLLY

  use meshfem_models_par, only: &
    HONOR_1D_SPHERICAL_MOHO

  use meshfem_par, only: &
    iregion_code, NPROCTOT

! boundary kernels
  use regions_mesh_par2, only: &
    NSPEC2D_MOHO, NSPEC2D_400, NSPEC2D_670, &
    ibelm_moho_top,ibelm_moho_bot,ibelm_400_top,ibelm_400_bot, &
    ibelm_670_top,ibelm_670_bot,normal_moho,normal_400,normal_670, &
    ispec2D_moho_top,ispec2D_moho_bot,ispec2D_400_top,ispec2D_400_bot, &
    ispec2D_670_top,ispec2D_670_bot

  use manager_hdf5
#endif

  implicit none

#ifdef USE_HDF5

  ! arrays for Moho, 400, 670
  integer, dimension(0:NPROCTOT-1) :: arr_nspec2d_moho_top
  integer, dimension(0:NPROCTOT-1) :: arr_nspec2d_moho_bot
  integer, dimension(0:NPROCTOT-1) :: arr_nspec2d_400_top
  integer, dimension(0:NPROCTOT-1) :: arr_nspec2d_400_bot
  integer, dimension(0:NPROCTOT-1) :: arr_nspec2d_670_top
  integer, dimension(0:NPROCTOT-1) :: arr_nspec2d_670_bot

  integer, dimension(0:NPROCTOT-1) :: act_arr_nspec2d_moho_top
  integer, dimension(0:NPROCTOT-1) :: act_arr_nspec2d_moho_bot
  integer, dimension(0:NPROCTOT-1) :: act_arr_nspec2d_400_top
  integer, dimension(0:NPROCTOT-1) :: act_arr_nspec2d_400_bot
  integer, dimension(0:NPROCTOT-1) :: act_arr_nspec2d_670_top
  integer, dimension(0:NPROCTOT-1) :: act_arr_nspec2d_670_bot

  ! processor dependent group names
  character(len=64) :: gname_region

  ! first check the number of surface elements are the same for Moho, 400, 670
  if (.not. SUPPRESS_CRUSTAL_MESH .and. HONOR_1D_SPHERICAL_MOHO) then
    if (ispec2D_moho_top /= NSPEC2D_MOHO .or. ispec2D_moho_bot /= NSPEC2D_MOHO) &
           call exit_mpi(myrank, 'Not the same number of Moho surface elements')
  endif
  if (ispec2D_400_top /= NSPEC2D_400 .or. ispec2D_400_bot /= NSPEC2D_400) &
           call exit_mpi(myrank,'Not the same number of 400 surface elements')
  if (ispec2D_670_top /= NSPEC2D_670 .or. ispec2D_670_bot /= NSPEC2D_670) &
           call exit_mpi(myrank,'Not the same number of 670 surface elements')

  ! gather the number of surface elements for Moho, 400, 670
  call gather_all_all_singlei(size(ibelm_moho_top), arr_nspec2d_moho_top, NPROCTOT)
  call gather_all_all_singlei(size(ibelm_moho_bot), arr_nspec2d_moho_bot, NPROCTOT)
  call gather_all_all_singlei(size(ibelm_400_top), arr_nspec2d_400_top, NPROCTOT)
  call gather_all_all_singlei(size(ibelm_400_bot), arr_nspec2d_400_bot, NPROCTOT)
  call gather_all_all_singlei(size(ibelm_670_top), arr_nspec2d_670_top, NPROCTOT)
  call gather_all_all_singlei(size(ibelm_670_bot), arr_nspec2d_670_bot, NPROCTOT)

  call gather_all_all_singlei(NSPEC2D_MOHO, act_arr_nspec2d_moho_top, NPROCTOT)
  call gather_all_all_singlei(NSPEC2D_MOHO, act_arr_nspec2d_moho_bot, NPROCTOT)
  call gather_all_all_singlei(NSPEC2D_400, act_arr_nspec2d_400_top, NPROCTOT)
  call gather_all_all_singlei(NSPEC2D_400, act_arr_nspec2d_400_bot, NPROCTOT)
  call gather_all_all_singlei(NSPEC2D_670, act_arr_nspec2d_670_top, NPROCTOT)
  call gather_all_all_singlei(NSPEC2D_670, act_arr_nspec2d_670_bot, NPROCTOT)

  ! file name
  name_database_hdf5 = LOCAL_PATH(1:len_trim(LOCAL_PATH))//'/'//'boundary_disc.h5'
  ! group name
  write(gname_region, "('reg',i1)") iregion_code

  ! create dataset
  if (myrank == 0) then
    ! create and open boundary.h5
    if (iregion_code == 1) then
      call h5_create_file(name_database_hdf5)
    else
      call h5_open_file(name_database_hdf5)
    endif

    ! create group for iregion_code
    call h5_create_group(gname_region)

    ! open group for iregion_code
    call h5_open_group(gname_region)

    ! write the number of surface elements for Moho, 400, 670
    call h5_create_dataset_gen_in_group('NSPEC2D_MOHO', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('NSPEC2D_400', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('NSPEC2D_670', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('sub_NSPEC2D_MOHO_top', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('sub_NSPEC2D_MOHO_bot', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('sub_NSPEC2D_400_top', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('sub_NSPEC2D_400_bot', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('sub_NSPEC2D_670_top', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('sub_NSPEC2D_670_bot', (/NPROCTOT/), 1, 1)

    ! create dataset for Moho, 400, 670
    call h5_create_dataset_gen_in_group('ibelm_moho_top', (/sum(arr_nspec2d_moho_top(:))/), 1, 1)
    call h5_create_dataset_gen_in_group('ibelm_moho_bot', (/sum(arr_nspec2d_moho_bot(:))/), 1, 1)
    call h5_create_dataset_gen_in_group('ibelm_400_top', (/sum(arr_nspec2d_400_top(:))/), 1, 1)
    call h5_create_dataset_gen_in_group('ibelm_400_bot', (/sum(arr_nspec2d_400_bot(:))/), 1, 1)
    call h5_create_dataset_gen_in_group('ibelm_670_top', (/sum(arr_nspec2d_670_top(:))/), 1, 1)
    call h5_create_dataset_gen_in_group('ibelm_670_bot', (/sum(arr_nspec2d_670_bot(:))/), 1, 1)
    call h5_create_dataset_gen_in_group('normal_moho', (/NDIM,NGLLX,NGLLY,sum(arr_nspec2d_moho_top(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('normal_400', (/NDIM,NGLLX,NGLLY,sum(arr_nspec2d_400_top(:))/), 4, CUSTOM_REAL)
    call h5_create_dataset_gen_in_group('normal_670', (/NDIM,NGLLX,NGLLY,sum(arr_nspec2d_670_top(:))/), 4, CUSTOM_REAL)

    call h5_close_group()
    call h5_close_file()

  endif

  !
  ! write datasets by all processors
  !
  ! open file and group
  call h5_open_file_p_collect(name_database_hdf5)
  call h5_open_group(gname_region)

  ! write datasets
  call h5_write_dataset_collect_hyperslab_in_group('NSPEC2D_MOHO', (/act_arr_nspec2d_moho_top(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('NSPEC2D_400', (/act_arr_nspec2d_400_top(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('NSPEC2D_670', (/act_arr_nspec2d_670_top(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('sub_NSPEC2D_MOHO_top', (/act_arr_nspec2d_moho_top(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('sub_NSPEC2D_MOHO_bot', (/act_arr_nspec2d_moho_bot(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('sub_NSPEC2D_400_top', (/act_arr_nspec2d_400_top(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('sub_NSPEC2D_400_bot', (/act_arr_nspec2d_400_bot(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('sub_NSPEC2D_670_top', (/act_arr_nspec2d_670_top(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('sub_NSPEC2D_670_bot', (/act_arr_nspec2d_670_bot(myrank)/), (/myrank/), H5_COL)

  call h5_write_dataset_collect_hyperslab_in_group('ibelm_moho_top', ibelm_moho_top, &
                                                   (/sum(arr_nspec2d_moho_top(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('ibelm_moho_bot', ibelm_moho_bot, &
                                                   (/sum(arr_nspec2d_moho_bot(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('ibelm_400_top', ibelm_400_top, (/sum(arr_nspec2d_400_top(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('ibelm_400_bot', ibelm_400_bot, (/sum(arr_nspec2d_400_bot(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('ibelm_670_top', ibelm_670_top, (/sum(arr_nspec2d_670_top(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('ibelm_670_bot', ibelm_670_bot, (/sum(arr_nspec2d_670_bot(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('normal_moho', normal_moho, &
                                                   (/0,0,0,sum(arr_nspec2d_moho_top(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('normal_400', normal_400, (/0,0,0,sum(arr_nspec2d_400_top(0:myrank-1))/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('normal_670', normal_670, (/0,0,0,sum(arr_nspec2d_670_top(0:myrank-1))/), H5_COL)

  ! close group for iregion_code
  call h5_close_group()
  ! close boundary_disc.h5
  call h5_close_file_p()

#else
  ! no HDF5 compilation

  ! user output
  print *
  print *, "Error: HDF5 routine save_databases_hdf5() called without HDF5 Support."
  print *, "To enable HDF5 support, reconfigure with --with-hdf5 flag."
  print *
  stop 'Error HDF5 save_databases_hdf5(): called without compilation support'
#endif

  end subroutine save_arrays_boundary_hdf5

!
!-------------------------------------------------------------------------------------------------
!

  subroutine save_MPI_arrays_hdf5(iregion_code,LOCAL_PATH, &
                                  num_interfaces,max_nibool_interfaces, &
                                  my_neighbors,nibool_interfaces, &
                                  ibool_interfaces, &
                                  nspec_inner,nspec_outer, &
                                  num_phase_ispec,phase_ispec_inner, &
                                  num_colors_outer,num_colors_inner, &
                                  num_elem_colors)
  use constants

#ifdef USE_HDF5
  use shared_parameters, only: H5_COL
  use meshfem_par, only: &
    NPROCTOT
  use manager_hdf5
#endif

  implicit none

  integer,intent(in) :: iregion_code

  character(len=MAX_STRING_LEN),intent(in) :: LOCAL_PATH

  ! MPI interfaces
  integer,intent(in) :: num_interfaces,max_nibool_interfaces
  integer, dimension(num_interfaces),intent(in) :: my_neighbors
  integer, dimension(num_interfaces),intent(in) :: nibool_interfaces
  integer, dimension(max_nibool_interfaces,num_interfaces),intent(in) :: &
    ibool_interfaces

  ! inner/outer elements
  integer,intent(in) :: nspec_inner,nspec_outer
  integer,intent(in) :: num_phase_ispec
  integer,dimension(num_phase_ispec,2),intent(in) :: phase_ispec_inner

  ! mesh coloring
  integer,intent(in) :: num_colors_outer,num_colors_inner
  integer, dimension(num_colors_outer + num_colors_inner),intent(in) :: &
    num_elem_colors

#ifdef USE_HDF5
  ! local parameters
  character(len=64) :: gname_region

  ! offset arrays
  integer, dimension(0:NPROCTOT-1) :: offset_num_interfaces
  integer, dimension(0:NPROCTOT-1) :: offset_max_nibool_interfaces
  integer, dimension(0:NPROCTOT-1) :: offset_nspec_inner
  integer, dimension(0:NPROCTOT-1) :: offset_nspec_outer
  integer, dimension(0:NPROCTOT-1) :: offset_num_phase_ispec
  integer, dimension(0:NPROCTOT-1) :: offset_num_colors_outer
  integer, dimension(0:NPROCTOT-1) :: offset_num_colors_inner

  ! gather the offsets
  call gather_all_all_singlei(num_interfaces, offset_num_interfaces, NPROCTOT)
  call gather_all_all_singlei(max_nibool_interfaces, offset_max_nibool_interfaces, NPROCTOT)
  call gather_all_all_singlei(nspec_inner, offset_nspec_inner, NPROCTOT)
  call gather_all_all_singlei(nspec_outer, offset_nspec_outer, NPROCTOT)
  call gather_all_all_singlei(num_phase_ispec, offset_num_phase_ispec, NPROCTOT)
  call gather_all_all_singlei(num_colors_outer, offset_num_colors_outer, NPROCTOT)
  call gather_all_all_singlei(num_colors_inner, offset_num_colors_inner, NPROCTOT)

  ! file name
  name_database_hdf5 = LOCAL_PATH(1:len_trim(LOCAL_PATH))//'/'//'solver_data_mpi.h5'
  ! group name
  write(gname_region, "('reg',i1)") iregion_code

  if (myrank == 0) then
    ! create and open solver_data_mpi.h5
    if (iregion_code == IREGION_CRUST_MANTLE) then
      call h5_create_file(name_database_hdf5)
    else
      call h5_open_file(name_database_hdf5)
    endif
    ! create group for iregion_code
    call h5_create_group(gname_region)
    ! open group for iregion_code
    call h5_open_group(gname_region)

    ! create datasets
    call h5_create_dataset_gen_in_group('offset_num_interfaces', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('offset_max_nibool_interfaces', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('offset_nspec_inner', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('offset_nspec_outer', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('offset_num_phase_ispec', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('offset_num_colors_outer', (/NPROCTOT/), 1, 1)
    call h5_create_dataset_gen_in_group('offset_num_colors_inner', (/NPROCTOT/), 1, 1)

    if (sum(offset_num_interfaces) > 0) then
      call h5_create_dataset_gen_in_group("max_nibool_interfaces", (/NPROCTOT/), 1, 1)
      call h5_create_dataset_gen_in_group("my_neighbors", (/sum(offset_num_interfaces(:))/), 1, 1)
      call h5_create_dataset_gen_in_group("nibool_interfaces", (/sum(offset_num_interfaces(:))/), 1, 1)
      call h5_create_dataset_gen_in_group("ibool_interfaces", &
                          (/maxval(offset_max_nibool_interfaces),sum(offset_num_interfaces(:))/), 2, 1)
    else
      ! dummy
      call h5_create_dataset_gen_in_group("max_nibool_interfaces", (/NPROCTOT/), 1, 1)
      call h5_create_dataset_gen_in_group("my_neighbors", (/NPROCTOT/), 1, 1)
      call h5_create_dataset_gen_in_group("nibool_interfaces", (/NPROCTOT/), 1, 1)
      call h5_create_dataset_gen_in_group("ibool_interfaces", (/1,NPROCTOT/), 2, 1)
    endif

    ! num_phase_ispec
    if (sum(offset_num_phase_ispec) > 0) then
      call h5_create_dataset_gen_in_group("phase_ispec_inner", (/sum(offset_num_phase_ispec(:)),2/), 2, 1)
    else
      ! dummy
      call h5_create_dataset_gen_in_group("phase_ispec_inner", (/NPROCTOT,2/), 2, 1)
    endif

    ! mesh coloring
    if (sum(offset_num_colors_outer) + sum(offset_num_colors_inner) > 0) then
      call h5_create_dataset_gen_in_group("num_elem_colors", &
                          (/sum(offset_num_colors_outer(:)) + sum(offset_num_colors_inner(:))/), 1, 1)
    else
      ! dummy
      call h5_create_dataset_gen_in_group("num_elem_colors", (/NPROCTOT/), 1, 1)
    endif

    ! close group for iregion_code
    call h5_close_group()
    ! close solver_data_mpi.h5
    call h5_close_file()

  endif

  !
  ! write datasets by all processors
  !

  ! open file and group
  call h5_open_file_p_collect(name_database_hdf5)
  call h5_open_group(gname_region)

  ! write datasets
  call h5_write_dataset_collect_hyperslab_in_group('offset_num_interfaces', (/offset_num_interfaces(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('offset_max_nibool_interfaces', &
                                                   (/offset_max_nibool_interfaces(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('offset_nspec_inner', (/offset_nspec_inner(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('offset_nspec_outer', (/offset_nspec_outer(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('offset_num_phase_ispec', &
                                                   (/offset_num_phase_ispec(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('offset_num_colors_outer', &
                                                   (/offset_num_colors_outer(myrank)/), (/myrank/), H5_COL)
  call h5_write_dataset_collect_hyperslab_in_group('offset_num_colors_inner', &
                                                   (/offset_num_colors_inner(myrank)/), (/myrank/), H5_COL)

  if (sum(offset_num_interfaces) > 0) then
    call h5_write_dataset_collect_hyperslab_in_group("max_nibool_interfaces", (/max_nibool_interfaces/), (/myrank/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group("my_neighbors", my_neighbors, &
                                                     (/sum(offset_num_interfaces(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group("nibool_interfaces", nibool_interfaces, &
                                                     (/sum(offset_num_interfaces(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group("ibool_interfaces", ibool_interfaces, &
                                                     (/0,sum(offset_num_interfaces(0:myrank-1))/), H5_COL)
  else
    ! dummy
    call h5_write_dataset_collect_hyperslab_in_group("max_nibool_interfaces", (/max_nibool_interfaces/), (/myrank/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group("my_neighbors", my_neighbors, (/myrank/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group("nibool_interfaces", nibool_interfaces, (/myrank/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group("ibool_interfaces", ibool_interfaces, (/0,myrank/), H5_COL)
  endif

  ! num_phase_ispec
  if (sum(offset_num_phase_ispec) > 0) then
    call h5_write_dataset_collect_hyperslab_in_group("phase_ispec_inner", phase_ispec_inner, &
                                                     (/sum(offset_num_phase_ispec(0:myrank-1)),0/), H5_COL)
  else
    ! dummy
    call h5_write_dataset_collect_hyperslab_in_group("phase_ispec_inner", phase_ispec_inner, (/myrank,0/), H5_COL)
  endif

  ! mesh coloring
  if (sum(offset_num_colors_outer) + sum(offset_num_colors_inner) > 0) then
    call h5_write_dataset_collect_hyperslab_in_group("num_elem_colors", num_elem_colors, &
                          (/sum(offset_num_colors_outer(0:myrank-1)) + sum(offset_num_colors_inner(0:myrank-1))/), H5_COL)
  else
    ! dummy
    call h5_write_dataset_collect_hyperslab_in_group("num_elem_colors", num_elem_colors, (/myrank/), H5_COL)
  endif

  ! close group for iregion_code
  call h5_close_group()
  ! close solver_data_mpi.h5
  call h5_close_file_p()

#else
  ! no HDF5 compilation

  ! to avoid compiler warnings
  integer :: idummy

  idummy = iregion_code
  idummy = len_trim(LOCAL_PATH)
  idummy = size(ibool_interfaces,kind=4)
  idummy = size(my_neighbors,kind=4)
  idummy = size(nibool_interfaces,kind=4)
  idummy = max(nspec_inner,nspec_outer)
  idummy = size(num_elem_colors)
  idummy = size(phase_ispec_inner)

  print * , "Error: HDF5 routine save_MPI_arrays_hdf5() called without HDF5 Support."
  print * , "To enable HDF5 support, reconfigure with --with-hdf5 flag."
  stop 'Error HDF5 save_MPI_arrays_hdf5(): called without compilation support'
#endif

  end subroutine save_MPI_arrays_hdf5

!
!-------------------------------------------------------------------------------------------------
!

  subroutine get_absorb_stacey_boundary_hdf5(iregion, num_abs_boundary_faces, &
                                         abs_boundary_ispec,abs_boundary_npoin, &
                                         abs_boundary_ijk,abs_boundary_normal,abs_boundary_jacobian2Dw)

  use constants, only: NDIM,NGLLX,NGLLY,NGLLSQUARE,CUSTOM_REAL,MAX_STRING_LEN,IREGION_CRUST_MANTLE

#ifdef USE_HDF5
  use constants, only: myrank
  use shared_parameters, only: H5_COL
  use meshfem_par, only: LOCAL_PATH, NPROCTOT
  use manager_hdf5
#endif

  implicit none

  integer,intent(in) :: iregion

  ! absorbing boundary arrays
  integer,intent(in) :: num_abs_boundary_faces
  integer, dimension(num_abs_boundary_faces), intent(in) :: abs_boundary_ispec
  integer, dimension(num_abs_boundary_faces), intent(in) :: abs_boundary_npoin
  integer, dimension(3,NGLLSQUARE,num_abs_boundary_faces), intent(in) :: abs_boundary_ijk
  real(kind=CUSTOM_REAL), dimension(NDIM,NGLLSQUARE,num_abs_boundary_faces), intent(in) :: abs_boundary_normal
  real(kind=CUSTOM_REAL), dimension(NGLLSQUARE,num_abs_boundary_faces), intent(in) :: abs_boundary_jacobian2Dw

#ifdef USE_HDF5
  ! local parameters
  integer, dimension(0:NPROCTOT-1) :: offset_num_abs_boundary_faces
  ! MPI parameters
  integer :: comm, info

  ! dummy arrays
  integer, dimension(1,1), parameter                    :: i2d_dummy = reshape((/0/),(/1,1/))
  integer, dimension(1,1,1), parameter                  :: i3d_dummy = reshape((/0/),(/1,1,1/))
  real(kind=CUSTOM_REAL), dimension(1,1), parameter     :: r2d_dummy = reshape((/0.0/),(/1,1/))
  real(kind=CUSTOM_REAL), dimension(1,1,1), parameter   :: r3d_dummy = reshape((/0.0/),(/1,1,1/))

  ! variables for HDF5
  character(len=64) :: gname_region

  ! get MPI parameters
  call world_get_comm(comm)
  call world_get_info_null(info)

  ! initialize HDF5
  call h5_initialize() ! called in initialize_mesher()
  ! set MPI
  call h5_set_mpi_info(comm, info, myrank, NPROCTOT)

  ! file name
  name_database_hdf5 = LOCAL_PATH(1:len_trim(LOCAL_PATH))//'/'//'stacey.h5'
  ! group name
  write(gname_region, "('reg',i1)") iregion

  ! get offset arrays
  call gather_all_all_singlei(num_abs_boundary_faces, offset_num_abs_boundary_faces, NPROCTOT)

  ! create datasets by myrank=0
  if (myrank == 0) then
    ! create and open stacey.h5
    if (iregion == IREGION_CRUST_MANTLE) then
      call h5_create_file(name_database_hdf5)
    else
      call h5_open_file(name_database_hdf5)
    endif

    ! create group for iregion_code
    call h5_create_group(gname_region)
    ! open group for iregion_code
    call h5_open_group(gname_region)

    call h5_create_dataset_gen_in_group("num_abs_boundary_faces", (/NPROCTOT/), 1, 1)

    if (sum(offset_num_abs_boundary_faces) > 0) then
      call h5_create_dataset_gen_in_group("abs_boundary_ispec", (/sum(offset_num_abs_boundary_faces(:))/), 1, 1)
      call h5_create_dataset_gen_in_group("abs_boundary_npoin", (/sum(offset_num_abs_boundary_faces(:))/), 1, 1)
      call h5_create_dataset_gen_in_group("abs_boundary_ijk", &
                                          (/NDIM,NGLLSQUARE,sum(offset_num_abs_boundary_faces(:))/), 3, CUSTOM_REAL)
      call h5_create_dataset_gen_in_group("abs_boundary_jacobian2Dw", &
                                          (/NGLLSQUARE,sum(offset_num_abs_boundary_faces(:))/), 2, CUSTOM_REAL)

      if (iregion == IREGION_CRUST_MANTLE) then
        call h5_create_dataset_gen_in_group("abs_boundary_normal", &
                                            (/NDIM,NGLLSQUARE,sum(offset_num_abs_boundary_faces(:))/), 3, CUSTOM_REAL)
      endif
    else
      ! dummy
      call h5_create_dataset_gen_in_group("abs_boundary_ispec", (/NPROCTOT/), 1, 1)
      call h5_create_dataset_gen_in_group("abs_boundary_npoin", (/NPROCTOT/), 1, 1)
      call h5_create_dataset_gen_in_group("abs_boundary_ijk", (/NDIM,NGLLSQUARE,NPROCTOT/), 3, CUSTOM_REAL)
      call h5_create_dataset_gen_in_group("abs_boundary_jacobian2Dw", (/NGLLSQUARE,NPROCTOT/), 2, CUSTOM_REAL)

      if (iregion == IREGION_CRUST_MANTLE) then
        call h5_create_dataset_gen_in_group("abs_boundary_normal", (/NDIM,NGLLSQUARE,NPROCTOT/), 3, CUSTOM_REAL)
      endif
    endif ! iregion == IREGION_CRUST_MANTEL

    ! close group for iregion_code
    call h5_close_group()
    ! close stacey.h5
    call h5_close_file()
  endif ! myrank == 0

  ! write datasets by all processors
  call h5_open_file_p_collect(name_database_hdf5)
  call h5_open_group(gname_region)

  ! write datasets
  call h5_write_dataset_collect_hyperslab_in_group("num_abs_boundary_faces", &
                                                   (/offset_num_abs_boundary_faces(myrank)/), (/myrank/), .true.)

  if (sum(offset_num_abs_boundary_faces) > 0) then
    call h5_write_dataset_collect_hyperslab_in_group("abs_boundary_ispec", abs_boundary_ispec, &
                                                     (/sum(offset_num_abs_boundary_faces(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group("abs_boundary_npoin", abs_boundary_npoin, &
                                                     (/sum(offset_num_abs_boundary_faces(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group("abs_boundary_ijk", abs_boundary_ijk, &
                                                     (/0,0,sum(offset_num_abs_boundary_faces(0:myrank-1))/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group("abs_boundary_jacobian2Dw", abs_boundary_jacobian2Dw, &
                                                     (/0,sum(offset_num_abs_boundary_faces(0:myrank-1))/), H5_COL)
    if (iregion == IREGION_CRUST_MANTLE) then
      call h5_write_dataset_collect_hyperslab_in_group("abs_boundary_normal", abs_boundary_normal, &
                                                       (/0,0,sum(offset_num_abs_boundary_faces(0:myrank-1))/), H5_COL)
    endif
  else
    ! dummy
    call h5_write_dataset_collect_hyperslab_in_group("abs_boundary_ispec", (/0/), (/myrank/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group("abs_boundary_npoin", (/0/), (/myrank/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group("abs_boundary_ijk", i3d_dummy, (/0,0,myrank/), H5_COL)
    call h5_write_dataset_collect_hyperslab_in_group("abs_boundary_jacobian2Dw", r2d_dummy, (/0,myrank/), H5_COL)
    if (iregion == IREGION_CRUST_MANTLE) then
      call h5_write_dataset_collect_hyperslab_in_group("abs_boundary_normal", r3d_dummy, (/0,0,myrank/), H5_COL)
    endif
  endif

  ! close group for iregion_code
  call h5_close_group()
  ! close stacey.h5
  call h5_close_file_p()

#else
  ! no HDF5 compilation

  ! to avoid compiler warnings
  integer :: idummy

  idummy = iregion
  idummy = size(abs_boundary_ispec,kind=4)
  idummy = size(abs_boundary_npoin)
  idummy = size(abs_boundary_ijk)
  idummy = size(abs_boundary_normal,kind=4)
  idummy = size(abs_boundary_jacobian2dw,kind=4)

  print * , "Error: HDF5 routine get_absorb_stacey_boundary_hdf5() called without HDF5 Support."
  print * , "To enable HDF5 support, reconfigure with --with-hdf5 flag."
  stop 'Error HDF5 get_absorb_stacey_boundary_hdf5(): called without compilation support'
#endif

  end subroutine get_absorb_stacey_boundary_hdf5
