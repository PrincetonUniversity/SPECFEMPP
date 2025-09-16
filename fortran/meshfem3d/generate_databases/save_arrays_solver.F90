!=====================================================================
!
!                          S p e c f e m 3 D
!                          -----------------
!
!     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                              CNRS, France
!                       and Princeton University, USA
!                 (there are currently many more authors!)
!                           (c) October 2017
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

module save_arrays_module
  implicit none
  contains

  subroutine save_control_nodes_indexing(nspec, indices)

    use constants, only: IOUT, CUSTOM_REAL
    use generate_databases_par, only: NGNOD

    implicit none

    integer, intent(in) :: nspec
    integer, dimension(:, :), intent(in) :: indices
    integer :: ispec

    ! Check if the size of the indices array is correct
    if (size(indices, 1) /= NGNOD .or. size(indices, 2) /= nspec) then
      write(*,*) 'Error: size of the indices array is not correct'
      stop
    endif

    ! Save the indices array element by element
    do ispec = 1, nspec
      write(IOUT) indices(:, ispec)
    end do

  end subroutine save_control_nodes_indexing

  subroutine save_control_nodes_array(nnodes, array)

    use constants, only: IOUT, CUSTOM_REAL
    use generate_databases_par, only: NGNOD

    implicit none

    integer, intent(in) :: nnodes
    integer :: i
    double precision, dimension(:, :), intent(in) :: array
    real(kind=CUSTOM_REAL), dimension(:), allocatable :: array_real

    ! Allocate the real array with the correct size
    allocate(array_real(3))
    ! Check if the size of the array is correct
    if (size(array, 1) /= 3 .or. size(array, 2) /= nnodes) then
      write(*,*) 'Error (control nodes): size of the array is not correct. ', &
                 'Expected size: (3, ', nnodes, ') Actual size: (', &
                 size(array, 1), ', ', size(array, 2), ')'
      stop
    endif

    ! Save the array node by node
    do i = 1, nnodes
      ! Convert the double precision array to real kind
      write(IOUT) real(array(:, i), kind=CUSTOM_REAL)
    end do

  end subroutine save_control_nodes_array

  subroutine save_global_arrays(nspec, array, name)

    use constants, only: IOUT, CUSTOM_REAL

    implicit none

    integer, intent(in) :: nspec
    integer :: ispec
    character(len=*), intent(in) :: name
    real(kind=CUSTOM_REAL), dimension(:, :, :, :), intent(in) :: array

    ! Check if the size of the array is correct
    if (size(array, 4) /= nspec) then
      write(*,*) 'Error (', name, '): size of the array is not correct. ', &
                 'Expected size: ', nspec, ' Actual size: ', size(array, 4)
      stop
    endif

    ! Save the array element by element
    do ispec = 1, nspec
      WRITE(IOUT) array(:, :, :, ispec)
    end do

  end subroutine save_global_arrays

  subroutine save_inner_outer_arrays(num_inner, array)

    use constants, only: IOUT, CUSTOM_REAL

    implicit none

    integer, intent(in) :: num_inner
    integer :: ispec
    integer, dimension(:, :), intent(in) :: array

    ! Check if the size of the array is correct
    if (size(array, 1) /= num_inner) then
      write(*,*) 'Error (inner/outer): size of the array is not correct. ', &
                 'Expected size: (', num_inner, ', 3) Actual size: (', &
                 size(array, 1), ', ', size(array, 2), ')'
      stop
    endif

    ! Save the array element by element
    do ispec = 1, num_inner
      WRITE(IOUT) array(ispec, :)
    end do

  end subroutine save_inner_outer_arrays

  subroutine save_global_arrays_with_components(nspec, array)

      use constants, only: IOUT, CUSTOM_REAL

      implicit none

      integer, intent(in) :: nspec
      integer :: ispec
      real(kind=CUSTOM_REAL), dimension(:, :, :, :, :), intent(in) :: array

      ! Check if the size of the array is correct
      if (size(array, 5) /= nspec) then
        write(*,*) 'Error (with components): size of the array is not correct. ', &
                   'Expected size: ', nspec, ' Actual size: ', size(array, 5)
        stop
      endif

      ! Save the array element by element
      do ispec = 1, nspec
        WRITE(IOUT) array(:, :, :, :, ispec)
      end do

  end subroutine save_global_arrays_with_components

  subroutine save_boundary_arrays(num_faces, spec_boundary, ijk_boundary, &
                            jacobian2Dw_boundary, normal_boundary)

    use constants, only: IOUT, CUSTOM_REAL
    implicit none

    integer, intent(in) :: num_faces
    integer :: iface
    integer, dimension(:), intent(in) :: spec_boundary
    integer, dimension(:,:,:), intent(in) :: ijk_boundary
    real(kind=CUSTOM_REAL), dimension(:,:), intent(in) :: jacobian2Dw_boundary
    real(kind=CUSTOM_REAL), dimension(:,:,:), intent(in) :: normal_boundary

    ! Save the spec_boundary array
    write(iout) spec_boundary

    ! save the ijk_boundary array
    do iface = 1, num_faces
      write(iout) ijk_boundary(:, :, iface)
    end do

    ! save the jacobian2dw_boundary array
    do iface = 1, num_faces
      write(iout) jacobian2dw_boundary(:, iface)
    end do

    ! save the normal_boundary array
    do iface = 1, num_faces
      write(iout) normal_boundary(:, :, iface)
    end do

  end subroutine save_boundary_arrays

  subroutine print_ijk(iface, ijk_boundary)

    use constants, only: IOUT, CUSTOM_REAL, NGLLSQUARE

    implicit none

    integer :: i
    integer, intent(in) :: iface
    integer, dimension(:,:,:), intent(in) :: ijk_boundary
    character(len=3) :: compname = "ijk"

    ! save the ijk_boundary array
    write (*,*) "Mapping values of ijk_boundary for face ", iface
    write (*,*) "----------------------------------------------"
    write (*,*) ""
    write (*,*) "   ---> igll square"
    write (*,*) "   "
    do i = 1, 3
      write(*, "(A)", advance='no') "ijk_boundary(", compname(i:i), ") = "
      write(*,*) ijk_boundary(i, :, iface)
    end do

  end subroutine print_ijk


  ! subroutine that converts a logical array to an integer array and writes it
  ! a file IOUT (see save_global_arrays)
  subroutine save_ispec_is_arrays(nspec, ispec_is_acoustic, ispec_is_elastic, ispec_is_poroelastic)

    use constants, only: IOUT

    implicit none

    integer, intent(in) :: nspec
    logical, dimension(nspec), intent(in) :: ispec_is_acoustic, ispec_is_elastic, ispec_is_poroelastic

    ! To be written to file
    integer, dimension(nspec) :: array_int

    integer :: i

    ! To be written to file
    integer :: nspec_acoustic = 0, nspec_elastic = 0, nspec_poroelastic = 0

    ! Loop over the number of element types in the array and convert the logical
    ! values to integer values
    do i = 1, nspec
      array_int(i) = 0
      if (ispec_is_acoustic(i)) then
        array_int(i) = 0
        nspec_acoustic = nspec_acoustic + 1
      elseif (ispec_is_elastic(i)) then
        array_int(i) = 1
        nspec_elastic = nspec_elastic + 1
      elseif (ispec_is_poroelastic(i)) then
        array_int(i) = 2
        nspec_poroelastic = nspec_poroelastic + 1
      else
        write(*,*) 'Error: ispec_is arrays are not correct'
        stop
      end if
    end do
    write(IOUT) nspec_acoustic
    write(IOUT) nspec_elastic
    write(IOUT) nspec_poroelastic
    write(IOUT) array_int

  end subroutine save_ispec_is_arrays

  subroutine print_ibool_element(ispec)

    use constants, only: NGLLX, NGLLY, NGLLZ
    use generate_databases_par, only: ibool

    implicit none

    integer, intent(in) :: ispec

    ! Local variables
    integer :: i,j,k

    write (*,*) "Mapping values of ibool for element ", ispec
    write (*,*) "----------------------------------------------"
    write (*,*) ""
    write (*,*) "   |---> igllx"
    write (*,*) "   |"
    write (*,*) "   V"
    write (*,*) "iglly"
    write (*,*) "  "
    write (*,*) "ibool:"

    ! Print the ibool array for a given ispec
    do k = 1, NGLLZ
      write(*,'(A,I0)', advance='no') "igllz=",k
      do j = 1, NGLLY
        if (j > 1) write(*,'(A)', advance='no') "       "
        write(*,*) ibool(:,j,k,ispec)
      end do
      write (*,*) ""
    end do

  end subroutine print_ibool_element

  subroutine print_unique_at_ispec(ispec, array_name, array)

    use constants, only: NGLLX, NGLLY, NGLLZ, CUSTOM_REAL

    use generate_databases_par, only: ibool

    implicit none

    character(len=*) :: array_name
    real(kind=CUSTOM_REAL), dimension(:), intent(in) :: array
    integer, intent(in) :: ispec

    ! Local variables
    integer :: i,j,k, iglob
    real(kind=CUSTOM_REAL), dimension(NGLLX, NGLLY, NGLLZ) :: sub_array
    write (*,*) "Mapping values of array", array_name
    write (*,*) "---------------------------------"
    write (*,*) ""
    write (*,*) "   |---> igllx"
    write (*,*) "   |"
    write (*,*) "   V"
    write (*,*) "iglly"
    write (*,*) "  "
    write (*,*) "array:"

    ! Print the array
    do i = 1, NGLLX
      do j = 1, NGLLY
        do k = 1, NGLLZ
          iglob = ibool(i,j,k,ispec)
          sub_array(i,j,k) = array(iglob)
        end do
      end do
    end do

    do k = 1, NGLLZ
      write(*,'(A,I0)', advance='no') "igllz=",k
      do j = 1, NGLLY
        if (j > 1) write(*,'(A)', advance='no') "       "
        write(*,*) sub_array(:,j,k)
      end do
      write (*,*) ""
    end do

  end subroutine print_unique_at_ispec

  subroutine print_global_array(ispec, sub_array, array_name)

    use constants, only: NGLLX, NGLLY, NGLLZ, CUSTOM_REAL

    implicit none

    real(kind=CUSTOM_REAL), dimension(:,:,:), intent(in) :: sub_array
    character(len=*) :: array_name
    integer, intent(in) :: ispec

    ! Local variables
    integer :: i,j,k

    write (*,*) "Mapping values of array", array_name, "for element ", ispec
    write (*,*) "-------------------------------------------------"
    write (*,*) ""
    write (*,*) "   |---> igllx"
    write (*,*) "   |"
    write (*,*) "   V"
    write (*,*) "iglly"
    write (*,*) "  "
    write (*,*) "array:"

    ! Print the sub_array
    do k = 1, NGLLZ
      write(*,'(A,I0)', advance='no') "igllz=",k
      do j = 1, NGLLY
        if (j > 1) write(*,'(A)', advance='no') "       "
        write(*,*) sub_array(:,j,k)
      end do
      write (*,*) ""
    end do

  end subroutine print_global_array

end module save_arrays_module
! for external mesh

  subroutine save_arrays_solver_mesh()

  use constants, only: IMAIN,IOUT,myrank
  use save_arrays_module, only: &
    save_global_arrays, &
    save_global_arrays_with_components, &
    save_ispec_is_arrays, &
    save_boundary_arrays, &
    save_inner_outer_arrays, &
    print_ibool_element, &
    print_unique_at_ispec, &
    print_global_array, &
    print_ijk, &
    save_control_nodes_indexing, &
    save_control_nodes_array

  use shared_parameters, only: ACOUSTIC_SIMULATION, ELASTIC_SIMULATION, POROELASTIC_SIMULATION, &
    APPROXIMATE_OCEAN_LOAD, SAVE_MESH_FILES, ANISOTROPY

  use shared_parameters, only: COUPLE_WITH_INJECTION_TECHNIQUE
  use generate_databases_par, only: MESH_A_CHUNK_OF_THE_EARTH,NGNOD

  ! global indices
  use generate_databases_par, only: nspec => NSPEC_AB, ibool

  use generate_databases_par, only: &
    nspec2D_xmin,nspec2D_xmax,nspec2D_ymin,nspec2D_ymax,NSPEC2D_BOTTOM,NSPEC2D_TOP, &
    ibelm_xmin,ibelm_xmax,ibelm_ymin,ibelm_ymax,ibelm_bottom,ibelm_top, &
    SIMULATION_TYPE,SAVE_FORWARD, &
    STACEY_ABSORBING_CONDITIONS,USE_MESH_COLORING_GPU

  ! MPI interfaces
  use generate_databases_par, only: num_interfaces_ext_mesh,my_neighbors_ext_mesh, &
    nibool_interfaces_ext_mesh,ibool_interfaces_ext_mesh

  ! PML
  use generate_databases_par, only: PML_CONDITIONS, &
    nspec_cpml,CPML_width_x,CPML_width_y,CPML_width_z,CPML_to_spec, &
    CPML_regions,is_CPML,min_distance_between_CPML_parameter, &
    d_store_x,d_store_y,d_store_z,k_store_x,k_store_y,k_store_z, &
    alpha_store_x,alpha_store_y,alpha_store_z, &
    nglob_interface_PML_acoustic,points_interface_PML_acoustic, &
    nglob_interface_PML_elastic,points_interface_PML_elastic, nnodes_ext_mesh, &
    nodes_coords_ext_mesh,elmnts_ext_mesh

  ! mesh surface
  use generate_databases_par, only: ispec_is_surface_external_mesh,iglob_is_surface_external_mesh, &
    nfaces_surface

  ! mesh adjacency
  use generate_databases_par, only: neighbors_xadj,neighbors_adjncy,num_neighbors_all

  use create_regions_mesh_ext_par

  use shared_parameters, only: ADIOS_FOR_MESH,HDF5_ENABLED

  !! setup wavefield discontinuity interface
  use shared_parameters, only: IS_WAVEFIELD_DISCONTINUITY
  use wavefield_discontinuity_generate_databases, only: &
                              save_arrays_wavefield_discontinuity


  implicit none

  ! local parameters
  integer, dimension(:,:), allocatable :: ibool_interfaces_ext_mesh_dummy
  integer :: max_nibool_interfaces_ext_mesh
  integer :: nglob, nspec_irregular_out
  integer :: ier,i,itest
  character(len=MAX_STRING_LEN) :: filename

  ! selects routine for file i/o format
  if (ADIOS_FOR_MESH) then
    ! ADIOS
    ! call save_arrays_solver_mesh_adios()

    print *, 'ADIOS is not supported for mesh databases'
    stop
    ! all done
    return
  else if (HDF5_ENABLED) then
    ! HDF5
    ! call save_arrays_solver_mesh_hdf5()

    print *, 'HDF5 is not supported for mesh databases'
    stop
    ! all done
    return
  else
    ! default binary
    ! implemented here below, continue
    continue
  endif

  ! number of unique global nodes
  nglob = nglob_unique

  ! database file name
  filename = prname(1:len_trim(prname))//'external_mesh.bin'

  ! user output
  if (myrank == 0) then
    write(IMAIN,*) '     using binary file format'
    write(IMAIN,*) '     database file (for rank 0): ',trim(filename)
    write(IMAIN,*)
    call flush_IMAIN()
  endif

  ! saves mesh file proc***_external_mesh.bin
  open(unit=IOUT,file=trim(filename),status='unknown',action='write',form='unformatted',iostat=ier)
  if (ier /= 0) stop 'error opening database proc######_external_mesh.bin'

  write(IOUT) nspec
  write(IOUT) nglob
  write(IOUT) NGNOD
  write(IOUT) nspec_irregular

  do i = 1, nspec
    write(IOUT) ibool(:,:,:,i)
  end do

  ! Debugging the array layout for a given element
  ! call print_ibool_element(1)
  ! call print_ibool_element(nspec)

  write(IOUT) xstore_unique
  write(IOUT) ystore_unique
  write(IOUT) zstore_unique

  ! Debugging the array layout for a given element
  ! call print_unique_at_ispec(1, 'x', xstore_unique)
  ! call print_unique_at_ispec(1, 'y', ystore_unique)
  ! call print_unique_at_ispec(1, 'z', zstore_unique)

  write(IOUT) irregular_element_number
  write(IOUT) xix_regular
  write(IOUT) jacobian_regular

  ! write (*,*) "Number of irregular elements: ", nspec_irregular
  ! write (*,*) "Size of xixstore: ", shape(xixstore)

  ! This is a fix for when the number irregular elements is 0.
  ! Because the mesh database reader still expects a single element
  ! 4-dimensional array.
  if (nspec_irregular > 0) then
    nspec_irregular_out = nspec_irregular
  else
    nspec_irregular_out = 1
  endif

  ! Debugging the array layout for a given element
  ! call print_global_array(1, xixstore(1,:,:,:), "xix")
  call save_global_arrays(nspec_irregular_out, xixstore, 'xix')
  call save_global_arrays(nspec_irregular_out, xiystore, 'xi')
  call save_global_arrays(nspec_irregular_out, xizstore, 'zi')
  call save_global_arrays(nspec_irregular_out, etaxstore, 'etax')
  call save_global_arrays(nspec_irregular_out, etaystore, 'etay')
  ! Debugging the array layout for a given element
  ! call print_global_array(1, etaystore(1,:,:,:), "etay")
  call save_global_arrays(nspec_irregular_out, etazstore, 'etaz')
  call save_global_arrays(nspec_irregular_out, gammaxstore, 'gammax')
  call save_global_arrays(nspec_irregular_out, gammaystore, 'gammay')
  call save_global_arrays(nspec_irregular_out, gammazstore, 'gammaz')
  ! Debugging the array layout for a given element
  ! call print_global_array(1, gammazstore(1,:,:,:), "gammaz")
  call save_global_arrays(nspec_irregular_out, jacobianstore, 'jacobian')


  ! write test value
  itest = 10000
  write(IOUT) itest

  ! Store arrays related to the control nodes
  call save_control_nodes_indexing(nspec, elmnts_ext_mesh)
  call save_control_nodes_array(nnodes_ext_mesh, nodes_coords_ext_mesh)

  call save_global_arrays(nspec, kappastore, 'kappa')
  call save_global_arrays(nspec, mustore, 'mu')

  ! save_ispec_is_arrays writes the following:
  ! the number of each type of element
  !   nspec_acoustic
  !   nspec_elastic
  !   nspec_poroelastic
  !
  ! single integer array of nspec
  ! represents the material type of each element
  ! 0 = acoustic, 1 = elastic, 2 = poroelastic

  call save_ispec_is_arrays(nspec, ispec_is_acoustic, &
                                   ispec_is_elastic, &
                                   ispec_is_poroelastic)

  ! stamp for checking i/o
  itest = 9999
  write(IOUT) itest

  ! acoustic
  if (ACOUSTIC_SIMULATION) then
    write(IOUT) rmass_acoustic
  endif

  ! this array is needed for acoustic simulations but also for elastic simulations with CPML,
  ! thus we allocate it and read it in all cases (whether the simulation is acoustic, elastic, or acoustic/elastic)
  call save_global_arrays(nspec, rhostore, 'rho')

  ! write test value
  itest = 9998
  write(IOUT) itest

  ! elastic
  if (ELASTIC_SIMULATION) then
    write(IOUT) rmass
    if (APPROXIMATE_OCEAN_LOAD) then
      write(IOUT) rmass_ocean_load
    endif
    ! Stacey
    call save_global_arrays(nspec, rho_vp, 'rho_vp')
    call save_global_arrays(nspec, rho_vs, 'rho_vs')
  endif

  ! Write a test value
  itest = 9997
  write(IOUT) itest

  ! poroelastic
  if (POROELASTIC_SIMULATION) then
    write(IOUT) rmass_solid_poroelastic
    write(IOUT) rmass_fluid_poroelastic

    call save_global_arrays_with_components(nspec, rhoarraystore)
    call save_global_arrays_with_components(nspec, kappaarraystore)
    call save_global_arrays(nspec, etastore, 'eta')
    call save_global_arrays(nspec, tortstore, 'tort')
    call save_global_arrays_with_components(nspec, permstore)
    call save_global_arrays(nspec, phistore, 'phi')
    call save_global_arrays(nspec, rho_vpI, 'rho_vpI')
    call save_global_arrays(nspec, rho_vpII, 'rho_vpII')
    call save_global_arrays(nspec, rho_vsI, 'rho_vsI')

  endif

  ! write test value
  itest = 9996
  write(IOUT) itest

  ! @Lucas & @Congyue need to uncomment this when implementing PML

  ! C-PML absorbing boundary conditions
  ! if (PML_CONDITIONS) then
  !   write(IOUT) nspec_cpml
  !   write(IOUT) CPML_width_x
  !   write(IOUT) CPML_width_y
  !   write(IOUT) CPML_width_z
  !   write(IOUT) min_distance_between_CPML_parameter
  !   if (nspec_cpml > 0) then
  !     write(IOUT) CPML_regions
  !     write(IOUT) CPML_to_spec
  !     write(IOUT) is_CPML
  !     write(IOUT) d_store_x
  !     write(IOUT) d_store_y
  !     write(IOUT) d_store_z
  !     write(IOUT) k_store_x
  !     write(IOUT) k_store_y
  !     write(IOUT) k_store_z
  !     write(IOUT) alpha_store_x
  !     write(IOUT) alpha_store_y
  !     write(IOUT) alpha_store_z
  !     ! --------------------------------------------------------------------------------------------
  !     ! for adjoint tomography
  !     ! save the array stored the points on interface between PML and interior computational domain
  !     ! --------------------------------------------------------------------------------------------
  !     if ((SIMULATION_TYPE == 1 .and. SAVE_FORWARD) .or. SIMULATION_TYPE == 3) then
  !       write(IOUT) nglob_interface_PML_acoustic
  !       write(IOUT) nglob_interface_PML_elastic
  !       if (nglob_interface_PML_acoustic > 0) write(IOUT) points_interface_PML_acoustic
  !       if (nglob_interface_PML_elastic > 0)  write(IOUT) points_interface_PML_elastic
  !     endif
  !   endif
  ! endif

  ! absorbing boundary surface
  write(IOUT) num_abs_boundary_faces
  if (num_abs_boundary_faces > 0) then

    ! Saves the arrays with num faces in the first dimension
    call save_boundary_arrays(num_abs_boundary_faces, &
                         abs_boundary_ispec, &
                         abs_boundary_ijk, &
                         abs_boundary_jacobian2Dw, &
                         abs_boundary_normal)

    ! write(*,*) "values of abs_boundary_ijk"
    ! write(*,*) "--------------------------"
    ! call print_ijk(1, abs_boundary_ijk)
    ! call print_ijk(num_abs_boundary_faces, abs_boundary_ijk)

    if (STACEY_ABSORBING_CONDITIONS .and. (.not. PML_CONDITIONS)) then
      ! store mass matrix contributions
      if (ELASTIC_SIMULATION) then
        write(IOUT) rmassx
        write(IOUT) rmassy
        write(IOUT) rmassz
      endif
      if (ACOUSTIC_SIMULATION) then
        write(IOUT) rmassz_acoustic
      endif
    endif
  endif

  ! stamp for checking i/o so far
  itest = 9995
  write(IOUT) itest

  ! boundaries
  write(IOUT) nspec2D_xmin
  write(IOUT) nspec2D_xmax
  write(IOUT) nspec2D_ymin
  write(IOUT) nspec2D_ymax
  write(IOUT) NSPEC2D_BOTTOM
  write(IOUT) NSPEC2D_TOP

  if (nspec2D_xmin > 0) write(IOUT) ibelm_xmin
  if (nspec2D_xmax > 0) write(IOUT) ibelm_xmax
  if (nspec2D_ymin > 0) write(IOUT) ibelm_ymin
  if (nspec2D_ymax > 0) write(IOUT) ibelm_ymax
  if (nspec2D_bottom > 0) write(IOUT) ibelm_bottom
  if (nspec2D_top > 0) write(IOUT) ibelm_top

  ! free surface
  write(IOUT) num_free_surface_faces
  if (num_free_surface_faces > 0) then

    ! Saves the arrays with num faces in the first dimension
    call save_boundary_arrays(num_free_surface_faces, &
                         free_surface_ispec, &
                         free_surface_ijk, &
                         free_surface_jacobian2Dw, &
                         free_surface_normal)
  endif

  ! write(*,*) "values of free_surface_ijk"
  ! write(*,*) "--------------------------"
  ! call print_ijk(1, free_surface_ijk)
  ! call print_ijk(num_free_surface_faces, free_surface_ijk)

  ! acoustic-elastic coupling surface
  write(IOUT) num_coupling_ac_el_faces
  if (num_coupling_ac_el_faces > 0) then

    ! Saves the arrays with num faces in the first dimension
    call save_boundary_arrays(num_coupling_ac_el_faces, &
                         coupling_ac_el_ispec, &
                         coupling_ac_el_ijk, &
                         coupling_ac_el_jacobian2Dw, &
                         coupling_ac_el_normal)

  endif

  ! acoustic-poroelastic coupling surface
  write(IOUT) num_coupling_ac_po_faces
  if (num_coupling_ac_po_faces > 0) then

    ! Saves the arrays with num faces in the first dimension
    call save_boundary_arrays(num_coupling_ac_po_faces, &
                         coupling_ac_po_ispec, &
                         coupling_ac_po_ijk, &
                         coupling_ac_po_jacobian2Dw, &
                         coupling_ac_po_normal)
  endif

  ! elastic-poroelastic coupling surface
  write(IOUT) num_coupling_el_po_faces
  if (num_coupling_el_po_faces > 0) then

    ! Saves the arrays with num faces in the first dimension
    call save_boundary_arrays(num_coupling_el_po_faces, &
                         coupling_el_po_ispec, &
                         coupling_el_po_ijk, &
                         coupling_el_po_jacobian2Dw, &
                         coupling_el_po_normal)

    ! Saves the arrays with num faces in the first dimension
    call save_boundary_arrays(num_coupling_el_po_faces, &
                         coupling_po_el_ispec, &
                         coupling_po_el_ijk, &
                         coupling_el_po_jacobian2Dw, &
                         coupling_el_po_normal)

  endif

  ! stamp for checking i/o
  itest = 9997
  write(IOUT) itest

  ! MPI interfaces
  if (num_interfaces_ext_mesh > 0) then
    max_nibool_interfaces_ext_mesh = maxval(nibool_interfaces_ext_mesh(:))
  else
    max_nibool_interfaces_ext_mesh = 0
  endif
  write(IOUT) num_interfaces_ext_mesh
  write(IOUT) max_nibool_interfaces_ext_mesh

  if (num_interfaces_ext_mesh > 0) then
    allocate(ibool_interfaces_ext_mesh_dummy(max_nibool_interfaces_ext_mesh,num_interfaces_ext_mesh),stat=ier)
    if (ier /= 0) call exit_MPI_without_rank('error allocating array 650')
    if (ier /= 0) stop 'error allocating array'
    ibool_interfaces_ext_mesh_dummy(:,:) = 0
    do i = 1, num_interfaces_ext_mesh
       ibool_interfaces_ext_mesh_dummy(:,i) = ibool_interfaces_ext_mesh(1:max_nibool_interfaces_ext_mesh,i)
    enddo
    write(IOUT) my_neighbors_ext_mesh
    write(IOUT) nibool_interfaces_ext_mesh
    write(IOUT) ibool_interfaces_ext_mesh_dummy
  endif

  ! stamp for checking i/o
  itest = 9996
  write(IOUT) itest

  ! material properties
  ! anisotropy
  if (ELASTIC_SIMULATION .and. ANISOTROPY) then

    call save_global_arrays(nspec, c11store, 'c11')
    call save_global_arrays(nspec, c12store, 'c12')
    call save_global_arrays(nspec, c13store, 'c13')
    call save_global_arrays(nspec, c14store, 'c14')
    call save_global_arrays(nspec, c15store, 'c15')
    call save_global_arrays(nspec, c16store, 'c16')
    call save_global_arrays(nspec, c22store, 'c22')
    call save_global_arrays(nspec, c23store, 'c23')
    call save_global_arrays(nspec, c24store, 'c24')
    call save_global_arrays(nspec, c25store, 'c25')
    call save_global_arrays(nspec, c26store, 'c26')
    call save_global_arrays(nspec, c33store, 'c33')
    call save_global_arrays(nspec, c34store, 'c34')
    call save_global_arrays(nspec, c35store, 'c35')
    call save_global_arrays(nspec, c36store, 'c36')
    call save_global_arrays(nspec, c44store, 'c44')
    call save_global_arrays(nspec, c45store, 'c45')
    call save_global_arrays(nspec, c46store, 'c46')
    call save_global_arrays(nspec, c55store, 'c55')
    call save_global_arrays(nspec, c56store, 'c56')
    call save_global_arrays(nspec, c66store, 'c66')

    ! write(IOUT) c11store
    ! write(IOUT) c12store
    ! write(IOUT) c13store
    ! write(IOUT) c14store
    ! write(IOUT) c15store
    ! write(IOUT) c16store
    ! write(IOUT) c22store
    ! write(IOUT) c23store
    ! write(IOUT) c24store
    ! write(IOUT) c25store
    ! write(IOUT) c26store
    ! write(IOUT) c33store
    ! write(IOUT) c34store
    ! write(IOUT) c35store
    ! write(IOUT) c36store
    ! write(IOUT) c44store
    ! write(IOUT) c45store
    ! write(IOUT) c46store
    ! write(IOUT) c55store
    ! write(IOUT) c56store
    ! write(IOUT) c66store
  endif

  ! Write test value 9995
  itest = 9995
  write(IOUT) itest

  ! inner/outer elements
  write(IOUT) ispec_is_inner

  if (ACOUSTIC_SIMULATION) then
    write(IOUT) nspec_inner_acoustic,nspec_outer_acoustic
    write(IOUT) num_phase_ispec_acoustic
    if (num_phase_ispec_acoustic > 0) then
      call save_inner_outer_arrays(num_phase_ispec_acoustic, phase_ispec_inner_acoustic)
    endif
  endif

  ! Write test value 9994
  itest = 9994
  write(IOUT) itest

  if (ELASTIC_SIMULATION) then
    write(IOUT) nspec_inner_elastic, nspec_outer_elastic
    write(IOUT) num_phase_ispec_elastic

    if (num_phase_ispec_elastic > 0) then
      call save_inner_outer_arrays(num_phase_ispec_elastic, phase_ispec_inner_elastic)
    endif
  endif

  ! Write test value 9993
  itest = 9993
  write(IOUT) itest

  if (POROELASTIC_SIMULATION) then
    write(IOUT) nspec_inner_poroelastic,nspec_outer_poroelastic
    write(IOUT) num_phase_ispec_poroelastic
    if (num_phase_ispec_poroelastic > 0) then
      call save_inner_outer_arrays(num_phase_ispec_poroelastic, phase_ispec_inner_poroelastic)
    end if
  endif

  ! Write test value 9992
  itest = 9992
  write(IOUT) itest

  ! mesh coloring
  if (USE_MESH_COLORING_GPU) then
    if (ACOUSTIC_SIMULATION) then
      write(IOUT) num_colors_outer_acoustic,num_colors_inner_acoustic
      write(IOUT) num_elem_colors_acoustic
    endif
    if (ELASTIC_SIMULATION) then
      write(IOUT) num_colors_outer_elastic,num_colors_inner_elastic
      write(IOUT) num_elem_colors_elastic
    endif
  endif

  ! Write test value 9991
  itest = 9991
  write(IOUT) itest


  ! surface points
  write(IOUT) nfaces_surface
  write(IOUT) ispec_is_surface_external_mesh
  write(IOUT) iglob_is_surface_external_mesh

  ! Write test value 9990
  itest = 9990
  write(IOUT) itest

  ! mesh adjacency
  write(IOUT) num_neighbors_all
  write(IOUT) neighbors_xadj
  write(IOUT) neighbors_adjncy

  ! stamp for checking i/o
  itest = 9989
  write(IOUT) itest

  close(IOUT)

  ! stores arrays in binary files
  if (SAVE_MESH_FILES) then
    call save_arrays_solver_files()
  endif

  ! if SAVE_MESH_FILES is true then the files have already been saved, no need to save them again
  if (COUPLE_WITH_INJECTION_TECHNIQUE .or. MESH_A_CHUNK_OF_THE_EARTH) then
    call save_arrays_solver_injection_boundary()
  endif

  !! setup wavefield discontinuity interface
  if (IS_WAVEFIELD_DISCONTINUITY) then
    call save_arrays_wavefield_discontinuity()
  endif

  ! synchronizes processes
  call synchronize_all()

  ! cleanup
  if (allocated(ibool_interfaces_ext_mesh_dummy)) then
    deallocate(ibool_interfaces_ext_mesh_dummy,stat=ier)
    if (ier /= 0) stop 'error deallocating array ibool_interfaces_ext_mesh_dummy'
  endif

  end subroutine save_arrays_solver_mesh

!
!-------------------------------------------------------------------------------------------------
!

  subroutine save_arrays_solver_files()

! outputs binary files for single mesh parameters (for example vp, vs, rho, ..)

  use constants, only: IDOMAIN_ACOUSTIC,IDOMAIN_ELASTIC,IDOMAIN_POROELASTIC, &
    NGLLX,NGLLY,NGLLZ,NGLLSQUARE,IMAIN,IOUT,FOUR_THIRDS,CUSTOM_REAL, &
    myrank

  use shared_parameters, only: ACOUSTIC_SIMULATION, ELASTIC_SIMULATION, POROELASTIC_SIMULATION, &
    NPROC

  ! global indices
  use generate_databases_par, only: nspec => NSPEC_AB, ibool

  ! MPI interfaces
  use generate_databases_par, only: nibool_interfaces_ext_mesh,ibool_interfaces_ext_mesh,num_interfaces_ext_mesh

  use create_regions_mesh_ext_par

  implicit none

  ! local parameters
  real(kind=CUSTOM_REAL), dimension(:,:,:,:), allocatable :: v_tmp
  integer,dimension(:),allocatable :: v_tmp_i
  integer :: ier,i,j
  integer, dimension(:), allocatable :: iglob_tmp
  integer :: inum, num_points
  character(len=MAX_STRING_LEN) :: filename

  !----------------------------------------------------------------------
  ! outputs mesh files in vtk-format for visualization
  ! (mostly for free-surface and acoustic/elastic coupling surfaces)
  logical,parameter :: SAVE_MESH_FILES_ADDITIONAL = .true.

  !----------------------------------------------------------------------

  if (myrank == 0) then
    write(IMAIN,*) '     saving mesh files for AVS, OpenDX, Paraview'
    call flush_IMAIN()
  endif

  ! mesh arrays used for example in combine_vol_data.f90
  !--- x coordinate
  open(unit=IOUT,file=prname(1:len_trim(prname))//'x.bin',status='unknown',form='unformatted',iostat=ier)
  if (ier /= 0) stop 'error opening file x.bin'
  write(IOUT) xstore_unique
  close(IOUT)

  !--- y coordinate
  open(unit=IOUT,file=prname(1:len_trim(prname))//'y.bin',status='unknown',form='unformatted',iostat=ier)
  if (ier /= 0) stop 'error opening file y.bin'
  write(IOUT) ystore_unique
  close(IOUT)

  !--- z coordinate
  open(unit=IOUT,file=prname(1:len_trim(prname))//'z.bin',status='unknown',form='unformatted',iostat=ier)
  if (ier /= 0) stop 'error opening file z.bin'
  write(IOUT) zstore_unique
  close(IOUT)

  ! ibool
  open(unit=IOUT,file=prname(1:len_trim(prname))//'ibool.bin',status='unknown',form='unformatted',iostat=ier)
  if (ier /= 0) stop 'error opening file ibool.bin'
  write(IOUT) ibool
  close(IOUT)

  allocate(v_tmp(NGLLX,NGLLY,NGLLZ,nspec), stat=ier)
  if (ier /= 0) call exit_MPI_without_rank('error allocating array 651')
  if (ier /= 0) call exit_MPI_without_rank('error allocating array')

  ! vp (for checking the mesh and model)
  !minimum = minval( abs(rho_vp) )
  !if (minimum(1) /= 0.0) then
  !  v_tmp = (FOUR_THIRDS * mustore + kappastore) / rho_vp
  !else
  !  v_tmp = 0.0
  !endif
  v_tmp = 0.0
  where( rho_vp /= 0._CUSTOM_REAL ) v_tmp = (FOUR_THIRDS * mustore + kappastore) / rho_vp
  open(unit=IOUT,file=prname(1:len_trim(prname))//'vp.bin',status='unknown',form='unformatted',iostat=ier)
  if (ier /= 0) stop 'error opening file vp.bin'
  write(IOUT) v_tmp
  close(IOUT)

  ! vp values - VTK file output
  filename = prname(1:len_trim(prname))//'vp'
  call write_VTK_data_gll_cr(nspec,nglob_unique, &
                             xstore_unique,ystore_unique,zstore_unique,ibool, &
                             v_tmp,filename)


  ! vs (for checking the mesh and model)
  !minimum = minval( abs(rho_vs) )
  !if (minimum(1) /= 0.0) then
  !  v_tmp = mustore / rho_vs
  !else
  !  v_tmp = 0.0
  !endif
  v_tmp = 0.0
  where( rho_vs /= 0._CUSTOM_REAL )  v_tmp = mustore / rho_vs
  open(unit=IOUT,file=prname(1:len_trim(prname))//'vs.bin',status='unknown',form='unformatted',iostat=ier)
  if (ier /= 0) stop 'error opening file vs.bin'
  write(IOUT) v_tmp
  close(IOUT)

  ! vs values - VTK file output
  filename = prname(1:len_trim(prname))//'vs'
  call write_VTK_data_gll_cr(nspec,nglob_unique, &
                             xstore_unique,ystore_unique,zstore_unique,ibool, &
                             v_tmp,filename)

  ! outputs density model for check
  v_tmp = 0.0
  where( rho_vp /= 0._CUSTOM_REAL ) v_tmp = rho_vp**2 / (FOUR_THIRDS * mustore + kappastore)
  open(unit=IOUT,file=prname(1:len_trim(prname))//'rho.bin',status='unknown',form='unformatted',iostat=ier)
  if (ier /= 0) stop 'error opening file rho.bin'
  write(IOUT) v_tmp
  close(IOUT)

  ! attenuation
  ! shear attenuation Qmu
  open(unit=IOUT,file=prname(1:len_trim(prname))//'qmu.bin',status='unknown',form='unformatted',iostat=ier)
  if (ier /= 0) stop 'error opening file qmu.bin'
  write(IOUT) qmu_attenuation_store
  close(IOUT)

  ! shear attenuation - VTK file output
  filename = prname(1:len_trim(prname))//'qmu'
  call write_VTK_data_gll_cr(nspec,nglob_unique, &
                             xstore_unique,ystore_unique,zstore_unique,ibool, &
                             qmu_attenuation_store,filename)

  ! bulk attenuation Qkappa
  open(unit=IOUT,file=prname(1:len_trim(prname))//'qkappa.bin',status='unknown',form='unformatted',iostat=ier)
  if (ier /= 0) stop 'error opening file qkappa.bin'
  write(IOUT) qkappa_attenuation_store
  close(IOUT)

  ! bulk attenuation - VTK file output
  filename = prname(1:len_trim(prname))//'qkappa'
  call write_VTK_data_gll_cr(nspec,nglob_unique, &
                             xstore_unique,ystore_unique,zstore_unique,ibool, &
                             qkappa_attenuation_store,filename)

  ! frees temporary array
  deallocate(v_tmp)

  ! additional VTK file output
  if (SAVE_MESH_FILES_ADDITIONAL) then
    ! user output
    call synchronize_all()
    if (myrank == 0) then
      write(IMAIN,*) '     saving additional mesh files with surface/coupling points'
      call flush_IMAIN()
    endif

    ! saves free surface points
    if (num_free_surface_faces > 0) then
      ! saves free surface interface points
      allocate(iglob_tmp(NGLLSQUARE*num_free_surface_faces),stat=ier)
      if (ier /= 0) call exit_MPI_without_rank('error allocating array 652')
      if (ier /= 0) stop 'error allocating array iglob_tmp'
      inum = 0
      iglob_tmp(:) = 0
      do i = 1,num_free_surface_faces
        do j = 1,NGLLSQUARE
          inum = inum+1
          iglob_tmp(inum) = ibool(free_surface_ijk(1,j,i), &
                                  free_surface_ijk(2,j,i), &
                                  free_surface_ijk(3,j,i), &
                                  free_surface_ispec(i) )
        enddo
      enddo
      filename = prname(1:len_trim(prname))//'free_surface'
      call write_VTK_data_points(nglob_unique, &
                                 xstore_unique,ystore_unique,zstore_unique, &
                                 iglob_tmp,NGLLSQUARE*num_free_surface_faces, &
                                 filename)

      deallocate(iglob_tmp)
    endif

    ! acoustic-elastic domains
    if (ACOUSTIC_SIMULATION .and. ELASTIC_SIMULATION) then
      ! saves points on acoustic-elastic coupling interface
      num_points = NGLLSQUARE*num_coupling_ac_el_faces
      allocate(iglob_tmp(num_points),stat=ier)
      if (ier /= 0) call exit_MPI_without_rank('error allocating array 653')
      if (ier /= 0) stop 'error allocating array iglob_tmp'
      inum = 0
      iglob_tmp(:) = 0
      do i = 1,num_coupling_ac_el_faces
        do j = 1,NGLLSQUARE
          inum = inum+1
          iglob_tmp(inum) = ibool(coupling_ac_el_ijk(1,j,i), &
                                  coupling_ac_el_ijk(2,j,i), &
                                  coupling_ac_el_ijk(3,j,i), &
                                  coupling_ac_el_ispec(i) )
        enddo
      enddo
      filename = prname(1:len_trim(prname))//'coupling_acoustic_elastic'
      call write_VTK_data_points(nglob_unique,xstore_unique,ystore_unique,zstore_unique, &
                                 iglob_tmp,num_points,filename)
      deallocate(iglob_tmp)
    endif !if (ACOUSTIC_SIMULATION .and. ELASTIC_SIMULATION )

    ! acoustic-poroelastic domains
    if (ACOUSTIC_SIMULATION .and. POROELASTIC_SIMULATION) then
      ! saves points on acoustic-poroelastic coupling interface
      num_points = NGLLSQUARE*num_coupling_ac_po_faces
      allocate( iglob_tmp(num_points),stat=ier)
      if (ier /= 0) call exit_MPI_without_rank('error allocating array 655')
      if (ier /= 0) stop 'error allocating array iglob_tmp'
      inum = 0
      iglob_tmp(:) = 0
      do i = 1,num_coupling_ac_po_faces
        do j = 1,NGLLSQUARE
          inum = inum+1
          iglob_tmp(inum) = ibool(coupling_ac_po_ijk(1,j,i), &
                                  coupling_ac_po_ijk(2,j,i), &
                                  coupling_ac_po_ijk(3,j,i), &
                                  coupling_ac_po_ispec(i) )
        enddo
      enddo
      filename = prname(1:len_trim(prname))//'coupling_acoustic_poroelastic'
      call write_VTK_data_points(nglob_unique,xstore_unique,ystore_unique,zstore_unique, &
                                 iglob_tmp,num_points,filename)
      deallocate(iglob_tmp)
    endif !if (ACOUSTIC_SIMULATION .and. POROELASTIC_SIMULATION )

    ! elastic-poroelastic domains
    if (ELASTIC_SIMULATION .and. POROELASTIC_SIMULATION) then
      ! saves points on elastic-poroelastic coupling interface
      num_points = NGLLSQUARE*num_coupling_el_po_faces
      allocate( iglob_tmp(num_points),stat=ier)
      if (ier /= 0) call exit_MPI_without_rank('error allocating array 657')
      if (ier /= 0) stop 'error allocating array iglob_tmp'
      inum = 0
      iglob_tmp(:) = 0
      do i = 1,num_coupling_el_po_faces
        do j = 1,NGLLSQUARE
          inum = inum+1
          iglob_tmp(inum) = ibool(coupling_el_po_ijk(1,j,i), &
                                  coupling_el_po_ijk(2,j,i), &
                                  coupling_el_po_ijk(3,j,i), &
                                  coupling_el_po_ispec(i) )
        enddo
      enddo
      filename = prname(1:len_trim(prname))//'coupling_elastic_poroelastic'
      call write_VTK_data_points(nglob_unique,xstore_unique,ystore_unique,zstore_unique, &
                                 iglob_tmp,num_points,filename)
      deallocate(iglob_tmp)
    endif !if (ACOUSTIC_SIMULATION .and. POROELASTIC_SIMULATION

    ! mixed simulations
    if ((ACOUSTIC_SIMULATION .and. ELASTIC_SIMULATION) .or. &
        (ACOUSTIC_SIMULATION .and. POROELASTIC_SIMULATION) .or. &
        (ELASTIC_SIMULATION .and. POROELASTIC_SIMULATION)) then
      ! saves acoustic/elastic/poroelastic flag
      allocate(v_tmp_i(nspec),stat=ier)
      if (ier /= 0) call exit_MPI_without_rank('error allocating array 656')
      if (ier /= 0) stop 'error allocating array v_tmp_i'
      do i = 1,nspec
        if (ispec_is_acoustic(i)) then
          v_tmp_i(i) = IDOMAIN_ACOUSTIC
        else if (ispec_is_elastic(i)) then
          v_tmp_i(i) = IDOMAIN_ELASTIC
        else if (ispec_is_poroelastic(i)) then
          v_tmp_i(i) = IDOMAIN_POROELASTIC
        else
          v_tmp_i(i) = 0
        endif
      enddo
      filename = prname(1:len_trim(prname))//'acoustic_elastic_poroelastic_flag'
      call write_VTK_data_elem_i(nspec,nglob_unique,xstore_unique,ystore_unique,zstore_unique,ibool, &
                                 v_tmp_i,filename)
      deallocate(v_tmp_i)
    endif

    ! MPI
    if (NPROC > 1) then
      ! saves MPI interface points
      num_points = sum(nibool_interfaces_ext_mesh(1:num_interfaces_ext_mesh))
      allocate( iglob_tmp(num_points),stat=ier)
      if (ier /= 0) call exit_MPI_without_rank('error allocating array 659')
      if (ier /= 0) stop 'error allocating array iglob_tmp'
      inum = 0
      iglob_tmp(:) = 0
      do i = 1,num_interfaces_ext_mesh
        do j = 1, nibool_interfaces_ext_mesh(i)
          inum = inum + 1
          iglob_tmp(inum) = ibool_interfaces_ext_mesh(j,i)
        enddo
      enddo

      filename = prname(1:len_trim(prname))//'MPI_points'
      call write_VTK_data_points(nglob_unique,xstore_unique,ystore_unique,zstore_unique, &
                                 iglob_tmp,num_points,filename)
      deallocate(iglob_tmp)
    endif ! NPROC > 1
  endif  !if (SAVE_MESH_FILES_ADDITIONAL)

  !debug
  ! saves ispec number
  !allocate(v_tmp_i(nspec),stat=ier)
  !if (ier /= 0) call exit_MPI_without_rank('error allocating array 656')
  !if (ier /= 0) stop 'error allocating array v_tmp_i'
  !do i = 1,nspec
  !  v_tmp_i(i) = i
  !enddo
  !filename = prname(1:len_trim(prname))//'ispec_number'
  !call write_VTK_data_elem_i(nspec,nglob_unique,xstore_unique,ystore_unique,zstore_unique,ibool, &
  !                           v_tmp_i,filename)
  !deallocate(v_tmp_i)

  end subroutine save_arrays_solver_files

!
!-------------------------------------------------------------------------------------------------
!

  subroutine save_arrays_solver_injection_boundary()

  use constants, only: myrank,NGLLSQUARE,IMAIN,IOUT

  use shared_parameters, only: COUPLE_WITH_INJECTION_TECHNIQUE
  use generate_databases_par, only: MESH_A_CHUNK_OF_THE_EARTH

  ! global indices
  use generate_databases_par, only: ibool

  use create_regions_mesh_ext_par

  implicit none

  ! local parameters
  integer :: ier,i,j,k
  integer :: iface, ispec, iglob, igll
  real(kind=CUSTOM_REAL) :: nx,ny,nz
  character(len=MAX_STRING_LEN) :: filename

  ! checks if anything to do
  if (.not. (COUPLE_WITH_INJECTION_TECHNIQUE .or. MESH_A_CHUNK_OF_THE_EARTH)) return

  if (myrank == 0) then
    write(IMAIN,*) '     saving mesh files for coupled injection boundary'
    call flush_IMAIN()
  endif

  filename = prname(1:len_trim(prname))//'absorb_dsm'
  open(IOUT,file=filename(1:len_trim(filename)),status='unknown',form='unformatted',iostat=ier)
  if (ier /= 0) stop 'error opening file absorb_dsm'
  write(IOUT) num_abs_boundary_faces
  write(IOUT) abs_boundary_ispec
  write(IOUT) abs_boundary_ijk
  write(IOUT) abs_boundary_jacobian2Dw
  write(IOUT) abs_boundary_normal
  close(IOUT)

  filename = prname(1:len_trim(prname))//'inner'
  open(IOUT,file=filename(1:len_trim(filename)),status='unknown',form='unformatted',iostat=ier)
  write(IOUT) ispec_is_inner
  write(IOUT) ispec_is_elastic
  close(IOUT)

  !! VM VM write an ascii file for instaseis input
  filename = prname(1:len_trim(prname))//'normal.txt'
  open(IOUT,file=filename(1:len_trim(filename)),status='unknown',iostat=ier)
  write(IOUT, *) ' number of points :', num_abs_boundary_faces*NGLLSQUARE

  do iface = 1,num_abs_boundary_faces
     ispec = abs_boundary_ispec(iface)
     if (ispec_is_elastic(ispec)) then
        do igll = 1,NGLLSQUARE

           ! gets local indices for GLL point
           i = abs_boundary_ijk(1,igll,iface)
           j = abs_boundary_ijk(2,igll,iface)
           k = abs_boundary_ijk(3,igll,iface)

           iglob = ibool(i,j,k,ispec)

           nx = abs_boundary_normal(1,igll,iface)
           ny = abs_boundary_normal(2,igll,iface)
           nz = abs_boundary_normal(3,igll,iface)

           write(IOUT,'(6f25.10)') xstore_unique(iglob), ystore_unique(iglob), zstore_unique(iglob), nx, ny, nz

        enddo
     endif
  enddo
  close(IOUT)

  end subroutine save_arrays_solver_injection_boundary
