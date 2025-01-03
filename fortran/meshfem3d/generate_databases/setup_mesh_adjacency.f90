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


  subroutine setup_mesh_adjacency()

! setups mesh adjacency array to search element neighbors for point searches

  use constants, only: myrank, &
    NGLLX,NGLLY,NGLLZ,MIDX,MIDY,MIDZ,IMAIN,MAX_STRING_LEN

  use generate_databases_par, only: NSPEC_AB,NGLOB_AB,ibool,NPROC,prname

  ! mesh adjacency for point searches
  use generate_databases_par, only: neighbors_xadj,neighbors_adjncy,num_neighbors_all

  use fault_generate_databases, only: ANY_FAULT_IN_THIS_PROC

  ! debugging
  use create_regions_mesh_ext_par, only: xstore => xstore_unique, ystore => ystore_unique, zstore => zstore_unique

  implicit none

  ! local parameters
  ! maximum number of neighbors
  integer,parameter :: MAX_NEIGHBORS_DIRECT = 80   ! maximum number of neighbors (for direct element neighbors only)
  integer,parameter :: MAX_NEIGHBORS = 300         ! maximum number of neighbors (with neighbor of neighbors)

  ! temporary
  integer,dimension(:),allocatable :: tmp_adjncy  ! temporary adjacency
  integer :: inum_neighbor

  ! timer MPI
  double precision :: time1,tCPU
  double precision, external :: wtime

  integer :: num_neighbors,num_neighbors_max
  integer :: num_elements_max
  integer :: ispec_ref,ispec,iglob,ier,icorner
  logical :: is_neighbor

  ! for all the elements in contact with the reference element
  integer, dimension(:,:), allocatable :: ibool_corner

  ! neighbors of neighbors
  integer :: ielem,ii,jj,ispec_neighbor
  integer :: num_neighbor_neighbors,num_neighbor_neighbors_max

  ! note: we add direct neighbors plus neighbors of neighbors.
  !       for very coarse meshes, the initial location guesses especially around doubling layers can be poor such that we need
  !       to enlarge the search of neighboring elements.
  logical, parameter :: ADD_NEIGHBOR_OF_NEIGHBORS = .true.

  ! debugging
  character(len=MAX_STRING_LEN) :: filename
  integer,dimension(:),allocatable :: tmp_num_neighbors
  logical,parameter :: DEBUG_VTK_OUTPUT = .false.

  ! Node-to-element reverse lookup
  integer, dimension(:,:), allocatable :: node_to_elem
  integer, dimension(:), allocatable :: node_to_elem_count
  integer :: icount
  integer, dimension(MAX_NEIGHBORS_DIRECT) :: elem_neighbors            ! direct neighbors
  integer, dimension(MAX_NEIGHBORS) :: elem_neighbors_of_neighbors      ! with neighboring element neighbors

  ! user output
  if (myrank == 0) then
    write(IMAIN,*)
    write(IMAIN,*) '     mesh adjacency:'
    write(IMAIN,*) '     total number of elements in this slice          = ',NSPEC_AB
    write(IMAIN,*)
    write(IMAIN,*) '     maximum number of neighbors allowed             = ',MAX_NEIGHBORS
    write(IMAIN,*) '     minimum array memory required per slice         = ', &
                                ((MAX_NEIGHBORS + 1) * NSPEC_AB * 4)/1024./1024.,"(MB)"
    call flush_IMAIN()
  endif

  ! get MPI starting time
  time1 = wtime()

  ! adjacency arrays
  !
  ! how to use:
  !  num_neighbors = neighbors_xadj(ispec+1)-neighbors_xadj(ispec)
  !  do i = 1,num_neighbors
  !    ! get neighbor
  !    ispec_neighbor = neighbors_adjncy(neighbors_xadj(ispec) + i)
  !    ..
  !  enddo
  allocate(neighbors_xadj(NSPEC_AB + 1),stat=ier)
  if (ier /= 0) stop 'Error allocating xadj'
  neighbors_xadj(:) = 0

  ! temporary helper array
  allocate(tmp_adjncy(MAX_NEIGHBORS * NSPEC_AB),stat=ier)
  if (ier /= 0) stop 'Error allocating tmp_adjncy'
  tmp_adjncy(:) = 0

  ! since we only need to check corner points for the adjacency,
  ! we build an extra ibool array with corner points only for faster accessing
  allocate(ibool_corner(8,NSPEC_AB),stat=ier)
  if (ier /= 0) stop 'Error allocating ibool_corner array'
  ibool_corner(:,:) = 0
  do ispec = 1,NSPEC_AB
    ibool_corner(1,ispec) = ibool(1,1,1,ispec)
    ibool_corner(2,ispec) = ibool(NGLLX,1,1,ispec)
    ibool_corner(3,ispec) = ibool(NGLLX,NGLLY,1,ispec)
    ibool_corner(4,ispec) = ibool(1,NGLLY,1,ispec)

    ibool_corner(5,ispec) = ibool(1,1,NGLLZ,ispec)
    ibool_corner(6,ispec) = ibool(NGLLX,1,NGLLZ,ispec)
    ibool_corner(7,ispec) = ibool(NGLLX,NGLLY,NGLLZ,ispec)
    ibool_corner(8,ispec) = ibool(1,NGLLY,NGLLZ,ispec)
  enddo

  ! setups node-to-element mapping counter
  allocate(node_to_elem_count(NGLOB_AB),stat=ier)
  if (ier /= 0) stop 'Error allocating node_to_elem_count arrays'
  node_to_elem_count(:) = 0
  num_elements_max = 0
  ! determines maximum count per node
  do ispec = 1,NSPEC_AB
    ! only corner nodes used to check for shared nodes (conforming mesh)
    do icorner = 1,8
      iglob = ibool_corner(icorner,ispec)

      ! add to mapping
      icount = node_to_elem_count(iglob) + 1

      ! gets maximum of elements sharing node
      if (icount > num_elements_max) num_elements_max = icount

      ! updates count
      node_to_elem_count(iglob) = icount
    enddo
  enddo

  ! user output
  if (myrank == 0) then
    write(IMAIN,*)
    write(IMAIN,*) '     maximum number of elements per shared node      = ',num_elements_max
    write(IMAIN,*) '     node-to-element array memory required per slice = ', &
                                (num_elements_max * NGLOB_AB * 4)/1024./1024.,"(MB)"
    write(IMAIN,*)
    call flush_IMAIN()
  endif

  ! Node-to-element reverse lookup
  allocate(node_to_elem(num_elements_max,NGLOB_AB),stat=ier)
  if (ier /= 0) stop 'Error allocating node_to_elem arrays'
  node_to_elem(:,:) = -1
  node_to_elem_count(:) = 0

  do ispec = 1,NSPEC_AB
    ! only corner nodes used to check for shared nodes (conforming mesh)
    do icorner = 1,8
      iglob = ibool_corner(icorner,ispec)

      ! add to mapping
      icount = node_to_elem_count(iglob) + 1

      ! adds entry to mapping array
      node_to_elem_count(iglob) = icount
      node_to_elem(icount,iglob) = ispec
    enddo
  enddo

  ! gets maximum number of neighbors
  inum_neighbor = 0       ! counts total number of neighbors added

  ! stats
  num_neighbors_max = 0
  num_neighbor_neighbors_max = 0

  do ispec_ref = 1,NSPEC_AB
    ! counts number of neighbors (for this element)
    num_neighbors = 0
    elem_neighbors(:) = 0

    ! only corner nodes used to check for shared nodes (conforming mesh)
    do icorner = 1,8
      iglob = ibool_corner(icorner,ispec_ref)

      ! loops over all other elements to add neighbors
      do ielem = 1,node_to_elem_count(iglob)
        ispec = node_to_elem(ielem,iglob)

        ! skip reference element
        if (ispec == ispec_ref) cycle

        ! checks if it is a new neighbor element
        is_neighbor = .false.

        ! checks if first element
        if (num_neighbors == 0) then
          is_neighbor = .true.
        else
          ! check if not added to list yet
          if (.not. any(elem_neighbors(1:num_neighbors) == ispec)) then
            is_neighbor = .true.
          endif
        endif

        ! adds as neighbor
        if (is_neighbor) then
          ! store neighbor elements
          num_neighbors = num_neighbors + 1

          ! check
          if (num_neighbors > MAX_NEIGHBORS_DIRECT) stop 'Error maximum of neighbors (direct) exceeded'

          ! store element
          elem_neighbors(num_neighbors) = ispec
        endif
      enddo
    enddo ! corners

    ! statistics
    if (num_neighbors > num_neighbors_max) num_neighbors_max = num_neighbors

    ! store to temporary adjacency array
    if (num_neighbors > 0) then
      ! updates total count
      inum_neighbor = inum_neighbor + num_neighbors

      ! checks
      if (inum_neighbor > MAX_NEIGHBORS * NSPEC_AB) stop 'Error maximum of total neighbors exceeded'

      ! adds to adjacency
      tmp_adjncy(inum_neighbor-num_neighbors+1:inum_neighbor) = elem_neighbors(1:num_neighbors)
    else
      ! no neighbors
      ! warning
      print *,'*** Warning: found mesh element with no neighbors : slice ',myrank, &
              ' - element ',ispec_ref,'out of',NSPEC_AB,' ***'
    endif

    ! again loop to get neighbors of neighbors
    if (ADD_NEIGHBOR_OF_NEIGHBORS) then
      ! counter for statistics
      num_neighbor_neighbors = 0
      elem_neighbors_of_neighbors(:) = 0

      ! loops over neighbor elements
      do ii = 1,num_neighbors
        ! get neighbor
        ispec_neighbor = elem_neighbors(ii)

        ! only corner nodes used to check for shared nodes (conforming mesh)
        do icorner = 1,8
          iglob = ibool_corner(icorner,ispec_neighbor)

          ! loops over all other elements to add neighbors
          do ielem = 1,node_to_elem_count(iglob)
            ispec = node_to_elem(ielem,iglob)

            ! skip reference element
            if (ispec == ispec_ref) cycle

            ! skip neighbor reference element (already added by elem_neighbors() array)
            if (ispec == ispec_neighbor) cycle

            ! checks if element has a corner iglob from reference element
            is_neighbor = .false.

            ! check if not added to neighbor list yet
            if (.not. any(elem_neighbors(1:num_neighbors) == ispec)) then
              ! check if added to neighbor of neighbors list
              if (num_neighbor_neighbors == 0) then
                is_neighbor = .true.
              else
                if (.not. any(elem_neighbors_of_neighbors(1:num_neighbor_neighbors) == ispec)) then
                  is_neighbor = .true.
                endif
              endif
            endif

            ! adds neighbors to reference element
            if (is_neighbor) then
              ! adds to adjacency
              num_neighbor_neighbors = num_neighbor_neighbors + 1

              ! check
              if (num_neighbor_neighbors > MAX_NEIGHBORS) stop 'Error maximum of neighbors (with neighbors) exceeded'

              ! store element
              elem_neighbors_of_neighbors(num_neighbor_neighbors) = ispec
            endif
          enddo
        enddo ! corners
      enddo ! num neighbors

      ! adds to temporary adjacency array
      if (num_neighbor_neighbors > 0) then
        ! updates total count
        inum_neighbor = inum_neighbor + num_neighbor_neighbors

        ! checks
        if (inum_neighbor > MAX_NEIGHBORS * NSPEC_AB) stop 'Error maximum of total neighbors (with neighbors) exceeded'

        ! adds elements
        tmp_adjncy(inum_neighbor-num_neighbor_neighbors+1:inum_neighbor) = elem_neighbors_of_neighbors(1:num_neighbor_neighbors)
      endif

      ! statistics
      if (num_neighbor_neighbors > num_neighbor_neighbors_max) num_neighbor_neighbors_max = num_neighbor_neighbors

    endif  ! ADD_NEIGHBOR_OF_NEIGHBORS

    ! adjacency indexing
    neighbors_xadj(ispec_ref + 1) = inum_neighbor
    ! how to use:
    !num_neighbors = neighbors_xadj(ispec+1)-neighbors_xadj(ispec)
    !do i = 1,num_neighbors
    !  ! get neighbor
    !  ispec_neighbor = neighbors_adjncy(neighbors_xadj(ispec) + i)
    !enddo

    ! user output progress
    if (myrank == 0) then
      if (mod(ispec_ref,max(NSPEC_AB/10,1)) == 0) then
        tCPU = wtime() - time1
        ! elapsed
        write(IMAIN,*) "    ",int(ispec_ref/(max(NSPEC_AB/10,1)) * 10)," %", &
                       " - elapsed time:",sngl(tCPU),"s"
        ! flushes file buffer for main output file (IMAIN)
        call flush_IMAIN()
      endif
    endif
  enddo ! ispec_ref

  ! frees temporary array
  deallocate(ibool_corner)
  deallocate(node_to_elem,node_to_elem_count)

  ! debug: for vtk output
  if (DEBUG_VTK_OUTPUT) then
    ! number of neighbors
    allocate(tmp_num_neighbors(NSPEC_AB),stat=ier)
    if (ier /= 0) stop 'Error allocating tmp_num_neighbors array'
    ! fills temporary array
    do ispec_ref = 1,NSPEC_AB
      ! gets number of neighbors
      num_neighbors = neighbors_xadj(ispec_ref+1) - neighbors_xadj(ispec_ref)
      tmp_num_neighbors(ispec_ref) = num_neighbors
    enddo
    filename = trim(prname) // 'mesh_neighbors'
    call write_VTK_data_elem_i(NSPEC_AB,NGLOB_AB,xstore,ystore,zstore,ibool,tmp_num_neighbors,filename)
    if (myrank == 0) then
      write(IMAIN,*) '     written file: ',trim(filename)//'.vtk'
      call flush_IMAIN()
    endif
    deallocate(tmp_num_neighbors)
  endif

  ! check if element has neighbors
  ! note: in case of a fault in this slice (splitting nodes) and/or scotch partitioning
  !       it can happen that an element has no neighbors
  if (NPROC == 1 .and. (.not. ANY_FAULT_IN_THIS_PROC)) then
    ! checks if neighbors were found
    do ispec_ref = 1,NSPEC_AB
      ! gets number of neighbors
      num_neighbors = neighbors_xadj(ispec_ref+1) - neighbors_xadj(ispec_ref)

      ! element should have neighbors, otherwise mesh is probably invalid
      if (num_neighbors == 0 .and. NSPEC_AB > 1) then
        ! midpoint
        iglob = ibool(MIDX,MIDY,MIDZ,ispec_ref)
        ! error info
        print *,'Error: rank ',myrank,' - element ',ispec_ref,'has no neighbors!'
        print *,'  element midpoint location: ',xstore(iglob),ystore(iglob),zstore(iglob)
        print *,'  maximum search elements  : ',num_elements_max
        call exit_MPI(myrank,'Error adjacency invalid')
      endif
    enddo
  endif

  ! total number of neighbors
  num_neighbors_all = inum_neighbor

  ! allocates compacted array
  allocate(neighbors_adjncy(num_neighbors_all),stat=ier)
  if (ier /= 0) stop 'Error allocating neighbors_adjncy'

  neighbors_adjncy(1:num_neighbors_all) = tmp_adjncy(1:num_neighbors_all)

  ! checks
  if (minval(neighbors_adjncy(:)) < 1 .or. maxval(neighbors_adjncy(:)) > NSPEC_AB) then
    print *,'Error: adjncy array invalid: slice ',myrank
    print *,'       number of all neighbors = ',num_neighbors_all
    print *,'       array min/max           = ',minval(neighbors_adjncy(:)),'/',maxval(neighbors_adjncy(:))
    stop 'Invalid adjncy array'
  endif
  ! frees temporary array
  deallocate(tmp_adjncy)

  ! checks neighbors for out-of-bounds indexing and duplicates
  do ispec_ref = 1,NSPEC_AB
    ! loops over neighbors
    num_neighbors = neighbors_xadj(ispec_ref+1) - neighbors_xadj(ispec_ref)
    do ii = 1,num_neighbors
      ! get neighbor entry
      ielem = neighbors_xadj(ispec_ref) + ii

      ! checks entry index
      if (ielem < 1 .or. ielem > num_neighbors_all) &
        stop 'Invalid element index in neighbors_xadj array'

      ! get neighbor element
      ispec = neighbors_adjncy(ielem)

      ! checks element index
      if (ispec < 1 .or. ispec > NSPEC_AB) &
        stop 'Invalid ispec index in neighbors_adjncy array'

      ! loops over all other neighbors
      do jj = ii+1,num_neighbors
        ! checks for duplicate
        if (neighbors_adjncy(neighbors_xadj(ispec_ref) + jj) == ispec) &
          stop 'Invalid ispec duplicate found in neighbors_adjncy array'
      enddo
    enddo
  enddo

  ! user output
  if (myrank == 0) then
    ! elapsed time since beginning of neighbor detection
    tCPU = wtime() - time1
    write(IMAIN,*)
    write(IMAIN,*) '     maximum neighbors found per element = ',num_neighbors_max
    if (ADD_NEIGHBOR_OF_NEIGHBORS) then
      write(IMAIN,*) '         (maximum neighbor of neighbors) = ',num_neighbor_neighbors_max
    endif
    write(IMAIN,*) '     total number of neighbors           = ',num_neighbors_all
    write(IMAIN,*)
    write(IMAIN,*) '     Elapsed time for detection of neighbors in seconds = ',sngl(tCPU)
    write(IMAIN,*)
    call flush_IMAIN()
  endif

  end subroutine setup_mesh_adjacency
