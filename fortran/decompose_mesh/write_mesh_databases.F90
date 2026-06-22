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


  subroutine write_mesh_databases()

! writes out new Databases files for each partition

  use decompose_mesh_par
  use constants, only: NGLLX, NGLLY, NGLLZ

  ! use fault_scotch, only: ANY_FAULT,write_fault_database,nodes_coords_open

  implicit none

  ! local parameters
  integer :: ier
  character(len=MAX_STRING_LEN) :: database_file
  character(len=32) :: proc_str_db, format_str_db
  integer :: ndigits_db

  ! user output
  print *, 'partitions: '
  print *, '  num = ',nparts
  print *

  if (COUPLE_WITH_INJECTION_TECHNIQUE .or. MESH_A_CHUNK_OF_THE_EARTH) open(124,file='Numglob2loc_elmn.txt')

  ! writes out Database file for each partition
  do ipart = 0, nparts-1

    ! opens output file — naming matches meshfem3d save_databases.F90 convention
    if (nparts <= 1) then
      database_file = trim(outputpath_name) // '/Database.bin'
    else
      ndigits_db = int(log10(real(nparts-1))) + 1
      write(format_str_db,'(a,i0,a,i0,a)') '(i', ndigits_db, '.', ndigits_db, ')'
      write(proc_str_db, format_str_db) ipart
      if (ipart == 0) &
        call execute_command_line('mkdir -p ' // trim(outputpath_name) // '/Database', wait=.true.)
      database_file = trim(outputpath_name) // '/Database/proc_' // trim(adjustl(proc_str_db)) // '.bin'
    endif

    open(unit=IIN_database, file=trim(database_file), &
         status='unknown', action='write', form='unformatted', iostat=ier)
    if (ier /= 0) then
      print *,'Error file open: ',trim(database_file)
      print *
      print *,'check if path exists: ',trim(outputpath_name)
      stop 'Error file open Database'
    endif

    ! file header — matches meshfem3d save_databases.F90 binary format
    write(IIN_database) MESH_A_CHUNK_OF_THE_EARTH
    write(IIN_database) UTM_PROJECTION_ZONE
    write(IIN_database) SUPPRESS_UTM_PROJECTION
    write(IIN_database) NGNOD

    ! gets number of local nodes nnodes_loc in this partition
    call write_glob2loc_nodes_database(IIN_database, ipart, nnodes_loc, nodes_coords, &
                                       glob2loc_nodes_nparts, glob2loc_nodes_parts, &
                                       glob2loc_nodes, nnodes, 1)

    ! gets number of local elements nspec_local in this parition
    call write_partition_database(IIN_database, ipart, nspec_local, nspec, elmnts, &
                                  glob2loc_elmnts, glob2loc_nodes_nparts, &
                                  glob2loc_nodes_parts, glob2loc_nodes, part, mat, NGNOD, 1)

    !debug
    !print *, ipart,": nspec_local=",nspec_local, " nnodes_local=", nnodes_loc

    ! writes out node coordinate locations
    write(IIN_database) nnodes_loc

    ! writes out coordinates nodes_coords for local nodes
    call write_glob2loc_nodes_database(IIN_database, ipart, nnodes_loc, nodes_coords, &
                                       glob2loc_nodes_nparts, glob2loc_nodes_parts, &
                                       glob2loc_nodes, nnodes, 2)

    call write_material_props_database(IIN_database,count_def_mat,count_undef_mat, &
                                       mat_prop, undef_mat_prop)

    ! writes out spectral element indices
    write(IIN_database) nspec_local
    write(IIN_database) NGLLZ, NGLLY, NGLLX

    call write_partition_database(IIN_database, ipart, nspec_local, nspec, elmnts, &
                                  glob2loc_elmnts, glob2loc_nodes_nparts, &
                                  glob2loc_nodes_parts, glob2loc_nodes, part, mat, NGNOD, 2)


    ! writes out absorbing/free-surface boundaries
    call write_boundaries_database(IIN_database, ipart, nspec, nspec2D_xmin, nspec2D_xmax, nspec2D_ymin, &
                                  nspec2D_ymax, nspec2D_bottom, nspec2D_top, &
                                  ibelm_xmin, ibelm_xmax, ibelm_ymin, &
                                  ibelm_ymax, ibelm_bottom, ibelm_top, &
                                  nodes_ibelm_xmin, nodes_ibelm_xmax, nodes_ibelm_ymin, &
                                  nodes_ibelm_ymax, nodes_ibelm_bottom, nodes_ibelm_top, &
                                  glob2loc_elmnts, glob2loc_nodes_nparts, &
                                  glob2loc_nodes_parts, glob2loc_nodes, part, NGNOD2D)

    ! writes out C-PML elements indices, CPML-regions and thickness of C-PML layer
    call write_cpml_database(IIN_database, ipart, nspec, nspec_cpml, CPML_to_spec, &
                             CPML_regions, is_CPML, glob2loc_elmnts, part)

    ! Note: MPI interface and moho surface database writing has been moved to
    ! the adjacency_database format (Issue #1802). Cross-partition connectivity
    ! is now computed from the 7-column MPI adjacency format read by
    ! read_adjacency_graph.cpp. Legacy interface database code removed.

    ! writes out element adjacency with face/edge/corner type codes
    call write_adjacency_database(IIN_database, ipart)

    close(IIN_database)
  enddo

  if (COUPLE_WITH_INJECTION_TECHNIQUE .or. MESH_A_CHUNK_OF_THE_EARTH) close(124)

  ! ! writes fault database
  ! if (ANY_FAULT) then
  !   do ipart = 0, nparts-1
  !     ! opens file
  !     write(prname, "(i6.6,'_Database_fault')") ipart

  !     open(unit=16,file=outputpath_name(1:len_trim(outputpath_name))//'/proc'//prname, &
  !          status='replace', action='write', form='unformatted', iostat = ier)
  !     if (ier /= 0) then
  !       print *,'Error file open:',outputpath_name(1:len_trim(outputpath_name))//'/proc'//prname
  !       print *
  !       print *,'check if path exists:',outputpath_name(1:len_trim(outputpath_name))
  !       stop
  !     endif

  !     ! writes out fault element indices
  !     call write_fault_database(16, ipart, nspec, &
  !                               glob2loc_elmnts, glob2loc_nodes_nparts, glob2loc_nodes_parts, &
  !                               glob2loc_nodes, part)

  !     ! gets number of local nodes nnodes_loc in this partition
  !     call write_glob2loc_nodes_database(16, ipart, nnodes_loc, nodes_coords_open, &
  !                                        glob2loc_nodes_nparts, glob2loc_nodes_parts, &
  !                                        glob2loc_nodes, nnodes, 1)

  !     ! ascii file format
  !     !write(16,*) nnodes_loc
  !     ! binary file format
  !     write(16) nnodes_loc

  !     ! writes out coordinates nodes_coords_open of local nodes
  !     ! note that we output here the "open" split node positions for the fault nodal points
  !     call write_glob2loc_nodes_database(16, ipart, nnodes_loc, nodes_coords_open, &
  !                                        glob2loc_nodes_nparts, glob2loc_nodes_parts, &
  !                                        glob2loc_nodes, nnodes, 2)
  !     close(16)
  !   enddo
  ! endif


  ! setting up wavefield discontinuity boundary
  if (IS_WAVEFIELD_DISCONTINUITY) then
    do ipart = 0, nparts-1
      call write_wavefield_discontinuity_database(ipart, outputpath_name, &
                                                  nb_wd, boundary_to_ispec_wd, side_wd, &
                                                  nspec, glob2loc_elmnts, part)
    enddo
  endif

  ! cleanup
  deallocate(CPML_to_spec,stat=ier); if (ier /= 0) stop 'Error deallocating array CPML_to_spec'
  deallocate(CPML_regions,stat=ier); if (ier /= 0) stop 'Error deallocating array CPML_regions'
  deallocate(is_CPML,stat=ier); if (ier /= 0) stop 'Error deallocating array is_CPML'

  ! user output
  print *
  print *, 'Databases files in directory: ',outputpath_name(1:len_trim(outputpath_name))
  print *

  end subroutine write_mesh_databases


  subroutine write_adjacency_database(IIN_database, ipart)

  ! Writes element adjacency matching the meshfem3d save_databases.F90 binary format:
  !   local (intra-partition) edges: total count, then (ispec_local, jspec_local, 1, adj_type)
  !   MPI (cross-partition) edges:   total count, then 7-column tuples matching the MPI
  !     adjacency format read by read_adjacency_graph.cpp

  use decompose_mesh_par, only: nspec, xadj_typed, adjncy_typed, adj_types_global, &
                                 glob2loc_elmnts, part, elmnts, NGNOD, nnonconforming_adjacencies, nonconforming_ispec_source, nonconforming_ispec_target,nonconforming_adjacency_type,nonconforming_adjacency_face

  implicit none

  integer, intent(in) :: IIN_database, ipart

  integer :: ispec, k, m_edge, jspec, ispec_local, jspec_local
  integer :: total_nadj_element, index
  integer :: num_mpi_adjacencies
  integer :: ia, ib, node_ia, node_ib
  integer :: my_conn_id, neighbor_conn_id, anchor_local, anchor_remote
  integer :: neighbor_proc
  logical :: reverse_found

  ! Mapping from elmnts array position (1-8) to element-absolute corner ID (19-26).
  ! Follows SPECFEM hex node ordering convention:
  !   pos 1 → 19 (BFL), pos 2 → 20 (BFR), pos 3 → 22 (BBR), pos 4 → 21 (BBL),
  !   pos 5 → 23 (TFL), pos 6 → 24 (TFR), pos 7 → 26 (TBR), pos 8 → 25 (TBL)
  integer, parameter :: pos_to_corner_id(8) = [19, 20, 22, 21, 23, 24, 26, 25]

  ! Count and write local adjacencies
  total_nadj_element = 0
  do ispec = 1, nspec
    if (part(ispec) /= ipart) cycle
    do k = xadj_typed(ispec), xadj_typed(ispec+1)-1
      if (part(adjncy_typed(k)) == ipart) total_nadj_element = total_nadj_element + 1
    end do
  end do

  ! total conforming + nonconforming adjacencies
  write(IIN_database) total_nadj_element + nnonconforming_adjacencies
  index = 0

  ! write weakly conforming
  do ispec = 1, nspec
    if (part(ispec) /= ipart) cycle
    ispec_local = glob2loc_elmnts(ispec-1) + 1
    do k = xadj_typed(ispec), xadj_typed(ispec+1)-1
      jspec = adjncy_typed(k)
      if (part(jspec) /= ipart) cycle
      jspec_local = glob2loc_elmnts(jspec-1) + 1
      write(IIN_database) ispec_local, jspec_local, 1, adj_types_global(k)
      index = index + 1
    end do
  end do

  if (index /= total_nadj_element) then
    print *,'Error: index ',index,' /= total_nadj_element ',total_nadj_element,' in partition ',ipart
    stop 'Error in write_adjacency_database (local)'
  endif

  ! write nonconforming
  do k = 1,nnonconforming_adjacencies
    write(IIN_database) nonconforming_ispec_source(k), nonconforming_ispec_target(k), nonconforming_adjacency_type(k), nonconforming_adjacency_face(k)
  end do


  ! Count MPI (cross-partition) adjacencies
  num_mpi_adjacencies = 0
  do ispec = 1, nspec
    if (part(ispec) /= ipart) cycle
    do k = xadj_typed(ispec), xadj_typed(ispec+1)-1
      if (part(adjncy_typed(k)) /= ipart) num_mpi_adjacencies = num_mpi_adjacencies + 1
    end do
  end do

  write(IIN_database) num_mpi_adjacencies

  ! Debug check: ensure adjacency graph is symmetric (optional, disabled by default)
  ! Each edge should have a corresponding reverse edge. This check is O(E²) where E is
  ! the number of edges, so only enable for small test cases or when debugging.
  ! To enable, set DEBUG_ADJACENCY_GRAPH = .true.
  do ispec = 1, nspec
    if (part(ispec) /= ipart) cycle
    do k = xadj_typed(ispec), xadj_typed(ispec+1)-1
      jspec = adjncy_typed(k)
      reverse_found = .false.
      do m_edge = xadj_typed(jspec), xadj_typed(jspec+1)-1
        if (adjncy_typed(m_edge) == ispec) then
          reverse_found = .true.
          exit
        endif
      end do
      if (.not. reverse_found) then
        print *,'Warning: asymmetric edge from ispec ',ispec,' to jspec ',jspec, &
                ' in partition ',ipart
      endif
    end do
  end do


  ! Write MPI adjacencies
  ! Format per row (7 columns), matching meshfem3d save_databases.F90:
  !   col 1: ispec_local      — local element index (1-based)
  !   col 2: neighbor_proc    — MPI rank of the neighbor partition
  !   col 3: jspec_local      — element index on neighbor partition (1-based)
  !   col 4: my_conn_id       — face/edge/corner connection ID on local element
  !   col 5: neighbor_conn_id — face/edge/corner connection ID on remote element
  !   col 6: anchor_local     — corner index (1-based) on local element for orientation
  !   col 7: anchor_remote    — matching corner index (1-based) on remote element
  do ispec = 1, nspec
    if (part(ispec) /= ipart) cycle
    ispec_local = glob2loc_elmnts(ispec-1) + 1
    do k = xadj_typed(ispec), xadj_typed(ispec+1)-1
      jspec = adjncy_typed(k)
      if (part(jspec) == ipart) cycle

      neighbor_proc = part(jspec)
      jspec_local   = glob2loc_elmnts(jspec-1) + 1
      my_conn_id    = adj_types_global(k)

      ! Find reverse edge jspec -> ispec to get neighbor_conn_id
      neighbor_conn_id = 0
      reverse_found = .false.
      do m_edge = xadj_typed(jspec), xadj_typed(jspec+1)-1
        if (adjncy_typed(m_edge) == ispec) then
          neighbor_conn_id = adj_types_global(m_edge)
          reverse_found = .true.
          exit
        endif
      end do

      ! Validate reverse edge was found (graph should be symmetric)
      if (.not. reverse_found) then
        print *,'ERROR: No reverse edge found from jspec ',jspec,' back to ispec ',ispec
        print *,'       in partition ',ipart
        stop 'Error in write_adjacency_database: asymmetric adjacency graph'
      endif

      ! Find anchor: first shared corner node between ispec and jspec.
      ! elmnts(ia, ispec) is a global node ID; equality means same node.
      ! Convert array positions (1-8) to element-absolute corner IDs (19-26).
      anchor_local  = 0
      anchor_remote = 0
      outer: do ia = 1, NGNOD
        node_ia = elmnts(ia, ispec)
        do ib = 1, NGNOD
          node_ib = elmnts(ib, jspec)
          if (node_ia == node_ib) then
            anchor_local  = pos_to_corner_id(ia)
            anchor_remote = pos_to_corner_id(ib)
            exit outer
          endif
        end do
      end do outer

      ! Validate anchors were found (elements must share at least one corner)
      if (anchor_local == 0 .or. anchor_remote == 0) then
        print *,'ERROR: No shared corner found between ispec ',ispec,' and jspec ',jspec
        print *,'       in partition ',ipart
        print *,'       ispec element nodes: ', elmnts(1:NGNOD, ispec)
        print *,'       jspec element nodes: ', elmnts(1:NGNOD, jspec)
        stop 'Error in write_adjacency_database: missing anchor node'
      endif

      write(IIN_database) ispec_local, neighbor_proc, jspec_local, &
                          my_conn_id, neighbor_conn_id, &
                          anchor_local, anchor_remote
    end do
  end do

  end subroutine write_adjacency_database
