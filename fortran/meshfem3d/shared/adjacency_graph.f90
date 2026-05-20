
module adjacency_graph_shared
  implicit none

contains

  ! Builds a CSR adjacency graph for a hexahedral mesh.
  !
  ! elmnts_flat(0:8*nspec-1): flat 0-based corner node array; element ispec occupies
  !   positions (ispec-1)*8 to ispec*8-1, in the canonical SPECFEM corner order matching
  !   ibool(1,1,1), ibool(NX,1,1), ibool(NX,NY,1), ibool(1,NY,1),
  !   ibool(1,1,NZ), ibool(NX,1,NZ), ibool(NX,NY,NZ), ibool(1,NY,NZ).
  !
  ! max_neighbors: upper bound on both (a) elements sharing any single node and
  !   (b) direct element-to-element neighbors. Use MAX_NEIGHBORS for meshfem3d or
  !   sup_neighbor (from check_valence) for decompose_mesh.
  !
  ! Output CSR arrays are 1-based:
  !   xadj_adj(1:nspec+1): xadj_adj(ispec) is the first index in adjncy_adj/adj_types
  !                         for element ispec's neighbors; xadj_adj(nspec+1) = nb_adj_edges+1
  !   adjncy_adj(1:nb_adj_edges): neighbor element IDs (1-based)
  !   adj_types(1:nb_adj_edges): connection type codes 1-26 (face 1-6, edge 7-18, corner 19-26)
  subroutine build_adjacency_graph_csr(elmnts_flat, nspec, nglob, max_neighbors, &
                                        xadj_adj, adjncy_adj, adj_types, nb_adj_edges)
    use constants, only: NGNOD_EIGHT_CORNERS
    implicit none

    integer, intent(in)               :: nspec, nglob, max_neighbors
    integer, intent(in)               :: elmnts_flat(0:NGNOD_EIGHT_CORNERS*nspec-1)
    integer, allocatable, intent(out) :: xadj_adj(:)
    integer, allocatable, intent(out) :: adjncy_adj(:)
    integer, allocatable, intent(out) :: adj_types(:)
    integer, intent(out)              :: nb_adj_edges

    integer :: ispec, corner, current_node, i, j, k, ier
    integer :: element1, element2
    integer :: num_shared, shared_nodes(NGNOD_EIGHT_CORNERS)
    integer :: type1, type2
    logical :: already_found

    ! node-to-element reverse mapping
    integer, allocatable :: nelements_for_each_node(:)
    integer, allocatable :: elements_for_each_node(:,:)

    ! flat adjacency storage before compaction: (0:max_neighbors-1, nspec)
    integer, allocatable :: adj_count(:)
    integer, allocatable :: adj_flat(:,:)
    integer, allocatable :: adj_type_flat(:,:)

    ! Step 1: build node-to-element mapping
    allocate(nelements_for_each_node(nglob), stat=ier)
    if (ier /= 0) stop 'Error allocating nelements_for_each_node'
    allocate(elements_for_each_node(0:max_neighbors-1, nglob), stat=ier)
    if (ier /= 0) stop 'Error allocating elements_for_each_node'
    nelements_for_each_node(:) = 0
    elements_for_each_node(:,:) = -1

    do ispec = 1, nspec
      do corner = 0, NGNOD_EIGHT_CORNERS-1
        current_node = elmnts_flat((ispec-1)*NGNOD_EIGHT_CORNERS + corner)
        if (nelements_for_each_node(current_node) >= max_neighbors) then
          write(*,*) 'Error: node', current_node, 'exceeds max_neighbors =', max_neighbors
          stop 'Error in build_adjacency_graph_csr: increase max_neighbors'
        end if
        elements_for_each_node(nelements_for_each_node(current_node), current_node) = ispec
        nelements_for_each_node(current_node) = nelements_for_each_node(current_node) + 1
      end do
    end do

    ! Step 2: pre-allocate flat neighbor storage
    allocate(adj_count(nspec), stat=ier)
    if (ier /= 0) stop 'Error allocating adj_count'
    allocate(adj_flat(0:max_neighbors-1, nspec), stat=ier)
    if (ier /= 0) stop 'Error allocating adj_flat'
    allocate(adj_type_flat(0:max_neighbors-1, nspec), stat=ier)
    if (ier /= 0) stop 'Error allocating adj_type_flat'
    adj_count(:) = 0
    adj_flat(:,:) = -1
    adj_type_flat(:,:) = 0

    ! Step 3: find element pairs and classify connections
    do current_node = 1, nglob
      do i = 0, nelements_for_each_node(current_node)-1
        element1 = elements_for_each_node(i, current_node)
        do j = i+1, nelements_for_each_node(current_node)-1
          element2 = elements_for_each_node(j, current_node)

          call get_shared_nodes_flat(elmnts_flat, nspec, element1, element2, &
                                     shared_nodes, num_shared)
          if (num_shared == 0) cycle

          ! skip if pair already recorded (element2 in element1's neighbor list)
          already_found = .false.
          do k = 0, adj_count(element1)-1
            if (adj_flat(k, element1) == element2) then
              already_found = .true.
              exit
            end if
          end do
          if (already_found) cycle

          select case (num_shared)
          case (1)
            call get_corner_id_flat(elmnts_flat, nspec, element1, shared_nodes(1), type1)
            call get_corner_id_flat(elmnts_flat, nspec, element2, shared_nodes(1), type2)
          case (2)
            call get_edge_id_flat(elmnts_flat, nspec, element1, &
                                  shared_nodes(1), shared_nodes(2), type1)
            call get_edge_id_flat(elmnts_flat, nspec, element2, &
                                  shared_nodes(1), shared_nodes(2), type2)
          case (4)
            call get_face_id_flat(elmnts_flat, nspec, element1, &
                                  shared_nodes(1), shared_nodes(2), shared_nodes(3), &
                                  shared_nodes(4), type1)
            call get_face_id_flat(elmnts_flat, nspec, element2, &
                                  shared_nodes(1), shared_nodes(2), shared_nodes(3), &
                                  shared_nodes(4), type2)
          case default
            write(*,*) 'Error: invalid shared node count', num_shared, &
                       'between elements', element1, 'and', element2
            stop 'Invalid mesh in build_adjacency_graph_csr'
          end select

          ! record bidirectionally
          if (adj_count(element1) >= max_neighbors) then
            write(*,*) 'Error: element', element1, 'exceeds max_neighbors =', max_neighbors
            stop 'Error in build_adjacency_graph_csr: increase max_neighbors (adj_flat)'
          end if
          adj_flat(adj_count(element1), element1) = element2
          adj_type_flat(adj_count(element1), element1) = type1
          adj_count(element1) = adj_count(element1) + 1

          if (adj_count(element2) >= max_neighbors) then
            write(*,*) 'Error: element', element2, 'exceeds max_neighbors =', max_neighbors
            stop 'Error in build_adjacency_graph_csr: increase max_neighbors (adj_flat)'
          end if
          adj_flat(adj_count(element2), element2) = element1
          adj_type_flat(adj_count(element2), element2) = type2
          adj_count(element2) = adj_count(element2) + 1

        end do
      end do
    end do

    deallocate(nelements_for_each_node)
    deallocate(elements_for_each_node)

    ! Step 4: compact to CSR
    nb_adj_edges = sum(adj_count)
    allocate(xadj_adj(nspec+1), stat=ier)
    if (ier /= 0) stop 'Error allocating xadj_adj'
    allocate(adjncy_adj(nb_adj_edges), stat=ier)
    if (ier /= 0) stop 'Error allocating adjncy_adj'
    allocate(adj_types(nb_adj_edges), stat=ier)
    if (ier /= 0) stop 'Error allocating adj_types'

    xadj_adj(1) = 1
    k = 1
    do ispec = 1, nspec
      do j = 0, adj_count(ispec)-1
        adjncy_adj(k) = adj_flat(j, ispec)
        adj_types(k) = adj_type_flat(j, ispec)
        k = k + 1
      end do
      xadj_adj(ispec+1) = k
    end do

    deallocate(adj_count, adj_flat, adj_type_flat)

  end subroutine build_adjacency_graph_csr


  subroutine get_shared_nodes_flat(elmnts_flat, nspec, element1, element2, &
                                    shared_nodes, num_shared)
    use constants, only: NGNOD_EIGHT_CORNERS
    implicit none
    integer, intent(in)  :: elmnts_flat(0:NGNOD_EIGHT_CORNERS*nspec-1)
    integer, intent(in)  :: nspec, element1, element2
    integer, intent(out) :: shared_nodes(NGNOD_EIGHT_CORNERS)
    integer, intent(out) :: num_shared

    integer :: i, j, node1, node2

    num_shared = 0
    shared_nodes(:) = -1
    do i = 0, NGNOD_EIGHT_CORNERS-1
      node1 = elmnts_flat((element1-1)*NGNOD_EIGHT_CORNERS + i)
      do j = 0, NGNOD_EIGHT_CORNERS-1
        node2 = elmnts_flat((element2-1)*NGNOD_EIGHT_CORNERS + j)
        if (node1 == node2) then
          num_shared = num_shared + 1
          shared_nodes(num_shared) = node1
        end if
      end do
    end do
  end subroutine get_shared_nodes_flat


  ! Corner IDs 19-26 are defined by the position of the shared node in the element's
  ! corner array (positions 0-7 in elmnts_flat, corresponding to ibool corners).
  !
  !   19 = bottom_front_left  (pos 0, ibool(1, 1, 1))
  !   20 = bottom_front_right (pos 1, ibool(NX,1, 1))
  !   21 = bottom_back_left   (pos 3, ibool(1, NY,1))
  !   22 = bottom_back_right  (pos 2, ibool(NX,NY,1))
  !   23 = top_front_left     (pos 4, ibool(1, 1, NZ))
  !   24 = top_front_right    (pos 5, ibool(NX,1, NZ))
  !   25 = top_back_left      (pos 7, ibool(1, NY,NZ))
  !   26 = top_back_right     (pos 6, ibool(NX,NY,NZ))
  subroutine get_corner_id_flat(elmnts_flat, nspec, element, node, corner_id)
    use constants, only: NGNOD_EIGHT_CORNERS
    implicit none
    integer, intent(in)  :: elmnts_flat(0:NGNOD_EIGHT_CORNERS*nspec-1)
    integer, intent(in)  :: nspec, element, node
    integer, intent(out) :: corner_id

    integer :: base
    base = (element-1)*NGNOD_EIGHT_CORNERS

    if      (elmnts_flat(base + 0) == node) then; corner_id = 19
    else if (elmnts_flat(base + 1) == node) then; corner_id = 20
    else if (elmnts_flat(base + 3) == node) then; corner_id = 21
    else if (elmnts_flat(base + 2) == node) then; corner_id = 22
    else if (elmnts_flat(base + 4) == node) then; corner_id = 23
    else if (elmnts_flat(base + 5) == node) then; corner_id = 24
    else if (elmnts_flat(base + 7) == node) then; corner_id = 25
    else if (elmnts_flat(base + 6) == node) then; corner_id = 26
    else
      write(*,*) 'Error: node', node, 'is not a corner of element', element
      stop 'Error in get_corner_id_flat: invalid corner node'
    end if
  end subroutine get_corner_id_flat


  ! Edge IDs 7-18.  Each edge is defined by two corner positions.
  !
  !    7 = bottom_left   (pos 0, pos 3)
  !    8 = bottom_right  (pos 1, pos 2)
  !    9 = top_right     (pos 5, pos 6)
  !   10 = top_left      (pos 4, pos 7)
  !   11 = front_bottom  (pos 0, pos 1)
  !   12 = front_top     (pos 4, pos 5)
  !   13 = front_left    (pos 0, pos 4)
  !   14 = front_right   (pos 1, pos 5)
  !   15 = back_bottom   (pos 3, pos 2)
  !   16 = back_top      (pos 7, pos 6)
  !   17 = back_left     (pos 3, pos 7)
  !   18 = back_right    (pos 2, pos 6)
  subroutine get_edge_id_flat(elmnts_flat, nspec, element, node1, node2, edge_id)
    use constants, only: NGNOD_EIGHT_CORNERS
    implicit none
    integer, intent(in)  :: elmnts_flat(0:NGNOD_EIGHT_CORNERS*nspec-1)
    integer, intent(in)  :: nspec, element, node1, node2
    integer, intent(out) :: edge_id

    integer :: base, n(0:7)
    base = (element-1)*NGNOD_EIGHT_CORNERS
    n(0) = elmnts_flat(base+0); n(1) = elmnts_flat(base+1)
    n(2) = elmnts_flat(base+2); n(3) = elmnts_flat(base+3)
    n(4) = elmnts_flat(base+4); n(5) = elmnts_flat(base+5)
    n(6) = elmnts_flat(base+6); n(7) = elmnts_flat(base+7)

    if      (edge_match(node1,node2,n(0),n(3))) then; edge_id = 7
    else if (edge_match(node1,node2,n(1),n(2))) then; edge_id = 8
    else if (edge_match(node1,node2,n(5),n(6))) then; edge_id = 9
    else if (edge_match(node1,node2,n(4),n(7))) then; edge_id = 10
    else if (edge_match(node1,node2,n(0),n(1))) then; edge_id = 11
    else if (edge_match(node1,node2,n(4),n(5))) then; edge_id = 12
    else if (edge_match(node1,node2,n(0),n(4))) then; edge_id = 13
    else if (edge_match(node1,node2,n(1),n(5))) then; edge_id = 14
    else if (edge_match(node1,node2,n(3),n(2))) then; edge_id = 15
    else if (edge_match(node1,node2,n(7),n(6))) then; edge_id = 16
    else if (edge_match(node1,node2,n(3),n(7))) then; edge_id = 17
    else if (edge_match(node1,node2,n(2),n(6))) then; edge_id = 18
    else
      write(*,*) 'Error: nodes', node1, node2, 'not an edge of element', element
      stop 'Error in get_edge_id_flat: invalid edge nodes'
    end if
  end subroutine get_edge_id_flat

  logical function edge_match(a, b, p, q)
    integer, intent(in) :: a, b, p, q
    edge_match = (a == p .and. b == q) .or. (a == q .and. b == p)
  end function edge_match


  ! Face IDs 1-6.  Each face is defined by four corner positions.
  !
  !   1 = bottom  (pos 0, 1, 2, 3)
  !   2 = right   (pos 1, 2, 6, 5)
  !   3 = top     (pos 4, 5, 6, 7)
  !   4 = left    (pos 0, 3, 7, 4)
  !   5 = front   (pos 0, 1, 5, 4)
  !   6 = back    (pos 3, 2, 6, 7)
  subroutine get_face_id_flat(elmnts_flat, nspec, element, &
                               node1, node2, node3, node4, face_id)
    use constants, only: NGNOD_EIGHT_CORNERS
    implicit none
    integer, intent(in)  :: elmnts_flat(0:NGNOD_EIGHT_CORNERS*nspec-1)
    integer, intent(in)  :: nspec, element
    integer, intent(in)  :: node1, node2, node3, node4
    integer, intent(out) :: face_id

    integer :: base
    integer :: input_nodes(4)
    integer :: bottom(4), right(4), top(4), left(4), front(4), back(4)

    base = (element-1)*NGNOD_EIGHT_CORNERS

    bottom = [elmnts_flat(base+0), elmnts_flat(base+1), &
              elmnts_flat(base+2), elmnts_flat(base+3)]
    right  = [elmnts_flat(base+1), elmnts_flat(base+2), &
              elmnts_flat(base+6), elmnts_flat(base+5)]
    top    = [elmnts_flat(base+4), elmnts_flat(base+5), &
              elmnts_flat(base+6), elmnts_flat(base+7)]
    left   = [elmnts_flat(base+0), elmnts_flat(base+3), &
              elmnts_flat(base+7), elmnts_flat(base+4)]
    front  = [elmnts_flat(base+0), elmnts_flat(base+1), &
              elmnts_flat(base+5), elmnts_flat(base+4)]
    back   = [elmnts_flat(base+3), elmnts_flat(base+2), &
              elmnts_flat(base+6), elmnts_flat(base+7)]

    input_nodes = [node1, node2, node3, node4]
    call sort4(input_nodes)
    call sort4(bottom); call sort4(right); call sort4(top)
    call sort4(left);   call sort4(front); call sort4(back)

    if      (all(input_nodes == bottom)) then; face_id = 1
    else if (all(input_nodes == right))  then; face_id = 2
    else if (all(input_nodes == top))    then; face_id = 3
    else if (all(input_nodes == left))   then; face_id = 4
    else if (all(input_nodes == front))  then; face_id = 5
    else if (all(input_nodes == back))   then; face_id = 6
    else
      write(*,*) 'Error: nodes', node1, node2, node3, node4, &
                 'not a face of element', element
      stop 'Error in get_face_id_flat: invalid face nodes'
    end if
  end subroutine get_face_id_flat

  subroutine sort4(arr)
    implicit none
    integer, dimension(4), intent(inout) :: arr
    integer :: i, j, tmp
    do i = 1, 3
      do j = 1, 4-i
        if (arr(j) > arr(j+1)) then
          tmp = arr(j); arr(j) = arr(j+1); arr(j+1) = tmp
        end if
      end do
    end do
  end subroutine sort4

end module adjacency_graph_shared
