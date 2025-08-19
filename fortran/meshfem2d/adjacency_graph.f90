

module adjacency_graph
   implicit none

contains

   subroutine compute_adjacency_graph()

      use constants, only: MAX_NEIGHBORS, NCORNERS
      use shared_parameters, only: NGNOD, nelmnts
      use part_unstruct_par, only: elmnts, num_adjacent, adjacency_type, adjacency_id, adjacent_elements

      implicit none

      integer :: i, ier
      integer, allocatable :: elmnts_bis(:)

      allocate(elmnts_bis(0:NCORNERS*nelmnts-1),stat=ier)
      allocate(num_adjacent(0:nelmnts-1),stat=ier)
      allocate(adjacency_type(0:nelmnts-1, 0:MAX_NEIGHBORS),stat=ier)
      allocate(adjacency_id(0:nelmnts-1, 0:MAX_NEIGHBORS),stat=ier)
      allocate(adjacent_elements(0:nelmnts-1, 0:MAX_NEIGHBORS),stat=ier)
      num_adjacent(:) = 0
      adjacency_type(:,:) = -1
      adjacency_id(:,:) = -1

      if (NGNOD == 9) then
         do i = 0, nelmnts-1
            elmnts_bis(i*NCORNERS:i*NCORNERS+NCORNERS-1) = elmnts(i*NGNOD:i*NGNOD+NCORNERS-1)
         enddo
      else if (NGNOD == 4) then
         elmnts_bis = elmnts
      else
         call stop_the_code('Error: NGNOD must be either 4 or 9 in adjacency_mapping')
      endif

      call get_adjacencies(elmnts_bis)

   end subroutine compute_adjacency_graph

   subroutine get_adjacencies(elmnts_bis)
      use constants, only: MAX_NEIGHBORS, MAX_NSIZE_SHARED, NCORNERS, ISTRONGLY_CONFORMING
      use part_unstruct_par, only: adjacency_type, adjacency_id, adjacent_elements, num_adjacent, nnodes
      use shared_parameters, only: nelmnts

      implicit none
      integer :: i, current_node, k, l, other_node
      integer :: current_elem, neighbor_elem
      integer :: edge_id1, edge_id2
      integer :: corner_id1, corner_id2
      integer, allocatable :: nnodes_elmnts(:)
      integer, allocatable :: nodes_elmnts(:,:)

      integer, intent(in) :: elmnts_bis(0:NCORNERS*nelmnts-1)

      ! List all elements at each corner node

      if (.not. allocated(nnodes_elmnts) ) allocate(nnodes_elmnts(0:nnodes-1))
      if (.not. allocated(nodes_elmnts) ) allocate(nodes_elmnts(0:nnodes-1, 0:MAX_NSIZE_SHARED-1))

      ! initializes
      nnodes_elmnts(:) = 0
      nodes_elmnts(:,:) = -1

      ! build the list of elements connected to each node
      do i = 0, NCORNERS*nelmnts-1
         nodes_elmnts(elmnts_bis(i), nnodes_elmnts(elmnts_bis(i))) = i / NCORNERS
         nnodes_elmnts(elmnts_bis(i)) = nnodes_elmnts(elmnts_bis(i)) + 1
      enddo

      ! now loop over all nodes and find adjacent elements
      do current_node = 0, nnodes-1
         do k = 0, nnodes_elmnts(current_node) - 1
            do l = k+1, nnodes_elmnts(current_node) - 1
               ! check if elements nodes share another node
               current_elem = nodes_elmnts(current_node, k)
               neighbor_elem = nodes_elmnts(current_node, l)
               if (current_elem == -1 .or. neighbor_elem == -1) then
                  call stop_the_code('Error: Invalid element index in adjacency graph')
               endif
               call check_shared_nodes(elmnts_bis, current_elem, neighbor_elem, current_node, other_node)
               if (other_node /= -1) then
                  ! They share two nodes, so they are strongly conforming along an edge
                  call get_edge(elmnts_bis, current_elem, current_node, other_node, edge_id1)
                  call get_edge(elmnts_bis, neighbor_elem, current_node, other_node, edge_id2)

                  ! count only if edge is not accounted yet
                  if (edge_id1 > 0) then
                     adjacent_elements(current_elem, num_adjacent(current_elem)) = neighbor_elem
                     adjacency_type(current_elem, num_adjacent(current_elem)) = ISTRONGLY_CONFORMING
                     adjacency_id(current_elem, num_adjacent(current_elem)) = edge_id1
                     num_adjacent(current_elem) = num_adjacent(current_elem) + 1
                  endif

                  ! count the neighbor's adjacency only if not accounted yet
                  if (edge_id2 > 0) then
                     adjacent_elements(neighbor_elem, num_adjacent(neighbor_elem)) = current_elem
                     adjacency_type(neighbor_elem, num_adjacent(neighbor_elem)) = ISTRONGLY_CONFORMING
                     adjacency_id(neighbor_elem, num_adjacent(neighbor_elem)) = edge_id2
                     num_adjacent(neighbor_elem) = num_adjacent(neighbor_elem) + 1
                  endif
               else ! Assign this a corner node
                  ! They share only one node, so they are strongly conforming at a corner
                  call get_corner_id(elmnts_bis, current_elem, current_node, corner_id1)
                  call get_corner_id(elmnts_bis, neighbor_elem, current_node, corner_id2)
                  adjacent_elements(current_elem, num_adjacent(current_elem)) = neighbor_elem
                  adjacency_type(current_elem, num_adjacent(current_elem)) = ISTRONGLY_CONFORMING
                  adjacency_id(current_elem, num_adjacent(current_elem)) = corner_id1
                  num_adjacent(current_elem) = num_adjacent(current_elem) + 1

                  adjacent_elements(neighbor_elem, num_adjacent(neighbor_elem)) = current_elem
                  adjacency_type(neighbor_elem, num_adjacent(neighbor_elem)) = ISTRONGLY_CONFORMING
                  adjacency_id(neighbor_elem, num_adjacent(neighbor_elem)) = corner_id2
                  num_adjacent(neighbor_elem) = num_adjacent(neighbor_elem) + 1
               endif

               if (num_adjacent(current_elem) > MAX_NEIGHBORS) then
                  write(*,*) 'Error: too many adjacent elements for element ', current_elem, ' (Modify the mesh)'
                  call stop_the_code('Error: too many adjacent elements (Modify the mesh)')
               endif
            enddo
         enddo
      enddo
   end subroutine get_adjacencies

   subroutine check_shared_nodes(elmnts_bis, elem1, elem2, current_node, other_node)
      use constants, only: NCORNERS
      use shared_parameters, only: nelmnts
      implicit none
      integer, intent(in) :: elmnts_bis(0:NCORNERS*nelmnts-1)
      integer, intent(in) :: elem1, elem2, current_node
      integer, intent(out) :: other_node

      integer :: i, j
      integer :: node1, node2
      integer :: count_shared

      other_node = -1
      count_shared = 0

      do i = 0, NCORNERS-1
         node1 = elmnts_bis(elem1*NCORNERS + i)
         if (node1 == current_node) cycle
         do j = 0, NCORNERS-1
            node2 = elmnts_bis(elem2*NCORNERS + j)
            if (node2 == current_node) cycle
            if (node1 == node2) then
               other_node = node1
               count_shared = count_shared + 1
            endif
         enddo
      enddo

      if (count_shared > 1) then
         write(*,*) 'Error: Elements ', elem1, ' and ', elem2, ' share more than two nodes'
         call stop_the_code('Error: Elements share more than two nodes')
      endif

      if (count_shared == 0) then
         other_node = -1
      endif
   end subroutine check_shared_nodes

   subroutine get_edge(elmnts_bis, elem, node1, node2, edge_id)
      use constants, only: NCORNERS, IBOTTOM_LEFT, IBOTTOM_RIGHT, ITOP_RIGHT, ITOP_LEFT
      use constants, only: IBOTTOM, IRIGHT, ITOP, ILEFT
      use shared_parameters, only: nelmnts
      implicit none
      integer, intent(in) :: elmnts_bis(0:NCORNERS*nelmnts-1)
      integer, intent(in) :: elem, node1, node2
      integer, intent(out) :: edge_id
      integer :: bottom_left_index, bottom_right_index
      integer :: top_right_index, top_left_index

      integer :: i
      integer :: n1, n2
      integer :: bottom_nodes(2), right_nodes(2), top_nodes(2), left_nodes(2)

      edge_id = -1

      bottom_nodes = [elmnts_bis(elem*NCORNERS + IBOTTOM_LEFT - 5), elmnts_bis(elem*NCORNERS + IBOTTOM_RIGHT - 5)]
      right_nodes = [elmnts_bis(elem*NCORNERS + IBOTTOM_RIGHT - 5), elmnts_bis(elem*NCORNERS + ITOP_RIGHT - 5)]
      top_nodes = [elmnts_bis(elem*NCORNERS + ITOP_LEFT - 5), elmnts_bis(elem*NCORNERS + ITOP_RIGHT - 5)]
      left_nodes = [elmnts_bis(elem*NCORNERS + IBOTTOM_LEFT - 5), elmnts_bis(elem*NCORNERS + ITOP_LEFT - 5)]

      if (node1 ==  bottom_nodes(1) .and. node2 == bottom_nodes(2)) then
         edge_id = IBOTTOM
         return
      endif

      if (node1 ==  bottom_nodes(2) .and. node2 == bottom_nodes(1)) then
         edge_id = 0 ! we set this to avoid double counting
         return
      endif

      if (node1 ==  right_nodes(1) .and. node2 == right_nodes(2)) then
         edge_id = IRIGHT
         return
      endif

      if (node1 ==  right_nodes(2) .and. node2 == right_nodes(1)) then
         edge_id = 0
         return
      endif

      if (node1 ==  top_nodes(1) .and. node2 == top_nodes(2)) then
         edge_id = ITOP
         return
      endif

      if (node1 ==  top_nodes(2) .and. node2 == top_nodes(1)) then
         edge_id = 0
         return
      endif

      if (node1 ==  left_nodes(1) .and. node2 == left_nodes(2)) then
         edge_id = ILEFT
         return
      endif

      if (node1 ==  left_nodes(2) .and. node2 == left_nodes(1)) then
         edge_id = 0
         return
      endif

      if (edge_id == -1) then
         write(*,*) 'Error: Could not find edge for element ', elem, ' with nodes ', node1, ' and ', node2
         call stop_the_code('Error: Could not find edge for element with nodes')
      endif

   end subroutine get_edge

   subroutine get_corner_id(elmnts_bis, elem, node, corner_id)
      use constants, only: IBOTTOM_LEFT, IBOTTOM_RIGHT, ITOP_RIGHT, ITOP_LEFT, NCORNERS
      use shared_parameters, only: nelmnts
      implicit none
      integer, intent(in) :: elem, node
      integer, intent(out) :: corner_id
      integer, intent(in) :: elmnts_bis(0:NCORNERS*nelmnts-1)

      integer :: i
      corner_id = -1

      if (elmnts_bis(elem*NCORNERS + IBOTTOM_LEFT - 5) == node) then
         corner_id = IBOTTOM_LEFT
      else if (elmnts_bis(elem*NCORNERS + IBOTTOM_RIGHT - 5) == node) then
         corner_id = IBOTTOM_RIGHT
      else if (elmnts_bis(elem*NCORNERS + ITOP_RIGHT - 5) == node) then
         corner_id = ITOP_RIGHT
      else if (elmnts_bis(elem*NCORNERS + ITOP_LEFT - 5) == node) then
         corner_id = ITOP_LEFT
      else
         write(*,*) 'Error: Node ', node, ' is not a corner of element ', elem
         call stop_the_code('Error: Node is not a corner of element')
      endif


   end subroutine get_corner_id

end module adjacency_graph
