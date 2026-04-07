
module adjacency_graph
   implicit none

contains

   subroutine compute_adjacency_graph(nglob)
      use constants_meshfem, only: NCORNERS, NGLLX_M, NGLLY_M, NGLLZ_M
      use meshfem_par, only: ibool, adjacency_matrix, nspec

      implicit none

      integer :: i, ier
      integer, allocatable :: elmnts_bis(:)
      integer :: ispec
      integer :: nglob

      allocate(elmnts_bis(0:NCORNERS*nspec-1), stat=ier)
      if (ier /= 0) stop 'Error allocating elmnts_bis array'

      allocate(adjacency_matrix(nspec, nspec), stat=ier)
      if (ier /= 0) stop 'Error allocating adjacency_matrix array'

      adjacency_matrix(:,:) = 0

      do ispec = 1, nspec
         elmnts_bis((ispec - 1) * NCORNERS + 0) = ibool(1,1,1,ispec)
         elmnts_bis((ispec - 1) * NCORNERS + 1) = ibool(NGLLX_M,1,1,ispec)
         elmnts_bis((ispec - 1) * NCORNERS + 2) = ibool(NGLLX_M,NGLLY_M,1,ispec)
         elmnts_bis((ispec - 1) * NCORNERS + 3) = ibool(1,NGLLY_M,1,ispec)
         elmnts_bis((ispec - 1) * NCORNERS + 4) = ibool(1,1,NGLLZ_M,ispec)
         elmnts_bis((ispec - 1) * NCORNERS + 5) = ibool(NGLLX_M,1,NGLLZ_M,ispec)
         elmnts_bis((ispec - 1) * NCORNERS + 6) = ibool(NGLLX_M,NGLLY_M,NGLLZ_M,ispec)
         elmnts_bis((ispec - 1) * NCORNERS + 7) = ibool(1,NGLLY_M,NGLLZ_M,ispec)
      end do

      call build_adjacency_graph(elmnts_bis, nglob)

      if (allocated(elmnts_bis)) then
         deallocate(elmnts_bis, stat=ier)
         if (ier /= 0) write(*,*) 'Warning: Error deallocating elmnts_bis array'
      endif

   end subroutine compute_adjacency_graph

   subroutine build_adjacency_graph(elmnts_bis, nglob)

      use constants_meshfem, only: NCORNERS, MAX_NEIGHBORS
      use meshfem_par, only: adjacency_matrix, nspec
      implicit none

      integer :: i, j, element, element1, element2
      integer :: current_node, corner
      integer :: num_shared
      integer :: corner_id1, corner_id2
      integer :: edge_id1, edge_id2
      integer :: face_id1, face_id2
      integer :: shared_nodes(NCORNERS)


      integer, intent(in) :: nglob
      integer, intent(in) :: elmnts_bis(0:NCORNERS*nspec-1)
      integer, allocatable :: nelements_for_each_node(:)
      integer, allocatable :: elements_for_each_node(:,:)

      allocate(nelements_for_each_node(nglob))
      allocate(elements_for_each_node(nglob, 0:MAX_NEIGHBORS-1))

      nelements_for_each_node(:) = 0
      elements_for_each_node(:,:) = -1

      do element = 1, nspec
         do corner = 0, NCORNERS-1
            current_node = elmnts_bis((element - 1) * NCORNERS + corner)
            if (nelements_for_each_node(current_node) >= MAX_NEIGHBORS) then
               write(*,*) 'Error: Node ', current_node, ' has more than ', MAX_NEIGHBORS, ' neighboring elements.'
               stop "Adjacency graph computation error"
            end if
            elements_for_each_node(current_node, nelements_for_each_node(current_node)) = element
            nelements_for_each_node(current_node) = nelements_for_each_node(current_node) + 1
         end do
      end do

      do current_node = 1, nglob
         do i = 0, nelements_for_each_node(current_node) - 1
            element1 = elements_for_each_node(current_node, i)
            do j = i + 1, nelements_for_each_node(current_node) - 1
               element2 = elements_for_each_node(current_node, j)
               ! get shared nodes between element1 and element2
               call get_shared_nodes(elmnts_bis, element1, element2, shared_nodes, num_shared)

               if (num_shared == 0) cycle

               if (num_shared == 1) then
                  ! Only a corner node is shared
                  call get_corner_id(element1, shared_nodes(1), corner_id1)
                  call get_corner_id(element2, shared_nodes(1), corner_id2)
                  adjacency_matrix(element1, element2) = corner_id1
                  adjacency_matrix(element2, element1) = corner_id2


               else if (num_shared == 2) then
                  ! An edge is shared
                  call get_edge_id(element1, shared_nodes(1), shared_nodes(2), edge_id1)
                  call get_edge_id(element2, shared_nodes(1), shared_nodes(2), edge_id2)
                  if (adjacency_matrix(element1, element2) == 0) then
                     adjacency_matrix(element1, element2) = edge_id1
                     adjacency_matrix(element2, element1) = edge_id2
                  else
                     ! Edge already recorded, do nothing
                  end if
               else if (num_shared == 4) then
                  ! A face is shared
                  call get_face_id(element1, shared_nodes(1), shared_nodes(2), shared_nodes(3), shared_nodes(4), face_id1)
                  call get_face_id(element2, shared_nodes(1), shared_nodes(2), shared_nodes(3), shared_nodes(4), face_id2)
                  if (adjacency_matrix(element1, element2) == 0) then
                     adjacency_matrix(element1, element2) = face_id1
                     adjacency_matrix(element2, element1) = face_id2
                  else
                     ! Face already recorded, do nothing
                  end if
               else
                  ! More than 4 nodes shared - should not happen in a valid mesh
                  write(*,*) 'Error: Wrong number of shared nodes between elements ', element1, ' and ', element2, ': ', num_shared
                  stop "Invalid mesh: wrong number of shared nodes"
               end if
            end do
         end do
      end do

   end subroutine build_adjacency_graph

   subroutine get_shared_nodes(elmnts_bis, element1, element2, shared_nodes, num_shared)
      use constants_meshfem, only: NCORNERS
      use meshfem_par, only: nspec
      implicit none
      integer, intent(in) :: elmnts_bis(0:NCORNERS*nspec-1)
      integer, intent(in) :: element1, element2
      integer, intent(out) :: shared_nodes(0:NCORNERS-1)
      integer, intent(out) :: num_shared

      integer :: i, j
      num_shared = 0
      shared_nodes(:) = -1
      do i = 0, NCORNERS-1
         do j = 0, NCORNERS-1
            if (elmnts_bis((element1-1) * NCORNERS + i) == elmnts_bis((element2-1) * NCORNERS + j)) then
               shared_nodes(num_shared) = elmnts_bis((element1-1) * NCORNERS + i)
               num_shared = num_shared + 1
            end if
         end do
      end do
   end subroutine get_shared_nodes

   subroutine get_corner_id(element, node, corner_id)
      use constants_meshfem, only: NGLLX_M, NGLLY_M, NGLLZ_M
      use meshfem_par, only: ibool
      implicit none

      integer, intent(in) :: element, node
      integer, intent(out) :: corner_id

      ! Corner IDs are 19 - 26
      ! bottom_front_left = 19,  ///< Bottom-front-left corner of the element
      ! bottom_front_right = 20, ///< Bottom-front-right corner of the element
      ! bottom_back_left = 21,   ///< Bottom-back-left corner of the element
      ! bottom_back_right = 22,  ///< Bottom-back-right corner of the element
      ! top_front_left = 23,     ///< Top-front-left corner of the element
      ! top_front_right = 24,    ///< Top-front-right corner of the element
      ! top_back_left = 25,      ///< Top-back-left corner of the element
      ! top_back_right = 26      ///< Top-back-right corner of the element

      if (ibool(1,1,1,element) == node) then
         corner_id = 19
      else if (ibool(NGLLX_M,1,1,element) == node) then
         corner_id = 20
      else if (ibool(1,NGLLY_M,1,element) == node) then
         corner_id = 21
      else if (ibool(NGLLX_M,NGLLY_M,1,element) == node) then
         corner_id = 22
      else if (ibool(1,1,NGLLZ_M,element) == node) then
         corner_id = 23
      else if (ibool(NGLLX_M,1,NGLLZ_M,element) == node) then
         corner_id = 24
      else if (ibool(1,NGLLY_M,NGLLZ_M,element) == node) then
         corner_id = 25
      else if (ibool(NGLLX_M,NGLLY_M,NGLLZ_M,element) == node) then
         corner_id = 26
      else
         write(*,*) 'Error: Node ', node, ' is not a corner of element ', element
         stop "Invalid corner node"
      end if
   end subroutine get_corner_id

   subroutine get_edge_id(element, node1, node2, edge_id)
      use constants_meshfem, only: NGLLX_M, NGLLY_M, NGLLZ_M
      use meshfem_par, only: ibool
      implicit none

      integer, intent(in) :: element, node1, node2
      integer, intent(out) :: edge_id

      ! Edge IDs are 7 - 18
      !   bottom_left = 7,         ///< Bottom-left edge of the element
      !   bottom_right = 8,        ///< Bottom-right edge of the element
      !   top_right = 9,           ///< Top-right edge of the element
      !   top_left = 10,           ///< Top-left edge of the element
      !   front_bottom = 11,       ///< Front-bottom edge of the element
      !   front_top = 12,          ///< Front-top edge of the element
      !   front_left = 13,         ///< Front-left edge of the element
      !   front_right = 14,        ///< Front-right edge of the element
      !   back_bottom = 15,        ///< Back-bottom edge of the element
      !   back_top = 16,           ///< Back-top edge of the element
      !   back_left = 17,          ///< Back-left edge of the element
      !   back_right = 18,         ///< Back-right edge of the element

      ! get nodes for each edge
      integer :: bottom_face, top_face, front_face, back_face, left_face, right_face
      integer, dimension(0:1) :: bottom_left_nodes, bottom_right_nodes, top_right_nodes, top_left_nodes, &
         front_bottom_nodes, front_top_nodes, front_left_nodes, front_right_nodes, &
         back_bottom_nodes, back_top_nodes, back_left_nodes, back_right_nodes

      bottom_face = 1
      top_face = NGLLZ_M
      front_face = 1
      back_face = NGLLY_M
      left_face = 1
      right_face = NGLLX_M

      bottom_left_nodes = [ibool(left_face, front_face, bottom_face, element), &
         ibool(left_face, back_face, bottom_face, element)]
      bottom_right_nodes = [ibool(right_face, front_face, bottom_face, element), &
         ibool(right_face, back_face, bottom_face, element)]
      top_right_nodes = [ibool(right_face, front_face, top_face, element), &
         ibool(right_face, back_face, top_face, element)]
      top_left_nodes = [ibool(left_face, front_face, top_face, element), &
         ibool(left_face, back_face, top_face, element)]
      front_bottom_nodes = [ibool(left_face, front_face, bottom_face, element), &
         ibool(right_face, front_face, bottom_face, element)]
      front_top_nodes = [ibool(left_face, front_face, top_face, element), &
         ibool(right_face, front_face, top_face, element)]
      front_left_nodes = [ibool(left_face, front_face, bottom_face, element), &
         ibool(left_face, front_face, top_face, element)]
      front_right_nodes = [ibool(right_face, front_face, bottom_face, element), &
         ibool(right_face, front_face, top_face, element)]
      back_bottom_nodes = [ibool(left_face, back_face, bottom_face, element), &
         ibool(right_face, back_face, bottom_face, element)]
      back_top_nodes = [ibool(left_face, back_face, top_face, element), &
         ibool(right_face, back_face, top_face, element)]
      back_left_nodes = [ibool(left_face, back_face, bottom_face, element), &
         ibool(left_face, back_face, top_face, element)]
      back_right_nodes = [ibool(right_face, back_face, bottom_face, element), &
         ibool(right_face, back_face, top_face, element)]

      if ((node1 == bottom_left_nodes(0) .and. node2 == bottom_left_nodes(1)) .or. &
         (node1 == bottom_left_nodes(1) .and. node2 == bottom_left_nodes(0))) then
         edge_id = 7
      else if ((node1 == bottom_right_nodes(0) .and. node2 == bottom_right_nodes(1)) .or. &
         (node1 == bottom_right_nodes(1) .and. node2 == bottom_right_nodes(0))) then
         edge_id = 8
      else if ((node1 == top_right_nodes(0) .and. node2 == top_right_nodes(1)) .or. &
         (node1 == top_right_nodes(1) .and. node2 == top_right_nodes(0))) then
         edge_id = 9
      else if ((node1 == top_left_nodes(0) .and. node2 == top_left_nodes(1)) .or. &
         (node1 == top_left_nodes(1) .and. node2 == top_left_nodes(0))) then
         edge_id = 10
      else if ((node1 == front_bottom_nodes(0) .and. node2 == front_bottom_nodes(1)) .or. &
         (node1 == front_bottom_nodes(1) .and. node2 == front_bottom_nodes(0))) then
         edge_id = 11
      else if ((node1 == front_top_nodes(0) .and. node2 == front_top_nodes(1)) .or. &
         (node1 == front_top_nodes(1) .and. node2 == front_top_nodes(0))) then
         edge_id = 12
      else if ((node1 == front_left_nodes(0) .and. node2 == front_left_nodes(1)) .or. &
         (node1 == front_left_nodes(1) .and. node2 == front_left_nodes(0))) then
         edge_id = 13
      else if ((node1 == front_right_nodes(0) .and. node2 == front_right_nodes(1)) .or. &
         (node1 == front_right_nodes(1) .and. node2 == front_right_nodes(0))) then
         edge_id = 14
      else if ((node1 == back_bottom_nodes(0) .and. node2 == back_bottom_nodes(1)) .or. &
         (node1 == back_bottom_nodes(1) .and. node2 == back_bottom_nodes(0))) then
         edge_id = 15
      else if ((node1 == back_top_nodes(0) .and. node2 == back_top_nodes(1)) .or. &
         (node1 == back_top_nodes(1) .and. node2 == back_top_nodes(0))) then
         edge_id = 16
      else if ((node1 == back_left_nodes(0) .and. node2 == back_left_nodes(1)) .or. &
         (node1 == back_left_nodes(1) .and. node2 == back_left_nodes(0))) then
         edge_id = 17
      else if ((node1 == back_right_nodes(0) .and. node2 == back_right_nodes(1)) .or. &
         (node1 == back_right_nodes(1) .and. node2 == back_right_nodes(0))) then
         edge_id = 18
      else
         write(*,*) 'Error: Nodes ', node1, ' and ', node2, ' do not form an edge of element ', element
         stop "Invalid edge nodes"
      end if

   end subroutine get_edge_id

   subroutine get_face_id(element, node1, node2, node3, node4, face_id)
      use constants_meshfem, only: NGLLX_M, NGLLY_M, NGLLZ_M
      use meshfem_par, only: ibool
      implicit none

      integer, intent(in) :: element
      integer, intent(in) :: node1, node2, node3, node4
      integer, intent(out) :: face_id

      ! Face IDs are 1 - 6
      !   bottom = 1,       ///< Bottom face of the element
      !   right = 2,        ///< Right face of the element
      !   top = 3,          ///< Top face of the element
      !   left = 4,         ///< Left face of the element
      !   front = 5,        ///< Front face of the element
      !   back = 6          ///< Back face of the element

      integer :: bottom_face, top_face, front_face, back_face, left_face, right_face
      integer, dimension(4) :: bottom_face_nodes, right_face_nodes, top_face_nodes, left_face_nodes, &
         front_face_nodes, back_face_nodes, input_nodes

      bottom_face = 1
      top_face = NGLLZ_M
      front_face = 1
      back_face = NGLLY_M
      left_face = 1
      right_face = NGLLX_M

      bottom_face_nodes = [ibool(left_face, front_face, bottom_face, element), &
         ibool(right_face, front_face, bottom_face, element), &
         ibool(right_face, back_face, bottom_face, element), &
         ibool(left_face, back_face, bottom_face, element)]
      right_face_nodes = [ibool(right_face, front_face, bottom_face, element), &
         ibool(right_face, back_face, bottom_face, element), &
         ibool(right_face, back_face, top_face, element), &
         ibool(right_face, front_face, top_face, element)]
      top_face_nodes = [ibool(left_face, front_face, top_face, element), &
         ibool(right_face, front_face, top_face, element), &
         ibool(right_face, back_face, top_face, element), &
         ibool(left_face, back_face, top_face, element)]
      left_face_nodes = [ibool(left_face, front_face, bottom_face, element), &
         ibool(left_face, back_face, bottom_face, element), &
         ibool(left_face, back_face, top_face, element), &
         ibool(left_face, front_face, top_face, element)]
      front_face_nodes = [ibool(left_face, front_face, bottom_face, element), &
         ibool(right_face, front_face, bottom_face, element), &
         ibool(right_face, front_face, top_face, element), &
         ibool(left_face, front_face, top_face, element)]
      back_face_nodes = [ibool(left_face, back_face, bottom_face, element), &
         ibool(right_face, back_face, bottom_face, element), &
         ibool(right_face, back_face, top_face, element), &
         ibool(left_face, back_face, top_face, element)]

      !! Sort nodes to ensure order does not matter
      input_nodes = [node1, node2, node3, node4]
      call sort(input_nodes)
      call sort(bottom_face_nodes)
      call sort(right_face_nodes)
      call sort(top_face_nodes)
      call sort(left_face_nodes)
      call sort(front_face_nodes)
      call sort(back_face_nodes)

      if (all(input_nodes == bottom_face_nodes)) then
         face_id = 1
      else if (all(input_nodes == right_face_nodes)) then
         face_id = 2
      else if (all(input_nodes == top_face_nodes)) then
         face_id = 3
      else if (all(input_nodes == left_face_nodes)) then
         face_id = 4
      else if (all(input_nodes == front_face_nodes)) then
         face_id = 5
      else if (all(input_nodes == back_face_nodes)) then
         face_id = 6
      else
         write(*,*) 'Error: Nodes ', node1, node2, node3, ' and ', node4, ' do not form a face of element ', element
         stop "Invalid face nodes"
      end if

   end subroutine get_face_id

   subroutine sort(arr)
      implicit none
      integer, dimension(4), intent(inout) :: arr
      integer :: i, j, temp

      do i = 1, 3
         do j = 1, 4-i
            if (arr(j) > arr(j+1)) then
               temp = arr(j)
               arr(j) = arr(j+1)
               arr(j+1) = temp
            end if
         end do
      end do
   end subroutine sort
end module adjacency_graph
