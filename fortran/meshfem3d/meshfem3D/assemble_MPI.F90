module assemble_MPI_par
  !! Parameters and shared state for MPI assembly of the mesh across
  !! processor slices.
  !!
  !! This module defines:
  !!   - Direction constants (W, E, S, N, NW, NE, SE, SW) that index
  !!     the 8 possible neighbor interfaces on a 2-D processor grid.
  !!   - MPI message tags for non-blocking exchange of element counts
  !!     (tag_send_*), interface definition buffers (tag_interface_*),
  !!     and face-corner coordinate buffers (tag_coordinates_*).
  !!   - Allocatable communication buffers (interface_buffer_*_send/recv,
  !!     coordinate_buffer_*_send/recv) shared between assemble_MPI
  !!     and downstream routines that consume the exchanged data.
  !!   - Interface metadata (nb_interfaces, interfaces, nspec_send,
  !!     nspec_recv) describing which directions are active and how
  !!     many elements each carries.
  !!
  !! Tags are chosen so that no two concurrent sends from the same
  !! rank share a tag value, preventing MPI message-matching ambiguity.

  implicit none

  ! Direction constants for MPI interface labeling
  integer, parameter :: W = 1, E = 2, S = 3, N = 4
  integer, parameter :: NW = 5, NE = 6, SE = 7, SW = 8

  ! MPI message tags for nspec count exchange
  integer, parameter :: tag_send_W = 1, tag_send_E = 2
  integer, parameter :: tag_send_S = 3, tag_send_N = 4
  integer, parameter :: tag_send_NW = 5, tag_send_NE = 6
  integer, parameter :: tag_send_SE = 7, tag_send_SW = 8

  ! MPI message tags for interface buffer exchange
  integer, parameter :: tag_interface_W = 11, tag_interface_E = 12
  integer, parameter :: tag_interface_S = 13, tag_interface_N = 14
  integer, parameter :: tag_interface_NW = 15, tag_interface_NE = 16
  integer, parameter :: tag_interface_SE = 17, tag_interface_SW = 18

  ! MPI message tags for coordinate buffer exchange
  integer, parameter :: tag_coordinates_W = 21, tag_coordinates_E = 22
  integer, parameter :: tag_coordinates_S = 23, tag_coordinates_N = 24
  integer, parameter :: tag_coordinates_NW = 25, tag_coordinates_NE = 26
  integer, parameter :: tag_coordinates_SE = 27, tag_coordinates_SW = 28

  ! Communication buffers for interface definitions (element index, interface type)
  integer, allocatable :: interface_buffer_W_send(:,:), interface_buffer_E_send(:,:)
  integer, allocatable :: interface_buffer_S_send(:,:), interface_buffer_N_send(:,:)
  integer, allocatable :: interface_buffer_NW_send(:,:), interface_buffer_NE_send(:,:)
  integer, allocatable :: interface_buffer_SE_send(:,:), interface_buffer_SW_send(:,:)
  integer, allocatable :: interface_buffer_W_recv(:,:), interface_buffer_E_recv(:,:)
  integer, allocatable :: interface_buffer_S_recv(:,:), interface_buffer_N_recv(:,:)
  integer, allocatable :: interface_buffer_NW_recv(:,:), interface_buffer_NE_recv(:,:)
  integer, allocatable :: interface_buffer_SE_recv(:,:), interface_buffer_SW_recv(:,:)

  ! Communication buffers for face-corner coordinates
  double precision, allocatable :: coordinate_buffer_W_send(:,:,:), coordinate_buffer_E_send(:,:,:)
  double precision, allocatable :: coordinate_buffer_S_send(:,:,:), coordinate_buffer_N_send(:,:,:)
  double precision, allocatable :: coordinate_buffer_NW_send(:,:,:), coordinate_buffer_NE_send(:,:,:)
  double precision, allocatable :: coordinate_buffer_SE_send(:,:,:), coordinate_buffer_SW_send(:,:,:)
  double precision, allocatable :: coordinate_buffer_W_recv(:,:,:), coordinate_buffer_E_recv(:,:,:)
  double precision, allocatable :: coordinate_buffer_S_recv(:,:,:), coordinate_buffer_N_recv(:,:,:)
  double precision, allocatable :: coordinate_buffer_NW_recv(:,:,:), coordinate_buffer_NE_recv(:,:,:)
  double precision, allocatable :: coordinate_buffer_SE_recv(:,:,:), coordinate_buffer_SW_recv(:,:,:)

  ! Communication partner info
  integer :: nb_interfaces
  logical :: interfaces(8)
  integer :: nspec_send(8), nspec_recv(8)

  ! MPI adjacency storage
  ! Each row: (myindex, neighbor_iproc, neighbor_index, my_connection_id, neighbor_connection_id)
  integer, allocatable :: mpi_adjacency(:,:)
  integer :: num_mpi_adjacencies = 0

end module assemble_MPI_par


subroutine assemble_MPI(nglob, iMPIcut_xi, iMPIcut_eta, nodes_coords)
  !! Set up and exchange MPI interface data between neighboring mesh
  !! slices on a 2-D processor grid (NPROC_XI x NPROC_ETA).
  !!
  !! The routine performs three stages:
  !!   1. Corner-index setup: builds ibool_corners_* arrays that map
  !!      each interface direction to the (ix, iy, iz) GLL indices of
  !!      the face/edge corner nodes.  Face interfaces (W, E, S, N)
  !!      have 4 corners; edge interfaces (NW, NE, SE, SW) have 2.
  !!   2. Partner discovery: calls get_communication_partners() to
  !!      determine which interfaces are active, count elements on
  !!      each, and exchange counts with neighbors.
  !!   3. Buffer exchange: for every active interface, fills element-
  !!      index and coordinate buffers via
  !!      prepare_interface_definition_buffers /
  !!      prepare_interface_coordinates_buffers, then posts
  !!      non-blocking sends and receives via
  !!      send_communication_buffers / receive_communication_buffers.
  !!      A final MPI_Waitall ensures all transfers complete before
  !!      the routine returns.
  !!
  !! This is a no-op when NPROC_XI == 1 and NPROC_ETA == 1.

#ifdef WITH_MPI
  use mpi
#endif
  use constants, only: NDIM, myrank
  use constants_meshfem, only: NGLLY_M, NGLLZ_M
  use meshfem_par, only: nspec, addressing, iproc_xi_current, iproc_eta_current
  use assemble_MPI_par

  implicit none

  integer, intent(in) :: nglob
  logical, intent(in) :: iMPIcut_xi(2, nspec), iMPIcut_eta(2, nspec)
  double precision, intent(in) :: nodes_coords(nglob, NDIM)

  ! ibool corner indices for each interface type
  ! Face interfaces (W,E,S,N) use all 4 corners;
  ! edge interfaces (NW,NE,SE,SW) use only rows 1-2, rows 3-4 are unused.
  integer :: ibool_corners_W(4, 3), ibool_corners_E(4, 3)
  integer :: ibool_corners_S(4, 3), ibool_corners_N(4, 3)
  integer :: ibool_corners_NW(4, 3), ibool_corners_NE(4, 3)
  integer :: ibool_corners_SE(4, 3), ibool_corners_SW(4, 3)

  ! MPI request handles and error flag
  integer :: requests(32)
  integer :: ireq, ierr

  ! Initialize request handles
#ifdef WITH_MPI
  requests(:) = MPI_REQUEST_NULL
#endif
  nb_interfaces = 0
  interfaces(:) = .false.
  nspec_send(:) = 0
  nspec_recv(:) = 0

  !! set up ibool_corners for each interface type (W, E, S, N, NW, NE, SE, SW)

  call assemble_ibool_corners(ibool_corners_W, ibool_corners_E, ibool_corners_S, &
    ibool_corners_N, ibool_corners_NW, ibool_corners_NE, &
    ibool_corners_SE, ibool_corners_SW)


  call get_communication_partners(iMPIcut_xi, iMPIcut_eta)

  !! Allocate the communication buffers based on the counts in nspec_send and
  !! nspec_recv for each interface.

  if (interfaces(W)) then
    allocate(interface_buffer_W_send(nspec_send(W), 2))
    allocate(coordinate_buffer_W_send(nspec_send(W), 4, NDIM))
    allocate(interface_buffer_W_recv(nspec_recv(W), 2))
    allocate(coordinate_buffer_W_recv(nspec_recv(W), 4, NDIM))
  endif
  if (interfaces(E)) then
    allocate(interface_buffer_E_send(nspec_send(E), 2))
    allocate(coordinate_buffer_E_send(nspec_send(E), 4, NDIM))
    allocate(interface_buffer_E_recv(nspec_recv(E), 2))
    allocate(coordinate_buffer_E_recv(nspec_recv(E), 4, NDIM))
  endif
  if (interfaces(S)) then
    allocate(interface_buffer_S_send(nspec_send(S), 2))
    allocate(coordinate_buffer_S_send(nspec_send(S), 4, NDIM))
    allocate(interface_buffer_S_recv(nspec_recv(S), 2))
    allocate(coordinate_buffer_S_recv(nspec_recv(S), 4, NDIM))
  endif
  if (interfaces(N)) then
    allocate(interface_buffer_N_send(nspec_send(N), 2))
    allocate(coordinate_buffer_N_send(nspec_send(N), 4, NDIM))
    allocate(interface_buffer_N_recv(nspec_recv(N), 2))
    allocate(coordinate_buffer_N_recv(nspec_recv(N), 4, NDIM))
  endif
  if (interfaces(NW)) then
    allocate(interface_buffer_NW_send(nspec_send(NW), 2))
    allocate(coordinate_buffer_NW_send(nspec_send(NW), 4, NDIM))
    allocate(interface_buffer_NW_recv(nspec_recv(NW), 2))
    allocate(coordinate_buffer_NW_recv(nspec_recv(NW), 4, NDIM))
  endif
  if (interfaces(NE)) then
    allocate(interface_buffer_NE_send(nspec_send(NE), 2))
    allocate(coordinate_buffer_NE_send(nspec_send(NE), 4, NDIM))
    allocate(interface_buffer_NE_recv(nspec_recv(NE), 2))
    allocate(coordinate_buffer_NE_recv(nspec_recv(NE), 4, NDIM))
  endif
  if (interfaces(SE)) then
    allocate(interface_buffer_SE_send(nspec_send(SE), 2))
    allocate(coordinate_buffer_SE_send(nspec_send(SE), 4, NDIM))
    allocate(interface_buffer_SE_recv(nspec_recv(SE), 2))
    allocate(coordinate_buffer_SE_recv(nspec_recv(SE), 4, NDIM))
  endif
  if (interfaces(SW)) then
    allocate(interface_buffer_SW_send(nspec_send(SW), 2))
    allocate(coordinate_buffer_SW_send(nspec_send(SW), 4, NDIM))
    allocate(interface_buffer_SW_recv(nspec_recv(SW), 2))
    allocate(coordinate_buffer_SW_recv(nspec_recv(SW), 4, NDIM))
  endif

  ireq = 1
  if (nb_interfaces > 0) then
    !! Send interface definitions and coordinates for all active interfaces
    if (interfaces(W)) then
      call prepare_interface_definition_buffers( &
        nspec, iMPIcut_xi(1,:), nspec_send(W), W, &
        interface_buffer_W_send)
      call prepare_interface_coordinates_buffers( &
        nspec, nglob, iMPIcut_xi(1,:), nodes_coords, nspec_send(W), W, &
        ibool_corners_W, coordinate_buffer_W_send)
      call send_communication_buffers( &
        nspec_send(W), &
        addressing(iproc_xi_current-1, iproc_eta_current), &
        interface_buffer_W_send, coordinate_buffer_W_send, &
        tag_interface_W, tag_coordinates_W, &
        requests(ireq:ireq+1))
    endif

    if (interfaces(E)) then
      ireq = ireq + 2
      call prepare_interface_definition_buffers( &
        nspec, iMPIcut_xi(2,:), nspec_send(E), E, &
        interface_buffer_E_send)
      call prepare_interface_coordinates_buffers( &
        nspec, nglob, iMPIcut_xi(2,:), nodes_coords, nspec_send(E), E, &
        ibool_corners_E, coordinate_buffer_E_send)
      call send_communication_buffers( &
        nspec_send(E), &
        addressing(iproc_xi_current+1, iproc_eta_current), &
        interface_buffer_E_send, coordinate_buffer_E_send, &
        tag_interface_E, tag_coordinates_E, &
        requests(ireq:ireq+1))
    endif

    if (interfaces(S)) then
      ireq = ireq + 2
      call prepare_interface_definition_buffers( &
        nspec, iMPIcut_eta(1,:), nspec_send(S), S, &
        interface_buffer_S_send)
      call prepare_interface_coordinates_buffers( &
        nspec, nglob, iMPIcut_eta(1,:), nodes_coords, nspec_send(S), S, &
        ibool_corners_S, coordinate_buffer_S_send)
      call send_communication_buffers( &
        nspec_send(S), &
        addressing(iproc_xi_current, iproc_eta_current-1), &
        interface_buffer_S_send, coordinate_buffer_S_send, &
        tag_interface_S, tag_coordinates_S, &
        requests(ireq:ireq+1))
    endif

    if (interfaces(N)) then
      ireq = ireq + 2
      call prepare_interface_definition_buffers( &
        nspec, iMPIcut_eta(2,:), nspec_send(N), N, &
        interface_buffer_N_send)
      call prepare_interface_coordinates_buffers( &
        nspec, nglob, iMPIcut_eta(2,:), nodes_coords, nspec_send(N), N, &
        ibool_corners_N, coordinate_buffer_N_send)
      call send_communication_buffers( &
        nspec_send(N), &
        addressing(iproc_xi_current, iproc_eta_current+1), &
        interface_buffer_N_send, coordinate_buffer_N_send, &
        tag_interface_N, tag_coordinates_N, &
        requests(ireq:ireq+1))
    endif

    if (interfaces(NW)) then
      ireq = ireq + 2
      call prepare_interface_definition_buffers( &
        nspec, (iMPIcut_xi(1,:) .and. iMPIcut_eta(2,:)), &
        nspec_send(NW), NW, interface_buffer_NW_send)
      call prepare_interface_coordinates_buffers( &
        nspec, nglob, (iMPIcut_xi(1,:) .and. iMPIcut_eta(2,:)), nodes_coords, &
        nspec_send(NW), NW, ibool_corners_NW, &
        coordinate_buffer_NW_send)
      call send_communication_buffers( &
        nspec_send(NW), &
        addressing(iproc_xi_current-1, iproc_eta_current+1), &
        interface_buffer_NW_send, coordinate_buffer_NW_send, &
        tag_interface_NW, tag_coordinates_NW, &
        requests(ireq:ireq+1))
    endif

    if (interfaces(NE)) then
      ireq = ireq + 2
      call prepare_interface_definition_buffers( &
        nspec, (iMPIcut_xi(2,:) .and. iMPIcut_eta(2,:)), &
        nspec_send(NE), NE, interface_buffer_NE_send)
      call prepare_interface_coordinates_buffers( &
        nspec, nglob, (iMPIcut_xi(2,:) .and. iMPIcut_eta(2,:)), nodes_coords, &
        nspec_send(NE), NE, ibool_corners_NE, &
        coordinate_buffer_NE_send)
      call send_communication_buffers( &
        nspec_send(NE), &
        addressing(iproc_xi_current+1, iproc_eta_current+1), &
        interface_buffer_NE_send, coordinate_buffer_NE_send, &
        tag_interface_NE, tag_coordinates_NE, &
        requests(ireq:ireq+1))
    endif

    if (interfaces(SE)) then
      ireq = ireq + 2
      call prepare_interface_definition_buffers( &
        nspec, (iMPIcut_xi(2,:) .and. iMPIcut_eta(1,:)), &
        nspec_send(SE), SE, interface_buffer_SE_send)
      call prepare_interface_coordinates_buffers( &
        nspec, nglob, (iMPIcut_xi(2,:) .and. iMPIcut_eta(1,:)), nodes_coords, &
        nspec_send(SE), SE, ibool_corners_SE, &
        coordinate_buffer_SE_send)
      call send_communication_buffers( &
        nspec_send(SE), &
        addressing(iproc_xi_current+1, iproc_eta_current-1), &
        interface_buffer_SE_send, coordinate_buffer_SE_send, &
        tag_interface_SE, tag_coordinates_SE, &
        requests(ireq:ireq+1))
    endif

    if (interfaces(SW)) then
      ireq = ireq + 2
      call prepare_interface_definition_buffers( &
        nspec, (iMPIcut_xi(1,:) .and. iMPIcut_eta(1,:)), &
        nspec_send(SW), SW, interface_buffer_SW_send)
      call prepare_interface_coordinates_buffers( &
        nspec, nglob, (iMPIcut_xi(1,:) .and. iMPIcut_eta(1,:)), nodes_coords, &
        nspec_send(SW), SW, ibool_corners_SW, &
        coordinate_buffer_SW_send)
      call send_communication_buffers( &
        nspec_send(SW), &
        addressing(iproc_xi_current-1, iproc_eta_current-1), &
        interface_buffer_SW_send, coordinate_buffer_SW_send, &
        tag_interface_SW, tag_coordinates_SW, &
        requests(ireq:ireq+1))
    endif

    !! Receives for all interfaces
    !! Don't reuse buffers since we might not have one-to-one matching
    !! of sends and receives
    if (interfaces(W)) then
      ireq = ireq + 2
      call receive_communication_buffers( &
        nspec_recv(W), &
        addressing(iproc_xi_current-1, iproc_eta_current), &
        interface_buffer_W_recv, coordinate_buffer_W_recv, &
        tag_interface_E, tag_coordinates_E, &
        requests(ireq:ireq+1))
    endif

    if (interfaces(E)) then
      ireq = ireq + 2
      call receive_communication_buffers( &
        nspec_recv(E), &
        addressing(iproc_xi_current+1, iproc_eta_current), &
        interface_buffer_E_recv, coordinate_buffer_E_recv, &
        tag_interface_W, tag_coordinates_W, &
        requests(ireq:ireq+1))
    endif

    if (interfaces(S)) then
      ireq = ireq + 2
      call receive_communication_buffers( &
        nspec_recv(S), &
        addressing(iproc_xi_current, iproc_eta_current-1), &
        interface_buffer_S_recv, coordinate_buffer_S_recv, &
        tag_interface_N, tag_coordinates_N, &
        requests(ireq:ireq+1))
    endif

    if (interfaces(N)) then
      ireq = ireq + 2
      call receive_communication_buffers( &
        nspec_recv(N), &
        addressing(iproc_xi_current, iproc_eta_current+1), &
        interface_buffer_N_recv, coordinate_buffer_N_recv, &
        tag_interface_S, tag_coordinates_S, &
        requests(ireq:ireq+1))
    endif

    if (interfaces(NW)) then
      ireq = ireq + 2
      call receive_communication_buffers( &
        nspec_recv(NW), &
        addressing(iproc_xi_current-1, iproc_eta_current+1), &
        interface_buffer_NW_recv, coordinate_buffer_NW_recv, &
        tag_interface_SE, tag_coordinates_SE, &
        requests(ireq:ireq+1))
    endif

    if (interfaces(NE)) then
      ireq = ireq + 2
      call receive_communication_buffers( &
        nspec_recv(NE), &
        addressing(iproc_xi_current+1, iproc_eta_current+1), &
        interface_buffer_NE_recv, coordinate_buffer_NE_recv, &
        tag_interface_SW, tag_coordinates_SW, &
        requests(ireq:ireq+1))
    endif

    if (interfaces(SE)) then
      ireq = ireq + 2
      call receive_communication_buffers( &
        nspec_recv(SE), &
        addressing(iproc_xi_current+1, iproc_eta_current-1), &
        interface_buffer_SE_recv, coordinate_buffer_SE_recv, &
        tag_interface_NW, tag_coordinates_NW, &
        requests(ireq:ireq+1))
    endif

    if (interfaces(SW)) then
      ireq = ireq + 2
      call receive_communication_buffers( &
        nspec_recv(SW), &
        addressing(iproc_xi_current-1, iproc_eta_current-1), &
        interface_buffer_SW_recv, coordinate_buffer_SW_recv, &
        tag_interface_NE, tag_coordinates_NE, &
        requests(ireq:ireq+1))
    endif

    !! Wait for all sends and receives to complete before returning
#ifdef WITH_MPI
    call MPI_Waitall(ireq+1, requests(1:ireq+1), &
      MPI_STATUSES_IGNORE, ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, &
      'MPI_Waitall failed in assemble_MPI')
#endif
  endif

end subroutine assemble_MPI

subroutine assemble_ibool_corners(ibool_corners_W, ibool_corners_E, ibool_corners_S, &
  ibool_corners_N, ibool_corners_NW, ibool_corners_NE, &
  ibool_corners_SE, ibool_corners_SW)
  !! Initialize the ibool corner index arrays for each interface type.
  !! This routine centralizes the hard-coded mapping of interface types
  !! to their corresponding GLL corner node indices (ix, iy, iz).
  !!
  !! Face interfaces (W,E,S,N) have 4 corners; edge interfaces (NW,NE,SE,SW)
  !! have 2 corners.  The specific index values are determined by the
  !! standard GLL node ordering within each spectral element and the
  !! definition of the processor grid layout.

  use constants_meshfem, only: NGLLY_M, NGLLZ_M
  implicit none

  integer, intent(inout) :: ibool_corners_W(4, 3), ibool_corners_E(4, 3)
  integer, intent(inout) :: ibool_corners_S(4, 3), ibool_corners_N(4, 3)
  integer, intent(inout) :: ibool_corners_NW(4, 3), ibool_corners_NE(4, 3)
  integer, intent(inout) :: ibool_corners_SE(4, 3), ibool_corners_SW(4, 3)


  ibool_corners_W(1, :) = [1, 1, 1] ! corner 1: (xi=1, eta=1)
  ibool_corners_W(2, :) = [1, NGLLY_M, 1] ! corner 2: (xi=1, eta=NGLLY_M)
  ibool_corners_W(3, :) = [1, 1, NGLLZ_M] ! corner 3: (xi=1, eta=1, zeta=NGLLZ_M)
  ibool_corners_W(4, :) = [1, NGLLY_M, NGLLZ_M] ! corner 4: (xi=1, eta=NGLLY_M, zeta=NGLLZ_M)

  ibool_corners_E(1, :) = [NGLLY_M, 1, 1] ! corner 1: (xi=NGLLY_M, eta=1)
  ibool_corners_E(2, :) = [NGLLY_M, NGLLY_M, 1] ! corner 2: (xi=NGLLY_M, eta=NGLLY_M)
  ibool_corners_E(3, :) = [NGLLY_M, 1, NGLLZ_M] ! corner 3: (xi=NGLLY_M, eta=1, zeta=NGLLZ_M)
  ibool_corners_E(4, :) = [NGLLY_M, NGLLY_M, NGLLZ_M] ! corner 4: (xi=NGLLY_M, eta=NGLLY_M, zeta=NGLLZ_M)

  ibool_corners_S(1, :) = [1, 1, 1] ! corner 1: (xi=1, eta=1)
  ibool_corners_S(2, :) = [NGLLY_M, 1, 1] ! corner 2: (xi=NGLLY_M, eta=1)
  ibool_corners_S(3, :) = [1, 1, NGLLZ_M] ! corner 3: (xi=1, eta=1, zeta=NGLLZ_M)
  ibool_corners_S(4, :) = [NGLLY_M, 1, NGLLZ_M] ! corner 4: (xi=NGLLY_M, eta=1, zeta=NGLLZ_M)

  ibool_corners_N(1, :) = [1, NGLLY_M, 1] ! corner 1: (xi=1, eta=NGLLY_M)
  ibool_corners_N(2, :) = [NGLLY_M, NGLLY_M, 1] ! corner 2: (xi=NGLLY_M, eta=NGLLY_M)
  ibool_corners_N(3, :) = [1, NGLLY_M, NGLLZ_M] ! corner 3: (xi=1, eta=NGLLY_M, zeta=NGLLZ_M)
  ibool_corners_N(4, :) = [NGLLY_M, NGLLY_M, NGLLZ_M] ! corner 4: (xi=NGLLY_M, eta=NGLLY_M, zeta=NGLLZ_M)

  ! Edge interfaces: only rows 1-2 are meaningful, rows 3-4 initialized to zero
  ibool_corners_NW(:, :) = 0
  ibool_corners_NW(1, :) = [1, NGLLY_M, 1] ! corner: (xi=1, eta=NGLLY_M)
  ibool_corners_NW(2, :) = [1, NGLLY_M, NGLLZ_M] ! corner: (xi=1, eta=NGLLY_M, zeta=NGLLZ_M)

  ibool_corners_NE(:, :) = 0
  ibool_corners_NE(1, :) = [NGLLY_M, NGLLY_M, 1] ! corner: (xi=NGLLY_M, eta=NGLLY_M)
  ibool_corners_NE(2, :) = [NGLLY_M, NGLLY_M, NGLLZ_M] ! corner: (xi=NGLLY_M, eta=NGLLY_M, zeta=NGLLZ_M)

  ibool_corners_SE(:, :) = 0
  ibool_corners_SE(1, :) = [NGLLY_M, 1, 1] ! corner: (xi=NGLLY_M, eta=1)
  ibool_corners_SE(2, :) = [NGLLY_M, 1, NGLLZ_M] ! corner: (xi=NGLLY_M, eta=1, zeta=NGLLZ_M)

  ibool_corners_SW(:, :) = 0
  ibool_corners_SW(1, :) = [1, 1, 1] ! corner: (xi=1, eta=1)
  ibool_corners_SW(2, :) = [1, 1, NGLLZ_M] ! corner: (xi=1, eta=1, zeta=NGLLZ_M)

end subroutine assemble_ibool_corners


subroutine get_communication_partners( &
  iMPIcut_xi, iMPIcut_eta)
  !! Determine MPI communication partners and exchange element counts
  !! for each mesh slice in a 2D processor grid (NPROC_XI x NPROC_ETA).
  !!
  !! For each slice, this routine:
  !!   1. Identifies active interfaces (W, E, S, N and diagonal
  !!      NW, NE, SE, SW) based on the slice's position in the
  !!      processor grid. Boundary slices disable interfaces at
  !!      model edges; interior slices additionally enable diagonal
  !!      interfaces where two cardinal neighbors exist.
  !!   2. Counts the number of spectral elements to send across
  !!      each active interface using the MPI cut masks
  !!      (iMPIcut_xi, iMPIcut_eta).
  !!   3. Posts non-blocking sends (MPI_ISend) and receives
  !!      (MPI_IRecv) to exchange nspec counts with each neighbor,
  !!      using complementary tags so that each send is matched by
  !!      the partner's corresponding receive.
  !!   4. Calls MPI_Waitall to block until all posted communication
  !!      completes, guaranteeing nspec_recv is fully populated
  !!      before the routine returns.
  !!
  !! All MPI calls are checked for errors; on failure the run is
  !! aborted via exit_MPI with a descriptive message.
  !!
  !! On exit, nb_interfaces holds the number of active interfaces,
  !! interfaces(:) flags which directions are active, and
  !! nspec_send(:) / nspec_recv(:) contain the element counts to
  !! exchange in each direction.
  !!
  !! This routine is a no-op when NPROC_XI == 1 and NPROC_ETA == 1
  !! (single-process run).

#ifdef WITH_MPI
  use mpi
#endif
  use constants, only: myrank
  use meshfem_par, only: nspec, addressing, NPROC_XI, NPROC_ETA, &
    iproc_xi_current, iproc_eta_current
  use assemble_MPI_par, only: nb_interfaces, interfaces, nspec_send, &
    nspec_recv, W, E, S, N, NW, NE, SE, SW, &
    tag_send_W, tag_send_E, tag_send_S, tag_send_N, &
    tag_send_NW, tag_send_NE, tag_send_SE, tag_send_SW

  implicit none

  ! Arguments
  logical, intent(in) :: iMPIcut_xi(2, nspec), iMPIcut_eta(2, nspec)

  ! Local variables
  integer :: iproc_partner, ierr
  integer :: requests(16), nrequests

  if (NPROC_XI == 1 .and. NPROC_ETA == 1) then
    nb_interfaces = 0
    interfaces(:) = .false.
    nspec_send(:) = 0
    nspec_recv(:) = 0
    return
  endif

  nrequests = 0

  ! determines number of MPI interfaces for each slice
  nb_interfaces = 4
  interfaces(W:N) = .true.
  interfaces(NW:SW) = .false.

  ! slices at model boundaries
  if (iproc_xi_current == 0) then
    nb_interfaces = nb_interfaces - 1
    interfaces(W) = .false.
  endif
  if (iproc_xi_current == NPROC_XI - 1) then
    nb_interfaces = nb_interfaces - 1
    interfaces(E) = .false.
  endif
  if (iproc_eta_current == 0) then
    nb_interfaces = nb_interfaces - 1
    interfaces(S) = .false.
  endif
  if (iproc_eta_current == NPROC_ETA - 1) then
    nb_interfaces = nb_interfaces - 1
    interfaces(N) = .false.
  endif

  ! slices in middle of model
  if (interfaces(W) .and. interfaces(N)) then
    interfaces(NW) = .true.
    nb_interfaces = nb_interfaces + 1
  endif
  if (interfaces(N) .and. interfaces(E)) then
    interfaces(NE) = .true.
    nb_interfaces = nb_interfaces + 1
  endif
  if (interfaces(E) .and. interfaces(S)) then
    interfaces(SE) = .true.
    nb_interfaces = nb_interfaces + 1
  endif
  if (interfaces(W) .and. interfaces(S)) then
    interfaces(SW) = .true.
    nb_interfaces = nb_interfaces + 1
  endif

  nspec_send(:) = 0
  if (interfaces(W)) &
    nspec_send(W) = count(iMPIcut_xi(1,:))
  if (interfaces(E)) &
    nspec_send(E) = count(iMPIcut_xi(2,:))
  if (interfaces(S)) &
    nspec_send(S) = count(iMPIcut_eta(1,:))
  if (interfaces(N)) &
    nspec_send(N) = count(iMPIcut_eta(2,:))
  if (interfaces(NW)) &
    nspec_send(NW) = count(iMPIcut_xi(1,:) .and. iMPIcut_eta(2,:))
  if (interfaces(NE)) &
    nspec_send(NE) = count(iMPIcut_xi(2,:) .and. iMPIcut_eta(2,:))
  if (interfaces(SE)) &
    nspec_send(SE) = count(iMPIcut_xi(2,:) .and. iMPIcut_eta(1,:))
  if (interfaces(SW)) &
    nspec_send(SW) = count(iMPIcut_xi(1,:) .and. iMPIcut_eta(1,:))

  !! Do all the sends first
#ifdef WITH_MPI
  if (interfaces(W)) then
    iproc_partner = addressing(iproc_xi_current-1, iproc_eta_current)
    nrequests = nrequests + 1
    call MPI_ISend(nspec_send(W), 1, MPI_INTEGER, &
      iproc_partner, tag_send_W, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_ISend failed for W interface')
  endif
  if (interfaces(E)) then
    iproc_partner = addressing(iproc_xi_current+1, iproc_eta_current)
    nrequests = nrequests + 1
    call MPI_ISend(nspec_send(E), 1, MPI_INTEGER, &
      iproc_partner, tag_send_E, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_ISend failed for E interface')
  endif
  if (interfaces(S)) then
    iproc_partner = addressing(iproc_xi_current, iproc_eta_current-1)
    nrequests = nrequests + 1
    call MPI_ISend(nspec_send(S), 1, MPI_INTEGER, &
      iproc_partner, tag_send_S, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_ISend failed for S interface')
  endif
  if (interfaces(N)) then
    iproc_partner = addressing(iproc_xi_current, iproc_eta_current+1)
    nrequests = nrequests + 1
    call MPI_ISend(nspec_send(N), 1, MPI_INTEGER, &
      iproc_partner, tag_send_N, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_ISend failed for N interface')
  endif
  if (interfaces(NW)) then
    iproc_partner = addressing(iproc_xi_current-1, iproc_eta_current+1)
    nrequests = nrequests + 1
    call MPI_ISend(nspec_send(NW), 1, MPI_INTEGER, &
      iproc_partner, tag_send_NW, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_ISend failed for NW interface')
  endif
  if (interfaces(NE)) then
    iproc_partner = addressing(iproc_xi_current+1, iproc_eta_current+1)
    nrequests = nrequests + 1
    call MPI_ISend(nspec_send(NE), 1, MPI_INTEGER, &
      iproc_partner, tag_send_NE, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_ISend failed for NE interface')
  endif
  if (interfaces(SE)) then
    iproc_partner = addressing(iproc_xi_current+1, iproc_eta_current-1)
    nrequests = nrequests + 1
    call MPI_ISend(nspec_send(SE), 1, MPI_INTEGER, &
      iproc_partner, tag_send_SE, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_ISend failed for SE interface')
  endif
  if (interfaces(SW)) then
    iproc_partner = addressing(iproc_xi_current-1, iproc_eta_current-1)
    nrequests = nrequests + 1
    call MPI_ISend(nspec_send(SW), 1, MPI_INTEGER, &
      iproc_partner, tag_send_SW, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_ISend failed for SW interface')
  endif
#endif

  nspec_recv(:) = 0

  !! Do all the receives
#ifdef WITH_MPI
  if (interfaces(W)) then
    iproc_partner = addressing(iproc_xi_current-1, iproc_eta_current)
    nrequests = nrequests + 1
    call MPI_IRecv(nspec_recv(W), 1, MPI_INTEGER, &
      iproc_partner, tag_send_E, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_IRecv failed for W interface')
  endif
  if (interfaces(E)) then
    iproc_partner = addressing(iproc_xi_current+1, iproc_eta_current)
    nrequests = nrequests + 1
    call MPI_IRecv(nspec_recv(E), 1, MPI_INTEGER, &
      iproc_partner, tag_send_W, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_IRecv failed for E interface')
  endif
  if (interfaces(S)) then
    iproc_partner = addressing(iproc_xi_current, iproc_eta_current-1)
    nrequests = nrequests + 1
    call MPI_IRecv(nspec_recv(S), 1, MPI_INTEGER, &
      iproc_partner, tag_send_N, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_IRecv failed for S interface')
  endif
  if (interfaces(N)) then
    iproc_partner = addressing(iproc_xi_current, iproc_eta_current+1)
    nrequests = nrequests + 1
    call MPI_IRecv(nspec_recv(N), 1, MPI_INTEGER, &
      iproc_partner, tag_send_S, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_IRecv failed for N interface')
  endif
  if (interfaces(NW)) then
    iproc_partner = addressing(iproc_xi_current-1, iproc_eta_current+1)
    nrequests = nrequests + 1
    call MPI_IRecv(nspec_recv(NW), 1, MPI_INTEGER, &
      iproc_partner, tag_send_SE, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_IRecv failed for NW interface')
  endif
  if (interfaces(NE)) then
    iproc_partner = addressing(iproc_xi_current+1, iproc_eta_current+1)
    nrequests = nrequests + 1
    call MPI_IRecv(nspec_recv(NE), 1, MPI_INTEGER, &
      iproc_partner, tag_send_SW, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_IRecv failed for NE interface')
  endif
  if (interfaces(SE)) then
    iproc_partner = addressing(iproc_xi_current+1, iproc_eta_current-1)
    nrequests = nrequests + 1
    call MPI_IRecv(nspec_recv(SE), 1, MPI_INTEGER, &
      iproc_partner, tag_send_NW, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_IRecv failed for SE interface')
  endif
  if (interfaces(SW)) then
    iproc_partner = addressing(iproc_xi_current-1, iproc_eta_current-1)
    nrequests = nrequests + 1
    call MPI_IRecv(nspec_recv(SW), 1, MPI_INTEGER, &
      iproc_partner, tag_send_NE, MPI_COMM_WORLD, &
      requests(nrequests), ierr)
    if (ierr /= MPI_SUCCESS) &
      call exit_MPI(myrank, 'MPI_IRecv failed for SW interface')
  endif

  !! Wait for all non-blocking sends and receives to complete
  call MPI_Waitall(nrequests, requests(1:nrequests), &
    MPI_STATUSES_IGNORE, ierr)
  if (ierr /= MPI_SUCCESS) &
    call exit_MPI(myrank, &
    'MPI_Waitall failed in get_communication_partners')
#endif

end subroutine get_communication_partners


subroutine prepare_interface_definition_buffers( &
  nspec, iMPIcut, ncommunication_elements, interface_type, buffer)
  !! Build the interface definition buffer for one MPI interface.
  !!
  !! Scans all nspec spectral elements and collects those flagged by
  !! iMPIcut (the MPI boundary cut mask for this direction).  For each
  !! flagged element, stores two integers per row:
  !!   buffer(i, 1) = ispec            (local element index, 1-based)
  !!   buffer(i, 2) = interface_type    (direction constant W=1..SW=8)
  !!
  !! The companion routine prepare_interface_coordinates_buffers fills
  !! the corresponding coordinate buffer with corner node positions.
  !! Together, these two buffers are sent to the neighbor via
  !! send_communication_buffers so the neighbor can identify which of
  !! its elements share the interface.
  !!
  !! Aborts if the count of flagged elements does not equal
  !! ncommunication_elements (a programming error).
  !!
  !! No-op when NPROC_XI == 1 and NPROC_ETA == 1.

  use constants, only: myrank
  use meshfem_par, only: NPROC_XI, NPROC_ETA

  implicit none

  ! Arguments
  integer, intent(in) :: nspec
  logical, intent(in) :: iMPIcut(nspec)
  integer, intent(in) :: ncommunication_elements
  integer, intent(in) :: interface_type
  integer, intent(out) :: buffer(ncommunication_elements, 2)

  ! Local variables
  integer :: index, ispec

  if (NPROC_XI == 1 .and. NPROC_ETA == 1) return

  !! buffer contains the index of the spectral elements and the
  !! interface type for all the elements to be sent across the
  !! interface defined by iMPIcut
  index = 0
  do ispec = 1, nspec
    if (iMPIcut(ispec)) then
      index = index + 1
      buffer(index, 1) = ispec
      buffer(index, 2) = interface_type
    endif
  enddo

  !! Check that the number of elements to send matches the expected count
  if (index /= ncommunication_elements) then
    call exit_MPI(myrank, &
      'Error in prepare_interface_definition_buffers: ' // &
      'count of elements does not match ncommunication_elements')
  endif

end subroutine prepare_interface_definition_buffers


subroutine prepare_interface_coordinates_buffers( &
  nspec, nglob, iMPIcut, nodes_coord, ncommunication_elements, interface_type, &
  ibool_corners, buffer)
  !! Fill buffer with the coordinates of the 4 face-corner nodes for
  !! each spectral element marked for communication across the
  !! interface defined by iMPIcut.
  !!
  !! ibool_corners(4,3) specifies the (ix,iy,iz) ibool indices for
  !! the 4 corner nodes of the interface face. For example, for a
  !! West (xi=1) face:
  !!   ibool_corners = reshape( &
  !!      [1,1,1,1, 1,NGLLY_M,1,NGLLY_M, 1,1,NGLLZ_M,NGLLZ_M], &
  !!      [4,3])
  !!
  !! For edge interfaces (interface_type > 4), only the first 2 rows
  !! of ibool_corners are used; corners 3-4 are flagged as -1.0.
  !!
  !! On exit, buffer(1:ncommunication_elements, 1:4, 1:NDIM) contains
  !! the coordinates of the 4 corner nodes for each interface element.

  use constants, only: NDIM, myrank
  use meshfem_par, only: ibool

  implicit none

  ! Arguments
  integer, intent(in) :: nspec, ncommunication_elements
  logical, intent(in) :: iMPIcut(nspec)
  integer, intent(in) :: interface_type
  integer, intent(in) :: ibool_corners(4, 3)
  double precision, intent(out) :: buffer(ncommunication_elements, 4, NDIM)

  integer, intent(in) :: nglob
  double precision, intent(in) :: nodes_coord(nglob, 3)

  ! Local variables
  integer :: index, ispec, icorner

  index = 0
  do ispec = 1, nspec
    if (iMPIcut(ispec)) then
      index = index + 1
      if (interface_type <= 4) then
        do icorner = 1, 4
          buffer(index, icorner, :) = nodes_coord( &
            ibool(ibool_corners(icorner, 1), &
            ibool_corners(icorner, 2), &
            ibool_corners(icorner, 3), ispec), :)
        enddo
      endif

      if (interface_type > 4) then
        do icorner = 1, 2
          buffer(index, icorner, :) = nodes_coord( &
            ibool(ibool_corners(icorner, 1), &
            ibool_corners(icorner, 2), &
            ibool_corners(icorner, 3), ispec), :)
        enddo
        ! flag: edge interface with only 2 corner nodes
        buffer(index, 3, :) = -1.0
        buffer(index, 4, :) = -1.0
      endif
    endif
  enddo

  !! Check that the number of elements to send matches the expected count
  if (index /= ncommunication_elements) then
    call exit_MPI(myrank, &
      'Error in prepare_interface_coordinates_buffers: ' // &
      'count of elements does not match ncommunication_elements')
  endif

end subroutine prepare_interface_coordinates_buffers


subroutine send_communication_buffers( &
  nspec_send, iproc_partner, interface_buffer, &
  coordinate_buffer, tag_interface, tag_coordinates, request)
  !! Post non-blocking sends for the interface definition and
  !! coordinate buffers to a single MPI partner.
  !!
  !! Two MPI_ISend calls are issued:
  !!   - interface_buffer(nspec_send, 2): element indices and
  !!     interface type
  !!   - coordinate_buffer(nspec_send, 4, NDIM): face-corner
  !!     coordinates
  !!
  !! The resulting request handles are stored in request(1:2) and
  !! must be waited on by the caller (e.g. via MPI_Waitall) before
  !! the send buffers can be safely reused. Each call is checked
  !! for errors.

#ifdef WITH_MPI
  use mpi
#endif
  use constants, only: NDIM, myrank
  use meshfem_par, only: NPROC_XI, NPROC_ETA

  implicit none

  ! Arguments
  integer, intent(in) :: nspec_send
  integer, intent(in) :: iproc_partner
  integer, intent(in) :: interface_buffer(nspec_send, 2)
  double precision, intent(in) :: coordinate_buffer(nspec_send, 4, NDIM)
  integer, intent(in) :: tag_interface, tag_coordinates
  integer, intent(inout) :: request(2)

  ! Local variables
  integer :: ierr

  if (NPROC_XI == 1 .and. NPROC_ETA == 1) return

#ifdef WITH_MPI
  call MPI_ISend(interface_buffer, nspec_send * 2, MPI_INTEGER, &
    iproc_partner, tag_interface, MPI_COMM_WORLD, &
    request(1), ierr)
  if (ierr /= MPI_SUCCESS) &
    call exit_MPI(myrank, 'MPI_ISend failed for interface buffer')

  call MPI_ISend(coordinate_buffer, nspec_send * 4 * NDIM, &
    MPI_DOUBLE_PRECISION, iproc_partner, tag_coordinates, MPI_COMM_WORLD, &
    request(2), ierr)
  if (ierr /= MPI_SUCCESS) &
    call exit_MPI(myrank, 'MPI_ISend failed for coordinate buffer')
#endif

end subroutine send_communication_buffers


subroutine receive_communication_buffers( &
  nspec_recv, iproc_partner, interface_buffer, &
  coordinate_buffer, tag_interface, tag_coordinates, request)
  !! Post non-blocking receives for the interface definition and
  !! coordinate buffers from a single MPI partner.
  !!
  !! Two MPI_IRecv calls are issued:
  !!   - interface_buffer(nspec_recv, 2): element indices and
  !!     interface type
  !!   - coordinate_buffer(nspec_recv, 4, NDIM): face-corner
  !!     coordinates
  !!
  !! The resulting request handles are stored in request(1:2) and
  !! must be waited on by the caller (e.g. via MPI_Waitall) before
  !! the receive buffers can be read. Each call is checked for
  !! errors.

#ifdef WITH_MPI
  use mpi
#endif
  use constants, only: NDIM, myrank
  use meshfem_par, only: NPROC_XI, NPROC_ETA

  implicit none

  ! Arguments
  integer, intent(in) :: nspec_recv
  integer, intent(in) :: iproc_partner
  integer, intent(out) :: interface_buffer(nspec_recv, 2)
  double precision, intent(out) :: coordinate_buffer(nspec_recv, 4, NDIM)
  integer, intent(in) :: tag_interface, tag_coordinates
  integer, intent(inout) :: request(2)

  ! Local variables
  integer :: ierr

  if (NPROC_XI == 1 .and. NPROC_ETA == 1) return

#ifdef WITH_MPI
  call MPI_IRecv(interface_buffer, nspec_recv * 2, MPI_INTEGER, &
    iproc_partner, tag_interface, MPI_COMM_WORLD, &
    request(1), ierr)
  if (ierr /= MPI_SUCCESS) &
    call exit_MPI(myrank, 'MPI_IRecv failed for interface buffer')

  call MPI_IRecv(coordinate_buffer, nspec_recv * 4 * NDIM, &
    MPI_DOUBLE_PRECISION, iproc_partner, tag_coordinates, MPI_COMM_WORLD, &
    request(2), ierr)
  if (ierr /= MPI_SUCCESS) &
    call exit_MPI(myrank, 'MPI_IRecv failed for coordinate buffer')
#endif

end subroutine receive_communication_buffers


subroutine compute_mpi_adjacency()
  !! Compute cross-MPI element adjacencies by coordinate matching.
  !!
  !! After assemble_MPI() has exchanged interface and coordinate
  !! buffers with all neighbor processors, this routine matches local
  !! interface elements against the received remote elements to build
  !! the mpi_adjacency table.
  !!
  !! Algorithm (two-pass):
  !!   Pass 1 — count_interface_matches: for each active direction
  !!     (W..SW), count how many local send-elements have a coordinate
  !!     match in the received buffer. This determines the total
  !!     allocation size for mpi_adjacency.
  !!   Pass 2 — match_interface_elements: repeat the same loop but now
  !!     fill rows of mpi_adjacency(num_mpi_adjacencies, 7).
  !!
  !! On exit, mpi_adjacency is allocated and populated. Each row has:
  !!   col 1 = myindex             — local element index (1-based)
  !!   col 2 = neighbor_iproc      — MPI rank of the neighbor
  !!   col 3 = neighbor_index      — element index local to neighbor
  !!   col 4 = my_connection_id    — face(1-6) or edge(7-18) ID on
  !!                                  the local element
  !!   col 5 = neighbor_connection_id — corresponding face/edge ID on
  !!                                     the remote element
  !!   col 6 = anchor_local_corner — corner index (1-4) on local element
  !!   col 7 = anchor_remote_corner — corner index (1-4) on remote element
  !!
  !! Connection ID convention (from adjacency_graph.f90):
  !!   Faces:  bottom=1, right=2, top=3, left=4, front=5, back=6
  !!   Edges:  bottom_left=7 .. back_right=18
  !!   Corners: bottom_front_left=19 .. top_back_right=26
  !!
  !! Does nothing when num_mpi_adjacencies == 0 (single-process or
  !! all interfaces empty).

  use constants, only: NDIM, myrank
  use meshfem_par, only: addressing, iproc_xi_current, iproc_eta_current
  use assemble_MPI_par

  implicit none

  integer :: dir, count, ier
  integer :: iproc_neighbor
  integer :: my_conn_id, neighbor_conn_id, ncorners
  integer :: nsend, nrecv

  ! First pass: count the total number of MPI adjacencies
  count = 0
  do dir = W, SW
    if (.not. interfaces(dir)) cycle

    call get_interface_properties(dir, my_conn_id, neighbor_conn_id, ncorners)
    call get_neighbor_iproc(dir, iproc_neighbor)

    nsend = nspec_send(dir)
    nrecv = nspec_recv(dir)

    call count_interface_matches(dir, nsend, nrecv, ncorners, count)
  end do

  num_mpi_adjacencies = count

  if (num_mpi_adjacencies == 0) return

  allocate(mpi_adjacency(num_mpi_adjacencies, 7), stat=ier)
  if (ier /= 0) call exit_MPI(myrank, 'Error allocating mpi_adjacency')

  ! Second pass: fill mpi_adjacency
  count = 0
  do dir = W, SW
    if (.not. interfaces(dir)) cycle

    call get_interface_properties(dir, my_conn_id, neighbor_conn_id, ncorners)
    call get_neighbor_iproc(dir, iproc_neighbor)

    nsend = nspec_send(dir)
    nrecv = nspec_recv(dir)

    call match_interface_elements(dir, nsend, nrecv, ncorners, &
      iproc_neighbor, my_conn_id, neighbor_conn_id, count)
  end do

end subroutine compute_mpi_adjacency


subroutine get_neighbor_iproc(dir, iproc_neighbor)
  !! Look up the MPI rank of the neighbor in a given direction.
  !!
  !! Uses the addressing(iproc_xi, iproc_eta) array which maps 2-D
  !! processor-grid coordinates to MPI ranks.  Each direction offsets
  !! the current processor's (iproc_xi_current, iproc_eta_current) by
  !! +/-1 in the xi and/or eta dimension:
  !!
  !!   Direction  |  xi offset  |  eta offset
  !!   ---------- | ----------- | -----------
  !!   W          |     -1      |      0
  !!   E          |     +1      |      0
  !!   S          |      0      |     -1
  !!   N          |      0      |     +1
  !!   NW         |     -1      |     +1
  !!   NE         |     +1      |     +1
  !!   SE         |     +1      |     -1
  !!   SW         |     -1      |     -1
  !!
  !! The caller must ensure that the direction is active (i.e.
  !! interfaces(dir) == .true.) before calling; otherwise the
  !! addressing lookup will be out of bounds.

  use meshfem_par, only: addressing, iproc_xi_current, iproc_eta_current
  use assemble_MPI_par, only: W, E, S, N, NW, NE, SE, SW

  implicit none

  integer, intent(in) :: dir
  integer, intent(out) :: iproc_neighbor

  select case (dir)
   case (W)
    iproc_neighbor = addressing(iproc_xi_current-1, iproc_eta_current)
   case (E)
    iproc_neighbor = addressing(iproc_xi_current+1, iproc_eta_current)
   case (S)
    iproc_neighbor = addressing(iproc_xi_current, iproc_eta_current-1)
   case (N)
    iproc_neighbor = addressing(iproc_xi_current, iproc_eta_current+1)
   case (NW)
    iproc_neighbor = addressing(iproc_xi_current-1, iproc_eta_current+1)
   case (NE)
    iproc_neighbor = addressing(iproc_xi_current+1, iproc_eta_current+1)
   case (SE)
    iproc_neighbor = addressing(iproc_xi_current+1, iproc_eta_current-1)
   case (SW)
    iproc_neighbor = addressing(iproc_xi_current-1, iproc_eta_current-1)
   case default
    stop 'Invalid direction in get_neighbor_iproc'
  end select

end subroutine get_neighbor_iproc


subroutine get_interface_properties(dir, my_conn_id, neighbor_conn_id, ncorners)
  !! Map an interface direction to connection IDs and match count.
  !!
  !! For cross-MPI adjacency matching we need to know:
  !!   (a) which face/edge of the LOCAL element sits on the interface,
  !!   (b) which face/edge of the REMOTE element sits on the same
  !!       interface (mirrored), and
  !!   (c) how many corner nodes to compare (4 for faces, 2 for edges).
  !!
  !! Cardinal directions share a FACE:
  !!   W  -> my left   face (4) <-> neighbor right  face (2), 4 corners
  !!   E  -> my right  face (2) <-> neighbor left   face (4), 4 corners
  !!   S  -> my front  face (5) <-> neighbor back   face (6), 4 corners
  !!   N  -> my back   face (6) <-> neighbor front  face (5), 4 corners
  !!
  !! Diagonal directions share an EDGE:
  !!   NW -> my back_left    edge (17) <-> neighbor front_right edge (14), 2 corners
  !!   NE -> my back_right   edge (18) <-> neighbor front_left  edge (13), 2 corners
  !!   SE -> my front_right  edge (14) <-> neighbor back_left   edge (17), 2 corners
  !!   SW -> my front_left   edge (13) <-> neighbor back_right  edge (18), 2 corners
  !!
  !! The connection IDs follow the convention in adjacency_graph.f90:
  !!   Faces  1-6:  bottom, right, top, left, front, back
  !!   Edges  7-18: bottom_left, bottom_right, top_right, top_left,
  !!                front_bottom, front_top, front_left, front_right,
  !!                back_bottom, back_top, back_left, back_right

  use assemble_MPI_par, only: W, E, S, N, NW, NE, SE, SW

  implicit none

  integer, intent(in) :: dir
  integer, intent(out) :: my_conn_id, neighbor_conn_id, ncorners

  select case (dir)
   case (W)
    my_conn_id = 4        ! left face
    neighbor_conn_id = 2   ! right face
    ncorners = 4
   case (E)
    my_conn_id = 2        ! right face
    neighbor_conn_id = 4   ! left face
    ncorners = 4
   case (S)
    my_conn_id = 5        ! front face
    neighbor_conn_id = 6   ! back face
    ncorners = 4
   case (N)
    my_conn_id = 6        ! back face
    neighbor_conn_id = 5   ! front face
    ncorners = 4
   case (NW)
    my_conn_id = 17       ! back_left edge
    neighbor_conn_id = 14  ! front_right edge
    ncorners = 2
   case (NE)
    my_conn_id = 18       ! back_right edge
    neighbor_conn_id = 13  ! front_left edge
    ncorners = 2
   case (SE)
    my_conn_id = 14       ! front_right edge
    neighbor_conn_id = 17  ! back_left edge
    ncorners = 2
   case (SW)
    my_conn_id = 13       ! front_left edge
    neighbor_conn_id = 18  ! back_right edge
    ncorners = 2
   case default
    stop 'Invalid direction in get_interface_properties'
  end select

end subroutine get_interface_properties


subroutine get_direction_buffers(dir, nsend, nrecv, &
  send_coords, recv_coords, send_iface, recv_iface)
  !! Copy the communication buffers for one direction into local arrays.
  !!
  !! The module assemble_MPI_par stores 16 named buffer pairs:
  !!   coordinate_buffer_{W,E,S,N,NW,NE,SE,SW}_{send,recv}
  !!   interface_buffer_{W,E,S,N,NW,NE,SE,SW}_{send,recv}
  !!
  !! Rather than selecting the right pair with a select-case in every
  !! caller, this routine centralizes the dispatch: given a direction
  !! constant (W=1..SW=8), it copies the relevant module buffers into
  !! the caller-provided arrays.
  !!
  !! Output arrays (must be pre-allocated by the caller):
  !!   send_coords(nsend, 4, NDIM) — corner coordinates of local elts
  !!   recv_coords(nrecv, 4, NDIM) — corner coordinates from neighbor
  !!   send_iface(nsend, 2)        — (element_index, interface_type)
  !!                                  for local interface elements
  !!   recv_iface(nrecv, 2)        — same, received from neighbor
  !!
  !! Note: For edge interfaces (NW/NE/SE/SW), only corners 1-2 of
  !! send_coords/recv_coords are valid; corners 3-4 are set to -1.0
  !! by prepare_interface_coordinates_buffers.

  use constants, only: NDIM
  use assemble_MPI_par

  implicit none

  integer, intent(in) :: dir, nsend, nrecv
  double precision, intent(out) :: send_coords(nsend, 4, NDIM)
  double precision, intent(out) :: recv_coords(nrecv, 4, NDIM)
  integer, intent(out) :: send_iface(nsend, 2)
  integer, intent(out) :: recv_iface(nrecv, 2)

  select case (dir)
   case (W)
    send_coords = coordinate_buffer_W_send
    recv_coords = coordinate_buffer_W_recv
    send_iface = interface_buffer_W_send
    recv_iface = interface_buffer_W_recv
   case (E)
    send_coords = coordinate_buffer_E_send
    recv_coords = coordinate_buffer_E_recv
    send_iface = interface_buffer_E_send
    recv_iface = interface_buffer_E_recv
   case (S)
    send_coords = coordinate_buffer_S_send
    recv_coords = coordinate_buffer_S_recv
    send_iface = interface_buffer_S_send
    recv_iface = interface_buffer_S_recv
   case (N)
    send_coords = coordinate_buffer_N_send
    recv_coords = coordinate_buffer_N_recv
    send_iface = interface_buffer_N_send
    recv_iface = interface_buffer_N_recv
   case (NW)
    send_coords = coordinate_buffer_NW_send
    recv_coords = coordinate_buffer_NW_recv
    send_iface = interface_buffer_NW_send
    recv_iface = interface_buffer_NW_recv
   case (NE)
    send_coords = coordinate_buffer_NE_send
    recv_coords = coordinate_buffer_NE_recv
    send_iface = interface_buffer_NE_send
    recv_iface = interface_buffer_NE_recv
   case (SE)
    send_coords = coordinate_buffer_SE_send
    recv_coords = coordinate_buffer_SE_recv
    send_iface = interface_buffer_SE_send
    recv_iface = interface_buffer_SE_recv
   case (SW)
    send_coords = coordinate_buffer_SW_send
    recv_coords = coordinate_buffer_SW_recv
    send_iface = interface_buffer_SW_send
    recv_iface = interface_buffer_SW_recv
   case default
    stop 'Invalid direction in get_direction_buffers'
  end select

end subroutine get_direction_buffers


subroutine count_interface_matches(dir, nsend, nrecv, ncorners, count)
  !! First pass of MPI adjacency: count matching element pairs.
  !!
  !! For one interface direction, retrieves the local send-buffer
  !! coordinates and the received neighbor coordinates, then does an
  !! O(nsend * nrecv) brute-force search comparing sorted corner
  !! coordinates via corners_match().  Each local element must have
  !! exactly one match in the received buffer — if not, the code
  !! aborts with a diagnostic message.
  !!
  !! The running total 'count' is incremented by one per match.
  !! After looping over all directions, count == num_mpi_adjacencies.
  !!
  !! Arguments:
  !!   dir      — direction constant (W=1..SW=8)
  !!   nsend    — number of local elements on this interface
  !!             (= nspec_send(dir))
  !!   nrecv    — number of received elements from the neighbor
  !!             (= nspec_recv(dir))
  !!   ncorners — number of corners to compare (4 for faces, 2 for
  !!             edges; from get_interface_properties)
  !!   count    — running total, incremented in-place

  use constants, only: NDIM
  use assemble_MPI_par

  implicit none

  integer, intent(in) :: dir, nsend, nrecv, ncorners
  integer, intent(inout) :: count

  double precision, allocatable :: send_coords(:,:,:), recv_coords(:,:,:)
  integer, allocatable :: send_iface(:,:), recv_iface(:,:)
  integer :: isend, irecv
  logical :: found
  logical, external :: corners_match

  allocate(send_coords(nsend, 4, NDIM))
  allocate(recv_coords(nrecv, 4, NDIM))
  allocate(send_iface(nsend, 2))
  allocate(recv_iface(nrecv, 2))

  call get_direction_buffers(dir, nsend, nrecv, &
    send_coords, recv_coords, send_iface, recv_iface)

  do isend = 1, nsend
    found = .false.
    do irecv = 1, nrecv
      if (corners_match(send_coords(isend,:,:), &
        recv_coords(irecv,:,:), ncorners)) then
        count = count + 1
        found = .true.
        exit
      endif
    end do
    if (.not. found) then
      write(*,*) 'Error: No matching element found for local element ', &
        send_iface(isend, 1), ' in direction ', dir
      stop 'MPI adjacency error: unmatched interface element'
    endif
  end do

  deallocate(send_coords, recv_coords, send_iface, recv_iface)

end subroutine count_interface_matches


subroutine match_interface_elements(dir, nsend, nrecv, ncorners, &
  iproc_neighbor, my_conn_id, neighbor_conn_id, count)
  !! Second pass of MPI adjacency: fill mpi_adjacency rows.
  !!
  !! Identical matching logic to count_interface_matches, but this
  !! time for each matched pair (isend, irecv) it writes a row into
  !! the pre-allocated mpi_adjacency array:
  !!
  !!   mpi_adjacency(count, 1) = local element index
  !!   mpi_adjacency(count, 2) = neighbor MPI rank
  !!   mpi_adjacency(count, 3) = neighbor's local element index
  !!   mpi_adjacency(count, 4) = local connection ID (face/edge)
  !!   mpi_adjacency(count, 5) = neighbor connection ID (face/edge)
  !!   mpi_adjacency(count, 6) = anchor corner index on local element
  !!   mpi_adjacency(count, 7) = anchor corner index on remote element
  !!
  !! The local/neighbor element indices come from column 1 of the
  !! interface_buffer_{dir}_{send,recv} arrays.  Connection IDs come
  !! from get_interface_properties and are the same for every element
  !! pair within a single direction.
  !!
  !! Aborts if any local send-element has no coordinate match in the
  !! received buffer (indicates corrupt or mismatched exchange).
  !!
  !! Arguments:
  !!   dir, nsend, nrecv, ncorners — same as count_interface_matches
  !!   iproc_neighbor              — MPI rank of neighbor processor
  !!   my_conn_id                  — face/edge ID on local element
  !!   neighbor_conn_id            — face/edge ID on remote element
  !!   count                       — running row index into
  !!                                  mpi_adjacency, incremented
  !!                                  in-place

  use constants, only: NDIM
  use assemble_MPI_par

  implicit none

  integer, intent(in) :: dir, nsend, nrecv, ncorners
  integer, intent(in) :: iproc_neighbor, my_conn_id, neighbor_conn_id
  integer, intent(inout) :: count

  double precision, allocatable :: send_coords(:,:,:), recv_coords(:,:,:)
  integer, allocatable :: send_iface(:,:), recv_iface(:,:)
  integer :: isend, irecv
  integer :: my_ispec, neighbor_ispec
  integer :: anchor_local, anchor_remote
  logical :: found
  logical, external :: corners_match

  allocate(send_coords(nsend, 4, NDIM))
  allocate(recv_coords(nrecv, 4, NDIM))
  allocate(send_iface(nsend, 2))
  allocate(recv_iface(nrecv, 2))

  call get_direction_buffers(dir, nsend, nrecv, &
    send_coords, recv_coords, send_iface, recv_iface)

  do isend = 1, nsend
    found = .false.
    do irecv = 1, nrecv
      if (corners_match(send_coords(isend,:,:), &
        recv_coords(irecv,:,:), ncorners)) then
        count = count + 1
        my_ispec = send_iface(isend, 1)
        neighbor_ispec = recv_iface(irecv, 1)

        ! Compute anchor point indices for orientation
        call compute_anchor_points_for_match(send_coords(isend,:,:), &
          recv_coords(irecv,:,:), ncorners, &
          my_conn_id, neighbor_conn_id, &
          anchor_local, anchor_remote)

        ! Fill mpi_adjacency row
        mpi_adjacency(count, 1) = my_ispec
        mpi_adjacency(count, 2) = iproc_neighbor
        mpi_adjacency(count, 3) = neighbor_ispec
        mpi_adjacency(count, 4) = my_conn_id
        mpi_adjacency(count, 5) = neighbor_conn_id
        mpi_adjacency(count, 6) = anchor_local
        mpi_adjacency(count, 7) = anchor_remote
        found = .true.
        exit
      endif
    end do
    if (.not. found) then
      write(*,*) 'Error: No matching element found for local element ', &
        send_iface(isend, 1), ' in direction ', dir
      stop 'MPI adjacency error: unmatched interface element'
    endif
  end do
  if (.not. found) then
    write(*,*) 'Error: No matching element found for local element ', &
      send_iface(isend, 1), ' in direction ', dir
    stop 'MPI adjacency error: unmatched interface element'
  endif
end do

deallocate(send_coords, recv_coords, send_iface, recv_iface)

end subroutine match_interface_elements


subroutine compute_anchor_points_for_match(coords_local, coords_remote, ncorners, &
  conn_id_local, conn_id_remote, anchor_local, anchor_remote)
  !! Compute anchor point indices from matched corner coordinates.
  !!
  !! After two interface elements have been matched by coordinate sets,
  !! this routine identifies a canonical pair of corner indices that
  !! connect the two elements. The corner indices returned are
  !! element-absolute IDs (19-26) as defined in adjacency_graph.f90,
  !! not interface-relative indices.
  !!
  !! Algorithm:
  !!   1. For each corner in the LOCAL element (unsorted), find the
  !!      closest matching corner in the REMOTE element by coordinate.
  !!   2. Return the lexicographically smallest local corner index
  !!      and its matched remote counterpart as the anchor pair.
  !!   3. Map interface corner indices (1-4 or 1-2) to element-absolute
  !!      corner IDs (19-26) using the face/edge ID mappings.
  !!   4. This ensures reproducible, deterministic anchor selection
  !!      independent of corner ordering.
  !!
  !! Arguments:
  !!   coords_local(4, 3)     — unsorted corner coordinates of local
  !!                             element interface face/edge
  !!   coords_remote(4, 3)    — unsorted corner coordinates of remote
  !!                             element interface face/edge
  !!   ncorners               — number of corners to match (2 or 4)
  !!   conn_id_local          — connection ID (face 1-6 or edge 7-18)
  !!                             on local element
  !!   conn_id_remote         — connection ID on remote element
  !!   anchor_local           — output: element corner ID (19-26) on local
  !!   anchor_remote          — output: element corner ID (19-26) on remote
  !!
  !! For a face (ncorners=4), interface corners are ordered as:
  !!   1 = bottom_left, 2 = bottom_right, 3 = top_right, 4 = top_left
  !!
  !! Returns the first (lexicographically smallest by coordinate)
  !! unmatched pair, mapped to element-absolute corner IDs.

  use constants, only: NDIM, myrank

  implicit none

  double precision, intent(in) :: coords_local(4, NDIM)
  double precision, intent(in) :: coords_remote(4, NDIM)
  integer, intent(in) :: ncorners
  integer, intent(in) :: conn_id_local, conn_id_remote
  integer, intent(out) :: anchor_local, anchor_remote

  double precision, parameter :: COORD_TOL = 1.0d-10
  integer :: ic, ic_remote
  integer :: ic_local_interface, ic_remote_interface
  double precision :: tol
  logical :: matched_remote(4)
  logical :: found

  matched_remote(:) = .false.

  ! Find the lexicographically smallest unmatched local corner
  ! and its matching remote corner
  do ic = 1, ncorners
    found = .false.
    do ic_remote = 1, ncorners
      if (matched_remote(ic_remote)) cycle

      ! Check if this pair of corners match within tolerance
      tol = COORD_TOL * (1.0d0 + maxval(dabs(coords_local(ic,:))))
      if (dabs(coords_local(ic,1) - coords_remote(ic_remote,1)) <= tol .and. &
        dabs(coords_local(ic,2) - coords_remote(ic_remote,2)) <= tol .and. &
        dabs(coords_local(ic,3) - coords_remote(ic_remote,3)) <= tol) then
        ic_local_interface = ic
        ic_remote_interface = ic_remote
        matched_remote(ic_remote) = .true.

        ! Convert interface-relative corner indices to element-absolute
        ! corner IDs (19-26) using connection ID mappings
        anchor_local = get_element_corner_id(conn_id_local, &
          ic_local_interface, ncorners)
        anchor_remote = get_element_corner_id(conn_id_remote, &
          ic_remote_interface, ncorners)
        return
      endif
    end do
  end do

  ! Error: no matching corner pair found (should not reach here if corners_match succeeded)
  call exit_MPI(myrank, &
    'Error in compute_anchor_points_for_match: ' // &
    'no matching anchor point pair detected. ' // &
    'This indicates a mismatch between corners_match and ' // &
    'the anchor detection algorithm.')

end subroutine compute_anchor_points_for_match


integer function get_element_corner_id(conn_id, interface_corner_idx, ncorners)
  !! Map interface corner index to element-absolute corner ID.
  !!
  !! Given a connection ID (face 1-6 or edge 7-18) and an interface
  !! corner index (1-4 for faces, 1-2 for edges), returns the element-
  !! absolute corner ID (19-26) using the mapping defined in
  !! adjacency_graph.f90.
  !!
  !! Corner ID convention:
  !!   19: bottom_front_left        20: bottom_front_right
  !!   21: bottom_back_left         22: bottom_back_right
  !!   23: top_front_left           24: top_front_right
  !!   25: top_back_left            26: top_back_right
  !!
  !! Arguments:
  !!   conn_id            — connection ID (1-18)
  !!   interface_corner_idx — corner position on face/edge (1-4 or 1-2)
  !!   ncorners           — number of corners (4 for faces, 2 for edges)
  !!
  !! Returns:
  !!   element_corner_id (19-26), or 0 if inputs are invalid

  implicit none

  integer, intent(in) :: conn_id, interface_corner_idx, ncorners

  ! Mapping tables: [interface corner 1-4] -> element corner ID
  ! Faces (1-6): 4 corners each
  integer, parameter :: face_1_map(4) = [19, 20, 22, 21]  ! bottom
  integer, parameter :: face_2_map(4) = [20, 24, 26, 22]  ! right
  integer, parameter :: face_3_map(4) = [23, 24, 26, 25]  ! top
  integer, parameter :: face_4_map(4) = [19, 23, 25, 21]  ! left
  integer, parameter :: face_5_map(4) = [19, 20, 24, 23]  ! front
  integer, parameter :: face_6_map(4) = [21, 25, 26, 22]  ! back

  ! Edges (7-18): 2 corners each
  integer, parameter :: edge_7_map(2) = [19, 21]   ! bottom_left
  integer, parameter :: edge_8_map(2) = [23, 25]   ! top_left
  integer, parameter :: edge_9_map(2) = [19, 20]   ! front_bottom
  integer, parameter :: edge_10_map(2) = [20, 22]  ! bottom_right
  integer, parameter :: edge_11_map(2) = [24, 26]  ! top_right
  integer, parameter :: edge_12_map(2) = [21, 22]  ! back_bottom
  integer, parameter :: edge_13_map(2) = [19, 23]  ! front_left
  integer, parameter :: edge_14_map(2) = [20, 24]  ! front_right
  integer, parameter :: edge_15_map(2) = [25, 21]  ! back_left
  integer, parameter :: edge_16_map(2) = [26, 22]  ! back_right
  integer, parameter :: edge_17_map(2) = [23, 24]  ! front_top
  integer, parameter :: edge_18_map(2) = [25, 26]  ! back_top

  ! Validate input
  if (interface_corner_idx < 1 .or. interface_corner_idx > ncorners) then
    get_element_corner_id = 0
    return
  endif

  ! Map based on connection ID
  select case (conn_id)
   case (1)
    get_element_corner_id = face_1_map(interface_corner_idx)
   case (2)
    get_element_corner_id = face_2_map(interface_corner_idx)
   case (3)
    get_element_corner_id = face_3_map(interface_corner_idx)
   case (4)
    get_element_corner_id = face_4_map(interface_corner_idx)
   case (5)
    get_element_corner_id = face_5_map(interface_corner_idx)
   case (6)
    get_element_corner_id = face_6_map(interface_corner_idx)
   case (7)
    get_element_corner_id = edge_7_map(interface_corner_idx)
   case (8)
    get_element_corner_id = edge_8_map(interface_corner_idx)
   case (9)
    get_element_corner_id = edge_9_map(interface_corner_idx)
   case (10)
    get_element_corner_id = edge_10_map(interface_corner_idx)
   case (11)
    get_element_corner_id = edge_11_map(interface_corner_idx)
   case (12)
    get_element_corner_id = edge_12_map(interface_corner_idx)
   case (13)
    get_element_corner_id = edge_13_map(interface_corner_idx)
   case (14)
    get_element_corner_id = edge_14_map(interface_corner_idx)
   case (15)
    get_element_corner_id = edge_15_map(interface_corner_idx)
   case (16)
    get_element_corner_id = edge_16_map(interface_corner_idx)
   case (17)
    get_element_corner_id = edge_17_map(interface_corner_idx)
   case (18)
    get_element_corner_id = edge_18_map(interface_corner_idx)
   case default
    get_element_corner_id = 0
  end select

end function get_element_corner_id


logical function corners_match(coords_a, coords_b, ncorners)
  !! Test whether two interface elements share the same boundary.
  !!
  !! Given the corner coordinates of two elements' interface (face or
  !! edge), determines if they represent the same geometric boundary
  !! by:
  !!   1. Copying each set of corners into a local array.
  !!   2. Sorting the first 'ncorners' rows lexicographically (x,y,z)
  !!      via sort_corners(), so the comparison is order-independent.
  !!   3. Comparing each sorted corner pair within a relative
  !!      tolerance:  tol = COORD_TOL * (1 + max(|x|,|y|,|z|))
  !!      where COORD_TOL = 1.0d-10.
  !!
  !! For face interfaces (ncorners=4), all 4 rows are compared.
  !! For edge interfaces (ncorners=2), only the first 2 rows are
  !! compared; rows 3-4 contain sentinel values (-1.0) and are
  !! ignored.
  !!
  !! Returns .true. if ALL ncorners corner pairs match within
  !! tolerance, .false. otherwise.

  use constants, only: NDIM

  implicit none

  double precision, intent(in) :: coords_a(4, NDIM), coords_b(4, NDIM)
  integer, intent(in) :: ncorners

  double precision :: sorted_a(4, NDIM), sorted_b(4, NDIM)
  double precision :: tol
  integer :: ic

  double precision, parameter :: COORD_TOL = 1.0d-10

  sorted_a = coords_a
  sorted_b = coords_b

  call sort_corners(sorted_a, ncorners)
  call sort_corners(sorted_b, ncorners)

  corners_match = .true.
  do ic = 1, ncorners
    tol = COORD_TOL * (1.0d0 + maxval(dabs(sorted_a(ic,:))))
    if (dabs(sorted_a(ic,1) - sorted_b(ic,1)) > tol .or. &
      dabs(sorted_a(ic,2) - sorted_b(ic,2)) > tol .or. &
      dabs(sorted_a(ic,3) - sorted_b(ic,3)) > tol) then
      corners_match = .false.
      return
    endif
  end do

end function corners_match


subroutine sort_corners(coords, ncorners)
  !! Sort corner coordinate rows in lexicographic order for matching.
  !!
  !! Sorts the first 'ncorners' rows of coords(4, NDIM) using a
  !! simple bubble sort with lexicographic ordering (x first, then y,
  !! then z).  Rows beyond ncorners are left untouched.
  !!
  !! This is needed because the same physical face/edge can have its
  !! corner nodes listed in different orders on the sending vs.
  !! receiving processor.  Sorting both sets canonicalizes the order
  !! so that a simple element-wise comparison suffices.
  !!
  !! The internal function coord_less_than(a, b) implements the
  !! lexicographic comparison a < b with a small epsilon (1e-14) to
  !! handle floating-point noise in coordinate values.
  !!
  !! Performance: O(ncorners^2) which is fine since ncorners <= 4.

  use constants, only: NDIM

  implicit none

  double precision, intent(inout) :: coords(4, NDIM)
  integer, intent(in) :: ncorners

  double precision :: temp(NDIM)
  integer :: i, j

  do i = 1, ncorners - 1
    do j = 1, ncorners - i
      if (coord_less_than(coords(j+1,:), coords(j,:))) then
        temp = coords(j,:)
        coords(j,:) = coords(j+1,:)
        coords(j+1,:) = temp
      endif
    end do
  end do

contains

  logical function coord_less_than(a, b)
    !! Lexicographic comparison: a < b by (x, y, z).
    double precision, intent(in) :: a(NDIM), b(NDIM)
    double precision, parameter :: EPS = 1.0d-14

    coord_less_than = .false.
    if (a(1) < b(1) - EPS) then
      coord_less_than = .true.
    else if (dabs(a(1) - b(1)) <= EPS) then
      if (a(2) < b(2) - EPS) then
        coord_less_than = .true.
      else if (dabs(a(2) - b(2)) <= EPS) then
        if (a(3) < b(3) - EPS) then
          coord_less_than = .true.
        endif
      endif
    endif
  end function coord_less_than

end subroutine sort_corners
