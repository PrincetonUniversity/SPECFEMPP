
module adjacency_graph
  implicit none

contains

  subroutine compute_adjacency_graph(nglob)
    use adjacency_graph_shared, only: build_adjacency_graph_csr
    use constants_meshfem, only: NCORNERS, NGLLX_M, NGLLY_M, NGLLZ_M, MAX_NEIGHBORS
    use meshfem_par, only: ibool, nspec, xadj_adj, adjncy_adj, adj_types_local, nb_adj_edges

    implicit none

    integer, intent(in) :: nglob

    integer :: ispec, ier
    integer, allocatable :: elmnts_flat(:)

    allocate(elmnts_flat(0:NCORNERS*nspec-1), stat=ier)
    if (ier /= 0) stop 'Error allocating elmnts_flat in compute_adjacency_graph'

    do ispec = 1, nspec
      elmnts_flat((ispec-1)*NCORNERS + 0) = ibool(1,       1,       1,       ispec)
      elmnts_flat((ispec-1)*NCORNERS + 1) = ibool(NGLLX_M, 1,       1,       ispec)
      elmnts_flat((ispec-1)*NCORNERS + 2) = ibool(NGLLX_M, NGLLY_M, 1,       ispec)
      elmnts_flat((ispec-1)*NCORNERS + 3) = ibool(1,       NGLLY_M, 1,       ispec)
      elmnts_flat((ispec-1)*NCORNERS + 4) = ibool(1,       1,       NGLLZ_M, ispec)
      elmnts_flat((ispec-1)*NCORNERS + 5) = ibool(NGLLX_M, 1,       NGLLZ_M, ispec)
      elmnts_flat((ispec-1)*NCORNERS + 6) = ibool(NGLLX_M, NGLLY_M, NGLLZ_M, ispec)
      elmnts_flat((ispec-1)*NCORNERS + 7) = ibool(1,       NGLLY_M, NGLLZ_M, ispec)
    end do

    call build_adjacency_graph_csr(elmnts_flat, nspec, nglob, MAX_NEIGHBORS, &
                                    xadj_adj, adjncy_adj, adj_types_local, nb_adj_edges)

    deallocate(elmnts_flat)

  end subroutine compute_adjacency_graph

end module adjacency_graph
