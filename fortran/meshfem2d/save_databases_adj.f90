module save_databases_adj
contains

subroutine save_databases_adjacency_map()

  use constants, only: IMAIN,IOUT,MAX_NEIGHBORS
  use part_unstruct_par, only: nelmnts,xadj_g,adjncy_g, nspec
  use shared_parameters, only: write_adjacency_map

  implicit none

  integer :: ielem, iadj
  write(IOUT) write_adjacency_map
  if (write_adjacency_map) then


    do ielem = 0, nelmnts-1
      write(IOUT) (xadj_g(ielem+1) - xadj_g(ielem))
      write(IOUT) adjncy_g(xadj_g(ielem):xadj_g(ielem+1)-1)
    enddo

    write(IMAIN,*)
    ! write(IOUT) xadj_g
    ! write(IOUT) adjncy_g

  endif

end subroutine save_databases_adjacency_map
end module save_databases_adj
