module save_databases_footer
contains

subroutine save_databases_additional_items()
  use constants, only: IMAIN,IOUT
  use save_databases_adj, only: save_databases_adjacency_map

  implicit none

  integer, dimension(3), parameter :: version = (/ 0, 0, 1 /)
  integer, parameter :: completion_footercode = 0

  write(IOUT) version(:)
  write(IMAIN,"(A,I0,A,I0,A,I0)") "  database footer: v", version(1) , ".", version(2), ".", version(3)


  call save_databases_adjacency_map()

  write(IOUT) completion_footercode


end subroutine save_databases_additional_items
end module save_databases_footer
