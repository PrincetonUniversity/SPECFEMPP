module save_databases_footer
contains

subroutine save_databases_additional_items()
  use constants, only: IMAIN,IOUT
  use save_databases_adj, only: save_databases_adjacency_map

  implicit none

  integer, parameter :: completion_footercode = 0


  call save_databases_adjacency_map()

  write(IOUT) completion_footercode


end subroutine save_databases_additional_items
end module save_databases_footer
