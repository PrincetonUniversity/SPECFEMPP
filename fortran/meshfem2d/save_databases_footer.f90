module save_databases_footer
contains

subroutine save_databases_additional_items()
  use constants, only: IMAIN,IOUT

  use shared_parameters, only: write_adjacency_map
  use save_databases_adj, only: save_databases_adjacency_map

  implicit none

  integer, parameter :: completion_footercode = 0

  if(write_adjacency_map) then
  call save_databases_adjacency_map()
  endif

  write(IOUT) completion_footercode


end subroutine save_databases_additional_items
end module save_databases_footer
