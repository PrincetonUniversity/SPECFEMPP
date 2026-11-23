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

! Wavefield-injection sides of the box, shared by xmeshfem3D and
! xdecompose_mesh so both write the same database boundary header.
!
! Side order everywhere: 1 = xmin, 2 = xmax, 3 = ymin, 4 = ymax,
! 5 = bottom, 6 = top (the database section numbers).

  subroutine resolve_injection_sides(stacey, top_free_surface, bottom_free_surface, &
                                     stacey_as_injection, requested, sides, kind)

! Resolves which sides carry injection faces and the injection-surface kind.
! Stops if a side is selected for injection that is not an absorbing side.

  use constants, only: INJECTION_SURFACE_NONE, INJECTION_SURFACE_BOX_SIDES

  implicit none

  logical, intent(in) :: stacey, top_free_surface, bottom_free_surface
  logical, intent(in) :: stacey_as_injection
  logical, dimension(6), intent(in) :: requested
  logical, dimension(6), intent(out) :: sides
  integer, intent(out) :: kind

  ! local parameters
  logical, dimension(6) :: absorbing
  integer :: iside
  character(len=6), dimension(6), parameter :: side_names = &
    (/ 'XMIN  ', 'XMAX  ', 'YMIN  ', 'YMAX  ', 'BOTTOM', 'TOP   ' /)

  ! a side is absorbing when the reader will classify its faces as Stacey
  absorbing(1:4) = stacey
  absorbing(5) = stacey .and. .not. bottom_free_surface
  absorbing(6) = stacey .and. .not. top_free_surface

  if (stacey_as_injection) then
    if (.not. stacey) &
      call exit_MPI_without_rank('Error: STACEY_AS_INJECTION_BOUNDARY requires STACEY_ABSORBING_CONDITIONS')
    sides(:) = absorbing(:)
  else
    do iside = 1, 6
      if (requested(iside) .and. .not. absorbing(iside)) then
        call exit_MPI_without_rank('Error: INJECTION_'//trim(side_names(iside))// &
                                   ' is set but that side is not a Stacey absorbing side')
      endif
    enddo
    sides(:) = requested(:)
  endif

  if (any(sides)) then
    kind = INJECTION_SURFACE_BOX_SIDES
  else
    kind = INJECTION_SURFACE_NONE
  endif

  end subroutine resolve_injection_sides

!
!-------------------------------------------------------------------------------------------------
!

  subroutine write_boundary_header(iunit, top_free_surface, bottom_free_surface, sides, kind)

! Writes the database boundary header read by the SPECFEM++ boundary reader:
!   (BOUNDARY_HEADER_MAGIC, BOUNDARY_HEADER_VERSION)
!   (TOP_FREE_SURFACE, BOTTOM_FREE_SURFACE)
!   (kind, sides(1:6))

  use constants, only: BOUNDARY_HEADER_MAGIC, BOUNDARY_HEADER_VERSION

  implicit none

  integer, intent(in) :: iunit
  logical, intent(in) :: top_free_surface, bottom_free_surface
  logical, dimension(6), intent(in) :: sides
  integer, intent(in) :: kind

  write(iunit) BOUNDARY_HEADER_MAGIC, BOUNDARY_HEADER_VERSION
  write(iunit) top_free_surface, bottom_free_surface
  write(iunit) kind, sides(1), sides(2), sides(3), sides(4), sides(5), sides(6)

  end subroutine write_boundary_header
