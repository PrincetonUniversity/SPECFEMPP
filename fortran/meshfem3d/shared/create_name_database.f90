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

  subroutine create_name_database(prname,iproc,LOCAL_PATH,sizeprocs)

! create the name of the database for the mesher and the solver

  use constants, only: MAX_STRING_LEN

  implicit none

  integer, intent(in) :: iproc, sizeprocs

  ! name of the database file
  character(len=MAX_STRING_LEN) :: prname, LOCAL_PATH

  ! local variables
  integer :: ndigits
  character(len=32) :: proc_str, format_str

  if (sizeprocs <= 1) then
    ! single process: no proc prefix, just append trailing slash
    prname = LOCAL_PATH(1:len_trim(LOCAL_PATH)) // '/'
  else
    ! calculate number of digits needed for processor numbering
    ndigits = int(log10(real(sizeprocs-1))) + 1

    ! create dynamic format string for zero-padding
    write(format_str,'(a,i0,a,i0,a)') '(i', ndigits, '.', ndigits, ')'
    write(proc_str, format_str) iproc

    ! construct: LOCAL_PATH/proc###_
    prname = LOCAL_PATH(1:len_trim(LOCAL_PATH)) // '/proc' // trim(proc_str) // '_'
  endif

  end subroutine create_name_database
