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

! read values from parameter file, ignoring white lines and comments

  subroutine read_value_integer(value_to_read, name, ier)

  use constants, only: MAX_STRING_LEN
  implicit none

  integer :: value_to_read
  character(len=*) :: name
  character(len=MAX_STRING_LEN) :: string_read
  integer :: ier

  call param_read(string_read, len(string_read), name, len(name), ier)
  if (ier /= 0) return
  read(string_read,*,iostat=ier) value_to_read

  end subroutine read_value_integer

!--------------------

  subroutine read_value_double_precision(value_to_read, name, ier)

  use constants, only: MAX_STRING_LEN
  implicit none

  double precision :: value_to_read
  character(len=*) :: name
  character(len=MAX_STRING_LEN) :: string_read
  integer :: ier

  call param_read(string_read, len(string_read), name, len(name), ier)
  if (ier /= 0) return
  read(string_read,*,iostat=ier) value_to_read

  end subroutine read_value_double_precision

!--------------------

  subroutine read_value_logical(value_to_read, name, ier)

  use constants, only: MAX_STRING_LEN
  implicit none

  logical :: value_to_read
  character(len=*) :: name
  character(len=MAX_STRING_LEN) :: string_read
  integer :: ier

  call param_read(string_read, len(string_read), name, len(name), ier)
  if (ier /= 0) return
  read(string_read,*,iostat=ier) value_to_read

  end subroutine read_value_logical

!--------------------

  subroutine read_value_string(value_to_read, name, ier)

  use constants, only: MAX_STRING_LEN
  implicit none

  character(len=*) :: value_to_read
  character(len=*) :: name
  character(len=MAX_STRING_LEN) :: string_read
  integer :: ier

  call param_read(string_read, len(string_read), name, len(name), ier)
  if (ier /= 0) return
  value_to_read = string_read

  end subroutine read_value_string

!--------------------

  subroutine open_parameter_file_from_main_only(ier)

  use constants, only: MAX_STRING_LEN
  use shared_input_parameters, only: Par_file

  implicit none

  integer :: ier
  character(len=MAX_STRING_LEN) :: filename_main,filename_run0001
  logical :: exists_main_Par_file,exists_run0001_Par_file

  call param_open(Par_file, len(Par_file), ier)
  if (ier /= 0) then
    print *, 'Error opening Par_file: ',trim(Par_file)
    stop
  endif

  end subroutine open_parameter_file_from_main_only

!--------------------

  subroutine open_parameter_file(ier)

  use constants, only: MAX_STRING_LEN
  use shared_input_parameters, only: Par_file

  implicit none

  integer :: ier
  character(len=MAX_STRING_LEN) :: filename_main, filename_run0001

  filename_main = Par_file

  call param_open(filename_main, len(filename_main), ier)

  if (ier /= 0) then
    ! checks second option with Par_file in run0001/DATA/
    print *, 'Error opening Par_file: ',trim(filename_main)
    stop
  endif

  end subroutine open_parameter_file

!--------------------

  subroutine close_parameter_file

  call param_close()

  end subroutine close_parameter_file
