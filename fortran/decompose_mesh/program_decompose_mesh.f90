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

program xdecompose_mesh

  use constants, only: MAX_STRING_LEN

  use decompose_mesh_par, only: nparts,outputpath_name,ADIOS_FOR_DATABASES

  use shared_parameters, only: LOCAL_PATH, NPROC

  implicit none

  integer :: i, n_args
  character(len=MAX_STRING_LEN) :: arg
  character(len=MAX_STRING_LEN) :: par_file_path

  ! user output
  print *
  print *,'**********************'
  print *,'Serial mesh decomposer'
  print *,'**********************'
  print *

  par_file_path = ''
  n_args = command_argument_count()

  if (n_args == 0) then
    call print_usage()
    stop ' Reenter command line options'
  endif

  i = 1
  do while (i <= n_args)
    call get_command_argument(i, arg)
    select case (trim(arg))
      case ('-h', '--help')
        call print_usage()
        stop
      case ('-p', '--Par_File')
        if (i == n_args) then
          print *, 'Error: missing argument for option '//trim(arg)
          stop ' Reenter command line options'
        endif
        i = i + 1
        call get_command_argument(i, par_file_path)
      case default
        print *, 'Error: unknown option '//trim(arg)
        call print_usage()
        stop ' Reenter command line options'
    end select
    i = i + 1
  enddo

  if (len_trim(par_file_path) == 0) then
    print *, 'Error: -p Par_file is required'
    call print_usage()
    stop ' Reenter command line options'
  endif

  ! reads the subset of Par_file parameters needed by xdecompose_mesh
  call read_decompose_mesh_parameters(par_file_path)

  ! number of partitions comes from NPROC in Par_file
  nparts = NPROC

  ! output path comes from LOCAL_PATH in the Par_file
  outputpath_name = LOCAL_PATH

  ! checks adios parameters
  if (ADIOS_FOR_DATABASES) then
    ! note: this decomposer runs as a single, serial process.
    !       writing out ADIOS files would require a parallel section, for each process - not clear yet how to do that...
    print *, 'Error: ADIOS_FOR_DATABASES set to .true. in Par_file'
    print *, 'ADIOS format for databases stored by xdecompose_mesh not implemented yet, please check your Par_file settings...'
    stop 'Error ADIOS_FOR_DATABASES setting; Reenter command line'
  endif

  ! reads in (CUBIT) mesh files: mesh_file,nodes_coord_file, ...
  call read_mesh_files()

  ! checks valence of nodes
  call check_valence()

  ! sets up elements for local time stepping
  call lts_setup_elements()

  ! partitions mesh (using scotch, metis, or patoh partitioners via constants.h)
  call decompose_mesh()

  call write_mesh_databases()

  ! user output
  print *
  print *,'finished successfully'
  print *

contains

  subroutine print_usage()
    print *
    print *, 'Usage: ./xdecompose_mesh -p Par_file'
    print *
    print *, '  Options:'
    print *, '    -p, --Par_File  path to Par_file (required)'
    print *, '    -h, --help      print this message and exit'
    print *
    print *, '  The Par_file must contain:'
    print *, '    NPROC                     = number of partitions'
    print *, '    LOCAL_PATH                = output directory for database files'
    print *, '    NODES_COORDS_FILE         = path to node coordinate file'
    print *, '    MESH_FILE                 = path to mesh connectivity file'
    print *, '    MATERIALS_FILE            = path to materials file'
    print *, '    NUMMATERIAL_VELOCITY_FILE = path to material velocity file'
    print *
    print *, '  Optional Par_file entries (skipped when absent):'
    print *, '    NUMMATERIAL_POROELASTIC_FILE, ABSORBING_SURFACE_FILE_XMIN/XMAX/YMIN/YMAX/BOTTOM,'
    print *, '    FREE_OR_ABSORBING_SURFACE_FILE_ZMAX, ABSORBING_CPML_FILE,'
    print *, '    MOHO_SURFACE_FILE, WAVEFIELD_DISCONTINUITY_INTERFACE_FILE'
    print *
  end subroutine print_usage

end program xdecompose_mesh
