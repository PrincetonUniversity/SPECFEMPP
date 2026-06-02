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


  module decompose_mesh_filenames

! paths to every input file read by xdecompose_mesh.
! populated at runtime by read_decompose_mesh_parameters() from the Par_file.
! optional files default to '' when absent from the Par_file.

  use constants, only: MAX_STRING_LEN

  implicit none

  ! mesh geometry (required)
  character(len=MAX_STRING_LEN) :: NODES_COORDS_FILE            = ''
  character(len=MAX_STRING_LEN) :: MESH_FILE                    = ''

  ! material definitions (required)
  character(len=MAX_STRING_LEN) :: MATERIALS_FILE               = ''
  character(len=MAX_STRING_LEN) :: NUMMATERIAL_VELOCITY_FILE    = ''

  ! poroelastic materials (optional — skipped when '')
  character(len=MAX_STRING_LEN) :: NUMMATERIAL_POROELASTIC_FILE = ''

  ! absorbing boundaries (optional — skipped when '')
  character(len=MAX_STRING_LEN) :: ABSORBING_SURFACE_FILE_XMIN   = ''
  character(len=MAX_STRING_LEN) :: ABSORBING_SURFACE_FILE_XMAX   = ''
  character(len=MAX_STRING_LEN) :: ABSORBING_SURFACE_FILE_YMIN   = ''
  character(len=MAX_STRING_LEN) :: ABSORBING_SURFACE_FILE_YMAX   = ''
  character(len=MAX_STRING_LEN) :: ABSORBING_SURFACE_FILE_BOTTOM = ''
  character(len=MAX_STRING_LEN) :: FREE_OR_ABSORBING_SURFACE_FILE_ZMAX = ''

  ! C-PML (optional — skipped when '')
  character(len=MAX_STRING_LEN) :: ABSORBING_CPML_FILE          = ''

  ! Moho surface (optional — skipped when '')
  character(len=MAX_STRING_LEN) :: MOHO_SURFACE_FILE            = ''

  ! wavefield discontinuity interface (optional — skipped when '')
  character(len=MAX_STRING_LEN) :: WAVEFIELD_DISCONTINUITY_INTERFACE_FILE = ''

  end module decompose_mesh_filenames


!
!-------------------------------------------------------------------------------------------------
!


  subroutine read_decompose_mesh_parameters(par_file_path)

! reads the subset of Par_file parameters required by xdecompose_mesh

  use shared_parameters, only: &
    NGNOD, NGNOD2D, NPROC, PARTITIONING_TYPE, LTS_MODE, ATTENUATION, &
    PML_CONDITIONS, SAVE_MESH_FILES, ADIOS_FOR_DATABASES, &
    COUPLE_WITH_INJECTION_TECHNIQUE, IS_WAVEFIELD_DISCONTINUITY, LOCAL_PATH

  use decompose_mesh_filenames

  implicit none

  character(len=*), intent(in) :: par_file_path

  integer :: ier
  logical :: file_exists

  ! validate path length
  if (len_trim(par_file_path) >= MAX_STRING_LEN) then
    print *, 'Error: Par_file path too long (max ', MAX_STRING_LEN, ' chars)'
    stop 'Par_file path length exceeded'
  endif

  ! validate file existence before opening
  inquire(file=par_file_path, exist=file_exists)
  if (.not. file_exists) then
    print *, 'Error: Par_file does not exist: '//trim(par_file_path)
    stop 'Par_file not found'
  endif

  call param_open(par_file_path, len_trim(par_file_path), ier)
  if (ier /= 0) stop 'Error opening Par_file in read_decompose_mesh_parameters'

  ! simulation / mesh parameters
  call read_value_integer(NPROC,                           'NPROC',                           ier)
  if (ier /= 0) stop 'Error reading NPROC from Par_file'

  call read_value_integer(NGNOD,                           'NGNOD',                           ier)
  if (ier /= 0) stop 'Error reading NGNOD from Par_file'

  call read_value_integer(PARTITIONING_TYPE,               'PARTITIONING_TYPE',               ier)
  if (ier /= 0) stop 'Error reading PARTITIONING_TYPE from Par_file'

  call read_value_logical(LTS_MODE,                        'LTS_MODE',                        ier)
  if (ier /= 0) stop 'Error reading LTS_MODE from Par_file'

  call read_value_logical(ATTENUATION,                     'ATTENUATION',                     ier)
  if (ier /= 0) stop 'Error reading ATTENUATION from Par_file'

  call read_value_logical(PML_CONDITIONS,                  'PML_CONDITIONS',                  ier)
  if (ier /= 0) stop 'Error reading PML_CONDITIONS from Par_file'

  call read_value_logical(SAVE_MESH_FILES,                 'SAVE_MESH_FILES',                 ier)
  if (ier /= 0) stop 'Error reading SAVE_MESH_FILES from Par_file'

  call read_value_logical(ADIOS_FOR_DATABASES,             'ADIOS_FOR_DATABASES',             ier)
  if (ier /= 0) stop 'Error reading ADIOS_FOR_DATABASES from Par_file'

  call read_value_logical(COUPLE_WITH_INJECTION_TECHNIQUE, 'COUPLE_WITH_INJECTION_TECHNIQUE', ier)
  if (ier /= 0) stop 'Error reading COUPLE_WITH_INJECTION_TECHNIQUE from Par_file'

  call read_value_string (LOCAL_PATH,                      'LOCAL_PATH',                      ier)
  if (ier /= 0) stop 'Error reading LOCAL_PATH from Par_file'

  ! optional flags — silently default to .false. if absent from Par_file
  call read_value_logical(IS_WAVEFIELD_DISCONTINUITY,'IS_WAVEFIELD_DISCONTINUITY',ier); ier = 0

  ! input mesh file paths (required)
  call read_value_string(NODES_COORDS_FILE,         'NODES_COORDS_FILE',         ier)
  if (ier /= 0) stop 'Error reading NODES_COORDS_FILE from Par_file'

  call read_value_string(MESH_FILE,                 'MESH_FILE',                 ier)
  if (ier /= 0) stop 'Error reading MESH_FILE from Par_file'

  call read_value_string(MATERIALS_FILE,            'MATERIALS_FILE',            ier)
  if (ier /= 0) stop 'Error reading MATERIALS_FILE from Par_file'

  call read_value_string(NUMMATERIAL_VELOCITY_FILE, 'NUMMATERIAL_VELOCITY_FILE', ier)
  if (ier /= 0) stop 'Error reading NUMMATERIAL_VELOCITY_FILE from Par_file'

  ! input mesh file paths (optional — silently default to '' if absent)
  call read_value_string(NUMMATERIAL_POROELASTIC_FILE,        'NUMMATERIAL_POROELASTIC_FILE',        ier)
  if (ier /= 0) then
    print *, '  warning: NUMMATERIAL_POROELASTIC_FILE not found in Par_file (optional)'
    NUMMATERIAL_POROELASTIC_FILE = ''
    ier = 0
  endif

  call read_value_string(ABSORBING_SURFACE_FILE_XMIN,         'ABSORBING_SURFACE_FILE_XMIN',         ier)
  if (ier /= 0) then
    ABSORBING_SURFACE_FILE_XMIN = ''
    ier = 0
  endif

  call read_value_string(ABSORBING_SURFACE_FILE_XMAX,         'ABSORBING_SURFACE_FILE_XMAX',         ier)
  if (ier /= 0) then
    ABSORBING_SURFACE_FILE_XMAX = ''
    ier = 0
  endif

  call read_value_string(ABSORBING_SURFACE_FILE_YMIN,         'ABSORBING_SURFACE_FILE_YMIN',         ier)
  if (ier /= 0) then
    ABSORBING_SURFACE_FILE_YMIN = ''
    ier = 0
  endif

  call read_value_string(ABSORBING_SURFACE_FILE_YMAX,         'ABSORBING_SURFACE_FILE_YMAX',         ier)
  if (ier /= 0) then
    ABSORBING_SURFACE_FILE_YMAX = ''
    ier = 0
  endif

  call read_value_string(ABSORBING_SURFACE_FILE_BOTTOM,       'ABSORBING_SURFACE_FILE_BOTTOM',       ier)
  if (ier /= 0) then
    ABSORBING_SURFACE_FILE_BOTTOM = ''
    ier = 0
  endif

  call read_value_string(FREE_OR_ABSORBING_SURFACE_FILE_ZMAX, 'FREE_OR_ABSORBING_SURFACE_FILE_ZMAX', ier)
  if (ier /= 0) then
    FREE_OR_ABSORBING_SURFACE_FILE_ZMAX = ''
    ier = 0
  endif

  call read_value_string(ABSORBING_CPML_FILE,                 'ABSORBING_CPML_FILE',                 ier)
  if (ier /= 0) then
    ABSORBING_CPML_FILE = ''
    ier = 0
  endif

  call read_value_string(MOHO_SURFACE_FILE,                       'MOHO_SURFACE_FILE',                       ier)
  if (ier /= 0) then
    MOHO_SURFACE_FILE = ''
    ier = 0
  endif

  call read_value_string(WAVEFIELD_DISCONTINUITY_INTERFACE_FILE, 'WAVEFIELD_DISCONTINUITY_INTERFACE_FILE', ier)
  if (ier /= 0) then
    WAVEFIELD_DISCONTINUITY_INTERFACE_FILE = ''
    ier = 0
  endif

  call close_parameter_file()

  ! derive NGNOD2D from NGNOD
  if (NGNOD == 8) then
    NGNOD2D = 4
  else if (NGNOD == 27) then
    NGNOD2D = 9
  else
    stop 'Error: NGNOD must be 8 or 27, please check Par_file'
  endif

  ! validate partitioner choice
  if (PARTITIONING_TYPE < 1 .or. PARTITIONING_TYPE > 4) &
    stop 'Error: PARTITIONING_TYPE must be 1 (SCOTCH), 2 (METIS), 3 (PATOH), or 4 (ROWS)'

  end subroutine read_decompose_mesh_parameters
