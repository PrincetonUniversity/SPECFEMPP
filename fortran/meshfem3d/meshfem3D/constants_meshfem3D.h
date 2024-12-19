
! number of material properties + element ID in input file Mesh_Par_file
  integer, parameter :: NUMBER_OF_MATERIAL_PROPERTIES = 18

!------------------------------------------------------
!----------- do not modify anything below -------------
!------------------------------------------------------

! number of GLL points not set in the mesher, do not modify this value
! we can use 2 for NGNOD == 8 only and faster meshing; or 3 to allow also for NGNOD == 27 meshing
  integer, parameter :: NGLLX_M = 3
  integer, parameter :: NGLLY_M = NGLLX_M
  integer, parameter :: NGLLZ_M = NGLLX_M

! number of points per spectral element
  integer, parameter :: NGLLCUBE_M = NGLLX_M * NGLLY_M * NGLLZ_M

! define flag for elements
  integer, parameter :: IFLAG_ONE_LAYER_TOPOGRAPHY = 1
  integer, parameter :: IFLAG_BASEMENT_TOPO = 2

! flag for the four edges of each slice and for the bottom edge
  integer, parameter :: XI_MIN = 1
  integer, parameter :: XI_MAX = 2
  integer, parameter :: ETA_MIN = 3
  integer, parameter :: ETA_MAX = 4
  integer, parameter :: BOTTOM = 5

! type of elements for heuristic rule
  integer, parameter :: ITYPE_UNUSUAL_1  = 1
  integer, parameter :: ITYPE_UNUSUAL_1p = 2
  integer, parameter :: ITYPE_UNUSUAL_4  = 3
  integer, parameter :: ITYPE_UNUSUAL_4p = 4

! define number of spectral elements and points in basic symmetric mesh doubling superbrick
  integer, parameter :: NSPEC_DOUBLING_SUPERBRICK = 32
  integer, parameter :: NGLOB_DOUBLING_SUPERBRICK = 67
  integer, parameter :: NSPEC_SUPERBRICK_1L = 28
  integer, parameter :: NGLOB_SUPERBRICK_1L = 58
