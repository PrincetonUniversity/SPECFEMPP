!
!=====================================================================
!
!  Thin mesh database writer for SPECFEM++
!
!  Writes one *merged* single-mesh file per MPI rank (all regions in one file), with
!  per-element region/medium/idoubling tags, so that CMB and ICB become ordinary
!  internal fluid-solid interfaces on the SPECFEM++ side.
!
!  It carries only what SPECFEM++ cannot cheaply recompute: anchor geometry,
!  per-element context, topology, boundaries, adjacency and the resolved model
!  selection. It deliberately does NOT carry Jacobians, mass matrices, ibool over the
!  GLL points, 2D boundary Jacobians/normals, tau_e, or (in ORACLE mode) any material
!  arrays -- SPECFEM++ rebuilds those itself, and gets material values by calling back
!  into the globe model routines with the MODEL_CONFIG block written here.
!
!  This writer is additive and opt-in: it runs only when SPECFEMPP_DATABASE = .true.
!  in the Par_file, and it does not modify anything save_arrays_solver.f90 produces.
!
!  ------------------------------------------------------------------------------
!  Record layout (format_version = 1)
!
!  Fortran sequential unformatted, one record per write statement. Coordinates and
!  radii are dimensionalized to SI metres on write (the mesher works in units of
!  R_PLANET internally). All node and element ids are 1-based.
!
!    -- HEADER
!    1  magic (character(len=32)), format_version (integer)
!    2  PLANET_TYPE (integer), R_PLANET (dp), RHOAV (dp)
!    3  NGNOD, NGLLX, NGLLY, NGLLZ, nregions (5 integers)
!    4  ELLIPTICITY, TOPOGRAPHY, GRAVITY, FULL_GRAVITY, ROTATION, ATTENUATION,
!       OCEANS, HAS_REFERENCE_GEOMETRY (8 logicals)
!    5  material_mode (integer: 1 = ORACLE, 2 = BAKED)
!
!    -- MODEL_CONFIG (the resolved model selection; replayed by SPECFEM++ into the
!       globe model routines, so that it never has to read the globe Par_file)
!    6  MODEL (character(len=MAX_STRING_LEN))
!    7  n_codes (integer), codes(n_codes) (integers), in this order:
!         REFERENCE_1D_MODEL, THREE_D_MODEL, THREE_D_MODEL_IC,
!         REFERENCE_CRUSTAL_MODEL, MODEL_GLL_TYPE
!    8  n_flags (integer), flags(n_flags) (logicals), in this order:
!         TRANSVERSE_ISOTROPY, CRUSTAL, ONE_CRUST, CASE_3D, ANISOTROPIC_3D_MANTLE,
!         ANISOTROPIC_INNER_CORE, MODEL_3D_MANTLE_PERTUBATIONS, HETEROGEN_3D_MANTLE,
!         ATTENUATION_3D, ATTENUATION_3D_BERKELEY, ATTENUATION_GLL,
!         HONOR_1D_SPHERICAL_MOHO, MODEL_GLL, USE_FULL_TISO_MANTLE,
!         REGIONAL_MOHO_MESH, EMC_MODEL
!
!    -- NODES (final, deformed geometry: what the Jacobian must be built from)
!    9  nnode (integer)
!   10  x(nnode), y(nnode), z(nnode) (3 dp arrays)
!
!    -- NODES_REFERENCE (written when HAS_REFERENCE_GEOMETRY; these are spherical
!       + Moho-stretched anchors captured before external/internal topography and
!       ellipticity deformation)
!   11  xref(nnode), yref(nnode), zref(nnode) (3 dp arrays)
!
!    -- ELEMENTS
!   12  nspec (integer)
!   13  region(nspec), medium_tag(nspec), property_tag(nspec), idoubling(nspec)
!       (4 integer arrays)
!   14  rmin(nspec), rmax(nspec) (2 dp arrays)
!   15  elem_in_crust(nspec) (logical array)
!   16  node_ids(NGNOD,nspec) (integer array)
!
!    -- BOUNDARY SURFACES: four blocks, in the order
!       free surface, CMB, ICB, ocean load. Each block is two records:
!         nfaces (integer)
!         ispec(nfaces), face_id(nfaces) (2 integer arrays; omitted when nfaces == 0)
!
!    -- ADJACENCY
!       nb_adj_edges (integer)
!       xadj(nspec+1), adjncy(nb_adj_edges), adj_type(nb_adj_edges) (3 integer arrays)
!       num_neighbors (integer)
!       then per neighbor: rank, n_shared (2 integers); node_ids(n_shared) (integers)
!
!    -- MATERIAL: omitted entirely when material_mode == ORACLE.
!
!  Codes used in the file (deliberately explicit, not enum ordinals of either code base):
!    region       1 = crust/mantle, 2 = outer core, 3 = inner core
!    medium_tag   1 = acoustic, 2 = elastic  (same convention as the Cartesian
!                 SPECFEM++ database domain_id)
!    property_tag 0 = isotropic, 1 = anisotropic (includes transversely isotropic)
!    face_id      follows the SPECFEM++ hexahedron face numbering: bottom = 1, top = 3
!    adj_type     1-26: face 1-6, edge 7-18, corner 19-26 (SPECFEM++ convention)
!
!=====================================================================

  module specfempp_database_par

  use constants, only: NGNOD

  implicit none

  ! magic string and version of the on-disk format
  character(len=32), parameter :: SPECFEMPP_DB_MAGIC = 'SPECFEMPP_GLOBE_DB              '
  integer, parameter :: SPECFEMPP_DB_VERSION = 1

  ! material_mode: material values are supplied by the model oracle at SPECFEM++ setup
  integer, parameter :: SPECFEMPP_MATERIAL_ORACLE = 1
  integer, parameter :: SPECFEMPP_MATERIAL_BAKED = 2

  ! SPECFEM++ hexahedron face ids
  integer, parameter :: SPECFEMPP_FACE_BOTTOM = 1
  integer, parameter :: SPECFEMPP_FACE_TOP = 3

  ! medium tags
  integer, parameter :: SPECFEMPP_MEDIUM_ACOUSTIC = 1
  integer, parameter :: SPECFEMPP_MEDIUM_ELASTIC = 2

  ! ---- accumulated per-rank mesh, filled region by region -------------------------

  ! elements kept so far (fictitious central-cube elements are excluded)
  integer :: db_nspec = 0
  ! number of regions contributing elements
  integer :: db_nregions = 0

  ! per-element context, dimension (db_nspec_max)
  integer, dimension(:), allocatable :: db_region,db_medium_tag,db_property_tag,db_idoubling
  double precision, dimension(:), allocatable :: db_rmin,db_rmax
  logical, dimension(:), allocatable :: db_elem_in_crust

  ! staged anchor coordinates, non-dimensional, laid out as (ispec-1)*NGNOD + ia.
  ! *_final is the deformed mesh; *_ref is spherical + Moho-stretched geometry.
  double precision, dimension(:), allocatable :: db_x,db_y,db_z
  double precision, dimension(:), allocatable :: db_xref,db_yref,db_zref

  ! boundary faces, in the merged element numbering
  integer :: db_n_free = 0,db_n_cmb = 0,db_n_icb = 0,db_n_ocean = 0
  integer, dimension(:), allocatable :: db_free_ispec,db_free_face
  integer, dimension(:), allocatable :: db_cmb_ispec,db_cmb_face
  integer, dimension(:), allocatable :: db_icb_ispec,db_icb_face
  integer, dimension(:), allocatable :: db_ocean_ispec,db_ocean_face

  ! MPI interfaces, accumulated as one raw entry per (region,interface) pair and
  ! merged by neighbor rank at write time. db_if_pts holds *staged* anchor indices,
  ! translated to merged node ids once the global numbering exists.
  integer :: db_n_if = 0
  integer, dimension(:), allocatable :: db_if_rank,db_if_n,db_if_off
  integer :: db_n_if_pts = 0
  integer, dimension(:), allocatable :: db_if_pts

  contains

!
!-------------------------------------------------------------------------------------------------
!

  logical function specfempp_has_reference_geometry() result(has_reference_geometry)

! returns whether the final mesh can differ from the spherical + Moho-stretched
! reference geometry. Keep this list synchronized with the geometry-changing
! branches after the reference capture in compute_element_properties().

  use constants, only: SUPPRESS_INTERNAL_TOPOGRAPHY, &
    THREE_D_MODEL_S362ANI,THREE_D_MODEL_S362WMANI, &
    THREE_D_MODEL_S362ANI_PREM,THREE_D_MODEL_S29EA, &
    THREE_D_MODEL_BKMNS_GLAD,THREE_D_MODEL_MANTLE_SH, &
    THREE_D_MODEL_SPIRAL

  use shared_parameters, only: TOPOGRAPHY,ELLIPTICITY,THREE_D_MODEL

  implicit none

  has_reference_geometry = TOPOGRAPHY .or. ELLIPTICITY
  if (has_reference_geometry .or. SUPPRESS_INTERNAL_TOPOGRAPHY) return

  select case (THREE_D_MODEL)
  case (THREE_D_MODEL_S362ANI,THREE_D_MODEL_S362WMANI, &
        THREE_D_MODEL_S362ANI_PREM,THREE_D_MODEL_S29EA, &
        THREE_D_MODEL_BKMNS_GLAD,THREE_D_MODEL_MANTLE_SH, &
        THREE_D_MODEL_SPIRAL)
    has_reference_geometry = .true.
  end select

  end function specfempp_has_reference_geometry

!
!-------------------------------------------------------------------------------------------------
!

  subroutine save_database_specfempp_accumulate(NSPEC2D_BOTTOM_REG,NSPEC2D_TOP_REG)

! accumulates the current region into the per-rank staging arrays.
!
! called from create_regions_mesh() in the second pass, at the same place as
! save_arrays_solver(), where the region's ibool/xstore/idoubling, its boundary element
! lists and its MPI interfaces are all still allocated.

  use constants, only: myrank,NGNOD, &
    IREGION_CRUST_MANTLE,IREGION_OUTER_CORE,IREGION_INNER_CORE, &
    IFLAG_IN_FICTITIOUS_CUBE

  use shared_parameters, only: NSPEC_REGIONS,ANISOTROPIC_3D_MANTLE,ANISOTROPIC_INNER_CORE

  use meshfem_par, only: nspec,nglob,iregion_code,ibool,idoubling,xstore,ystore,zstore

  use regions_mesh_par2, only: ibelm_top,ibelm_bottom,ispec_is_tiso, &
    xelm_ref_store,yelm_ref_store,zelm_ref_store, &
    rmin_store,rmax_store,elem_in_crust_store


  implicit none

  integer,intent(in) :: NSPEC2D_BOTTOM_REG,NSPEC2D_TOP_REG

  ! local parameters
  integer, dimension(NGNOD) :: anchor_iax,anchor_iay,anchor_iaz
  ! staged index of the first anchor occurrence of each region-local global point
  integer, dimension(:), allocatable :: stage_index
  ! region-local element -> merged element index (0 when the element is dropped)
  integer, dimension(:), allocatable :: ispec_map
  integer :: ispec,ispec_db,ia,i,j,k,iglob,istage,ier
  integer :: is_anisotropic

  ! only the three solid-Earth regions go into the merged mesh; the transition-to-infinite
  ! and infinite regions used by full gravity are not part of the wave-propagation mesh
  if (iregion_code /= IREGION_CRUST_MANTLE .and. &
      iregion_code /= IREGION_OUTER_CORE .and. &
      iregion_code /= IREGION_INNER_CORE) return

  if (nspec == 0) return

  ! allocates the staging arrays on first use; the final size is known up front
  call save_database_specfempp_alloc(sum(NSPEC_REGIONS(IREGION_CRUST_MANTLE:IREGION_INNER_CORE)))

  db_nregions = db_nregions + 1

  ! (i,j,k) positions of the 27 anchors within the GLL grid
  call hex_nodes_anchor_ijk(anchor_iax,anchor_iay,anchor_iaz)

  allocate(stage_index(nglob),ispec_map(nspec),stat=ier)
  if (ier /= 0) call exit_MPI(myrank,'Error allocating stage_index in save_database_specfempp')
  stage_index(:) = 0
  ispec_map(:) = 0

  ! is this region modelled with a full anisotropic tensor?
  is_anisotropic = 0
  if (iregion_code == IREGION_CRUST_MANTLE .and. ANISOTROPIC_3D_MANTLE) is_anisotropic = 1
  if (iregion_code == IREGION_INNER_CORE .and. ANISOTROPIC_INNER_CORE) is_anisotropic = 1

  ! ---- elements -------------------------------------------------------------------
  do ispec = 1,nspec

    ! skips the fictitious central-cube elements: they duplicate cube elements owned by
    ! another slice and must not enter a merged single mesh
    if (idoubling(ispec) == IFLAG_IN_FICTITIOUS_CUBE) cycle

    db_nspec = db_nspec + 1
    ispec_db = db_nspec
    ispec_map(ispec) = ispec_db

    db_region(ispec_db) = iregion_code
    if (iregion_code == IREGION_OUTER_CORE) then
      db_medium_tag(ispec_db) = SPECFEMPP_MEDIUM_ACOUSTIC
    else
      db_medium_tag(ispec_db) = SPECFEMPP_MEDIUM_ELASTIC
    endif

    ! transversely isotropic elements are anisotropic as far as SPECFEM++'s property
    ! containers are concerned, so they share the anisotropic tag
    if (is_anisotropic == 1 .or. ispec_is_tiso(ispec)) then
      db_property_tag(ispec_db) = 1
    else
      db_property_tag(ispec_db) = 0
    endif

    db_idoubling(ispec_db) = idoubling(ispec)
    db_rmin(ispec_db) = rmin_store(ispec)
    db_rmax(ispec_db) = rmax_store(ispec)
    db_elem_in_crust(ispec_db) = elem_in_crust_store(ispec)

    do ia = 1,NGNOD
      i = anchor_iax(ia)
      j = anchor_iay(ia)
      k = anchor_iaz(ia)

      istage = (ispec_db - 1) * NGNOD + ia

      ! note: coordinates stay non-dimensional while staged, because the node merge
      ! below compares them with a non-dimensional tolerance. they are scaled by
      ! R_PLANET only on write.
      db_x(istage) = xstore(i,j,k,ispec)
      db_y(istage) = ystore(i,j,k,ispec)
      db_z(istage) = zstore(i,j,k,ispec)

      if (specfempp_has_reference_geometry()) then
        db_xref(istage) = xelm_ref_store(ia,ispec)
        db_yref(istage) = yelm_ref_store(ia,ispec)
        db_zref(istage) = zelm_ref_store(ia,ispec)
      endif

      iglob = ibool(i,j,k,ispec)
      if (stage_index(iglob) == 0) stage_index(iglob) = istage
    enddo
  enddo

  ! ---- boundary surfaces ------------------------------------------------------------
  !
  ! ibelm_bottom lists the elements whose k = 1 face is on the bottom of the region,
  ! ibelm_top those whose k = NGLLZ face is on the top (see get_jacobian_boundaries).
  ! Region adjacency then names the interfaces:
  !   crust/mantle top    -> free surface (and the ocean load acts on the same faces)
  !   crust/mantle bottom -> CMB, solid side
  !   outer core top      -> CMB, fluid side
  !   outer core bottom   -> ICB, fluid side
  !   inner core top      -> ICB, solid side
  select case (iregion_code)
  case (IREGION_CRUST_MANTLE)
    call save_database_specfempp_add_faces(ibelm_top,NSPEC2D_TOP_REG,SPECFEMPP_FACE_TOP, &
                                           ispec_map,nspec,db_free_ispec,db_free_face,db_n_free)
    call save_database_specfempp_add_faces(ibelm_bottom,NSPEC2D_BOTTOM_REG,SPECFEMPP_FACE_BOTTOM, &
                                           ispec_map,nspec,db_cmb_ispec,db_cmb_face,db_n_cmb)
  case (IREGION_OUTER_CORE)
    call save_database_specfempp_add_faces(ibelm_top,NSPEC2D_TOP_REG,SPECFEMPP_FACE_TOP, &
                                           ispec_map,nspec,db_cmb_ispec,db_cmb_face,db_n_cmb)
    call save_database_specfempp_add_faces(ibelm_bottom,NSPEC2D_BOTTOM_REG,SPECFEMPP_FACE_BOTTOM, &
                                           ispec_map,nspec,db_icb_ispec,db_icb_face,db_n_icb)
  case (IREGION_INNER_CORE)
    call save_database_specfempp_add_faces(ibelm_top,NSPEC2D_TOP_REG,SPECFEMPP_FACE_TOP, &
                                           ispec_map,nspec,db_icb_ispec,db_icb_face,db_n_icb)
  end select

  ! ---- MPI interfaces ---------------------------------------------------------------
  call save_database_specfempp_add_mpi(stage_index,nglob)

  deallocate(stage_index,ispec_map)

  end subroutine save_database_specfempp_accumulate

!
!-------------------------------------------------------------------------------------------------
!

  subroutine save_database_specfempp_alloc(nspec_total)

! allocates the staging arrays; a no-op after the first call

  use constants, only: myrank,NGNOD

  implicit none

  integer,intent(in) :: nspec_total

  ! local parameters
  integer :: ier

  if (allocated(db_region)) return

  allocate(db_region(nspec_total), &
           db_medium_tag(nspec_total), &
           db_property_tag(nspec_total), &
           db_idoubling(nspec_total), &
           db_rmin(nspec_total), &
           db_rmax(nspec_total), &
           db_elem_in_crust(nspec_total),stat=ier)
  if (ier /= 0) call exit_MPI(myrank,'Error allocating element arrays in save_database_specfempp')

  allocate(db_x(NGNOD*nspec_total), &
           db_y(NGNOD*nspec_total), &
           db_z(NGNOD*nspec_total),stat=ier)
  if (ier /= 0) call exit_MPI(myrank,'Error allocating anchor arrays in save_database_specfempp')

  if (specfempp_has_reference_geometry()) then
    allocate(db_xref(NGNOD*nspec_total), &
             db_yref(NGNOD*nspec_total), &
             db_zref(NGNOD*nspec_total),stat=ier)
    if (ier /= 0) call exit_MPI(myrank,'Error allocating reference arrays in save_database_specfempp')
  endif

  db_region(:) = 0
  db_medium_tag(:) = 0
  db_property_tag(:) = 0
  db_idoubling(:) = 0
  db_rmin(:) = 0.d0
  db_rmax(:) = 0.d0
  db_elem_in_crust(:) = .false.

  db_x(:) = 0.d0; db_y(:) = 0.d0; db_z(:) = 0.d0
  if (allocated(db_xref)) then
    db_xref(:) = 0.d0; db_yref(:) = 0.d0; db_zref(:) = 0.d0
  endif

  end subroutine save_database_specfempp_alloc

!
!-------------------------------------------------------------------------------------------------
!

  subroutine save_database_specfempp_add_faces(ibelm,nfaces,face_id,ispec_map,nspec, &
                                               list_ispec,list_face,list_n)

! appends a region's boundary element list to one of the accumulated boundary blocks,
! translating region-local element indices into the merged numbering

  use constants, only: myrank

  implicit none

  integer,intent(in) :: nfaces,face_id,nspec
  integer,dimension(nfaces),intent(in) :: ibelm
  integer,dimension(nspec),intent(in) :: ispec_map
  integer,dimension(:),allocatable,intent(inout) :: list_ispec,list_face
  integer,intent(inout) :: list_n

  ! local parameters
  integer :: iface,ispec,ispec_db

  if (nfaces <= 0) return

  call save_database_specfempp_grow(list_ispec,list_n + nfaces)
  call save_database_specfempp_grow(list_face,list_n + nfaces)

  do iface = 1,nfaces
    ispec = ibelm(iface)
    if (ispec < 1 .or. ispec > nspec) cycle

    ispec_db = ispec_map(ispec)
    ! a boundary face on a dropped (fictitious) element has no counterpart in the
    ! merged mesh
    if (ispec_db == 0) cycle

    list_n = list_n + 1
    list_ispec(list_n) = ispec_db
    list_face(list_n) = face_id
  enddo

  end subroutine save_database_specfempp_add_faces

!
!-------------------------------------------------------------------------------------------------
!

  subroutine save_database_specfempp_add_mpi(stage_index,nglob_region)

! appends the current region's MPI interfaces.
!
! The globe stores interfaces as lists of shared global GLL points, already ordered
! identically on the two ranks of a pair (that is what makes assemble_MPI_* correct).
! We keep only the anchor points of that list. Whether a shared point is an anchor is a
! property of the physical point -- with NGLLX = 5 the anchors sit at element-local
! indices 1, 3 and 5, and the chunk-to-chunk interfaces are conforming face contacts --
! so both ranks classify each shared point the same way and the filtered subsequence
! stays in a common order.

  use constants, only: myrank, &
    IREGION_CRUST_MANTLE,IREGION_OUTER_CORE,IREGION_INNER_CORE

  use meshfem_par, only: iregion_code

  use MPI_crust_mantle_par, only: num_interfaces_crust_mantle,my_neighbors_crust_mantle, &
    nibool_interfaces_crust_mantle,ibool_interfaces_crust_mantle

  use MPI_outer_core_par, only: num_interfaces_outer_core,my_neighbors_outer_core, &
    nibool_interfaces_outer_core,ibool_interfaces_outer_core

  use MPI_inner_core_par, only: num_interfaces_inner_core,my_neighbors_inner_core, &
    nibool_interfaces_inner_core,ibool_interfaces_inner_core


  implicit none

  integer,intent(in) :: nglob_region
  integer,dimension(nglob_region),intent(in) :: stage_index

  ! local parameters
  integer :: num_interfaces,iinterface,ipoin,iglob,nkept,istart

  num_interfaces = 0
  select case (iregion_code)
  case (IREGION_CRUST_MANTLE)
    num_interfaces = num_interfaces_crust_mantle
  case (IREGION_OUTER_CORE)
    num_interfaces = num_interfaces_outer_core
  case (IREGION_INNER_CORE)
    num_interfaces = num_interfaces_inner_core
  end select

  if (num_interfaces <= 0) return

  do iinterface = 1,num_interfaces
    call save_database_specfempp_grow(db_if_rank,db_n_if + 1)
    call save_database_specfempp_grow(db_if_n,db_n_if + 1)
    call save_database_specfempp_grow(db_if_off,db_n_if + 1)

    istart = db_n_if_pts
    nkept = 0

    select case (iregion_code)
    case (IREGION_CRUST_MANTLE)
      call save_database_specfempp_grow(db_if_pts,db_n_if_pts + nibool_interfaces_crust_mantle(iinterface))
      do ipoin = 1,nibool_interfaces_crust_mantle(iinterface)
        iglob = ibool_interfaces_crust_mantle(ipoin,iinterface)
        if (iglob < 1 .or. iglob > nglob_region) cycle
        if (stage_index(iglob) == 0) cycle
        db_n_if_pts = db_n_if_pts + 1
        nkept = nkept + 1
        db_if_pts(db_n_if_pts) = stage_index(iglob)
      enddo
      db_if_rank(db_n_if + 1) = my_neighbors_crust_mantle(iinterface)

    case (IREGION_OUTER_CORE)
      call save_database_specfempp_grow(db_if_pts,db_n_if_pts + nibool_interfaces_outer_core(iinterface))
      do ipoin = 1,nibool_interfaces_outer_core(iinterface)
        iglob = ibool_interfaces_outer_core(ipoin,iinterface)
        if (iglob < 1 .or. iglob > nglob_region) cycle
        if (stage_index(iglob) == 0) cycle
        db_n_if_pts = db_n_if_pts + 1
        nkept = nkept + 1
        db_if_pts(db_n_if_pts) = stage_index(iglob)
      enddo
      db_if_rank(db_n_if + 1) = my_neighbors_outer_core(iinterface)

    case (IREGION_INNER_CORE)
      call save_database_specfempp_grow(db_if_pts,db_n_if_pts + nibool_interfaces_inner_core(iinterface))
      do ipoin = 1,nibool_interfaces_inner_core(iinterface)
        iglob = ibool_interfaces_inner_core(ipoin,iinterface)
        if (iglob < 1 .or. iglob > nglob_region) cycle
        if (stage_index(iglob) == 0) cycle
        db_n_if_pts = db_n_if_pts + 1
        nkept = nkept + 1
        db_if_pts(db_n_if_pts) = stage_index(iglob)
      enddo
      db_if_rank(db_n_if + 1) = my_neighbors_inner_core(iinterface)
    end select

    db_n_if = db_n_if + 1
    db_if_off(db_n_if) = istart
    db_if_n(db_n_if) = nkept
  enddo

  end subroutine save_database_specfempp_add_mpi

!
!-------------------------------------------------------------------------------------------------
!

  subroutine save_database_specfempp_grow(array,needed)

! grows an allocatable integer array to at least `needed` entries, preserving contents

  use constants, only: myrank

  implicit none

  integer,dimension(:),allocatable,intent(inout) :: array
  integer,intent(in) :: needed

  ! local parameters
  integer,dimension(:),allocatable :: tmp
  integer :: capacity,new_capacity,ier

  if (needed <= 0) return

  if (.not. allocated(array)) then
    allocate(array(max(needed,128)),stat=ier)
    if (ier /= 0) call exit_MPI(myrank,'Error allocating array in save_database_specfempp_grow')
    array(:) = 0
    return
  endif

  capacity = size(array)
  if (capacity >= needed) return

  new_capacity = capacity
  do while (new_capacity < needed)
    new_capacity = 2 * new_capacity
  enddo

  allocate(tmp(new_capacity),stat=ier)
  if (ier /= 0) call exit_MPI(myrank,'Error growing array in save_database_specfempp_grow')
  tmp(:) = 0
  tmp(1:capacity) = array(1:capacity)

  call move_alloc(tmp,array)

  end subroutine save_database_specfempp_grow

!
!-------------------------------------------------------------------------------------------------
!

  subroutine save_database_specfempp_write()

! merges the accumulated regions into a single mesh and writes the per-rank database.
!
! called once after the region loop in create_meshes().

  use constants, only: myrank,IOUT,NGNOD,NGLLX,NGLLY,NGLLZ,MAX_STRING_LEN,IMAIN, &
    NGNOD_EIGHT_CORNERS

  use shared_parameters, only: LOCAL_PATH,PLANET_TYPE,R_PLANET,RHOAV, &
    ELLIPTICITY,TOPOGRAPHY,GRAVITY,FULL_GRAVITY,ROTATION,ATTENUATION,OCEANS, &
    MODEL,REFERENCE_1D_MODEL,THREE_D_MODEL,THREE_D_MODEL_IC,REFERENCE_CRUSTAL_MODEL, &
    MODEL_GLL_TYPE,TRANSVERSE_ISOTROPY,CRUSTAL,ONE_CRUST,CASE_3D, &
    ANISOTROPIC_3D_MANTLE,ANISOTROPIC_INNER_CORE,MODEL_3D_MANTLE_PERTUBATIONS, &
    HETEROGEN_3D_MANTLE,ATTENUATION_3D,ATTENUATION_3D_BERKELEY,ATTENUATION_GLL, &
    HONOR_1D_SPHERICAL_MOHO,MODEL_GLL,USE_FULL_TISO_MANTLE,REGIONAL_MOHO_MESH,EMC_MODEL

  use adjacency_graph_shared, only: build_adjacency_graph_csr


  implicit none

  ! local parameters
  integer, parameter :: N_CODES = 5
  integer, parameter :: N_FLAGS = 16

  character(len=MAX_STRING_LEN) :: filename
  integer :: npointot,nnode,nspec_total,ier,ispec,ia,istage,inode
  integer :: nb_adj_edges,max_valence,i

  ! merged global numbering of the staged anchors
  integer, dimension(:), allocatable :: iglob_merged
  double precision, dimension(:), allocatable :: xp,yp,zp
  ! node coordinate tables in the merged numbering
  double precision, dimension(:), allocatable :: xn,yn,zn,xnr,ynr,znr
  ! per-element anchor node ids
  integer, dimension(:,:), allocatable :: node_ids
  ! adjacency
  integer, dimension(:), allocatable :: elmnts_flat,xadj_adj,adjncy_adj,adj_types
  integer, dimension(:), allocatable :: valence
  ! merged-by-rank MPI neighbors
  integer :: num_neighbors
  integer, dimension(:), allocatable :: nb_rank,nb_count,nb_offset,nb_nodes

  integer, dimension(N_CODES) :: codes
  logical, dimension(N_FLAGS) :: flags
  logical :: has_reference_geometry

  ! nothing meshed on this rank
  nspec_total = db_nspec
  has_reference_geometry = specfempp_has_reference_geometry()

  if (myrank == 0) then
    write(IMAIN,*)
    write(IMAIN,*) '  ...saving thin SPECFEM++ mesh database'
    call flush_IMAIN()
  endif

  if (nspec_total == 0) then
    call save_database_specfempp_free()
    return
  endif

  ! ---- merged node numbering ---------------------------------------------------------
  !
  ! get_global() welds coincident anchors. Nodes shared between two regions (the CMB and
  ! the ICB) are welded into a single node on purpose: it is what makes the fluid-solid
  ! adjacency edges appear below. SPECFEM++ re-splits the degrees of freedom by medium
  ! when it numbers its quadrature points, so a shared control node does not weld the
  ! acoustic and elastic fields together.
  npointot = NGNOD * nspec_total

  allocate(xp(npointot),yp(npointot),zp(npointot),iglob_merged(npointot),stat=ier)
  if (ier /= 0) call exit_MPI(myrank,'Error allocating merge arrays in save_database_specfempp_write')

  ! get_global() clobbers its coordinate arguments, so it gets scratch copies
  xp(:) = db_x(1:npointot)
  yp(:) = db_y(1:npointot)
  zp(:) = db_z(1:npointot)
  iglob_merged(:) = 0

  call get_global(npointot,xp,yp,zp,iglob_merged,nnode)

  deallocate(xp,yp,zp)

  ! node coordinate tables, taken from the first staged occurrence of each merged node
  allocate(xn(nnode),yn(nnode),zn(nnode),stat=ier)
  if (ier /= 0) call exit_MPI(myrank,'Error allocating node arrays in save_database_specfempp_write')
  xn(:) = 0.d0; yn(:) = 0.d0; zn(:) = 0.d0

  if (has_reference_geometry) then
    allocate(xnr(nnode),ynr(nnode),znr(nnode),stat=ier)
    if (ier /= 0) call exit_MPI(myrank,'Error allocating reference nodes in save_database_specfempp_write')
    xnr(:) = 0.d0; ynr(:) = 0.d0; znr(:) = 0.d0
  endif

  allocate(node_ids(NGNOD,nspec_total),stat=ier)
  if (ier /= 0) call exit_MPI(myrank,'Error allocating node_ids in save_database_specfempp_write')

  do ispec = 1,nspec_total
    do ia = 1,NGNOD
      istage = (ispec - 1) * NGNOD + ia
      inode = iglob_merged(istage)
      node_ids(ia,ispec) = inode
      xn(inode) = db_x(istage)
      yn(inode) = db_y(istage)
      zn(inode) = db_z(istage)
      if (has_reference_geometry) then
        xnr(inode) = db_xref(istage)
        ynr(inode) = db_yref(istage)
        znr(inode) = db_zref(istage)
      endif
    enddo
  enddo

  ! ---- local adjacency ---------------------------------------------------------------
  !
  ! anchors 1-8 are the element corners, in exactly the order build_adjacency_graph_csr
  ! expects: ibool(1,1,1), (NX,1,1), (NX,NY,1), (1,NY,1), (1,1,NZ), (NX,1,NZ),
  ! (NX,NY,NZ), (1,NY,NZ).
  allocate(elmnts_flat(0:NGNOD_EIGHT_CORNERS*nspec_total-1),valence(nnode),stat=ier)
  if (ier /= 0) call exit_MPI(myrank,'Error allocating adjacency input in save_database_specfempp_write')

  valence(:) = 0
  do ispec = 1,nspec_total
    do ia = 1,NGNOD_EIGHT_CORNERS
      elmnts_flat((ispec-1)*NGNOD_EIGHT_CORNERS + ia - 1) = node_ids(ia,ispec)
      valence(node_ids(ia,ispec)) = valence(node_ids(ia,ispec)) + 1
    enddo
  enddo

  ! build_adjacency_graph_csr bounds both the elements-per-node and the neighbors-per-
  ! element tables by max_valence
  max_valence = max(maxval(valence),40)
  deallocate(valence)

  call build_adjacency_graph_csr(elmnts_flat,nspec_total,nnode,max_valence, &
                                 xadj_adj,adjncy_adj,adj_types,nb_adj_edges)

  deallocate(elmnts_flat)

  ! ---- MPI neighbors ------------------------------------------------------------------
  call save_database_specfempp_merge_mpi(iglob_merged,npointot, &
                                         num_neighbors,nb_rank,nb_count,nb_offset,nb_nodes)

  call save_database_specfempp_check_mpi(num_neighbors,nb_rank,nb_count)

  deallocate(iglob_merged)

  ! ---- write ---------------------------------------------------------------------------
  write(filename,'(a,i6.6,a)') trim(LOCAL_PATH)//'/proc',myrank,'_specfempp_database.bin'

  open(unit=IOUT,file=trim(filename),status='unknown',form='unformatted',action='write',iostat=ier)
  if (ier /= 0) call exit_MPI(myrank,'Error opening '//trim(filename))

  ! header
  write(IOUT) SPECFEMPP_DB_MAGIC,SPECFEMPP_DB_VERSION
  write(IOUT) PLANET_TYPE,R_PLANET,RHOAV
  write(IOUT) NGNOD,NGLLX,NGLLY,NGLLZ,db_nregions
  write(IOUT) ELLIPTICITY,TOPOGRAPHY,GRAVITY,FULL_GRAVITY,ROTATION,ATTENUATION,OCEANS, &
              has_reference_geometry
  write(IOUT) SPECFEMPP_MATERIAL_ORACLE

  ! model config
  codes = (/ REFERENCE_1D_MODEL,THREE_D_MODEL,THREE_D_MODEL_IC, &
             REFERENCE_CRUSTAL_MODEL,MODEL_GLL_TYPE /)
  flags = (/ TRANSVERSE_ISOTROPY,CRUSTAL,ONE_CRUST,CASE_3D, &
             ANISOTROPIC_3D_MANTLE,ANISOTROPIC_INNER_CORE, &
             MODEL_3D_MANTLE_PERTUBATIONS,HETEROGEN_3D_MANTLE, &
             ATTENUATION_3D,ATTENUATION_3D_BERKELEY,ATTENUATION_GLL, &
             HONOR_1D_SPHERICAL_MOHO,MODEL_GLL,USE_FULL_TISO_MANTLE, &
             REGIONAL_MOHO_MESH,EMC_MODEL /)

  write(IOUT) MODEL
  write(IOUT) N_CODES,codes
  write(IOUT) N_FLAGS,flags

  ! nodes: dimensionalized to SI metres on the way out
  write(IOUT) nnode
  write(IOUT) xn(:)*R_PLANET,yn(:)*R_PLANET,zn(:)*R_PLANET

  if (has_reference_geometry) then
    write(IOUT) xnr(:)*R_PLANET,ynr(:)*R_PLANET,znr(:)*R_PLANET
  endif

  ! elements
  write(IOUT) nspec_total
  write(IOUT) db_region(1:nspec_total),db_medium_tag(1:nspec_total), &
              db_property_tag(1:nspec_total),db_idoubling(1:nspec_total)
  write(IOUT) db_rmin(1:nspec_total)*R_PLANET,db_rmax(1:nspec_total)*R_PLANET
  write(IOUT) db_elem_in_crust(1:nspec_total)
  write(IOUT) node_ids

  ! boundary surfaces
  call save_database_specfempp_write_faces(db_free_ispec,db_free_face,db_n_free)
  call save_database_specfempp_write_faces(db_cmb_ispec,db_cmb_face,db_n_cmb)
  call save_database_specfempp_write_faces(db_icb_ispec,db_icb_face,db_n_icb)
  if (OCEANS) then
    ! the ocean load acts on the free surface of the crust/mantle region
    call save_database_specfempp_write_faces(db_free_ispec,db_free_face,db_n_free)
  else
    call save_database_specfempp_write_faces(db_ocean_ispec,db_ocean_face,db_n_ocean)
  endif

  ! adjacency
  write(IOUT) nb_adj_edges
  write(IOUT) xadj_adj(1:nspec_total+1)
  write(IOUT) adjncy_adj(1:nb_adj_edges)
  write(IOUT) adj_types(1:nb_adj_edges)

  write(IOUT) num_neighbors
  do i = 1,num_neighbors
    write(IOUT) nb_rank(i),nb_count(i)
    if (nb_count(i) > 0) then
      write(IOUT) nb_nodes(nb_offset(i)+1:nb_offset(i)+nb_count(i))
    endif
  enddo

  ! material block: nothing to write in ORACLE mode

  close(IOUT)

  ! user output
  if (myrank == 0) then
    write(IMAIN,*) '    number of elements   : ',nspec_total
    write(IMAIN,*) '    number of nodes      : ',nnode
    write(IMAIN,*) '    adjacency edges      : ',nb_adj_edges
    write(IMAIN,*) '    MPI neighbors        : ',num_neighbors
    write(IMAIN,*) '    written to           : ',trim(filename)
    call flush_IMAIN()
  endif

  deallocate(xn,yn,zn,node_ids)
  if (allocated(xnr)) deallocate(xnr,ynr,znr)
  deallocate(xadj_adj,adjncy_adj,adj_types)
  if (allocated(nb_rank)) deallocate(nb_rank,nb_count,nb_offset)
  if (allocated(nb_nodes)) deallocate(nb_nodes)

  call save_database_specfempp_free()

  end subroutine save_database_specfempp_write

!
!-------------------------------------------------------------------------------------------------
!

  subroutine save_database_specfempp_write_faces(list_ispec,list_face,list_n)

! writes one boundary surface block

  use constants, only: IOUT

  implicit none

  integer,dimension(:),allocatable,intent(in) :: list_ispec,list_face
  integer,intent(in) :: list_n

  write(IOUT) list_n
  if (list_n > 0) write(IOUT) list_ispec(1:list_n),list_face(1:list_n)

  end subroutine save_database_specfempp_write_faces

!
!-------------------------------------------------------------------------------------------------
!

  subroutine save_database_specfempp_merge_mpi(iglob_merged,npointot, &
                                               num_neighbors,nb_rank,nb_count,nb_offset,nb_nodes)

! merges the per-region interface entries into one entry per neighbor rank.
!
! Region order (crust/mantle, outer core, inner core) is the same on every rank, and so
! is the point order within each region's interface, so the concatenation is consistent
! between the two ranks of a pair. A node on a region boundary can appear in two regions'
! interfaces and weld to the same merged node, so duplicates are removed keeping the
! first occurrence -- again identically on both sides.

  use constants, only: myrank


  implicit none

  integer,intent(in) :: npointot
  integer,dimension(npointot),intent(in) :: iglob_merged
  integer,intent(out) :: num_neighbors
  integer,dimension(:),allocatable,intent(out) :: nb_rank,nb_count,nb_offset,nb_nodes

  ! local parameters
  integer :: i,j,k,inode,ier,total,pos
  logical :: is_known
  integer, dimension(:), allocatable :: seen

  num_neighbors = 0

  if (db_n_if == 0) then
    allocate(nb_rank(1),nb_count(1),nb_offset(1),nb_nodes(1),stat=ier)
    if (ier /= 0) call exit_MPI(myrank,'Error allocating empty neighbor arrays')
    nb_rank(:) = -1; nb_count(:) = 0; nb_offset(:) = 0; nb_nodes(:) = 0
    return
  endif

  ! distinct neighbor ranks, in order of first appearance
  allocate(nb_rank(db_n_if),nb_count(db_n_if),nb_offset(db_n_if),stat=ier)
  if (ier /= 0) call exit_MPI(myrank,'Error allocating neighbor arrays')
  nb_rank(:) = -1; nb_count(:) = 0; nb_offset(:) = 0

  do i = 1,db_n_if
    is_known = .false.
    do j = 1,num_neighbors
      if (nb_rank(j) == db_if_rank(i)) then
        is_known = .true.
        exit
      endif
    enddo
    if (.not. is_known) then
      num_neighbors = num_neighbors + 1
      nb_rank(num_neighbors) = db_if_rank(i)
    endif
  enddo

  ! upper bound on the concatenated node list
  total = db_n_if_pts
  allocate(nb_nodes(max(total,1)),stat=ier)
  if (ier /= 0) call exit_MPI(myrank,'Error allocating neighbor node list')
  nb_nodes(:) = 0

  ! `seen` marks, per neighbor, the merged nodes already collected, so that a node shared
  ! across two regions is not counted twice
  allocate(seen(maxval(iglob_merged)),stat=ier)
  if (ier /= 0) call exit_MPI(myrank,'Error allocating seen array')
  seen(:) = 0

  pos = 0
  do j = 1,num_neighbors
    nb_offset(j) = pos
    do i = 1,db_n_if
      if (db_if_rank(i) /= nb_rank(j)) cycle
      do k = 1,db_if_n(i)
        inode = iglob_merged(db_if_pts(db_if_off(i) + k))
        if (seen(inode) == j) cycle
        seen(inode) = j
        pos = pos + 1
        nb_nodes(pos) = inode
      enddo
    enddo
    nb_count(j) = pos - nb_offset(j)
  enddo

  deallocate(seen)

  end subroutine save_database_specfempp_merge_mpi

!
!-------------------------------------------------------------------------------------------------
!

  subroutine save_database_specfempp_check_mpi(num_neighbors,nb_rank,nb_count)

! verifies that the two ranks of every neighbor pair agree on how many anchor nodes they
! share. The anchor filter and the fictitious-element exclusion above are both local
! decisions, so this is the cheap guard that they came out symmetric; a mismatch would
! otherwise show up much later as a silently wrong MPI assembly.

  use constants, only: myrank

  implicit none

  integer,intent(in) :: num_neighbors
  integer,dimension(:),allocatable,intent(in) :: nb_rank,nb_count

  ! local parameters
  integer, parameter :: itag = 0
  integer :: i,neighbor,ier
  double precision, dimension(:), allocatable :: sendbuf,recvbuf
  integer, dimension(:), allocatable :: req_send,req_recv
  character(len=128) :: msg

  if (num_neighbors <= 0) return

  ! the exchange has to be non-blocking: neighbor lists are not in the same order on
  ! every rank, so any blocking send/recv ordering can end up in a cycle
  allocate(sendbuf(num_neighbors),recvbuf(num_neighbors), &
           req_send(num_neighbors),req_recv(num_neighbors),stat=ier)
  if (ier /= 0) call exit_MPI(myrank,'Error allocating exchange buffers in save_database_specfempp_check_mpi')

  do i = 1,num_neighbors
    sendbuf(i) = dble(nb_count(i))
    recvbuf(i) = -1.d0
  enddo

  do i = 1,num_neighbors
    call irecv_dp(recvbuf(i:i),1,nb_rank(i),itag,req_recv(i))
  enddo
  do i = 1,num_neighbors
    call isend_dp(sendbuf(i:i),1,nb_rank(i),itag,req_send(i))
  enddo
  do i = 1,num_neighbors
    call wait_req(req_recv(i))
    call wait_req(req_send(i))
  enddo

  do i = 1,num_neighbors
    neighbor = nb_rank(i)
    if (nint(recvbuf(i)) /= nb_count(i)) then
      write(msg,'(a,i6,a,i6,a,i8,a,i8)') 'asymmetric SPECFEM++ MPI interface: rank ',myrank, &
        ' <-> ',neighbor,' shares ',nb_count(i),' vs ',nint(recvbuf(i))
      call exit_MPI(myrank,trim(msg))
    endif
  enddo

  deallocate(sendbuf,recvbuf,req_send,req_recv)

  end subroutine save_database_specfempp_check_mpi

!
!-------------------------------------------------------------------------------------------------
!

  subroutine save_database_specfempp_free()

! releases the staging arrays


  implicit none

  if (allocated(db_region)) deallocate(db_region,db_medium_tag,db_property_tag,db_idoubling)
  if (allocated(db_rmin)) deallocate(db_rmin,db_rmax)
  if (allocated(db_elem_in_crust)) deallocate(db_elem_in_crust)
  if (allocated(db_x)) deallocate(db_x,db_y,db_z)
  if (allocated(db_xref)) deallocate(db_xref,db_yref,db_zref)

  if (allocated(db_free_ispec)) deallocate(db_free_ispec,db_free_face)
  if (allocated(db_cmb_ispec)) deallocate(db_cmb_ispec,db_cmb_face)
  if (allocated(db_icb_ispec)) deallocate(db_icb_ispec,db_icb_face)
  if (allocated(db_ocean_ispec)) deallocate(db_ocean_ispec,db_ocean_face)

  if (allocated(db_if_rank)) deallocate(db_if_rank,db_if_n,db_if_off)
  if (allocated(db_if_pts)) deallocate(db_if_pts)

  db_nspec = 0
  db_nregions = 0
  db_n_free = 0; db_n_cmb = 0; db_n_icb = 0; db_n_ocean = 0
  db_n_if = 0
  db_n_if_pts = 0

  end subroutine save_database_specfempp_free

  end module specfempp_database_par
