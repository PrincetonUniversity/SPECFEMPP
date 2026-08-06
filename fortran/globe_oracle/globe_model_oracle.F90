!=====================================================================
!
!  globe_model_oracle -- ISO_C_BINDING wrappers exposing the
!                        SPECFEM3D_GLOBE model catalog as a callable
!                        "oracle" for SPECFEM++.
!
!  See issue #2001. This file is SPECFEM++ code; the vendored mesher
!  tree under fortran/meshfem3d_globe/ is deliberately left untouched
!  so it stays a clean upstream mirror.
!
!  Three production entry points plus one test-only shim:
!    globe_oracle_dims            -- compile-time NGLL / N_SLS query
!    globe_oracle_init            -- one-time model setup
!    globe_oracle_get_element     -- material for one element's GLL points
!    globe_oracle_prem_reference  -- TEST ONLY, see note at its definition
!
!  Contracts (see the plan for provenance):
!    * Coordinates and radii cross the boundary in SI metres. This module
!      non-dimensionalizes by R_PLANET, because the catalog works in
!      non-dimensional radius internally (get_model.F90:164-165 compares
!      against non-dimensional rmin/rmax, and get_model_check_idoubling
!      re-dimensionalizes at get_model.F90:381).
!    * Call single-threaded, at setup only. The catalog holds global module
!      state and a handful of `save` variables.
!    * GLL point ordering is the mesher's: k outermost, i innermost.
!
!=====================================================================

  module globe_oracle_par

! module state private to the oracle wrappers

  use iso_c_binding, only: c_int

  implicit none

  ! guards against a second globe_oracle_init() -- the catalog's state is
  ! global, so two independent configurations cannot coexist.
  logical :: is_initialized = .false.

  ! tracks whether this module opened the IMAIN unit, so a failed init followed
  ! by a retry does not attempt to re-open an already-connected unit
  logical :: imain_is_open = .false.

  ! status codes returned across the C boundary
  integer(c_int), parameter :: GLOBE_ORACLE_OK = 0
  integer(c_int), parameter :: GLOBE_ORACLE_ALREADY_INITIALIZED = 1
  integer(c_int), parameter :: GLOBE_ORACLE_NOT_INITIALIZED = 2
  integer(c_int), parameter :: GLOBE_ORACLE_UNSUPPORTED_MODEL = 3
  integer(c_int), parameter :: GLOBE_ORACLE_IMAIN_OPEN_FAILED = 4
  integer(c_int), parameter :: GLOBE_ORACLE_BAD_ARGUMENT = 5

  end module globe_oracle_par


!
!-------------------------------------------------------------------------------------------------
!


  subroutine globe_oracle_reset_sticky_flags()

! Resets catalog flags that get_model_parameters_flags() does NOT re-initialize
! at its top, so a rejected or superseded configuration cannot leak into the
! next one.
!
! get_model_parameters_flags() defensively resets almost every model flag before
! its `select case` (get_model_parameters.F90:260-297) -- MODEL_GLL, CEM_*,
! EMC_*, HETEROGEN_3D_MANTLE, ATTENUATION_3D, ATTENUATION_3D_BERKELEY. It misses
! ATTENUATION_GLL, which is only ever assigned .true. (:913) and otherwise relies
! on its declaration initializer (shared_par.f90:331).
!
! That is harmless upstream, where the mesher calls get_model_parameters() once
! per process. It is not harmless here: the oracle can be configured more than
! once, so a single "gll_qmu" attempt would otherwise latch ATTENUATION_GLL true
! and poison every later model.

  use shared_parameters, only: ATTENUATION_GLL

  implicit none

  ATTENUATION_GLL = .false.

  end subroutine globe_oracle_reset_sticky_flags


!
!-------------------------------------------------------------------------------------------------
!


  subroutine globe_oracle_dims(ngllx_out, nglly_out, ngllz_out, n_sls_out) &
    bind(C, name="globe_oracle_dims")

! reports the compile-time quadrature sizes the catalog was built with, so the
! caller can assert its own NGLL matches before trusting any returned values.

  use iso_c_binding, only: c_int
  use constants, only: NGLLX, NGLLY, NGLLZ, N_SLS

  implicit none

  integer(c_int), intent(out) :: ngllx_out, nglly_out, ngllz_out, n_sls_out

  ngllx_out = int(NGLLX, kind=c_int)
  nglly_out = int(NGLLY, kind=c_int)
  ngllz_out = int(NGLLZ, kind=c_int)
  n_sls_out = int(N_SLS, kind=c_int)

  end subroutine globe_oracle_dims


!
!-------------------------------------------------------------------------------------------------
!


  integer(c_int) function globe_oracle_init(model_name, name_len, &
                                            imain_path, imain_path_len, &
                                            planet_type_in, nchunks_in, &
                                            nex_xi_in, nex_eta_in, &
                                            ellipticity_in, topography_in, oceans_in, &
                                            attenuation_in, gravity_in, rotation_in, &
                                            min_attenuation_period_in, &
                                            max_attenuation_period_in, &
                                            comm_f) &
    bind(C, name="globe_oracle_init")

! One-time model setup. Deliberately does NOT read the globe Par_file: the
! model is identified by name only, and get_model_parameters() re-derives every
! flag and discontinuity radius from it. That is the whole point of the oracle
! design -- there is exactly one place a model is chosen (the mesher), and
! replaying the name here cannot drift from it.
!
! Returns GLOBE_ORACLE_OK (0) on success, a nonzero status otherwise.
!
! Caveat: an unrecognized model name cannot be reported gracefully.
! get_model_parameters_flags() terminates the process with a bare `stop` in its
! `case default` (get_model_parameters.F90:978-982), and likewise on several
! flag-consistency violations (:1026-1060). Only models that parse successfully
! but land on a code path the oracle cannot reproduce come back as
! GLOBE_ORACLE_UNSUPPORTED_MODEL.

  use iso_c_binding, only: c_int, c_char, c_double

  use globe_oracle_par, only: is_initialized, imain_is_open, &
    GLOBE_ORACLE_OK, GLOBE_ORACLE_ALREADY_INITIALIZED, &
    GLOBE_ORACLE_UNSUPPORTED_MODEL, GLOBE_ORACLE_IMAIN_OPEN_FAILED, &
    GLOBE_ORACLE_BAD_ARGUMENT

  use constants, only: MAX_STRING_LEN, IMAIN, ISTANDARD_OUTPUT, myrank

  use shared_parameters, only: MODEL, LOCAL_PATH, &
    PLANET_TYPE, NCHUNKS, NEX_XI, NEX_ETA, &
    ELLIPTICITY, GRAVITY, ROTATION, TOPOGRAPHY, OCEANS, ATTENUATION, &
    SAVE_MESH_FILES, ABSORBING_CONDITIONS, &
    MIN_ATTENUATION_PERIOD, MAX_ATTENUATION_PERIOD, &
    MODEL_GLL, CEM_ACCEPT, CEM_REQUEST, HETEROGEN_3D_MANTLE, ATTENUATION_GLL, &
    EMC_MODEL, REGIONAL_MESH_CUTOFF, ADD_SCATTERING_PERTURBATIONS

  use my_mpi, only: my_local_mpi_comm_world

  implicit none

  character(kind=c_char), dimension(*), intent(in) :: model_name
  integer(c_int), value, intent(in) :: name_len
  character(kind=c_char), dimension(*), intent(in) :: imain_path
  integer(c_int), value, intent(in) :: imain_path_len
  ! NOTE: every dummy below carries an `_in` suffix on purpose. Fortran is
  ! case-insensitive, so a dummy named e.g. `ellipticity` would shadow the
  ! shared_parameters variable ELLIPTICITY this routine has to write to -- and
  ! gfortran rejects the whole `use ... only:` list on the first such clash.
  integer(c_int), value, intent(in) :: planet_type_in, nchunks_in
  integer(c_int), value, intent(in) :: nex_xi_in, nex_eta_in
  integer(c_int), value, intent(in) :: ellipticity_in, topography_in, oceans_in
  integer(c_int), value, intent(in) :: attenuation_in, gravity_in, rotation_in
  real(c_double), value, intent(in) :: min_attenuation_period_in
  real(c_double), value, intent(in) :: max_attenuation_period_in
  integer(c_int), value, intent(in) :: comm_f

  ! local parameters
  character(len=MAX_STRING_LEN) :: model_string, imain_string
  integer :: i, ier

  ! refuse a second configuration: the catalog's state is global module state,
  ! so re-initializing would silently mutate an oracle the caller still holds.
  if (is_initialized) then
    globe_oracle_init = GLOBE_ORACLE_ALREADY_INITIALIZED
    return
  endif

  if (name_len <= 0 .or. name_len > MAX_STRING_LEN) then
    globe_oracle_init = GLOBE_ORACLE_BAD_ARGUMENT
    return
  endif
  if (imain_path_len < 0 .or. imain_path_len > MAX_STRING_LEN) then
    globe_oracle_init = GLOBE_ORACLE_BAD_ARGUMENT
    return
  endif

  ! marshals the C strings (not NUL-terminated on this side; length is explicit)
  model_string = ''
  do i = 1, int(name_len)
    model_string(i:i) = model_name(i)
  enddo

  imain_string = ''
  do i = 1, int(imain_path_len)
    imain_string(i:i) = imain_path(i)
  enddo

  ! ---- MPI ----------------------------------------------------------------
  ! The catalog broadcasts through my_local_mpi_comm_world (parallel.f90:72),
  ! so adopting the caller's communicator is a single assignment rather than
  ! plumbing a handle through every bcast_all_* routine.
  my_local_mpi_comm_world = int(comm_f)
  call world_rank(myrank)

  ! ---- state that has no default initializer in shared_par.f90 ------------
  MODEL = trim(model_string)
  PLANET_TYPE = int(planet_type_in)
  NCHUNKS = int(nchunks_in)
  NEX_XI = int(nex_xi_in)
  NEX_ETA = int(nex_eta_in)

  ELLIPTICITY = (ellipticity_in /= 0)
  TOPOGRAPHY = (topography_in /= 0)
  OCEANS = (oceans_in /= 0)
  ATTENUATION = (attenuation_in /= 0)
  GRAVITY = (gravity_in /= 0)
  ROTATION = (rotation_in /= 0)

  ! The attenuation period band is NOT derivable from the model name: the mesher
  ! computes it in rcp_set_compute_parameters/get_timestep_and_layers, which the
  ! oracle deliberately does not call. So it has to be supplied, and the database
  ! has to carry it. With ATTENUATION on and either bound left at zero,
  ! attenuation_tau_sigma aborts the process (model_attenuation.f90:625-628);
  ! rejecting it here turns that abort into a status code.
  MIN_ATTENUATION_PERIOD = real(min_attenuation_period_in, kind=8)
  MAX_ATTENUATION_PERIOD = real(max_attenuation_period_in, kind=8)

  if (ATTENUATION) then
    if (MIN_ATTENUATION_PERIOD <= 0.d0 .or. MAX_ATTENUATION_PERIOD <= 0.d0 .or. &
        MAX_ATTENUATION_PERIOD <= MIN_ATTENUATION_PERIOD) then
      globe_oracle_init = GLOBE_ORACLE_BAD_ARGUMENT
      return
    endif
  endif

  ! MUST be false: it gates the unguarded VTK writers in
  ! meshfem3D_models.F90 (meshfem3D_plot_VTK_crust_moho / _topo_bathy), which
  ! write to IMAIN and to disk without a myrank check.
  SAVE_MESH_FILES = .false.

  ! the oracle drops get_model's rho_vp/rho_vs Stacey branch entirely
  ABSORBING_CONDITIONS = .false.

  ! only consulted when TOPOGRAPHY is on (model_topo_bathy_broadcast)
  LOCAL_PATH = '.'

  ! ---- state that already defaults correctly; pinned here so a future change
  !      to shared_par.f90 cannot silently alter the oracle's behaviour ------
  REGIONAL_MESH_CUTOFF = .false.
  ADD_SCATTERING_PERTURBATIONS = .false.

  ! ---- IMAIN --------------------------------------------------------------
  ! Mandatory. The model catalog contains ~513 write(IMAIN,...) statements in
  ! meshfem3D/model_*.f90, largely unguarded by a myrank check. Mirrors
  ! initialize_mesher.f90:57-58.
  if (IMAIN /= ISTANDARD_OUTPUT .and. .not. imain_is_open) then
    if (len_trim(imain_string) > 0) then
      open(unit=IMAIN, file=trim(imain_string), status='unknown', &
           action='write', iostat=ier)
    else
      open(unit=IMAIN, file='/dev/null', status='unknown', &
           action='write', iostat=ier)
    endif
    if (ier /= 0) then
      globe_oracle_init = GLOBE_ORACLE_IMAIN_OPEN_FAILED
      return
    endif
    imain_is_open = .true.
  endif

  ! clears flags a previous configuration may have latched (see the routine's
  ! own comment for why get_model_parameters_flags cannot be relied on here)
  call globe_oracle_reset_sticky_flags()

  ! ---- resolve the model --------------------------------------------------
  ! get_model_parameters() = get_model_parameters_flags()
  !                        + get_model_planet_constants()
  !                        + get_model_parameters_radii()
  ! Derives every model flag from the MODEL string and every discontinuity
  ! radius (RCMB/RICB/R670/...) from PLANET_TYPE + REFERENCE_1D_MODEL.
  call get_model_parameters()

  ! ---- reject model paths the oracle cannot reproduce ----------------------
  ! These four are exactly the consumers of (ispec,i,j,k) inside the catalog:
  !   model_gll_impose_val        (meshfem3D_models.F90:1888)
  !   request_cem                 (meshfem3D_models.F90:1049)
  !   model_heterogen_mantle      (meshfem3D_models.F90:971,1002)
  !   model_attenuation_gll       (meshfem3D_models.F90:1731)
  ! They index a per-GLL array belonging to the mesher's own discretization, so
  ! a position-only oracle cannot evaluate them. Fail loudly rather than return
  ! a plausible wrong answer.
  if (MODEL_GLL .or. CEM_ACCEPT .or. CEM_REQUEST .or. &
      HETEROGEN_3D_MANTLE .or. ATTENUATION_GLL .or. EMC_MODEL) then
    ! leaves the catalog's flags as we found them, so a rejected model does not
    ! affect a subsequent, supported one
    call globe_oracle_reset_sticky_flags()
    globe_oracle_init = GLOBE_ORACLE_UNSUPPORTED_MODEL
    return
  endif

  ! ---- build the model ----------------------------------------------------
  ! Single entry point: topo/bathy, 1D reference, 3D mantle, crust,
  ! attenuation, scattering.
  call meshfem3D_models_broadcast()

  is_initialized = .true.
  globe_oracle_init = GLOBE_ORACLE_OK

  end function globe_oracle_init


!
!-------------------------------------------------------------------------------------------------
!


  integer(c_int) function globe_oracle_scales(length_scale, density_scale, &
                                              velocity_scale) &
    bind(C, name="globe_oracle_scales")

! Reports the factors that convert the oracle's outputs to SI.
!
! The catalog works in, and this oracle returns, the globe's NON-DIMENSIONAL
! units -- density near 1, velocities near 2. That is deliberate: returning the
! catalog's own numbers untouched is what makes the values bit-for-bit the
! mesher's. Converting here would introduce a rounding step the mesher does not
! have.
!
! The scalings are fixed by model_prem.f90:400-406 and used identically by every
! other reference model:
!   rho_SI = rho * RHOAV
!   v_SI   = v   * R_PLANET * sqrt(PI * GRAV * RHOAV)
!   r_SI   = r   * R_PLANET
!
! Qmu/Qkappa and eta are dimensionless; cij carries density * velocity^2.
!
! Requires a configured oracle: R_PLANET and RHOAV are planet constants set by
! get_model_planet_constants() during init.

  use iso_c_binding, only: c_int, c_double

  use globe_oracle_par, only: is_initialized, &
    GLOBE_ORACLE_OK, GLOBE_ORACLE_NOT_INITIALIZED

  use constants, only: PI, GRAV

  use shared_parameters, only: R_PLANET, RHOAV

  implicit none

  real(c_double), intent(out) :: length_scale, density_scale, velocity_scale

  if (.not. is_initialized) then
    length_scale = 0.d0
    density_scale = 0.d0
    velocity_scale = 0.d0
    globe_oracle_scales = GLOBE_ORACLE_NOT_INITIALIZED
    return
  endif

  length_scale = R_PLANET
  density_scale = RHOAV
  velocity_scale = R_PLANET * sqrt(PI * GRAV * RHOAV)

  globe_oracle_scales = GLOBE_ORACLE_OK

  end function globe_oracle_scales


!
!-------------------------------------------------------------------------------------------------
!


  integer(c_int) function globe_oracle_finalize() &
    bind(C, name="globe_oracle_finalize")

! Releases what the oracle itself owns and clears the initialization guard, so a
! different model can be configured afterwards. Idempotent.
!
! LIMITATION: this releases the oracle's own resources (the IMAIN unit and the
! topo/bathy array allocated by meshfem3D_models_broadcast), but it cannot
! deallocate state owned by individual model_*.f90 files -- upstream provides no
! teardown hooks for those. Re-initializing is therefore reliable for the
! analytic 1D reference models, which allocate nothing, and may fail inside the
! catalog for 3D models that do. Full teardown is tracked as a follow-up; the
! production path configures once per process and never needs this.

  use iso_c_binding, only: c_int

  use globe_oracle_par, only: is_initialized, imain_is_open, GLOBE_ORACLE_OK

  use constants, only: IMAIN, ISTANDARD_OUTPUT

  use meshfem_models_par, only: ibathy_topo

  implicit none

  ! local parameters
  integer :: ier

  if (allocated(ibathy_topo)) deallocate(ibathy_topo)

  call globe_oracle_reset_sticky_flags()

  if (imain_is_open .and. IMAIN /= ISTANDARD_OUTPUT) then
    close(unit=IMAIN, iostat=ier)
    imain_is_open = .false.
  endif

  is_initialized = .false.
  globe_oracle_finalize = GLOBE_ORACLE_OK

  end function globe_oracle_finalize


!
!-------------------------------------------------------------------------------------------------
!


  integer(c_int) function globe_oracle_get_element(iregion_code, idoubling, &
                                                   rmin_si, rmax_si, &
                                                   elem_in_crust, elem_in_mantle, &
                                                   xyz_si, &
                                                   rho_out, vpv_out, vph_out, &
                                                   vsv_out, vsh_out, eta_out, &
                                                   vp_iso_out, vs_iso_out, &
                                                   qmu_out, qkappa_out, &
                                                   cij_out, gc_prime_out, gs_prime_out, &
                                                   is_anisotropic) &
    bind(C, name="globe_oracle_get_element")

! Evaluates the model at every GLL point of one element.
!
! This is get_model()'s body (get_model.F90:104-247) with the array stores and
! the MODEL_GLL branch removed, returning values instead. It makes the identical
! meshfem3D_models_get1D_val / get3Dmntl_val / get3Dcrust_val / getatten_val
! calls, in the identical order.
!
! PER-ELEMENT, NOT PER-POINT, and that is load-bearing rather than a
! performance choice. `moho` and `sediment` are element-scoped in the original
! (get_model.F90:100-101, under a comment warning that this placement caused the
! "s362ani + attenuation" bug in 2013/2014), and they genuinely carry across GLL
! points: meshfem3D_models_get3Dcrust_val returns early at
! meshfem3D_models.F90:1300 (`if (r < R_DEEPEST_CRUST) return`) BEFORE resetting
! moho at :1311, so a point deeper than ~80 km inherits the moho of an earlier
! point in the same element -- and that value is read at :1745 to relocate the
! attenuation sampling radius. A per-point entry point would not be bit-for-bit
! for ATTENUATION_3D models.

  use iso_c_binding, only: c_int, c_double

  use globe_oracle_par, only: is_initialized, &
    GLOBE_ORACLE_OK, GLOBE_ORACLE_NOT_INITIALIZED

  use constants, only: NGLLX, NGLLY, NGLLZ, N_SLS, CUSTOM_REAL, &
    TINYVAL, IREGION_CRUST_MANTLE, IREGION_INNER_CORE, IREGION_OUTER_CORE

  use shared_parameters, only: R_PLANET, ADD_SCATTERING_PERTURBATIONS, &
    RCMB, RICB, R670, RMOHO, RTOPDDOUBLEPRIME, R220, R771, R400, R120, R80, &
    RMIDDLE_CRUST, &
    ANISOTROPIC_3D_MANTLE, ANISOTROPIC_INNER_CORE, ATTENUATION, CRUSTAL, &
    CEM_ACCEPT

  use regions_mesh_par2, only: tau_s_store

  implicit none

  integer, parameter :: NGLL_CUBE = NGLLX * NGLLY * NGLLZ

  integer(c_int), value, intent(in) :: iregion_code, idoubling
  real(c_double), value, intent(in) :: rmin_si, rmax_si
  integer(c_int), value, intent(in) :: elem_in_crust, elem_in_mantle

  real(c_double), dimension(3, NGLL_CUBE), intent(in) :: xyz_si

  real(c_double), dimension(NGLL_CUBE), intent(out) :: rho_out, vpv_out, vph_out
  real(c_double), dimension(NGLL_CUBE), intent(out) :: vsv_out, vsh_out, eta_out
  real(c_double), dimension(NGLL_CUBE), intent(out) :: vp_iso_out, vs_iso_out
  real(c_double), dimension(NGLL_CUBE), intent(out) :: qmu_out, qkappa_out
  real(c_double), dimension(21, NGLL_CUBE), intent(out) :: cij_out
  real(c_double), dimension(NGLL_CUBE), intent(out) :: gc_prime_out, gs_prime_out
  integer(c_int), intent(out) :: is_anisotropic

  ! local parameters -- mirroring get_model.F90:74-91
  double precision :: xmesh, ymesh, zmesh
  double precision :: c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26, c33, &
                      c34, c35, c36, c44, c45, c46, c55, c56, c66
  double precision :: A, C, L, N, F, Gc, Gs, Gc_prime, Gs_prime, mu0
  double precision :: Qkappa, Qmu
  double precision, dimension(N_SLS) :: tau_e, tau_s
  double precision :: rho, vs, vp
  double precision :: vpv, vph, vsv, vsh, eta_aniso
  double precision :: r, r_prem, moho, sediment
  double precision :: theta, phi
  double precision :: rmin, rmax
  logical :: in_crust, in_mantle

  integer :: ipoint

  if (.not. is_initialized) then
    globe_oracle_get_element = GLOBE_ORACLE_NOT_INITIALIZED
    return
  endif

  ! SI -> non-dimensional. The catalog compares radii against non-dimensional
  ! rmin/rmax and re-dimensionalizes internally where it needs metres.
  rmin = rmin_si / R_PLANET
  rmax = rmax_si / R_PLANET

  in_crust = (elem_in_crust /= 0)
  in_mantle = (elem_in_mantle /= 0)

  ! reports which storage tier the caller should use for this element,
  ! mirroring the branches at get_model.F90:268 and :276
  if ((ANISOTROPIC_INNER_CORE .and. iregion_code == IREGION_INNER_CORE) .or. &
      (ANISOTROPIC_3D_MANTLE .and. iregion_code == IREGION_CRUST_MANTLE)) then
    is_anisotropic = 1
  else
    is_anisotropic = 0
  endif

  ! it is *CRUCIAL* that these are initialized here, outside the point loop:
  ! see the note in the header comment above and get_model.F90:98-101.
  moho = 0.d0
  sediment = 0.d0

  ! tau_s is a per-run constant that model_attenuation_broadcast() cached in
  ! tau_s_store during init, exactly as get_model.F90:148 reads it. It does not
  ! influence the Qmu the oracle returns (the caller builds its own SLS
  ! coefficients), but taking the real value keeps the call faithful and means
  ! extending the ABI to return tau_e later needs no change here.
  !
  ! Note tau_s_store is a fixed-size module array, not one of the allocatables
  ! this oracle deliberately never allocates.
  tau_s(:) = tau_s_store(:)

  ! loops over the element's GLL points in the mesher's order (k,j,i with i
  ! innermost); the caller packs xyz_si the same way.
  do ipoint = 1, NGLL_CUBE

    ! initializes values -- get_model.F90:108-148
    rho = 0.d0
    vpv = 0.d0
    vph = 0.d0
    vsv = 0.d0
    vsh = 0.d0

    eta_aniso = 1.d0 ! default for isotropic element

    c11 = 0.d0
    c12 = 0.d0
    c13 = 0.d0
    c14 = 0.d0
    c15 = 0.d0
    c16 = 0.d0
    c22 = 0.d0
    c23 = 0.d0
    c24 = 0.d0
    c25 = 0.d0
    c26 = 0.d0
    c33 = 0.d0
    c34 = 0.d0
    c35 = 0.d0
    c36 = 0.d0
    c44 = 0.d0
    c45 = 0.d0
    c46 = 0.d0
    c55 = 0.d0
    c56 = 0.d0
    c66 = 0.d0

    mu0 = 0.d0
    Gc = 0.d0
    Gs = 0.d0
    Gc_prime = 0.d0
    Gs_prime = 0.d0

    Qmu = 0.d0
    Qkappa = 0.d0
    tau_e(:) = 0.d0

    ! non-dimensionalized GLL point position
    xmesh = xyz_si(1, ipoint) / R_PLANET
    ymesh = xyz_si(2, ipoint) / R_PLANET
    zmesh = xyz_si(3, ipoint) / R_PLANET

    ! gets point's (geocentric) position theta/phi, and exact point radius
    call xyz_2_rthetaphi_dble(xmesh, ymesh, zmesh, r, theta, phi)

    ! puts theta in range [0,PI] / phi in range [0,2PI]
    call reduce(theta, phi)

    ! make sure we are within the right shell in PREM to honor discontinuities
    ! use small geometrical tolerance
    r_prem = r
    if (r <= rmin*1.000001d0) r_prem = rmin*1.000001d0
    if (r >= rmax*0.999999d0) r_prem = rmax*0.999999d0

    ! checks r_prem,rmin/rmax and assigned idoubling
    ! note: this aborts the process on an inconsistent flag (see get_model.F90
    !       :388 onwards). Converting those to status codes is tracked as a
    !       follow-up; keeping the call here validates the idoubling contract.
    call get_model_check_idoubling(r_prem, theta, phi, rmin, rmax, int(idoubling), &
                                   RICB, RCMB, RTOPDDOUBLEPRIME, R220, R670)

    ! gets reference model values: rho,vpv,vph,vsv,vsh and eta_aniso
    call meshfem3D_models_get1D_val(int(iregion_code), int(idoubling), &
                                    r_prem, rho, vpv, vph, vsv, vsh, eta_aniso, &
                                    Qkappa, Qmu, RICB, RCMB, &
                                    RTOPDDOUBLEPRIME, R80, R120, R220, R400, R670, R771, &
                                    RMOHO, RMIDDLE_CRUST)

    ! isotropic Voigt average -- get_model.F90:181-190.
    ! In the outer core this yields vs = 0 exactly, which is the acoustic tag.
    if (iregion_code == IREGION_OUTER_CORE) then
      ! fluid with zero shear speed
      vs = 0.d0
    else
      vs = sqrt(((1.d0-2.d0*eta_aniso)*vph*vph + vpv*vpv &
                + 5.d0*vsh*vsh + (6.d0+4.d0*eta_aniso)*vsv*vsv)/15.d0)
    endif

    ! 1D isotropic mu0 = rho * Vs*Vs.
    !
    ! DO NOT "simplify" the round-trip below. The mesher writes mu0 into
    ! mu0store as real(...,kind=CUSTOM_REAL) (get_model.F90:191) and reads it
    ! back as a double before normalizing Gc/Gs (get_model.F90:309-312). In a
    ! single-precision build that truncation is observable in Gc_prime/Gs_prime,
    ! so reproducing it is required for bit-for-bit agreement.
    mu0 = real(real(rho * vs*vs, kind=CUSTOM_REAL), kind=8)

    ! gets the 3-D model parameters for the mantle.
    ! the trailing ispec,i,j,k are only read by model paths rejected in
    ! globe_oracle_init(), so passing 1s here is safe and never observable.
    call meshfem3D_models_get3Dmntl_val(int(iregion_code), r_prem, rho, &
                                        vpv, vph, vsv, vsh, eta_aniso, &
                                        RCMB, RMOHO, &
                                        r, theta, phi, &
                                        c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26, &
                                        c33, c34, c35, c36, c44, c45, c46, c55, c56, c66, &
                                        1, 1, 1, 1)

    ! gets the 3-D crustal model
    if (CRUSTAL .and. .not. CEM_ACCEPT) then
      if (.not. in_mantle) &
        call meshfem3D_models_get3Dcrust_val(int(iregion_code), r, theta, phi, &
                                             vpv, vph, vsv, vsh, rho, eta_aniso, &
                                             c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26, &
                                             c33, c34, c35, c36, c44, c45, c46, c55, c56, c66, &
                                             in_crust, moho, sediment)
    endif

    ! note: get_model.F90:214-219 (MODEL_GLL) is intentionally absent -- that
    !       path is rejected in globe_oracle_init().

    ! adds scattering perturbations
    if (ADD_SCATTERING_PERTURBATIONS) then
      call model_scattering_add_perturbations(int(iregion_code), xmesh, ymesh, zmesh, &
                                              vpv, vph, vsv, vsh, rho, eta_aniso, &
                                              c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26, &
                                              c33, c34, c35, c36, c44, c45, c46, c55, c56, c66)
    endif

    ! attenuation: defined after Moho stretch and before TOPOGRAPHY/ELLIPTICITY
    if (ATTENUATION) then
      call meshfem3D_models_getatten_val(int(iregion_code), int(idoubling), &
                                         r_prem, theta, phi, &
                                         1, 1, 1, 1, &
                                         tau_e, tau_s, &
                                         moho, Qmu, Qkappa, in_crust, rho)
    endif

    ! Isotropic P velocity, the Voigt companion to vs above. get_model does not
    ! keep this (it stores kappa/mu instead), but the caller needs it for
    ! isotropic media and for the CFL time-step estimate, and the averaging
    ! convention belongs in exactly one place. Formula is
    ! meshfem3D_models_get1D_val's (meshfem3D_models.F90:446-448).
    !
    ! Applied unconditionally, matching how get_model treats vs: for a truly
    ! isotropic point it reduces algebraically to vpv, but reproducing the
    ! arithmetic rather than short-circuiting keeps it bit-identical to the
    ! reference path in globe_oracle_prem_reference().
    vp = sqrt(((8.d0+4.d0*eta_aniso)*vph*vph + 3.d0*vpv*vpv &
              + (8.d0-8.d0*eta_aniso)*vsv*vsv)/15.d0)

    ! azimuthal anisotropy -- get_model.F90:299-317
    if (ANISOTROPIC_3D_MANTLE .and. iregion_code == IREGION_CRUST_MANTLE) then
      call rotate_tensor_global_to_azi(theta, phi, &
                                       A, C, N, L, F, &
                                       Gc, Gs, &
                                       c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26, &
                                       c33, c34, c35, c36, c44, c45, c46, c55, c56, c66)
      if (abs(mu0) > TINYVAL) then
        Gc_prime = Gc / mu0
        Gs_prime = Gs / mu0
      else
        ! get_model.F90:314 stops here; returning zeros lets the caller decide
        Gc_prime = 0.d0
        Gs_prime = 0.d0
      endif
    endif

    ! ---- hand back this point's values ----
    rho_out(ipoint) = rho
    vpv_out(ipoint) = vpv
    vph_out(ipoint) = vph
    vsv_out(ipoint) = vsv
    vsh_out(ipoint) = vsh
    eta_out(ipoint) = eta_aniso

    vp_iso_out(ipoint) = vp
    vs_iso_out(ipoint) = vs

    qmu_out(ipoint) = Qmu
    qkappa_out(ipoint) = Qkappa

    cij_out(1, ipoint) = c11
    cij_out(2, ipoint) = c12
    cij_out(3, ipoint) = c13
    cij_out(4, ipoint) = c14
    cij_out(5, ipoint) = c15
    cij_out(6, ipoint) = c16
    cij_out(7, ipoint) = c22
    cij_out(8, ipoint) = c23
    cij_out(9, ipoint) = c24
    cij_out(10, ipoint) = c25
    cij_out(11, ipoint) = c26
    cij_out(12, ipoint) = c33
    cij_out(13, ipoint) = c34
    cij_out(14, ipoint) = c35
    cij_out(15, ipoint) = c36
    cij_out(16, ipoint) = c44
    cij_out(17, ipoint) = c45
    cij_out(18, ipoint) = c46
    cij_out(19, ipoint) = c55
    cij_out(20, ipoint) = c56
    cij_out(21, ipoint) = c66

    gc_prime_out(ipoint) = Gc_prime
    gs_prime_out(ipoint) = Gs_prime

  enddo

  globe_oracle_get_element = GLOBE_ORACLE_OK

  end function globe_oracle_get_element


!
!-------------------------------------------------------------------------------------------------
!


  integer(c_int) function globe_oracle_prem_reference(r_si, idoubling, iregion_code, &
                                                      rho_out, vpv_out, vph_out, &
                                                      vsv_out, vsh_out, eta_out, &
                                                      vp_iso_out, vs_iso_out, &
                                                      qkappa_out, qmu_out) &
    bind(C, name="globe_oracle_prem_reference")

! TEST-ONLY reference evaluation, not part of the production oracle contract.
!
! Calls model_prem_iso / model_prem_aniso directly, choosing between them the
! same way meshfem3D_models_get1D_val does -- on the TRANSVERSE_ISOTROPY flag,
! not on radius (meshfem3D_models.F90:439-486) -- and applies the same Voigt
! reduction. This lets the tests check the oracle against the catalog's own
! reference routine instead of against a hand-written table of expected values,
! which would only ever confirm that the table matches itself.

  use iso_c_binding, only: c_int, c_double

  use globe_oracle_par, only: is_initialized, &
    GLOBE_ORACLE_OK, GLOBE_ORACLE_NOT_INITIALIZED

  use constants, only: IREGION_OUTER_CORE

  use shared_parameters, only: R_PLANET, TRANSVERSE_ISOTROPY, CRUSTAL, &
    REGIONAL_MESH_CUTOFF

  implicit none

  real(c_double), value, intent(in) :: r_si
  integer(c_int), value, intent(in) :: idoubling, iregion_code
  real(c_double), intent(out) :: rho_out, vpv_out, vph_out, vsv_out, vsh_out, eta_out
  real(c_double), intent(out) :: vp_iso_out, vs_iso_out, qkappa_out, qmu_out

  ! local parameters
  double precision :: x, rho, drhodr, vp, vs, vpv, vph, vsv, vsh, eta_aniso
  double precision :: Qkappa, Qmu
  logical :: check_doubling_flag

  if (.not. is_initialized) then
    globe_oracle_prem_reference = GLOBE_ORACLE_NOT_INITIALIZED
    return
  endif

  x = r_si / R_PLANET

  ! mirrors meshfem3D_models_get1D_val:411-418
  check_doubling_flag = .not. REGIONAL_MESH_CUTOFF

  rho = 0.d0
  drhodr = 0.d0
  vp = 0.d0
  vs = 0.d0
  vpv = 0.d0
  vph = 0.d0
  vsv = 0.d0
  vsh = 0.d0
  eta_aniso = 1.d0
  Qkappa = 0.d0
  Qmu = 0.d0

  ! picks the reference routine exactly as meshfem3D_models_get1D_val does --
  ! on the TRANSVERSE_ISOTROPY flag, not on radius
  ! (meshfem3D_models.F90:439-486)
  if (TRANSVERSE_ISOTROPY) then
    call model_prem_aniso(x, rho, vpv, vph, vsv, vsh, eta_aniso, Qkappa, Qmu, &
                          int(idoubling), CRUSTAL, check_doubling_flag)
  else
    call model_prem_iso(x, rho, drhodr, vp, vs, Qkappa, Qmu, &
                        int(idoubling), CRUSTAL, check_doubling_flag)
    vpv = vp
    vph = vp
    vsv = vs
    vsh = vs
    eta_aniso = 1.d0
  endif

  ! Applies the same Voigt reduction the oracle applies, from the same inputs,
  ! so vp_iso/vs_iso compare exactly rather than to within an ULP. The
  ! load-bearing assertions in the tests are on the five Love velocities and
  ! rho above, which pass through untouched on both paths.
  vp = sqrt(((8.d0+4.d0*eta_aniso)*vph*vph + 3.d0*vpv*vpv &
            + (8.d0-8.d0*eta_aniso)*vsv*vsv)/15.d0)

  if (iregion_code == IREGION_OUTER_CORE) then
    vs = 0.d0
  else
    vs = sqrt(((1.d0-2.d0*eta_aniso)*vph*vph + vpv*vpv &
              + 5.d0*vsh*vsh + (6.d0+4.d0*eta_aniso)*vsv*vsv)/15.d0)
  endif

  rho_out = rho
  vpv_out = vpv
  vph_out = vph
  vsv_out = vsv
  vsh_out = vsh
  eta_out = eta_aniso
  vp_iso_out = vp
  vs_iso_out = vs
  qkappa_out = Qkappa
  qmu_out = Qmu

  globe_oracle_prem_reference = GLOBE_ORACLE_OK

  end function globe_oracle_prem_reference
