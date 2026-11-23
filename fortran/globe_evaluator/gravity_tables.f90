! Radial setup from SPECFEM3D_GLOBE's prepare_gravity.f90. The spline and
! the analytical profiles use the catalog's non-dimensional units.
subroutine globe_evaluator_gravity_size(size_out) bind(C, name="globe_evaluator_gravity_size")
  use iso_c_binding, only: c_int
  use constants, only: NRAD_GRAVITY
  implicit none
  integer(c_int), intent(out) :: size_out

  size_out = NRAD_GRAVITY
end subroutine globe_evaluator_gravity_size

integer(c_int) function globe_evaluator_gravity_tables( &
    capacity, range_max, minus_gravity, minus_deriv_gravity, density, &
    d_ln_density_dr, minus_rho_g_over_kappa_fluid) bind(C, name="globe_evaluator_gravity_tables")
  use iso_c_binding, only: c_int, c_double
  use constants, only: NR_DENSITY, NRAD_GRAVITY, REFERENCE_MODEL_SOHL_B
  use shared_parameters, only: PLANET_TYPE, IPLANET_EARTH, IPLANET_MARS, IPLANET_MOON, &
    REFERENCE_1D_MODEL, R_PLANET, R_PLANET_KM, TOPO_MAXIMUM, RICB, RCMB
  use globe_evaluator_par, only: is_initialized, &
    GLOBE_EVALUATOR_OK, GLOBE_EVALUATOR_NOT_INITIALIZED, GLOBE_EVALUATOR_BAD_ARGUMENT, GLOBE_EVALUATOR_UNSUPPORTED_MODEL
  implicit none

  integer(c_int), value, intent(in) :: capacity
  real(c_double), intent(out) :: range_max
  real(c_double), dimension(NRAD_GRAVITY), intent(out) :: minus_gravity, minus_deriv_gravity, density, &
    d_ln_density_dr, minus_rho_g_over_kappa_fluid

  integer :: nspl, i, idummy, lower, upper
  double precision :: radii(NR_DENSITY), gravity(NR_DENSITY), gravity2(NR_DENSITY)
  double precision :: radius, g, rho, drhodr, vp, vs, qkappa, qmu, lower_km, upper_km, radius_km

  range_max = 0.d0
  globe_evaluator_gravity_tables = GLOBE_EVALUATOR_NOT_INITIALIZED
  if (.not. is_initialized) return
  globe_evaluator_gravity_tables = GLOBE_EVALUATOR_BAD_ARGUMENT
  if (capacity < NRAD_GRAVITY) return

  ! make_gravity calls Sohl_density, which executes STOP for model B.
  if (REFERENCE_1D_MODEL == REFERENCE_MODEL_SOHL_B) then
    globe_evaluator_gravity_tables = GLOBE_EVALUATOR_UNSUPPORTED_MODEL
    return
  endif

  ! TOPO_MAXIMUM comes from the selected planet's catalog constants.
  range_max = (R_PLANET + dble(TOPO_MAXIMUM)) / R_PLANET
  call make_gravity(nspl, radii, gravity, gravity2)

  do i = 1, NRAD_GRAVITY
    radius = dble(i) / dble(NRAD_GRAVITY) * range_max
    call spline_evaluation(radii, gravity, gravity2, nspl, radius, g)
    idummy = 0
    select case (PLANET_TYPE)
    case (IPLANET_EARTH)
      call model_prem_iso(radius, rho, drhodr, vp, vs, qkappa, qmu, idummy, .false., .false.)
    case (IPLANET_MARS)
      call model_Sohl(radius, rho, drhodr, vp, vs, qkappa, qmu, idummy, .false., .false.)
    case (IPLANET_MOON)
      call model_vpremoon(radius, rho, drhodr, vp, vs, qkappa, qmu, idummy, .false., .false., idummy)
    case default
      return
    end select
    minus_gravity(i) = -g
    minus_deriv_gravity(i) = -(4.d0 * rho - 2.d0 * g / radius)
    density(i) = rho
    minus_rho_g_over_kappa_fluid(i) = -g / vp**2
  enddo

  ! Globe populates this table only on its gravity-disabled path.
  d_ln_density_dr(:) = 0.d0

  ! Extend fluid values into both solids using samples 3 km inside the
  ! outer core. This also covers points moved by boundary topography.
  lower_km = RICB / 1000.d0 + 3.d0
  upper_km = RCMB / 1000.d0 - 3.d0
  lower = nint(lower_km / R_PLANET_KM * NRAD_GRAVITY / range_max)
  upper = nint(upper_km / R_PLANET_KM * NRAD_GRAVITY / range_max)
  if (lower < 1 .or. upper > NRAD_GRAVITY .or. lower >= upper) return

  do i = 1, NRAD_GRAVITY
    radius_km = dble(i) / dble(NRAD_GRAVITY) * range_max * R_PLANET_KM
    if (radius_km > upper_km) minus_rho_g_over_kappa_fluid(i) = minus_rho_g_over_kappa_fluid(upper)
    if (radius_km < lower_km) minus_rho_g_over_kappa_fluid(i) = minus_rho_g_over_kappa_fluid(lower)
  enddo
  globe_evaluator_gravity_tables = GLOBE_EVALUATOR_OK
end function globe_evaluator_gravity_tables
