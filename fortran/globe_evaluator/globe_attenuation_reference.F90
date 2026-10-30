! TEST ONLY: reference routines from SPECFEM3D_GLOBE
! src/specfem3D/get_attenuation.f90 at 9c312cb2c991b47484a7f302775f4f01ed9470f8
! Only symbol names and scale_t=1 (physical SI units) are adapted.

  subroutine globe_reference_get_attenuation_property_values(tau_s, tau_e, factor_common, one_minus_sum_beta)

  use constants

  implicit none

  double precision, dimension(N_SLS),intent(in) :: tau_s, tau_e
  double precision, dimension(N_SLS),intent(out) :: factor_common
  double precision,intent(out) ::  one_minus_sum_beta

  ! local parameters
  double precision, dimension(N_SLS) :: tauinv,beta

  ! debugging
  !double precision :: omsb1,omsb2

  integer :: i

  tauinv(:) = 0.d0
  where(abs(tau_s(:)) > 0.d0) tauinv(:) = 1.0d0 / tau_s(:)

  beta(:) = 1.0d0 - tau_e(:) * tauinv(:)     ! 1 - tau_e / tau_s

  ! factor to scale from relaxed to unrelaxed moduli: see e.g. Komatitsch & Tromp 1999, eq. (7)
  one_minus_sum_beta = 1.0d0
  do i = 1,N_SLS
     one_minus_sum_beta = one_minus_sum_beta - beta(i)
  enddo

  ! note: Komatitsch & Tromp 1999, eq. (7) (or Komatitsch 2002 eq. 10), defines the unrelaxed modulus as
  !          C_unrelaxed = C_relaxed * [ 1 - sum(1 - tau_eps/tau_sigma) ]
  !       with a scaling factor one_minus_sum_beta = [ 1 - sum(1 - tau_eps/tau_sigma) ]
  !
  !       this is different to Liu (1976) who mentions
  !          M_U = M_R / [ 1 - sum( (tau_eps - tau_sigma)/tau_eps ) ] = M_R / [ 1 - sum(1 - tau_sigma/tau_eps) ]
  !       with a scaling factor = 1 / [ 1 - sum(1 - tau_sigma/tau_eps) ]
  !
  ! debugging - to compare these two factors:
  !if (.false.) then
  !  ! factor to scale from relaxed to unrelaxed moduli: see Komatitsch
  !  omsb1 = 1.0d0
  !  do i = 1,N_SLS
  !    omsb1  = omsb1 - (1.d0 - tau_e(i)/tau_s(i))
  !  enddo
  !  ! factor to scale from relaxed to unrelaxed moduli: see Liu
  !  omsb2 = 1.0d0
  !  do i = 1,N_SLS
  !    omsb2 = omsb2 - (1.d0 - tau_s(i)/tau_e(i))
  !  enddo
  !  print *,'debug: one_minus_sum_beta1,2 = ',omsb1,1.d0/omsb2
  !endif
  ! results:
  ! Q ~    65: one_minus_sum_beta1,2 = 1.0648735275165400        1.0677815840121743   -> ratio = 0.997276
  ! Q ~   184: one_minus_sum_beta1,2 = 1.0233224020801992        1.0236877302481358   -> ratio = 0.999643
  ! Q ~ 19577: one_minus_sum_beta1,2 = 1.0002203256626838        1.0002203577661830   -> ratio = 0.999999967
  !
  ! -> for large enough Q, these factors are almost identical

  ! see Savage (2010), eq. (11) where the two terms with previous strain \epsilon' and current strain \epsilon
  ! both have a factor 2 \Delta M_i / \tau_{sigma_i}, where \Delta M_i is the modulus defect (M_unrelaxed - M_relaxed)_i
  !
  ! C_unrelaxed = C_relaxed * [ 1 - sum(1 - tau_eps/tau_sigma) ] = C_relaxed * [ 1 - sum(beta(:))]
  ! -> \Delta M_i =  (C_unrelaxed - C_relaxed)_i = C_relaxed * [1 - beta_i] - C_relaxed = C_relaxed * [1 - beta_i - 1]
  !               = C_relaxed * [ - beta_i] with beta_i = 1 - tau_e_i/tau_s_i
  ! (see also Komatitsch, 1999, eq. 8)

!ZN beware, here the expression differs from the strain used in memory variable equation (6) in D. Komatitsch and J. Tromp 1999,
!ZN here Brian Savage uses the engineering strain which are epsilon = 1/2*(grad U + (grad U)^T),
!ZN where U is the displacement vector and grad the gradient operator, i.e. there is a 1/2 factor difference between the two.
!ZN Both expressions are fine, but we need to keep in mind that if we have put the 1/2 factor there we need to remove it
!ZN from the expression in which we use the strain here in the code.
!ZN This is why here Brian Savage multiplies beta(:) * tauinv(:) by 2.0 to compensate for the 1/2 factor used before
  factor_common(:) = 2.0d0 * (- beta(:)) * tauinv(:)

  end subroutine globe_reference_get_attenuation_property_values


  subroutine globe_reference_get_attenuation_scale_factor(T_c_source, tau_mu, tau_sigma, Q_mu, scale_factor, scale_factor_relaxed, f0_reference)

  use constants, only: ZERO,ONE,TWO,PI,TWO_PI,N_SLS,myrank
  ! Physical units in this reference wrapper: scale_t = 1.

  implicit none

  double precision,intent(in) :: T_c_source  ! center period
  double precision,intent(in) :: Q_mu
  double precision, dimension(N_SLS),intent(in) :: tau_mu, tau_sigma

  double precision,intent(out) :: scale_factor,scale_factor_relaxed

  ! model reference frequency
  double precision,intent(in) :: f0_reference

  ! local parameters
  double precision :: f_c_source, w_c_source,f_0_model
  double precision :: factor_scale_mu0,factor_scale_mu
  double precision :: a_val, b_val, denom
  double precision :: big_omega
  integer :: i

  ! compute central angular frequency of source (non dimensionalized)
  if (T_c_source > 0.d0) then
    f_c_source = ONE / T_c_source
  else
    f_c_source = 0.d0
  endif
  w_c_source = TWO_PI * f_c_source

  ! model reference frequency (e.g., PREM reference of 1 second)
  ! non-dimensionalizes frequency
  f_0_model = f0_reference       ! original f_0_prem = ONE / ( ONE / scale_t) or f_reference = ATTENUATION_f0_REFERENCE

!--- quantity by which to scale mu_0 to get mu
! this formula can be found for instance in
! Liu, H. P., Anderson, D. L. and Kanamori, H., Velocity dispersion due to
! anelasticity: implications for seismology and mantle composition,
! Geophys. J. R. Astron. Soc., vol. 47, pp. 41-58 (1976)
! and in Aki, K. and Richards, P. G., Quantitative seismology, theory and methods,
! W. H. Freeman, (1980), second edition, sections 5.5 and 5.5.2, eq. (5.81) p. 170
  if (f_0_model > 0.d0) then
    factor_scale_mu0 = ONE + TWO * log(f_c_source / f_0_model) / (PI * Q_mu)
  else
    factor_scale_mu0 = ONE
  endif

  !--- compute a, b and Omega parameters
  a_val = ONE
  b_val = ZERO

  do i = 1,N_SLS
    denom = 1.d0 + w_c_source * w_c_source * tau_mu(i) * tau_mu(i)
    if (denom /= 0.d0) then
      a_val = a_val - w_c_source * w_c_source * tau_mu(i) * (tau_mu(i) - tau_sigma(i)) / denom
      b_val = b_val + w_c_source * (tau_mu(i) - tau_sigma(i)) / denom
    endif
  enddo

  if (a_val /= 0.d0) then
    big_omega = a_val*(sqrt(1.d0 + b_val*b_val/(a_val*a_val))-1.d0)
  else
    big_omega = 0.d0
  endif

  !--- quantity by which to scale mu to get mu_relaxed
  !
  ! note: by assuming that the input velocities are given at a reference frequency (f_0_model),
  !       we first shift them to the center frequency (f_c_source or angular frequency w_c_source) of the absorption-band
  !       and determine the corresponding relaxed moduli based on:
  !         M_relaxed = M(w_c_source) * B**2 / Omega
  !
  !       see Liu 1976 eq. (20), where here we use the expression for modulus, Omega = Omega(w_c_source), B = B(w_c_source).
  !       eq. 20 would be given for phase velocities v_p = sqrt( M(w_c_source)/rho) and v_e = sqrt( M_relaxed/rho)
  !
  !       This shifting to the center frequency of the absorption-band also guarantees that the relaxation times tau_**
  !       are optimal (for a good Q approximation within the band) and thus should give a reliable scaling to
  !       the relaxed modulus as well.
  !
  !       For the stiffness, we will need the moduli to be scaled to unrelaxed ones for the computations of the stress tensor:
  !         T = C_unrelaxed : grad(s) - sum( R_l)  (see e.g. Komatitsch, 1999, eq. 5)
  !
  !       A relaxed modulus can be seen as the modulus at zero frequency, unrelaxed modulus at infinite frequency (instantaneous)

  !       These factors together with typos and the debate about a missing 1/L factor leads to some confusion.
  !       It has been shown that as long as one uses a consistent set of equations, the 1/L factor expressions are equivalent
  !       to the ones used by Liu (1976) without it and implemented here.
  !
  !       Unfortunately, note that Liu (1976) starts with expressions for compliance J = 1/M rather than M, however turns
  !       to complex modulus M_c which again is a stress relaxation function sigma(t) = M_c(t) epsilon(t).
  !
  !       For Liu 1976, the quality factor Q is defined as tan delta = 1/Q = B/A
  !
  !       We further assume the reference velocities are given at a certain reference frequency - and not as relaxed or unrelaxed
  !       velocities, thus velocities need to be shifted to a absorption-band and the relaxed/unrelaxed moduli must be determined.
  !       This is a convention and could also be defined in different ways, see e.g. Carcione (1993) who defines
  !       the elastic model given by relaxed values.
  !
  if (big_omega /= 0.d0) then
    factor_scale_mu = b_val * b_val / (TWO * big_omega)
  else
    factor_scale_mu = ONE
  endif

  !--- total factor by which to scale mu0
  scale_factor = factor_scale_mu * factor_scale_mu0

  ! factor to get relaxed modulus from shifted value
  scale_factor_relaxed = factor_scale_mu

  ! note:
  !  * if reference frequency and center frequency are the same, i.e., f_0_model == f_c_source, then factor_scale_mu0 == 1
  !  * for f_c_source == 0, i.e. at zero frequency, the scaling to relaxed factor_scale_mu should be 1
  !  * Liu 1976 defines tan delta = 1/Q = B/A -> Q = A/B gives the approximated Q value for the given SLS
  !print *,'debug: factor scale mu0,mu = ',factor_scale_mu0,factor_scale_mu,'approximated Q = A/B',a_val/b_val

  !--- check that the correction factor is close to one
  ! note: for very coarse global simulations, the reference frequency (usually 1Hz for global models)
  !       and center frequency of the simulation can be far apart.
  !       the scale_factor limiting range here is somewhat set arbitrarily, and enlarged to [0.5,1.5]
  !       to allow for very coarse global simulation tests with sponge attenuation.
  if (scale_factor < 0.5d0 .or. scale_factor > 1.5d0) then
    print *,'Error: incorrect scale factor: ', scale_factor
    print *,'  scale factor: ', scale_factor,' should be between 0.5 to 1.5'
    print *,'  factor scale_mu = ',factor_scale_mu,' factor scale_mu0 = ',factor_scale_mu0
    print *,'  Q value = ',Q_mu
    print *,'  central period = ',T_c_source,' frequency = ',f_c_source
    print *,'  model frequency = ',f_0_model
    print *,'Please check your reference frequency ATTENUATION_f0_REFERENCE in file constants.h'
    call exit_MPI(myrank,'incorrect correction factor in attenuation model')
  endif

  end subroutine globe_reference_get_attenuation_scale_factor


! Independently fit Q using the actual vendored globe simplex, then compute
! the relaxed scale, unrelaxed ratio and memory coefficients above.
subroutine globe_evaluator_attenuation_reference(fmin, fmax, q, fc, f0, &
    tau_s, tau_e, relaxed_scale, ratio, rates) &
    bind(C, name="globe_evaluator_attenuation_reference")
  use iso_c_binding, only: c_double
  use constants, only: N_SLS
  implicit none
  real(c_double), value, intent(in) :: fmin, fmax, q, fc, f0
  real(c_double), intent(out) :: tau_s(N_SLS), tau_e(N_SLS)
  real(c_double), intent(out) :: relaxed_scale, ratio, rates(N_SLS)
  real(c_double) :: unused,effective_q
  ! model_attenuation_storage:517-518; fit the limited Q, not raw input Q.
  effective_q = q
  if (effective_q <= 0.d0 .or. effective_q > 9000.d0) effective_q = 9000.d0
  call attenuation_invert_by_simplex(1.d0/fmax, 1.d0/fmin, N_SLS, effective_q, tau_s, tau_e)
  call globe_reference_get_attenuation_property_values(tau_s, tau_e, rates, ratio)
  call globe_reference_get_attenuation_scale_factor(1.d0/fc, tau_e, tau_s, effective_q, &
      relaxed_scale, unused, f0)
end subroutine globe_evaluator_attenuation_reference
