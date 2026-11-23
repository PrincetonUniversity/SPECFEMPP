! TEST ONLY: SPECFEM3D_GLOBE's anisotropic modulus shift
! (src/specfem3D/prepare_attenuation.f90 and prepare_elastic_elements.F90 at
! 9c312cb2c991b47484a7f302775f4f01ed9470f8), adapted to physical SI units.
!
! The attenuation fit itself (relaxed scale, unrelaxed ratio and modulus-free
! memory coefficients) is supplied by the caller: these routines check the
! tensor algebra of the shift, independently of the formulation of the fit.

! TEST ONLY: double-precision reference for prepare_elastic_elements.F90's
! full anisotropic radial round trip. Uses the vendored globe rotations.
subroutine globe_evaluator_shift_cij(theta, phi, increment, input, output) &
    bind(C, name="globe_evaluator_shift_cij")
  use iso_c_binding, only: c_double
  implicit none
  real(c_double), value, intent(in) :: theta, phi, increment
  real(c_double), intent(in) :: input(21)
  real(c_double), intent(out) :: output(21)
  real(c_double) :: unused
  call globe_reference_shift_cij(theta, phi, increment, input, output, unused)
end subroutine globe_evaluator_shift_cij

subroutine globe_reference_shift_cij(theta, phi, increment, input, output, muvstore)
  use iso_c_binding, only: c_double
  implicit none
  real(c_double), intent(in) :: theta, phi, increment
  real(c_double), intent(in) :: input(21)
  real(c_double), intent(out) :: output(21), muvstore
  real(c_double) :: d(21), a, n, l, f, eta, muv, muh
  call rotate_tensor_global_to_radial(theta, phi, &
      d(1),d(2),d(3),d(4),d(5),d(6),d(7),d(8),d(9),d(10),d(11), &
      d(12),d(13),d(14),d(15),d(16),d(17),d(18),d(19),d(20),d(21), &
      input(1),input(2),input(3),input(4),input(5),input(6),input(7), &
      input(8),input(9),input(10),input(11),input(12),input(13),input(14), &
      input(15),input(16),input(17),input(18),input(19),input(20),input(21))
  a = 0.125d0 * (3.d0*d(1) + 3.d0*d(7) + 2.d0*d(2) + 4.d0*d(21))
  n = 0.125d0 * (d(1) + d(7) - 2.d0*d(2) + 4.d0*d(21))
  l = 0.5d0 * (d(16) + d(19))
  f = 0.5d0 * (d(3) + d(8))
  eta = f / (a - 2.d0*l)
  muv = l * increment
  muh = n * increment
  d(1) = d(1) + (4.d0/3.d0)*muh
  d(2) = d(2) - (2.d0/3.d0)*muh
  d(3) = d(3) + eta*((4.d0/3.d0)*muh - 2.d0*muv)
  d(7) = d(7) + (4.d0/3.d0)*muh
  d(8) = d(8) + eta*((4.d0/3.d0)*muh - 2.d0*muv)
  d(12) = d(12) + (4.d0/3.d0)*muv
  d(16) = d(16) + muv
  d(19) = d(19) + muv
  d(21) = d(21) + muh
  muvstore = d(16)
  call rotate_tensor_radial_to_global(theta, phi, &
      d(1),d(2),d(3),d(4),d(5),d(6),d(7),d(8),d(9),d(10),d(11), &
      d(12),d(13),d(14),d(15),d(16),d(17),d(18),d(19),d(20),d(21), &
      output(1),output(2),output(3),output(4),output(5),output(6),output(7), &
      output(8),output(9),output(10),output(11),output(12),output(13),output(14), &
      output(15),output(16),output(17),output(18),output(19),output(20),output(21))
end subroutine globe_reference_shift_cij

! Two-stage shift of a Cartesian tensor: reference to relaxed, then relaxed
! to unrelaxed, each in the radial frame. rates holds modulus-free memory
! coefficients on input and modulus-weighted ones on output.
subroutine globe_evaluator_attenuate_cij(theta, phi, relaxed_scale, ratio, &
    input, output, muvstore, rates) bind(C, name="globe_evaluator_attenuate_cij")
  use iso_c_binding, only: c_double
  use constants, only: N_SLS
  implicit none
  real(c_double), value, intent(in) :: theta, phi, relaxed_scale, ratio
  real(c_double), intent(in) :: input(21)
  real(c_double), intent(out) :: output(21), muvstore
  real(c_double), intent(inout) :: rates(N_SLS)
  real(c_double) :: relaxed(21), unused
  call globe_reference_shift_cij(theta,phi,relaxed_scale-1.d0,input,relaxed,unused)
  call globe_reference_shift_cij(theta,phi,ratio-1.d0,relaxed,output,muvstore)
  ! prepare_elastic_elements.F90:536 divides factor_common by the unrelaxed
  ! ratio before compute_element_att_memory multiplies by the unrelaxed
  ! muvstore.
  rates(:) = rates(:) / ratio * muvstore
end subroutine globe_evaluator_attenuate_cij

! TEST ONLY: five-coefficient inner-core shift. No radial rotation.
subroutine globe_evaluator_attenuate_inner_core(relaxed_scale, ratio, &
    input,output,muvstore,rates) &
    bind(C, name="globe_evaluator_attenuate_inner_core")
  use iso_c_binding, only: c_double
  use constants, only: N_SLS
  implicit none
  real(c_double), value, intent(in) :: relaxed_scale, ratio
  real(c_double), intent(in) :: input(21)
  real(c_double), intent(out) :: output(21),muvstore
  real(c_double), intent(inout) :: rates(N_SLS)
  real(c_double) :: s,g,mul,c11,c12,c13,c33,c44
  s = relaxed_scale
  g = ratio
  ! prepare_attenuation.f90:367-382
  mul = input(16)
  c11 = input(1) + (4.d0/3.d0)*(s-1.d0)*mul
  c12 = input(2) - (2.d0/3.d0)*(s-1.d0)*mul
  c13 = input(3) - (2.d0/3.d0)*(s-1.d0)*mul
  c33 = input(12) + (4.d0/3.d0)*(s-1.d0)*mul
  c44 = input(16) + (s-1.d0)*mul
  ! prepare_elastic_elements.F90:628-644
  mul = c44*(g-1.d0)
  c11 = c11 + (4.d0/3.d0)*mul
  c12 = c12 - (2.d0/3.d0)*mul
  c13 = c13 - (2.d0/3.d0)*mul
  c33 = c33 + (4.d0/3.d0)*mul
  c44 = c44 + mul
  output(:) = 0.d0
  output(1) = c11
  output(2) = c12
  output(3) = c13
  output(7) = c11
  output(8) = c13
  output(12) = c33
  output(16) = c44
  output(19) = c44
  output(21) = 0.5d0*(c11-c12)
  muvstore = c44
  ! prepare_elastic_elements.F90:754
  rates(:) = rates(:)/g*muvstore
end subroutine globe_evaluator_attenuate_inner_core
