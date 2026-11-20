/*
 * globe_attenuation_reference.h -- TEST ONLY C ABI for independent
 * SPECFEM3D_GLOBE attenuation references.
 *
 * Implemented by globe_attenuation_reference.F90 and built into the
 * globe_evaluator::test_reference target, which is never installed and is not
 * linked into the production specfem::globe library. Inputs and outputs are in
 * physical SI units (scale_t = 1).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Independent globe Q fit and physical-unit coefficients.
 * Array lengths are N_SLS; rates are the modulus-free factor_common values.
 */
void globe_evaluator_attenuation_reference(double fmin, double fmax, double q,
    double fc, double f0, double *tau_sigma, double *tau_epsilon,
    double *relaxed_scale, double *unrelaxed_ratio, double *rates);

/* Full independent globe attenuation setup for Cartesian cij. */
void globe_evaluator_attenuate_cij(double theta, double phi, double fmin,
    double fmax, double q, double fc, double f0, const double *input,
    double *output, double *muvstore, double *rates);

/* Unrotated five-coefficient inner-core attenuation setup. */
void globe_evaluator_attenuate_inner_core(double fmin, double fmax, double q,
    double fc, double f0, const double *input, double *output,
    double *muvstore, double *rates);

/* Globe's radial unrelaxed-moduli shift; 21 Voigt entries. */
void globe_evaluator_shift_cij(double theta, double phi, double increment,
                               const double *input, double *output);

#ifdef __cplusplus
} /* extern "C" */
#endif
