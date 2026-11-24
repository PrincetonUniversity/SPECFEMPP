/*
 * globe_attenuation_reference.h -- TEST ONLY C ABI for SPECFEM3D_GLOBE's
 * anisotropic modulus shift.
 *
 * Implemented by globe_attenuation_reference.F90 and built into the
 * globe_evaluator::test_reference target, which is never installed and is not
 * linked into the production specfem::globe library. Inputs and outputs are in
 * physical SI units (scale_t = 1). The attenuation fit is supplied by the
 * caller, so these references check the tensor algebra only.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Two-stage globe shift of Cartesian cij (21 Voigt entries) at a point of
 * colatitude theta and longitude phi. rates (length N_SLS) holds modulus-free
 * memory coefficients on input and modulus-weighted ones on output.
 */
void globe_evaluator_attenuate_cij(double theta, double phi,
    double relaxed_scale, double ratio, const double *input, double *output,
    double *muvstore, double *rates);

/* Unrotated five-coefficient inner-core shift; same conventions. */
void globe_evaluator_attenuate_inner_core(double relaxed_scale, double ratio,
    const double *input, double *output, double *muvstore, double *rates);

/* Globe's radial unrelaxed-moduli shift; 21 Voigt entries. */
void globe_evaluator_shift_cij(double theta, double phi, double increment,
                               const double *input, double *output);

#ifdef __cplusplus
} /* extern "C" */
#endif
