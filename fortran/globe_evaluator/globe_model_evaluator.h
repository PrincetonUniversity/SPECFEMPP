/*
 * globe_model_evaluator.h -- C ABI for the SPECFEM3D_GLOBE model evaluator.
 *
 * Implemented by globe_model_evaluator.F90. See issue #2001.
 *
 * Contracts:
 *   - INPUT coordinates and radii are SI metres. The Fortran side
 *     non-dimensionalizes by R_PLANET.
 *   - OUTPUT material is in the globe's NON-DIMENSIONAL units, not SI:
 *     density is near 1 and velocities near 2. This is deliberate -- returning
 *     the catalog's own numbers untouched is what makes them bit-for-bit the
 *     mesher's. Use globe_evaluator_scales to convert.
 *   - globe_evaluator_init must be called exactly once, before any
 *     globe_evaluator_get_element call. The underlying catalog holds global
 *     module state, so a second init is refused rather than honoured.
 *   - Call single-threaded, at setup only. Never from a solver hot loop.
 *   - Output arrays are indexed over the element's GLL points in the mesher's
 *     order (k outermost, i innermost) and must be at least
 *     NGLLX*NGLLY*NGLLZ long -- query the sizes with globe_evaluator_dims rather
 *     than assuming 125.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes, mirroring globe_evaluator_par in globe_model_evaluator.F90. */
#define GLOBE_EVALUATOR_OK 0
#define GLOBE_EVALUATOR_ALREADY_INITIALIZED 1
#define GLOBE_EVALUATOR_NOT_INITIALIZED 2
#define GLOBE_EVALUATOR_UNSUPPORTED_MODEL 3
#define GLOBE_EVALUATOR_IMAIN_OPEN_FAILED 4
#define GLOBE_EVALUATOR_BAD_ARGUMENT 5

/*
 * Reports the quadrature sizes the catalog was compiled with. Call this and
 * check it against your own NGLL before trusting any returned values: a
 * mismatch means the evaluator is sampling a different rule than the caller.
 */
void globe_evaluator_dims(int *ngllx, int *nglly, int *ngllz, int *n_sls);

/*
 * One-time model setup. Identifies the model by name only -- no globe Par_file
 * is read; the catalog re-derives every flag and discontinuity radius from the
 * name via get_model_parameters().
 *
 * model_name / imain_path are NOT required to be NUL-terminated; the lengths
 * are explicit. Pass imain_path_len == 0 to send the catalog's log output to
 * /dev/null.
 *
 * In MPI builds, comm_f must be a Fortran communicator handle returned by
 * MPI_Comm_c2f(). Serial builds pass the serial communicator value zero.
 *
 * Boolean toggles are 0 (false) / nonzero (true).
 *
 * min/max_attenuation_period are in seconds and consulted only when
 * attenuation is nonzero. They are NOT derivable from the model name -- the
 * mesher computes them in rcp_set_compute_parameters, which the evaluator does not
 * call -- so the database must carry them. Supplying a non-positive or inverted
 * band with attenuation on returns GLOBE_EVALUATOR_BAD_ARGUMENT rather than letting
 * the catalog abort the process.
 *
 * Returns GLOBE_EVALUATOR_OK, or a nonzero status above.
 *
 * WARNING: an unrecognized model name cannot be reported. The catalog
 * terminates the process with a bare Fortran `stop` in that case
 * (get_model_parameters.F90:978-982), as it does for several flag-consistency
 * violations. GLOBE_EVALUATOR_UNSUPPORTED_MODEL is returned only for models that
 * parse successfully but require the per-GLL indexing a position-only evaluator
 * cannot provide (MODEL_GLL, CEM, HETEROGEN_3D_MANTLE, ATTENUATION_GLL, EMC).
 */
int globe_evaluator_init(const char *model_name, int name_len,
                      const char *imain_path, int imain_path_len,
                      int planet_type, int nchunks, int nex_xi, int nex_eta,
                      int ellipticity, int topography, int oceans,
                      int attenuation, int gravity, int rotation,
                      double min_attenuation_period,
                      double max_attenuation_period, int comm_f);

/*
 * Reports the factors converting this evaluator's outputs to SI:
 *
 *   rho_SI = rho * density_scale        (density_scale = RHOAV)
 *   v_SI   = v   * velocity_scale       (= R_PLANET * sqrt(PI*GRAV*RHOAV))
 *   r_SI   = r   * length_scale         (= R_PLANET)
 *
 * eta and Qmu/Qkappa are dimensionless; cij scales as density * velocity^2.
 * Requires a configured evaluator; returns GLOBE_EVALUATOR_NOT_INITIALIZED otherwise
 * and writes zeros.
 */
int globe_evaluator_scales(double *length_scale, double *density_scale,
                        double *velocity_scale);

/*
 * Releases what the evaluator owns (its log unit and the topo/bathy array) and
 * clears the initialization guard so a different model may be configured.
 * Idempotent; always returns GLOBE_EVALUATOR_OK.
 *
 * Does NOT deallocate state owned by individual model_*.f90 files -- upstream
 * exposes no teardown hooks for those. Re-initializing is therefore reliable
 * for the analytic 1D reference models and may fail inside the catalog for 3D
 * models that allocate. The production path configures once per process.
 */
int globe_evaluator_finalize(void);

/*
 * Evaluates the model at every GLL point of one element.
 *
 * Per-element rather than per-point by necessity, not for speed: the catalog's
 * `moho`/`sediment` state is element-scoped and genuinely carries across GLL
 * points (meshfem3D_models.F90:1300 returns before the reset at :1311, and the
 * stale value is read at :1745). Evaluating point-by-point from outside would
 * not reproduce the mesher for ATTENUATION_3D models.
 *
 *   iregion_code    IREGION_CRUST_MANTLE=1, OUTER_CORE=2, INNER_CORE=3
 *   idoubling       the element's radial-zone IFLAG_* value
 *   rmin_si/rmax_si the element's radial shell bounds, SI metres
 *   xyz_si          [3 * npoints], point-major: {x0,y0,z0, x1,y1,z1, ...}
 *   cij             [21 * npoints], point-major, Voigt order
 *                   c11,c12,c13,c14,c15,c16,c22,c23,c24,c25,c26,
 *                   c33,c34,c35,c36,c44,c45,c46,c55,c56,c66
 *   is_anisotropic  1 if the caller should store this element from cij
 *
 * vs_iso is exactly 0.0 in the outer core -- that is the acoustic tag, and it
 * is worth cross-checking against the database's own medium_tag.
 *
 * Returns GLOBE_EVALUATOR_OK or GLOBE_EVALUATOR_NOT_INITIALIZED. Note that an
 * inconsistent idoubling aborts the process from inside the catalog
 * (get_model_check_idoubling); converting that to a status code is a tracked
 * follow-up.
 */
int globe_evaluator_get_element(int iregion_code, int idoubling, double rmin_si,
                             double rmax_si, int elem_in_crust,
                             int elem_in_mantle, const double *xyz_si,
                             double *rho, double *vpv, double *vph, double *vsv,
                             double *vsh, double *eta, double *vp_iso,
                             double *vs_iso, double *qmu, double *qkappa,
                             double *cij, double *gc_prime, double *gs_prime,
                             int *is_anisotropic);

/*
 * TEST ONLY -- not part of the production evaluator contract.
 *
 * Evaluates the 1D PREM reference directly (model_prem_iso / model_prem_aniso,
 * selected on TRANSVERSE_ISOTROPY exactly as meshfem3D_models_get1D_val does)
 * and applies the same Voigt reduction as globe_evaluator_get_element. Exists so
 * tests can check the evaluator against the catalog's own reference routine rather
 * than against a hand-written table of expected values.
 */
int globe_evaluator_prem_reference(double r_si, int idoubling, int iregion_code,
                                double *rho, double *vpv, double *vph,
                                double *vsv, double *vsh, double *eta,
                                double *vp_iso, double *vs_iso, double *qkappa,
                                double *qmu);

#ifdef __cplusplus
} /* extern "C" */
#endif
