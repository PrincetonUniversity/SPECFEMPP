#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace specfem {
namespace globe_model {

/**
 * @brief Quadrature sizes the SPECFEM3D_GLOBE model catalog was compiled with.
 *
 * Check these against SPECFEM++'s own NGLL before trusting any evaluated
 * material: a mismatch means the evaluator samples a different quadrature rule
 * than the caller, which produces plausible but wrong properties.
 */
struct Dims {
  int ngllx = 0;
  int nglly = 0;
  int ngllz = 0;
  int n_sls = 0;

  /** @brief Number of GLL points per element, @f$ N_x N_y N_z @f$. */
  std::size_t points_per_element() const {
    return static_cast<std::size_t>(ngllx) * static_cast<std::size_t>(nglly) *
           static_cast<std::size_t>(ngllz);
  }
};

/**
 * @brief Resolved model selection, replayed into the Fortran catalog.
 *
 * The model is identified by @ref model_name alone; the catalog re-derives
 * every model flag and every discontinuity radius from it. SPECFEM++ therefore
 * never parses the globe `Par_file`, and there is exactly one place a model is
 * chosen (the mesher, which writes this name into the database).
 */
struct ModelConfig {
  std::string model_name;

  /** @brief Path for the catalog's own log output. Empty sends it to
   * /dev/null. */
  std::string log_path;

  int planet_type = 1; ///< IPLANET_EARTH
  int nchunks = 6;
  int nex_xi = 64;
  int nex_eta = 64;

  bool ellipticity = false;
  bool topography = false;
  bool oceans = false;
  bool attenuation = false;
  bool gravity = false;
  bool rotation = false;

  /// @name Attenuation period band, in seconds
  /// @{
  /// Consulted only when @ref attenuation is set. Unlike everything else here,
  /// this is *not* derivable from @ref model_name: the mesher computes it in
  /// `rcp_set_compute_parameters`, which the evaluator deliberately does not
  /// call, so the database has to carry it. The defaults below are typical
  /// global values and a placeholder -- production must take them from the
  /// mesher.
  double min_attenuation_period = 20.0;
  double max_attenuation_period = 1000.0;
  /// @}
};

/**
 * @brief Factors converting the evaluator's output to SI.
 *
 * The catalog works in -- and @ref ElementProperties therefore reports -- the
 * globe's non-dimensional units, where density is near 1 and velocities near 2.
 * That is deliberate: handing back the catalog's own numbers untouched is what
 * makes them bit-for-bit the mesher's, whereas converting inside the evaluator
 * would add a rounding step the mesher does not have. Callers re-dimensionalize
 * here, in one place.
 *
 * \f[
 *   \rho_{SI} = \rho \cdot \rho_{scale}, \qquad
 *   v_{SI} = v \cdot v_{scale}, \qquad
 *   r_{SI} = r \cdot L
 * \f]
 *
 * with \f$ \rho_{scale} = \overline{\rho} \f$ and
 * \f$ v_{scale} = R \sqrt{\pi G \overline{\rho}} \f$.
 *
 * @ref ElementProperties::eta and the quality factors are dimensionless;
 * `cij` scales as density times velocity squared.
 */
struct Scales {
  double length = 0.0;   ///< R_PLANET, in metres
  double density = 0.0;  ///< RHOAV, in kg/m^3
  double velocity = 0.0; ///< R_PLANET * sqrt(PI * GRAV * RHOAV), in m/s

  /** @brief Converts a non-dimensional elastic modulus to Pa. */
  double modulus() const { return density * velocity * velocity; }
};

/**
 * @brief Material properties at every GLL point of one spectral element.
 *
 * All arrays are sized @ref Dims::points_per_element and indexed in the
 * mesher's GLL order (@f$ k @f$ outermost, @f$ i @f$ innermost).
 *
 * @warning Values are **non-dimensional**, not SI. See @ref Scales.
 */
struct ElementProperties {
  /// @name Transversely isotropic Love velocities and density
  /// @{
  std::vector<double> rho, vpv, vph, vsv, vsh, eta;
  /// @}

  /// @name Isotropic Voigt averages, for isotropic media and the CFL estimate
  /// @{
  /// In the outer core @ref vs_iso is exactly zero -- the acoustic tag.
  std::vector<double> vp_iso, vs_iso;
  /// @}

  /// @name Attenuation quality factors. SLS coefficients are the caller's job.
  /// @{
  std::vector<double> qmu, qkappa;
  /// @}

  /**
   * @brief The 21 elastic coefficients, point-major: `cij[21 * ipoint + n]`.
   *
   * Voigt order: c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26, c33,
   * c34, c35, c36, c44, c45, c46, c55, c56, c66.
   */
  std::vector<double> cij;

  /// @name Azimuthal anisotropy, already normalized by the 1D reference
  /// @f$ \mu_0 @f$
  /// @{
  std::vector<double> gc_prime, gs_prime;
  /// @}

  /** @brief Whether the caller should store this element from @ref cij. */
  bool is_anisotropic = false;
};

/**
 * @brief Evaluates the SPECFEM3D_GLOBE model catalog at arbitrary positions.
 *
 * Wraps the Fortran catalog (~90 `model_*.f90` files) so SPECFEM++ can
 * regenerate material at its own GLL points instead of reimplementing it. See
 * issue #2001.
 *
 * Two invariants this class exists to enforce, both of which produce silently
 * wrong results rather than errors when violated:
 *
 * 1. **Single instance.** The catalog's configuration lives in Fortran module
 *    variables, so two `Evaluator` objects would share one model. Constructing
 * a second while one is alive throws.
 * 2. **Setup only, single-threaded.** Several catalog routines cache state in
 *    `save` variables. Never call @ref evaluate_element from a solver loop or
 *    from multiple threads.
 *
 * The caller must additionally evaluate at the mesher's *reference* coordinates
 * (spherical, Moho-stretched -- before topography and ellipticity displace the
 * points), because that is where the mesher sampled the model. Evaluating at
 * final deformed positions samples the wrong depth near the surface. That
 * cannot be checked here.
 */
class Evaluator {
public:
  /**
   * @brief Configures the catalog. Collective over the active communicator.
   *
   * @param config Resolved model selection, from the database header.
   * @throws std::runtime_error if the catalog rejects the configuration --
   *         notably for models needing per-GLL indexing the evaluator cannot
   *         provide (`MODEL_GLL`, CEM, `HETEROGEN_3D_MANTLE`,
   *         `ATTENUATION_GLL`, EMC), or if an `Evaluator` already exists.
   *
   * @warning An unrecognized model name cannot be reported as an exception:
   *          the Fortran catalog terminates the process with a bare `stop`.
   */
  explicit Evaluator(const ModelConfig &config);

  /**
   * @brief Releases the catalog's configuration so another model may be built.
   *
   * @warning Only the evaluator's own resources are released. State allocated
   * by individual `model_*.f90` files cannot be freed -- upstream exposes no
   * teardown hooks -- so destroying an `Evaluator` and building one for a
   * *different 3D model* in the same process may fail inside the catalog. The
   * analytic 1D reference models allocate nothing and re-initialize cleanly.
   * Production configures once per process.
   */
  ~Evaluator();

  Evaluator(const Evaluator &) = delete;
  Evaluator &operator=(const Evaluator &) = delete;
  Evaluator(Evaluator &&) = delete;
  Evaluator &operator=(Evaluator &&) = delete;

  /**
   * @brief Evaluates the model at one element's GLL points.
   *
   * Per-element rather than per-point by necessity: the catalog's crustal
   * `moho`/`sediment` state is element-scoped and genuinely carries across GLL
   * points, so evaluating point-by-point would not reproduce the mesher for
   * 3D-attenuation models.
   *
   * @param iregion_code 1 = crust/mantle, 2 = outer core, 3 = inner core.
   * @param idoubling The element's radial-zone `IFLAG_*` value.
   * @param rmin_si,rmax_si The element's radial shell bounds, in metres.
   * @param elem_in_crust,elem_in_mantle Per-element flags set by Moho
   *        stretching. Both come from the database; they cannot be recomputed
   *        here without the crustal dataset.
   * @param xyz_si Point-major coordinates in metres: `{x0,y0,z0, x1,y1,z1,
   *        ...}`, of length `3 * dims().points_per_element()`.
   * @return Material at each GLL point.
   * @throws std::invalid_argument if @p xyz_si has the wrong length.
   * @throws std::runtime_error if the catalog reports a failure.
   *
   * @warning An inconsistent @p idoubling aborts the process from inside the
   *          Fortran catalog rather than returning an error.
   */
  ElementProperties evaluate_element(int iregion_code, int idoubling,
                                     double rmin_si, double rmax_si,
                                     bool elem_in_crust, bool elem_in_mantle,
                                     const std::vector<double> &xyz_si) const;

  /** @brief Quadrature sizes the catalog was compiled with. */
  static Dims dims();

  /**
   * @brief Factors converting evaluated material to SI.
   * @throws std::runtime_error if no @ref Evaluator is configured -- the scales
   *         depend on the planet constants that construction resolves.
   */
  static Scales scales();

  /** @brief Whether an @ref Evaluator is currently alive. */
  static bool is_active();

private:
  static bool is_active_;
};

/**
 * @brief Evaluates the 1D PREM reference directly. **Test use only.**
 *
 * Bypasses the evaluator path and calls the catalog's own PREM routines, so
 * tests can check @ref Evaluator against the reference implementation rather
 * than against a hand-written table of expected values. Not part of the
 * production contract; requires a live @ref Evaluator.
 */
struct ReferencePoint {
  double rho, vpv, vph, vsv, vsh, eta;
  double vp_iso, vs_iso;
  double qkappa, qmu;
};

/** @copydoc ReferencePoint */
ReferencePoint prem_reference(double r_si, int idoubling, int iregion_code);

} // namespace globe_model
} // namespace specfem
