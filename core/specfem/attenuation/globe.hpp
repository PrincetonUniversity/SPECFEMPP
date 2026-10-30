#pragma once

#include "compute_factors.hpp"
#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/band.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <map>
#include <stdexcept>

namespace specfem {
namespace attenuation {

/**
 * @brief SPECFEM3D_GLOBE attenuation formulation, used by every 3D mesh.
 *
 * Ports the Q fit (src/meshfem3D/model_attenuation.f90) and the modulus
 * factors (src/specfem3D/get_attenuation.f90). The 2D solver keeps the
 * 1/L-normalized routines in specfem::attenuation.
 *
 * Differences from those routines: the Maxwell real part is unnormalized,
 * \f$A=1-N+\sum_i\ldots\f$; the fit runs in double precision with
 * specfem::optimization::Fminsearch; Q is limited to 9000 and bucketed as in
 * globe's model_attenuation_storage; one_minus_sum_beta is
 * \f$1+\sum_i(\tau_{\epsilon_i}/\tau_{\sigma_i}-1)\f$; and the dispersion
 * factor is split into a reference-to-relaxed scale (Liu et al. 1976) and the
 * relaxed-to-unrelaxed ratio.
 */
namespace globe {

/// Globe's ATTENUATION_COMP_MAXIMUM.
constexpr type_real maximum_quality_factor = 9000;
/// Globe's 10**ATTENUATION_COMP_RESOLUTION.
constexpr int quality_factor_resolution = 10;

/**
 * @brief Apply globe's Q limit before fitting and dispersion correction.
 * @param q Input quality factor
 * @return q, or 9000 when q is nonpositive, infinite or above 9000
 * @throws std::invalid_argument if q is NaN
 */
inline type_real effective_quality_factor(const type_real q) {
  if (std::isnan(q)) {
    throw std::invalid_argument("Quality factor must not be NaN");
  }
  return q <= 0 || q > maximum_quality_factor ? maximum_quality_factor : q;
}

/**
 * @brief Storage-table index of globe's model_attenuation_storage.
 * @param q Quality factor
 * @return \f$\max(1, \lfloor 10\,Q_\text{eff} \rfloor)\f$
 */
inline int quality_factor_bucket(const type_real q) {
  const int key =
      static_cast<int>(static_cast<double>(effective_quality_factor(q)) *
                       quality_factor_resolution);
  return key < 1 ? 1 : key;
}

/**
 * @brief Globe's Q storage table: the first Q seen in a bucket represents it.
 *
 * Globe fits tau_epsilon once per bucket and replaces later Q values in the
 * same bucket with the stored one, which is then used for the dispersion
 * correction and written to the mesh files.
 */
class QualityFactorTable {
public:
  /**
   * @brief Return the representative Q of q's bucket, registering q if new.
   * @param q Quality factor
   * @return First effective Q stored in q's bucket
   */
  type_real representative(const type_real q) {
    return table_
        .try_emplace(quality_factor_bucket(q), effective_quality_factor(q))
        .first->second;
  }

private:
  std::map<int, type_real> table_;
};

/**
 * @brief Real and imaginary moduli of globe's generalized Maxwell solid.
 * @tparam NF Number of frequencies
 */
template <int NF> struct MaxwellFactors {
  Kokkos::View<double[NF], Kokkos::LayoutRight, Kokkos::HostSpace>
      real; ///< \f$A(\omega)\f$
  Kokkos::View<double[NF], Kokkos::LayoutRight, Kokkos::HostSpace>
      imag; ///< \f$B(\omega)\f$
};

/**
 * @brief Globe's attenuation_maxwell with a unit relaxed modulus.
 *
 * \f[A(\omega)=1-N_{SLS}+\sum_i
 * \frac{1+\omega^2\tau_{\epsilon_i}\tau_{\sigma_i}}
 * {1+\omega^2\tau_{\sigma_i}^2},\qquad
 * B(\omega)=\sum_i\frac{\omega(\tau_{\epsilon_i}-\tau_{\sigma_i})}
 * {1+\omega^2\tau_{\sigma_i}^2}.\f]
 *
 * @tparam NF Number of frequencies
 * @tparam N_SLS Number of standard linear solids
 * @param f Frequencies (Hz)
 * @param tau_s Stress relaxation times \f$\tau_\sigma\f$
 * @param tau_eps Strain relaxation times \f$\tau_\epsilon\f$
 * @return \f$A\f$ and \f$B\f$ at each frequency
 */
template <int NF, int N_SLS>
MaxwellFactors<NF> maxwell(
    const Kokkos::View<double[NF], Kokkos::LayoutRight, Kokkos::HostSpace> &f,
    const Kokkos::View<double[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_s,
    const Kokkos::View<double[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_eps);

/**
 * @brief Globe's attenuation_eval misfit for the Q fit.
 *
 * \f$\sum_i |\tan\delta(f_i)-1/Q|/(1/Q)\f$ with \f$\tan\delta=B/A\f$, using
 * globe's HUGEVAL in place of \f$\tan\delta\f$ when \f$A=0\f$.
 *
 * @tparam N_SLS Number of standard linear solids
 */
template <int N_SLS> struct AttenuationObjective {
  double iQ = 0; ///< \f$1/Q\f$, the target loss tangent
  Kokkos::View<double[specfem::constants::NF_ATTENUATION], Kokkos::LayoutRight,
               Kokkos::HostSpace>
      f; ///< Evaluation frequencies (Hz)
  Kokkos::View<double[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma; ///< Stress relaxation times \f$\tau_\sigma\f$

  /**
   * @brief Evaluate the misfit for candidate strain relaxation times.
   * @param tau_eps Candidate \f$\tau_\epsilon\f$
   * @return Sum of absolute relative \f$1/Q\f$ misfits
   */
  double operator()(const Kokkos::View<double[N_SLS], Kokkos::LayoutRight,
                                       Kokkos::HostSpace> &tau_eps) const;
};

/**
 * @brief Fit strain relaxation times with globe's
 * attenuation_invert_by_simplex.
 *
 * Uses 100 log-spaced frequencies, the initial guess
 * \f$\tau_\sigma(1+2/Q)\f$ and specfem::optimization::Fminsearch, all in double
 * precision regardless of type_real.
 *
 * @tparam N_SLS Number of standard linear solids (> 1)
 * @tparam T Unit of the band (frequency or period)
 * @param Q Target Q, limited by effective_quality_factor()
 * @param tau_sigma Stress relaxation times from compute_tau_sigma()
 * @param band Attenuation band
 * @return Strain relaxation times \f$\tau_\epsilon\f$
 * @throws std::runtime_error if the fit exhausts globe's budgets
 * @throws std::invalid_argument if Q is NaN
 */
template <int N_SLS, typename T>
Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
compute_tau_eps(type_real Q,
                const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                                   Kokkos::HostSpace> &tau_sigma,
                const specfem::utilities::Band<T> &band);

/**
 * @brief Globe's get_attenuation_property_values.
 *
 * Returns positive defects \f$\beta_i=\tau_{\epsilon_i}/\tau_{\sigma_i}-1\f$
 * (globe stores \f$-\beta_i\f$) and
 * \f$\text{one\_minus\_sum\_beta}=1+\sum_i\beta_i\f$, the relaxed-to-unrelaxed
 * multiplier. Evaluates \f$1-\tau_\epsilon\,\tau_\sigma^{-1}\f$ with a guarded
 * reciprocal, in globe's order.
 *
 * @tparam N_SLS Number of standard linear solids
 * @param tau_s Stress relaxation times \f$\tau_\sigma\f$
 * @param tau_eps Strain relaxation times \f$\tau_\epsilon\f$
 * @return Defects and one_minus_sum_beta
 */
template <int N_SLS>
specfem::attenuation::AttenuationPropertyValues<N_SLS>
get_attenuation_property_values(
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_s,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_eps);

/** @brief Modulus multipliers for globe's two setup stages. */
struct ModulusScales {
  double relaxed = 1.0; ///< Reference-frequency to relaxed multiplier.
  double ratio = 1.0;   ///< Relaxed to unrelaxed multiplier.
  /**
   * @brief Total multiplier used by scalar property storage.
   * @return Reference-frequency to unrelaxed multiplier in runtime precision
   */
  type_real unrelaxed() const { return relaxed * ratio; }
};

/**
 * @brief Globe's get_attenuation_scale_factor, plus the unrelaxed ratio.
 *
 * The relaxed scale is \f$[1+2\ln(f_c/f_0)/(\pi Q)]\,B^2/(2\Omega)\f$ with
 * \f$\Omega=A(\sqrt{1+B^2/A^2}-1)\f$ evaluated at \f$f_c\f$, including
 * globe's zero-denominator fallbacks. The ratio is
 * \f$1+\sum_i(\tau_{\epsilon_i}/\tau_{\sigma_i}-1)\f$.
 *
 * @tparam N_SLS Number of standard linear solids
 * @param f_c_source Band-center frequency (Hz)
 * @param tau_eps Strain relaxation times \f$\tau_\epsilon\f$
 * @param tau_sigma Stress relaxation times \f$\tau_\sigma\f$
 * @param Q_val Quality factor, limited by effective_quality_factor()
 * @param attenuation_f0_reference Model reference frequency (Hz)
 * @return Relaxed scale and unrelaxed ratio
 * @throws std::runtime_error if the relaxed scale is outside [0.5, 1.5]
 */
template <int N_SLS>
ModulusScales get_attenuation_modulus_scales(
    type_real f_c_source,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_eps,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_sigma,
    type_real Q_val, type_real attenuation_f0_reference);

/**
 * @brief Everything globe derives from one quality factor.
 * @tparam N_SLS Number of standard linear solids
 */
template <int N_SLS> struct QualityFactorFit {
  /// Strain relaxation times \f$\tau_\epsilon\f$
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps;
  /// Defects and one_minus_sum_beta
  specfem::attenuation::AttenuationPropertyValues<N_SLS> properties;
  ModulusScales scales; ///< Relaxed scale and unrelaxed ratio
};

/**
 * @brief Fit Q and derive its property values and modulus scales.
 *
 * Runs compute_tau_eps(), get_attenuation_property_values() and
 * get_attenuation_modulus_scales() in turn.
 *
 * @tparam N_SLS Number of standard linear solids (> 1)
 * @tparam T Unit of the band (frequency or period)
 * @param Q Quality factor, limited by effective_quality_factor()
 * @param tau_sigma Stress relaxation times from compute_tau_sigma()
 * @param band Attenuation band
 * @param f_c_source Band-center frequency (Hz)
 * @param attenuation_f0_reference Model reference frequency (Hz)
 * @return The fit and everything derived from it
 */
template <int N_SLS, typename T>
QualityFactorFit<N_SLS>
fit_quality_factor(type_real Q,
                   const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                                      Kokkos::HostSpace> &tau_sigma,
                   const specfem::utilities::Band<T> &band,
                   type_real f_c_source, type_real attenuation_f0_reference);

} // namespace globe
} // namespace attenuation
} // namespace specfem

#include "globe.tpp"
