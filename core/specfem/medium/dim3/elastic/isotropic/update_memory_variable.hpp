#include "specfem/datatype.hpp"
#include "specfem/point/attenuation_factors.hpp"
#include "specfem/point/integration_factors.hpp"
#include "specfem/point/memory.hpp"
#include "specfem/point/strain.hpp"
#include <type_traits>

namespace specfem::medium_physics::impl {

/**
 * Runge Kutta integration of the memory variable for constant isotropic
 * attenuation in elastic media.
 *
 * See Savage 2010 Eqs. 8-11 for the formulation of the memory variable update
 * and Tromp 2018 notes for the specific form of the attenuation factors.
 *
 *
 * The code below could be optimized by explicitly only computing the 6
 * necessary components of the strain and memory variable for 3D elastic media,
 * rather than computing all 9 components and relying on the fact that the
 * off-diagonal components of the strain and memory variable are zero for
 * isotropic media.
 */
template <typename Tags,
          std::enable_if_t<
              (Tags::dimension_tag == specfem::element::dimension_tag::dim3) &&
                  (Tags::medium_tag == specfem::element::medium_tag::elastic) &&
                  (Tags::attenuation_tag ==
                   specfem::element::attenuation_tag::constant_isotropic),
              int> = 0>
KOKKOS_INLINE_FUNCTION void compute_update_memory_variable(
    const specfem::point::strain<Tags::dimension_tag, Tags::medium_tag, UseSIMD>
        &point_current_strain,
    const specfem::point::strain<Tags::dimension_tag, Tags::medium_tag, UseSIMD>
        &point_future_strain,
    const specfem::point::attenuation_factors<Tags::dimension_tag,
                                              Tags::medium_tag, UseSIMD>
        &point_attenuation_factors,
    const specfem::point::integration_factors<Tags::dimension_tag,
                                              Tags::medium_tag, UseSIMD>
        &point_integration_factors,
    specfem::point::memory<Tags::dimension_tag, Tags::medium_tag, N_SLS,
                           UseSIMD> &point_memory_variable) {

  /** Savage 2010 eq. 11 need 2 * deltaMi / tau_sigma_i Delta M is the
   * modulus defect and delta M i are the individual modulus defect
   * tau_sigma_i)
   * Tromp attenuation notes equation 33 and 44
   *  \f$ (M_R/L)(\tau_\epsilon_^\ell/\tau_\sigma_^\ell - 1), \f$
   * where  \f$ M_R  \f$ is the relaxed modulus and  \f$ M_L \f$ is the
   * unrelaxed modulus. We can compute the factor as
   * (M_R/L)(tau_epsilon/tau_sigma So, combining savage 2010 with tromp notes
   * from 2018
   *
   * @code
   * kappa_defect(i) = (kappa_relaxed / N_SLS) (tau_epsilon(i)/tau_sigma(i) -
   * 1)} kappa_defect_factor(i) = 2 * kappa_defect(i) / \tau_sigma(i)}
   * @endcode
   *
   * where `kappa_defect_factor(i)` is stored per GLL point
   *
   * Similarly, we have a factor for the shear attenuation
   * mu_defect_factor(i) = 2 * mu_defect(i) / tau_sigma(i)
   */
  const auto A_kappa = point_attenuation_factors.A_kappa;
  const auto A_mu = point_attenuation_factors.A_mu;

  const auto &alpha = point_integration_factors.alpha();
  const auto &beta = point_integration_factors.beta();
  const auto &gamma = point_integration_factors.gamma();

  auto &R_xx = point_memory_variable.R_xx;
  auto &R_yy = point_memory_variable.R_yy;
  auto &R_xy = point_memory_variable.R_xy;
  auto &R_xz = point_memory_variable.R_xz;
  auto &R_yz = point_memory_variable.R_yz;
  auto &R_kappa = point_memory_variable.R_kappa;
  const auto S = point_current_strain;
  const auto Snp = point_future_strain;
  const auto trace_S = point_current_strain.trace();
  const auto trace_Snp = point_future_strain.trace();

  for (int isls = 0; isls < N_SLS; ++isls) {

    R_kappa(isls) =
        alpha(isls) * R_kappa(isls) +
        A_kappa(isls) * (beta(isls) * trace_S - gamma(isls) * trace_Snp);
    R_xx(isls) = alpha(isls) * R_xx(isls) +
                 A_mu(isls) * (beta(isls) * S.xx - alpha(isls) * Snp.xx);
    R_yy(isls) = alpha(isls) * R_yy(isls) +
                 A_mu(isls) * (beta(isls) * S.yy - alpha(isls) * Snp.yy);
    R_xy(isls) = alpha(isls) * R_xy(isls) +
                 A_mu(isls) * (beta(isls) * S.xy - alpha(isls) * Snp.xy);
    R_xz(isls) = alpha(isls) * R_xz(isls) +
                 A_mu(isls) * (beta(isls) * S.xz - alpha(isls) * Snp.xz);
    R_yz(isls) = alpha(isls) * R_yz(isls) +
                 A_mu(isls) * (beta(isls) * S.yz - alpha(isls) * Snp.yz);
  }
};

} // namespace specfem::medium_physics::impl
