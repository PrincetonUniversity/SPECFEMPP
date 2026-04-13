#pragma once

#include "specfem/assembly/attenuation/load_on_device.hpp"
#include "specfem/assembly/attenuation/store_on_device.hpp"
#include "specfem/constants.hpp"
#include "specfem/element.hpp"
#include "specfem/point/attenuation.hpp"
#include "specfem/point/field_derivatives.hpp"
#include "specfem/point/stress.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

// ---------------------------------------------------------------------------
// SFINAE helper shared by all three impl functions
// ---------------------------------------------------------------------------

template <typename Tags>
using enable_if_dim2_psv_iso_att = std::enable_if_t<
    Tags::dimension_tag == specfem::element::dimension_tag::dim2 &&
        Tags::medium_tag == specfem::element::medium_tag::elastic_psv &&
        Tags::property_tag == specfem::element::property_tag::isotropic &&
        Tags::attenuation_tag ==
            specfem::element::attenuation_tag::constant_isotropic,
    int>;

// ---------------------------------------------------------------------------
// impl_add_relaxation_to_stress
// ---------------------------------------------------------------------------

/**
 * @brief Add SLS memory variable contributions to the elastic stress tensor
 *        for 2D elastic P-SV isotropic media with constant-Q attenuation.
 *
 * Follows SPECFEM3D convention (subtraction from stress):
 *   sigma_xx -= sum_j(R_xx[j]) + sum_j(R_kappa[j])
 *   sigma_zz += sum_j(R_xx[j]) - sum_j(R_kappa[j])   (Rzz = -Rxx, traceless)
 *   sigma_xz -= sum_j(R_xz[j])
 *
 * @tparam Tags Element tags (dim2, elastic_psv, isotropic, constant_isotropic)
 * @param point_attenuation Per-point attenuation state (memory variables)
 * @param point_stress      Stress tensor to be modified in-place
 */
template <typename Tags, enable_if_dim2_psv_iso_att<Tags> = 0>
KOKKOS_INLINE_FUNCTION void impl_add_relaxation_to_stress(
    const specfem::point::attenuation<
        specfem::element::dimension_tag::dim2,
        specfem::element::medium_tag::elastic_psv,
        specfem::element::attenuation_tag::constant_isotropic, Tags::using_simd>
        &point_attenuation,
    specfem::point::stress<Tags> &point_stress) {

  using datatype =
      typename specfem::datatype::simd<type_real, Tags::using_simd>::datatype;

  constexpr int N = specfem::constants::N_SLS;

  datatype R_xx_sum{ 0 };
  datatype R_xz_sum{ 0 };
  datatype R_kappa_sum{ 0 };

  for (int j = 0; j < N; ++j) {
    R_xx_sum += point_attenuation.Rxx(j);
    R_xz_sum += point_attenuation.Rxz(j);
    R_kappa_sum += point_attenuation.Rkappa(j);
  }

  // T layout: T(0,0)=sigma_xx, T(0,1)=sigma_xz, T(1,0)=sigma_xz,
  // T(1,1)=sigma_zz sigma_xx -= R_xx + R_kappa
  point_stress.T(0, 0) -= R_xx_sum + R_kappa_sum;
  // sigma_zz += R_xx - R_kappa   (Rzz = -Rxx for traceless deviatoric)
  point_stress.T(1, 1) += R_xx_sum - R_kappa_sum;
  // sigma_xz -= R_xz
  point_stress.T(0, 1) -= R_xz_sum;
  point_stress.T(1, 0) -= R_xz_sum;
}

// ---------------------------------------------------------------------------
// impl_integrate_memory_variables
// ---------------------------------------------------------------------------

/**
 * @brief Advance SLS memory variables by one RK step for 2D elastic P-SV
 *        isotropic media.
 *
 * Implements Savage et al. (2010) BSSA eqs. 8-11:
 *   Snp1 = deviatoric(du + dt * dv)   (Taylor-expanded displacement strain)
 *   R[j] = alpha[j]*R[j] + rate[j] * (beta[j]*Sn + gamma[j]*Snp1)
 *
 * @tparam Tags Element tags
 * @param point_attenuation Attenuation state; memory variables updated in-place
 * @param Sn   Strain from previous time step (loaded from
 * field_derivative_storage)
 * @param du   Displacement gradient (current step, from gradient pack)
 * @param dv   Velocity gradient (current step, from gradient pack)
 * @param deltat Time step size
 */
template <typename Tags, enable_if_dim2_psv_iso_att<Tags> = 0>
KOKKOS_INLINE_FUNCTION void impl_integrate_memory_variables(
    specfem::point::attenuation<
        specfem::element::dimension_tag::dim2,
        specfem::element::medium_tag::elastic_psv,
        specfem::element::attenuation_tag::constant_isotropic, Tags::using_simd>
        &point_attenuation,
    const specfem::point::field_derivatives<Tags> &Sn,
    const specfem::point::field_derivatives<Tags> &du,
    const specfem::point::field_derivatives<Tags> &dv, const type_real deltat) {

  using datatype =
      typename specfem::datatype::simd<type_real, Tags::using_simd>::datatype;

  constexpr int N = specfem::constants::N_SLS;

  // du_att = du + dt*dv  (first-order Taylor expansion, mirrors SPECFEM3D)
  const auto du_att = du + deltat * dv;

  // Traces
  const datatype trace_Sn = Sn.du(0, 0) + Sn.du(1, 1);
  const datatype trace_Snp1 = du_att.du(0, 0) + du_att.du(1, 1);

  // Deviatoric components: epsilondev = strain - trace/2  (2D)
  constexpr type_real one_third =
      static_cast<type_real>(1) / static_cast<type_real>(3);

  // For 2D P-SV we follow the 3D convention and subtract trace/3
  // (plane-strain approximation; see Komatitsch & Tromp 1999)
  const datatype epsilondev_xx_Sn = Sn.du(0, 0) - one_third * trace_Sn;
  const datatype epsilondev_zz_Sn = Sn.du(1, 1) - one_third * trace_Sn;
  // xz: symmetrized shear (0.5*(dux/dz + duz/dx)), indices (0,1) and (1,0)
  const datatype epsilondev_xz_Sn =
      static_cast<type_real>(0.5) * (Sn.du(0, 1) + Sn.du(1, 0));

  const datatype epsilondev_xx_Snp1 = du_att.du(0, 0) - one_third * trace_Snp1;
  // epsilondev_zz not stored separately (Rzz = -(Rxx) enforced by traceless)
  const datatype epsilondev_xz_Snp1 =
      static_cast<type_real>(0.5) * (du_att.du(0, 1) + du_att.du(1, 0));

  for (int j = 0; j < N; ++j) {
    const datatype alpha = point_attenuation.alpha_rk(j);
    const datatype beta = point_attenuation.beta_rk(j);
    const datatype gamma = point_attenuation.gamma_rk(j);
    const datatype mu_rate = point_attenuation.mu_relaxation_rate(j);
    const datatype kappa_rate = point_attenuation.kappa_relaxation_rate(j);

    point_attenuation.Rxx(j) =
        alpha * point_attenuation.Rxx(j) +
        mu_rate * (beta * epsilondev_xx_Sn + gamma * epsilondev_xx_Snp1);

    point_attenuation.Rxz(j) =
        alpha * point_attenuation.Rxz(j) +
        mu_rate * (beta * epsilondev_xz_Sn + gamma * epsilondev_xz_Snp1);

    point_attenuation.Rkappa(j) =
        alpha * point_attenuation.Rkappa(j) +
        kappa_rate * (beta * trace_Sn + gamma * trace_Snp1);
  }
}

// ---------------------------------------------------------------------------
// impl_compute_attenuation
// ---------------------------------------------------------------------------

/**
 * @brief Full attenuation update for a single GLL point: load state, modify
 *        stress, advance memory variables, and write back.
 *
 * @tparam Tags          Element tags (dim2, elastic_psv, isotropic,
 *                       constant_isotropic)
 * @tparam IndexType     Point index type
 * @tparam GradientPackType  stiffness_gradient_pack type (provides
 * get_du/get_dv)
 * @tparam AttenuationContainer        Assembly attenuation container
 *
 * @param index                  GLL point index
 * @param point_stress           Elastic stress (modified in-place)
 * @param grad_pack              Gradient pack (contains du and dv)
 * @param attenuation            Assembly attenuation container
 */
template <typename Tags, typename IndexType, typename GradientPackType,
          typename AttenuationContainer, enable_if_dim2_psv_iso_att<Tags> = 0>
KOKKOS_INLINE_FUNCTION void
impl_compute_attenuation(const IndexType &index,
                         specfem::point::stress<Tags> &point_stress,
                         const GradientPackType &grad_pack,
                         const AttenuationContainer &attenuation) {

  using PointAttenuationType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim2,
      specfem::element::medium_tag::elastic_psv,
      specfem::element::attenuation_tag::constant_isotropic, Tags::using_simd>;

  const auto du = grad_pack.template get_du<Tags>();
  const auto dv = grad_pack.template get_dv<Tags>();

  PointAttenuationType point_attenuation;
  specfem::assembly::load_on_device(index, attenuation, point_attenuation);

  // Load Sn (strain from previous time step) from point_attenuation.du
  specfem::point::field_derivatives<Tags> Sn;
  constexpr int components = PointAttenuationType::components;
  constexpr int num_dimensions = PointAttenuationType::num_dimensions;
  for (int ic = 0; ic < components; ++ic) {
    for (int id = 0; id < num_dimensions; ++id) {
      Sn.du[ic][id] = point_attenuation.du[ic][id];
    }
  }

  // 1. Add memory variable contributions to stress
  impl_add_relaxation_to_stress<Tags>(point_attenuation, point_stress);

  // 2. Advance memory variables
  impl_integrate_memory_variables<Tags>(point_attenuation, Sn, du, dv,
                                        attenuation.deltat);

  // 3. Compute du_att = du + dt*dv for next time step
  const auto du_att = du + attenuation.deltat * dv;

  // 4. Store du_att in point_attenuation.du for next time step
  for (int ic = 0; ic < components; ++ic) {
    for (int id = 0; id < num_dimensions; ++id) {
      point_attenuation.du[ic][id] = du_att.du[ic][id];
    }
  }

  // 5. Write back updated memory variables and du (next Sn)
  specfem::assembly::store_on_device(index, attenuation, point_attenuation);
}

} // namespace medium_physics
} // namespace specfem
