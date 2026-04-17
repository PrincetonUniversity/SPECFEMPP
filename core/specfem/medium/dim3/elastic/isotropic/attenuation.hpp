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
// impl_add_relaxation_to_stress
// ---------------------------------------------------------------------------

/**
 * @brief Add SLS memory variable contributions to the elastic stress tensor
 *        for 3D elastic isotropic media with constant-Q attenuation.
 *
 * Follows SPECFEM3D convention (subtraction from stress).
 * Rzz = -(Rxx + Ryy) is not stored; its contribution is absorbed into T(2,2):
 *   sigma_xx -= sum_j(R_xx[j]) + sum_j(R_kappa[j])
 *   sigma_yy -= sum_j(R_yy[j]) + sum_j(R_kappa[j])
 *   sigma_zz += sum_j(R_xx[j] + R_yy[j]) - sum_j(R_kappa[j])
 *   sigma_xy -= sum_j(R_xy[j])
 *   sigma_xz -= sum_j(R_xz[j])
 *   sigma_yz -= sum_j(R_yz[j])
 *
 * @tparam Tags Element tags (dim3, elastic, isotropic, constant_isotropic)
 * @param point_attenuation Per-point attenuation state (memory variables)
 * @param point_stress      Stress tensor to be modified in-place
 */
template <
    typename Tags,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
            Tags::medium_tag == specfem::element::medium_tag::elastic &&
            Tags::property_tag == specfem::element::property_tag::isotropic &&
            Tags::attenuation_tag ==
                specfem::element::attenuation_tag::constant_isotropic,
        int> = 0>
KOKKOS_INLINE_FUNCTION void impl_add_relaxation_to_stress(
    const specfem::point::attenuation<
        specfem::element::dimension_tag::dim3,
        specfem::element::medium_tag::elastic,
        specfem::element::attenuation_tag::constant_isotropic, Tags::using_simd>
        &point_attenuation,
    specfem::point::stress<Tags> &point_stress) {

  using datatype =
      typename specfem::datatype::simd<type_real, Tags::using_simd>::datatype;

  constexpr int N = specfem::constants::N_SLS;

  datatype R_xx_sum{ 0 };
  datatype R_yy_sum{ 0 };
  datatype R_xy_sum{ 0 };
  datatype R_xz_sum{ 0 };
  datatype R_yz_sum{ 0 };
  datatype R_kappa_sum{ 0 };

  for (int j = 0; j < N; ++j) {
    R_xx_sum += point_attenuation.Rxx(j);
    R_yy_sum += point_attenuation.Ryy(j);
    R_xy_sum += point_attenuation.Rxy(j);
    R_xz_sum += point_attenuation.Rxz(j);
    R_yz_sum += point_attenuation.Ryz(j);
    R_kappa_sum += point_attenuation.Rkappa(j);
  }

  // T layout: T(0,0)=sigma_xx, T(1,1)=sigma_yy, T(2,2)=sigma_zz
  //           T(0,1)=T(1,0)=sigma_xy, T(0,2)=T(2,0)=sigma_xz,
  //           T(1,2)=T(2,1)=sigma_yz
  // sigma_xx -= R_xx + R_kappa
  point_stress.T(0, 0) -= R_xx_sum + R_kappa_sum;
  // sigma_yy -= R_yy + R_kappa
  point_stress.T(1, 1) -= R_yy_sum + R_kappa_sum;
  // sigma_zz -= Rzz + R_kappa; Rzz = -(Rxx+Ryy) => += (Rxx+Ryy) - R_kappa
  point_stress.T(2, 2) += R_xx_sum + R_yy_sum - R_kappa_sum;
  // sigma_xy -= R_xy
  point_stress.T(0, 1) -= R_xy_sum;
  point_stress.T(1, 0) -= R_xy_sum;
  // sigma_xz -= R_xz
  point_stress.T(0, 2) -= R_xz_sum;
  point_stress.T(2, 0) -= R_xz_sum;
  // sigma_yz -= R_yz
  point_stress.T(1, 2) -= R_yz_sum;
  point_stress.T(2, 1) -= R_yz_sum;
}

// ---------------------------------------------------------------------------
// impl_integrate_memory_variables
// ---------------------------------------------------------------------------

/**
 * @brief Advance SLS memory variables by one RK step for 3D elastic
 *        isotropic media.
 *
 * Implements Savage et al. (2010) BSSA eqs. 8-11 extended to 3D:
 *   Snp1 = symmetrised(du + dt * dv)   (Taylor-expanded displacement strain)
 *   R[j] = alpha[j]*R[j] + rate[j] * (beta[j]*Sn + gamma[j]*Snp1)
 *
 * Sn (previous symmetrised strain) is read from point_attenuation.epsilon_*.
 * Snp1 is computed internally and written back to point_attenuation.epsilon_*.
 * Five deviatoric components (Rxx, Ryy, Rxy, Rxz, Ryz) plus Rkappa are
 * tracked. Rzz = -(Rxx+Ryy) is not stored.
 *
 * @tparam Tags Element tags (dim3, elastic, isotropic, constant_isotropic)
 * @param point_attenuation Attenuation state; R and epsilon updated in-place
 * @param du   Displacement gradient (current step, from gradient pack)
 * @param dv   Velocity gradient (current step, from gradient pack)
 * @param deltat Time step size
 */
template <
    typename Tags,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
            Tags::medium_tag == specfem::element::medium_tag::elastic &&
            Tags::property_tag == specfem::element::property_tag::isotropic &&
            Tags::attenuation_tag ==
                specfem::element::attenuation_tag::constant_isotropic,
        int> = 0>
KOKKOS_INLINE_FUNCTION void impl_integrate_memory_variables(
    specfem::point::attenuation<
        specfem::element::dimension_tag::dim3,
        specfem::element::medium_tag::elastic,
        specfem::element::attenuation_tag::constant_isotropic, Tags::using_simd>
        &point_attenuation,
    const specfem::point::field_derivatives<Tags> &du,
    const specfem::point::field_derivatives<Tags> &dv, const type_real deltat) {

  using datatype =
      typename specfem::datatype::simd<type_real, Tags::using_simd>::datatype;

  constexpr int N = specfem::constants::N_SLS;
  constexpr type_real one_third =
      static_cast<type_real>(1) / static_cast<type_real>(3);

  // Read Sn from stored strain fields
  const datatype epsilon_xx_Sn = point_attenuation.epsilon_xx;
  const datatype epsilon_yy_Sn = point_attenuation.epsilon_yy;
  const datatype epsilon_zz_Sn = point_attenuation.epsilon_zz;
  const datatype epsilon_xy_Sn = point_attenuation.epsilon_xy;
  const datatype epsilon_xz_Sn = point_attenuation.epsilon_xz;
  const datatype epsilon_yz_Sn = point_attenuation.epsilon_yz;

  // Compute Snp1 from Taylor-predicted displacement gradient
  const auto du_att = du + deltat * dv;
  const datatype epsilon_xx_Snp1 = du_att.du(0, 0);
  const datatype epsilon_yy_Snp1 = du_att.du(1, 1);
  const datatype epsilon_zz_Snp1 = du_att.du(2, 2);
  const datatype epsilon_xy_Snp1 =
      static_cast<type_real>(0.5) * (du_att.du(0, 1) + du_att.du(1, 0));
  const datatype epsilon_xz_Snp1 =
      static_cast<type_real>(0.5) * (du_att.du(0, 2) + du_att.du(2, 0));
  const datatype epsilon_yz_Snp1 =
      static_cast<type_real>(0.5) * (du_att.du(1, 2) + du_att.du(2, 1));

  // Traces
  const datatype trace_Sn = epsilon_xx_Sn + epsilon_yy_Sn + epsilon_zz_Sn;
  const datatype trace_Snp1 =
      epsilon_xx_Snp1 + epsilon_yy_Snp1 + epsilon_zz_Snp1;

  // Deviatoric normal components (trace/3 subtracted)
  const datatype epsilondev_xx_Sn = epsilon_xx_Sn - one_third * trace_Sn;
  const datatype epsilondev_yy_Sn = epsilon_yy_Sn - one_third * trace_Sn;
  const datatype epsilondev_xx_Snp1 = epsilon_xx_Snp1 - one_third * trace_Snp1;
  const datatype epsilondev_yy_Snp1 = epsilon_yy_Snp1 - one_third * trace_Snp1;
  // epsilondev_zz not stored (Rzz = -(Rxx+Ryy), enforced by tracelessness)

  // Shear components: already symmetrised
  const datatype &epsilondev_xy_Sn = epsilon_xy_Sn;
  const datatype &epsilondev_xz_Sn = epsilon_xz_Sn;
  const datatype &epsilondev_yz_Sn = epsilon_yz_Sn;
  const datatype &epsilondev_xy_Snp1 = epsilon_xy_Snp1;
  const datatype &epsilondev_xz_Snp1 = epsilon_xz_Snp1;
  const datatype &epsilondev_yz_Snp1 = epsilon_yz_Snp1;

  for (int j = 0; j < N; ++j) {
    const datatype alpha = point_attenuation.alpha_rk(j);
    const datatype beta = point_attenuation.beta_rk(j);
    const datatype gamma = point_attenuation.gamma_rk(j);
    const datatype mu_rate = point_attenuation.mu_relaxation_rate(j);
    const datatype kappa_rate = point_attenuation.kappa_relaxation_rate(j);

    point_attenuation.Rxx(j) =
        alpha * point_attenuation.Rxx(j) +
        mu_rate * (beta * epsilondev_xx_Sn + gamma * epsilondev_xx_Snp1);

    point_attenuation.Ryy(j) =
        alpha * point_attenuation.Ryy(j) +
        mu_rate * (beta * epsilondev_yy_Sn + gamma * epsilondev_yy_Snp1);

    point_attenuation.Rxy(j) =
        alpha * point_attenuation.Rxy(j) +
        mu_rate * (beta * epsilondev_xy_Sn + gamma * epsilondev_xy_Snp1);

    point_attenuation.Rxz(j) =
        alpha * point_attenuation.Rxz(j) +
        mu_rate * (beta * epsilondev_xz_Sn + gamma * epsilondev_xz_Snp1);

    point_attenuation.Ryz(j) =
        alpha * point_attenuation.Ryz(j) +
        mu_rate * (beta * epsilondev_yz_Sn + gamma * epsilondev_yz_Snp1);

    point_attenuation.Rkappa(j) =
        alpha * point_attenuation.Rkappa(j) +
        kappa_rate * (beta * trace_Sn + gamma * trace_Snp1);
  }

  // Write back Snp1 for the next time step
  point_attenuation.epsilon_xx = epsilon_xx_Snp1;
  point_attenuation.epsilon_yy = epsilon_yy_Snp1;
  point_attenuation.epsilon_zz = epsilon_zz_Snp1;
  point_attenuation.epsilon_xy = epsilon_xy_Snp1;
  point_attenuation.epsilon_xz = epsilon_xz_Snp1;
  point_attenuation.epsilon_yz = epsilon_yz_Snp1;
}

// ---------------------------------------------------------------------------
// impl_compute_attenuation
// ---------------------------------------------------------------------------

/**
 * @brief Full attenuation update for a single GLL point: load state, modify
 *        stress, advance memory variables, and write back.
 *
 * @tparam Tags          Element tags (dim3, elastic, isotropic,
 *                       constant_isotropic)
 * @tparam IndexType     Point index type
 * @tparam GradientPackType  stiffness_gradient_pack type (provides
 *                           get_du/get_dv)
 * @tparam AttenuationContainer  Assembly attenuation container
 *
 * @param index          GLL point index
 * @param point_stress   Elastic stress (modified in-place)
 * @param grad_pack      Gradient pack (contains du and dv)
 * @param attenuation    Assembly attenuation container
 */
template <
    typename Tags, typename IndexType, typename GradientPackType,
    typename AttenuationContainer,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
            Tags::medium_tag == specfem::element::medium_tag::elastic &&
            Tags::property_tag == specfem::element::property_tag::isotropic &&
            Tags::attenuation_tag ==
                specfem::element::attenuation_tag::constant_isotropic,
        int> = 0>
KOKKOS_INLINE_FUNCTION void
impl_compute_attenuation(const IndexType &index,
                         specfem::point::stress<Tags> &point_stress,
                         const GradientPackType &grad_pack,
                         const AttenuationContainer &attenuation) {

  using PointAttenuationType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic,
      specfem::element::attenuation_tag::constant_isotropic, Tags::using_simd>;

  const auto du = grad_pack.template get_du<Tags>();
  const auto dv = grad_pack.template get_dv<Tags>();

  PointAttenuationType point_attenuation;
  specfem::assembly::load_on_device(index, attenuation, point_attenuation);

  // 1. Add memory variable contributions to stress
  impl_add_relaxation_to_stress<Tags>(point_attenuation, point_stress);

  // 2. Advance memory variables (reads Sn from point_attenuation.epsilon_*,
  //    writes Snp1 back)
  impl_integrate_memory_variables<Tags>(point_attenuation, du, dv,
                                        attenuation.deltat);

  // 3. Write back updated memory variables and strain
  specfem::assembly::store_on_device(index, attenuation, point_attenuation);
}

} // namespace medium_physics
} // namespace specfem
