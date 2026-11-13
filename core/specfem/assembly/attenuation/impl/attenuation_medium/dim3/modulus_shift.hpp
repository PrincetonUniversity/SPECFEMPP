#pragma once

#include "specfem/assembly/attenuation/impl/attenuation_medium/dim3/unrelaxed_tensor.hpp"
#include "specfem/setup.hpp"
#include <array>

namespace specfem::assembly::impl {

/** @brief Unrelaxed moduli of one GLL point that set its relaxation rates. */
struct UnrelaxedModuli {
  type_real kappa = 0; ///< Bulk modulus
  type_real mu = 0;    ///< Shear modulus
};

/**
 * @brief Shift an isotropic GLL point from reference to unrelaxed moduli.
 *
 * Scales kappa and mu by their total reference-to-unrelaxed multipliers,
 * as SPECFEM3D's prepare_attenuation.f90.
 *
 * @tparam PropsContainer Property container exposing h_kappa and h_mu.
 * @tparam ScaleView Host view of the container's per-GLL modulus scales.
 */
template <typename PropsContainer, typename ScaleView>
struct IsotropicModulusShift {
  /// Isotropic moduli need no radial frame.
  static constexpr bool needs_spherical_coordinates = false;

  const PropsContainer &props; ///< Properties, shifted in place
  ScaleView kappa_scale;       ///< Output: per-GLL kappa multiplier
  ScaleView mu_scale;          ///< Output: per-GLL mu multiplier
  int offset = 0;              ///< Attenuation-to-property element index offset

  /**
   * @brief Shift one GLL point.
   * @tparam Fit Fit of one Q (scales and relaxation properties).
   * @param i Attenuation-local element index.
   * @param iz, iy, ix GLL indices.
   * @param kappa_fit Bulk fit (identity when bulk attenuation is off).
   * @param mu_fit Shear fit.
   * @return Unrelaxed kappa and mu.
   */
  template <typename Fit>
  UnrelaxedModuli operator()(const int i, const int /*ispec*/, const int iz,
                             const int iy, const int ix,
                             const AttenuationElementContext & /*context*/,
                             const Fit &kappa_fit, const Fit &mu_fit) const {
    const type_real kappa_multiplier = kappa_fit.scales.unrelaxed();
    const type_real mu_multiplier = mu_fit.scales.unrelaxed();
    const type_real kappa =
        props.h_kappa(offset + i, iz, iy, ix) * kappa_multiplier;
    const type_real mu = props.h_mu(offset + i, iz, iy, ix) * mu_multiplier;
    props.h_kappa(offset + i, iz, iy, ix) = kappa;
    props.h_mu(offset + i, iz, iy, ix) = mu;
    kappa_scale(i, iz, iy, ix) = kappa_multiplier;
    mu_scale(i, iz, iy, ix) = mu_multiplier;
    return { kappa, mu };
  }
};

/**
 * @brief Shift an anisotropic GLL point from reference to unrelaxed moduli.
 *
 * Shear: globe's Love-parameter shift in the point's radial frame (see
 * shift_shear_to_unrelaxed()). Bulk: an isotropic increment of the Voigt
 * bulk modulus (see shift_bulk()), the identity when bulk attenuation is off.
 * The shear modulus driving the memory variables is the final radial c44,
 * globe's muvstore; the bulk one is the unrelaxed Voigt bulk modulus.
 *
 * @tparam CijView Host view type of one stiffness coefficient.
 * @tparam ScaleView Host view of the container's per-GLL modulus scales.
 * @tparam CorrectionView Host view of the container's cij increments.
 * @tparam SphericalView Host spherical coordinate cache.
 */
template <typename CijView, typename ScaleView, typename CorrectionView,
          typename SphericalView>
struct AnisotropicModulusShift {
  /// Globe mantle points are shifted in their geocentric frame.
  static constexpr bool needs_spherical_coordinates = true;

  std::array<CijView, 21> cij; ///< c11..c66, shifted in place
  ScaleView kappa_scale;       ///< Output: per-GLL bulk multiplier
  ScaleView mu_scale;          ///< Output: per-GLL shear multiplier
  CorrectionView correction;   ///< Output: unrelaxed minus reference cij
  SphericalView spherical;     ///< (r, theta, phi) per global GLL point
  int offset = 0;              ///< Attenuation-to-property element index offset

  /**
   * @brief Shift one GLL point.
   * @tparam Fit Fit of one Q (scales and relaxation properties).
   * @param i Attenuation-local element index.
   * @param ispec Global element index.
   * @param iz, iy, ix GLL indices.
   * @param context Element metadata selecting the radial frame.
   * @param kappa_fit Bulk fit (identity when bulk attenuation is off).
   * @param mu_fit Shear fit.
   * @return Unrelaxed bulk modulus and radial c44.
   */
  template <typename Fit>
  UnrelaxedModuli operator()(const int i, const int ispec, const int iz,
                             const int iy, const int ix,
                             const AttenuationElementContext &context,
                             const Fit &kappa_fit, const Fit &mu_fit) const {
    using Tensor = specfem::medium_physics::ElasticityTensor<
        specfem::coordinate_systems::CartesianFrame, double>;
    const int iprop = offset + i;
    double coefficients[21];
    for (int k = 0; k < 21; ++k)
      coefficients[k] = cij[k](iprop, iz, iy, ix);
    const Tensor reference(coefficients);

    const bool rotates = context.needs_spherical_coordinates();
    const auto frame = RadialFrame::of(
        context, rotates ? spherical(ispec, iz, iy, ix, 1) : 0.0,
        rotates ? spherical(ispec, iz, iy, ix, 2) : 0.0);
    double radial_c44 = 0;
    auto unrelaxed = shift_shear_to_unrelaxed<type_real>(
        frame, reference, mu_fit.scales.relaxed,
        mu_fit.properties.one_minus_sum_beta, radial_c44);
    const type_real kappa_multiplier = kappa_fit.scales.unrelaxed();
    const double kappa = shift_bulk(unrelaxed, reference, kappa_multiplier);

    for (int k = 0; k < 21; ++k) {
      cij[k](iprop, iz, iy, ix) = unrelaxed[k];
      correction(k, i, iz, iy, ix) =
          cij[k](iprop, iz, iy, ix) - coefficients[k];
    }
    kappa_scale(i, iz, iy, ix) = kappa_multiplier;
    mu_scale(i, iz, iy, ix) = mu_fit.scales.unrelaxed();
    // Globe's muvstore holds this c44 in storage precision.
    return { static_cast<type_real>(kappa),
             static_cast<type_real>(radial_c44) };
  }
};

} // namespace specfem::assembly::impl
