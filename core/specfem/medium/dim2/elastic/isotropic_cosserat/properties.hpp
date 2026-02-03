#pragma once

#include "specfem/medium_container/impl/point_container.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium_container::properties {

/**
 * @defgroup specfem_point_properties_dim2_elastic_isotropic_cosserat 2D Elastic
 * Isotropic Cosserat Properties
 * @{
 */

/**
 * @ingroup specfem_point_properties_dim2_elastic_isotropic_cosserat
 * @brief Template specialization for 2D elastic isotropic cosserat media
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 *
 * Parameters:
 * - rho: Density @f$ \rho @f$
 * - kappa: Bulk modulus @f$ \kappa @f$
 * - mu: Shear modulus @f$ \mu @f$
 * - nu: Coupling parameter @f$ \nu @f$
 * - j: Inertia density @f$ j @f$
 * - lambda_c: Cosserat Lame's first parameter @f$ \lambda_c @f$
 * - mu_c: Cosserat shear modulus @f$ \mu_c @f$
 * - nu_c: Cosserat Coupling parameter @f$ \nu_c @f$
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct point_container<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::isotropic_cosserat, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    /// @cond
    : public PropertyAccessor<
          specfem::dimension::type::dim2, MediumTag,
          specfem::element::property_tag::isotropic_cosserat, UseSIMD>
/// @endcond
{

private:
  using base_type =
      PropertyAccessor<specfem::dimension::type::dim2, MediumTag,
                       specfem::element::property_tag::isotropic_cosserat,
                       UseSIMD>; ///< Base type of the
                                 ///< point properties

public:
  using value_type = typename base_type::value_type; ///< Type of the properties
  using simd = typename base_type::simd;

  POINT_CONTAINER(rho, kappa, mu, nu, j, lambda_c, mu_c, nu_c)

  // lambda + 2 mu = kappa - 2/3*mu + 6/3 * mu
  KOKKOS_INLINE_FUNCTION const value_type lambdaplus2mu() const {
    return kappa() + static_cast<value_type>(4.0) /
                         static_cast<value_type>(3.0) *
                         mu(); ///< @f$ \lambda + 2\mu @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type lambda() const {
    return kappa() - static_cast<value_type>(2.0) /
                         static_cast<value_type>(3.0) *
                         mu(); ///< @f$ \lambda + 2\mu @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type rho_vp() const {
    return Kokkos::sqrt((kappa() + static_cast<value_type>(4.0) /
                                       static_cast<value_type>(3.0) * mu()) *
                        rho()); ///< @f$ \rho v_p @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type rho_vs() const {
    return Kokkos::sqrt(rho() * mu()); ///< @f$ \rho v_s @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type vp() const {
    return Kokkos::sqrt(lambdaplus2mu() / rho()); ///< @f$ v_P @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type vs() const {
    return Kokkos::sqrt(mu() / rho()); ///< @f$ v_S @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type vmax() const {
    return Kokkos::max(vp(), vs()); ///< @f$ v_{max} = \max(v_P, v_S) @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type vmin() const {
    return Kokkos::min(vp(), vs()); ///< @f$ v_{min} = \min(v_P, v_S) @f$
  }
};
///@} end of group specfem_point_properties_dim2_elastic_isotropic_cosserat

} // namespace specfem::medium_container::properties
