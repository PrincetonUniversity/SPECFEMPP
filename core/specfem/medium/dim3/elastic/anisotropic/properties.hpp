#pragma once

#include "specfem/medium_container/impl/point_container.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium_container::properties {

/**
 * @defgroup specfem_point_properties_dim3_elastic_anisotropic 3D Elastic
 * Anisotropic Point Properties
 * @{
 */

/**
 * @ingroup specfem_point_properties_dim3_elastic_anisotropic
 * @brief Properties of a 3D anisotropic elastic medium at one quadrature point.
 *
 * The container stores the 21 independent stiffnesses of the symmetric
 * \f$6\times6\f$ Voigt matrix and density. The scalar wave-speed accessors use
 * the Voigt isotropic average
 *
 * \f[
 * K_V = \frac{c_{11}+c_{22}+c_{33}+2(c_{12}+c_{13}+c_{23})}{9},
 * \f]
 * \f[
 * G_V = \frac{c_{11}+c_{22}+c_{33}-c_{12}-c_{13}-c_{23}
 *              +3(c_{44}+c_{55}+c_{66})}{15}.
 * \f]
 *
 * Consequently, \f$v_p=\sqrt{(K_V+4G_V/3)/\rho}\f$ and
 * \f$v_s=\sqrt{G_V/\rho}\f$. These are isotropic-equivalent speeds for CFL and
 * resolution reporting, not directional phase velocities.
 *
 * @tparam MediumTag Physical medium type; must be elastic.
 * @tparam UseSIMD Whether values use SIMD lanes.
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct point_container<
    specfem::element::dimension_tag::dim3, MediumTag,
    specfem::element::property_tag::anisotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value>>
    : public PropertyAccessor<specfem::element::dimension_tag::dim3, MediumTag,
                              specfem::element::property_tag::anisotropic,
                              UseSIMD> {
private:
  using base_type =
      PropertyAccessor<specfem::element::dimension_tag::dim3, MediumTag,
                       specfem::element::property_tag::anisotropic, UseSIMD>;

public:
  using value_type = typename base_type::value_type; ///< Property value type.
  using simd = typename base_type::simd;             ///< SIMD configuration.

  POINT_CONTAINER(c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26, c33,
                  c34, c35, c36, c44, c45, c46, c55, c56, c66, rho)

  /**
   * @brief Return the Voigt-average bulk modulus.
   * @return Voigt-average bulk modulus \f$K_V\f$.
   */
  KOKKOS_INLINE_FUNCTION const value_type voigt_bulk_modulus() const {
    return (c11() + c22() + c33() +
            static_cast<type_real>(2.0) * (c12() + c13() + c23())) /
           static_cast<type_real>(9.0);
  }

  /**
   * @brief Return the Voigt-average shear modulus.
   * @return Voigt-average shear modulus \f$G_V\f$.
   */
  KOKKOS_INLINE_FUNCTION const value_type voigt_shear_modulus() const {
    return (c11() + c22() + c33() - c12() - c13() - c23() +
            static_cast<type_real>(3.0) * (c44() + c55() + c66())) /
           static_cast<type_real>(15.0);
  }

  /**
   * @brief Return density times the Voigt-average P-wave speed.
   * @return Product \f$\rho v_p\f$.
   */
  KOKKOS_INLINE_FUNCTION const value_type rho_vp() const {
    return vp() * rho();
  }

  /**
   * @brief Return density times the Voigt-average S-wave speed.
   * @return Product \f$\rho v_s\f$.
   */
  KOKKOS_INLINE_FUNCTION const value_type rho_vs() const {
    return vs() * rho();
  }

  /**
   * @brief Return the isotropic-equivalent Voigt-average P-wave speed.
   * @return Voigt-average P-wave speed \f$v_p\f$.
   */
  KOKKOS_INLINE_FUNCTION const value_type vp() const {
    return Kokkos::sqrt(
        (voigt_bulk_modulus() +
         static_cast<type_real>(4.0 / 3.0) * voigt_shear_modulus()) /
        rho());
  }

  /**
   * @brief Return the isotropic-equivalent Voigt-average S-wave speed.
   * @return Voigt-average S-wave speed \f$v_s\f$.
   */
  KOKKOS_INLINE_FUNCTION const value_type vs() const {
    return Kokkos::sqrt(voigt_shear_modulus() / rho());
  }

  /**
   * @brief Return the larger Voigt-average wave speed.
   * @return Maximum of \f$v_p\f$ and \f$v_s\f$.
   */
  KOKKOS_INLINE_FUNCTION const value_type vmax() const {
    return Kokkos::max(vp(), vs());
  }

  /**
   * @brief Return the smaller Voigt-average wave speed.
   * @return Minimum of \f$v_p\f$ and \f$v_s\f$.
   */
  KOKKOS_INLINE_FUNCTION const value_type vmin() const {
    return Kokkos::min(vp(), vs());
  }
};

/** @} */

} // namespace specfem::medium_container::properties
