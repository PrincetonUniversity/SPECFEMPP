#pragma once
#include "impl/point_data.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {
/**
 * @brief Store frechet kernels for a quadrature point
 *
 * @tparam DimensionType Dimension of the element where the quadrature point is
 * located
 * @tparam MediumTag Medium of the element where the quadrature point is located
 * @tparam PropertyTag  Property of the element where the quadrature point is
 * located
 * @tparam UseSIMD  Use SIMD instructions
 */
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD,
          typename Enable = void>
struct kernels;

/**
 * @brief Template specialization for the kernels struct for 2D elastic
 * isotropic elements
 *
 * @tparam UseSIMD  Use SIMD instructions
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct kernels<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::isotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : public impl::point_data<6, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<6, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  constexpr static bool is_point_properties = true;
  ///@}

  using base_type::base_type;

  /**
   * @name Misfit Kernels
   *
   */
  ///@{
  DEFINE_POINT_VALUE(rho, 0)   ///< \f$ K_{\rho} \f$
  DEFINE_POINT_VALUE(mu, 1)    ///< \f$ K_{\mu} \f$
  DEFINE_POINT_VALUE(kappa, 2) ///< \f$ K_{\kappa} \f$
  DEFINE_POINT_VALUE(rhop, 3)  ///< \f$ K_{\rho'} \f$
  DEFINE_POINT_VALUE(alpha, 4) ///< \f$ K_{\alpha} \f$
  DEFINE_POINT_VALUE(beta, 5)  ///< \f$ K_{\beta} \f$
  ///@}
};
// end elastic isotropic

/**
 * @brief Template specialization for the kernels struct for 2D elastic
 * anisotropic elements
 *
 * @tparam UseSIMD  Use SIMD instructions
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct kernels<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::anisotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : public impl::point_data<7, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<7, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag =
      specfem::element::property_tag::anisotropic;

  constexpr static bool is_point_properties = true;
  ///@}

  using base_type::base_type;

  /**
   * @name Misfit Kernels
   *
   */
  ///@{
  DEFINE_POINT_VALUE(rho, 0) ///< \f$ K_{\rho} \f$
  DEFINE_POINT_VALUE(c11, 1) ///< \f$ K_{c_{11}} \f$
  DEFINE_POINT_VALUE(c13, 2) ///< \f$ K_{c_{13}} \f$
  DEFINE_POINT_VALUE(c15, 3) ///< \f$ K_{c_{15}} \f$
  DEFINE_POINT_VALUE(c33, 4) ///< \f$ K_{c_{33}} \f$
  DEFINE_POINT_VALUE(c35, 5) ///< \f$ K_{c_{35}} \f$
  DEFINE_POINT_VALUE(c55, 6) ///< \f$ K_{c_{55}} \f$
  ///@}
};
// end elastic anisotropic

/**
 * @brief Template specialization for the kernels struct for 2D
 *        elastic isotropic spin elements
 *
 * This specialization is not implemented and should throw an error upon
 * compilation should the code try to use it.
 *
 * @tparam UseSIMD  Use SIMD instructions
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct kernels<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::isotropic_cosserat, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >;

/**
 * @brief Template specialization for the kernels struct for 2D acoustic
 * isotropic elements
 *
 * @tparam UseSIMD  Use SIMD instructions
 */
template <bool UseSIMD>
struct kernels<specfem::dimension::type::dim2,
               specfem::element::medium_tag::acoustic,
               specfem::element::property_tag::isotropic, UseSIMD>
    : public impl::point_data<4, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<4, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::acoustic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  constexpr static bool is_point_properties = true;
  ///@}

  using base_type::base_type;

  /**
   * @brief Constructor
   *
   * @param rho \f$ K_{\rho} \f$
   * @param kappa \f$ K_{\kappa} \f$
   */
  KOKKOS_FUNCTION
  kernels(const value_type rho, const value_type kappa)
      : kernels(rho, kappa, rho * kappa, static_cast<type_real>(2.0) * kappa) {}

  /**
   * @name Misfit Kernels
   *
   */
  ///@{
  DEFINE_POINT_VALUE(rho, 0)   ///< \f$ K_{\rho} \f$
  DEFINE_POINT_VALUE(kappa, 1) ///< \f$ K_{\kappa} \f$
  DEFINE_POINT_VALUE(rhop, 2)  ///< \f$ K_{\rho'} \f$
  DEFINE_POINT_VALUE(alpha, 3) ///< \f$ K_{\alpha} \f$
  ///@}
};

/**
 * @brief Template specialization for the kernels struct for 2D
 *        electromagnetic isotropic elements
 *
 * This specialization is not implemented and should throw an error upon
 * compilation should the code try to use it.
 *
 * @tparam UseSIMD  Use SIMD instructions
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct kernels<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::isotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_electromagnetic<MediumTag>::value> >;

/**
 * @brief Template specialization for the kernels struct for 2D poroelastic
 * isotropic elements
 *
 * @tparam UseSIMD  Use SIMD instructions
 */
template <bool UseSIMD>
struct kernels<specfem::dimension::type::dim2,
               specfem::element::medium_tag::poroelastic,
               specfem::element::property_tag::isotropic, UseSIMD>
    : public impl::point_data<19, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<19, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::poroelastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  constexpr static bool is_point_properties = true;
  ///@}

  using base_type::base_type;
  /**
   * @brief Constructor
   *
   * @param
   */
  KOKKOS_FUNCTION
  kernels(const value_type rhot, const value_type rhof, const value_type eta,
          const value_type sm, const value_type mu_fr, const value_type B,
          const value_type C, const value_type M, const value_type cpI,
          const value_type cpII, const value_type cs, const value_type rhobb,
          const value_type rhofbb, const value_type ratio,
          const value_type phib)
      : kernels(rhot, rhof, eta, sm, mu_fr, B, C, M, mu_fr, (rhot + B + mu_fr),
                (rhof + C + M + sm), (static_cast<value_type>(-1.0) * (sm + M)),
                cpI, cpII, cs, rhobb, rhofbb, ratio, phib) {}

  /**
   * @name Misfit Kernels
   *
   */
  ///@{

  /// Primary Kernels
  DEFINE_POINT_VALUE(rhot, 0)
  DEFINE_POINT_VALUE(rhof, 1)
  DEFINE_POINT_VALUE(eta, 2)
  DEFINE_POINT_VALUE(sm, 3)
  DEFINE_POINT_VALUE(mu_fr, 4)
  DEFINE_POINT_VALUE(B, 5)
  DEFINE_POINT_VALUE(C, 6)
  DEFINE_POINT_VALUE(M, 7)

  /// Density Normalized Kernels
  DEFINE_POINT_VALUE(mu_frb, 8)
  DEFINE_POINT_VALUE(rhob, 9)
  DEFINE_POINT_VALUE(rhofb, 10)
  DEFINE_POINT_VALUE(phi, 11)

  /// wavespeed kernels
  DEFINE_POINT_VALUE(cpI, 12)
  DEFINE_POINT_VALUE(cpII, 13)
  DEFINE_POINT_VALUE(cs, 14)
  DEFINE_POINT_VALUE(rhobb, 15)
  DEFINE_POINT_VALUE(rhofbb, 16)
  DEFINE_POINT_VALUE(ratio, 17)
  DEFINE_POINT_VALUE(phib, 18)
  ///@}
};

} // namespace point
} // namespace specfem
