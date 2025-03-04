#pragma once

#include "impl/point_data.hpp"

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
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct kernels;

/**
 * @brief Template specialization for the kernels struct for 2D elastic
 * isotropic elements
 *
 * @tparam UseSIMD  Use SIMD instructions
 */
template <bool UseSIMD>
struct kernels<specfem::dimension::type::dim2,
               specfem::element::medium_tag::elastic,
               specfem::element::property_tag::isotropic, UseSIMD>
    : public impl::point_data<6, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<6, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  constexpr static bool is_point_properties = true;
  constexpr static int _counter = __COUNTER__;
  ///@}

  using base_type::base_type;

  /**
   * @name Misfit Kernels
   *
   */
  ///@{
  DEFINE_POINT_VALUE(rho)   ///< \f$ K_{\rho} \f$
  DEFINE_POINT_VALUE(mu)    ///< \f$ K_{\mu} \f$
  DEFINE_POINT_VALUE(kappa) ///< \f$ K_{\kappa} \f$
  DEFINE_POINT_VALUE(rhop)  ///< \f$ K_{\rho'} \f$
  DEFINE_POINT_VALUE(alpha) ///< \f$ K_{\alpha} \f$
  DEFINE_POINT_VALUE(beta)  ///< \f$ K_{\beta} \f$
  ///@}
};
// end elastic isotropic

/**
 * @brief Template specialization for the kernels struct for 2D elastic
 * anisotropic elements
 *
 * @tparam UseSIMD  Use SIMD instructions
 */
template <bool UseSIMD>
struct kernels<specfem::dimension::type::dim2,
               specfem::element::medium_tag::elastic,
               specfem::element::property_tag::anisotropic, UseSIMD>
    : public impl::point_data<7, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<7, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::anisotropic;

  constexpr static bool is_point_properties = true;
  constexpr static int _counter = __COUNTER__;
  ///@}

  using base_type::base_type;

  /**
   * @name Misfit Kernels
   *
   */
  ///@{
  DEFINE_POINT_VALUE(rho) ///< \f$ K_{\rho} \f$
  DEFINE_POINT_VALUE(c11) ///< \f$ K_{c_{11}} \f$
  DEFINE_POINT_VALUE(c13) ///< \f$ K_{c_{13}} \f$
  DEFINE_POINT_VALUE(c15) ///< \f$ K_{c_{15}} \f$
  DEFINE_POINT_VALUE(c33) ///< \f$ K_{c_{33}} \f$
  DEFINE_POINT_VALUE(c35) ///< \f$ K_{c_{35}} \f$
  DEFINE_POINT_VALUE(c55) ///< \f$ K_{c_{55}} \f$
  ///@}
};
// end elastic anisotropic

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
  constexpr static int _counter = __COUNTER__;
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
  DEFINE_POINT_VALUE(rho)   ///< \f$ K_{\rho} \f$
  DEFINE_POINT_VALUE(kappa) ///< \f$ K_{\kappa} \f$
  DEFINE_POINT_VALUE(rhop)  ///< \f$ K_{\rho'} \f$
  DEFINE_POINT_VALUE(alpha) ///< \f$ K_{\alpha} \f$
  ///@}
};

} // namespace point
} // namespace specfem
