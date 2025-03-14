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

template <bool UseSIMD>
struct kernels<specfem::dimension::type::dim2,
               specfem::element::medium_tag::elastic_sv,
               specfem::element::property_tag::isotropic, UseSIMD>
    : public kernels<specfem::dimension::type::dim2,
                     specfem::element::medium_tag::elastic,
                     specfem::element::property_tag::isotropic, UseSIMD> {

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd =
      typename specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  using value_type =
      typename simd::datatype; ///< Underlying data type to store the kernels
  ///@}

  constexpr static auto medium_tag = specfem::element::medium_tag::elastic_sv;

private:
  using base_type = kernels<specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic, UseSIMD>;

public:
  using base_type::base_type;
};

template <bool UseSIMD>
struct kernels<specfem::dimension::type::dim2,
               specfem::element::medium_tag::elastic_sh,
               specfem::element::property_tag::isotropic, UseSIMD>
    : public kernels<specfem::dimension::type::dim2,
                     specfem::element::medium_tag::elastic,
                     specfem::element::property_tag::isotropic, UseSIMD> {

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd =
      typename specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  using value_type =
      typename simd::datatype; ///< Underlying data type to store the kernels
  ///@}

  constexpr static auto medium_tag = specfem::element::medium_tag::elastic_sh;

private:
  using base_type = kernels<specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic, UseSIMD>;

public:
  using base_type::base_type;
};

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

template <bool UseSIMD>
struct kernels<specfem::dimension::type::dim2,
               specfem::element::medium_tag::elastic_sv,
               specfem::element::property_tag::anisotropic, UseSIMD>
    : public kernels<specfem::dimension::type::dim2,
                     specfem::element::medium_tag::elastic,
                     specfem::element::property_tag::anisotropic, UseSIMD> {

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd =
      typename specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  using value_type =
      typename simd::datatype; ///< Underlying data type to store the kernels
  ///@}
  constexpr static auto medium_tag = specfem::element::medium_tag::elastic_sv;

private:
  using base_type =
      kernels<specfem::dimension::type::dim2,
              specfem::element::medium_tag::elastic,
              specfem::element::property_tag::anisotropic, UseSIMD>;

public:
  using base_type::base_type;
};

template <bool UseSIMD>
struct kernels<specfem::dimension::type::dim2,
               specfem::element::medium_tag::elastic_sh,
               specfem::element::property_tag::anisotropic, UseSIMD>
    : public kernels<specfem::dimension::type::dim2,
                     specfem::element::medium_tag::elastic,
                     specfem::element::property_tag::anisotropic, UseSIMD> {

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd =
      typename specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  using value_type =
      typename simd::datatype; ///< Underlying data type to store the kernels
  ///@}

  constexpr static auto medium_tag = specfem::element::medium_tag::elastic_sh;

private:
  using base_type =
      kernels<specfem::dimension::type::dim2,
              specfem::element::medium_tag::elastic,
              specfem::element::property_tag::anisotropic, UseSIMD>;

public:
  using base_type::base_type;
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
 * @brief Template specialization for the kernels struct for 2D electromagnetic
 * isotropic elements
 * @tparam UseSIMD  Use SIMD instructions
 */
template <bool UseSIMD>
struct kernels<specfem::dimension::type::dim2,
               specfem::element::medium_tag::electromagnetic_sv,
               specfem::element::property_tag::isotropic, UseSIMD>
    : public impl::point_data<10, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<10, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag =
      specfem::element::medium_tag::electromagnetic_sv;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  constexpr static bool is_point_properties = true;
  ///@}

  using base_type::base_type;

  /**
   * @brief Constructor
   *
   * @param mu0 \f$ K_{{\mu}_{0}} \f$ Magnetic permeability in henry per meter
   * @param e0  \f$ K_{{\epsilon}_{0}} \f$ Effective permittivity in farad per
   * meter
   * @param e11 \f$ K_{{\epsilon}^{11}{\epsilon}_{0}} \f$ effective permittivity
   * in farad per meter
   * @param e33 \f$ K_{{\epsilon}^{33}{\epsilon}_{0}} \f$ effective permittivity
   * in farad per meter
   * @param sig11 \f$ K_{{\sigma}^{11}{\sigma}_{0}} \f$ of effective
   * conductivity in siemens per meter
   * @param sig33 \f$ K_{{\sigma}^{33}{\sigma}_{0}} \f$  of effective
   * conductivity in siemens per meter
   * @param Qe11 \f$ K_{ Q_{ \epsilon^{11} } } \f$ Quality factor of e11 for
   * attenuation
   * @param Qe33 \f$ K_{ Q_{ \epsilon^{33} } } \f$ Quality factor of e33 for
   * attenuation
   * @param Qs11 \f$ K_{ Q_{ \sigma^{11} } } \f$ Quality factor of sig11 for
   * attenuation
   * @param Qs33 \f$ K_{ Q_{ \sigma^{11} } } \f$ Quality factor of sig33 for
   * attenuation
   */
  KOKKOS_FUNCTION
  kernels(const value_type mu0, const value_type e0, const value_type e11,
          const value_type e33, const value_type sig11, const value_type sig33,
          const value_type Qe11, const value_type Qe33, const value_type Qs11,
          const value_type Qs33)
      : kernels(mu0, e0, e11, e33, sig11, sig33, Qe11, Qe33, Qs11, Qs33) {}

  /**
   * @name Misfit Kernels
   *
   */
  ///@{
  DEFINE_POINT_VALUE(mu0, 0)   ///< \f$ K_{{\mu}_{0}} \f$
  DEFINE_POINT_VALUE(e0, 1)    ///< \f$ K_{{\epsilon}_{0}} \f$
  DEFINE_POINT_VALUE(e11, 2)   ///< \f$ K_{{\epsilon}^{11}{\epsilon}_{0}} \f$
                               ///< effective permittivity in farad per meter
  DEFINE_POINT_VALUE(e33, 3)   ///< \f$ K_{{\epsilon}^{33}{\epsilon}_{0}} \f$
                               ///< effective permittivity in farad per meter
  DEFINE_POINT_VALUE(sig11, 4) ///< \f$ K_{{\sigma}^{11}{\sigma}_{0}} \f$ of
                               ///< effective conductivity in siemens per meter
  DEFINE_POINT_VALUE(sig33, 5) ///< \f$ K_{{\sigma}^{33}{\sigma}_{0}} \f$  of
                               ///< effective conductivity in siemens per meter
  DEFINE_POINT_VALUE(Qe11, 6)  ///< \f$ K_{ Q_{ \epsilon^{11} } } \f$ Quality
                               ///< factor of e11 for attenuation
  DEFINE_POINT_VALUE(Qe33, 7)  ///< \f$ K_{ Q_{ \epsilon^{33} } } \f$ Quality
                               ///< factor of e33 for attenuation
  DEFINE_POINT_VALUE(Qs11, 8)  ///< \f$ K_{ Q_{ \sigma^{11} } } \f$ Quality
                               ///< factor of sig11 for attenuation
  DEFINE_POINT_VALUE(Qs33, 9)  ///< \f$ K_{ Q_{ \sigma^{11} } } \f$ Quality
                               ///< factor of sig33 for attenuation
  ///@}
};

} // namespace point
} // namespace specfem
