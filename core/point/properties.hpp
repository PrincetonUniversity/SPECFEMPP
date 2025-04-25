#pragma once

#include "impl/point_data.hpp"

namespace specfem {
namespace point {

/**
 * @brief Store properties of the medium at a quadrature point
 *
 * @tparam DimensionTag Dimension of the spectral element
 * @tparam MediumTag Tag indicating the medium of the element
 * @tparam PropertyTag Tag indicating the property of the medium
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD,
          typename Enable = void>
struct properties;

/**
 * @brief Template specialization for 2D isotropic elastic media
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct properties<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::isotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : public impl::point_data<3, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<3, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  constexpr static bool is_point_properties = true;
  ///@}

  using base_type::base_type;

  DEFINE_POINT_VALUE(lambdaplus2mu,
                     0)      ///< Lame's parameter @f$ \lambda + 2\mu @f$
  DEFINE_POINT_VALUE(mu, 1)  ///< shear modulus @f$ \mu @f$
  DEFINE_POINT_VALUE(rho, 2) ///< density @f$ \rho @f$

  KOKKOS_INLINE_FUNCTION const value_type rho_vp() const {
    return Kokkos::sqrt(rho() * lambdaplus2mu()); ///< P-wave velocity @f$ \rho
                                                  ///< v_p @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type rho_vs() const {
    return Kokkos::sqrt(rho() * mu()); ///< S-wave velocity @f$ \rho v_s @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type lambda() const {
    return lambdaplus2mu() - (static_cast<value_type>(2.0)) *
                                 mu(); ///< Lame's parameter @f$ \lambda @f$
  }
};

template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct properties<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::anisotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : public impl::point_data<10, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<10, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag =
      specfem::element::property_tag::anisotropic;

  constexpr static bool is_point_properties = true;
  ///@}

  using base_type::base_type;

  /**
   * @name Elastic constants
   *
   */
  ///@{
  DEFINE_POINT_VALUE(c11, 0) ///< @f$ c_{11} @f$
  DEFINE_POINT_VALUE(c13, 1) ///< @f$ c_{13} @f$
  DEFINE_POINT_VALUE(c15, 2) ///< @f$ c_{15} @f$
  DEFINE_POINT_VALUE(c33, 3) ///< @f$ c_{33} @f$
  DEFINE_POINT_VALUE(c35, 4) ///< @f$ c_{35} @f$
  DEFINE_POINT_VALUE(c55, 5) ///< @f$ c_{55} @f$
  DEFINE_POINT_VALUE(c12, 6) ///< @f$ c_{12} @f$
  DEFINE_POINT_VALUE(c23, 7) ///< @f$ c_{23} @f$
  DEFINE_POINT_VALUE(c25, 8) ///< @f$ c_{25} @f$
  DEFINE_POINT_VALUE(rho, 9) ///< Density @f$ \rho @f$
  ///@}

  KOKKOS_INLINE_FUNCTION const value_type rho_vp() const {
    return Kokkos::sqrt(rho() * c33()); ///< P-wave velocity @f$ \rho v_p @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type rho_vs() const {
    return Kokkos::sqrt(rho() * c55()); ///< S-wave velocity @f$ \rho v_s @f$
  }
};

/**
 * @brief Template specialization for 2D isotropic acoustic media
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct properties<specfem::dimension::type::dim2,
                  specfem::element::medium_tag::acoustic,
                  specfem::element::property_tag::isotropic, UseSIMD>
    : public impl::point_data<2, UseSIMD> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<2, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::acoustic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  constexpr static bool is_point_properties = true;
  ///@}

  using base_type::base_type;

  DEFINE_POINT_VALUE(rho_inverse, 0) ///< @f$ \frac{1}{\rho} @f$
  DEFINE_POINT_VALUE(kappa, 1)       ///< Bulk modulus @f$ \kappa @f$

  KOKKOS_INLINE_FUNCTION const value_type kappa_inverse() const {
    return (static_cast<value_type>(1.0)) /
           kappa(); ///< @f$ \frac{1}{\lambda + 2\mu} @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type rho_vpinverse() const {
    return Kokkos::sqrt(rho_inverse() * kappa_inverse()); ///< @f$ \frac{1}{\rho
                                                          ///< v_p} @f$
  }
};

template <bool UseSIMD>
struct properties<specfem::dimension::type::dim2,
                  specfem::element::medium_tag::poroelastic,
                  specfem::element::property_tag::isotropic, UseSIMD>
    : public impl::point_data<12, UseSIMD> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<12, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::poroelastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  constexpr static bool is_point_properties = true;
  ///@}

  using base_type::base_type;

  DEFINE_POINT_VALUE(phi, 0)        ///< porosity @f$ \phi @f$
  DEFINE_POINT_VALUE(rho_s, 1)      ///< solid density @f$ \rho_s @f$
  DEFINE_POINT_VALUE(rho_f, 2)      ///< fluid density @f$ \rho_f @f$
  DEFINE_POINT_VALUE(tortuosity, 3) ///< tortuosity @f$ \tau @f$
  DEFINE_POINT_VALUE(mu_G, 4)
  DEFINE_POINT_VALUE(H_Biot, 5)
  DEFINE_POINT_VALUE(C_Biot, 6)
  DEFINE_POINT_VALUE(M_Biot, 7)
  DEFINE_POINT_VALUE(permxx, 8)
  DEFINE_POINT_VALUE(permxz, 9)
  DEFINE_POINT_VALUE(permzz, 10)
  DEFINE_POINT_VALUE(eta_f, 11) ///< Viscosity @f$ \eta @f$

  /**
   * @brief Compute Lame's parameter @f$ \lambda @f$
   *
   * @return Lame's parameter @f$ \lambda @f$
   */
  KOKKOS_INLINE_FUNCTION const value_type lambda_G() const {
    return H_Biot() - (static_cast<value_type>(2.0)) * mu_G();
  }

  KOKKOS_INLINE_FUNCTION const value_type lambdaplus2mu_G() const {
    return lambda_G() + (static_cast<value_type>(2.0)) * mu_G();
  }

  KOKKOS_INLINE_FUNCTION const value_type inverse_permxx() const {
    const value_type determinant =
        permxx() * permzz() - permxz() * permxz(); ///< determinant of the
                                                   ///< permeability tensor
    return permzz() / determinant; ///< inverse of the permeability tensor
  }

  KOKKOS_INLINE_FUNCTION const value_type inverse_permxz() const {
    const value_type determinant =
        permxx() * permzz() - permxz() * permxz(); ///< determinant of the
                                                   ///< permeability tensor
    return static_cast<value_type>(-1.0) * permxz() /
           determinant; ///< inverse of the permeability tensor
  }

  KOKKOS_INLINE_FUNCTION const value_type inverse_permzz() const {
    const value_type determinant =
        permxx() * permzz() - permxz() * permxz(); ///< determinant of the
                                                   ///< permeability tensor
    return permxx() / determinant; ///< inverse of the permeability tensor
  }

  KOKKOS_INLINE_FUNCTION const value_type rho_bar() const {
    return (((static_cast<type_real>(1.0) - this->phi()) * this->rho_s()) +
            (this->phi() * this->rho_f()));
  }

  KOKKOS_INLINE_FUNCTION const value_type vpI() const {
    ///< Helper variable for readability
    const auto phi_over_tort = this->phi() / this->tortuosity();
    const auto afactor = rho_bar() - phi_over_tort * rho_f();
    const auto bfactor =
        this->H_Biot() + phi_over_tort * rho_bar() / rho_f() * this->M_Biot() -
        static_cast<value_type>(2.0) * phi_over_tort * this->C_Biot();
    const auto cfactor =
        phi_over_tort / rho_f() *
        (this->H_Biot() * this->M_Biot() - this->C_Biot() * this->C_Biot());

    return Kokkos::sqrt(bfactor + Kokkos::sqrt(bfactor * bfactor -
                                               static_cast<type_real>(4.0) *
                                                   afactor * cfactor)) /
           (static_cast<type_real>(2.0) * afactor);
  }

  KOKKOS_INLINE_FUNCTION const value_type vpII() const {
    ///< Helper variable for readability
    const auto phi_over_tort = this->phi() / this->tortuosity();
    const auto afactor = rho_bar() - phi_over_tort * rho_f();
    const auto bfactor =
        this->H_Biot() + phi_over_tort * rho_bar() / rho_f() * this->M_Biot() -
        static_cast<value_type>(2.0) * phi_over_tort * this->C_Biot();
    const auto cfactor =
        phi_over_tort / rho_f() *
        (this->H_Biot() * this->M_Biot() - this->C_Biot() * this->C_Biot());

    return Kokkos::sqrt(bfactor - Kokkos::sqrt(bfactor * bfactor -
                                               static_cast<type_real>(4.0) *
                                                   afactor * cfactor)) /
           (static_cast<type_real>(2.0) * afactor);
  }

  KOKKOS_INLINE_FUNCTION const value_type vs() const {
    ///< Helper variable for readability
    const auto phi_over_tort = this->phi() / this->tortuosity();
    const auto afactor = rho_bar() - phi_over_tort * rho_f();
    return Kokkos::sqrt(mu_G() / afactor);
  }
};

/**
 * @brief Template specialization for 2D isotropic electromagnetic media
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct properties<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::isotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_electromagnetic<MediumTag>::value> >
    : public impl::point_data<5, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::point_data<5, UseSIMD>;
  using value_type = typename base_type::value_type;

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag =
      specfem::element::medium_tag::electromagnetic_te;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  constexpr static bool is_point_properties = true;
  ///@}

  using base_type::base_type;

  /**
   * @name Material properties
   *
   */
  ///@{
  DEFINE_POINT_VALUE(mu0_inv,
                     0) ///< Inverse magnetic permeability @f$ \mu_{0}^{-1} @f$
  DEFINE_POINT_VALUE(eps11, 1) ///< @f$ \epsilon_{11} @f$ component of the
                               ///< permittivity tensor
  DEFINE_POINT_VALUE(eps33, 2) ///< @f$ \epsilon_{33} @f$ component of the
                               ///< permittivity tensor
  DEFINE_POINT_VALUE(sig11, 3) ///< @f$ \sigma_{11} @f$ component of the
                               ///< conductivity tensor
  DEFINE_POINT_VALUE(sig33, 4) ///< @f$ \sigma_{33} @f$ component of the
                               ///< conductivity tensor
  ///@}
};

} // namespace point
} // namespace specfem
