#pragma once

#include "impl/point_container.hpp"

namespace specfem {
namespace point {

namespace impl {

namespace properties {

template <bool UseSIMD>
struct data_container<specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, UseSIMD>
    : point_traits<specfem::dimension::type::dim2,
                   specfem::element::medium_tag::acoustic,
                   specfem::element::property_tag::isotropic, UseSIMD> {

  using base_type =
      point_traits<specfem::dimension::type::dim2,
                   specfem::element::medium_tag::acoustic,
                   specfem::element::property_tag::isotropic, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  POINT_CONTAINER(rho_inverse,
                  kappa) ///< Inverse density @f$ \rho^{-1} @f$ and bulk modulus
  ///< @f$ \kappa @f$

  KOKKOS_INLINE_FUNCTION const value_type kappa_inverse() const {
    return (static_cast<value_type>(1.0)) /
           kappa(); ///< @f$ \frac{1}{\lambda + 2\mu} @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type rho_vpinverse() const {
    return Kokkos::sqrt(rho_inverse() * kappa_inverse()); ///< @f$ \frac{1}{\rho
                                                          ///< v_p} @f$
  }
};

template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct data_container<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::isotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : public point_traits<specfem::dimension::type::dim2, MediumTag,
                          specfem::element::property_tag::isotropic, UseSIMD> {

  using base_type =
      point_traits<specfem::dimension::type::dim2, MediumTag,
                   specfem::element::property_tag::isotropic, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  POINT_CONTAINER(lambdaplus2mu, mu, rho)

  KOKKOS_INLINE_FUNCTION const value_type rho_vp() const {
    return Kokkos::sqrt(rho() * lambdaplus2mu());
  }

  KOKKOS_INLINE_FUNCTION const value_type rho_vs() const {
    return Kokkos::sqrt(rho() * mu());
  }

  KOKKOS_INLINE_FUNCTION const value_type lambda() const {
    return lambdaplus2mu() - (static_cast<value_type>(2.0)) * mu();
  }
};

template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct data_container<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::anisotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : public point_traits<specfem::dimension::type::dim2, MediumTag,
                          specfem::element::property_tag::anisotropic,
                          UseSIMD> {

  using base_type =
      point_traits<specfem::dimension::type::dim2, MediumTag,
                   specfem::element::property_tag::anisotropic, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  POINT_CONTAINER(c11, c13, c15, c33, c35, c55, c12, c23, c25, rho)

  KOKKOS_INLINE_FUNCTION const value_type rho_vp() const {
    return Kokkos::sqrt(rho() * c33()); ///< P-wave velocity @f$ \rho v_p @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type rho_vs() const {
    return Kokkos::sqrt(rho() * c55()); ///< S-wave velocity @f$ \rho v_s @f$
  }
};

template <bool UseSIMD>
struct data_container<specfem::dimension::type::dim2,
                      specfem::element::medium_tag::poroelastic,
                      specfem::element::property_tag::isotropic, UseSIMD>
    : public point_traits<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::poroelastic,
                          specfem::element::property_tag::isotropic, UseSIMD> {

  using base_type =
      point_traits<specfem::dimension::type::dim2,
                   specfem::element::medium_tag::poroelastic,
                   specfem::element::property_tag::isotropic, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  POINT_CONTAINER(phi, rho_s, rho_f, tortuosity, mu_G, H_Biot, C_Biot, M_Biot,
                  permxx, permxz, permzz, eta_f)

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
struct data_container<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::isotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_electromagnetic<MediumTag>::value> >
    : public point_traits<specfem::dimension::type::dim2, MediumTag,
                          specfem::element::property_tag::isotropic, UseSIMD> {

  using base_type =
      point_traits<specfem::dimension::type::dim2, MediumTag,
                   specfem::element::property_tag::isotropic, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  POINT_CONTAINER(mu0_inv, eps11, eps33, sig11, sig33)
};

} // namespace properties

} // namespace impl

template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct properties : impl::properties::data_container<Dimension, MediumTag,
                                                     PropertyTag, UseSIMD> {
  using base_type = impl::properties::data_container<Dimension, MediumTag,
                                                     PropertyTag, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  using base_type::base_type;
};

} // namespace point
} // namespace specfem
