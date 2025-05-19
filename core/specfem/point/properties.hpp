#pragma once

#include "impl/point_container.hpp"

namespace specfem {
namespace point {

namespace impl {
namespace properties {

/**
 * @brief Data container to hold properties of 2D acoustic media at a quadrature
 * point
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 *
 * @fn const value_type rho_inverse() const
 *   @brief Get the inverse of density @f$ \frac{1}{\rho} @f$
 *   @return const value_type @f$ \frac{1}{\rho} @f$
 *
 * @fn const value_type kappa() const
 *   @brief Get the bulk modulus @f$ \kappa @f$
 *   @return const value_type @f$ \kappa @f$
 *
 * @fn const value_type kappa_inverse() const
 *   @brief Get the inverse of bulk modulus @f$ \frac{1}{\kappa} @f$
 *   @return const value_type @f$ \frac{1}{\kappa} @f$
 *
 * @fn const value_type rho_vpinverse() const
 *   @brief Get the inverse of the product of density and P-wave velocity
 *   @f$ \frac{1}{\rho v_p} @f$
 *   @return const value_type @f$ \frac{1}{\rho v_p} @f$
 *
 */
template <bool UseSIMD>
struct data_container<specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, UseSIMD>
    : traits<specfem::dimension::type::dim2,
             specfem::element::medium_tag::acoustic,
             specfem::element::property_tag::isotropic, UseSIMD> {

  using base_type = traits<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::acoustic,
                           specfem::element::property_tag::isotropic, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  POINT_CONTAINER(rho_inverse, kappa)

  KOKKOS_INLINE_FUNCTION const value_type kappa_inverse() const {
    return (static_cast<type_real>(1.0)) /
           kappa(); ///< @f$ \frac{1}{\lambda + 2\mu} @f$
  }

  KOKKOS_INLINE_FUNCTION const value_type rho_vpinverse() const {
    return Kokkos::sqrt(rho_inverse() * kappa_inverse()); ///< @f$ \frac{1}{\rho
                                                          ///< v_p} @f$
  }
};

/**
 * @defgroup specfem_point_properties_dim2_elastic_isotropic 2D Elastic
 * Isotropic Properties
 * @{
 */

/**
 * @brief Data container to hold properties of 2D elastic media at a quadrature
 * point
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 *
 * @fn const value_type lambdaplus2mu() const
 *   @brief Get Lame's parameter @f$ \lambda + 2\mu @f$
 *   @return const value_type @f$ \lambda + 2\mu @f$
 *
 * @fn const value_type mu() const
 *   @brief Get shear modulus @f$ \mu @f$
 *   @return const value_type @f$ \mu @f$
 *
 * @fn const value_type rho() const
 *   @brief Get density @f$ \rho @f$
 *   @return const value_type @f$ \rho @f$
 *
 * @fn const value_type rho_vp() const
 *   @brief Compute the product of density and P-wave
 *   velocity squared, i.e., @f$ \rho v_p^2 = \rho
 *   (\lambda + 2\mu) @f$
 *   @return const value_type The value of @f$ \rho
 *   (\lambda + 2\mu) @f$
 *
 * @fn const value_type rho_vs() const
 *   @brief Compute the product of density and S-wave
 *   velocity squared, i.e., @f$ \rho v_s^2 = \rho \mu
 *   @f$
 *   @return const value_type The value of @f$ \rho \mu
 *   @f$
 *
 * @fn const value_type lambda() const
 *   @brief Get Lame's first parameter @f$ \lambda @f$
 *   from @f$ \lambda + 2\mu @f$ and @f$ \mu @f$
 *   @return const value_type The value of @f$ \lambda =
 *   (\lambda + 2\mu) - 2\mu @f$
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct data_container<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::isotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : public traits<specfem::dimension::type::dim2, MediumTag,
                    specfem::element::property_tag::isotropic, UseSIMD> {

  using base_type = traits<specfem::dimension::type::dim2, MediumTag,
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
    return lambdaplus2mu() - (static_cast<type_real>(2.0)) * mu();
  }
};
///@} end of group specfem_point_properties_dim2_elastic_isotropic

/**
 * @defgroup specfem_point_properties_dim2_elastic_anisotropic 2D Elastic
 * Anisotropic Properties
 * @{
 */
/**
 * @brief Data container to hold properties of 2D anisotropic elastic media at a
 * quadrature point
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 *
 * @fn const value_type c11() const
 *   @brief Get stiffness tensor component @f$ c_{11} @f$.
 *   @return The value of @f$ c_{11} @f$.
 *
 * @fn const value_type c13() const
 *   @brief Get stiffness tensor component @f$ c_{13} @f$.
 *   @return The value of @f$ c_{13} @f$.
 *
 * @fn const value_type c15() const
 *   @brief Get stiffness tensor component @f$ c_{15} @f$.
 *   @return The value of @f$ c_{15} @f$.
 *
 * @fn const value_type c33() const
 *   @brief Get stiffness tensor component @f$ c_{33} @f$.
 *   @return The value of @f$ c_{33} @f$.
 *
 * @fn const value_type c35() const
 *   @brief Get stiffness tensor component @f$ c_{35} @f$.
 *   @return The value of @f$ c_{35} @f$.
 *
 * @fn const value_type c55() const
 *   @brief Get stiffness tensor component @f$ c_{55} @f$.
 *   @return The value of @f$ c_{55} @f$.
 *
 * @fn const value_type c12() const
 *   @brief Get stiffness tensor component @f$ c_{12} @f$.
 *   @return The value of @f$ c_{12} @f$.
 *
 * @fn const value_type c23() const
 *   @brief Get stiffness tensor component @f$ c_{23} @f$.
 *   @return The value of @f$ c_{23} @f$.
 *
 * @fn const value_type c25() const
 *   @brief Get stiffness tensor component @f$ c_{25} @f$.
 *   @return The value of @f$ c_{25} @f$.
 *
 * @fn const value_type rho() const
 *   @brief Get the density @f$ \rho @f$.
 *   @return The value of @f$ \rho @f$.
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct data_container<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::anisotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : public traits<specfem::dimension::type::dim2, MediumTag,
                    specfem::element::property_tag::anisotropic, UseSIMD> {

  using base_type =
      traits<specfem::dimension::type::dim2, MediumTag,
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
///@} end of group specfem_point_properties_dim2_elastic_anisotropic

/**
 * @brief Data container to hold properties of 2D poroelastic media at a
 * quadrature point
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 *
 * @fn const value_type phi() const
 *   @brief Get porosity @f$ \phi @f$
 *   @return The value of @f$ \phi @f$
 *
 * @fn const value_type rho_s() const
 *   @brief Get solid density @f$ \rho_s @f$
 *   @return The value of @f$ \rho_s @f$
 *
 * @fn const value_type rho_f() const
 *   @brief Get fluid density @f$ \rho_f @f$
 *   @return The value of @f$ \rho_f @f$
 *
 * @fn const value_type tortuosity() const
 *   @brief Get tortuosity @f$ \tau @f$
 *   @return The value of @f$ \tau @f$
 *
 * @fn const value_type mu_G() const
 *   @brief Get shear modulus @f$ \mu_G @f$
 *   @return The value of @f$ \mu_G @f$
 *
 * @fn const value_type H_Biot() const
 *   @brief Get Biot's modulus @f$ H_Biot @f$
 *   @return The value of @f$ H_Biot @f$
 *
 * @fn const value_type C_Biot() const
 *   @brief Get Biot's modulus @f$ C_Biot @f$
 *   @return The value of @f$ C_Biot @f$
 *
 * @fn const value_type M_Biot() const
 *   @brief Get Biot's modulus @f$ M_Biot @f$
 *   @return The value of @f$ M_Biot @f$
 *
 * @fn const value_type permxx() const
 *   @brief Get permeability tensor component @f$ k_{xx} @f$
 *   @return The value of @f$ k_{xx} @f$
 *
 * @fn const value_type permxz() const
 *   @brief Get permeability tensor component @f$ k_{xz} @f$
 *   @return The value of @f$ k_{xz} @f$
 *
 * @fn const value_type permzz() const
 *   @brief Get permeability tensor component @f$ k_{zz} @f$
 *   @return The value of @f$ k_{zz} @f$
 *
 * @fn const value_type eta_f() const
 *   @brief Get fluid viscosity @f$ \eta_f @f$
 *   @return The value of @f$ \eta_f @f$
 *
 * @fn const value_type lambda_G() const
 *   @brief Get Lame's parameter @f$ \lambda_G @f$
 *   @return The value of @f$ \lambda_G @f$
 *
 * @fn const value_type lambdaplus2mu_G() const
 *   @brief Get Lame's parameter @f$ \lambda + 2\mu_G @f$
 *   @return The value of @f$ \lambda + 2\mu_G @f$
 *
 * @fn const value_type inverse_permxx() const
 *   @brief Get the inverse of permeability tensor component @f$ k_{xx} @f$
 *   @return The value of @f$ \frac{1}{k_{xx}} @f$
 *
 * @fn const value_type inverse_permxz() const
 *   @brief Get the inverse of permeability tensor component @f$ k_{xz} @f$
 *   @return The value of @f$ \frac{1}{k_{xz}} @f$
 *
 * @fn const value_type inverse_permzz() const
 *   @brief Get the inverse of permeability tensor component @f$ k_{zz} @f$
 *   @return The value of @f$ \frac{1}{k_{zz}} @f$
 *
 * @fn const value_type rho_bar() const
 *   @brief Get the average density @f$ \rho_{bar} @f$
 *   @return The value of @f$ \rho_{bar} @f$
 *
 * @fn const value_type vpI() const
 *   @brief Get the P-wave velocity @f$ v_{pI} @f$
 *   @return The value of @f$ v_{pI} @f$
 *
 * @fn const value_type vpII() const
 *   @brief Get the P-wave velocity @f$ v_{pII} @f$
 *   @return The value of @f$ v_{pII} @f$
 *
 * @fn const value_type vs() const
 *   @brief Get the S-wave velocity @f$ v_s @f$
 *   @return The value of @f$ v_s @f$
 */
template <bool UseSIMD>
struct data_container<specfem::dimension::type::dim2,
                      specfem::element::medium_tag::poroelastic,
                      specfem::element::property_tag::isotropic, UseSIMD>
    : public traits<specfem::dimension::type::dim2,
                    specfem::element::medium_tag::poroelastic,
                    specfem::element::property_tag::isotropic, UseSIMD> {

  using base_type = traits<specfem::dimension::type::dim2,
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
    return H_Biot() - (static_cast<type_real>(2.0)) * mu_G();
  }

  KOKKOS_INLINE_FUNCTION const value_type lambdaplus2mu_G() const {
    return lambda_G() + (static_cast<type_real>(2.0)) * mu_G();
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
    return static_cast<type_real>(-1.0) * permxz() /
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
        static_cast<type_real>(2.0) * phi_over_tort * this->C_Biot();
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
        static_cast<type_real>(2.0) * phi_over_tort * this->C_Biot();
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
 * @brief Data container to hold properties of 2D electromagnetic media at a
 * quadrature point
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 *
 * @fn const value_type mu0_inv() const
 *   @brief Get the inverse of permeability @f$ \frac{1}{\mu_0} @f$
 *   @return The value of @f$ \frac{1}{\mu_0} @f$
 *
 * @fn const value_type eps11() const
 *   @brief Get permittivity tensor component @f$ \epsilon_{11} @f$
 *   @return The value of @f$ \epsilon_{11} @f$
 *
 * @fn const value_type eps33() const
 *   @brief Get permittivity tensor component @f$ \epsilon_{33} @f$
 *   @return The value of @f$ \epsilon_{33} @f$
 *
 * @fn const value_type sig11() const
 *   @brief Get conductivity tensor component @f$ \sigma_{11} @f$
 *   @return The value of @f$ \sigma_{11} @f$
 *
 * @fn const value_type sig33() const
 *   @brief Get conductivity tensor component @f$ \sigma_{33} @f$
 *   @return The value of @f$ \sigma_{33} @f$
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct data_container<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::isotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_electromagnetic<MediumTag>::value> >
    : public traits<specfem::dimension::type::dim2, MediumTag,
                    specfem::element::property_tag::isotropic, UseSIMD> {

  using base_type = traits<specfem::dimension::type::dim2, MediumTag,
                           specfem::element::property_tag::isotropic, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  POINT_CONTAINER(mu0_inv, eps11, eps33, sig11, sig33)
};

} // namespace properties

} // namespace impl

/**
 * @brief Properties of a quadrature point in a 2D medium
 *
 * @tparam Dimension The dimension of the medium
 * @tparam MediumTag The type of the medium
 * @tparam PropertyTag The type of the properties
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 */
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
