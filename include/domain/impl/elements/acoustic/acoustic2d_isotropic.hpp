#ifndef _DOMAIN_ACOUSTIC_ELEMENTS2D_IMPLEMENTATION_HPP
#define _DOMAIN_ACOUSTIC_ELEMENTS2D_IMPLEMENTATION_HPP

#include "compute/interface.hpp"
#include "domain/impl/elements/acoustic/acoustic2d.hpp"
#include "domain/impl/elements/element.hpp"
#include "kokkos_abstractions.h"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

/**
 * @brief Decltype for the field subviewed at particular global index
 *
 */
using field_type = Kokkos::Subview<
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>, int,
    std::remove_const_t<decltype(Kokkos::ALL)> >;

namespace specfem {
namespace domain {
namespace impl {
namespace elements {
/**
 * @brief Elastic 2D isotropic element class with number of quadrature points
 * defined at compile time
 *
 * @tparam N Number of Gauss-Lobatto-Legendre quadrature points
 */
template <int N>
class element<specfem::enums::element::dimension::dim2,
              specfem::enums::element::medium::elastic,
              specfem::enums::element::quadrature::static_quadrature_points<N>,
              specfem::enums::element::property::isotropic>
    : public element<
          specfem::enums::element::dimension::dim2,
          specfem::enums::element::medium::elastic,
          specfem::enums::element::quadrature::static_quadrature_points<N> > {
public:
  /**
   * @brief Number of Gauss-Lobatto-Legendre quadrature points
   */
  using quadrature_points =
      specfem::enums::element::quadrature::static_quadrature_points<N>;

  /**
   * @brief Use the scratch view type from the quadrature points
   *
   * @tparam T Type of the scratch view
   */
  template <typename T>
  using ScratchViewType =
      typename quadrature_points::template ScratchViewType<T>;

  /**
   * @brief Construct a new element object
   *
   */
  KOKKOS_FUNCTION
  element() = default;

  /**
   * @brief Construct a new element object
   *
   * @param ispec Index of the element
   * @param partial_derivatives partial derivatives
   * @param properties Properties of the element
   */
  KOKKOS_FUNCTION
  element(const int ispec,
          const specfem::compute::partial_derivatives partial_derivatives,
          const specfem::compute::properties properties);

  /**
   * @brief Compute the gradient of the field at a particular
   * Gauss-Lobatto-Legendre quadrature point
   *
   * @param xz Index of Gauss-Lobatto-Legendre quadrature point
   * @param s_hprime_xx Scratch view of derivative of Lagrange polynomial in x
   * direction
   * @param s_hprime_zz Scratch view of derivative of Lagrange polynomial in z
   * direction
   * @param field_p Scratch view of potential field
   * @param dpdxl Computed partial derivative of field \f$ \frac{\partial
   * u_x}{\partial x} \f$
   * @param dpdzl Computed partial derivative of field \f$ \frac{\partial
   * u_x}{\partial z} \f$
   */
  KOKKOS_INLINE_FUNCTION void
  compute_gradient(const int &xz, const ScratchViewType<type_real> s_hprime_xx,
                   const ScratchViewType<type_real> s_hprime_zz,
                   const ScratchViewType<type_real> field_p, type_real *dpdxl,
                   type_real *dpdzl) const override;

  /**
   * @brief Compute the stress integrand at a particular Gauss-Lobatto-Legendre
   * quadrature point
   *
   * @param xz Index of Gauss-Lobatto-Legendre quadrature point
   * @param dpdxl Partial derivative of field \f$ \frac{\partial p}{\partial
   * x} \f$
   * @param dpdzl Partial derivative of field \f$ \frac{\partial p}{\partial
   * z} \f$
   * @param stress_integrand_1l Stress integrand jacobianl * (sigma_xx * xixl +
   * sigma_xz * xizl) at a particular Gauss-Lobatto-Legendre quadrature point xz
   * @param stress_integrand_2l Stress integrand jacobianl * (sigma_xz * xixl +
   * sigma_zz * xizl) at a particular Gauss-Lobatto-Legendre quadrature point xz
   * @param stress_integrand_3l Stress integrand jacobianl * (sigma_xx * gammaxl
   * + sigma_xz * gammazl) at a particular Gauss-Lobatto-Legendre quadrature
   * point xz
   * @param stress_integrand_4l Stress integrand jacobianl * (sigma_xz * gammaxl
   * + sigma_zz * gammazl) at a particular Gauss-Lobatto-Legendre quadrature
   * point xz
   * @return KOKKOS_FUNCTION
   */
  KOKKOS_INLINE_FUNCTION void
  compute_stress(const int &xz, const type_real &duxdxl,
                 const type_real &duxdzl, const type_real &duzdxl,
                 const type_real &duzdzl, type_real *stress_integrand_1l,
                 type_real *stress_integrand_2l, type_real *stress_integrand_3l,
                 type_real *stress_integrand_4l) const override;

  /**
   * @brief Update the acceleration at a particular Gauss-Lobatto-Legendre
   * quadrature point
   *
   * @param xz Index of Gauss-Lobatto-Legendre quadrature point
   * @param wxglll Weight of the Gauss-Lobatto-Legendre quadrature point in x
   * direction
   * @param wzglll Weight of the Gauss-Lobatto-Legendre quadrature point in z
   * direction
   * @param stress_integrand_1 Stress integrand jacobianl * (sigma_xx * xixl +
   * sigma_xz * xizl) as computed by compute_stress
   * @param stress_integrand_2 Stress integrand jacobianl * (sigma_xz * xixl +
   * sigma_zz * xizl) as computed by compute_stress
   * @param stress_integrand_3 Stress integrand jacobianl * (sigma_xx * gammaxl
   * + sigma_xz * gammazl) as computed by compute_stress
   * @param stress_integrand_4 Stress integrand jacobianl * (sigma_xz * gammaxl
   * + sigma_zz * gammazl) as computed by compute_stress
   * @param s_hprimewgll_xx Scratch view hprime_xx * wxgll
   * @param s_hprimewgll_zz Scratch view hprime_zz * wzgll
   * @param field_dot_dot Acceleration of the field subviewed at global index xz
   */
  KOKKOS_INLINE_FUNCTION void
  update_acceleration(const int &xz, const type_real &wxglll,
                      const type_real &wzglll,
                      const ScratchViewType<type_real> stress_integrand_1,
                      const ScratchViewType<type_real> stress_integrand_2,
                      const ScratchViewType<type_real> stress_integrand_3,
                      const ScratchViewType<type_real> stress_integrand_4,
                      const ScratchViewType<type_real> s_hprimewgll_xx,
                      const ScratchViewType<type_real> s_hprimewgll_zz,
                      field_type field_dot_dot) const override;

  /**
   * @brief Get the index of the element
   *
   * @return int Index of the element
   */
  KOKKOS_INLINE_FUNCTION int get_ispec() const override { return this->ispec; }

private:
  int ispec;                                         ///< Index of the element
  specfem::kokkos::DeviceView2d<type_real> xix;      ///< xix
  specfem::kokkos::DeviceView2d<type_real> xiz;      ///< xiz
  specfem::kokkos::DeviceView2d<type_real> gammax;   ///< gammax
  specfem::kokkos::DeviceView2d<type_real> gammaz;   ///< gammaz
  specfem::kokkos::DeviceView2d<type_real> jacobian; ///< jacobian
  specfem::kokkos::DeviceView2d<type_real> lambdaplus2mu; ///< lambda +
                                                          ///< 2 * mu
  specfem::kokkos::StaticDeviceView2d<type_real, N> mu;   ///< mu
};
} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
