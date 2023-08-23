#ifndef _DOMAIN_ELASTIC_ELEMENTS2D_HPP
#define _DOMAIN_ELASTIC_ELEMENTS2D_HPP

#include "domain/impl/elements/element.hpp"
#include "kokkos_abstractions.h"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"

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
 * @brief Elastic element in 2D
 *
 * Base class to define specialized elastic elements in 2D. This class defines
 * pure virtual methods to compute the gradient and stress at the quadrature
 * points of the element. The specific implementaions are left to the derived
 * classes.
 *
 * @tparam quadrature_points Type for number of quadrature points defined either
 * at compile time or run time
 */
template <typename qp_type>
class element<specfem::enums::element::dimension::dim2,
              specfem::enums::element::medium::elastic, qp_type> {

public:
  using dimension = specfem::enums::element::dimension::dim2;
  using medium = specfem::enums::element::medium::elastic;
  using quadrature_points = qp_type;

  /**
   * @brief Scratch view type as defined by the quadrature points (either at
   * compile time or run time)
   *
   * @tparam T Type of the scratch view
   */
  template <typename T>
  using ScratchViewType =
      typename quadrature_points::template ScratchViewType<T>;

  /**
   * @brief Compute the mass matrix component ($ m_{\alpha, \beta} $) for a
   * given quadrature point
   *
   * Mass matrix is given by \\f$ M =  \sum_{\Omega_e} \sum_{\alpha, \beta}
   * \omega_{\alpha} \omega_{\beta}  m_{\alpha, \beta} \\f$
   *
   * @param xz index of the quadrature point
   * @return type_real mass matrix component
   */
  KOKKOS_INLINE_FUNCTION virtual type_real
      [medium::components] compute_mass_matrix_component(
          const int &xz) const = 0;

  /**
   * @brief Compute the gradient of the field at the quadrature point xz
   *
   * @param xz Index of the quadrature point
   * @param s_hprime_xx lagrange polynomial derivative in x direction
   * @param s_hprime_zz lagraange polynomial derivative in z direction
   * @param field_x field in x direction
   * @param field_z field in z direction
   * @param duxdxl \f$ \frac{\partial u_x}{\partial x} \f$
   * @param duxdzl \f$ \frac{\partial u_x}{\partial z} \f$
   * @param duzdxl \f$ \frac{\partial u_z}{\partial x} \f$
   * @param duzdzl \f$ \frac{\partial u_z}{\partial z} \f$
   */
  KOKKOS_INLINE_FUNCTION virtual void
  compute_gradient(const int &xz, const ScratchViewType<type_real> s_hprime_xx,
                   const ScratchViewType<type_real> s_hprime_zz,
                   const ScratchViewType<type_real> field_x,
                   const ScratchViewType<type_real> field_z, type_real *duxdxl,
                   type_real *duxdzl, type_real *duzdxl,
                   type_real *duzdzl) const = 0;

  /**
   * @brief Compute the stress at the quadrature point xz
   *
   * @param xz Index of the quadrature point
   * @param duxdxl \f$ \frac{\partial u_x}{\partial x} \f$ as computed by
   * compute_gradient
   * @param duxdzl \f$ \frac{\partial u_x}{\partial z} \f$ as computed by
   * compute_gradient
   * @param duzdxl \f$ \frac{\partial u_z}{\partial x} \f$ as computed by
   * compute_gradient
   * @param duzdzl \f$ \frac{\partial u_z}{\partial z} \f$ as computed by
   * compute_gradient
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
   */
  KOKKOS_INLINE_FUNCTION virtual void
  compute_stress(const int &xz, const type_real &duxdxl,
                 const type_real &duxdzl, const type_real &duzdxl,
                 const type_real &duzdzl, type_real *stress_integrand_1l,
                 type_real *stress_integrand_2l, type_real *stress_integrand_3l,
                 type_real *stress_integrand_4l) const = 0;

  /**
   * @brief Update the acceleration at the quadrature point xz
   *
   * @param xz Index of the quadrature point
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
  KOKKOS_INLINE_FUNCTION virtual void
  update_acceleration(const int &xz, const type_real &wxglll,
                      const type_real &wzglll,
                      const ScratchViewType<type_real> stress_integrand_1,
                      const ScratchViewType<type_real> stress_integrand_2,
                      const ScratchViewType<type_real> stress_integrand_3,
                      const ScratchViewType<type_real> stress_integrand_4,
                      const ScratchViewType<type_real> s_hprimewgll_xx,
                      const ScratchViewType<type_real> s_hprimewgll_zz,
                      field_type field_dot_dot) const = 0;

  /**
   * @brief Get the global index of the element
   *
   */
  KOKKOS_INLINE_FUNCTION virtual int get_ispec() const = 0;
}; // namespace element

} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
