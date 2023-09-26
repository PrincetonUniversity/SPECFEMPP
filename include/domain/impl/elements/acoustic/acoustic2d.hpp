#ifndef _DOMAIN_ACOUSTIC_ELEMENTS2D_HPP
#define _DOMAIN_ACOUSTIC_ELEMENTS2D_HPP

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
 * @brief Acoustic element in 2D
 *
 * Base class to define specialized acoustic elements in 2D. This class defines
 * pure virtual methods to compute the gradient and stress at the quadrature
 * points of the element. The specific implementaions are left to the derived
 * classes, e.g.
 *
 * @tparam quadrature_points Type for number of quadrature points defined either
 * at compile time or run time
 */
template <typename quadrature_points>
class element<specfem::enums::element::dimension::dim2,
              specfem::enums::element::medium::acoustic, quadrature_points> {

public:
  using dimension = specfem::enums::element::dimension::dim2;
  using medium = specfem::enums::element::medium::acoustic;
  /**
   * @brief Scratch view type as defined by the quadrature points (either at
   * compile time or run time)
   *
   * @tparam T Type of the scratch view
   */
  template <typename T, int N>
  using ScratchViewType =
      typename quadrature_points::template ScratchViewType<T, N>;

  /**
   * @brief Compute the mass matrix component ($ m_{\alpha, \beta} $) for a
   * given quadrature point
   *
   * Mass matrix is given by \\f$ M =  \sum_{\Omega_e} \sum_{\alpha, \beta}
   * \omega_{\alpha} \omega_{\beta}  m_{\alpha, \beta} \\f$
   *
   * @param xz index of the quadrature point
   * @param mass_matrix mass matrix component
   */
  KOKKOS_INLINE_FUNCTION virtual void
  compute_mass_matrix_component(const int &ispec, const int &xz,
                                type_real *mass_matrix) const = 0;

  /**
   * @brief Compute the gradient of the field at the quadrature point xz
   * (Komatisch & Tromp, 2002-I., eq. 22(theory, OC),44,45)
   *
   * @param xz Index of the quadrature point
   * @param s_hprime_xx lagrange polynomial derivative in x direction
   * @param s_hprime_zz lagraange polynomial derivative in z direction
   * @param field_chi potential field
   * @param dchidxl \f$ \frac{\partial \chi}{\partial x} \f$
   * @param dchidzl \f$ \frac{\partial \chi}{\partial z} \f$
   */
  KOKKOS_INLINE_FUNCTION virtual void compute_gradient(
      const int &ispec, const int &xz,
      const ScratchViewType<type_real, 1> s_hprime_xx,
      const ScratchViewType<type_real, 1> s_hprime_zz,
      const ScratchViewType<type_real, medium::components> field_chi,
      type_real *dchidxl, type_real *dchidzl) const = 0;

  /**
   * @brief Compute the stress integrand at a particular Gauss-Lobatto-Legendre
   * quadrature point. Note the collapse of jacobian elements into the
   * integrands for the gradient of \f$ w \f$
   * (Komatisch & Tromp, 2002-I., eq. 44,45)
   *
   * @param xz Index of Gauss-Lobatto-Legendre quadrature point
   * @param dchidxl Partial derivative of field \f$ \frac{\partial
   * \chi}{\partial x} \f$
   * @param dchipdzl Partial derivative of field \f$ \frac{\partial
   * \chi}{\partial z} \f$
   * @param stress_integrand_xi Stress integrand  wrt. \f$ \xi \f$
   * \f$ J^{\alpha\gamma} * {\rho^{\alpha\gamma}}^{-1}
   * \partial_x \chi \partial_x \xi
   * + \partial_z \chi * \partial_z \xi \f$
   * @param stress_integrand_gamma Stress integrand  wrt. \f$\gamma\f$
   * \f$ J^{\alpha\gamma} * {\rho^{\alpha\gamma}}^{-1}
   * \partial_x \chi \partial_x \gamma
   * + \partial_z \chi * \partial_z \gamma \f$
   * @return KOKKOS_FUNCTION
   */
  KOKKOS_INLINE_FUNCTION virtual void
  compute_stress(const int &ispec, const int &xz, const type_real *dchidxl,
                 const type_real *dchidzl, type_real *stress_integrand_xi,
                 type_real *stress_integrand_gamma) const = 0;

  /**
   * @brief Update the acceleration at the quadrature point xz
   *
   * @param xz Index of the quadrature point
   * @param wxglll Weight of the Gauss-Lobatto-Legendre quadrature point in x
   * direction
   * @param wzglll Weight of the Gauss-Lobatto-Legendre quadrature point in z
   * direction
   * @param stress_integrand_xi Stress integrand wrt. \f$ \xi \f$
   * J^{\alpha\gamma} * {\rho^{\alpha\gamma}}^{-1} \partial_x \chi \partial_x
   * \xi + \partial_z \chi * \partial_z \xi as computed by compute_stress
   * @param stress_integrand_gamma Stress integrand wrt. \f$\gamma\f$
   * \f$ J^{\alpha\gamma} * {\rho^{\alpha\gamma}}^{-1}
   * \partial_x \chi \partial_x \gamma + \partial_z \chi * \partial_z \gamma \f$
   * as computed by compute_stress
   * @param s_hprimewgll_xx Scratch view hprime_xx * wxgll
   * @param s_hprimewgll_zz Scratch view hprime_zz * wzgll
   * @param field_dot_dot Acceleration of the field subviewed at global index xz
   */
  KOKKOS_INLINE_FUNCTION virtual void update_acceleration(
      const int &xz, const type_real &wxglll, const type_real &wzglll,
      const ScratchViewType<type_real, medium::components> stress_integrand_xi,
      const ScratchViewType<type_real, medium::components>
          stress_integrand_gamma,
      const ScratchViewType<type_real, 1> s_hprimewgll_xx,
      const ScratchViewType<type_real, 1> s_hprimewgll_zz,
      type_real *acceleration) const = 0;

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
