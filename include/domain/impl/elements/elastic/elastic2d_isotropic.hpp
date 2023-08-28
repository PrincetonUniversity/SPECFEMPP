#ifndef _DOMAIN_ELASTIC_ELEMENTS2D_ISOTROPIC_HPP
#define _DOMAIN_ELASTIC_ELEMENTS2D_ISOTROPIC_HPP

#include "compute/interface.hpp"
#include "domain/impl/elements/elastic/elastic2d.hpp"
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
template <int NGLL>
class element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>
    : public element<specfem::enums::element::dimension::dim2,
                     specfem::enums::element::medium::elastic,
                     specfem::enums::element::quadrature::
                         static_quadrature_points<NGLL> > {
public:
  /**
   * @brief Dimension of the element
   *
   */
  using dimension = specfem::enums::element::dimension::dim2;
  /**
   * @brief Medium of the element
   *
   */
  using medium_type = specfem::enums::element::medium::elastic;
  /**
   * @brief Number of Gauss-Lobatto-Legendre quadrature points
   */
  using quadrature_points_type =
      specfem::enums::element::quadrature::static_quadrature_points<NGLL>;
  /**
   * @brief Use the scratch view type from the quadrature points
   *
   * @tparam T Type of the scratch view
   */
  template <typename T, int N>
  using ScratchViewType =
      typename quadrature_points_type::template ScratchViewType<T, N>;

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
   * @brief Compute the mass matrix component ($ m_{\alpha, \beta} $) for a
   * given quadrature point
   *
   * Mass matrix is given by \\f$ M =  \sum_{\Omega_e} \sum_{\alpha, \beta}
   * \omega_{\alpha} \omega_{\beta}  m_{\alpha, \beta} \\f$
   *
   * @param xz index of the quadrature point
   * @param mass_matrix mass matrix component
   */
  KOKKOS_INLINE_FUNCTION
  void compute_mass_matrix_component(const int &xz,
                                     type_real *mass_matrix) const override;

  /**
   * @brief Compute the gradient of the field at a particular
   * Gauss-Lobatto-Legendre quadrature point
   *
   * @param xz Index of Gauss-Lobatto-Legendre quadrature point
   * @param s_hprime_xx Scratch view of derivative of Lagrange polynomial in x
   * direction
   * @param s_hprime_zz Scratch view of derivative of Lagrange polynomial in z
   * direction
   * @param u Scrath view of field. For elastic domains the field has 2
   * components
   * @param dudxl Computed partial derivative of field \f$ \frac{\partial
   * \tilde{u}}{\partial x} \f$
   * @param dudzl Computed partial derivative of field \f$ \frac{\partial
   * \tilde{u}}{\partial z} \f$
   */
  KOKKOS_INLINE_FUNCTION void
  compute_gradient(const int &xz,
                   const ScratchViewType<type_real, 1> s_hprime_xx,
                   const ScratchViewType<type_real, 1> s_hprime_zz,
                   const ScratchViewType<type_real, medium_type::components> u,
                   type_real *dudxl, type_real *dudzl) const override;

  /**
   * @brief Compute the stress integrand at a particular Gauss-Lobatto-Legendre
   * quadrature point.
   *
   * @param xz Index of Gauss-Lobatto-Legendre quadrature point
   * @param dudxl Partial derivative of field \f$ \frac{\partial
   * \tilde{u}}{\partial x} \f$
   * @param dudzl Partial derivative of field \f$ \frac{\partial
   * \tilde{u}}{\partial z} \f$
   * @param stress_integrand_xi Stress integrand  wrt. \f$ \xi \f$ \f$
   * J^{\alpha, \gamma} \partial_x u \partial_x \xi + \partial_z u * \partial_z
   * \xi \f$
   * @param stress_integrand_gamma Stress integrand  wrt. \f$\gamma\f$ \f$
   * J^{\alpha, \gamma} \partial_x u \partial_x \gamma + \partial_z u *
   * \partial_z \gamma \f$
   */
  KOKKOS_INLINE_FUNCTION void
  compute_stress(const int &xz, const type_real *dudxl, const type_real *dudzl,
                 type_real *stress_integrand_xi,
                 type_real *stress_integrand_gamma) const override;

  /**
   * @brief Update the acceleration at the quadrature point xz
   *
   * @param xz Index of the quadrature point
   * @param wxglll Weight of the Gauss-Lobatto-Legendre quadrature point in x
   * direction
   * @param wzglll Weight of the Gauss-Lobatto-Legendre quadrature point in z
   * direction
   * @param stress_integrand_xi Stress integrand wrt. \f$ \xi \f$ \f$ J^{\alpha,
   * \gamma} \partial_x u \partial_x \xi + \partial_z u * \partial_z \xi \f$ as
   * computed by compute_stress
   * @param stress_integrand_gamma Stress integrand wrt. \f$\gamma\f$ \f$
   * J^{\alpha, \gamma} \partial_x u \partial_x \gamma + \partial_z u *
   * \partial_z \gamma \f$ as computed by compute_stress
   * @param s_hprimewgll_xx Scratch view hprime_xx * wxgll
   * @param s_hprimewgll_zz Scratch view hprime_zz * wzgll
   * @param field_dot_dot Acceleration of the field subviewed at global index xz
   */
  KOKKOS_INLINE_FUNCTION void
  update_acceleration(const int &xz, const type_real &wxglll,
                      const type_real &wzglll,
                      const ScratchViewType<type_real, medium_type::components>
                          stress_integrand_xi,
                      const ScratchViewType<type_real, medium_type::components>
                          stress_integrand_gamma,
                      const ScratchViewType<type_real, 1> s_hprimewgll_xx,
                      const ScratchViewType<type_real, 1> s_hprimewgll_zz,
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
  specfem::kokkos::DeviceView2d<type_real> mu;            ///< mu
  specfem::kokkos::DeviceView2d<type_real> rho;           ///< rho
};
} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
