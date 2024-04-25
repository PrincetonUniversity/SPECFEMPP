#ifndef _DOMAIN_ACOUSTIC_ELEMENTS2D_IMPLEMENTATION_HPP
#define _DOMAIN_ACOUSTIC_ELEMENTS2D_IMPLEMENTATION_HPP

#include "compute/interface.hpp"
// #include "domain/impl/elements/acoustic/acoustic2d.hpp"
#include "domain/impl/elements/element.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace elements {
/**
 * @brief Element specialization for 2D elastic isotropic spectral elements with
 * static quadrature points
 *
 * @tparam NGLL Number of Gauss-Lobatto-Legendre quadrature points defined at
 * compile time
 *
 * @tparam BC Boundary conditions if void then neumann boundary conditions are
 * assumed
 */
template <int NGLL, specfem::element::boundary_tag BC>
class element<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, BC,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL> > {
private:
  constexpr static auto DimensionType = specfem::dimension::type::dim2;
  constexpr static auto MediumTag = specfem::element::medium_tag::acoustic;
  using ElementQuadratureViewType = typename specfem::element::quadrature<
      NGLL, DimensionType, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, true>::ViewType;
  using ElementFieldViewType =
      typename specfem::element::field<NGLL, DimensionType, MediumTag,
                                       specfem::kokkos::DevScratchSpace,
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>,
                                       true, true, true, false>::ViewType;

public:
  using dimension =
      specfem::dimension::dimension<specfem::dimension::type::dim2>;
  using quadrature_points_type =
      specfem::enums::element::quadrature::static_quadrature_points<NGLL>;
  using medium_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::acoustic,
                              specfem::element::property_tag::isotropic>;
  using boundary_conditions_type = specfem::boundary::boundary<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic, BC, quadrature_points_type>;

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
  element(const specfem::compute::assembly &assembly,
          const quadrature_points_type &quadrature_points)
      : boundary_conditions(assembly.boundaries, quadrature_points){};

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
  void compute_mass_matrix_component(
      const specfem::point::properties<medium_type::medium_tag,
                                       medium_type::property_tag> &properties,
      const specfem::point::partial_derivatives2<true> &partial_derivatives,
      specfem::kokkos::array_type<type_real, 1> &mass_matrix) const;

  template <specfem::enums::time_scheme::type time_scheme>
  KOKKOS_INLINE_FUNCTION void mass_time_contribution(
      const int &xz, const type_real &dt,
      const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
      const specfem::point::partial_derivatives2<true> &partial_derivatives,
      const specfem::point::properties<medium_type::medium_tag,
                                       medium_type::property_tag> &properties,
      const specfem::point::boundary &boundary_type,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &rmass_inverse) const;

  /**
   * @brief Compute the gradient of the field at the quadrature point xz
   * (Komatisch & Tromp, 2002-I., eq. 22(theory, OC),44,45)
   *
   * @param xz Index of the quadrature point
   * @param s_hprime_xx lagrange polynomial derivative in x direction
   * @param s_hprime_zz lagraange polynomial derivative in z direction
   * @param field_chi potential field
   * @param dchidxl Computed partial derivative of field \f$ \frac{\partial
   * \chi}{\partial x} \f$
   * @param dchidzl Computed partial derivative of field \f$ \frac{\partial
   * \chi}{\partial z} \f$
   */
  template <bool with_jacobian>
  KOKKOS_INLINE_FUNCTION void compute_gradient(
      const int xz, const ElementQuadratureViewType s_hprime,
      const ElementFieldViewType field_chi,
      const specfem::point::partial_derivatives2<with_jacobian>
          &partial_derivatives,
      const specfem::point::boundary &boundary_type,
      specfem::kokkos::array_type<type_real, medium_type::components> &dudxl,
      specfem::kokkos::array_type<type_real, medium_type::components> &dudzl)
      const;

  /**
   * @brief Compute the stress integrand at a particular
   Gauss-Lobatto-Legendre
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
  KOKKOS_INLINE_FUNCTION void compute_stress(
      const int xz,
      const specfem::kokkos::array_type<type_real, medium_type::components>
          &dudxl,
      const specfem::kokkos::array_type<type_real, medium_type::components>
          &dudzl,
      const specfem::point::partial_derivatives2<true> &partial_derivatives,
      const specfem::point::properties<medium_type::medium_tag,
                                       medium_type::property_tag> &properties,
      const specfem::point::boundary &boundary_type,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &stress_integrand_xi,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &stress_integrand_gamma) const;

  /**
   * @brief Update the acceleration at a particular Gauss-Lobatto-Legendre
   * quadrature point
   *
   * @param xz Index of Gauss-Lobatto-Legendre quadrature point
   * @param wxglll Weight of the Gauss-Lobatto-Legendre quadrature point in
   x
   * direction
   * @param wzglll Weight of the Gauss-Lobatto-Legendre quadrature point in
   z
   * direction
   * @param stress_integrand_xi Stress integrand  wrt. \f$ \xi \f$
   * \f$ J^{\alpha\gamma} * {\rho^{\alpha\gamma}}^{-1}
   * \partial_x \chi \partial_x \xi + \partial_z \chi * \partial_z \xi \f$
   as
   * computed by compute_stress
   * @param stress_integrand_gamma Stress integrand  wrt. \f$\gamma\f$
   * \f$ J^{\alpha\gamma} * {\rho^{\alpha\gamma}}^{-1}
   * \partial_x \chi \partial_x \gamma + \partial_z \chi * \partial_z
   \gamma \f$
   * as computed by compute_stress
   * @param s_hprimewgll_xx Scratch view hprime_xx * wxgll
   * @param s_hprimewgll_zz Scratch view hprime_zz * wzgll
   * @param field_dot_dot Acceleration of the field subviewed at global
   index xz
   */
  KOKKOS_INLINE_FUNCTION void compute_acceleration(
      const int &xz,
      const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
      const ElementFieldViewType stress_integrand_xi,
      const ElementFieldViewType stress_integrand_gamma,
      const ElementQuadratureViewType s_hprimewgll,
      const specfem::point::partial_derivatives2<true> &partial_derivatives,
      const specfem::point::properties<medium_type::medium_tag,
                                       medium_type::property_tag> &properties,
      const specfem::point::boundary &boundary_type,
      const specfem::kokkos::array_type<type_real, medium_type::components>
          &velocity,
      specfem::kokkos::array_type<type_real, 1> &acceleration) const;

private:
  boundary_conditions_type boundary_conditions; ///< boundary conditions
};
} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
