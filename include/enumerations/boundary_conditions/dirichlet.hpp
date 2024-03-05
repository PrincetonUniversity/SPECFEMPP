#ifndef _ENUMS_BOUNDARY_CONDITIONS_DIRICHLET_HPP_
#define _ENUMS_BOUNDARY_CONDITIONS_DIRICHLET_HPP_

// #include "compute/compute_boundaries.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/interface.hpp"
#include "enumerations/quadrature.hpp"
#include "enumerations/specfem_enums.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace boundary {

/**
 * @brief Dirichlet boundary condition
 *
 * @tparam dim Dimension of the boundary.
 * @tparam medium Medium type for the element where the boundary is located.
 * @tparam property Property type for the element where the boundary is located.
 * @tparam qp_type Quadrature points object to define the quadrature points
 * either at compile time or run time.
 */
template <specfem::element::medium_tag medium,
          specfem::element::property_tag property, typename qp_type>
class boundary<specfem::dimension::type::dim2, medium, property,
               specfem::element::boundary_tag::acoustic_free_surface, qp_type> {
public:
  using quadrature_points_type = qp_type; ///< Quadrature points type
  using medium_type =
      specfem::medium::medium<specfem::dimension::type::dim2, medium,
                              property>; ///< Medium type

  constexpr static specfem::enums::element::boundary_tag value = specfem::
      enums::element::boundary_tag::acoustic_free_surface; ///< boundary
                                                           ///< tag

  /**
   * @brief Construct a new dirichlet object
   *
   */
  boundary(){};

  /**
   * @brief Construct a new dirichlet object
   *
   * @param quadrature_points Quadrature points object to define the quadrature
   * points either at compile time or run time.
   * @param type type of the edge on an element on the boundary.
   */
  boundary(const quadrature_points_type &quadrature_points,
           const specfem::kokkos::DeviceView1d<specfem::point::boundary> &type)
      : quadrature_points(quadrature_points), type(type) {}

  /**
   * @brief Construct a new dirichlet object
   *
   * @param boundary_conditions boundary conditions object specifying the
   * boundary conditions
   * @param quadrature_points Quadrature points object to define the quadrature
   * points either at compile time or run time.
   */
  boundary(const specfem::compute::boundaries &boundary_conditions,
           const quadrature_points_type &quadrature_points){};

  /**
   * @brief Compute the mass time contribution for the boundary condition
   *
   * @tparam time_scheme Time scheme to use when computing the mass time
   * contribution
   * @param ielement index of the element
   * @param xz index of the quadrature point
   * @param dt time step
   * @param weight weights(x,z) for the quadrature point
   * @param partial_derivatives partial derivatives of the shape functions
   * @param properties properties of the element at the quadrature point
   * @param mass_matrix mass matrix to update
   */
  template <specfem::enums::time_scheme::type time_scheme>
  KOKKOS_INLINE_FUNCTION void mass_time_contribution(
      const int &xz, const type_real &dt,
      const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
      const specfem::point::partial_derivatives2 &partial_derivatives,
      const specfem::point::properties<medium_type::value,
                                       medium_type::property_value> &properties,
      const specfem::point::boundary &boundary_type,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &rmass_inverse) const {};

  /**
   * @brief Compute the contribuition of BC to the gradient term
   *
   * @param ielement index of the element
   * @param xz index of the quadrature point
   * @param partial_derivatives spacial derivatives at the quadrature point
   * @param df_dx Gradient of field in x-direction to update
   * @param df_dz Gradient of field in z-direction to update
   */
  KOKKOS_INLINE_FUNCTION void enforce_gradient(
      const int &xz,
      const specfem::point::partial_derivatives2 &partial_derivatives,
      const specfem::point::boundary &boundary_type,
      specfem::kokkos::array_type<type_real, medium_type::components> &df_dx,
      specfem::kokkos::array_type<type_real, medium_type::components> &df_dz)
      const {};

  /**
   * @brief Compute the contribution of BC to the stress term
   *
   * @param ielement index of the element
   * @param xz index of the quadrature point
   * @param partial_derivatives spacial derivatives at the quadrature point
   * @param properties properties of the element at the quadrature point
   * @param stress_integrand_xi /f$ \sigma_{\xi} /f$ to update
   * @param stress_integrand_xgamma /f$ \sigma_{\gamma} /f$ to update
   * @return KOKKOS_INLINE_FUNCTION
   */
  KOKKOS_INLINE_FUNCTION void enforce_stress(
      const int &xz,
      const specfem::point::partial_derivatives2 &partial_derivatives,
      const specfem::point::properties<medium_type::value,
                                       medium_type::property_value> &properties,
      const specfem::point::boundary &boundary_type,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &stress_integrand_xi,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &stress_integrand_xgamma) const {};

  /**
   * @brief Compute the contribution of BC to the traction term
   *
   * @param ielement index of the element
   * @param xz index of the quadrature point
   * @param weight weights(x,z) for the quadrature point
   * @param partial_derivatives partial derivatives of the shape functions
   * @param properties properties of the element at the quadrature point
   * @param velocity first derivative of the field computed from previous time
   * step
   * @param accelation second derivative of the field to update
   * @return KOKKOS_INLINE_FUNCTION
   */
  KOKKOS_FUNCTION void enforce_traction(
      const int &xz,
      const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
      const specfem::point::partial_derivatives2 &partial_derivatives,
      const specfem::point::properties<medium_type::value,
                                       medium_type::property_value> &properties,
      const specfem::point::boundary &boundary_type,
      const specfem::kokkos::array_type<type_real, medium_type::components>
          &field_dot,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &field_dot_dot) const;

  /**
   * @brief Convert the boundary to a string
   *
   */
  inline static std::string to_string() { return "Dirichlet"; }

private:
  specfem::kokkos::DeviceView1d<specfem::point::boundary>
      type; ///< type of the edge on an element on the boundary.
  quadrature_points_type quadrature_points; ///< Quadrature points object.
};

} // namespace boundary
} // namespace specfem

#endif /* _ENUMS_BOUNDARY_CONDITIONS_DIRICHLET_HPP_ */
