#ifndef _ENUMS_BOUNDARY_CONDITIONS_DIRICHLET_HPP_
#define _ENUMS_BOUNDARY_CONDITIONS_DIRICHLET_HPP_

#include "compute/compute_boundaries.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/interface.hpp"
#include "enumerations/quadrature.hpp"
#include "enumerations/specfem_enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace enums {
namespace boundary_conditions {

/**
 * @brief Dirichlet boundary condition
 *
 * @tparam dim Dimension of the boundary.
 * @tparam medium Medium type for the element where the boundary is located.
 * @tparam property Property type for the element where the boundary is located.
 * @tparam qp_type Quadrature points object to define the quadrature points
 * either at compile time or run time.
 */
template <typename dim, typename medium, typename property, typename qp_type>
class dirichlet {
public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  /**
   * @brief Medium type of the boundary.
   *
   */
  using medium_type = medium;
  /**
   * @brief Dimension of the boundary.
   *
   */
  using dimension = dim;
  /**
   * @brief Quadrature points object to define the quadrature points either at
   * compile time or run time.
   *
   */
  using quadrature_points_type = qp_type;

  /**
   * @brief
   *
   */
  using property_type = property;
  ///@}

  constexpr static specfem::enums::element::boundary_tag value = specfem::
      enums::element::boundary_tag::acoustic_free_surface; ///< boundary
                                                           ///< tag

  /**
   * @brief Construct a new dirichlet object
   *
   */
  dirichlet(){};

  /**
   * @brief Construct a new dirichlet object
   *
   * @param quadrature_points Quadrature points object to define the quadrature
   * points either at compile time or run time.
   * @param type type of the edge on an element on the boundary.
   */
  dirichlet(const quadrature_points_type &quadrature_points,
            const specfem::kokkos::DeviceView1d<
                specfem::compute::access::boundary_types> &type)
      : quadrature_points(quadrature_points), type(type) {}

  /**
   * @brief Construct a new dirichlet object
   *
   * @param boundary_conditions boundary conditions object specifying the
   * boundary conditions
   * @param quadrature_points Quadrature points object to define the quadrature
   * points either at compile time or run time.
   */
  dirichlet(const specfem::compute::boundaries &boundary_conditions,
            const quadrature_points_type &quadrature_points);

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
      const int &ielement, const int &xz, const type_real &dt,
      const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
      const specfem::compute::element_partial_derivatives &partial_derivatives,
      const specfem::compute::element_properties<
          medium_type::value, property_type::value> &properties,
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
      const int &ielement, const int &xz,
      const specfem::compute::element_partial_derivatives &partial_derivatives,
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
      const int &ielement, const int &xz,
      const specfem::compute::element_partial_derivatives &partial_derivatives,
      const specfem::compute::element_properties<
          medium_type::value, property_type::value> &properties,
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
      const int &ielement, const int &xz,
      const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
      const specfem::compute::element_partial_derivatives &partial_derivatives,
      const specfem::compute::element_properties<
          medium_type::value, property_type::value> &properties,
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
  specfem::kokkos::DeviceView1d<specfem::compute::access::boundary_types>
      type; ///< type of the edge on an element on the boundary.
  quadrature_points_type quadrature_points; ///< Quadrature points object.
};

} // namespace boundary_conditions
} // namespace enums
} // namespace specfem

#endif /* _ENUMS_BOUNDARY_CONDITIONS_DIRICHLET_HPP_ */
