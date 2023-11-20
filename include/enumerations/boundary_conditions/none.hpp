#ifndef _ENUMS_BOUNDARY_CONDITIONS_NONE_HPP_
#define _ENUMS_BOUNDARY_CONDITIONS_NONE_HPP_

#include "compute/interface.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/quadrature.hpp"
#include "enumerations/specfem_enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace enums {
namespace boundary_conditions {

template <typename dim, typename medium, typename property, typename qp_type>
class none {

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
   * @brief Property type of the boundary.
   *
   */
  using property_type = property;
  ///@}

  constexpr static specfem::enums::element::boundary_tag value =
      specfem::enums::element::boundary_tag::none; ///< boundary tag

  none(){};

  none(const specfem::compute::boundaries &boundary_conditions,
       const quadrature_points_type &quadrature_points){};

  template <specfem::enums::time_scheme::type time_scheme>
  KOKKOS_INLINE_FUNCTION void mass_time_contribution(
      const int &ielement, const int &xz, const type_real &dt,
      const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
      const specfem::compute::element_partial_derivatives &partial_derivatives,
      const specfem::compute::element_properties<
          medium_type::value, property_type::value> &properties,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &rmass_inverse) const {};

  KOKKOS_INLINE_FUNCTION void enforce_gradient(
      const int &ielement, const int &xz,
      const specfem::compute::element_partial_derivatives &partial_derivatives,
      specfem::kokkos::array_type<type_real, medium_type::components> &df_dx,
      specfem::kokkos::array_type<type_real, medium_type::components> &df_dz)
      const {};

  KOKKOS_INLINE_FUNCTION void enforce_stress(
      const int &ielement, const int &xz,
      const specfem::compute::element_partial_derivatives &partial_derivatives,
      const specfem::compute::element_properties<
          medium_type::value, property_type::value> &properties,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &stress_integrand_xi,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &stress_integrand_xgamma) const {};

  KOKKOS_INLINE_FUNCTION void enforce_traction(
      const int &ielement, const int &xz,
      const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
      const specfem::compute::element_partial_derivatives &partial_derivatives,
      const specfem::compute::element_properties<
          medium_type::value, property_type::value> &properties,
      const specfem::kokkos::array_type<type_real, medium_type::components>
          &field_dot,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &field_dot_dot) const {};

  __inline__ static std::string to_string() { return ""; }
};

} // namespace boundary_conditions
} // namespace enums
} // namespace specfem

#endif /* _ENUMS_BOUNDARY_CONDITIONS_NONE_HPP_ */
