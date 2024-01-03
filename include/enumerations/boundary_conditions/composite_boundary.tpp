#ifndef _ENUMS_BOUNDARY_CONDITIONS_COMPOSITE_BOUNDARY_TPP_
#define _ENUMS_BOUNDARY_CONDITIONS_COMPOSITE_BOUNDARY_TPP_

#include "compute/interface.hpp"
#include "enumerations/boundary_conditions/composite_boundary.hpp"
#include "enumerations/boundary_conditions/dirichlet.hpp"
#include "enumerations/boundary_conditions/stacey/interface.hpp"
#include "enumerations/quadrature.hpp"
#include "enumerations/specfem_enums.hpp"

template <typename... properties>
specfem::enums::boundary_conditions::composite_boundary<
    specfem::enums::boundary_conditions::stacey<properties...>,
    specfem::enums::boundary_conditions::dirichlet<properties...> >::
    composite_boundary(const specfem::compute::boundaries &boundary_conditions,
                       const quadrature_points_type &quadrature_points)
    : stacey(quadrature_points,
             boundary_conditions.composite_stacey_dirichlet.type),
      dirichlet(quadrature_points,
                boundary_conditions.composite_stacey_dirichlet.type) {

  static_assert(
      std::is_same<typename stacey_type::medium_type,
                   typename dirichlet_type::medium_type>::value,
      "Medium types must be the same for composite boundary conditions.");
  static_assert(
      std::is_same<typename stacey_type::dimension,
                   typename dirichlet_type::dimension>::value,
      "Dimensions must be the same for composite boundary conditions.");
  static_assert(
      std::is_same<typename stacey_type::quadrature_points_type,
                   typename dirichlet_type::quadrature_points_type>::value,
      "Quadrature points must be the same for composite boundary conditions.");
  static_assert(
      std::is_same<typename stacey_type::property_type,
                   typename dirichlet_type::property_type>::value,
      "Property types must be the same for composite boundary conditions.");
}

#endif /* _ENUMS_BOUNDARY_CONDITIONS_COMPOSITE_BOUNDARY_TPP_ */
