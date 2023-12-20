#ifndef _ENUMS_BOUNDARY_CONDITIONS_HPP
#define _ENUMS_BOUNDARY_CONDITIONS_HPP

#include "compute/interface.hpp"
#include "enumerations/boundary_conditions/dirichlet.hpp"
#include "enumerations/boundary_conditions/stacey/interface.hpp"
#include "enumerations/quadrature.hpp"
#include "enumerations/specfem_enums.hpp"

namespace specfem {
namespace enums {
namespace boundary_conditions {

template <typename... BCs> class composite_boundary {};

template <typename... properties>
class composite_boundary<
    specfem::enums::boundary_conditions::stacey<properties...>,
    specfem::enums::boundary_conditions::dirichlet<properties...> > {
private:
  using stacey_type =
      typename specfem::enums::boundary_conditions::stacey<properties...>;
  using dirichlet_type =
      typename specfem::enums::boundary_conditions::dirichlet<properties...>;

public:
  using medium_type = typename stacey_type::medium_type;
  using dimension = typename stacey_type::dimension;
  using quadrature_points_type = typename stacey_type::quadrature_points_type;
  using property_type = typename stacey_type::property_type;

  constexpr static specfem::enums::element::boundary_tag value =
      specfem::enums::element::boundary_tag::composite_stacey_dirichlet;

  composite_boundary() = default;

  composite_boundary(const specfem::compute::boundaries &boundary_conditions,
                     const quadrature_points_type &quadrature_points);

  template <specfem::enums::time_scheme::type time_scheme, typename... Args>
  KOKKOS_INLINE_FUNCTION void mass_time_contribution(Args &&...args) const {
    stacey.template mass_time_contribution<time_scheme>(
        std::forward<Args>(args)...);
    dirichlet.template mass_time_contribution<time_scheme>(
        std::forward<Args>(args)...);
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void enforce_gradient(Args &&...args) const {
    stacey.enforce_gradient(std::forward<Args>(args)...);
    dirichlet.enforce_gradient(std::forward<Args>(args)...);
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void enforce_stress(Args &&...args) const {
    stacey.enforce_stress(std::forward<Args>(args)...);
    dirichlet.enforce_stress(std::forward<Args>(args)...);
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void enforce_traction(Args &&...args) const {
    // The order of operations is important here. The Dirichlet boundary
    // conditions must be applied last.
    stacey.enforce_traction(std::forward<Args>(args)...);
    dirichlet.enforce_traction(std::forward<Args>(args)...);
  }

  inline static std::string to_string() {
    return "Composite: Stacey, Dirichlet";
  }

private:
  stacey_type stacey;
  dirichlet_type dirichlet;
};

} /* namespace boundary_conditions */
} /* namespace enums */
} /* namespace specfem */

#endif /* _ENUMS_BOUNDARY_CONDITIONS_HPP */
