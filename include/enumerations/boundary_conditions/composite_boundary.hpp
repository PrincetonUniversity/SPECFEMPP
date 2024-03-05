#ifndef _ENUMS_BOUNDARY_CONDITIONS_HPP
#define _ENUMS_BOUNDARY_CONDITIONS_HPP

#include "compute/interface.hpp"
#include "enumerations/boundary_conditions/dirichlet.hpp"
#include "enumerations/boundary_conditions/dirichlet.tpp"
#include "enumerations/boundary_conditions/stacey/interface.hpp"
#include "enumerations/quadrature.hpp"
#include "enumerations/specfem_enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace boundary {

template <specfem::element::medium_tag medium,
          specfem::element::property_tag property, typename qp_type>
class boundary<specfem::dimension::type::dim2, medium, property,
               specfem::element::boundary_tag::composite_stacey_dirichlet,
               qp_type> {
private:
  using stacey_type = typename specfem::boundary::boundary<
      specfem::dimension::type::dim2, medium, property,
      specfem::element::boundary_tag::stacey, qp_type>;
  using dirichlet_type = typename specfem::boundary::boundary<
      specfem::dimension::type::dim2, medium, property,
      specfem::element::boundary_tag::acoustic_free_surface, qp_type>;

public:
  using quadrature_points_type = qp_type; ///< Quadrature points type
  using medium_type =
      specfem::medium::medium<specfem::dimension::type::dim2, medium,
                              property>; ///< Medium type

  constexpr static specfem::enums::element::boundary_tag value =
      specfem::enums::element::boundary_tag::
          composite_stacey_dirichlet; ///< boundary tag
  /**
   * @brief Construct a new composite boundary object
   *
   */
  boundary(){};

  /**
   * @brief Construct a new stacey object
   *
   * @param boundary_conditions boundary conditions object specifying the
   * boundary conditions
   * @param quadrature_points Quadrature points object
   */
  boundary(const specfem::compute::boundaries &boundary_conditions,
           const quadrature_points_type &quadrature_points){};

  /**
   * @brief Compute the contribution of composite boundaries to the mass term
   *
   * @tparam time_scheme Time scheme to be used
   * @param Args Arguments to be passed to the function. The underlying BCs are
   * called with the same arguments
   */
  template <specfem::enums::time_scheme::type time_scheme, typename... Args>
  KOKKOS_INLINE_FUNCTION void mass_time_contribution(Args &&...args) const {
    stacey.template mass_time_contribution<time_scheme>(
        std::forward<Args>(args)...);
    dirichlet.template mass_time_contribution<time_scheme>(
        std::forward<Args>(args)...);
  }

  /**
   * @brief Compute the contribution of composite boundaries to the gradient
   * term
   *
   * @param Args Arguments to be passed to the function. The underlying BCs are
   * called with the same arguments
   */
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void enforce_gradient(Args &&...args) const {
    stacey.enforce_gradient(std::forward<Args>(args)...);
    dirichlet.enforce_gradient(std::forward<Args>(args)...);
  }

  /**
   * @brief Compute the contribution of composite boundaries to the stress term
   *
   * @param Args Arguments to be passed to the function. The underlying BCs are
   * called with the same arguments
   */
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void enforce_stress(Args &&...args) const {
    stacey.enforce_stress(std::forward<Args>(args)...);
    dirichlet.enforce_stress(std::forward<Args>(args)...);
  }

  /**
   * @brief Compute the contribution of composite boundaries to the traction
   * term
   *
   * @param Args Arguments to be passed to the function. The underlying BCs are
   * called with the same arguments
   */
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void enforce_traction(Args &&...args) const {
    // The order of operations is important here. The Dirichlet boundary
    // conditions must be applied last.
    stacey.enforce_traction(std::forward<Args>(args)...);
    dirichlet.enforce_traction(std::forward<Args>(args)...);
  }

  /**
   * @brief Convert Stacey BC to string
   *
   */
  inline static std::string to_string() {
    return "Composite: Stacey, Dirichlet";
  }

private:
  stacey_type stacey;       ///< Stacey boundary conditions
  dirichlet_type dirichlet; ///< Dirichlet boundary conditions
};

} // namespace boundary
} // namespace specfem

#endif /* _ENUMS_BOUNDARY_CONDITIONS_HPP */
