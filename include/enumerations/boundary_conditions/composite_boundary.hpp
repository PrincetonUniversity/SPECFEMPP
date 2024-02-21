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
namespace enums {
namespace boundary_conditions {
/**
 * @brief Composite boundary conditions
 *
 * @tparam BCs Boundary conditions to be combined in the composite boundary
 */
template <typename... BCs> class composite_boundary {};

/**
 * @brief Composite boundary conditions for Stacey BC and Dirichlet BC
 *
 * @tparam properties Properties of the Boundary conditions
 */
template <typename... properties>
class composite_boundary<
    specfem::enums::boundary_conditions::stacey<properties...>,
    specfem::enums::boundary_conditions::dirichlet<properties...> > {
private:
  using stacey_type = typename specfem::enums::boundary_conditions::stacey<
      properties...>; ///< Stacey boundary conditions
  using dirichlet_type =
      typename specfem::enums::boundary_conditions::dirichlet<
          properties...>; ///< Dirichlet boundary conditions

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using medium_type =
      typename stacey_type::medium_type; ///< Medium type of the boundary.
  using dimension =
      typename stacey_type::dimension; ///< Dimension of the boundary.
  using quadrature_points_type =
      typename stacey_type::quadrature_points_type; ///< Quadrature points
                                                    ///< object to define the
                                                    ///< quadrature points
                                                    ///< either at compile time
                                                    ///< or run time.
  using property_type =
      typename stacey_type::property_type; ///< Property type of the boundary.
  ///@}

  constexpr static specfem::enums::element::boundary_tag value =
      specfem::enums::element::boundary_tag::
          composite_stacey_dirichlet; ///< boundary tag

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

  /**
   * @brief Construct a new composite boundary object
   *
   */
  composite_boundary(){};

  /**
   * @brief Construct a new stacey object
   *
   * @param boundary_conditions boundary conditions object specifying the
   * boundary conditions
   * @param quadrature_points Quadrature points object
   */
  composite_boundary(const specfem::compute::boundaries &boundary_conditions,
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

} /* namespace boundary_conditions */
} /* namespace enums */
} /* namespace specfem */

#endif /* _ENUMS_BOUNDARY_CONDITIONS_HPP */
