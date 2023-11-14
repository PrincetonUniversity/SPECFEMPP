#ifndef _ENUMS_BOUNDARY_CONDITIONS_STACEY_2D_HPP_
#define _ENUMS_BOUNDARY_CONDITIONS_STACEY_2D_HPP_

#include "compute/interface.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/quadrature.hpp"
#include "enumerations/specfem_enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace enums {
namespace boundary_conditions {
template <typename dim, typename medium, typename qp_type> class stacey {

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
  ///@}

  constexpr static specfem::enums::element::boundary_tag value =
      specfem::enums::element::boundary_tag::stacey; ///< boundary tag

  stacey(){};

  /**
   * @brief Convert Stacey BC to string
   *
   */
  __inline__ static std::string to_string() { return "Stacey"; }
};
} // namespace boundary_conditions
} // namespace enums
} // namespace specfem

#endif /* end of include guard: _ENUMS_BOUNDARY_CONDITIONS_STACEY_2D_HPP_ */
