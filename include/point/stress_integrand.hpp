#pragma once

#include "datatypes/point_view.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Store stress integrands for a quadrature point
 *
 * For elastic domains the stress integrand is given by:
 * \f$ F_{ik} = \sum_{j=1}^{n} T_{ij} \partial_j \xi_{k} \f$ where \f$ T \f$ is
 * the stress tensor. Equation (35) & (36) from Komatitsch and Tromp 2002 I. -
 * Validation
 *
 * For acoustic domains the stress integrand is given by:
 * \f$ F_{ik} = \rho^{-1} \partial_i \xi_{k} \partial_k \chi_{k} \f$. Equation
 * (44) & (45) from Komatitsch and Tromp 2002 I. - Validation
 *
 * @tparam DimensionType The dimension of the element where the quadrature point
 * is located
 * @tparam MediumTag The medium of the element where the quadrature point is
 * located
 * @tparam UseSIMD Use SIMD instructions
 */
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct stress_integrand {
  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int dimension =
      specfem::element::attributes<DimensionType, MediumTag>::dimension;
  constexpr static int components =
      specfem::element::attributes<DimensionType, MediumTag>::components;
  ///@}

  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type

  using ViewType =
      specfem::datatype::VectorPointViewType<type_real, dimension, components,
                                             UseSIMD>; ///< Underlying view type
                                                       ///< to store the stress
                                                       ///< integrand
  ///@}

  ViewType F; ///< View to store the stress integrand

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION stress_integrand() = default;

  /**
   * @brief Constructor
   *
   * @param F Stress integrands
   */
  KOKKOS_FUNCTION stress_integrand(const ViewType &F) : F(F) {}
  ///@}
};

} // namespace point
} // namespace specfem
