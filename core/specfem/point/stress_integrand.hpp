#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem/datatype.hpp"
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
 * @tparam DimensionTag The dimension of the element where the quadrature point
 * is located
 * @tparam MediumTag The medium of the element where the quadrature point is
 * located
 * @tparam UseSIMD Use SIMD instructions
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct stress_integrand
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::stress_integrand, DimensionTag,
          UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::stress_integrand, DimensionTag,
      UseSIMD>; ///< Base accessor
                ///< type
public:
  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int dimension =
      specfem::element::attributes<DimensionTag, MediumTag>::dimension;
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;
  ///@}

  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = typename base_type::template simd<type_real>; ///< SIMD type
  using value_type =
      typename base_type::template tensor_type<type_real, components,
                                               dimension>; ///< Underlying view
                                                           ///< type to store
                                                           ///< the stress
                                                           ///< integrand
  ///@}

  value_type F; ///< View to store the stress integrand

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
  KOKKOS_FUNCTION stress_integrand(const value_type &F) : F(F) {}
  ///@}
};

} // namespace point
} // namespace specfem
