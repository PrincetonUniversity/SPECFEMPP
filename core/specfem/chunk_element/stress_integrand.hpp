#pragma once

#include "datatypes/chunk_element_view.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace chunk_element {

/**
 * @brief Struct to hold the stress integrand at every quadrature point within a
 * chunk of elements.
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
 * @tparam NumberElements Number of elements in the chunk.
 * @tparam NGLL Number of Gauss-Lobatto-Legendre points.
 * @tparam DimensionTag Dimension type for elements within the chunk.
 * @tparam MediumTag Medium tag for elements within the chunk.
 * @tparam MemorySpace Memory space for data storage.
 * @tparam MemoryTraits Memory traits for data storage.
 * @tparam UseSIMD Flag to indicate if SIMD should be used.
 */
template <int NumberElements, int NGLL, specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, typename MemorySpace,
          typename MemoryTraits, bool UseSIMD>
struct stress_integrand {

public:
  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static int num_elements =
      NumberElements; ///< Number of elements in the chunk.
  constexpr static auto dimension =
      DimensionTag; ///< Dimension type for elements.
  ///@}

private:
  constexpr static int num_dimensions =
      specfem::element::attributes<DimensionTag,
                                   MediumTag>::dimension; ///< Number of
  ///< dimensions.
  constexpr static int components =
      specfem::element::attributes<DimensionTag,
                                   MediumTag>::components; ///< Number of
                                                           ///< components.

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type.

  using ViewType = specfem::datatype::TensorChunkViewType<
      type_real, DimensionTag, NumberElements, NGLL, components, num_dimensions,
      UseSIMD, MemorySpace,
      MemoryTraits>; ///< Underlying view used to store data.
  ///@}

  ViewType F; ///< Stress integrand

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor.
   *
   */
  KOKKOS_FUNCTION stress_integrand() = default;

  /**
   * @brief Constructor that initializes the stress integrand with a given view.
   *
   * @param F Stress integrand view.
   */
  KOKKOS_FUNCTION stress_integrand(const ViewType &F) : F(F) {}

  /**
   * @brief Constructor that initializes the stress integrand within Scratch
   * Memory.
   *
   * @tparam MemberType Kokos team member type.
   * @param team Kokkos team member.
   */
  template <typename MemberType>
  KOKKOS_FUNCTION stress_integrand(const MemberType &team)
      : F(team.team_scratch(0)) {}

  ///@}

  /**
   * @brief Get the amount memory in bytes required for shared memory
   *
   * @return int  Amount of shared memory required
   */
  constexpr static int shmem_size() { return ViewType::shmem_size(); }
};

} // namespace chunk_element
} // namespace specfem
