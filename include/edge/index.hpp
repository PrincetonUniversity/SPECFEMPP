#pragma once
#include "enumerations/specfem_enums.hpp"

namespace specfem {
namespace edge {

// this is similar to include/point/coordinates.hpp:index

template <specfem::dimension::type DimensionType, bool using_simd = false>
struct index;

template <> struct index<specfem::dimension::type::dim2, false> {
  int ispec;                            ///< Index of the spectral element
  specfem::enums::edge::type edge_type; ///< edge type within the spectral
                                        ///< element

  constexpr static bool using_simd =
      false; ///< Flag to indicate that SIMD is not being used

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  index() = default;

  /**
   * @brief Construct a new index object
   *
   * @param ispec Index of the spectral element
   * @param edge_type edge type within the spectral element
   */
  KOKKOS_FUNCTION
  index(const int &ispec, const specfem::enums::edge::type &edge_type)
      : ispec(ispec), edge_type(edge_type) {}
};

/**
 * @brief Template specialization for 2D elements
 *
 * @copydoc simd_index
 *
 */
template <> struct index<specfem::dimension::type::dim2, true> {
  int ispec; ///< Index associated with the spectral element at the start
             ///< of the SIMD vector
  specfem::enums::edge::type edge_type; ///< edge type within the spectral
                                        ///< element
  int number_elements; ///< Number of elements stored in the SIMD vector

  constexpr static bool using_simd =
      true; ///< Flag to indicate that SIMD is being used

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  index() = default;

  /**
   * @brief Construct a new simd index object
   *
   * @param ispec Index of the spectral element
   * @param number_elements Number of elements
   * @param edge_type edge type within the spectral element
   */
  KOKKOS_FUNCTION
  index(const int &ispec, const int &number_elements,
        const specfem::enums::edge::type &edge_type)
      : ispec(ispec), number_elements(number_elements), edge_type(edge_type) {}

  /**
   * @brief Returns a boolean mask to check if the SIMD index is within the SIMD
   * vector
   *
   * @param lane SIMD lane
   * @return bool True if the SIMD index is within the SIMD vector
   */
  KOKKOS_INLINE_FUNCTION
  bool mask(const std::size_t &lane) const {
    return int(lane) < number_elements;
  }
};

/**
 * @brief Alias for the simd index
 *
 * @tparam DimensionType Dimension of the element where the quadrature point is
 * located
 */
template <specfem::dimension::type DimensionType>
using simd_index = index<DimensionType, true>;

} // namespace edge
} // namespace specfem
