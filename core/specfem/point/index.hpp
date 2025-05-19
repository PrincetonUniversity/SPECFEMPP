#pragma once

#include "enumerations/dimension.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace point {

/**
 * @brief Struct to store the index associated with a quadrature point
 *
 * @tparam DimensionTag Dimension of the element where the quadrature point is
 * located
 * @tparam using_simd Flag to indicate if this is a simd index
 */
template <specfem::dimension::type DimensionTag, bool using_simd = false>
struct index;

/**
 * @brief Alias for the simd index
 *
 * @tparam DimensionTag Dimension of the element where the quadrature point is
 * located
 */
template <specfem::dimension::type DimensionTag>
using simd_index = index<DimensionTag, true>;

//--------------------------- 2D Specializations -----------------------------//

/**
 * @brief 2D specialization of the index struct for the non-SIMD case
 *
 */
template <> struct index<specfem::dimension::type::dim2, false> {
  int ispec; ///< Index of the spectral element
  int iz;    ///< Index of the quadrature point in the z direction within the
             ///< spectral element
  int ix;    ///< Index of the quadrature point in the x direction within the
             ///< spectral element

  constexpr static bool using_simd =
      false; ///< Flag to indicate that SIMD is not being used'

  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension type

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
   * @param iz Index of the quadrature point in the z direction within the
   * spectral element
   * @param ix Index of the quadrature point in the x direction within the
   * spectral element
   */
  KOKKOS_FUNCTION
  index(const int &ispec, const int &iz, const int &ix)
      : ispec(ispec), iz(iz), ix(ix) {}
};

/**
 * @brief 2D specialization of the index struct for the SIMD case
 *
 * @copydoc simd_index
 *
 */
template <> struct index<specfem::dimension::type::dim2, true> {
  int ispec; ///< Index associated with the spectral element at the start
             ///< of the SIMD vector
  int number_elements; ///< Number of elements stored in the SIMD vector
  int iz; ///< Index of the quadrature point in the z direction within
          ///< the spectral element
  int ix; ///< Index of the quadrature point in the x direction within
          ///< the spectral element

  constexpr static bool using_simd =
      true; ///< Flag to indicate that SIMD is being used

  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension type

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
   * @param iz Index of the quadrature point in the z direction within the
   * spectral element
   * @param ix Index of the quadrature point in the x direction within the
   * spectral element
   */
  KOKKOS_FUNCTION
  index(const int &ispec, const int &number_elements, const int &iz,
        const int &ix)
      : ispec(ispec), number_elements(number_elements), iz(iz), ix(ix) {}

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

//-------------------------- 3D Specializations ------------------------------//

/**
 * @brief Template specialization for 3D elements for the non-SIMD index
 *        implementation.
 *
 */
template <> struct index<specfem::dimension::type::dim3, false> {
  int ispec; ///< Index of the spectral element
  int iz;    ///< Index of the quadrature point in the z direction within the
             ///< spectral element
  int iy;    ///< Index of the quadrature point in the y direction within the
             ///< spectral element
  int ix;    ///< Index of the quadrature point in the x direction within the
             ///< spectral element

  constexpr static bool using_simd =
      false; ///< Flag to indicate that SIMD is not being used'

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension type

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
   * @param iz Index of the quadrature point in the z direction within the
   * spectral element
   * @param iy Index of the quadrature point in the y direction within the
   * spectral element
   * @param ix Index of the quadrature point in the x direction within the
   * spectral element
   */
  KOKKOS_FUNCTION
  index(const int &ispec, const int &iz, const int &iy, const int &ix)
      : ispec(ispec), iz(iz), iy(iy), ix(ix) {};
};

/**
 * @brief Template specialization for 2D elements for the SIMD index
 * implementation.
 *
 * @copydoc simd_index
 *
 */
template <> struct index<specfem::dimension::type::dim3, true> {
  int ispec; ///< Index associated with the spectral element at the start
             ///< of the SIMD vector
  int number_elements; ///< Number of elements stored in the SIMD vector
  int iz; ///< Index of the quadrature point in the z direction within
          ///< the spectral element
  int iy; ///< Index of the quadrature point in the y direction within the
          ///< spectral element
  int ix; ///< Index of the quadrature point in the x direction within
          ///< the spectral element

  constexpr static bool using_simd =
      true; ///< Flag to indicate that SIMD is being used

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension type

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
   * @param iz Index of the quadrature point in the z direction within the
   * spectral element
   * @param iy Index of the quadrature point in the y direction within the
   * spectral element
   * @param ix Index of the quadrature point in the x direction within the
   * spectral element
   */
  KOKKOS_FUNCTION
  index(const int &ispec, const int &number_elements, const int &iz,
        const int &iy, const int &ix)
      : ispec(ispec), number_elements(number_elements), iz(iz), iy(iy), ix(ix) {
  }

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

} // namespace point
} // namespace specfem
