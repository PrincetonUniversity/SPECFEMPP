#pragma once

#include "enumerations/dimension.hpp"
#include "specfem/data_access.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace point {

/**
 * @brief Local element indexing system for quadrature points within spectral
 * elements.
 *
 * The index struct provides a comprehensive indexing system for uniquely
 * identifying quadrature points within individual spectral elements. This local
 * indexing system is essential for accessing element-specific data, performing
 * local operations, and mapping between different coordinate systems within the
 * spectral element framework.
 *
 * @tparam DimensionTag Spatial dimension of the element (dim2 or dim3).
 * @tparam using_simd Boolean flag enabling SIMD vectorization support.
 *                   When `true`, enables vectorized operations on multiple
 * indices. When `false`, uses scalar indexing for single points.
 *
 * @note The local indexing follows the spectral element convention where
 * indices typically range from 0 to N-1, where N is the polynomial degree + 1.
 *
 * @see specfem::point::simd_index for SIMD-enabled vectorized indexing
 * @see specfem::point::assembly_index for global DOF indexing
 *
 * @code
 * // Example: 2D local indexing
 * using LocalIndex = specfem::point::index<specfem::dimension::type::dim2,
 * false>; LocalIndex local_idx; local_idx.ispec = element_id; local_idx.ix = 3;
 * // 4th point in x-direction (0-indexed) local_idx.iz = 2;  // 3rd point in
 * z-direction
 *
 * // Use for field access
 * auto field_value = element_fields(local_idx.ispec, local_idx.iz,
 * local_idx.ix);
 * @endcode
 */
template <specfem::dimension::type DimensionTag, bool using_simd = false>
struct index;

/**
 * @brief Type alias for SIMD-vectorized local element indexing.
 *
 * This alias provides convenient access to the SIMD-enabled specialization of
 * the index template, which supports vectorized operations on multiple
 * quadrature points simultaneously. SIMD indexing is crucial for achieving
 * optimal performance on modern vector architectures by processing multiple
 * points in parallel.
 *
 * @tparam DimensionTag Spatial dimension determining the indexing structure.
 *
 * @see specfem::point::index<DimensionTag, true> for the underlying SIMD
 * implementation
 * @see specfem::datatype::simd for SIMD type definitions
 *
 * @code
 * // Example: SIMD vectorized indexing
 * using SIMDIndex = specfem::point::simd_index<specfem::dimension::type::dim2>;
 * SIMDIndex simd_idx(element_id, base_iz, base_ix, num_valid_points);
 *
 * // Process multiple points with vectorization
 * for (int lane = 0; lane < num_valid_points; ++lane) {
 *   if (simd_idx.mask(lane)) {
 *     // Process valid lane
 *   }
 * }
 * @endcode
 */
template <specfem::dimension::type DimensionTag>
using simd_index = index<DimensionTag, true>;

//--------------------------- 2D Specializations -----------------------------//

/**
 * @brief 2D scalar specialization for local element quadrature point indexing.
 *
 * This specialization provides indexing for 2D spectral elements using a
 * three-component system: element index (ispec) and two spatial indices (ix,
 * iz). The indexing follows the tensor-product structure characteristic of
 * spectral element methods, where each spatial direction is independently
 * indexed.
 *
 * The 2D indexing convention typically follows:
 * - ispec: Element identifier within the mesh [0, num_elements)
 * - ix: Index along the x-direction within the element [0, NGLL_X)
 * - iz: Index along the z-direction within the element [0, NGLL_Z)
 *
 * This indexing system enables efficient access to:
 * - Element-local field arrays organized as [ispec][iz][ix]
 * - Basis function coefficients and quadrature weights
 * - Local mass and stiffness matrix entries
 * - Geometric transformation data (Jacobians, coordinate mappings)
 *
 * @note NGLL_X and NGLL_Z are typically (polynomial_degree + 1) in each
 * direction, commonly 5 for degree-4 polynomials in standard spectral element
 * setups.
 *
 * @see specfem::quadrature for GLL point ordering conventions
 * @see specfem::assembly for element assembly operations
 *
 * @code
 * // Example: Traversing all quadrature points in a 2D element
 * specfem::point::index<specfem::dimension::type::dim2, false> idx;
 * idx.ispec = element_id;
 *
 * for (int iz = 0; iz < NGLL_Z; ++iz) {
 *   for (int ix = 0; ix < NGLL_X; ++ix) {
 *     idx.iz = iz; idx.ix = ix;
 *
 *     // Access field values at this quadrature point
 *     auto field_val = element_fields(idx.ispec, idx.iz, idx.ix);
 *
 *     // Perform local operations
 *     process_quadrature_point(idx, field_val);
 *   }
 * }
 * @endcode
 */
template <>
struct index<specfem::dimension::type::dim2, false>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::index,
          specfem::dimension::type::dim2, false> {
  int ispec; ///< Index of the spectral element
  int iz;    ///< Index of the quadrature point in the z direction within the
             ///< spectral element
  int ix;    ///< Index of the quadrature point in the x direction within the
             ///< spectral element

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
template <>
struct index<specfem::dimension::type::dim2, true>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::index,
          specfem::dimension::type::dim2, true> {
  int ispec; ///< Index associated with the spectral element at the start
             ///< of the SIMD vector
  int number_elements; ///< Number of elements stored in the SIMD vector
  int iz; ///< Index of the quadrature point in the z direction within
          ///< the spectral element
  int ix; ///< Index of the quadrature point in the x direction within
          ///< the spectral element

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
template <>
struct index<specfem::dimension::type::dim3, false>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::index,
          specfem::dimension::type::dim3, false> {
  int ispec; ///< Index of the spectral element
  int iz;    ///< Index of the quadrature point in the z direction within the
             ///< spectral element
  int iy;    ///< Index of the quadrature point in the y direction within the
             ///< spectral element
  int ix;    ///< Index of the quadrature point in the x direction within the
             ///< spectral element

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
template <>
struct index<specfem::dimension::type::dim3, true>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::index,
          specfem::dimension::type::dim3, true> {
  int ispec; ///< Index associated with the spectral element at the start
             ///< of the SIMD vector
  int number_elements; ///< Number of elements stored in the SIMD vector
  int iz; ///< Index of the quadrature point in the z direction within
          ///< the spectral element
  int iy; ///< Index of the quadrature point in the y direction within the
          ///< spectral element
  int ix; ///< Index of the quadrature point in the x direction within
          ///< the spectral element

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
