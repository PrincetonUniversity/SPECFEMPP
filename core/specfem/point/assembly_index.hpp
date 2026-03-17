#pragma once

#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @struct assembly_index
 * @brief Struct to store the assembled index for a quadrature point
 *
 * @tparam using_simd Flag to indicate if this is a simd index
 */
template <bool using_simd = false> struct assembly_index;

/**
 * @struct assembly_index<false>
 * @brief Struct to store the assembled index for a quadrature point
 *
 * This struct stores a 1D index that corresponds to a global numbering of the
 * quadrature point within the mesh.
 *
 */
template <>
struct assembly_index<false>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::assembly_index,
          specfem::element::dimension_tag::dim2, false> {
  /**
   * @brief Global index number of the quadrature point.
   *
   * This index uniquely identifies the quadrature point within the global mesh
   * numbering system. It is used for gathering and scattering data to global
   * arrays.
   */
  int iglob;

  /**
   * @name Constructors
   * @brief Constructors for initializing the assembly index.
   */
  ///@{
  /**
   * @brief Default constructor.
   *
   * Initializes an empty assembly index.
   */
  KOKKOS_FUNCTION
  assembly_index() = default;

  /**
   * @brief Constructor with global index.
   *
   * Initializes the assembly index with a specific global index.
   *
   * @param iglob Global index number of the quadrature point.
   */
  KOKKOS_FUNCTION
  assembly_index(const int &iglob) : iglob(iglob) {}
  ///@}
};

/**
 * @struct assembly_index<true>
 * @brief Struct to store the SIMD assembled indices for a quadrature point
 *
 * SIMD indices are intended to be used for loading @c load_on_device and
 * storing @c store_on_device data into SIMD vectors and operating on those data
 * using SIMD instructions.
 *
 */
template <>
struct assembly_index<true>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::assembly_index,
          specfem::element::dimension_tag::dim2, true> {
  /**
   * @brief Number of active points in the SIMD vector.
   *
   * Indicates how many lanes in the SIMD vector contain valid data.
   * This is used for masking operations at boundaries or when the number of
   * points is not a multiple of the SIMD width.
   */
  int number_points;

  /**
   * @brief Global index number of the first quadrature point in the SIMD batch.
   *
   * Represents the starting global index. Subsequent points in the SIMD vector
   * are typically assumed to be contiguous or strided based on the access
   * pattern.
   */
  int iglob;

  /**
   * @brief Mask function to determine if a SIMD lane is valid.
   *
   * Checks if the given lane index is within the valid range of points
   * for this SIMD batch.
   *
   * @param lane The SIMD lane index to check (0 to vector_width-1).
   * @return true if the lane is active/valid, false otherwise.
   */
  KOKKOS_FUNCTION
  bool mask(const std::size_t &lane) const { return int(lane) < number_points; }

  /**
   * @brief Returns a SIMD mask for valid lanes.
   *
   * For full chunks (all lanes valid) returns an all-true mask using the cheap
   * broadcast constructor, avoiding the per-lane generator. For the rare
   * partial last chunk falls back to per-lane evaluation.
   *
   * @tparam simd_type SIMD wrapper type (e.g. specfem::datatype::simd<float,
   * true>)
   * @return SIMD mask with true for valid lanes
   */
  template <typename simd_type>
  KOKKOS_INLINE_FUNCTION typename simd_type::mask_type get_mask() const {
    if (number_points >= simd_type::size()) {
      return typename simd_type::mask_type(true);
    }
    return typename simd_type::mask_type(
        [&](std::size_t lane) { return int(lane) < number_points; });
  }

  /**
   * @name Constructors
   * @brief Constructors for initializing the SIMD assembly index.
   */
  ///@{
  /**
   * @brief Default constructor.
   *
   * Initializes an empty SIMD assembly index.
   */
  KOKKOS_FUNCTION
  assembly_index() = default;

  /**
   * @brief Constructor with values.
   *
   * Initializes the SIMD assembly index with a starting global index and
   * the number of valid points.
   *
   * @param iglob Global index number of the first quadrature point.
   * @param number_points Number of valid points in the SIMD vector.
   */
  KOKKOS_FUNCTION
  assembly_index(const int &iglob, const int &number_points)
      : number_points(number_points), iglob(iglob) {}
  ///@}
};

} // namespace point
} // namespace specfem
