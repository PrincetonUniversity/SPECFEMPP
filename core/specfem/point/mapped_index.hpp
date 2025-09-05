#pragma once

#include "enumerations/interface.hpp"
#include "index.hpp"
#include "specfem/data_access.hpp"

namespace specfem {
namespace point {

/**
 * @brief Struct associated to store the index of a quadrature point, along with
 * a mapping to reference another parameter to be associated with the qudrature
 * point.
 *
 * This struct is used to associated a vector paramter with a quadrature point
 * index.
 * @code
 * View<int*> mapped_indices("mapped_indices", n_points);
 * // The following code associates the 0-th mapped index with the quadrature
 * point at (0, 1, 1)
 * specfem::point::mapped_index<specfem::dimension::type::dim2, false> index(
 *     specfem::point::index(0, 1, 1), 0);
 * @endcode
 * @tparam DimensionTag Dimension of the element where the quadrature point is
 * located
 * @tparam UseSIMD Flag to indicate if this is a SIMD index
 */
template <specfem::dimension::type DimensionTag, bool UseSIMD>
struct mapped_index : public index<DimensionTag, UseSIMD> {
private:
  using base_type = index<DimensionTag, UseSIMD>;

public:
  int imap; ///< Index of the mapped element

  /**
   * @brief Constructor for the mapped index
   *
   * @param index Index to store the location of the quadrature point
   * @param imap Index of the mapped element to be associated with the
   * quadrature point
   */
  KOKKOS_INLINE_FUNCTION
  mapped_index(const base_type &index, const int &imap)
      : base_type(index), imap(imap) {}
};

} // namespace point
} // namespace specfem
