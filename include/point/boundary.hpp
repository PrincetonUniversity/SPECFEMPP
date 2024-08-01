#ifndef _POINT_BOUNDARY_HPP
#define _POINT_BOUNDARY_HPP

#include "datatypes/simd.hpp"
#include "enumerations/boundary.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

template <bool UseSIMD, specfem::element::boundary_tag BoundaryTag>
struct boundary {
public:
  constexpr static auto boundary_tag = BoundaryTag; ///< Boundary tag
  constexpr static bool isPointBoundaryType =
      true; ///< Is a point boundary type

  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type

private:
  constexpr static int simd_size =
      specfem::datatype::simd<type_real, UseSIMD>::size(); ///< SIMD size

public:
  specfem::element::boundary_tag_container tags[simd_size]; ///< Boundary
                                                            ///< tag

  /**
   * @brief Construct a new boundary object
   *
   */
  KOKKOS_FUNCTION boundary() = default;

  template <specfem::element::boundary_tag UTag>
  KOKKOS_FUNCTION boundary(const boundary<UseSIMD, UTag> &other) {
    for (int i = 0; i < simd_size; i++) {
      tags[i] = other.tags[i];
    }
  }

  // move constructor
  template <specfem::element::boundary_tag UTag>
  KOKKOS_FUNCTION boundary(boundary<UseSIMD, UTag> &&other)
      : tags(std::move(other.tags)) {}
};

// struct boundary {
//   specfem::element::boundary_tag top =
//       specfem::element::boundary_tag::none; ///< top boundary tag
//   specfem::element::boundary_tag bottom =
//       specfem::element::boundary_tag::none; ///< bottom boundary tag
//   specfem::element::boundary_tag left =
//       specfem::element::boundary_tag::none; ///< left boundary tag
//   specfem::element::boundary_tag right =
//       specfem::element::boundary_tag::none; ///< right boundary tag
//   specfem::element::boundary_tag bottom_right =
//       specfem::element::boundary_tag::none; ///< bottom right boundary
//                                             ///< tag
//   specfem::element::boundary_tag bottom_left =
//       specfem::element::boundary_tag::none; ///< bottom left boundary tag
//   specfem::element::boundary_tag top_right =
//       specfem::element::boundary_tag::none; ///< top right boundary tag
//   specfem::element::boundary_tag top_left =
//       specfem::element::boundary_tag::none; ///< top left boundary tag

//   /**
//    * @brief Construct a new boundary types object
//    *
//    */
//   KOKKOS_FUNCTION boundary() = default;

//   /**
//    * @brief Update the tag for a given boundary type
//    *
//    * @param type Type of the boundary to update - defines an edge or node
//    * @param tag Tag to update the boundary with
//    */
//   void update_boundary(const specfem::enums::boundaries::type &type,
//                        const specfem::element::boundary_tag &tag);
// };

// KOKKOS_FUNCTION
// bool is_on_boundary(const specfem::element::boundary_tag &tag,
//                     const specfem::point::boundary &type, const int &iz,
//                     const int &ix, const int &ngllz, const int &ngllx);

} // namespace point
} // namespace specfem

#endif
