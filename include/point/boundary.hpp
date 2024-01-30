#ifndef _POINT_BOUNDARY_HPP
#define _POINT_BOUNDARY_HPP

#include "enumerations/specfem_enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {
struct boundary {
  specfem::enums::element::boundary_tag top =
      specfem::enums::element::boundary_tag::none; ///< top boundary tag
  specfem::enums::element::boundary_tag bottom =
      specfem::enums::element::boundary_tag::none; ///< bottom boundary tag
  specfem::enums::element::boundary_tag left =
      specfem::enums::element::boundary_tag::none; ///< left boundary tag
  specfem::enums::element::boundary_tag right =
      specfem::enums::element::boundary_tag::none; ///< right boundary tag
  specfem::enums::element::boundary_tag bottom_right =
      specfem::enums::element::boundary_tag::none; ///< bottom right boundary
                                                   ///< tag
  specfem::enums::element::boundary_tag bottom_left =
      specfem::enums::element::boundary_tag::none; ///< bottom left boundary tag
  specfem::enums::element::boundary_tag top_right =
      specfem::enums::element::boundary_tag::none; ///< top right boundary tag
  specfem::enums::element::boundary_tag top_left =
      specfem::enums::element::boundary_tag::none; ///< top left boundary tag

  /**
   * @brief Construct a new boundary types object
   *
   */
  KOKKOS_FUNCTION boundary() = default;

  /**
   * @brief Update the tag for a given boundary type
   *
   * @param type Type of the boundary to update - defines an edge or node
   * @param tag Tag to update the boundary with
   */
  void update_boundary(const specfem::enums::boundaries::type &type,
                       const specfem::enums::element::boundary_tag &tag);
};

KOKKOS_FUNCTION
bool is_on_boundary(const specfem::enums::element::boundary_tag &tag,
                    const specfem::point::boundary &type, const int &iz,
                    const int &ix, const int &ngllz, const int &ngllx);

} // namespace point
} // namespace specfem

#endif
