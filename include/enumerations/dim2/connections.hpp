#pragma once

#include "../mesh_entities.hpp"
#include <functional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace specfem::connections {

/**
 * @brief 2D coordinate mapping between adjacent spectral elements.
 *
 * Maps coordinates between mesh entities (edges, corners) of adjacent
 * quadrilateral elements.
 */
template <> class connection_mapping<specfem::dimension::type::dim2> {
private:
  using ElementIndexView =
      Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>;

public:
  constexpr static auto dimension_tag = specfem::dimension::type::dim2;

  /**
   * @brief Construct 2D connection mapping.
   * @param ngllz Number of GLL points in z-direction
   * @param ngllx Number of GLL points in x-direction
   * @param element1 Control node indices for first element
   * @param element2 Control node indices for second element
   * @throws std::runtime_error if elements have different node counts
   */
  connection_mapping(const int ngllz, const int ngllx,
                     ElementIndexView element1, ElementIndexView element2)
      : ngllz(ngllz), ngllx(ngllx), element1(element1), element2(element2) {
    if (element1.extent(0) != element2.extent(0)) {
      throw std::runtime_error(
          "Elements must have the same number of control nodes.");
    }
  }

  /**
   * @brief Map coordinates between mesh entities.
   * @param from Source entity on element1
   * @param to Target entity on element2
   * @param iz Grid index in z-direction
   * @param ix Grid index in x-direction
   * @return Mapped coordinates (iz', ix') on target entity
   */
  std::tuple<int, int>
  map_coordinates(const specfem::mesh_entity::dim2::type &from,
                  const specfem::mesh_entity::dim2::type &to, const int iz,
                  const int ix) const;

  /**
   * @brief Map coordinates for corner entities.
   * @param from Source entity on element1
   * @param to Target entity on element2
   * @return Base mapped coordinates for entity transformation
   */
  std::tuple<int, int>
  map_coordinates(const specfem::mesh_entity::dim2::type &from,
                  const specfem::mesh_entity::dim2::type &to) const;

private:
  int ngllz; ///< Number of GLL points in z-direction
  int ngllx; ///< Number of GLL points in x-direction

  ElementIndexView element1; ///< Control node indices for first element
  ElementIndexView element2; ///< Control node indices for second element
};

} // namespace specfem::connections
