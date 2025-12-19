/**
 * @brief 3D coordinate mapping between adjacent hexahedral spectral elements.
 *
 * Handles face-to-face, edge-to-edge, and corner coordinate transformations
 * with orientation handling for 3D hexahedral elements.
 */

#pragma once

#include "../mesh_entities.hpp"
#include <functional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace specfem::connections {

/**
 * @brief 3D coordinate mapping between adjacent hexahedral elements.
 *
 * Maps coordinates between faces, edges, and corners of adjacent 3D elements
 * using affine transformations and permutation handling.
 *
 * @code
 * connection_mapping<dim3> mapping(5, 5, 5, elem1_nodes, elem2_nodes);
 * auto [iz, iy, ix] = mapping.map_coordinates(
 *     mesh_entity::left, mesh_entity::right, 2, 1, 0);
 * @endcode
 */
template <> class connection_mapping<specfem::dimension::type::dim3> {
private:
  /// Kokkos view for element control node indices
  using ElementIndexView =
      Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>;

public:
  /** @brief Dimension tag for 3D spectral elements */
  constexpr static auto dimension_tag = specfem::dimension::type::dim3;

  /**
   * @brief Construct 3D connection mapping.
   * @param ngllz Number of GLL points in z-direction
   * @param nglly Number of GLL points in y-direction
   * @param ngllx Number of GLL points in x-direction
   * @param element1 Control node indices for first element
   * @param element2 Control node indices for second element
   * @throws std::runtime_error if elements have different node counts
   */
  connection_mapping(const int ngllz, const int nglly, const int ngllx,
                     ElementIndexView element1, ElementIndexView element2)
      : ngllz(ngllz), nglly(nglly), ngllx(ngllx), element1(element1),
        element2(element2) {
    if (element1.extent(0) != element2.extent(0)) {
      throw std::runtime_error(
          "The 2 elements must have the same number of control nodes.");
    }
  }

  /**
   * @brief Map coordinates between mesh entities.
   * @param from Source entity on element1
   * @param to Target entity on element2
   * @param iz Grid index in z-direction
   * @param iy Grid index in y-direction
   * @param ix Grid index in x-direction
   * @return Mapped coordinates (iz', iy', ix') on target entity
   */
  std::tuple<int, int, int>
  map_coordinates(const specfem::mesh_entity::dim3::type &from,
                  const specfem::mesh_entity::dim3::type &to, const int iz,
                  const int iy, const int ix) const;

  /**
   * @brief Map coordinates for corner entities.
   * @param from Source entity on element1
   * @param to Target entity on element2
   * @return Base mapped coordinates for entity transformation
   */
  std::tuple<int, int, int>
  map_coordinates(const specfem::mesh_entity::dim3::type &from,
                  const specfem::mesh_entity::dim3::type &to) const;

private:
  int ngllz; ///< Number of GLL points in z-direction
  int nglly; ///< Number of GLL points in y-direction
  int ngllx; ///< Number of GLL points in x-direction

  ElementIndexView element1; ///< Control node indices for first element
  ElementIndexView element2; ///< Control node indices for second element
};

} // namespace specfem::connections
