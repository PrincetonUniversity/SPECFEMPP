
#pragma once

#include "edge_index.hpp"
#include "face_index.hpp"
#include "specfem/element.hpp"

namespace specfem::point {
/**
 * @brief Index pair for coupled interface points
 *
 * This struct holds the indices of corresponding points on the self and coupled
 * sides of an interface. It provides a unified way to access the local
 * coordinates of the interface points for both sides, which is essential for
 * computing fluxes, and applying coupling conditions in multi-domain
 * simulations.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag> class interface_index;

/**
 * @brief Specialization of interface_index for 2D
 *
 * In 2D, the interface is a line where two edges meet. This specialization
 * contains edge indices for both sides of the interface, allowing for
 * efficient access to corresponding points on either side.
 */
template <> class interface_index<specfem::element::dimension_tag::dim2> {
public:
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim2;
  /**
   * @brief Edge index on the self side of the interface.
   */
  specfem::point::edge_index<dimension_tag> self_index;

  /**
   * @brief Edge index on the coupled side of the interface.
   */
  specfem::point::edge_index<dimension_tag> coupled_index;

  /**
   * @brief Default constructor.
   */
  KOKKOS_INLINE_FUNCTION
  interface_index() = default;

  /**
   * @brief Constructs interface index from self and coupled edge indices
   *
   * @param self_index Edge index on the self side of interface
   * @param coupled_index Edge index on the coupled side of interface
   */
  KOKKOS_INLINE_FUNCTION
  interface_index(
      const specfem::point::edge_index<dimension_tag> &self_index,
      const specfem::point::edge_index<dimension_tag> &coupled_index)
      : self_index(self_index), coupled_index(coupled_index) {}
};

/**
 * @brief Specialization of interface_index for 3D
 *
 * In 3D, the interface is a surface where two faces meet. This specialization
 * contains face indices for both sides of the interface, allowing for
 * efficient access to corresponding points on either side.
 */
template <> class interface_index<specfem::element::dimension_tag::dim3> {
public:
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;

  /**
   * @brief Face index on the self side of the interface.
   */
  specfem::point::face_index<dimension_tag> self_index;

  /**
   * @brief Face index on the coupled side of the interface.
   */
  specfem::point::face_index<dimension_tag> coupled_index;

  /**
   * @brief Default constructor.
   */
  KOKKOS_INLINE_FUNCTION
  interface_index() = default;

  /**
   * @brief Constructs interface index from self and coupled face indices
   *
   * @param self_index Face index on the self side of interface
   * @param coupled_index Face index on the coupled side of interface
   */
  KOKKOS_INLINE_FUNCTION
  interface_index(
      const specfem::point::face_index<dimension_tag> &self_index,
      const specfem::point::face_index<dimension_tag> &coupled_index)
      : self_index(self_index), coupled_index(coupled_index) {}
};

} // namespace specfem::point
