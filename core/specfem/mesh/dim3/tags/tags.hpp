#pragma once

#include "specfem/enums.hpp"
#include "specfem/mesh/dim3/adjacency_graph/adjacency_graph.hpp"
#include "specfem/mesh/dim3/boundaries/boundaries.hpp"
#include "specfem/mesh/dim3/materials/materials.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::mesh {

template <specfem::element::dimension_tag DimensionTag> struct tags;

/**
 * @brief Element tagging system for 3D spectral elements
 *
 * Extracts material classification from MESHFEM3D materials containers and
 * creates tags_container objects for each spectral element. Uses parallel
 * initialization to assign medium and property tags based on material data.
 * Assigns medium, property, and boundary tags to every spectral element
 * using material data and the two boundary sub-structures
 * (absorbing_boundary and acoustic_free_surface).
 *
 * @code
 * specfem::mesh::materials<specfem::element::dimension_tag::dim3>
 * materials;
 * // ... populate materials ...
 *
 * specfem::mesh::tags<specfem::element::dimension_tag::dim3> tags(
 *     nspec, materials, adjacency_graph);
 * @endcode
 *
 * @see specfem::mesh::materials::get_material_type
 * @see specfem::mesh::impl::tags_container
 */
template <> struct tags<specfem::element::dimension_tag::dim3> {
  /**
   * @brief Dimension tag for compile-time type identification
   */
  static constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;

  /** @name Constructors */
  /** @{ */

  tags() = default;

  /**
   * @brief Construct tags from materials and boundary information.
   *
   * Extracts material classification (medium and property tags) from the
   * materials container for each element using parallel initialization.
   * Assigns medium/property tags from materials and derives boundary tags
   * directly from the boundary sub-structures:
   * - Elements in @c boundaries.absorbing_boundary → stacey
   * - Acoustic elements in @c boundaries.acoustic_free_surface →
   *   acoustic_free_surface
   * Classifies each element as MPI inner or outer from
   * the adjacency graph's MPI connections.
   *
   * @param nspec Total number of spectral elements in the mesh
   * @param materials MESHFEM3D materials container with material data
   * @param adjacency_graph Mesh-domain adjacency graph; its MPI connections
   *                        mark elements touching a partition boundary as outer
   * @param boundaries Domain boundary face information (two sub-structs)
   *
   * @see specfem::mesh::materials::get_material_type
   */
  tags(const int nspec, specfem::mesh::materials<dimension_tag> &materials,
       const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph,
       const specfem::mesh::boundaries<dimension_tag> &boundaries);

  /** @} */

  /** @name Data Members */
  /** @{ */

  /** @brief Total number of spectral elements in the mesh */
  int nspec;

  /**
   * @brief Kokkos host view containing tags for all elements
   *
   * Each entry contains medium, property, and boundary tags for one spectral
   * element. Boundary tags are currently set to `none` for all elements.
   */
  Kokkos::View<specfem::mesh::impl::tags_container *,
               Kokkos::DefaultHostExecutionSpace>
      tags_container;

  /** @} */
};

} // namespace specfem::mesh
