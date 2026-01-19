#pragma once

#include "impl/adjacency_graph.hpp"
#include "impl/control_nodes.hpp"
#include "impl/mesh_to_compute_mapping.hpp"
#include "impl/points.hpp"
#include "impl/shape_functions.hpp"
#include "specfem/assembly/mesh/impl/quadrature.hpp"
#include "specfem/mesh.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"
#include "specfem_setup.hpp"
#include <vector>

namespace specfem::assembly {

/**
 * @brief 2D assembly-optimized mesh for spectral element computations.
 *
 * Combines all mesh components (points, control nodes, shape functions, etc.)
 * with compute-optimized ordering for efficient assembly operations.
 *
 * Inherits functionality from:
 * - points: Quadrature point coordinates and indexing
 * - quadrature: GLL quadrature points and weights
 * - control_nodes: Element control node data
 * - mesh_to_compute_mapping: Element reordering for performance
 * - shape_functions: Shape function values and derivatives
 * - adjacency_graph: Element connectivity information
 *
 * @see specfem::mesh::mesh
 */
template <>
struct mesh<specfem::dimension::type::dim2>
    : public specfem::assembly::mesh_impl::points<
          specfem::dimension::type::dim2>,
      public specfem::assembly::mesh_impl::quadrature<
          specfem::dimension::type::dim2>,
      public specfem::assembly::mesh_impl::control_nodes<
          specfem::dimension::type::dim2>,
      public specfem::assembly::mesh_impl::mesh_to_compute_mapping<
          specfem::dimension::type::dim2>,
      public specfem::assembly::mesh_impl::shape_functions<
          specfem::dimension::type::dim2>,
      public specfem::assembly::mesh_impl::adjacency_graph<
          specfem::dimension::type::dim2> {

public:
  constexpr static auto dimension_tag = specfem::dimension::type::dim2;
  constexpr static auto ndim = 2;

  int nspec; ///< Number of spectral elements
  int ngnod; ///< Number of control nodes per element
  specfem::mesh_entity::element_grid<dimension_tag> element_grid; ///< GLL grid
                                                                  ///< info

  /**
   * @brief Default constructor.
   */
  mesh() = default;

  /**
   * @brief Constructor from mesh components.
   *
   * Builds assembly mesh from source mesh data with compute optimization.
   *
   * @param tags Element tags for reordering
   * @param control_nodes Element control node data
   * @param quadratures GLL quadrature information
   * @param adjacency_graph Element connectivity
   */
  mesh(const specfem::mesh::tags<dimension_tag> &tags,
       const specfem::mesh::control_nodes<dimension_tag> &control_nodes,
       const specfem::quadrature::quadratures &quadratures,
       const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph);

  /**
   * @brief Assemble quadrature point coordinates.
   *
   * Computes physical coordinates for all quadrature points using
   * control nodes and shape functions.
   */
  void assemble();
};
} // namespace specfem::assembly
