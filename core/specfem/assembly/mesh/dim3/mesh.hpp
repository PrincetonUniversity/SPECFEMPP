#pragma once

#include "impl/control_nodes.hpp"
#include "impl/points.hpp"
#include "impl/shape_functions.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly/mesh/impl/quadrature.hpp"
#include "specfem/mesh.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem::assembly {

/**
 * @brief 3D assembly mesh for spectral element computations.
 *
 * Combines mesh components (points, control nodes, shape functions, quadrature)
 * for efficient 3D hexahedral spectral element assembly operations.
 *
 * Inherits functionality from:
 * - points: Quadrature point coordinates and indexing
 * - quadrature: GLL quadrature points and weights
 * - control_nodes: Element control node data
 * - shape_functions: Shape function values and derivatives
 *
 * @see specfem::mesh::mesh
 */
template <>
struct mesh<specfem::dimension::type::dim3>
    : public specfem::assembly::mesh_impl::points<
          specfem::dimension::type::dim3>,
      public specfem::assembly::mesh_impl::quadrature<
          specfem::dimension::type::dim3>,
      public specfem::assembly::mesh_impl::control_nodes<
          specfem::dimension::type::dim3>,
      public specfem::assembly::mesh_impl::shape_functions<
          specfem::dimension::type::dim3> {

public:
  constexpr static auto dimension_tag = specfem::dimension::type::dim3;
  constexpr static int ndim = 3;

  int nspec; ///< Number of spectral elements
  int ngnod; ///< Number of control nodes per element

  specfem::mesh_entity::element<dimension_tag> element_grid; ///< 3D GLL grid
                                                             ///< info

  /**
   * @brief Default constructor.
   */
  mesh() = default;

  /**
   * @brief Constructor from mesh components.
   *
   * Builds 3D assembly mesh from source mesh data.
   *
   * @param nspec Number of spectral elements
   * @param ngnod Number of control nodes per element
   * @param ngllz Number of GLL points in z direction
   * @param nglly Number of GLL points in y direction
   * @param ngllx Number of GLL points in x direction
   * @param adjacency_graph Element connectivity
   * @param control_nodes Element control node data
   * @param quadrature GLL quadrature information
   */
  mesh(const int nspec, const int ngnod, const int ngllz, const int nglly,
       const int ngllx,
       const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph,
       const specfem::mesh::control_nodes<dimension_tag> &control_nodes,
       const specfem::quadrature::quadratures &quadrature);
};

} // namespace specfem::assembly
