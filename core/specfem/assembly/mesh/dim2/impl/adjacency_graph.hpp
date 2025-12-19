#pragma once

#include "mesh/mesh.hpp"

namespace specfem::assembly::mesh_impl {

/**
 * @brief 2D adjacency graph with compute-optimized element ordering.
 *
 * Inherits from specfem::mesh::adjacency_graph but uses compute-optimized
 * element indices instead of mesh ordering for better assembly performance.
 *
 * @see specfem::mesh::adjacency_graph
 */
template <>
class adjacency_graph<specfem::dimension::type::dim2>
    : public specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> {

private:
  using base_type =
      specfem::mesh::adjacency_graph<specfem::dimension::type::dim2>;

public:
  /**
   * @brief Inherit all constructors from base class.
   */
  using base_type::base_type;
};

} // namespace specfem::assembly::mesh_impl
