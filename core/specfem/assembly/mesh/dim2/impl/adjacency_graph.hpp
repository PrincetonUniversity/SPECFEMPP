#pragma once

#include "specfem/mesh.hpp"

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
class adjacency_graph<specfem::element::dimension_tag::dim2>
    : public specfem::mesh::adjacency_graph<
          specfem::element::dimension_tag::dim2> {

private:
  using base_type =
      specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim2>;

public:
  /**
   * @brief Inherit all constructors from base class.
   */
  using base_type::base_type;

  /**
   * @brief Get mutable reference to the local Boost graph
   *
   * The assembly adjacency graph only contains intra-partition edges
   * (copied from the mesh graph by @c build_assembly_adjacency_graph),
   * so graph() delegates directly to local_connections().
   */
  auto &graph() { return base_type::local_connections(); }

  /// @overload
  const auto &graph() const { return base_type::local_connections(); }

  /// Deleted — assembly graph contains only local edges; use graph() instead.
  auto &local_connections() const = delete;

  /// Deleted — assembly graph has no cross-partition edges; not applicable.
  auto &mpi_connections() const = delete;
};

} // namespace specfem::assembly::mesh_impl
