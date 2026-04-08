#pragma once

#include "mesh_to_compute_mapping.hpp"
#include "specfem/element.hpp"
#include "specfem/mesh.hpp"

namespace specfem::assembly::mesh_impl {

/**
 * @brief 3D adjacency graph with compute-optimized element ordering.
 *
 * Inherits from specfem::mesh::adjacency_graph but uses compute-optimized
 * element indices instead of mesh ordering for better assembly performance.
 *
 * @see specfem::mesh::adjacency_graph
 */
template <>
class adjacency_graph<specfem::element::dimension_tag::dim3>
    : public specfem::mesh::adjacency_graph<
          specfem::element::dimension_tag::dim3> {

private:
  using base_type =
      specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim3>;

public:
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;
  /**
   * @brief Inherit all constructors from base class.
   */
  using base_type::base_type;

  /**
   * @brief Construct a new adjacency graph with compute-optimized ordering.
   *
   * Reorders the adjacency graph based on the provided mesh to compute mapping
   * to optimize for assembly performance.
   *
   * @param nspec Number of spectral elements
   * @param mapping Mapping between mesh and compute element indices
   * @param mesh_adjacency_graph Original adjacency graph in mesh ordering
   * @return adjacency_graph Reordered adjacency graph in compute ordering
   */
  adjacency_graph(const int nspec,
                  const mesh_to_compute_mapping<dimension_tag> &mapping,
                  const specfem::mesh::adjacency_graph<dimension_tag>
                      &mesh_adjacency_graph);

  auto &graph() { return base_type::local_connections(); }

  const auto &graph() const { return base_type::local_connections(); }

private:
  using base_local_connections_type =
      decltype(std::declval<base_type &>().local_connections());
  using base_local_connections_const_type =
      decltype(std::declval<const base_type &>().local_connections());
  using base_mpi_connections_type =
      decltype(std::declval<base_type &>().mpi_connections());
  using base_mpi_connections_const_type =
      decltype(std::declval<const base_type &>().mpi_connections());

public:
  // local_connections() and mpi_connections() are deleted to force callers
  // to use graph() instead, which returns the compute-optimized ordering.
  // Note: because these are non-virtual, callers holding a base_type&
  // reference can still invoke the base versions; this delete only applies
  // when called through the derived type directly.
  base_local_connections_type &local_connections() = delete;
  base_local_connections_const_type &local_connections() const = delete;
  base_mpi_connections_type &mpi_connections() = delete;
  base_mpi_connections_const_type &mpi_connections() const = delete;
};

} // namespace specfem::assembly::mesh_impl
