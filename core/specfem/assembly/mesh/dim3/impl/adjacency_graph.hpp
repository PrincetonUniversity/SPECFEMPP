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

  auto &local_connections() = delete;
  const auto &local_connections() const = delete;
  auto &mpi_connections() = delete;
  const auto &mpi_connections() const = delete;
};

} // namespace specfem::assembly::mesh_impl
