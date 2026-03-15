#pragma once

#include "specfem/element_coupling.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/mpi.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <memory>

namespace specfem::mesh {

/**
 * @brief Adjacency graph for mesh connectivity representation
 *
 * This class represents the adjacency relationships between mesh elements
 * using a Boost graph structure. It stores connectivity information between
 * spectral elements including the type of connection and orientation.
 *
 * @tparam Dimension The spatial dimension of the mesh (dim2 or dim3)
 */
template <specfem::element::dimension_tag Dimension> struct adjacency_graph {

public:
  /**
   * @brief Properties associated with graph edges
   *
   * This structure stores information about the connection between two
   * adjacent mesh elements, including the type of connection and the
   * orientation of the shared interface.
   */
  struct EdgeProperties {
    /** @brief Type of connection between adjacent elements */
    specfem::element_connections::type connection;

    /** @brief Orientation of the shared mesh entity (left, right, ...,
     * bottom_left, bottom_right, etc.) */
    specfem::mesh_entity::dim2::type orientation;

    /** @brief Partition rank of the neighboring element (if local connection
     * then @code local_partition = specfem::mpi::rank() @endcode, if
     * cross-partition connection then @code neighbor_partition = partition rank
     * of neighbor @endcode) */
    int neighbor_partition = -1;

    /**
     * @brief Default constructor
     *
     * Initializes edge properties with default values.
     */
    EdgeProperties() = default;

    /**
     * @brief Constructor with connection and orientation parameters
     *
     * @param conn Type of connection between elements
     * @param orient Orientation of the shared mesh entity
     * @param neighbor_part Partition rank of neighbor (-1 for intra-partition)
     */
    EdgeProperties(const specfem::element_connections::type conn,
                   const specfem::mesh_entity::dim2::type orient,
                   const int neighbor_part = -1)
        : connection(conn), orientation(orient),
          neighbor_partition(neighbor_part) {}
  };

private:
  /**
   * @brief Boost graph type definition
   *
   * Defines a directed adjacency list using:
   * - Vector storage for vertices (vecS)
   * - Vector storage for edges (vecS)
   * - Directed graph structure (directedS)
   * - No vertex properties
   * - EdgeProperties for edge data
   */
  using Graph =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
                            boost::no_property, EdgeProperties>;

  /** @brief The underlying Boost graph storing adjacency relationships */
  std::shared_ptr<Graph> p_graph_;

public:
  /**
   * @brief Default constructor
   *
   * Creates an empty adjacency graph with no vertices or edges.
   */
  adjacency_graph() : p_graph_(std::make_shared<Graph>(0)) {}

  /**
   * @brief Constructor with specified number of spectral elements
   *
   * Creates an adjacency graph with the specified number of vertices
   * (spectral elements) but no edges initially.
   *
   * @param nspec Number of spectral elements in the mesh
   */
  adjacency_graph(const int nspec) : p_graph_(std::make_shared<Graph>(nspec)) {}

  /**
   * @brief Get mutable reference to the underlying graph
   *
   * Provides direct access to the Boost graph structure for modification
   * operations such as adding edges or vertices.
   *
   * @return Mutable reference to the Boost adjacency_list graph
   */
  Graph &graph() { return *p_graph_; }

  /**
   * @brief Get const reference to the underlying graph
   *
   * Provides read-only access to the Boost graph structure for
   * query operations such as traversing edges or checking connectivity.
   *
   * @return Const reference to the Boost adjacency_list graph
   */
  const Graph &graph() const { return *p_graph_; }

  /**
   * @brief Get a filtered view containing only intra-partition edges
   *
   * Returns a Boost filtered_graph that includes only edges whose
   * neighbor_partition equals the current MPI rank (i.e. both source
   * and target elements belong to this partition).
   *
   * @return Filtered graph view over local (intra-partition) edges
   */
  auto local_connections() const {
    const auto my_id = specfem::MPI::get_rank();
    auto filter = [&my_id, this](const auto &edge) {
      return this->graph()[edge].neighbor_partition == my_id;
    };
    return boost::make_filtered_graph(this->graph(), filter);
  }

  /**
   * @brief Get a filtered view containing only cross-partition (MPI) edges
   *
   * Returns a Boost filtered_graph that includes only edges whose
   * neighbor_partition differs from the current MPI rank (i.e. the
   * target element belongs to a remote partition).
   *
   * @return Filtered graph view over cross-partition edges
   */
  auto mpi_connections() const {
    const auto my_id = specfem::MPI::get_rank();
    auto filter = [&my_id, this](const auto &edge) {
      return this->graph()[edge].neighbor_partition != my_id;
    };
    return boost::make_filtered_graph(this->graph(), filter);
  }

  /**
   * @brief Assert that the adjacency graph is symmetric
   *
   * Verifies that for every intra-partition directed edge from vertex A
   * to vertex B, there exists a corresponding edge from vertex B to
   * vertex A. Cross-partition edges are skipped because their reverse
   * edge resides in the remote partition's graph.
   *
   * @throws std::runtime_error if a local edge has no symmetric reverse
   */
  void assert_symmetry() const;
};

} // namespace specfem::mesh
