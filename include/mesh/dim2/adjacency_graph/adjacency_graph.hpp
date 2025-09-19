#pragma once

#include "enumerations/interface.hpp"
#include <boost/graph/adjacency_list.hpp>

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
template <specfem::dimension::type Dimension> struct adjacency_graph {

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
    specfem::connections::type connection;

    /** @brief Orientation of the shared mesh entity (left, right, ...,
     * bottom_left, bottom_right, etc.) */
    specfem::mesh_entity::type orientation;

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
     */
    EdgeProperties(const specfem::connections::type conn,
                   const specfem::mesh_entity::type orient)
        : connection(conn), orientation(orient) {}
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
  Graph graph_;

public:
  /**
   * @brief Default constructor
   *
   * Creates an empty adjacency graph with no vertices or edges.
   */
  adjacency_graph() = default;

  /**
   * @brief Constructor with specified number of spectral elements
   *
   * Creates an adjacency graph with the specified number of vertices
   * (spectral elements) but no edges initially.
   *
   * @param nspec Number of spectral elements in the mesh
   */
  adjacency_graph(const int nspec) : graph_(nspec) {}

  /**
   * @brief Get mutable reference to the underlying graph
   *
   * Provides direct access to the Boost graph structure for modification
   * operations such as adding edges or vertices.
   *
   * @return Mutable reference to the Boost adjacency_list graph
   */
  Graph &graph() { return graph_; }

  /**
   * @brief Get const reference to the underlying graph
   *
   * Provides read-only access to the Boost graph structure for
   * query operations such as traversing edges or checking connectivity.
   *
   * @return Const reference to the Boost adjacency_list graph
   */
  const Graph &graph() const { return graph_; }

  // TODO(Rohit: ADJ_GRAPH_DEFAULT)
  // Graph should never be empty after it is made as default
  /**
   * @brief Check if the adjacency graph is empty
   *
   * Determines whether the graph contains any vertices. An empty graph
   * has no spectral elements and therefore no adjacency relationships.
   *
   * @return true if the graph has no vertices, false otherwise
   */
  bool empty() const { return boost::num_vertices(graph_) == 0; }

  /**
   * @brief Assert that the adjacency graph is symmetric
   *
   * Verifies that for every directed edge from vertex A to vertex B,
   * there exists a corresponding edge from vertex B to vertex A.
   * This is required for proper mesh connectivity in spectral element methods.
   *
   * @throws std::runtime_error if the graph is not symmetric
   */
  void assert_symmetry() const;
};

} // namespace specfem::mesh
