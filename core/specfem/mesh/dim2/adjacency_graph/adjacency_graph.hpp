#pragma once

#include "specfem/element_coupling.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh_entity.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <cstddef>
#include <memory>
#include <vector>

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
    EdgeProperties(const specfem::element_connections::type conn,
                   const specfem::mesh_entity::dim2::type orient)
        : connection(conn), orientation(orient) {}
  };

  /**
   * @brief Properties for cross-partition (MPI) adjacency edges
   *
   * Extends EdgeProperties with partition and remote-index information
   * for edges that cross MPI partition boundaries. These edges are stored
   * separately from the Boost graph (in a flat vector) because the
   * neighbor element belongs to a different partition.
   */
  struct MPIEdgeProperties : public EdgeProperties {
    int neighbor_partition; ///< MPI rank of the neighboring partition
    specfem::mesh_entity::dim2::type neighbor_orientation; ///< Orientation of
                                                           ///< the shared
                                                           ///< entity in the
                                                           ///< neighboring
                                                           ///< partition
    size_t neighbor_local_index; ///< Local index of the neighboring element in
                                 ///< its partition
    size_t local_index; ///< Local index of the element in this partition
    specfem::mesh_entity::dim2::type local_anchor;    ///< Anchor point on the
                                                      ///< local element
    specfem::mesh_entity::dim2::type neighbor_anchor; ///< Anchor point on the
                                                      ///< neighboring element

    MPIEdgeProperties() = default;

    MPIEdgeProperties(
        const specfem::element_connections::type conn,
        const specfem::mesh_entity::dim2::type orient, const size_t local_idx,
        const size_t neighbor_index, const int neighbor_part,
        const specfem::mesh_entity::dim2::type neighbor_orient,
        const specfem::mesh_entity::dim2::type local_anchor_idx,
        const specfem::mesh_entity::dim2::type neighbor_anchor_idx)
        : EdgeProperties(conn, orient), local_index(local_idx),
          neighbor_local_index(neighbor_index),
          neighbor_partition(neighbor_part),
          neighbor_orientation(neighbor_orient), local_anchor(local_anchor_idx),
          neighbor_anchor(neighbor_anchor_idx) {}
  }
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
using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
                                    boost::no_property, EdgeProperties>;

/** @brief The underlying Boost graph storing adjacency relationships */
std::shared_ptr<Graph> p_graph_;
/** @brief Cross-partition edges stored outside the Boost graph */
std::vector<MPIEdgeProperties> mpi_edge_properties_;

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
 * @brief Get mutable reference to the local (intra-partition) graph
 *
 * Provides direct access to the Boost graph containing only edges
 * where both source and target elements belong to this partition.
 *
 * @return Mutable reference to the Boost adjacency_list graph
 */
Graph &local_connections() { return *p_graph_; }

/**
 * @brief Get const reference to the local (intra-partition) graph
 *
 * Provides read-only access to the Boost graph containing only edges
 * where both source and target elements belong to this partition.
 *
 * @return Const reference to the Boost adjacency_list graph
 */
const Graph &local_connections() const { return *p_graph_; }

/**
 * @brief Get mutable reference to cross-partition (MPI) edge list
 *
 * Returns the vector of MPIEdgeProperties describing edges that cross
 * partition boundaries. These edges are not stored in the Boost graph.
 *
 * @return Mutable reference to the MPI edge properties vector
 */
auto &mpi_connections() { return mpi_edge_properties_; }

/**
 * @brief Get const reference to cross-partition (MPI) edge list
 *
 * @return Const reference to the MPI edge properties vector
 */
const auto &mpi_connections() const { return mpi_edge_properties_; }

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
