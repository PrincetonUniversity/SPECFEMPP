#include "specfem/mesh/dim2/adjacency_graph/adjacency_graph.hpp"
#include "enumerations/interface.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {

/**
 * @brief Helper function to add a bidirectional edge between two elements
 */
void add_bidirectional_edge(
    specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> &graph,
    int source, int target, specfem::connections::type conn_type,
    specfem::mesh_entity::dim2::type orientation1,
    specfem::mesh_entity::dim2::type orientation2) {
  auto &g = graph.graph();
  boost::add_edge(
      source, target,
      specfem::mesh::adjacency_graph<
          specfem::dimension::type::dim2>::EdgeProperties(conn_type,
                                                          orientation1),
      g);
  boost::add_edge(
      target, source,
      specfem::mesh::adjacency_graph<
          specfem::dimension::type::dim2>::EdgeProperties(conn_type,
                                                          orientation2),
      g);
}

/**
 * @brief Helper function to add a unidirectional edge between two elements
 */
void add_unidirectional_edge(
    specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> &graph,
    int source, int target, specfem::connections::type conn_type,
    specfem::mesh_entity::dim2::type orientation) {
  auto &g = graph.graph();
  boost::add_edge(
      source, target,
      specfem::mesh::adjacency_graph<
          specfem::dimension::type::dim2>::EdgeProperties(conn_type,
                                                          orientation),
      g);
}
} // namespace

// ===== Constructor Tests =====

/**
 * @brief Test default constructor
 */
TEST(AdjacencyGraphTest, DefaultConstructor) {
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph;

  EXPECT_EQ(boost::num_vertices(graph.graph()), 0);
}

/**
 * @brief Test constructor with specified number of elements
 */
TEST(AdjacencyGraphTest, ConstructorWithElements) {
  const int nspec = 10;
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph(nspec);

  EXPECT_EQ(boost::num_vertices(graph.graph()), nspec);
  EXPECT_EQ(boost::num_edges(graph.graph()), 0);
}

/**
 * @brief Test constructor with zero elements
 */
TEST(AdjacencyGraphTest, ConstructorWithZeroElements) {
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph(0);

  EXPECT_EQ(boost::num_vertices(graph.graph()), 0);
  EXPECT_EQ(boost::num_edges(graph.graph()), 0);
}

// ===== Graph Access Tests =====

/**
 * @brief Test mutable graph access
 */
TEST(AdjacencyGraphTest, MutableGraphAccess) {
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph(3);

  auto &g = graph.graph();
  EXPECT_EQ(boost::num_vertices(g), 3);

  // Add an edge using mutable reference
  boost::add_edge(
      0, 1,
      specfem::mesh::adjacency_graph<specfem::dimension::type::dim2>::
          EdgeProperties(specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::right),
      g);

  EXPECT_EQ(boost::num_edges(g), 1);
}

/**
 * @brief Test const graph access
 */
TEST(AdjacencyGraphTest, ConstGraphAccess) {
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph(3);
  add_bidirectional_edge(graph, 0, 1,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::right,
                         specfem::mesh_entity::dim2::type::left);

  const auto &const_graph = graph;
  const auto &g = const_graph.graph();

  EXPECT_EQ(boost::num_vertices(g), 3);
  EXPECT_EQ(boost::num_edges(g), 2);
}

// ===== EdgeProperties Tests =====

/**
 * @brief Test EdgeProperties default constructor
 */
TEST(AdjacencyGraphTest, EdgePropertiesDefaultConstructor) {
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2>::EdgeProperties
      props;

  // Default values should be initialized (exact values depend on enum defaults)
  // We mainly test that the constructor doesn't crash
  SUCCEED();
}

/**
 * @brief Test EdgeProperties parameterized constructor
 */
TEST(AdjacencyGraphTest, EdgePropertiesParameterizedConstructor) {
  auto conn_type = specfem::connections::type::strongly_conforming;
  auto orientation = specfem::mesh_entity::dim2::type::top;

  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2>::EdgeProperties
      props(conn_type, orientation);

  EXPECT_EQ(props.connection, conn_type);
  EXPECT_EQ(props.orientation, orientation);
}

/**
 * @brief Test EdgeProperties with all mesh entity types
 */
TEST(AdjacencyGraphTest, EdgePropertiesAllOrientations) {
  auto conn_type = specfem::connections::type::strongly_conforming;

  std::vector<specfem::mesh_entity::dim2::type> orientations = {
    specfem::mesh_entity::dim2::type::bottom,
    specfem::mesh_entity::dim2::type::right,
    specfem::mesh_entity::dim2::type::top,
    specfem::mesh_entity::dim2::type::left,
    specfem::mesh_entity::dim2::type::bottom_left,
    specfem::mesh_entity::dim2::type::bottom_right,
    specfem::mesh_entity::dim2::type::top_right,
    specfem::mesh_entity::dim2::type::top_left
  };

  for (const auto &orientation : orientations) {
    specfem::mesh::adjacency_graph<
        specfem::dimension::type::dim2>::EdgeProperties props(conn_type,
                                                              orientation);
    EXPECT_EQ(props.connection, conn_type);
    EXPECT_EQ(props.orientation, orientation);
  }
}

// ===== Symmetry Tests =====

/**
 * @brief Test assert_symmetry() with symmetric graph
 */
TEST(AdjacencyGraphTest, AssertSymmetryWithSymmetricGraph) {
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph(4);

  // Add symmetric edges
  add_bidirectional_edge(graph, 0, 1,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::right,
                         specfem::mesh_entity::dim2::type::left);

  add_bidirectional_edge(graph, 1, 2,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::bottom,
                         specfem::mesh_entity::dim2::type::top);

  add_bidirectional_edge(graph, 2, 3,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::right,
                         specfem::mesh_entity::dim2::type::left);

  // Should not throw
  EXPECT_NO_THROW(graph.assert_symmetry());
}

/**
 * @brief Test assert_symmetry() with asymmetric graph
 */
TEST(AdjacencyGraphTest, AssertSymmetryWithAsymmetricGraph) {
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph(3);

  // Add only one direction of the edge
  add_unidirectional_edge(graph, 0, 1,
                          specfem::connections::type::strongly_conforming,
                          specfem::mesh_entity::dim2::type::right);

  // Should throw std::runtime_error
  EXPECT_THROW(graph.assert_symmetry(), std::runtime_error);
}

/**
 * @brief Test assert_symmetry() with empty graph
 */
TEST(AdjacencyGraphTest, AssertSymmetryWithEmptyGraph) {
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> empty_graph;

  // Empty graph should be considered symmetric
  EXPECT_NO_THROW(empty_graph.assert_symmetry());
}

/**
 * @brief Test assert_symmetry() with graph with no edges
 */
TEST(AdjacencyGraphTest, AssertSymmetryWithNoEdges) {
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph(5);

  // Graph with vertices but no edges should be symmetric
  EXPECT_NO_THROW(graph.assert_symmetry());
}

/**
 * @brief Test assert_symmetry() with self-loops
 */
TEST(AdjacencyGraphTest, AssertSymmetryWithSelfLoops) {
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph(2);

  // Add self-loop (should be symmetric by definition)
  add_unidirectional_edge(graph, 0, 0,
                          specfem::connections::type::strongly_conforming,
                          specfem::mesh_entity::dim2::type::bottom);

  // Self-loops are inherently symmetric
  EXPECT_NO_THROW(graph.assert_symmetry());
}

/**
 * @brief Test assert_symmetry() with partially symmetric graph
 */
TEST(AdjacencyGraphTest, AssertSymmetryWithPartiallySymmetricGraph) {
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph(4);

  // Add some symmetric edges
  add_bidirectional_edge(graph, 0, 1,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::right,
                         specfem::mesh_entity::dim2::type::left);

  // Add an asymmetric edge
  add_unidirectional_edge(graph, 2, 3,
                          specfem::connections::type::strongly_conforming,
                          specfem::mesh_entity::dim2::type::top);

  // Should throw because of the asymmetric edge
  EXPECT_THROW(graph.assert_symmetry(), std::runtime_error);
}

// ===== Complex Graph Tests =====

/**
 * @brief Test with complex symmetric graph
 */
TEST(AdjacencyGraphTest, ComplexSymmetricGraph) {
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph(6);

  // Create a more complex symmetric graph structure
  add_bidirectional_edge(graph, 0, 1,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::right,
                         specfem::mesh_entity::dim2::type::left);
  add_bidirectional_edge(graph, 1, 2,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::bottom,
                         specfem::mesh_entity::dim2::type::top);
  add_bidirectional_edge(graph, 2, 3,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::right,
                         specfem::mesh_entity::dim2::type::left);
  add_bidirectional_edge(graph, 3, 4,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::bottom,
                         specfem::mesh_entity::dim2::type::top);
  add_bidirectional_edge(graph, 4, 5,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::right,
                         specfem::mesh_entity::dim2::type::left);
  add_bidirectional_edge(graph, 0, 3,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::bottom,
                         specfem::mesh_entity::dim2::type::top);

  EXPECT_EQ(boost::num_vertices(graph.graph()), 6);
  EXPECT_EQ(boost::num_edges(graph.graph()),
            12); // 6 bidirectional edges = 12 directed edges
  EXPECT_NO_THROW(graph.assert_symmetry());
}

/**
 * @brief Test graph edge iteration
 */
TEST(AdjacencyGraphTest, EdgeIteration) {
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph(3);

  add_bidirectional_edge(graph, 0, 1,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::right,
                         specfem::mesh_entity::dim2::type::left);

  const auto &g = graph.graph();
  int edge_count = 0;

  for (auto edge_it = boost::edges(g); edge_it.first != edge_it.second;
       ++edge_it.first) {
    const auto &edge = *edge_it.first;
    const auto source = boost::source(edge, g);
    const auto target = boost::target(edge, g);
    const auto &props = g[edge];

    EXPECT_TRUE(source < 3);
    EXPECT_TRUE(target < 3);
    EXPECT_EQ(props.connection,
              specfem::connections::type::strongly_conforming);

    edge_count++;
  }

  EXPECT_EQ(edge_count,
            2); // Two directed edges for one bidirectional connection
}

/**
 * @brief Test vertex degree checking
 */
TEST(AdjacencyGraphTest, VertexDegreeChecking) {
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph(4);

  // Add edges to create specific degree patterns
  add_bidirectional_edge(graph, 0, 1,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::right,
                         specfem::mesh_entity::dim2::type::left);
  add_bidirectional_edge(graph, 0, 2,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::bottom,
                         specfem::mesh_entity::dim2::type::top);
  add_bidirectional_edge(graph, 1, 3,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::bottom,
                         specfem::mesh_entity::dim2::type::top);

  const auto &g = graph.graph();

  // Check out-degrees
  EXPECT_EQ(boost::out_degree(0, g), 2); // Connected to 1 and 2
  EXPECT_EQ(boost::out_degree(1, g), 2); // Connected to 0 and 3
  EXPECT_EQ(boost::out_degree(2, g), 1); // Connected to 0
  EXPECT_EQ(boost::out_degree(3, g), 1); // Connected to 1
}

// ===== Edge Case Tests =====

/**
 * @brief Test with maximum reasonable graph size
 */
TEST(AdjacencyGraphTest, LargeGraph) {
  const int large_size = 1000;
  specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> graph(
      large_size);

  EXPECT_EQ(boost::num_vertices(graph.graph()), large_size);

  // Add a few edges to test functionality with large graphs
  add_bidirectional_edge(graph, 0, 1,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::right,
                         specfem::mesh_entity::dim2::type::left);
  add_bidirectional_edge(graph, large_size - 2, large_size - 1,
                         specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim2::type::bottom,
                         specfem::mesh_entity::dim2::type::top);

  EXPECT_NO_THROW(graph.assert_symmetry());
}
