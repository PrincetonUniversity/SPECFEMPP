#pragma once
#include "enumerations/connections.hpp"
#include "enumerations/mesh_entities.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mesh/dim2/adjacency_graph/adjacency_graph.hpp"
#include <string>
#include <type_traits>

namespace specfem::testing::predicate {
/**
 * @brief Stores a single mesher-side connection with optional matching of
 * parameters. This is used to check if an adjacency map has the expected
 * connections.
 *
 */
struct connects {
public:
  int ispec;
  int jspec;
  specfem::mesh_entity::dim2::type ispec_mesh_entity;
  specfem::mesh_entity::dim2::type jspec_mesh_entity;
  specfem::connections::type connection_type;
  bool check_ispec_mesh_entity;
  bool check_jspec_mesh_entity;
  bool check_connection_type;

  /**
   * @brief Construct a new connects object that tests if a mesh connects two
   * elements, without verifying orientation or connection type.
   *
   * @param ispec - element 1
   * @param jspec - element 2
   */
  connects(const int &ispec, const int &jspec)
      : ispec(ispec), jspec(jspec), check_connection_type(false),
        check_ispec_mesh_entity(false), check_jspec_mesh_entity(false) {};

  /**
   * @brief Construct a new connects object that tests if a mesh connects two
   * elements, without verifying connection type, and only testing ispec's
   * orientation.
   *
   * @param ispec - element 1
   * @param ispec_mesh_entity - orientation on ispec
   * @param jspec - element 2
   */
  connects(const int &ispec,
           const specfem::mesh_entity::dim2::type &ispec_mesh_entity,
           const int &jspec)
      : ispec(ispec), jspec(jspec), check_connection_type(false),
        check_ispec_mesh_entity(true), ispec_mesh_entity(ispec_mesh_entity),
        check_jspec_mesh_entity(false) {};
  /**
   * @brief Construct a new connects object that tests if a mesh connects two
   * elements, without verifying connection type, and only testing jspec's
   * orientation.
   *
   * @param ispec - element 1
   * @param jspec - element 2
   * @param jspec_mesh_entity - orientation on jspec
   */
  connects(const int &ispec, const int &jspec,
           const specfem::mesh_entity::dim2::type &jspec_mesh_entity)
      : ispec(ispec), jspec(jspec), check_connection_type(false),
        check_ispec_mesh_entity(false), check_jspec_mesh_entity(true),
        jspec_mesh_entity(jspec_mesh_entity) {};
  /**
   * @brief Construct a new connects object that tests if a mesh connects two
   * elements, without verifying connection type.
   *
   * @param ispec - element 1
   * @param ispec_mesh_entity - orientation on ispec
   * @param jspec - element 2
   * @param jspec_mesh_entity - orientation on jspec
   */
  connects(const int &ispec,
           const specfem::mesh_entity::dim2::type &ispec_mesh_entity,
           const int &jspec,
           const specfem::mesh_entity::dim2::type &jspec_mesh_entity)
      : ispec(ispec), jspec(jspec), check_connection_type(false),
        check_ispec_mesh_entity(true), ispec_mesh_entity(ispec_mesh_entity),
        check_jspec_mesh_entity(false), jspec_mesh_entity(jspec_mesh_entity) {};

  /**
   * @brief Adds ispec's orientation to test
   *
   * @param ispec_mesh_entity - orientation to set
   * @return connects& - *this to be used in a builder pattern
   */
  connects &i(const specfem::mesh_entity::dim2::type &ispec_mesh_entity) {
    this->ispec_mesh_entity = ispec_mesh_entity;
    check_ispec_mesh_entity = true;
    return *this;
  }

  /**
   * @brief Adds jspec's orientation to test
   *
   * @param ispec_mesh_entity - orientation to set
   * @return connects& - *this to be used in a builder pattern
   */
  connects &j(const specfem::mesh_entity::dim2::type &jspec_mesh_entity) {
    this->jspec_mesh_entity = jspec_mesh_entity;
    check_jspec_mesh_entity = true;
    return *this;
  }

  /**
   * @brief Adds connection type to test
   *
   * @param connection_type - type to ensure
   * @return connects& - *this to be used in a builder pattern
   */
  connects &with(const specfem::connections::type &connection_type) {
    this->connection_type = connection_type;
    check_connection_type = true;
    return *this;
  }

  /**
   * @brief Runs the test on this object. specfem::testing::predicate::verify()
   * delegates to this method for `connects`. If the graph has the desired
   * connection, nothing is returned. Otherwise, FAIL()s.
   *
   * @tparam dimension - dimension of the adjacency graph
   * @param adjacency_graph - graph to verify has the connection
   */
  template <specfem::dimension::type dimension>
  void expect_in(
      const specfem::mesh::adjacency_graph<dimension> &adjacency_graph) const;

  std::string ispec_to_jspec_string() const;
  std::string jspec_to_ispec_string() const;
  std::string str() const;
};

/**
 * @brief To assert the number of edges from some ispec is equal to a known
 * value.
 *
 */
struct number_of_out_edges {
public:
  int ispec;
  int edge_count;

  /**
   * @brief Construct a test to ensure the element ispec has exactly edge_count
   * number of adjacencies. Only the out-edges are counted, which for a
   * symmetric graph, is equal to the number of adjacent elements.
   *
   * @param ispec - mesh index of the element
   * @param edge_count - number of adjacencies to expect
   */
  number_of_out_edges(const int &ispec, const int &edge_count)
      : ispec(ispec), edge_count(edge_count) {}

  /**
   * @brief Runs the test on this object. specfem::testing::predicate::verify()
   * delegates to this method for `number_of_out_edges`. If the graph has the
   * desired connection, nothing is returned. Otherwise, FAIL()s.
   *
   * @tparam dimension - dimension of the adjacency graph
   * @param adjacency_graph - graph to verify has the connections
   */
  template <specfem::dimension::type dimension>
  void expect_in(
      const specfem::mesh::adjacency_graph<dimension> &adjacency_graph) const;
};

using variant = std::variant<connects, number_of_out_edges>;

/**
 * @brief Ensures the adjacency graph satisfies the given predicate. FAIL()s if
 * not.
 *
 * @tparam dimension - dimension of the adjacency graph
 * @param predicate - test to apply to adjacency_graph
 * @param adjacency_graph - graph to check
 */
template <specfem::dimension::type dimension>
void verify(const variant &predicate,
            const specfem::mesh::adjacency_graph<dimension> &adjacency_graph);

} // namespace specfem::testing::predicate
