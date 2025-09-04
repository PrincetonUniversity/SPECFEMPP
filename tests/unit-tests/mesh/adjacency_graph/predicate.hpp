#pragma once
#include "enumerations/connections.hpp"
#include "enumerations/mesh_entities.hpp"
#include "mesh/dim2/adjacency_graph/adjacency_graph.hpp"
#include "mesh/mesh.hpp"
#include <string>
#include <type_traits>

namespace predicate {
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
  specfem::mesh_entity::type ispec_mesh_entity;
  specfem::mesh_entity::type jspec_mesh_entity;
  specfem::connections::type connection_type;
  bool check_ispec_mesh_entity;
  bool check_jspec_mesh_entity;
  bool check_connection_type;
  connects(const int &ispec, const int &jspec)
      : ispec(ispec), jspec(jspec), check_connection_type(false),
        check_ispec_mesh_entity(false), check_jspec_mesh_entity(false) {};
  connects(const int &ispec,
           const specfem::mesh_entity::type &ispec_mesh_entity,
           const int &jspec)
      : ispec(ispec), jspec(jspec), check_connection_type(false),
        check_ispec_mesh_entity(true), ispec_mesh_entity(ispec_mesh_entity),
        check_jspec_mesh_entity(false) {};
  connects(const int &ispec, const int &jspec,
           const specfem::mesh_entity::type &jspec_mesh_entity)
      : ispec(ispec), jspec(jspec), check_connection_type(false),
        check_ispec_mesh_entity(false), check_jspec_mesh_entity(true),
        jspec_mesh_entity(jspec_mesh_entity) {};
  connects(const int &ispec,
           const specfem::mesh_entity::type &ispec_mesh_entity,
           const int &jspec,
           const specfem::mesh_entity::type &jspec_mesh_entity)
      : ispec(ispec), jspec(jspec), check_connection_type(false),
        check_ispec_mesh_entity(true), ispec_mesh_entity(ispec_mesh_entity),
        check_jspec_mesh_entity(false), jspec_mesh_entity(jspec_mesh_entity) {};

  // builder pattern
  connects &i(const specfem::mesh_entity::type &ispec_mesh_entity) {
    this->ispec_mesh_entity = ispec_mesh_entity;
    check_ispec_mesh_entity = true;
    return *this;
  }
  connects &j(const specfem::mesh_entity::type &jspec_mesh_entity) {
    this->jspec_mesh_entity = jspec_mesh_entity;
    check_jspec_mesh_entity = true;
    return *this;
  }
  connects &with(const specfem::connections::type &connection_type) {
    this->connection_type = connection_type;
    check_connection_type = true;
    return *this;
  }

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
  number_of_out_edges(const int &ispec, const int &edge_count)
      : ispec(ispec), edge_count(edge_count) {}

  template <specfem::dimension::type dimension>
  void expect_in(
      const specfem::mesh::adjacency_graph<dimension> &adjacency_graph) const;
};

using variant = std::variant<connects, number_of_out_edges>;

template <specfem::dimension::type dimension>
void verify(const variant &predicate,
            const specfem::mesh::adjacency_graph<dimension> &adjacency_graph);

} // namespace predicate
