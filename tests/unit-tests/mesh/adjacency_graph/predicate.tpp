#pragma once

#include "enumerations/connections.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/mesh_entities.hpp"
#include "predicate.hpp"
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>

namespace predicate {
std::string connects::ispec_to_jspec_string() const {
  std::stringstream str;
  str << "[" << ispec << " -> " << jspec << "]";
  return str.str();
}
std::string connects::jspec_to_ispec_string() const {
  std::stringstream str;
  str << "[" << jspec << " -> " << ispec << "]";
  return str.str();
}
std::string connects::str() const {
  std::stringstream str;
  str << "[" << ispec;
  if (check_ispec_mesh_entity) {
    str << " (" << specfem::mesh_entity::to_string(ispec_mesh_entity) << ")";
  }
  str << " <-";
  if (check_connection_type) {
    str << specfem::connections::to_string(connection_type);
  }
  str << "-> " << jspec;
  if (check_jspec_mesh_entity) {
    str << " (" << specfem::mesh_entity::to_string(jspec_mesh_entity) << ")";
  }
  str << "]";
  return str.str();
}

template <specfem::dimension::type dimension>
void connects::expect_in(
    const specfem::mesh::adjacency_graph<dimension> &adjacency_graph) const {
  const auto &g = adjacency_graph.graph();
  { // test ispec -> jspec edge
    const auto [edge_, exists] = boost::edge(ispec, jspec, g);
    const auto edge = g[edge_];
    if (!exists) {
      std::ostringstream msg;
      msg << "Failed expected adjacency " << str() << ":\n";
      msg << "  Adjacency graph did not contain " << ispec_to_jspec_string()
          << "\n";
      FAIL() << msg.str();
    }
    if (check_connection_type && (edge.connection != connection_type)) {
      std::ostringstream msg;
      msg << "Failed expected adjacency " << str() << ":\n";
      msg << "  Found connection type "
          << specfem::connections::to_string(edge.connection) << " for edge "
          << ispec_to_jspec_string() << "\n";
      FAIL() << msg.str();
    }
    if (check_ispec_mesh_entity && (edge.orientation != ispec_mesh_entity)) {
      std::ostringstream msg;
      msg << "Failed expected adjacency " << str() << ":\n";
      msg << "  Found ispec orientation "
          << specfem::mesh_entity::to_string(edge.orientation) << " for edge "
          << ispec_to_jspec_string() << "\n";
      FAIL() << msg.str();
    }
  }
  { // test jspec -> ispec edge
    const auto [edge_, exists] = boost::edge(ispec, jspec, g);
    const auto edge = g[edge_];
    if (!exists) {
      std::ostringstream msg;
      msg << "Failed expected adjacency " << str() << ":\n";
      msg << "  Adjacency graph did not contain " << jspec_to_ispec_string()
          << "\n";
      FAIL() << msg.str();
    }
    if (check_connection_type && (edge.connection != connection_type)) {
      std::ostringstream msg;
      msg << "Failed expected adjacency " << str() << ":\n";
      msg << "  Found connection type "
          << specfem::connections::to_string(edge.connection) << " for edge "
          << jspec_to_ispec_string() << "\n";
      FAIL() << msg.str();
    }
    if (check_jspec_mesh_entity && (edge.orientation != jspec_mesh_entity)) {
      std::ostringstream msg;
      msg << "Failed expected adjacency " << str() << ":\n";
      msg << "  Found jspec orientation "
          << specfem::mesh_entity::to_string(edge.orientation) << " for edge "
          << jspec_to_ispec_string() << "\n";
      FAIL() << msg.str();
    }
  }
}
template <specfem::dimension::type dimension>
void number_of_out_edges::expect_in(
    const specfem::mesh::adjacency_graph<dimension> &adjacency_graph) const {

  const auto &g = adjacency_graph.graph();

  const auto out_edges = boost::out_edges(ispec, g);
  std::vector<int> computed_neighbors;
  for (auto edge_it = out_edges.first; edge_it != out_edges.second; ++edge_it) {
    const int neighbor = boost::target(*edge_it, g);
    computed_neighbors.push_back(neighbor);
  }

  if (computed_neighbors.size() != edge_count) {
    std::ostringstream msg;
    msg << "Length mismatch for node " << ispec << ":\n"
        << "Expected length: " << edge_count << "\n"
        << "Actual length: " << computed_neighbors.size() << "\n";
    FAIL() << msg.str();
  }
}

template <specfem::dimension::type dimension>
void verify(const variant &predicate,
            const specfem::mesh::adjacency_graph<dimension> &adjacency_graph) {
  std::visit(
      [&adjacency_graph](auto &&predicate) {
        using PredicateType = std::decay_t<decltype(predicate)>;
        if constexpr (std::is_same_v<PredicateType, connects> ||
                      std::is_same_v<PredicateType, number_of_out_edges>) {
          predicate.expect_in(adjacency_graph);
        }
      },
      predicate);
}

} // namespace predicate
