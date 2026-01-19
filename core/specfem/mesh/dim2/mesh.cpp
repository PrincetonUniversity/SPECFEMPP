#include "specfem/mesh.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "medium/material.hpp"

#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <limits>
#include <list>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

// Mesher uses a indexing that looks like following for the corners of a 2D:
// 4----3
// |    |
// |    |
// 1----2
// We need to convert the corners from specfem::mesh_entity::dim2::type to
// mesher indexing
template <typename T>
std::vector<int> convert_corners_to_mesher_index(const T corners) {

  std::vector<int> return_value;
  std::transform(corners.begin(), corners.end(),
                 std::back_inserter(return_value), [](const auto &corner) {
                   if (!specfem::mesh_entity::contains(
                           specfem::mesh_entity::dim2::corners, corner)) {
                     throw std::runtime_error("The argument is not a corner");
                   }

                   return static_cast<int>(corner) -
                          5; // Convert to mesher indexing
                 });
  return return_value;
}

bool check_nodes_on_domain_edge(const std::list<int> &nodes_on_domain_edge,
                                const std::set<int> &control_nodes) {
  // Check if all control nodes are on the domain edge
  for (const auto &node : control_nodes) {
    if (std::find(nodes_on_domain_edge.begin(), nodes_on_domain_edge.end(),
                  node) == nodes_on_domain_edge.end()) {
      return false;
    }
  }
  return true;
}

void check_adjacency_graph(
    const specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> &graph,
    const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
        &control_nodes) {
  // Get xmin and xmax from the control nodes
  type_real xmin = std::numeric_limits<type_real>::max();
  type_real xmax = std::numeric_limits<type_real>::lowest();
  type_real zmin = std::numeric_limits<type_real>::max();
  type_real zmax = std::numeric_limits<type_real>::lowest();

  for (int i = 0; i < control_nodes.ngnod; ++i) {
    xmin = std::min(xmin, control_nodes.coord(0, i));
    xmax = std::max(xmax, control_nodes.coord(0, i));
    zmin = std::min(zmin, control_nodes.coord(1, i));
    zmax = std::max(zmax, control_nodes.coord(1, i));
  }

  std::list<int> nodes_on_domain_edge;
  for (int i = 0; i < control_nodes.ngnod; ++i) {
    if (std::abs(control_nodes.coord(0, i) - xmin) < 1e-6 ||
        std::abs(control_nodes.coord(0, i) - xmax) < 1e-6 ||
        std::abs(control_nodes.coord(1, i) - zmin) < 1e-6 ||
        std::abs(control_nodes.coord(1, i) - zmax) < 1e-6) {
      nodes_on_domain_edge.push_back(i);
    }
  }

  const auto tolerance = 1e-6 * std::min(xmax - xmin, zmax - zmin);

  const auto &g = graph.graph();
  // Filter out strongly conforming connections
  auto filter = [&g](const auto &edge) {
    return g[edge].connection ==
           specfem::connections::type::strongly_conforming;
  };

  // Create a filtered graph view
  const auto fg = boost::make_filtered_graph(g, filter);

  for (const auto &edge : boost::make_iterator_range(boost::edges(fg))) {
    const int ispec_mesh = boost::source(edge, fg);
    const int jspec_mesh = boost::target(edge, fg);

    const auto iedge = g[edge].orientation;

    if (specfem::mesh_entity::contains(specfem::mesh_entity::dim2::edges,
                                       iedge)) {
      const auto returned_edge = boost::edge(jspec_mesh, ispec_mesh, fg).first;
      const auto jedge = g[returned_edge].orientation;

      const auto corners1 = convert_corners_to_mesher_index(
          specfem::mesh_entity::corners_of_edge(iedge));
      const auto corners2 = convert_corners_to_mesher_index(
          specfem::mesh_entity::corners_of_edge(jedge));

      std::set<int> control_nodes1 = {
        control_nodes.knods(corners1[0], ispec_mesh),
        control_nodes.knods(corners1[1], ispec_mesh)
      };

      std::set<int> control_nodes2 = {
        control_nodes.knods(corners2[0], jspec_mesh),
        control_nodes.knods(corners2[1], jspec_mesh)
      };

      if (check_nodes_on_domain_edge(nodes_on_domain_edge, control_nodes1) &&
          check_nodes_on_domain_edge(nodes_on_domain_edge, control_nodes2)) {
        // If both edges are on the domain edge, we can skip the check
        // Periodic boundaries may have strongly conforming edges that do not
        // match. But both the edges will be on the domain boundary.
        continue;
      }

      // Make sure that the two edges share the same two corners
      if (control_nodes1 != control_nodes2) {

        std::ostringstream oss;
        oss << "Error: Edges do not share the same corners between elements "
            << ispec_mesh << " and " << jspec_mesh
            << ". Edges: " << static_cast<int>(iedge) << " and "
            << static_cast<int>(jedge);

        throw std::runtime_error(oss.str());
      }
    } else { // the connection is a corner
      const auto icorner = iedge;
      const auto returned_edge = boost::edge(jspec_mesh, ispec_mesh, fg).first;
      const auto jcorner = g[returned_edge].orientation;
      const auto corners1 = convert_corners_to_mesher_index(
          std::list<specfem::mesh_entity::dim2::type>{ icorner });
      const auto corners2 = convert_corners_to_mesher_index(
          std::list<specfem::mesh_entity::dim2::type>{ jcorner });

      std::set<int> control_nodes1 = { control_nodes.knods(corners1[0],
                                                           ispec_mesh) };
      std::set<int> control_nodes2 = { control_nodes.knods(corners2[0],
                                                           jspec_mesh) };

      if (check_nodes_on_domain_edge(nodes_on_domain_edge, control_nodes1) &&
          check_nodes_on_domain_edge(nodes_on_domain_edge, control_nodes2)) {
        // If both corners are on the domain edge, we can skip the check
        continue;
      }

      // Make sure that the two corners are the same
      if (control_nodes1 != control_nodes2) {

        std::ostringstream oss;
        oss << "Error: Corners do not match between elements " << ispec_mesh
            << " and " << jspec_mesh
            << ". Corners: " << static_cast<int>(icorner) << " and "
            << static_cast<int>(jcorner);

        throw std::runtime_error(oss.str());
      }
    }
  }
}

std::string specfem::mesh::mesh<specfem::dimension::type::dim2>::print() const {

  int n_elastic;
  int n_acoustic;

  Kokkos::parallel_reduce(
      "specfem::mesh::mesh::print", specfem::kokkos::HostRange(0, this->nspec),
      KOKKOS_CLASS_LAMBDA(const int ispec, int &n_elastic, int &n_acoustic) {
        if (this->materials.material_index_mapping(ispec).type ==
            specfem::element::medium_tag::elastic_psv) {
          n_elastic++;
        } else if (this->materials.material_index_mapping(ispec).type ==
                   specfem::element::medium_tag::acoustic) {
          n_acoustic++;
        }
      },
      n_elastic, n_acoustic);

  std::ostringstream message;

  message
      << "Spectral element information:\n"
      << "------------------------------\n"
      << "Total number of spectral elements : " << this->nspec << "\n"
      << "Total number of spectral elements assigned to elastic material : "
      << n_elastic << "\n"
      << "Total number of spectral elements assigned to acoustic material : "
      << n_acoustic << "\n"
      << "Total number of geometric points : " << this->npgeo << "\n";

  return message.str();
}

void specfem::mesh::mesh<specfem::dimension::type::dim2>::check_consistency()
    const {
  if (this->npgeo <= 0) {
    throw std::runtime_error(
        "Number of geometric points must be greater than 0.");
  }

  if (this->nspec <= 0) {
    throw std::runtime_error(
        "Number of spectral elements must be greater than 0.");
  }

  check_adjacency_graph(this->adjacency_graph, this->control_nodes);
}

void specfem::mesh::mesh<specfem::dimension::type::dim2>::
    setup_coupled_interfaces(
        const std::set<std::pair<int, int> > &coupled_interfaces) {
  auto &graph = this->adjacency_graph.graph();
  auto &materials = this->materials;

  int n_matched = 0;
  for (const auto v : boost::make_iterator_range(boost::vertices(graph))) {
    for (const auto e :
         boost::make_iterator_range(boost::out_edges(v, graph))) {
      const auto target = boost::target(e, graph);
      auto &edge_props = graph[e];

      const auto [self_medium, self_property] = materials.get_material_type(v);
      const auto [neighbor_medium, neighbor_property] =
          materials.get_material_type(target);
      if ((self_medium != neighbor_medium) &&
          (edge_props.connection ==
           specfem::connections::type::strongly_conforming)) {
        // Change strongly conforming to weakly conforming if media differ
        edge_props.connection = specfem::connections::type::weakly_conforming;
        if (coupled_interfaces.count(std::make_pair(v, target))) {
          n_matched++;
        }
      }
    }
  }

  if (n_matched != static_cast<int>(coupled_interfaces.size())) {
    std::cout << "Number of coupled interfaces specified: "
              << coupled_interfaces.size() << "\n";
    std::cout << "Number of coupled interfaces matched in the mesh: "
              << n_matched << "\n";
    throw std::runtime_error(
        "Not all coupled interfaces were matched in the mesh adjacency "
        "graph.");
  }

  this->adjacency_graph.assert_symmetry();

  return;
}
