
#include "compute_intersection.hpp"
#include "enumerations/mesh_entities.hpp"
#include "specfem_setup.hpp"
#include <sstream>
#include <stdexcept>

inline std::pair<
    specfem::point::global_coordinates<specfem::dimension::type::dim2>,
    specfem::point::global_coordinates<specfem::dimension::type::dim2> >
edge_extents(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const int &ispec, const specfem::mesh_entity::type &side) {
  specfem::point::global_coordinates<specfem::dimension::type::dim2> p1, p2;
  switch (side) {
  case specfem::mesh_entity::type::bottom:
    // control nodes 0 and 1
    return { { mesh.h_control_node_coord(0, ispec, 0),
               mesh.h_control_node_coord(1, ispec, 0) },
             { mesh.h_control_node_coord(0, ispec, 1),
               mesh.h_control_node_coord(1, ispec, 1) } };
  case specfem::mesh_entity::type::right:
    // control nodes 1 and 2
    return { { mesh.h_control_node_coord(0, ispec, 1),
               mesh.h_control_node_coord(1, ispec, 1) },
             { mesh.h_control_node_coord(0, ispec, 2),
               mesh.h_control_node_coord(1, ispec, 2) } };
  case specfem::mesh_entity::type::top:
    // control nodes 3 and 2 (order for increasing xi)
    return { { mesh.h_control_node_coord(0, ispec, 3),
               mesh.h_control_node_coord(1, ispec, 3) },
             { mesh.h_control_node_coord(0, ispec, 2),
               mesh.h_control_node_coord(1, ispec, 2) } };
  case specfem::mesh_entity::type::left:
    // control nodes 0 and 3 (order for increasing gamma)
    return { { mesh.h_control_node_coord(0, ispec, 0),
               mesh.h_control_node_coord(1, ispec, 0) },
             { mesh.h_control_node_coord(0, ispec, 3),
               mesh.h_control_node_coord(1, ispec, 3) } };
  default:
    throw std::runtime_error(
        "compute_intersection was given a corner, not an edge.");
  }
}

std::vector<std::pair<type_real, type_real> >
specfem::assembly::nonconforming_interfaces::compute_intersection(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const boost::graph_traits<specfem::mesh::adjacency_graph<
        specfem::dimension::type::dim2>::GraphType>::edge_descriptor &edge,
    const specfem::quadrature::quadrature &mortar_quadrature) {
  const int nquad = mortar_quadrature.get_N();

  std::vector<std::pair<type_real, type_real> > intersections(nquad);

  const auto &graph = mesh.graph();

  // i is source, j is target
  const int ispec = boost::source(edge, graph);
  const int jspec = boost::target(edge, graph);

  const specfem::mesh_entity::type iorientation = graph[edge].orientation;
  const auto [edge_inv, exists] = boost::edge(jspec, ispec, graph);
  if (!exists) {
    throw std::runtime_error(
        "Non-symmetric adjacency graph detected in `compute_intersection`.");
  }
  const specfem::mesh_entity::type jorientation = graph[edge_inv].orientation;

  // endpoints of edge on either element
  const auto [p1i, p2i] = edge_extents(mesh, ispec, iorientation);
  const auto [p1j, p2j] = edge_extents(mesh, jspec, jorientation);

  // recover local coordinates on opposite side
  const auto [p1j_in_i, p1j_inside_i] =
      specfem::algorithms::locate_point_on_edge(p1j, mesh, ispec, iorientation);
  const auto [p2j_in_i, p2j_inside_i] =
      specfem::algorithms::locate_point_on_edge(p2j, mesh, ispec, iorientation);
  const auto [p1i_in_j, p1i_inside_j] =
      specfem::algorithms::locate_point_on_edge(p1i, mesh, jspec, jorientation);
  const auto [p2i_in_j, p2i_inside_j] =
      specfem::algorithms::locate_point_on_edge(p2i, mesh, jspec, jorientation);

  // recover the bounds of the intersection in local coordinates:
  // locate_point_on_edge returns values within [-1,1], so clamping is not
  // necessary.
  type_real j_to_i_lo = std::min(p1j_in_i, p2j_in_i);
  type_real j_to_i_hi = std::max(p1j_in_i, p2j_in_i);
  type_real i_to_j_lo = std::min(p1i_in_j, p2i_in_j);
  type_real i_to_j_hi = std::max(p1i_in_j, p2i_in_j);

  if (j_to_i_lo >= j_to_i_hi || i_to_j_lo >= i_to_j_hi) {
    // no intersection
    std::ostringstream oss;

    oss << "When computing intersections between " << ispec << " ("
        << specfem::mesh_entity::to_string(iorientation) << ") and " << jspec
        << " (" << specfem::mesh_entity::to_string(jorientation)
        << "), no intersection was found, despite the adjacency map declaring "
           "such an adjacency.\n"
        << "\n"
        << "Edge " << jspec << " ("
        << specfem::mesh_entity::to_string(jorientation)
        << ") spans edge coordinates [" << j_to_i_lo << "," << j_to_i_hi
        << "] on element " << ispec << " and edge " << ispec << " ("
        << specfem::mesh_entity::to_string(iorientation)
        << ") spans edge coordinates [" << i_to_j_lo << "," << i_to_j_hi
        << "] on element " << jspec << ".\n";

    throw std::runtime_error(oss.str());
  }

  /* we may get better accuracy if we properly space out the mortar quadrature
   * points to have |ds| approx constant, but for now just scale points linearly
   * along ispec's local coordinates.
   */
  for (int iquad = 0; iquad < nquad; ++iquad) {
    const type_real xi_mortar = mortar_quadrature.get_xi()(iquad);
    // map from [-1,1] to [j_to_i_lo,j_to_i_hi]
    const type_real xi_i =
        0.5 * ((j_to_i_hi - j_to_i_lo) * xi_mortar + (j_to_i_hi + j_to_i_lo));
    intersections[iquad].first = xi_i;

    // find corresponding coordinate on jspec through global mapping:
    const auto [xi_j, is_inside] = specfem::algorithms::locate_point_on_edge(
        specfem::algorithms::locate_point_on_edge(xi_i, mesh, ispec,
                                                  iorientation),
        mesh, jspec, jorientation);

    // map from [-1,1] to [i_to_j_lo,i_to_j_hi]
    intersections[iquad].second = xi_j;
  }

  return intersections;
}
