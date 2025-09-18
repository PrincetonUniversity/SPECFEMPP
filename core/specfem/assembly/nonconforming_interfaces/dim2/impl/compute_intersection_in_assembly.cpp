
#include "compute_intersection.hpp"
#include "enumerations/mesh_entities.hpp"
#include "compute_intersection_in_assembly.hpp"
#include "specfem_setup.hpp"
#include <sstream>
#include <stdexcept>

template <typename EdgeType>
std::vector<std::pair<type_real, type_real> >
specfem::assembly::nonconforming_interfaces_impl::compute_intersection(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const EdgeType &edge,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &mortar_quadrature) {

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

  const int ngnod = mesh.ngnod;

  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg1("coorg1", ngnod);
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg2("coorg2", ngnod);

  for (int i = 0; i < ngnod; i++) {
    coorg1(i).x = mesh.h_control_node_coord(0, ispec, i);
    coorg2(i).z = mesh.h_control_node_coord(1, jspec, i);
  }

  return specfem::assembly::nonconforming_interfaces_impl::
      compute_intersection(coorg1, coorg2, iorientation, jorientation,
                           mortar_quadrature);
}
