
#include "compute_intersection_in_assembly.hpp"
#include "compute_intersection.hpp"
#include "enumerations/mesh_entities.hpp"
#include "specfem_setup.hpp"
#include <sstream>
#include <stdexcept>

template <typename EdgeType>
inline std::tuple<
    Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace>,
    Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace>,
    specfem::mesh_entity::type, specfem::mesh_entity::type>
expand_edge_index(
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
  return { coorg1, coorg2, iorientation, jorientation };
}

template <typename EdgeType>
std::vector<std::pair<type_real, type_real> >
specfem::assembly::nonconforming_interfaces_impl::compute_intersection(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const EdgeType &edge, const Kokkos::View<type_real *> &mortar_quadrature) {
  const auto edgeinfo = expand_edge_index(mesh, edge);
  return specfem::assembly::nonconforming_interfaces_impl::compute_intersection(
      std::get<0>(edgeinfo), std::get<1>(edgeinfo), std::get<2>(edgeinfo),
      std::get<2>(edgeinfo), mortar_quadrature);
}

template <typename EdgeType>
void set_transfer_functions(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const EdgeType &edge,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &mortar_quadrature,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &element_quadrature,
    Kokkos::View<type_real **, Kokkos::HostSpace> &transfer_function1,
    Kokkos::View<type_real **, Kokkos::HostSpace> &transfer_function2) {
  const auto edgeinfo = expand_edge_index(mesh, edge);
  return specfem::assembly::nonconforming_interfaces_impl::
      set_transfer_functions(
          std::get<0>(edgeinfo), std::get<1>(edgeinfo), std::get<2>(edgeinfo),
          std::get<2>(edgeinfo), mortar_quadrature, mortar_quadrature,
          element_quadrature, transfer_function1, transfer_function2);
}

template <typename EdgeType>
void set_transfer_functions(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const EdgeType &edge,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &mortar_quadrature,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &element_quadrature,
    Kokkos::View<type_real **, Kokkos::HostSpace> &transfer_function1,
    Kokkos::View<type_real **, Kokkos::HostSpace> &transfer_function1_prime,
    Kokkos::View<type_real **, Kokkos::HostSpace> &transfer_function2,
    Kokkos::View<type_real **, Kokkos::HostSpace> &transfer_function2_prime) {
  const auto edgeinfo = expand_edge_index(mesh, edge);
  return specfem::assembly::nonconforming_interfaces_impl::
      set_transfer_functions(std::get<0>(edgeinfo), std::get<1>(edgeinfo),
                             std::get<2>(edgeinfo), std::get<2>(edgeinfo),
                             mortar_quadrature, element_quadrature,
                             transfer_function1, transfer_function1_prime,
                             transfer_function2, transfer_function2_prime);
}
