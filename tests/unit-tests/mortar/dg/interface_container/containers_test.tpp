#include "Kokkos_Core.hpp"
#include "containers_test.hpp"
#include <fstream>
#include <type_traits>

#include "edge_policy_test.tpp"
#include "enumerations/specfem_enums.hpp"

static specfem::enums::edge::type edge_from_int(const int edgetype) {
  switch (edgetype) {
  case 1:
    return specfem::enums::edge::type::BOTTOM;
  case 2:
    return specfem::enums::edge::type::RIGHT;
  case 3:
    return specfem::enums::edge::type::TOP;
  case 4:
    return specfem::enums::edge::type::LEFT;
  default:
    return specfem::enums::edge::type::NONE;
  }
}

namespace test_configuration::interface_containers {

template <typename ContainerType>
ContainerType load_interfaces(const test_configuration::mesh &mesh_config,
                              const int ngll) {
  // temporary measure while adjacency_graph is WIP
  // interface file is stored in 3 chunks, the size of each specified in the
  // first line.
  //
  // chunk 1: medium 1 edges  (ispec, edgetype)
  // chunk 2: medium 2 edges  (ispec, edgetype)
  // chunk 3: interfaces   (edge1, edge2, param_start_edge1,
  //                param_end_edge1, param_start_edge2, param_end_edge2)
  std::ifstream interfaces_in(mesh_config.interface_file);
  int num_edges_1, num_edges_2, num_interfaces;
  interfaces_in >> num_edges_1 >> num_edges_2 >> num_interfaces;
  ContainerType container(specfem::assembly::interface::initializer(
      num_edges_1, num_edges_2, num_interfaces, ngll, ngll, ngll));

  // medium 1:
  int ispec, edgetype;
  for (int iedge = 0; iedge < num_edges_1; iedge++) {
    interfaces_in >> ispec >> edgetype;
    container.template index_at<1, true>(iedge) = ispec - 1;
    container.template edge_type_at<1, true>(iedge) = edge_from_int(edgetype);
  }

  // medium 2 behavior depends on single or double edge
  // single edge medium 2 is appended onto medium 1, and index_at<2> points to
  // first medium.

  for (int iedge = 0; iedge < num_edges_2; iedge++) {
    interfaces_in >> ispec >> edgetype;
    const int ind = iedge + ContainerType::is_single_edge ? iedge : 0;
    container.template index_at<2, true>(ind) = ispec - 1;
    container.template index_at<2, true>(ind) = edge_from_int(edgetype);
  }

  container.template sync_edge_container<specfem::sync::kind::HostToDevice>();

  interfaces_in.close();
  return container;
}

template <typename ContainerType>
void test_interface(
    const test_configuration::mesh &mesh_config,
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {

  ContainerType container =
      load_interfaces<ContainerType>(mesh_config, assembly.mesh.ngllz);
  test_edge_policy(container, mesh, assembly);
}

void test_on_mesh(
    const test_configuration::mesh &mesh_config,
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {
  if (mesh_config.interface_fluid_2d) {
    test_interface<fluid_2d>(mesh_config, mesh, assembly);
  }
  if (mesh_config.interface_solid_2d) {
    test_interface<solid_2d>(mesh_config, mesh, assembly);
  }
  if (mesh_config.interface_fluid_fluid_2d) {
    test_interface<fluid_fluid_2d>(mesh_config, mesh, assembly);
  }
  if (mesh_config.interface_solid_fluid_2d) {
    test_interface<solid_fluid_2d>(mesh_config, mesh, assembly);
  }
  if (mesh_config.interface_fluid_solid_2d) {
    test_interface<fluid_solid_2d>(mesh_config, mesh, assembly);
  }
  if (mesh_config.interface_solid_solid_2d) {
    test_interface<solid_solid_2d>(mesh_config, mesh, assembly);
  }
}
} // namespace test_configuration::interface_containers
