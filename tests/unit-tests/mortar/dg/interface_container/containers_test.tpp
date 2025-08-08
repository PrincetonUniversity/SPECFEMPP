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
ContainerType load_interfaces(const test_configuration::mesh &mesh_config) {
  // temporary measure while adjacency_graph is WIP
  std::ifstream interfaces_in(mesh_config.interface_file);
  int num_edges;
  interfaces_in >> num_edges;

  // break into cases: 1 or 2 edge container;

  if constexpr (ContainerType::is_single_edge) {
    ContainerType container(num_edges * 2);
    int iedge = 0, ispec, edgetype;
    while (interfaces_in >> ispec >> edgetype) {
      container.h_index_mapping(iedge) = ispec - 1;
      container.h_edge_type(iedge) = edge_from_int(edgetype);
      iedge ++;
    }
    Kokkos::deep_copy(container.index_mapping, container.h_index_mapping);
    Kokkos::deep_copy(container.edge_type, container.h_edge_type);
    interfaces_in.close();
    return container;
  } else {
    ContainerType container(num_edges, num_edges);
    int iedge = 0, ispec1, edgetype1, ispec2, edgetype2;
    while (interfaces_in >> ispec1 >> edgetype1 >> ispec2 >> edgetype2) {
      container.h_medium1_index_mapping(iedge) = ispec1 - 1;
      container.h_medium1_edge_type(iedge) = edge_from_int(edgetype1);
      container.h_medium2_index_mapping(iedge) = ispec2 - 1;
      container.h_medium2_edge_type(iedge) = edge_from_int(edgetype2);
      iedge ++;
    }
    Kokkos::deep_copy(container.medium1_index_mapping,
                      container.h_medium1_index_mapping);
    Kokkos::deep_copy(container.medium1_edge_type,
                      container.h_medium1_edge_type);
    Kokkos::deep_copy(container.medium2_index_mapping,
                      container.h_medium2_index_mapping);
    Kokkos::deep_copy(container.medium2_edge_type,
                      container.h_medium2_edge_type);
    interfaces_in.close();
    return container;
  }
}

template <typename ContainerType>
void test_interface(
    const test_configuration::mesh &mesh_config,
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {

  ContainerType container = load_interfaces<ContainerType>(mesh_config);
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
