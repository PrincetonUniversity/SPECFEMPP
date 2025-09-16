#include "io/mesh/impl/fortran/dim2/read_interfaces.hpp"
#include "io/fortranio/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"

namespace {

std::vector<int> get_common_nodes(std::vector<int> element1_nodes,
                                  std::vector<int> element2_nodes) {
  std::sort(element1_nodes.begin(), element1_nodes.end());
  std::sort(element2_nodes.begin(), element2_nodes.end());

  std::vector<int> common_nodes;
  std::set_intersection(element1_nodes.begin(), element1_nodes.end(),
                        element2_nodes.begin(), element2_nodes.end(),
                        std::back_inserter(common_nodes));
  return common_nodes;
}

std::tuple<specfem::mesh_entity::type, specfem::mesh_entity::type>
compute_connected_edges(Kokkos::View<int **, Kokkos::HostSpace> knods,
                        const int ispec1, const int ispec2) {
  std::vector<int> element1_nodes{ knods(0, ispec1), knods(1, ispec1),
                                   knods(2, ispec1), knods(3, ispec1) };

  std::vector<int> element2_nodes{ knods(0, ispec2), knods(1, ispec2),
                                   knods(2, ispec2), knods(3, ispec2) };

  // Find the common nodes between 2 elements
  const auto common_nodes = get_common_nodes(element1_nodes, element2_nodes);

  if (common_nodes.size() != 2)
    throw std::runtime_error("Error: Elements " + std::to_string(ispec1) +
                             " and " + std::to_string(ispec2) +
                             " do not share an edge.");

  std::vector<int> edge1_node_indices;
  std::vector<int> edge2_node_indices;

  for (const auto &node : common_nodes) {
    if (element1_nodes[0] == node)
      edge1_node_indices.push_back(0);
    if (element1_nodes[1] == node)
      edge1_node_indices.push_back(1);
    if (element1_nodes[2] == node)
      edge1_node_indices.push_back(2);
    if (element1_nodes[3] == node)
      edge1_node_indices.push_back(3);
  }

  for (const auto &node : common_nodes) {
    if (element2_nodes[0] == node)
      edge2_node_indices.push_back(0);
    if (element2_nodes[1] == node)
      edge2_node_indices.push_back(1);
    if (element2_nodes[2] == node)
      edge2_node_indices.push_back(2);
    if (element2_nodes[3] == node)
      edge2_node_indices.push_back(3);
  }

  if (edge1_node_indices.size() != 2 || edge2_node_indices.size() != 2)
    throw std::runtime_error("Error: Could not determine edge nodes.");

  specfem::mesh_entity::type edge1, edge2;

  // 0, 1 -> bottom
  // 1, 2 -> right
  // 2, 3 -> top
  // 3, 0 -> left

  if ((edge1_node_indices == std::vector<int>{ 0, 1 } ||
       edge1_node_indices == std::vector<int>{ 1, 0 })) {
    edge1 = specfem::mesh_entity::type::bottom;
  } else if ((edge1_node_indices == std::vector<int>{ 1, 2 } ||
              edge1_node_indices == std::vector<int>{ 2, 1 })) {
    edge1 = specfem::mesh_entity::type::right;
  } else if ((edge1_node_indices == std::vector<int>{ 2, 3 } ||
              edge1_node_indices == std::vector<int>{ 3, 2 })) {
    edge1 = specfem::mesh_entity::type::top;
  } else if ((edge1_node_indices == std::vector<int>{ 3, 0 } ||
              edge1_node_indices == std::vector<int>{ 0, 3 })) {
    edge1 = specfem::mesh_entity::type::left;
  } else {
    throw std::runtime_error("Error: Could not determine edge1 type.");
  }

  if ((edge2_node_indices == std::vector<int>{ 0, 1 } ||
       edge2_node_indices == std::vector<int>{ 1, 0 })) {
    edge2 = specfem::mesh_entity::type::bottom;
  } else if ((edge2_node_indices == std::vector<int>{ 1, 2 } ||
              edge2_node_indices == std::vector<int>{ 2, 1 })) {
    edge2 = specfem::mesh_entity::type::right;
  } else if ((edge2_node_indices == std::vector<int>{ 2, 3 } ||
              edge2_node_indices == std::vector<int>{ 3, 2 })) {
    edge2 = specfem::mesh_entity::type::top;
  } else if ((edge2_node_indices == std::vector<int>{ 3, 0 } ||
              edge2_node_indices == std::vector<int>{ 0, 3 })) {
    edge2 = specfem::mesh_entity::type::left;
  } else {
    throw std::runtime_error("Error: Could not determine edge2 type.");
  }

  return { edge1, edge2 };
}

} // namespace

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag medium1,
          specfem::element::medium_tag medium2>
specfem::mesh::interface_container<DimensionTag, medium1, medium2>
specfem::io::mesh::impl::fortran::dim2::read_interfaces(
    const int num_interfaces,
    Kokkos::View<int **, Kokkos::LayoutRight, Kokkos::HostSpace> knods,
    std::ifstream &stream, const specfem::MPI::MPI *mpi) {

  specfem::mesh::interface_container<DimensionTag, medium1, medium2> interface(
      num_interfaces);

  if (!num_interfaces)
    return interface;

  int medium1_ispec_l, medium2_ispec_l;

  for (int i = 0; i < num_interfaces; i++) {
    specfem::io::fortran_read_line(stream, &medium2_ispec_l, &medium1_ispec_l);
    interface.medium1_index_mapping(i) = medium1_ispec_l - 1;
    interface.medium2_index_mapping(i) = medium2_ispec_l - 1;

    const auto [edge1, edge2] = compute_connected_edges(
        knods, medium1_ispec_l - 1, medium2_ispec_l - 1);

    interface.medium1_edge_type(i) = edge1;
    interface.medium2_edge_type(i) = edge2;
  }

  return interface;
}

// Explicit instantiation of the template function for the different medium
// interfaces elastic/acoustic
template specfem::mesh::interface_container<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv,
    specfem::element::medium_tag::acoustic>
specfem::io::mesh::impl::fortran::dim2::read_interfaces<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv,
    specfem::element::medium_tag::acoustic>(
    const int num_interfaces,
    Kokkos::View<int **, Kokkos::LayoutRight, Kokkos::HostSpace> knods,
    std::ifstream &stream, const specfem::MPI::MPI *mpi);

// acoustic/poroelastic
template specfem::mesh::interface_container<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::poroelastic>
specfem::io::mesh::impl::fortran::dim2::read_interfaces<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::poroelastic>(
    const int num_interfaces,
    Kokkos::View<int **, Kokkos::LayoutRight, Kokkos::HostSpace> knods,
    std::ifstream &stream, const specfem::MPI::MPI *mpi);

// elastic/poroelastic
template specfem::mesh::interface_container<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv,
    specfem::element::medium_tag::poroelastic>
specfem::io::mesh::impl::fortran::dim2::read_interfaces<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv,
    specfem::element::medium_tag::poroelastic>(
    const int num_interfaces,
    Kokkos::View<int **, Kokkos::LayoutRight, Kokkos::HostSpace> knods,
    std::ifstream &stream, const specfem::MPI::MPI *mpi);

specfem::mesh::coupled_interfaces<specfem::dimension::type::dim2>
specfem::io::mesh::impl::fortran::dim2::read_coupled_interfaces(
    std::ifstream &stream, const int num_interfaces_elastic_acoustic,
    const int num_interfaces_acoustic_poroelastic,
    const int num_interfaces_elastic_poroelastic,
    Kokkos::View<int **, Kokkos::LayoutRight, Kokkos::HostSpace> knods,
    const specfem::MPI::MPI *mpi) {

  auto elastic_acoustic =
      specfem::io::mesh::impl::fortran::dim2::read_interfaces<
          specfem::dimension::type::dim2,
          specfem::element::medium_tag::elastic_psv,
          specfem::element::medium_tag::acoustic>(
          num_interfaces_elastic_acoustic, knods, stream, mpi);

  auto acoustic_poroelastic =
      specfem::io::mesh::impl::fortran::dim2::read_interfaces<
          specfem::dimension::type::dim2,
          specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::poroelastic>(
          num_interfaces_acoustic_poroelastic, knods, stream, mpi);

  auto elastic_poroelastic =
      specfem::io::mesh::impl::fortran::dim2::read_interfaces<
          specfem::dimension::type::dim2,
          specfem::element::medium_tag::elastic_psv,
          specfem::element::medium_tag::poroelastic>(
          num_interfaces_elastic_poroelastic, knods, stream, mpi);

  return specfem::mesh::coupled_interfaces<specfem::dimension::type::dim2>(
      elastic_acoustic, acoustic_poroelastic, elastic_poroelastic);
}
