#include "specfem/io/mesh/impl/fortran/dim3_globe/read_adjacency_graph.hpp"

#include "specfem/io.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include "specfem/mpi.hpp"

#include <algorithm>
#include <array>
#include <boost/graph/adjacency_list.hpp>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace specfem::io::mesh::impl::fortran::dim3_globe_impl {

int corner_entity(const int local_corner) {
  constexpr std::array<int, 8> entities = { 19, 20, 22, 21, 23, 24, 26, 25 };
  return entities.at(local_corner);
}

int entity_from_corners(const std::set<int> &corners) {
  static const std::array<std::pair<int, std::set<int>>, 26> entities = {
    { { 1, { 0, 1, 2, 3 } }, { 2, { 1, 2, 5, 6 } }, { 3, { 4, 5, 6, 7 } },
      { 4, { 0, 3, 4, 7 } }, { 5, { 0, 1, 4, 5 } }, { 6, { 2, 3, 6, 7 } },
      { 7, { 0, 3 } },       { 8, { 1, 2 } },       { 9, { 5, 6 } },
      { 10, { 4, 7 } },      { 11, { 0, 1 } },      { 12, { 4, 5 } },
      { 13, { 0, 4 } },      { 14, { 1, 5 } },      { 15, { 2, 3 } },
      { 16, { 6, 7 } },      { 17, { 3, 7 } },      { 18, { 2, 6 } },
      { 19, { 0 } },         { 20, { 1 } },         { 21, { 3 } },
      { 22, { 2 } },         { 23, { 4 } },         { 24, { 5 } },
      { 25, { 7 } },         { 26, { 6 } } }
  };
  for (const auto &[entity, entity_corners] : entities) {
    if (corners == entity_corners) {
      return entity;
    }
  }
  throw std::runtime_error(
      "MPI anchor nodes do not form a hex face, edge, or corner");
}

struct mpi_element_description {
  int local_index = -1;
  int orientation = 0;
  int nshared = 0;
  int anchor = 0;
  std::array<double, 12> coordinates{};
};

std::vector<mpi_element_description> describe_mpi_interface(
    const specfem::mesh::globe_mpi_interface &interface,
    const specfem::mesh::control_nodes<specfem::element::dimension_tag::dim3>
        &nodes) {
  std::set<int> shared_nodes(interface.node_ids.begin(),
                             interface.node_ids.end());
  std::vector<mpi_element_description> descriptions;
  for (int ispec = 0; ispec < nodes.nspec; ++ispec) {
    std::set<int> local_corners;
    struct Point {
      std::array<double, 3> xyz;
      int local_corner;
      bool operator<(const Point &other) const { return xyz < other.xyz; }
    };
    std::vector<Point> points;
    for (int corner = 0; corner < 8; ++corner) {
      const int inode = nodes.control_node_index(ispec, corner);
      if (shared_nodes.contains(inode)) {
        local_corners.insert(corner);
        points.push_back(
            { { nodes.coordinates(inode, 0), nodes.coordinates(inode, 1),
                nodes.coordinates(inode, 2) },
              corner });
      }
    }
    if (points.empty()) {
      continue;
    }
    if (points.size() != 1 && points.size() != 2 && points.size() != 4) {
      throw std::runtime_error(
          "Invalid number of shared MPI corners on globe element " +
          std::to_string(ispec));
    }
    std::sort(points.begin(), points.end());
    mpi_element_description description;
    description.local_index = ispec;
    description.orientation = entity_from_corners(local_corners);
    description.nshared = static_cast<int>(points.size());
    description.anchor = corner_entity(points.front().local_corner);
    for (int i = 0; i < description.nshared; ++i) {
      for (int component = 0; component < 3; ++component) {
        description.coordinates[3 * i + component] = points[i].xyz[component];
      }
    }
    descriptions.push_back(description);
  }
  return descriptions;
}

bool same_interface_entity(const mpi_element_description &left,
                           const mpi_element_description &right) {
  if (left.nshared != right.nshared) {
    return false;
  }
  constexpr double coordinate_tolerance = 1.0e-4;
  for (int i = 0; i < 3 * left.nshared; ++i) {
    if (std::abs(left.coordinates[i] - right.coordinates[i]) >
        coordinate_tolerance) {
      return false;
    }
  }
  return true;
}

void build_mpi_adjacency(specfem::mesh::globe3d_mesh &mesh) {
  if (mesh.globe.mpi_interfaces.empty()) {
    return;
  }
#ifndef SPECFEM_ENABLE_MPI
  throw std::runtime_error(
      "A partitioned globe database requires an MPI-enabled build");
#else
  auto &interfaces = mesh.globe.mpi_interfaces;
  const int ninterfaces = static_cast<int>(interfaces.size());
  std::vector<std::vector<mpi_element_description>> local(ninterfaces);
  std::vector<std::vector<mpi_element_description>> remote(ninterfaces);
  std::vector<int> send_counts(ninterfaces), receive_counts(ninterfaces);
  std::vector<MPI_Request> requests(2 * ninterfaces, MPI_REQUEST_NULL);
  const auto comm = specfem::MPI::communicator();

  for (int i = 0; i < ninterfaces; ++i) {
    local[i] = describe_mpi_interface(interfaces[i], mesh.control_nodes);
    send_counts[i] = static_cast<int>(local[i].size());
    SPECFEM_MPI_SAFECALL(MPI_Irecv(&receive_counts[i], 1, MPI_INT,
                                   interfaces[i].neighbor_rank, 29001, comm,
                                   &requests[2 * i]));
    SPECFEM_MPI_SAFECALL(MPI_Isend(&send_counts[i], 1, MPI_INT,
                                   interfaces[i].neighbor_rank, 29001, comm,
                                   &requests[2 * i + 1]));
  }
  SPECFEM_MPI_SAFECALL(MPI_Waitall(static_cast<int>(requests.size()),
                                   requests.data(), MPI_STATUSES_IGNORE));

  std::fill(requests.begin(), requests.end(), MPI_REQUEST_NULL);
  for (int i = 0; i < ninterfaces; ++i) {
    remote[i].resize(receive_counts[i]);
    const std::size_t send_bytes =
        local[i].size() * sizeof(mpi_element_description);
    const std::size_t receive_bytes =
        remote[i].size() * sizeof(mpi_element_description);
    if (send_bytes > std::numeric_limits<int>::max() ||
        receive_bytes > std::numeric_limits<int>::max()) {
      throw std::runtime_error("Globe MPI interface description is too large");
    }
    SPECFEM_MPI_SAFECALL(
        MPI_Irecv(remote[i].data(), static_cast<int>(receive_bytes), MPI_BYTE,
                  interfaces[i].neighbor_rank, 29002, comm, &requests[2 * i]));
    SPECFEM_MPI_SAFECALL(MPI_Isend(
        local[i].data(), static_cast<int>(send_bytes), MPI_BYTE,
        interfaces[i].neighbor_rank, 29002, comm, &requests[2 * i + 1]));
  }
  SPECFEM_MPI_SAFECALL(MPI_Waitall(static_cast<int>(requests.size()),
                                   requests.data(), MPI_STATUSES_IGNORE));

  for (int i = 0; i < ninterfaces; ++i) {
    std::vector<bool> matched(remote[i].size(), false);
    for (const auto &local_element : local[i]) {
      int match = -1;
      for (int candidate = 0; candidate < static_cast<int>(remote[i].size());
           ++candidate) {
        if (!matched[candidate] &&
            same_interface_entity(local_element, remote[i][candidate])) {
          if (match != -1) {
            throw std::runtime_error(
                "Ambiguous element match across globe MPI interface");
          }
          match = candidate;
        }
      }
      if (match == -1) {
        throw std::runtime_error(
            "Could not match an element across globe MPI interface");
      }
      matched[match] = true;
      const auto &remote_element = remote[i][match];
      mesh.adjacency_graph.mpi_connections().emplace_back(
          specfem::element_connections::type::strongly_conforming,
          static_cast<specfem::mesh_entity::dim3::type>(
              local_element.orientation),
          interfaces[i].neighbor_rank,
          static_cast<specfem::mesh_entity::dim3::type>(
              remote_element.orientation),
          local_element.local_index, remote_element.local_index,
          static_cast<specfem::mesh_entity::dim3::type>(local_element.anchor),
          static_cast<specfem::mesh_entity::dim3::type>(remote_element.anchor));
    }
    if (std::find(matched.begin(), matched.end(), false) != matched.end()) {
      throw std::runtime_error(
          "Neighbor reported unmatched elements on globe MPI interface");
    }
  }
#endif
}

} // namespace specfem::io::mesh::impl::fortran::dim3_globe_impl

void specfem::io::mesh::impl::fortran::dim3_globe::read_adjacency_graph(
    std::ifstream &stream, specfem::mesh::globe3d_mesh &mesh, const int nnode) {
  using Dimension = specfem::element::dimension_tag;

  int nadjacencies = 0;
  specfem::io::fortran_read_line(stream, &nadjacencies);
  std::vector<int> xadj(mesh.nspec + 1), adjncy(nadjacencies),
      adjacency_types(nadjacencies);
  specfem::io::fortran_read_line(stream, &xadj);
  specfem::io::fortran_read_line(stream, &adjncy);
  specfem::io::fortran_read_line(stream, &adjacency_types);

  mesh.adjacency_graph =
      specfem::mesh::adjacency_graph<Dimension::dim3>(mesh.nspec);
  auto &graph = mesh.adjacency_graph.local_connections();
  using EdgeProperties =
      specfem::mesh::adjacency_graph<Dimension::dim3>::EdgeProperties;
  for (int ispec = 0; ispec < mesh.nspec; ++ispec) {
    if (xadj[ispec] < 1 || xadj[ispec + 1] < xadj[ispec] ||
        xadj[ispec + 1] > nadjacencies + 1) {
      throw std::runtime_error("Invalid CSR adjacency in globe database");
    }
    for (int offset = xadj[ispec] - 1; offset < xadj[ispec + 1] - 1; ++offset) {
      const int neighbor = adjncy[offset] - 1;
      if (neighbor < 0 || neighbor >= mesh.nspec) {
        throw std::runtime_error("Invalid adjacency element in globe database");
      }
      boost::add_edge(
          ispec, neighbor,
          EdgeProperties(
              specfem::element_connections::type::strongly_conforming,
              static_cast<specfem::mesh_entity::dim3::type>(
                  adjacency_types[offset])),
          graph);
    }
  }
  mesh.adjacency_graph.assert_symmetry();

  int nneighbors = 0;
  specfem::io::fortran_read_line(stream, &nneighbors);
  mesh.globe.mpi_interfaces.resize(nneighbors);
  for (auto &interface : mesh.globe.mpi_interfaces) {
    int nshared = 0;
    specfem::io::fortran_read_line(stream, &interface.neighbor_rank, &nshared);
    interface.node_ids.resize(nshared);
    if (nshared > 0) {
      specfem::io::fortran_read_line(stream, &interface.node_ids);
      for (auto &inode : interface.node_ids) {
        --inode;
        if (inode < 0 || inode >= nnode) {
          throw std::runtime_error(
              "Invalid MPI anchor node ID in globe database");
        }
      }
    }
  }
  specfem::io::mesh::impl::fortran::dim3_globe_impl::build_mpi_adjacency(mesh);
}
