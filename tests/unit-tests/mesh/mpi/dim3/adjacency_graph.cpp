
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"
#include "test_fixture.hpp"

namespace specfem::test_configuration {

struct MPIConnection {

  constexpr static specfem::element::dimension_tag dimension =
      specfem::element::dimension_tag::dim3;
  struct EdgeProperties {
    size_t rank;  ///< MPI rank of the edge
    size_t index; ///< Local element index associated with this edge
    specfem::mesh_entity::dim3::type connection_id;
    specfem::mesh_entity::dim3::type anchor;

    EdgeProperties(size_t rank, size_t index,
                   specfem::mesh_entity::dim3::type connection_id,
                   specfem::mesh_entity::dim3::type anchor)
        : rank(rank), index(index), connection_id(connection_id),
          anchor(anchor) {}
    EdgeProperties(
        const std::tuple<size_t, size_t, specfem::mesh_entity::dim3::type,
                         specfem::mesh_entity::dim3::type> &properties)
        : EdgeProperties(std::get<0>(properties), std::get<1>(properties),
                         std::get<2>(properties), std::get<3>(properties)) {}
  };

  EdgeProperties edge1; ///< Properties of edge 1 in MPI connection
  EdgeProperties edge2; ///< Properties of edge 2 in MPI connection

  MPIConnection(const EdgeProperties &edge1, const EdgeProperties &edge2)
      : edge1(edge1), edge2(edge2) {}

  MPIConnection(
      const std::tuple<size_t, size_t, specfem::mesh_entity::dim3::type,
                       specfem::mesh_entity::dim3::type> &edge1_properties,
      const std::tuple<size_t, size_t, specfem::mesh_entity::dim3::type,
                       specfem::mesh_entity::dim3::type> &edge2_properties)
      : edge1(edge1_properties), edge2(edge2_properties) {}

  bool expect_in(const std::vector<specfem::mesh::adjacency_graph<
                     dimension>::MPIEdgeProperties> &mpi_connections) const {

    const auto my_rank = static_cast<size_t>(specfem::MPI::get_rank());
    if (my_rank != edge1.rank && my_rank != edge2.rank) {
      return false; // This connection was not expected on this rank
    }

    /// Check if the expected MPI connection exists in the list of MPI
    /// connections
    bool found = false;
    for (const auto &mpi_edge : mpi_connections) {
      if (my_rank == edge1.rank) {
        // We are edge1, look for matching edge2
        if (mpi_edge.neighbor_partition == edge2.rank &&
            mpi_edge.orientation == edge1.connection_id &&
            mpi_edge.neighbor_orientation == edge2.connection_id &&
            mpi_edge.local_index == edge1.index &&
            mpi_edge.neighbor_local_index == edge2.index &&
            mpi_edge.local_anchor_point == edge1.anchor &&
            mpi_edge.neighbor_anchor_point == edge2.anchor) {
          found = true;
          break;
        }
      } else {
        // We are edge2, look for matching edge1
        if (mpi_edge.neighbor_partition == edge1.rank &&
            mpi_edge.orientation == edge2.connection_id &&
            mpi_edge.neighbor_orientation == edge1.connection_id &&
            mpi_edge.local_index == edge2.index &&
            mpi_edge.neighbor_local_index == edge1.index &&
            mpi_edge.local_anchor_point == edge2.anchor &&
            mpi_edge.neighbor_anchor_point == edge1.anchor) {
          found = true;
          break;
        }
      }
    }

    return found;
  }
};

struct ExpectedMPIAdjacency3D {
  constexpr static specfem::element::dimension_tag dimension =
      specfem::element::dimension_tag::dim3;

  std::vector<MPIConnection> mpi_connections; ///< List of expected MPI
                                              ///< connections

  ExpectedMPIAdjacency3D(
      const std::initializer_list<MPIConnection> &mpi_connections)
      : mpi_connections(mpi_connections) {}

  void check(
      const specfem::mesh::adjacency_graph<dimension> &adjacency_graph) const {
    const auto &mpi_conns = adjacency_graph.mpi_connections();
    for (const auto &expected_mpi_conn : mpi_connections) {
      auto local_found = expected_mpi_conn.expect_in(mpi_conns);
      int local_found_int = local_found ? 1 : 0;
      int found_int = 0;
      SPECFEM_MPI_SAFECALL(MPI_Reduce(&local_found_int, &found_int, 1, MPI_INT,
                                      MPI_LOR, 0,
                                      specfem::MPI::world_communicator()));
      bool found = (found_int != 0);

      if (!found) {
        SPECFEM_MPI_ON_ROOT({
          std::ostringstream msg;
          msg << "Failed expected MPI adjacency between rank "
              << expected_mpi_conn.edge1.rank << " (element "
              << expected_mpi_conn.edge1.index << ", connection ID "
              << static_cast<int>(expected_mpi_conn.edge1.connection_id)
              << ", anchor " << static_cast<int>(expected_mpi_conn.edge1.anchor)
              << ") and rank " << expected_mpi_conn.edge2.rank << " (element "
              << expected_mpi_conn.edge2.index << ", connection ID "
              << static_cast<int>(expected_mpi_conn.edge2.connection_id)
              << ", anchor " << static_cast<int>(expected_mpi_conn.edge2.anchor)
              << ")\n";
          msg << "  No matching MPI connection found in adjacency graph.\n";
          ADD_FAILURE() << msg.str();
        });
      }

      SPECFEM_MPI_SAFECALL(MPI_Barrier(specfem::MPI::world_communicator()));
    }
    SPECFEM_MPI_ON_ROOT({
      SUCCEED() << "All expected MPI connections are present and correct."
                << std::endl;
    });
  }
};

} // namespace specfem::test_configuration

using namespace specfem::test_configuration;

static const std::unordered_map<std::string, ExpectedMPIAdjacency3D>
    expected_mpi_adjacency_map = {
      { "HomogeneousMediumMPI4x4",
        ExpectedMPIAdjacency3D(
            { // Left - Right connections
              MPIConnection(
                  std::make_tuple(
                      0, 1, specfem::mesh_entity::dim3::type::right,
                      specfem::mesh_entity::dim3::type::bottom_front_right),
                  std::make_tuple(
                      1, 2, specfem::mesh_entity::dim3::type::left,
                      specfem::mesh_entity::dim3::type::bottom_front_left)),
              MPIConnection(
                  std::make_tuple(
                      2, 1, specfem::mesh_entity::dim3::type::right,
                      specfem::mesh_entity::dim3::type::bottom_front_right),
                  std::make_tuple(
                      3, 2, specfem::mesh_entity::dim3::type::left,
                      specfem::mesh_entity::dim3::type::bottom_front_left)),
              // Front - Back connections
              MPIConnection(
                  std::make_tuple(
                      0, 2, specfem::mesh_entity::dim3::type::back,
                      specfem::mesh_entity::dim3::type::bottom_back_left),
                  std::make_tuple(
                      2, 0, specfem::mesh_entity::dim3::type::front,
                      specfem::mesh_entity::dim3::type::bottom_front_left)),
              MPIConnection(
                  std::make_tuple(
                      1, 3, specfem::mesh_entity::dim3::type::back,
                      specfem::mesh_entity::dim3::type::bottom_back_right),
                  std::make_tuple(
                      3, 1, specfem::mesh_entity::dim3::type::front,
                      specfem::mesh_entity::dim3::type::bottom_front_right)),
              // Diagonal Edge connections
              MPIConnection(
                  std::make_tuple(
                      0, 3, specfem::mesh_entity::dim3::type::back_right,
                      specfem::mesh_entity::dim3::type::bottom_back_right),
                  std::make_tuple(
                      3, 0, specfem::mesh_entity::dim3::type::front_left,
                      specfem::mesh_entity::dim3::type::bottom_front_left)),
              MPIConnection(
                  std::make_tuple(
                      1, 2, specfem::mesh_entity::dim3::type::back_left,
                      specfem::mesh_entity::dim3::type::bottom_back_left),
                  std::make_tuple(2, 1,
                                  specfem::mesh_entity::dim3::type::front_right,
                                  specfem::mesh_entity::dim3::type::
                                      bottom_front_right)) }) }
    };

TEST_P(MPIMesh3DTest, MPIAdjacencyGraph) {
  const auto &param_name = GetParam();

  // Skip cleanly if there is no MPI ground truth for this parameter.
  const auto it = expected_mpi_adjacency_map.find(param_name);
  if (it == expected_mpi_adjacency_map.end()) {
    GTEST_SKIP() << "No MPI adjacency ground truth defined for test case: "
                 << param_name << std::endl;
    return;
  }

  const auto &mesh = getMesh();
  const auto &adjacency_graph = mesh.adjacency_graph;
  const auto &expected = it->second;
  expected.check(adjacency_graph);
}
