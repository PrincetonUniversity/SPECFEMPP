#include "../test_fixture.hpp"
#include "SPECFEM_Environment.hpp"
#include "enumerations/connections.hpp"
#include "enumerations/interface.hpp"
#include "enumerations/mesh_entities.hpp"
#include "io/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <gtest/gtest.h>

namespace specfem::assembly_test {

struct TotalQuadraturePoints {
  int nelements;
  int ngllz;
  int nglly;
  int ngllx;
  int npoints;
  TotalQuadraturePoints(int nelements, int ngllz, int nglly, int ngllx,
                        int npoints)
      : nelements(nelements), ngllz(ngllz), nglly(nglly), ngllx(ngllx),
        npoints(npoints) {}
};

struct ExpectedMapping {
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3;
  TotalQuadraturePoints total_quadrature_points;
  std::string database_file;

  ExpectedMapping(TotalQuadraturePoints total_quadrature_points,
                  const std::string &database_file)
      : total_quadrature_points(total_quadrature_points),
        database_file(database_file) {}

  void
  check(const specfem::assembly::mesh_impl::points<dimension> &points) const {

    // Get the mesh
    const auto expected_mesh =
        specfem::io::read_3d_mesh(database_file, SPECFEMEnvironment::get_mpi());

    ASSERT_EQ(points.nspec, total_quadrature_points.nelements)
        << "Number of spectral elements mismatch. "
        << "Expected: " << total_quadrature_points.nelements << ", "
        << "Got: " << points.nspec << std::endl;
    ASSERT_EQ(points.ngllz, total_quadrature_points.ngllz)
        << "Number of GLL points in z-direction mismatch. "
        << "Expected: " << total_quadrature_points.ngllz << ", "
        << "Got: " << points.ngllz << std::endl;
    ASSERT_EQ(points.nglly, total_quadrature_points.nglly)
        << "Number of GLL points in y-direction mismatch. "
        << "Expected: " << total_quadrature_points.nglly << ", "
        << "Got: " << points.nglly << std::endl;
    ASSERT_EQ(points.ngllx, total_quadrature_points.ngllx)
        << "Number of GLL points in x-direction mismatch. "
        << "Expected: " << total_quadrature_points.ngllx << ", "
        << "Got: " << points.ngllx << std::endl;
    ASSERT_EQ(points.nglob, total_quadrature_points.npoints)
        << "Total number of global points mismatch. "
        << "Expected: " << total_quadrature_points.npoints << ", "
        << "Got: " << points.nglob << std::endl;

    // Check that views are allocated correctly
    ASSERT_TRUE(points.index_mapping.extent(0) == points.nspec)
        << "Index mapping extent 0 mismatch.";
    ASSERT_TRUE(points.index_mapping.extent(1) == points.ngllz)
        << "Index mapping extent 1 mismatch.";
    ASSERT_TRUE(points.index_mapping.extent(2) == points.nglly)
        << "Index mapping extent 2 mismatch.";
    ASSERT_TRUE(points.index_mapping.extent(3) == points.ngllx)
        << "Index mapping extent 3 mismatch.";
    ASSERT_TRUE(points.coord.extent(0) == points.nspec)
        << "Coordinate view extent 0 mismatch.";
    ASSERT_TRUE(points.coord.extent(1) == points.ngllz)
        << "Coordinate view extent 1 mismatch.";
    ASSERT_TRUE(points.coord.extent(2) == points.nglly)
        << "Coordinate view extent 2 mismatch.";
    ASSERT_TRUE(points.coord.extent(3) == points.ngllx)
        << "Coordinate view extent 3 mismatch.";
    ASSERT_TRUE(points.coord.extent(4) == 3)
        << "Coordinate view extent 4 mismatch.";

    const auto &adjacency_graph = expected_mesh.adjacency_graph;
    const auto &graph = adjacency_graph.graph();

    // Filter out strongly conforming connections
    auto filter = [&graph](const auto &edge) {
      return graph[edge].connection ==
             specfem::connections::type::strongly_conforming;
    };

    // Create a filtered graph view
    const auto fg = boost::make_filtered_graph(graph, filter);

    const auto mapping = specfem::mesh_entity::element(
        total_quadrature_points.ngllz, total_quadrature_points.nglly,
        total_quadrature_points.ngllx);

    // Validate index mapping and coordinates against expected mesh
    for (int e = 0; e < points.nspec; e++) {
      for (const auto edge :
           boost::make_iterator_range(boost::out_edges(e, fg))) {
        const int neighbor = boost::target(edge, graph);
        const auto orientation1 = fg[edge].orientation;
        const auto other_edge = boost::edge(neighbor, e, graph).first;
        const auto orientation2 = fg[other_edge].orientation;

        const auto connections = specfem::connections::connection_mapping(
            points.ngllz, points.nglly, points.ngllx,
            Kokkos::subview(expected_mesh.control_nodes.control_node_index, e,
                            Kokkos::ALL),
            Kokkos::subview(expected_mesh.control_nodes.control_node_index,
                            neighbor, Kokkos::ALL));

        if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::corners,
                                           orientation1)) {
          const auto [iz, iy, ix] = mapping.map_coordinates(orientation1);
          const auto [mapped_iz, mapped_iy, mapped_ix] =
              connections.map_coordinates(orientation1, orientation2);
          const int global_index_1 = points.h_index_mapping(e, iz, iy, ix);
          const int global_index_2 =
              points.h_index_mapping(neighbor, mapped_iz, mapped_iy, mapped_ix);
          ASSERT_EQ(global_index_1, global_index_2)
              << "Global index mismatch between element " << e
              << " and neighbor " << neighbor << " at corner on orientation "
              << specfem::mesh_entity::dim3::to_string(orientation1) << ". "
              << "Expected: " << global_index_2 << ", "
              << "Got: " << global_index_1 << std::endl;
          continue; // Corner test complete
        }

        const int npoints =
            mapping.number_of_points_on_orientation(orientation1);
        for (int p = 0; p < npoints; p++) {
          const auto [iz, iy, ix] = mapping.map_coordinates(orientation1, p);
          const auto [mapped_iz, mapped_iy, mapped_ix] =
              connections.map_coordinates(orientation1, orientation2, iz, iy,
                                          ix);
          const int global_index_1 = points.h_index_mapping(e, iz, iy, ix);
          const int global_index_2 =
              points.h_index_mapping(neighbor, mapped_iz, mapped_iy, mapped_ix);
          ASSERT_EQ(global_index_1, global_index_2)
              << "Global index mismatch between element " << e
              << " and neighbor " << neighbor << " at point " << p
              << " on orientation "
              << specfem::mesh_entity::dim3::to_string(orientation1) << ". "
              << "Expected: " << global_index_2 << ", "
              << "Got: " << global_index_1 << std::endl;
        }
      }
    }

    SUCCEED() << "All points mapping and coordinates are correct." << std::endl;
    return;
  }
};

} // namespace specfem::assembly_test

using namespace specfem::assembly_test;

const std::unordered_map<std::string, ExpectedMapping> expected_mappings = {
  { "EightNodeElastic",
    // Total quadrature points:
    //    Interior points: 8 elements * 3 * 3 * 3 = 216
    //    Face points: 36 faces * 3 * 3 = 324
    //    Edge points: 54 edges * 3 = 162
    //    Corner points: 27 corners * 1 = 27
    //    Total = 216 + 324 + 162 + 27 = 729
    ExpectedMapping(TotalQuadraturePoints(8, 5, 5, 5, 729),
                    "data/dim3/EightNodeElastic/database.bin") }
};

TEST_P(Assembly3DTest, Points) {
  const auto &param_name = GetParam();
  if (expected_mappings.find(param_name) == expected_mappings.end()) {
    GTEST_SKIP() << "No expected mapping defined for test case: " << param_name
                 << std::endl;
    return;
  }
  const auto &assembly = getAssembly();
  const auto &points = assembly.mesh;

  const auto &expected_mapping = expected_mappings.at(param_name);

  expected_mapping.check(points);
}
