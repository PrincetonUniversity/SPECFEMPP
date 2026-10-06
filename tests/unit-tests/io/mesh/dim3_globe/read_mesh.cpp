#include "specfem/attenuation.hpp"
#include "specfem/element.hpp"
#include "specfem/element_connections.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/mpi.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <gtest/gtest.h>
#include <iomanip>
#include <sstream>
#include <string>

namespace globe_mesh_test_impl {

constexpr int nproc = 1;
const std::string database_directory =
    "data/dim3_globe/GlobalSmallMesh/DATABASES_MPI";

std::string database_path(const std::string &directory, const int rank) {
  std::ostringstream path;
  path << directory << "/proc" << std::setw(6) << std::setfill('0') << rank
       << "_specfempp_database.bin";
  return path.str();
}

void check() {
  const int rank = specfem::MPI::get_rank();
  ASSERT_EQ(specfem::MPI::get_size(), nproc);

  const auto mesh = specfem::io::read_globe_mesh(
      database_path(database_directory, rank), specfem::attenuation::Setup{});
  const auto &globe = mesh.globe;
  const auto &config = globe.model_config;

  EXPECT_EQ(globe.format_version, 5);
  EXPECT_EQ(mesh.control_nodes.ngnod, 27);
  EXPECT_GT(mesh.nspec, 0);
  EXPECT_GT(mesh.control_nodes.nnodes, 0);
  EXPECT_EQ(globe.nregions, 3);
  EXPECT_TRUE(globe.has_reference_geometry);
  EXPECT_EQ(config.model_name, "1D_isotropic_prem");
  EXPECT_EQ(config.nchunks, 1);
  EXPECT_EQ(config.nex_xi, 32);
  EXPECT_EQ(config.nex_eta, 32);
  EXPECT_TRUE(config.ellipticity);
  EXPECT_FALSE(config.topography);
  EXPECT_FALSE(config.gravity);
  EXPECT_FALSE(config.rotation);
  EXPECT_FALSE(config.attenuation);
  EXPECT_FALSE(config.oceans);
  EXPECT_EQ(globe.model_verification.codes.size(), 5);
  EXPECT_EQ(globe.model_verification.flags.size(), 16);
  EXPECT_FALSE(globe.free_surface.elements.empty());
  EXPECT_FALSE(globe.cmb.elements.empty());
  EXPECT_FALSE(globe.icb.elements.empty());
  EXPECT_TRUE(mesh.adjacency_graph.mpi_connections().empty());
}

/**
 * @brief Check that every medium-contrast face separates the regions the
 * globe's layering requires.
 *
 * The outer core is the only fluid region: it sits below the mantle at the CMB
 * and above the inner core at the ICB. So on every weakly-conforming face one
 * side is outer core, and the fluid element's face says which interface it is.
 */
void check_fluid_solid_regions() {
  using specfem::element::region_tag;
  using specfem::mesh_entity::dim3::type;

  const int rank = specfem::MPI::get_rank();
  const auto mesh = specfem::io::read_globe_mesh(
      database_path(database_directory, rank), specfem::attenuation::Setup{});
  const auto &context = mesh.globe.element_context;
  const auto &graph = mesh.adjacency_graph.local_connections();

  int cmb_faces = 0;
  int icb_faces = 0;
  for (const auto v : boost::make_iterator_range(boost::vertices(graph))) {
    for (const auto e :
         boost::make_iterator_range(boost::out_edges(v, graph))) {
      const auto &edge = graph[e];
      if (edge.connection !=
              specfem::element_connections::type::weakly_conforming ||
          !specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                          edge.orientation)) {
        continue;
      }
      // Each face is seen once from each side; count it from the fluid side.
      if (context[v].region != region_tag::outer_core) {
        EXPECT_EQ(context[boost::target(e, graph)].region,
                  region_tag::outer_core)
            << "element " << (v + 1)
            << " has a fluid-solid face with no outer-core neighbor";
        continue;
      }
      const auto solid_region = context[boost::target(e, graph)].region;
      if (edge.orientation == type::top) {
        ++cmb_faces;
        EXPECT_EQ(solid_region, region_tag::crust_mantle)
            << "outer-core element " << (v + 1);
      } else {
        EXPECT_EQ(edge.orientation, type::bottom)
            << "outer-core element " << (v + 1);
        ++icb_faces;
        EXPECT_EQ(solid_region, region_tag::inner_core)
            << "outer-core element " << (v + 1);
      }
    }
  }

  EXPECT_GT(cmb_faces, 0);
  EXPECT_GT(icb_faces, 0);

  // Recorded rather than asserted: these are the first measured face counts on
  // this path, and they belong in the test report so a future mesher change
  // that shifts them is visible rather than silent.
  ::testing::Test::RecordProperty("cmb_face_pairs", cmb_faces);
  ::testing::Test::RecordProperty("icb_face_pairs", icb_faces);
  ::testing::Test::RecordProperty(
      "database_cmb_faces", static_cast<int>(mesh.globe.cmb.elements.size()));
  ::testing::Test::RecordProperty(
      "database_icb_faces", static_cast<int>(mesh.globe.icb.elements.size()));
}

} // namespace globe_mesh_test_impl

TEST(ReadGlobeMeshTests, GlobalSmallMesh) { globe_mesh_test_impl::check(); }

TEST(ReadGlobeMeshTests, FluidSolidFacesSeparateExpectedRegions) {
  globe_mesh_test_impl::check_fluid_solid_regions();
}
