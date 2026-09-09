#include "specfem/attenuation.hpp"
#include "specfem/io.hpp"
#include "specfem/mpi.hpp"

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

  EXPECT_EQ(globe.format_version, 2);
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
  EXPECT_TRUE(globe.mpi_interfaces.empty());
  EXPECT_TRUE(mesh.adjacency_graph.mpi_connections().empty());
}

} // namespace globe_mesh_test_impl

TEST(ReadGlobeMeshTests, GlobalSmallMesh) { globe_mesh_test_impl::check(); }
