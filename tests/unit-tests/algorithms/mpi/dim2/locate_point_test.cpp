// MPI tests for specfem::algorithms::locate_point
//
// Domain (HomogeneousMediumMPI4Procs): x ∈ [0, 4000] m, z ∈ [0, 3000] m
// 80×60 spectral elements, NGNOD=9, 4 MPI processes.

#include "SPECFEM_Environment.hpp"

#include "specfem/algorithms/locate_point.hpp"
#include "specfem/algorithms/locate_point/locate_point_impl.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {
constexpr auto dim2 = specfem::element::dimension_tag::dim2;
using GlobalCoord = specfem::point::global_coordinates<dim2>;

// Points distributed across the domain [0,4000]×[0,3000].
// Element size ≈ 50×50 m; points are placed away from element edges to
// avoid ambiguity when the same GLL node is shared by two elements.
const std::vector<GlobalCoord> TEST_POINTS = {
  { 200.0, 150.0 },   // lower-left quadrant
  { 2000.0, 1500.0 }, // geometric centre
  { 3800.0, 2850.0 }, // upper-right quadrant
  { 1000.0, 1500.0 }, // near expected partition boundary (left half)
  { 3000.0, 1500.0 }, // near expected partition boundary (right half)
  { 500.0, 2500.0 },  // upper-left area
  { 3500.0, 500.0 },  // lower-right area
  { 1600.0, 800.0 },  // arbitrary interior point
};
} // namespace

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class LocatePointMPI2DTest : public ::testing::TestWithParam<std::string> {
protected:
  specfem::assembly::mesh<dim2> assembly_mesh;

  void SetUp() override {
    if (!SPECFEMEnvironment::IsMPISizeValid()) {
      GTEST_SKIP() << SPECFEMEnvironment::GetMPISizeError();
    }
    if (specfem::MPI::communicator() == MPI_COMM_NULL) {
      GTEST_SKIP() << "Test designed for 4 processes. Rank "
                   << specfem::MPI::get_rank()
                   << " is outside the participating range [0-3].";
    }

    const std::string database =
        "data/mpi/dim2/" + GetParam() + "/Database.bin";
    const auto proc_db = specfem::MPI::format_proc_filename(database);

    auto mesh_data = specfem::io::read_2d_mesh(
        proc_db, specfem::enums::elastic_wave::psv,
        specfem::enums::electromagnetic_wave::te, /*attenuation=*/false);

    specfem::quadrature::gll::gll gll(0.0, 0.0, 5);
    specfem::quadrature::quadratures quadratures(gll);

    assembly_mesh =
        specfem::assembly::mesh<dim2>(mesh_data.tags, mesh_data.control_nodes,
                                      quadratures, mesh_data.adjacency_graph);
  }
};

// ---------------------------------------------------------------------------
// Test: every point has a valid owning rank in [0, nproc)
// ---------------------------------------------------------------------------
TEST_P(LocatePointMPI2DTest, EachPointOwnedByExactlyOneRank) {
  const auto [lcoords, owners] =
      specfem::algorithms::locate_point(TEST_POINTS, assembly_mesh);

  const int nproc = specfem::MPI::get_size();
  const int npts = static_cast<int>(TEST_POINTS.size());

  for (int i = 0; i < npts; ++i) {
    EXPECT_GE(owners[i], 0)
        << "Point " << i << " (" << TEST_POINTS[i].x << ", " << TEST_POINTS[i].z
        << ") has no owner (partition_index = " << owners[i] << ")";
    EXPECT_LT(owners[i], nproc) << "Point " << i << " owner rank " << owners[i]
                                << " exceeds nproc-1 = " << nproc - 1;
  }

  SPECFEM_MPI_SAFECALL(MPI_Barrier(specfem::MPI::communicator()));
}

// ---------------------------------------------------------------------------
// Test: the owning rank stores valid inside local coordinates
// ---------------------------------------------------------------------------
TEST_P(LocatePointMPI2DTest, OwnerHasInsideLocalCoords) {
  const auto [lcoords, owners] =
      specfem::algorithms::locate_point(TEST_POINTS, assembly_mesh);

  const int myrank = specfem::MPI::get_rank();
  const int npts = static_cast<int>(TEST_POINTS.size());
  constexpr type_real tol = type_real(1e-6);

  for (int i = 0; i < npts; ++i) {
    if (owners[i] != myrank)
      continue;

    EXPECT_GE(lcoords[i].ispec, 0)
        << "Owning rank has ispec = -1 for point " << i;
    EXPECT_LE(std::abs(lcoords[i].xi), type_real(1) + tol)
        << "xi out of [-1,1] for point " << i;
    EXPECT_LE(std::abs(lcoords[i].gamma), type_real(1) + tol)
        << "gamma out of [-1,1] for point " << i;
  }

  SPECFEM_MPI_SAFECALL(MPI_Barrier(specfem::MPI::communicator()));
}

// ---------------------------------------------------------------------------
// Test: non-owning ranks store ispec = -1
// ---------------------------------------------------------------------------
TEST_P(LocatePointMPI2DTest, NonOwnerHasInvalidIspec) {
  const auto [lcoords, owners] =
      specfem::algorithms::locate_point(TEST_POINTS, assembly_mesh);

  const int myrank = specfem::MPI::get_rank();
  const int npts = static_cast<int>(TEST_POINTS.size());

  for (int i = 0; i < npts; ++i) {
    if (owners[i] == myrank)
      continue;

    EXPECT_EQ(lcoords[i].ispec, -1)
        << "Non-owning rank " << myrank << " has ispec = " << lcoords[i].ispec
        << " for point " << i << " (owned by rank " << owners[i] << ")";
  }

  SPECFEM_MPI_SAFECALL(MPI_Barrier(specfem::MPI::communicator()));
}

// ---------------------------------------------------------------------------
// Test: back-projecting the returned local coordinate recovers the original
//       global coordinate within a tight tolerance.
// ---------------------------------------------------------------------------
TEST_P(LocatePointMPI2DTest, BackProjectionAccuracy) {
  const auto [lcoords, owners] =
      specfem::algorithms::locate_point(TEST_POINTS, assembly_mesh);

  const int myrank = specfem::MPI::get_rank();
  const int npts = static_cast<int>(TEST_POINTS.size());
  // Element size ≈ 50 m; allow up to 1 m round-trip error.
  constexpr type_real tol = type_real(1);

  for (int i = 0; i < npts; ++i) {
    if (owners[i] != myrank)
      continue;

    const auto recovered = specfem::algorithms::locate_point_impl::locate_point(
        lcoords[i], assembly_mesh);
    const type_real dist = specfem::point::distance(TEST_POINTS[i], recovered);

    EXPECT_LE(dist, tol) << "Back-projection error " << dist << " m for point "
                         << i << " at (" << TEST_POINTS[i].x << ", "
                         << TEST_POINTS[i].z << ")";
  }

  SPECFEM_MPI_SAFECALL(MPI_Barrier(specfem::MPI::communicator()));
}

// ---------------------------------------------------------------------------
// Instantiation + main
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(LocatePointMPI2DTests, LocatePointMPI2DTest,
                         ::testing::Values("HomogeneousMediumMPI4Procs"));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment(4));
  return RUN_ALL_TESTS();
}
