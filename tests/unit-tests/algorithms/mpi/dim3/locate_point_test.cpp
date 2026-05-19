// MPI tests for specfem::algorithms::locate_point (3D)
//
// Domain (HomogeneousMediumMPI4x4):
//   x ∈ [0, 100000] m (longitude, NEX_XI=4 → element width ≈ 25000 m)
//   y ∈ [0,  80000] m (latitude,  NEX_ETA=4 → element depth ≈ 20000 m)
//   z ∈ [0,  60000] m (depth,     NZ=1      → full layer)
//   NGNOD=8, 2×2 MPI decomposition (4 processes total).

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
constexpr auto dim3 = specfem::element::dimension_tag::dim3;
using GlobalCoord = specfem::point::global_coordinates<dim3>;

// Points spread across the domain [0,100000]×[0,80000]×[0,60000].
// Element sizes ≈ 25000×20000×60000 m.  Points are placed at element
// centroids to avoid boundary ambiguity across MPI partitions.
const std::vector<GlobalCoord> TEST_POINTS = {
  { 12500.0, 10000.0, -30000.0 }, // element (0,0,0) centroid
  { 37500.0, 10000.0, -30000.0 }, // element (1,0,0) centroid
  { 62500.0, 10000.0, -30000.0 }, // element (2,0,0) centroid
  { 87500.0, 10000.0, -30000.0 }, // element (3,0,0) centroid
  { 12500.0, 50000.0, -30000.0 }, // element (0,2,0) centroid
  { 37500.0, 50000.0, -30000.0 }, // element (1,2,0) centroid
  { 62500.0, 50000.0, -30000.0 }, // element (2,2,0) centroid
  { 87500.0, 50000.0, -30000.0 }, // element (3,2,0) centroid
};
} // namespace

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class LocatePointMPI3DTest : public ::testing::TestWithParam<std::string> {
protected:
  specfem::assembly::mesh<dim3> assembly_mesh;

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
        "data/mpi/dim3/" + GetParam() + "/Database.bin";
    const auto proc_db = specfem::MPI::format_proc_filename(database);

    auto mesh_data =
        specfem::io::read_3d_mesh(proc_db, specfem::attenuation::Setup{});

    specfem::quadrature::gll::gll gll(0.0, 0.0, 5);
    specfem::quadrature::quadratures quadratures(gll);

    constexpr int ngll = 5;
    assembly_mesh = specfem::assembly::mesh<dim3>(
        mesh_data.nspec, mesh_data.control_nodes.ngnod, ngll, ngll, ngll,
        mesh_data.tags, mesh_data.adjacency_graph, mesh_data.control_nodes,
        quadratures);
  }
};

// ---------------------------------------------------------------------------
// Test: every point has a valid owning rank in [0, nproc)
// ---------------------------------------------------------------------------
TEST_P(LocatePointMPI3DTest, EachPointOwnedByExactlyOneRank) {
  const auto [lcoords, owners] =
      specfem::algorithms::locate_point(TEST_POINTS, assembly_mesh);

  const int nproc = specfem::MPI::get_size();
  const int npts = static_cast<int>(TEST_POINTS.size());

  for (int i = 0; i < npts; ++i) {
    EXPECT_GE(owners[i], 0)
        << "Point " << i << " (" << TEST_POINTS[i].x << ", " << TEST_POINTS[i].y
        << ", " << TEST_POINTS[i].z
        << ") has no owner (partition_index = " << owners[i] << ")";
    EXPECT_LT(owners[i], nproc) << "Point " << i << " owner rank " << owners[i]
                                << " exceeds nproc-1 = " << nproc - 1;
  }

  SPECFEM_MPI_SAFECALL(MPI_Barrier(specfem::MPI::communicator()));
}

// ---------------------------------------------------------------------------
// Test: the owning rank stores valid inside local coordinates
// ---------------------------------------------------------------------------
TEST_P(LocatePointMPI3DTest, OwnerHasInsideLocalCoords) {
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
    EXPECT_LE(std::abs(lcoords[i].eta), type_real(1) + tol)
        << "eta out of [-1,1] for point " << i;
    EXPECT_LE(std::abs(lcoords[i].gamma), type_real(1) + tol)
        << "gamma out of [-1,1] for point " << i;
  }

  SPECFEM_MPI_SAFECALL(MPI_Barrier(specfem::MPI::communicator()));
}

// ---------------------------------------------------------------------------
// Test: non-owning ranks store ispec = -1
// ---------------------------------------------------------------------------
TEST_P(LocatePointMPI3DTest, NonOwnerHasInvalidIspec) {
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
TEST_P(LocatePointMPI3DTest, BackProjectionAccuracy) {
  const auto [lcoords, owners] =
      specfem::algorithms::locate_point(TEST_POINTS, assembly_mesh);

  const int myrank = specfem::MPI::get_rank();
  const int npts = static_cast<int>(TEST_POINTS.size());
  // Allow up to 100 m round-trip error (element sizes are O(10000 m)).
  constexpr type_real tol = type_real(100);

  for (int i = 0; i < npts; ++i) {
    if (owners[i] != myrank)
      continue;

    const auto recovered = specfem::algorithms::locate_point_impl::locate_point(
        lcoords[i], assembly_mesh);
    const type_real dist = specfem::point::distance(TEST_POINTS[i], recovered);

    EXPECT_LE(dist, tol) << "Back-projection error " << dist << " m for point "
                         << i << " at (" << TEST_POINTS[i].x << ", "
                         << TEST_POINTS[i].y << ", " << TEST_POINTS[i].z << ")";
  }

  SPECFEM_MPI_SAFECALL(MPI_Barrier(specfem::MPI::communicator()));
}

// ---------------------------------------------------------------------------
// Instantiation + main
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(LocatePointMPI3DTests, LocatePointMPI3DTest,
                         ::testing::Values("HomogeneousMediumMPI4x4"));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment(4));
  return RUN_ALL_TESTS();
}
