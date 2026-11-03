// MPI interfaces of a six-chunk globe mesh (issue #2051).
//
// One rank per cubed-sphere chunk. The databases are not committed (~47 MB);
// the globe_small_6rank_mesh CTest fixture generates them into
// generated/dim3_globe/global_small_6rank/ before this test runs. See
// tests/unit-tests/data/dim3_globe/global_small_6rank/README.md.

#include "SPECFEM_Environment.hpp"
#include "globe_fixture.hpp"
#include "specfem/element/to_string.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tag_dispatch/for_each.hpp"
#include "specfem/tags.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace globe_6rank_test_impl {

constexpr auto dimension = specfem::element::dimension_tag::dim3;
constexpr auto forward = specfem::simulation::field_type::forward;

const std::string databases_mpi =
    "generated/dim3_globe/global_small_6rank/DATABASES_MPI";

/** @brief Quantized position of a GLL point. */
using CellKey = std::array<std::int64_t, 3>;

struct CellKeyHash {
  std::size_t operator()(const CellKey &key) const {
    std::size_t h = 1469598103934665603ULL;
    for (const auto v : key) {
      h ^= static_cast<std::size_t>(v) + 0x9e3779b97f4a7c15ULL + (h << 6) +
           (h >> 2);
    }
    return h;
  }
};

/**
 * @brief Storage for the fixture shared by every test in the process.
 *
 * Released by GlobeMPI3DTest::TearDownTestSuite, before Kokkos is finalized;
 * a function-local static would free its views after that.
 */
std::unique_ptr<specfem::test_configuration::GlobeMPI3D> &fixture_storage() {
  static std::unique_ptr<specfem::test_configuration::GlobeMPI3D> instance;
  return instance;
}

/**
 * @brief Build the fixture once per process; every test reads it.
 *
 * CTest runs each case in its own launch, so this only saves work when the
 * binary is run by hand with several cases selected.
 */
const specfem::test_configuration::GlobeMPI3D &fixture() {
  auto &instance = fixture_storage();
  if (!instance) {
    instance = std::make_unique<specfem::test_configuration::GlobeMPI3D>(
        databases_mpi);
  }
  return *instance;
}

/**
 * @brief Sum a per-iglob scalar across ranks through the exchange patterns.
 *
 * Sends the local value at each packed point and adds what each neighbor
 * sends at the matching unpacked point -- the same pairing the solver uses
 * for field assembly.
 *
 * @tparam MediumTag Medium whose patterns and iglob numbering are used
 * @param local Per-iglob values on this rank (size: nglob of the medium)
 * @return Values after cross-rank summation
 */
template <specfem::element::medium_tag MediumTag>
std::vector<double> exchange_sum(const std::vector<double> &local) {
  using MediumTags = specfem::tags::Tags<dimension, MediumTag>;
  const auto &patterns = fixture()
                             .mpi_interfaces.forward_communication_patterns
                             .template get<MediumTags>();
  const MPI_Comm comm = specfem::MPI::communicator();
  const int my_rank = specfem::MPI::get_rank();

  std::unordered_map<unsigned int, std::vector<double>> send, recv;
  std::vector<MPI_Request> requests;
  for (const auto &[neighbor, pattern] : patterns) {
    auto &rbuf = recv[neighbor];
    rbuf.resize(pattern.unpack.nglob);
    MPI_Request request;
    SPECFEM_MPI_SAFECALL(MPI_Irecv(rbuf.data(), static_cast<int>(rbuf.size()),
                                   MPI_DOUBLE, static_cast<int>(neighbor),
                                   static_cast<int>(neighbor) * 100 + my_rank,
                                   comm, &request));
    requests.push_back(request);
  }
  for (const auto &[neighbor, pattern] : patterns) {
    auto &sbuf = send[neighbor];
    sbuf.resize(pattern.pack.nglob);
    for (unsigned int i = 0; i < pattern.pack.nglob; ++i) {
      sbuf[i] = local[pattern.pack.h_mapping(i)];
    }
    SPECFEM_MPI_SAFECALL(MPI_Send(sbuf.data(), static_cast<int>(sbuf.size()),
                                  MPI_DOUBLE, static_cast<int>(neighbor),
                                  my_rank * 100 + static_cast<int>(neighbor),
                                  comm));
  }
  SPECFEM_MPI_SAFECALL(MPI_Waitall(static_cast<int>(requests.size()),
                                   requests.data(), MPI_STATUSES_IGNORE));

  std::vector<double> total = local;
  for (const auto &[neighbor, pattern] : patterns) {
    const auto &rbuf = recv.at(neighbor);
    for (unsigned int i = 0; i < pattern.unpack.nglob; ++i) {
      total[pattern.unpack.h_mapping(i)] += rbuf[i];
    }
  }
  return total;
}

/**
 * @brief Number of ranks holding each of this rank's points, from geometry.
 *
 * Independent of the exchange patterns: every rank gathers the quantized
 * coordinates of all GLL points on its outer elements of @p MediumTag (a
 * shared point always lies on one), and a point's share count is the number
 * of ranks holding a point in the same or an adjacent cell. Points of
 * different media are counted separately, so a CMB point is never shared
 * between the fluid and solid counts.
 *
 * @tparam MediumTag Medium whose iglob numbering is used
 * @return Share count per iglob (1 for points on no other rank)
 */
template <specfem::element::medium_tag MediumTag>
std::vector<int> geometric_share_count() {
  const auto &f = fixture();
  const auto &field = f.fields.template get_simulation_field<forward>();
  const int nglob = field.template get_nglob<MediumTag>();
  const auto &grid = f.assembly_mesh.element_grid;
  const auto &coord = f.assembly_mesh.h_coord;
  const int nspec = f.assembly_mesh.nspec;
  const MPI_Comm comm = specfem::MPI::communicator();
  const int nranks = specfem::MPI::get_size();
  const int my_rank = specfem::MPI::get_rank();

  // Quantum: 1e-6 of the radius (~6 m). The closest distinct GLL points in
  // this mesh are ~75 m apart, and coordinates are single precision in the
  // default build (~0.5 m ulp at the surface), so copies of one point land in
  // the same or an adjacent cell while distinct points never do.
  double scale = 0.0;
  for (int ispec = 0; ispec < nspec; ++ispec) {
    for (int c = 0; c < 3; ++c) {
      scale = std::max(scale,
                       std::abs(static_cast<double>(coord(ispec, 0, 0, 0, c))));
    }
  }
  SPECFEM_MPI_SAFECALL(
      MPI_Allreduce(MPI_IN_PLACE, &scale, 1, MPI_DOUBLE, MPI_MAX, comm));
  const double quantum = 1e-6 * scale;

  std::vector<CellKey> key_of(nglob);
  std::vector<bool> candidate(nglob, false);
  for (int ispec = 0; ispec < nspec; ++ispec) {
    if (f.element_types.get_medium_tag(ispec) != MediumTag ||
        f.element_types.get_mpi_tag(ispec) !=
            specfem::element::mpi_tag::outer) {
      continue;
    }
    for (int iz = 0; iz < grid.ngllz; ++iz) {
      for (int iy = 0; iy < grid.nglly; ++iy) {
        for (int ix = 0; ix < grid.ngllx; ++ix) {
          const int iglob =
              field.template get_iglob<false, MediumTag>(ispec, iz, iy, ix);
          if (candidate[iglob])
            continue;
          candidate[iglob] = true;
          for (int c = 0; c < 3; ++c) {
            key_of[iglob][c] = static_cast<std::int64_t>(std::llround(
                static_cast<double>(coord(ispec, iz, iy, ix, c)) / quantum));
          }
        }
      }
    }
  }

  std::vector<std::int64_t> mine;
  for (int iglob = 0; iglob < nglob; ++iglob) {
    if (candidate[iglob]) {
      mine.insert(mine.end(), key_of[iglob].begin(), key_of[iglob].end());
    }
  }
  int count = static_cast<int>(mine.size());
  std::vector<int> counts(nranks), displs(nranks, 0);
  SPECFEM_MPI_SAFECALL(
      MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm));
  for (int r = 1; r < nranks; ++r)
    displs[r] = displs[r - 1] + counts[r - 1];
  std::vector<std::int64_t> all(displs.back() + counts.back());
  SPECFEM_MPI_SAFECALL(MPI_Allgatherv(mine.data(), count, MPI_INT64_T,
                                      all.data(), counts.data(), displs.data(),
                                      MPI_INT64_T, comm));

  std::unordered_map<CellKey, std::uint32_t, CellKeyHash> ranks_at;
  for (int r = 0; r < nranks; ++r) {
    for (int i = displs[r]; i < displs[r] + counts[r]; i += 3) {
      ranks_at[{ all[i], all[i + 1], all[i + 2] }] |= (1u << r);
    }
  }

  std::vector<int> share(nglob, 1);
  for (int iglob = 0; iglob < nglob; ++iglob) {
    if (!candidate[iglob])
      continue;
    std::uint32_t mask = 0;
    const auto &k = key_of[iglob];
    for (int dz = -1; dz <= 1; ++dz)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const auto it = ranks_at.find({ k[0] + dx, k[1] + dy, k[2] + dz });
          if (it != ranks_at.end())
            mask |= it->second;
        }
    EXPECT_TRUE(mask & (1u << my_rank));
    share[iglob] = std::popcount(mask);
  }
  return share;
}

} // namespace globe_6rank_test_impl

class GlobeMPI3DTest : public ::testing::Test {
protected:
  static void TearDownTestSuite() {
    globe_6rank_test_impl::fixture_storage().reset();
  }

  void SetUp() override {
    if (!SPECFEMEnvironment::IsMPISizeValid()) {
      GTEST_SKIP() << SPECFEMEnvironment::GetMPISizeError();
    }
  }
};

// Every element is either inner or outer, and outer is exactly "touches an MPI
// connection" -- which for a six-chunk mesh is non-empty on every rank.
TEST_F(GlobeMPI3DTest, InnerOuterCoverNspec) {
  const auto &f = globe_6rank_test_impl::fixture();
  const int nspec = f.mesh.nspec;

  std::unordered_set<std::size_t> touching;
  for (const auto &conn : f.mesh.adjacency_graph.mpi_connections()) {
    touching.insert(conn.local_index);
  }

  int ninner = 0;
  int nouter = 0;
  for (int ispec = 0; ispec < nspec; ++ispec) {
    if (f.element_types.get_mpi_tag(ispec) ==
        specfem::element::mpi_tag::outer) {
      ++nouter;
    } else {
      ++ninner;
    }
  }
  EXPECT_EQ(ninner + nouter, nspec);
  EXPECT_EQ(nouter, static_cast<int>(touching.size()));
  EXPECT_GT(nouter, 0);
  EXPECT_GT(ninner, 0);

  // Translated through mesh_to_compute: the compute-order element of every
  // mesh-order MPI element must be tagged outer.
  for (const auto ispec_mesh : touching) {
    const int ispec =
        f.assembly_mesh.h_mesh_to_compute(static_cast<int>(ispec_mesh));
    EXPECT_EQ(f.element_types.get_mpi_tag(ispec),
              specfem::element::mpi_tag::outer)
        << "mesh element " << ispec_mesh;
  }
}

// Every element reached by an exchange pattern is outer and of the pattern's
// medium, every exchanged iglob is in that medium's numbering, and both media
// exchange on every rank (the outer core reaches the chunk boundaries).
TEST_F(GlobeMPI3DTest, MediaExchangeSeparately) {
  const auto &f = globe_6rank_test_impl::fixture();
  const auto &field =
      f.fields.template get_simulation_field<globe_6rank_test_impl::forward>();

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim3) * MEDIUM_SET(acoustic, elastic),
      [&]<typename TagsType>() {
        constexpr auto medium = TagsType::medium_tag;
        const auto &patterns =
            f.mpi_interfaces.forward_communication_patterns
                .template get<specfem::tags::Tags<
                    globe_6rank_test_impl::dimension, medium>>();
        const int nglob = field.template get_nglob<medium>();
        const std::string name = specfem::element::to_string(medium);
        EXPECT_FALSE(patterns.empty()) << name << " has no exchange";

        for (const auto &[neighbor, pattern] : patterns) {
          const auto &unpack = pattern.unpack;
          for (const auto &elements :
               { unpack.h_face_elements, unpack.h_edge_elements,
                 unpack.h_corner_elements }) {
            for (std::size_t i = 0; i < elements.extent(0); ++i) {
              EXPECT_EQ(f.element_types.get_medium_tag(elements(i)), medium)
                  << name << " exchange with rank " << neighbor
                  << " reaches element " << elements(i);
              EXPECT_EQ(f.element_types.get_mpi_tag(elements(i)),
                        specfem::element::mpi_tag::outer);
            }
          }
          for (unsigned int i = 0; i < pattern.pack.nglob; ++i) {
            EXPECT_LT(pattern.pack.h_mapping(i), nglob) << name;
          }
          for (unsigned int i = 0; i < unpack.nglob; ++i) {
            EXPECT_LT(unpack.h_mapping(i), nglob) << name;
          }
        }
      });
}

// The neighbor relation is symmetric: rank a exchanges with b iff b with a.
TEST_F(GlobeMPI3DTest, NeighborsAreSymmetric) {
  const auto &f = globe_6rank_test_impl::fixture();
  const int nranks = specfem::MPI::get_size();
  const int my_rank = specfem::MPI::get_rank();

  std::vector<int> row(nranks, 0);
  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim3) * MEDIUM_SET(acoustic, elastic),
      [&]<typename TagsType>() {
        for (const auto &[neighbor, pattern] :
             f.mpi_interfaces.forward_communication_patterns
                 .template get<TagsType>()) {
          EXPECT_NE(static_cast<int>(neighbor), my_rank);
          row[neighbor] = 1;
        }
      });
  std::vector<int> matrix(nranks * nranks);
  SPECFEM_MPI_SAFECALL(MPI_Allgather(row.data(), nranks, MPI_INT, matrix.data(),
                                     nranks, MPI_INT,
                                     specfem::MPI::communicator()));
  for (int a = 0; a < nranks; ++a) {
    for (int b = 0; b < nranks; ++b) {
      EXPECT_EQ(matrix[a * nranks + b], matrix[b * nranks + a])
          << "ranks " << a << " and " << b;
    }
  }
}

// Every exchanged point lands on the same physical point: the coordinates a
// neighbor packs for slot i coincide with ours at the slot-i unpack target.
TEST_F(GlobeMPI3DTest, ExchangedPointsCoincide) {
  const auto &f = globe_6rank_test_impl::fixture();
  const MPI_Comm comm = specfem::MPI::communicator();
  const int my_rank = specfem::MPI::get_rank();

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim3) * MEDIUM_SET(acoustic, elastic),
      [&]<typename TagsType>() {
        constexpr auto medium = TagsType::medium_tag;
        const auto &field = f.fields.template get_simulation_field<
            globe_6rank_test_impl::forward>();
        const auto &grid = f.assembly_mesh.element_grid;
        const auto &coord = f.assembly_mesh.h_coord;

        std::vector<std::array<double, 3>> xyz(
            field.template get_nglob<medium>());
        double scale = 0.0;
        for (int ispec = 0; ispec < f.assembly_mesh.nspec; ++ispec) {
          if (f.element_types.get_medium_tag(ispec) != medium)
            continue;
          for (int iz = 0; iz < grid.ngllz; ++iz)
            for (int iy = 0; iy < grid.nglly; ++iy)
              for (int ix = 0; ix < grid.ngllx; ++ix) {
                auto &x = xyz[field.template get_iglob<false, medium>(ispec, iz,
                                                                      iy, ix)];
                for (int c = 0; c < 3; ++c) {
                  x[c] = coord(ispec, iz, iy, ix, c);
                  scale = std::max(scale, std::abs(x[c]));
                }
              }
        }
        // Distinct points are >= ~75 m apart; single-precision copies of one
        // point differ by about a metre.
        const double tolerance = 1e-6 * scale * 4;

        const auto &patterns = f.mpi_interfaces.forward_communication_patterns
                                   .template get<TagsType>();
        std::unordered_map<unsigned int, std::vector<double>> send, recv;
        std::vector<MPI_Request> requests;
        for (const auto &[neighbor, pattern] : patterns) {
          auto &rbuf = recv[neighbor];
          rbuf.resize(3 * pattern.unpack.nglob);
          MPI_Request request;
          SPECFEM_MPI_SAFECALL(MPI_Irecv(
              rbuf.data(), static_cast<int>(rbuf.size()), MPI_DOUBLE,
              static_cast<int>(neighbor),
              static_cast<int>(neighbor) * 100 + my_rank, comm, &request));
          requests.push_back(request);
        }
        for (const auto &[neighbor, pattern] : patterns) {
          auto &sbuf = send[neighbor];
          for (unsigned int i = 0; i < pattern.pack.nglob; ++i) {
            const auto &x = xyz[pattern.pack.h_mapping(i)];
            sbuf.insert(sbuf.end(), x.begin(), x.end());
          }
          SPECFEM_MPI_SAFECALL(
              MPI_Send(sbuf.data(), static_cast<int>(sbuf.size()), MPI_DOUBLE,
                       static_cast<int>(neighbor),
                       my_rank * 100 + static_cast<int>(neighbor), comm));
        }
        SPECFEM_MPI_SAFECALL(MPI_Waitall(static_cast<int>(requests.size()),
                                         requests.data(), MPI_STATUSES_IGNORE));

        for (const auto &[neighbor, pattern] : patterns) {
          const auto &rbuf = recv.at(neighbor);
          int nwrong = 0;
          for (unsigned int i = 0; i < pattern.unpack.nglob; ++i) {
            const auto &x = xyz[pattern.unpack.h_mapping(i)];
            double d2 = 0.0;
            double r_mine = 0.0;
            double r_theirs = 0.0;
            for (int c = 0; c < 3; ++c) {
              d2 += (x[c] - rbuf[3 * i + c]) * (x[c] - rbuf[3 * i + c]);
              r_mine += x[c] * x[c];
              r_theirs += rbuf[3 * i + c] * rbuf[3 * i + c];
            }
            if (std::sqrt(d2) > tolerance && ++nwrong <= 5) {
              ADD_FAILURE()
                  << specfem::element::to_string(medium) << " slot " << i
                  << " from rank " << neighbor << ": points " << std::sqrt(d2)
                  << " m apart (r = " << std::sqrt(r_mine) << " here, "
                  << std::sqrt(r_theirs) << " there)";
            }
          }
          EXPECT_EQ(nwrong, 0)
              << specfem::element::to_string(medium) << " rank " << my_rank
              << " <- rank " << neighbor << " of " << pattern.unpack.nglob;
        }
      });
}

// Assembling a field of ones without normalization gives each point the number
// of ranks that hold it, as counted from geometry.
TEST_F(GlobeMPI3DTest, PartitionOfUnity) {
  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim3) * MEDIUM_SET(acoustic, elastic),
      [&]<typename TagsType>() {
        constexpr auto medium = TagsType::medium_tag;
        const auto share =
            globe_6rank_test_impl::geometric_share_count<medium>();
        const auto total = globe_6rank_test_impl::exchange_sum<medium>(
            std::vector<double>(share.size(), 1.0));

        int nshared = 0;
        int nwrong = 0;
        for (std::size_t iglob = 0; iglob < share.size(); ++iglob) {
          nshared += share[iglob] > 1;
          if (total[iglob] != static_cast<double>(share[iglob]) &&
              ++nwrong <= 10) {
            ADD_FAILURE() << specfem::element::to_string(medium) << " iglob "
                          << iglob << ": assembled " << total[iglob]
                          << ", shared by " << share[iglob] << " ranks";
          }
        }
        EXPECT_EQ(nwrong, 0) << specfem::element::to_string(medium);
        EXPECT_GT(nshared, 0) << specfem::element::to_string(medium);
      });
}

// A constant field split across ranks by multiplicity reassembles to the
// constant everywhere.
TEST_F(GlobeMPI3DTest, ConstantFieldIsPreserved) {
  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim3) * MEDIUM_SET(acoustic, elastic),
      [&]<typename TagsType>() {
        constexpr auto medium = TagsType::medium_tag;
        const auto share =
            globe_6rank_test_impl::geometric_share_count<medium>();
        std::vector<double> local(share.size());
        for (std::size_t iglob = 0; iglob < share.size(); ++iglob) {
          local[iglob] = 1.0 / share[iglob];
        }
        const auto total = globe_6rank_test_impl::exchange_sum<medium>(local);

        int nwrong = 0;
        for (std::size_t iglob = 0; iglob < total.size(); ++iglob) {
          if (std::abs(total[iglob] - 1.0) > 1e-12 && ++nwrong <= 10) {
            ADD_FAILURE() << specfem::element::to_string(medium) << " iglob "
                          << iglob << ": " << total[iglob];
          }
        }
        EXPECT_EQ(nwrong, 0) << specfem::element::to_string(medium);
      });
}
