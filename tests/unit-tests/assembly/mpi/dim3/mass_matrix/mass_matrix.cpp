/**
 * @file mass_matrix.cpp
 * @brief Unit test for MPI consistency of the assembled mass matrix in 3D.
 *
 * Builds the mass matrix through the production solver path (per-medium
 * outer/inner compute with overlapped cross-rank exchange, then inversion),
 * then verifies that every shared boundary node has the identical assembled
 * inverse mass value on all ranks that share it. Uses the same MPI exchange
 * pattern as the CommunicationPattern test: pack local values, send to
 * neighbor, receive neighbor's values, compare against local unpack mapping.
 */

#include "SPECFEM_Environment.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/solver/impl/update_medium.hpp"
#include "specfem/solver/mpi_buffers.hpp"
#include "specfem/tags.hpp"
#include <gtest/gtest.h>
#include <type_traits>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class MassMatrixMPI3DTest : public ::testing::TestWithParam<std::string> {
protected:
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;

  specfem::mesh::mesh<dimension> mesh;
  specfem::assembly::assembly<dimension> assembly;

  void SetUp() override {
    if (!SPECFEMEnvironment::IsMPISizeValid()) {
      GTEST_SKIP() << SPECFEMEnvironment::GetMPISizeError();
    }
    if (!specfem::MPI::is_active()) {
      GTEST_SKIP() << "Rank " << specfem::MPI::get_rank()
                   << " outside participating range.";
    }

    const auto &folder = GetParam();
    const std::string database = "data/mpi/dim3/" + folder + "/Database.bin";
    const auto mpi_database = specfem::MPI::format_proc_filename(database);
    mesh =
        specfem::io::read_3d_mesh(mpi_database, specfem::attenuation::Setup{});

    const auto quadratures =
        specfem::quadrature::quadratures(specfem::quadrature::gll::gll{});

    std::vector<std::shared_ptr<specfem::sources::source<dimension>>> sources;
    const std::vector<std::shared_ptr<specfem::receivers::receiver<dimension>>>
        receivers;
    const std::vector<specfem::enums::wavefield> stypes;

    assembly = specfem::assembly::assembly<dimension>(
        mesh, quadratures, sources, receivers, stypes,
        /* t0 */ 0.0, /* dt */ 0.001,
        /* max_timesteps */ 1, /* max_sig_step */ 1,
        /* nsteps_between_samples */ 1, specfem::simulation::type::forward,
        /* allocate_boundary_values */ false,
        /* property_reader */ nullptr);

    // Build the mass matrix through the production solver path: per-medium
    // outer/inner compute with overlapped cross-rank exchange, then invert.
    constexpr auto forward = specfem::simulation::field_type::forward;
    auto mpi_buffers = specfem::solver::make_mpi_buffers(
        assembly, specfem::simulation::type::forward);

    specfem::tag_dispatch::for_each(
        specfem::tag_dispatch::dimension_set<dimension>{} *
            MEDIUM_SET(acoustic, elastic, elastic_psv, elastic_sh, poroelastic,
                       elastic_psv_t),
        [&]<typename ElementTags>() {
          specfem::solver::impl::init_medium_mass<
              5, specfem::tags::expand<ElementTags, forward>>(
              assembly, mpi_buffers, assembly.dt);
        });
  }

  void TearDown() override {}
};

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

TEST_P(MassMatrixMPI3DTest, MassMatrixMPIConsistency) {
  assembly.fields.copy_to_host();

  const auto forward_field = assembly.fields.template get_simulation_field<
      specfem::simulation::field_type::forward>();

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim3) * MEDIUM_SET(elastic, acoustic),
      [&]<typename ElementTags>() {
        constexpr auto medium_tag = ElementTags::medium_tag;
        using MediumTags =
            specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                                medium_tag>;

        const auto &field_impl = forward_field.template get_field<medium_tag>();
        const auto h_mass = field_impl.get_host_mass_inverse();
        constexpr int ncomp =
            std::remove_cvref_t<decltype(field_impl)>::components;

        const auto &patterns =
            assembly.mpi_interfaces.forward_communication_patterns
                .template get<MediumTags>();

        const MPI_Comm comm = specfem::MPI::communicator();
        const int my_rank = specfem::MPI::get_rank();
        const int n_patterns = static_cast<int>(patterns.size());

        // Post non-blocking receives, then send, then wait
        std::unordered_map<unsigned int, std::vector<double>> send_bufs;
        std::unordered_map<unsigned int, std::vector<double>> recv_bufs;
        std::vector<MPI_Request> recv_reqs;
        recv_reqs.reserve(n_patterns);

        for (const auto &[neighbor, pattern] : patterns) {
          const int n_send = pattern.pack.nglob;
          const int n_recv = pattern.unpack.nglob;

          // Pack local assembled mass values for sending
          auto &sbuf = send_bufs[neighbor];
          sbuf.resize(ncomp * n_send);
          for (int i = 0; i < n_send; i++) {
            const int iglob = pattern.pack.h_mapping(i);
            for (int c = 0; c < ncomp; c++)
              sbuf[ncomp * i + c] = h_mass(iglob, c);
          }

          // Post receive for neighbor's packed values
          auto &rbuf = recv_bufs[neighbor];
          rbuf.resize(ncomp * n_recv);
          const int neighbor_rank = static_cast<int>(neighbor);
          const int recv_tag = neighbor_rank * 10000 + my_rank;
          MPI_Request req;
          SPECFEM_MPI_SAFECALL(MPI_Irecv(rbuf.data(), ncomp * n_recv,
                                         MPI_DOUBLE, neighbor_rank, recv_tag,
                                         comm, &req));
          recv_reqs.push_back(req);
        }

        // Send packed values to each neighbor
        for (const auto &[neighbor, pattern] : patterns) {
          const int neighbor_rank = static_cast<int>(neighbor);
          const int send_tag = my_rank * 10000 + neighbor_rank;
          SPECFEM_MPI_SAFECALL(MPI_Send(send_bufs[neighbor].data(),
                                        ncomp * pattern.pack.nglob, MPI_DOUBLE,
                                        neighbor_rank, send_tag, comm));
        }

        // Wait for all receives
        SPECFEM_MPI_SAFECALL(
            MPI_Waitall(n_patterns, recv_reqs.data(), MPI_STATUSES_IGNORE));

        // Compare: received neighbor values must match local assembled values
        for (const auto &[neighbor, pattern] : patterns) {
          const auto &rbuf = recv_bufs[neighbor];
          for (int i = 0; i < pattern.unpack.nglob; i++) {
            const int iglob = pattern.unpack.h_mapping(i);
            for (int c = 0; c < ncomp; c++) {
              const double local_val = h_mass(iglob, c);
              const double neighbor_val = rbuf[ncomp * i + c];
              const double tol = 1e-10 * std::max(std::abs(local_val), 1.0);
              EXPECT_NEAR(neighbor_val, local_val, tol)
                  << "Inverse mass mismatch at shared node iglob=" << iglob
                  << " component=" << c << " rank=" << my_rank
                  << " neighbor=" << neighbor;
            }
          }
        }
      });
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(MassMatrixMPI3DTests, MassMatrixMPI3DTest,
                         ::testing::Values("HomogeneousMediumMPI4x4"));
