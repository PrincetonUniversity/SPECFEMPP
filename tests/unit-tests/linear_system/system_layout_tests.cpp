#include "../SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/io.hpp"
#include "specfem/linear_system/system_layout.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace system_layout_test {

constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;
constexpr auto elastic_tag = specfem::element::medium_tag::elastic;
constexpr int NGLL = 5;
constexpr int ncomp = 3;

using AssemblyType = specfem::assembly::assembly<dim3_tag>;
using LayoutTags =
    specfem::tags::Tags<dim3_tag, elastic_tag,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none>;
using LayoutType = specfem::linear_system::SystemLayout<LayoutTags>;

// Build a full assembly from a Newmark displacement-test dataset. Paths are
// relative to TEST_OUTPUT_DIR, where the displacement_tests data tree is
// linked (see SERIAL_LINK_DIRS in serial.cmake).
std::unique_ptr<AssemblyType> build_assembly_3d(const std::string &test_name) {
  const std::string test_path =
      "displacement_tests/Newmark/serial/dim3/" + test_name;

  specfem::runtime_configuration::setup setup(test_path +
                                              "/specfem_config.yaml");

  const auto database_filename = setup.get_databases();
  const auto &source_entries = setup.get_source_entries();
  const auto stations_node = setup.get_stations();
  const auto quadratures = setup.instantiate_quadrature();

  auto mesh = specfem::io::read_3d_mesh(database_filename,
                                        setup.get_attenuation_setup());

  const type_real dt = setup.get_dt();
  const int nsteps = setup.get_nsteps();

  auto [sources, t0, starttime] = specfem::io::read_sources<dim3_tag>(
      source_entries, nsteps, setup.get_t0(), dt, setup.get_simulation_type());
  (void)starttime;
  setup.update_t0(t0);

  auto receivers = specfem::io::read_3d_receivers(stations_node);

  return std::make_unique<AssemblyType>(
      mesh, quadratures, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, setup.get_max_seismogram_step(),
      setup.get_nstep_between_samples(), setup.get_simulation_type(),
      setup.allocate_boundary_values(), setup.instantiate_property_reader());
}

// The layout is built once for the whole suite; it holds a pointer to the
// assembly, so both are torn down together before the global environment
// finalizes Kokkos.
class SystemLayout3D : public ::testing::Test {
protected:
  static void TearDownTestSuite() {
    layout_.reset();
    delete assembly_;
    assembly_ = nullptr;
  }

  static AssemblyType &assembly() {
    if (assembly_ == nullptr) {
      assembly_ = build_assembly_3d("HomogeneousHalfspaceSmallNoABCForceSource")
                      .release();
    }
    return *assembly_;
  }

  static const LayoutType &layout() {
    if (!layout_) {
      layout_ =
          std::make_unique<LayoutType>(LayoutType::from_assembly(assembly()));
    }
    return *layout_;
  }

  static AssemblyType *assembly_;
  static std::unique_ptr<LayoutType> layout_;
};

AssemblyType *SystemLayout3D::assembly_ = nullptr;
std::unique_ptr<LayoutType> SystemLayout3D::layout_;

TEST_F(SystemLayout3D, NumberingAndMaps) {
  const auto &layout = this->layout();

  EXPECT_EQ(layout.ncomp(), ncomp);
  EXPECT_GT(layout.nglob(), 0);
  EXPECT_EQ(layout.num_global_dofs(),
            static_cast<specfem::linear_system::global_ordinal_type>(ncomp) *
                layout.nglob());

  EXPECT_EQ(layout.owned_map()->getGlobalNumElements(),
            static_cast<Tpetra::global_size_t>(layout.num_global_dofs()));
  // Serial: the overlap map carries no extra shared-interface dofs.
  EXPECT_TRUE(layout.overlap_map()->isSameAs(*layout.owned_map()));

  const auto vector = layout.create_vector();
  EXPECT_EQ(static_cast<std::size_t>(vector->getGlobalLength()),
            static_cast<std::size_t>(layout.num_global_dofs()));
  EXPECT_EQ(vector->normInf(), 0) << "create_vector must be zero-initialized";
}

TEST_F(SystemLayout3D, ElementColumnGidsFollowLocalDofIndex) {
  const auto &layout = this->layout();
  const auto elements =
      assembly().element_types.get_elements_on_host(elastic_tag);
  ASSERT_GT(elements.size(), 0);

  const auto &cols = layout.element_column_gids(elements(0));
  ASSERT_EQ(cols.size(), static_cast<std::size_t>(ncomp) * NGLL * NGLL * NGLL);

  // A second call must hit the cache and return the identical buffer.
  EXPECT_EQ(&cols, &layout.element_column_gids(elements(0)));
}

TEST_F(SystemLayout3D, FullMatrixIsEmptyOnACachedFillCompleteGraph) {
  const auto &layout = this->layout();

  const auto matrix = layout.full_matrix();
  const auto dofs =
      static_cast<Tpetra::global_size_t>(layout.num_global_dofs());

  EXPECT_EQ(matrix->getGlobalNumRows(), dofs);
  EXPECT_EQ(matrix->getGlobalNumCols(), dofs);
  EXPECT_GT(matrix->getGlobalNumEntries(), 0u);
  // Structure only: the assembler supplies the values.
  EXPECT_EQ(matrix->getFrobeniusNorm(), 0);

  // The graph is cached, so a second matrix shares it -- this is what lets
  // the implicit solver build its operator A on K's own graph.
  const auto other = layout.full_matrix();
  EXPECT_EQ(matrix->getCrsGraph().get(), other->getCrsGraph().get());
}

TEST_F(SystemLayout3D, BlockDiagonalDefaultAdmitsEveryPoint) {
  const auto &layout = this->layout();

  const auto matrix = layout.block_diagonal_matrix();

  // Exactly ncomp entries per row, and every point carries its whole block.
  EXPECT_EQ(matrix->getGlobalNumEntries(),
            static_cast<std::size_t>(ncomp) *
                static_cast<std::size_t>(layout.num_global_dofs()));
  for (int iglob = 0; iglob < layout.nglob(); ++iglob) {
    ASSERT_TRUE(layout.has_point_block(*matrix, iglob))
        << "point " << iglob << " is missing its block";
  }
}

TEST_F(SystemLayout3D, BlockDiagonalMaskSelectsPoints) {
  const auto &layout = this->layout();

  const auto matrix = layout.block_diagonal_matrix(
      [](const int iglob) { return iglob % 2 == 0; });

  for (int iglob = 0; iglob < layout.nglob(); ++iglob) {
    ASSERT_EQ(layout.has_point_block(*matrix, iglob), iglob % 2 == 0)
        << "point " << iglob << " does not match the mask";
  }
}

// The invariant ImplicitNewmarkSolver::form_operator relies on: every entry
// of a block-diagonal operator is also an entry of the full graph, so C can
// be summed onto K. Sharing one layout makes this structural.
TEST_F(SystemLayout3D, BlockDiagonalEntriesLieInTheFullGraph) {
  const auto &layout = this->layout();

  auto full = layout.full_matrix();
  LayoutType::host_field_view_type zero_block("system_layout_test::zero_block",
                                              ncomp, ncomp);

  for (int iglob = 0; iglob < layout.nglob(); ++iglob) {
    ASSERT_TRUE(layout.has_point_block(*full, iglob))
        << "point " << iglob << " block is not contained in the full graph";
    // The operation form_operator performs when it sums C onto K.
    ASSERT_NO_THROW(layout.scatter_point_block(*full, iglob, zero_block))
        << "point " << iglob;
  }
}

// scatter and gather must be exact inverses -- this is the property that
// makes changing the dof ordering safe, since no consumer addresses rows
// directly any more.
TEST_F(SystemLayout3D, ScatterGatherRoundTrip) {
  const auto &layout = this->layout();
  const int nglob = layout.nglob();

  LayoutType::host_field_view_type source("system_layout_test::source", nglob,
                                          ncomp);
  LayoutType::host_field_view_type roundtrip("system_layout_test::roundtrip",
                                             nglob, ncomp);

  // Distinct value per (iglob, icomp) so a transposed or collapsed index
  // cannot survive the round trip.
  for (int iglob = 0; iglob < nglob; ++iglob) {
    for (int icomp = 0; icomp < ncomp; ++icomp) {
      source(iglob, icomp) = static_cast<type_real>(iglob * ncomp + icomp + 1);
    }
  }

  const auto vector = layout.scatter(source);
  ASSERT_EQ(static_cast<std::size_t>(vector->getGlobalLength()),
            static_cast<std::size_t>(layout.num_global_dofs()));
  layout.gather(*vector, roundtrip);

  for (int iglob = 0; iglob < nglob; ++iglob) {
    for (int icomp = 0; icomp < ncomp; ++icomp) {
      ASSERT_EQ(roundtrip(iglob, icomp), source(iglob, icomp))
          << "round trip changed (" << iglob << ", " << icomp << ")";
    }
  }

  // The in-place overload must agree with the allocating one.
  auto reused = layout.create_vector();
  layout.scatter(source, *reused);
  reused->update(static_cast<specfem::linear_system::scalar_type>(-1), *vector,
                 static_cast<specfem::linear_system::scalar_type>(1));
  EXPECT_EQ(reused->normInf(), 0);
}

TEST_F(SystemLayout3D, TransfersRejectMisshapedFields) {
  const auto &layout = this->layout();
  LayoutType::host_field_view_type wrong("system_layout_test::wrong",
                                         layout.nglob() + 1, ncomp);
  auto vector = layout.create_vector();
  EXPECT_THROW(layout.scatter(wrong, *vector), std::runtime_error);
  EXPECT_THROW(layout.gather(*vector, wrong), std::runtime_error);
}

} // namespace system_layout_test

#else // !SPECFEM_ENABLE_TRILINOS

TEST(SystemLayout3D, SkippedWithoutTrilinos) {
  GTEST_SKIP() << "SPECFEM++ was built without Trilinos "
                  "(SPECFEM_ENABLE_TRILINOS=OFF); the system layout is "
                  "unavailable.";
}

#endif // SPECFEM_ENABLE_TRILINOS

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
