#include "SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/io.hpp"
#include "specfem/linear_system/sparse_matrix_view/fe_assembly.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace sparse_matrix_view_fe_assembly_test {

constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;
constexpr auto elastic_tag = specfem::element::medium_tag::elastic;

using AssemblyType = specfem::assembly::assembly<dim3_tag>;
using MappingType = specfem::linear_system::FEMapping<dim3_tag, elastic_tag>;
using FEAssemblyType = specfem::linear_system::FEAssembly<MappingType>;
using global_ordinal_type = specfem::linear_system::global_ordinal_type;

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

// Sorted global column ids of one row of a fill-complete graph.
std::vector<global_ordinal_type>
row_columns(const specfem::linear_system::crs_graph_type &graph,
            const global_ordinal_type row) {
  using inds_type = specfem::linear_system::crs_graph_type::
      nonconst_global_inds_host_view_type;

  const std::size_t num_entries = graph.getNumEntriesInGlobalRow(row);
  inds_type indices("row_columns", num_entries);
  std::size_t num_returned = 0;
  graph.getGlobalRowCopy(row, indices, num_returned);

  std::vector<global_ordinal_type> columns(indices.data(),
                                           indices.data() + num_returned);
  std::sort(columns.begin(), columns.end());
  return columns;
}

// Mesh with natural boundaries only.
class FEAssemblyNaturalBoundary3D : public ::testing::Test {
protected:
  static void TearDownTestSuite() {
    fe_.reset();
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

  static FEAssemblyType &fe() {
    if (!fe_) {
      fe_ = std::make_unique<FEAssemblyType>(MappingType(assembly()));
    }
    return *fe_;
  }

  static AssemblyType *assembly_;
  static std::unique_ptr<FEAssemblyType> fe_;
};

AssemblyType *FEAssemblyNaturalBoundary3D::assembly_ = nullptr;
std::unique_ptr<FEAssemblyType> FEAssemblyNaturalBoundary3D::fe_;

// Mesh with Stacey absorbing boundaries, the dataset the damping assembler
// tests use.
class FEAssemblyStacey3D : public ::testing::Test {
protected:
  static void TearDownTestSuite() {
    fe_.reset();
    delete assembly_;
    assembly_ = nullptr;
  }

  static AssemblyType &assembly() {
    if (assembly_ == nullptr) {
      assembly_ = build_assembly_3d("HomogeneousHalfSpaceStacey").release();
    }
    return *assembly_;
  }

  static FEAssemblyType &fe() {
    if (!fe_) {
      fe_ = std::make_unique<FEAssemblyType>(MappingType(assembly()));
    }
    return *fe_;
  }

  static AssemblyType *assembly_;
  static std::unique_ptr<FEAssemblyType> fe_;
};

AssemblyType *FEAssemblyStacey3D::assembly_ = nullptr;
std::unique_ptr<FEAssemblyType> FEAssemblyStacey3D::fe_;

TEST_F(FEAssemblyStacey3D, DampingGraphIsBlockDiagonal) {
  const auto &map = fe().mapping();
  const auto graph = fe().damping_matrix_graph();
  ASSERT_FALSE(graph.is_null());

  const auto ncomp = static_cast<std::size_t>(map.ncomp());
  std::size_t nonempty_rows = 0;

  for (global_ordinal_type row = 0; row < map.num_global_dofs(); ++row) {
    const auto row_entries = graph->getNumEntriesInGlobalRow(row);
    if (row_entries == 0) {
      continue;
    }
    ++nonempty_rows;
    EXPECT_EQ(row_entries, ncomp)
        << "damping row " << row << " is not a single ncomp block";
  }

  EXPECT_GT(nonempty_rows, 0u) << "the Stacey mesh must damp somewhere";
  EXPECT_LT(nonempty_rows, static_cast<std::size_t>(map.num_global_dofs()))
      << "interior rows must stay empty";
  EXPECT_EQ(nonempty_rows % ncomp, 0u)
      << "all components of a damping point must participate";
}

TEST_F(FEAssemblyStacey3D, DampingGraphAgreesWithMask) {
  const auto &map = fe().mapping();
  const auto graph = fe().damping_matrix_graph();

  // Every dof of a masked point carries a block; no other row does.
  for (int point = 0; point < map.nglob(); ++point) {
    const std::size_t expected =
        map.is_damping_point(point) ? static_cast<std::size_t>(map.ncomp()) : 0;
    for (int icomp = 0; icomp < map.ncomp(); ++icomp) {
      ASSERT_EQ(graph->getNumEntriesInGlobalRow(map(point, icomp)), expected)
          << "graph disagrees with the mask at (iglob=" << point
          << ", icomp=" << icomp << ")";
    }
  }
}

} // namespace sparse_matrix_view_fe_assembly_test

#else // !SPECFEM_ENABLE_TRILINOS

TEST(FEAssembly3D, SkippedWithoutTrilinos) {
  GTEST_SKIP() << "SPECFEM++ was built without Trilinos "
                  "(SPECFEM_ENABLE_TRILINOS=OFF); the finite-element linear "
                  "system is unavailable.";
}

#endif // SPECFEM_ENABLE_TRILINOS

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
