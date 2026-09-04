#include "SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/io.hpp"
#include "specfem/linear_system/damping_assembler.hpp"
#include "specfem/linear_system/dof_map.hpp"
#include "specfem/linear_system/sparse_matrix_view/fe_assembly.hpp"
#include "specfem/linear_system/tpetra_assembler.hpp"
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
using MediumTags =
    specfem::tags::Tags<dim3_tag, elastic_tag,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none>;
using StiffnessAssemblerType =
    specfem::linear_system::StiffnessAssembler<MediumTags>;
using DampingAssemblerType =
    specfem::linear_system::DampingAssembler<MediumTags>;
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

// Mesh with natural boundaries only. The stiffness assembler rejects Stacey
// meshes, so the full-matrix cross-check has to live here.
class FEAssemblyNaturalBoundary3D : public ::testing::Test {
protected:
  static void TearDownTestSuite() {
    matrix_ = Teuchos::null;
    assembler_.reset();
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

  static StiffnessAssemblerType &assembler() {
    if (!assembler_) {
      assembler_ = std::make_unique<StiffnessAssemblerType>(assembly());
    }
    return *assembler_;
  }

  static Teuchos::RCP<specfem::linear_system::crs_matrix_type> matrix() {
    if (matrix_.is_null()) {
      matrix_ = assembler().assemble();
    }
    return matrix_;
  }

  static AssemblyType *assembly_;
  static std::unique_ptr<FEAssemblyType> fe_;
  static std::unique_ptr<StiffnessAssemblerType> assembler_;
  static Teuchos::RCP<specfem::linear_system::crs_matrix_type> matrix_;
};

AssemblyType *FEAssemblyNaturalBoundary3D::assembly_ = nullptr;
std::unique_ptr<FEAssemblyType> FEAssemblyNaturalBoundary3D::fe_;
std::unique_ptr<StiffnessAssemblerType> FEAssemblyNaturalBoundary3D::assembler_;
Teuchos::RCP<specfem::linear_system::crs_matrix_type>
    FEAssemblyNaturalBoundary3D::matrix_;

// The graph comparisons below are only meaningful if both paths number dofs
// the same way.
TEST_F(FEAssemblyNaturalBoundary3D, DofIdsMatchDofMap) {
  const auto &map = fe().mapping();
  const specfem::linear_system::DofMap dof_map(assembly(), MediumTags{});

  ASSERT_EQ(map.num_global_dofs(), dof_map.num_global_dofs());

  for (int iglob = 0; iglob < map.nglob(); ++iglob) {
    for (int icomp = 0; icomp < map.ncomp(); ++icomp) {
      ASSERT_EQ(map(iglob, icomp), dof_map.gid(iglob, icomp))
          << "dof id layout diverged at (iglob=" << iglob << ", icomp=" << icomp
          << ")";
    }
  }
}

TEST_F(FEAssemblyNaturalBoundary3D, GraphDimensionsMatchDofCount) {
  const auto &map = fe().mapping();
  const auto graph = fe().full_matrix_graph();

  ASSERT_FALSE(graph.is_null());
  EXPECT_TRUE(graph->isFillComplete());
  EXPECT_EQ(static_cast<global_ordinal_type>(graph->getGlobalNumRows()),
            map.num_global_dofs());
  EXPECT_EQ(static_cast<global_ordinal_type>(graph->getGlobalNumCols()),
            map.num_global_dofs());
}

TEST_F(FEAssemblyNaturalBoundary3D, FullGraphMatchesStiffnessAssembler) {
  const auto &map = fe().mapping();
  const auto graph = fe().full_matrix_graph();
  const auto reference = matrix()->getCrsGraph();

  ASSERT_FALSE(reference.is_null());
  ASSERT_EQ(graph->getGlobalNumEntries(), reference->getGlobalNumEntries())
      << "the FEAssembly graph has a different number of nonzeros than the "
         "graph the stiffness assembler builds";

  for (global_ordinal_type row = 0; row < map.num_global_dofs(); ++row) {
    const auto columns = row_columns(*graph, row);
    const auto reference_columns = row_columns(*reference, row);
    ASSERT_EQ(columns, reference_columns)
        << "row " << row << " couples to a different set of columns than in "
        << "the graph the stiffness assembler builds";
  }
}

TEST_F(FEAssemblyNaturalBoundary3D, DampingGraphEmpty) {
  const auto graph = fe().damping_matrix_graph();

  ASSERT_FALSE(graph.is_null());
  EXPECT_TRUE(graph->isFillComplete());
  EXPECT_EQ(graph->getGlobalNumEntries(), 0u)
      << "a mesh without absorbing boundaries must yield an empty damping "
         "graph";
}

// Mesh with Stacey absorbing boundaries, the dataset the damping assembler
// tests use.
class FEAssemblyStacey3D : public ::testing::Test {
protected:
  static void TearDownTestSuite() {
    matrix_ = Teuchos::null;
    dof_map_.reset();
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

  static Teuchos::RCP<specfem::linear_system::crs_matrix_type> matrix() {
    if (matrix_.is_null()) {
      if (!dof_map_) {
        dof_map_ = std::make_unique<specfem::linear_system::DofMap>(
            assembly(), MediumTags{});
      }
      DampingAssemblerType assembler(assembly(), *dof_map_);
      matrix_ = assembler.assemble();
    }
    return matrix_;
  }

  static AssemblyType *assembly_;
  static std::unique_ptr<FEAssemblyType> fe_;
  static std::unique_ptr<specfem::linear_system::DofMap> dof_map_;
  static Teuchos::RCP<specfem::linear_system::crs_matrix_type> matrix_;
};

AssemblyType *FEAssemblyStacey3D::assembly_ = nullptr;
std::unique_ptr<FEAssemblyType> FEAssemblyStacey3D::fe_;
std::unique_ptr<specfem::linear_system::DofMap> FEAssemblyStacey3D::dof_map_;
Teuchos::RCP<specfem::linear_system::crs_matrix_type>
    FEAssemblyStacey3D::matrix_;

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

TEST_F(FEAssemblyStacey3D, DampingGraphMatchesDampingAssembler) {
  const auto &map = fe().mapping();
  const auto graph = fe().damping_matrix_graph();
  const auto reference = matrix()->getCrsGraph();
  ASSERT_FALSE(reference.is_null());

  // The two masks are derived independently: the assembler probes the
  // matrix-free kernel and tests for exact nonzeros, this graph reads the
  // boundary tags. They must agree on which points damp.
  ASSERT_EQ(graph->getGlobalNumEntries(), reference->getGlobalNumEntries())
      << "the FEAssembly damping graph has a different number of nonzeros "
         "than the graph the damping assembler builds";

  for (global_ordinal_type row = 0; row < map.num_global_dofs(); ++row) {
    const auto columns = row_columns(*graph, row);
    const auto reference_columns = row_columns(*reference, row);
    ASSERT_EQ(columns, reference_columns)
        << "damping row " << row << " couples to a different set of columns "
        << "than in the graph the damping assembler builds";
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
