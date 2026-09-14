#include "../SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/compute/impl/compute_stiffness_interaction.hpp"
#include "specfem/compute/initialize_mass_matrix.hpp"
#include "specfem/io.hpp"
#include "specfem/linear_system/damping_assembler.hpp"
#include "specfem/linear_system/mass_vector.hpp"
#include "specfem/linear_system/tpetra_assembler.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace damping_assembler_test {

constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;
constexpr auto elastic_tag = specfem::element::medium_tag::elastic;
constexpr auto forward_tag = specfem::simulation::field_type::forward;
constexpr int NGLL = 5;
constexpr int ncomp = 3;

constexpr bool single_precision = sizeof(type_real) == sizeof(float);
const type_real rel_tol = single_precision ? static_cast<type_real>(2e-3)
                                           : static_cast<type_real>(1e-10);

using AssemblyType = specfem::assembly::assembly<dim3_tag>;
using DampingTags =
    specfem::tags::Tags<dim3_tag, elastic_tag,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none>;
using DampingAssemblerType =
    specfem::linear_system::DampingAssembler<DampingTags>;
using MappingType = specfem::linear_system::FEMapping<dim3_tag, elastic_tag>;
using FEAssemblyType = specfem::linear_system::FEAssembly<MappingType>;
using StiffnessAssemblerType =
    specfem::linear_system::StiffnessAssembler<DampingTags>;
using VectorType = specfem::linear_system::vector_type;

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

// Shared fixture: the Stacey halfspace assembly and its damping matrix,
// built once on first access so a throwing constructor fails inside a test
// body, not in SetUpTestSuite.
class DampingAssembler3D : public ::testing::Test {
protected:
  static void TearDownTestSuite() {
    matrix_ = Teuchos::null;
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

  static const FEAssemblyType &fe() {
    if (!fe_) {
      fe_ = std::make_unique<FEAssemblyType>(MappingType(assembly()));
    }
    return *fe_;
  }

  static Teuchos::RCP<specfem::linear_system::crs_matrix_type> matrix() {
    if (matrix_.is_null()) {
      DampingAssemblerType assembler(assembly(), fe());
      matrix_ = assembler.assemble();
    }
    return matrix_;
  }

  static AssemblyType *assembly_;
  static std::unique_ptr<FEAssemblyType> fe_;
  static Teuchos::RCP<specfem::linear_system::crs_matrix_type> matrix_;
};

AssemblyType *DampingAssembler3D::assembly_ = nullptr;
std::unique_ptr<FEAssemblyType> DampingAssembler3D::fe_;
Teuchos::RCP<specfem::linear_system::crs_matrix_type>
    DampingAssembler3D::matrix_;

// Largest absolute matrix entry -- the natural scale for tolerances.
type_real max_abs_entry(
    const Teuchos::RCP<specfem::linear_system::crs_matrix_type> &matrix) {
  const auto values = matrix->getLocalMatrixHost().values;
  type_real scale = 0;
  for (std::size_t k = 0; k < values.extent(0); ++k) {
    scale = std::max(scale, std::abs(values(k)));
  }
  return scale;
}

TEST(DampingAssemblerScope3D, EmptyOnNaturalBoundaryMesh) {
  const auto assembly =
      build_assembly_3d("HomogeneousHalfspaceSmallNoABCForceSource");
  const FEAssemblyType fe{ MappingType(*assembly) };
  DampingAssemblerType assembler(*assembly, fe);
  const auto matrix = assembler.assemble();
  EXPECT_EQ(matrix->getGlobalNumEntries(), 0u)
      << "a mesh without Stacey boundaries must yield an empty damping "
         "matrix";
}

TEST(DampingAssemblerScope3D, WithStaceyScopeAcceptsStaceyMesh) {
  const auto assembly = build_assembly_3d("HomogeneousHalfSpaceStacey");
  // The default (natural-boundaries) scope rejects this mesh -- covered by
  // StiffnessAssemblerScope3D.RejectsStaceyBoundaries -- while the opt-in
  // admits it because the displacement probe runs at zero velocity.
  EXPECT_NO_THROW(StiffnessAssemblerType assembler(
      *assembly, StiffnessAssemblerType::default_batch_size,
      specfem::linear_system::StiffnessScope::with_stacey));
}

TEST_F(DampingAssembler3D, BlockDiagonalWithEmptyInteriorRows) {
  const auto matrix = this->matrix();
  const auto &mapping = fe().mapping();

  const auto graph = matrix->getCrsGraph();
  std::size_t nonempty_rows = 0;
  for (std::size_t row = 0;
       row < static_cast<std::size_t>(mapping.num_global_dofs()); ++row) {
    const auto row_entries = graph->getNumEntriesInGlobalRow(
        static_cast<specfem::linear_system::global_ordinal_type>(row));
    if (row_entries == 0) {
      continue;
    }
    ++nonempty_rows;
    EXPECT_EQ(row_entries, static_cast<std::size_t>(ncomp))
        << "damping row " << row << " is not a single ncomp block";
  }

  // The Stacey fixture has boundary points (nonempty rows) and interior
  // points (empty rows); all components of a damping point carry a block.
  EXPECT_GT(nonempty_rows, 0u);
  EXPECT_LT(nonempty_rows, static_cast<std::size_t>(mapping.num_global_dofs()));
  EXPECT_EQ(nonempty_rows % ncomp, 0u);
}

TEST_F(DampingAssembler3D, SymmetricPositiveSemidefinite) {
  const auto matrix = this->matrix();
  const auto &mapping = fe().mapping();

  VectorType x(fe().owned_map()), z(fe().owned_map());
  x.randomize();
  z.randomize();

  VectorType c_x(fe().owned_map()), c_z(fe().owned_map());
  matrix->apply(x, c_x);
  matrix->apply(z, c_z);

  // Symmetry through the bilinear form: z' C x == x' C z.
  const type_real z_c_x = z.dot(c_x);
  const type_real x_c_z = x.dot(c_z);
  const type_real scale =
      max_abs_entry(matrix) * static_cast<type_real>(x.norm2() * z.norm2());
  ASSERT_GT(scale, static_cast<type_real>(0));
  EXPECT_LE(std::abs(x_c_z - z_c_x), rel_tol * scale)
      << "damping matrix is not symmetric";

  // Positive semidefinite: the dashpot only ever removes energy.
  const type_real x_c_x = x.dot(c_x);
  EXPECT_GE(x_c_x, -rel_tol * scale)
      << "damping quadratic form is not positive semidefinite";
}

TEST_F(DampingAssembler3D, MatchesProductionKernelForRandomVelocity) {
  const auto matrix = this->matrix();
  const auto &mapping = fe().mapping();

  auto &field = assembly().fields.template get_simulation_field<forward_tag>();
  const auto &field_impl = field.template get_field<elastic_tag>();
  const auto h_u = field_impl.get_host_field();
  const auto h_v = field_impl.get_host_field_dot();
  const auto h_a = field_impl.get_host_field_dot_dot();
  const int nglob = mapping.nglob();

  // Random velocity over the whole mesh; displacement and acceleration zero,
  // so the production kernel computes accel = -C v exactly.
  std::mt19937 generator(98765);
  std::uniform_real_distribution<type_real> distribution(-1, 1);
  Kokkos::deep_copy(h_u, 0);
  Kokkos::deep_copy(h_a, 0);
  for (int iglob = 0; iglob < nglob; ++iglob) {
    for (int icomp = 0; icomp < ncomp; ++icomp) {
      h_v(iglob, icomp) = distribution(generator);
    }
  }
  assembly().fields.copy_to_device();

  using BaseTags = specfem::tags::Tags<dim3_tag, forward_tag, elastic_tag>;
  specfem::compute::impl::compute_stiffness_interaction<
      NGLL, specfem::tags::expand<BaseTags, specfem::element::mpi_tag::outer>>(
      assembly(), 0);
  specfem::compute::impl::compute_stiffness_interaction<
      NGLL, specfem::tags::expand<BaseTags, specfem::element::mpi_tag::inner>>(
      assembly(), 0);
  assembly().fields.copy_to_host();

  VectorType v(fe().owned_map()), c_v(fe().owned_map());
  {
    auto view = v.getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (int iglob = 0; iglob < nglob; ++iglob) {
      for (int icomp = 0; icomp < ncomp; ++icomp) {
        view(static_cast<std::size_t>(mapping(iglob, icomp)), 0) =
            h_v(iglob, icomp);
      }
    }
  }
  matrix->apply(v, c_v);

  type_real scale = 0;
  type_real max_diff = 0;
  {
    auto view = c_v.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (int iglob = 0; iglob < nglob; ++iglob) {
      for (int icomp = 0; icomp < ncomp; ++icomp) {
        const type_real expected = -h_a(iglob, icomp);
        const type_real actual =
            view(static_cast<std::size_t>(mapping(iglob, icomp)), 0);
        scale = std::max(scale, std::abs(expected));
        max_diff = std::max(max_diff, std::abs(expected - actual));
      }
    }
  }
  ASSERT_GT(scale, static_cast<type_real>(0));
  EXPECT_LE(max_diff, rel_tol * scale)
      << "assembled C v disagrees with the production kernel's velocity "
         "path";

  // Leave the probe scratch as found for the other tests.
  Kokkos::deep_copy(h_u, 0);
  Kokkos::deep_copy(h_v, 0);
  Kokkos::deep_copy(h_a, 0);
  assembly().fields.copy_to_device();
}

// Independent cross-path check: the explicit solver folds the Stacey term
// into the lumped mass as M(dt) = M(0) + (dt/2) C 1 (the mass path evaluates
// the traction at velocity -dt/2, a different code path than the velocity
// probe). The assembled C must reproduce that difference exactly.
TEST_F(DampingAssembler3D, MassPathCrossCheck) {
  const auto matrix = this->matrix();
  const auto &mapping = fe().mapping();
  const int nglob = mapping.nglob();

  // dt large enough that the Stacey term is well separated from float
  // cancellation in M(dt) - M(0).
  const type_real dt = static_cast<type_real>(0.5);

  const auto mass_zero =
      specfem::linear_system::assemble_mass_vector<DampingTags>(assembly(),
                                                                fe());

  // Manual M(dt) accumulation through the production path, using the same
  // scratch discipline as assemble_mass_vector.
  auto &field = assembly().fields.template get_simulation_field<forward_tag>();
  const auto &field_impl = field.template get_field<elastic_tag>();
  const auto mass = field_impl.get_mass_inverse();
  const auto h_mass = field_impl.get_host_mass_inverse();

  Kokkos::deep_copy(mass, 0);
  using BaseTags = specfem::tags::Tags<dim3_tag, forward_tag, elastic_tag>;
  specfem::compute::compute_mass_matrix<
      NGLL, specfem::tags::expand<BaseTags, specfem::element::mpi_tag::outer>>(
      assembly(), dt);
  specfem::compute::compute_mass_matrix<
      NGLL, specfem::tags::expand<BaseTags, specfem::element::mpi_tag::inner>>(
      assembly(), dt);
  Kokkos::deep_copy(h_mass, mass);

  std::vector<type_real> mass_dt(
      static_cast<std::size_t>(mapping.num_global_dofs()));
  for (int iglob = 0; iglob < nglob; ++iglob) {
    for (int icomp = 0; icomp < ncomp; ++icomp) {
      mass_dt[static_cast<std::size_t>(mapping(iglob, icomp))] =
          h_mass(iglob, icomp);
    }
  }
  Kokkos::deep_copy(mass, 0);
  Kokkos::deep_copy(h_mass, 0);

  // (dt/2) C 1: row sums of the damping matrix.
  VectorType ones(fe().owned_map()), row_sums(fe().owned_map());
  ones.putScalar(static_cast<type_real>(1));
  matrix->apply(ones, row_sums);

  type_real scale = 0;
  type_real max_diff = 0;
  {
    const auto m0 = mass_zero->getLocalViewHost(Tpetra::Access::ReadOnly);
    const auto sums = row_sums.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (std::size_t dof = 0;
         dof < static_cast<std::size_t>(mapping.num_global_dofs()); ++dof) {
      const type_real expected = dt / 2 * sums(dof, 0);
      const type_real actual = mass_dt[dof] - m0(dof, 0);
      scale = std::max(scale, std::abs(expected));
      max_diff = std::max(max_diff, std::abs(expected - actual));
    }
  }
  ASSERT_GT(scale, static_cast<type_real>(0));
  EXPECT_LE(max_diff, rel_tol * scale)
      << "M(dt) - M(0) disagrees with (dt/2) C 1";
}

} // namespace damping_assembler_test

#else // !SPECFEM_ENABLE_TRILINOS

TEST(DampingAssembler3D, SkippedWithoutTrilinos) {
  GTEST_SKIP() << "SPECFEM++ was built without Trilinos "
                  "(SPECFEM_ENABLE_TRILINOS=OFF); the damping assembler is "
                  "unavailable.";
}

#endif // SPECFEM_ENABLE_TRILINOS

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
