#include "../SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/solver/implicit_solver.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace implicit_solver_test {

constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;

constexpr bool single_precision = sizeof(type_real) == sizeof(float);
const type_real rel_tol = single_precision ? static_cast<type_real>(2e-3)
                                           : static_cast<type_real>(1e-10);

using AssemblyType = specfem::assembly::assembly<dim3_tag>;
using SolverTags =
    specfem::tags::Tags<dim3_tag, specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none>;
using SolverType = specfem::solver::ImplicitNewmarkSolver<SolverTags>;
using VectorType = specfem::linear_system::vector_type;

// Assembly plus the time scheme built from the same fixture configuration.
struct TestCase {
  std::unique_ptr<AssemblyType> assembly;
  std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme;
};

// Build a full assembly + time scheme from a Newmark displacement-test
// dataset. Paths are relative to TEST_OUTPUT_DIR, where the
// displacement_tests data tree is linked (see SERIAL_LINK_DIRS in
// serial.cmake).
TestCase build_case_3d(const std::string &test_name) {
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

  TestCase test_case;
  test_case.assembly = std::make_unique<AssemblyType>(
      mesh, quadratures, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, setup.get_max_seismogram_step(),
      setup.get_nstep_between_samples(), setup.get_simulation_type(),
      setup.allocate_boundary_values(), setup.instantiate_property_reader());
  test_case.time_scheme =
      setup.instantiate_timescheme(test_case.assembly->fields);
  return test_case;
}

TEST(ImplicitSolverScope3D, RejectsExplicitBeta) {
  auto test_case = build_case_3d("HomogeneousHalfspaceSmallNoABCForceSource");
  specfem::solver::ImplicitSolverConfig config;
  config.newmark.beta = 0;
  EXPECT_THROW(
      SolverType solver(test_case.time_scheme, {}, *test_case.assembly, config),
      std::runtime_error);
}

TEST(ImplicitSolverScope3D, RejectsMixedMediumMesh) {
  auto test_case = build_case_3d("AcousticElasticForce");
  EXPECT_THROW(
      SolverType solver(test_case.time_scheme, {}, *test_case.assembly),
      std::runtime_error);
}

// Shared fixture: the solver on the Stacey halfspace (K, C, M, and A all
// nontrivial), built once on first access.
class ImplicitSolver3D : public ::testing::Test {
protected:
  static void TearDownTestSuite() {
    solver_.reset();
    delete test_case_;
    test_case_ = nullptr;
  }

  static TestCase &test_case() {
    if (test_case_ == nullptr) {
      test_case_ = new TestCase(build_case_3d("HomogeneousHalfSpaceStacey"));
    }
    return *test_case_;
  }

  static SolverType &solver() {
    if (!solver_) {
      solver_ = std::make_unique<SolverType>(
          test_case().time_scheme,
          std::vector<std::shared_ptr<
              specfem::periodic_tasks::periodic_task<dim3_tag>>>{},
          *test_case().assembly);
    }
    return *solver_;
  }

  static TestCase *test_case_;
  static std::unique_ptr<SolverType> solver_;
};

TestCase *ImplicitSolver3D::test_case_ = nullptr;
std::unique_ptr<SolverType> ImplicitSolver3D::solver_;

TEST_F(ImplicitSolver3D, ConstructsOnStaceyMesh) {
  const auto &solver = this->solver();
  EXPECT_GT(solver.stiffness()->getGlobalNumEntries(), 0u);
  EXPECT_GT(solver.damping()->getGlobalNumEntries(), 0u)
      << "the Stacey fixture must produce a nonempty damping matrix";
  EXPECT_EQ(solver.system_operator()->getGlobalNumEntries(),
            solver.stiffness()->getGlobalNumEntries())
      << "A must live on K's graph";
}

// A x must equal M/(beta dt^2) x + gamma/(beta dt) C x + K x entry by entry
// -- validates the value plumbing of form_operator (K copy, scaled C sum,
// mass diagonal) through independent applies of the constituent operators.
TEST_F(ImplicitSolver3D, OperatorMatchesConstituents) {
  const auto &solver = this->solver();
  const auto &dof_map = solver.dof_map();

  const type_real dt = test_case().time_scheme->get_timestep();
  const specfem::solver::ImplicitSolverConfig config; // defaults the solver
                                                      // used
  const type_real beta = config.newmark.beta;
  const type_real gamma = config.newmark.gamma;
  const type_real mass_coefficient = 1 / (beta * dt * dt);
  const type_real damping_coefficient = gamma / (beta * dt);

  VectorType x(dof_map.owned_map());
  x.randomize();

  VectorType a_x(dof_map.owned_map()), reference(dof_map.owned_map()),
      scratch(dof_map.owned_map());
  solver.system_operator()->apply(x, a_x);

  solver.stiffness()->apply(x, reference);
  solver.damping()->apply(x, scratch);
  reference.update(damping_coefficient, scratch, 1);
  reference.elementWiseMultiply(mass_coefficient, *solver.mass(), x, 1);

  type_real scale = 0;
  type_real max_diff = 0;
  {
    const auto a_view = a_x.getLocalViewHost(Tpetra::Access::ReadOnly);
    const auto ref_view = reference.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (std::size_t dof = 0;
         dof < static_cast<std::size_t>(dof_map.num_global_dofs()); ++dof) {
      scale = std::max(scale, std::abs(ref_view(dof, 0)));
      max_diff =
          std::max(max_diff, std::abs(a_view(dof, 0) - ref_view(dof, 0)));
    }
  }
  ASSERT_GT(scale, static_cast<type_real>(0));
  EXPECT_LE(max_diff, rel_tol * scale)
      << "A x disagrees with M/(beta dt^2) x + gamma/(beta dt) C x + K x";
}

} // namespace implicit_solver_test

#else // !SPECFEM_ENABLE_TRILINOS

TEST(ImplicitSolver3D, SkippedWithoutTrilinos) {
  GTEST_SKIP() << "SPECFEM++ was built without Trilinos "
                  "(SPECFEM_ENABLE_TRILINOS=OFF); the implicit solver is "
                  "unavailable.";
}

#endif // SPECFEM_ENABLE_TRILINOS

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
