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
  test_case.time_scheme = setup.instantiate_timescheme(
      test_case.assembly->fields, setup.get_simulation_type());
  return test_case;
}

// beta = 0 has no displacement-form operator: at beta = 0, u_{n+1} does not
// depend on a_{n+1}, so the map the form must invert is singular. This is a
// property of the form, not of the solver -- see the acceleration-form test
// below, which accepts the same beta.
TEST(ImplicitSolverScope3D, RejectsExplicitBetaInDisplacementForm) {
  auto test_case = build_case_3d("HomogeneousHalfspaceSmallNoABCForceSource");
  specfem::solver::ImplicitSolverConfig config;
  config.form = specfem::solver::NewmarkForm::displacement;
  config.newmark.beta = 0;
  EXPECT_THROW(
      SolverType solver(test_case.time_scheme, {}, *test_case.assembly, config),
      std::runtime_error);
}

// The acceleration form's operator is M + gamma dt C + beta dt^2 K, which is
// regular at beta = 0. On a mesh without Stacey boundaries C is empty, so the
// operator collapses to the lumped mass -- the explicit scheme's effective
// mass, which is what makes the beta = 0 run reproduce the explicit solver.
TEST(ImplicitSolver3D, AcceptsExplicitBetaInAccelerationForm) {
  auto test_case = build_case_3d("HomogeneousHalfspaceSmallNoABCForceSource");
  specfem::solver::ImplicitSolverConfig config;
  config.form = specfem::solver::NewmarkForm::acceleration;
  config.newmark.beta = 0;
  config.newmark.gamma = static_cast<type_real>(0.5);
  SolverType solver(test_case.time_scheme, {}, *test_case.assembly, config);

  EXPECT_EQ(solver.damping()->getGlobalNumEntries(), 0u)
      << "the no-ABC fixture must produce an empty damping matrix";

  // A x == M x exactly: beta dt^2 K vanishes and C is empty.
  const auto &dof_map = solver.dof_map();
  VectorType x(dof_map.owned_map());
  x.randomize();

  VectorType a_x(dof_map.owned_map()), reference(dof_map.owned_map());
  solver.system_operator()->apply(x, a_x);
  reference.elementWiseMultiply(static_cast<type_real>(1), *solver.mass(), x,
                                static_cast<type_real>(0));

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
      << "at beta = 0 with no damping the operator must be the lumped mass";
}

TEST(ImplicitSolverScope3D, RejectsMixedMediumMesh) {
  auto test_case = build_case_3d("AcousticElasticForce");
  EXPECT_THROW(
      SolverType solver(test_case.time_scheme, {}, *test_case.assembly),
      std::runtime_error);
}

// One test covers construction and the operator identity: ctest runs every
// test case in its own process, and the solver construction (stiffness
// probe + preconditioner setup, ~20 s serial) dominates -- splitting the
// assertions would pay it once per case.
//
// A x must equal (k K + c C + m M) x entry by entry, with the coefficient
// triple of the configured form -- this validates the value plumbing of
// form_operator (scaled K copy, scaled C sum, mass diagonal) through
// independent applies of the constituent operators. Parameterized over the
// form because the two share that plumbing and differ only in the triple.
class ImplicitSolverFormTest
    : public ::testing::TestWithParam<specfem::solver::NewmarkForm> {};

TEST_P(ImplicitSolverFormTest, ConstructsAndOperatorMatchesOnStaceyMesh) {
  auto test_case = build_case_3d("HomogeneousHalfSpaceStacey");
  specfem::solver::ImplicitSolverConfig config;
  config.form = GetParam();
  SolverType solver(test_case.time_scheme, {}, *test_case.assembly, config);

  EXPECT_GT(solver.stiffness()->getGlobalNumEntries(), 0u);
  EXPECT_GT(solver.damping()->getGlobalNumEntries(), 0u)
      << "the Stacey fixture must produce a nonempty damping matrix";
  EXPECT_EQ(solver.system_operator()->getGlobalNumEntries(),
            solver.stiffness()->getGlobalNumEntries())
      << "A must live on K's graph";

  const auto &dof_map = solver.dof_map();

  const type_real dt = test_case.time_scheme->get_timestep();
  const type_real beta = config.newmark.beta;
  const type_real gamma = config.newmark.gamma;

  const bool acceleration_form =
      config.form == specfem::solver::NewmarkForm::acceleration;
  const type_real stiffness_coefficient =
      acceleration_form ? beta * dt * dt : static_cast<type_real>(1);
  const type_real damping_coefficient =
      acceleration_form ? gamma * dt : gamma / (beta * dt);
  const type_real mass_coefficient =
      acceleration_form ? static_cast<type_real>(1) : 1 / (beta * dt * dt);

  VectorType x(dof_map.owned_map());
  x.randomize();

  VectorType a_x(dof_map.owned_map()), reference(dof_map.owned_map()),
      scratch(dof_map.owned_map());
  solver.system_operator()->apply(x, a_x);

  solver.stiffness()->apply(x, scratch);
  reference.update(stiffness_coefficient, scratch, 0);
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
      << "A x disagrees with the configured form's k K x + c C x + m M x";
}

INSTANTIATE_TEST_SUITE_P(
    ImplicitSolver3D, ImplicitSolverFormTest,
    ::testing::Values(specfem::solver::NewmarkForm::displacement,
                      specfem::solver::NewmarkForm::acceleration),
    [](const ::testing::TestParamInfo<specfem::solver::NewmarkForm> &info) {
      return info.param == specfem::solver::NewmarkForm::displacement
                 ? "DisplacementForm"
                 : "AccelerationForm";
    });

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
