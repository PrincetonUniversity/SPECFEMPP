#include "../SPECFEM_Environment.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/compute/impl/stiffness_kernels.hpp"
#include "specfem/execution.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <string>

// The stiffness kernels must build every scratch view on the scratch level
// the launch policy reserved. compute_stiffness_interaction_core falls back to
// level-1 (global-memory) scratch when neither kernel fits level 0; this suite
// forces both levels on the same input and requires identical accelerations.
//
// On Serial/OpenMP both levels are carved from one host scratch pool with no
// bounds checking, so a view built on the wrong level still lands in valid
// memory and these tests pass even with the mismatch. They exercise the
// level-1 path on host but only detect a mismatch on a CUDA/HIP build.
namespace stiffness_scratch_level_test {

constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;
constexpr auto elastic_tag = specfem::element::medium_tag::elastic;
constexpr auto forward_tag = specfem::simulation::field_type::forward;
constexpr int NGLL = 5;

constexpr bool single_precision = sizeof(type_real) == sizeof(float);

using AssemblyType = specfem::assembly::assembly<dim3_tag>;

template <specfem::element::mpi_tag MpiTag>
using KernelTags = specfem::tags::Tags<
    dim3_tag, elastic_tag, specfem::element::property_tag::isotropic,
    specfem::element::attenuation_tag::none,
    specfem::element::boundary_tag::none, forward_tag, MpiTag>;

using HostField = Kokkos::View<type_real **, Kokkos::LayoutLeft,
                               Kokkos::DefaultHostExecutionSpace>;

// Same fixture as element_stiffness_tests.cpp.
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
  [[maybe_unused]] const auto unused_starttime = starttime;
  setup.update_t0(t0);

  auto receivers = specfem::io::read_3d_receivers(stations_node);

  return std::make_unique<AssemblyType>(
      mesh, quadratures, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, setup.get_max_seismogram_step(),
      setup.get_nstep_between_samples(), setup.get_simulation_type(),
      setup.allocate_boundary_values(), setup.instantiate_property_reader());
}

// Launch Kernel on the elements carrying MpiTag, reserving and building all
// scratch on `scratch_level`. Mirrors compute_stiffness_interaction_core.
template <template <int, typename> class Kernel,
          specfem::element::mpi_tag MpiTag>
int launch_on_level(const AssemblyType &assembly, const int scratch_level) {
  using Tags = KernelTags<MpiTag>;
  using KernelType = Kernel<NGLL, Tags>;

  const auto elements = assembly.element_types.get_elements_on_device(
      Tags::medium_tag, Tags::property_tag, Tags::attenuation_tag,
      Tags::boundary_tag, Tags::mpi_tag);
  const int nelements = elements.extent(0);
  if (nelements == 0)
    return 0;

  specfem::mesh_entity::element_grid<dim3_tag, specfem::mesh_entity::Grid<NGLL>>
      element_grid{};
  specfem::execution::ChunkedDomainIterator chunk(
      typename KernelType::ParallelConfig(), elements, element_grid);

  if (KernelType::shmem_size() > chunk.scratch_size_max(scratch_level)) {
    return -1;
  }

  KernelType{ assembly, 0, scratch_level }(chunk.set_scratch_size(
      scratch_level, Kokkos::PerTeam(KernelType::shmem_size())));
  Kokkos::fence();
  return nelements;
}

class StiffnessScratchLevel : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    assembly_ = build_assembly_3d("HomogeneousHalfspaceSmallNoABCForceSource")
                    .release();
  }

  static void TearDownTestSuite() {
    delete assembly_;
    assembly_ = nullptr;
  }

  // Fill displacement and velocity with a fixed-seed random field, zero the
  // acceleration, run Kernel on `scratch_level` over all elements and return
  // a host copy of the resulting acceleration.
  template <template <int, typename> class Kernel>
  static HostField acceleration_on_level(const int scratch_level) {
    auto &field = assembly_->fields.get_simulation_field<forward_tag>();
    auto &field_impl = field.get_field<elastic_tag>();
    const auto h_u = field_impl.get_host_field();
    const auto h_v = field_impl.get_host_field_dot();
    const auto h_a = field_impl.get_host_field_dot_dot();

    std::mt19937 rng(20251104);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int iglob = 0; iglob < static_cast<int>(h_u.extent(0)); ++iglob) {
      for (int icomp = 0; icomp < static_cast<int>(h_u.extent(1)); ++icomp) {
        h_u(iglob, icomp) = static_cast<type_real>(dist(rng));
        h_v(iglob, icomp) = static_cast<type_real>(dist(rng));
      }
    }
    Kokkos::deep_copy(h_a, 0);
    assembly_->fields.copy_to_device();

    const int n_inner =
        launch_on_level<Kernel, specfem::element::mpi_tag::inner>(
            *assembly_, scratch_level);
    const int n_outer =
        launch_on_level<Kernel, specfem::element::mpi_tag::outer>(
            *assembly_, scratch_level);
    if (n_inner < 0 || n_outer < 0) {
      return HostField();
    }
    EXPECT_GT(n_inner + n_outer, 0) << "no elastic isotropic elements found";

    assembly_->fields.copy_to_host();

    HostField result("acceleration", h_a.extent(0), h_a.extent(1));
    Kokkos::deep_copy(result, h_a);
    return result;
  }

  static void expect_close(const HostField &expected, const HostField &actual,
                           const std::string &what) {
    ASSERT_EQ(expected.extent(0), actual.extent(0)) << what;
    ASSERT_EQ(expected.extent(1), actual.extent(1)) << what;

    type_real scale = 0;
    for (int i = 0; i < static_cast<int>(expected.extent(0)); ++i)
      for (int c = 0; c < static_cast<int>(expected.extent(1)); ++c)
        scale = std::max(scale, std::abs(expected(i, c)));
    ASSERT_GT(scale, type_real(0)) << what << ": acceleration is all zero";

    // Atomic accumulation into the global field makes summation order
    // nondeterministic on GPUs; compare relative to the field's magnitude.
    const type_real tol =
        static_cast<type_real>(single_precision ? 1e-5 : 1e-12) * scale;
    for (int i = 0; i < static_cast<int>(expected.extent(0)); ++i)
      for (int c = 0; c < static_cast<int>(expected.extent(1)); ++c)
        ASSERT_NEAR(expected(i, c), actual(i, c), tol)
            << what << " at iglob=" << i << " icomp=" << c;
  }

  static AssemblyType *assembly_;
};

AssemblyType *StiffnessScratchLevel::assembly_ = nullptr;

TEST_F(StiffnessScratchLevel, GatherLevel1MatchesLevel0) {
  const auto a0 =
      acceleration_on_level<specfem::compute::impl::gather_kernel>(0);
  const auto a1 =
      acceleration_on_level<specfem::compute::impl::gather_kernel>(1);
  if (a0.extent(0) == 0 || a1.extent(0) == 0)
    GTEST_SKIP() << "gather kernel does not fit one of the scratch levels";
  expect_close(a0, a1, "gather level 1 vs level 0");
}

TEST_F(StiffnessScratchLevel, ScatterLevel1MatchesLevel0) {
  const auto a0 =
      acceleration_on_level<specfem::compute::impl::scatter_kernel>(0);
  const auto a1 =
      acceleration_on_level<specfem::compute::impl::scatter_kernel>(1);
  if (a0.extent(0) == 0 || a1.extent(0) == 0)
    GTEST_SKIP() << "scatter kernel does not fit one of the scratch levels";
  expect_close(a0, a1, "scatter level 1 vs level 0");
}

TEST_F(StiffnessScratchLevel, GatherMatchesScatterOnLevel1) {
  const auto gather =
      acceleration_on_level<specfem::compute::impl::gather_kernel>(1);
  const auto scatter =
      acceleration_on_level<specfem::compute::impl::scatter_kernel>(1);
  if (gather.extent(0) == 0 || scatter.extent(0) == 0)
    GTEST_SKIP() << "a kernel does not fit level-1 scratch";
  expect_close(gather, scatter, "gather vs scatter on level 1");
}

} // namespace stiffness_scratch_level_test

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
