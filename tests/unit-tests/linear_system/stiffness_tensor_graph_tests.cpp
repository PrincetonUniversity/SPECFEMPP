#include "../SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

#ifdef SPECFEM_ENABLE_TENSOROPS

#include "specfem/assembly/assembly.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/io.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

// A/B test holding the tensor-graph kernel to the probe kernel: same inputs,
// same K_e contract, agreement to roundoff. The probe is the oracle -- it is
// correct by construction (it applies the production matrix-free operator one
// unit column at a time), while the tensor-graph kernel evaluates the same
// action on all columns at once through a TensorOperations level graph.
namespace stiffness_tensor_graph_test {

constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;
constexpr auto elastic_tag = specfem::element::medium_tag::elastic;
constexpr int NGLL = 5;
constexpr int ncomp = 3;
constexpr int ndof = ncomp * NGLL * NGLL * NGLL;

constexpr bool single_precision = sizeof(type_real) == sizeof(float);

using AssemblyType = specfem::assembly::assembly<dim3_tag>;
using StiffnessTags =
    specfem::tags::Tags<dim3_tag, elastic_tag,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none>;
using StiffnessView = Kokkos::View<type_real ***, Kokkos::LayoutRight,
                                   Kokkos::DefaultExecutionSpace>;
using KernelImpl = specfem::linear_system::StiffnessKernelImpl;

// Build a full assembly from a Newmark displacement-test dataset; same
// fixture as element_stiffness_tests.cpp.
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

class TensorGraphStiffness3D : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    assembly_ = build_assembly_3d("HomogeneousHalfspaceSmallNoABCForceSource")
                    .release();
  }

  static void TearDownTestSuite() {
    delete assembly_;
    assembly_ = nullptr;
  }

  // The contiguous compute-domain range covering every elastic element; the
  // single-medium fixture guarantees contiguity (asserted).
  static specfem::datatype::ElementIndexRange elastic_range() {
    const auto elements =
        assembly_->element_types.get_elements_on_host(elastic_tag);
    const int nelements = elements.size();
    EXPECT_GT(nelements, 0);
    EXPECT_EQ(elements(nelements - 1) - elements(0), nelements - 1)
        << "fixture's elastic elements are not contiguous";
    return { elements(0), elements(0) + nelements };
  }

  // Fill blocks for the whole range with the given kernel; returns wall time
  // (kernel launch through fence) in milliseconds.
  static double fill_blocks(const StiffnessView &k_e, const KernelImpl impl,
                            const specfem::datatype::ElementIndexRange &range) {
    Kokkos::fence();
    Kokkos::Timer timer;
    specfem::linear_system::compute_element_stiffness<StiffnessTags>(
        *assembly_, range, k_e, impl);
    return timer.seconds() * 1e3;
  }

  static AssemblyType *assembly_;
};

AssemblyType *TensorGraphStiffness3D::assembly_ = nullptr;

TEST_F(TensorGraphStiffness3D, AgreesWithProbeKernelWithTiming) {
  const auto range = elastic_range();
  const int nelements = range.size();

  StiffnessView k_probe("k_probe", nelements, ndof, ndof);
  StiffnessView k_graph("k_graph", nelements, ndof, ndof);

  // Warm-up runs first (view allocations, first-touch); the timed pair after.
  fill_blocks(k_probe, KernelImpl::probe, range);
  fill_blocks(k_graph, KernelImpl::tensor_graph, range);
  const double probe_ms = fill_blocks(k_probe, KernelImpl::probe, range);
  const double graph_ms = fill_blocks(k_graph, KernelImpl::tensor_graph, range);

  std::printf("[ timing   ] probe: %.2f ms, tensor_graph: %.2f ms "
              "(%d elements, speedup %.1fx)\n",
              probe_ms, graph_ms, nelements, probe_ms / graph_ms);

  auto h_probe =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, k_probe);
  auto h_graph =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, k_graph);

  type_real scale = 0;
  for (int e = 0; e < nelements; ++e) {
    for (int i = 0; i < ndof; ++i) {
      for (int j = 0; j < ndof; ++j) {
        scale = std::max(scale, std::abs(h_probe(e, i, j)));
      }
    }
  }
  ASSERT_GT(scale, static_cast<type_real>(0));

  type_real max_diff = 0;
  int worst_e = 0, worst_i = 0, worst_j = 0;
  for (int e = 0; e < nelements; ++e) {
    for (int i = 0; i < ndof; ++i) {
      for (int j = 0; j < ndof; ++j) {
        const type_real diff = std::abs(h_probe(e, i, j) - h_graph(e, i, j));
        if (diff > max_diff) {
          max_diff = diff;
          worst_e = e;
          worst_i = i;
          worst_j = j;
        }
      }
    }
  }

  // The two kernels evaluate the same action through different operation
  // orders (serialized probes against fused level contractions), so they
  // agree only to roundoff of the largest entry.
  const type_real tol = (single_precision ? static_cast<type_real>(1e-4)
                                          : static_cast<type_real>(1e-12)) *
                        scale;
  EXPECT_LE(max_diff, tol) << "worst entry at (e=" << worst_e
                           << ", i=" << worst_i << ", j=" << worst_j
                           << "): probe=" << h_probe(worst_e, worst_i, worst_j)
                           << " tensor_graph="
                           << h_graph(worst_e, worst_i, worst_j)
                           << " scale=" << scale;
}

TEST_F(TensorGraphStiffness3D, SymmetricWithRigidBodyNullSpace) {
  const auto range = elastic_range();
  const specfem::datatype::ElementIndexRange first(range.begin_index(),
                                                   range.begin_index() + 1);
  StiffnessView k_e("k_e", 1, ndof, ndof);
  specfem::linear_system::compute_element_stiffness<StiffnessTags>(
      *assembly_, first, k_e, KernelImpl::tensor_graph);
  auto h_k = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, k_e);

  type_real scale = 0;
  for (int i = 0; i < ndof; ++i) {
    for (int j = 0; j < ndof; ++j) {
      scale = std::max(scale, std::abs(h_k(0, i, j)));
    }
  }
  ASSERT_GT(scale, static_cast<type_real>(0));

  // K is symmetric for elastic isotropic media; the kernel computes K(i, j)
  // and K(j, i) from independent unit columns, so they match only up to
  // roundoff.
  type_real max_asymmetry = 0;
  for (int i = 0; i < ndof; ++i) {
    for (int j = i + 1; j < ndof; ++j) {
      max_asymmetry =
          std::max(max_asymmetry, std::abs(h_k(0, i, j) - h_k(0, j, i)));
    }
  }
  const type_real symmetry_tol =
      (single_precision ? static_cast<type_real>(1e-4)
                        : static_cast<type_real>(1e-12)) *
      scale;
  EXPECT_LE(max_asymmetry, symmetry_tol);

  // Rigid translations produce zero strain, so every row of K must sum to
  // zero over the columns of each component block.
  type_real max_null = 0;
  for (int icomp = 0; icomp < ncomp; ++icomp) {
    for (int i = 0; i < ndof; ++i) {
      type_real row_sum = 0;
      for (int p = 0; p < NGLL * NGLL * NGLL; ++p) {
        row_sum += h_k(0, i, icomp * NGLL * NGLL * NGLL + p);
      }
      max_null = std::max(max_null, std::abs(row_sum));
    }
  }
  const type_real null_tol =
      (single_precision ? static_cast<type_real>(5e-3)
                        : static_cast<type_real>(1e-10)) *
      scale;
  EXPECT_LE(max_null, null_tol);
}

} // namespace stiffness_tensor_graph_test

#else // !SPECFEM_ENABLE_TENSOROPS

TEST(TensorGraphStiffness3D, SkippedWithoutTensorOps) {
  GTEST_SKIP() << "SPECFEM++ was built without TensorOperations "
                  "(SPECFEM_ENABLE_TENSOROPS=OFF); the tensor-graph stiffness "
                  "kernel is unavailable.";
}

#endif // SPECFEM_ENABLE_TENSOROPS

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
