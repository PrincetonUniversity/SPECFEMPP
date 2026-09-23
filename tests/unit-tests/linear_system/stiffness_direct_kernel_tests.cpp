#include "../SPECFEM_Environment.hpp"
#include "specfem/algorithms.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/io.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/mesh.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

// Tests of the direct (sum-factored) element stiffness formulation.
//
// The closed form is
//   K_e(a,i; b,j) = sum_{r,s} sum_q h(q, i_r) M(a,b,r,s; q in slot r, i
//   elsewhere) S_{r,s}
// with the weighted reference-frame constitutive tensor
//   M(a,b,r,s; q) = w(q) J(q) sum_{c,d} xi_{r,c} C_{acbd} xi_{s,d}
// and
//   S_{r,s} = h(q, j_r) prod_{t != r} [i_t == j_t]                  (s == r)
//   S_{r,s} = [q == j_r] h(i_s, j_s) prod_{t != r,s} [i_t == j_t]   (s != r)
//
// The host reference below implements exactly that from the host mirrors of
// the mesh and the constitutive_tensor accessor, independently of every
// device kernel. The probe kernel is the oracle it is held to.
namespace stiffness_direct_kernel_test {

constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;
constexpr auto elastic_tag = specfem::element::medium_tag::elastic;
constexpr int NGLL = 5;
constexpr int ncomp = 3;
constexpr int ndim = 3;
constexpr int npoints = NGLL * NGLL * NGLL;
constexpr int ndof = ncomp * npoints;

constexpr bool single_precision = sizeof(type_real) == sizeof(float);

type_real scaled_tolerance(const double single_tol, const double double_tol,
                           const type_real scale) {
  return static_cast<type_real>(single_precision ? single_tol : double_tol) *
         scale;
}

using AssemblyType = specfem::assembly::assembly<dim3_tag>;
using StiffnessTags =
    specfem::tags::Tags<dim3_tag, elastic_tag,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none>;
using PointTags =
    specfem::tags::Tags<dim3_tag, elastic_tag,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none, false>;
using StiffnessView = Kokkos::View<type_real ***, Kokkos::LayoutRight,
                                   Kokkos::DefaultExecutionSpace>;
using KernelImpl = specfem::linear_system::StiffnessKernelImpl;
using PointIndexType = specfem::point::index<dim3_tag, false>;
using PointJacobianMatrixType =
    specfem::point::jacobian_matrix<dim3_tag, true, false>;
using PointPropertyType = specfem::point::properties<PointTags>;
using PointFieldDerivativesType = specfem::point::field_derivatives<PointTags>;

// Same fixture as element_stiffness_tests.cpp / stiffness_tensor_graph_tests.
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

/// Weighted reference-frame constitutive tensor of one element on the host:
/// M(a,b,r,s; z,y,x) = w J sum_{c,d} xi_{r,c} C_{acbd} xi_{s,d}, flattened as
/// ((((a * ncomp + b) * ndim + r) * ndim + s) * npoints + point) with
/// point = (iz * NGLL + iy) * NGLL + ix.
struct ReferenceConstitutive {
  std::vector<double> values;

  ReferenceConstitutive() : values(ncomp * ncomp * ndim * ndim * npoints) {}

  static int point_index(const int iz, const int iy, const int ix) {
    return (iz * NGLL + iy) * NGLL + ix;
  }

  double &operator()(const int a, const int b, const int r, const int s,
                     const int point) {
    return values[(((a * ncomp + b) * ndim + r) * ndim + s) * npoints + point];
  }
  double operator()(const int a, const int b, const int r, const int s,
                    const int point) const {
    return values[(((a * ncomp + b) * ndim + r) * ndim + s) * npoints + point];
  }
};

ReferenceConstitutive reference_constitutive(const AssemblyType &assembly,
                                             const int ispec) {
  const auto &h_weights = assembly.mesh.h_weights;
  ReferenceConstitutive M;

  for (int iz = 0; iz < NGLL; ++iz) {
    for (int iy = 0; iy < NGLL; ++iy) {
      for (int ix = 0; ix < NGLL; ++ix) {
        const PointIndexType index(ispec, iz, iy, ix);

        PointJacobianMatrixType jacobian;
        specfem::assembly::load_on_host(index, assembly.jacobian_matrix,
                                        jacobian);
        PointPropertyType properties;
        specfem::assembly::load_on_host(index, assembly.properties, properties);

        const double wJ = static_cast<double>(h_weights(ix)) *
                          static_cast<double>(h_weights(iy)) *
                          static_cast<double>(h_weights(iz)) *
                          static_cast<double>(jacobian.jacobian());

        // xi(c, r) = d xi_r / d x_c (row = spatial, column = reference).
        const auto &xi = jacobian.tensor();
        const int point = ReferenceConstitutive::point_index(iz, iy, ix);

        for (int a = 0; a < ncomp; ++a) {
          for (int b = 0; b < ncomp; ++b) {
            for (int r = 0; r < ndim; ++r) {
              for (int s = 0; s < ndim; ++s) {
                double sum = 0;
                for (int c = 0; c < ndim; ++c) {
                  for (int d = 0; d < ndim; ++d) {
                    sum += static_cast<double>(xi(c, r)) *
                           static_cast<double>(
                               specfem::medium_physics::constitutive_tensor<
                                   PointTags>(properties, a, c, b, d)) *
                           static_cast<double>(xi(d, s));
                  }
                }
                M(a, b, r, s, point) = wJ * sum;
              }
            }
          }
        }
      }
    }
  }
  return M;
}

/// Closed-form K_e of one element on the host, row/column ordering per
/// specfem::linear_system::local_dof_index.
std::vector<double> reference_stiffness(const AssemblyType &assembly,
                                        const int ispec) {
  const auto &h = assembly.mesh.h_hprime; // h(point, function)
  const auto M = reference_constitutive(assembly, ispec);

  std::vector<double> k(static_cast<std::size_t>(ndof) * ndof, 0.0);

  // i = (i_x, i_y, i_z) so that slot index r matches the direction index
  // used by the Jacobian (0 = x, 1 = y, 2 = z).
  std::array<int, ndim> i{};
  std::array<int, ndim> j{};

  for (int a = 0; a < ncomp; ++a) {
    for (i[2] = 0; i[2] < NGLL; ++i[2]) {
      for (i[1] = 0; i[1] < NGLL; ++i[1]) {
        for (i[0] = 0; i[0] < NGLL; ++i[0]) {
          const int row = specfem::linear_system::local_dof_index<NGLL>(
              a, i[2], i[1], i[0]);
          for (int b = 0; b < ncomp; ++b) {
            for (j[2] = 0; j[2] < NGLL; ++j[2]) {
              for (j[1] = 0; j[1] < NGLL; ++j[1]) {
                for (j[0] = 0; j[0] < NGLL; ++j[0]) {
                  const int col = specfem::linear_system::local_dof_index<NGLL>(
                      b, j[2], j[1], j[0]);

                  double value = 0;
                  for (int r = 0; r < ndim; ++r) {
                    for (int s = 0; s < ndim; ++s) {
                      // Predicates on the directions other than r (and s).
                      bool same = true;
                      for (int t = 0; t < ndim; ++t) {
                        if (t != r && t != s && i[t] != j[t]) {
                          same = false;
                        }
                      }
                      if (!same) {
                        continue;
                      }
                      for (int q = 0; q < NGLL; ++q) {
                        std::array<int, ndim> p = i;
                        p[r] = q;
                        const int point = ReferenceConstitutive::point_index(
                            p[2], p[1], p[0]);
                        const double m = M(a, b, r, s, point);
                        const double row_factor =
                            static_cast<double>(h(q, i[r]));
                        double column_factor = 0;
                        if (s == r) {
                          column_factor = static_cast<double>(h(q, j[r]));
                        } else if (q == j[r]) {
                          column_factor = static_cast<double>(h(i[s], j[s]));
                        }
                        value += row_factor * m * column_factor;
                      }
                    }
                  }
                  k[static_cast<std::size_t>(row) * ndof + col] = value;
                }
              }
            }
          }
        }
      }
    }
  }
  return k;
}

class DirectStiffness3D : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    assembly_ = build_assembly_3d("HomogeneousHalfspaceSmallNoABCForceSource")
                    .release();
  }

  static void TearDownTestSuite() {
    delete assembly_;
    assembly_ = nullptr;
  }

  // Contiguous compute-domain range over every elastic element (asserted).
  static specfem::datatype::ElementIndexRange elastic_range() {
    const auto elements =
        assembly_->element_types.get_elements_on_host(elastic_tag);
    const int nelements = elements.size();
    EXPECT_GT(nelements, 0);
    EXPECT_EQ(elements(nelements - 1) - elements(0), nelements - 1)
        << "fixture's elastic elements are not contiguous";
    return { elements(0), elements(0) + nelements };
  }

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

AssemblyType *DirectStiffness3D::assembly_ = nullptr;

// The reference-frame constitutive tensor built from the accessor equals the
// one obtained by pushing a unit reference gradient through the production
// chain_rule -> compute_stress -> stress * jacobian pipeline at every point
// of one element. Pins constitutive_tensor to compute_stress inside the
// exact index conventions the kernel relies on.
TEST_F(DirectStiffness3D, ReferenceConstitutiveMatchesStressPipeline) {
  const auto range = elastic_range();
  const int ispec = range.begin_index();
  const auto M = reference_constitutive(*assembly_, ispec);
  const auto &h_weights = assembly_->mesh.h_weights;

  double scale = 0;
  double max_diff = 0;
  for (int iz = 0; iz < NGLL; ++iz) {
    for (int iy = 0; iy < NGLL; ++iy) {
      for (int ix = 0; ix < NGLL; ++ix) {
        const PointIndexType index(ispec, iz, iy, ix);
        PointJacobianMatrixType jacobian;
        specfem::assembly::load_on_host(index, assembly_->jacobian_matrix,
                                        jacobian);
        PointPropertyType properties;
        specfem::assembly::load_on_host(index, assembly_->properties,
                                        properties);
        const double w = static_cast<double>(h_weights(ix)) *
                         static_cast<double>(h_weights(iy)) *
                         static_cast<double>(h_weights(iz));
        const int point = ReferenceConstitutive::point_index(iz, iy, ix);

        for (int s = 0; s < ndim; ++s) {
          for (int b = 0; b < ncomp; ++b) {
            type_real du_dxi[ncomp] = { 0, 0, 0 };
            type_real du_deta[ncomp] = { 0, 0, 0 };
            type_real du_dgamma[ncomp] = { 0, 0, 0 };
            (s == 0 ? du_dxi : (s == 1 ? du_deta : du_dgamma))[b] = 1;
            const PointFieldDerivativesType field_derivatives(
                specfem::algorithms::chain_rule(jacobian, du_dxi, du_deta,
                                                du_dgamma));
            const auto F = specfem::medium_physics::compute_stress<PointTags>(
                               properties, field_derivatives) *
                           jacobian;
            for (int a = 0; a < ncomp; ++a) {
              for (int r = 0; r < ndim; ++r) {
                const double expected = w * static_cast<double>(F(a, r));
                const double actual = M(a, b, r, s, point);
                scale = std::max(scale, std::abs(expected));
                max_diff = std::max(max_diff, std::abs(expected - actual));
              }
            }
          }
        }
      }
    }
  }
  ASSERT_GT(scale, 0.0);
  EXPECT_LE(max_diff, static_cast<double>(scaled_tolerance(
                          1e-5, 1e-12, static_cast<type_real>(scale))));
}

// The closed form, evaluated on the host from the accessor and the host
// quadrature/Jacobian data, reproduces the probe kernel's blocks.
TEST_F(DirectStiffness3D, ClosedFormReferenceMatchesProbe) {
  const auto range = elastic_range();
  const int nelements = range.size();

  StiffnessView k_probe("k_probe", nelements, ndof, ndof);
  const double probe_ms = fill_blocks(k_probe, KernelImpl::probe, range);
  auto h_probe =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, k_probe);

  // The host reference is O(ndof^2 * 45) per element; a stride keeps the
  // test quick while still covering elements across the whole mesh.
  const int stride = std::max(1, nelements / 48);

  type_real scale = 0;
  type_real max_diff = 0;
  int worst_e = 0, worst_i = 0, worst_j = 0;
  int checked = 0;
  Kokkos::Timer timer;
  for (int e = 0; e < nelements; e += stride) {
    const auto k_ref = reference_stiffness(*assembly_, range.begin_index() + e);
    ++checked;
    for (int i = 0; i < ndof; ++i) {
      for (int j = 0; j < ndof; ++j) {
        const type_real probe = h_probe(e, i, j);
        const type_real ref = static_cast<type_real>(
            k_ref[static_cast<std::size_t>(i) * ndof + j]);
        scale = std::max(scale, std::abs(probe));
        const type_real diff = std::abs(probe - ref);
        if (diff > max_diff) {
          max_diff = diff;
          worst_e = e;
          worst_i = i;
          worst_j = j;
        }
      }
    }
  }
  const double reference_ms = timer.seconds() * 1e3;
  std::printf("[ timing   ] probe: %.2f ms (%d elements); host closed-form "
              "reference: %.2f ms (%d elements)\n",
              probe_ms, nelements, reference_ms, checked);

  ASSERT_GT(scale, static_cast<type_real>(0));
  const type_real tol = scaled_tolerance(1e-4, 1e-12, scale);
  EXPECT_LE(max_diff, tol) << "worst entry at (e=" << worst_e
                           << ", i=" << worst_i << ", j=" << worst_j
                           << "): probe=" << h_probe(worst_e, worst_i, worst_j)
                           << " scale=" << scale;
}

} // namespace stiffness_direct_kernel_test

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
