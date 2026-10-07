#include "../SPECFEM_Environment.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/compute/impl/compute_stiffness_interaction.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/io.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;
constexpr auto elastic_tag = specfem::element::medium_tag::elastic;
constexpr auto forward_tag = specfem::simulation::field_type::forward;
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

// Decode a local dof index back into (icomp, iz, iy, ix); inverse of
// specfem::linear_system::local_dof_index<NGLL>.
struct LocalDof {
  int icomp;
  int iz;
  int iy;
  int ix;
};

LocalDof decode_local_dof(const int jdof) {
  const int points = NGLL * NGLL * NGLL;
  const int icomp = jdof / points;
  const int rem = jdof % points;
  return { icomp, rem / (NGLL * NGLL), (rem / NGLL) % NGLL, rem % NGLL };
}

// Shared fixture: the small homogeneous elastic mesh without absorbing
// boundaries is built once for the whole suite and torn down before the
// global environment finalizes Kokkos.
class ElementStiffness3D : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    assembly_ = build_assembly_3d("HomogeneousHalfspaceSmallNoABCForceSource")
                    .release();
  }

  static void TearDownTestSuite() {
    delete assembly_;
    assembly_ = nullptr;
  }

  // Compute the dense stiffness block of a single element.
  static StiffnessView::host_mirror_type element_block(const int ispec) {
    const specfem::datatype::ElementIndexRange batch(ispec, ispec + 1);
    StiffnessView k_e("k_e", 1, ndof, ndof);
    specfem::linear_system::compute_element_stiffness<StiffnessTags>(
        *assembly_, batch, k_e);
    auto h_k = Kokkos::create_mirror_view(k_e);
    Kokkos::deep_copy(h_k, k_e);
    return h_k;
  }

  static AssemblyType *assembly_;
};

AssemblyType *ElementStiffness3D::assembly_ = nullptr;

TEST_F(ElementStiffness3D, ValidatesScopeOnCleanMesh) {
  EXPECT_NO_THROW(
      specfem::linear_system::validate_stiffness_scope<StiffnessTags>(
          *assembly_));
}

TEST_F(ElementStiffness3D, SymmetricWithRigidBodyNullSpace) {
  const auto elements =
      assembly_->element_types.get_elements_on_host(elastic_tag);
  ASSERT_GT(elements.size(), 0);

  const auto h_k = element_block(elements(0));

  type_real scale = 0;
  for (int i = 0; i < ndof; ++i) {
    for (int j = 0; j < ndof; ++j) {
      scale = std::max(scale, std::abs(h_k(0, i, j)));
    }
  }
  ASSERT_GT(scale, static_cast<type_real>(0));

  // K = integral of grad(phi_i) : C : grad(phi_j) is symmetric for elastic
  // isotropic media; the probe computes K(i, j) and K(j, i) through
  // independent operator applications, so they match only up to roundoff.
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

TEST_F(ElementStiffness3D, MatchesMatrixFreeOperator) {
  const auto elements =
      assembly_->element_types.get_elements_on_host(elastic_tag);
  ASSERT_GT(elements.size(), 0);
  const int ispec = elements(0);

  const auto h_k = element_block(ispec);

  auto &field = assembly_->fields.template get_simulation_field<forward_tag>();
  const auto &field_impl = field.template get_field<elastic_tag>();
  const auto h_u = field_impl.get_host_field();
  const auto h_a = field_impl.get_host_field_dot_dot();
  const auto h_v = field_impl.get_host_field_dot();
  const int nglob = field_impl.nglob;

  auto zero_fields_on_host = [&]() {
    for (int iglob = 0; iglob < nglob; ++iglob) {
      for (int icomp = 0; icomp < ncomp; ++icomp) {
        h_u(iglob, icomp) = 0;
        h_v(iglob, icomp) = 0;
        h_a(iglob, icomp) = 0;
      }
    }
  };

  // Apply the production matrix-free stiffness operator (velocity == 0, no
  // mass division) and pull the assembled acceleration back to the host.
  auto apply_matrix_free = [&]() {
    assembly_->fields.copy_to_device();
    using BaseTags = specfem::tags::Tags<dim3_tag, forward_tag, elastic_tag>;
    specfem::compute::impl::compute_stiffness_interaction<
        NGLL,
        specfem::tags::expand<BaseTags, specfem::element::mpi_tag::outer>>(
        *assembly_, 0);
    specfem::compute::impl::compute_stiffness_interaction<
        NGLL,
        specfem::tags::expand<BaseTags, specfem::element::mpi_tag::inner>>(
        *assembly_, 0);
    assembly_->fields.copy_to_host();
  };

  auto local_displacement = [&]() {
    std::vector<type_real> u_e(ndof);
    for (int jdof = 0; jdof < ndof; ++jdof) {
      const auto dof = decode_local_dof(jdof);
      const int iglob = field.template get_iglob<false, elastic_tag>(
          ispec, dof.iz, dof.iy, dof.ix);
      u_e[jdof] = h_u(iglob, dof.icomp);
    }
    return u_e;
  };

  // The matrix-free kernel accumulates accel += -(divergence result), so the
  // assembled acceleration equals -K u (before mass division).
  auto expected_row = [&](const std::vector<type_real> &u_e, const int idof) {
    type_real value = 0;
    for (int jdof = 0; jdof < ndof; ++jdof) {
      value += h_k(0, idof, jdof) * u_e[jdof];
    }
    return -value;
  };

  std::mt19937 generator(12345);
  std::uniform_real_distribution<type_real> distribution(-1, 1);

  const type_real rel_tol = single_precision ? static_cast<type_real>(2e-3)
                                             : static_cast<type_real>(1e-10);

  // Probe 1: displacement supported on the element's interior GLL points.
  // No other element sees a nonzero displacement, so the assembled
  // acceleration equals -K_e u_e at ALL of this element's dofs -- this
  // checks every row of the block.
  {
    zero_fields_on_host();
    for (int iz = 1; iz < NGLL - 1; ++iz) {
      for (int iy = 1; iy < NGLL - 1; ++iy) {
        for (int ix = 1; ix < NGLL - 1; ++ix) {
          const int iglob =
              field.template get_iglob<false, elastic_tag>(ispec, iz, iy, ix);
          for (int icomp = 0; icomp < ncomp; ++icomp) {
            h_u(iglob, icomp) = distribution(generator);
          }
        }
      }
    }
    apply_matrix_free();

    const auto u_e = local_displacement();
    type_real scale = 0;
    type_real max_diff = 0;
    for (int idof = 0; idof < ndof; ++idof) {
      const auto dof = decode_local_dof(idof);
      const int iglob = field.template get_iglob<false, elastic_tag>(
          ispec, dof.iz, dof.iy, dof.ix);
      const type_real expected = expected_row(u_e, idof);
      const type_real actual = h_a(iglob, dof.icomp);
      scale = std::max(scale, std::abs(expected));
      max_diff = std::max(max_diff, std::abs(expected - actual));
    }
    ASSERT_GT(scale, static_cast<type_real>(0));
    EXPECT_LE(max_diff, rel_tol * scale)
        << "K_e rows disagree with the matrix-free operator";
  }

  // Probe 2: random displacement over the whole mesh. Interior GLL points
  // belong exclusively to this element, so the assembled acceleration there
  // equals -K_e u_e -- this checks every column of the block.
  {
    zero_fields_on_host();
    for (int iglob = 0; iglob < nglob; ++iglob) {
      for (int icomp = 0; icomp < ncomp; ++icomp) {
        h_u(iglob, icomp) = distribution(generator);
      }
    }
    apply_matrix_free();

    const auto u_e = local_displacement();
    type_real scale = 0;
    type_real max_diff = 0;
    for (int iz = 1; iz < NGLL - 1; ++iz) {
      for (int iy = 1; iy < NGLL - 1; ++iy) {
        for (int ix = 1; ix < NGLL - 1; ++ix) {
          const int iglob =
              field.template get_iglob<false, elastic_tag>(ispec, iz, iy, ix);
          for (int icomp = 0; icomp < ncomp; ++icomp) {
            const int idof = specfem::linear_system::local_dof_index<NGLL>(
                icomp, iz, iy, ix);
            const type_real expected = expected_row(u_e, idof);
            const type_real actual = h_a(iglob, icomp);
            scale = std::max(scale, std::abs(expected));
            max_diff = std::max(max_diff, std::abs(expected - actual));
          }
        }
      }
    }
    ASSERT_GT(scale, static_cast<type_real>(0));
    EXPECT_LE(max_diff, rel_tol * scale)
        << "K_e columns disagree with the matrix-free operator";
  }
}

TEST(ElementStiffnessScope3D, RejectsStaceyBoundaries) {
  const auto stacey_assembly = build_assembly_3d("HomogeneousHalfSpaceStacey");
  EXPECT_THROW(specfem::linear_system::validate_stiffness_scope<StiffnessTags>(
                   *stacey_assembly),
               std::runtime_error);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
