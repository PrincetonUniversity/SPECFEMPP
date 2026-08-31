#include "../SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/compute/impl/compute_stiffness_interaction.hpp"
#include "specfem/io.hpp"
#include "specfem/linear_system/tpetra_assembler.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

namespace stiffness_assembler_test {

constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;
constexpr auto elastic_tag = specfem::element::medium_tag::elastic;
constexpr auto forward_tag = specfem::simulation::field_type::forward;
constexpr int NGLL = 5;
constexpr int ncomp = 3;

constexpr bool single_precision = sizeof(type_real) == sizeof(float);

using AssemblyType = specfem::assembly::assembly<dim3_tag>;
using HostFieldView = specfem::linear_system::SystemLayout<specfem::tags::Tags<
    dim3_tag, elastic_tag, specfem::element::property_tag::isotropic,
    specfem::element::attenuation_tag::none>>::host_field_view_type;
using StiffnessTags =
    specfem::tags::Tags<dim3_tag, elastic_tag,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none>;
using AssemblerType = specfem::linear_system::StiffnessAssembler<StiffnessTags>;
using VectorType =
    Tpetra::Vector<specfem::linear_system::scalar_type,
                   specfem::linear_system::crs_matrix_type::local_ordinal_type,
                   specfem::linear_system::global_ordinal_type>;

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

// Shared fixture: assemble the matrix of the small homogeneous elastic mesh
// once for the whole suite; assembly happens on the first access so that a
// throwing constructor fails inside a test body, not in SetUpTestSuite.
class StiffnessAssembler3D : public ::testing::Test {
protected:
  static void TearDownTestSuite() {
    matrix_ = Teuchos::null;
    assembler_.reset();
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

  static AssemblerType &assembler() {
    if (!assembler_) {
      assembler_ = std::make_unique<AssemblerType>(assembly());
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
  static std::unique_ptr<AssemblerType> assembler_;
  static Teuchos::RCP<specfem::linear_system::crs_matrix_type> matrix_;
};

AssemblyType *StiffnessAssembler3D::assembly_ = nullptr;
std::unique_ptr<AssemblerType> StiffnessAssembler3D::assembler_;
Teuchos::RCP<specfem::linear_system::crs_matrix_type>
    StiffnessAssembler3D::matrix_;

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

TEST_F(StiffnessAssembler3D, GlobalDimensionsMatchLayout) {
  const auto matrix = this->matrix();
  const auto &layout = assembler().layout();

  const auto num_dofs =
      static_cast<Tpetra::global_size_t>(layout.num_global_dofs());
  EXPECT_EQ(num_dofs,
            static_cast<Tpetra::global_size_t>(ncomp) * layout.nglob());
  EXPECT_EQ(matrix->getGlobalNumRows(), num_dofs);
  EXPECT_EQ(matrix->getGlobalNumCols(), num_dofs);
  EXPECT_TRUE(matrix->isFillComplete());
  EXPECT_GT(matrix->getGlobalNumEntries(), 0u);
  EXPECT_GT(max_abs_entry(matrix), static_cast<type_real>(0));
}

TEST_F(StiffnessAssembler3D, SymmetricBilinearForm) {
  const auto matrix = this->matrix();
  const auto map = assembler().layout().owned_map();

  // K is symmetric, so x' K z == z' K x for any x, z. The bilinear form
  // probes symmetry of the whole assembled matrix at matrix-vector cost.
  std::mt19937 generator(2026);
  std::uniform_real_distribution<type_real> distribution(-1, 1);

  VectorType x(map), z(map), k_x(map), k_z(map);
  {
    auto x_view = x.getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto z_view = z.getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (std::size_t i = 0; i < x_view.extent(0); ++i) {
      x_view(i, 0) = distribution(generator);
      z_view(i, 0) = distribution(generator);
    }
  }

  matrix->apply(x, k_x);
  matrix->apply(z, k_z);

  const type_real x_k_z = static_cast<type_real>(x.dot(k_z));
  const type_real z_k_x = static_cast<type_real>(z.dot(k_x));

  const type_real scale = std::max(std::max(std::abs(x_k_z), std::abs(z_k_x)),
                                   max_abs_entry(matrix));
  const type_real tol = (single_precision ? static_cast<type_real>(1e-3)
                                          : static_cast<type_real>(1e-10)) *
                        scale;
  EXPECT_NEAR(x_k_z, z_k_x, tol);
}

TEST_F(StiffnessAssembler3D, RigidBodyTranslationNullspace) {
  const auto matrix = this->matrix();
  const auto &layout = assembler().layout();
  const auto map = layout.owned_map();

  const type_real scale = max_abs_entry(matrix);
  ASSERT_GT(scale, static_cast<type_real>(0));

  // With natural boundary conditions only, a rigid translation of one
  // component produces zero strain, so K annihilates it. Rows sum over up to
  // ~8 * 375 float entries, hence the loose relative tolerance.
  const type_real tol = (single_precision ? static_cast<type_real>(5e-3)
                                          : static_cast<type_real>(1e-10)) *
                        scale;

  HostFieldView translation_field("stiffness_assembler_test::translation",
                                  layout.nglob(), ncomp);
  for (int icomp = 0; icomp < ncomp; ++icomp) {
    Kokkos::deep_copy(translation_field, 0);
    for (int iglob = 0; iglob < layout.nglob(); ++iglob) {
      translation_field(iglob, icomp) = 1;
    }

    VectorType translation(map), response(map);
    layout.scatter(translation_field, translation);

    matrix->apply(translation, response);
    EXPECT_LE(response.normInf(), tol)
        << "translation nullspace violated for component " << icomp;
  }
}

TEST_F(StiffnessAssembler3D, MatchesMatrixFreeOperatorGlobally) {
  const auto matrix = this->matrix();
  const auto &layout = assembler().layout();

  auto &field = assembly().fields.template get_simulation_field<forward_tag>();
  const auto &field_impl = field.template get_field<elastic_tag>();
  const auto h_u = field_impl.get_host_field();
  const auto h_v = field_impl.get_host_field_dot();
  const auto h_a = field_impl.get_host_field_dot_dot();
  const int nglob = field_impl.nglob;
  ASSERT_EQ(nglob, layout.nglob());

  // Random displacement over the whole mesh; velocity and acceleration zero.
  std::mt19937 generator(54321);
  std::uniform_real_distribution<type_real> distribution(-1, 1);
  for (int iglob = 0; iglob < nglob; ++iglob) {
    for (int icomp = 0; icomp < ncomp; ++icomp) {
      h_u(iglob, icomp) = distribution(generator);
      h_v(iglob, icomp) = 0;
      h_a(iglob, icomp) = 0;
    }
  }

  // Production matrix-free operator (no mass division): accel = -K u.
  assembly().fields.copy_to_device();
  using BaseTags = specfem::tags::Tags<dim3_tag, forward_tag, elastic_tag>;
  specfem::compute::impl::compute_stiffness_interaction<
      NGLL, specfem::tags::expand<BaseTags, specfem::element::mpi_tag::outer>>(
      assembly(), 0);
  specfem::compute::impl::compute_stiffness_interaction<
      NGLL, specfem::tags::expand<BaseTags, specfem::element::mpi_tag::inner>>(
      assembly(), 0);
  assembly().fields.copy_to_host();

  // Assembled operator applied to the same displacement.
  VectorType u(layout.owned_map()), k_u(layout.owned_map());
  layout.scatter(h_u, u);
  matrix->apply(u, k_u);

  type_real scale = 0;
  type_real max_diff = 0;
  {
    HostFieldView k_u_field("stiffness_assembler_test::k_u", nglob, ncomp);
    layout.gather(k_u, k_u_field);
    for (int iglob = 0; iglob < nglob; ++iglob) {
      for (int icomp = 0; icomp < ncomp; ++icomp) {
        const type_real expected = -h_a(iglob, icomp);
        const type_real actual = k_u_field(iglob, icomp);
        scale = std::max(scale, std::abs(expected));
        max_diff = std::max(max_diff, std::abs(expected - actual));
      }
    }
  }
  ASSERT_GT(scale, static_cast<type_real>(0));

  const type_real rel_tol = single_precision ? static_cast<type_real>(2e-3)
                                             : static_cast<type_real>(1e-10);
  EXPECT_LE(max_diff, rel_tol * scale)
      << "assembled K u disagrees with the matrix-free operator";
}

TEST(StiffnessAssemblerScope3D, RejectsMultiMediumMeshes) {
  const auto mixed_assembly = build_assembly_3d("AcousticElasticForce");
  EXPECT_THROW(AssemblerType assembler(*mixed_assembly), std::runtime_error);
}

TEST(StiffnessAssemblerScope3D, RejectsStaceyBoundaries) {
  const auto stacey_assembly = build_assembly_3d("HomogeneousHalfSpaceStacey");
  EXPECT_THROW(AssemblerType assembler(*stacey_assembly), std::runtime_error);
}

} // namespace stiffness_assembler_test

#else // !SPECFEM_ENABLE_TRILINOS

TEST(StiffnessAssembler3D, SkippedWithoutTrilinos) {
  GTEST_SKIP() << "SPECFEM++ was built without Trilinos "
                  "(SPECFEM_ENABLE_TRILINOS=OFF); the stiffness assembler is "
                  "unavailable.";
}

#endif // SPECFEM_ENABLE_TRILINOS

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
