#include "../SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/io.hpp"
#include "specfem/linear_system/mass_vector.hpp"
#include "specfem/linear_system/sparse_matrix_view/fe_assembly.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>

namespace mass_vector_test {

constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;
constexpr auto elastic_tag = specfem::element::medium_tag::elastic;
constexpr int ncomp = 3;

using AssemblyType = specfem::assembly::assembly<dim3_tag>;
using MassTags = specfem::tags::Tags<dim3_tag, elastic_tag,
                                     specfem::element::property_tag::isotropic,
                                     specfem::element::attenuation_tag::none>;
using MappingType = specfem::linear_system::FEMapping<dim3_tag, elastic_tag>;
using FEAssemblyType = specfem::linear_system::FEAssembly<MappingType>;

// Homogeneous density of both fixtures (see the meshfem3D Mesh_Par_file
// under each fixture's provenance/ directory).
constexpr type_real fixture_density = 2700;

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

void check_mass_vector(const std::string &fixture) {
  const auto assembly = build_assembly_3d(fixture);
  const FEAssemblyType fe{ MappingType(*assembly) };
  const auto &mapping = fe.mapping();
  const auto mass =
      specfem::linear_system::assemble_mass_vector<MassTags>(*assembly, fe);

  ASSERT_EQ(static_cast<std::size_t>(mass->getGlobalLength()),
            static_cast<std::size_t>(mapping.num_global_dofs()));

  const auto view = mass->getLocalViewHost(Tpetra::Access::ReadOnly);
  const int nglob = mapping.nglob();

  // Lumped mass is strictly positive everywhere and, for an isotropic
  // medium, identical across components (same rho * w * J accumulation; the
  // tolerance only covers accumulation-order rounding on threaded backends).
  double total_mass = 0;
  for (int iglob = 0; iglob < nglob; ++iglob) {
    const type_real reference =
        view(static_cast<std::size_t>(mapping(iglob, 0)), 0);
    ASSERT_GT(reference, 0) << "non-positive lumped mass at point " << iglob;
    total_mass += static_cast<double>(reference);
    for (int icomp = 1; icomp < ncomp; ++icomp) {
      const type_real other =
          view(static_cast<std::size_t>(mapping(iglob, icomp)), 0);
      EXPECT_NEAR(other, reference, 1e-5 * reference)
          << "component " << icomp << " mass differs at point " << iglob;
    }
  }

  // Per component, the lumped masses sum to the domain mass rho * V. The
  // fixtures are homogeneous boxes, so V comes exactly from the coordinate
  // extents; GLL quadrature integrates the constant density exactly.
  const auto &mesh = assembly->mesh;
  const double volume = static_cast<double>(mesh.xmax - mesh.xmin) *
                        static_cast<double>(mesh.ymax - mesh.ymin) *
                        static_cast<double>(mesh.zmax - mesh.zmin);
  const double expected_mass = static_cast<double>(fixture_density) * volume;
  EXPECT_NEAR(total_mass, expected_mass, 1e-3 * expected_mass)
      << "total lumped mass disagrees with rho * V on " << fixture;
}

TEST(MassVector3D, PositiveAndMatchesDomainMassOnNaturalBoundaryMesh) {
  check_mass_vector("HomogeneousHalfspaceSmallNoABCForceSource");
}

// dt = 0 kills the Stacey (dt/2) C 1 lumped term exactly, so the mass on a
// Stacey mesh is the same pure rho-integral as on a natural-boundary mesh;
// the cross-check against (dt/2) C 1 lives in the damping assembler tests.
TEST(MassVector3D, PositiveAndMatchesDomainMassOnStaceyMesh) {
  check_mass_vector("HomogeneousHalfSpaceStacey");
}

} // namespace mass_vector_test

#else // !SPECFEM_ENABLE_TRILINOS

TEST(MassVector3D, SkippedWithoutTrilinos) {
  GTEST_SKIP() << "SPECFEM++ was built without Trilinos "
                  "(SPECFEM_ENABLE_TRILINOS=OFF); the mass vector assembly "
                  "is unavailable.";
}

#endif // SPECFEM_ENABLE_TRILINOS

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
