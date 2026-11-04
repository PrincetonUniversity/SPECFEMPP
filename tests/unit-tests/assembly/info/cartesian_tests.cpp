#include "../dim3/test_fixture.hpp"
#include "specfem/assembly/info.hpp"
#include "specfem/assembly/info/impl/globe.hpp"
#include <sstream>
#include <vector>

TEST_P(Assembly3DTest, CartesianInfoOutputUnchanged) {
  const auto &assembly = getAssembly();
  const specfem::assembly::Info<dimension> info(
      assembly.mesh, assembly.properties, assembly.element_types);
  EXPECT_TRUE(info.regions.empty());
  EXPECT_TRUE(info.elements_per_region.empty());
  EXPECT_EQ(info.minimum_spacing_location.rank, -1);
  EXPECT_EQ(info.total_volume, 0);
  EXPECT_EQ(info.total_mass, 0);
  // Preserve the complete pre-globe format, including whitespace and labels.
  std::ostringstream expected;
  expected << "Mesh Information (3D):\n"
           << " Domain X: ............... [" << info.domain_bounds.x().min
           << ", " << info.domain_bounds.x().max << "]\n"
           << " Domain Y: ............... [" << info.domain_bounds.y().min
           << ", " << info.domain_bounds.y().max << "]\n"
           << " Domain Z: ............... [" << info.domain_bounds.z().min
           << ", " << info.domain_bounds.z().max << "]\n"
           << " VP: ..................... [" << info.vp.min << ", "
           << info.vp.max << "]\n"
           << " VS: ..................... [" << info.vs.min << ", "
           << info.vs.max << "]\n"
           << " V: .......................[" << info.v.min << ", " << info.v.max
           << "]\n"
           << " Rho: .................... [" << info.rho.min << ", "
           << info.rho.max << "]\n"
           << " Element Size: ........... [" << info.element_size.min << ", "
           << info.element_size.max << "]\n"
           << " GLL Distance: ........... [" << info.gll_distance.min << ", "
           << info.gll_distance.max << "]\n"
           << " Minimum Period: . " << info.largest_minimum_period << "\n"
           << " Suggested Time Step: .... " << info.suggested_time_step << "\n";
  EXPECT_EQ(info.string(), expected.str());
}

TEST_P(Assembly3DTest, IntegratesConstantDensityAndHandlesAbsentRegions) {
  auto &assembled = assembly.assembly;
  using Region = specfem::element::region_tag;
  assembled.element_types.regions = decltype(assembled.element_types.regions)(
      "synthetic_region", assembled.mesh.nspec);
  Kokkos::deep_copy(assembled.element_types.regions, Region::crust_mantle);
  const specfem::assembly::Info<dimension> info(
      assembled.mesh, assembled.properties, assembled.element_types,
      &assembled.jacobian_matrix);
  // Both eight-element fixtures are rectangular boxes with constant density.
  const double volume = info.domain_bounds.x().length() *
                        info.domain_bounds.y().length() *
                        info.domain_bounds.z().length();
  EXPECT_NEAR(info.total_volume, volume, 1.e-5 * volume);
  EXPECT_NEAR(info.total_mass, info.rho.min * volume,
              1.e-5 * info.rho.min * volume);
  EXPECT_EQ(info.regions.at(Region::crust_mantle).element_count,
            assembled.mesh.nspec);
  for (const auto region : { Region::outer_core, Region::inner_core }) {
    const auto &statistics = info.regions.at(region);
    EXPECT_EQ(statistics.element_count, 0);
    EXPECT_NEAR(statistics.radius.min, 0, 0);
    EXPECT_NEAR(statistics.radius.max, 0, 0);
    EXPECT_NEAR(statistics.suggested_time_step, 0, 0);
  }
  EXPECT_EQ(info.minimum_spacing_location.rank, 0);
  EXPECT_EQ(info.cfl_location.rank, 0);
}

namespace specfem::test_configuration {

/** @brief Minimal scatter results with deliberately different CFL values. */
struct InfoScatterResults {
  struct BoundsResults {
    std::vector<specfem::assembly::info::impl::Bounds> values;
    auto get_all_bounds() const { return values; }
  };
  BoundsResults element_v;
  BoundsResults element_gll_distance;
  Kokkos::View<type_real *> element_dt;
};

} // namespace specfem::test_configuration

TEST_P(Assembly3DTest, GlobeLocationUsesDeviceTimeSteps) {
  auto &assembled = assembly.assembly;
  using Region = specfem::element::region_tag;
  assembled.element_types.regions = decltype(assembled.element_types.regions)(
      "synthetic_region", assembled.mesh.nspec);
  Kokkos::deep_copy(assembled.element_types.regions, Region::crust_mantle);
  specfem::test_configuration::InfoScatterResults scatters;
  scatters.element_v.values.resize(assembled.mesh.nspec, { 1, 2 });
  scatters.element_gll_distance.values.resize(assembled.mesh.nspec, { 10, 20 });
  scatters.element_dt = decltype(scatters.element_dt)(
      "synthetic_device_time_steps", assembled.mesh.nspec);
  Kokkos::deep_copy(scatters.element_dt, type_real(1));
  auto host_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                     scatters.element_dt);
  host_dt(assembled.mesh.nspec - 1) = type_real(0.125);
  Kokkos::deep_copy(scatters.element_dt, host_dt);
  specfem::assembly::Info<dimension> info;
  info.suggested_time_step = type_real(0.125);
  info.gll_distance.min = 10;
  // Recomputing from the synthetic speed/spacing would give 2.5, and would
  // select element 0. Preserve the device result and its actual winning
  // element.
  specfem::assembly::info_impl::compute_globe_info(
      assembled.mesh, assembled.properties, assembled.element_types,
      assembled.jacobian_matrix, scatters, info);
  EXPECT_EQ(info.cfl_location.compute_element, assembled.mesh.nspec - 1);
  EXPECT_NEAR(info.regions.at(Region::crust_mantle).suggested_time_step, 0.125,
              0);
}
