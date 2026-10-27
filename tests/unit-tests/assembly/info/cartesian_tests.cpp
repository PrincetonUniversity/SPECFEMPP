#include "../dim3/test_fixture.hpp"
#include "specfem/assembly/info.hpp"
#include <sstream>

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
