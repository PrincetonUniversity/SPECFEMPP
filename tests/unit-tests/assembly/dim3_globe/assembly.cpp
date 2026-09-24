#include "test_fixture.hpp"

#include "specfem/element.hpp"
#include <cmath>
#include <numeric>
#include <string>
#include <yaml-cpp/yaml.h>

TEST_F(GlobeAssembly3DTest, ConstructsCompleteAssembly) {
  auto &globe_assembly = assembly->assembly;

  EXPECT_TRUE(globe_assembly.element_types.has_element_context());
  ASSERT_TRUE(globe_assembly.planet_constants.has_value());
  EXPECT_EQ(globe_assembly.planet_constants->planet(),
            specfem::globe::Planet::earth);

  const int nelements = globe_assembly.get_total_number_of_elements();
  const int degrees_of_freedom = globe_assembly.get_total_degrees_of_freedom();
  EXPECT_GT(nelements, 0);
  const auto &grid = globe_assembly.mesh.element_grid;
  const int interior_points_per_element =
      (grid.ngllx - 2) * (grid.nglly - 2) * (grid.ngllz - 2);
  EXPECT_GE(degrees_of_freedom, nelements * interior_points_per_element);

  ASSERT_EQ(globe_assembly.info.elements_per_region.size(), 3);
  const int region_elements = std::accumulate(
      globe_assembly.info.elements_per_region.begin(),
      globe_assembly.info.elements_per_region.end(), 0,
      [](const int total, const auto &entry) { return total + entry.second; });
  EXPECT_EQ(region_elements, nelements);

  const auto &info = globe_assembly.info;
  ASSERT_EQ(info.regions.size(), 3);
  for (const auto &[region_tag, region] : info.regions) {
    EXPECT_GT(region.element_count, 0)
        << specfem::element::to_string(region_tag);
    EXPECT_LT(region.radius.min, region.radius.max);
    EXPECT_GT(region.gll_distance.min, 0.0);
    EXPECT_LE(region.gll_distance.min, region.gll_distance.max);
    EXPECT_GT(region.v.min, 0.0);
    EXPECT_LE(region.v.min, region.v.max);
    EXPECT_GT(region.suggested_time_step, 0.0);
  }

  const auto &radii = globe_assembly.planet_constants->radii();
  EXPECT_NEAR(
      info.regions.at(specfem::element::region_tag::crust_mantle).radius.min,
      radii.r_cmb, 1.0);
  EXPECT_NEAR(
      info.regions.at(specfem::element::region_tag::outer_core).radius.min,
      radii.r_icb, 1.0);
  EXPECT_NEAR(
      info.regions.at(specfem::element::region_tag::outer_core).radius.max,
      radii.r_cmb, 1.0);
  EXPECT_NEAR(
      info.regions.at(specfem::element::region_tag::inner_core).radius.max,
      radii.r_icb, 1.0);

  EXPECT_GE(info.minimum_gll_distance_location.element, 0);
  EXPECT_GE(info.cfl_limit_location.element, 0);
  EXPECT_LT(info.minimum_gll_distance_location.radius.min,
            info.minimum_gll_distance_location.radius.max);
  EXPECT_LT(info.cfl_limit_location.radius.min,
            info.cfl_limit_location.radius.max);

  const YAML::Node provenance = YAML::LoadFile(
      "data/dim3_globe/GlobalSmallMesh/provenance/mesh_info.yaml");
  const double mesher_volume = provenance["volume"].as<double>();
  const double planet_radius =
      globe_assembly.planet_constants->values().r_planet;
  const double expected_volume = mesher_volume * std::pow(planet_radius, 3.0);
  EXPECT_NEAR(info.total_volume, expected_volume, 1.0e-3 * expected_volume);

  // Scaling this one elliptic chunk by six is approximate, but catches unit
  // errors cheaply while the six-chunk MPI fixture provides the precise check.
  constexpr double earth_mass = 5.972e24;
  EXPECT_NEAR(6.0 * info.total_mass, earth_mass, 1.0e-2 * earth_mass);

  const std::string summary = globe_assembly.print();
  EXPECT_NE(summary.find("Elements per region:"), std::string::npos);
  EXPECT_NE(summary.find("Minimum GLL Distance at:"), std::string::npos);
  EXPECT_NE(summary.find("Total Volume:"), std::string::npos);
  EXPECT_NE(summary.find("Total Mass:"), std::string::npos);
  EXPECT_NE(summary.find("All elements accounted for."), std::string::npos);
}
