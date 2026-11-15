#include "test_fixture.hpp"

#include "specfem/element.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <string>

TEST_F(GlobeAssembly3DTest, ConstructsCompleteAssembly) {
  auto &globe_assembly = assembly->assembly;

  EXPECT_TRUE(globe_assembly.element_types.has_element_context());
  ASSERT_TRUE(assembly->mesh.globe.planet_constants.has_value());
  EXPECT_EQ(assembly->mesh.globe.planet_constants->planet(),
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

  const std::string summary = globe_assembly.print();
  EXPECT_NE(summary.find("Elements per region:"), std::string::npos);
  EXPECT_NE(summary.find("All elements accounted for."), std::string::npos);
}

TEST_F(GlobeAssembly3DTest, ReportsGlobeStatistics) {
  const auto &assembled = assembly->assembly;
  const auto &info = assembled.info;
  ASSERT_EQ(info.regions.size(), 3);
  type_real minimum_spacing = std::numeric_limits<type_real>::max();
  type_real minimum_dt = minimum_spacing;
  for (const auto &[region, statistics] : info.regions) {
    EXPECT_EQ(statistics.element_count, info.elements_per_region.at(region));
    EXPECT_GT(statistics.element_count, 0);
    EXPECT_GE(statistics.radius.min, 0);
    EXPECT_GT(statistics.radius.max, statistics.radius.min);
    EXPECT_GT(statistics.minimum_gll_spacing, 0);
    EXPECT_GT(statistics.wave_speed.min, 0);
    EXPECT_GE(statistics.wave_speed.max, statistics.wave_speed.min);
    EXPECT_GT(statistics.suggested_time_step, 0);
    minimum_spacing = std::min(minimum_spacing, statistics.minimum_gll_spacing);
    minimum_dt = std::min(minimum_dt, statistics.suggested_time_step);
  }
  EXPECT_EQ(minimum_spacing, info.gll_distance.min);
  EXPECT_EQ(minimum_dt, info.suggested_time_step);
  EXPECT_GT(info.total_volume, 0);
  EXPECT_GT(info.total_mass, 0);
  for (const auto &location :
       { info.minimum_spacing_location, info.cfl_location }) {
    EXPECT_EQ(location.rank, 0);
    ASSERT_GE(location.compute_element, 0);
    ASSERT_LT(location.compute_element, assembled.mesh.nspec);
    EXPECT_EQ(location.mesh_element,
              assembled.mesh.h_compute_to_mesh(location.compute_element));
    EXPECT_EQ(location.region,
              assembled.element_types.get_region_tag(location.compute_element));
    EXPECT_GE(location.radius, info.regions.at(location.region).radius.min);
    EXPECT_LE(location.radius, info.regions.at(location.region).radius.max);
  }
  const auto summary = info.string();
  EXPECT_NE(summary.find("Globe region statistics (SI):"), std::string::npos);
  EXPECT_NE(summary.find("CFL limiting location:"), std::string::npos);
  EXPECT_NE(summary.find("Total mesh volume:"), std::string::npos);
  EXPECT_NE(summary.find("Total mesh mass:"), std::string::npos);

  // The original three-argument API also computes the globe integrals.
  const specfem::assembly::Info<specfem::element::dimension_tag::dim3>
      standalone(assembled.mesh, assembled.properties, assembled.element_types);
  EXPECT_EQ(standalone.string(), summary);
}
