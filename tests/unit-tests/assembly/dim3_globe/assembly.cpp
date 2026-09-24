#include "test_fixture.hpp"

#include "specfem/element.hpp"
#include <numeric>
#include <string>

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

  const std::string summary = globe_assembly.print();
  EXPECT_NE(summary.find("Elements per region:"), std::string::npos);
  EXPECT_NE(summary.find("All elements accounted for."), std::string::npos);
}
