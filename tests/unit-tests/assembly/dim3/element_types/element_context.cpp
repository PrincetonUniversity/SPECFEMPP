#include "../test_fixture.hpp"
#include "specfem/element.hpp"
#include <gtest/gtest.h>
#include <stdexcept>

// Cartesian meshes carry no globe context: the views must stay unallocated.
TEST_P(Assembly3DTest, NoGlobeElementContext) {
  const auto &element_types = getAssembly().element_types;
  EXPECT_FALSE(element_types.has_element_context());
  EXPECT_EQ(element_types.regions.extent(0), 0);
  EXPECT_EQ(element_types.idoubling.extent(0), 0);
  EXPECT_EQ(element_types.rmin.extent(0), 0);
  EXPECT_EQ(element_types.rmax.extent(0), 0);
  EXPECT_EQ(element_types.elem_in_crust.extent(0), 0);
  EXPECT_EQ(element_types.elem_in_mantle.extent(0), 0);
  EXPECT_EQ(element_types.get_number_of_elements(
                specfem::element::region_tag::crust_mantle),
            0);
  EXPECT_THROW(element_types.get_region_tag(0), std::runtime_error);
}
