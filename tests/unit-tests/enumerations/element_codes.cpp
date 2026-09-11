#include "specfem/element.hpp"
#include <gtest/gtest.h>
#include <stdexcept>

TEST(ElementCodes, RegionRoundTrip) {
  using specfem::element::region_tag;
  for (const auto region : { region_tag::crust_mantle, region_tag::outer_core,
                             region_tag::inner_core }) {
    EXPECT_EQ(specfem::element::region_tag_from_code(
                  specfem::element::to_code(region)),
              region);
  }
  EXPECT_EQ(specfem::element::region_tag_from_code(1),
            region_tag::crust_mantle);
  EXPECT_EQ(specfem::element::region_tag_from_code(2), region_tag::outer_core);
  EXPECT_EQ(specfem::element::region_tag_from_code(3), region_tag::inner_core);
  EXPECT_THROW(specfem::element::region_tag_from_code(0), std::runtime_error);
  EXPECT_THROW(specfem::element::region_tag_from_code(4), std::runtime_error);
}

TEST(ElementCodes, MediumCodes) {
  using specfem::element::medium_tag;
  EXPECT_EQ(specfem::element::medium_tag_from_code(1), medium_tag::acoustic);
  EXPECT_EQ(specfem::element::medium_tag_from_code(2), medium_tag::elastic);
  EXPECT_EQ(specfem::element::to_code(medium_tag::acoustic), 1);
  EXPECT_EQ(specfem::element::to_code(medium_tag::elastic), 2);
  EXPECT_THROW(specfem::element::medium_tag_from_code(0), std::runtime_error);
  EXPECT_THROW(specfem::element::medium_tag_from_code(3), std::runtime_error);
  EXPECT_THROW(specfem::element::to_code(medium_tag::poroelastic),
               std::runtime_error);
}

TEST(ElementCodes, PropertyCodes) {
  using specfem::element::property_tag;
  EXPECT_EQ(specfem::element::property_tag_from_code(0),
            property_tag::isotropic);
  EXPECT_EQ(specfem::element::property_tag_from_code(1),
            property_tag::anisotropic);
  EXPECT_EQ(specfem::element::to_code(property_tag::isotropic), 0);
  EXPECT_EQ(specfem::element::to_code(property_tag::anisotropic), 1);
  EXPECT_THROW(specfem::element::property_tag_from_code(2), std::runtime_error);
  EXPECT_THROW(specfem::element::to_code(property_tag::isotropic_cosserat),
               std::runtime_error);
}

TEST(ElementCodes, RegionToString) {
  using specfem::element::region_tag;
  EXPECT_EQ(specfem::element::to_string(region_tag::crust_mantle),
            "crust_mantle");
  EXPECT_EQ(specfem::element::to_string(region_tag::outer_core), "outer_core");
  EXPECT_EQ(specfem::element::to_string(region_tag::inner_core), "inner_core");
}
