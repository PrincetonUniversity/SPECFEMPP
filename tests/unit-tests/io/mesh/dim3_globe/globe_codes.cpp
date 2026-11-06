#include "specfem/io/mesh/impl/fortran/dim3_globe/globe_codes.hpp"
#include "specfem/element.hpp"
#include "specfem/globe/region_codes.hpp"
#include <gtest/gtest.h>
#include <stdexcept>

namespace globe_codes = specfem::io::mesh::impl::fortran::dim3_globe;

TEST(GlobeCodes, RegionRoundTrip) {
  using specfem::element::region_tag;
  EXPECT_EQ(specfem::globe::to_region_tag(1), region_tag::crust_mantle);
  EXPECT_EQ(specfem::globe::to_region_tag(2), region_tag::outer_core);
  EXPECT_EQ(specfem::globe::to_region_tag(3), region_tag::inner_core);
  EXPECT_THROW(specfem::globe::to_region_tag(0), std::runtime_error);
  EXPECT_THROW(specfem::globe::to_region_tag(4), std::runtime_error);

  EXPECT_EQ(specfem::globe::to_region_code(region_tag::crust_mantle), 1);
  EXPECT_EQ(specfem::globe::to_region_code(region_tag::outer_core), 2);
  EXPECT_EQ(specfem::globe::to_region_code(region_tag::inner_core), 3);
  for (int code = 1; code <= 3; ++code) {
    EXPECT_EQ(
        specfem::globe::to_region_code(specfem::globe::to_region_tag(code)),
        code);
  }
  EXPECT_THROW(specfem::globe::to_region_code(static_cast<region_tag>(-1)),
               std::runtime_error);

  EXPECT_EQ(specfem::element::to_string(region_tag::crust_mantle),
            "crust_mantle");
  EXPECT_EQ(specfem::element::to_string(region_tag::outer_core), "outer_core");
  EXPECT_EQ(specfem::element::to_string(region_tag::inner_core), "inner_core");
}

TEST(GlobeCodes, MediumAndPropertyCodes) {
  using specfem::element::medium_tag;
  using specfem::element::property_tag;
  EXPECT_EQ(globe_codes::to_medium_tag(1), medium_tag::acoustic);
  EXPECT_EQ(globe_codes::to_medium_tag(2), medium_tag::elastic);
  EXPECT_THROW(globe_codes::to_medium_tag(0), std::runtime_error);
  EXPECT_THROW(globe_codes::to_medium_tag(3), std::runtime_error);

  EXPECT_EQ(globe_codes::to_property_tag(0), property_tag::isotropic);
  EXPECT_EQ(globe_codes::to_property_tag(1), property_tag::anisotropic);
  EXPECT_THROW(globe_codes::to_property_tag(2), std::runtime_error);
}

TEST(GlobeCodes, EntityCodes) {
  using specfem::mesh_entity::dim3::type;
  EXPECT_EQ(globe_codes::to_face(1), type::bottom);
  EXPECT_EQ(globe_codes::to_face(3), type::top);
  EXPECT_EQ(globe_codes::to_entity(7), type::bottom_left);
  EXPECT_EQ(globe_codes::to_entity(26), type::top_back_right);
  EXPECT_EQ(globe_codes::to_anchor(19), type::bottom_front_left);
  for (int code = 1; code <= 26; ++code) {
    EXPECT_EQ(static_cast<int>(globe_codes::to_entity(code)), code);
  }
  EXPECT_THROW(globe_codes::to_entity(0), std::runtime_error);
  EXPECT_THROW(globe_codes::to_entity(27), std::runtime_error);
  EXPECT_THROW(globe_codes::to_face(0), std::runtime_error);
  EXPECT_THROW(globe_codes::to_face(7), std::runtime_error);
  EXPECT_THROW(globe_codes::to_anchor(18), std::runtime_error);
}
