#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>

TEST(MassMatrix, ElasticAnisotropic3D) {
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::anisotropic, false>;
  using Properties = specfem::point::properties<Tags>;
  using Mass = specfem::point::mass_inverse<Tags>;

  const type_real rho = 10.0;
  const Properties properties(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, rho);

  const Mass mass =
      specfem::medium_physics::mass_matrix_component<Tags>(properties);
  EXPECT_TRUE(mass == Mass(rho, rho, rho));
}

TEST(Material, ElasticAnisotropic3DPropertyOrder) {
  using Material = specfem::medium_container::material<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::anisotropic,
      specfem::element::attenuation_tag::none>;

  const Material material(22.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                          10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                          19.0, 20.0, 21.0);
  const auto properties = material.get_properties();

  for (int i = 0; i < 21; ++i) {
    EXPECT_EQ(properties[i], static_cast<type_real>(i + 1));
  }
  EXPECT_EQ(properties.rho(), static_cast<type_real>(22.0));
}
