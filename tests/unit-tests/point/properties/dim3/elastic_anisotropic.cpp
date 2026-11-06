#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem/utilities.hpp"
#include <gtest/gtest.h>

TYPED_TEST(PointPropertiesTest, ElasticAnisotropic3DVoigtSpeeds) {
  constexpr bool using_simd = TypeParam::value;
  using Tags = specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                                   specfem::element::medium_tag::elastic,
                                   specfem::element::property_tag::anisotropic,
                                   using_simd>;
  using Properties = specfem::point::properties<Tags>;
  using value_type = typename Properties::value_type;

  const value_type rho = static_cast<type_real>(4.0);
  const value_type bulk = static_cast<type_real>(12.0);
  const value_type shear = static_cast<type_real>(3.0);
  const value_type lambda = bulk - static_cast<type_real>(2.0 / 3.0) * shear;
  const value_type normal = lambda + static_cast<type_real>(2.0) * shear;
  const value_type zero = static_cast<type_real>(0.0);

  const Properties properties(normal, lambda, lambda, zero, zero, zero, normal,
                              lambda, zero, zero, zero, normal, zero, zero,
                              zero, shear, zero, zero, shear, zero, shear, rho);

  const value_type expected_vp =
      Kokkos::sqrt((bulk + static_cast<type_real>(4.0 / 3.0) * shear) / rho);
  const value_type expected_vs = Kokkos::sqrt(shear / rho);

  EXPECT_EQ(Properties::nprops, 22);
  EXPECT_TRUE(
      specfem::utilities::is_close(properties.voigt_bulk_modulus(), bulk));
  EXPECT_TRUE(
      specfem::utilities::is_close(properties.voigt_shear_modulus(), shear));
  EXPECT_TRUE(specfem::utilities::is_close(properties.vp(), expected_vp));
  EXPECT_TRUE(specfem::utilities::is_close(properties.vs(), expected_vs));
  EXPECT_TRUE(
      specfem::utilities::is_close(properties.rho_vp(), rho * expected_vp));
  EXPECT_TRUE(
      specfem::utilities::is_close(properties.rho_vs(), rho * expected_vs));
  EXPECT_TRUE(specfem::utilities::is_close(properties.vmax(), expected_vp));
  EXPECT_TRUE(specfem::utilities::is_close(properties.vmin(), expected_vs));
}
