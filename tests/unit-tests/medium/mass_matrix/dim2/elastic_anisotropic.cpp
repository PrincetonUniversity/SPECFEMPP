#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

TEST(MassMatrix, ElasticPSVAnIsotropicTrivialSolution2D) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::anisotropic;

  using Tags =
      specfem::tags::Tags<dimension, specfem::element::medium_tag::elastic_psv,
                          property_tag, specfem::element::attenuation_tag::none,
                          false>;
  using PointPSVPropertiesType = specfem::point::properties<Tags>;
  using PointPSVMassMatrixType = specfem::point::mass_inverse<Tags>;

  const PointPSVPropertiesType properties(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0);

  const PointPSVMassMatrixType mass_matrix =
      specfem::medium_physics::mass_matrix_component<Tags>(properties);

  const PointPSVMassMatrixType expected_mass_matrix(0.0, 0.0);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}

TEST(MassMatrix, ElasticSHAnIsotropicTrivialSolution2D) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::anisotropic;

  using Tags =
      specfem::tags::Tags<dimension, specfem::element::medium_tag::elastic_sh,
                          property_tag, specfem::element::attenuation_tag::none,
                          false>;

  using PointSHPropertiesType = specfem::point::properties<Tags>;
  using PointSHMassMatrixType = specfem::point::mass_inverse<Tags>;

  const PointSHPropertiesType properties(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         0.0, 0.0);

  const PointSHMassMatrixType mass_matrix =
      specfem::medium_physics::mass_matrix_component<Tags>(properties);

  const PointSHMassMatrixType expected_mass_matrix(0.0);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}

TEST(MassMatrix, ElasticPSVAnIsotropic2D) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::anisotropic;

  using Tags =
      specfem::tags::Tags<dimension, specfem::element::medium_tag::elastic_psv,
                          property_tag, specfem::element::attenuation_tag::none,
                          false>;

  using PointPSVPropertiesType = specfem::point::properties<Tags>;
  using PointPSVMassMatrixType = specfem::point::mass_inverse<Tags>;

  const type_real rho = 10.0;

  const PointPSVPropertiesType properties(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, rho);

  const PointPSVMassMatrixType mass_matrix =
      specfem::medium_physics::mass_matrix_component<Tags>(properties);

  const PointPSVMassMatrixType expected_mass_matrix(rho, rho);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}

TEST(MassMatrix, ElasticSHAnIsotropic2D) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::anisotropic;

  using Tags =
      specfem::tags::Tags<dimension, specfem::element::medium_tag::elastic_sh,
                          property_tag, specfem::element::attenuation_tag::none,
                          false>;
  using PointSHPropertiesType = specfem::point::properties<Tags>;
  using PointSHMassMatrixType = specfem::point::mass_inverse<Tags>;

  const type_real rho = 10.0;

  const PointSHPropertiesType properties(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         0.0, rho);

  const PointSHMassMatrixType mass_matrix =
      specfem::medium_physics::mass_matrix_component<Tags>(properties);

  const PointSHMassMatrixType expected_mass_matrix(rho);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}
