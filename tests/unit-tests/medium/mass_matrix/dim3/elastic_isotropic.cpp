#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

TEST(MassMatrix, ElasticIsotropicTrivialSolution3D) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using Tags =
      specfem::tags::Tags<dimension, specfem::element::medium_tag::elastic,
                          property_tag, false>;

  using PointPropertiesType = specfem::point::properties<Tags>;
  using PointMassMatrixType = specfem::point::mass_inverse<Tags>;

  const PointPropertiesType properties(0.0, 0.0, 0.0);

  const PointMassMatrixType mass_matrix =
      specfem::medium_physics::mass_matrix_component<Tags>(properties);

  const PointMassMatrixType expected_mass_matrix(0.0, 0.0, 0.0);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}

TEST(MassMatrix, ElasticIsotropic3D) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using Tags =
      specfem::tags::Tags<dimension, specfem::element::medium_tag::elastic,
                          property_tag, false>;
  using PointPropertiesType = specfem::point::properties<Tags>;
  using PointMassMatrixType = specfem::point::mass_inverse<Tags>;

  const type_real rho = 10.0;

  const PointPropertiesType properties(0.0, 0.0, rho);

  const PointMassMatrixType mass_matrix =
      specfem::medium_physics::mass_matrix_component<Tags>(properties);

  const PointMassMatrixType expected_mass_matrix(rho, rho, rho);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}
