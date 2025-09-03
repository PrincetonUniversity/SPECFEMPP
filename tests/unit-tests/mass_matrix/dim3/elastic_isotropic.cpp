#include "medium/compute_mass_matrix.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

TEST(MassMatrix, ElasticIsotropicTrivialSolution3D) {
  static constexpr auto dimension = specfem::dimension::type::dim3;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using PointPropertiesType = specfem::point::properties<
      dimension, specfem::element::medium_tag::elastic, property_tag, false>;
  using PointMassMatrixType = specfem::point::mass_inverse<
      dimension, specfem::element::medium_tag::elastic, false>;

  const PointPropertiesType properties(0.0, 0.0, 0.0);

  const PointMassMatrixType mass_matrix =
      specfem::medium::mass_matrix_component(properties);

  const PointMassMatrixType expected_mass_matrix(0.0, 0.0, 0.0);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}

TEST(MassMatrix, ElasticIsotropic3D) {
  static constexpr auto dimension = specfem::dimension::type::dim3;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using PointPropertiesType = specfem::point::properties<
      dimension, specfem::element::medium_tag::elastic, property_tag, false>;
  using PointMassMatrixType = specfem::point::mass_inverse<
      dimension, specfem::element::medium_tag::elastic, false>;

  const type_real rho = 10.0;

  const PointPropertiesType properties(0.0, 0.0, rho);

  const PointMassMatrixType mass_matrix =
      specfem::medium::mass_matrix_component(properties);

  const PointMassMatrixType expected_mass_matrix(rho, rho, rho);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}
