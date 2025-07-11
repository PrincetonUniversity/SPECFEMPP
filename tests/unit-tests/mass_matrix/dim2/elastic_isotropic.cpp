#include "medium/compute_mass_matrix.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

TEST(MassMatrix, ElasticPSVIsotropicTrivialSolution2D) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using PointPSVPropertiesType =
      specfem::point::properties<dimension,
                                 specfem::element::medium_tag::elastic_psv,
                                 property_tag, false>;
  using PointPSVMassMatrixType =
      specfem::point::field<dimension,
                            specfem::element::medium_tag::elastic_psv, false,
                            false, false, true, false>;

  const PointPSVPropertiesType properties(0.0, 0.0, 0.0);

  const PointPSVMassMatrixType mass_matrix =
      specfem::medium::mass_matrix_component(properties);

  const PointPSVMassMatrixType expected_mass_matrix(0.0, 0.0);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}

TEST(MassMatrix, ElasticSHIsotropicTrivialSolution2D) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using PointSHPropertiesType = specfem::point::properties<
      dimension, specfem::element::medium_tag::elastic_sh, property_tag, false>;
  using PointSHMassMatrixType =
      specfem::point::field<dimension, specfem::element::medium_tag::elastic_sh,
                            false, false, false, true, false>;

  const PointSHPropertiesType properties(0.0, 0.0, 0.0);

  const PointSHMassMatrixType mass_matrix =
      specfem::medium::mass_matrix_component(properties);

  const PointSHMassMatrixType expected_mass_matrix(0.0);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}

TEST(MassMatrix, ElasticPSVIsotropic2D) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using PointPSVPropertiesType =
      specfem::point::properties<dimension,
                                 specfem::element::medium_tag::elastic_psv,
                                 property_tag, false>;
  using PointPSVMassMatrixType =
      specfem::point::field<dimension,
                            specfem::element::medium_tag::elastic_psv, false,
                            false, false, true, false>;

  const type_real rho = 10.0;

  const PointPSVPropertiesType properties(0.0, 0.0, rho);

  const PointPSVMassMatrixType mass_matrix =
      specfem::medium::mass_matrix_component(properties);

  const PointPSVMassMatrixType expected_mass_matrix(rho, rho);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}

TEST(MassMatrix, ElasticSHIsotropic2D) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using PointSHPropertiesType = specfem::point::properties<
      dimension, specfem::element::medium_tag::elastic_sh, property_tag, false>;
  using PointSHMassMatrixType =
      specfem::point::field<dimension, specfem::element::medium_tag::elastic_sh,
                            false, false, false, true, false>;

  const type_real rho = 10.0;

  const PointSHPropertiesType properties(0.0, 0.0, rho);

  const PointSHMassMatrixType mass_matrix =
      specfem::medium::mass_matrix_component(properties);

  const PointSHMassMatrixType expected_mass_matrix(rho);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}
