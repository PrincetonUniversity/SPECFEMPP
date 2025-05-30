#include "medium/compute_mass_matrix.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

constexpr auto dimension = specfem::dimension::type::dim2;
constexpr auto property_tag = specfem::element::property_tag::anisotropic;

using PointPSVPropertiesType = specfem::point::properties<
    dimension, specfem::element::medium_tag::elastic_psv, property_tag, false>;

using PointPSVMassMatrixType =
    specfem::point::field<dimension, specfem::element::medium_tag::elastic_psv,
                          false, false, false, true, false>;

using PointSHPropertiesType = specfem::point::properties<
    dimension, specfem::element::medium_tag::elastic_sh, property_tag, false>;

using PointSHMassMatrixType =
    specfem::point::field<dimension, specfem::element::medium_tag::elastic_sh,
                          false, false, false, true, false>;

TEST(MassMatrix, ElasticPSVAnIsotropicTrivialSolution2D) {

  const PointPSVPropertiesType properties(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0);

  const PointPSVMassMatrixType mass_matrix =
      specfem::medium::mass_matrix_component(properties);

  const PointPSVMassMatrixType expected_mass_matrix(0.0, 0.0);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}

TEST(MassMatrix, ElasticSHAnIsotropicTrivialSolution2D) {

  const PointSHPropertiesType properties(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         0.0, 0.0);

  const PointSHMassMatrixType mass_matrix =
      specfem::medium::mass_matrix_component(properties);

  const PointSHMassMatrixType expected_mass_matrix(0.0);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}

TEST(MassMatrix, ElasticPSVAnIsotropic2D) {

  const type_real rho = 10.0;

  const PointPSVPropertiesType properties(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, rho);

  const PointPSVMassMatrixType mass_matrix =
      specfem::medium::mass_matrix_component(properties);

  const PointPSVMassMatrixType expected_mass_matrix(rho, rho);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}

TEST(MassMatrix, ElasticSHAnIsotropic2D) {

  const type_real rho = 10.0;

  const PointSHPropertiesType properties(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         0.0, rho);

  const PointSHMassMatrixType mass_matrix =
      specfem::medium::mass_matrix_component(properties);

  const PointSHMassMatrixType expected_mass_matrix(rho);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}
