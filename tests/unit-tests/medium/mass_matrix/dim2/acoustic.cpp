#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

TEST(MassMatrix, AcousticIsotropic2D) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::acoustic;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;
  using Tags =
      specfem::tags::Tags<dimension, medium_tag, property_tag,
                          specfem::element::attenuation_tag::none, false>;
  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<dimension, true, false>;
  using PointPropertiesType = specfem::point::properties<Tags>;
  using PointMassMatrixType = specfem::point::mass_inverse<Tags>;

  const type_real kappa = 10.0;

  const PointPropertiesType properties(0.0, kappa);

  const PointMassMatrixType mass_matrix =
      specfem::medium_physics::mass_matrix_component<Tags>(properties);

  const PointMassMatrixType expected_mass_matrix(static_cast<type_real>(1.0) /
                                                 kappa);

  std::ostringstream message;
  message << "Mass matrix is not equal to expected value: " << mass_matrix(0)
          << " != " << expected_mass_matrix(0);
  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}
