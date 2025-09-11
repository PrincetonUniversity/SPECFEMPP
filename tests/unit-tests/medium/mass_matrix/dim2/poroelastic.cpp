#include "medium/compute_mass_matrix.hpp"
#include "specfem/point.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <sstream>

TEST(MassMatrix, PoroelasticIsotropic2DZeroPorosity) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::poroelastic;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<dimension, true, false>;
  using PointPropertiesType =
      specfem::point::properties<dimension, medium_tag, property_tag, false>;
  using PointMassMatrixType =
      specfem::point::mass_inverse<dimension, medium_tag, false>;

  const type_real rho_s = 2.0;
  const type_real phi = 0.0;
  const type_real rho_f = 1.0;
  const type_real tortuosity = 1.0;

  const type_real solid_component = rho_s;
  const PointPropertiesType properties(phi, rho_s, rho_f, tortuosity, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  const PointMassMatrixType mass_matrix =
      specfem::medium::mass_matrix_component(properties);

  EXPECT_NEAR(mass_matrix(0), solid_component, 1e-6 * solid_component)
      << "Mass matrix is not equal to expected value: " << mass_matrix(0)
      << " != " << solid_component;
  EXPECT_NEAR(mass_matrix(1), solid_component, 1e-6 * solid_component)
      << "Mass matrix is not equal to expected value: " << mass_matrix(1)
      << " != " << solid_component;

  EXPECT_TRUE(std::isinf(mass_matrix(2)))
      << "Fluid component is not infinity: " << mass_matrix(2);

  EXPECT_TRUE(std::isinf(mass_matrix(3)))
      << "Fluid component is not infinity: " << mass_matrix(3);
}

TEST(MassMatrix, PoroelasticIsotropic2D) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::poroelastic;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<dimension, true, false>;
  using PointPropertiesType =
      specfem::point::properties<dimension, medium_tag, property_tag, false>;
  using PointMassMatrixType =
      specfem::point::mass_inverse<dimension, medium_tag, false>;

  const type_real rho_s = 2.0;
  const type_real phi = 0.5;
  const type_real rho_f = 1.0;
  const type_real tortuosity = 10.2;

  const auto rho_bar = ((1.0 - phi) * rho_s) + (phi * rho_f);

  const auto solid_component = (rho_bar - phi * rho_f / tortuosity);
  const auto fluid_component =
      (rho_f * tortuosity * rho_bar - phi * rho_f * rho_f) / (phi * rho_bar);

  const PointPropertiesType properties(phi, rho_s, rho_f, tortuosity, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  const PointMassMatrixType mass_matrix =
      specfem::medium::mass_matrix_component(properties);

  const PointMassMatrixType expected_mass_matrix(
      solid_component, solid_component, fluid_component, fluid_component);

  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}
