#include "specfem/enums.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

using AnisotropicTags =
    specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::anisotropic, false>;

using IsotropicTags =
    specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic, false>;

/**
 * @brief Fill the nine displacement gradients with distinct values.
 */
template <typename FieldDerivativesType>
FieldDerivativesType make_field_derivatives() {
  FieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 1.0; // du_x/dx
  field_derivatives.du(1, 1) = 2.0; // du_y/dy
  field_derivatives.du(2, 2) = 3.0; // du_z/dz
  field_derivatives.du(0, 1) = 4.0; // du_x/dy
  field_derivatives.du(1, 0) = 5.0; // du_y/dx
  field_derivatives.du(0, 2) = 6.0; // du_x/dz
  field_derivatives.du(2, 0) = 7.0; // du_z/dx
  field_derivatives.du(1, 2) = 8.0; // du_y/dz
  field_derivatives.du(2, 1) = 9.0; // du_z/dy
  return field_derivatives;
}

TEST(Stress, ElasticAnisotropic3D_Basic) {
  using PropertiesType = specfem::point::properties<AnisotropicTags>;
  using FieldDerivativesType =
      specfem::point::field_derivatives<AnisotropicTags>;
  using StressType = specfem::point::stress<AnisotropicTags>;

  // Distinct, deliberately asymmetric stiffnesses so that a transposed or
  // mis-ordered Voigt row would change the result.
  const type_real c11 = 1.0, c12 = 2.0, c13 = 3.0, c14 = 4.0, c15 = 5.0,
                  c16 = 6.0, c22 = 7.0, c23 = 8.0, c24 = 9.0, c25 = 10.0,
                  c26 = 11.0, c33 = 12.0, c34 = 13.0, c35 = 14.0, c36 = 15.0,
                  c44 = 16.0, c45 = 17.0, c46 = 18.0, c55 = 19.0, c56 = 20.0,
                  c66 = 21.0;
  const type_real rho = 22.0; // Density is not used in stress computation

  const PropertiesType properties(c11, c12, c13, c14, c15, c16, c22, c23, c24,
                                  c25, c26, c33, c34, c35, c36, c44, c45, c46,
                                  c55, c56, c66, rho);

  const auto field_derivatives = make_field_derivatives<FieldDerivativesType>();

  const StressType stress =
      specfem::medium_physics::compute_stress<AnisotropicTags>(
          properties, field_derivatives);

  // Voigt strain vector, shear entries as engineering strains.
  const type_real e_xx = 1.0;
  const type_real e_yy = 2.0;
  const type_real e_zz = 3.0;
  const type_real g_yz = 8.0 + 9.0;
  const type_real g_xz = 6.0 + 7.0;
  const type_real g_xy = 4.0 + 5.0;

  StressType expected_stress;
  expected_stress.T(0, 0) = c11 * e_xx + c12 * e_yy + c13 * e_zz + c14 * g_yz +
                            c15 * g_xz + c16 * g_xy;
  expected_stress.T(1, 1) = c12 * e_xx + c22 * e_yy + c23 * e_zz + c24 * g_yz +
                            c25 * g_xz + c26 * g_xy;
  expected_stress.T(2, 2) = c13 * e_xx + c23 * e_yy + c33 * e_zz + c34 * g_yz +
                            c35 * g_xz + c36 * g_xy;
  const type_real sigma_yz = c14 * e_xx + c24 * e_yy + c34 * e_zz + c44 * g_yz +
                             c45 * g_xz + c46 * g_xy;
  const type_real sigma_xz = c15 * e_xx + c25 * e_yy + c35 * e_zz + c45 * g_yz +
                             c55 * g_xz + c56 * g_xy;
  const type_real sigma_xy = c16 * e_xx + c26 * e_yy + c36 * e_zz + c46 * g_yz +
                             c56 * g_xz + c66 * g_xy;
  expected_stress.T(0, 1) = sigma_xy;
  expected_stress.T(1, 0) = sigma_xy;
  expected_stress.T(0, 2) = sigma_xz;
  expected_stress.T(2, 0) = sigma_xz;
  expected_stress.T(1, 2) = sigma_yz;
  expected_stress.T(2, 1) = sigma_yz;

  std::ostringstream message;
  message << "3D anisotropic stress tensor is not equal to expected value: \n"
          << "Computed:\n"
          << stress.print() << "\n"
          << "Expected:\n"
          << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

/**
 * @brief The isotropic-equivalent Voigt matrix must reproduce Hooke's law.
 *
 * This is the degenerate case of the `model_aniso` port in
 * `io/mesh/impl/fortran/dim3/read_materials.cpp` (anisotropy flag 0), and the
 * baseline that flags 1 and 2 perturb. An element tagged anisotropic with
 * these stiffnesses must propagate waves identically to the isotropic
 * material it was built from.
 */
TEST(Stress, ElasticAnisotropic3D_IsotropicEquivalentMatchesIsotropic) {
  using AnisotropicPropertiesType = specfem::point::properties<AnisotropicTags>;
  using IsotropicPropertiesType = specfem::point::properties<IsotropicTags>;

  const type_real rho = 2300.0;
  const type_real vp = 2800.0;
  const type_real vs = 1500.0;

  const type_real lambda_plus_two_mu = rho * vp * vp;
  const type_real mu = rho * vs * vs;
  const type_real lambda = lambda_plus_two_mu - 2.0 * mu;
  const type_real kappa = lambda + (2.0 / 3.0) * mu;
  const type_real zero = 0.0;

  const AnisotropicPropertiesType anisotropic_properties(
      lambda_plus_two_mu, lambda, lambda, zero, zero, zero, lambda_plus_two_mu,
      lambda, zero, zero, zero, lambda_plus_two_mu, zero, zero, zero, mu, zero,
      zero, mu, zero, mu, rho);

  const IsotropicPropertiesType isotropic_properties(kappa, mu, rho);

  const auto anisotropic_stress =
      specfem::medium_physics::compute_stress<AnisotropicTags>(
          anisotropic_properties,
          make_field_derivatives<
              specfem::point::field_derivatives<AnisotropicTags>>());

  const auto isotropic_stress =
      specfem::medium_physics::compute_stress<IsotropicTags>(
          isotropic_properties,
          make_field_derivatives<
              specfem::point::field_derivatives<IsotropicTags>>());

  const type_real tolerance = 1e-3 * lambda_plus_two_mu;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(anisotropic_stress.T(i, j), isotropic_stress.T(i, j),
                  tolerance)
          << "Mismatch at T(" << i << ", " << j << ")\n"
          << "Anisotropic:\n"
          << anisotropic_stress.print() << "\n"
          << "Isotropic:\n"
          << isotropic_stress.print() << "\n";
    }
  }

  // The Voigt averages of the isotropic-equivalent matrix must recover the
  // moduli they were built from; wavefield output relies on kappa().
  EXPECT_NEAR(anisotropic_properties.kappa(), kappa, 1e-3 * kappa);
  EXPECT_NEAR(anisotropic_properties.mu(), mu, 1e-3 * mu);
}

} // namespace
