#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>

TEST(ConstitutiveTensor, ElasticIsotropic3DMatchesComputeStress) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using Tags =
      specfem::tags::Tags<dimension, specfem::element::medium_tag::elastic,
                          property_tag, false>;
  using PointPropertiesType = specfem::point::properties<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using StressType = specfem::point::stress<Tags>;

  const type_real kappa = 2.7;
  const type_real mu = 1.3;
  const type_real rho = 2.0;

  const PointPropertiesType properties(kappa, mu, rho);

  FieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 0.3;
  field_derivatives.du(0, 1) = -1.1;
  field_derivatives.du(0, 2) = 0.7;
  field_derivatives.du(1, 0) = -0.4;
  field_derivatives.du(1, 1) = 1.5;
  field_derivatives.du(1, 2) = -0.9;
  field_derivatives.du(2, 0) = 0.6;
  field_derivatives.du(2, 1) = -1.3;
  field_derivatives.du(2, 2) = 0.2;

  const auto &du = field_derivatives.du;

  const StressType stress = specfem::medium_physics::compute_stress<Tags>(
      properties, field_derivatives);

  type_real max_abs_stress = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int k = 0; k < 3; ++k) {
      const type_real abs_stress =
          (stress.T(a, k) < 0.0) ? -stress.T(a, k) : stress.T(a, k);
      if (abs_stress > max_abs_stress) {
        max_abs_stress = abs_stress;
      }
    }
  }

  const type_real tol = (sizeof(type_real) == sizeof(float))
                            ? static_cast<type_real>(1e-5) * max_abs_stress
                            : static_cast<type_real>(1e-12) * max_abs_stress;

  for (int a = 0; a < 3; ++a) {
    for (int k = 0; k < 3; ++k) {
      type_real sigma_ak = 0.0;
      for (int b = 0; b < 3; ++b) {
        for (int l = 0; l < 3; ++l) {
          sigma_ak += specfem::medium_physics::constitutive_tensor<Tags>(
                          properties, a, k, b, l) *
                      du(b, l);
        }
      }
      EXPECT_NEAR(sigma_ak, stress.T(a, k), tol)
          << "Mismatch between constitutive_tensor contraction and "
             "compute_stress at (a, k) = ("
          << a << ", " << k << ")";
    }
  }
}

TEST(ConstitutiveTensor, ElasticIsotropic3DSymmetries) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using Tags =
      specfem::tags::Tags<dimension, specfem::element::medium_tag::elastic,
                          property_tag, false>;
  using PointPropertiesType = specfem::point::properties<Tags>;

  const type_real kappa = 2.7;
  const type_real mu = 1.3;
  const type_real rho = 2.0;

  const PointPropertiesType properties(kappa, mu, rho);

  for (int a = 0; a < 3; ++a) {
    for (int k = 0; k < 3; ++k) {
      for (int b = 0; b < 3; ++b) {
        for (int l = 0; l < 3; ++l) {
          const type_real C_akbl =
              specfem::medium_physics::constitutive_tensor<Tags>(properties, a,
                                                                 k, b, l);
          const type_real C_kabl =
              specfem::medium_physics::constitutive_tensor<Tags>(properties, k,
                                                                 a, b, l);
          const type_real C_aklb =
              specfem::medium_physics::constitutive_tensor<Tags>(properties, a,
                                                                 k, l, b);
          const type_real C_blak =
              specfem::medium_physics::constitutive_tensor<Tags>(properties, b,
                                                                 l, a, k);

          EXPECT_EQ(C_akbl, C_kabl) << "C_akbl != C_kabl at (" << a << ", " << k
                                    << ", " << b << ", " << l << ")";
          EXPECT_EQ(C_akbl, C_aklb) << "C_akbl != C_aklb at (" << a << ", " << k
                                    << ", " << b << ", " << l << ")";
          EXPECT_EQ(C_akbl, C_blak) << "C_akbl != C_blak at (" << a << ", " << k
                                    << ", " << b << ", " << l << ")";
        }
      }
    }
  }
}

TEST(ConstitutiveTensor, ElasticIsotropic3DLameEntries) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using Tags =
      specfem::tags::Tags<dimension, specfem::element::medium_tag::elastic,
                          property_tag, false>;
  using PointPropertiesType = specfem::point::properties<Tags>;

  const type_real kappa = 2.7;
  const type_real mu = 1.3;
  const type_real rho = 2.0;

  const PointPropertiesType properties(kappa, mu, rho);
  const type_real lambda = properties.lambda();

  const type_real C_0000 = specfem::medium_physics::constitutive_tensor<Tags>(
      properties, 0, 0, 0, 0);
  const type_real C_0011 = specfem::medium_physics::constitutive_tensor<Tags>(
      properties, 0, 0, 1, 1);
  const type_real C_0101 = specfem::medium_physics::constitutive_tensor<Tags>(
      properties, 0, 1, 0, 1);
  const type_real C_0110 = specfem::medium_physics::constitutive_tensor<Tags>(
      properties, 0, 1, 1, 0);
  const type_real C_0001 = specfem::medium_physics::constitutive_tensor<Tags>(
      properties, 0, 0, 0, 1);

  EXPECT_NEAR(C_0000, lambda + 2.0 * mu,
              static_cast<type_real>(1e-6) * (lambda + 2.0 * mu));
  EXPECT_NEAR(C_0011, lambda, static_cast<type_real>(1e-6) * lambda);
  EXPECT_NEAR(C_0101, mu, static_cast<type_real>(1e-6) * mu);
  EXPECT_NEAR(C_0110, mu, static_cast<type_real>(1e-6) * mu);
  EXPECT_NEAR(C_0001, 0.0, static_cast<type_real>(1e-6));
}
