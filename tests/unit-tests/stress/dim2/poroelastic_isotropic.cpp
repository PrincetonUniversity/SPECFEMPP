#include "enumerations/interface.hpp"
#include "medium/compute_stress.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

TEST(Stress, PoroelasticIsotropic2D_Basic) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;
  static constexpr auto PoroTag = specfem::element::medium_tag::poroelastic;

  using PoroPropertiesType =
      specfem::point::properties<dimension, PoroTag, property_tag, false>;
  using PoroFieldDerivativesType =
      specfem::point::field_derivatives<dimension, PoroTag, false>;
  using PoroStressType = specfem::point::stress<dimension, PoroTag, false>;
  // Set up properties (arbitrary but nonzero values)
  const type_real phi = 0.25;
  const type_real rho_s = 2.5;
  const type_real rho_f = 1.2;
  const type_real tortuosity = 2.0;
  const type_real mu_G = 3.0;
  const type_real H_Biot = 8.0;
  const type_real C_Biot = 1.5;
  const type_real M_Biot = 2.5;
  const type_real permxx = 0.1;
  const type_real permxz = 0.02;
  const type_real permzz = 0.12;
  const type_real eta_f = 0.001;
  const PoroPropertiesType properties(phi, rho_s, rho_f, tortuosity, mu_G,
                                      H_Biot, C_Biot, M_Biot, permxx, permxz,
                                      permzz, eta_f);

  PoroFieldDerivativesType field_derivatives;
  // du(0,0): dux/dx, du(1,1): duz/dz, du(0,1): dux/dz, du(1,0): duz/dx
  // du(2,0): dwx/dx, du(3,1): dwz/dz
  field_derivatives.du(0, 0) = 1.0;
  field_derivatives.du(1, 1) = 2.0;
  field_derivatives.du(0, 1) = 3.0;
  field_derivatives.du(1, 0) = 4.0;
  field_derivatives.du(2, 0) = 5.0;
  field_derivatives.du(3, 1) = 6.0;

  // Compute expected values using property accessors
  const type_real lambda_G = properties.lambda_G();
  const type_real lambdaplus2mu_G = properties.lambdaplus2mu_G();
  const type_real rho_bar = properties.rho_bar();

  const type_real sigma_xx =
      lambdaplus2mu_G * 1.0 + lambda_G * 2.0 + C_Biot * (5.0 + 6.0);
  const type_real sigma_zz =
      lambdaplus2mu_G * 2.0 + lambda_G * 1.0 + C_Biot * (5.0 + 6.0);
  const type_real sigma_xz = mu_G * (3.0 + 4.0);
  const type_real sigmap = C_Biot * (1.0 + 2.0) + M_Biot * (5.0 + 6.0);

  PoroStressType expected_stress;
  expected_stress.T(0, 0) = sigma_xx - phi / tortuosity * sigmap;
  expected_stress.T(1, 0) = sigma_xz;
  expected_stress.T(0, 1) = sigma_xz;
  expected_stress.T(1, 1) = sigma_zz - phi / tortuosity * sigmap;
  expected_stress.T(2, 0) = sigmap - rho_f / rho_bar * sigma_xx;
  expected_stress.T(3, 0) = -rho_f / rho_bar * sigma_xz;
  expected_stress.T(2, 1) = -rho_f / rho_bar * sigma_xz;
  expected_stress.T(3, 1) = sigmap - rho_f / rho_bar * sigma_zz;

  const PoroStressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  std::ostringstream message;
  message << "Poroelastic stress tensor is not equal to expected value: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

TEST(Stress, PoroelasticIsotropic2D_ZeroDerivatives) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;
  static constexpr auto PoroTag = specfem::element::medium_tag::poroelastic;

  using PoroPropertiesType =
      specfem::point::properties<dimension, PoroTag, property_tag, false>;
  using PoroFieldDerivativesType =
      specfem::point::field_derivatives<dimension, PoroTag, false>;
  using PoroStressType = specfem::point::stress<dimension, PoroTag, false>;
  const type_real phi = 0.25;
  const type_real rho_s = 2.5;
  const type_real rho_f = 1.2;
  const type_real tortuosity = 2.0;
  const type_real mu_G = 3.0;
  const type_real H_Biot = 8.0;
  const type_real C_Biot = 1.5;
  const type_real M_Biot = 2.5;
  const type_real permxx = 0.1;
  const type_real permxz = 0.02;
  const type_real permzz = 0.12;
  const type_real eta_f = 0.001;
  const PoroPropertiesType properties(phi, rho_s, rho_f, tortuosity, mu_G,
                                      H_Biot, C_Biot, M_Biot, permxx, permxz,
                                      permzz, eta_f);

  PoroFieldDerivativesType field_derivatives;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      field_derivatives.du(i, j) = 0.0;
    }
  }

  const PoroStressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  PoroStressType expected_stress;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      expected_stress.T(i, j) = 0.0;
    }
  }

  std::ostringstream message;
  message << "Poroelastic stress tensor should be zero for zero derivatives: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

} // namespace
