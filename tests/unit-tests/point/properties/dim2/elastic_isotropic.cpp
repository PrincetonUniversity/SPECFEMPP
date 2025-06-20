#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include "utilities/simd.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Elastic Isotropic Tests
// ============================================================================
TYPED_TEST(PointPropertiesTest, ElasticIsotropic2D) {
  constexpr bool using_simd = TypeParam::value; ///< Use SIMD if true

  // Get the SIMD size from the implementation
  using simd_type =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Declare variables for properties
  simd_type kappa;
  simd_type mu;
  simd_type rho;
  simd_type lambdaplus2mu_val;
  simd_type lambda_val;
  simd_type rho_vp_val;
  simd_type rho_vs_val;

  if constexpr (using_simd) {
    // Setup test data for SIMD
    for (int i = 0; i < simd_size; ++i) {
      rho[i] =
          static_cast<type_real>(2700.0) +
          static_cast<type_real>(i) * static_cast<type_real>(50.0); // kg/m³
      const type_real vp =
          static_cast<type_real>(6000.0) +
          static_cast<type_real>(i) * static_cast<type_real>(100.0); // m/s
      const type_real vs =
          static_cast<type_real>(3500.0) +
          static_cast<type_real>(i) * static_cast<type_real>(50.0); // m/s

      mu[i] = static_cast<type_real>(rho[i]) * static_cast<type_real>(vs) *
              static_cast<type_real>(vs);
      kappa[i] = static_cast<type_real>(rho[i]) *
                 (static_cast<type_real>(vp) * static_cast<type_real>(vp) -
                  static_cast<type_real>(4.0 / 3.0) *
                      static_cast<type_real>(vs) * static_cast<type_real>(vs));
      lambdaplus2mu_val[i] =
          static_cast<type_real>(kappa[i]) +
          static_cast<type_real>(4.0 / 3.0) * static_cast<type_real>(mu[i]);
      lambda_val[i] =
          static_cast<type_real>(lambdaplus2mu_val[i]) -
          static_cast<type_real>(2.0) * static_cast<type_real>(mu[i]);
      rho_vp_val[i] =
          static_cast<type_real>(rho[i]) * static_cast<type_real>(vp);
      rho_vs_val[i] =
          static_cast<type_real>(rho[i]) * static_cast<type_real>(vs);
    }
  } else {
    // Granite-like material for scalar test
    constexpr type_real rho_val = 2700.0;           // kg/m³
    constexpr type_real vp = 6000.0;                // m/s
    constexpr type_real vs = 3500.0;                // m/s
    constexpr type_real mu_val = rho_val * vs * vs; // shear modulus

    // For 2D, we use kappa (bulk modulus) instead of lambda + 2*mu
    constexpr type_real kappa_val =
        rho_val * (vp * vp - (4.0 / 3.0) * vs * vs); // bulk modulus
    constexpr type_real lambdaplus2mu_scalar = kappa_val + (4.0 / 3.0) * mu_val;
    constexpr type_real lambda_scalar = lambdaplus2mu_scalar - 2.0 * mu_val;

    // Assign to our variables
    kappa = kappa_val;
    mu = mu_val;
    rho = rho_val;
    lambdaplus2mu_val = lambdaplus2mu_scalar;
    lambda_val = lambda_scalar;
    rho_vp_val = rho_val * vp;
    rho_vs_val = rho_val * vs;
  }

  // Create the properties object
  specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic, using_simd>
      props(kappa, mu, rho);

  EXPECT_TRUE(specfem::utilities::is_close(props.kappa(), kappa))
      << ExpectedGot(kappa, props.kappa());
  EXPECT_TRUE(
      specfem::utilities::is_close(props.lambdaplus2mu(), lambdaplus2mu_val))
      << ExpectedGot(lambdaplus2mu_val, props.lambdaplus2mu());
  EXPECT_TRUE(specfem::utilities::is_close(props.mu(), mu))
      << ExpectedGot(mu, props.mu());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho(), rho))
      << ExpectedGot(rho, props.rho());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho_vp(), rho_vp_val))
      << ExpectedGot(rho_vp_val, props.rho_vp());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho_vs(), rho_vs_val))
      << ExpectedGot(rho_vs_val, props.rho_vs());

  // See note in original code about lambda precision
  EXPECT_TRUE(specfem::utilities::is_close(
      props.lambda(), lambdaplus2mu_val - static_cast<type_real>(2.0) * mu))
      << ExpectedGot(lambda_val, props.lambda());
}
