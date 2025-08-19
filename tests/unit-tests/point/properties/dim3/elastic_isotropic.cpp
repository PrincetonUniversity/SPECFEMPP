#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include "utilities/interface.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

const type_real tol = 1e-6; ///< Tolerance for floating point comparisons

// ============================================================================
// 3D Elastic Tests
// ============================================================================
TYPED_TEST(PointPropertiesTest, ElasticIsotropic3D) {
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
      rho[i] = 2700.0 + i * 50.0;              // kg/m³
      const type_real vp = 6000.0 + i * 100.0; // m/s
      const type_real vs = 3500.0 + i * 50.0;  // m/s

      mu[i] = rho[i] * vs * vs;
      kappa[i] = rho[i] * (vp * vp - (4.0 / 3.0) * vs * vs);
      lambdaplus2mu_val[i] = kappa[i] + (4.0 / 3.0) * mu[i];
      lambda_val[i] = lambdaplus2mu_val[i] - 2.0 * mu[i];
      rho_vp_val[i] = rho[i] * vp;
      rho_vs_val[i] = rho[i] * vs;
    }
  } else {
    // Granite-like material for scalar test
    constexpr type_real rho_val = 2700.0;           // kg/m³
    constexpr type_real vp = 6000.0;                // m/s
    constexpr type_real vs = 3500.0;                // m/s
    constexpr type_real mu_val = rho_val * vs * vs; // shear modulus

    // For 3D, we use kappa (bulk modulus) instead of lambda + 2*mu
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
  using PointPropertiesType = specfem::point::properties<
      specfem::dimension::type::dim3, specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic, using_simd>;
  PointPropertiesType props(kappa, mu, rho);

  EXPECT_TRUE(specfem::utilities::is_close(props.kappa(), kappa))
      << ExpectedGot(kappa, props.kappa());
  EXPECT_TRUE(specfem::utilities::is_close(props.mu(), mu))
      << ExpectedGot(mu, props.mu());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho(), rho))
      << ExpectedGot(rho, props.rho());

  EXPECT_TRUE(
      specfem::utilities::is_close(props.lambdaplus2mu(), lambdaplus2mu_val))
      << ExpectedGot(lambdaplus2mu_val, props.lambdaplus2mu());
  EXPECT_TRUE(specfem::utilities::is_close(props.lambda(), lambda_val))
      << ExpectedGot(lambda_val, props.lambda());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho_vp(), rho_vp_val))
      << ExpectedGot(rho_vp_val, props.rho_vp());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho_vs(), rho_vs_val))
      << ExpectedGot(rho_vs_val, props.rho_vs());

  // Additional constructors and assignment tests
  PointPropertiesType props2;
  props2.kappa() = kappa;
  props2.mu() = mu;
  props2.rho() = rho;

  simd_type data[] = { kappa, mu, rho };
  PointPropertiesType props3(data);

  PointPropertiesType props4(kappa);

  EXPECT_TRUE(props2 == props) << ExpectedGot(props2.kappa(), props.kappa())
                               << ExpectedGot(props2.mu(), props.mu())
                               << ExpectedGot(props2.rho(), props.rho());

  EXPECT_TRUE(props2 == props3) << ExpectedGot(props3.kappa(), props2.kappa())
                                << ExpectedGot(props3.mu(), props2.mu())
                                << ExpectedGot(props3.rho(), props2.rho());

  EXPECT_TRUE(specfem::utilities::is_close(props4.kappa(), kappa));
  EXPECT_TRUE(specfem::utilities::is_close(props4.mu(), kappa));
  EXPECT_TRUE(specfem::utilities::is_close(props4.rho(), kappa));
}
