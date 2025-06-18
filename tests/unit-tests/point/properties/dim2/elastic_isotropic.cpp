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
  simd_type lambdaplus2mu;
  simd_type mu;
  simd_type rho;
  simd_type lambda;
  simd_type rho_vp;
  simd_type rho_vs;
  simd_type vp;
  simd_type vs;
  simd_type wrong_value;

  if constexpr (using_simd) {
    // Setup test data for SIMD
    for (int i = 0; i < simd_size; ++i) {
      rho[i] =
          static_cast<type_real>(2700.0) +
          static_cast<type_real>(i) * static_cast<type_real>(50.0); // kg/m³
      vp[i] = static_cast<type_real>(6000.0) +
              static_cast<type_real>(i) * static_cast<type_real>(100.0); // m/s
      vs[i] = static_cast<type_real>(3500.0) +
              static_cast<type_real>(i) * static_cast<type_real>(50.0); // m/s

      mu[i] = static_cast<type_real>(rho[i]) * static_cast<type_real>(vs[i]) *
              static_cast<type_real>(vs[i]);
      lambda[i] = static_cast<type_real>(rho[i]) *
                      static_cast<type_real>(vp[i]) *
                      static_cast<type_real>(vp[i]) -
                  static_cast<type_real>(2.0) * static_cast<type_real>(mu[i]);
      lambdaplus2mu[i] =
          static_cast<type_real>(lambda[i]) +
          static_cast<type_real>(2.0) * static_cast<type_real>(mu[i]);
      rho_vp[i] =
          static_cast<type_real>(rho[i]) * static_cast<type_real>(vp[i]);
      rho_vs[i] =
          static_cast<type_real>(rho[i]) * static_cast<type_real>(vs[i]);
    }
  } else {
    // Granite-like material for scalar test
    rho = 2700.0;                    // kg/m³
    vp = 6000.0;                     // m/s
    vs = 3500.0;                     // m/s
    mu = rho * vs * vs;              // shear modulus
    lambda = rho * vp * vp - 2 * mu; // first Lamé parameter
    lambdaplus2mu = lambda + 2 * mu;
    rho_vp = rho * vp; // density times P-wave velocity
    rho_vs = rho * vs; // density times S-wave velocity
  }

  // Create the properties object
  specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic, using_simd>
      props(lambdaplus2mu, mu, rho);

  EXPECT_TRUE(
      specfem::utilities::is_close(props.lambdaplus2mu(), lambdaplus2mu))
      << ExpectedGot(lambdaplus2mu, props.lambdaplus2mu());
  EXPECT_TRUE(specfem::utilities::is_close(props.mu(), mu))
      << ExpectedGot(mu, props.mu());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho(), rho))
      << ExpectedGot(rho, props.rho());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho_vp(), rho_vp))
      << ExpectedGot(rho_vp, props.rho_vp());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho_vs(), rho_vs))
      << ExpectedGot(rho_vs, props.rho_vs());

  // See note in original code about lambda precision
  EXPECT_TRUE(specfem::utilities::is_close(
      props.lambda(), lambdaplus2mu - static_cast<type_real>(2.0) * mu))
      << ExpectedGot(lambda, props.lambda());
}
