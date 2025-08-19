#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include "utilities/interface.hpp"
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
  using T = typename specfem::datatype::simd<type_real, using_simd>::base_type;
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
    T rho_arr[simd_size];
    T kappa_arr[simd_size];
    T vp_arr[simd_size];
    T vs_arr[simd_size];
    T mu_arr[simd_size];
    T lambda_arr[simd_size];
    T lambdaplus2mu_arr[simd_size];
    T rho_vp_arr[simd_size];
    T rho_vs_arr[simd_size];

    // Setup test data for SIMD
    for (int i = 0; i < simd_size; ++i) {
      rho_arr[i] =
          static_cast<type_real>(2700.0) +
          static_cast<type_real>(i) * static_cast<type_real>(50.0); // kg/m³
      vp_arr[i] =
          static_cast<type_real>(6000.0) +
          static_cast<type_real>(i) * static_cast<type_real>(100.0); // m/s
      vs_arr[i] =
          static_cast<type_real>(3500.0) +
          static_cast<type_real>(i) * static_cast<type_real>(50.0); // m/s

      mu_arr[i] = static_cast<type_real>(rho_arr[i]) *
                  static_cast<type_real>(vs_arr[i]) *
                  static_cast<type_real>(vs_arr[i]);

      kappa_arr[i] = static_cast<type_real>(rho_arr[i]) *
                         static_cast<type_real>(vp_arr[i]) *
                         static_cast<type_real>(vp_arr[i]) -
                     static_cast<type_real>(4.0) / static_cast<type_real>(3.0) *
                         static_cast<type_real>(mu_arr[i]);

      lambda_arr[i] =
          static_cast<type_real>(rho_arr[i]) *
              static_cast<type_real>(vp_arr[i]) *
              static_cast<type_real>(vp_arr[i]) -
          static_cast<type_real>(2.0) * static_cast<type_real>(mu_arr[i]);
      lambdaplus2mu_arr[i] =
          static_cast<type_real>(lambda_arr[i]) +
          static_cast<type_real>(2.0) * static_cast<type_real>(mu_arr[i]);
      rho_vp_arr[i] = static_cast<type_real>(rho_arr[i]) *
                      static_cast<type_real>(vp_arr[i]);
      rho_vs_arr[i] = static_cast<type_real>(rho_arr[i]) *
                      static_cast<type_real>(vs_arr[i]);
    }

    // Copy to SIMD types
    rho.copy_from(rho_arr, Kokkos::Experimental::simd_flag_default);
    kappa.copy_from(kappa_arr, Kokkos::Experimental::simd_flag_default);
    mu.copy_from(mu_arr, Kokkos::Experimental::simd_flag_default);
    lambda_val.copy_from(lambda_arr, Kokkos::Experimental::simd_flag_default);
    lambdaplus2mu_val.copy_from(lambdaplus2mu_arr,
                                Kokkos::Experimental::simd_flag_default);
    rho_vp_val.copy_from(rho_vp_arr, Kokkos::Experimental::simd_flag_default);
    rho_vs_val.copy_from(rho_vs_arr, Kokkos::Experimental::simd_flag_default);
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
  using PointPropertiesType = specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic, using_simd>;
  PointPropertiesType props(kappa, mu, rho);

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
  std::cout << "lambdaplus2mu: " << std::endl;
  std::cout << props.lambdaplus2mu() << std::endl;
  std::cout << "lambdaplus2mu_val: " << std::endl;
  std::cout << lambdaplus2mu_val << std::endl;
  std::cout << "lambda: " << std::endl;
  std::cout << props.lambda() << std::endl;
  EXPECT_TRUE(specfem::utilities::is_close(
      props.lambda(), lambdaplus2mu_val - (static_cast<type_real>(2.0)) * mu))
      << ExpectedGot(lambda_val, props.lambda());

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
