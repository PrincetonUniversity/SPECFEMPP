#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Elastic Tests
// ============================================================================
TYPED_TEST(PointPropertiesTest, ElasticIsotropicCosserat2D) {
  constexpr bool using_simd = TypeParam::value; ///< Use SIMD if true

  // Get the SIMD size from the implementation
  using simd_type =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;
  using T = typename specfem::datatype::simd<type_real, using_simd>::base_type;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Declare variables for properties
  simd_type rho;
  simd_type kappa;
  simd_type mu;
  simd_type nu;
  simd_type j;
  simd_type lambda_c;
  simd_type mu_c;
  simd_type nu_c;
  simd_type lambdaplus2mu_val;
  simd_type lambda_val;
  simd_type rho_vp_val;
  simd_type rho_vs_val;
  simd_type vp_val;
  simd_type vs_val;
  simd_type vmax_val;
  simd_type vmin_val;

  if constexpr (using_simd) {
    T rho_arr[simd_size];
    T kappa_arr[simd_size];
    T mu_arr[simd_size];
    T nu_arr[simd_size];
    T j_arr[simd_size];
    T lambda_c_arr[simd_size];
    T mu_c_arr[simd_size];
    T nu_c_arr[simd_size];

    T vp_arr[simd_size];
    T vs_arr[simd_size];

    T lambda_arr[simd_size];
    T lambdaplus2mu_arr[simd_size];
    T rho_vp_arr[simd_size];
    T rho_vs_arr[simd_size];
    T vmax_arr[simd_size];
    T vmin_arr[simd_size];

    // Setup test data for SIMD
    for (int i = 0; i < simd_size; ++i) {
      rho_arr[i] =
          static_cast<type_real>(1.0e5) +
          static_cast<type_real>(i) * static_cast<type_real>(5.0e3); // kg/m^3
      kappa_arr[i] =
          static_cast<type_real>(22.667e9) +
          static_cast<type_real>(i) * static_cast<type_real>(1e9); // Pa
      mu_arr[i] = static_cast<type_real>(4e9) +
                  static_cast<type_real>(i) * static_cast<type_real>(1e8);
      nu_arr[i] = static_cast<type_real>(2e9) +
                  static_cast<type_real>(i) * static_cast<type_real>(1e8);
      j_arr[i] = static_cast<type_real>(1e4) +
                 static_cast<type_real>(i) * static_cast<type_real>(500.0);
      lambda_c_arr[i] =
          static_cast<type_real>(1e8) +
          static_cast<type_real>(i) * static_cast<type_real>(2.5e6);
      mu_c_arr[i] = static_cast<type_real>(1.936e8) +
                    static_cast<type_real>(i) * static_cast<type_real>(2.5e6);
      nu_c_arr[i] = static_cast<type_real>(3.0464e9) +
                    static_cast<type_real>(i) * static_cast<type_real>(2.5e7);

      vp_arr[i] =
          static_cast<type_real>(6000) +
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

      lambda_arr[i] = static_cast<type_real>(kappa_arr[i]) -
                      static_cast<type_real>(2.0) /
                          static_cast<type_real>(3.0) *
                          static_cast<type_real>(mu_arr[i]);
      lambdaplus2mu_arr[i] =
          static_cast<type_real>(lambda_arr[i]) +
          static_cast<type_real>(2.0) * static_cast<type_real>(mu_arr[i]);
      vp_arr[i] = std::sqrt(lambdaplus2mu_arr[i] / rho_arr[i]);
      vs_arr[i] = std::sqrt(mu_arr[i] / rho_arr[i]);

      rho_vp_arr[i] = static_cast<type_real>(rho_arr[i]) *
                      static_cast<type_real>(vp_arr[i]);
      rho_vs_arr[i] = static_cast<type_real>(rho_arr[i]) *
                      static_cast<type_real>(vs_arr[i]);
      vmax_arr[i] = std::max(static_cast<type_real>(vp_arr[i]),
                             static_cast<type_real>(vs_arr[i]));
      vmin_arr[i] = std::min(static_cast<type_real>(vp_arr[i]),
                             static_cast<type_real>(vs_arr[i]));
    }

    // Copy to SIMD types
    rho = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        rho_arr, Kokkos::Experimental::simd_flag_default);
    kappa = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        kappa_arr, Kokkos::Experimental::simd_flag_default);
    mu = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        mu_arr, Kokkos::Experimental::simd_flag_default);
    nu = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        nu_arr, Kokkos::Experimental::simd_flag_default);
    j = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        j_arr, Kokkos::Experimental::simd_flag_default);
    lambda_c = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        lambda_c_arr, Kokkos::Experimental::simd_flag_default);
    mu_c = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        mu_c_arr, Kokkos::Experimental::simd_flag_default);
    nu_c = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        nu_c_arr, Kokkos::Experimental::simd_flag_default);
    lambda_val = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        lambda_arr, Kokkos::Experimental::simd_flag_default);
    lambdaplus2mu_val = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        lambdaplus2mu_arr, Kokkos::Experimental::simd_flag_default);
    rho_vp_val = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        rho_vp_arr, Kokkos::Experimental::simd_flag_default);
    rho_vs_val = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        rho_vs_arr, Kokkos::Experimental::simd_flag_default);
    vp_val = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        vp_arr, Kokkos::Experimental::simd_flag_default);
    vs_val = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        vs_arr, Kokkos::Experimental::simd_flag_default);
    vmax_val = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        vmax_arr, Kokkos::Experimental::simd_flag_default);
    vmin_val = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        vmin_arr, Kokkos::Experimental::simd_flag_default);
  } else {
    // Kulesh (2009) material properties for scalar test
    constexpr type_real rho_val = 1.0e5;      // kg/m^3
    constexpr type_real kappa_val = 22.667e9; // Pa
    constexpr type_real mu_val = 4e9;         // Pa
    constexpr type_real nu_val = 2e9;         // Pa
    constexpr type_real j_val = 1e4;          // kg/m
    constexpr type_real lambda_c_val = 1e8;   // N
    constexpr type_real mu_c_val = 1.936e8;   // N
    constexpr type_real nu_c_val = 3.0464e9;  // N

    constexpr type_real lambda_scalar =
        kappa_val - static_cast<type_real>(2.0) / static_cast<type_real>(3.0) *
                        mu_val; // Lamé's first parameter
    constexpr type_real lambdaplus2mu_scalar = kappa_val + (4.0 / 3.0) * mu_val;
    const type_real vp = std::sqrt(lambdaplus2mu_scalar / rho_val);
    const type_real vs = std::sqrt(mu_val / rho_val);

    // Assign to our variables
    rho = rho_val;
    kappa = kappa_val;
    mu = mu_val;
    nu = nu_val;
    j = j_val;
    lambda_c = lambda_c_val;
    mu_c = mu_c_val;
    nu_c = nu_c_val;
    lambdaplus2mu_val = lambdaplus2mu_scalar;
    lambda_val = lambda_scalar;
    rho_vp_val = rho_val * vp;
    rho_vs_val = rho_val * vs;
    vp_val = vp;
    vs_val = vs;
    vmax_val = std::max(vp, vs);
    vmin_val = std::min(vp, vs);
  }

  // Create the properties object
  using PointPropertiesType = specfem::point::properties<specfem::tags::Tags<
      specfem::element::dimension_tag::dim2,
      specfem::element::medium_tag::elastic_spin,
      specfem::element::property_tag::isotropic_cosserat, using_simd> >;
  PointPropertiesType props(rho, kappa, mu, nu, j, lambda_c, mu_c, nu_c);

  EXPECT_TRUE(specfem::utilities::is_close(props.rho(), rho))
      << ExpectedGot(rho, props.rho());
  EXPECT_TRUE(specfem::utilities::is_close(props.kappa(), kappa))
      << ExpectedGot(kappa, props.kappa());
  EXPECT_TRUE(specfem::utilities::is_close(props.mu(), mu))
      << ExpectedGot(mu, props.mu());
  EXPECT_TRUE(specfem::utilities::is_close(props.nu(), nu))
      << ExpectedGot(nu, props.nu());
  EXPECT_TRUE(specfem::utilities::is_close(props.j(), j))
      << ExpectedGot(j, props.j());
  EXPECT_TRUE(specfem::utilities::is_close(props.lambda_c(), lambda_c))
      << ExpectedGot(lambda_c, props.lambda_c());
  EXPECT_TRUE(specfem::utilities::is_close(props.mu_c(), mu_c))
      << ExpectedGot(mu_c, props.mu_c());
  EXPECT_TRUE(specfem::utilities::is_close(props.nu_c(), nu_c))
      << ExpectedGot(nu_c, props.nu_c());

  EXPECT_TRUE(
      specfem::utilities::is_close(props.lambdaplus2mu(), lambdaplus2mu_val))
      << ExpectedGot(lambdaplus2mu_val, props.lambdaplus2mu());
  EXPECT_TRUE(specfem::utilities::is_close(props.lambda(), lambda_val))
      << ExpectedGot(lambda_val, props.lambda());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho_vp(), rho_vp_val))
      << ExpectedGot(rho_vp_val, props.rho_vp());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho_vs(), rho_vs_val))
      << ExpectedGot(rho_vs_val, props.rho_vs());

  // New property checks
  EXPECT_TRUE(specfem::utilities::is_close(props.vp(), vp_val))
      << ExpectedGot(vp_val, props.vp());
  EXPECT_TRUE(specfem::utilities::is_close(props.vs(), vs_val))
      << ExpectedGot(vs_val, props.vs());
  EXPECT_TRUE(specfem::utilities::is_close(props.vmax(), vmax_val))
      << ExpectedGot(vmax_val, props.vmax());
  EXPECT_TRUE(specfem::utilities::is_close(props.vmin(), vmin_val))
      << ExpectedGot(vmin_val, props.vmin());

  // Additional constructors and assignment tests
  PointPropertiesType props2;
  props2.rho() = rho;
  props2.kappa() = kappa;
  props2.mu() = mu;
  props2.nu() = nu;
  props2.j() = j;
  props2.lambda_c() = lambda_c;
  props2.mu_c() = mu_c;
  props2.nu_c() = nu_c;

  simd_type data[] = { rho, kappa, mu, nu, j, lambda_c, mu_c, nu_c };
  PointPropertiesType props3(data);

  PointPropertiesType props4(kappa);

  EXPECT_TRUE(props2 == props)
      << ExpectedGot(props2.rho(), props.rho())
      << ExpectedGot(props2.kappa(), props.kappa())
      << ExpectedGot(props2.mu(), props.mu())
      << ExpectedGot(props2.nu(), props.nu())
      << ExpectedGot(props2.j(), props.j())
      << ExpectedGot(props2.lambda_c(), props.lambda_c())
      << ExpectedGot(props2.mu_c(), props.mu_c())
      << ExpectedGot(props2.nu_c(), props.nu_c());

  EXPECT_TRUE(props2 == props3)
      << ExpectedGot(props2.rho(), props3.rho())
      << ExpectedGot(props2.kappa(), props3.kappa())
      << ExpectedGot(props2.mu(), props3.mu())
      << ExpectedGot(props2.nu(), props3.nu())
      << ExpectedGot(props2.j(), props3.j())
      << ExpectedGot(props2.lambda_c(), props3.lambda_c())
      << ExpectedGot(props2.mu_c(), props3.mu_c())
      << ExpectedGot(props2.nu_c(), props3.nu_c());
  EXPECT_TRUE(specfem::utilities::is_close(props4.rho(), kappa));
  EXPECT_TRUE(specfem::utilities::is_close(props4.kappa(), kappa));
  EXPECT_TRUE(specfem::utilities::is_close(props4.mu(), kappa));
  EXPECT_TRUE(specfem::utilities::is_close(props4.nu(), kappa));
  EXPECT_TRUE(specfem::utilities::is_close(props4.j(), kappa));
  EXPECT_TRUE(specfem::utilities::is_close(props4.lambda_c(), kappa));
  EXPECT_TRUE(specfem::utilities::is_close(props4.mu_c(), kappa));
  EXPECT_TRUE(specfem::utilities::is_close(props4.nu_c(), kappa));
}
