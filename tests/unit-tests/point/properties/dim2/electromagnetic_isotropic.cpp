#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include "utilities/interface.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Electromagnetic Tests
// ============================================================================
TYPED_TEST(PointPropertiesTest, ElectromagneticIsotropic2D) {
  constexpr bool using_simd = TypeParam::value; ///< Use SIMD if true

  // Get the SIMD size from the implementation
  using simd_type =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;
  using T = typename specfem::datatype::simd<type_real, using_simd>::base_type;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Declare variables for properties
  simd_type mu0_inv;
  simd_type eps11;
  simd_type eps33;
  simd_type sig11;
  simd_type sig33;

  if constexpr (using_simd) {
    T mu0_inv_arr[simd_size];
    T eps11_arr[simd_size];
    T eps33_arr[simd_size];
    T sig11_arr[simd_size];
    T sig33_arr[simd_size];
    // Setup test data for SIMD
    for (int i = 0; i < simd_size; ++i) {
      mu0_inv_arr[i] = 7.957747e5 + i * 1.0e4; // 1/μ₀ (1/H/m)
      eps11_arr[i] = 8.85e-12 + i * 1.0e-13;   // permittivity in xx (F/m)
      eps33_arr[i] = 8.85e-12 + i * 1.0e-13;   // permittivity in zz (F/m)
      sig11_arr[i] = 1.0e-2 + i * 1.0e-3;      // conductivity in xx (S/m)
      sig33_arr[i] = 1.0e-2 + i * 1.0e-3;      // conductivity in zz (S/m)
    }
    // Copy to SIMD types
    mu0_inv.copy_from(mu0_inv_arr, Kokkos::Experimental::simd_flag_default);
    eps11.copy_from(eps11_arr, Kokkos::Experimental::simd_flag_default);
    eps33.copy_from(eps33_arr, Kokkos::Experimental::simd_flag_default);
    sig11.copy_from(sig11_arr, Kokkos::Experimental::simd_flag_default);
    sig33.copy_from(sig33_arr, Kokkos::Experimental::simd_flag_default);
  } else {
    // Electromagnetic medium properties for scalar case
    mu0_inv = 7.957747e5; // 1/μ₀ (1/H/m)
    eps11 = 8.85e-12;     // permittivity in xx (F/m)
    eps33 = 8.85e-12;     // permittivity in zz (F/m)
    sig11 = 1.0e-2;       // conductivity in xx (S/m)
    sig33 = 1.0e-2;       // conductivity in zz (S/m)
  }

  // Create the properties object
  using PointPropertiesType =
      specfem::point::properties<specfem::dimension::type::dim2,
                                 specfem::element::medium_tag::electromagnetic,
                                 specfem::element::property_tag::isotropic,
                                 using_simd>;
  PointPropertiesType props(mu0_inv, eps11, eps33, sig11, sig33);

  EXPECT_TRUE(specfem::utilities::is_close(props.mu0_inv(), mu0_inv))
      << ExpectedGot(mu0_inv, props.mu0_inv());
  EXPECT_TRUE(specfem::utilities::is_close(props.eps11(), eps11))
      << ExpectedGot(eps11, props.eps11());
  EXPECT_TRUE(specfem::utilities::is_close(props.eps33(), eps33))
      << ExpectedGot(eps33, props.eps33());
  EXPECT_TRUE(specfem::utilities::is_close(props.sig11(), sig11))
      << ExpectedGot(sig11, props.sig11());
  EXPECT_TRUE(specfem::utilities::is_close(props.sig33(), sig33))
      << ExpectedGot(sig33, props.sig33());

  // Additional constructors and assignment tests
  PointPropertiesType props2;
  props2.mu0_inv() = mu0_inv;
  props2.eps11() = eps11;
  props2.eps33() = eps33;
  props2.sig11() = sig11;
  props2.sig33() = sig33;

  simd_type data[] = { mu0_inv, eps11, eps33, sig11, sig33 };
  PointPropertiesType props3(data);

  PointPropertiesType props4(mu0_inv);

  EXPECT_TRUE(props2 == props) << ExpectedGot(props2.mu0_inv(), props.mu0_inv())
                               << ExpectedGot(props2.eps11(), props.eps11())
                               << ExpectedGot(props2.eps33(), props.eps33())
                               << ExpectedGot(props2.sig11(), props.sig11())
                               << ExpectedGot(props2.sig33(), props.sig33());

  EXPECT_TRUE(props2 == props3)
      << ExpectedGot(props3.mu0_inv(), props2.mu0_inv())
      << ExpectedGot(props3.eps11(), props2.eps11())
      << ExpectedGot(props3.eps33(), props2.eps33())
      << ExpectedGot(props3.sig11(), props2.sig11())
      << ExpectedGot(props3.sig33(), props2.sig33());

  EXPECT_TRUE(specfem::utilities::is_close(props4.mu0_inv(), mu0_inv));
  EXPECT_TRUE(specfem::utilities::is_close(props4.eps11(), mu0_inv));
  EXPECT_TRUE(specfem::utilities::is_close(props4.eps33(), mu0_inv));
  EXPECT_TRUE(specfem::utilities::is_close(props4.sig11(), mu0_inv));
  EXPECT_TRUE(specfem::utilities::is_close(props4.sig33(), mu0_inv));
}
