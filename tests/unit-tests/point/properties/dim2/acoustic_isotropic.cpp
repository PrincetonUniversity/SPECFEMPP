#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Acoustic Tests
// ============================================================================
TYPED_TEST(PointPropertiesTest, AcousticIsotropic2D) {

  constexpr bool using_simd = TypeParam::value; ///< Use SIMD if true

  // Get the SIMD size from the implementation
  using simd_type =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;
  using T = typename specfem::datatype::simd<type_real, using_simd>::base_type;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Variables to hold property values
  simd_type rho;
  simd_type vp;
  simd_type rho_inv;
  simd_type kappa;
  simd_type kappa_inv;
  simd_type rho_vpinv;

  if constexpr (using_simd) {
    // For SIMD case, we can use array indexing syntax
    T rho_arr[simd_size];
    T vp_arr[simd_size];
    T kappa_arr[simd_size];
    T rho_inv_arr[simd_size];
    T kappa_inv_arr[simd_size];
    T rho_vpinv_arr[simd_size];
    for (int i = 0; i < simd_size; ++i) {
      rho_arr[i] = 1000.0 + i * 20.0; // kg/m³
      vp_arr[i] = 1500.0 + i * 50.0;  // m/s
      kappa_arr[i] = static_cast<type_real>(rho_arr[i]) *
                     static_cast<type_real>(vp_arr[i]) *
                     static_cast<type_real>(vp_arr[i]);
      rho_inv_arr[i] = 1.0 / static_cast<type_real>(rho_arr[i]);
      kappa_inv_arr[i] = 1.0 / static_cast<type_real>(kappa_arr[i]);
      rho_vpinv_arr[i] = 1.0 / (static_cast<type_real>(rho_arr[i]) *
                                static_cast<type_real>(vp_arr[i]));
    }
    rho.copy_from(rho_arr, Kokkos::Experimental::simd_flag_default);
    vp.copy_from(vp_arr, Kokkos::Experimental::simd_flag_default);
    kappa.copy_from(kappa_arr, Kokkos::Experimental::simd_flag_default);
    rho_inv.copy_from(rho_inv_arr, Kokkos::Experimental::simd_flag_default);
    kappa_inv.copy_from(kappa_inv_arr, Kokkos::Experimental::simd_flag_default);
    rho_vpinv.copy_from(rho_vpinv_arr, Kokkos::Experimental::simd_flag_default);
  } else {
    // For scalar case, we need direct assignment
    // Water-like material
    rho = 1000.0;          // kg/m³
    vp = 1500.0;           // m/s
    kappa = rho * vp * vp; // bulk modulus
    rho_inv = 1.0 / rho;
    kappa_inv = 1.0 / kappa;
    rho_vpinv = 1.0 / (rho * vp);
  }

  // Create the properties object
  using PointPropertiesType = specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic, using_simd>;
  PointPropertiesType props(rho_inv, kappa);

  EXPECT_TRUE(specfem::utilities::is_close(props.rho_inverse(), rho_inv))
      << ExpectedGot(kappa, props.rho_inverse());

  EXPECT_TRUE(specfem::utilities::is_close(props.kappa(), kappa))
      << ExpectedGot(kappa, props.kappa());
  EXPECT_TRUE(specfem::utilities::is_close(props.kappa_inverse(), kappa_inv))
      << ExpectedGot(kappa_inv, props.kappa_inverse());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho_vpinverse(), rho_vpinv))
      << ExpectedGot(rho_vpinv, props.rho_vpinverse());

  PointPropertiesType props2;
  props2.rho_inverse() = rho_inv;
  props2.kappa() = kappa;

  simd_type data[] = { rho_inv, kappa };
  PointPropertiesType props3(data);

  PointPropertiesType props4(rho_inv);

  EXPECT_TRUE(props2 == props)
      << ExpectedGot(props2.rho_inverse(), props.rho_inverse())
      << ExpectedGot(props2.kappa(), props.kappa());

  EXPECT_TRUE(props2 == props3)
      << ExpectedGot(props3.rho_inverse(), props2.rho_inverse())
      << ExpectedGot(props3.kappa(), props2.kappa());

  EXPECT_TRUE(specfem::utilities::is_close(props4.rho_inverse(), rho_inv));
  EXPECT_TRUE(specfem::utilities::is_close(props4.kappa(), rho_inv));
}
