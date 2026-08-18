#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Elastic Anisotropic Tests
// ============================================================================
TYPED_TEST(PointPropertiesTest, ElasticAnisotropic2D) {

  constexpr bool using_simd = TypeParam::value; ///< Use SIMD if true

  // Get the SIMD size from the implementation
  using simd_type =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;
  using T = typename specfem::datatype::simd<type_real, using_simd>::base_type;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Declare variables for properties
  simd_type c11;
  simd_type c13;
  simd_type c15;
  simd_type c33;
  simd_type c35;
  simd_type c55;
  simd_type c12;
  simd_type c23;
  simd_type c25;
  simd_type rho;
  simd_type rho_vp_val;
  simd_type rho_vs_val;
  simd_type vp_val;
  simd_type vs_val;
  simd_type vmax_val;
  simd_type vmin_val;

  if constexpr (using_simd) {
    T rho_arr[simd_size];
    T c11_arr[simd_size];
    T c13_arr[simd_size];
    T c15_arr[simd_size];
    T c33_arr[simd_size];
    T c35_arr[simd_size];
    T c55_arr[simd_size];
    T c12_arr[simd_size];
    T c23_arr[simd_size];
    T c25_arr[simd_size];
    T rho_vp_val_arr[simd_size];
    T rho_vs_val_arr[simd_size];
    T vp_val_arr[simd_size];
    T vs_val_arr[simd_size];
    T vmax_val_arr[simd_size];
    T vmin_val_arr[simd_size];
    // Setup test data for SIMD
    for (int i = 0; i < simd_size; ++i) {
      rho_arr[i] = 2500.0 + i * 50.0;  // kg/m³
      c11_arr[i] = 75.0e9 + i * 1.0e9; // Pa
      c13_arr[i] = 15.0e9 + i * 0.5e9; // Pa
      c15_arr[i] = i * 0.0;            // Pa (zero for simplicity)
      c33_arr[i] = 55.0e9 + i * 1.0e9; // Pa
      c35_arr[i] = i * 0.0;            // Pa (zero for simplicity)
      c55_arr[i] = 20.0e9 + i * 0.5e9; // Pa
      c12_arr[i] = 15.0e9 + i * 0.5e9; // Pa
      c23_arr[i] = 10.0e9 + i * 0.5e9; // Pa
      c25_arr[i] = i * 0.0;            // Pa (zero for simplicity)

      // Computed values for verification
      rho_vp_val_arr[i] = std::sqrt(static_cast<type_real>(rho_arr[i]) *
                                    static_cast<type_real>(c33_arr[i]));
      rho_vs_val_arr[i] = std::sqrt(static_cast<type_real>(rho_arr[i]) *
                                    static_cast<type_real>(c55_arr[i]));
      vp_val_arr[i] = std::sqrt(static_cast<type_real>(c33_arr[i]) /
                                static_cast<type_real>(rho_arr[i]));
      vs_val_arr[i] = std::sqrt(static_cast<type_real>(c55_arr[i]) /
                                static_cast<type_real>(rho_arr[i]));
      vmax_val_arr[i] = std::max(vp_val_arr[i], vs_val_arr[i]);
      vmin_val_arr[i] = std::min(vp_val_arr[i], vs_val_arr[i]);
    }
    // Copy to SIMD types
    rho = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        rho_arr, Kokkos::Experimental::simd_flag_default);
    c11 = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        c11_arr, Kokkos::Experimental::simd_flag_default);
    c13 = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        c13_arr, Kokkos::Experimental::simd_flag_default);
    c15 = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        c15_arr, Kokkos::Experimental::simd_flag_default);
    c33 = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        c33_arr, Kokkos::Experimental::simd_flag_default);
    c35 = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        c35_arr, Kokkos::Experimental::simd_flag_default);
    c55 = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        c55_arr, Kokkos::Experimental::simd_flag_default);
    c12 = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        c12_arr, Kokkos::Experimental::simd_flag_default);
    c23 = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        c23_arr, Kokkos::Experimental::simd_flag_default);
    c25 = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        c25_arr, Kokkos::Experimental::simd_flag_default);
    rho_vp_val = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        rho_vp_val_arr, Kokkos::Experimental::simd_flag_default);
    rho_vs_val = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        rho_vs_val_arr, Kokkos::Experimental::simd_flag_default);
    vp_val = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        vp_val_arr, Kokkos::Experimental::simd_flag_default);
    vs_val = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        vs_val_arr, Kokkos::Experimental::simd_flag_default);
    vmax_val = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        vmax_val_arr, Kokkos::Experimental::simd_flag_default);
    vmin_val = Kokkos::Experimental::simd_unchecked_load<simd_type>(
        vmin_val_arr, Kokkos::Experimental::simd_flag_default);
  } else {
    // Anisotropic material values (e.g., shale-like)
    constexpr type_real rho_val = 2500.0; // kg/m³
    constexpr type_real c11_val = 75.0e9; // Pa
    constexpr type_real c13_val = 15.0e9; // Pa
    constexpr type_real c15_val = 0.0;    // Pa (zero for simplicity)
    constexpr type_real c33_val = 55.0e9; // Pa
    constexpr type_real c35_val = 0.0;    // Pa (zero for simplicity)
    constexpr type_real c55_val = 20.0e9; // Pa
    constexpr type_real c12_val = 15.0e9; // Pa
    constexpr type_real c23_val = 10.0e9; // Pa
    constexpr type_real c25_val = 0.0;    // Pa (zero for simplicity)

    // Assign to our variables
    rho = rho_val;
    c11 = c11_val;
    c13 = c13_val;
    c15 = c15_val;
    c33 = c33_val;
    c35 = c35_val;
    c55 = c55_val;
    c12 = c12_val;
    c23 = c23_val;
    c25 = c25_val;

    // Computed values for verification
    rho_vp_val = std::sqrt(rho_val * c33_val);
    rho_vs_val = std::sqrt(rho_val * c55_val);
    vp_val = std::sqrt(c33_val / rho_val);
    vs_val = std::sqrt(c55_val / rho_val);
    vmax_val = std::max(vp_val, vs_val);
    vmin_val = std::min(vp_val, vs_val);
  }

  // Create the properties object
  using PointPropertiesType = specfem::point::properties<specfem::tags::Tags<
      specfem::element::dimension_tag::dim2,
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::anisotropic, using_simd> >;
  PointPropertiesType props(c11, c13, c15, c33, c35, c55, c12, c23, c25, rho);

  // Additional constructors and assignment tests
  PointPropertiesType props2;
  props2.c11() = c11;
  props2.c13() = c13;
  props2.c15() = c15;
  props2.c33() = c33;
  props2.c35() = c35;
  props2.c55() = c55;
  props2.c12() = c12;
  props2.c23() = c23;
  props2.c25() = c25;
  props2.rho() = rho;

  simd_type data[] = { c11, c13, c15, c33, c35, c55, c12, c23, c25, rho };
  PointPropertiesType props3(data);

  PointPropertiesType props4(c11);

  EXPECT_TRUE(props2 == props) << ExpectedGot(props2.c11(), props.c11())
                               << ExpectedGot(props2.c13(), props.c13())
                               << ExpectedGot(props2.c15(), props.c15())
                               << ExpectedGot(props2.c33(), props.c33())
                               << ExpectedGot(props2.c35(), props.c35())
                               << ExpectedGot(props2.c55(), props.c55())
                               << ExpectedGot(props2.c12(), props.c12())
                               << ExpectedGot(props2.c23(), props.c23())
                               << ExpectedGot(props2.c25(), props.c25())
                               << ExpectedGot(props2.rho(), props.rho());

  EXPECT_TRUE(props2 == props3) << ExpectedGot(props3.c11(), props2.c11())
                                << ExpectedGot(props3.c13(), props2.c13())
                                << ExpectedGot(props3.c15(), props2.c15())
                                << ExpectedGot(props3.c33(), props2.c33())
                                << ExpectedGot(props3.c35(), props2.c35())
                                << ExpectedGot(props3.c55(), props2.c55())
                                << ExpectedGot(props3.c12(), props2.c12())
                                << ExpectedGot(props3.c23(), props2.c23())
                                << ExpectedGot(props3.c25(), props2.c25())
                                << ExpectedGot(props3.rho(), props2.rho());

  EXPECT_TRUE(specfem::utilities::is_close(props4.c11(), c11));
  EXPECT_TRUE(specfem::utilities::is_close(props4.c13(), c11));
  EXPECT_TRUE(specfem::utilities::is_close(props4.c15(), c11));
  EXPECT_TRUE(specfem::utilities::is_close(props4.c33(), c11));
  EXPECT_TRUE(specfem::utilities::is_close(props4.c35(), c11));
  EXPECT_TRUE(specfem::utilities::is_close(props4.c55(), c11));
  EXPECT_TRUE(specfem::utilities::is_close(props4.c12(), c11));
  EXPECT_TRUE(specfem::utilities::is_close(props4.c23(), c11));
  EXPECT_TRUE(specfem::utilities::is_close(props4.c25(), c11));
  EXPECT_TRUE(specfem::utilities::is_close(props4.rho(), c11));

  EXPECT_TRUE(specfem::utilities::is_close(props.c11(), c11))
      << ExpectedGot(c11, props.c11());
  EXPECT_TRUE(specfem::utilities::is_close(props.c13(), c13))
      << ExpectedGot(c13, props.c13());
  EXPECT_TRUE(specfem::utilities::is_close(props.c15(), c15))
      << ExpectedGot(c15, props.c15());
  EXPECT_TRUE(specfem::utilities::is_close(props.c33(), c33))
      << ExpectedGot(c33, props.c33());
  EXPECT_TRUE(specfem::utilities::is_close(props.c35(), c35))
      << ExpectedGot(c35, props.c35());
  EXPECT_TRUE(specfem::utilities::is_close(props.c55(), c55))
      << ExpectedGot(c55, props.c55());
  EXPECT_TRUE(specfem::utilities::is_close(props.c12(), c12))
      << ExpectedGot(c12, props.c12());
  EXPECT_TRUE(specfem::utilities::is_close(props.c23(), c23))
      << ExpectedGot(c23, props.c23());
  EXPECT_TRUE(specfem::utilities::is_close(props.c25(), c25))
      << ExpectedGot(c25, props.c25());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho(), rho))
      << ExpectedGot(rho, props.rho());

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
}
