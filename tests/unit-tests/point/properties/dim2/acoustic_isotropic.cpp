#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include "utilities/simd.hpp"
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
    for (int i = 0; i < simd_size; ++i) {
      rho[i] = 1000.0 + i * 20.0; // kg/m³
      vp[i] = 1500.0 + i * 50.0;  // m/s
      kappa[i] = static_cast<type_real>(rho[i]) *
                 static_cast<type_real>(vp[i]) * static_cast<type_real>(vp[i]);
      rho_inv[i] = 1.0 / static_cast<type_real>(rho[i]);
      kappa_inv[i] = 1.0 / static_cast<type_real>(kappa[i]);
      rho_vpinv[i] = 1.0 / (static_cast<type_real>(rho[i]) *
                            static_cast<type_real>(vp[i]));
    }
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
  specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic, using_simd>
      props(rho_inv, kappa);

  EXPECT_TRUE(specfem::utilities::is_close(props.rho_inverse(), rho_inv))
      << ExpectedGot(kappa, props.rho_inverse());

  EXPECT_TRUE(specfem::utilities::is_close(props.kappa(), kappa))
      << ExpectedGot(kappa, props.kappa());
  EXPECT_TRUE(specfem::utilities::is_close(props.kappa_inverse(), kappa_inv))
      << ExpectedGot(kappa_inv, props.kappa_inverse());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho_vpinverse(), rho_vpinv))
      << ExpectedGot(rho_vpinv, props.rho_vpinverse());
}
