#include "specfem/io/mesh/impl/fortran/dim3/read_materials.hpp"
#include <gtest/gtest.h>

namespace {

namespace dim3_impl = specfem::io::mesh::impl::fortran::dim3_impl;

constexpr type_real rho = 2300.0;
constexpr type_real vp = 2800.0;
constexpr type_real vs = 1500.0;

// Voigt indices into the returned array, in constructor order.
enum : std::size_t {
  c11 = 0,
  c12,
  c13,
  c14,
  c15,
  c16,
  c22,
  c23,
  c24,
  c25,
  c26,
  c33,
  c34,
  c35,
  c36,
  c44,
  c45,
  c46,
  c55,
  c56,
  c66
};

/**
 * @brief Check the stiffnesses that carry no anisotropic perturbation.
 *
 * With SPECFEM3D's default factors every term except the one-zeta P term
 * vanishes, so only c14, c24 and c34 depart from the isotropic-equivalent
 * matrix. @p scale is the zeta-independent scaling the model applies.
 */
void expect_isotropic_equivalent_part(const dim3_impl::stiffness_array &c,
                                      const type_real scale) {
  const type_real lambda_plus_two_mu = rho * vp * vp;
  const type_real mu = rho * vs * vs;
  const type_real lambda = lambda_plus_two_mu - 2.0 * mu;
  const type_real tolerance = 1e-6 * lambda_plus_two_mu;

  EXPECT_NEAR(c[c11], scale * lambda_plus_two_mu, tolerance);
  EXPECT_NEAR(c[c22], scale * lambda_plus_two_mu, tolerance);
  EXPECT_NEAR(c[c33], scale * lambda_plus_two_mu, tolerance);

  EXPECT_NEAR(c[c12], scale * lambda, tolerance);
  EXPECT_NEAR(c[c13], scale * lambda, tolerance);
  EXPECT_NEAR(c[c23], scale * lambda, tolerance);

  EXPECT_NEAR(c[c44], scale * mu, tolerance);
  EXPECT_NEAR(c[c55], scale * mu, tolerance);
  EXPECT_NEAR(c[c66], scale * mu, tolerance);

  for (const auto index : { c15, c16, c25, c26, c35, c36, c45, c46, c56 }) {
    EXPECT_NEAR(c[index], 0.0, tolerance) << "at Voigt slot " << index;
  }
}

TEST(ModelAniso, IsotropicFlagGivesIsotropicEquivalentMatrix) {
  const auto c = dim3_impl::model_aniso(0, rho, vp, vs);

  expect_isotropic_equivalent_part(c, 1.0);

  // No perturbation at all: the three coupling terms vanish too.
  const type_real tolerance = 1e-6 * rho * vp * vp;
  EXPECT_NEAR(c[c14], 0.0, tolerance);
  EXPECT_NEAR(c[c24], 0.0, tolerance);
  EXPECT_NEAR(c[c34], 0.0, tolerance);
}

TEST(ModelAniso, Model1AddsOneZetaPerturbation) {
  const auto c = dim3_impl::model_aniso(1, rho, vp, vs);

  // Model 1 leaves the zeta-independent parameters untouched.
  expect_isotropic_equivalent_part(c, 1.0);

  // The one-zeta P term, FACTOR_CS1p_A = 0.2, reaches c14, c24 and c34 as
  // -2 * 0.2 * rho * vp^2 through the d -> c frame rotation.
  const type_real aa = rho * vp * vp;
  const type_real expected = -0.4 * aa;
  const type_real tolerance = 1e-6 * aa;
  EXPECT_NEAR(c[c14], expected, tolerance);
  EXPECT_NEAR(c[c24], expected, tolerance);
  EXPECT_NEAR(c[c34], expected, tolerance);
}

TEST(ModelAniso, Model2ScalesZetaIndependentParametersByTenPercent) {
  const auto c = dim3_impl::model_aniso(2, rho, vp, vs);

  expect_isotropic_equivalent_part(c, 1.1);

  // The zeta-dependent term is built from the unperturbed aa, so the coupling
  // stiffnesses are unchanged from model 1.
  const type_real aa = rho * vp * vp;
  const type_real tolerance = 1e-6 * aa;
  EXPECT_NEAR(c[c14], -0.4 * aa, tolerance);
  EXPECT_NEAR(c[c24], -0.4 * aa, tolerance);
  EXPECT_NEAR(c[c34], -0.4 * aa, tolerance);
}

TEST(ModelAniso, RejectsUnsupportedFlag) {
  EXPECT_THROW(dim3_impl::model_aniso(3, rho, vp, vs), std::runtime_error);
}

} // namespace
