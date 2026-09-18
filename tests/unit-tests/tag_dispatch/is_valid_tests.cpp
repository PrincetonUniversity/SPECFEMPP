// Unit tests for specfem/tag_dispatch/is_valid.hpp
//
// Tests all four constexpr validity predicates:
//   is_valid_medium_combo   (dim × medium)
//   is_valid_property_combo (dim × medium × property)
//   is_valid_material_combo (dim × medium × property × attenuation)
//   is_valid_boundary_combo (dim × medium × property × boundary)
//   is_valid_full_combo     (dim × medium × property × attenuation × boundary)
//
// Both runtime (EXPECT_TRUE/FALSE) and compile-time (static_assert) coverage.

#include "specfem/tag_dispatch/is_valid.hpp"
#include <gtest/gtest.h>

using specfem::tag_dispatch::impl::BoundaryComboTuple;
using specfem::tag_dispatch::impl::ElementTagTuple;
using specfem::tag_dispatch::impl::is_valid_boundary_combo;
using specfem::tag_dispatch::impl::is_valid_full_combo;
using specfem::tag_dispatch::impl::is_valid_material_combo;
using specfem::tag_dispatch::impl::is_valid_medium_combo;
using specfem::tag_dispatch::impl::is_valid_property_combo;
using specfem::tag_dispatch::impl::MaterialTagTuple;
using specfem::tag_dispatch::impl::MediumTagTuple;
using specfem::tag_dispatch::impl::PropertyComboTuple;

using D = specfem::element::dimension_tag;
using M = specfem::element::medium_tag;
using P = specfem::element::property_tag;
using A = specfem::element::attenuation_tag;
using B = specfem::element::boundary_tag;

// ── is_valid_medium_combo ─────────────────────────────────────────────

TEST(IsValidMedium, Dim2ValidMedia) {
  EXPECT_TRUE(is_valid_medium_combo({ D::dim2, M::elastic_psv }));
  EXPECT_TRUE(is_valid_medium_combo({ D::dim2, M::elastic_sh }));
  EXPECT_TRUE(is_valid_medium_combo({ D::dim2, M::elastic_psv_t }));
  EXPECT_TRUE(is_valid_medium_combo({ D::dim2, M::acoustic }));
  EXPECT_TRUE(is_valid_medium_combo({ D::dim2, M::poroelastic }));
  EXPECT_TRUE(is_valid_medium_combo({ D::dim2, M::electromagnetic_te }));
}

TEST(IsValidMedium, Dim2InvalidMedia) {
  EXPECT_FALSE(is_valid_medium_combo({ D::dim2, M::elastic }));
  EXPECT_FALSE(is_valid_medium_combo({ D::dim2, M::elastic_spin }));
}

TEST(IsValidMedium, Dim3ValidMedia) {
  EXPECT_TRUE(is_valid_medium_combo({ D::dim3, M::elastic }));
  EXPECT_TRUE(is_valid_medium_combo({ D::dim3, M::acoustic }));
  EXPECT_TRUE(is_valid_medium_combo({ D::dim3, M::elastic_spin }));
}

TEST(IsValidMedium, Dim3InvalidMedia) {
  EXPECT_FALSE(is_valid_medium_combo({ D::dim3, M::elastic_psv }));
  EXPECT_FALSE(is_valid_medium_combo({ D::dim3, M::elastic_sh }));
  EXPECT_FALSE(is_valid_medium_combo({ D::dim3, M::elastic_psv_t }));
  EXPECT_FALSE(is_valid_medium_combo({ D::dim3, M::poroelastic }));
  EXPECT_FALSE(is_valid_medium_combo({ D::dim3, M::electromagnetic_te }));
}

// ── is_valid_property_combo ───────────────────────────────────────────

TEST(IsValidProperty, ElasticPSV_SH_IsotropicOrAnisotropic) {
  EXPECT_TRUE(
      is_valid_property_combo({ D::dim2, M::elastic_psv, P::isotropic }));
  EXPECT_TRUE(
      is_valid_property_combo({ D::dim2, M::elastic_psv, P::anisotropic }));
  EXPECT_FALSE(is_valid_property_combo(
      { D::dim2, M::elastic_psv, P::isotropic_cosserat }));
  EXPECT_TRUE(
      is_valid_property_combo({ D::dim2, M::elastic_sh, P::isotropic }));
  EXPECT_TRUE(
      is_valid_property_combo({ D::dim2, M::elastic_sh, P::anisotropic }));
  EXPECT_FALSE(is_valid_property_combo(
      { D::dim2, M::elastic_sh, P::isotropic_cosserat }));
}

TEST(IsValidProperty, ElasticDim3_IsotropicOrAnisotropic) {
  EXPECT_TRUE(is_valid_property_combo({ D::dim3, M::elastic, P::isotropic }));
  EXPECT_TRUE(is_valid_property_combo({ D::dim3, M::elastic, P::anisotropic }));
  EXPECT_FALSE(
      is_valid_property_combo({ D::dim3, M::elastic, P::isotropic_cosserat }));
}

TEST(IsValidProperty, Cosserat_IsotropicCosseratOnly) {
  EXPECT_TRUE(is_valid_property_combo(
      { D::dim2, M::elastic_psv_t, P::isotropic_cosserat }));
  EXPECT_FALSE(
      is_valid_property_combo({ D::dim2, M::elastic_psv_t, P::isotropic }));
  EXPECT_FALSE(
      is_valid_property_combo({ D::dim2, M::elastic_psv_t, P::anisotropic }));
  EXPECT_TRUE(is_valid_property_combo(
      { D::dim3, M::elastic_spin, P::isotropic_cosserat }));
  EXPECT_FALSE(
      is_valid_property_combo({ D::dim3, M::elastic_spin, P::isotropic }));
}

TEST(IsValidProperty, Acoustic_Poroelastic_EMTE_IsotropicOnly) {
  EXPECT_TRUE(is_valid_property_combo({ D::dim2, M::acoustic, P::isotropic }));
  EXPECT_FALSE(
      is_valid_property_combo({ D::dim2, M::acoustic, P::anisotropic }));
  EXPECT_TRUE(
      is_valid_property_combo({ D::dim2, M::poroelastic, P::isotropic }));
  EXPECT_FALSE(
      is_valid_property_combo({ D::dim2, M::poroelastic, P::anisotropic }));
  EXPECT_TRUE(is_valid_property_combo(
      { D::dim2, M::electromagnetic_te, P::isotropic }));
  EXPECT_FALSE(is_valid_property_combo(
      { D::dim2, M::electromagnetic_te, P::anisotropic }));
}

TEST(IsValidProperty, PropagatesInvalidMedium) {
  // dim3 × elastic_psv → medium is invalid → property also invalid
  EXPECT_FALSE(
      is_valid_property_combo({ D::dim3, M::elastic_psv, P::isotropic }));
  EXPECT_FALSE(
      is_valid_property_combo({ D::dim3, M::elastic_psv, P::anisotropic }));
}

// ── is_valid_material_combo ───────────────────────────────────────────

TEST(IsValidMaterial, Cosserat_NoAttenuation) {
  EXPECT_TRUE(is_valid_material_combo(
      { D::dim2, M::elastic_psv_t, P::isotropic_cosserat, A::none }));
  EXPECT_FALSE(is_valid_material_combo({ D::dim2, M::elastic_psv_t,
                                         P::isotropic_cosserat,
                                         A::constant_isotropic }));
  EXPECT_TRUE(is_valid_material_combo(
      { D::dim3, M::elastic_spin, P::isotropic_cosserat, A::none }));
  EXPECT_FALSE(
      is_valid_material_combo({ D::dim3, M::elastic_spin, P::isotropic_cosserat,
                                A::constant_isotropic }));
}

TEST(IsValidMaterial, EMTE_NoAttenuation) {
  EXPECT_TRUE(is_valid_material_combo(
      { D::dim2, M::electromagnetic_te, P::isotropic, A::none }));
  EXPECT_FALSE(is_valid_material_combo(
      { D::dim2, M::electromagnetic_te, P::isotropic, A::constant_isotropic }));
}

TEST(IsValidMaterial, OthersAllowBothAttenuation) {
  EXPECT_TRUE(is_valid_material_combo(
      { D::dim2, M::elastic_psv, P::isotropic, A::none }));
  EXPECT_TRUE(is_valid_material_combo(
      { D::dim2, M::elastic_psv, P::isotropic, A::constant_isotropic }));
  EXPECT_TRUE(
      is_valid_material_combo({ D::dim2, M::acoustic, P::isotropic, A::none }));
  EXPECT_TRUE(is_valid_material_combo(
      { D::dim2, M::acoustic, P::isotropic, A::constant_isotropic }));
  EXPECT_TRUE(
      is_valid_material_combo({ D::dim3, M::elastic, P::isotropic, A::none }));
  EXPECT_TRUE(is_valid_material_combo(
      { D::dim3, M::elastic, P::isotropic, A::constant_isotropic }));
  EXPECT_TRUE(is_valid_material_combo(
      { D::dim3, M::elastic, P::anisotropic, A::none }));
  EXPECT_FALSE(is_valid_material_combo(
      { D::dim3, M::elastic, P::anisotropic, A::constant_isotropic }));
}

// ── is_valid_boundary_combo ───────────────────────────────────────────

TEST(IsValidBoundary, Dim3_ElasticNoneOrStacey) {
  EXPECT_TRUE(
      is_valid_boundary_combo({ D::dim3, M::elastic, P::isotropic, B::none }));
  EXPECT_TRUE(is_valid_boundary_combo(
      { D::dim3, M::elastic, P::isotropic, B::stacey }));
  // composite_stacey_dirichlet requires acoustic medium
  EXPECT_FALSE(is_valid_boundary_combo(
      { D::dim3, M::elastic, P::isotropic, B::composite_stacey_dirichlet }));
  EXPECT_FALSE(is_valid_boundary_combo(
      { D::dim3, M::elastic, P::isotropic, B::acoustic_free_surface }));
}

TEST(IsValidBoundary, Dim3_AcousticAllBoundaries) {
  EXPECT_TRUE(
      is_valid_boundary_combo({ D::dim3, M::acoustic, P::isotropic, B::none }));
  EXPECT_TRUE(is_valid_boundary_combo(
      { D::dim3, M::acoustic, P::isotropic, B::acoustic_free_surface }));
  EXPECT_TRUE(is_valid_boundary_combo(
      { D::dim3, M::acoustic, P::isotropic, B::stacey }));
  EXPECT_TRUE(is_valid_boundary_combo(
      { D::dim3, M::acoustic, P::isotropic, B::composite_stacey_dirichlet }));
}

TEST(IsValidBoundary, Acoustic_AllBoundaries) {
  EXPECT_TRUE(
      is_valid_boundary_combo({ D::dim2, M::acoustic, P::isotropic, B::none }));
  EXPECT_TRUE(is_valid_boundary_combo(
      { D::dim2, M::acoustic, P::isotropic, B::acoustic_free_surface }));
  EXPECT_TRUE(is_valid_boundary_combo(
      { D::dim2, M::acoustic, P::isotropic, B::stacey }));
  EXPECT_TRUE(is_valid_boundary_combo(
      { D::dim2, M::acoustic, P::isotropic, B::composite_stacey_dirichlet }));
}

TEST(IsValidBoundary, EMTE_NoneOnly) {
  EXPECT_TRUE(is_valid_boundary_combo(
      { D::dim2, M::electromagnetic_te, P::isotropic, B::none }));
  EXPECT_FALSE(is_valid_boundary_combo(
      { D::dim2, M::electromagnetic_te, P::isotropic, B::stacey }));
  EXPECT_FALSE(
      is_valid_boundary_combo({ D::dim2, M::electromagnetic_te, P::isotropic,
                                B::acoustic_free_surface }));
}

TEST(IsValidBoundary, ElasticPSV_NoneOrStacey) {
  EXPECT_TRUE(is_valid_boundary_combo(
      { D::dim2, M::elastic_psv, P::isotropic, B::none }));
  EXPECT_TRUE(is_valid_boundary_combo(
      { D::dim2, M::elastic_psv, P::isotropic, B::stacey }));
  EXPECT_FALSE(is_valid_boundary_combo(
      { D::dim2, M::elastic_psv, P::isotropic, B::acoustic_free_surface }));
  EXPECT_FALSE(is_valid_boundary_combo({ D::dim2, M::elastic_psv, P::isotropic,
                                         B::composite_stacey_dirichlet }));
}

TEST(IsValidBoundary, ElasticSH_NoneOrStacey) {
  EXPECT_TRUE(is_valid_boundary_combo(
      { D::dim2, M::elastic_sh, P::isotropic, B::none }));
  EXPECT_TRUE(is_valid_boundary_combo(
      { D::dim2, M::elastic_sh, P::isotropic, B::stacey }));
  EXPECT_FALSE(is_valid_boundary_combo(
      { D::dim2, M::elastic_sh, P::isotropic, B::acoustic_free_surface }));
  EXPECT_FALSE(is_valid_boundary_combo(
      { D::dim2, M::elastic_sh, P::isotropic, B::composite_stacey_dirichlet }));
}

TEST(IsValidBoundary, Poroelastic_NoneOrStacey) {
  EXPECT_TRUE(is_valid_boundary_combo(
      { D::dim2, M::poroelastic, P::isotropic, B::none }));
  EXPECT_TRUE(is_valid_boundary_combo(
      { D::dim2, M::poroelastic, P::isotropic, B::stacey }));
  EXPECT_FALSE(is_valid_boundary_combo(
      { D::dim2, M::poroelastic, P::isotropic, B::acoustic_free_surface }));
  EXPECT_FALSE(is_valid_boundary_combo({ D::dim2, M::poroelastic, P::isotropic,
                                         B::composite_stacey_dirichlet }));
}

// ── is_valid_full_combo ───────────────────────────────────────────────

TEST(IsValidFull, ValidCombos) {
  EXPECT_TRUE(is_valid_full_combo(
      { D::dim2, M::elastic_psv, P::isotropic, A::none, B::none }));
  EXPECT_TRUE(is_valid_full_combo({ D::dim2, M::elastic_psv, P::isotropic,
                                    A::constant_isotropic, B::stacey }));
  EXPECT_TRUE(is_valid_full_combo({ D::dim2, M::acoustic, P::isotropic, A::none,
                                    B::acoustic_free_surface }));
  EXPECT_TRUE(is_valid_full_combo(
      { D::dim3, M::elastic, P::isotropic, A::constant_isotropic, B::none }));
}

TEST(IsValidFull, InvalidByMaterial) {
  // Cosserat does not allow constant_isotropic attenuation
  EXPECT_FALSE(
      is_valid_full_combo({ D::dim2, M::elastic_psv_t, P::isotropic_cosserat,
                            A::constant_isotropic, B::none }));
}

TEST(IsValidFull, InvalidByBoundary) {
  // acoustic_free_surface not allowed on elastic_psv (dim2)
  EXPECT_FALSE(is_valid_full_combo({ D::dim2, M::elastic_psv, P::isotropic,
                                     A::none, B::acoustic_free_surface }));
  // composite_stacey_dirichlet not allowed on elastic (dim3)
  EXPECT_FALSE(is_valid_full_combo({ D::dim3, M::elastic, P::isotropic, A::none,
                                     B::composite_stacey_dirichlet }));
}

TEST(IsValidFull, Dim3StaceyAndAcousticFreeSurface) {
  // stacey is now valid on dim3 for both elastic and acoustic
  EXPECT_TRUE(is_valid_full_combo(
      { D::dim3, M::elastic, P::isotropic, A::none, B::stacey }));
  EXPECT_TRUE(is_valid_full_combo({ D::dim3, M::acoustic, P::isotropic, A::none,
                                    B::acoustic_free_surface }));
  EXPECT_TRUE(is_valid_full_combo(
      { D::dim3, M::acoustic, P::isotropic, A::none, B::stacey }));
  EXPECT_TRUE(is_valid_full_combo({ D::dim3, M::acoustic, P::isotropic, A::none,
                                    B::composite_stacey_dirichlet }));
}

TEST(IsValidFull, InvalidByMedium) {
  // elastic (dim3-only medium) used on dim2
  EXPECT_FALSE(is_valid_full_combo(
      { D::dim2, M::elastic, P::isotropic, A::none, B::none }));
}

// ── compile-time static_assert ────────────────────────────────────────

namespace {

static_assert(is_valid_medium_combo({ D::dim2, M::elastic_psv }));
static_assert(is_valid_medium_combo({ D::dim3, M::elastic }));
static_assert(!is_valid_medium_combo({ D::dim2, M::elastic }));
static_assert(!is_valid_medium_combo({ D::dim3, M::elastic_psv }));

static_assert(is_valid_property_combo({ D::dim2, M::elastic_psv,
                                        P::isotropic }));
static_assert(is_valid_property_combo({ D::dim2, M::elastic_psv,
                                        P::anisotropic }));
static_assert(!is_valid_property_combo({ D::dim2, M::elastic_psv,
                                         P::isotropic_cosserat }));
static_assert(is_valid_property_combo({ D::dim3, M::elastic, P::anisotropic }));

static_assert(is_valid_material_combo({ D::dim2, M::elastic_psv, P::isotropic,
                                        A::none }));
static_assert(is_valid_material_combo({ D::dim2, M::elastic_psv, P::isotropic,
                                        A::constant_isotropic }));
static_assert(!is_valid_material_combo({ D::dim2, M::elastic_psv_t,
                                         P::isotropic_cosserat,
                                         A::constant_isotropic }));

static_assert(is_valid_boundary_combo({ D::dim2, M::acoustic, P::isotropic,
                                        B::acoustic_free_surface }));
static_assert(is_valid_boundary_combo({ D::dim3, M::elastic, P::isotropic,
                                        B::stacey }));
static_assert(is_valid_boundary_combo({ D::dim3, M::acoustic, P::isotropic,
                                        B::acoustic_free_surface }));
static_assert(!is_valid_boundary_combo({ D::dim3, M::elastic, P::isotropic,
                                         B::composite_stacey_dirichlet }));

static_assert(is_valid_full_combo({ D::dim2, M::elastic_psv, P::isotropic,
                                    A::none, B::stacey }));
static_assert(!is_valid_full_combo({ D::dim2, M::elastic_psv, P::isotropic,
                                     A::none, B::acoustic_free_surface }));

} // namespace
