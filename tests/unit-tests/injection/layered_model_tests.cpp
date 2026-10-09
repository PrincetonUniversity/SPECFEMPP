#include "specfem/injection/fk/eval_points.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Valid ocean-over-solid model (acoustic top, elastic below, elastic
// half-space)
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidOceanOverSolid) {
  specfem::injection::fk::AcousticLayer water;
  water.density = 1000.0;
  water.p_velocity = 1500.0;
  water.thickness = 1000.0;

  specfem::injection::fk::ElasticIsotropicLayer crust;
  crust.density = 2700.0;
  crust.p_velocity = 6000.0;
  crust.s_velocity = 3500.0;
  crust.thickness = 20000.0;

  specfem::injection::fk::ElasticIsotropicLayer halfspace;
  halfspace.density = 3300.0;
  halfspace.p_velocity = 8000.0;
  halfspace.s_velocity = 4500.0;
  halfspace.thickness = 0.0;

  specfem::injection::fk::LayeredModel model({ water }, { crust, halfspace });

  EXPECT_EQ(model.number_of_fluid_layers(), 1);
  EXPECT_EQ(model.number_of_elastic_layers(), 2);
  EXPECT_EQ(model.total_number_of_layers(), 3);
  EXPECT_TRUE(model.has_fluid_layer());

  // shear_modulus(0) == rho*vs^2 of crust (elastic_index 0).
  EXPECT_NEAR(static_cast<double>(model.shear_modulus(0)),
              static_cast<double>(crust.density) *
                  static_cast<double>(crust.s_velocity) *
                  static_cast<double>(crust.s_velocity),
              1.0e5);

  // shear_modulus(1) == rho*vs^2 of halfspace (elastic_index 1).
  EXPECT_NEAR(static_cast<double>(model.shear_modulus(1)),
              static_cast<double>(halfspace.density) *
                  static_cast<double>(halfspace.s_velocity) *
                  static_cast<double>(halfspace.s_velocity),
              1.0e5);

  // Cumulative top depths in global order (fluid first, then elastic):
  //   index 0 (water)     → 0
  //   index 1 (crust)     → 1000
  //   index 2 (halfspace) → 21000
  const std::vector<type_real> &depths = model.layer_top_depths();
  ASSERT_EQ(static_cast<int>(depths.size()), 3);
  EXPECT_NEAR(static_cast<double>(depths[0]), 0.0, 1.0e-3);
  EXPECT_NEAR(static_cast<double>(depths[1]), 1000.0, 1.0e-3);
  EXPECT_NEAR(static_cast<double>(depths[2]), 21000.0, 1.0e-2);
}

// ---------------------------------------------------------------------------
// All-elastic model (no fluid layers)
// ---------------------------------------------------------------------------

TEST(LayeredModel, AllElasticModel) {
  specfem::injection::fk::ElasticIsotropicLayer layer;
  layer.density = 2700.0;
  layer.p_velocity = 6000.0;
  layer.s_velocity = 3500.0;
  layer.thickness = 10000.0;

  specfem::injection::fk::ElasticIsotropicLayer halfspace;
  halfspace.density = 3300.0;
  halfspace.p_velocity = 8000.0;
  halfspace.s_velocity = 4500.0;
  halfspace.thickness = 0.0;

  specfem::injection::fk::LayeredModel model({}, { layer, halfspace });

  EXPECT_EQ(model.number_of_fluid_layers(), 0);
  EXPECT_EQ(model.number_of_elastic_layers(), 2);
  EXPECT_EQ(model.total_number_of_layers(), 2);
  EXPECT_FALSE(model.has_fluid_layer());

  const std::vector<type_real> &depths = model.layer_top_depths();
  ASSERT_EQ(static_cast<int>(depths.size()), 2);
  EXPECT_NEAR(static_cast<double>(depths[0]), 0.0, 1.0e-3);
  EXPECT_NEAR(static_cast<double>(depths[1]), 10000.0, 1.0e-3);
}

// ---------------------------------------------------------------------------
// validate() throws: empty elastic block
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidateThrowsEmptyElastic) {
  EXPECT_THROW(specfem::injection::fk::LayeredModel({}, {}),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// validate() throws: fluid layer density <= 0
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidateThrowsFluidDensity) {
  specfem::injection::fk::AcousticLayer bad_fluid;
  bad_fluid.density = -100.0;
  bad_fluid.p_velocity = 1500.0;
  bad_fluid.thickness = 500.0;

  specfem::injection::fk::ElasticIsotropicLayer hs;
  hs.density = 2700.0;
  hs.p_velocity = 6000.0;
  hs.s_velocity = 3500.0;
  hs.thickness = 0.0;

  EXPECT_THROW(specfem::injection::fk::LayeredModel({ bad_fluid }, { hs }),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// validate() throws: fluid layer p_velocity <= 0
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidateThrowsFluidPVelocity) {
  specfem::injection::fk::AcousticLayer bad_fluid;
  bad_fluid.density = 1000.0;
  bad_fluid.p_velocity = 0.0;
  bad_fluid.thickness = 500.0;

  specfem::injection::fk::ElasticIsotropicLayer hs;
  hs.density = 2700.0;
  hs.p_velocity = 6000.0;
  hs.s_velocity = 3500.0;
  hs.thickness = 0.0;

  EXPECT_THROW(specfem::injection::fk::LayeredModel({ bad_fluid }, { hs }),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// validate() throws: elastic layer density <= 0
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidateThrowsElasticDensity) {
  specfem::injection::fk::ElasticIsotropicLayer bad;
  bad.density = -100.0;
  bad.p_velocity = 5000.0;
  bad.s_velocity = 3000.0;
  bad.thickness = 0.0;

  EXPECT_THROW(specfem::injection::fk::LayeredModel({}, { bad }),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// validate() throws: elastic layer p_velocity <= 0
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidateThrowsElasticPVelocity) {
  specfem::injection::fk::ElasticIsotropicLayer bad;
  bad.density = 2700.0;
  bad.p_velocity = 0.0;
  bad.s_velocity = 3000.0;
  bad.thickness = 0.0;

  EXPECT_THROW(specfem::injection::fk::LayeredModel({}, { bad }),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// validate() throws: elastic layer s_velocity <= 0
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidateThrowsElasticZeroShear) {
  specfem::injection::fk::ElasticIsotropicLayer bad;
  bad.density = 2700.0;
  bad.p_velocity = 6000.0;
  bad.s_velocity = 0.0;
  bad.thickness = 0.0;

  EXPECT_THROW(specfem::injection::fk::LayeredModel({}, { bad }),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// locate() tests — helpers in a named impl namespace (no anonymous namespaces)
// ---------------------------------------------------------------------------

namespace layered_model_tests_impl {

// All-elastic model: two layers + half-space.
//   layer 0: thickness 10000 m  → top   0, bottom 10000
//   layer 1: thickness 20000 m  → top 10000, bottom 30000
//   half-space (layer 2)        → top 30000
specfem::injection::fk::LayeredModel make_elastic_only_model() {
  specfem::injection::fk::ElasticIsotropicLayer layer0;
  layer0.density = 2700.0;
  layer0.p_velocity = 6000.0;
  layer0.s_velocity = 3500.0;
  layer0.thickness = 10000.0;

  specfem::injection::fk::ElasticIsotropicLayer layer1;
  layer1.density = 3000.0;
  layer1.p_velocity = 7000.0;
  layer1.s_velocity = 4000.0;
  layer1.thickness = 20000.0;

  specfem::injection::fk::ElasticIsotropicLayer hs;
  hs.density = 3300.0;
  hs.p_velocity = 8000.0;
  hs.s_velocity = 4500.0;
  hs.thickness = 0.0;

  return specfem::injection::fk::LayeredModel({}, { layer0, layer1, hs });
}

// A1 acoustic+elastic model:
//   fluid layer 0: thickness 2000 m  → top    0, bottom  2000
//   fluid layer 1: thickness 1000 m  → top 2000, bottom  3000
//   elastic layer 0 (crust): thickness 10000 m → top 3000, bottom 13000
//   half-space (elastic layer 1)     → top 13000
specfem::injection::fk::LayeredModel make_a1_model() {
  specfem::injection::fk::AcousticLayer fluid0;
  fluid0.density = 1000.0;
  fluid0.p_velocity = 1500.0;
  fluid0.thickness = 2000.0;

  specfem::injection::fk::AcousticLayer fluid1;
  fluid1.density = 1030.0;
  fluid1.p_velocity = 1600.0;
  fluid1.thickness = 1000.0;

  specfem::injection::fk::ElasticIsotropicLayer crust;
  crust.density = 2700.0;
  crust.p_velocity = 6000.0;
  crust.s_velocity = 3500.0;
  crust.thickness = 10000.0;

  specfem::injection::fk::ElasticIsotropicLayer hs;
  hs.density = 3300.0;
  hs.p_velocity = 8000.0;
  hs.s_velocity = 4500.0;
  hs.thickness = 0.0;

  return specfem::injection::fk::LayeredModel({ fluid0, fluid1 },
                                              { crust, hs });
}

} // namespace layered_model_tests_impl

// ---------------------------------------------------------------------------
// Elastic-only model: mid-layer locate
// ---------------------------------------------------------------------------

TEST(LayeredModel, LocateMidFirstElasticLayer) {
  specfem::injection::fk::LayeredModel model =
      layered_model_tests_impl::make_elastic_only_model();

  specfem::injection::fk::LayerLocation loc = model.locate(
      5000.0, specfem::injection::fk::point_medium::from_depth, 1.0);

  EXPECT_EQ(loc.medium, specfem::injection::fk::point_medium::elastic);
  EXPECT_EQ(loc.layer, 0);
  EXPECT_NEAR(loc.height, 5000.0, 1.0e-6); // bottom=10000, depth=5000
  EXPECT_FALSE(loc.in_halfspace);
  EXPECT_FALSE(loc.at_free_surface);
}

// ---------------------------------------------------------------------------
// Elastic-only model: interface tie → upper layer (layer 0 at height 0)
// ---------------------------------------------------------------------------

TEST(LayeredModel, LocateInterfaceTieUpperLayer) {
  specfem::injection::fk::LayeredModel model =
      layered_model_tests_impl::make_elastic_only_model();

  // depth == 10000: exactly at the layer 0/layer 1 interface.
  specfem::injection::fk::LayerLocation loc = model.locate(
      10000.0, specfem::injection::fk::point_medium::from_depth, 1.0);

  EXPECT_EQ(loc.medium, specfem::injection::fk::point_medium::elastic);
  EXPECT_EQ(loc.layer, 0); // upper layer, not layer 1
  EXPECT_NEAR(loc.height, 0.0, 1.0e-6);
  EXPECT_FALSE(loc.in_halfspace);
}

// ---------------------------------------------------------------------------
// Elastic-only model: half-space top and below
// ---------------------------------------------------------------------------

TEST(LayeredModel, LocateHalfspaceTop) {
  specfem::injection::fk::LayeredModel model =
      layered_model_tests_impl::make_elastic_only_model();

  // depth == 30000: exactly at layer 1 / half-space interface.
  specfem::injection::fk::LayerLocation loc_top = model.locate(
      30000.0, specfem::injection::fk::point_medium::from_depth, 1.0);
  EXPECT_EQ(loc_top.layer, 1); // layer 1 (upper), not the half-space
  EXPECT_NEAR(loc_top.height, 0.0, 1.0e-6);
  EXPECT_FALSE(loc_top.in_halfspace);

  // depth == 35000: clearly inside the half-space.
  specfem::injection::fk::LayerLocation loc_hs = model.locate(
      35000.0, specfem::injection::fk::point_medium::from_depth, 1.0);
  EXPECT_EQ(loc_hs.layer, 2);
  EXPECT_TRUE(loc_hs.in_halfspace);
  // hs_top = 30000; depth = 35000 → depth_below_halfspace_top = 30000-35000
  EXPECT_NEAR(loc_hs.depth_below_halfspace_top, -5000.0, 1.0e-6);
}

// ---------------------------------------------------------------------------
// A1 model: fluid-layer resolution
// ---------------------------------------------------------------------------

TEST(LayeredModel, LocateA1FluidLayers) {
  specfem::injection::fk::LayeredModel model =
      layered_model_tests_impl::make_a1_model();

  // depth == 0: at the free surface, acoustic layer 0.
  specfem::injection::fk::LayerLocation loc_surf =
      model.locate(0.0, specfem::injection::fk::point_medium::from_depth, 1.0);
  EXPECT_EQ(loc_surf.medium, specfem::injection::fk::point_medium::acoustic);
  EXPECT_EQ(loc_surf.layer, 0);
  EXPECT_TRUE(loc_surf.at_free_surface);

  // depth == 1000: inside fluid layer 0 (bottom 2000), height = 1000.
  specfem::injection::fk::LayerLocation loc_mid = model.locate(
      1000.0, specfem::injection::fk::point_medium::from_depth, 1.0);
  EXPECT_EQ(loc_mid.medium, specfem::injection::fk::point_medium::acoustic);
  EXPECT_EQ(loc_mid.layer, 0);
  EXPECT_NEAR(loc_mid.height, 1000.0, 1.0e-6);
  EXPECT_FALSE(loc_mid.at_free_surface);
}

// ---------------------------------------------------------------------------
// A1 model: fluid-solid interface (depth == 3000) resolves as elastic by
// default (from_depth), as acoustic with forced acoustic, as elastic with
// forced elastic.
// ---------------------------------------------------------------------------

TEST(LayeredModel, LocateA1FluidSolidInterface) {
  specfem::injection::fk::LayeredModel model =
      layered_model_tests_impl::make_a1_model();
  const double tol = 1.0;

  // from_depth at the interface → elastic (crust, layer 0, height 10000).
  specfem::injection::fk::LayerLocation loc_fd = model.locate(
      3000.0, specfem::injection::fk::point_medium::from_depth, tol);
  EXPECT_EQ(loc_fd.medium, specfem::injection::fk::point_medium::elastic);
  EXPECT_EQ(loc_fd.layer, 0);
  EXPECT_NEAR(loc_fd.height, 10000.0, 1.0e-6);

  // forced acoustic at the interface → acoustic (fluid layer 1, height 0).
  specfem::injection::fk::LayerLocation loc_ac =
      model.locate(3000.0, specfem::injection::fk::point_medium::acoustic, tol);
  EXPECT_EQ(loc_ac.medium, specfem::injection::fk::point_medium::acoustic);
  EXPECT_EQ(loc_ac.layer, 1);
  EXPECT_NEAR(loc_ac.height, 0.0, 1.0e-6);

  // forced elastic at the interface → elastic (crust, layer 0, height 10000).
  specfem::injection::fk::LayerLocation loc_el =
      model.locate(3000.0, specfem::injection::fk::point_medium::elastic, tol);
  EXPECT_EQ(loc_el.medium, specfem::injection::fk::point_medium::elastic);
  EXPECT_EQ(loc_el.layer, 0);
  EXPECT_NEAR(loc_el.height, 10000.0, 1.0e-6);
}

// ---------------------------------------------------------------------------
// locate() throw cases
// ---------------------------------------------------------------------------

TEST(LayeredModel, LocateThrowsAboveSurface) {
  specfem::injection::fk::LayeredModel model =
      layered_model_tests_impl::make_elastic_only_model();

  EXPECT_THROW(model.locate(-100.0,
                            specfem::injection::fk::point_medium::from_depth,
                            1.0),
               std::runtime_error);
}

TEST(LayeredModel, LocateThrowsAcousticBelowFluidColumn) {
  // depth == 8000 is deep in the elastic half of the A1 model.
  specfem::injection::fk::LayeredModel model =
      layered_model_tests_impl::make_a1_model();

  EXPECT_THROW(
      model.locate(8000.0, specfem::injection::fk::point_medium::acoustic, 1.0),
      std::runtime_error);
}

TEST(LayeredModel, LocateThrowsAcousticOnElasticOnlyModel) {
  specfem::injection::fk::LayeredModel model =
      layered_model_tests_impl::make_elastic_only_model();

  EXPECT_THROW(
      model.locate(1000.0, specfem::injection::fk::point_medium::acoustic, 1.0),
      std::runtime_error);
}

// ---------------------------------------------------------------------------
// fluid_thickness() and minimum_velocity()
// ---------------------------------------------------------------------------

TEST(LayeredModel, FluidThicknessAndMinVelocity) {
  // Elastic-only: fluid_thickness == 0; minimum_velocity == min(vs) == 3500.
  specfem::injection::fk::LayeredModel elastic_model =
      layered_model_tests_impl::make_elastic_only_model();
  EXPECT_NEAR(elastic_model.fluid_thickness(), 0.0, 1.0e-9);
  EXPECT_NEAR(elastic_model.minimum_velocity(), 3500.0, 1.0e-6);

  // A1 model: fluid_thickness == 2000+1000 == 3000;
  // minimum_velocity == min(vp_fluid0=1500, vp_fluid1=1600,
  //                         vs_crust=3500, vs_hs=4500) == 1500.
  specfem::injection::fk::LayeredModel a1_model =
      layered_model_tests_impl::make_a1_model();
  EXPECT_NEAR(a1_model.fluid_thickness(), 3000.0, 1.0e-6);
  EXPECT_NEAR(a1_model.minimum_velocity(), 1500.0, 1.0e-6);
}

// ---------------------------------------------------------------------------
// Free-surface flag is set ONLY at the top of the fluid column (fluid layer 0),
// never at a deeper fluid layer whose in-layer height happens to equal the top
// layer's thickness. (The old driver's |height - H0| test zeroed the latter.)
// ---------------------------------------------------------------------------

TEST(LayeredModel, FreeSurfaceOnlyAtFluidTop) {
  // Fluid F0 (H=1000) over fluid F1 (H=3000) over crust + half-space. A point
  // at z=3000 sits inside F1 with in-layer height 1000 == H0.
  specfem::injection::fk::AcousticLayer f0, f1;
  f0.density = 1020;
  f0.p_velocity = 1500;
  f0.thickness = 1000;
  f1.density = 1020;
  f1.p_velocity = 1520;
  f1.thickness = 3000;
  specfem::injection::fk::ElasticIsotropicLayer crust, hs;
  crust.density = 2800;
  crust.p_velocity = 6000;
  crust.s_velocity = 3500;
  crust.thickness = 10000;
  hs.density = 3300;
  hs.p_velocity = 8100;
  hs.s_velocity = 4500;
  hs.thickness = 0;
  specfem::injection::fk::LayeredModel model({ f0, f1 }, { crust, hs });
  const double tol = 1.0e-3;

  // True free surface: top of fluid layer 0.
  const auto top =
      model.locate(0.0, specfem::injection::fk::point_medium::from_depth, tol);
  EXPECT_EQ(top.medium, specfem::injection::fk::point_medium::acoustic);
  EXPECT_EQ(top.layer, 0);
  EXPECT_TRUE(top.at_free_surface);

  // Deep point in fluid layer 1 (height-in-F1 == H0 == 1000): NOT free surface.
  const auto deep = model.locate(
      3000.0, specfem::injection::fk::point_medium::from_depth, tol);
  EXPECT_EQ(deep.medium, specfem::injection::fk::point_medium::acoustic);
  EXPECT_EQ(deep.layer, 1);
  EXPECT_FALSE(deep.at_free_surface);
}
