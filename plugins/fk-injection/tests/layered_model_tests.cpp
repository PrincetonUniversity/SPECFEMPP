#include "fk_injection/layered_model.hpp"
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace layered_model_tests_impl {

// The what() of the std::runtime_error thrown by the callable; empty if none.
template <typename Callable> std::string rejection_reason(Callable &&callable) {
  try {
    callable();
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return "";
}

} // namespace layered_model_tests_impl

// ---------------------------------------------------------------------------
// Valid ocean-over-solid model (acoustic top, elastic below, elastic
// half-space)
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidOceanOverSolid) {
  fk_injection::AcousticLayer water;
  water.density = 1000.0;
  water.p_velocity = 1500.0;
  water.thickness = 1000.0;

  fk_injection::ElasticIsotropicLayer crust;
  crust.density = 2700.0;
  crust.p_velocity = 6000.0;
  crust.s_velocity = 3500.0;
  crust.thickness = 20000.0;

  fk_injection::ElasticIsotropicLayer halfspace;
  halfspace.density = 3300.0;
  halfspace.p_velocity = 8000.0;
  halfspace.s_velocity = 4500.0;
  halfspace.thickness = 0.0;

  fk_injection::LayeredModel model({ water, crust, halfspace });

  ASSERT_EQ(model.number_of_layers(), 3);
  EXPECT_TRUE(
      std::holds_alternative<fk_injection::AcousticLayer>(model.layers()[0]));
  EXPECT_TRUE(std::holds_alternative<fk_injection::ElasticIsotropicLayer>(
      model.layers()[1]));

  // Water is 1000 m thick, so the half-space top is at 1000 + 20000 = 21000.
  EXPECT_NEAR(model.fluid_thickness(), 1000.0, 1.0e-3);
  EXPECT_NEAR(model.locate(0.0, specfem::plugin::Medium::acoustic, 1.0)
                  .depth_below_halfspace_top,
              21000.0, 1.0e-2);

  // Layers are indexed top-down across both media.
  const fk_injection::LayerLocation crust_location =
      model.locate(5000.0, specfem::plugin::Medium::elastic, 1.0);
  EXPECT_EQ(crust_location.layer, 1);
  EXPECT_FALSE(crust_location.in_halfspace);
  EXPECT_TRUE(model.locate(30000.0, specfem::plugin::Medium::elastic, 1.0)
                  .in_halfspace);
  EXPECT_EQ(model.halfspace().density, halfspace.density);
}

// ---------------------------------------------------------------------------
// All-elastic model (no fluid layers)
// ---------------------------------------------------------------------------

TEST(LayeredModel, AllElasticModel) {
  fk_injection::ElasticIsotropicLayer layer;
  layer.density = 2700.0;
  layer.p_velocity = 6000.0;
  layer.s_velocity = 3500.0;
  layer.thickness = 10000.0;

  fk_injection::ElasticIsotropicLayer halfspace;
  halfspace.density = 3300.0;
  halfspace.p_velocity = 8000.0;
  halfspace.s_velocity = 4500.0;
  halfspace.thickness = 0.0;

  fk_injection::LayeredModel model({ layer, halfspace });

  ASSERT_EQ(model.number_of_layers(), 2);
  EXPECT_TRUE(std::holds_alternative<fk_injection::ElasticIsotropicLayer>(
      model.layers()[0]));

  EXPECT_NEAR(model.fluid_thickness(), 0.0, 1.0e-9);
  EXPECT_NEAR(model.locate(0.0, specfem::plugin::Medium::elastic, 1.0)
                  .depth_below_halfspace_top,
              10000.0, 1.0e-3);
}

// ---------------------------------------------------------------------------
// validate() throws: empty elastic block
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidateThrowsEmptyElastic) {
  EXPECT_THROW(fk_injection::LayeredModel(std::vector<fk_injection::Layer>{}),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// Fluid-only model: the last layer has to be an elastic half-space.
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidateThrowsFluidOnly) {
  fk_injection::AcousticLayer fluid;
  fluid.density = static_cast<fk_injection::type_real>(1000);
  fluid.p_velocity = static_cast<fk_injection::type_real>(1500);
  fluid.thickness = static_cast<fk_injection::type_real>(2000);
  EXPECT_THROW(fk_injection::LayeredModel({ fluid }), std::runtime_error);
}

// ---------------------------------------------------------------------------
// validate() throws: fluid layer density <= 0
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidateThrowsFluidDensity) {
  fk_injection::AcousticLayer bad_fluid;
  bad_fluid.density = -100.0;
  bad_fluid.p_velocity = 1500.0;
  bad_fluid.thickness = 500.0;

  fk_injection::ElasticIsotropicLayer hs;
  hs.density = 2700.0;
  hs.p_velocity = 6000.0;
  hs.s_velocity = 3500.0;
  hs.thickness = 0.0;

  EXPECT_THROW(fk_injection::LayeredModel({ bad_fluid, hs }),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// validate() throws: fluid layer p_velocity <= 0
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidateThrowsFluidPVelocity) {
  fk_injection::AcousticLayer bad_fluid;
  bad_fluid.density = 1000.0;
  bad_fluid.p_velocity = 0.0;
  bad_fluid.thickness = 500.0;

  fk_injection::ElasticIsotropicLayer hs;
  hs.density = 2700.0;
  hs.p_velocity = 6000.0;
  hs.s_velocity = 3500.0;
  hs.thickness = 0.0;

  EXPECT_THROW(fk_injection::LayeredModel({ bad_fluid, hs }),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// validate() throws: elastic layer density <= 0
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidateThrowsElasticDensity) {
  fk_injection::ElasticIsotropicLayer bad;
  bad.density = -100.0;
  bad.p_velocity = 5000.0;
  bad.s_velocity = 3000.0;
  bad.thickness = 0.0;

  EXPECT_THROW(fk_injection::LayeredModel({ bad }), std::runtime_error);
}

// ---------------------------------------------------------------------------
// validate() throws: elastic layer p_velocity <= 0
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidateThrowsElasticPVelocity) {
  fk_injection::ElasticIsotropicLayer bad;
  bad.density = 2700.0;
  bad.p_velocity = 0.0;
  bad.s_velocity = 3000.0;
  bad.thickness = 0.0;

  EXPECT_THROW(fk_injection::LayeredModel({ bad }), std::runtime_error);
}

// ---------------------------------------------------------------------------
// validate() throws: elastic layer s_velocity <= 0
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidateThrowsElasticZeroShear) {
  fk_injection::ElasticIsotropicLayer bad;
  bad.density = 2700.0;
  bad.p_velocity = 6000.0;
  bad.s_velocity = 0.0;
  bad.thickness = 0.0;

  EXPECT_THROW(fk_injection::LayeredModel({ bad }), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Validation rejects a layer thickness that is not positive, except on the
// half-space, whose thickness is ignored.
// ---------------------------------------------------------------------------

TEST(LayeredModel, ValidateThrowsFluidThickness) {
  fk_injection::AcousticLayer bad_fluid;
  bad_fluid.density = 1000.0;
  bad_fluid.p_velocity = 1500.0;
  bad_fluid.thickness = 0.0;

  fk_injection::ElasticIsotropicLayer hs;
  hs.density = 2700.0;
  hs.p_velocity = 6000.0;
  hs.s_velocity = 3500.0;

  const std::string reason = layered_model_tests_impl::rejection_reason(
      [&] { fk_injection::LayeredModel({ bad_fluid, hs }); });
  EXPECT_NE(reason.find("layer 0 (acoustic) has non-positive thickness"),
            std::string::npos)
      << reason;
}

TEST(LayeredModel, ValidateThrowsElasticThicknessAboveHalfspace) {
  fk_injection::ElasticIsotropicLayer bad;
  bad.density = 2700.0;
  bad.p_velocity = 6000.0;
  bad.s_velocity = 3500.0;
  bad.thickness = -10.0;

  fk_injection::ElasticIsotropicLayer hs;
  hs.density = 3300.0;
  hs.p_velocity = 8000.0;
  hs.s_velocity = 4500.0;

  const std::string reason = layered_model_tests_impl::rejection_reason(
      [&] { fk_injection::LayeredModel({ bad, hs }); });
  EXPECT_NE(reason.find("layer 0 (elastic) has non-positive thickness"),
            std::string::npos)
      << reason;
}

TEST(LayeredModel, ValidateThrowsFluidBelowSolid) {
  fk_injection::ElasticIsotropicLayer crust;
  crust.density = 2700.0;
  crust.p_velocity = 6000.0;
  crust.s_velocity = 3500.0;
  crust.thickness = 1000.0;

  fk_injection::AcousticLayer fluid;
  fluid.density = 1000.0;
  fluid.p_velocity = 1500.0;
  fluid.thickness = 500.0;

  fk_injection::ElasticIsotropicLayer hs;
  hs.density = 3300.0;
  hs.p_velocity = 8000.0;
  hs.s_velocity = 4500.0;

  const std::string reason = layered_model_tests_impl::rejection_reason(
      [&] { fk_injection::LayeredModel({ crust, fluid, hs }); });
  EXPECT_NE(reason.find("acoustic layer 1 lies below an elastic layer"),
            std::string::npos)
      << reason;
}

TEST(LayeredModel, AcceptsAnyHalfspaceThickness) {
  fk_injection::ElasticIsotropicLayer hs;
  hs.density = 3300.0;
  hs.p_velocity = 8000.0;
  hs.s_velocity = 4500.0;
  hs.thickness = 0.0;

  EXPECT_NO_THROW(fk_injection::LayeredModel({ hs }));
}

// ---------------------------------------------------------------------------
// locate() tests — helpers in a named impl namespace (no anonymous namespaces)
// ---------------------------------------------------------------------------

namespace layered_model_tests_impl {

// All-elastic model: two layers + half-space.
//   layer 0: thickness 10000 m  → top   0, bottom 10000
//   layer 1: thickness 20000 m  → top 10000, bottom 30000
//   half-space (layer 2)        → top 30000
fk_injection::LayeredModel make_elastic_only_model() {
  fk_injection::ElasticIsotropicLayer layer0;
  layer0.density = 2700.0;
  layer0.p_velocity = 6000.0;
  layer0.s_velocity = 3500.0;
  layer0.thickness = 10000.0;

  fk_injection::ElasticIsotropicLayer layer1;
  layer1.density = 3000.0;
  layer1.p_velocity = 7000.0;
  layer1.s_velocity = 4000.0;
  layer1.thickness = 20000.0;

  fk_injection::ElasticIsotropicLayer hs;
  hs.density = 3300.0;
  hs.p_velocity = 8000.0;
  hs.s_velocity = 4500.0;
  hs.thickness = 0.0;

  return fk_injection::LayeredModel({ layer0, layer1, hs });
}

// A1 acoustic+elastic model:
//   fluid layer 0: thickness 2000 m  → top    0, bottom  2000
//   fluid layer 1: thickness 1000 m  → top 2000, bottom  3000
//   elastic layer 0 (crust): thickness 10000 m → top 3000, bottom 13000
//   half-space (elastic layer 1)     → top 13000
fk_injection::LayeredModel make_a1_model() {
  fk_injection::AcousticLayer fluid0;
  fluid0.density = 1000.0;
  fluid0.p_velocity = 1500.0;
  fluid0.thickness = 2000.0;

  fk_injection::AcousticLayer fluid1;
  fluid1.density = 1030.0;
  fluid1.p_velocity = 1600.0;
  fluid1.thickness = 1000.0;

  fk_injection::ElasticIsotropicLayer crust;
  crust.density = 2700.0;
  crust.p_velocity = 6000.0;
  crust.s_velocity = 3500.0;
  crust.thickness = 10000.0;

  fk_injection::ElasticIsotropicLayer hs;
  hs.density = 3300.0;
  hs.p_velocity = 8000.0;
  hs.s_velocity = 4500.0;
  hs.thickness = 0.0;

  return fk_injection::LayeredModel({ fluid0, fluid1, crust, hs });
}

} // namespace layered_model_tests_impl

// ---------------------------------------------------------------------------
// Elastic-only model: mid-layer locate
// ---------------------------------------------------------------------------

TEST(LayeredModel, LocateMidFirstElasticLayer) {
  fk_injection::LayeredModel model =
      layered_model_tests_impl::make_elastic_only_model();

  fk_injection::LayerLocation loc =
      model.locate(5000.0, specfem::plugin::Medium::elastic, 1.0);

  EXPECT_EQ(loc.medium, specfem::plugin::Medium::elastic);
  EXPECT_EQ(loc.layer, 0);
  EXPECT_NEAR(loc.height, 5000.0, 1.0e-6); // bottom=10000, depth=5000
  EXPECT_FALSE(loc.in_halfspace);
  EXPECT_FALSE(loc.at_free_surface);
}

// ---------------------------------------------------------------------------
// Elastic-only model: interface tie → upper layer (layer 0 at height 0)
// ---------------------------------------------------------------------------

TEST(LayeredModel, LocateInterfaceTieUpperLayer) {
  fk_injection::LayeredModel model =
      layered_model_tests_impl::make_elastic_only_model();

  // depth == 10000: exactly at the layer 0/layer 1 interface.
  fk_injection::LayerLocation loc =
      model.locate(10000.0, specfem::plugin::Medium::elastic, 1.0);

  EXPECT_EQ(loc.medium, specfem::plugin::Medium::elastic);
  EXPECT_EQ(loc.layer, 0); // upper layer, not layer 1
  EXPECT_NEAR(loc.height, 0.0, 1.0e-6);
  EXPECT_FALSE(loc.in_halfspace);
}

// ---------------------------------------------------------------------------
// Elastic-only model: half-space top and below
// ---------------------------------------------------------------------------

TEST(LayeredModel, LocateHalfspaceTop) {
  fk_injection::LayeredModel model =
      layered_model_tests_impl::make_elastic_only_model();

  // depth == 30000: exactly at layer 1 / half-space interface.
  fk_injection::LayerLocation loc_top =
      model.locate(30000.0, specfem::plugin::Medium::elastic, 1.0);
  EXPECT_EQ(loc_top.layer, 1); // layer 1 (upper), not the half-space
  EXPECT_NEAR(loc_top.height, 0.0, 1.0e-6);
  EXPECT_FALSE(loc_top.in_halfspace);

  // depth == 35000: clearly inside the half-space.
  fk_injection::LayerLocation loc_hs =
      model.locate(35000.0, specfem::plugin::Medium::elastic, 1.0);
  EXPECT_EQ(loc_hs.layer, 2);
  EXPECT_TRUE(loc_hs.in_halfspace);
  // hs_top = 30000; depth = 35000 → depth_below_halfspace_top = 30000-35000
  EXPECT_NEAR(loc_hs.depth_below_halfspace_top, -5000.0, 1.0e-6);
}

// ---------------------------------------------------------------------------
// A1 model: fluid-layer resolution
// ---------------------------------------------------------------------------

TEST(LayeredModel, LocateA1FluidLayers) {
  fk_injection::LayeredModel model = layered_model_tests_impl::make_a1_model();

  // depth == 0: at the free surface, acoustic layer 0.
  fk_injection::LayerLocation loc_surf =
      model.locate(0.0, specfem::plugin::Medium::acoustic, 1.0);
  EXPECT_EQ(loc_surf.medium, specfem::plugin::Medium::acoustic);
  EXPECT_EQ(loc_surf.layer, 0);
  EXPECT_TRUE(loc_surf.at_free_surface);

  // depth == 1000: inside fluid layer 0 (bottom 2000), height = 1000.
  fk_injection::LayerLocation loc_mid =
      model.locate(1000.0, specfem::plugin::Medium::acoustic, 1.0);
  EXPECT_EQ(loc_mid.medium, specfem::plugin::Medium::acoustic);
  EXPECT_EQ(loc_mid.layer, 0);
  EXPECT_NEAR(loc_mid.height, 1000.0, 1.0e-6);
  EXPECT_FALSE(loc_mid.at_free_surface);
}

// ---------------------------------------------------------------------------
// A1 model: the fluid-solid interface (depth == 3000) resolves in either
// medium.
// ---------------------------------------------------------------------------

TEST(LayeredModel, LocateA1FluidSolidInterface) {
  fk_injection::LayeredModel model = layered_model_tests_impl::make_a1_model();
  const double tol = 1.0;

  // Acoustic at the interface → acoustic (fluid layer 1, height 0).
  fk_injection::LayerLocation loc_ac =
      model.locate(3000.0, specfem::plugin::Medium::acoustic, tol);
  EXPECT_EQ(loc_ac.medium, specfem::plugin::Medium::acoustic);
  EXPECT_EQ(loc_ac.layer, 1);
  EXPECT_NEAR(loc_ac.height, 0.0, 1.0e-6);

  // Elastic at the interface → elastic (crust, layer 2, height 10000).
  fk_injection::LayerLocation loc_el =
      model.locate(3000.0, specfem::plugin::Medium::elastic, tol);
  EXPECT_EQ(loc_el.medium, specfem::plugin::Medium::elastic);
  EXPECT_EQ(loc_el.layer, 2);
  EXPECT_NEAR(loc_el.height, 10000.0, 1.0e-6);
}

// ---------------------------------------------------------------------------
// locate() throw cases
// ---------------------------------------------------------------------------

TEST(LayeredModel, LocateThrowsAboveSurface) {
  fk_injection::LayeredModel model =
      layered_model_tests_impl::make_elastic_only_model();

  EXPECT_THROW(model.locate(-100.0, specfem::plugin::Medium::elastic, 1.0),
               std::runtime_error);
}

TEST(LayeredModel, LocateThrowsAcousticBelowFluidColumn) {
  // depth == 8000 is deep in the elastic half of the A1 model.
  fk_injection::LayeredModel model = layered_model_tests_impl::make_a1_model();

  EXPECT_THROW(model.locate(8000.0, specfem::plugin::Medium::acoustic, 1.0),
               std::runtime_error);
}

TEST(LayeredModel, LocateThrowsAcousticOnElasticOnlyModel) {
  fk_injection::LayeredModel model =
      layered_model_tests_impl::make_elastic_only_model();

  EXPECT_THROW(model.locate(1000.0, specfem::plugin::Medium::acoustic, 1.0),
               std::runtime_error);
}

TEST(LayeredModel, LocateRejectsElasticSpin) {
  fk_injection::LayeredModel model =
      layered_model_tests_impl::make_elastic_only_model();

  const std::string reason = layered_model_tests_impl::rejection_reason([&] {
    model.locate(1000.0, specfem::plugin::Medium::elastic_spin, 1.0);
  });
  EXPECT_NE(reason.find("acoustic and elastic media only"), std::string::npos)
      << reason;
}

TEST(LayeredModel, LocateRejectsNegativeTolerance) {
  fk_injection::LayeredModel model =
      layered_model_tests_impl::make_elastic_only_model();

  const std::string reason = layered_model_tests_impl::rejection_reason(
      [&] { model.locate(1000.0, specfem::plugin::Medium::elastic, -1.0); });
  EXPECT_NE(reason.find("tolerance must be non-negative"), std::string::npos)
      << reason;
}

// ---------------------------------------------------------------------------
// fluid_thickness()
// ---------------------------------------------------------------------------

TEST(LayeredModel, FluidThickness) {
  fk_injection::LayeredModel elastic_model =
      layered_model_tests_impl::make_elastic_only_model();
  EXPECT_NEAR(elastic_model.fluid_thickness(), 0.0, 1.0e-9);

  // A1 model: fluid_thickness == 2000+1000 == 3000.
  fk_injection::LayeredModel a1_model =
      layered_model_tests_impl::make_a1_model();
  EXPECT_NEAR(a1_model.fluid_thickness(), 3000.0, 1.0e-6);
}

// ---------------------------------------------------------------------------
// A point at the fluid-solid interface finds the last fluid layer at zero
// tolerance. These thicknesses accumulate to different values in float and in
// double, which must not send the search past every layer.
// ---------------------------------------------------------------------------

TEST(LayeredModel, LocateInterfaceFindsLastFluidLayerAtZeroTolerance) {
  std::vector<fk_injection::AcousticLayer> fluid(3);
  fluid[0].thickness = 11863.953125;
  fluid[1].thickness = 2134.3206;
  fluid[2].thickness = 6416.8496;
  for (fk_injection::AcousticLayer &layer : fluid) {
    layer.density = 1000.0;
    layer.p_velocity = 1500.0;
  }

  fk_injection::ElasticIsotropicLayer hs;
  hs.density = 3300.0;
  hs.p_velocity = 8000.0;
  hs.s_velocity = 4500.0;

  std::vector<fk_injection::Layer> layers(fluid.begin(), fluid.end());
  layers.push_back(hs);
  const fk_injection::LayeredModel model(std::move(layers));
  const fk_injection::LayerLocation location = model.locate(
      model.fluid_thickness(), specfem::plugin::Medium::acoustic, 0.0);

  EXPECT_EQ(location.layer, 2);
  EXPECT_NEAR(location.height, 0.0, 1.0e-9);
}

// ---------------------------------------------------------------------------
// Free-surface flag is set ONLY at the top of the fluid column (fluid layer 0),
// never at a deeper fluid layer whose in-layer height happens to equal the top
// layer's thickness. (The old driver's |height - H0| test zeroed the latter.)
// ---------------------------------------------------------------------------

TEST(LayeredModel, FreeSurfaceOnlyAtFluidTop) {
  // Fluid F0 (H=1000) over fluid F1 (H=3000) over crust + half-space. A point
  // at z=3000 sits inside F1 with in-layer height 1000 == H0.
  fk_injection::AcousticLayer f0, f1;
  f0.density = 1020;
  f0.p_velocity = 1500;
  f0.thickness = 1000;
  f1.density = 1020;
  f1.p_velocity = 1520;
  f1.thickness = 3000;
  fk_injection::ElasticIsotropicLayer crust, hs;
  crust.density = 2800;
  crust.p_velocity = 6000;
  crust.s_velocity = 3500;
  crust.thickness = 10000;
  hs.density = 3300;
  hs.p_velocity = 8100;
  hs.s_velocity = 4500;
  hs.thickness = 0;
  fk_injection::LayeredModel model({ f0, f1, crust, hs });
  const double tol = 1.0e-3;

  // True free surface: top of fluid layer 0.
  const auto top = model.locate(0.0, specfem::plugin::Medium::acoustic, tol);
  EXPECT_EQ(top.medium, specfem::plugin::Medium::acoustic);
  EXPECT_EQ(top.layer, 0);
  EXPECT_TRUE(top.at_free_surface);

  // Deep point in fluid layer 1 (height-in-F1 == H0 == 1000): NOT free surface.
  const auto deep =
      model.locate(3000.0, specfem::plugin::Medium::acoustic, tol);
  EXPECT_EQ(deep.medium, specfem::plugin::Medium::acoustic);
  EXPECT_EQ(deep.layer, 1);
  EXPECT_FALSE(deep.at_free_surface);
}

TEST(LayeredModel, HalfspaceTopDepthSumsFiniteLayers) {
  fk_injection::AcousticLayer fluid;
  fluid.density = 1000;
  fluid.p_velocity = 1500;
  fluid.thickness = 2000;

  fk_injection::ElasticIsotropicLayer crust;
  crust.density = 2800;
  crust.p_velocity = 6000;
  crust.s_velocity = 3500;
  crust.thickness = 10000;

  fk_injection::ElasticIsotropicLayer halfspace;
  halfspace.density = 3300;
  halfspace.p_velocity = 8100;
  halfspace.s_velocity = 4500;

  EXPECT_NEAR(fk_injection::LayeredModel({ halfspace }).halfspace_top_depth(),
              0.0, 1.0e-9);
  EXPECT_NEAR(fk_injection::LayeredModel({ fluid, crust, halfspace })
                  .halfspace_top_depth(),
              12000.0, 1.0e-9);
}
