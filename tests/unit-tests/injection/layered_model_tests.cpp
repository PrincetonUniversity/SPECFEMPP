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
