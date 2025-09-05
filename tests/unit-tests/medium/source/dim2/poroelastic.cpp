#include "enumerations/interface.hpp"
#include "enumerations/wavefield.hpp"
#include "medium/compute_source.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

TEST(Source, PoroelasticIsotropic2D) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::poroelastic;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using PointPropertiesType =
      specfem::point::properties<dimension, medium_tag, property_tag, false>;
  using PointSourceType =
      specfem::point::source<dimension, medium_tag,
                             specfem::wavefield::simulation_field::forward>;
  using PointAccelerationType =
      specfem::point::acceleration<dimension, medium_tag, false>;

  const type_real phi = 0.3;
  const type_real rho_s = 2500.0;
  const type_real rho_f = 1000.0;
  const type_real tortuosity = 1.2;
  const type_real mu_G = 10e9;
  const type_real H_Biot = 36e9;
  const type_real C_Biot = 2.2e9;
  const type_real M_Biot = 2.2e9;
  const type_real permxx = 1e-12;
  const type_real permxz = 0.0;
  const type_real permzz = 1e-12;
  const type_real eta_f = 1e-3;

  const PointPropertiesType properties(phi, rho_s, rho_f, tortuosity, mu_G,
                                       H_Biot, C_Biot, M_Biot, permxx, permxz,
                                       permzz, eta_f);

  PointSourceType point_source;
  point_source.stf(0) = 1.0;
  point_source.stf(1) = 2.0;
  point_source.stf(2) = 3.0;
  point_source.stf(3) = 4.0;
  point_source.lagrange_interpolant(0) = 1.5;
  point_source.lagrange_interpolant(1) = 2.5;
  point_source.lagrange_interpolant(2) = 3.5;
  point_source.lagrange_interpolant(3) = 4.5;

  const PointAccelerationType acceleration =
      specfem::medium::compute_source_contribution(point_source, properties);

  PointAccelerationType expected_acceleration;
  // Values from actual implementation - calculated using the property methods
  expected_acceleration(0) = 1.125;   // 1.0 * 1.5 * (1.0 - 0.3/1.2) = 1.125
  expected_acceleration(1) = 3.75;    // 2.0 * 2.5 * (1.0 - 0.3/1.2) = 3.75
  expected_acceleration(2) = 5.37805; // 3.0 * 3.5 * (1.0 - rho_f/rho_bar)
  expected_acceleration(3) = 9.21951; // 4.0 * 4.5 * (1.0 - rho_f/rho_bar)

  std::ostringstream message;
  message << "Source acceleration is not equal to expected value: \n"
          << "Computed: " << acceleration.print() << "\n"
          << "Expected: " << expected_acceleration.print() << "\n";

  EXPECT_TRUE(acceleration == expected_acceleration) << message.str();
}

TEST(Source, PoroelasticIsotropic2D_ZeroSource) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::poroelastic;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using PointPropertiesType =
      specfem::point::properties<dimension, medium_tag, property_tag, false>;
  using PointSourceType =
      specfem::point::source<dimension, medium_tag,
                             specfem::wavefield::simulation_field::forward>;
  using PointAccelerationType =
      specfem::point::acceleration<dimension, medium_tag, false>;

  const type_real phi = 0.3;
  const type_real rho_s = 2500.0;
  const type_real rho_f = 1000.0;
  const type_real tortuosity = 1.2;
  const type_real mu_G = 10e9;
  const type_real H_Biot = 36e9;
  const type_real C_Biot = 2.2e9;
  const type_real M_Biot = 2.2e9;
  const type_real permxx = 1e-12;
  const type_real permxz = 0.0;
  const type_real permzz = 1e-12;
  const type_real eta_f = 1e-3;

  const PointPropertiesType properties(phi, rho_s, rho_f, tortuosity, mu_G,
                                       H_Biot, C_Biot, M_Biot, permxx, permxz,
                                       permzz, eta_f);

  PointSourceType point_source;
  point_source.stf(0) = 0.0;
  point_source.stf(1) = 0.0;
  point_source.stf(2) = 0.0;
  point_source.stf(3) = 0.0;
  point_source.lagrange_interpolant(0) = 1.5;
  point_source.lagrange_interpolant(1) = 2.5;
  point_source.lagrange_interpolant(2) = 3.5;
  point_source.lagrange_interpolant(3) = 4.5;

  const PointAccelerationType acceleration =
      specfem::medium::compute_source_contribution(point_source, properties);

  PointAccelerationType expected_acceleration;
  expected_acceleration(0) = 0.0;
  expected_acceleration(1) = 0.0;
  expected_acceleration(2) = 0.0;
  expected_acceleration(3) = 0.0;

  std::ostringstream message;
  message << "Source acceleration should be zero for zero STF: \n"
          << "Computed: " << acceleration.print() << "\n"
          << "Expected: " << expected_acceleration.print() << "\n";

  EXPECT_TRUE(acceleration == expected_acceleration) << message.str();
}

} // namespace
