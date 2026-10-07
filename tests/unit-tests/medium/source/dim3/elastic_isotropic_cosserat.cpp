#include "specfem/enums.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

TEST(Source, ElasticIsotropicCosserat3D) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto medium_tag = specfem::element::medium_tag::elastic_spin;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic_cosserat;

  using PointPropertiesType = specfem::point::properties<
      specfem::tags::Tags<dimension, medium_tag, property_tag, false> >;
  using PointSourceType =
      specfem::point::source<dimension, medium_tag,
                             specfem::simulation::field_type::forward>;
  using PointAccelerationType = specfem::point::acceleration<
      specfem::tags::Tags<dimension, medium_tag, false> >;

  const type_real rho = 2700.0;
  const type_real kappa = 50e9;
  const type_real mu = 40e9;
  const type_real nu = 0.25;
  const type_real j = 0.01;
  const type_real lambda_c = 10e9;
  const type_real mu_c = 5e9;
  const type_real nu_c = 0.1;
  const PointPropertiesType properties(rho, kappa, mu, nu, j, lambda_c, mu_c,
                                       nu_c);

  PointSourceType point_source;
  point_source.stf(0) = 1.5;
  point_source.stf(1) = 2.5;
  point_source.stf(2) = 3.5;
  point_source.lagrange_interpolant(0) = 2.0;
  point_source.lagrange_interpolant(1) = 3.0;
  point_source.lagrange_interpolant(2) = 4.0;

  const PointAccelerationType acceleration =
      specfem::medium_physics::compute_source_contribution(point_source,
                                                           properties);

  PointAccelerationType expected_acceleration;
  expected_acceleration(0) =
      point_source.stf(0) * point_source.lagrange_interpolant(0);
  expected_acceleration(1) =
      point_source.stf(1) * point_source.lagrange_interpolant(1);
  expected_acceleration(2) =
      point_source.stf(2) * point_source.lagrange_interpolant(2);

  std::ostringstream message;
  message << "Source acceleration is not equal to expected value: \n"
          << "Computed: " << acceleration.print() << "\n"
          << "Expected: " << expected_acceleration.print() << "\n";

  EXPECT_TRUE(acceleration == expected_acceleration) << message.str();
}

TEST(Source, ElasticIsotropicCosserat3D_ZeroSource) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto medium_tag = specfem::element::medium_tag::elastic_spin;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic_cosserat;

  using PointPropertiesType = specfem::point::properties<
      specfem::tags::Tags<dimension, medium_tag, property_tag, false> >;
  using PointSourceType =
      specfem::point::source<dimension, medium_tag,
                             specfem::simulation::field_type::forward>;
  using PointAccelerationType = specfem::point::acceleration<
      specfem::tags::Tags<dimension, medium_tag, false> >;

  const type_real rho = 2700.0;
  const type_real kappa = 50e9;
  const type_real mu = 40e9;
  const type_real nu = 0.25;
  const type_real j = 0.01;
  const type_real lambda_c = 10e9;
  const type_real mu_c = 5e9;
  const type_real nu_c = 0.1;
  const PointPropertiesType properties(rho, kappa, mu, nu, j, lambda_c, mu_c,
                                       nu_c);

  PointSourceType point_source;
  point_source.stf(0) = 0.0;
  point_source.stf(1) = 0.0;
  point_source.stf(2) = 0.0;
  point_source.lagrange_interpolant(0) = 2.0;
  point_source.lagrange_interpolant(1) = 3.0;
  point_source.lagrange_interpolant(2) = 4.0;

  const PointAccelerationType acceleration =
      specfem::medium_physics::compute_source_contribution(point_source,
                                                           properties);

  PointAccelerationType expected_acceleration;
  expected_acceleration(0) = 0.0;
  expected_acceleration(1) = 0.0;
  expected_acceleration(2) = 0.0;

  std::ostringstream message;
  message << "Source acceleration should be zero for zero STF: \n"
          << "Computed: " << acceleration.print() << "\n"
          << "Expected: " << expected_acceleration.print() << "\n";

  EXPECT_TRUE(acceleration == expected_acceleration) << message.str();
}

} // namespace
