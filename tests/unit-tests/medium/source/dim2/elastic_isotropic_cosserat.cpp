#include "enumerations/interface.hpp"
#include "enumerations/wavefield.hpp"
#include "medium/compute_source.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

TEST(Source, ElasticIsotropicCosserat2D) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto medium_tag =
      specfem::element::medium_tag::elastic_psv_t;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic_cosserat;

  using PointPropertiesType =
      specfem::point::properties<dimension, medium_tag, property_tag, false>;
  using PointSourceType =
      specfem::point::source<dimension, medium_tag,
                             specfem::wavefield::simulation_field::forward>;
  using PointAccelerationType =
      specfem::point::acceleration<dimension, medium_tag, false>;

  const type_real rho = 2000.0;
  const type_real kappa = 40e9;
  const type_real mu = 30e9;
  const type_real nu = 0.25;
  const type_real j = 1e-6;
  const type_real lambda_c = 20e9;
  const type_real mu_c = 5e9;
  const type_real nu_c = 0.2;
  const PointPropertiesType properties(rho, kappa, mu, nu, j, lambda_c, mu_c,
                                       nu_c);

  PointSourceType point_source;
  point_source.stf(0) = 2.0;
  point_source.stf(1) = 3.0;
  point_source.stf(2) = 4.0;
  point_source.lagrange_interpolant(0) = 1.5;
  point_source.lagrange_interpolant(1) = 2.5;
  point_source.lagrange_interpolant(2) = 3.5;

  const PointAccelerationType acceleration =
      specfem::medium::compute_source_contribution(point_source, properties);

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

TEST(Source, ElasticIsotropicCosserat2D_ZeroSource) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto medium_tag =
      specfem::element::medium_tag::elastic_psv_t;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic_cosserat;

  using PointPropertiesType =
      specfem::point::properties<dimension, medium_tag, property_tag, false>;
  using PointSourceType =
      specfem::point::source<dimension, medium_tag,
                             specfem::wavefield::simulation_field::forward>;
  using PointAccelerationType =
      specfem::point::acceleration<dimension, medium_tag, false>;

  const type_real rho = 2000.0;
  const type_real kappa = 40e9;
  const type_real mu = 30e9;
  const type_real nu = 0.25;
  const type_real j = 1e-6;
  const type_real lambda_c = 20e9;
  const type_real mu_c = 5e9;
  const type_real nu_c = 0.2;
  const PointPropertiesType properties(rho, kappa, mu, nu, j, lambda_c, mu_c,
                                       nu_c);

  PointSourceType point_source;
  point_source.stf(0) = 0.0;
  point_source.stf(1) = 0.0;
  point_source.stf(2) = 0.0;
  point_source.lagrange_interpolant(0) = 1.5;
  point_source.lagrange_interpolant(1) = 2.5;
  point_source.lagrange_interpolant(2) = 3.5;

  const PointAccelerationType acceleration =
      specfem::medium::compute_source_contribution(point_source, properties);

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
