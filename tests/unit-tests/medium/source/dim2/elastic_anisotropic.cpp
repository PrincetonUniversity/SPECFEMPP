#include "enumerations/interface.hpp"
#include "enumerations/wavefield.hpp"
#include "medium/compute_source.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

TEST(Source, ElasticAnisotropicPSV2D) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::elastic_psv;
  static constexpr auto property_tag =
      specfem::element::property_tag::anisotropic;

  using PointPropertiesType =
      specfem::point::properties<dimension, medium_tag, property_tag, false>;
  using PointSourceType =
      specfem::point::source<dimension, medium_tag,
                             specfem::wavefield::simulation_field::forward>;
  using PointAccelerationType =
      specfem::point::acceleration<dimension, medium_tag, false>;

  const type_real c11 = 50e9, c13 = 30e9, c15 = 0.0;
  const type_real c33 = 45e9, c35 = 0.0, c55 = 25e9;
  const type_real rho = 2000.0;
  const type_real c12 = 25e9, c23 = 20e9, c25 = 0.0;
  const PointPropertiesType properties(c11, c13, c15, c33, c35, c55, c12, c23,
                                       c25, rho);

  PointSourceType point_source;
  point_source.stf(0) = 6.0;
  point_source.stf(1) = 4.0;
  point_source.lagrange_interpolant(0) = 2.5;
  point_source.lagrange_interpolant(1) = 3.5;

  const PointAccelerationType acceleration =
      specfem::medium::compute_source_contribution(point_source, properties);

  PointAccelerationType expected_acceleration;
  expected_acceleration(0) =
      point_source.stf(0) * point_source.lagrange_interpolant(0);
  expected_acceleration(1) =
      point_source.stf(1) * point_source.lagrange_interpolant(1);

  std::ostringstream message;
  message << "Source acceleration is not equal to expected value: \n"
          << "Computed: " << acceleration.print() << "\n"
          << "Expected: " << expected_acceleration.print() << "\n";

  EXPECT_TRUE(acceleration == expected_acceleration) << message.str();
}

TEST(Source, ElasticAnisotropicSH2D) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::elastic_sh;
  static constexpr auto property_tag =
      specfem::element::property_tag::anisotropic;

  using PointPropertiesType =
      specfem::point::properties<dimension, medium_tag, property_tag, false>;
  using PointSourceType =
      specfem::point::source<dimension, medium_tag,
                             specfem::wavefield::simulation_field::forward>;
  using PointAccelerationType =
      specfem::point::acceleration<dimension, medium_tag, false>;

  const type_real c11 = 30e9, c13 = 0.0, c15 = 0.0;
  const type_real c33 = 30e9, c35 = 0.0, c55 = 25e9;
  const type_real rho = 2000.0;
  const type_real c12 = 0.0, c23 = 0.0, c25 = 0.0;
  const PointPropertiesType properties(c11, c13, c15, c33, c35, c55, c12, c23,
                                       c25, rho);

  PointSourceType point_source;
  point_source.stf(0) = 8.0;
  point_source.lagrange_interpolant(0) = 1.2;

  const PointAccelerationType acceleration =
      specfem::medium::compute_source_contribution(point_source, properties);

  PointAccelerationType expected_acceleration;
  expected_acceleration(0) =
      point_source.stf(0) * point_source.lagrange_interpolant(0);

  std::ostringstream message;
  message << "Source acceleration is not equal to expected value: \n"
          << "Computed: " << acceleration.print() << "\n"
          << "Expected: " << expected_acceleration.print() << "\n";

  EXPECT_TRUE(acceleration == expected_acceleration) << message.str();
}

} // namespace
