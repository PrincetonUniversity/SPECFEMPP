#include "enumerations/interface.hpp"
#include "enumerations/wavefield.hpp"
#include "medium/compute_source.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

TEST(Source, AcousticIsotropic2D) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::acoustic;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using PointPropertiesType =
      specfem::point::properties<dimension, medium_tag, property_tag, false>;
  using PointSourceType =
      specfem::point::source<dimension, medium_tag,
                             specfem::wavefield::simulation_field::forward>;
  using PointAccelerationType =
      specfem::point::acceleration<dimension, medium_tag, false>;

  const type_real rho_inverse = 2.0;
  const type_real kappa = 10.0;
  const PointPropertiesType properties(rho_inverse, kappa);

  PointSourceType point_source;
  point_source.stf(0) = 5.0;
  point_source.lagrange_interpolant(0) = 3.0;

  const PointAccelerationType acceleration =
      specfem::medium::compute_source_contribution(point_source, properties);

  PointAccelerationType expected_acceleration;
  expected_acceleration(0) =
      -point_source.stf(0) * point_source.lagrange_interpolant(0) / kappa;

  std::ostringstream message;
  message << "Source acceleration is not equal to expected value: \n"
          << "Computed: " << acceleration.print() << "\n"
          << "Expected: " << expected_acceleration.print() << "\n";

  EXPECT_TRUE(acceleration == expected_acceleration) << message.str();
}

TEST(Source, AcousticIsotropic2D_ZeroSource) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::acoustic;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using PointPropertiesType =
      specfem::point::properties<dimension, medium_tag, property_tag, false>;
  using PointSourceType =
      specfem::point::source<dimension, medium_tag,
                             specfem::wavefield::simulation_field::forward>;
  using PointAccelerationType =
      specfem::point::acceleration<dimension, medium_tag, false>;

  const type_real rho_inverse = 2.0;
  const type_real kappa = 10.0;
  const PointPropertiesType properties(rho_inverse, kappa);

  PointSourceType point_source;
  point_source.stf(0) = 0.0;
  point_source.lagrange_interpolant(0) = 3.0;

  const PointAccelerationType acceleration =
      specfem::medium::compute_source_contribution(point_source, properties);

  PointAccelerationType expected_acceleration;
  expected_acceleration(0) = 0.0;

  std::ostringstream message;
  message << "Source acceleration should be zero for zero STF: \n"
          << "Computed: " << acceleration.print() << "\n"
          << "Expected: " << expected_acceleration.print() << "\n";

  EXPECT_TRUE(acceleration == expected_acceleration) << message.str();
}

} // namespace
