#include "medium/compute_coupling.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

struct ElasticAcousticTestParams {
  type_real edge_factor;
  std::array<type_real, 2> normal;
  type_real acceleration;
  std::array<type_real, 2> expected_result;
  type_real tolerance;
  std::string name;
};

class ElasticAcousticCouplingTest
    : public ::testing::TestWithParam<ElasticAcousticTestParams> {};

std::ostream &operator<<(std::ostream &os,
                         const ElasticAcousticTestParams &params) {
  os << params.name;
  return os;
}

TEST_P(ElasticAcousticCouplingTest, CouplingCalculation) {
  const auto &params = GetParam();

  // Create interface data
  specfem::point::coupled_interface<
      specfem::dimension::type::dim2,
      specfem::connections::type::weakly_conforming,
      specfem::interface::interface_tag::elastic_acoustic,
      specfem::element::boundary_tag::none>
      interface_data(params.edge_factor,
                     { params.normal[0], params.normal[1] });

  // Create coupled field (acceleration from elastic medium)
  specfem::point::acceleration<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::acoustic, false>
      coupled_field;
  coupled_field(0) = params.acceleration;

  specfem::point::acceleration<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::elastic_psv, false>
      self_field;

  // Perform coupling computation
  specfem::medium::compute_coupling(interface_data, coupled_field, self_field);

  // Verify results
  EXPECT_NEAR(self_field(0), params.expected_result[0], params.tolerance);
  EXPECT_NEAR(self_field(1), params.expected_result[1], params.tolerance);
}

INSTANTIATE_TEST_SUITE_P(
    ElasticAcousticVariations, ElasticAcousticCouplingTest,
    ::testing::Values(
        // Basic coupling calculation
        ElasticAcousticTestParams{ 1.5,          // edge_factor
                                   { 0.8, 0.6 }, // normal
                                   2.0,          // acceleration
                                   { 2.4, 1.8 }, // expected_result (1.5 * 0.8
                                                 // * 2.0 = 2.4, 1.5 * 0.6 * 2.0
                                                 // = 1.8)
                                   1e-6,         // tolerance
                                   "BasicCouplingCalculation" },
        // Zero edge factor test
        ElasticAcousticTestParams{ 0.0,          // edge_factor
                                   { 1.0, 0.0 }, // normal
                                   3.0,          // acceleration
                                   { 0.0, 0.0 }, // expected_result
                                   1e-12,        // tolerance
                                   "ZeroEdgeFactorTest" },
        // Zero acceleration test
        ElasticAcousticTestParams{ 2.5,          // edge_factor
                                   { 0.6, 0.8 }, // normal
                                   0.0,          // acceleration (zero)
                                   { 0.0, 0.0 }, // expected_result
                                   1e-12,        // tolerance
                                   "ZeroAccelerationTest" },
        // Negative values test
        ElasticAcousticTestParams{
            -2.0,          // edge_factor
            { -0.8, 0.6 }, // normal
            -1.5,          // acceleration
            { -2.4, 1.8 }, // expected_result ((-2.0) * (-0.8) * (-1.5) = -2.4,
                           // (-2.0) * 0.6 * (-1.5) = 1.8)
            1e-6,          // tolerance
            "NegativeValuesTest" },
        // Unit normal test (x-direction)
        ElasticAcousticTestParams{
            1.0,          // edge_factor
            { 1.0, 0.0 }, // normal (unit normal in x-direction)
            3.0,          // acceleration
            { 3.0, 0.0 }, // expected_result (1.0 * 1.0 * 3.0 = 3.0, 1.0 * 0.0
                          // * 3.0 = 0.0)
            1e-6,         // tolerance
            "UnitNormalTest" },
        // Diagonal normal test (45 degrees)
        ElasticAcousticTestParams{
            1.0,                          // edge_factor
            { 0.707106781, 0.707106781 }, // normal (√2/2, √2/2)
            2.0,                          // acceleration
            { 1.414213562, 1.414213562 }, // expected_result (1.0 * 0.707106781
                                          // * 2.0 ≈ 1.414)
            1e-6, // tolerance (relaxed for floating point precision)
            "DiagonalNormalTest" }));
