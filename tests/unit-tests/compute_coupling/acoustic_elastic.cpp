#include "medium/compute_coupling.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

struct AcousticElasticTestParams {
  type_real edge_factor;
  std::array<type_real, 2> normal;
  std::array<type_real, 2> displacement_value;
  type_real expected_result;
  type_real tolerance;
  std::string name;
};

std::ostream &operator<<(std::ostream &os,
                         const AcousticElasticTestParams &params) {
  os << params.name;
  return os;
}

class AcousticElasticCouplingTest
    : public ::testing::TestWithParam<AcousticElasticTestParams> {};

TEST_P(AcousticElasticCouplingTest, CouplingCalculation) {
  const auto &params = GetParam();

  // Create interface data
  specfem::point::coupled_interface<
      specfem::dimension::type::dim2,
      specfem::connections::type::weakly_conforming,
      specfem::interface::interface_tag::acoustic_elastic,
      specfem::element::boundary_tag::none>
      interface_data(params.edge_factor,
                     { params.normal[0], params.normal[1] });

  // Create coupled field (displacement from acoustic medium)
  specfem::point::displacement<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::elastic_psv, false>
      coupled_field;
  coupled_field(0) = params.displacement_value[0];
  coupled_field(1) = params.displacement_value[1];

  // Create self field (acceleration in elastic medium)
  specfem::point::acceleration<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::acoustic, false>
      self_field;

  // Perform coupling computation
  specfem::medium::compute_coupling(interface_data, coupled_field, self_field);

  // Verify result
  EXPECT_NEAR(self_field(0), params.expected_result, params.tolerance);
}

INSTANTIATE_TEST_SUITE_P(
    AcousticElasticVariations, AcousticElasticCouplingTest,
    ::testing::Values(
        // Basic coupling calculation
        AcousticElasticTestParams{ 2.0,          // edge_factor
                                   { 0.6, 0.8 }, // normal
                                   { 1.5, 1.5 }, // displacement_value
                                   4.2,  // expected_result (2.0 * (0.6 * 1.5 +
                                         // 0.8 * 1.5) = 4.2)
                                   1e-6, // tolerance
                                   "BasicCouplingCalculation" },
        // Zero edge factor test
        AcousticElasticTestParams{ 0.0,          // edge_factor
                                   { 1.0, 0.0 }, // normal
                                   { 5.0, 5.0 }, // displacement_value
                                   0.0,          // expected_result
                                   1e-12,        // tolerance
                                   "ZeroEdgeFactorTest" },
        // Zero displacement test
        AcousticElasticTestParams{ 1.5,          // edge_factor
                                   { 0.8, 0.6 }, // normal
                                   { 0.0, 0.0 }, // displacement_value
                                   0.0,          // expected_result
                                   1e-12,        // tolerance
                                   "ZeroDisplacementTest" },
        // Negative values test
        AcousticElasticTestParams{ -2.0,          // edge_factor
                                   { -0.6, 0.8 }, // normal
                                   { -2.0, 0.0 }, // displacement_value
                                   -2.4, // expected_result ((-2.0) * ((-0.6) *
                                         // (-2.0) + (-2.0) * (0.8) * 0.0) =
                                         // -2.4)
                                   1e-6, // tolerance
                                   "NegativeValuesTest" },
        // Large values test
        AcousticElasticTestParams{
            1e6,                          // edge_factor
            { 0.707106781, 0.707106781 }, // normal (1/sqrt(2), 1/sqrt(2))
            { 1e-6, 1e-6 },               // displacement_value
            1.414213562, // expected_result (1e6 * (0.707106781 * 1e-6 +
                         // 0.707106781 * 1e-6) = 1.414213562)
            1e-6,        // tolerance (relaxed for large numbers)
            "LargeValuesTest" }));
