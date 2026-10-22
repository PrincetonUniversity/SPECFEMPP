#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <array>
#include <ostream>
#include <string>

/**
 * @brief One pointwise acoustic-elastic coupling case in 3D.
 *
 * Self is the acoustic (fluid) side, which receives a scalar acceleration from
 * the normal projection of the solid side's displacement.
 */
struct AcousticElasticDim3TestParams {
  type_real face_factor;
  std::array<type_real, 3> normal;
  std::array<type_real, 3> displacement_value;
  type_real expected_result;
  type_real tolerance;
  std::string name;
};

void PrintTo(const AcousticElasticDim3TestParams &, std::ostream *os) {
  *os << "";
}

std::ostream &operator<<(std::ostream &os,
                         const AcousticElasticDim3TestParams &params) {
  os << params.name;
  return os;
}

class AcousticElasticDim3CouplingTest
    : public ::testing::TestWithParam<AcousticElasticDim3TestParams> {};

TEST_P(AcousticElasticDim3CouplingTest, CouplingCalculation) {
  const auto &params = GetParam();

  specfem::point::conforming_interface<
      specfem::element::dimension_tag::dim3,
      specfem::element_coupling::interface_tag::acoustic_elastic,
      specfem::element::boundary_tag::none>
      interface_data(params.face_factor,
                     { params.normal[0], params.normal[1], params.normal[2] });

  // Coupled field: displacement on the elastic side.
  specfem::point::displacement<
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic, false>>
      coupled_field;
  coupled_field(0) = params.displacement_value[0];
  coupled_field(1) = params.displacement_value[1];
  coupled_field(2) = params.displacement_value[2];

  // Self field: scalar acceleration on the acoustic side.
  specfem::point::acceleration<
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::acoustic, false>>
      self_field;

  specfem::medium_physics::compute_coupling(interface_data, coupled_field,
                                            self_field);

  EXPECT_NEAR(self_field(0), params.expected_result, params.tolerance);
}

INSTANTIATE_TEST_SUITE_P(
    AcousticElasticDim3Variations, AcousticElasticDim3CouplingTest,
    ::testing::Values(
        // n . u over all three components.
        AcousticElasticDim3TestParams{ 2.0,               // face_factor
                                       { 0.6, 0.8, 0.0 }, // normal
                                       { 1.5, 1.5, 1.5 }, // displacement
                                       4.2, // 2.0 * (0.9 + 1.2 + 0.0)
                                       1e-6,
                                       "BasicCouplingCalculation" },
        // CMB sense: the fluid outer core lies below the mantle, so the fluid
        // element's outward normal on its top face points along +r.
        AcousticElasticDim3TestParams{ 1.5,                // face_factor
                                       { 0.0, 0.0, 1.0 },  // normal
                                       { 0.3, -0.4, 2.0 }, // displacement
                                       3.0,                // 1.5 * 2.0
                                       1e-6,
                                       "OutwardRadialNormal" },
        // ICB sense: the fluid lies above the inner core, so the same geometry
        // produces the opposite normal and the coupling term changes sign. A
        // convention that works at the CMB but not here is the bug this guards.
        AcousticElasticDim3TestParams{ 1.5,                // face_factor
                                       { 0.0, 0.0, -1.0 }, // normal
                                       { 0.3, -0.4, 2.0 }, // displacement
                                       -3.0,               // 1.5 * (-2.0)
                                       1e-6,
                                       "InwardRadialNormalFlipsSign" },
        AcousticElasticDim3TestParams{ 0.0,               // face_factor
                                       { 1.0, 0.0, 0.0 }, // normal
                                       { 5.0, 5.0, 5.0 }, // displacement
                                       0.0,
                                       1e-12,
                                       "ZeroFaceFactorTest" },
        AcousticElasticDim3TestParams{ 1.5,               // face_factor
                                       { 0.8, 0.6, 0.0 }, // normal
                                       { 0.0, 0.0, 0.0 }, // displacement
                                       0.0,
                                       1e-12,
                                       "ZeroDisplacementTest" },
        // Fully oblique normal, so no component is spuriously ignored.
        AcousticElasticDim3TestParams{
            1.0,                                    // face_factor
            { 0.57735027, 0.57735027, 0.57735027 }, // normal (1/sqrt(3))
            { 1.0, 1.0, 1.0 },                      // displacement
            1.73205081,                             // sqrt(3)
            1e-5,
            "ObliqueNormalTest" }),
    [](const ::testing::TestParamInfo<AcousticElasticDim3TestParams> &info)
        -> std::string { return info.param.name; });
