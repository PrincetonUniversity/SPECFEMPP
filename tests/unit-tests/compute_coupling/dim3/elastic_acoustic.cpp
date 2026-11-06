#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <array>
#include <ostream>
#include <string>

/**
 * @brief One pointwise elastic-acoustic coupling case in 3D.
 *
 * Self is the elastic (solid) side, which receives a vector acceleration from
 * the fluid side's scalar acceleration projected along the face normal.
 */
struct ElasticAcousticDim3TestParams {
  type_real face_factor;
  std::array<type_real, 3> normal;
  type_real acceleration;
  std::array<type_real, 3> expected_result;
  type_real tolerance;
  std::string name;
};

void PrintTo(const ElasticAcousticDim3TestParams &, std::ostream *os) {
  *os << "";
}

std::ostream &operator<<(std::ostream &os,
                         const ElasticAcousticDim3TestParams &params) {
  os << params.name;
  return os;
}

class ElasticAcousticDim3CouplingTest
    : public ::testing::TestWithParam<ElasticAcousticDim3TestParams> {};

TEST_P(ElasticAcousticDim3CouplingTest, CouplingCalculation) {
  const auto &params = GetParam();

  specfem::point::conforming_interface<
      specfem::element::dimension_tag::dim3,
      specfem::element_coupling::interface_tag::elastic_acoustic,
      specfem::element::boundary_tag::none>
      interface_data(params.face_factor,
                     { params.normal[0], params.normal[1], params.normal[2] });

  // Coupled field: scalar acceleration on the acoustic side.
  specfem::point::acceleration<
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::acoustic, false>>
      coupled_field;
  coupled_field(0) = params.acceleration;

  // Self field: vector acceleration on the elastic side.
  specfem::point::acceleration<
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic, false>>
      self_field;

  specfem::medium_physics::compute_coupling(interface_data, coupled_field,
                                            self_field);

  EXPECT_NEAR(self_field(0), params.expected_result[0], params.tolerance);
  EXPECT_NEAR(self_field(1), params.expected_result[1], params.tolerance);
  EXPECT_NEAR(self_field(2), params.expected_result[2], params.tolerance);
}

INSTANTIATE_TEST_SUITE_P(
    ElasticAcousticDim3Variations, ElasticAcousticDim3CouplingTest,
    ::testing::Values(
        ElasticAcousticDim3TestParams{ 1.5,               // face_factor
                                       { 0.8, 0.6, 0.0 }, // normal
                                       2.0,               // acceleration
                                       { 2.4, 1.8, 0.0 }, // 1.5 * n * 2.0
                                       1e-6,
                                       "BasicCouplingCalculation" },
        // CMB sense: the solid mantle sits above the fluid outer core, so the
        // solid element's outward normal on its bottom face points along -r.
        ElasticAcousticDim3TestParams{ 1.0,                // face_factor
                                       { 0.0, 0.0, -1.0 }, // normal
                                       3.0,                // acceleration
                                       { 0.0, 0.0, -3.0 },
                                       1e-6,
                                       "InwardRadialNormal" },
        // ICB sense: the solid inner core sits below the fluid, so the normal
        // reverses and so must every component of the coupling term.
        ElasticAcousticDim3TestParams{ 1.0,               // face_factor
                                       { 0.0, 0.0, 1.0 }, // normal
                                       3.0,               // acceleration
                                       { 0.0, 0.0, 3.0 },
                                       1e-6,
                                       "OutwardRadialNormalFlipsSign" },
        ElasticAcousticDim3TestParams{ 0.0,               // face_factor
                                       { 1.0, 0.0, 0.0 }, // normal
                                       3.0,               // acceleration
                                       { 0.0, 0.0, 0.0 },
                                       1e-12,
                                       "ZeroFaceFactorTest" },
        ElasticAcousticDim3TestParams{ 2.0,               // face_factor
                                       { 0.8, 0.6, 0.0 }, // normal
                                       0.0,               // acceleration
                                       { 0.0, 0.0, 0.0 },
                                       1e-12,
                                       "ZeroAccelerationTest" },
        // Fully oblique normal, so no component is spuriously ignored.
        ElasticAcousticDim3TestParams{ 2.0,                       // face_factor
                                       { 0.5, -0.5, 0.70710678 }, // normal
                                       4.0, // acceleration
                                       { 4.0, -4.0, 5.65685424 }, // 2.0 * n
                                                                  // * 4.0
                                       1e-5,
                                       "ObliqueNormalTest" }),
    [](const ::testing::TestParamInfo<ElasticAcousticDim3TestParams> &info)
        -> std::string { return info.param.name; });
