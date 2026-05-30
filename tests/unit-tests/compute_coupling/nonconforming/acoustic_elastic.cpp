#include "nonconforming.hpp"
#include "nonconforming_3d.hpp"
#include "specfem/element_coupling.hpp"
#include "utilities/include/fixture/nonconforming_interface/analytical_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/edge_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/intersection_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/quadrature.hpp"

#include <tuple>

namespace {

using namespace specfem::test_fixture;

using Asymm4to5_VectorizedPower = std::tuple<
    TransferFunctionInitializer2D::FromQuadratureRules<
        QuadraturePoints::Asymm4Point, QuadraturePoints::Asymm5Point>,
    IntersectionFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Chain<AnalyticalFunctionType::Power<0>,
                                      AnalyticalFunctionType::Power<1>>,
        QuadraturePoints::Asymm5Point>,
    EdgeFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Chain<AnalyticalFunctionType::Power<1>,
                                      AnalyticalFunctionType::Power<2>>,
        QuadraturePoints::Asymm4Point>,
    IntersectionFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Sum<AnalyticalFunctionType::Power<1>,
                                    AnalyticalFunctionType::Power<3>>,
        QuadraturePoints::Asymm5Point>>;

TEST(NonconformingAcousticElastic, Asymm4to5_VectorizedPower) {
  specfem::compute_coupling_test::nonconforming::run_case<
      specfem::element_coupling::interface_tag::acoustic_elastic,
      specfem::element_coupling::flux_scheme_tag::natural,
      Asymm4to5_VectorizedPower>();
}

TEST(NonconformingAcousticElastic, InitialSimple3D) {
  // TODO finalize namespace of this execute.
  specfem::compute_coupling_test::compute_coupling::execute<
      specfem::element_coupling::interface_tag::acoustic_elastic>(
      NoisyFaceQuadraturePoints3D<1, 4>(-1, 0, 0, 1),
      specfem::test_fixture::FaceFunction3D<
          specfem::test_fixture::FaceFunctionInitializer3D::
              FromAnalyticalFunction<
                  specfem::test_fixture::AnalyticalFunctionType::Chain<
                      specfem::test_fixture::AnalyticalFunctionType::Power2D<2,
                                                                             1>,
                      specfem::test_fixture::AnalyticalFunctionType::Power2D<1,
                                                                             2>,
                      specfem::test_fixture::AnalyticalFunctionType::Power2D<
                          1, 1>>,
                  specfem::test_fixture::QuadraturePoints::GLL2>>({}),
      specfem::test_fixture::FaceFunction3D<
          specfem::test_fixture::FaceFunctionInitializer3D::
              FromAnalyticalFunction<
                  specfem::test_fixture::AnalyticalFunctionType::Chain<
                      specfem::test_fixture::AnalyticalFunctionType::Power2D<0,
                                                                             0>,
                      specfem::test_fixture::AnalyticalFunctionType::Power2D<0,
                                                                             0>,
                      specfem::test_fixture::AnalyticalFunctionType::Power2D<
                          0, 0>>,
                  specfem::test_fixture::QuadraturePoints::
                      Asymm4Point /*TODO change*/>>({}),
      "Sample 4-point uniform on upper-left corner of GLL2-point-defined "
      "field: x^2 * y (in local coordinates)");
}

} // anonymous namespace
