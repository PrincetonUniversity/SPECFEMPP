#include "enumerations/coupled_interface.hpp"
#include "nonconforming.hpp"
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
                                      AnalyticalFunctionType::Power<1> >,
        QuadraturePoints::Asymm5Point>,
    EdgeFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Chain<AnalyticalFunctionType::Power<1>,
                                      AnalyticalFunctionType::Power<2> >,
        QuadraturePoints::Asymm4Point>,
    IntersectionFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Sum<AnalyticalFunctionType::Power<1>,
                                    AnalyticalFunctionType::Power<3> >,
        QuadraturePoints::Asymm5Point> >;

TEST(NonconformingAcousticElastic, Asymm4to5_VectorizedPower) {
  specfem::compute_coupling_test::nonconforming::run_case<
      specfem::interface::interface_tag::acoustic_elastic,
      specfem::interface::flux_scheme_tag::natural,
      Asymm4to5_VectorizedPower>();
}

} // anonymous namespace
