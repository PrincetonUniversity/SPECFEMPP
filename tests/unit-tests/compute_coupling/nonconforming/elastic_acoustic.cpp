#include "enumerations/coupled_interface.hpp"
#include "nonconforming.hpp"
#include "utilities/include/fixture/nonconforming_interface/analytical_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/edge_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/intersection_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/quadrature.hpp"

#include <tuple>

namespace {

using namespace specfem::test_fixture;

using GLL2_Constant = std::tuple<
    TransferFunctionInitializer2D::FromQuadratureRules<QuadraturePoints::GLL2,
                                                       QuadraturePoints::GLL2>,
    IntersectionFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Chain<AnalyticalFunctionType::Power<0>,
                                      AnalyticalFunctionType::Power<1> >,
        QuadraturePoints::GLL2>,
    EdgeFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Power<0>, QuadraturePoints::GLL2>,
    IntersectionFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Chain<AnalyticalFunctionType::Power<0>,
                                      AnalyticalFunctionType::Power<1> >,
        QuadraturePoints::GLL2> >;

using Asymm4to5_HigherOrder = std::tuple<
    TransferFunctionInitializer2D::FromQuadratureRules<
        QuadraturePoints::Asymm4Point, QuadraturePoints::Asymm5Point>,
    IntersectionFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Chain<AnalyticalFunctionType::Power<3>,
                                      AnalyticalFunctionType::Power<2> >,
        QuadraturePoints::Asymm5Point>,
    EdgeFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Power<1>, QuadraturePoints::Asymm4Point>,
    IntersectionFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Chain<AnalyticalFunctionType::Power<4>,
                                      AnalyticalFunctionType::Power<3> >,
        QuadraturePoints::Asymm5Point> >;

TEST(NonconformingElasticAcoustic, GLL2_Constant) {
  specfem::compute_coupling_test::nonconforming::run_case<
      specfem::interface::interface_tag::elastic_acoustic,
      specfem::interface::flux_scheme_tag::natural, GLL2_Constant>();
}

TEST(NonconformingElasticAcoustic, Asymm4to5_HigherOrder) {
  specfem::compute_coupling_test::nonconforming::run_case<
      specfem::interface::interface_tag::elastic_acoustic,
      specfem::interface::flux_scheme_tag::natural, Asymm4to5_HigherOrder>();
}

} // namespace
