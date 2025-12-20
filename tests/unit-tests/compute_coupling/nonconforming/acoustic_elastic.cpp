#include "enumerations/coupled_interface.hpp"
#include "enumerations/medium.hpp"
#include "nonconforming_tester.hpp"
#include "utilities/include/fixture/nonconforming_interface/quadrature.hpp"
#include <gtest/gtest.h>
#include <tuple>
#include <type_traits>

template <typename IntersectionData2D, typename EdgeFunction2D>
std::array<
    std::array<
        std::array<type_real,
                   specfem::element::attributes<
                       dimension_tag,
                       specfem::interface::attributes<
                           dimension_tag,
                           specfem::interface::interface_tag::
                               acoustic_elastic>::self_medium()>::components>,
        std::tuple_element_t<0, typename IntersectionData2D::packed_accessors>::
            nquad_intersection>,
    1 /*num_edges*/>
expected_solution(
    std::integral_constant<specfem::interface::interface_tag,
                           specfem::interface::interface_tag::acoustic_elastic>,
    const IntersectionData2D &intersection_data,
    const EdgeFunction2D &coupled_field) {
  static constexpr specfem::interface::interface_tag interface_tag =
      specfem::interface::interface_tag::acoustic_elastic;
  static constexpr int num_edges = 1;
  static constexpr int nquad_intersection = std::tuple_element_t<
      0, typename IntersectionData2D::packed_accessors>::nquad_intersection;
  static constexpr int nquad_edge = std::tuple_element_t<
      0, typename IntersectionData2D::packed_accessors>::nquad_edge;
  static constexpr auto self_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::self_medium();
  static constexpr auto coupled_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::coupled_medium();
  static constexpr int ncomp_self =
      specfem::element::attributes<dimension_tag, self_medium>::components;
  static constexpr int ncomp_coupled =
      specfem::element::attributes<dimension_tag, coupled_medium>::components;

  // break apart intersection_data
  const auto &transfer_function = static_cast<
      std::tuple_element_t<0, typename IntersectionData2D::packed_accessors> >(
      intersection_data);
  const auto &normal = static_cast<
      std::tuple_element_t<1, typename IntersectionData2D::packed_accessors> >(
      intersection_data);

  std::array<std::array<std::array<type_real, ncomp_self>, nquad_intersection>,
             num_edges>
      expected;

  std::array<
      std::array<std::array<type_real, ncomp_coupled>, nquad_intersection>,
      num_edges>
      intersection_field;

  // transfer edge -> intersection
  for (int i = 0; i < num_edges; ++i) {
    for (int j = 0; j < nquad_intersection; ++j) {
      for (int l = 0; l < ncomp_coupled; ++l) {
        intersection_field[i][j][l] = 0;
        for (int k = 0; k < nquad_edge; ++k) {
          intersection_field[i][j][l] +=
              transfer_function(i, k, j) * coupled_field(i, k, l);
        }
      }
    }
  }

  // this differs between schemes: compute expectation
  for (int i = 0; i < num_edges; ++i) {
    for (int j = 0; j < nquad_intersection; ++j) {
      expected[i][j][0] = 0;
      for (int k = 0; k < ncomp_coupled; k++) {
        expected[i][j][0] += intersection_field[i][j][k] * normal(i, j, k);
      }
    }
  }

  return expected;
};

TEST(NonconformingComputeCoupling, AcousticElastic) {
  static constexpr specfem::interface::interface_tag interface_tag =
      specfem::interface::interface_tag::acoustic_elastic;

  using EdgeQuadrature = specfem::test::fixture::QuadraturePoints::Asymm5Point;
  using IntersectionQuadrature =
      specfem::test::fixture::QuadraturePoints::Asymm4Point;

  using CoupledFieldInitializerComponent =
      specfem::test::fixture::EdgeFunctionInitializer2D::FromAnalyticalFunction<
          specfem::test::fixture::AnalyticalFunctionType1D::Power<3>,
          EdgeQuadrature>;
  // this is incredibly verbose -- this will be reworked with flux-scheme
  // testing introduction, when data-packs get a minor rework.
  execute(
      std::integral_constant<specfem::interface::interface_tag,
                             interface_tag>(),
      specfem::test::fixture::IntersectionDataPack<
          interface_tag,
          specfem::test::fixture::TransferFunction2D<
              specfem::test::fixture::TransferFunctionInitializer2D::
                  FromQuadratureRules<EdgeQuadrature, IntersectionQuadrature> >,
          specfem::test::fixture::IntersectionFunction2D<
              specfem::test::fixture::IntersectionFunctionInitializer2D::
                  ElementaryBasisVector2D<IntersectionQuadrature::nquad, 0> > >(
          { specfem::test::fixture::TransferFunctionInitializer2D::
                FromQuadratureRules<EdgeQuadrature, IntersectionQuadrature>() },
          { specfem::test::fixture::IntersectionFunctionInitializer2D::
                ElementaryBasisVector2D<IntersectionQuadrature::nquad, 0>() }),
      specfem::test::fixture::EdgeFunction2D<
          specfem::test::fixture::EdgeFunctionInitializer2D::
              StackEdgeFunctionsComponentwise<CoupledFieldInitializerComponent,
                                              CoupledFieldInitializerComponent>,
          specfem::data_access::Accessor<
              specfem::data_access::AccessorType::chunk_edge,
              specfem::data_access::DataClassType::displacement,
              specfem::dimension::type::dim2, false>,
          CoupledMediumTagInheritor<interface_tag> >({}));
}
