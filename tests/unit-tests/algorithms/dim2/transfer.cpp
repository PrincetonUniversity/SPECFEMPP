
#include <array>
#include <tuple>
#include <vector>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "algorithms/transfer.hpp"
#include "datatypes/chunk_edge_view.hpp"
#include "enumerations/interface.hpp"
#include "specfem/chunk_edge.hpp"
#include "specfem/data_access.hpp"
#include "utilities/include/fixture/nonconforming_interface.hpp"
#include "utilities/interface.hpp"

#include "SPECFEM_Environment.hpp"

namespace specfem::algorithms_test {

/**
 * @brief Test index type for chunk edge operations.
 */
class ChunkEdgeIndex {
public:
  static constexpr auto accessor_type =
      specfem::data_access::AccessorType::chunk_edge;
  using KokkosIndexType = Kokkos::TeamPolicy<>::member_type;

  /**
   * @brief Get Kokkos team member index.
   * @return Reference to Kokkos team member
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType &get_policy_index() const {
    return this->kokkos_index;
  }

  /**
   * @brief Construct chunk edge index.
   * @param nedges Number of edges in chunk
   * @param kokkos_index Kokkos team member
   */
  KOKKOS_INLINE_FUNCTION
  ChunkEdgeIndex(const int nedges, const KokkosIndexType &kokkos_index)
      : kokkos_index(kokkos_index), _nedges(nedges) {}

  /**
   * @brief Get number of edges.
   * @return Edge count
   */
  KOKKOS_INLINE_FUNCTION int nedges() const { return _nedges; }

private:
  int _nedges;                  ///< Number of edges in the chunk
  KokkosIndexType kokkos_index; /**< Kokkos team member for this chunk */
};

/** Test dimension (2D) */
constexpr static auto dimension_tag = specfem::dimension::type::dim2;
/** Interface type (dummy for testing) */
constexpr static auto interface_tag =
    specfem::interface::interface_tag::acoustic_elastic;
/** Boundary type (dummy for testing) */
constexpr static auto boundary_tag = specfem::element::boundary_tag::none;

/**
 * @brief Compute expected result of transfer function operation.
 * @tparam TransferFunction2D Transfer function type
 * @tparam EdgeFunction2D Field type
 * @param transfer_function Transfer function data
 * @param field Input field data
 * @return Expected transferred field values
 */
template <typename TransferFunction2D, typename EdgeFunction2D>
std::vector<std::array<std::array<type_real, EdgeFunction2D::num_components>,
                       TransferFunction2D::nquad_intersection> >
expected_solution(const TransferFunction2D &transfer_function,
                  const EdgeFunction2D &field) {
  const int n_edges = TransferFunction2D::num_edges;
  std::vector<std::array<std::array<type_real, EdgeFunction2D::num_components>,
                         TransferFunction2D::nquad_intersection> >
      result_field(
          n_edges,
          std::array<std::array<type_real, EdgeFunction2D::num_components>,
                     TransferFunction2D::nquad_intersection>{ 0.0 });
  for (int i = 0; i < n_edges; ++i) {
    for (int j = 0; j < TransferFunction2D::nquad_intersection; ++j) {
      for (int k = 0; k < EdgeFunction2D::num_components; ++k) {

        for (int l = 0; l < TransferFunction2D::nquad_edge; ++l) {
          result_field[i][j][k] += transfer_function(i, l, j) * field(i, l, k);
        }
      }
    }
  }
  return result_field;
}

/*
 * Specialization: We are transfering a function f =
 * AnalyticalFunction::evaluate using an actual quadrature rule.
 */
template <typename AnalyticalFunction, typename EdgeQuadraturePoints,
          typename IntersectionQuadraturePoints>
std::vector<
    std::array<std::array<type_real, 1>, IntersectionQuadraturePoints::nquad> >
expected_solution(
    const specfem::test::fixture::TransferFunction2D<
        specfem::test::fixture::TransferFunctionInitializer2D::
            FromQuadratureRules<EdgeQuadraturePoints,
                                IntersectionQuadraturePoints> >
        &transfer_function,
    const specfem::test::fixture::EdgeFunction2D<
        specfem::test::fixture::EdgeFunctionInitializer2D::
            FromAnalyticalFunction<AnalyticalFunction, EdgeQuadraturePoints> >
        &field) {
  using TransferFunction2D = specfem::test::fixture::TransferFunction2D<
      specfem::test::fixture::TransferFunctionInitializer2D::
          FromQuadratureRules<EdgeQuadraturePoints,
                              IntersectionQuadraturePoints> >;
  using EdgeFunction2D = specfem::test::fixture::EdgeFunction2D<
      specfem::test::fixture::EdgeFunctionInitializer2D::FromAnalyticalFunction<
          AnalyticalFunction, EdgeQuadraturePoints> >;

  const int n_edges = TransferFunction2D::num_edges;
  std::vector<std::array<std::array<type_real, EdgeFunction2D::num_components>,
                         TransferFunction2D::nquad_intersection> >
      result_field(
          n_edges,
          std::array<std::array<type_real, EdgeFunction2D::num_components>,
                     TransferFunction2D::nquad_intersection>{ 0.0 });
  for (int i = 0; i < n_edges; ++i) {
    for (int j = 0; j < TransferFunction2D::nquad_intersection; ++j) {
      for (int k = 0; k < EdgeFunction2D::num_components; ++k) {

        result_field[i][j][k] = AnalyticalFunction::evaluate(
            IntersectionQuadraturePoints::quadrature_points[j]);
      }
    }
  }
  return result_field;
}

using ZeroTransferFunction = specfem::test::fixture::TransferFunction2D<
    specfem::test::fixture::TransferFunctionInitializer2D::Zero>;
using UniformEdgeFunction = specfem::test::fixture::EdgeFunction2D<
    specfem::test::fixture::EdgeFunctionInitializer2D::Uniform>;

/**
 * @brief Compute transferred field for zero transfer function and uniform
 * field.
 * @param transfer_function Zero transfer function
 * @param field Uniform field
 * @return Zero field result
 */

std::vector<
    std::array<std::array<type_real, UniformEdgeFunction::num_components>,
               ZeroTransferFunction::nquad_intersection> >
expected_solution(const ZeroTransferFunction &transfer_function,
                  const UniformEdgeFunction &field) {
  using TransferFunction2D = ZeroTransferFunction;
  using EdgeFunction2D = UniformEdgeFunction;
  // Result field is a zero field
  const int n_edges = EdgeFunction2D::num_edges;
  return std::vector<
      std::array<std::array<type_real, EdgeFunction2D::num_components>,
                 TransferFunction2D::nquad_intersection> >(n_edges, [] {
    std::array<std::array<type_real, EdgeFunction2D::num_components>,
               TransferFunction2D::nquad_intersection>
        arr{};
    return arr;
  }());
}

/**
 * @brief Execute transfer function test with validation.
 * @tparam TransferFunction2D Transfer function type
 * @tparam EdgeFunction2D Field type
 * @param transfer_function Transfer function data
 * @param function Input function data
 */
template <typename TransferFunction2D, typename EdgeFunction2D>
void execute(const TransferFunction2D &transfer_function,
             const EdgeFunction2D &function) {
  auto expected = expected_solution(transfer_function, function);

  const int n_edges = TransferFunction2D::num_edges;
  using TransferFunctionType = specfem::chunk_edge::impl::transfer_function<
      dimension_tag, 1, TransferFunction2D::nquad_intersection,
      TransferFunction2D::nquad_edge,
      specfem::data_access::DataClassType::transfer_function_self,
      interface_tag, boundary_tag, typename TransferFunction2D::memory_space,
      Kokkos::MemoryTraits<> >;
  using FunctionType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, dimension_tag, 1, TransferFunction2D::nquad_edge,
      EdgeFunction2D::num_components, false,
      typename TransferFunction2D::memory_space, Kokkos::MemoryTraits<> >;

  const auto transfer_function_view = transfer_function.get_view();
  const auto function_view = function.get_view();

  const auto results_view_name = "result_view";

  Kokkos::View<type_real * [TransferFunction2D::nquad_intersection]
                               [EdgeFunction2D::num_components],
               typename TransferFunction2D::memory_space>
      result_view(results_view_name, n_edges);

  Kokkos::parallel_for(
      "transfer_function_test", Kokkos::TeamPolicy<>(n_edges, 1, 1),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member) {
        const int iedge = team_member.league_rank();
        const TransferFunctionType TF(Kokkos::subview(
            transfer_function_view, Kokkos::make_pair(iedge, iedge + 1),
            Kokkos::ALL(), Kokkos::ALL()));
        const FunctionType F(
            Kokkos::subview(function_view, Kokkos::make_pair(iedge, iedge + 1),
                            Kokkos::ALL(), Kokkos::ALL()));
        specfem::algorithms::transfer(
            ChunkEdgeIndex(1, team_member), TF, F,
            [&](const auto &index, const auto &point) {
              for (int icomp = 0; icomp < EdgeFunction2D::num_components;
                   ++icomp) {
                Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
                  result_view(index(0), index(1), icomp) = point(icomp);
                });
              }
            });
      });

  Kokkos::fence();

  auto result_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result_view);

  for (int i = 0; i < n_edges; ++i) {
    for (int j = 0; j < TransferFunction2D::nquad_intersection; ++j) {
      if (!specfem::utilities::is_close(result_host(i, j, 0),
                                        expected[i][j][0])) {
        std::ostringstream oss;
        oss << "-- Transfer function --\n"
            << TransferFunction2D::description() << std::endl
            << "-- Edge Function --\n"
            << EdgeFunction2D::description() << std::endl
            << "\n-- Failure --\n"
            << "Transfer function test failed at edge " << i << ": expected "
            << expected[i][j][0] << "\n got " << result_host(i, j, 0)
            << std::endl;

        ADD_FAILURE() << oss.str();
      }
    }
  }
}

} // namespace specfem::algorithms_test

using namespace specfem::algorithms_test;

/**
 * @brief Test fixture for 2D transfer function algorithms.
 * @tparam TestingTypes Tuple of (TransferFunctionInitializer,
 * FunctionInitializer)
 */
template <typename TestingTypes>
struct TransferFunctionTest2D : public ::testing::Test {
  using TransferFunctionInitializer = std::tuple_element_t<0, TestingTypes>;
  using FunctionInitializer = std::tuple_element_t<1, TestingTypes>;

  /**
   * @brief Set up test with initialized transfer function and field.
   */
  TransferFunctionTest2D()
      : transfer_function(TransferFunctionInitializer()),
        function(FunctionInitializer()) {}

  specfem::test::fixture::TransferFunction2D<TransferFunctionInitializer>
      transfer_function; /**< Test
    transfer
    function
  */
  specfem::test::fixture::EdgeFunction2D<FunctionInitializer> function; /**<
                                                                           Test
                                                                           field
                                                                         */
};

using specfem::test::fixture::AnalyticalFunctionType1D::Power;
using specfem::test::fixture::EdgeFunctionInitializer2D::FromAnalyticalFunction;
using specfem::test::fixture::QuadraturePoints::Asymm4Point;
using specfem::test::fixture::QuadraturePoints::Asymm5Point;
using specfem::test::fixture::QuadraturePoints::GLL1;
using specfem::test::fixture::QuadraturePoints::GLL2;
using specfem::test::fixture::TransferFunctionInitializer2D::
    FromQuadratureRules;

/** Test type combinations for parameterized testing */
using TransferFunctionTestTypes2D = ::testing::Types<
    std::tuple<specfem::test::fixture::TransferFunctionInitializer2D::Zero,
               specfem::test::fixture::EdgeFunctionInitializer2D::Uniform>,
    std::tuple<FromQuadratureRules<GLL1, GLL2>,
               FromAnalyticalFunction<Power<0>, GLL1> >,
    std::tuple<FromQuadratureRules<GLL2, GLL1>,
               FromAnalyticalFunction<Power<1>, GLL2> >,
    std::tuple<FromQuadratureRules<GLL2, GLL2>,
               FromAnalyticalFunction<Power<2>, GLL2> >,
    std::tuple<FromQuadratureRules<Asymm4Point, Asymm5Point>,
               FromAnalyticalFunction<Power<3>, Asymm4Point> >,
    std::tuple<FromQuadratureRules<Asymm5Point, Asymm4Point>,
               FromAnalyticalFunction<Power<4>, Asymm5Point> >,
    std::tuple<FromQuadratureRules<Asymm5Point, Asymm5Point>,
               FromAnalyticalFunction<Power<5>, Asymm5Point> > >;

TYPED_TEST_SUITE(TransferFunctionTest2D, TransferFunctionTestTypes2D);

TYPED_TEST(TransferFunctionTest2D, ExecuteTransferFunction) {
  execute(this->transfer_function, this->function);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
