#pragma once

#include "initializers.hpp"
#include "specfem_setup.hpp"

#include <array>
#include <type_traits>

namespace specfem::test::fixture {
/**
 * @brief Types for TransferFunction2D. These contain the compile-time values of
 * the transfer functions
 *
 */
namespace TransferFunctionInitializer2D {

/** Zero transfer function initializer with ngll = nquad_intersection = 5 */
struct Zero : TransferFunctionInitializer2D {
  static constexpr int num_edges = 1;
  static constexpr int nquad_edge = 5;
  static constexpr int nquad_intersection = 5;

private:
  using ArrayType = std::array<
      std::array<std::array<type_real, nquad_intersection>, nquad_edge>,
      num_edges>;

public:
  static ArrayType init_transfer_function() { return { 0 }; }

  static std::string description() {
    return "A blank transfer function. This should zero out all values";
  }
};

template <typename EdgeQuadraturePoints_,
          typename IntersectionQuadraturePoints_>
struct FromQuadratureRules : TransferFunctionInitializer2D {
  static constexpr int num_edges = 1;
  using EdgeQuadraturePoints = EdgeQuadraturePoints_;
  using IntersectionQuadraturePoints = IntersectionQuadraturePoints_;

  using EdgeQuadrature = QuadratureRule<EdgeQuadraturePoints>;
  static constexpr int nquad_edge = EdgeQuadraturePoints::nquad;
  static constexpr int nquad_intersection = IntersectionQuadraturePoints::nquad;

  static constexpr auto edge_quadrature_points =
      EdgeQuadraturePoints::quadrature_points;
  static constexpr auto intersection_quadrature_points =
      IntersectionQuadraturePoints::quadrature_points;

private:
  using ArrayType = std::array<
      std::array<std::array<type_real, nquad_intersection>, nquad_edge>,
      num_edges>;

public:
  static ArrayType init_transfer_function() {
    ArrayType transfer_function;

    for (int ipoint_edge = 0; ipoint_edge < nquad_edge; ipoint_edge++) {
      for (int ipoint_intersection = 0;
           ipoint_intersection < nquad_intersection; ipoint_intersection++) {
        transfer_function[0][ipoint_edge][ipoint_intersection] =
            EdgeQuadrature::evaluate_lagrange_polynomial(
                ipoint_edge,
                intersection_quadrature_points[ipoint_intersection]);
      }
    }
    return transfer_function;
  }
  static std::string description() {
    return "GLL1 ({-1, 1} on the edge), GLL2 ({-1, 0, 1} on the intersection)";
  }
};

} // namespace TransferFunctionInitializer2D

// =================================================================================================

template <typename Initializer> struct TransferFunction2D;

/**
 * @brief Test transfer function container.
 * @tparam Initializer Transfer function initialization strategy
 */
template <typename Initializer> struct TransferFunction2D {
  static_assert(
      std::is_base_of_v<
          TransferFunctionInitializer2D::TransferFunctionInitializer2D,
          Initializer>,
      "TransferFunction2D needs an TransferFunctionInitializer2D!");

public:
  using TransferFunctionInitializer = Initializer;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  static constexpr int num_edges = Initializer::num_edges;

  static constexpr int nquad_edge = Initializer::nquad_edge;

  static constexpr int nquad_intersection = Initializer::nquad_intersection;

  static std::string description() { return Initializer::description(); }

private:
  std::array<std::array<std::array<type_real, nquad_intersection>, nquad_edge>,
             num_edges>
      _transfer_function;
  using TransferFunctionView =
      Kokkos::View<type_real[num_edges][nquad_edge][nquad_intersection],
                   memory_space>;

public:
  /**
   * @brief Construct transfer function with initializer.
   * @param initializer Initialization strategy
   */
  TransferFunction2D(const Initializer &initializer) {
    _transfer_function = Initializer::init_transfer_function();
  }

  /**
   * @brief Get Kokkos view of transfer function data.
   * @return Kokkos view for device access
   */
  TransferFunctionView get_view() const {
    TransferFunctionView view("transfer_function_view");
    for (size_t i = 0; i < num_edges; ++i) {
      for (size_t j = 0; j < nquad_edge; ++j) {
        for (size_t k = 0; k < nquad_intersection; ++k) {
          view(i, j, k) = (*this)(i, j, k);
        }
      }
    }
    return view;
  }

  /**
   * @brief Access transfer function values.
   * @param i Edge index
   * @param j Element quadrature index
   * @param k Intersection quadrature index
   * @return Reference to transfer function value
   */
  type_real &operator()(const int i, const int j, const int k) {
    return _transfer_function[i][j][k];
  }

  const type_real &operator()(const int i, const int j, const int k) const {
    return _transfer_function[i][j][k];
  }
};

} // namespace specfem::test::fixture
