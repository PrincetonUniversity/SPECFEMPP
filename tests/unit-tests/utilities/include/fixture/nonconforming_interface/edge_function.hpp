#pragma once

#include "initializers.hpp"
#include "specfem_setup.hpp"

#include <type_traits>

namespace specfem::test::fixture {

/**
 * @brief Types for EdgeFunction2D. These contain the compile-time values of the
 * edge function
 *
 */
namespace EdgeFunctionInitializer2D {

struct Uniform : EdgeFunctionInitializer2D {
  static constexpr int num_edges = 1;
  static constexpr int num_components = 1;
  static constexpr int nquad_edge = 5;

private:
  using ArrayType =
      std::array<std::array<std::array<type_real, num_components>, nquad_edge>,
                 num_edges>;

public:
  static ArrayType init_function() {
    ArrayType edge_function;
    for (size_t i = 0; i < num_edges; ++i) {
      for (size_t j = 0; j < nquad_edge; ++j) {
        for (size_t k = 0; k < num_components; ++k) {
          edge_function[i][j][k] = 1;
        }
      }
    }
    return edge_function;
  }

  static std::string description() {
    return std::string("ngll = 5 field of all ones");
  }
};

template <typename AnalyticalFunctionType, typename EdgePoints>
struct FromAnalyticalFunction : EdgeFunctionInitializer2D {
  using AnalyticalFunction = AnalyticalFunctionType;
  static_assert(
      std::is_base_of_v<AnalyticalFunctionType1D::AnalyticalFunctionType1D,
                        AnalyticalFunction>,
      "FromAnalyticalFunction expects its first template argument to be "
      "an AnalyticalFunctionType1D!");

  using EdgeQuadraturePoints = EdgePoints;

  static constexpr int num_edges = 1;
  static constexpr int num_components = AnalyticalFunction::num_components;
  static constexpr int nquad_edge = EdgeQuadraturePoints::nquad;
  static constexpr auto edge_quadrature_points =
      EdgeQuadraturePoints::quadrature_points;

  static std::string description() {
    return std::string("EdgeField-initialized function (description: \"") +
           AnalyticalFunction::description() + "\")";
  }

private:
  using ArrayType =
      std::array<std::array<std::array<type_real, num_components>, nquad_edge>,
                 num_edges>;

public:
  static ArrayType init_function() {
    ArrayType edge_function;
    for (size_t i = 0; i < num_edges; ++i) {
      for (size_t j = 0; j < nquad_edge; ++j) {
        const auto eval =
            AnalyticalFunction::evaluate(edge_quadrature_points[j]);
        // if num_components is 1, we may see a type_real returned, rather than
        // an array.
        if constexpr (std::is_arithmetic_v<decltype(eval)>) {
          edge_function[i][j][0] = eval;
        } else {
          for (size_t k = 0; k < num_components; ++k) {
            edge_function(i, j, k) = eval[k];
          }
        }
      }
    }
    return edge_function;
  }
};

} // namespace EdgeFunctionInitializer2D

// =================================================================================================

/**
 * @brief Test field container.
 * @tparam Initializer Field initialization strategy
 */
template <typename Initializer> struct EdgeFunction2D {
  static_assert(
      std::is_base_of_v<EdgeFunctionInitializer2D::EdgeFunctionInitializer2D,
                        Initializer>,
      "EdgeFunction2D needs an EdgeFunctionInitializer2D!");

public:
  using FunctionInitializer = Initializer;
  static constexpr int num_edges = Initializer::num_edges;
  static constexpr int num_components = Initializer::num_components;
  static constexpr int nquad_edge = Initializer::nquad_edge;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  static std::string description() { return Initializer::description(); }

private:
  std::array<std::array<std::array<type_real, num_components>, nquad_edge>,
             num_edges>
      _field;
  using FieldView =
      Kokkos::View<type_real[num_edges][nquad_edge][num_components],
                   memory_space>;

public:
  /**
   * @brief Construct field with initializer.
   * @param initializer Initialization strategy
   */
  EdgeFunction2D(const FunctionInitializer &initializer) {
    _field = FunctionInitializer::init_function();
  }

  /**
   * @brief Get Kokkos view of field data.
   * @return Kokkos view for device access
   */
  FieldView get_view() const {
    FieldView view("field_view");
    for (size_t i = 0; i < num_edges; ++i) {
      for (size_t j = 0; j < nquad_edge; ++j) {
        for (size_t k = 0; k < num_components; ++k) {
          view(i, j, k) = (*this)(i, j, k);
        }
      }
    }
    return view;
  }

  /**
   * @brief Access field values.
   * @param i Edge index
   * @param j Element quadrature index
   * @param k Component index
   * @return Reference to field value
   */
  type_real &operator()(const int i, const int j, const int k) {
    return _field[i][j][k];
  }
  const type_real &operator()(const int i, const int j, const int k) const {
    return _field[i][j][k];
  }
};
} // namespace specfem::test::fixture
