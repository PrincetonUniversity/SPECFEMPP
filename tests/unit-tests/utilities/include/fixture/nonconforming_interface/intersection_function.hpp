#pragma once

#include "initializers.hpp"
#include "specfem_setup.hpp"

#include <type_traits>

namespace specfem::test::fixture {

/**
 * @brief Types for IntersectionFunction2D. These contain the compile-time
 * values of the edge function
 *
 */
namespace IntersectionFunctionInitializer2D {

template <typename AnalyticalFunctionType, typename IntersectionPoints>
struct FromAnalyticalFunction : IntersectionFunctionInitializer2D {
  using AnalyticalFunction = AnalyticalFunctionType;
  static_assert(
      std::is_base_of_v<AnalyticalFunctionType1D::AnalyticalFunctionType1D,
                        AnalyticalFunction>,
      "FromAnalyticalFunction expects its first template argument to be "
      "an AnalyticalFunctionType1D!");

  using IntersectionQuadraturePoints = IntersectionPoints;
  static_assert(
      std::is_base_of_v<QuadraturePoints::QuadraturePoints, IntersectionPoints>,
      "FromAnalyticalFunction expects its second template argument to be "
      "of QuadraturePoints type!");

  static constexpr int num_edges = 1;
  static constexpr int num_components = AnalyticalFunction::num_components;
  static constexpr int nquad_intersection = IntersectionQuadraturePoints::nquad;
  static constexpr auto intersection_quadrature_points =
      IntersectionQuadraturePoints::quadrature_points;

  static std::string description() {
    return std::string("EdgeField-initialized function (description: \"") +
           AnalyticalFunction::description() + "\")";
  }

private:
  using ArrayType = std::array<
      std::array<std::array<type_real, num_components>, nquad_intersection>,
      num_edges>;

public:
  static ArrayType init_function() {
    ArrayType intersection_function;
    for (size_t i = 0; i < num_edges; ++i) {
      for (size_t j = 0; j < nquad_intersection; ++j) {
        const auto eval =
            AnalyticalFunction::evaluate(intersection_quadrature_points[j]);
        // if num_components is 1, we may see a type_real returned, rather than
        // an array.
        if constexpr (std::is_arithmetic_v<decltype(eval)>) {
          intersection_function[i][j][0] = eval;
        } else {
          for (size_t k = 0; k < num_components; ++k) {
            intersection_function(i, j, k) = eval[k];
          }
        }
      }
    }
    return intersection_function;
  }
};

template <int nquad_intersection_, int index>
struct ElementaryBasisVector2D : IntersectionFunctionInitializer2D {
  static constexpr int num_edges = 1;
  static constexpr int num_components = 2;
  static constexpr int nquad_intersection = nquad_intersection_;
  static std::string description() {
    return std::string("Basis vector e_") + std::to_string(index);
  }

private:
  using ArrayType = std::array<
      std::array<std::array<type_real, num_components>, nquad_intersection>,
      num_edges>;

public:
  static ArrayType init_function() {
    ArrayType intersection_function;
    for (size_t i = 0; i < num_edges; ++i) {
      for (size_t j = 0; j < nquad_intersection; ++j) {
        for (size_t k = 0; k < num_components; ++k) {
          intersection_function[i][j][k] = (index == k) ? 1 : 0;
        }
      }
    }
    return intersection_function;
  }
};

} // namespace IntersectionFunctionInitializer2D

// =================================================================================================

/**
 * @brief Test field container.
 * @tparam Initializer Field initialization strategy
 */
template <typename Initializer> struct IntersectionFunction2D {
  static_assert(
      std::is_base_of_v<
          IntersectionFunctionInitializer2D::IntersectionFunctionInitializer2D,
          Initializer>,
      "IntersectionFunction2D needs an IntersectionFunctionInitializer2D!");

public:
  static constexpr specfem::connections::type connection_tag =
      specfem::connections::type::nonconforming;
  using FunctionInitializer = Initializer;
  static constexpr int num_edges = Initializer::num_edges;
  static constexpr int num_components = Initializer::num_components;
  static constexpr int nquad_intersection = Initializer::nquad_intersection;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  static std::string description() { return Initializer::description(); }

private:
  std::array<
      std::array<std::array<type_real, num_components>, nquad_intersection>,
      num_edges>
      _field;
  using FieldView =
      Kokkos::View<type_real[num_edges][nquad_intersection][num_components],
                   memory_space>;

public:
  /**
   * @brief Construct field with initializer.
   * @param initializer Initialization strategy
   */
  IntersectionFunction2D(const FunctionInitializer &initializer) {
    _field = FunctionInitializer::init_function();
  }

  /**
   * @brief Get Kokkos view of field data.
   * @return Kokkos view for device access
   */
  FieldView get_view() const {
    FieldView view("field_view");
    for (size_t i = 0; i < num_edges; ++i) {
      for (size_t j = 0; j < nquad_intersection; ++j) {
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
