#pragma once

#include "../impl/descriptions.hpp"
#include "initializers.hpp"
#include "specfem/setup.hpp"

#include <type_traits>

// TODO consider merging with EdgeFunction

namespace specfem::test_fixture {

/**
 * @brief Types for IntersectionFunction2D. These contain the compile-time
 * values of the Intersection function
 *
 */
namespace IntersectionFunctionInitializer2D {

struct Uniform : IntersectionFunctionInitializer2D {
  static constexpr int num_edges = 1;
  static constexpr int num_components = 1;
  static constexpr int nquad_intersection = 5;

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
          intersection_function[i][j][k] = 1;
        }
      }
    }
    return intersection_function;
  }

  static std::string description() {
    return std::string("ngll = 5 field of all ones");
  }
  static std::string name() { return "Uniform(1,5,1)"; }
};

template <typename AnalyticalFunction, typename IntersectionPoints>
struct FromAnalyticalFunction : IntersectionFunctionInitializer2D {
  static constexpr bool is_from_analytical_function = true;
  static_assert(
      std::is_base_of_v<AnalyticalFunctionType::AnalyticalFunctionType,
                        AnalyticalFunction>,
      "FromAnalyticalFunction expects its first template argument to be "
      "an AnalyticalFunctionType1D!");
  using AnalyticalFunctionType = AnalyticalFunction;

  using IntersectionQuadraturePoints = IntersectionPoints;

  static constexpr int num_edges = 1;
  static constexpr int num_components = AnalyticalFunction::num_components;
  static constexpr int nquad_intersection = IntersectionQuadraturePoints::nquad;
  static constexpr auto Intersection_quadrature_points =
      IntersectionQuadraturePoints::quadrature_points;

  static std::string name() {
    return std::string("FromAnalyticalFunction(") +
           specfem::test_fixture::impl::name<AnalyticalFunction>::get() + ")";
  }
  static std::string description() {
    return std::string("IntersectionField-initialized function:\n") +
           specfem::test_fixture::impl::description<AnalyticalFunction>::get(2);
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
            AnalyticalFunction::evaluate(Intersection_quadrature_points[j]);
        for (size_t k = 0; k < num_components; ++k) {
          intersection_function[i][j][k] = eval[k];
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
    auto host_view =
        Kokkos::create_mirror_view(view); // Create host mirror to copy data
    for (size_t i = 0; i < num_edges; ++i) {
      for (size_t j = 0; j < nquad_intersection; ++j) {
        for (size_t k = 0; k < num_components; ++k) {
          host_view(i, j, k) = (*this)(i, j, k);
        }
      }
    }
    Kokkos::deep_copy(view, host_view);
    return view;
  }

  /**
   * @brief Access field values.
   * @param i Intersection index
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
} // namespace specfem::test_fixture
