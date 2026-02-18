#pragma once

#include "../impl/descriptions.hpp"
#include "initializers.hpp"
#include "specfem/setup.hpp"

#include <type_traits>

namespace specfem::test_fixture {

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

  static std::string description() { return "ngll = 5 field of all ones"; }
  static std::string name() { return "Uniform(1,5,1)"; }
};

template <typename AnalyticalFunction, typename EdgePoints>
struct FromAnalyticalFunction : EdgeFunctionInitializer2D {
  static constexpr bool is_from_analytical_function = true;
  static_assert(
      std::is_base_of_v<AnalyticalFunctionType::AnalyticalFunctionType,
                        AnalyticalFunction>,
      "FromAnalyticalFunction expects its first template argument to be "
      "an AnalyticalFunctionType1D!");
  using AnalyticalFunctionType = AnalyticalFunction;

  using EdgeQuadraturePoints = EdgePoints;

  static constexpr int num_edges = 1;
  static constexpr int num_components = AnalyticalFunction::num_components;
  static constexpr int nquad_edge = EdgeQuadraturePoints::nquad;
  static constexpr auto edge_quadrature_points =
      EdgeQuadraturePoints::quadrature_points;

  static std::string description() {
    /*    Format:
     *
     * Edge Function from analytical function:
     *   AnalyticalFunction description with this indent
     *
     */
    return std::string("Edge Function from analytical function:\n") +
           specfem::test_fixture::impl::description<
               AnalyticalFunctionType>::get(2);
  }
  static std::string name() {
    return std::string("FromAnalytical(") +
           specfem::test_fixture::impl::name<AnalyticalFunctionType>::get() +
           ")";
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
        for (size_t k = 0; k < num_components; ++k) {
          edge_function[i][j][k] = eval[k];
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

  static std::string description(const int &indent = 0) {
    return specfem::test_fixture::impl::description<Initializer>::get(indent);
  }
  static std::string initializer_name() {
    return specfem::test_fixture::impl::name<Initializer>::get();
  }

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
    auto host_view =
        Kokkos::create_mirror_view(view); // Create host mirror to copy data
    for (size_t i = 0; i < num_edges; ++i) {
      for (size_t j = 0; j < nquad_edge; ++j) {
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
} // namespace specfem::test_fixture
