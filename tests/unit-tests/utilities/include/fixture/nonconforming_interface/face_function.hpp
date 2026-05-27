#pragma once

#include "../impl/descriptions.hpp"
#include "initializers.hpp"
#include "specfem/setup.hpp"
#include <tuple>
#include <type_traits>

namespace specfem::test_fixture {

/**
 * @brief Types for FaceFunction3D. These contain the compile-time values of the
 * edge function
 *
 */
namespace FaceFunctionInitializer3D {

template <typename AnalyticalFunction, typename EdgePoints>
struct FromAnalyticalFunction : FaceFunctionInitializer3D {
  static constexpr bool is_from_analytical_function = true;
  static_assert(
      std::is_base_of_v<AnalyticalFunctionType::AnalyticalFunctionType,
                        AnalyticalFunction> &&
          AnalyticalFunction::num_arguments == 2,
      "FromAnalyticalFunction expects its first template argument to be "
      "an AnalyticalFunctionType2D!");
  using AnalyticalFunctionType = AnalyticalFunction;

  // edge, since we assume both edges use same rule
  using EdgeQuadraturePoints = EdgePoints;

  static constexpr int stack_size = 1;
  static constexpr int num_components = AnalyticalFunction::num_components;
  static constexpr int nquad_element = EdgeQuadraturePoints::nquad;
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
      std::array<std::array<std::array<std::array<type_real, num_components>,
                                       nquad_element>,
                            nquad_element>,
                 stack_size>;

public:
  static ArrayType init_function() {
    ArrayType face_function;
    for (size_t i = 0; i < stack_size; ++i) {
      for (size_t j = 0; j < nquad_element; ++j) {
        for (size_t ell = 0; ell < nquad_element; ++ell) {
          const auto eval = AnalyticalFunction::evaluate(
              edge_quadrature_points[j], edge_quadrature_points[ell]);
          for (size_t k = 0; k < num_components; ++k) {
            face_function[i][j][ell][k] = eval[k];
          }
        }
      }
    }
    return face_function;
  }
};

template <typename... FaceFunctionInitializers>
struct Stack : FaceFunctionInitializer3D {
  static constexpr bool is_stacked_initializer = true;

private:
  using FirstFaceFunctionInitializer =
      std::tuple_element_t<0, std::tuple<FaceFunctionInitializers...>>;

public:
  static constexpr int stack_size =
      ((FaceFunctionInitializers::stack_size) + ...);
  static constexpr int num_components =
      FirstFaceFunctionInitializer::num_components;
  static constexpr int nquad_element =
      FirstFaceFunctionInitializer::nquad_element;

  static_assert(
      ((std::is_base_of_v<FaceFunctionInitializer3D,
                          FaceFunctionInitializers>) &&
       ...),
      "Stack needs all of its parameters to be of FaceFunctionInitializer3D!");
  static_assert(
      ((FaceFunctionInitializers::num_components == num_components) && ...),
      "Stack needs all of its parameters to have same num_components!");
  static_assert(
      ((FaceFunctionInitializers::nquad_element == nquad_element) && ...),
      "Stack needs all of its parameters to have same nquad_element!");

  static std::string description() {
    return std::string("Stack (") +
           std::to_string(sizeof...(FaceFunctionInitializers)) +
           " initializers)\n" +
           ((specfem::test_fixture::impl::description<
                 FaceFunctionInitializers>::get(2) +
             "\n") +
            ...);
  }
  static std::string name() {
    return std::string("Stack(") +
           ((specfem::test_fixture::impl::name<
                 FaceFunctionInitializers>::get() +
             ",") +
            ...) +
           ")";
  }

private:
  using ArrayType =
      std::array<std::array<std::array<std::array<type_real, num_components>,
                                       nquad_element>,
                            nquad_element>,
                 stack_size>;

public:
  static ArrayType init_function() {
    ArrayType face_function;
    auto it = face_function.begin();
    (
        [&]() {
          const auto sub = FaceFunctionInitializers::init_function();
          std::copy(sub.begin(), sub.end(), it);
          it += FaceFunctionInitializers::stack_size;
        }(),
        ...);

    return face_function;
  }
};

} // namespace FaceFunctionInitializer3D

/**
 * @brief Test field container.
 * @tparam Initializer Field initialization strategy
 */
template <typename Initializer> struct FaceFunction3D {
  static_assert(
      std::is_base_of_v<FaceFunctionInitializer3D::FaceFunctionInitializer3D,
                        Initializer>,
      "FaceFunction3D needs an FaceFunctionInitializer3D!");

public:
  using FunctionInitializer = Initializer;
  static constexpr int stack_size = Initializer::stack_size;
  static constexpr int num_components = Initializer::num_components;
  static constexpr int nquad_element = Initializer::nquad_element;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  static std::string description(const int &indent = 0) {
    return specfem::test_fixture::impl::description<Initializer>::get(indent);
  }
  static std::string initializer_name() {
    return specfem::test_fixture::impl::name<Initializer>::get();
  }

private:
  std::array<std::array<std::array<std::array<type_real, num_components>,
                                   nquad_element>,
                        nquad_element>,
             stack_size>
      _field;
  using FieldView = Kokkos::View<
      type_real[stack_size][nquad_element][nquad_element][num_components],
      memory_space>;
  using ResizedFieldView =
      Kokkos::View<type_real *[nquad_element][nquad_element][num_components],
                   memory_space>;

public:
  /**
   * @brief Construct field with initializer.
   * @param initializer Initialization strategy
   */
  FaceFunction3D(const FunctionInitializer &initializer) {
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
    for (size_t i = 0; i < stack_size; ++i) {
      for (size_t j = 0; j < nquad_element; ++j) {
        for (size_t ell = 0; ell < nquad_element; ++ell) {
          for (size_t k = 0; k < num_components; ++k) {
            host_view(i, j, ell, k) = (*this)(i, j, ell, k);
          }
        }
      }
    }
    Kokkos::deep_copy(view, host_view);
    return view;
  }

  /**
   * @brief Get Kokkos view of field data, set to different size.
   *
   * This is used to pad the View so that each chunk can get a full-sized
   * subview to mock the scratch view.
   *
   * @return Kokkos view for device access
   */
  ResizedFieldView get_resized_view(const int &stack_size) const {
    ResizedFieldView view("field_view", stack_size);
    auto host_view =
        Kokkos::create_mirror_view(view); // Create host mirror to copy data
    for (size_t i = 0; i < std::min(stack_size, this->stack_size); ++i) {
      for (size_t j = 0; j < nquad_element; ++j) {
        for (size_t ell = 0; ell < nquad_element; ++ell) {
          for (size_t k = 0; k < num_components; ++k) {
            host_view(i, j, ell, k) = (*this)(i, j, ell, k);
          }
        }
      }
    }
    Kokkos::deep_copy(view, host_view);
    return view;
  }

  template <int new_stack_size>
  Kokkos::View<
      type_real[new_stack_size][nquad_element][nquad_element][num_components],
      memory_space>
  get_resized_view() const {
    Kokkos::View<
        type_real[new_stack_size][nquad_element][nquad_element][num_components],
        memory_space>
        view("field_view");
    auto host_view =
        Kokkos::create_mirror_view(view); // Create host mirror to copy data
    for (size_t i = 0; i < std::min(new_stack_size, this->stack_size); ++i) {
      for (size_t j = 0; j < nquad_element; ++j) {
        for (size_t ell = 0; ell < nquad_element; ++ell) {
          for (size_t k = 0; k < num_components; ++k) {
            host_view(i, j, ell, k) = (*this)(i, j, ell, k);
          }
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
  type_real &operator()(const int i, const int j, const int ell, const int k) {
    return _field[i][j][ell][k];
  }
  const type_real &operator()(const int i, const int j, const int ell,
                              const int k) const {
    return _field[i][j][ell][k];
  }
};
} // namespace specfem::test_fixture
