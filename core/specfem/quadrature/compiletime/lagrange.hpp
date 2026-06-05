#pragma once

#include "specfem/quadrature/compiletime/legendre.hpp"
#include "specfem/quadrature/compiletime/rational_polynomial.hpp"
#include "specfem/setup.hpp"

#include "impl/lagrange.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>
#include <utility>

namespace specfem::quadrature::compiletime {
template <typename NodeInitializer> struct Lagrange {

  /**
   * @brief Replaces xi Kokkos::View (size 0 struct with operator())
   *
   */
  struct Nodes {
  private:
    using initializer_node_type = decltype(NodeInitializer::get_nodes());

  public:
    using numerical_type =
        std::decay_t<decltype(NodeInitializer::get_nodes()[0])>;
    static constexpr std::size_t N =
        sizeof(initializer_node_type) / sizeof(numerical_type);

    static_assert(std::is_same_v<std::decay_t<initializer_node_type>,
                                 Kokkos::Array<numerical_type, N>>,
                  "NodeInitializer() should be type Kokkos::Array<T,N>");

    // bypass NodeInitializer to force compile time with an explicit consteval
    static consteval Kokkos::Array<numerical_type, N> get_nodes() {
      return NodeInitializer::get_nodes();
    }

  private:
    static constexpr Kokkos::Array<numerical_type, N> nodes = get_nodes();

  public:
    /**
     * @brief Retrieves the node corresponding to the given index at runtime.
     */
    static constexpr KOKKOS_INLINE_FUNCTION numerical_type
    node(const int &node_index) {
      constexpr auto nodes_ = nodes;
      // so for some reason this works, while directly referencing nodes has
      // nvcc crying
      return nodes_[node_index];
    }
    KOKKOS_INLINE_FUNCTION numerical_type
    operator()(const int &node_index) const {
      constexpr auto nodes_ = nodes;
      return nodes_[node_index];
    }
  };

  using numerical_type = Nodes::numerical_type;
  static constexpr std::size_t N = Nodes::N;

  // make_lagrange_coeff_table is already consteval.
  static constexpr auto coeff_table =
      impl::make_lagrange_coeff_table(Nodes::get_nodes());

public:
  Lagrange(const NodeInitializer &) {}

  static constexpr KOKKOS_INLINE_FUNCTION numerical_type
  node(const int &node_index) {
    return Nodes::node(node_index);
  }

  static constexpr KOKKOS_INLINE_FUNCTION numerical_type
  poly_coeff(const int &polynomial_index, const int &coefficient_index) {
    constexpr auto coeff_table_ = coeff_table;

    // same thing as Nodes::node(const int&)
    return coeff_table_[polynomial_index][coefficient_index];
  }

  /**
   * @brief Evaluate all five basis polynomials at @p x using Horner's method.
   * @param x  Evaluation point in @f$ [-1,1] @f$.
   * @param L  Output array; @c L[i] receives @f$ L_i(x) @f$.
   */
  template <typename SampleNumberType>
  KOKKOS_INLINE_FUNCTION static void
  eval_all(SampleNumberType x, Kokkos::Array<SampleNumberType, N> &L) {
    for (int i = 0; i < N; ++i) {
      float v = poly_coeff(i, N - 1);
      for (int k = N - 2; k >= 0; --k)
        v = v * x + poly_coeff(i, k);
      L[i] = v;
    }
  }
};

template <int NGLL, typename numerical_type = type_real>
struct gll_initializer {
  static consteval Kokkos::Array<numerical_type, NGLL> get_nodes() {
    Kokkos::Array<numerical_type, NGLL> knots;
    knots[0] = -1;

    using Lp =
        decltype(specfem::quadrature::compiletime::RationalPolynomialWithRoots(
            typename specfem::quadrature::compiletime::LegendrePolynomial<
                NGLL - 1>::derivative()));

    for (int i = 1; i < NGLL - 1; i++) {
      knots[i] = Lp::roots[i - 1];
    }

    knots[NGLL - 1] = 1;
    return knots;
  }
};

template <int NGLL> using gll = Lagrange<gll_initializer<NGLL>>;
} // namespace specfem::quadrature::compiletime
