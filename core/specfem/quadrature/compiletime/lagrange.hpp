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

  private:
    static_assert(std::is_same_v<std::decay_t<initializer_node_type>,
                                 std::array<numerical_type, N>>,
                  "NodeInitializer() should be type std::array<T,N>");

    // bypass NodeInitializer to force compile time
    static consteval std::array<numerical_type, N> get_nodes() {
      return NodeInitializer::get_nodes();
    }

  public:
    static constexpr std::array<numerical_type, N> nodes = get_nodes();

    /**
     * @brief Retrieves the node corresponding to the given index at runtime.
     */
    KOKKOS_INLINE_FUNCTION numerical_type
    operator()(const int &node_index) const {
      return nodes[node_index];
    }
  };

  using numerical_type = Nodes::numerical_type;
  static constexpr std::size_t N = Nodes::N;

public:
  Lagrange(const NodeInitializer &) {}

  static constexpr std::array<numerical_type, N> nodes = Nodes::nodes;

  // consteval means this exists at compile time.
  static constexpr auto poly_coeffs =
      impl::make_lagrange_coeff_table(Nodes::nodes);

  /**
   * @brief Evaluate all five basis polynomials at @p x using Horner's method.
   * @param x  Evaluation point in @f$ [-1,1] @f$.
   * @param L  Output array; @c L[i] receives @f$ L_i(x) @f$.
   */
  template <typename SampleNumberType>
  KOKKOS_INLINE_FUNCTION static void
  eval_all(SampleNumberType x, std::array<SampleNumberType, N> &L) {
    for (int i = 0; i < N; ++i) {
      float v = poly_coeffs[i][N - 1];
      for (int k = N - 2; k >= 0; --k)
        v = v * x + poly_coeffs[i][k];
      L[i] = v;
    }
  }
};

template <int NGLL, typename numerical_type = type_real>
struct gll_initializer {
  static consteval std::array<numerical_type, NGLL> get_nodes() {
    std::array<numerical_type, NGLL> knots;
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
