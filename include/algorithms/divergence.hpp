#pragma once

#include "compute/compute_partial_derivatives.hpp"
#include "datatypes/point_view.hpp"
#include "point/coordinates.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {

/**
 * @defgroup AlgorithmsDivergence
 *
 */

/**
 * @brief Compute the divergence of a vector field f using the spectral element
 * formulation (eqn: A7 in Komatitsch and Tromp, 1999)
 *
 * @ingroup AlgorithmsDivergence
 *
 *
 * @tparam MemberType Kokkos team member type
 * @tparam IteratorType Iterator type (Chunk iterator)
 * @tparam MemberType Kokkos team member type
 * @tparam IteratorType Iterator type (Chunk iterator)
 * @tparam VectorFieldType Vector field view type (Chunk view)
 * @tparam QuadratureType Quadrature view type
 * @tparam CallableType Callback functor type
 * @param team Kokkos team member
 * @param iterator Chunk iterator
 * @param partial_derivatives Partial derivatives of basis functions
 * @param weights Weights for the quadrature
 * @param hprimewgll Integration quadrature
 * @param f Field to compute the divergence of
 * @param callback Callback functor. Callback signature must be:
 * @code void(const typename IteratorType::index_type, const
 * specfem::datatype::ScalarPointViewType<type_real, ViewType::components>)
 * @endcode
 */
template <typename MemberType, typename IteratorType, typename VectorFieldType,
          typename QuadratureType, typename CallableType,
          std::enable_if_t<(VectorFieldType::isChunkViewType), int> = 0>
NOINLINE KOKKOS_FUNCTION void divergence(
    const MemberType &team, const IteratorType &iterator,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const Kokkos::View<type_real *,
                       typename MemberType::execution_space::memory_space>
        &weights,
    const QuadratureType &hprimewgll, const VectorFieldType &f,
    CallableType callback) {

  constexpr int components = VectorFieldType::components;
  constexpr int NGLL = VectorFieldType::ngll;
  constexpr static bool using_simd = VectorFieldType::simd::using_simd;

  constexpr bool is_host_space =
      std::is_same<typename MemberType::execution_space::memory_space,
                   Kokkos::HostSpace>::value;

  using ScalarPointViewType =
      specfem::datatype::ScalarPointViewType<type_real, components, using_simd>;

  static_assert(VectorFieldType::isVectorViewType,
                "ViewType must be a vector field view type");

  static_assert(
      std::is_invocable_v<CallableType, typename IteratorType::index_type,
                          ScalarPointViewType>,
      "CallableType must be invocable with arguments (int, "
      "specfem::point::index, "
      "specfem::datatype::ScalarPointViewType<type_real, components>)");

  using datatype = typename IteratorType::simd::datatype;

  // Compute the integral
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, iterator.chunk_size()), [&](const int i) {
        const auto iterator_index = iterator(i);
        const int ielement = iterator_index.ielement;
        const int ispec = iterator_index.index.ispec;
        const int iz = iterator_index.index.iz;
        const int ix = iterator_index.index.ix;

        const datatype jacobian =
            (is_host_space) ? partial_derivatives.h_jacobian(ispec, iz, ix)
                            : partial_derivatives.jacobian(ispec, iz, ix);

        datatype temp1l[components] = { 0.0 };
        datatype temp2l[components] = { 0.0 };

        for (int l = 0; l < NGLL; ++l) {
          for (int icomp = 0; icomp < components; ++icomp) {
            temp1l[icomp] +=
                f(ielement, iz, l, 0, icomp) * hprimewgll(ix, l) * jacobian;
          }
          for (int icomp = 0; icomp < components; ++icomp) {
            temp2l[icomp] +=
                f(ielement, l, ix, 1, icomp) * hprimewgll(iz, l) * jacobian;
          }
        }

        ScalarPointViewType result;

        for (int icomp = 0; icomp < components; ++icomp) {
          result(icomp) =
              weights(iz) * temp1l[icomp] + weights(ix) * temp2l[icomp];
        }

        callback(iterator_index, result);
      });

  return;
}

} // namespace algorithms
} // namespace specfem
