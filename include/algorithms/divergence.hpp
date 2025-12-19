#pragma once

#include "datatypes/point_view.hpp"
#include "execution/for_each_level.hpp"
#include "specfem/assembly.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {

namespace impl {
template <typename TensorFieldType, typename WeightsType,
          typename QuadratureType,
          typename std::enable_if_t<TensorFieldType::dimension_tag ==
                                        specfem::dimension::type::dim2,
                                    int> = 0>
KOKKOS_FORCEINLINE_FUNCTION auto
element_divergence(const TensorFieldType &f,
                   const typename TensorFieldType::index_type &local_index,
                   const WeightsType &weights,
                   const QuadratureType &lagrange_derivative) {
  using datatype = typename TensorFieldType::simd::datatype;

  using VectorPointViewType = specfem::datatype::VectorPointViewType<
      type_real, TensorFieldType::components, TensorFieldType::using_simd>;
  const int iz = local_index.iz;
  const int ix = local_index.ix;
  const int ielement = local_index.ispec;
  constexpr int components = TensorFieldType::components;
  constexpr int ngll = TensorFieldType::ngll;

  datatype temp1l[components] = { 0.0 };
  datatype temp2l[components] = { 0.0 };

  /// We omit the divergence here since we multiplied it when computing F.
  for (int l = 0; l < ngll; ++l) {
    for (int icomp = 0; icomp < components; ++icomp) {
      temp1l[icomp] += f(ielement, iz, l, icomp, 0) *
                       lagrange_derivative.xi(l, ix) * weights(l);
    }
    for (int icomp = 0; icomp < components; ++icomp) {
      temp2l[icomp] += f(ielement, l, ix, icomp, 1) *
                       lagrange_derivative.gamma(l, iz) * weights(l);
    }
  }
  VectorPointViewType result;
  for (int icomp = 0; icomp < components; ++icomp) {
    result(icomp) = weights(iz) * temp1l[icomp] + weights(ix) * temp2l[icomp];
  }
  return result;
}

template <typename TensorFieldType, typename WeightsType,
          typename QuadratureType,
          typename std::enable_if_t<TensorFieldType::dimension_tag ==
                                        specfem::dimension::type::dim3,
                                    int> = 0>
KOKKOS_FORCEINLINE_FUNCTION auto
element_divergence(const TensorFieldType &f,
                   const typename TensorFieldType::index_type &local_index,
                   const WeightsType &weights,
                   const QuadratureType &lagrange_derivative) {
  using datatype = typename TensorFieldType::simd::datatype;

  using VectorPointViewType = specfem::datatype::VectorPointViewType<
      type_real, TensorFieldType::components, TensorFieldType::using_simd>;
  const int iz = local_index.iz;
  const int iy = local_index.iy;
  const int ix = local_index.ix;
  const int ielement = local_index.ispec;
  constexpr int components = TensorFieldType::components;
  constexpr int ngll = TensorFieldType::ngll;

  datatype temp1l[components] = { 0.0 };
  datatype temp2l[components] = { 0.0 };
  datatype temp3l[components] = { 0.0 };

  /// We omit the divergence here since we multiplied it when computing F.
  for (int l = 0; l < ngll; ++l) {
    for (int icomp = 0; icomp < components; ++icomp) {
      temp1l[icomp] += f(ielement, iz, iy, l, icomp, 0) *
                       lagrange_derivative.xi(l, ix) * weights(l);
    }
    for (int icomp = 0; icomp < components; ++icomp) {
      temp2l[icomp] += f(ielement, iz, l, ix, icomp, 1) *
                       lagrange_derivative.eta(l, iy) * weights(l);
    }
    for (int icomp = 0; icomp < components; ++icomp) {
      temp3l[icomp] += f(ielement, l, iy, ix, icomp, 2) *
                       lagrange_derivative.gamma(l, iz) * weights(l);
    }
  }
  VectorPointViewType result;
  for (int icomp = 0; icomp < components; ++icomp) {
    result(icomp) = weights(iz) * weights(iy) * temp1l[icomp] +
                    weights(iz) * weights(ix) * temp2l[icomp] +
                    weights(iy) * weights(ix) * temp3l[icomp];
  }
  return result;
}
} // namespace impl

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
 * @tparam ChunkIndexType Chunk index type
 * @tparam MemberType Kokkos team member type
 * @tparam IteratorType Iterator type (Chunk iterator)
 * @tparam TensorFieldType Vector field view type (Chunk view)
 * @tparam QuadratureType Quadrature view type
 * @tparam CallableType Callback functor type
 * @param chunk_index Chunk index specifying the elements within this chunk
 * @param jacobian_matrix Jacobian matrix of basis functions
 * @param weights Weights for the quadrature
 * @param hprime Integration quadrature
 * @param f Field to compute the divergence of
 * @param callback Callback functor. Callback signature must be:
 * @code void(const typename IteratorType::index_type, const
 * specfem::datatype::VectorPointViewType<type_real, ViewType::components>)
 * @endcode
 */
template <typename ChunkIndexType, typename TensorFieldType,
          typename WeightsType, typename QuadratureType, typename CallableType,
          std::enable_if_t<
              specfem::data_access::is_chunk_element<TensorFieldType>::value,
              int> = 0>
KOKKOS_FUNCTION void
divergence(const ChunkIndexType &chunk_index, const WeightsType &weights,
           const QuadratureType &hprime, const TensorFieldType &f,
           const CallableType &callback) {

  using VectorPointViewType = specfem::datatype::VectorPointViewType<
      type_real, TensorFieldType::components, TensorFieldType::using_simd>;

  static_assert(
      std::is_invocable_v<CallableType,
                          typename ChunkIndexType::iterator_type::index_type,
                          VectorPointViewType>,
      "CallableType must be invocable with arguments (int, "
      "specfem::point::index, "
      "specfem::datatype::VectorPointViewType<type_real, components>)");

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto local_index = iterator_index.get_local_index();
        const auto result =
            impl::element_divergence(f, local_index, weights, hprime);
        callback(iterator_index, result);
      });

  return;
}

} // namespace algorithms
} // namespace specfem
