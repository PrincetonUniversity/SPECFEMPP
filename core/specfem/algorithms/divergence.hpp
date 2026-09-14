#pragma once

#include "specfem/datatype.hpp"
#include "specfem/execution.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

/**
 * @file divergence.hpp
 * @brief Algorithms for computing divergence of tensor fields in spectral
 * elements
 * @ingroup AlgorithmsDivergence
 */

namespace specfem {
namespace algorithms {

/**
 * @brief Combine the three directional quadrature sums of a 3D divergence at
 * one quadrature point with the transverse weights.
 *
 * \f$ w_{iz} w_{iy} t_\xi + w_{iz} w_{ix} t_\eta + w_{iy} w_{ix} t_\gamma \f$
 * -- each direction's sum carries the two weights transverse to it (the
 * summed direction's own weight already rides inside the sum). This is the
 * single owner of the transverse-weight combine:
 * `specfem::algorithms::impl::element_divergence`'s result stage and any
 * kernel that produces the directional sums by other means (e.g. the
 * tensor-graph stiffness kernel's contractions) both delegate here.
 *
 * @ingroup AlgorithmsDivergence
 *
 * @tparam WeightsType Quadrature weights view
 * @tparam Datatype Scalar or SIMD datatype of the directional sums
 * @param weights Quadrature weights, one per GLL index
 * @param iz Quadrature index in z
 * @param iy Quadrature index in y
 * @param ix Quadrature index in x
 * @param t_xi Directional sum over \f$ \xi \f$
 * @param t_eta Directional sum over \f$ \eta \f$
 * @param t_gamma Directional sum over \f$ \gamma \f$
 * @return The weighted divergence value at the point
 */
template <typename WeightsType, typename Datatype>
KOKKOS_FORCEINLINE_FUNCTION Datatype transverse_weighted_sum(
    const WeightsType &weights, const int iz, const int iy, const int ix,
    const Datatype &t_xi, const Datatype &t_eta, const Datatype &t_gamma) {
  return weights(iz) * weights(iy) * t_xi + weights(iz) * weights(ix) * t_eta +
         weights(iy) * weights(ix) * t_gamma;
}

/// @brief Implementation details
namespace impl {
/**
 * @brief Compute the divergence of a tensor field at a specific point in a 2D
 * spectral element
 *
 * @tparam TensorFieldType Type of the tensor field (must be 2D)
 * @tparam WeightsType Type of the quadrature weights
 * @tparam QuadratureType Type of the Lagrange derivative polynomial
 * @param f Tensor field to compute divergence of
 * @param local_index Local indices within the spectral element
 * @param weights Quadrature weights for integration
 * @param lagrange_derivative Lagrange derivative polynomials for
 * differentiation
 * @return Auto-deduced return type containing the divergence values
 */
template <typename TensorFieldType, typename WeightsType,
          typename QuadratureType,
          typename std::enable_if_t<TensorFieldType::dimension_tag ==
                                        specfem::element::dimension_tag::dim2,
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

/**
 * @brief Compute the divergence of a tensor field at a specific point in a 3D
 * spectral element
 *
 * @tparam TensorFieldType Type of the tensor field (must be 3D)
 * @tparam WeightsType Type of the quadrature weights
 * @tparam QuadratureType Type of the Lagrange derivative polynomial
 * @param f Tensor field to compute divergence of
 * @param local_index Local indices within the spectral element
 * @param weights Quadrature weights for integration
 * @param lagrange_derivative Lagrange derivative polynomials for
 * differentiation
 * @return Auto-deduced return type containing the divergence values
 */
template <typename TensorFieldType, typename WeightsType,
          typename QuadratureType,
          typename std::enable_if_t<TensorFieldType::dimension_tag ==
                                        specfem::element::dimension_tag::dim3,
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
    result(icomp) = specfem::algorithms::transverse_weighted_sum(
        weights, iz, iy, ix, temp1l[icomp], temp2l[icomp], temp3l[icomp]);
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
