#pragma once

#include "datatypes/point_view.hpp"
#include "execution/for_each_level.hpp"
#include "specfem/assembly.hpp"
#include "specfem/point.hpp"
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
 * @tparam ChunkIndexType Chunk index type
 * @tparam MemberType Kokkos team member type
 * @tparam IteratorType Iterator type (Chunk iterator)
 * @tparam VectorFieldType Vector field view type (Chunk view)
 * @tparam QuadratureType Quadrature view type
 * @tparam CallableType Callback functor type
 * @param chunk_index Chunk index specifying the elements within this chunk
 * @param jacobian_matrix Jacobian matrix of basis functions
 * @param weights Weights for the quadrature
 * @param hprimewgll Integration quadrature
 * @param f Field to compute the divergence of
 * @param callback Callback functor. Callback signature must be:
 * @code void(const typename IteratorType::index_type, const
 * specfem::datatype::VectorPointViewType<type_real, ViewType::components>)
 * @endcode
 */
template <
    typename ChunkIndexType, typename VectorFieldType, typename WeightsType,
    typename QuadratureType, typename CallableType,
    std::enable_if_t<VectorFieldType::accessor_type ==
                         specfem::data_access::AccessorType::chunk_element,
                     int> = 0>
KOKKOS_FUNCTION void divergence(
    const ChunkIndexType &chunk_index,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    const WeightsType &weights, const QuadratureType &hprimewgll,
    const VectorFieldType &f, const CallableType &callback) {

  constexpr int components = VectorFieldType::components;
  constexpr int NGLL = VectorFieldType::ngll;
  constexpr static bool using_simd = VectorFieldType::simd::using_simd;

  using VectorPointViewType =
      specfem::datatype::VectorPointViewType<type_real, components, using_simd>;

  static_assert(
      std::is_invocable_v<CallableType,
                          typename ChunkIndexType::iterator_type::index_type,
                          VectorPointViewType>,
      "CallableType must be invocable with arguments (int, "
      "specfem::point::index, "
      "specfem::datatype::VectorPointViewType<type_real, components>)");

  using simd = typename VectorFieldType::simd;
  using datatype = typename VectorFieldType::simd::datatype;
  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true,
                                      using_simd>;

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto ielement = iterator_index.get_local_index().ispec;
        const auto index = iterator_index.get_index();
        const int iz = index.iz;
        const int ix = index.ix;

        datatype temp1l[components] = { 0.0 };
        datatype temp2l[components] = { 0.0 };

        /// We omit the divergence here since we multiplied it when computing F.
        for (int l = 0; l < NGLL; ++l) {
          for (int icomp = 0; icomp < components; ++icomp) {
            temp1l[icomp] += f(ielement, iz, l, icomp, 0) * hprimewgll(ix, l);
          }
          for (int icomp = 0; icomp < components; ++icomp) {
            temp2l[icomp] += f(ielement, l, ix, icomp, 1) * hprimewgll(iz, l);
          }
        }
        VectorPointViewType result;
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
