#pragma once

#include "execution/for_each_level.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {

/**
 * @defgroup AlgorithmsGradient
 *
 */

/**
 * @brief Compute the gradient of a scalar field f using the spectral element
 * formulation (eqn: 29 in Komatitsch and Tromp, 1999)
 *
 * @ingroup AlgorithmsGradient
 *
 * @tparam ChunkIndexType Chunk index type
 * @tparam ViewType Field view type (Chunk view)
 * @tparam QuadratureType Quadrature view type
 * @tparam CallbackFunctor Callback functor type
 * @param chunk_index Chunk index specifying the elements within this chunk
 * @param jacobian_matrix Jacobian matrix of basis functions
 * @param quadrature Integration quadrature
 * @param f Field to compute the gradient of
 * @param callback Callback functor. Callback signature must be:
 * @code void(const typename IteratorType::index_type, const
 * specfem::datatype::TensorPointViewType<type_real, 2, ViewType::components>)
 * @endcode
 */
template <typename ChunkIndexType, typename ViewType, typename QuadratureType,
          typename CallbackFunctor,
          std::enable_if_t<ViewType::isChunkViewType, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void gradient(
    const ChunkIndexType &chunk_index,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    const QuadratureType &quadrature, const ViewType &f,
    const CallbackFunctor &callback) {
  constexpr int components = ViewType::components;
  constexpr bool using_simd = ViewType::simd::using_simd;
  constexpr int dimension = 2;

  constexpr int NGLL = ViewType::ngll;

  using TensorPointViewType =
      specfem::datatype::TensorPointViewType<type_real, components, dimension,
                                             using_simd>;

  using datatype = typename ViewType::simd::datatype;

  static_assert(ViewType::isScalarViewType,
                "ViewType must be a scalar field view type");

  static_assert(
      std::is_invocable_v<CallbackFunctor,
                          typename ChunkIndexType::iterator_type::index_type,
                          TensorPointViewType>,
      "CallbackFunctor must be invocable with the following signature: "
      "void(const int, const specfem::point::index, const "
      "specfem::kokkos::array_type<type_real, components>, const "
      "specfem::kokkos::array_type<type_real, components>)");

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto ielement = iterator_index.get_local_index().ispec;
        const auto point_index = iterator_index.get_index();
        const int ispec = point_index.ispec;
        const int iz = point_index.iz;
        const int ix = point_index.ix;

        datatype df_dxi[components] = { 0.0 };
        datatype df_dgamma[components] = { 0.0 };

        for (int l = 0; l < NGLL; ++l) {
          for (int icomponent = 0; icomponent < components; ++icomponent) {
            df_dxi[icomponent] +=
                quadrature(ix, l) * f(ielement, iz, l, icomponent);
            df_dgamma[icomponent] +=
                quadrature(iz, l) * f(ielement, l, ix, icomponent);
          }
        }

        specfem::point::jacobian_matrix<specfem::dimension::type::dim2, false,
                                        using_simd>
            point_jacobian_matrix;

        specfem::assembly::load_on_device(point_index, jacobian_matrix,
                                          point_jacobian_matrix);
        TensorPointViewType df;
        for (int icomponent = 0; icomponent < components; ++icomponent) {
          df(icomponent, 0) =
              point_jacobian_matrix.xix * df_dxi[icomponent] +
              point_jacobian_matrix.gammax * df_dgamma[icomponent];

          df(icomponent, 1) =
              point_jacobian_matrix.xiz * df_dxi[icomponent] +
              point_jacobian_matrix.gammaz * df_dgamma[icomponent];
        }

        callback(iterator_index, df);
      });

  return;
}

/**
 * @brief Compute the gradient of a field f & g using the spectral element
 * formulation (eqn: 29 in Komatitsch and Tromp, 1999)
 *
 * @ingroup AlgorithmsGradient
 *
 * @tparam ChunkIndexType Chunk index type
 * @tparam ViewType Field view type (Chunk view)
 * @tparam QuadratureType Quadrature view type
 * @tparam CallbackFunctor Callback functor type
 * @param chunk_index Chunk index specifying the elements within this chunk
 * @param jacobian_matrix Jacobian matrix of basis functions
 * @param quadrature Integration quadrature
 * @param f Field to compute the gradient of
 * @param g Field to compute the gradient of
 * @param callback Callback functor. Callback signature must be:
 * @code void(const typename IteratorType::index_type, const
 * specfem::datatype::TensorPointViewType<type_real, 2, ViewType::components>,
 * const specfem::datatype::TensorPointViewType<type_real, 2,
 * ViewType::components>)
 * @endcode
 */
template <typename ChunkIndexType, typename ViewType, typename QuadratureType,
          typename CallbackFunctor,
          std::enable_if_t<ViewType::isChunkViewType, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void gradient(
    const ChunkIndexType &chunk_index,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    const QuadratureType &quadrature, const ViewType &f, const ViewType &g,
    const CallbackFunctor &callback) {
  constexpr int components = ViewType::components;
  constexpr bool using_simd = ViewType::simd::using_simd;
  constexpr int dimension = 2;

  constexpr int NGLL = ViewType::ngll;

  using TensorPointViewType =
      specfem::datatype::TensorPointViewType<type_real, components, dimension,
                                             using_simd>;

  static_assert(ViewType::isScalarViewType,
                "ViewType must be a scalar field view type");

  using datatype = typename ViewType::simd::datatype;

  static_assert(
      std::is_invocable_v<CallbackFunctor,
                          typename ChunkIndexType::iterator_type::index_type,
                          TensorPointViewType, TensorPointViewType>,
      "CallbackFunctor must be invocable with the following signature: "
      "void(const ChunkIndexType::iterator_type::index_type, "
      "const specfem::datatype::TensorPointViewType<type_real, 2, components>, "
      "const specfem::datatype::TensorPointViewType<type_real, 2, "
      "components>)");

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto ielement = iterator_index.get_local_index().ispec;
        const auto point_index = iterator_index.get_index();
        const int ispec = point_index.ispec;
        const int iz = point_index.iz;
        const int ix = point_index.ix;

        datatype df_dxi[components] = { 0.0 };
        datatype df_dgamma[components] = { 0.0 };

        for (int l = 0; l < NGLL; ++l) {
          for (int icomponent = 0; icomponent < components; ++icomponent) {
            df_dxi[icomponent] +=
                quadrature(ix, l) * f(ielement, iz, l, icomponent);
            df_dgamma[icomponent] +=
                quadrature(iz, l) * f(ielement, l, ix, icomponent);
          }
        }

        specfem::point::jacobian_matrix<specfem::dimension::type::dim2, false,
                                        using_simd>
            point_jacobian_matrix;

        specfem::assembly::load_on_device(point_index, jacobian_matrix,
                                          point_jacobian_matrix);

        TensorPointViewType df;

        for (int icomponent = 0; icomponent < components; ++icomponent) {
          df(icomponent, 0) =
              point_jacobian_matrix.xix * df_dxi[icomponent] +
              point_jacobian_matrix.gammax * df_dgamma[icomponent];

          df(icomponent, 1) =
              point_jacobian_matrix.xiz * df_dxi[icomponent] +
              point_jacobian_matrix.gammaz * df_dgamma[icomponent];
        }

        for (int icomponent = 0; icomponent < components; ++icomponent) {
          df_dxi[icomponent] = 0.0;
          df_dgamma[icomponent] = 0.0;
        }

        for (int l = 0; l < NGLL; ++l) {
          for (int icomponent = 0; icomponent < components; ++icomponent) {
            df_dxi[icomponent] +=
                quadrature(ix, l) * g(ielement, iz, l, icomponent);
            df_dgamma[icomponent] +=
                quadrature(iz, l) * g(ielement, l, ix, icomponent);
          }
        }

        TensorPointViewType dg;
        for (int icomponent = 0; icomponent < components; ++icomponent) {
          dg(icomponent, 0) =
              point_jacobian_matrix.xix * df_dxi[icomponent] +
              point_jacobian_matrix.gammax * df_dgamma[icomponent];

          dg(icomponent, 1) =
              point_jacobian_matrix.xiz * df_dxi[icomponent] +
              point_jacobian_matrix.gammaz * df_dgamma[icomponent];
        }
        callback(iterator_index, df, dg);
      });

  return;
}
} // namespace algorithms
} // namespace specfem
