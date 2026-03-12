#pragma once

#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/chunk_element/field_pack.hpp"
#include "specfem/data_access.hpp"
#include "specfem/execution.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <tuple>

/**
 * @file gradient.hpp
 * @brief Algorithms for computing gradients of vector fields in spectral
 * elements
 * @ingroup AlgorithmsGradient
 */

namespace specfem {
namespace algorithms {
/// @brief Implementation details
namespace impl {
/**
 * @brief Compute the gradient of a vector field at a specific point in a 2D
 * spectral element
 *
 * @tparam VectorFieldType Type of the vector field (must be 2D)
 * @tparam QuadratureType Type of the Lagrange derivative polynomial
 * @param f Vector field to compute gradient of
 * @param local_index Local indices within the spectral element
 * @param point_jacobian_matrix Jacobian matrix for coordinate transformation
 * @param lagrange_derivative Lagrange derivative polynomials for
 * differentiation
 * @param df_dxi Output array for derivatives with respect to xi
 * @param df_dgamma Output array for derivatives with respect to gamma
 * @return Auto-deduced return type containing the gradient values
 */
template <typename VectorFieldType, typename QuadratureType,
          typename std::enable_if_t<VectorFieldType::dimension_tag ==
                                        specfem::element::dimension_tag::dim2,
                                    int> = 0>
KOKKOS_FORCEINLINE_FUNCTION auto element_gradient(
    const VectorFieldType &f,
    const specfem::point::index<specfem::element::dimension_tag::dim2,
                                VectorFieldType::using_simd> &local_index,
    const specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2,
                                          false, VectorFieldType::using_simd>
        &point_jacobian_matrix,
    const QuadratureType &lagrange_derivative,
    typename VectorFieldType::simd::datatype (
        &df_dxi)[VectorFieldType::components],
    typename VectorFieldType::simd::datatype (
        &df_dgamma)[VectorFieldType::components]) {

  constexpr int dimension = 2;
  constexpr int components = VectorFieldType::components;
  constexpr int ngll = VectorFieldType::ngll;
  using TensorPointViewType = specfem::datatype::TensorPointViewType<
      type_real, VectorFieldType::components, dimension,
      VectorFieldType::simd::using_simd>;
  const int ielement = local_index.ispec;
  const int iz = local_index.iz;
  const int ix = local_index.ix;

  for (int l = 0; l < ngll; ++l) {
    for (int icomponent = 0; icomponent < components; ++icomponent) {
      df_dxi[icomponent] +=
          lagrange_derivative.xi(ix, l) * f(ielement, iz, l, icomponent);
      df_dgamma[icomponent] +=
          lagrange_derivative.gamma(iz, l) * f(ielement, l, ix, icomponent);
    }
  }

  TensorPointViewType df;

  for (int icomponent = 0; icomponent < components; ++icomponent) {
    df(icomponent, 0) = point_jacobian_matrix.xix * df_dxi[icomponent] +
                        point_jacobian_matrix.gammax * df_dgamma[icomponent];

    df(icomponent, 1) = point_jacobian_matrix.xiz * df_dxi[icomponent] +
                        point_jacobian_matrix.gammaz * df_dgamma[icomponent];
  }
  return df;
}
/**
 * @brief Compute the gradient of a vector field at a specific point in a 3D
 * spectral element
 *
 * @tparam VectorFieldType Type of the vector field (must be 3D)
 * @tparam QuadratureType Type of the Lagrange derivative polynomial
 * @param f Vector field to compute gradient of
 * @param local_index Local indices within the spectral element
 * @param point_jacobian_matrix Jacobian matrix for coordinate transformation
 * @param lagrange_derivative Lagrange derivative polynomials for
 * differentiation
 * @param df_dxi Output array for derivatives with respect to xi
 * @param df_deta Output array for derivatives with respect to eta
 * @param df_dgamma Output array for derivatives with respect to gamma
 * @return Auto-deduced return type containing the gradient values
 */
template <typename VectorFieldType, typename QuadratureType,
          typename std::enable_if_t<VectorFieldType::dimension_tag ==
                                        specfem::element::dimension_tag::dim3,
                                    int> = 0>
KOKKOS_FORCEINLINE_FUNCTION auto element_gradient(
    const VectorFieldType &f,
    const specfem::point::index<specfem::element::dimension_tag::dim3,
                                VectorFieldType::using_simd> &local_index,
    const specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                          false, VectorFieldType::using_simd>
        &point_jacobian_matrix,
    const QuadratureType &lagrange_derivative,
    typename VectorFieldType::simd::datatype (
        &df_dxi)[VectorFieldType::components],
    typename VectorFieldType::simd::datatype (
        &df_deta)[VectorFieldType::components],
    typename VectorFieldType::simd::datatype (
        &df_dgamma)[VectorFieldType::components]) {

  constexpr int dimension = 3;
  constexpr int components = VectorFieldType::components;
  constexpr int ngll = VectorFieldType::ngll;
  using TensorPointViewType = specfem::datatype::TensorPointViewType<
      type_real, VectorFieldType::components, dimension,
      VectorFieldType::simd::using_simd>;
  const int ielement = local_index.ispec;
  const int iz = local_index.iz;
  const int iy = local_index.iy;
  const int ix = local_index.ix;

  for (int l = 0; l < ngll; ++l) {
    for (int icomponent = 0; icomponent < components; ++icomponent) {
      df_dxi[icomponent] +=
          lagrange_derivative.xi(ix, l) * f(ielement, iz, iy, l, icomponent);
      df_deta[icomponent] +=
          lagrange_derivative.eta(iy, l) * f(ielement, iz, l, ix, icomponent);
      df_dgamma[icomponent] +=
          lagrange_derivative.gamma(iz, l) * f(ielement, l, iy, ix, icomponent);
    }
  }

  TensorPointViewType df;

  for (int icomponent = 0; icomponent < components; ++icomponent) {
    df(icomponent, 0) = point_jacobian_matrix.xix * df_dxi[icomponent] +
                        point_jacobian_matrix.etax * df_deta[icomponent] +
                        point_jacobian_matrix.gammax * df_dgamma[icomponent];

    df(icomponent, 1) = point_jacobian_matrix.xiy * df_dxi[icomponent] +
                        point_jacobian_matrix.etay * df_deta[icomponent] +
                        point_jacobian_matrix.gammay * df_dgamma[icomponent];

    df(icomponent, 2) = point_jacobian_matrix.xiz * df_dxi[icomponent] +
                        point_jacobian_matrix.etaz * df_deta[icomponent] +
                        point_jacobian_matrix.gammaz * df_dgamma[icomponent];
  }
  return df;
}
} // namespace impl

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
 * @tparam VectorFieldType Field view type (Chunk view)
 * @tparam QuadratureType Quadrature view type
 * @tparam CallbackFunctor Callback functor type
 * @param chunk_index Chunk index specifying the elements within this chunk
 * @param jacobian_matrix Jacobian matrix of basis functions
 * @param quadrature Integration quadrature
 * @param f Field to compute the gradient of
 * @param callback Callback functor. Callback signature must be:
 * @code void(const typename IteratorType::index_type, const
 * specfem::datatype::TensorPointViewType<type_real, 2,
 * VectorFieldType::components>)
 * @endcode
 */
template <typename ChunkIndexType, typename VectorFieldType,
          typename QuadratureType, typename CallbackFunctor,
          std::enable_if_t<
              specfem::data_access::is_chunk_element<VectorFieldType>::value &&
                  VectorFieldType::dimension_tag ==
                      specfem::element::dimension_tag::dim2,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
gradient(const ChunkIndexType &chunk_index,
         const specfem::assembly::jacobian_matrix<
             specfem::element::dimension_tag::dim2> &jacobian_matrix,
         const QuadratureType &quadrature, const VectorFieldType &f,
         const CallbackFunctor &callback) {
  constexpr int components = VectorFieldType::components;
  constexpr int dimension = 2;
  constexpr bool using_simd = VectorFieldType::simd::using_simd;

  using TensorPointViewType =
      specfem::datatype::TensorPointViewType<type_real, components, dimension,
                                             using_simd>;

  using datatype = typename VectorFieldType::simd::datatype;

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
        const auto index = iterator_index.get_index();
        const auto local_index = iterator_index.get_local_index();
        datatype df_dxi[components] = { 0.0 };
        datatype df_dgamma[components] = { 0.0 };
        specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2,
                                        false, using_simd>
            point_jacobian_matrix;

        specfem::assembly::load_on_device(index, jacobian_matrix,
                                          point_jacobian_matrix);

        const auto df =
            impl::element_gradient(f, local_index, point_jacobian_matrix,
                                   quadrature, df_dxi, df_dgamma);
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
 * @tparam VectorFieldType Field view type (Chunk view)
 * @tparam QuadratureType Quadrature view type
 * @tparam CallbackFunctor Callback functor type
 * @param chunk_index Chunk index specifying the elements within this chunk
 * @param jacobian_matrix Jacobian matrix of basis functions
 * @param quadrature Integration quadrature
 * @param f Field to compute the gradient of
 * @param g Field to compute the gradient of
 * @param callback Callback functor. Callback signature must be:
 * @code void(const typename IteratorType::index_type, const
 * specfem::datatype::TensorPointViewType<type_real, 2,
 * VectorFieldType::components>, const
 * specfem::datatype::TensorPointViewType<type_real, 2,
 * VectorFieldType::components>)
 * @endcode
 */
template <typename ChunkIndexType, typename VectorFieldType,
          typename QuadratureType, typename CallbackFunctor,
          std::enable_if_t<
              specfem::data_access::is_chunk_element<VectorFieldType>::value &&
                  VectorFieldType::dimension_tag ==
                      specfem::element::dimension_tag::dim2,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
gradient(const ChunkIndexType &chunk_index,
         const specfem::assembly::jacobian_matrix<
             specfem::element::dimension_tag::dim2> &jacobian_matrix,
         const QuadratureType &quadrature, const VectorFieldType &f,
         const VectorFieldType &g, const CallbackFunctor &callback) {
  constexpr int components = VectorFieldType::components;
  constexpr bool using_simd = VectorFieldType::simd::using_simd;
  constexpr int dimension = 2;

  using TensorPointViewType =
      specfem::datatype::TensorPointViewType<type_real, components, dimension,
                                             using_simd>;

  using datatype = typename VectorFieldType::simd::datatype;

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
        const auto index = iterator_index.get_index();
        const auto local_index = iterator_index.get_local_index();
        datatype df_dxi[components] = { 0.0 };
        datatype df_dgamma[components] = { 0.0 };
        specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2,
                                        false, using_simd>
            point_jacobian_matrix;

        specfem::assembly::load_on_device(index, jacobian_matrix,
                                          point_jacobian_matrix);

        const auto df =
            impl::element_gradient(f, local_index, point_jacobian_matrix,
                                   quadrature, df_dxi, df_dgamma);
        const auto dg =
            impl::element_gradient(g, local_index, point_jacobian_matrix,
                                   quadrature, df_dxi, df_dgamma);
        callback(iterator_index, df, dg);
      });

  return;
}

/**
 * @brief Compute the gradient of a scalar field f using the spectral element
 * formulation for 3D elements
 *
 * @ingroup AlgorithmsGradient
 *
 * @tparam ChunkIndexType Chunk index type
 * @tparam VectorFieldType Field view type (Chunk view)
 * @tparam QuadratureType Quadrature view type
 * @tparam CallbackFunctor Callback functor type
 * @param chunk_index Chunk index specifying the elements within this chunk
 * @param jacobian_matrix Jacobian matrix of basis functions
 * @param quadrature Integration quadrature
 * @param f Field to compute the gradient of
 * @param callback Callback functor. Callback signature must be:
 * @code void(const typename IteratorType::index_type, const
 * specfem::datatype::TensorPointViewType<type_real, 3,
 * VectorFieldType::components>)
 * @endcode
 */
template <typename ChunkIndexType, typename VectorFieldType,
          typename QuadratureType, typename CallbackFunctor,
          std::enable_if_t<
              specfem::data_access::is_chunk_element<VectorFieldType>::value &&
                  VectorFieldType::dimension_tag ==
                      specfem::element::dimension_tag::dim3,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
gradient(const ChunkIndexType &chunk_index,
         const specfem::assembly::jacobian_matrix<
             specfem::element::dimension_tag::dim3> &jacobian_matrix,
         const QuadratureType &quadrature, const VectorFieldType &f,
         const CallbackFunctor &callback) {
  constexpr int components = VectorFieldType::components;
  constexpr int dimension = 3;
  constexpr bool using_simd = VectorFieldType::simd::using_simd;

  using TensorPointViewType =
      specfem::datatype::TensorPointViewType<type_real, components, dimension,
                                             using_simd>;

  using datatype = typename VectorFieldType::simd::datatype;

  static_assert(
      std::is_invocable_v<CallbackFunctor,
                          typename ChunkIndexType::iterator_type::index_type,
                          TensorPointViewType>,
      "CallbackFunctor must be invocable with the following signature: "
      "void(const int, const specfem::point::index, const "
      "specfem::kokkos::array_type<type_real, components>, const "
      "specfem::kokkos::array_type<type_real, components>, const "
      "specfem::kokkos::array_type<type_real, components>)");

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const auto local_index = iterator_index.get_local_index();
        datatype df_dxi[components] = { 0.0 };
        datatype df_deta[components] = { 0.0 };
        datatype df_dgamma[components] = { 0.0 };
        specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                        false, using_simd>
            point_jacobian_matrix;

        specfem::assembly::load_on_device(index, jacobian_matrix,
                                          point_jacobian_matrix);

        const auto df =
            impl::element_gradient(f, local_index, point_jacobian_matrix,
                                   quadrature, df_dxi, df_deta, df_dgamma);
        callback(iterator_index, df);
      });

  return;
}

/**
 * @brief Compute the gradient of a field f & g using the spectral element
 * formulation for 3D elements
 *
 * @ingroup AlgorithmsGradient
 *
 * @tparam ChunkIndexType Chunk index type
 * @tparam VectorFieldType Field view type (Chunk view)
 * @tparam QuadratureType Quadrature view type
 * @tparam CallbackFunctor Callback functor type
 * @param chunk_index Chunk index specifying the elements within this chunk
 * @param jacobian_matrix Jacobian matrix of basis functions
 * @param quadrature Integration quadrature
 * @param f Field to compute the gradient of
 * @param g Field to compute the gradient of
 * @param callback Callback functor. Callback signature must be:
 * @code void(const typename IteratorType::index_type, const
 * specfem::datatype::TensorPointViewType<type_real, 3,
 * VectorFieldType::components>, const
 * specfem::datatype::TensorPointViewType<type_real, 3,
 * VectorFieldType::components>)
 * @endcode
 */
template <typename ChunkIndexType, typename VectorFieldType,
          typename QuadratureType, typename CallbackFunctor,
          std::enable_if_t<
              specfem::data_access::is_chunk_element<VectorFieldType>::value &&
                  VectorFieldType::dimension_tag ==
                      specfem::element::dimension_tag::dim3,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
gradient(const ChunkIndexType &chunk_index,
         const specfem::assembly::jacobian_matrix<
             specfem::element::dimension_tag::dim3> &jacobian_matrix,
         const QuadratureType &quadrature, const VectorFieldType &f,
         const VectorFieldType &g, const CallbackFunctor &callback) {
  constexpr int components = VectorFieldType::components;
  constexpr bool using_simd = VectorFieldType::simd::using_simd;
  constexpr int dimension = 3;

  using TensorPointViewType =
      specfem::datatype::TensorPointViewType<type_real, components, dimension,
                                             using_simd>;

  using datatype = typename VectorFieldType::simd::datatype;

  static_assert(
      std::is_invocable_v<CallbackFunctor,
                          typename ChunkIndexType::iterator_type::index_type,
                          TensorPointViewType, TensorPointViewType>,
      "CallbackFunctor must be invocable with the following signature: "
      "void(const ChunkIndexType::iterator_type::index_type, "
      "const specfem::datatype::TensorPointViewType<type_real, 3, components>, "
      "const specfem::datatype::TensorPointViewType<type_real, 3, "
      "components>)");

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto local_index = iterator_index.get_local_index();
        const auto index = iterator_index.get_index();
        datatype df_dxi[components] = { 0.0 };
        datatype df_deta[components] = { 0.0 };
        datatype df_dgamma[components] = { 0.0 };
        datatype dg_dxi[components] = { 0.0 };
        datatype dg_deta[components] = { 0.0 };
        datatype dg_dgamma[components] = { 0.0 };
        specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                        false, using_simd>
            point_jacobian_matrix;

        specfem::assembly::load_on_device(index, jacobian_matrix,
                                          point_jacobian_matrix);

        const auto df =
            impl::element_gradient(f, local_index, point_jacobian_matrix,
                                   quadrature, df_dxi, df_deta, df_dgamma);
        const auto dg =
            impl::element_gradient(g, local_index, point_jacobian_matrix,
                                   quadrature, dg_dxi, dg_deta, dg_dgamma);
        callback(iterator_index, df, dg);
      });

  return;
}

/**
 * @brief Compute the gradient of a FieldPack using the spectral element
 * formulation. Dispatches to the single- or dual-field gradient overloads
 * based on FieldPack::size (1 or 2). The callback receives a GradientPack
 * whose gradient holders mirror the input field holders via their nested
 * `gradient_holder<V>` alias.
 *
 * @ingroup AlgorithmsGradient
 *
 * @tparam ChunkIndexType Chunk index type
 * @tparam Holders        Named holder types in the FieldPack
 * @tparam DimTag         Dimension tag (dim2 or dim3), deduced from
 *                        the jacobian_matrix argument
 * @tparam QuadratureType Quadrature view type
 * @tparam CallbackFunctor Callback functor receiving
 *         (iterator_index, GradientPack<GH0[, GH1]>)
 */
template <typename ChunkIndexType, typename... Holders,
          specfem::element::dimension_tag DimTag, typename QuadratureType,
          typename CallbackFunctor>
KOKKOS_FORCEINLINE_FUNCTION void
gradient(const ChunkIndexType &chunk_index,
         const specfem::assembly::jacobian_matrix<DimTag> &jacobian_matrix,
         const QuadratureType &quadrature,
         const specfem::chunk_element::FieldPack<Holders...> &field_pack,
         const CallbackFunctor &callback) {

  using PackType = specfem::chunk_element::FieldPack<Holders...>;

  if constexpr (PackType::size == 1) {
    using H0 = std::tuple_element_t<0, std::tuple<Holders...> >;
    gradient(chunk_index, jacobian_matrix, quadrature,
             static_cast<const H0 &>(field_pack).get(),
             [&](const auto &iterator_index, const auto &g0) {
               using GH0 = typename H0::template gradient_holder<
                   std::decay_t<decltype(g0)> >;
               callback(iterator_index,
                        specfem::point::GradientPack<GH0>(GH0(g0)));
             });
  } else if constexpr (PackType::size == 2) {
    using H0 = std::tuple_element_t<0, std::tuple<Holders...> >;
    using H1 = std::tuple_element_t<1, std::tuple<Holders...> >;
    gradient(chunk_index, jacobian_matrix, quadrature,
             static_cast<const H0 &>(field_pack).get(),
             static_cast<const H1 &>(field_pack).get(),
             [&](const auto &iterator_index, const auto &g0, const auto &g1) {
               using GH0 = typename H0::template gradient_holder<
                   std::decay_t<decltype(g0)> >;
               using GH1 = typename H1::template gradient_holder<
                   std::decay_t<decltype(g1)> >;
               callback(iterator_index, specfem::point::GradientPack<GH0, GH1>(
                                            GH0(g0), GH1(g1)));
             });
  }
}

} // namespace algorithms
} // namespace specfem
