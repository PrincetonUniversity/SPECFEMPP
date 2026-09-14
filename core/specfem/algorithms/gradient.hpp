#pragma once

#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/chunk_element/field_pack.hpp"
#include "specfem/data_access.hpp"
#include "specfem/element.hpp"
#include "specfem/execution.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

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
    df(icomponent, 0) = point_jacobian_matrix.xix() * df_dxi[icomponent] +
                        point_jacobian_matrix.gammax() * df_dgamma[icomponent];

    df(icomponent, 1) = point_jacobian_matrix.xiz() * df_dxi[icomponent] +
                        point_jacobian_matrix.gammaz() * df_dgamma[icomponent];
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
    df(icomponent, 0) = point_jacobian_matrix.xix() * df_dxi[icomponent] +
                        point_jacobian_matrix.etax() * df_deta[icomponent] +
                        point_jacobian_matrix.gammax() * df_dgamma[icomponent];

    df(icomponent, 1) = point_jacobian_matrix.xiy() * df_dxi[icomponent] +
                        point_jacobian_matrix.etay() * df_deta[icomponent] +
                        point_jacobian_matrix.gammay() * df_dgamma[icomponent];

    df(icomponent, 2) = point_jacobian_matrix.xiz() * df_dxi[icomponent] +
                        point_jacobian_matrix.etaz() * df_deta[icomponent] +
                        point_jacobian_matrix.gammaz() * df_dgamma[icomponent];
  }
  return df;
}
} // namespace impl

/**
 * @defgroup AlgorithmsGradient
 *
 */

/**
 * @brief Compute the gradient of a vector field f using the spectral element
 * formulation (eqn: 29 in Komatitsch and Tromp, 1999)
 *
 * @ingroup AlgorithmsGradient
 *
 * @tparam ChunkIndexType  Chunk index type
 * @tparam VectorFieldType Field view type (Chunk view)
 * @tparam QuadratureType  Quadrature view type
 * @tparam CallbackFunctor Callback functor type
 * @param chunk_index      Chunk index specifying the elements within this chunk
 * @param jacobian_matrix  Jacobian matrix of basis functions
 * @param quadrature       Integration quadrature
 * @param f                Field to compute the gradient of
 * @param callback         Callback functor receiving
 *        (iterator_index, TensorPointViewType<components, dim>)
 */
template <typename ChunkIndexType, typename VectorFieldType,
          typename JacobianMatrixType, typename QuadratureType,
          typename CallbackFunctor,
          std::enable_if_t<
              specfem::data_access::is_chunk_element<VectorFieldType>::value,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
gradient(const ChunkIndexType &chunk_index,
         const JacobianMatrixType &jacobian_matrix,
         const QuadratureType &quadrature, const VectorFieldType &f,
         const CallbackFunctor &callback) {

  constexpr specfem::element::dimension_tag dimension_tag =
      VectorFieldType::dimension_tag;
  constexpr int components = VectorFieldType::components;
  constexpr int dimension = specfem::element::dimension<dimension_tag>::dim;
  constexpr bool using_simd = VectorFieldType::simd::using_simd;

  using TensorPointViewType =
      specfem::datatype::TensorPointViewType<type_real, components, dimension,
                                             using_simd>;
  using datatype = typename VectorFieldType::simd::datatype;

  static_assert(
      std::is_invocable_v<CallbackFunctor,
                          typename ChunkIndexType::iterator_type::index_type,
                          TensorPointViewType>,
      "CallbackFunctor must be invocable with (iterator_index, "
      "TensorPointViewType)");

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const auto local_index = iterator_index.get_local_index();
        datatype df_dxi[components] = { 0.0 };
        datatype df_dgamma[components] = { 0.0 };
        specfem::point::jacobian_matrix<dimension_tag, false, using_simd>
            point_jacobian_matrix;
        specfem::assembly::load_on_device(index, jacobian_matrix,
                                          point_jacobian_matrix);
        if constexpr (dimension_tag == specfem::element::dimension_tag::dim3) {
          datatype df_deta[components] = { 0.0 };
          callback(iterator_index, impl::element_gradient(
                                       f, local_index, point_jacobian_matrix,
                                       quadrature, df_dxi, df_deta, df_dgamma));
        } else {
          callback(iterator_index,
                   impl::element_gradient(f, local_index, point_jacobian_matrix,
                                          quadrature, df_dxi, df_dgamma));
        }
      });
}

/**
 * @brief Compute the gradients of two vector fields f and g using the spectral
 * element formulation (eqn: 29 in Komatitsch and Tromp, 1999)
 *
 * @ingroup AlgorithmsGradient
 *
 * @tparam ChunkIndexType  Chunk index type
 * @tparam VectorFieldType Field view type (Chunk view)
 * @tparam QuadratureType  Quadrature view type
 * @tparam CallbackFunctor Callback functor type
 * @param chunk_index      Chunk index specifying the elements within this chunk
 * @param jacobian_matrix  Jacobian matrix of basis functions
 * @param quadrature       Integration quadrature
 * @param f                First field to compute the gradient of
 * @param g                Second field to compute the gradient of
 * @param callback         Callback functor receiving
 *        (iterator_index, TensorPointViewType<components, dim>,
 *         TensorPointViewType<components, dim>)
 */
template <typename ChunkIndexType, typename VectorFieldType,
          typename JacobianMatrixType, typename QuadratureType,
          typename CallbackFunctor,
          std::enable_if_t<
              specfem::data_access::is_chunk_element<VectorFieldType>::value,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
gradient(const ChunkIndexType &chunk_index,
         const JacobianMatrixType &jacobian_matrix,
         const QuadratureType &quadrature, const VectorFieldType &f,
         const VectorFieldType &g, const CallbackFunctor &callback) {

  constexpr specfem::element::dimension_tag dimension_tag =
      VectorFieldType::dimension_tag;
  constexpr int components = VectorFieldType::components;
  constexpr int dimension = specfem::element::dimension<dimension_tag>::dim;
  constexpr bool using_simd = VectorFieldType::simd::using_simd;

  using TensorPointViewType =
      specfem::datatype::TensorPointViewType<type_real, components, dimension,
                                             using_simd>;
  using datatype = typename VectorFieldType::simd::datatype;

  static_assert(
      std::is_invocable_v<CallbackFunctor,
                          typename ChunkIndexType::iterator_type::index_type,
                          TensorPointViewType, TensorPointViewType>,
      "CallbackFunctor must be invocable with (iterator_index, "
      "TensorPointViewType, TensorPointViewType)");

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const auto local_index = iterator_index.get_local_index();
        datatype df_dxi[components] = { 0.0 };
        datatype df_dgamma[components] = { 0.0 };
        datatype dg_dxi[components] = { 0.0 };
        datatype dg_dgamma[components] = { 0.0 };
        specfem::point::jacobian_matrix<dimension_tag, false, using_simd>
            point_jacobian_matrix;
        specfem::assembly::load_on_device(index, jacobian_matrix,
                                          point_jacobian_matrix);
        if constexpr (dimension_tag == specfem::element::dimension_tag::dim3) {
          datatype df_deta[components] = { 0.0 };
          datatype dg_deta[components] = { 0.0 };
          callback(
              iterator_index,
              impl::element_gradient(f, local_index, point_jacobian_matrix,
                                     quadrature, df_dxi, df_deta, df_dgamma),
              impl::element_gradient(g, local_index, point_jacobian_matrix,
                                     quadrature, dg_dxi, dg_deta, dg_dgamma));
        } else {
          callback(iterator_index,
                   impl::element_gradient(f, local_index, point_jacobian_matrix,
                                          quadrature, df_dxi, df_dgamma),
                   impl::element_gradient(g, local_index, point_jacobian_matrix,
                                          quadrature, dg_dxi, dg_dgamma));
        }
      });
}

/**
 * @brief Compute the gradient of a FieldPack<F> using the spectral element
 * formulation. The callback receives a GradientPack<TF>.
 *
 * @ingroup AlgorithmsGradient
 *
 * @tparam ChunkIndexType  Chunk index type
 * @tparam VectorFieldType   Chunk element field type held as .f
 * @tparam QuadratureType  Quadrature view type
 * @tparam CallbackFunctor Callback functor receiving
 *         (iterator_index, GradientPack<TF>)
 */
template <typename ChunkIndexType, typename VectorFieldType,
          typename JacobianMatrixType, typename QuadratureType,
          typename CallbackFunctor>
KOKKOS_FORCEINLINE_FUNCTION void
gradient(const ChunkIndexType &chunk_index,
         const JacobianMatrixType &jacobian_matrix,
         const QuadratureType &quadrature,
         const specfem::chunk_element::FieldPack<VectorFieldType> &field_pack,
         const CallbackFunctor &callback) {
  constexpr specfem::element::dimension_tag dimension_tag =
      VectorFieldType::dimension_tag;
  constexpr bool using_simd = VectorFieldType::simd::using_simd;
  constexpr int dimension = specfem::element::dimension<dimension_tag>::dim;
  constexpr int components = VectorFieldType::components;

  using TensorPointViewTypeF =
      specfem::datatype::TensorPointViewType<type_real, components, dimension,
                                             using_simd>;
  using datatypeF = typename VectorFieldType::simd::datatype;

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const auto local_index = iterator_index.get_local_index();
        datatypeF df_dxi[components] = { 0.0 };
        datatypeF df_dgamma[components] = { 0.0 };
        specfem::point::jacobian_matrix<dimension_tag, false, using_simd>
            point_jacobian_matrix;
        specfem::assembly::load_on_device(index, jacobian_matrix,
                                          point_jacobian_matrix);
        if constexpr (dimension_tag == specfem::element::dimension_tag::dim3) {
          datatypeF df_deta[components] = { 0.0 };
          callback(iterator_index,
                   specfem::point::GradientPack<TensorPointViewTypeF>{
                       impl::element_gradient(
                           static_cast<const VectorFieldType &>(field_pack),
                           local_index, point_jacobian_matrix, quadrature,
                           df_dxi, df_deta, df_dgamma) });
        } else {
          callback(iterator_index,
                   specfem::point::GradientPack<TensorPointViewTypeF>{
                       impl::element_gradient(
                           static_cast<const VectorFieldType &>(field_pack),
                           local_index, point_jacobian_matrix, quadrature,
                           df_dxi, df_dgamma) });
        }
      });
}

/**
 * @brief Compute the gradients of a FieldPack<F, G> using the spectral element
 * formulation. The callback receives a GradientPack<TF, TG>.
 *
 * @ingroup AlgorithmsGradient
 *
 * @tparam ChunkIndexType   Chunk index type
 * @tparam VectorFieldTypeF Chunk element field type held as .f
 * @tparam VectorFieldTypeG Chunk element field type held as .g
 * @tparam QuadratureType   Quadrature view type
 * @tparam CallbackFunctor  Callback functor receiving
 *         (iterator_index, GradientPack<TF, TG>)
 */
template <typename ChunkIndexType, typename VectorFieldTypeF,
          typename VectorFieldTypeG, typename JacobianMatrixType,
          typename QuadratureType, typename CallbackFunctor>
KOKKOS_FORCEINLINE_FUNCTION void gradient(
    const ChunkIndexType &chunk_index,
    const JacobianMatrixType &jacobian_matrix, const QuadratureType &quadrature,
    const specfem::chunk_element::FieldPack<VectorFieldTypeF, VectorFieldTypeG>
        &field_pack,
    const CallbackFunctor &callback) {

  static_assert(VectorFieldTypeF::dimension_tag ==
                VectorFieldTypeG::dimension_tag);

  constexpr specfem::element::dimension_tag dimension_tag =
      VectorFieldTypeF::dimension_tag;
  constexpr int dimension = specfem::element::dimension<dimension_tag>::dim;

  constexpr int componentsF = VectorFieldTypeF::components;
  constexpr int componentsG = VectorFieldTypeG::components;
  constexpr bool using_simd = VectorFieldTypeF::simd::using_simd;

  using TensorPointViewTypeF =
      specfem::datatype::TensorPointViewType<type_real, componentsF, dimension,
                                             using_simd>;
  using TensorPointViewTypeG = specfem::datatype::TensorPointViewType<
      type_real, componentsG, dimension, VectorFieldTypeG::simd::using_simd>;

  using datatypeF = typename VectorFieldTypeF::simd::datatype;
  using datatypeG = typename VectorFieldTypeG::simd::datatype;

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const auto local_index = iterator_index.get_local_index();
        datatypeF df_dxi[componentsF] = { 0.0 };
        datatypeF df_dgamma[componentsF] = { 0.0 };
        datatypeG dg_dxi[componentsG] = { 0.0 };
        datatypeG dg_dgamma[componentsG] = { 0.0 };
        specfem::point::jacobian_matrix<dimension_tag, false, using_simd>
            point_jacobian_matrix;
        specfem::assembly::load_on_device(index, jacobian_matrix,
                                          point_jacobian_matrix);
        if constexpr (dimension_tag == specfem::element::dimension_tag::dim3) {
          datatypeF df_deta[componentsF] = { 0.0 };
          datatypeG dg_deta[componentsG] = { 0.0 };
          callback(iterator_index,
                   specfem::point::GradientPack<TensorPointViewTypeF,
                                                TensorPointViewTypeG>{
                       impl::element_gradient(
                           static_cast<const VectorFieldTypeF &>(field_pack),
                           local_index, point_jacobian_matrix, quadrature,
                           df_dxi, df_deta, df_dgamma),
                       impl::element_gradient(
                           static_cast<const VectorFieldTypeG &>(field_pack),
                           local_index, point_jacobian_matrix, quadrature,
                           dg_dxi, dg_deta, dg_dgamma) });
        } else {
          callback(iterator_index,
                   specfem::point::GradientPack<TensorPointViewTypeF,
                                                TensorPointViewTypeG>{
                       impl::element_gradient(
                           static_cast<const VectorFieldTypeF &>(field_pack),
                           local_index, point_jacobian_matrix, quadrature,
                           df_dxi, df_dgamma),
                       impl::element_gradient(
                           static_cast<const VectorFieldTypeG &>(field_pack),
                           local_index, point_jacobian_matrix, quadrature,
                           dg_dxi, dg_dgamma) });
        }
      });
}

} // namespace algorithms
} // namespace specfem
