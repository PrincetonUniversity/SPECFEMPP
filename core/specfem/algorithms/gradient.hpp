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
 * @brief Accumulate partial derivatives of a 2D vector field in reference
 * coordinates.
 *
 * Computes partial derivatives with respect to reference coordinates (ξ, γ)
 * by accumulating contributions from Lagrange polynomial derivatives.
 * Reads field data and quadrature information from scratch memory.
 *
 * @tparam VectorFieldType Field type with dimension_tag=dim2
 * @tparam QuadratureType Quadrature type providing lagrange derivatives
 * @param f Input vector field
 * @param local_index Element and point indices (ispec, iz, ix)
 * @param lagrange_derivative Quadrature derivative values
 * @param df_dq Output: accumulated partial derivatives (components × 2 tensor)
 */
template <typename VectorFieldType, typename QuadratureType,
          typename std::enable_if_t<VectorFieldType::dimension_tag ==
                                        specfem::element::dimension_tag::dim2,
                                    int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void element_accumulate(
    const VectorFieldType &f,
    const specfem::point::index<specfem::element::dimension_tag::dim2,
                                VectorFieldType::using_simd> &local_index,
    const QuadratureType &lagrange_derivative,
    specfem::datatype::TensorPointViewType<
        type_real, VectorFieldType::components, 2,
        VectorFieldType::simd::using_simd> &df_dq) {

  constexpr int components = VectorFieldType::components;
  constexpr int ngll = VectorFieldType::ngll;
  const int ielement = local_index.ispec;
  const int iz = local_index.iz;
  const int ix = local_index.ix;

  for (int l = 0; l < ngll; ++l) {
    for (int icomponent = 0; icomponent < components; ++icomponent) {
      df_dq(icomponent, 0) +=
          lagrange_derivative.xi(ix, l) * f(ielement, iz, l, icomponent);
      df_dq(icomponent, 1) +=
          lagrange_derivative.gamma(iz, l) * f(ielement, l, ix, icomponent);
    }
  }
}

/**
 * @brief Transform 2D reference-frame derivatives to physical coordinates.
 *
 * Applies Jacobian matrix transformation to convert partial derivatives
 * from reference coordinates to physical coordinates using the chain rule.
 *
 * @tparam VectorFieldType Field type with dimension_tag=dim2
 * @param point_jacobian_matrix Jacobian transformation matrix
 * @param df_dq Partial derivatives in reference frame (components × 2)
 * @return Partial derivatives in physical coordinates
 */
template <typename VectorFieldType,
          typename std::enable_if_t<VectorFieldType::dimension_tag ==
                                        specfem::element::dimension_tag::dim2,
                                    int> = 0>
KOKKOS_FORCEINLINE_FUNCTION auto element_transform(
    const specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2,
                                          false, VectorFieldType::using_simd>
        &point_jacobian_matrix,
    const specfem::datatype::TensorPointViewType<
        type_real, VectorFieldType::components, 2,
        VectorFieldType::simd::using_simd> &df_dq) {

  constexpr int dimension = 2;
  constexpr int components = VectorFieldType::components;
  using TensorPointViewType =
      specfem::datatype::TensorPointViewType<type_real, components, dimension,
                                             VectorFieldType::simd::using_simd>;

  TensorPointViewType df;

  for (int icomponent = 0; icomponent < components; ++icomponent) {
    df(icomponent, 0) = point_jacobian_matrix.xix * df_dq(icomponent, 0) +
                        point_jacobian_matrix.gammax * df_dq(icomponent, 1);

    df(icomponent, 1) = point_jacobian_matrix.xiz * df_dq(icomponent, 0) +
                        point_jacobian_matrix.gammaz * df_dq(icomponent, 1);
  }
  return df;
}

/**
 * @brief Accumulate partial derivatives of a 3D vector field in reference
 * coordinates.
 *
 * Computes partial derivatives with respect to reference coordinates (ξ, η, γ)
 * by accumulating contributions from Lagrange polynomial derivatives.
 * Reads field data and quadrature information from scratch memory.
 *
 * @tparam VectorFieldType Field type with dimension_tag=dim3
 * @tparam QuadratureType Quadrature type providing lagrange derivatives
 * @param f Input vector field
 * @param local_index Element and point indices (ispec, iz, iy, ix)
 * @param lagrange_derivative Quadrature derivative values
 * @param df_dq Output: accumulated partial derivatives (components × 3 tensor)
 */
template <typename VectorFieldType, typename QuadratureType,
          typename std::enable_if_t<VectorFieldType::dimension_tag ==
                                        specfem::element::dimension_tag::dim3,
                                    int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void element_accumulate(
    const VectorFieldType &f,
    const specfem::point::index<specfem::element::dimension_tag::dim3,
                                VectorFieldType::using_simd> &local_index,
    const QuadratureType &lagrange_derivative,
    specfem::datatype::TensorPointViewType<
        type_real, VectorFieldType::components, 3,
        VectorFieldType::simd::using_simd> &df_dq) {

  constexpr int components = VectorFieldType::components;
  constexpr int ngll = VectorFieldType::ngll;
  const int ielement = local_index.ispec;
  const int iz = local_index.iz;
  const int iy = local_index.iy;
  const int ix = local_index.ix;

  for (int l = 0; l < ngll; ++l) {
    for (int icomponent = 0; icomponent < components; ++icomponent) {
      df_dq(icomponent, 0) +=
          lagrange_derivative.xi(ix, l) * f(ielement, iz, iy, l, icomponent);
      df_dq(icomponent, 1) +=
          lagrange_derivative.eta(iy, l) * f(ielement, iz, l, ix, icomponent);
      df_dq(icomponent, 2) +=
          lagrange_derivative.gamma(iz, l) * f(ielement, l, iy, ix, icomponent);
    }
  }
}

/**
 * @brief Transform 3D reference-frame derivatives to physical coordinates.
 *
 * Applies Jacobian matrix transformation to convert partial derivatives
 * from reference coordinates (ξ, η, γ) to physical coordinates (x, y, z)
 * using the chain rule.
 *
 * @tparam VectorFieldType Field type with dimension_tag=dim3
 * @param point_jacobian_matrix Jacobian transformation matrix (9 components)
 * @param df_dq Partial derivatives in reference frame (components × 3)
 * @return Partial derivatives in physical coordinates
 */
template <typename VectorFieldType,
          typename std::enable_if_t<VectorFieldType::dimension_tag ==
                                        specfem::element::dimension_tag::dim3,
                                    int> = 0>
KOKKOS_FORCEINLINE_FUNCTION auto element_transform(
    const specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                          false, VectorFieldType::using_simd>
        &point_jacobian_matrix,
    const specfem::datatype::TensorPointViewType<
        type_real, VectorFieldType::components, 3,
        VectorFieldType::simd::using_simd> &df_dq) {

  constexpr int dimension = 3;
  constexpr int components = VectorFieldType::components;
  using TensorPointViewType =
      specfem::datatype::TensorPointViewType<type_real, components, dimension,
                                             VectorFieldType::simd::using_simd>;

  TensorPointViewType df;

  for (int icomponent = 0; icomponent < components; ++icomponent) {
    df(icomponent, 0) = point_jacobian_matrix.xix * df_dq(icomponent, 0) +
                        point_jacobian_matrix.etax * df_dq(icomponent, 1) +
                        point_jacobian_matrix.gammax * df_dq(icomponent, 2);

    df(icomponent, 1) = point_jacobian_matrix.xiy * df_dq(icomponent, 0) +
                        point_jacobian_matrix.etay * df_dq(icomponent, 1) +
                        point_jacobian_matrix.gammay * df_dq(icomponent, 2);

    df(icomponent, 2) = point_jacobian_matrix.xiz * df_dq(icomponent, 0) +
                        point_jacobian_matrix.etaz * df_dq(icomponent, 1) +
                        point_jacobian_matrix.gammaz * df_dq(icomponent, 2);
  }
  return df;
}

} // namespace impl

/**
 * @brief Compute gradients of a vector field in spectral elements.
 *
 * Computes field gradients at quadrature points using the spectral element
 * method (Komatitsch & Tromp, 1999). Transforms from reference to physical
 * coordinates via Jacobian matrices. Invokes callback for each quadrature
 * point.
 *
 * @tparam ChunkIndexType Chunk element index
 * @tparam VectorFieldType Field container
 * @tparam JacobianMatrixType Jacobian matrix container
 * @tparam QuadratureType Lagrange derivative quadrature
 * @tparam CallbackFunctor Callback taking (iterator_index, gradient_tensor)
 *
 * @param chunk_index Elements in chunk
 * @param jacobian_matrix Coordinate transformation matrices
 * @param quadrature Lagrange polynomial derivatives
 * @param f Input field
 * @param callback Invoked with gradient tensor at each quadrature point
 *
 * @ingroup AlgorithmsGradient
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

  using TPViewType =
      specfem::datatype::TensorPointViewType<type_real, components, dimension,
                                             using_simd>;

  static_assert(
      std::is_invocable_v<CallbackFunctor,
                          typename ChunkIndexType::iterator_type::index_type,
                          TPViewType>,
      "CallbackFunctor must be invocable with (iterator_index, "
      "TensorPointViewType)");

  specfem::execution::for_each_level(
      specfem::execution::prefetch_ahead<4>(
          chunk_index,
          [&](const auto &prefetch_index) {
            specfem::assembly::prefetch_on_device(prefetch_index.get_index(),
                                                  jacobian_matrix);
          })
          .get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const auto local_index = iterator_index.get_local_index();
        TPViewType df_dq;
        impl::element_accumulate(f, local_index, quadrature, df_dq);
        specfem::point::jacobian_matrix<dimension_tag, false, using_simd>
            point_jacobian_matrix;
        specfem::assembly::load_on_device(index, jacobian_matrix,
                                          point_jacobian_matrix);
        callback(iterator_index, impl::element_transform<VectorFieldType>(
                                     point_jacobian_matrix, df_dq));
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

  using TPViewType =
      specfem::datatype::TensorPointViewType<type_real, components, dimension,
                                             using_simd>;

  static_assert(
      std::is_invocable_v<CallbackFunctor,
                          typename ChunkIndexType::iterator_type::index_type,
                          TPViewType, TPViewType>,
      "CallbackFunctor must be invocable with (iterator_index, "
      "TensorPointViewType, TensorPointViewType)");

  specfem::execution::for_each_level(
      specfem::execution::prefetch_ahead<4>(
          chunk_index,
          [&](const auto &prefetch_index) {
            specfem::assembly::prefetch_on_device(prefetch_index.get_index(),
                                                  jacobian_matrix);
          })
          .get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const auto local_index = iterator_index.get_local_index();
        TPViewType df_dq;
        TPViewType dg_dq;
        impl::element_accumulate(f, local_index, quadrature, df_dq);
        impl::element_accumulate(g, local_index, quadrature, dg_dq);
        specfem::point::jacobian_matrix<dimension_tag, false, using_simd>
            point_jacobian_matrix;
        specfem::assembly::load_on_device(index, jacobian_matrix,
                                          point_jacobian_matrix);
        callback(iterator_index,
                 impl::element_transform<VectorFieldType>(point_jacobian_matrix,
                                                          df_dq),
                 impl::element_transform<VectorFieldType>(point_jacobian_matrix,
                                                          dg_dq));
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

  specfem::execution::for_each_level(
      specfem::execution::prefetch_ahead<4>(
          chunk_index,
          [&](const auto &prefetch_index) {
            specfem::assembly::prefetch_on_device(prefetch_index.get_index(),
                                                  jacobian_matrix);
          })
          .get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const auto local_index = iterator_index.get_local_index();
        TensorPointViewTypeF df_dq;
        impl::element_accumulate(
            static_cast<const VectorFieldType &>(field_pack), local_index,
            quadrature, df_dq);
        specfem::point::jacobian_matrix<dimension_tag, false, using_simd>
            point_jacobian_matrix;
        specfem::assembly::load_on_device(index, jacobian_matrix,
                                          point_jacobian_matrix);
        callback(iterator_index,
                 specfem::point::GradientPack<TensorPointViewTypeF>{
                     impl::element_transform<VectorFieldType>(
                         point_jacobian_matrix, df_dq) });
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

  specfem::execution::for_each_level(
      specfem::execution::prefetch_ahead<4>(
          chunk_index,
          [&](const auto &prefetch_index) {
            specfem::assembly::prefetch_on_device(prefetch_index.get_index(),
                                                  jacobian_matrix);
          })
          .get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const auto local_index = iterator_index.get_local_index();
        TensorPointViewTypeF df_dq;
        TensorPointViewTypeG dg_dq;
        impl::element_accumulate(
            static_cast<const VectorFieldTypeF &>(field_pack), local_index,
            quadrature, df_dq);
        impl::element_accumulate(
            static_cast<const VectorFieldTypeG &>(field_pack), local_index,
            quadrature, dg_dq);
        specfem::point::jacobian_matrix<dimension_tag, false, using_simd>
            point_jacobian_matrix;
        specfem::assembly::load_on_device(index, jacobian_matrix,
                                          point_jacobian_matrix);
        callback(iterator_index,
                 specfem::point::GradientPack<TensorPointViewTypeF,
                                              TensorPointViewTypeG>{
                     impl::element_transform<VectorFieldTypeF>(
                         point_jacobian_matrix, df_dq),
                     impl::element_transform<VectorFieldTypeG>(
                         point_jacobian_matrix, dg_dq) });
      });
}

} // namespace algorithms
} // namespace specfem
