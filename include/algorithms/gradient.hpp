#ifndef _ALGORITHMS_GRADIENT_HPP
#define _ALGORITHMS_GRADIENT_HPP

#include "kokkos_abstractions.h"
#include "point/field_derivatives.hpp"
#include "point/partial_derivatives.hpp"
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
 * @tparam MemberType Kokkos team member type
 * @tparam IteratorType Iterator type (Chunk iterator)
 * @tparam ViewType Field view type (Chunk view)
 * @tparam QuadratureType Quadrature view type
 * @tparam CallbackFunctor Callback functor type
 * @param team Kokkos team member
 * @param iterator Chunk iterator
 * @param partial_derivatives Partial derivatives of basis functions
 * @param quadrature Integration quadrature
 * @param f Field to compute the gradient of
 * @param callback Callback functor. Callback signature must be:
 * @code void(const typename IteratorType::index_type, const
 * specfem::datatype::VectorPointViewType<type_real, 2, ViewType::components>)
 * @endcode
 */
template <typename MemberType, typename IteratorType, typename ViewType,
          typename QuadratureType, typename CallbackFunctor,
          std::enable_if_t<ViewType::isChunkViewType, int> = 0>
NOINLINE KOKKOS_FUNCTION void
gradient(const MemberType &team, const IteratorType &iterator,
         const specfem::compute::partial_derivatives &partial_derivatives,
         const QuadratureType &quadrature, const ViewType &f,
         CallbackFunctor callback) {
  constexpr int components = ViewType::components;
  constexpr bool using_simd = ViewType::simd::using_simd;

  constexpr int NGLL = ViewType::ngll;

  using VectorPointViewType =
      specfem::datatype::VectorPointViewType<type_real, 2, components,
                                             using_simd>;

  using datatype = typename IteratorType::simd::datatype;

  static_assert(ViewType::isScalarViewType,
                "ViewType must be a scalar field view type");

  static_assert(
      std::is_same_v<typename IteratorType::simd, typename ViewType::simd>,
      "IteratorType and ViewType must have the same simd type");

  static_assert(
      std::is_invocable_v<CallbackFunctor, typename IteratorType::index_type,
                          VectorPointViewType>,
      "CallbackFunctor must be invocable with the following signature: "
      "void(const int, const specfem::point::index, const "
      "specfem::kokkos::array_type<type_real, components>, const "
      "specfem::kokkos::array_type<type_real, components>)");

  static_assert(
      Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                 typename ViewType::memory_space>::accessible,
      "ViewType memory space is not accessible from the member execution "
      "space");

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, iterator.chunk_size()), [&](const int &i) {
        const auto iterator_index = iterator(i);
        const auto &index = iterator_index.index;
        const int &ielement = iterator_index.ielement;
        const int &ix = index.ix;
        const int &iz = index.iz;

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

        specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                            false, using_simd>
            point_partial_derivatives;

        specfem::compute::load_on_device(index, partial_derivatives,
                                         point_partial_derivatives);

        VectorPointViewType df;

        for (int icomponent = 0; icomponent < components; ++icomponent) {
          df(0, icomponent) =
              point_partial_derivatives.xix * df_dxi[icomponent] +
              point_partial_derivatives.gammax * df_dgamma[icomponent];

          df(1, icomponent) =
              point_partial_derivatives.xiz * df_dxi[icomponent] +
              point_partial_derivatives.gammaz * df_dgamma[icomponent];
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
 * @tparam MemberType Kokkos team member type
 * @tparam IteratorType Iterator type (Chunk iterator)
 * @tparam ViewType Field view type (Chunk view)
 * @tparam QuadratureType Quadrature view type
 * @tparam CallbackFunctor Callback functor type
 * @param team Kokkos team member
 * @param iterator Chunk iterator
 * @param partial_derivatives Partial derivatives of basis functions
 * @param quadrature Integration quadrature
 * @param f Field to compute the gradient of
 * @param g Field to compute the gradient of
 * @param callback Callback functor. Callback signature must be:
 * @code void(const typename IteratorType::index_type, const
 * specfem::datatype::VectorPointViewType<type_real, 2, ViewType::components>,
 * const specfem::datatype::VectorPointViewType<type_real, 2,
 * ViewType::components>)
 * @endcode
 */
template <typename MemberType, typename IteratorType, typename ViewType,
          typename QuadratureType, typename CallbackFunctor,
          std::enable_if_t<ViewType::isChunkViewType, int> = 0>
NOINLINE KOKKOS_FUNCTION void
gradient(const MemberType &team, const IteratorType &iterator,
         const specfem::compute::partial_derivatives &partial_derivatives,
         const QuadratureType &quadrature, const ViewType &f, const ViewType &g,
         CallbackFunctor callback) {
  constexpr int components = ViewType::components;
  constexpr bool using_simd = ViewType::simd::using_simd;

  constexpr int NGLL = ViewType::ngll;

  using VectorPointViewType =
      specfem::datatype::VectorPointViewType<type_real, 2, components,
                                             using_simd>;

  static_assert(ViewType::isScalarViewType,
                "ViewType must be a scalar field view type");

  using datatype = typename IteratorType::simd::datatype;

  static_assert(
      std::is_same_v<typename IteratorType::simd, typename ViewType::simd>,
      "IteratorType and ViewType must have the same simd type");

  static_assert(
      std::is_invocable_v<CallbackFunctor, typename IteratorType::index_type,
                          VectorPointViewType, VectorPointViewType>,
      "CallbackFunctor must be invocable with the following signature: "
      "void(const int, const specfem::point::index, const "
      "pecfem::datatype::VectorPointViewType<type_real, 2, components>, "
      "const "
      "pecfem::datatype::VectorPointViewType<type_real, 2, components>)");

  static_assert(
      Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                 typename ViewType::memory_space>::accessible,
      "ViewType memory space is not accessible from the member execution "
      "space");

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, iterator.chunk_size()), [=](const int &i) {
        const auto iterator_index = iterator(i);
        const auto &index = iterator_index.index;
        const int &ielement = iterator_index.ielement;
        const int &ix = index.ix;
        const int &iz = index.iz;

        datatype df_dxi[components];
        datatype df_dgamma[components];

        for (int icomponent = 0; icomponent < components; ++icomponent) {
          df_dxi[icomponent] = 0.0;
          df_dgamma[icomponent] = 0.0;
        }

        for (int l = 0; l < NGLL; ++l) {
          for (int icomponent = 0; icomponent < components; ++icomponent) {
            df_dxi[icomponent] +=
                quadrature(ix, l) * f(ielement, iz, l, icomponent);
            df_dgamma[icomponent] +=
                quadrature(iz, l) * f(ielement, l, ix, icomponent);
          }
        }

        specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                            false, using_simd>
            point_partial_derivatives;

        specfem::compute::load_on_device(index, partial_derivatives,
                                         point_partial_derivatives);

        VectorPointViewType df;

        for (int icomponent = 0; icomponent < components; ++icomponent) {
          df(0, icomponent) =
              point_partial_derivatives.xix * df_dxi[icomponent] +
              point_partial_derivatives.gammax * df_dgamma[icomponent];

          df(1, icomponent) =
              point_partial_derivatives.xiz * df_dxi[icomponent] +
              point_partial_derivatives.gammaz * df_dgamma[icomponent];
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

        VectorPointViewType dg;

        for (int icomponent = 0; icomponent < components; ++icomponent) {
          dg(0, icomponent) =
              point_partial_derivatives.xix * df_dxi[icomponent] +
              point_partial_derivatives.gammax * df_dgamma[icomponent];

          dg(1, icomponent) =
              point_partial_derivatives.xiz * df_dxi[icomponent] +
              point_partial_derivatives.gammaz * df_dgamma[icomponent];
        }

        callback(iterator_index, df, dg);
      });

  return;
}
} // namespace algorithms
} // namespace specfem

#endif /* _ALGORITHMS_GRADIENT_HPP */
