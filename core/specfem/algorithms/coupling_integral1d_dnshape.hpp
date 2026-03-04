#pragma once

#include "specfem/assembly.hpp"
#include "specfem/element_coupling.hpp"
#include "specfem/execution.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

/**
 * @brief Temporary file for integral in issue # 1550. this function should be
 * merged with `coupling_integral1d.hpp:coupling_integral` once the intersection
 * field typing is updated to store two fields.
 *
 */

namespace specfem::algorithms {

/**
 * @brief Takes a field `intersection_field` on the intersection and computes,
 * for each self GLL point, the integral of `intersection_field` times the
 * normal derivative of the shape function at that point. `intersection_field`
 * should be call-accessible (e.g. Kokkos::View) with shape:
 *
 * (chunk_size, n_quad_intersection, self::components())
 *
 * After handling any other intersection forces, boundary conditions, etc. the
 * result can be `atomic_add`ed to the acceleration field.
 *
 * @tparam IndexType The chunk_edge iterator type
 * @tparam IntersectionFieldViewType The type of `intersection_field`
 * @tparam IntersectionFactor A nonconforming chunk_edge accessor
 * holding `intersection_factor`
 * @tparam ShapeFunctionNormalDerivativesType a view holding the normal
 * derivatives of the shape function (implementation may change)
 * @tparam CallableType The callback function, which will be given the point
 * index and corresponding evaluated integral
 *
 * @param ngllz - number of quadrature points in z
 * @param ngllx - number of quadrature points in x
 * @param chunk_index - the outer index (chunk_edge) that gets iterated for
 * points
 * @param intersection_field - the field to integrate
 * @param intersection_factor - nonconforming chunk_edge accessor holding
 * `intersection_factor`
 * @param shape_function_normal_derivatives - (TEMPORARY, TO BE LOADED LATER BY
 * THIS FUNCTION) normal derivatives of the shape functions at each intersection
 * point.
 * @param callback - callback function to capture integral values
 * @ingroup AlgorithmsIntegration
 */
template <typename IndexType, typename IntersectionFieldViewType,
          typename IntersectionFactor,
          typename ShapeFunctionNormalDerivativesType, typename CallableType>
KOKKOS_FUNCTION void coupling_integral_dnshape(
    const int &ngllz, const int &ngllx, const IndexType &chunk_index,
    const IntersectionFieldViewType &intersection_field,
    const IntersectionFactor &intersection_factor,
    const ShapeFunctionNormalDerivativesType &shape_function_normal_derivatives,
    const CallableType &callback) {
  constexpr specfem::element::dimension_tag dimension_tag =
      specfem::element::dimension_tag::dim2;

  constexpr auto self_medium_tag = specfem::element_coupling::attributes<
      dimension_tag, IntersectionFactor::interface_tag>::self_medium();

  using PointIndexType =
      typename IndexType::iterator_type::index_type::index_type;
  using PointFieldType =
      specfem::point::acceleration<dimension_tag, self_medium_tag,
                                   IntersectionFieldViewType::using_simd>;

  constexpr int ncomp = PointFieldType::components;
  constexpr int nquad_intersection = IntersectionFactor::n_quad_intersection;

  const auto &team = chunk_index.get_policy_index();
  const int &num_edges = chunk_index.nedges();
  specfem::execution::for_each_level(
      specfem::execution::TeamThreadMDRangeIterator(team, num_edges, ngllz,
                                                    ngllx),
      [&](const auto &index) {
        const int iedge = index(0);
        const int iz = index(1);
        const int ix = index(2);

        PointFieldType result;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
        for (int icomp = 0; icomp < ncomp; icomp++) {
          result(icomp) = 0;
        }

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
        for (int iquad = 0; iquad < nquad_intersection; iquad++) {

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
          for (int icomp = 0; icomp < ncomp; icomp++) {
            result(icomp) +=
                intersection_field(iedge, iquad, icomp) *
                intersection_factor(iedge, iquad) *
                shape_function_normal_derivatives(iedge, iz, ix, iquad);
          }
        }
        // this index will always have ipoint = 0, but we will not use it
        const auto edge_index = chunk_index.get_iterator()(iedge).get_index();

        specfem::point::index<dimension_tag> self_index(edge_index.ispec, iz,
                                                        ix);
        callback(self_index, result);
      });
}

} // namespace specfem::algorithms
