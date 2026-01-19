#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly/nonconforming_interfaces.hpp"
#include "specfem/execution.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {
template <specfem::dimension::type DimensionTag> struct assembly;
}

/**
 * @file coupling_integral.hpp
 * @brief 1D coupling integrals for interface computations
 * @ingroup AlgorithmsIntegration
 *
 * Provides algorithms for computing integrals over 1D intersections
 * between different mesh elements in coupled interface problems.
 */

namespace specfem::algorithms {

/**
 * @brief Takes a field `intersection_field` on the intersection and computes,
 * for each self GLL point, the integral of `intersection_field` times the shape
 * function at that point. `intersection_field` should be call-accessible (e.g.
 * Kokkos::View) with shape:
 *
 * (chunk_size, n_quad_intersection, self::components())
 *
 * After handling any other intersection forces, boundary conditions, etc. the
 * result can be `atomic_add`ed to the acceleration field.
 *
 * @tparam dimension_tag dimension of the simulation
 * @tparam IndexType The chunk_edge iterator type
 * @tparam IntersectionFieldViewType The type of `intersection_field`
 * @tparam ChunkEdgeWeightJacobianType A nonconforming chunk_edge accessor
 * holding `intersection_factor`
 * @tparam CallableType The callback function, which will be given the point
 * index and corresponding evaluated integral
 * @param assembly - assembly struct
 * @param chunk_index - the outer index (chunk_edge) that gets iterated for
 * points
 * @param intersection_field - the field to integrate
 * @param weight_jacobian - nonconforming chunk_edge accessor holding
 * `intersection_factor`
 * @param callback - callback function to capture integral values
 * @ingroup AlgorithmsIntegration
 */
template <specfem::dimension::type dimension_tag, typename IndexType,
          typename IntersectionFieldViewType, typename IntersectionFactor,
          typename CallableType>
KOKKOS_FUNCTION void
coupling_integral(const specfem::assembly::assembly<dimension_tag> &assembly,
                  const IndexType &chunk_index,
                  const IntersectionFieldViewType &intersection_field,
                  const IntersectionFactor &intersection_factor,
                  const CallableType &callback) {

  constexpr auto self_medium_tag = specfem::interface::attributes<
      dimension_tag, IntersectionFactor::interface_tag>::self_medium();

  using PointIndexType =
      typename IndexType::iterator_type::index_type::index_type;
  using PointFieldType =
      specfem::point::acceleration<dimension_tag, self_medium_tag,
                                   IntersectionFieldViewType::using_simd>;
  using SelfTransferFunctionType = specfem::point::transfer_function_self<
      IntersectionFactor::n_quad_intersection, dimension_tag,
      IntersectionFactor::interface_tag, IntersectionFactor::boundary_tag>;

  constexpr int ncomp = PointFieldType::components;
  constexpr int nquad_intersection = IntersectionFactor::n_quad_intersection;

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename IndexType::iterator_type::index_type &index) {
        const auto self_index = index.get_index();
        const auto self_index_local = index.get_local_index();

        SelfTransferFunctionType transfer_function_self;
        specfem::assembly::load_on_device(self_index,
                                          assembly.nonconforming_interfaces,
                                          transfer_function_self);

        PointFieldType result;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
        for (int icomp = 0; icomp < ncomp; icomp++) {
          result(icomp) = 0;
        }
        const int &iedge = self_index_local.iedge;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
        for (int iquad = 0; iquad < nquad_intersection; iquad++) {

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
          for (int icomp = 0; icomp < ncomp; icomp++) {
            result(icomp) += intersection_field(iedge, iquad, icomp) *
                             intersection_factor(iedge, iquad) *
                             transfer_function_self(iquad);
          }
        }

        callback(self_index, result);
      });
}

} // namespace specfem::algorithms

/**
 * @defgroup AlgorithmsIntegration Integration Algorithms
 * @ingroup Algorithms
 */
