#pragma once

#include "enumerations/interface.hpp"
#include "execution/for_each_level.hpp"
#include "execution/team_thread_md_range_iterator.hpp"
#include "specfem/assembly.hpp"
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
 * @tparam dimension_tag dimension of the simulation
 * @tparam IndexType The chunk_edge iterator type
 * @tparam IntersectionFieldViewType The type of `intersection_field`
 * @tparam ChunkEdgeWeightJacobianType A nonconforming chunk_edge accessor
 * holding `intersection_factor`
 * @tparam CallableType The callback function, which will be given the point
 * index and corresponding evaluated integral
 * @param nonconforming_interfaces - assembly.nonconforming_interfaces struct
 * @param lagrange_derivative - Local derivatives of shape functions on the
 * element
 * @param chunk_index - the outer index (chunk_edge) that gets iterated for
 * points
 * @param intersection_field - the field to integrate
 * @param weight_jacobian - nonconforming chunk_edge accessor holding
 * `intersection_factor`
 * @param intersection_normal_covariant_edgelocal - the normal vector at each
 * intersection point in contravariant local coordinate basis, with the first
 * component in the normal direction.
 * @param callback - callback function to capture integral values
 * @ingroup AlgorithmsIntegration
 */
template <specfem::dimension::type dimension_tag, typename QuadratureType,
          typename IndexType, typename IntersectionFieldViewType,
          typename IntersectionFactor, typename IntersectionJacobianType,
          typename TransferFunctionDerivativeType, typename CallableType>
KOKKOS_FUNCTION void coupling_integral_dnshape(
    const specfem::assembly::nonconforming_interfaces<dimension_tag>
        &nonconforming_interfaces,
    const QuadratureType &lagrange_derivative, const IndexType &chunk_index,
    const IntersectionFieldViewType &intersection_field,
    const IntersectionFactor &intersection_factor,
    const IntersectionJacobianType &intersection_normal_covariant_edgelocal,
    const TransferFunctionDerivativeType
        &transfer_function_self_derivative /* This could be loaded just like
                                         transfer_function_self */
    ,
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
  constexpr int nquad_element = IntersectionFactor::n_quad_element;

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename IndexType::iterator_type::index_type &index) {
        const auto self_index = index.get_index();
        const auto self_index_local = index.get_local_index();
        const bool ipoint_n_is_ix =
            self_index.edge_type == specfem::mesh_entity::dim2::type::left ||
            self_index.edge_type == specfem::mesh_entity::dim2::type::right;
        const int &iedge = self_index_local.iedge;
        SelfTransferFunctionType transfer_function_self;
        specfem::assembly::load_on_device(self_index, nonconforming_interfaces,
                                          transfer_function_self);

        const int &ipoint_s = self_index.ipoint;
        for (int ipoint_n = 0; ipoint_n < nquad_element; ++ipoint_n) {
          // iterate backwards from the edge (this gets called for each gllxz)
          specfem::point::index<specfem::dimension::type::dim2> interior_index(
              self_index.ispec, self_index.iz, self_index.ix);

          // sample derivative of interior_index shape function, in normal
          // (covector) direction, at edge coordinate.
          const type_real dlagrange_dn =
              ipoint_n_is_ix
                  ? lagrange_derivative.xi(ipoint_n, self_index.ix)
                  : lagrange_derivative.gamma(ipoint_n, self_index.iz);

          // we may want to refactor this at some point. For now, this should be
          // fine to update the index point.
          switch (self_index.edge_type) {
          case specfem::mesh_entity::dim2::type::right:
            interior_index.ix = (nquad_element - 1) - ipoint_n;
            break;
          case specfem::mesh_entity::dim2::type::top:
            interior_index.iz = (nquad_element - 1) - ipoint_n;
            break;
          case specfem::mesh_entity::dim2::type::left:
            interior_index.ix = ipoint_n;
            break;
          case specfem::mesh_entity::dim2::type::bottom:
            interior_index.iz = ipoint_n;
            break;
          }

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
              // dshape_dn, local-normal derivative (derivative of
              // normal-direction L, which is normally kronecker delta
              // indicating on edge)
              type_real dshape_dn =
                  intersection_normal_covariant_edgelocal(iedge, iquad, 0) *
                  (dlagrange_dn * transfer_function_self(iquad));
              // dshape_dn, local-tangential derivative (differentiate
              // transfer_function_self instead of normal-direction L)
              if (ipoint_n == 0) {
                dshape_dn +=
                    intersection_normal_covariant_edgelocal(iedge, iquad, 1) *
                    transfer_function_self_derivative(iedge, ipoint_s, iquad);
              }
              result(icomp) += intersection_field(iedge, iquad, icomp) *
                               intersection_factor(iedge, iquad) * dshape_dn;
            }
          }

          callback(self_index, result);
        }
      });
}

} // namespace specfem::algorithms
