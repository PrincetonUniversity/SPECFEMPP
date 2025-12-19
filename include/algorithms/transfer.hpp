#pragma once

#include "enumerations/interface.hpp"
#include "execution/for_each_level.hpp"
#include "execution/team_thread_md_range_iterator.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::algorithms {

/**
 * @brief Takes a chunk_edge::field and maps it onto the intersection, using a
 single-sided
 * transfer-function container.
 *
 * @tparam CoupledInterfaceType transfer function container type
 (specfem::assembly::coupled_interfaces_impl::stores_transfer_function_coupled<CoupledInterfaceType>::value
 must be true)
 * @tparam EdgeFieldType The chunk_edge field type
 * @tparam IntersectionFieldViewType - a view that the intersection field should
 be stored into
 * @param interface_data transfer function container
 * @param coupled_field The chunk_edge field to map from
 * @param intersection_field a view that the intersection field should be stored
 into
 */
template <
    typename IndexType, typename TransferFunctionType,
    typename EdgeFunctionType, typename IntersectionReturnCallback,
    typename std::enable_if_t<TransferFunctionType::connection_tag ==
                                  specfem::connections::type::nonconforming,
                              int> = 0>
KOKKOS_INLINE_FUNCTION void
transfer(const IndexType &chunk_edge_index,
         const TransferFunctionType &transfer_function,
         const EdgeFunctionType &edge_function,
         const IntersectionReturnCallback &callback) {

  static_assert(
      specfem::data_access::is_chunk_edge<IndexType>::value,
      "The index for a nonconforming compute_coupling must be a chunk_edge.");

  static_assert(specfem::data_access::is_chunk_edge<EdgeFunctionType>::value,
                "EdgeFunctionType must be a chunk_edge data type.");

  // TODO future consideration: use load_on_device for coupled field here.
  // We would want it to be a specialization, since we want to transfer more
  // things than just fields is there a better way of recovering global index?
  const auto &team = chunk_edge_index.get_policy_index();
  const int &num_edges = chunk_edge_index.nedges();

  using VectorPointViewType = specfem::datatype::VectorPointViewType<
      type_real, EdgeFunctionType::components, EdgeFunctionType::using_simd>;

  constexpr int ncomp = EdgeFunctionType::components;

  specfem::execution::for_each_level(
      specfem::execution::TeamThreadMDRangeIterator(
          team, num_edges, TransferFunctionType::n_quad_intersection),
      [&](const auto &index) {
        const int iedge = index(0);
        const int iquad = index(1);
        VectorPointViewType intersection_point_view;

        for (int icomp = 0; icomp < ncomp; icomp++) {
          intersection_point_view(icomp) = 0;

          for (int ipoint_edge = 0;
               ipoint_edge < TransferFunctionType::n_quad_element;
               ipoint_edge++) {
            intersection_point_view(icomp) +=
                edge_function(iedge, ipoint_edge, icomp) *
                transfer_function(iedge, ipoint_edge, iquad);
          }
        }

        callback(index, intersection_point_view);
      });
}

} // namespace specfem::algorithms
