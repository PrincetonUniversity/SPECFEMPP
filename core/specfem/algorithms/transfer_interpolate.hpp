#pragma once

#include "Kokkos_Macros.hpp"
#include "specfem/data_access.hpp"
#include "specfem/enums.hpp"
#include "specfem/execution.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::algorithms {
/**
 * @brief Takes a chunk_edge or chunk_face field and maps it by coordinates.
 *
 * @tparam IndexType chunk_edge or chunk_face index
 * @tparam TransferFunctionType transfer function container type
 (should be of DataClassType transfer_coordinates)
 * @tparam FaceFunctionType The chunk_edge or chunk_face field type
 * @tparam IntersectionReturnCallback the callback function that retrieves the
 pointwise value at each coordinate.
 * @ingroup AlgorithmsTransfer
 */
template <typename IndexType, typename TransferFunctionType,
          typename FaceFunctionType, typename IntersectionReturnCallback,
          typename = void>
KOKKOS_INLINE_FUNCTION void
transfer_interpolate(const IndexType &chunk_face_index,
                     const TransferFunctionType &transfer_function,
                     const FaceFunctionType &edge_function,
                     const IntersectionReturnCallback &callback);
// can we code-share with core/specfem/quadrature/gll/lagrange_poly.cpp?

template <
    typename IndexType, typename TransferFunctionType,
    typename FaceFunctionType, typename IntersectionReturnCallback,
    std::enable_if_t<specfem::data_access::is_chunk_edge<IndexType>::value>>
KOKKOS_INLINE_FUNCTION void
transfer_interpolate(const IndexType &chunk_face_index,
                     const TransferFunctionType &transfer_function,
                     const FaceFunctionType &edge_function,
                     const IntersectionReturnCallback &callback) {
  static_assert(specfem::data_access::is_chunk_edge<FaceFunctionType>::value,
                "EdgeFunctionType must be a chunk_edge data type.");

  // TODO future consideration: use load_on_device for coupled field here.
  // We would want it to be a specialization, since we want to transfer more
  // things than just fields is there a better way of recovering global index?
  const auto &team = chunk_face_index.get_policy_index();
  const int &num_edges = chunk_face_index.nedges();

  using VectorPointViewType = specfem::datatype::VectorPointViewType<
      type_real, FaceFunctionType::components, FaceFunctionType::using_simd>;

  constexpr int ncomp = FaceFunctionType::components;

  specfem::execution::for_each_level(
      specfem::execution::TeamThreadMDRangeIterator(
          team, num_edges, TransferFunctionType::n_quad_element),
      [&](const auto &index) {
        // TODO
      });
}

template <
    typename IndexType, typename TransferFunctionType,
    typename FaceFunctionType, typename IntersectionReturnCallback,
    std::enable_if_t<specfem::data_access::is_chunk_face<IndexType>::value>>
KOKKOS_INLINE_FUNCTION void
transfer_interpolate(const IndexType &chunk_face_index,
                     const TransferFunctionType &transfer_function,
                     const FaceFunctionType &face_function,
                     const IntersectionReturnCallback &callback) {
  static_assert(specfem::data_access::is_chunk_face<FaceFunctionType>::value,
                "FaceFunctionType must be a chunk_face data type.");

  // TODO future consideration: use load_on_device for coupled field here.
  // We would want it to be a specialization, since we want to transfer more
  // things than just fields is there a better way of recovering global index?
  const auto &team = chunk_face_index.get_policy_index();
  const int &num_faces = chunk_face_index.nfaces();

  using VectorPointViewType = specfem::datatype::VectorPointViewType<
      type_real, FaceFunctionType::components, FaceFunctionType::using_simd>;

  constexpr int ncomp = FaceFunctionType::components;

  specfem::execution::for_each_level(
      specfem::execution::TeamThreadMDRangeIterator(
          team, num_faces, TransferFunctionType::n_quad_element,
          TransferFunctionType::n_quad_element),
      [&](const auto &index) {
        const int iedge = index(0);
        const int iquad1 = index(1);
        const int iquad2 = index(2);

        const type_real face_coord1 =
            transfer_function(iedge, iquad1, iquad2, 0);
        const type_real face_coord2 =
            transfer_function(iedge, iquad1, iquad2, 1);

        VectorPointViewType intersection_point_view;

        for (int icomp = 0; icomp < ncomp; icomp++) {
          intersection_point_view(icomp) = 0;
        }
        for (int ipoint_axis1; ipoint_axis1 < FaceFunctionType::n_quad_element;
             ipoint_axis1++) {
          const type_real coeff_axis1 =
              0; // TODO compute L_ipoint_axis1(facecoord1)
          for (int ipoint_axis2;
               ipoint_axis2 < FaceFunctionType::n_quad_element;
               ipoint_axis2++) {

            const type_real coeff_axis2 =
                0; // TODO compute L_ipoint_axis2(facecoord2)
            const type_real transfer_coeff = coeff_axis1 * coeff_axis2;

            for (int icomp = 0; icomp < ncomp; icomp++) {
              intersection_point_view(icomp) +=
                  face_function(iedge, ipoint_axis1, ipoint_axis2, icomp) *
                  transfer_coeff;
            }
          }
        }

        callback(index, intersection_point_view);
      });
}
} // namespace specfem::algorithms

/**
 * @defgroup AlgorithmsTransfer Transfer Algorithms
 * @ingroup Algorithms
 */
