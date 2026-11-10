#pragma once

#include "specfem/data_access.hpp"
#include "specfem/enums.hpp"

#include "specfem/datatype.hpp"

#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::algorithms {

namespace transfer_interpolate_impl {

/**
 * @brief Interpolates a face function at one point from precomputed 1D
 * interpolants.
 *
 * Computes \f$ \sum_{k_i, k_j} f(k_i, k_j) \ell_{k_i}(\xi) \ell_{k_j}(\eta)
 * \f$, with `interpolants(k, 0)` \f$ = \ell_k(\xi) \f$ and
 * `interpolants(k, 1)` \f$ = \ell_k(\eta) \f$. Zero interpolants give a zero
 * result.
 *
 * @tparam InterpolantsType Accessor indexable as `(igll, axis)`
 * @tparam FaceFunctionType Chunk-face field with `ngll` points per axis
 * @tparam PointVectorType Point vector receiving the result
 * @param ielem Face index within the chunk
 * @param interpolants 1D interpolants of the face at the point
 * @param face_function Field on the face
 * @param result Interpolated value, one entry per component
 */
template <typename InterpolantsType, typename FaceFunctionType,
          typename PointVectorType>
KOKKOS_INLINE_FUNCTION void
interpolate_point(const int &ielem, const InterpolantsType &interpolants,
                  const FaceFunctionType &face_function,
                  PointVectorType &result) {
  constexpr int ncomp = PointVectorType::components;

  for (int icomp = 0; icomp < ncomp; icomp++) {
    result(icomp) = 0;
  }

  // interpolate on tensor-product element
  for (int ipoint_axis1 = 0; ipoint_axis1 < FaceFunctionType::ngll;
       ipoint_axis1++) {
    const type_real coeff_axis1 = interpolants(ipoint_axis1, 0);
    for (int ipoint_axis2 = 0; ipoint_axis2 < FaceFunctionType::ngll;
         ipoint_axis2++) {
      const type_real transfer_coeff =
          coeff_axis1 * interpolants(ipoint_axis2, 1);

      for (int icomp = 0; icomp < ncomp; icomp++) {
        result(icomp) +=
            face_function(ielem, ipoint_axis1, ipoint_axis2, icomp) *
            transfer_coeff;
      }
    }
  }
}
} // namespace transfer_interpolate_impl

template <typename IndexType, typename TransferFunctionType,
          typename FaceFunctionType, typename PointwiseReturnType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<
    specfem::data_access::is_chunk_face<FaceFunctionType>::value &&
        specfem::data_access::is_point<IndexType>::value,
    void>
transfer_interpolate(const IndexType &point_index,
                     const TransferFunctionType &transfer_function,
                     const FaceFunctionType &face_function,
                     PointwiseReturnType &self_mapped_field) {

  const int &ielem = point_index.iface;

  transfer_interpolate_impl::interpolate_point(
      ielem, transfer_function.interpolants, face_function, self_mapped_field);
}

/**
 * @brief Applies the transposed interpolation at one guest-face node.
 *
 * For a 3D nonconforming pair, the host face's GLL nodes \f$ c = (c_i, c_j)
 * \f$ are the quadrature points. With the guest face's 1D interpolants stored
 * at each host node,
 * \f[
 *   T_{c,k} = \texttt{interpolants}(p, c_i, c_j, k_i, 0) \,
 *             \texttt{interpolants}(p, c_i, c_j, k_j, 1),
 * \f]
 * this computes, at guest node \f$ k = (k_i, k_j) \f$ of pair \f$ p \f$,
 * \f$ \mathrm{result}_d = \sum_{c_i, c_j} T_{c,k} \, F_d(c) \f$
 * for every component \f$ d \f$, where \f$ F \f$ is a value per host node.
 *
 * This is the transpose of `transfer_interpolate`, which evaluates
 * \f$ \sum_k T_{c,k} u(k) \f$ at a host node with the same \f$ T \f$. Host
 * nodes whose interpolant rows are zero (nodes owned by another pair)
 * contribute zero.
 *
 * @tparam IndexType Point face index (`iface`, `ipoint_i`, `ipoint_j`)
 * @tparam LocalIndexType Point face index whose `iface` is local to the chunk
 * @tparam InterpolantsViewType Rank-5 view indexable as
 * `(ipair, ci, cj, igll, axis)`
 * @tparam FluxViewType Rank-4 view `(iface_local, ci, cj, icomp)` with static
 * GLL and component extents (a plain static `Kokkos::View` or a
 * `specfem::datatype::VectorChunkFaceViewType`)
 * @tparam PointFieldType Point vector with `components` entries
 *
 * @param index Guest node: `iface` is the pair index \f$ p \f$ into
 * `interpolants`, `(ipoint_i, ipoint_j)` is \f$ (k_i, k_j) \f$
 * @param local_index The same node; its `iface` is the face position inside
 * the chunk, used to index `flux`
 * @param interpolants Guest-face 1D interpolants at the host nodes, by pair
 * @param flux Values \f$ F(c) \f$ at the host nodes of the chunk
 * @param result \f$ \sum_c T_{c,k} F(c) \f$, one entry per component (output)
 */
template <typename IndexType, typename LocalIndexType,
          typename InterpolantsViewType, typename FluxViewType,
          typename PointFieldType>
KOKKOS_INLINE_FUNCTION void
transfer_integrate(const IndexType &index, const LocalIndexType &local_index,
                   const InterpolantsViewType &interpolants,
                   const FluxViewType &flux, PointFieldType &result) {

  constexpr int ncomp = PointFieldType::components;
  constexpr int ngll = static_cast<int>(FluxViewType::static_extent(1));

  static_assert(InterpolantsViewType::rank() == 5,
                "interpolants must be indexed (ipair, ci, cj, igll, axis)");
  static_assert(FluxViewType::rank() == 4,
                "flux must be indexed (iface_local, ci, cj, icomp)");
  static_assert(ngll > 0 && FluxViewType::static_extent(2) == ngll,
                "flux must have equal static GLL extents");
  static_assert(FluxViewType::static_extent(3) == ncomp,
                "flux and result must have the same number of components");

  const int ipair = index.iface;
  const int iface_local = local_index.iface;
  const int ki = index.ipoint_i;
  const int kj = index.ipoint_j;

  for (int icomp = 0; icomp < ncomp; icomp++) {
    result(icomp) = 0;
  }

  for (int ci = 0; ci < ngll; ci++) {
    for (int cj = 0; cj < ngll; cj++) {
      const type_real transfer_coeff = interpolants(ipair, ci, cj, ki, 0) *
                                       interpolants(ipair, ci, cj, kj, 1);
      for (int icomp = 0; icomp < ncomp; icomp++) {
        result(icomp) += transfer_coeff * flux(iface_local, ci, cj, icomp);
      }
    }
  }
}
} // namespace specfem::algorithms

/**
 * @defgroup AlgorithmsTransfer Transfer Algorithms
 * @ingroup Algorithms
 */
