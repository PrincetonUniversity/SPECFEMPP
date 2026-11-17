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
} // namespace specfem::algorithms

/**
 * @defgroup AlgorithmsTransfer Transfer Algorithms
 * @ingroup Algorithms
 */
