#pragma once

#include "specfem/assembly/field_derivative_storage/impl/field_derivative_medium.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

/**
 * @defgroup FieldDerivativeStorageDataAccess Field Derivative Storage Data
 * Access Functions
 *
 */

// ---------------------------------------------------------------------------
// Non-empty (non-none attenuation), non-SIMD index
// ---------------------------------------------------------------------------

/**
 * @brief Load field derivatives from compact storage at a non-SIMD GLL point.
 *
 * @ingroup FieldDerivativeStorageDataAccess
 *
 * @tparam PointFDType      Point field-derivatives type
 * (specfem::point::field_derivatives)
 * @tparam IndexType        Index type (non-SIMD, has ispec/iz/ix)
 * @tparam D,M,P,A          Tag template parameters of the storage medium
 *
 * @param index    GLL point index
 * @param medium   Compact field-derivative storage for this tag combination
 * @param point_fd Output point field derivatives to populate
 */
template <typename PointFDType, typename IndexType,
          specfem::element::dimension_tag D, specfem::element::medium_tag M,
          specfem::element::property_tag P, specfem::element::attenuation_tag A,
          std::enable_if_t<!IndexType::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(
    const IndexType &index,
    const specfem::assembly::impl::field_derivative_medium<D, M, P, A> &medium,
    PointFDType &point_fd) {
  const int i = medium.ispec_to_compact(index.ispec);
  for (int ic = 0; ic < PointFDType::components; ++ic) {
    for (int id = 0; id < PointFDType::num_dimensions; ++id) {
      point_fd.du[ic][id] = medium.du_storage(i, index.iz, index.ix, ic, id);
    }
  }
}

// ---------------------------------------------------------------------------
// Non-empty, SIMD index — gather across lanes
// ---------------------------------------------------------------------------

/**
 * @brief Load field derivatives from compact storage at a SIMD GLL point
 *        (gather across SIMD lanes).
 *
 * @ingroup FieldDerivativeStorageDataAccess
 */
template <typename PointFDType, typename IndexType,
          specfem::element::dimension_tag D, specfem::element::medium_tag M,
          specfem::element::property_tag P, specfem::element::attenuation_tag A,
          std::enable_if_t<IndexType::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(
    const IndexType &index,
    const specfem::assembly::impl::field_derivative_medium<D, M, P, A> &medium,
    PointFDType &point_fd) {
  using simd_type = typename PointFDType::simd;
  constexpr int simd_size = simd_type::size();
  for (int lane = 0; lane < simd_size; ++lane) {
    const int i = medium.ispec_to_compact(index.ispec + lane);
    for (int ic = 0; ic < PointFDType::components; ++ic) {
      for (int id = 0; id < PointFDType::num_dimensions; ++id) {
        point_fd.du[ic][id][lane] =
            medium.du_storage(i, index.iz, index.ix, ic, id);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Empty specialization (attenuation_none) — compile-time no-op
// ---------------------------------------------------------------------------

/**
 * @brief No-op load for attenuation_none — zero overhead at compile time.
 *
 * @ingroup FieldDerivativeStorageDataAccess
 */
template <typename PointFDType, typename IndexType,
          specfem::element::dimension_tag D, specfem::element::medium_tag M,
          specfem::element::property_tag P>
KOKKOS_FORCEINLINE_FUNCTION void
load_on_device(const IndexType &,
               const specfem::assembly::impl::field_derivative_medium<
                   D, M, P, specfem::element::attenuation_tag::none> &,
               PointFDType &) {
  // intentionally empty
}

} // namespace specfem::assembly
