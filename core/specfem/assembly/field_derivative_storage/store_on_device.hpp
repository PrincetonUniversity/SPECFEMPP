#pragma once

#include "specfem/assembly/field_derivative_storage.hpp"
#include "specfem/assembly/field_derivative_storage/impl/field_derivative_medium.hpp"
#include "specfem/data_access/check_compatibility.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief Store field derivatives for a GLL point into the assembly
 *        FieldDerivativeStorage container (device kernel).
 *
 * Dispatches to the matching field_derivative_medium via get_container<M,P,A>()
 * and delegates to its store_device_values() method.
 *
 * If @p point_fd is specfem::point::null_field_derivatives this function
 * returns immediately — no storage exists for this element combination.
 *
 * @ingroup FieldDerivativeStorageDataAccess
 *
 * @tparam IndexType                      Index type
 * (specfem::point::index<...>)
 * @tparam FieldDerivativesContainerType  Assembly FieldDerivativeStorage
 * @tparam PointFieldDerivativesType      specfem::point::field_derivatives<...>
 *                                        or
 * specfem::point::null_field_derivatives
 *
 * @param index     GLL point index (global ispec, iz, [iy,] ix).
 * @param container Assembly field-derivative storage container
 *                  (device-accessible).
 * @param point_fd  Input: point field-derivatives struct with values to store,
 *                  or null_field_derivatives for a no-op.
 */
template <typename IndexType, typename FieldDerivativesContainerType,
          typename PointFieldDerivativesType,
          typename std::enable_if_t<
              specfem::data_access::is_index_type<IndexType>::value &&
                  specfem::data_access::is_field_derivatives<
                      FieldDerivativesContainerType>::value &&
                  specfem::data_access::is_field_derivatives<
                      PointFieldDerivativesType>::value,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
store_on_device(const IndexType &index,
                const FieldDerivativesContainerType &container,
                const PointFieldDerivativesType &point_fd) {

  auto compatibility = specfem::data_access::check_compatibility<
      IndexType, FieldDerivativesContainerType, PointFieldDerivativesType>();
  // Incompatible types; this likely indicates a programming error in the
  // caller. Trigger a compile-time error with a helpful message.
  static_assert(compatibility::value,
                "Incompatible types for field derivative storing.");

  constexpr auto medium_tag =
      PointFieldDerivativesType::field_derivatives_medium_type::medium_tag;

  container.template get_container<medium_tag>().store_device_values(index,
                                                                     point_fd);
}

template <typename IndexType, typename FieldDerivativesContainerType>
KOKKOS_FORCEINLINE_FUNCTION void
store_on_device(const IndexType &index,
                const FieldDerivativesContainerType &container,
                specfem::point::null_field_derivatives_type point_fd) {
  // No-op for null_field_derivatives (no storage allocated)
}

} // namespace specfem::assembly
