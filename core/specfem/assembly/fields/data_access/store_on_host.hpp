#pragma once

#include "impl/store_access_functions.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/check_accessor_compatibility.hpp"
#include "specfem/assembly/fields/impl/store_access_functions.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

namespace fields_impl {
template <typename IndexType, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              ((specfem::data_access::is_assembly_index<IndexType>::value) &&
               (specfem::data_access::is_field<AccessorTypes>::value && ...)),
              int> = 0>
void store_on_host(const IndexType &index, const ContainerType &field,
                   AccessorTypes &...accessors) {

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  fields_impl::check_accessor_compatibility<AccessorTypes...>();

  const auto &current_field = field.template get_field<MediumTag>();

  fields_impl::store_after_field_access<false>(index, current_field,
                                               accessors...);

  return;
}

template <typename IndexType, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              ((specfem::data_access::is_index_type<IndexType>::value) &&
               (specfem::data_access::is_field<AccessorTypes>::value && ...)),
              int> = 0>
void store_on_host(const IndexType &index, const ContainerType &field,
                   AccessorTypes &...accessors) {

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  fields_impl::check_accessor_compatibility<AccessorTypes...>();

  using simd_accessor_type =
      std::integral_constant<bool, IndexType::using_simd>;

  simulation_field_impl::store_after_simd_dispatch<false>(
      simd_accessor_type(), index, field, accessors...);

  return;
}
} // namespace fields_impl

/**
 * @brief Host-side field data storage for spectral elements and debugging
 *
 * Public interface for storing field data from accessors into simulation fields
 * on the host (CPU). This function provides a unified interface for writing
 * values from multiple field components simultaneously for host-based
 * computations and debugging.
 *
 * @ingroup FieldsDataAccess
 *
 * @tparam IndexType Index type (specfem::point::assembly_index,
 * specfem::point::index)
 * @tparam ContainerType Simulation field container (2D/3D specializations)
 * @tparam AccessorTypes Variadic field accessor types
 *
 * @param index Spatial index (element + quadrature point information)
 * @param field Simulation field container holding medium-specific field data to
 * be modified
 * @param accessors Variable number of field accessors containing data to store
 *
 * @pre All accessors must have the same medium tag (e.g., all elastic or all
 * acoustic)
 * @pre All accessors must be field accessor types
 *
 * @note All accessors must target the same medium type (enforced at
 * compile-time)
 * @note This function is host-only and should be called from host code
 * @note For device operations, use store_on_device instead
 * @note No atomic operations needed on host as there's typically no thread
 * contention
 *
 * Usage Examples:
 *
 * @code
 * // Host-side field initialization or debugging
 * auto disp = specfem::point::displacement<...>(...);
 * auto vel = specfem::point::velocity<...>(...);
 *
 * // Set initial or computed values
 * disp(0) = initial_disp_x;
 * disp(1) = initial_disp_z;
 * velocity(0) = initial_vel_x;
 *
 * // Store values to field on host
 * store_on_host(assembly_index, elastic_field, disp, vel);
 *
 * // Host-based field processing and storage
 * for (int i = 0; i < num_points; ++i) {
 *   // Compute field values on host
 *   store_on_host(host_index[i], field, accessor1, accessor2);
 * }
 * @endcode
 */
template <
    typename IndexType, typename ContainerType, typename... AccessorTypes,
    typename std::enable_if_t<
        (specfem::data_access::is_field<AccessorTypes>::value && ...), int> = 0>
void store_on_host(const IndexType &index, const ContainerType &field,
                   AccessorTypes &...accessors) {
  fields_impl::store_on_host(index, field, accessors...);
  return;
}

} // namespace specfem::assembly
