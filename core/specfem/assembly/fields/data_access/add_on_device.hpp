#pragma once

#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/add_access_functions.hpp"
#include "specfem/assembly/fields/impl/check_accessor_compatibility.hpp"
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
KOKKOS_FORCEINLINE_FUNCTION void add_on_device(const IndexType &index,
                                               const ContainerType &field,
                                               AccessorTypes &...accessors) {

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  fields_impl::check_accessor_compatibility<AccessorTypes...>();

  const auto &current_field = field.template get_field<MediumTag>();

  fields_impl::add_after_field_access<true>(index, current_field, accessors...);

  return;
}
} // namespace fields_impl

/**
 * @defgroup FieldsDataAccess Fields Data Access Functions
 *
 */

/**
 * @brief High-performance device-side field accumulation for spectral elements
 *
 * Public interface for accumulating values into simulation fields on the default compute device (GPU if GPU support is enabled)
 * devices. This function provides a unified interface for adding values to
 * multiple field components simultaneously.
 *
 *
 * @ingroup FieldsDataAccess
 *
 * @tparam IndexType Index type (specfem::point::index,
 * specfem::point::assembly_index)
 * @tparam ContainerType Simulation field container (2D/3D specializations)
 * @tparam AccessorTypes Variadic field accessor types
 * (specfem::point::displacement, specfem::point::velocity,
 * specfem::point::acceleration)
 *
 * @param index Spatial index (element + quadrature point information)
 * @param field Simulation field container holding medium-specific field data
 * @param accessors Variable number of field accessors for simultaneous updates
 *
 * @note All accessors must target the same medium type (enforced at
 * compile-time)
 * @note This function should be called from device kernels
 *
 * Usage Examples:
 *
 * @code
 * // Elastic medium: Add to displacement and velocity fields
 * auto disp = specfem::point::displacement<...>(...);
 * auto vel = specfem::point::velocity<...>(...);
 *
 * // Update fields at a specific quadrature point
 * disp(0) += delta_disp_x;
 * disp(1) += delta_disp_z;
 *
 * // Single kernel call updates multiple components
 * add_on_device(assembly_index, elastic_field, disp, vel);
 *
 * @endcode
 */
template <
    typename IndexType, typename ContainerType, typename... AccessorTypes,
    typename std::enable_if_t<
        (specfem::data_access::is_field<AccessorTypes>::value && ...), int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void add_on_device(const IndexType &index,
                                               const ContainerType &field,
                                               AccessorTypes &...accessors) {
  // Delegate to implementation with additional assembly index validation
  fields_impl::add_on_device(index, field, accessors...);
  return;
}
} // namespace specfem::assembly
