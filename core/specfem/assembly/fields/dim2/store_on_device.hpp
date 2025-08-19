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
KOKKOS_FORCEINLINE_FUNCTION void store_on_device(const IndexType &index,
                                                 const ContainerType &field,
                                                 AccessorTypes &...accessors) {

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  fields_impl::check_accessor_compatibility<AccessorTypes...>();

  const auto &current_field = field.template get_field<MediumTag>();

  fields_impl::store_after_field_access<true>(index, current_field,
                                              accessors...);

  return;
}
} // namespace fields_impl

/**
 * @brief Stores field data on the device at the specified index from the given
 * accessors.
 *
 * @ingroup FieldDataAccess
 *
 * This function stores field data from accessors into the field container at
 * the specified index location. The operation is performed on the device (GPU)
 * and is optimized for parallel execution within Kokkos kernels.
 *
 * @tparam IndexType The type of the index (assembly index, point, or chunk
 * element).
 * @tparam ContainerType The type of the container holding the field data.
 * @tparam AccessorTypes The types of the accessors used to access the field
 * data.
 *
 * @param index The index specifying the location or entity for field storage.
 * @param field The container holding the field data to be modified.
 * @param accessors One or more accessor objects containing data to store in the
 * field.
 *
 * @pre All accessors must have the same medium tag (e.g., all elastic or all
 * acoustic).
 * @pre All accessors must be field accessor types.
 *
 * @note This function is device-only (KOKKOS_FORCEINLINE_FUNCTION) and should
 * be called from device kernels.
 *
 */
template <
    typename IndexType, typename ContainerType, typename... AccessorTypes,
    typename std::enable_if_t<
        (specfem::data_access::is_field<AccessorTypes>::value && ...), int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void store_on_device(const IndexType &index,
                                                 const ContainerType &field,
                                                 AccessorTypes &...accessors) {
  fields_impl::store_on_device(index, field, accessors...);
  return;
}

} // namespace specfem::assembly
