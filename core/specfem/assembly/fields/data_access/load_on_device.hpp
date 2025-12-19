#pragma once

#include "impl/load_access_functions.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/check_accessor_compatibility.hpp"
#include "specfem/assembly/fields/impl/load_access_functions.hpp"
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
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const IndexType &index,
                                                const ContainerType &field,
                                                AccessorTypes &...accessors) {

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  fields_impl::check_accessor_compatibility<AccessorTypes...>();

  const auto &current_field = field.template get_field<MediumTag>();

  fields_impl::load_after_field_access<true>(index, current_field,
                                             accessors...);

  return;
}

template <typename IndexType, typename ContainerType, typename... AccessorTypes,
          typename std::enable_if_t<
              ((specfem::data_access::is_index_type<IndexType>::value) &&
               (specfem::data_access::is_point<IndexType>::value ||
                specfem::data_access::is_chunk_element<IndexType>::value ||
                specfem::data_access::is_chunk_edge<IndexType>::value) &&
               (specfem::data_access::is_field<AccessorTypes>::value && ...)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const IndexType &index,
                                                const ContainerType &field,
                                                AccessorTypes &...accessors) {

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  fields_impl::check_accessor_compatibility<AccessorTypes...>();

  using simd_accessor_type =
      std::integral_constant<bool, IndexType::using_simd>;
  simulation_field_impl::load_after_simd_dispatch<true>(
      simd_accessor_type(), index, field, accessors...);
  return;
}
} // namespace fields_impl

/**
 * @brief Device-side field data loading for spectral elements
 *
 * Public interface for loading field data from simulation fields into accessors
 * on GPU devices. This function provides a unified interface for retrieving
 * values from multiple field components simultaneously with optimized device
 * memory access patterns.
 *
 * @ingroup FieldsDataAccess
 *
 * @tparam IndexType Index type (specfem::point::index,
 * specfem::chunk::element_index, specfem::chunk::edge_index with SIMD support)
 * @tparam ContainerType Simulation field container (2D/3D specializations)
 * @tparam AccessorTypes Variadic field accessor types
 *
 * @param index Spatial index (element + quadrature point information with
 * optional SIMD support)
 * @param field Simulation field container holding medium-specific field data
 * @param accessors Variable number of field accessors to populate with loaded
 * data
 *
 * @pre All accessors must have the same medium tag (e.g., all elastic or all
 * acoustic)
 * @pre All accessors must be field accessor types
 *
 * @note All accessors must target the same medium type (enforced at
 * compile-time)
 * @note This function is device-only (KOKKOS_FORCEINLINE_FUNCTION) and should
 * be called from device kernels
 * @note Supports both SIMD and non-SIMD index types for optimal performance
 *
 * Usage Examples:
 *
 * @code
 * // Elastic medium: Load displacement and velocity fields
 * auto disp = specfem::point::displacement<...>(...);
 * auto vel = specfem::point::velocity<...>(...);
 *
 * // Single kernel call loads multiple components from field
 * load_on_device(simd_index, elastic_field, disp, vel);
 *
 * // Access loaded values
 * auto disp_x = disp(0);
 * auto disp_z = disp(1);
 *
 * // Use in device kernels for field retrieval
 * Kokkos::parallel_for("load_kernel", range, KOKKOS_LAMBDA(int i) {
 *   load_on_device(index[i], field, accessor1, accessor2);
 *   // Process loaded field data
 * });
 * @endcode
 */
template <
    typename IndexType, typename ContainerType, typename... AccessorTypes,
    typename std::enable_if_t<
        (specfem::data_access::is_field<AccessorTypes>::value && ...), int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const IndexType &index,
                                                const ContainerType &field,
                                                AccessorTypes &...accessors) {
  fields_impl::load_on_device(index, field, accessors...);
  return;
}
} // namespace specfem::assembly
