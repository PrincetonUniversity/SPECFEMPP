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
KOKKOS_FORCEINLINE_FUNCTION void load_on_host(const IndexType &index,
                                              const ContainerType &field,
                                              AccessorTypes &...accessors) {

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  fields_impl::check_accessor_compatibility<AccessorTypes...>();

  const auto &current_field = field.template get_field<MediumTag>();

  fields_impl::load_after_field_access<false>(index, current_field,
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
KOKKOS_FORCEINLINE_FUNCTION void load_on_host(const IndexType &index,
                                              const ContainerType &field,
                                              AccessorTypes &...accessors) {

  constexpr static auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;

  // Check that all accessors have the same medium tag
  fields_impl::check_accessor_compatibility<AccessorTypes...>();

  using simd_accessor_type =
      std::integral_constant<bool, IndexType::using_simd>;
  simulation_field_impl::load_after_simd_dispatch<false>(
      simd_accessor_type(), index, field, accessors...);
  return;
}
} // namespace fields_impl

/**
 * @brief Host-side field data loading for spectral elements and debugging
 *
 * Public interface for loading field data from simulation fields into accessors
 * on the host (CPU). This function provides a unified interface for retrieving
 * values from multiple field components simultaneously for host-based
 * computations and debugging.
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
 * @note This function is host-only and should be called from host code
 * @note For device operations, use load_on_device instead
 * @note Supports both SIMD and non-SIMD index types
 *
 * Usage Examples:
 *
 * @code
 * // Host-side debugging: Load displacement field values
 * auto disp = specfem::point::displacement<...>(...);
 *
 * // Load field data on host for analysis
 * load_on_host(assembly_index, elastic_field, disp);
 *
 * // Access loaded values for debugging
 * std::cout << "Displacement X: " << disp(0) << std::endl;
 * std::cout << "Displacement Z: " << disp(1) << std::endl;
 *
 * // Host-based field processing
 * for (int i = 0; i < num_points; ++i) {
 *   load_on_host(host_index[i], field, accessor);
 *   // Process field data on host
 * }
 * @endcode
 */
template <
    typename IndexType, typename ContainerType, typename... AccessorTypes,
    typename std::enable_if_t<
        (specfem::data_access::is_field<AccessorTypes>::value && ...), int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_host(const IndexType &index,
                                              const ContainerType &field,
                                              AccessorTypes &...accessors) {
  fields_impl::load_on_host(index, field, accessors...);
  return;
}
} // namespace specfem::assembly
