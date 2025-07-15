#pragma once

#include "dim2/data_access.hpp"
#include "specfem/data_access.hpp"

namespace specfem::assembly {

/**
 * @defgroup FieldDataAccess
 */

/**
 * @brief Load fields at a given quadrature point index on the device
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::assembly::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param field Wavefield container
 * @param point_field Point field to store the field values (output)
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<
              specfem::data_access::is_accessor<ViewType>::value &&
                  specfem::data_access::is_field<ViewType>::value,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const IndexType &index,
                                                const WavefieldContainer &field,
                                                ViewType &point_field) {
  fields_impl::impl_load<true>(index, field, point_field);
}

/**
 * @brief Load fields at a given quadrature point index on the host
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::assembly::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param field Wavefield container
 * @param point_field Point field to store the field values (output)
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<
              specfem::data_access::is_accessor<ViewType>::value &&
                  specfem::data_access::is_field<ViewType>::value,
              int> = 0>
inline void load_on_host(const IndexType &index,
                         const WavefieldContainer &field,
                         ViewType &point_field) {
  fields_impl::impl_load<false>(index, field, point_field);
}

/**
 * @brief Store fields at a given quadrature point index on the device
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::assembly::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param point_field Point field to store the field values from
 * @param field Wavefield container
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<
              specfem::data_access::is_accessor<ViewType>::value &&
                  specfem::data_access::is_field<ViewType>::value,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
store_on_device(const IndexType &index, const ViewType &point_field,
                const WavefieldContainer &field) {
  fields_impl::impl_store<true>(index, point_field, field);
}

/**
 * @brief Store fields at a given quadrature point index on the host
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::assembly::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param point_field Point field to store the field values from
 * @param field Wavefield container
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<
              specfem::data_access::is_accessor<ViewType>::value &&
                  specfem::data_access::is_field<ViewType>::value,
              int> = 0>
inline void store_on_host(const IndexType &index, const ViewType &point_field,
                          const WavefieldContainer &field) {
  fields_impl::impl_store<false>(index, point_field, field);
}

/**
 * @brief Add fields at a given quadrature point index on the device
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::assembly::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param point_field Point field to add to the field values from
 * @param field Wavefield container
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<
              specfem::data_access::is_accessor<ViewType>::value &&
                  specfem::data_access::is_field<ViewType>::value,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
add_on_device(const IndexType &index, const ViewType &point_field,
              const WavefieldContainer &field) {
  fields_impl::impl_add<true>(index, point_field, field);
}

/**
 * @brief Add fields at a given quadrature point index on the host
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::assembly::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param point_field Point field to add to the field values from
 * @param field Wavefield container
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<
              specfem::data_access::is_accessor<ViewType>::value &&
                  specfem::data_access::is_field<ViewType>::value,
              int> = 0>
inline void add_on_host(const IndexType &index, const ViewType &point_field,
                        const WavefieldContainer &field) {
  fields_impl::impl_add<false>(index, point_field, field);
}

/**
 * @brief Atomic add fields at a given quadrature point index on the device
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::assembly::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param point_field Point field to add to the field values from
 * @param field Wavefield container
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<
              specfem::data_access::is_accessor<ViewType>::value &&
                  specfem::data_access::is_field<ViewType>::value,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
atomic_add_on_device(const IndexType &index, const ViewType &point_field,
                     const WavefieldContainer &field) {
  fields_impl::impl_atomic_add<true>(index, point_field, field);
}

/**
 * @brief Atomic add fields at a given quadrature point index on the host
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::assembly::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param point_field Point field to add to the field values from
 * @param field Wavefield container
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<
              specfem::data_access::is_accessor<ViewType>::value &&
                  specfem::data_access::is_field<ViewType>::value,
              int> = 0>
inline void atomic_add_on_host(const IndexType &index,
                               const ViewType &point_field,
                               const WavefieldContainer &field) {
  fields_impl::impl_atomic_add<false>(index, point_field, field);
}

/**
 * @brief Load fields at a given element on the device
 *
 * @ingroup FieldDataAccess
 *
 * @tparam MemberType Member type. Needs to be of @ref Kokkos::TeamPolicy
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::assembly::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::element::field
 * @param member Team member
 * @param index Spectral element index
 * @param field Wavefield container
 * @param element_field Element field to store the field values (output)
 */
template <typename MemberType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isElementFieldType, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_on_device(const MemberType &member, const int &index,
               const WavefieldContainer &field, ViewType &element_field) {
  fields_impl::impl_load<true>(member, index, field, element_field);
}

/**
 * @brief Load fields at a given element on the host
 *
 * @ingroup FieldDataAccess
 *
 * @tparam MemberType Member type. Needs to be of @ref Kokkos::TeamPolicy
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::assembly::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::element::field
 * @param member Team member
 * @param index Spectral element index
 * @param field Wavefield container
 * @param element_field Element field to store the field values (output)
 */
template <typename MemberType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isElementFieldType, int> = 0>
inline void load_on_host(const MemberType &member, const int &index,
                         const WavefieldContainer &field,
                         ViewType &element_field) {
  fields_impl::impl_load<false>(member, index, field, element_field);
}

/**
 * @brief Store fields for a given chunk of elements on the device
 *
 * @ingroup FieldDataAccess
 *
 * @tparam MemberType Member type. Needs to be of @ref Kokkos::TeamPolicy
 * @tparam ChunkIteratorType Chunk iterator type. Needs to be of @ref
 * specfem::iterator::chunk_iterator
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::assembly::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::element::field
 * @param member Team member
 * @param iterator Chunk iterator specifying the elements
 * @param field Wavefield container
 * @param chunk_field Chunk field to store the field values (output)
 */
template <typename MemberType, typename ChunkIteratorType,
          typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isChunkFieldType, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_on_device(const MemberType &member, const ChunkIteratorType &iterator,
               const WavefieldContainer &field, ViewType &chunk_field) {
  fields_impl::impl_load<true>(member, iterator, field, chunk_field);
}

/**
 * @brief Store fields for a given chunk of elements on the host
 *
 * @ingroup FieldDataAccess
 *
 * @tparam MemberType Member type. Needs to be of @ref Kokkos::TeamPolicy
 * @tparam ChunkIteratorType Chunk iterator type. Needs to be of @ref
 * specfem::iterator::chunk_iterator
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::assembly::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::element::field
 * @param member Team member
 * @param iterator Chunk iterator specifying the elements
 * @param field Wavefield container
 * @param chunk_field Chunk field to store the field values (output)
 */
template <typename MemberType, typename ChunkIteratorType,
          typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isChunkFieldType, int> = 0>
inline void
load_on_host(const MemberType &member, const ChunkIteratorType &iterator,
             const WavefieldContainer &field, ViewType &chunk_field) {
  fields_impl::impl_load<false>(member, iterator, field, chunk_field);
}

template <typename ChunkIndexType, typename WavefieldContainer,
          typename ViewType,
          typename std::enable_if_t<ViewType::isChunkFieldType, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const ChunkIndexType &index,
                                                const WavefieldContainer &field,
                                                ViewType &chunk_field) {
  fields_impl::impl_load<true>(index, field, chunk_field);
}

template <typename ChunkIndexType, typename WavefieldContainer,
          typename ViewType,
          typename std::enable_if_t<ViewType::isChunkFieldType, int> = 0>
inline void load_on_host(const ChunkIndexType &index,
                         const WavefieldContainer &field,
                         ViewType &chunk_field) {
  fields_impl::impl_load<false>(index, field, chunk_field);
}

} // namespace specfem::assembly
