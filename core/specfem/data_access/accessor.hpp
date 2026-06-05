#pragma once

#include "data_class.hpp"
#include "specfem/datatype/accessor_type.hpp"
#include "specfem/enums.hpp"
#include <type_traits>

namespace specfem::data_access {

/**
 * @brief Type-safe data accessor for simulation components.
 *
 * Provides specialized access patterns for different data types and
 * computational contexts. Enables efficient data loading/storing with proper
 * indexing and vectorization support.
 *
 * @tparam AccessorType Access pattern (point/element/chunk)
 * @tparam DataClass Type of data (properties/fields/indices)
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @tparam UseSIMD Enable SIMD vectorization
 */
template <specfem::datatype::AccessorType AccessorType,
          specfem::data_access::DataClassType DataClass,
          specfem::element::dimension_tag DimensionTag, bool UseSIMD>
struct Accessor;

/**
 * @brief Type trait to detect accessor types.
 *
 * Checks if a type implements the Accessor interface by detecting
 * the accessor_type static member.
 */
template <typename T, typename = void> struct is_accessor : std::false_type {};

template <typename T>
struct is_accessor<
    T, std::enable_if_t<std::is_same_v<decltype(T::accessor_type),
                                       specfem::datatype::AccessorType>>>
    : std::true_type {};

/**
 * @brief Type trait to detect a codimension-1 accessor type (chunk_edge for
 * dim2, chunk_face for dim3).
 */
template <typename T, typename = void>
struct is_codim1_chunk : std::false_type {};

template <typename T>
struct is_codim1_chunk<
    T, std::enable_if_t<
           (T::accessor_type == specfem::datatype::AccessorType::chunk_edge &&
            T::dimension_tag == specfem::element::dimension_tag::dim2) ||
           (T::accessor_type == specfem::datatype::AccessorType::chunk_face &&
            T::dimension_tag == specfem::element::dimension_tag::dim3)>>
    : std::true_type {};

/**
 * @brief A data accessor that holds no data, and contains null shmem info, etc.
 *
 * This container can be used as a placeholder when one kernel needs some data
 * type (say in a scratch view) only for certain template arguments.
 */
struct EmptyAccessor {

  /**
   * @brief Capture any constructer configuration. Should do nothing.
   */
  template <typename... Args> EmptyAccessor(Args...){};
  /**
   * @brief Get shared memory size requirement
   * @return Size in bytes needed for scratch memory
   */
  constexpr static int shmem_size() { return 0; }
};

} // namespace specfem::data_access

#include "accessor/chunk_edge.hpp"
#include "accessor/chunk_element.hpp"
#include "accessor/chunk_face.hpp"
#include "accessor/element.hpp"
#include "accessor/point_accessor.hpp"
