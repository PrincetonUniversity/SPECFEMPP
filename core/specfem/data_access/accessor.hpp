#pragma once

#include "data_class.hpp"
#include "enumerations/accessor_type.hpp"
#include "enumerations/interface.hpp"
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
template <specfem::data_access::AccessorType AccessorType,
          specfem::data_access::DataClassType DataClass,
          specfem::dimension::type DimensionTag, bool UseSIMD>
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
                                       specfem::data_access::AccessorType> > >
    : std::true_type {};

} // namespace specfem::data_access

#include "accessor/chunk_edge.hpp"
#include "accessor/chunk_element.hpp"
#include "accessor/element.hpp"
#include "accessor/point_accessor.hpp"
