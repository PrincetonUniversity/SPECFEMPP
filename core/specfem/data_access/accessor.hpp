#pragma once

#include "data_class.hpp"
#include "enumerations/interface.hpp"
#include <type_traits>

namespace specfem::data_access {
enum class AccessorType { point, chunk_element, chunk_edge };

template <specfem::data_access::AccessorType AccessorType,
          specfem::data_access::DataClassType DataClass,
          specfem::dimension::type DimensionTag, bool UseSIMD>
struct Accessor;

template <typename T, typename = void> struct is_accessor : std::false_type {};

template <typename T>
struct is_accessor<
    T, std::enable_if_t<std::is_same_v<decltype(T::accessor_type),
                                       specfem::data_access::AccessorType> > >
    : std::true_type {};

} // namespace specfem::data_access

#include "accessor/chunk_edge.hpp"
#include "accessor/chunk_element.hpp"
#include "accessor/point_accessor.hpp"
