#pragma once

#include "enumerations/interface.hpp"

#include "specfem/datatype.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::data_access {

template <specfem::data_access::DataClassType DataClass,
          specfem::element::dimension_tag DimensionTag, bool UseSIMD>
struct Accessor<specfem::datatype::AccessorType::element, DataClass,
                DimensionTag, UseSIMD> {
  constexpr static auto accessor_type =
      specfem::datatype::AccessorType::element;
  constexpr static auto data_class = DataClass;
  constexpr static auto dimension_tag = DimensionTag;
  constexpr static bool using_simd = UseSIMD;
};

template <typename T, typename = void> struct is_element : std::false_type {};

template <typename T>
struct is_element<T,
                  std::enable_if_t<T::accessor_type ==
                                   specfem::datatype::AccessorType::element> >
    : std::true_type {};

} // namespace specfem::data_access
