#pragma once

#include "data_class.hpp"
#include "datatypes/point_view.hpp"
#include "enumerations/interface.hpp"
#include <type_traits>

namespace specfem::data_access {
enum class AccessorType { point, chunk_element };

namespace impl {
template <specfem::data_access::AccessorType AccessorType>
struct AccessorValueType;

template <>
struct AccessorValueType<specfem::data_access::AccessorType::point> {
  template <typename T, bool UseSIMD>
  using scalar_type = typename specfem::datatype::simd<T, UseSIMD>::datatype;

  template <typename T, int dimension, bool UseSIMD>
  using vector_type =
      typename specfem::datatype::VectorPointViewType<T, dimension, UseSIMD>;

  template <typename T, int components, int dimension, bool UseSIMD>
  using tensor_type =
      typename specfem::datatype::TensorPointViewType<T, components, dimension,
                                                      UseSIMD>;
};
} // namespace impl

template <specfem::data_access::AccessorType AccessorType,
          specfem::data_access::DataClassType DataClass,
          specfem::dimension::type DimensionTag, bool UseSIMD>
struct Accessor {
  constexpr static auto accessor_type = AccessorType;
  constexpr static auto data_class = DataClass;
  constexpr static auto dimension_tag = DimensionTag;
  constexpr static bool using_simd = UseSIMD;

  template <typename T>
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type

  template <typename T>
  using scalar_type = typename impl::AccessorValueType<
      AccessorType>::template scalar_type<T, UseSIMD>;

  template <typename T, int dimension>
  using vector_type = typename impl::AccessorValueType<
      AccessorType>::template vector_type<T, dimension, UseSIMD>;

  template <typename T, int components, int dimension>
  using tensor_type = typename impl::AccessorValueType<
      AccessorType>::template tensor_type<T, components, dimension, UseSIMD>;
};

template <typename T, typename = void> struct is_point : std::false_type {};

template <typename T>
struct is_point<T, std::enable_if_t<T::accessor_type ==
                                    specfem::data_access::AccessorType::point> >
    : std::true_type {};

template <typename T, typename = void>
struct is_chunk_element : std::false_type {};

template <typename T>
struct is_chunk_element<
    T, std::enable_if_t<T::accessor_type ==
                        specfem::data_access::AccessorType::chunk_element> >
    : std::true_type {};

template <typename T, typename = void> struct is_accessor : std::false_type {};

template <typename T>
struct is_accessor<
    T, std::enable_if_t<(specfem::data_access::is_point<T>::value ||
                         specfem::data_access::is_chunk_element<T>::value)> >
    : std::true_type {};

} // namespace specfem::data_access
