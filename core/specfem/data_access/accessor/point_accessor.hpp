#pragma once

#include "datatypes/point_view.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/interface.hpp"
#include "specfem/data_access/accessor.hpp"

namespace specfem::data_access {

template <specfem::data_access::DataClassType DataClass,
          specfem::dimension::type DimensionTag, bool UseSIMD>
struct Accessor<specfem::data_access::AccessorType::point, DataClass,
                DimensionTag, UseSIMD> {
  constexpr static auto accessor_type =
      specfem::data_access::AccessorType::point;
  constexpr static auto data_class = DataClass;
  constexpr static auto dimension_tag = DimensionTag;
  constexpr static bool using_simd = UseSIMD;

  template <typename T>
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type

  template <typename T> using scalar_type = typename simd<T>::datatype;

  template <typename T, int dimension>
  using vector_type =
      typename specfem::datatype::VectorPointViewType<T, dimension, UseSIMD>;

  template <typename T, int components, int dimension>
  using tensor_type =
      typename specfem::datatype::TensorPointViewType<T, components, dimension,
                                                      UseSIMD>;
};

template <typename T, typename = void> struct is_point : std::false_type {};

template <typename T>
struct is_point<T, std::enable_if_t<T::accessor_type ==
                                    specfem::data_access::AccessorType::point> >
    : std::true_type {};

} // namespace specfem::data_access
