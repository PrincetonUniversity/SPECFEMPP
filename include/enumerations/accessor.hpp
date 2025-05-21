#include "data_class.hpp"
#include "dimension.hpp"

namespace specfem::accessor {
enum class type { point, chunk_element };

namespace impl {
template <specfem::accessor::type AccessorType> struct AccessorValueType;

template <> struct AccessorValueType<specfem::accessor::type::point> {
  template <typename T, bool UseSIMD>
  using scalar_type = typename specfem::datatype::simd<T, UseSIMD>::datatype;

  template <typename T, int dimension, bool UseSIMD>
  using vector_type =
      typename specfem::datatype::ScalarPointViewType<T, dimension, UseSIMD>;

  template <typename T, int dimension, int components, bool UseSIMD>
  using tensor_type =
      typename specfem::datatype::VectorPointViewType<T, dimension, components,
                                                      UseSIMD>;
};
} // namespace impl

template <specfem::accessor::type AccessorType,
          specfem::data_class::type DataClass,
          specfem::dimension::type DimensionTag, bool UseSIMD>
struct Accessor {
  constexpr static auto accessor_type = AccessorType;
  constexpr static auto data_class = DataClass;
  constexpr static auto dimension_tag = DimensionTag;
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD data type

  template <typename T>
  using scalar_type = typename impl::AccessorValueType<
      AccessorType>::template scalar_type<type_real, UseSIMD>;

  template <typename T, int dimension>
  using vector_type = typename impl::AccessorValueType<
      AccessorType>::template vector_type<type_real, dimension, UseSIMD>;

  template <typename T, int dimension, int components>
  using tensor_type = typename impl::AccessorValueType<
      AccessorType>::template tensor_type<type_real, dimension, 3, UseSIMD>;
};

} // namespace specfem::accessor
