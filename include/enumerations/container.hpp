#include "data_class.hpp"
#include "dimension.hpp"

namespace specfem::container {
enum class type { boundary, interface, domain };

template <specfem::container::type ContainerType> struct ContainerValueType;

template <> struct ContainerValueType<specfem::container::type::domain> {
  template <typename T, typename MemorySpace>
  using scalar_type =
      specfem::datatype::ScalarDomainViewType<T, 3, MemorySpace>;
  template <typename T, typename MemorySpace>
  using vector_type =
      specfem::datatype::VectorDomainViewType<T, 4, MemorySpace>;
  template <typename T, typename MemorySpace>
  using tensor_type =
      specfem::datatype::VectorDomainViewType<T, 5, MemorySpace>;
};

template <specfem::container::type ContainerType,
          specfem::data_class::type DataClass,
          specfem::dimension::type DimensionTag>
struct Container {
  constexpr static auto container_type = ContainerType;
  constexpr static auto data_class = DataClass;
  constexpr static auto dimension_tag = DimensionTag;
};

} // namespace specfem::container
