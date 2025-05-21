#pragma once

#include "data_class.hpp"
#include "dimension.hpp"
#include "domain_view.hpp"

namespace specfem::container {
enum class type { boundary, interface, domain };

namespace impl {
template <specfem::container::type ContainerType> struct ContainerValueType;

template <> struct ContainerValueType<specfem::container::type::domain> {
  template <typename T, int N, typename MemorySpace>
  using scalar_type = specfem::kokkos::DomainView2d<T, N, MemorySpace>;
  template <typename T, int N, typename MemorySpace>
  using vector_type = specfem::kokkos::DomainView2d<T, N, MemorySpace>;
  template <typename T, int N, typename MemorySpace>
  using tensor_type = specfem::kokkos::DomainView2d<T, N, MemorySpace>;
};
} // namespace impl

template <specfem::container::type ContainerType,
          specfem::data_class::type DataClass,
          specfem::dimension::type DimensionTag>
struct Container {
  constexpr static auto container_type = ContainerType;
  constexpr static auto data_class = DataClass;
  constexpr static auto dimension_tag = DimensionTag;

  template <typename T, typename MemorySpace>
  using scalar_type = typename impl::ContainerValueType<
      ContainerType>::template scalar_type<T, 3, MemorySpace>;

  template <typename T, typename MemorySpace>
  using vector_type = typename impl::ContainerValueType<
      ContainerType>::template vector_type<T, 4, MemorySpace>;

  template <typename T, typename MemorySpace>
  using tensor_type = typename impl::ContainerValueType<
      ContainerType>::template tensor_type<T, 5, MemorySpace>;
};

} // namespace specfem::container
