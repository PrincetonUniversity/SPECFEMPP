#pragma once

#include "accessor.hpp"
#include "data_class.hpp"
#include "domain_view.hpp"
#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::data_access {
enum class ContainerType { boundary, edge, domain };

namespace impl {
template <specfem::data_access::ContainerType ContainerType,
          specfem::dimension::type DimensionTag>
struct ContainerValueType;

template <>
struct ContainerValueType<specfem::data_access::ContainerType::domain,
                          specfem::dimension::type::dim2> {
  template <typename T, typename MemorySpace>
  using scalar_type = specfem::kokkos::DomainView2d<T, 3, MemorySpace>;
  template <typename T, typename MemorySpace>
  using vector_type = specfem::kokkos::DomainView2d<T, 4, MemorySpace>;
  template <typename T, typename MemorySpace>
  using tensor_type = specfem::kokkos::DomainView2d<T, 5, MemorySpace>;
};

template <>
struct ContainerValueType<specfem::data_access::ContainerType::domain,
                          specfem::dimension::type::dim3> {
  template <typename T, typename MemorySpace>
  using scalar_type = Kokkos::View<T ****, Kokkos::LayoutLeft, MemorySpace>;
  template <typename T, typename MemorySpace>
  using vector_type = Kokkos::View<T *****, Kokkos::LayoutLeft, MemorySpace>;
  template <typename T, typename MemorySpace>
  using tensor_type = Kokkos::View<T ******, Kokkos::LayoutLeft, MemorySpace>;
};

template <>
struct ContainerValueType<specfem::data_access::ContainerType::edge,
                          specfem::dimension::type::dim2> {
  template <typename T, typename MemorySpace>
  using scalar_type = Kokkos::View<T **, Kokkos::LayoutLeft, MemorySpace>;
  template <typename T, typename MemorySpace>
  using vector_type = Kokkos::View<T ***, Kokkos::LayoutLeft, MemorySpace>;
  template <typename T, typename MemorySpace>
  using tensor_type = Kokkos::View<T ****, Kokkos::LayoutLeft, MemorySpace>;
};
} // namespace impl

template <specfem::data_access::ContainerType ContainerType,
          specfem::data_access::DataClassType DataClass,
          specfem::dimension::type DimensionTag>
struct Container {
  constexpr static auto container_type = ContainerType;
  constexpr static auto data_class = DataClass;
  constexpr static auto dimension_tag = DimensionTag;

  template <typename T, typename MemorySpace>
  using scalar_type = typename impl::ContainerValueType<
      ContainerType, dimension_tag>::template scalar_type<T, MemorySpace>;

  template <typename T, typename MemorySpace>
  using vector_type = typename impl::ContainerValueType<
      ContainerType, dimension_tag>::template vector_type<T, MemorySpace>;

  template <typename T, typename MemorySpace>
  using tensor_type = typename impl::ContainerValueType<
      ContainerType, dimension_tag>::template tensor_type<T, MemorySpace>;
};

template <typename T, typename = void> struct is_container : std::false_type {};

template <typename T>
struct is_container<T, std::void_t<decltype(T::container_type)> >
    : std::true_type {};

} // namespace specfem::data_access
