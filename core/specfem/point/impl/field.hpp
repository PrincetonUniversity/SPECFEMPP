#pragma once

#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::point::impl {

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::data_access::DataClassType DataClass, bool UseSIMD>
class field : public specfem::data_access::Accessor<
                  specfem::data_access::AccessorType::point, DataClass,
                  DimensionTag, UseSIMD> {
private:
  using base_type =
      specfem::data_access::Accessor<specfem::data_access::AccessorType::point,
                                     DataClass, DimensionTag, UseSIMD>;
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

public:
  using simd = typename base_type::template simd<type_real>; ///< SIMD type
  using value_type =
      typename base_type::template vector_type<type_real, components>;
  constexpr static auto medium_tag = MediumTag;

private:
  value_type m_data;

public:
  KOKKOS_FORCEINLINE_FUNCTION field() = default;

  KOKKOS_FORCEINLINE_FUNCTION constexpr
  field(const typename value_type::value_type initializer) {
    for (std::size_t icomp = 0; icomp < components; ++icomp)
      m_data(icomp) = initializer;
  }

  template <typename... Args,
            typename = std::enable_if_t<sizeof...(Args) == components> >
  KOKKOS_FORCEINLINE_FUNCTION constexpr field(Args &&...args)
      : m_data(std::forward<Args>(args)...) {}

  KOKKOS_FORCEINLINE_FUNCTION typename value_type::value_type
  operator()(const std::size_t icomp) const {
    return m_data(icomp);
  }

  KOKKOS_FORCEINLINE_FUNCTION typename value_type::value_type &
  operator()(const std::size_t icomp) {
    return m_data(icomp);
  }

  KOKKOS_FORCEINLINE_FUNCTION bool operator==(const field &other) const {
    return (this->m_data == other.m_data);
  }

  KOKKOS_FORCEINLINE_FUNCTION bool operator!=(const field &other) const {
    return !(*this == other);
  }
};

} // namespace specfem::point::impl
