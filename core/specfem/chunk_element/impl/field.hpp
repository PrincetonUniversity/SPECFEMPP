#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::chunk_element::impl {

template <int ChunkSize, int NGLL, specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD,
          typename ValueType>
class field_without_accessor {
public:
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;
  constexpr static int nelements = ChunkSize;
  constexpr static int ngll = NGLL;
  constexpr static auto medium_tag = MediumTag;
  using value_type = ValueType;
  constexpr static bool isChunkViewType = true;
  constexpr static bool isScalarViewType = true;
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type

private:
  value_type m_data;

public:
  KOKKOS_FORCEINLINE_FUNCTION field_without_accessor() = default;

  KOKKOS_FORCEINLINE_FUNCTION
  field_without_accessor(const value_type &data_in) : m_data(data_in) {}

  // Index operator for accessing components
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION typename value_type::value_type &
  operator()(Indices... indices) const {
    return m_data(indices...);
  }
};

template <int ChunkSize, int NGLL, specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD,
          typename ValueType>
class remove_accessor_attribute {
public:
  using type = field_without_accessor<ChunkSize, NGLL, DimensionTag, MediumTag,
                                      UseSIMD, ValueType>;
};

template <int ChunkSize, int NGLL, specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::data_access::DataClassType DataClass, bool UseSIMD>
class field : public specfem::data_access::Accessor<
                  specfem::data_access::AccessorType::chunk_element, DataClass,
                  DimensionTag, UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::chunk_element, DataClass,
      DimensionTag, UseSIMD>;

public:
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;
  constexpr static int nelements = ChunkSize;
  constexpr static int ngll = NGLL;
  using simd = typename base_type::template simd<type_real>; ///< SIMD type
  using value_type =
      typename base_type::template vector_type<type_real, nelements, ngll,
                                               components>;
  constexpr static auto medium_tag = MediumTag;

  constexpr static bool isChunkViewType = true;
  constexpr static bool isScalarViewType = true;

private:
  value_type m_data;

public:
  KOKKOS_FORCEINLINE_FUNCTION field() = default;

  template <typename ScratchMemorySpace>
  KOKKOS_FORCEINLINE_FUNCTION field(const ScratchMemorySpace &scratch_space)
      : m_data(scratch_space) {}

  // Index operator for accessing components
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION typename value_type::value_type &
  operator()(Indices... indices) const {
    return m_data(indices...);
  }

  constexpr static std::size_t shmem_size() { return value_type::shmem_size(); }

  KOKKOS_INLINE_FUNCTION const typename remove_accessor_attribute<
      ChunkSize, NGLL, DimensionTag, MediumTag, UseSIMD, value_type>::type
  field_without_accessor() const {
    return { m_data };
  }
};

} // namespace specfem::chunk_element::impl
