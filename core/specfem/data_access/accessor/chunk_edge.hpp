#pragma once

#include "datatypes/simd.hpp"
#include "enumerations/interface.hpp"
#include "specfem/data_access/accessor.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::data_access {

template <specfem::data_access::DataClassType DataClass,
          specfem::dimension::type DimensionTag, bool UseSIMD>
struct Accessor<specfem::data_access::AccessorType::chunk_edge, DataClass,
                DimensionTag, UseSIMD> {
  constexpr static auto accessor_type =
      specfem::data_access::AccessorType::chunk_edge;
  constexpr static auto data_class = DataClass;
  constexpr static auto dimension_tag = DimensionTag;
  constexpr static bool using_simd = UseSIMD;

  template <typename T>
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type

  template <typename T, int nelements, int ngll>
  using scalar_type =
      Kokkos::View<typename simd<T>::datatype[nelements][ngll],
                   Kokkos::DefaultExecutionSpace::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  template <typename T, int nelements, int ngll, int components>
  using vector_type =
      Kokkos::View<typename simd<T>::datatype[nelements][ngll][components],
                   Kokkos::DefaultExecutionSpace::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  template <typename T, int nelements, int ngll, int components, int dimension>
  using tensor_type = Kokkos::View<
      typename simd<T>::datatype[nelements][ngll][components][dimension],
      Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
};

template <typename T, typename = void>
struct is_chunk_edge : std::false_type {};

template <typename T>
struct is_chunk_edge<
    T, std::enable_if_t<T::accessor_type ==
                        specfem::data_access::AccessorType::chunk_edge> >
    : std::true_type {};

} // namespace specfem::data_access
