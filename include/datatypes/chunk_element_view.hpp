#pragma once

#include <Kokkos_Core.hpp>

namespace specfem {
namespace datatype {

template <typename T, int NumberOfElements, int NumberOfGLLPoints,
          int Components, typename MemorySpace, typename MemoryTraits,
          bool UseSIMD = false>
struct ScalarChunkViewType
    : public Kokkos::View<typename specfem::datatype::simd<T, UseSIMD>::datatype
                              [NumberOfElements][NumberOfGLLPoints]
                              [NumberOfGLLPoints][Components],
                          MemorySpace, MemoryTraits> {
  using simd = specfem::datatype::simd<T, UseSIMD>;
  using type =
      Kokkos::View<typename simd::datatype[NumberOfElements][NumberOfGLLPoints]
                                          [NumberOfGLLPoints][Components],
                   MemorySpace, MemoryTraits>;
  using value_type = typename type::value_type;
  using base_type = T;
  constexpr static int nelements = NumberOfElements;
  constexpr static int ngll = NumberOfGLLPoints;
  constexpr static int components = Components;
  constexpr static bool isPointViewType = false;
  constexpr static bool isElementViewType = false;
  constexpr static bool isChunkViewType = true;
  constexpr static bool isDomainViewType = false;
  constexpr static bool isScalarViewType = true;
  constexpr static bool isVectorViewType = false;

  KOKKOS_FUNCTION
  ScalarChunkViewType() = default;

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION ScalarChunkViewType(
      const ScratchMemorySpace &scratch_memory_space)
      : Kokkos::View<value_type[NumberOfElements][NumberOfGLLPoints]
                               [NumberOfGLLPoints][Components],
                     MemorySpace, MemoryTraits>(scratch_memory_space) {}
};

template <typename T, int NumberOfElements, int NumberOfGLLPoints,
          int Components, int NumberOfDimensions, typename MemorySpace,
          typename MemoryTraits, bool UseSIMD = false>
struct VectorChunkViewType
    : public Kokkos::View<
          typename specfem::datatype::simd<T, UseSIMD>::datatype
              [NumberOfElements][NumberOfGLLPoints][NumberOfGLLPoints]
              [NumberOfDimensions][Components],
          MemorySpace, MemoryTraits> {
  using simd = specfem::datatype::simd<T, UseSIMD>;
  using type = typename Kokkos::View<
      typename simd::datatype[NumberOfElements][NumberOfGLLPoints]
                             [NumberOfGLLPoints][NumberOfDimensions]
                             [Components],
      MemorySpace, MemoryTraits>;
  using value_type = typename type::value_type;
  using base_type = T;
  constexpr static int nelements = NumberOfElements;
  constexpr static int ngll = NumberOfGLLPoints;
  constexpr static int components = Components;
  constexpr static int dimensions = NumberOfDimensions;
  constexpr static bool isPointViewType = false;
  constexpr static bool isElementViewType = false;
  constexpr static bool isChunkViewType = true;
  constexpr static bool isDomainViewType = false;
  constexpr static bool isScalarViewType = false;
  constexpr static bool isVectorViewType = true;

  KOKKOS_FUNCTION
  VectorChunkViewType() = default;

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION VectorChunkViewType(
      const ScratchMemorySpace &scratch_memory_space)
      : Kokkos::View<
            value_type[NumberOfElements][NumberOfGLLPoints][NumberOfGLLPoints]
                      [NumberOfDimensions][Components],
            MemorySpace, MemoryTraits>(scratch_memory_space) {}
};

} // namespace datatype
} // namespace specfem
