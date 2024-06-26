#pragma once

#include <Kokkos_Core.hpp>

namespace specfem {
namespace datatype {

template <typename T, int NumberOfElements, int NumberOfGLLPoints,
          int Components, typename MemorySpace, typename MemoryTraits>
struct ScalarChunkViewType
    : public Kokkos::View<
          T[NumberOfElements][NumberOfGLLPoints][NumberOfGLLPoints][Components],
          Kokkos::LayoutLeft, MemorySpace, MemoryTraits> {
  using type = Kokkos::View<
      T[NumberOfElements][NumberOfGLLPoints][NumberOfGLLPoints][Components],
      Kokkos::LayoutLeft, MemorySpace, MemoryTraits>;
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
      : Kokkos::View<T[NumberOfElements][NumberOfGLLPoints][NumberOfGLLPoints]
                      [Components],
                     Kokkos::LayoutLeft, MemorySpace, MemoryTraits>(
            scratch_memory_space) {}
};

template <typename T, int NumberOfElements, int NumberOfGLLPoints,
          int Components, int NumberOfDimensions, typename MemorySpace,
          typename MemoryTraits>
struct VectorChunkViewType
    : public Kokkos::View<T[NumberOfElements][NumberOfGLLPoints]
                           [NumberOfGLLPoints][NumberOfDimensions][Components],
                          Kokkos::LayoutLeft, MemorySpace, MemoryTraits> {
  using type =
      Kokkos::View<T[NumberOfElements][NumberOfGLLPoints][NumberOfGLLPoints]
                    [NumberOfDimensions][Components],
                   Kokkos::LayoutLeft, MemorySpace, MemoryTraits>;
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
      : Kokkos::View<T[NumberOfElements][NumberOfGLLPoints][NumberOfGLLPoints]
                      [NumberOfDimensions][Components],
                     Kokkos::LayoutLeft, MemorySpace, MemoryTraits>(
            scratch_memory_space) {}
};

} // namespace datatype
} // namespace specfem
