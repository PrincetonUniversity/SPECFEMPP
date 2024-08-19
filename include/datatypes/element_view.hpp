#pragma once

#include <Kokkos_Core.hpp>

namespace specfem {
namespace datatype {
template <typename T, int NumberOfGLLPoints, int Components,
          typename MemorySpace, typename MemoryTraits, bool UseSIMD = false>
struct ScalarElementViewType
    : public Kokkos::View<
          typename specfem::datatype::simd<T, UseSIMD>::datatype
              [NumberOfGLLPoints][NumberOfGLLPoints][Components],
          Kokkos::LayoutLeft, MemorySpace, MemoryTraits> {
  using simd = specfem::datatype::simd<T, UseSIMD>;
  using type = Kokkos::View<
      typename simd::datatype[NumberOfGLLPoints][NumberOfGLLPoints][Components],
      Kokkos::LayoutLeft, MemorySpace, MemoryTraits>;
  using value_type = typename type::value_type;
  using base_type = T;
  constexpr static int ngll = NumberOfGLLPoints;
  constexpr static int components = Components;
  constexpr static bool isPointViewType = false;
  constexpr static bool isElementViewType = true;
  constexpr static bool isChunkViewType = false;
  constexpr static bool isDomainViewType = false;
  constexpr static bool isScalarViewType = true;
  constexpr static bool isVectorViewType = false;

  KOKKOS_FUNCTION
  ScalarElementViewType() = default;

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION ScalarElementViewType(const ScratchMemorySpace &scratch_space)
      : Kokkos::View<
            value_type[NumberOfGLLPoints][NumberOfGLLPoints][Components],
            Kokkos::LayoutLeft, MemorySpace, MemoryTraits>(scratch_space) {}
};

template <typename T, int NumberOfGLLPoints, int Components,
          int NumberOfDimensions, typename MemorySpace, typename MemoryTraits,
          bool UseSIMD = false>
struct VectorElementViewType
    : public Kokkos::View<typename specfem::datatype::simd<T, UseSIMD>::datatype
                              [NumberOfGLLPoints][NumberOfGLLPoints]
                              [NumberOfDimensions][Components],
                          Kokkos::LayoutLeft, MemorySpace, MemoryTraits> {
  using simd = specfem::datatype::simd<T, UseSIMD>;
  using type =
      Kokkos::View<typename simd::datatype[NumberOfGLLPoints][NumberOfGLLPoints]
                                          [NumberOfDimensions][Components],
                   Kokkos::LayoutLeft, MemorySpace, MemoryTraits>;
  using value_type = typename type::value_type;
  using base_type = T;
  constexpr static int ngll = NumberOfGLLPoints;
  constexpr static int components = Components;
  constexpr static int dimensions = NumberOfDimensions;
  constexpr static bool isPointViewType = false;
  constexpr static bool isElementViewType = true;
  constexpr static bool isChunkViewType = false;
  constexpr static bool isDomainViewType = false;
  constexpr static bool isScalarViewType = false;
  constexpr static bool isVectorViewType = true;

  KOKKOS_FUNCTION
  VectorElementViewType() = default;

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION VectorElementViewType(const ScratchMemorySpace &scratch_space)
      : Kokkos::View<value_type[NumberOfGLLPoints][NumberOfGLLPoints]
                               [NumberOfDimensions][Components],
                     Kokkos::LayoutLeft, MemorySpace, MemoryTraits>(
            scratch_space) {}
};

} // namespace datatype
} // namespace specfem
