#pragma once

#include <Kokkos_Core.hpp>

namespace specfem {
namespace datatype {

template <typename T, int NumberOfGLLPoints, int Components,
          typename MemorySpace, typename MemoryTraits>
struct ScalarElementViewType
    : public Kokkos::View<T[NumberOfGLLPoints][NumberOfGLLPoints][Components],
                          Kokkos::LayoutLeft, MemorySpace, MemoryTraits> {
  using type = Kokkos::View<T[NumberOfGLLPoints][NumberOfGLLPoints],
                            Kokkos::LayoutLeft, MemorySpace, MemoryTraits>;
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

  KOKKOS_FUNCTION
  ScalarElementViewType(const type_real *data_ptr)
      : Kokkos::View<T[NumberOfGLLPoints][NumberOfGLLPoints][Components]>(
            data_ptr) {}

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION ScalarElementViewType(const ScratchMemorySpace &scratch_space)
      : Kokkos::View<T[NumberOfGLLPoints][NumberOfGLLPoints][Components],
                     Kokkos::LayoutLeft, MemorySpace, MemoryTraits>(
            scratch_space) {}
};

template <typename T, int NumberOfGLLPoints, int Components,
          int NumberOfDimensions, typename MemorySpace, typename MemoryTraits>
struct VectorElementViewType
    : public Kokkos::View<T[NumberOfGLLPoints][NumberOfGLLPoints]
                           [NumberOfDimensions][Components],
                          Kokkos::LayoutLeft, MemorySpace, MemoryTraits> {
  using type = Kokkos::View<
      T[NumberOfGLLPoints][NumberOfGLLPoints][NumberOfDimensions][Components],
      Kokkos::LayoutLeft, MemorySpace, MemoryTraits>;
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

  KOKKOS_FUNCTION
  VectorElementViewType(const type_real *data_ptr)
      : Kokkos::View<T[NumberOfGLLPoints][NumberOfGLLPoints][NumberOfDimensions]
                      [Components]>(data_ptr) {}

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION VectorElementViewType(const ScratchMemorySpace &scratch_space)
      : Kokkos::View<T[NumberOfGLLPoints][NumberOfGLLPoints][NumberOfDimensions]
                      [Components],
                     Kokkos::LayoutLeft, MemorySpace, MemoryTraits>(
            scratch_space) {}
};

} // namespace datatype
} // namespace specfem
