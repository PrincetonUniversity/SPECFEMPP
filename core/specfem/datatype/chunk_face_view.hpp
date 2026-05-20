#pragma once

#include "chunk_ndim_view.hpp"
#include <utility>

namespace specfem {
namespace datatype {
template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfFaces, int NumberOfGLLPoints, bool UseSIMD = false,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged>>
struct ScalarChunkFaceViewType
    : ChunkNDimViewType<T, DimensionTag, NumberOfFaces, NumberOfGLLPoints, 2,
                        std::integer_sequence<int>> {};

template <
    typename T, specfem::element::dimension_tag DimensionTag, int NumberOfFaces,
    int Components, int NumberOfGLLPoints, bool UseSIMD = false,
    typename MemorySpace = Kokkos::DefaultExecutionSpace::scratch_memory_space,
    typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged>>
struct VectorChunkFaceViewType
    : ChunkNDimViewType<T, DimensionTag, NumberOfFaces, NumberOfGLLPoints, 2,
                        std::integer_sequence<int, Components>> {
  constexpr static int components = Components;
};

template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfFaces, int Components, int NumberOfDimensions,
          int NumberOfGLLPoints, bool UseSIMD = false,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged>>
struct TensorChunkFaceViewType
    : ChunkNDimViewType<
          T, DimensionTag, NumberOfFaces, NumberOfGLLPoints, 2,
          std::integer_sequence<int, Components, NumberOfDimensions>> {
  constexpr static int components = Components;
  constexpr static int dimensions = NumberOfDimensions;
};
} // namespace datatype
} // namespace specfem
