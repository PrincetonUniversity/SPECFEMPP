#pragma once

#include "chunk_ndim_view.hpp"
#include "specfem/datatype/accessor_type.hpp"
#include <utility>

namespace specfem {
namespace datatype {
template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfFaces, int NumberOfGLLPoints, bool UseSIMD = false,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged>>
struct ScalarChunkFaceViewType
    : public ChunkNDimViewType<T, DimensionTag, NumberOfFaces,
                               NumberOfGLLPoints, 2, std::integer_sequence<int>,
                               specfem::datatype::AccessorType::chunk_face,
                               UseSIMD, MemorySpace, MemoryTraits> {
  constexpr static auto accessor_type =
      specfem::datatype::AccessorType::chunk_face; ///< Accessor type for
                                                   ///< identifying the
                                                   ///< class
};

template <
    typename T, specfem::element::dimension_tag DimensionTag, int NumberOfFaces,
    int Components, int NumberOfGLLPoints, bool UseSIMD = false,
    typename MemorySpace = Kokkos::DefaultExecutionSpace::scratch_memory_space,
    typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged>>
struct VectorChunkFaceViewType
    : public ChunkNDimViewType<T, DimensionTag, NumberOfFaces,
                               NumberOfGLLPoints, 2,
                               std::integer_sequence<int, Components>,
                               specfem::datatype::AccessorType::chunk_face,
                               UseSIMD, MemorySpace, MemoryTraits> {
  constexpr static auto accessor_type =
      specfem::datatype::AccessorType::chunk_face; ///< Accessor type for
                                                   ///< identifying the
                                                   ///< class
  constexpr static int components = Components;
};

template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfFaces, int Components, int NumberOfDimensions,
          int NumberOfGLLPoints, bool UseSIMD = false,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged>>
struct TensorChunkFaceViewType
    : public ChunkNDimViewType<
          T, DimensionTag, NumberOfFaces, NumberOfGLLPoints, 2,
          std::integer_sequence<int, Components, NumberOfDimensions>,
          specfem::datatype::AccessorType::chunk_face, UseSIMD, MemorySpace,
          MemoryTraits> {
  constexpr static auto accessor_type =
      specfem::datatype::AccessorType::chunk_face; ///< Accessor type for
                                                   ///< identifying the
                                                   ///< class
  constexpr static int components = Components;
  constexpr static int dimensions = NumberOfDimensions;
};
} // namespace datatype
} // namespace specfem
