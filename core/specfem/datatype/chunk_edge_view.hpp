#pragma once

#include "chunk_ndim_view.hpp"

// Forward declarations
namespace specfem::point {
template <specfem::element::dimension_tag DimensionTag, bool UseSIMD>
struct index;
} // namespace specfem::point

namespace specfem {
namespace datatype {
template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfFaces, int NumberOfGLLPoints, bool UseSIMD = false,
          typename... ViewParameters>
struct ScalarChunkEdgeViewType
    : public ChunkNDimViewType<T, DimensionTag, NumberOfFaces,
                               NumberOfGLLPoints, 1 /* edge: 1 coordinate */,
                               std::integer_sequence<int>,
                               specfem::datatype::AccessorType::chunk_face,
                               UseSIMD, ViewParameters...> {
  using chunk_ndim_view_type =
      ChunkNDimViewType<T, DimensionTag, NumberOfFaces, NumberOfGLLPoints, 1,
                        std::integer_sequence<int>,
                        specfem::datatype::AccessorType::chunk_face, UseSIMD,
                        ViewParameters...>;

  // explicit constructor pass-through because nvcc hates you and will do
  // anything in its power to keep you from ever seeing world peace
  using chunk_ndim_view_type::chunk_ndim_view_type;
};

template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfFaces, int NumberOfGLLPoints, int Components,
          bool UseSIMD = false, typename... ViewParameters>
struct VectorChunkEdgeViewType
    : public ChunkNDimViewType<T, DimensionTag, NumberOfFaces,
                               NumberOfGLLPoints, 1 /* edge: 1 coordinate */,
                               std::integer_sequence<int, Components>,
                               specfem::datatype::AccessorType::chunk_face,
                               UseSIMD, ViewParameters...> {
  using chunk_ndim_view_type =
      ChunkNDimViewType<T, DimensionTag, NumberOfFaces, NumberOfGLLPoints, 1,
                        std::integer_sequence<int, Components>,
                        specfem::datatype::AccessorType::chunk_face, UseSIMD,
                        ViewParameters...>;

  constexpr static int components = Components;

  // explicit constructor pass-through because nvcc hates you and will do
  // anything in its power to keep you from ever seeing world peace
  using chunk_ndim_view_type::chunk_ndim_view_type;
};

template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfFaces, int NumberOfGLLPoints, int Components,
          int NumberOfDimensions, bool UseSIMD = false,
          typename... ViewParameters>
struct TensorChunkEdgeViewType
    : public ChunkNDimViewType<
          T, DimensionTag, NumberOfFaces, NumberOfGLLPoints,
          1 /* edge: 1 coordinate */,
          std::integer_sequence<int, Components, NumberOfDimensions>,
          specfem::datatype::AccessorType::chunk_face, UseSIMD,
          ViewParameters...> {

  using chunk_ndim_view_type = ChunkNDimViewType<
      T, DimensionTag, NumberOfFaces, NumberOfGLLPoints, 1,
      std::integer_sequence<int, Components, NumberOfDimensions>,
      specfem::datatype::AccessorType::chunk_face, UseSIMD, ViewParameters...>;

  constexpr static int components = Components;
  constexpr static int dimensions = NumberOfDimensions;

  // explicit constructor pass-through because nvcc hates you and will do
  // anything in its power to keep you from ever seeing world peace
  using chunk_ndim_view_type::chunk_ndim_view_type;
};

} // namespace datatype
} // namespace specfem
