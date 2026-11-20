#pragma once

#include "chunk_ndim_view.hpp"
#include "impl/chunk_face_subview.hpp"
#include "specfem/datatype/accessor_type.hpp"
#include "specfem/datatype/point_view.hpp"
#include <utility>

namespace specfem {
namespace datatype {
template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfFaces, int NumberOfGLLPoints, bool UseSIMD = false,
          typename... ViewParameters>
struct ScalarChunkFaceViewType
    : public ChunkNDimViewType<T, DimensionTag, NumberOfFaces,
                               NumberOfGLLPoints, 2 /*face: 2 coordinates */,
                               std::integer_sequence<int>,
                               specfem::datatype::AccessorType::chunk_face,
                               UseSIMD, ViewParameters...> {
  using base_type =
      ChunkNDimViewType<T, DimensionTag, NumberOfFaces, NumberOfGLLPoints, 2,
                        std::integer_sequence<int>,
                        specfem::datatype::AccessorType::chunk_face, UseSIMD,
                        ViewParameters...>;

  // explicit constructor pass-through because nvcc hates you and will do
  // anything in its power to keep you from ever seeing world peace
  using base_type::base_type;
};

template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfFaces, int NumberOfGLLPoints, int Components,
          bool UseSIMD = false, typename... ViewParameters>
struct VectorChunkFaceViewType
    : public ChunkNDimViewType<T, DimensionTag, NumberOfFaces,
                               NumberOfGLLPoints, 2 /*face: 2 coordinates */,
                               std::integer_sequence<int, Components>,
                               specfem::datatype::AccessorType::chunk_face,
                               UseSIMD, ViewParameters...> {
  using base_type =
      ChunkNDimViewType<T, DimensionTag, NumberOfFaces, NumberOfGLLPoints, 2,
                        std::integer_sequence<int, Components>,
                        specfem::datatype::AccessorType::chunk_face, UseSIMD,
                        ViewParameters...>;
  using point_view_type =
      VectorPointViewType<T, Components, UseSIMD>; ///< Point view type for
                                                   ///< component access

  constexpr static int components = Components;

  // explicit constructor pass-through because nvcc hates you and will do
  // anything in its power to keep you from ever seeing world peace
  using base_type::base_type;
  using base_type::operator();

  /**
   * @brief Get vector subview by a chunk-local face point index.
   *
   * @param index Face point index whose `iface` is the position of the face
   * within the chunk
   */
  KOKKOS_INLINE_FUNCTION
  impl::VectorChunkFaceSubview<VectorChunkFaceViewType>
  operator()(const base_type::index_type &index) {
    return { *this, index };
  }
};

template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfFaces, int NumberOfGLLPoints, int Components,
          int NumberOfDimensions, bool UseSIMD = false,
          typename... ViewParameters>
struct TensorChunkFaceViewType
    : public ChunkNDimViewType<
          T, DimensionTag, NumberOfFaces, NumberOfGLLPoints,
          2 /*face: 2 coordinates */,
          std::integer_sequence<int, Components, NumberOfDimensions>,
          specfem::datatype::AccessorType::chunk_face, UseSIMD,
          ViewParameters...> {

  using base_type = ChunkNDimViewType<
      T, DimensionTag, NumberOfFaces, NumberOfGLLPoints, 2,
      std::integer_sequence<int, Components, NumberOfDimensions>,
      specfem::datatype::AccessorType::chunk_face, UseSIMD, ViewParameters...>;

  constexpr static int components = Components;
  constexpr static int dimensions = NumberOfDimensions;

  // explicit constructor pass-through because nvcc hates you and will do
  // anything in its power to keep you from ever seeing world peace
  using base_type::base_type;
};
} // namespace datatype
} // namespace specfem
