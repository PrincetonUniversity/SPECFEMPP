#pragma once

#include "chunk_ndim_view.hpp"
#include "impl/chunk_edge_subview.hpp"

// Forward declarations
namespace specfem::point {
template <specfem::element::dimension_tag DimensionTag, bool UseSIMD>
struct index;
} // namespace specfem::point

namespace specfem {
namespace datatype {
template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfEdges, int NumberOfGLLPoints, bool UseSIMD = false,
          typename... ViewParameters>
struct ScalarChunkEdgeViewType
    : public ChunkNDimViewType<T, DimensionTag, NumberOfEdges,
                               NumberOfGLLPoints, 1 /* edge: 1 coordinate */,
                               std::integer_sequence<int>,
                               specfem::datatype::AccessorType::chunk_edge,
                               UseSIMD, ViewParameters...> {
  using chunk_ndim_view_type =
      ChunkNDimViewType<T, DimensionTag, NumberOfEdges, NumberOfGLLPoints, 1,
                        std::integer_sequence<int>,
                        specfem::datatype::AccessorType::chunk_edge, UseSIMD,
                        ViewParameters...>;

  // explicit constructor pass-through because nvcc hates you and will do
  // anything in its power to keep you from ever seeing world peace
  using chunk_ndim_view_type::chunk_ndim_view_type;
  constexpr static int nedges = NumberOfEdges;
  using chunk_ndim_view_type::operator();

  /**
   * @brief Get scalar value by a point index.
   *
   * @param index Point index
   */
  KOKKOS_INLINE_FUNCTION
  constexpr chunk_ndim_view_type::value_type &
  operator()(chunk_ndim_view_type::index_type index) {
    return (*this)(index.ispec, index.ipoint);
  }
};

template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfEdges, int NumberOfGLLPoints, int Components,
          bool UseSIMD = false, typename... ViewParameters>
struct VectorChunkEdgeViewType
    : public ChunkNDimViewType<T, DimensionTag, NumberOfEdges,
                               NumberOfGLLPoints, 1 /* edge: 1 coordinate */,
                               std::integer_sequence<int, Components>,
                               specfem::datatype::AccessorType::chunk_edge,
                               UseSIMD, ViewParameters...> {
  using chunk_ndim_view_type =
      ChunkNDimViewType<T, DimensionTag, NumberOfEdges, NumberOfGLLPoints, 1,
                        std::integer_sequence<int, Components>,
                        specfem::datatype::AccessorType::chunk_edge, UseSIMD,
                        ViewParameters...>;

  constexpr static int components = Components;

  // explicit constructor pass-through because nvcc hates you and will do
  // anything in its power to keep you from ever seeing world peace
  using chunk_ndim_view_type::chunk_ndim_view_type;
  constexpr static int nedges = NumberOfEdges;
  using chunk_ndim_view_type::operator();

  /**
   * @brief Get vector subview by a point index.
   *
   * @param index Point index
   */
  KOKKOS_INLINE_FUNCTION
  impl::VectorChunkEdgeSubview<VectorChunkEdgeViewType>
  operator()(const chunk_ndim_view_type::index_type &index) {
    return { *this, index };
  }
};

template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfEdges, int NumberOfGLLPoints, int Components,
          int NumberOfDimensions, bool UseSIMD = false,
          typename... ViewParameters>
struct TensorChunkEdgeViewType
    : public ChunkNDimViewType<
          T, DimensionTag, NumberOfEdges, NumberOfGLLPoints,
          1 /* edge: 1 coordinate */,
          std::integer_sequence<int, Components, NumberOfDimensions>,
          specfem::datatype::AccessorType::chunk_edge, UseSIMD,
          ViewParameters...> {

  using chunk_ndim_view_type = ChunkNDimViewType<
      T, DimensionTag, NumberOfEdges, NumberOfGLLPoints, 1,
      std::integer_sequence<int, Components, NumberOfDimensions>,
      specfem::datatype::AccessorType::chunk_edge, UseSIMD, ViewParameters...>;

  constexpr static int components = Components;
  constexpr static int dimensions = NumberOfDimensions;

  // explicit constructor pass-through because nvcc hates you and will do
  // anything in its power to keep you from ever seeing world peace
  using chunk_ndim_view_type::chunk_ndim_view_type;
  constexpr static int nedges = NumberOfEdges;

  using chunk_ndim_view_type::operator();
  /**
   * @brief Get tensor subview by a point index.
   *
   * @param index Point index
   */
  KOKKOS_INLINE_FUNCTION
  impl::TensorChunkEdgeSubview<TensorChunkEdgeViewType>
  operator()(const chunk_ndim_view_type::index_type &index) {
    return { *this, index };
  }
};

} // namespace datatype
} // namespace specfem
