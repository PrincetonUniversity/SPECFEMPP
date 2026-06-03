#pragma once

#include "accessor_type.hpp"
#include "chunk_ndim_view.hpp"
#include "impl/chunk_element_subview.hpp"
#include <Kokkos_Core.hpp>

// Forward declarations
namespace specfem::point {
template <specfem::element::dimension_tag DimensionTag, bool UseSIMD>
struct index;
} // namespace specfem::point

namespace specfem {
namespace datatype {

/**
 * @brief Datatype used to scalar values within chunk of elements. Data is
 * stored within a Kokkos view located in the memory space specified by
 * MemorySpace.
 *
 * @tparam T Data type of the scalar values
 * @tparam NumberOfElements Number of elements in the chunk
 * @tparam NumberOfGLLPoints Number of GLL points in each element
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfElements, int NumberOfGLLPoints, bool UseSIMD = false,
          typename... ViewParameters>
struct ScalarChunkElementViewType
    : public ChunkNDimViewType<T, DimensionTag, NumberOfElements,
                               NumberOfGLLPoints,
                               specfem::element::dimension<DimensionTag>::dim,
                               std::integer_sequence<int>,
                               specfem::datatype::AccessorType::chunk_element,
                               UseSIMD, ViewParameters...> {
  using base_type =
      ChunkNDimViewType<T, DimensionTag, NumberOfElements, NumberOfGLLPoints,
                        specfem::element::dimension<DimensionTag>::dim,
                        std::integer_sequence<int>,
                        specfem::datatype::AccessorType::chunk_element, UseSIMD,
                        ViewParameters...>;

  constexpr static int nelements = NumberOfElements; ///< Number of elements in
                                                     ///< the chunk
};

/**
 * @brief 2D Datatype used to vector values within chunk of elements. Data is
 * stored within a Kokkos view located in the memory space specified by
 * MemorySpace.
 *
 * @tparam T Data type of the vector values
 * @tparam NumberOfElements Number of elements in the chunk
 * @tparam NumberOfGLLPoints Number of GLL points in each element
 * @tparam Components Number of vector values (components) at each GLL point
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfElements, int NumberOfGLLPoints, int Components,
          bool UseSIMD = false, typename... ViewParameters>
struct VectorChunkElementViewType
    : public ChunkNDimViewType<T, DimensionTag, NumberOfElements,
                               NumberOfGLLPoints,
                               specfem::element::dimension<DimensionTag>::dim,
                               std::integer_sequence<int, Components>,
                               specfem::datatype::AccessorType::chunk_element,
                               UseSIMD, ViewParameters...> {
  using base_type =
      ChunkNDimViewType<T, DimensionTag, NumberOfElements, NumberOfGLLPoints,
                        specfem::element::dimension<DimensionTag>::dim,
                        std::integer_sequence<int, Components>,
                        specfem::datatype::AccessorType::chunk_element, UseSIMD,
                        ViewParameters...>;
  using point_view_type =
      VectorPointViewType<T, Components, UseSIMD>; ///< Point view type for
                                                   ///< component access
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int nelements = NumberOfElements; ///< Number of elements in
                                                     ///< the chunk
  constexpr static int components = Components; ///< Number of vector values at
                                                ///< each GLL point
  ///@}

  using base_type::operator();
  using base_type::base_type;

  /**
   * @brief Get vector subview by a point index.
   *
   * @param index Point index
   */
  KOKKOS_INLINE_FUNCTION
  impl::VectorChunkElementSubview<VectorChunkElementViewType>
  operator()(const base_type::index_type &index) {
    return { *this, index };
  }
};

/**
 * @brief Datatype used to tensor values within chunk of elements. Data is
 * stored within a Kokkos view located in the memory space specified by
 * MemorySpace.
 *
 * @tparam T Data type of the tensor values
 * @tparam NumberOfElements Number of elements in the chunk
 * @tparam NumberOfGLLPoints Number of GLL points in each element
 * @tparam Components Number of vector values (components) at each GLL point
 * @tparam NumberOfDimensions Number of dimensions of the tensor
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, specfem::element::dimension_tag DimensionTag,
          int NumberOfElements, int NumberOfGLLPoints, int Components,
          int NumberOfDimensions, bool UseSIMD = false,
          typename... ViewParameters>
struct TensorChunkElementViewType
    : public ChunkNDimViewType<
          T, DimensionTag, NumberOfElements, NumberOfGLLPoints,
          specfem::element::dimension<DimensionTag>::dim,
          std::integer_sequence<int, Components, NumberOfDimensions>,
          specfem::datatype::AccessorType::chunk_element, UseSIMD,
          ViewParameters...> {

  using base_type = ChunkNDimViewType<
      T, DimensionTag, NumberOfElements, NumberOfGLLPoints,
      specfem::element::dimension<DimensionTag>::dim,
      std::integer_sequence<int, Components, NumberOfDimensions>,
      specfem::datatype::AccessorType::chunk_element, UseSIMD,
      ViewParameters...>;

  constexpr static int nelements = NumberOfElements; ///< Number of elements in
  ///< the chunk
  constexpr static int components = Components;
  constexpr static int dimensions = NumberOfDimensions;
  using point_view_type = TensorPointViewType<T, Components, NumberOfDimensions,
                                              UseSIMD>; ///< Point view type for
                                                        ///< component access

  using base_type::operator();
  using base_type::base_type;
  /**
   * @brief Get tensor subview by a point index.
   *
   * @param index Point index
   */
  KOKKOS_INLINE_FUNCTION
  impl::TensorChunkElementSubview<TensorChunkElementViewType>
  operator()(const base_type::index_type &index) {
    return { *this, index };
  }
};

} // namespace datatype
} // namespace specfem
